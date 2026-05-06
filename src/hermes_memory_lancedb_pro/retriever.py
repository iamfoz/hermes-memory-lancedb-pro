"""
Hybrid retrieval engine with vector-dominant fusion and multi-stage scoring.

Architecture from CortexReach memory-lancedb-pro:
  1. Vector search + BM25 FTS search in parallel
  2. Vector-score-dominant fusion with BM25 confirmation bonus
  3. Multi-stage scoring pipeline:
     - Length normalisation
     - Hard min score filter (BEFORE decay)
     - Composite decay scoring (Weibull)
  4. Optional cross-encoder reranking (LangSearch or Google Ranking API)
  5. MMR diversity demotion
  6. BM25 ghost entry protection via store.check_ids()
  7. Lifecycle hooks: access count increment + tier evaluation on recall

Reranker selection
------------------
Set ``MEMORY_RERANKER`` to choose explicitly:

  - ``langsearch``  — LangSearch cross-encoder (requires LANGSEARCH_API_KEY)
  - ``google``      — Google Discovery Engine Ranking API (requires
                      GOOGLE_API_KEY + GOOGLE_CLOUD_PROJECT)
  - ``disabled``    — skip reranking entirely
  - ``auto``        — (default) choose the one whose keys are present;
                      if both are configured, log a warning and disable
                      until the user sets MEMORY_RERANKER explicitly.

Env-var timing note
-------------------
API keys are read inside ``MemoryRetriever.__init__``, not at module-import
time.  This is intentional: hermes-agent (and many other hosts) load
``~/.hermes/.env`` into ``os.environ`` *after* Python imports the plugin
modules, so a module-level ``os.environ.get("GOOGLE_API_KEY")`` would
always return ``""`` even when the key is present in the file.  By
deferring to ``__init__`` — which is called at provider-instantiation time,
well after dotenv loading — the retriever always sees the fully-populated
environment.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import requests

from .decay import (
    ScoringPipeline,
    _coerce_metadata,
    evaluate_all_tiers,
    is_noise,
    mmr_diversity_filter,
)
from .store import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static config — safe at module level because these never come from .env
# ---------------------------------------------------------------------------

LANGSEARCH_BASE_URL = "https://api.langsearch.com/v1/rerank"
LANGSEARCH_MODEL = "langsearch-reranker-v1"
LANGSEARCH_MAX_DOCS = 50   # API limit per request
LANGSEARCH_TIMEOUT = 15    # seconds

GOOGLE_RANKING_BASE_URL = (
    "https://discoveryengine.googleapis.com/v1/projects/{project}"
    "/locations/global/rankingConfigs/default_ranking_config:rank"
)
GOOGLE_RANKING_MAX_DOCS = 200   # API limit per request
GOOGLE_RANKING_TIMEOUT = 15     # seconds

# ---------------------------------------------------------------------------
# Tuning constants — these are non-secret; reading at import is fine because
# they're set in the system environment, not in the user's .env file, and
# their defaults are always usable even if the env var isn't present yet.
# ---------------------------------------------------------------------------

TIER_EVAL_FREQUENCY = int(os.environ.get("MEMORY_TIER_EVAL_FREQUENCY", "10"))
TIER_EVAL_BATCH = int(os.environ.get("MEMORY_TIER_EVAL_BATCH", "500"))
DEFAULT_MIN_RECALL_SCORE: float = float(
    os.environ.get("MEMORY_MIN_RECALL_SCORE", "0.0")
)


class MemoryRetriever:
    """
    Hybrid retrieval with vector-dominant fusion and spec-compliant scoring.

    Fusion strategy (per CortexReach spec):
      - Vector score is the primary signal
      - BM25 hit provides a confirmation boost (not a full parallel ranking)
      - Pure BM25 hits allowed but with a lower floor
      - BM25 ghost entries filtered via store.check_ids()
      - Preservation floor for high BM25 lexical hits to prevent reranker kills
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self.scoring = ScoringPipeline()
        self._lifecycle_call_count = 0

        # -----------------------------------------------------------------
        # Read API keys HERE, not at module level.
        #
        # hermes-agent loads ~/.hermes/.env into os.environ during startup,
        # AFTER Python has already imported all plugin modules.  Reading keys
        # at module level (e.g. at the top of this file) means they're always
        # empty because the import runs before dotenv loading.
        #
        # MemoryRetriever.__init__ is called when the provider is
        # instantiated, which happens after dotenv loading, so by this point
        # os.environ is fully populated.
        # -----------------------------------------------------------------
        self._langsearch_api_key: str = os.environ.get("LANGSEARCH_API_KEY", "")
        self._google_api_key: str = os.environ.get("GOOGLE_API_KEY", "")
        self._google_cloud_project: str = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GOOGLE_PROJECT_ID")
            or os.environ.get("GOOGLE_PROJECT")
            or ""
        )
        self._google_ranking_model: str = os.environ.get(
            "MEMORY_GOOGLE_RANKING_MODEL", "semantic-ranker-512@latest"
        )

        # Resolve which backend to use (logs once, here at construction time)
        self._active_reranker: str = self._resolve_reranker(
            langsearch_key=self._langsearch_api_key,
            google_key=self._google_api_key,
            google_project=self._google_cloud_project,
            setting=os.environ.get("MEMORY_RERANKER", "auto").strip().lower(),
        )

        # One requests.Session per backend, created lazily on first use
        self._langsearch_session: requests.Session | None = None
        self._google_session: requests.Session | None = None

        # Per-backend kill-switch: tripped on persistent 401/403/429
        self._langsearch_disabled: bool = False
        self._google_disabled: bool = False

    # ----- public -----

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        scope: str | None = None,
        source: str = "manual",
        *,
        session_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: search text (the user's current message, typically).
            limit: maximum number of results to return.
            category, scope: optional column filters.
            source: "manual" / "auto-recall" / "cli" — drives the recall
                lifecycle hooks. Use "cli" to skip access-count bumps.
            session_id: when set, results are restricted to memories whose
                ``metadata.source_session`` matches this id, plus memories
                explicitly flagged ``cross_session`` or living in the core
                tier.
            min_score: drop final results below this threshold. Defaults to
                ``DEFAULT_MIN_RECALL_SCORE`` (env ``MEMORY_MIN_RECALL_SCORE``).
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        now_ms = int(time.time() * 1000)
        if min_score is None:
            min_score = DEFAULT_MIN_RECALL_SCORE

        # 1. Parallel vector + BM25 search (wide initial net)
        candidate_pool = min(max(limit * 4, 16), 40)
        vector_results = self.store._vector_search(
            query, candidate_pool, category, scope, session_id=session_id,
        )
        bm25_results = self.store._bm25_search(
            query, candidate_pool, category, scope, session_id=session_id,
        )

        # 1.5 BM25 ghost entry protection
        bm25_results = self._filter_bm25_ghosts(bm25_results, vector_results)

        # 2. Vector-dominant fusion with BM25 confirmation bonus
        fused = self._vector_dominant_fusion(vector_results, bm25_results)
        if not fused:
            return []

        # 3. Scoring pipeline (length norm -> hardMinScore -> decay -> sort)
        scored = self.scoring.apply_scoring(fused, now_ms)

        # 4. Noise filter
        scored = [e for e in scored if not is_noise(e.get("text", ""))]

        # 5. Cross-encoder rerank (best relevance signal — runs before MMR
        # so diversity operates on the most accurate ordering)
        scored = self._rerank(query, scored, top_n=min(5, limit))

        # 6. MMR diversity (demotes near-duplicates)
        scored_tuples = [(e, float(e.get("_final_score", 0.0))) for e in scored]
        diverse = mmr_diversity_filter(scored_tuples, similarity_threshold=0.85)
        scored = [e for e, score in diverse]
        for entry, score in diverse:
            entry["_mmr_score"] = score

        # 7. Apply min_score gate
        if min_score and min_score > 0.0:
            scored = [
                e for e in scored
                if float(e.get("_final_score", 0.0)) >= min_score
            ]

        # 8. Lifecycle hooks
        if source in ("manual", "auto-recall"):
            self._run_recall_lifecycle(scored[:limit])

        return scored[:limit]

    # ----- reranker resolution -----

    @staticmethod
    def _resolve_reranker(
        *,
        langsearch_key: str,
        google_key: str,
        google_project: str,
        setting: str,
    ) -> str:
        """
        Choose which reranker backend to activate.

        Returns ``"langsearch"``, ``"google"``, or ``""`` (disabled).

        Accepts the key values and the ``MEMORY_RERANKER`` setting as
        explicit arguments rather than reading ``os.environ`` directly, so
        that tests can call it without environment manipulation.
        """
        have_langsearch = bool(langsearch_key)
        # Google requires BOTH key AND project (project appears in the URL)
        have_google = bool(google_key and google_project)

        if setting == "disabled":
            return ""

        if setting == "langsearch":
            if not have_langsearch:
                logger.warning(
                    "MEMORY_RERANKER=langsearch but LANGSEARCH_API_KEY is not set; "
                    "reranking disabled."
                )
                return ""
            return "langsearch"

        if setting == "google":
            if not have_google:
                missing = []
                if not google_key:
                    missing.append("GOOGLE_API_KEY")
                if not google_project:
                    missing.append("GOOGLE_CLOUD_PROJECT (or GOOGLE_PROJECT_ID)")
                logger.warning(
                    "MEMORY_RERANKER=google but %s not set; reranking disabled.",
                    " and ".join(missing),
                )
                return ""
            return "google"

        # auto — pick the configured one; warn if ambiguous
        if have_langsearch and have_google:
            logger.warning(
                "Both LANGSEARCH_API_KEY and GOOGLE_API_KEY+GOOGLE_CLOUD_PROJECT are "
                "set. Set MEMORY_RERANKER=langsearch or MEMORY_RERANKER=google to "
                "choose one. Reranking is disabled until configured."
            )
            return ""
        if have_langsearch:
            logger.info("Reranker: LangSearch (auto-selected, LANGSEARCH_API_KEY found)")
            return "langsearch"
        if have_google:
            logger.info("Reranker: Google Ranking API (auto-selected)")
            return "google"

        return ""

    # ----- helpers -----

    def _filter_bm25_ghosts(
        self,
        bm25_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop BM25-only hits that don't exist (or are archived) in the store."""
        vector_ids = {r["id"] for r in vector_results}
        bm25_only_ids = [
            entry["id"] for entry in bm25_results if entry["id"] not in vector_ids
        ]
        confirmed = (
            set(self.store.check_ids(bm25_only_ids)) if bm25_only_ids else set()
        )
        return [
            entry for entry in bm25_results
            if entry["id"] in vector_ids or entry["id"] in confirmed
        ]

    def _vector_dominant_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Vector-score-dominant fusion with BM25 confirmation bonus."""
        vector_map = {r["id"]: r for r in vector_results}
        vector_ranks = {r["id"]: i + 1 for i, r in enumerate(vector_results)}
        bm25_map = {r["id"]: r for r in bm25_results}
        bm25_ranks = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}

        ordered_ids = list(vector_map.keys()) + [
            mid for mid in bm25_map if mid not in vector_map
        ]

        vector_dists = [r.get("_distance", 0.0) for r in vector_results]
        max_dist = max(vector_dists) if vector_dists else 1.0
        min_dist = min(vector_dists) if vector_dists else 0.0
        dist_range = max_dist - min_dist

        fused: list[dict[str, Any]] = []
        for mid in ordered_ids:
            entry = vector_map.get(mid) or bm25_map.get(mid)
            if entry is None:
                continue

            in_vector = mid in vector_map
            in_bm25 = mid in bm25_map

            if in_vector:
                raw_dist = entry.get("_distance", 0.0)
                norm_dist = (raw_dist - min_dist) / dist_range if dist_range > 0 else 0.0
                fusion_score = max(0.01, 1.0 - norm_dist)
                if in_bm25:
                    bm25_rank = bm25_ranks[mid]
                    fusion_score *= 1.0 + (0.3 / (1 + math.log1p(bm25_rank)))
                entry["_fusion_source"] = "both" if in_bm25 else "vector"
            else:
                bm25_rank = bm25_ranks[mid]
                fusion_score = 0.15 / (1 + math.log1p(bm25_rank))
                importance = self._extract_importance(entry)
                if importance >= 0.8:
                    fusion_score = max(fusion_score, 0.12)
                elif importance >= 0.6:
                    fusion_score = max(fusion_score, 0.08)
                entry["_fusion_source"] = "bm25"

            entry["_fusion_score"] = fusion_score
            entry["_vector_rank"] = vector_ranks.get(mid) if in_vector else None
            entry["_bm25_rank"] = bm25_ranks.get(mid)
            fused.append(entry)

        fused.sort(key=lambda e: e.get("_fusion_score", 0.0), reverse=True)
        return fused

    @staticmethod
    def _extract_importance(entry: dict[str, Any]) -> float:
        val = entry.get("importance")
        if val is None:
            val = _coerce_metadata(entry.get("metadata", {})).get("importance", 0.5)
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return 0.5

    def _run_recall_lifecycle(self, results: list[dict[str, Any]]) -> None:
        """Lifecycle hooks after recall: access-count increment + tier eval."""
        self._lifecycle_call_count += 1

        for entry in results:
            mem_id = entry.get("id")
            if mem_id:
                try:
                    self.store.increment_access_count(mem_id)
                except Exception as e:
                    logger.warning("increment_access_count failed for %s: %s", mem_id, e)

        if TIER_EVAL_FREQUENCY <= 0:
            return
        if self._lifecycle_call_count % TIER_EVAL_FREQUENCY != 0:
            return

        try:
            all_memories = self.store.list_memories(limit=TIER_EVAL_BATCH)
        except Exception as e:
            logger.warning("Tier eval list_memories failed: %s", e)
            return

        try:
            tier_changes = evaluate_all_tiers(all_memories)
        except Exception as e:
            logger.warning("evaluate_all_tiers failed: %s", e)
            return

        for mem_id, new_tier in tier_changes.items():
            try:
                self.store.update(mem_id, tier=new_tier)
            except Exception as e:
                logger.warning("Tier update for %s failed: %s", mem_id, e)

    # ----- reranking dispatcher -----

    def _rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Dispatch to the active reranker backend, or return results unchanged."""
        if self._active_reranker == "langsearch":
            return self._rerank_langsearch(query, results, top_n)
        if self._active_reranker == "google":
            return self._rerank_google(query, results, top_n)
        return results

    # ----- LangSearch backend -----

    def _rerank_langsearch(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Cross-encoder reranking via LangSearch."""
        if self._langsearch_disabled:
            return results
        if len(results) <= 1:
            return results

        candidate_count = min(top_n * 3, LANGSEARCH_MAX_DOCS, len(results))
        if candidate_count <= 1:
            return results

        candidates = results[:candidate_count]
        documents = [str(c.get("text", "") or "") for c in candidates]
        if not any(documents):
            return results

        if self._langsearch_session is None:
            self._langsearch_session = requests.Session()
            self._langsearch_session.headers.update({
                "Authorization": f"Bearer {self._langsearch_api_key}",
                "Content-Type": "application/json",
            })

        payload = {
            "model": LANGSEARCH_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, candidate_count),
            "return_documents": False,
        }

        data = None
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._langsearch_session.post(
                    LANGSEARCH_BASE_URL,
                    json=payload,
                    timeout=LANGSEARCH_TIMEOUT,
                )
                if resp.status_code in (401, 403, 429):
                    self._langsearch_disabled = True
                    logger.warning(
                        "LangSearch rerank disabled for this session "
                        "(HTTP %d). Check LANGSEARCH_API_KEY or quota. "
                        "Falling back to fusion-only ranking.",
                        resp.status_code,
                    )
                    return results
                if 500 <= resp.status_code < 600 and attempt == 0:
                    last_err = Exception(f"5xx on rerank: {resp.status_code}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                last_err = None
                break
            except requests.RequestException as e:
                last_err = e
                if attempt == 0 and isinstance(e, (requests.ConnectionError, requests.Timeout)):
                    continue
                break
            except ValueError as e:
                last_err = e
                break

        if data is None:
            if last_err is not None:
                logger.warning("LangSearch rerank failed: %s", last_err)
            return results

        if isinstance(data, dict) and data.get("code") not in (None, 200):
            logger.warning(
                "LangSearch rerank returned error: %s",
                data.get("msg", "unknown"),
            )
            return results

        lang_results = (data or {}).get("results") or []
        if not lang_results:
            return results

        score_map: dict[int, float] = {}
        for item in lang_results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if isinstance(idx, int) and isinstance(score, (int, float)):
                score_map[idx] = float(score)
        if not score_map:
            return results

        return self._apply_rerank_scores(candidates, results, candidate_count, score_map)

    # ----- Google Ranking API backend -----

    def _rerank_google(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Cross-encoder reranking via Google Discovery Engine Ranking API.

        Uses ``GOOGLE_API_KEY`` (via ``x-goog-api-key`` header) and
        ``GOOGLE_CLOUD_PROJECT`` (in the request URL).  Both are read at
        ``MemoryRetriever.__init__`` time, not at module-import time, so
        they correctly pick up values loaded from ``~/.hermes/.env``.

        Model: ``semantic-ranker-512@latest`` (overridable via
        ``MEMORY_GOOGLE_RANKING_MODEL``).

        Pricing: 1,000 free queries/month; ~$0.001/query thereafter.
        """
        if self._google_disabled:
            return results
        if len(results) <= 1:
            return results

        candidate_count = min(top_n * 3, GOOGLE_RANKING_MAX_DOCS, len(results))
        if candidate_count <= 1:
            return results

        candidates = results[:candidate_count]
        if not any(c.get("text") for c in candidates):
            return results

        if self._google_session is None:
            self._google_session = requests.Session()
            self._google_session.headers.update({
                "x-goog-api-key": self._google_api_key,
                "Content-Type": "application/json",
            })

        records = [
            {"id": str(i), "content": str(c.get("text", "") or "")}
            for i, c in enumerate(candidates)
        ]
        payload = {
            "model": self._google_ranking_model,
            "topN": candidate_count,
            "query": query,
            "records": records,
        }
        url = GOOGLE_RANKING_BASE_URL.format(project=self._google_cloud_project)

        data = None
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._google_session.post(
                    url,
                    json=payload,
                    timeout=GOOGLE_RANKING_TIMEOUT,
                )
                if resp.status_code in (401, 403, 429):
                    self._google_disabled = True
                    logger.warning(
                        "Google rerank disabled for this session "
                        "(HTTP %d). Check GOOGLE_API_KEY / GOOGLE_CLOUD_PROJECT "
                        "or quota. Falling back to fusion-only ranking.",
                        resp.status_code,
                    )
                    return results
                if resp.status_code == 404:
                    self._google_disabled = True
                    logger.warning(
                        "Google rerank got 404. Ensure the Discovery Engine API "
                        "is enabled for project %r and GOOGLE_CLOUD_PROJECT is "
                        "correct. Falling back to fusion-only ranking.",
                        self._google_cloud_project,
                    )
                    return results
                if 500 <= resp.status_code < 600 and attempt == 0:
                    last_err = Exception(f"5xx on Google rerank: {resp.status_code}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                last_err = None
                break
            except requests.RequestException as e:
                last_err = e
                if attempt == 0 and isinstance(e, (requests.ConnectionError, requests.Timeout)):
                    continue
                break
            except ValueError as e:
                last_err = e
                break

        if data is None:
            if last_err is not None:
                logger.warning("Google rerank failed: %s", last_err)
            return results

        google_records = (data or {}).get("records") or []
        if not google_records:
            return results

        score_map: dict[int, float] = {}
        for rec in google_records:
            try:
                idx = int(rec["id"])
            except (KeyError, TypeError, ValueError):
                continue
            score = rec.get("relevanceScore")
            if isinstance(score, (int, float)):
                score_map[idx] = float(score)
        if not score_map:
            return results

        # Normalise within the batch — Google's scores cluster in a narrow
        # band and need rescaling before the 70/30 blend.
        if score_map:
            max_s = max(score_map.values())
            min_s = min(score_map.values())
            span = max_s - min_s
            if span > 0:
                score_map = {k: (v - min_s) / span for k, v in score_map.items()}

        return self._apply_rerank_scores(candidates, results, candidate_count, score_map)

    # ----- shared rerank blend logic -----

    @staticmethod
    def _apply_rerank_scores(
        candidates: list[dict[str, Any]],
        results: list[dict[str, Any]],
        candidate_count: int,
        score_map: dict[int, float],
    ) -> list[dict[str, Any]]:
        """
        Blend reranker scores into ``_final_score`` and re-sort.

        ``score_map`` maps candidate-list index → normalised relevance score.
        Candidates without a score entry are penalised.  The tail of
        ``results`` beyond ``candidate_count`` is preserved unchanged.
        """
        for i, entry in enumerate(candidates):
            existing = float(entry.get("_final_score", 0.0))
            if i in score_map:
                rerank = score_map[i]
                entry["_rerank_score"] = rerank
                entry["_final_score"] = 0.7 * rerank + 0.3 * existing
            else:
                entry["_final_score"] = 0.3 * existing
                entry["_rerank_score"] = 0.0

        candidates.sort(key=lambda e: e.get("_final_score", 0.0), reverse=True)
        results[:candidate_count] = candidates
        return results


# Back-compat alias for callers that imported the old name
HybridRetriever = MemoryRetriever
