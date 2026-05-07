"""
Hybrid retrieval engine with vector-dominant fusion and multi-stage scoring.

Architecture from CortexReach memory-lancedb-pro:
  1. Vector search + BM25 FTS search in parallel
  2. Vector-score-dominant fusion with BM25 confirmation bonus
  3. Multi-stage scoring pipeline:
     - Length normalisation
     - Hard min score filter (BEFORE decay)
     - Composite decay scoring (Weibull)
     - Noise filter
     - MMR diversity
  4. BM25 ghost entry protection via store.hasId()
  5. Lifecycle hooks: access count increment + tier evaluation on recall
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .store import MemoryStore
from .decay import (
    ScoringPipeline,
    mmr_diversity_filter,
    is_noise,
    evaluate_all_tiers,
)

logger = logging.getLogger(__name__)

# LangSearch Reranking API config
LANGSEARCH_API_KEY = os.environ.get("LANGSEARCH_API_KEY", "")
LANGSEARCH_BASE_URL = "https://api.langsearch.com/v1/rerank"
LANGSEARCH_MODEL = "langsearch-reranker-v1"
LANGSEARCH_MAX_DOCS = 50  # API limit per request
LANGSEARCH_TIMEOUT = 15  # seconds


class HybridRetriever:
    """
    Hybrid retrieval with vector-dominant fusion and spec-compliant scoring.

    Fusion strategy (per CortexReach spec):
      - Vector score is the primary signal
      - BM25 hit provides a confirmation boost (not a full parallel ranking)
      - Pure BM25 hits allowed but with a lower floor
      - BM25 ghost entries filtered via store.hasId()
      - Preservation floor for high BM25 lexical hits to prevent reranker kills
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self.scoring = ScoringPipeline()

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
        scope: Optional[str] = None,
        source: str = "manual",
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with full scoring pipeline.

        Args:
            query: Search query text
            limit: Maximum number of results
            category: Optional category filter
            scope: Optional scope filter
            source: Retrieval source ("manual", "auto-recall", "cli")

        Returns:
            List of memory entries with scores
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        now_ms = int(time.time() * 1000)

        # Step 1: Parallel vector + BM25 search
        candidate_pool = min(limit * 4, 40)  # Wide initial net

        vector_results = self.store._vector_search(query, candidate_pool, category, scope)
        bm25_results = self.store._bm25_search(query, candidate_pool, category, scope)

        # Step 1.5: BM25 ghost entry protection
        bm25_results = self._filter_bm25_ghosts(bm25_results, vector_results)

        # Step 2: Vector-dominant fusion with BM25 confirmation bonus
        fused = self._vector_dominant_fusion(vector_results, bm25_results)

        if not fused:
            return []

        # Step 3: Apply scoring pipeline (length norm -> hardMinScore -> decay -> sort)
        scored = self.scoring.apply_scoring(fused, now_ms)

        # Step 4: Noise filter
        scored = [e for e in scored if not is_noise(e.get("text", ""))]

        # Step 5: Cross-encoder reranking (best relevance signal — before diversity)
        scored = self._rerank(query, scored, top_n=5)

        # Step 6: MMR diversity (filter after rerank so diversity operates on correct ordering)
        scored_tuples = [(e, e.get("_final_score", 0)) for e in scored]
        diverse = mmr_diversity_filter(scored_tuples, similarity_threshold=0.85)
        scored = [e for e, _ in diverse]

        # Step 6: Lifecycle hooks — increment access count for recalled items
        if source in ("manual", "auto-recall"):
            self._run_recall_lifecycle(scored)

        # Step 7: Limit
        return scored[:limit]

    def _filter_bm25_ghosts(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter BM25-only results that are ghost entries (not in vector index).

        Per spec: BM25-only results checked via store.hasId() to avoid FTS
        residual ghost entries. Batch query to avoid N+1 database calls.
        """
        vector_ids = {r["id"] for r in vector_results}
        # BM25-only IDs not in vector results
        bm25_only_ids = [entry["id"] for entry in bm25_results if entry["id"] not in vector_ids]
        
        # Batch check store for BM25-only IDs
        if bm25_only_ids:
            confirmed = set(self.store.check_ids(bm25_only_ids))
        else:
            confirmed = set()
        
        filtered = []
        for entry in bm25_results:
            mid = entry["id"]
            # Keep if also in vector results (definitely real)
            if mid in vector_ids:
                filtered.append(entry)
            elif mid in confirmed:
                filtered.append(entry)
            # Otherwise it's a ghost — drop it
        return filtered

    def _vector_dominant_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Vector-score-dominant fusion with BM25 confirmation bonus.

        Strategy (per CortexReach spec):
          - For vector hits: fusion_score = 1 - vector_distance (normalised)
          - BM25 confirmation bonus: if entry also appears in BM25 results,
            apply a multiplicative boost (1.0 + 0.3 / bm25_rank)
          - Pure BM25 hits: allowed with a base score (0.3 / rank)
          - High BM25 lexical hits get a preservation floor to prevent
            reranker kills

        Returns:
            List of entries with _fusion_score set
        """
        # Build lookup maps + rank dicts
        vector_map = {r["id"]: r for r in vector_results}
        vector_ranks = {r["id"]: i + 1 for i, r in enumerate(vector_results)}
        bm25_map = {r["id"]: r for r in bm25_results}
        bm25_ranks = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}

        # Ordered union: vector results first (in rank order), then BM25-only hits
        all_ids = list(vector_map.keys()) + [mid for mid in bm25_map if mid not in vector_map]

        # Normalise vector distances to [0, 1] range across this batch
        # (raw cosine distances can exceed 1.0 for very dissimilar vectors)
        vector_dists = [r.get("_distance", 0.0) for r in vector_results]
        max_dist = max(vector_dists) if vector_dists else 1.0
        min_dist = min(vector_dists) if vector_dists else 0.0

        fused = []
        for mid in all_ids:
            entry = vector_map.get(mid) or bm25_map.get(mid)
            if entry is None:
                continue

            in_vector = mid in vector_map
            in_bm25 = mid in bm25_map

            if in_vector:
                # Primary signal: normalised vector proximity
                raw_dist = entry.get("_distance", 0.0)
                if max_dist > min_dist:
                    norm_dist = (raw_dist - min_dist) / (max_dist - min_dist)
                else:
                    norm_dist = 0.0
                # Score: close to 1.0 for good matches, clamped to [0.01, 1.0]
                fusion_score = max(0.01, 1.0 - norm_dist)

                # BM25 confirmation bonus
                if in_bm25:
                    bm25_rank = bm25_ranks[mid]
                    bonus = 1.0 + (0.3 / (1 + math.log1p(bm25_rank)))
                    fusion_score *= bonus

                entry["_fusion_source"] = "both" if in_bm25 else "vector"

            elif in_bm25:
                # Pure BM25 hit — lower base score, but with preservation floor
                # for high-importance entries (per spec: prevent reranker kills)
                bm25_rank = bm25_ranks[mid]
                fusion_score = 0.15 / (1 + math.log1p(bm25_rank))
                # Importance preservation floor
                metadata = entry.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                importance = metadata.get("importance", 0.5)
                if importance >= 0.8:
                    fusion_score = max(fusion_score, 0.12)
                elif importance >= 0.6:
                    fusion_score = max(fusion_score, 0.08)
                entry["_fusion_source"] = "bm25"
            else:
                continue

            entry["_fusion_score"] = fusion_score
            entry["_vector_rank"] = vector_ranks.get(mid) if in_vector else None
            entry["_bm25_rank"] = bm25_ranks.get(mid)
            fused.append(entry)

        # Sort by fusion score descending
        fused.sort(key=lambda e: e.get("_fusion_score", 0), reverse=True)

        return fused

    # Throttle counter for lifecycle evaluation
    _lifecycle_call_count = 0

    def _run_recall_lifecycle(
        self,
        results: List[Dict[str, Any]],
    ) -> None:
        """
        Lifecycle hooks after recall (per CortexReach spec):
          1. Increment access count for each recalled item
          2. Update last_accessed_at
          3. Evaluate tier promotions/demotions for affected scope

        Throttling: Access count increment happens every call, but the
        expensive full-store tier evaluation (list_memories + evaluate_all_tiers
        + write-back) only runs every 10th call. Without this throttle, every
        search triggers hundreds of DB operations — list_memories(limit=500),
        decay computation on every memory, then conditional writes back.
        """
        HybridRetriever._lifecycle_call_count += 1

        # Always increment access counts — this is lightweight per-item
        for entry in results:
            mem_id = entry.get("id")
            if mem_id:
                self.store.increment_access_count(mem_id)

        # Throttle full-store tier evaluation to every 10th call
        if HybridRetriever._lifecycle_call_count % 10 != 0:
            return

        # Evaluate tier changes across the full store (not just recalled items)
        # This is the "scoreAll" + "evaluateAll" cycle from the spec
        all_memories = self.store.list_memories(limit=500)
        tier_changes = evaluate_all_tiers(all_memories)

        # Write back tier changes
        for mem_id, new_tier in tier_changes.items():
            self.store.update(mem_id, tier=new_tier)

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic reranking using the LangSearch Reranking API.

        Sends the top-N fused results to LangSearch's dedicated reranker
        which scores each document's semantic relevance to the query
        (0.0–1.0). Far more accurate than the previous LLM-prompt approach.

        Args:
            query: Original search query
            results: Pre-fused and scored results
            top_n: Number of results to return after reranking

        Returns:
            Reranked list of results
        """
        if not LANGSEARCH_API_KEY:
            logger.warning("LANGSEARCH_API_KEY not set, skipping reranking")
            return results

        if len(results) <= 1:
            return results

        # LangSearch accepts up to 50 documents; cap reasonably
        candidates = results[:min(top_n * 3, LANGSEARCH_MAX_DOCS)]

        # Extract document texts for LangSearch
        documents = []
        for entry in candidates:
            text = entry.get("text", "") or ""
            documents.append(text)

        if not documents or all(not d for d in documents):
            return results

        try:
            resp = requests.post(
                LANGSEARCH_BASE_URL,
                headers={
                    "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LANGSEARCH_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": False,
                },
                timeout=LANGSEARCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 200:
                logger.warning("LangSearch rerank returned error: %s", data.get("msg", "unknown"))
                return results

            lang_results = data.get("results", [])
            if not lang_results:
                return results

            # Build a map: index -> relevance_score
            score_map = {}
            for item in lang_results:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)
                if idx is not None:
                    score_map[idx] = score

            # Apply reranking scores — blend: 70% LangSearch, 30% fusion
            for i, entry in enumerate(candidates):
                if i in score_map:
                    entry["_rerank_score"] = score_map[i]
                    entry["_final_score"] = (
                        entry.get("_final_score", 0) * 0.3
                        + score_map[i] * 0.7
                    )

            # Re-sort by final score descending
            results[:len(candidates)] = sorted(
                candidates,
                key=lambda e: e.get("_final_score", 0),
                reverse=True,
            )
            return results

        except Exception as e:
            logger.warning("LangSearch reranking failed: %s", e)
            return results
