"""Smart memory extractor — LLM-driven 6-category extraction with fallback.

Ports CortexReach's `smart-extractor.ts`. Pipeline (LLM mode):
    conversation_text → strip envelope → extract candidates (LLM)
    → batch dedup (cosine) → admission control (optional)
    → for each candidate: vector search neighbours → LLM dedup decision
      → CREATE / MERGE (LLM) / SKIP / SUPERSEDE / SUPPORT / CONTEXTUALIZE / CONTRADICT

Switchable fallback:
    When `llm=None` (no LLM available), `extract_and_persist` falls back
    to the legacy "store raw user + assistant turn" behaviour. Identical
    on-disk shape to what the hermes-agent provider was doing before
    smart extraction landed — so existing memories don't migrate.

Entry point — see `SmartExtractor.extract_and_persist(...)`."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .admission_control import (
    AdmissionAuditRecord,
    AdmissionController,
)
from .batch_dedup import batch_dedup
from .decay import is_noise
from .extraction_prompts import (
    build_dedup_prompt,
    build_extraction_prompt,
    build_merge_prompt,
)
from .memory_categories import (
    ALWAYS_MERGE_CATEGORIES,
    MERGE_SUPPORTED_CATEGORIES,
    TEMPORAL_VERSIONED_CATEGORIES,
    CandidateMemory,
    SmartCategory,
    normalize_category,
)
from .reflection.retry import run_with_reflection_transient_retry_once
from .temporal_classifier import classify_temporal, infer_expiry

if TYPE_CHECKING:  # pragma: no cover
    from .llm_client import LlmClient
    from .store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — keep in sync with TS source
# ---------------------------------------------------------------------------

MAX_MEMORIES_PER_EXTRACTION = 10
MAX_SIMILAR_FOR_PROMPT = 5
SIMILARITY_THRESHOLD = 0.55  # min cosine similarity for vector-search dedup pool
DEFAULT_EXTRACT_MAX_CHARS = 8000

DedupDecision = Literal[
    "create", "merge", "skip", "supersede",
    "support", "contextualize", "contradict",
]
_VALID_DECISIONS: frozenset[str] = frozenset({
    "create", "merge", "skip", "supersede",
    "support", "contextualize", "contradict",
})
_DESTRUCTIVE_DECISIONS: frozenset[str] = frozenset({"supersede", "contradict"})


@dataclass
class DedupResult:
    decision: DedupDecision
    reason: str = ""
    match_id: str | None = None
    context_label: str | None = None


@dataclass
class ExtractionStats:
    created: int = 0
    merged: int = 0
    skipped: int = 0
    rejected: int = 0
    supported: int = 0
    superseded: int = 0
    boundary_skipped: int = 0


@dataclass
class SmartExtractorConfig:
    extract_max_chars: int = DEFAULT_EXTRACT_MAX_CHARS
    user: str = "User"
    default_scope: str = "global"
    persist_admission_audit: bool = True
    # Legacy fallback config: when llm is None, write user_content + assistant_content
    # as separate memories using these defaults
    legacy_user_importance: float = 0.4
    legacy_assistant_importance: float = 0.4
    legacy_category: str = "other"
    # Max concurrent candidates to process (each does a vector search + LLM
    # dedup call). Default 1 = serial, current behaviour. Setting > 1 cuts
    # multi-candidate latency but adds threading complexity and may stress
    # the upstream LLM endpoint. 4 is a reasonable upper bound for most
    # APIs (rate limits, fairness with other tenants).
    dedup_max_workers: int = 1


# ---------------------------------------------------------------------------
# Envelope metadata stripping
# ---------------------------------------------------------------------------
# Mirrors CortexReach's `stripEnvelopeMetadata`. The "leading zone" is the
# stretch of platform wrappers (subagent context, system headers, untrusted
# metadata blocks) that prefix incoming conversation text. We strip those
# but preserve the trailing zone (real conversation), with one exception:
# any inline boilerplate that immediately follows a wrapper prefix gets
# scrubbed too.

_WRAPPER_LINE_RE = re.compile(
    r"^\[(?:Subagent Context|Subagent Task)\](?:\s|$)?", re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"^(?:Results auto-announce to your requester\.?"
    r"|do not busy-poll for status\.?"
    r"|Reply with a brief acknowledgment only\.?"
    r"|Do not use any memory tools\.?)$",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_BOILERPLATE_RE = re.compile(
    r"^(?:(?:You are running as a subagent\b.*?(?:(?<=\.)\s+|$)"
    r"|Results auto-announce to your requester\.?\s*"
    r"|do not busy-poll for status\.?\s*"
    r"|Reply with a brief acknowledgment only\.?\s*"
    r"|Do not use any memory tools\.?\s*))+",
    re.IGNORECASE,
)
_SUBAGENT_RUNNING_RE = re.compile(r"^You are running as a subagent\b", re.IGNORECASE)

# Untrusted-metadata block markers. The TS source matches a few specific
# headers + the JSON code block that follows.
_UNTRUSTED_HEADERS = (
    "Conversation info (untrusted metadata):",
    "Sender (untrusted metadata):",
    "Replied message (untrusted, for context):",
)

_SYSTEM_HEADER_RE = re.compile(
    r"^System:\s*\[\d{4}-\d{2}-\d{2}[^\]]*\][^\n]*$",
    re.MULTILINE,
)


def strip_envelope_metadata(text: str) -> str:
    """Remove platform envelope metadata that pollutes LLM extraction.

    Drops:
      - `[Subagent Context]` / `[Subagent Task]` wrapper lines (and any
        inline boilerplate that follows)
      - `System: [YYYY-MM-DD ...] Channel[acct] ...` header lines
      - `Conversation info (untrusted metadata):` / similar headers + their
        following JSON code blocks
      - Standalone JSON blocks containing message_id / sender_id fields

    Preserves the trailing zone (the real conversation)."""
    if not text:
        return text

    # 1. Strip the leading zone of subagent wrappers + inline boilerplate
    lines = text.split("\n")
    cleaned: list[str] = []
    still_in_leading_zone = True

    for raw_line in lines:
        line = raw_line
        if still_in_leading_zone:
            wrapper_match = _WRAPPER_LINE_RE.match(line)
            if wrapper_match:
                # Strip the wrapper prefix; scrub inline boilerplate from
                # whatever follows. The trailing remainder, if any, is
                # preserved (post-wrapper inline content can be legitimate).
                remainder = line[wrapper_match.end():]
                remainder = _INLINE_BOILERPLATE_RE.sub("", remainder)
                remainder = re.sub(r"\s{2,}", " ", remainder).strip()
                if remainder:
                    cleaned.append(remainder)
                continue
            if _BOILERPLATE_RE.match(line.strip()):
                continue
            if _SUBAGENT_RUNNING_RE.match(line.strip()):
                continue
            if line.strip():
                # First non-wrapper line ends the leading zone
                still_in_leading_zone = False
        cleaned.append(line)

    out = "\n".join(cleaned)

    # 2. Drop System: timestamp header lines anywhere in the text
    out = _SYSTEM_HEADER_RE.sub("", out)

    # 3. Drop untrusted-metadata blocks: header + following ```json...``` block
    for header in _UNTRUSTED_HEADERS:
        # Match "<header>\n<optional code fence>...<close fence>" non-greedily
        pattern = re.compile(
            re.escape(header) + r"\s*(?:```(?:json)?\n.*?```)?",
            re.DOTALL,
        )
        out = pattern.sub("", out)

    # 4. Drop standalone JSON blocks that mention message_id / sender_id
    out = re.sub(
        r"```(?:json)?\s*\n\s*\{[^`]*?(?:message_id|sender_id)[^`]*?\}\s*\n\s*```",
        "",
        out,
        flags=re.DOTALL,
    )

    # Collapse runs of blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# SmartExtractor
# ---------------------------------------------------------------------------

class SmartExtractor:
    """LLM-driven memory extraction with switchable LLM fallback.

    Constructor:
        store              — MemoryStore (required)
        llm                — ExtractorLLM / LlmClient. **Optional**: when
                             None, `extract_and_persist` falls back to
                             writing raw user/assistant turns as memories
                             (the legacy hermes-agent provider behaviour).
        admission_controller — optional AdmissionController; gates writes
                             before dedup. Pass None to disable admission.
        config             — SmartExtractorConfig (optional)

    The extractor never raises in `extract_and_persist`: errors in any
    individual candidate are logged and the pipeline carries on with the
    rest. The fallback path is invoked even on LLM API failures."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm: LlmClient | None = None,
        admission_controller: AdmissionController | None = None,
        config: SmartExtractorConfig | None = None,
    ):
        self._store = store
        self._llm = llm
        self._admission = admission_controller
        self.config = config or SmartExtractorConfig()

    # -- public properties -----------------------------------------------------

    @property
    def has_llm(self) -> bool:
        """True iff smart extraction is active. False → legacy fallback."""
        return self._llm is not None

    def _llm_complete_json_with_retry(
        self,
        prompt: str,
        *,
        label: str,
    ) -> dict[str, Any] | None:
        """Wrap the LLM's `complete_json` in `run_with_reflection_transient_retry_once`
        so transient upstream blips (502 / connection reset / timeout) get one
        retry with a 1-3s jittered delay. Non-transient errors (auth, quota,
        content_policy) are NOT retried — bouncing those wastes quota.

        Returns None on any unrecoverable failure (matching the contract of
        the underlying `complete_json`)."""
        if self._llm is None:
            return None

        def _call() -> dict[str, Any] | None:
            assert self._llm is not None
            return self._llm.complete_json(prompt, label=label)

        try:
            return run_with_reflection_transient_retry_once(_call)
        except Exception as e:
            logger.warning(
                "smart-extractor: %s LLM call raised after retry: %s", label, e,
            )
            return None

    # -- main entry ------------------------------------------------------------

    def extract_and_persist(
        self,
        conversation_text: str | None = None,
        *,
        session_key: str = "unknown",
        scope: str | None = None,
        scope_filter: list[str] | None = None,
        user_content: str = "",
        assistant_content: str = "",
    ) -> ExtractionStats:
        """Extract memories from a conversation and persist them.

        - When the extractor has no LLM: fall back to writing
          `user_content` and `assistant_content` as separate memories
          (matches the legacy hermes-agent `sync_turn` shape).
        - When the extractor has an LLM: build `conversation_text` from
          `user_content` / `assistant_content` if not supplied, then run
          the full LLM pipeline.
        Returns an `ExtractionStats` describing what happened."""
        target_scope = scope or self.config.default_scope
        effective_filter = scope_filter if scope_filter is not None else [target_scope]

        if self._llm is None:
            return self._legacy_fallback(
                user_content=user_content,
                assistant_content=assistant_content,
                session_key=session_key,
                scope=target_scope,
            )

        if conversation_text is None:
            conversation_text = self._format_conversation(user_content, assistant_content)
        if not conversation_text or not conversation_text.strip():
            return ExtractionStats()

        return self._run_llm_pipeline(
            conversation_text=conversation_text,
            session_key=session_key,
            target_scope=target_scope,
            scope_filter=effective_filter,
        )

    # -- legacy fallback -------------------------------------------------------

    def _legacy_fallback(
        self,
        *,
        user_content: str,
        assistant_content: str,
        session_key: str,
        scope: str,
    ) -> ExtractionStats:
        """Legacy path: store user & assistant turns as separate memories.

        This is exactly what `LanceDBProMemoryProvider.sync_turn` was doing
        before smart extraction landed — kept identical so users without an
        LLM don't see their write shape change."""
        stats = ExtractionStats()
        cfg = self.config
        meta_extra = {
            "source_session": session_key,
            "source": "agent_turn",
        }
        try:
            if user_content and user_content.strip():
                self._store.store(
                    text=user_content.strip(),
                    category=cfg.legacy_category,
                    scope=scope,
                    importance=cfg.legacy_user_importance,
                    metadata_extra={**meta_extra, "role": "user"},
                )
                stats.created += 1
        except Exception as e:
            logger.warning("legacy fallback user write failed: %s", e)

        try:
            if assistant_content and assistant_content.strip():
                self._store.store(
                    text=assistant_content.strip(),
                    category=cfg.legacy_category,
                    scope=scope,
                    importance=cfg.legacy_assistant_importance,
                    metadata_extra={**meta_extra, "role": "assistant"},
                )
                stats.created += 1
        except Exception as e:
            logger.warning("legacy fallback assistant write failed: %s", e)
        return stats

    @staticmethod
    def _format_conversation(user_content: str, assistant_content: str) -> str:
        parts = []
        if user_content and user_content.strip():
            parts.append(f"User: {user_content.strip()}")
        if assistant_content and assistant_content.strip():
            parts.append(f"Assistant: {assistant_content.strip()}")
        return "\n\n".join(parts)

    # -- LLM pipeline ----------------------------------------------------------

    def _run_llm_pipeline(
        self,
        *,
        conversation_text: str,
        session_key: str,
        target_scope: str,
        scope_filter: list[str],
    ) -> ExtractionStats:
        stats = ExtractionStats()
        candidates = self._extract_candidates(conversation_text)
        if not candidates:
            logger.debug("smart-extractor: no candidates extracted")
            return stats

        # Cap and batch-dedup before per-candidate LLM calls
        capped = candidates[:MAX_MEMORIES_PER_EXTRACTION]
        survivors = self._batch_dedup_candidates(capped, stats)

        # Pre-compute vectors for non-profile candidates in a single batch
        precomputed = self._precompute_vectors(survivors)

        max_workers = max(1, int(self.config.dedup_max_workers))
        if max_workers <= 1 or len(survivors) <= 1:
            # Serial path — same behaviour as before.
            for i, candidate in enumerate(survivors):
                try:
                    self._process_candidate(
                        candidate=candidate,
                        conversation_text=conversation_text,
                        session_key=session_key,
                        stats=stats,
                        target_scope=target_scope,
                        scope_filter=scope_filter,
                        precomputed_vector=precomputed.get(i),
                    )
                except Exception as e:
                    logger.warning(
                        "smart-extractor: failed to process candidate [%s] %s: %s",
                        candidate.category, candidate.abstract[:60], e,
                    )
        else:
            # Concurrent path — bounded thread pool. Stats mutation is the
            # only shared mutable state; serialise it under a Lock. LanceDB
            # writes are tolerant of concurrent calls; the LLM endpoint may
            # rate-limit but that's the caller's tune.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from threading import Lock
            stats_lock = Lock()

            # Wrap _process_candidate so it acquires the lock around the
            # `stats` argument (the call mutates stats fields). This keeps
            # the parallel work outside the critical section.
            def _worker(i: int, candidate: CandidateMemory) -> None:
                # Each worker owns a local stats deltas object.
                local = ExtractionStats()
                try:
                    self._process_candidate(
                        candidate=candidate,
                        conversation_text=conversation_text,
                        session_key=session_key,
                        stats=local,
                        target_scope=target_scope,
                        scope_filter=scope_filter,
                        precomputed_vector=precomputed.get(i),
                    )
                except Exception as e:
                    logger.warning(
                        "smart-extractor: failed to process candidate [%s] %s: %s",
                        candidate.category, candidate.abstract[:60], e,
                    )
                with stats_lock:
                    stats.created += local.created
                    stats.merged += local.merged
                    stats.skipped += local.skipped
                    stats.rejected += local.rejected
                    stats.supported += local.supported
                    stats.superseded += local.superseded
                    stats.boundary_skipped += local.boundary_skipped

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_worker, i, c)
                    for i, c in enumerate(survivors)
                ]
                for fut in as_completed(futures):
                    # Re-raise should not happen — _worker swallows; this
                    # is a defensive guard for unexpected pool exceptions.
                    exc = fut.exception()
                    if exc is not None:
                        logger.warning(
                            "smart-extractor: worker raised: %s", exc,
                        )
        return stats

    # -- step 1: LLM extraction ------------------------------------------------

    def _extract_candidates(self, conversation_text: str) -> list[CandidateMemory]:
        max_chars = self.config.extract_max_chars
        truncated = (
            conversation_text[-max_chars:]
            if len(conversation_text) > max_chars
            else conversation_text
        )
        cleaned = strip_envelope_metadata(truncated)

        prompt = build_extraction_prompt(cleaned, self.config.user)
        result = self._llm_complete_json_with_retry(
            prompt, label="extract-candidates",
        )
        if not result:
            return []
        memories = result.get("memories")
        if not isinstance(memories, list):
            return []

        out: list[CandidateMemory] = []
        for raw in memories:
            if not isinstance(raw, dict):
                continue
            category = normalize_category(raw.get("category"))
            if not category:
                continue
            abstract = (raw.get("abstract") or "").strip()
            overview = (raw.get("overview") or "").strip()
            content = (raw.get("content") or "").strip()
            if not abstract or len(abstract) < 5:
                continue
            if is_noise(abstract):
                continue
            out.append(CandidateMemory(
                category=category,
                abstract=abstract,
                overview=overview,
                content=content,
            ))
        return out

    # -- step 1b: batch dedup -------------------------------------------------

    def _batch_dedup_candidates(
        self,
        capped: list[CandidateMemory],
        stats: ExtractionStats,
    ) -> list[CandidateMemory]:
        if len(capped) <= 1:
            return capped
        try:
            abstracts = [c.abstract for c in capped]
            vectors = self._store.encode_batch(abstracts)
            result = batch_dedup(abstracts, vectors)
        except Exception as e:
            logger.warning("smart-extractor: batch dedup failed: %s", e)
            return capped

        if not result.duplicate_indices:
            return capped
        stats.skipped += len(result.duplicate_indices)
        return [capped[i] for i in result.surviving_indices]

    def _precompute_vectors(
        self,
        candidates: list[CandidateMemory],
    ) -> dict[int, list[float]]:
        """Batch-embed non-profile candidates in one call. Profile category
        always-merges via a different code path (handle_profile_merge), so
        we skip it here."""
        non_profile_indices: list[int] = []
        non_profile_texts: list[str] = []
        for i, c in enumerate(candidates):
            if c.category in ALWAYS_MERGE_CATEGORIES:
                continue
            non_profile_indices.append(i)
            non_profile_texts.append(f"{c.abstract} {c.content}")

        if not non_profile_texts:
            return {}
        try:
            vectors = self._store.encode_batch(non_profile_texts)
        except Exception as e:
            logger.warning("smart-extractor: pre-embed batch failed: %s", e)
            return {}
        return {
            non_profile_indices[j]: vectors[j]
            for j in range(len(non_profile_indices))
            if vectors[j]
        }

    # -- step 2: per-candidate dedup + persist --------------------------------

    def _process_candidate(
        self,
        *,
        candidate: CandidateMemory,
        conversation_text: str,
        session_key: str,
        stats: ExtractionStats,
        target_scope: str,
        scope_filter: list[str],
        precomputed_vector: list[float] | None,
    ) -> None:
        # Profile bypass: always merge into the existing profile entry (or
        # create one if none exists)
        if candidate.category in ALWAYS_MERGE_CATEGORIES:
            outcome = self._handle_profile_merge(
                candidate=candidate,
                conversation_text=conversation_text,
                session_key=session_key,
                target_scope=target_scope,
                scope_filter=scope_filter,
            )
            if outcome == "rejected":
                stats.rejected += 1
            elif outcome == "created":
                stats.created += 1
            else:
                stats.merged += 1
            return

        vector = precomputed_vector or self._store.encode(
            f"{candidate.abstract} {candidate.content}"
        )
        if not vector:
            logger.warning("smart-extractor: embed failed; storing as-is")
            self._store_candidate(candidate, vector=[], session_key=session_key,
                                  target_scope=target_scope)
            stats.created += 1
            return

        # Admission control gate
        admission_audit: AdmissionAuditRecord | None = None
        if self._admission is not None:
            candidate.vector = vector  # admission needs a vector for novelty
            admission = self._admission.evaluate(
                candidate, conversation_text, scope_filter=scope_filter,
            )
            if admission.decision == "reject":
                stats.rejected += 1
                return
            admission_audit = admission.audit

        decision = self._deduplicate(candidate, vector, scope_filter)

        if decision.decision == "create":
            self._store_candidate(candidate, vector=vector, session_key=session_key,
                                  target_scope=target_scope, admission_audit=admission_audit)
            stats.created += 1
        elif decision.decision == "merge":
            if decision.match_id and candidate.category in MERGE_SUPPORTED_CATEGORIES:
                self._handle_merge(
                    candidate=candidate, match_id=decision.match_id,
                    target_scope=target_scope, scope_filter=scope_filter,
                    context_label=decision.context_label,
                    admission_audit=admission_audit,
                )
                stats.merged += 1
            else:
                self._store_candidate(candidate, vector=vector, session_key=session_key,
                                      target_scope=target_scope, admission_audit=admission_audit)
                stats.created += 1
        elif decision.decision == "skip":
            stats.skipped += 1
        elif decision.decision == "supersede":
            if decision.match_id and candidate.category in TEMPORAL_VERSIONED_CATEGORIES:
                self._handle_supersede(
                    candidate=candidate, vector=vector,
                    match_id=decision.match_id, session_key=session_key,
                    target_scope=target_scope, scope_filter=scope_filter,
                    admission_audit=admission_audit,
                )
                stats.created += 1
                stats.superseded += 1
            else:
                self._store_candidate(candidate, vector=vector, session_key=session_key,
                                      target_scope=target_scope, admission_audit=admission_audit)
                stats.created += 1
        elif decision.decision == "support":
            if decision.match_id:
                self._handle_support(
                    match_id=decision.match_id, scope_filter=scope_filter,
                    context_label=decision.context_label,
                    admission_audit=admission_audit,
                )
                stats.supported += 1
            else:
                self._store_candidate(candidate, vector=vector, session_key=session_key,
                                      target_scope=target_scope, admission_audit=admission_audit)
                stats.created += 1
        elif decision.decision == "contextualize":
            if decision.match_id:
                self._handle_contextualize_or_contradict(
                    candidate=candidate, vector=vector, match_id=decision.match_id,
                    session_key=session_key, target_scope=target_scope,
                    relation_type="contextualizes",
                    context_label=decision.context_label,
                    admission_audit=admission_audit,
                )
            else:
                self._store_candidate(candidate, vector=vector, session_key=session_key,
                                      target_scope=target_scope, admission_audit=admission_audit)
            stats.created += 1
        elif decision.decision == "contradict":
            if decision.match_id:
                # Temporal-versioned + general-context contradict promotes to supersede
                if (candidate.category in TEMPORAL_VERSIONED_CATEGORIES
                        and decision.context_label == "general"):
                    self._handle_supersede(
                        candidate=candidate, vector=vector,
                        match_id=decision.match_id, session_key=session_key,
                        target_scope=target_scope, scope_filter=scope_filter,
                        admission_audit=admission_audit,
                    )
                    stats.created += 1
                    stats.superseded += 1
                else:
                    self._handle_contextualize_or_contradict(
                        candidate=candidate, vector=vector,
                        match_id=decision.match_id, session_key=session_key,
                        target_scope=target_scope,
                        relation_type="contradicts",
                        context_label=decision.context_label,
                        admission_audit=admission_audit,
                        record_contradiction_on=decision.match_id,
                        scope_filter=scope_filter,
                    )
                    stats.created += 1
            else:
                self._store_candidate(candidate, vector=vector, session_key=session_key,
                                      target_scope=target_scope, admission_audit=admission_audit)
                stats.created += 1

    # -- dedup pipeline --------------------------------------------------------

    def _deduplicate(
        self,
        candidate: CandidateMemory,
        candidate_vector: list[float],
        scope_filter: list[str] | None,
    ) -> DedupResult:
        # Stage 1: vector pre-filter (only one scope at a time supported by
        # the underlying API; use the first if a list)
        scope = scope_filter[0] if scope_filter else None
        try:
            similar = self._store.search_by_vector(
                candidate_vector,
                limit=MAX_SIMILAR_FOR_PROMPT,
                scope=scope,
                keep_vector=False,
            )
        except Exception as e:
            logger.warning("smart-extractor: vector search failed: %s", e)
            return DedupResult(decision="create", reason=f"vector search failed: {e}")

        if not similar:
            return DedupResult(decision="create", reason="No similar memories found")

        # Stage 2: LLM decision
        return self._llm_dedup_decision(candidate, similar)

    def _llm_dedup_decision(
        self,
        candidate: CandidateMemory,
        similar: list[dict[str, Any]],
    ) -> DedupResult:
        # Format the existing memories for the prompt
        existing = []
        for r in similar[:MAX_SIMILAR_FOR_PROMPT]:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            existing.append({
                "text": r.get("text", ""),
                "category": meta.get("memory_category") or r.get("category") or "",
                "abstract": meta.get("l0_abstract") or r.get("text", ""),
                "overview": meta.get("l1_overview") or "",
                "score": 1.0 - float(r.get("_distance") or 0.0),
            })

        prompt = build_dedup_prompt(
            candidate.abstract, candidate.overview, candidate.content, existing,
        )
        data = self._llm_complete_json_with_retry(prompt, label="dedup-decision")
        if not data:
            return DedupResult(decision="create", reason="LLM returned no JSON")

        decision_raw = (data.get("decision") or "").lower()
        if decision_raw not in _VALID_DECISIONS:
            return DedupResult(decision="create", reason=f"unknown decision: {decision_raw}")

        idx = data.get("match_index")
        has_valid_index = isinstance(idx, int) and 1 <= idx <= len(similar)
        match_entry = similar[idx - 1] if has_valid_index else (similar[0] if similar else None)

        # Destructive decisions without a valid match_index → degrade to create
        if decision_raw in _DESTRUCTIVE_DECISIONS and not has_valid_index:
            return DedupResult(
                decision="create",
                reason=f"{decision_raw} degraded: missing match_index",
            )

        match_id = (
            match_entry.get("id") if match_entry and decision_raw != "create" else None
        )
        return DedupResult(
            decision=decision_raw,  # type: ignore[arg-type]
            reason=str(data.get("reason") or ""),
            match_id=match_id,
            context_label=data.get("context_label") if isinstance(data.get("context_label"), str) else None,
        )

    # -- handlers --------------------------------------------------------------

    def _handle_profile_merge(
        self,
        *,
        candidate: CandidateMemory,
        conversation_text: str,
        session_key: str,
        target_scope: str,
        scope_filter: list[str],
    ) -> Literal["merged", "created", "rejected"]:
        vector = self._store.encode(f"{candidate.abstract} {candidate.content}")
        admission_audit: AdmissionAuditRecord | None = None
        if self._admission is not None and vector:
            candidate.vector = vector
            admission = self._admission.evaluate(
                candidate, conversation_text, scope_filter=scope_filter,
            )
            if admission.decision == "reject":
                return "rejected"
            admission_audit = admission.audit

        scope = scope_filter[0] if scope_filter else None
        try:
            similar = self._store.search_by_vector(
                vector or [], limit=1, scope=scope, keep_vector=False,
            )
        except Exception as e:
            logger.warning("smart-extractor: profile search failed: %s", e)
            similar = []

        match = None
        for r in similar:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            if meta.get("memory_category") == "profile":
                match = r
                break

        if match and match.get("id"):
            self._handle_merge(
                candidate=candidate, match_id=match["id"],
                target_scope=target_scope, scope_filter=scope_filter,
                context_label=None, admission_audit=admission_audit,
            )
            return "merged"
        self._store_candidate(
            candidate, vector=vector or [], session_key=session_key,
            target_scope=target_scope, admission_audit=admission_audit,
        )
        return "created"

    def _handle_merge(
        self,
        *,
        candidate: CandidateMemory,
        match_id: str,
        target_scope: str,
        scope_filter: list[str],
        context_label: str | None,
        admission_audit: AdmissionAuditRecord | None,
    ) -> None:
        from .smart_metadata import (
            parse_smart_metadata,
            parse_support_info,
            update_support_stats,
        )

        try:
            existing = self._store.get_by_id(match_id)
        except Exception as e:
            logger.warning("smart-extractor: merge get_by_id failed: %s", e)
            existing = None
        if not existing:
            # Existing entry vanished — fall through to create
            self._store_candidate(
                candidate,
                vector=self._store.encode(f"{candidate.abstract} {candidate.content}") or [],
                session_key="merge-fallback", target_scope=target_scope,
                admission_audit=admission_audit,
            )
            return

        existing_meta = parse_smart_metadata(existing.get("metadata"), existing)
        existing_abstract = existing_meta.l0_abstract or existing.get("text", "")
        existing_overview = existing_meta.l1_overview or ""
        existing_content = existing_meta.l2_content or existing.get("text", "")

        prompt = build_merge_prompt(
            existing_abstract, existing_overview, existing_content,
            candidate.abstract, candidate.overview, candidate.content,
            candidate.category,
        )
        merged = self._llm_complete_json_with_retry(prompt, label="merge-memory")
        if not merged:
            return

        merged_abstract = (merged.get("abstract") or "").strip()
        merged_overview = (merged.get("overview") or "").strip()
        merged_content = (merged.get("content") or "").strip()
        if not merged_abstract:
            return

        # MemoryStore.update(text=...) takes the supersede path and re-encodes
        # the new vector internally; we don't need to pre-embed here.
        merged_meta_extras: dict[str, Any] = {
            "l0_abstract": merged_abstract,
            "l1_overview": merged_overview,
            "l2_content": merged_content,
            "memory_category": candidate.category,
            "merged_from": candidate.abstract,
        }
        if admission_audit and self.config.persist_admission_audit:
            merged_meta_extras["admission_control"] = json.dumps(self._audit_to_json(admission_audit))

        # Update the row text + metadata. Our MemoryStore.update supports
        # text-supersede OR metadata-only path; for merge we want the
        # supersede behaviour (text changed → embed changed).
        # store.update() returns the new row's ID after supersede — capture
        # it so the support-stats update targets the live row, not the
        # archived one.
        try:
            new_id = self._store.update(
                match_id,
                text=merged_abstract,
                metadata_extra=merged_meta_extras,
            )
        except Exception as e:
            logger.warning("smart-extractor: merge update failed: %s", e)
            return

        # Best-effort: update support stats on the merged (live) memory.
        # Use new_id returned by the supersede; match_id now points at the
        # archived predecessor and must not be used here.
        try:
            live_id = new_id or match_id
            updated = self._store.get_by_id(live_id)
            if updated:
                meta = parse_smart_metadata(updated.get("metadata"), updated)
                support = parse_support_info(meta.support_info)
                new_support = update_support_stats(support, context_label, "support")
                meta.support_info = new_support
                self._store.update(
                    live_id,
                    metadata_extra={"support_info": _support_info_to_dict(new_support)},
                )
        except Exception as e:
            logger.debug("smart-extractor: merge support-stats update failed (non-critical): %s", e)

    def _handle_supersede(
        self,
        *,
        candidate: CandidateMemory,
        vector: list[float],
        match_id: str,
        session_key: str,
        target_scope: str,
        scope_filter: list[str],
        admission_audit: AdmissionAuditRecord | None,
    ) -> None:
        # Use MemoryStore's existing supersede path: update with new text
        # archives the old row + creates a new one. We skip our own
        # supersede chain because the store has the canonical implementation.
        try:
            self._store.update(
                match_id,
                text=candidate.abstract,
                category=self._map_to_store_category(candidate.category),
                metadata_extra={
                    "l0_abstract": candidate.abstract,
                    "l1_overview": candidate.overview,
                    "l2_content": candidate.content,
                    "memory_category": candidate.category,
                    "source": "auto-capture",
                    "source_session": session_key,
                    "supersedes": match_id,
                    "memory_temporal_type": classify_temporal(
                        candidate.content or candidate.abstract
                    ),
                    "valid_until": infer_expiry(candidate.content or candidate.abstract),
                },
            )
        except Exception as e:
            logger.warning("smart-extractor: supersede update failed: %s", e)

    def _handle_support(
        self,
        *,
        match_id: str,
        scope_filter: list[str],
        context_label: str | None,
        admission_audit: AdmissionAuditRecord | None,
    ) -> None:
        from .smart_metadata import (
            parse_smart_metadata,
            parse_support_info,
            update_support_stats,
        )

        try:
            existing = self._store.get_by_id(match_id)
            if not existing:
                return
            meta = parse_smart_metadata(existing.get("metadata"), existing)
            support = parse_support_info(meta.support_info)
            new_support = update_support_stats(support, context_label, "support")
            self._store.update(
                match_id,
                metadata_extra={"support_info": _support_info_to_dict(new_support)},
            )
        except Exception as e:
            logger.warning("smart-extractor: support update failed: %s", e)

    def _handle_contextualize_or_contradict(
        self,
        *,
        candidate: CandidateMemory,
        vector: list[float],
        match_id: str,
        session_key: str,
        target_scope: str,
        relation_type: Literal["contextualizes", "contradicts"],
        context_label: str | None,
        admission_audit: AdmissionAuditRecord | None,
        record_contradiction_on: str | None = None,
        scope_filter: list[str] | None = None,
    ) -> None:
        """Shared path for contextualize + contradict — both create a new
        memory linked back to the matched original via metadata.relations.
        Contradict additionally records a contradiction on the original
        memory's support stats."""
        from .smart_metadata import (
            parse_smart_metadata,
            parse_support_info,
            update_support_stats,
        )

        if record_contradiction_on:
            try:
                existing = self._store.get_by_id(record_contradiction_on)
                if existing:
                    meta = parse_smart_metadata(existing.get("metadata"), existing)
                    support = parse_support_info(meta.support_info)
                    updated = update_support_stats(support, context_label, "contradict")
                    self._store.update(
                        record_contradiction_on,
                        metadata_extra={"support_info": _support_info_to_dict(updated)},
                    )
            except Exception as e:
                logger.warning("smart-extractor: contradiction record failed: %s", e)

        # Now write the new entry
        self._store_candidate(
            candidate, vector=vector, session_key=session_key,
            target_scope=target_scope, admission_audit=admission_audit,
            extra_metadata={
                "relations": [{"type": relation_type, "target_id": match_id}],
                "contexts": [context_label] if context_label else [],
            },
        )

    # -- store helpers --------------------------------------------------------

    def _store_candidate(
        self,
        candidate: CandidateMemory,
        *,
        vector: list[float],
        session_key: str,
        target_scope: str,
        admission_audit: AdmissionAuditRecord | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write a candidate as a new memory. Uses store.store() (which
        handles encoding when vector is empty) so we get the standard
        metadata defaults; we then patch in the smart fields via
        metadata_extra."""
        store_category = self._map_to_store_category(candidate.category)
        classify_text = candidate.content or candidate.abstract

        meta_extra: dict[str, Any] = {
            "l0_abstract": candidate.abstract,
            "l1_overview": candidate.overview,
            "l2_content": candidate.content,
            "memory_category": candidate.category,
            "source_session": session_key,
            "source": "auto-capture",
            "memory_temporal_type": classify_temporal(classify_text),
        }
        expiry = infer_expiry(classify_text)
        if expiry is not None:
            meta_extra["valid_until"] = expiry
        if admission_audit and self.config.persist_admission_audit:
            meta_extra["admission_control"] = self._audit_to_json(admission_audit)
        if extra_metadata:
            meta_extra.update(extra_metadata)

        return self._store.store(
            text=candidate.abstract,
            category=store_category,
            scope=target_scope,
            importance=self._default_importance(candidate.category),
            tier="working",
            confidence=0.7,
            metadata_extra=meta_extra,
        )

    @staticmethod
    def _map_to_store_category(category: SmartCategory) -> str:
        """Map the 6-category smart taxonomy to the legacy column values
        in MemoryStore. Smart category lives in metadata.memory_category;
        the column gets its closest legacy match."""
        return {
            "profile": "fact",
            "preferences": "preference",
            "entities": "entity",
            "events": "decision",
            "cases": "fact",
            "patterns": "other",
        }.get(category, "other")

    @staticmethod
    def _default_importance(category: SmartCategory) -> float:
        return {
            "profile": 0.9,
            "preferences": 0.8,
            "entities": 0.7,
            "events": 0.6,
            "cases": 0.8,
            "patterns": 0.85,
        }.get(category, 0.5)

    @staticmethod
    def _audit_to_json(audit: AdmissionAuditRecord) -> dict[str, Any]:
        """Convert an AdmissionAuditRecord dataclass to a JSON-friendly dict."""
        from dataclasses import asdict, is_dataclass
        if is_dataclass(audit):
            return asdict(audit)
        if isinstance(audit, dict):
            return dict(audit)
        return {"raw": str(audit)}


# ---------------------------------------------------------------------------
# Rate limiter (Feature 7 from the TS source)
# ---------------------------------------------------------------------------

@dataclass
class ExtractionRateLimiter:
    """Sliding-window rate limiter for extractor invocations. Mostly
    useful for hosts running with a paid LLM that want a hard cap on
    spend per hour."""
    max_per_hour: int = 30
    _timestamps_ms: list[int] = field(default_factory=list)

    def _prune(self) -> None:
        cutoff = int(time.time() * 1000) - 60 * 60 * 1000
        self._timestamps_ms = [t for t in self._timestamps_ms if t >= cutoff]

    def is_rate_limited(self) -> bool:
        self._prune()
        return len(self._timestamps_ms) >= self.max_per_hour

    def record_extraction(self) -> None:
        self._prune()
        self._timestamps_ms.append(int(time.time() * 1000))

    def get_recent_count(self) -> int:
        self._prune()
        return len(self._timestamps_ms)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _support_info_to_dict(support: Any) -> dict[str, Any]:
    """Convert a SupportInfoV2 dataclass / dict into a JSON-friendly dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(support):
        return asdict(support)
    if isinstance(support, dict):
        return dict(support)
    return {}


__all__ = [
    "DEFAULT_EXTRACT_MAX_CHARS",
    "MAX_MEMORIES_PER_EXTRACTION",
    "MAX_SIMILAR_FOR_PROMPT",
    "SIMILARITY_THRESHOLD",
    "DedupDecision",
    "DedupResult",
    "ExtractionRateLimiter",
    "ExtractionStats",
    "SmartExtractor",
    "SmartExtractorConfig",
    "strip_envelope_metadata",
]
