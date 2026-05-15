"""Hermes Agent MemoryProvider adapter.

Wraps `MemoryStore` + `MemoryRetriever` in the `agent.memory_provider.MemoryProvider`
ABC so hermes-agent can drop this plugin into
`~/.hermes/plugins/memory/lancedb_pro/` and have it be discoverable, with
proper session scoping wired through.

This module imports `agent.memory_provider` lazily — the rest of the package
remains usable as a standalone library, and tests / non-Hermes consumers
don't need hermes-agent installed.

USAGE (in your `~/.hermes/plugins/memory/lancedb_pro/__init__.py`):

    from hermes_memory_lancedb_pro.provider import register

That's all hermes-agent's plugin discovery needs. The provider:

  * passes `session_id` through to `MemoryRetriever.retrieve()` and
    `MemoryStore.store()` — fixing the cross-session memory bleed
    (the "stickiness" symptom)
  * applies a configurable `min_score` floor so unrelated memories
    don't get injected on weak matches
  * batches `sync_turn` writes and increments access counts via the
    throttled `mark_recall_used` API
  * runs `sync_turn` in a daemon thread so hermes-agent is never
    blocked by the write path
  * isolates the database under `hermes_home` when supplied by
    hermes-agent's `initialize()` call
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from ._sql import ARCHIVED_STATE as _ARCHIVED_STATE
from .decay import is_noise as _is_noise
from .memory_compactor import (
    CompactionConfig,
    record_compaction_run,
    run_compaction,
    should_run_compaction,
)
from .retriever import DEFAULT_MIN_RECALL_SCORE, MemoryRetriever
from .store import MemoryStore

logger = logging.getLogger(__name__)

# Defer the heavy import until we know hermes-agent is on PYTHONPATH.
if TYPE_CHECKING:  # pragma: no cover
    pass

PROVIDER_NAME = "lancedb_pro"

# Default recall limit when prefetch fires. The hermes-agent prefetch path
# currently doesn't pass an explicit limit, so we own the default.
DEFAULT_PREFETCH_LIMIT: int = int(os.environ.get("MEMORY_PREFETCH_LIMIT", "5"))

# ---------------------------------------------------------------------------
# Auto-purge configuration
# ---------------------------------------------------------------------------
# Purge cooldown: minimum hours between automatic purge runs.  Set 0 to
# disable auto-purge entirely (you'll need to call purge_archived() manually
# or use `hermes-memory doctor` to see the recommendation).
_AUTO_PURGE_COOLDOWN_HOURS: int = int(
    os.environ.get("MEMORY_AUTO_PURGE_COOLDOWN_HOURS", "24")
)
# Grace period: archived rows younger than this many days are left alone even
# when a purge runs.  30 days gives a comfortable audit window.
_AUTO_PURGE_GRACE_DAYS: int = int(
    os.environ.get("MEMORY_PURGE_GRACE_DAYS", "30")
)
# State-file name — lives alongside the database so it follows the store.
_PURGE_STATE_FILENAME = ".purge-state.json"

# ---------------------------------------------------------------------------
# Session-summary configuration
# ---------------------------------------------------------------------------
# Char budget for the compressed transcript written on session end. Set 0 to
# disable session-summary memory writes entirely.
_SESSION_SUMMARY_MAX_CHARS: int = int(
    os.environ.get("MEMORY_SESSION_SUMMARY_MAX_CHARS", "4000")
)
# Minimum number of messages before a session summary is written. Skips
# trivial one-turn sessions.
_SESSION_SUMMARY_MIN_MESSAGES: int = int(
    os.environ.get("MEMORY_SESSION_SUMMARY_MIN_MESSAGES", "2")
)

# ---------------------------------------------------------------------------
# Auto-compaction configuration
# ---------------------------------------------------------------------------
# Hours between automatic compaction runs. Compaction clusters near-duplicate
# old memories and merges each cluster into one consolidated entry. Defaults
# to weekly; set 0 to disable.
_AUTO_COMPACT_COOLDOWN_HOURS: int = int(
    os.environ.get("MEMORY_AUTO_COMPACT_COOLDOWN_HOURS", "168")
)
_COMPACT_STATE_FILENAME = ".compact-state.json"

# ---------------------------------------------------------------------------
# Reflection configuration
# ---------------------------------------------------------------------------
# Reflection captures durable "invariants" and short-lived "derived" insights
# at session end (requires an LLM) and replays them on recall. Set
# MEMORY_REFLECTION=off to disable both the write and the read path.
_REFLECTION_ENABLED: bool = os.environ.get(
    "MEMORY_REFLECTION", "on"
).strip().lower() not in ("off", "0", "false", "no", "disabled")
# Rows scanned when loading reflection slices for recall.
_REFLECTION_SCAN_LIMIT: int = int(
    os.environ.get("MEMORY_REFLECTION_SCAN_LIMIT", "200")
)
# Agent identity used for reflection ownership. Single-agent setups can leave
# this at the default; multi-agent hosts pass `agent_id` to `initialize()`.
_REFLECTION_AGENT_ID: str = os.environ.get(
    "MEMORY_REFLECTION_AGENT_ID", "main"
).strip() or "main"

# ---------------------------------------------------------------------------
# Admission-control configuration
# ---------------------------------------------------------------------------
# Preset for the AMAC-v1 admission gate wired into the smart extractor:
# `balanced` / `conservative` / `high-recall`, or `off` to disable the gate.
_ADMISSION_PRESET: str = os.environ.get(
    "MEMORY_ADMISSION_PRESET", "balanced"
).strip().lower()

# ---------------------------------------------------------------------------
# Extraction rate-limit configuration
# ---------------------------------------------------------------------------
# Maximum LLM extraction calls per hour. When the cap is hit, sync_turn falls
# back to legacy raw writes for the remainder of the hour. 0 disables the cap.
_EXTRACTION_RATE_LIMIT: int = int(
    os.environ.get("MEMORY_EXTRACTION_RATE_LIMIT", "0")
)


def _extract_message_texts(messages: Any) -> list[str]:
    """Coerce hermes-agent's session-end ``messages`` arg to a flat list of
    text strings. Accepts a list of dicts (``{"content": ...}``) or raw
    strings; silently drops anything else.

    Also handles Anthropic-style content blocks (``content`` is a list of
    ``{"type": "text", "text": "..."}`` dicts), which a tool-using model
    routinely emits — without this branch those turns disappear from the
    session-summary."""
    texts: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("text") or ""
        elif isinstance(msg, str):
            content = msg
        else:
            content = ""
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(p for p in parts if p)
        if isinstance(content, str) and content.strip():
            texts.append(content)
    return texts


def _load_memory_provider_base():
    """Import hermes-agent's MemoryProvider ABC. Returns None if hermes-agent
    isn't on the import path — which is fine for tests / standalone use."""
    try:
        from agent.memory_provider import MemoryProvider
        return MemoryProvider
    except ImportError:
        return None


def _maybe_build_default_smart_extractor(store: MemoryStore) -> Any:
    """Try to build a `SmartExtractor` with an env-detected LLM client.

    Returns None when no LLM is configured (the env-detect helper finds
    nothing) — sync_turn then falls back to legacy raw-turn writes. Any
    exception is swallowed and reported via debug log; the provider must
    NEVER fail to construct just because LLM detection went sideways."""
    try:
        from .llm_client import create_llm_client_from_env
        from .smart_extractor import ExtractionRateLimiter, SmartExtractor
    except ImportError as e:
        logger.debug("lancedb_pro: smart_extractor unavailable: %s", e)
        return None
    try:
        llm = create_llm_client_from_env()
    except Exception as e:
        logger.debug("lancedb_pro: LLM env-detect failed: %s", e)
        return None
    if llm is None:
        return None
    admission = _maybe_build_admission_controller(store, llm)
    rate_limiter = (
        ExtractionRateLimiter(max_per_hour=_EXTRACTION_RATE_LIMIT)
        if _EXTRACTION_RATE_LIMIT > 0
        else None
    )
    try:
        return SmartExtractor(
            store, llm=llm, admission_controller=admission, rate_limiter=rate_limiter,
        )
    except Exception as e:
        logger.debug("lancedb_pro: SmartExtractor construction failed: %s", e)
        return None


def _maybe_build_admission_controller(store: MemoryStore, llm: Any) -> Any:
    """Build an `AdmissionController` from `MEMORY_ADMISSION_PRESET`.

    Returns None when the preset is `off` or construction fails — the
    extractor then runs without an admission gate. An unrecognised preset
    falls back to `balanced` rather than disabling the gate silently."""
    if _ADMISSION_PRESET in ("off", "disabled", "none", ""):
        return None
    preset = (
        _ADMISSION_PRESET
        if _ADMISSION_PRESET in ("balanced", "conservative", "high-recall")
        else "balanced"
    )
    try:
        from .admission_control import AdmissionController, get_preset
        return AdmissionController(store, config=get_preset(preset), llm=llm)
    except Exception as e:
        logger.debug("lancedb_pro: admission controller unavailable: %s", e)
        return None


def _spawn_warmup(store: MemoryStore) -> None:
    """Pre-load the embedding model in a daemon thread.

    First-time users pay a 10-30 s model-load + JIT cost on the first
    `encode()`. Running it here, off the calling thread, means that cost
    lands while the user is composing their first message instead of
    stalling their first turn. Best-effort: failures are logged at debug."""
    def _run() -> None:
        try:
            store.warmup()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("lancedb_pro warmup failed: %s", e)

    threading.Thread(target=_run, daemon=True, name="lancedb-pro-warmup").start()


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce an LLM-returned field to a clean list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _build_reflection_markdown(
    invariants: list[str], derived: list[str]
) -> str:
    """Render invariant / derived lines into the `## Invariants` /
    `## Derived` markdown that the reflection layer's parser expects."""
    parts: list[str] = []
    if invariants:
        parts.append("## Invariants")
        parts.extend(f"- {line}" for line in invariants)
    if derived:
        if parts:
            parts.append("")
        parts.append("## Derived")
        parts.extend(f"- {line}" for line in derived)
    return "\n".join(parts)


def _maybe_auto_compact(store: MemoryStore) -> None:
    """Run cooldown-gated memory compaction.

    Clusters near-duplicate old memories and merges each cluster into one
    consolidated entry. Runs once per `MEMORY_AUTO_COMPACT_COOLDOWN_HOURS`
    (default: weekly). Compaction runs per-scope so a merge never spans
    scopes. Set `MEMORY_AUTO_COMPACT_COOLDOWN_HOURS=0` to disable."""
    if _AUTO_COMPACT_COOLDOWN_HOURS <= 0:
        return

    state_file = os.path.join(store.db_path, _COMPACT_STATE_FILENAME)
    if not should_run_compaction(
        state_file, cooldown_hours=_AUTO_COMPACT_COOLDOWN_HOURS
    ):
        return

    try:
        cfg = CompactionConfig()
        deleted = created = 0
        for scope in ("agent", "user"):
            result = run_compaction(store, cfg, scopes=[scope])
            deleted += result.memories_deleted
            created += result.memories_created
        record_compaction_run(state_file)
        if deleted or created:
            logger.info(
                "Auto-compaction: merged clusters → -%d +%d memories. "
                "Next run in ~%dh.",
                deleted, created, _AUTO_COMPACT_COOLDOWN_HOURS,
            )
        else:
            logger.debug("Auto-compaction: no clusters to merge.")
    except Exception as e:
        logger.warning("Auto-compaction failed (will retry next session): %s", e)


_TOKEN_RE = re.compile(r"[a-z']{2,}")


def _response_references_memory(response_lower: str, memory_text: str) -> bool:
    """Heuristic: did the assistant response reference this memory?

    Looks for any 3-word phrase from the memory in the response. Robust
    to paraphrasing — "user prefers Vim" recalled, response mentions
    "your Vim shortcuts" — the 3-word "your vim shortcuts" wouldn't
    match, but "prefers vim shortcuts" or any 3-word window from the
    memory that the response also contains will hit.

    For very short memories (< 3 tokens) falls back to substring match.
    """
    mem_lower = (memory_text or "").lower().strip()
    if not mem_lower or not response_lower:
        return False
    tokens = _TOKEN_RE.findall(mem_lower)
    if len(tokens) < 3:
        return mem_lower in response_lower
    for i in range(len(tokens) - 2):
        phrase = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
        if phrase in response_lower:
            return True
    # Fallback: a long memory might lose its 3-grams to paraphrasing.
    # Check if the response contains 3+ distinctive (length > 4) tokens
    # from the memory.
    distinctive = {t for t in tokens if len(t) > 4}
    if not distinctive:
        return False
    hits = sum(1 for t in distinctive if t in response_lower)
    return hits >= 3


def _format_recall(results: list[dict[str, Any]]) -> str:
    """Format a list of recall results into the text block hermes-agent
    injects under `<memory-context>`. Returns "" for an empty result so
    hermes-agent's prefetch_all skips injection entirely."""
    if not results:
        return ""
    lines = []
    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        cat = r.get("category") or "other"
        score = next(
            (r[k] for k in ("_final_score", "_rrf_score", "score") if r.get(k) is not None),
            0.0,
        )
        lines.append(f"- [{cat}] {text} (score={score:.2f})")
    return "\n".join(lines) if lines else ""


def _maybe_auto_purge(store: MemoryStore) -> None:
    """Run purge_archived() if the cooldown has elapsed since the last run.

    Called at session end. The check is a fast JSON stat; the purge only
    executes every ``MEMORY_AUTO_PURGE_COOLDOWN_HOURS`` hours (default: 24).

    Set ``MEMORY_AUTO_PURGE_COOLDOWN_HOURS=0`` to disable entirely.
    Adjust the minimum age of rows to delete with ``MEMORY_PURGE_GRACE_DAYS``
    (default: 30 days).
    """
    if _AUTO_PURGE_COOLDOWN_HOURS <= 0:
        return

    state_file = os.path.join(store.db_path, _PURGE_STATE_FILENAME)
    if not should_run_compaction(state_file, cooldown_hours=_AUTO_PURGE_COOLDOWN_HOURS):
        return

    try:
        n = store.purge_archived(grace_period_days=_AUTO_PURGE_GRACE_DAYS)
        record_compaction_run(state_file)
        if n:
            logger.info(
                "Auto-purge: removed %d archived row(s) "
                "(grace_period_days=%d). Next run in ~%dh.",
                n,
                _AUTO_PURGE_GRACE_DAYS,
                _AUTO_PURGE_COOLDOWN_HOURS,
            )
        else:
            logger.debug(
                "Auto-purge: no archived rows older than %d days to remove.",
                _AUTO_PURGE_GRACE_DAYS,
            )
    except Exception as e:
        logger.warning("Auto-purge failed (will retry next session): %s", e)


def _build_provider_class():
    """Construct the LanceDBProMemoryProvider class lazily.

    Done as a factory so importing this module doesn't fail when
    hermes-agent isn't installed (e.g. during unit tests for the
    underlying store)."""
    base = _load_memory_provider_base()

    if base is None:
        # Hermes-agent isn't installed. Provide a stub that raises a
        # clear error if anyone tries to instantiate it, so the user
        # gets an actionable message instead of a confusing ImportError
        # buried in the discovery code.
        class _StubProvider:
            def __init__(self, *args: Any, **kwargs: Any):
                raise ImportError(
                    "hermes-agent is not on PYTHONPATH; "
                    "LanceDBProMemoryProvider needs `agent.memory_provider` "
                    "to be importable. Install hermes-agent or use "
                    "MemoryStore / MemoryRetriever directly."
                )

        return _StubProvider

    class LanceDBProMemoryProvider(base):  # type: ignore[misc, valid-type]
        """LanceDB-backed memory provider for hermes-agent.

        Honours `session_id` on every read and write so memories stay
        scoped to the conversation that created them — modulo
        cross-session memories (core tier or explicit cross_session
        flag) which surface globally."""

        def __init__(
            self,
            store: MemoryStore | None = None,
            retriever: MemoryRetriever | None = None,
            *,
            min_score: float | None = None,
            prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
            smart_extractor: Any = None,
            auto_smart_extraction: bool = True,
        ):
            self._explicit_store = store is not None
            self._store = store or MemoryStore.get_instance()
            self._retriever = retriever or MemoryRetriever(self._store)
            self._min_score = (
                min_score if min_score is not None else DEFAULT_MIN_RECALL_SCORE
            )
            self._prefetch_limit = prefetch_limit
            self._session_id: str = ""
            self._sync_thread: threading.Thread | None = None
            # Protects _sync_thread reference against concurrent sync_turn /
            # on_session_end / shutdown calls from different threads.
            self._thread_lock = threading.Lock()
            # Serializes the join+create+start sequence so two concurrent
            # sync_turn callers cannot each launch their own write thread.
            self._dispatch_lock = threading.Lock()
            # Lock protecting _pending_used_ids — dict is mutated from the
            # calling thread (prefetch/before_prompt_build) and from the
            # sync_turn daemon thread simultaneously.
            self._pending_lock = threading.Lock()
            # Cache last-prefetched ids per session so we can mark them
            # "used" on the next sync_turn (i.e. only when we actually
            # forwarded the recall to the LLM and got a response back).
            self._pending_used_ids: dict[str, list[str]] = {}
            # Smart extractor — optional. If the caller doesn't supply one,
            # auto_smart_extraction tries to construct one from env vars
            # (`MEMORY_EXTRACTION_*` overrides, then `OPENAI_API_KEY` /
            # `ANTHROPIC_API_KEY`). When neither resolves, sync_turn falls
            # back to writing raw user/assistant turns — the same shape this
            # provider always wrote, so existing stores don't migrate.
            self._auto_smart_extraction = auto_smart_extraction
            self._smart_extractor = smart_extractor
            if smart_extractor is None and auto_smart_extraction:
                self._smart_extractor = _maybe_build_default_smart_extractor(self._store)
            # Embedding-model warmup runs once, off the first turn's path.
            self._warmed_up = False
            # Agent identity for reflection ownership; may be overridden by
            # hermes-agent via `initialize(agent_id=...)`.
            self._agent_id = _REFLECTION_AGENT_ID
            # Reflection recall block cached per session — reflection rows
            # only change at session end, so the set is stable mid-session.
            self._reflection_lock = threading.Lock()
            self._reflection_cache: dict[str, str] = {}

        # ---- ABC requirements --------------------------------------------

        @property
        def name(self) -> str:
            return PROVIDER_NAME

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **kwargs: Any) -> None:
            """Called by hermes-agent before the first turn of each session.

            Stores the session ID and re-points the store at the profile-
            isolated ``hermes_home`` directory when hermes-agent supplies it.
            Passing ``hermes_home`` keeps each Hermes profile's memories in
            a separate database tree (e.g. ``~/.hermes/memory-lancedb``)
            rather than the process-wide default path."""
            self._session_id = session_id
            agent_id = kwargs.get("agent_id")
            if agent_id and str(agent_id).strip():
                self._agent_id = str(agent_id).strip()
            hermes_home = kwargs.get("hermes_home")
            if hermes_home and not self._explicit_store:
                db_path = os.path.join(str(hermes_home), "memory-lancedb")
                self._store = MemoryStore.get_instance(db_path=db_path)
                self._retriever = MemoryRetriever(self._store)
                if self._auto_smart_extraction:
                    self._smart_extractor = _maybe_build_default_smart_extractor(
                        self._store
                    )
            elif self._explicit_store:
                # get_instance() calls _initialise() internally, but an
                # explicitly-supplied store may not have been opened yet.
                self._store._initialise()

            # Warm the embedding model once, in the background, so the
            # cold-start cost never lands on the user's first turn.
            if not self._warmed_up:
                self._warmed_up = True
                _spawn_warmup(self._store)

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return []  # context-only provider; no tool calls

        def handle_tool_call(self, name: str, args: dict[str, Any]) -> Any:
            return None  # no tools registered; should never be called

        def get_config_schema(self) -> list[dict[str, Any]]:
            """Declare configuration for `hermes memory setup`.

            Kept minimal per spec guidance — only fields the user must
            configure are prompted here. Advanced tuning knobs
            (MEMORY_PREFETCH_LIMIT, MEMORY_ADMISSION_PRESET, etc.) are
            documented in the README and set via environment variables
            directly.
            """
            return [
                {
                    "key": "extraction_api_key",
                    "env_var": "MEMORY_EXTRACTION_API_KEY",
                    "description": (
                        "API key for LLM-driven memory extraction (optional). "
                        "Without this, the provider stores raw turns; with it, "
                        "a 6-category smart extractor runs on every turn. "
                        "Accepts OpenAI-compatible keys or ANTHROPIC_API_KEY."
                    ),
                    "secret": True,
                    "required": False,
                },
                {
                    "key": "extraction_base_url",
                    "env_var": "MEMORY_EXTRACTION_BASE_URL",
                    "description": (
                        "Base URL for a custom or self-hosted LLM extraction "
                        "endpoint (optional, e.g. http://localhost:11434/v1). "
                        "Leave blank to use the default OpenAI / Anthropic endpoint."
                    ),
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "extraction_model",
                    "env_var": "MEMORY_EXTRACTION_MODEL",
                    "description": (
                        "Model name for LLM extraction, e.g. gpt-4o-mini or "
                        "claude-haiku-4-5-20251001 (optional). Defaults to the "
                        "provider's own default when blank."
                    ),
                    "secret": False,
                    "required": False,
                },
            ]

        def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
            """Persist setup values to ``<hermes_home>/.env``.

            Reads the existing file, replaces lines for any env vars
            being updated, then appends new ones. Empty/None values are
            skipped — the user can clear them by editing the file directly."""
            if not values or not hermes_home:
                return
            schema = {entry["key"]: entry["env_var"] for entry in self.get_config_schema()}
            to_write = {
                schema[key]: str(val)
                for key, val in values.items()
                if key in schema and val is not None and str(val).strip()
            }
            if not to_write:
                return
            env_path = os.path.join(hermes_home, ".env")
            existing: list[str] = []
            if os.path.exists(env_path):
                with open(env_path, encoding="utf-8") as fh:
                    existing = fh.readlines()
            # Drop lines we're overwriting, preserve everything else.
            kept = [
                line for line in existing
                if not any(line.startswith(f"{var}=") for var in to_write)
            ]
            for env_var, value in to_write.items():
                kept.append(f"{env_var}={value}\n")
            os.makedirs(hermes_home, exist_ok=True)
            with open(env_path, "w", encoding="utf-8") as fh:
                fh.writelines(kept)

        # ---- Read path ----------------------------------------------------

        def _do_recall(self, query: str, session_id: str) -> str:
            """Shared implementation for both `prefetch` and
            `before_prompt_build`. Runs a session-scoped recall, caches
            the returned ids in `_pending_used_ids[session_id]` so we
            can credit them later, prepends the reflection block, and
            returns the combined text.

            Relevance-based recall can miss earlier task framing when the
            current query is semantically distant (e.g. "check slot 7"
            doesn't match "stress test my memory"). To keep context
            continuity, the two most-recently-written session memories are
            injected as anchors, deduplicated against the relevance results."""
            if not query or not query.strip():
                return ""
            try:
                results = self._retriever.retrieve(
                    query,
                    limit=self._prefetch_limit,
                    session_id=session_id or None,
                    min_score=self._min_score,
                    source="auto-recall",
                )
            except Exception as e:
                logger.warning("lancedb_pro recall failed: %s", e)
                results = []

            # Session anchors — always append the 2 oldest (task framing) and
            # 2 most-recently-written session memories so context continuity
            # holds regardless of how many turns have passed.  Without the
            # "first" anchors, task framing from turn 1 falls out of the
            # recency window after turn 3 and is only recoverable by relevance
            # search, which fails when the current query is semantically
            # distant (e.g. "check slot 7" vs "stress test my memory").
            if session_id:
                try:
                    existing_ids = {r.get("id") for r in results}
                    first_anchors = self._store.first_for_session(session_id, limit=2)
                    recent_anchors = self._store.recent_for_session(session_id, limit=2)
                    seen: set[str | None] = set(existing_ids)
                    extra_anchors = []
                    for m in first_anchors + recent_anchors:
                        mid = m.get("id")
                        if mid not in seen:
                            extra_anchors.append(m)
                            seen.add(mid)
                    results = results + extra_anchors
                except Exception as e:
                    logger.debug("lancedb_pro session anchor lookup failed: %s", e)

            if results and session_id:
                with self._pending_lock:
                    self._pending_used_ids[session_id] = [
                        r["id"] for r in results if r.get("id")
                    ]

            recall_block = _format_recall(results)
            reflection_block = self._reflection_block(session_id)
            if reflection_block and recall_block:
                return f"{reflection_block}\n{recall_block}"
            return reflection_block or recall_block

        def _reflection_block(self, session_id: str) -> str:
            """Return the formatted reflection-recall block for this
            session. Computed once and cached for the session's lifetime
            — reflection rows are only written at session end, so the set
            is stable mid-session."""
            if not _REFLECTION_ENABLED:
                return ""
            cache_key = session_id or "_global"
            with self._reflection_lock:
                cached = self._reflection_cache.get(cache_key)
            if cached is not None:
                return cached
            block = self._compute_reflection_block()
            with self._reflection_lock:
                self._reflection_cache[cache_key] = block
            return block

        def _compute_reflection_block(self) -> str:
            """Load and rank reflection slices, format them as recall
            lines. Best-effort: any failure yields an empty block."""
            try:
                from .reflection import load_agent_reflection_slices_from_entries
                entries = self._store.list_memories(
                    limit=_REFLECTION_SCAN_LIMIT, category="reflection"
                )
                slices = load_agent_reflection_slices_from_entries(
                    entries=entries, agent_id=self._agent_id,
                )
            except Exception as e:
                logger.debug("lancedb_pro reflection load failed: %s", e)
                return ""
            lines = [f"- [reflection/invariant] {s}" for s in slices.invariants]
            lines += [f"- [reflection/derived] {s}" for s in slices.derived]
            return "\n".join(lines)

        def prefetch(self, query: str, session_id: str | None = None) -> str:
            """User-message memory injection (legacy hermes-agent path).

            Returns the formatted recall block. On hermes-agent versions
            that support `before_prompt_build`, this method is NOT
            called — the host detects our override and skips prefetch
            to avoid double-injection. On older hermes-agent, this is
            the only injection point."""
            self._flush_pending_write()
            return self._do_recall(query, session_id or self._session_id)

        def before_prompt_build(self, turn_state: dict[str, Any]) -> str:
            """System-prompt memory injection (new hermes-agent path).

            On hosts that support the hook (introduced via the
            corresponding hermes-agent PR), this places the recall
            block in the system prompt — a more authoritative position
            than the user message. The host calls this instead of
            `prefetch` for providers that override it; we override it,
            so on a new host we'll always go through here.

            Older hosts never call this method, so it's dormant for
            users who haven't picked up the hermes-agent change. The
            plugin keeps both methods so the SAME wheel works against
            both old and new hermes-agent."""
            self._flush_pending_write()
            query = str(turn_state.get("query") or "")
            session_id = str(turn_state.get("session_id") or "") or self._session_id
            return self._do_recall(query, session_id)

        # ---- Write path ---------------------------------------------------

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
        ) -> None:
            """Persist a completed turn in a daemon thread (non-blocking).

            hermes-agent must not be blocked by the write path; all I/O
            happens in a background daemon thread. We join any still-running
            previous thread first (with a 5-second cap) so writes remain
            ordered per session.

            When a `smart_extractor` is configured, sync_turn delegates the
            write to it (LLM-driven 6-category extraction). Otherwise we
            fall back to writing raw user / assistant turns — same shape
            this provider has always used."""
            effective_session_id = session_id or self._session_id
            # Capture store and extractor at dispatch time so a concurrent
            # initialize() call cannot swap them out mid-write and redirect
            # this turn's data to a different session's database.
            _extractor = self._smart_extractor
            _store = self._store

            def _do() -> None:
                if _extractor is not None:
                    try:
                        _extractor.extract_and_persist(
                            user_content=user_content,
                            assistant_content=assistant_content,
                            session_key=effective_session_id,
                            scope="agent",
                        )
                    except Exception as e:
                        # The extractor's own pipeline catches per-candidate
                        # errors; if the orchestrator itself blows up, fall
                        # back to legacy raw writes so the turn still lands.
                        logger.warning(
                            "lancedb_pro smart_extractor sync_turn failed; "
                            "falling back to raw writes: %s", e,
                        )
                        self._raw_sync_turn(
                            user_content, assistant_content, effective_session_id,
                            _store_override=_store,
                        )
                else:
                    self._raw_sync_turn(
                        user_content, assistant_content, effective_session_id,
                        _store_override=_store,
                    )

                # Credit the memories the model saw in its prefetch — bypasses
                # the per-recall throttle because we now know they were actually
                # injected into a turn.
                with self._pending_lock:
                    used = (
                        self._pending_used_ids.pop(effective_session_id, None)
                        if effective_session_id
                        else None
                    )
                if used:
                    try:
                        _store.mark_recall_used(used, session_id=effective_session_id)
                    except Exception as e:
                        logger.warning("lancedb_pro mark_recall_used failed: %s", e)

            with self._dispatch_lock:
                with self._thread_lock:
                    prev = self._sync_thread
                if prev and prev.is_alive():
                    prev.join(timeout=5.0)
                new_thread = threading.Thread(target=_do, daemon=True)
                with self._thread_lock:
                    self._sync_thread = new_thread
                new_thread.start()

        def _flush_pending_write(self, timeout: float = 2.0) -> None:
            """Wait briefly for the previous sync_turn write thread to finish.

            Called at the top of prefetch / before_prompt_build so that
            the previous turn's memories are visible to the upcoming recall.
            Without this, a slow embedding (e.g. first-ever model load on a
            brand-new install) causes the read to race the write and return
            empty results for the first several turns."""
            with self._thread_lock:
                thread = self._sync_thread
            if thread and thread.is_alive():
                thread.join(timeout=timeout)

        def _raw_sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            session_id: str,
            *,
            _store_override: MemoryStore | None = None,
        ) -> None:
            """Legacy raw-turn write path. Used when no smart_extractor is
            configured, or as a fail-safe if the extractor orchestrator
            itself raises (per-candidate failures don't reach here).

            Only user-side content is stored, and only after passing the noise
            filter. Assistant responses are deliberately excluded: they are
            verbose, agent-side text that creates a feedback loop when recalled
            (e.g. an early greeting gets injected back later, causing the agent
            to re-greet). The smart_extractor path handles both sides properly
            by extracting facts rather than storing raw turns.

            ``_store_override`` lets the sync_turn daemon thread pass the
            store it captured at dispatch time, preventing a concurrent
            initialize() from redirecting writes to the wrong database."""
            store = _store_override or self._store
            metadata_extra = (
                {"source_session": session_id, "source": "agent_turn"}
                if session_id else {"source": "agent_turn"}
            )
            try:
                text = (user_content or "").strip()
                if text and not _is_noise(text):
                    store.store(
                        text=text,
                        category="other",
                        scope="agent",
                        importance=0.4,
                        metadata_extra={**metadata_extra, "role": "user"},
                    )
            except Exception as e:
                logger.warning("lancedb_pro sync_turn user write failed: %s", e)

        # ---- Lifecycle ----------------------------------------------------

        def on_session_switch(
            self,
            new_session_id: str,
            *,
            parent_session_id: str = "",
            reset: bool = False,
            **_kwargs: Any,
        ) -> None:
            self._session_id = new_session_id
            # Drop any pending used-ids for the old session — we're not
            # going to credit recalls that were never confirmed.
            if parent_session_id:
                with self._pending_lock:
                    self._pending_used_ids.pop(parent_session_id, None)
                with self._reflection_lock:
                    self._reflection_cache.pop(parent_session_id, None)

        def on_recall_used(
            self,
            response_text: str,
            *,
            session_id: str = "",
        ) -> None:
            """Credit memories the response actually referenced.

            On hermes-agent hosts that support this hook, fires once per
            turn with the full assistant response. We do a phrase-overlap
            match between each prefetched memory and the response and
            credit only the matches — far more precise than the legacy
            "credit everything we prefetched" approach.

            When this hook fires, we consume the per-session
            `_pending_used_ids` ledger so `sync_turn`'s legacy
            timing-based crediting becomes a no-op (no double-credit)."""
            effective_session_id = session_id or self._session_id
            with self._pending_lock:
                ids = (
                    self._pending_used_ids.pop(effective_session_id, None)
                    if effective_session_id
                    else None
                )
            if not ids:
                return

            response_lower = (response_text or "").lower()
            if not response_lower.strip():
                return

            used: list[str] = []
            for mem_id in ids:
                try:
                    row = self._store.get_by_id(mem_id)
                except Exception:
                    continue
                if not row:
                    continue
                if _response_references_memory(response_lower, row.get("text") or ""):
                    used.append(mem_id)

            if used:
                try:
                    self._store.mark_recall_used(used, session_id=effective_session_id)
                except Exception as e:
                    logger.warning(
                        "lancedb_pro mark_recall_used (on_recall_used) failed: %s", e,
                    )

        def on_tool_call_observed(
            self,
            tool_name: str,
            args: dict[str, Any],
            result: Any,
            *,
            session_id: str = "",
            success: bool = True,
        ) -> None:
            """Hook for observing every tool call. Currently a no-op
            stub — placeholder for future entity-extraction logic
            ('agent kept calling read_file on /foo' → high-utility
            entity). Fires for both successful and failed tool calls."""
            # Intentionally minimal. The hook is wired so future
            # versions of the plugin can extract entities here without
            # requiring another hermes-agent change.
            return

        def on_memory_write(
            self,
            action: str,
            target: str,
            content: str,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            """Mirror writes from the built-in memory tool into our store
            so hermes-agent's `/memory` commands and our recall stay in
            sync.

            ``add``: stores ``content`` with provenance from ``target``
            (namespace: "user" → preference/user scope, else other/agent).

            ``edit``: BM25-searches for memories matching ``target`` (the
            old text), then supersedes each match with ``content`` (the new
            text).  Pass ``metadata={"replace_all": True}`` to update every
            matching entry; without it only the single best match is updated.

            ``delete``: BM25-searches for memories matching ``target`` (or
            ``content`` when target is a namespace keyword) and soft-archives
            each match.  ``replace_all`` applies here too."""
            if action not in ("add", "edit", "delete"):
                return

            if action in ("edit", "delete"):
                replace_all = bool((metadata or {}).get("replace_all", False))
                # target carries the old text for edit/delete; content may
                # carry it too when target is a namespace keyword.
                query = (
                    target
                    if target and target not in ("user", "agent")
                    else content
                )
                if not query or not query.strip():
                    logger.debug(
                        "lancedb_pro on_memory_write %r: empty query — skip", action
                    )
                    return
                try:
                    candidates = self._store.search(
                        query.strip(), mode="bm25", limit=20
                    )
                except Exception as e:
                    logger.warning(
                        "lancedb_pro on_memory_write %r search failed: %s", action, e
                    )
                    return
                query_lower = query.strip().lower()
                exact = [
                    c for c in candidates
                    if query_lower in c.get("text", "").lower()
                ]
                matches = exact if exact else (candidates[:1] if candidates else [])
                if not matches:
                    logger.debug(
                        "lancedb_pro on_memory_write %r: no match for %r — skip",
                        action, query,
                    )
                    return
                if len(matches) > 1 and not replace_all:
                    matches = matches[:1]
                    logger.debug(
                        "lancedb_pro on_memory_write %r: %d candidates, using top "
                        "(pass replace_all=True to update all)",
                        action, len(exact) or len(candidates),
                    )
                if action == "edit":
                    new_text = content.strip()
                    if not new_text:
                        return
                    for m in matches:
                        try:
                            self._store.update(m["id"], text=new_text)
                        except Exception as e:
                            logger.warning(
                                "lancedb_pro on_memory_write edit id=%s: %s",
                                m.get("id"), e,
                            )
                else:  # delete
                    now_ms = int(time.time() * 1000)
                    for m in matches:
                        try:
                            self._store.update(
                                m["id"],
                                metadata_extra={
                                    "state": _ARCHIVED_STATE,
                                    "invalidated_at": now_ms,
                                },
                            )
                        except Exception as e:
                            logger.warning(
                                "lancedb_pro on_memory_write delete id=%s: %s",
                                m.get("id"), e,
                            )
                return

            if not content.strip():
                return
            sess = (metadata or {}).get("session_id") or ""
            extra = {"source": f"hermes_{target}"}
            if sess:
                extra["source_session"] = sess
            if metadata:
                # Pass through any provenance the agent supplied
                extra.update(
                    {k: v for k, v in metadata.items() if k not in ("session_id", "replace_all")}
                )
            try:
                self._store.store(
                    text=content.strip(),
                    category="preference" if target == "user" else "other",
                    scope="user" if target == "user" else "agent",
                    importance=0.6,
                    # Built-in memory writes are user-curated and should
                    # surface across sessions.
                    metadata_extra={**extra, "cross_session": True},
                )
            except Exception as e:
                logger.warning("lancedb_pro on_memory_write failed: %s", e)

        def on_session_end(self, messages: list) -> None:
            """Called by hermes-agent at conversation end (not process exit).

            Joins any pending sync_turn thread so writes complete first,
            writes a session-summary memory from the conversation history,
            flushes the pending-recall ledger, then triggers the
            cooldown-gated auto-purge.

            Holds `_dispatch_lock` for the whole barrier so a concurrent
            `sync_turn` cannot launch a new write thread between our join
            and the summary write."""
            with self._dispatch_lock:
                with self._thread_lock:
                    thread = self._sync_thread
                if thread and thread.is_alive():
                    thread.join(timeout=10.0)

                try:
                    self._write_session_summary(messages)
                except Exception as e:
                    logger.warning("lancedb_pro session-summary write failed: %s", e)

                try:
                    self._maybe_write_reflection(_extract_message_texts(messages))
                except Exception as e:
                    logger.warning("lancedb_pro reflection write failed: %s", e)

                with self._pending_lock:
                    self._pending_used_ids.clear()
            with self._reflection_lock:
                self._reflection_cache.clear()
            _maybe_auto_purge(self._store)
            _maybe_auto_compact(self._store)

        def _maybe_write_reflection(self, texts: list[str]) -> None:
            """Generate a session reflection via the extractor's LLM and
            persist it through the reflection layer.

            No-op when reflection is disabled, no LLM is configured (the
            reflection summary needs one to be generated), or the
            transcript is empty. Best-effort throughout — the caller
            already wraps this in a try/except."""
            if not _REFLECTION_ENABLED:
                return
            extractor = self._smart_extractor
            if extractor is None or not getattr(extractor, "has_llm", False):
                return
            llm = getattr(extractor, "llm", None)
            if llm is None:
                return
            conversation = "\n".join(texts).strip()
            if not conversation:
                return

            from .extraction_prompts import build_reflection_prompt
            result = llm.complete_json(
                build_reflection_prompt(conversation), label="reflection",
            )
            if not isinstance(result, dict):
                return
            invariants = _coerce_str_list(result.get("invariants"))
            derived = _coerce_str_list(result.get("derived"))
            if not invariants and not derived:
                return

            from .reflection import (
                MemoryStoreReflectionAdapter,
                store_reflection_to_lancedb,
            )
            store_reflection_to_lancedb(
                MemoryStoreReflectionAdapter(self._store),
                reflection_text=_build_reflection_markdown(invariants, derived),
                session_key=self._session_id or "unknown",
                session_id=self._session_id or "unknown",
                agent_id=self._agent_id,
                command="session-end",
                scope="agent",
                run_at=int(time.time() * 1000),
            )

        def _write_session_summary(self, messages: Any) -> None:
            """Compress the session transcript and write it as a single
            ``metadata_type=session-summary`` memory.

            Honours ``MEMORY_SESSION_SUMMARY_MAX_CHARS`` (0 disables) and
            ``MEMORY_SESSION_SUMMARY_MIN_MESSAGES``. Decay's ``evaluate_tier``
            already exempts ``session-summary`` rows from tier mutation so
            the summary persists at its initial tier."""
            if _SESSION_SUMMARY_MAX_CHARS <= 0:
                return
            if not self._session_id:
                return
            texts = _extract_message_texts(messages)
            if len(texts) < _SESSION_SUMMARY_MIN_MESSAGES:
                return
            from .session_compressor import compress_texts
            result = compress_texts(texts, max_chars=_SESSION_SUMMARY_MAX_CHARS)
            if not result.texts:
                return
            summary = "\n".join(result.texts)
            # compress_texts honours max_chars softly: a single boundary
            # message larger than the budget is preserved intact. Cap the
            # stored summary at 2x the budget so a degenerate session
            # can't write an unbounded blob.
            hard_cap = _SESSION_SUMMARY_MAX_CHARS * 2
            if len(summary) > hard_cap:
                summary = summary[:hard_cap] + "\n[...truncated]"
            self._store.store(
                text=summary,
                category="other",
                scope="agent",
                importance=0.5,
                metadata_extra={
                    "metadata_type": "session-summary",
                    "source": "session_end",
                    "source_session": self._session_id,
                    "summary_message_count": len(texts),
                    "summary_kept_count": len(result.texts),
                    "summary_dropped_count": result.dropped,
                    "cross_session": False,
                },
            )

        def shutdown(self) -> None:
            """Called by hermes-agent at process exit."""
            with self._thread_lock:
                thread = self._sync_thread
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            with self._pending_lock:
                self._pending_used_ids.clear()
            with self._reflection_lock:
                self._reflection_cache.clear()
            _maybe_auto_purge(self._store)
            _maybe_auto_compact(self._store)

    return LanceDBProMemoryProvider


# Build the class once at import time; it's either real or a stub.
LanceDBProMemoryProvider = _build_provider_class()


def register(ctx: Any) -> None:
    """Plugin entry point per the Hermes memory-provider plugin spec.

    Called by hermes-agent's plugin discovery when it loads
    ``~/.hermes/plugins/lancedb_pro/``. Registers a configured
    LanceDBProMemoryProvider with the host context.

    A `~/.hermes/plugins/lancedb_pro/__init__.py` shim needs only:

        from hermes_memory_lancedb_pro.provider import register

        __all__ = ["register"]
    """
    base = _load_memory_provider_base()
    if base is None:
        raise ImportError(
            "hermes-agent is not on PYTHONPATH; "
            "register() can only be called from inside hermes-agent."
        )
    ctx.register_memory_provider(LanceDBProMemoryProvider())


def register_memory_provider(_ctx: Any = None) -> Any:
    """Backwards-compatible alias; prefer ``register(ctx)`` for new installs.

    Returns a configured LanceDBProMemoryProvider for callers that use
    the old return-value convention instead of the ``ctx.register_*``
    pattern."""
    base = _load_memory_provider_base()
    if base is None:
        raise ImportError(
            "hermes-agent is not on PYTHONPATH; "
            "register_memory_provider() can only be called from inside hermes-agent."
        )
    return LanceDBProMemoryProvider()


__all__ = [
    "LanceDBProMemoryProvider",
    "PROVIDER_NAME",
    "register",
    "register_memory_provider",
]


def _self_check() -> str:  # pragma: no cover — exercised by smoke test
    """Cheap smoke for "is the provider class wired?" — used by tests."""
    return "stub" if _load_memory_provider_base() is None else "real"
