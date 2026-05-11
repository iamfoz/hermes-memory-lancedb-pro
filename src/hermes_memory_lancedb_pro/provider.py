"""Hermes Agent MemoryProvider adapter.

Wraps `MemoryStore` + `MemoryRetriever` in the `agent.memory_provider.MemoryProvider`
ABC so hermes-agent can drop this plugin into `~/.hermes/plugins/lancedb_pro/`
and have it be discoverable, with proper session scoping wired through.

This module imports `agent.memory_provider` lazily — the rest of the package
remains usable as a standalone library, and tests / non-Hermes consumers
don't need hermes-agent installed.

USAGE (in your `~/.hermes/plugins/lancedb_pro/__init__.py`):

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
from typing import TYPE_CHECKING, Any

from .memory_compactor import record_compaction_run, should_run_compaction
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


def _extract_message_texts(messages: Any) -> list[str]:
    """Coerce hermes-agent's session-end ``messages`` arg to a flat list of
    text strings. Accepts a list of dicts (``{"content": ...}``) or raw
    strings; silently drops anything else."""
    texts: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("text") or ""
        elif isinstance(msg, str):
            content = msg
        else:
            content = ""
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
        from .smart_extractor import SmartExtractor
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
    try:
        return SmartExtractor(store, llm=llm)
    except Exception as e:
        logger.debug("lancedb_pro: SmartExtractor construction failed: %s", e)
        return None


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
        score = r.get("_final_score") or r.get("_rrf_score") or r.get("score") or 0.0
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

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return []  # context-only provider; no tool calls

        def get_config_schema(self) -> list[dict[str, Any]]:
            """Declare env-var configuration for `hermes memory setup`."""
            return [
                {
                    "key": "extraction_api_key",
                    "env_var": "MEMORY_EXTRACTION_API_KEY",
                    "description": "API key for LLM-driven memory extraction (optional)",
                    "secret": True,
                    "required": False,
                },
                {
                    "key": "extraction_base_url",
                    "env_var": "MEMORY_EXTRACTION_BASE_URL",
                    "description": "Base URL for LLM extraction endpoint (optional)",
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "extraction_model",
                    "env_var": "MEMORY_EXTRACTION_MODEL",
                    "description": "Model name for LLM extraction (optional)",
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "prefetch_limit",
                    "env_var": "MEMORY_PREFETCH_LIMIT",
                    "description": "Max memories injected per turn (default: 5)",
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "auto_purge_cooldown_hours",
                    "env_var": "MEMORY_AUTO_PURGE_COOLDOWN_HOURS",
                    "description": "Hours between automatic archive purges (default: 24; 0 = off)",
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "purge_grace_days",
                    "env_var": "MEMORY_PURGE_GRACE_DAYS",
                    "description": "Min age in days before archived rows are deleted (default: 30)",
                    "secret": False,
                    "required": False,
                },
                {
                    "key": "session_summary_max_chars",
                    "env_var": "MEMORY_SESSION_SUMMARY_MAX_CHARS",
                    "description": "Char budget for session-summary memories (default: 4000; 0 = off)",
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
            can credit them later, and returns the formatted block."""
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
                return ""

            if results and session_id:
                with self._pending_lock:
                    self._pending_used_ids[session_id] = [
                        r["id"] for r in results if r.get("id")
                    ]
            return _format_recall(results)

        def prefetch(self, query: str) -> str:
            """User-message memory injection (legacy hermes-agent path).

            Returns the formatted recall block. On hermes-agent versions
            that support `before_prompt_build`, this method is NOT
            called — the host detects our override and skips prefetch
            to avoid double-injection. On older hermes-agent, this is
            the only injection point."""
            return self._do_recall(query, self._session_id)

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

            with self._thread_lock:
                prev = self._sync_thread
            if prev and prev.is_alive():
                prev.join(timeout=5.0)
            new_thread = threading.Thread(target=_do, daemon=True)
            with self._thread_lock:
                self._sync_thread = new_thread
            new_thread.start()

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

            ``_store_override`` lets the sync_turn daemon thread pass the
            store it captured at dispatch time, preventing a concurrent
            initialize() from redirecting writes to the wrong database."""
            store = _store_override or self._store
            metadata_extra = (
                {"source_session": session_id, "source": "agent_turn"}
                if session_id else {"source": "agent_turn"}
            )
            try:
                if user_content and user_content.strip():
                    store.store(
                        text=user_content.strip(),
                        category="other",
                        scope="agent",
                        importance=0.4,
                        metadata_extra={**metadata_extra, "role": "user"},
                    )
            except Exception as e:
                logger.warning("lancedb_pro sync_turn user write failed: %s", e)

            try:
                if assistant_content and assistant_content.strip():
                    store.store(
                        text=assistant_content.strip(),
                        category="other",
                        scope="agent",
                        importance=0.4,
                        metadata_extra={**metadata_extra, "role": "assistant"},
                    )
            except Exception as e:
                logger.warning("lancedb_pro sync_turn assistant write failed: %s", e)

        # ---- Lifecycle ----------------------------------------------------

        def on_session_switch(
            self,
            new_session_id: str,
            *,
            parent_session_id: str = "",
            reset: bool = False,
            **_kwargs: Any,
        ) -> None:
            # Drop any pending used-ids for the old session — we're not
            # going to credit recalls that were never confirmed.
            if parent_session_id:
                with self._pending_lock:
                    self._pending_used_ids.pop(parent_session_id, None)

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
            sync. Idempotent on duplicate writes — we just add a row.

            ``edit`` and ``delete`` actions are noted in the debug log but
            not yet wired to store mutations — the exact `target`/`content`
            semantics for those actions are not yet finalised in the spec."""
            if action not in ("add", "edit", "delete"):
                return
            if action in ("edit", "delete"):
                logger.debug(
                    "lancedb_pro on_memory_write: action %r not yet handled "
                    "(target=%r); built-in and LanceDB stores may diverge",
                    action, target,
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
                    {k: v for k, v in metadata.items() if k != "session_id"}
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
            cooldown-gated auto-purge."""
            with self._thread_lock:
                thread = self._sync_thread
            if thread and thread.is_alive():
                thread.join(timeout=10.0)

            try:
                self._write_session_summary(messages)
            except Exception as e:
                logger.warning("lancedb_pro session-summary write failed: %s", e)

            with self._pending_lock:
                self._pending_used_ids.clear()
            _maybe_auto_purge(self._store)

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
            _maybe_auto_purge(self._store)

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
