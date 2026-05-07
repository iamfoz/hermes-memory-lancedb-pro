"""Hermes Agent MemoryProvider adapter.

Wraps `MemoryStore` + `MemoryRetriever` in the `agent.memory_provider.MemoryProvider`
ABC so hermes-agent can drop this plugin into `~/.hermes/plugins/lancedb_pro/`
and have it be discoverable, with proper session scoping wired through.

This module imports `agent.memory_provider` lazily — the rest of the package
remains usable as a standalone library, and tests / non-Hermes consumers
don't need hermes-agent installed.

USAGE (in your `~/.hermes/plugins/lancedb_pro/__init__.py`):

    from hermes_memory_lancedb_pro.provider import (
        LanceDBProMemoryProvider,
        register_memory_provider,
    )

That's all hermes-agent's plugin discovery needs. The provider:

  * passes `session_id` through to `MemoryRetriever.retrieve()` and
    `MemoryStore.store()` — fixing the cross-session memory bleed
    (the "stickiness" symptom)
  * applies a configurable `min_score` floor so unrelated memories
    don't get injected on weak matches
  * batches `sync_turn` writes and increments access counts via the
    throttled `mark_recall_used` API
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

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
        score = r.get("_final_score") or r.get("_rrf_score") or 0.0
        lines.append(f"- [{cat}] {text} (score={score:.2f})")
    return "\n".join(lines) if lines else ""


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
            self._store = store or MemoryStore.get_instance()
            self._retriever = retriever or MemoryRetriever(self._store)
            self._min_score = (
                min_score if min_score is not None else DEFAULT_MIN_RECALL_SCORE
            )
            self._prefetch_limit = prefetch_limit
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
            self._smart_extractor = smart_extractor
            if smart_extractor is None and auto_smart_extraction:
                self._smart_extractor = _maybe_build_default_smart_extractor(self._store)

        # ---- ABC requirements --------------------------------------------

        @property
        def name(self) -> str:
            return PROVIDER_NAME

        def is_available(self) -> bool:
            return True

        def initialize(self, session_id: str, **_kwargs: Any) -> None:
            self._store._initialise()

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            return []  # context-only provider; no tool calls

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
                self._pending_used_ids[session_id] = [
                    r["id"] for r in results if r.get("id")
                ]
            return _format_recall(results)

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            """User-message memory injection (legacy hermes-agent path).

            Returns the formatted recall block. On hermes-agent versions
            that support `before_prompt_build`, this method is NOT
            called — the host detects our override and skips prefetch
            to avoid double-injection. On older hermes-agent, this is
            the only injection point."""
            return self._do_recall(query, session_id)

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
            session_id = str(turn_state.get("session_id") or "")
            return self._do_recall(query, session_id)

        # ---- Write path ---------------------------------------------------

        def sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            *,
            session_id: str = "",
        ) -> None:
            """Persist a completed turn and credit the memories that were
            actually used. We tag each new entry with `source_session` so
            future recalls in this session can find them and other
            sessions' recalls won't.

            When a `smart_extractor` is configured, sync_turn delegates the
            write to it (LLM-driven 6-category extraction). Otherwise we
            fall back to writing raw user / assistant turns — same shape
            this provider has always used."""
            if self._smart_extractor is not None:
                try:
                    self._smart_extractor.extract_and_persist(
                        user_content=user_content,
                        assistant_content=assistant_content,
                        session_key=session_id,
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
                    self._raw_sync_turn(user_content, assistant_content, session_id)
            else:
                self._raw_sync_turn(user_content, assistant_content, session_id)

            # Credit the memories the model saw in its prefetch — bypasses
            # the per-recall throttle because we now know they were actually
            # injected into a turn.
            used = self._pending_used_ids.pop(session_id, None) if session_id else None
            if used:
                try:
                    # Pass session_id so the cross-session auto-promotion
                    # ledger can track distinct-session usage.
                    self._store.mark_recall_used(used, session_id=session_id)
                except Exception as e:
                    logger.warning("lancedb_pro mark_recall_used failed: %s", e)

        def _raw_sync_turn(
            self,
            user_content: str,
            assistant_content: str,
            session_id: str,
        ) -> None:
            """Legacy raw-turn write path. Used when no smart_extractor is
            configured, or as a fail-safe if the extractor orchestrator
            itself raises (per-candidate failures don't reach here)."""
            metadata_extra = (
                {"source_session": session_id, "source": "agent_turn"}
                if session_id else {"source": "agent_turn"}
            )
            try:
                if user_content and user_content.strip():
                    self._store.store(
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
                    self._store.store(
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
            ids = self._pending_used_ids.pop(session_id or "", None) if session_id else None
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
                    self._store.mark_recall_used(used, session_id=session_id)
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
            sync. Idempotent on duplicate writes — we just add a row."""
            if action != "add" or not content.strip():
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

        def shutdown(self) -> None:
            self._pending_used_ids.clear()

    return LanceDBProMemoryProvider


# Build the class once at import time; it's either real or a stub.
LanceDBProMemoryProvider = _build_provider_class()


def register_memory_provider(_ctx: Any = None) -> Any:
    """Plugin entry point: returns a configured LanceDBProMemoryProvider.

    Called by hermes-agent's `_load_provider_from_dir`. The `_ctx` arg is
    accepted for compatibility with the plugin contract; we don't need
    its contents because we read MemoryStore config from env vars.

    A `~/.hermes/plugins/lancedb_pro/__init__.py` shim should look like:

        from hermes_memory_lancedb_pro.provider import (
            LanceDBProMemoryProvider,
            register_memory_provider,
        )

        __all__ = ["LanceDBProMemoryProvider", "register_memory_provider"]
    """
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
    "register_memory_provider",
]


def _self_check() -> str:  # pragma: no cover — exercised by smoke test
    """Cheap smoke for "is the provider class wired?" — used by tests."""
    return "stub" if _load_memory_provider_base() is None else "real"
