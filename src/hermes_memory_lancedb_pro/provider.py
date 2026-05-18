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

import json
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
from .task_ledger import build_control_block as _build_task_control_block

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

# ---------------------------------------------------------------------------
# Recall guardrails
# ---------------------------------------------------------------------------
# Categories that are NEVER injected into the recall block regardless of score.
# Comma-separated; e.g. MEMORY_NEVER_CATEGORIES=greeting,ephemeral_chat,old_task_state
_RECALL_NEVER_CATEGORIES: frozenset[str] = frozenset(
    c.strip()
    for c in os.environ.get("MEMORY_NEVER_CATEGORIES", "greeting,ephemeral_chat").split(",")
    if c.strip()
)
# Approximate character budget for the full recall block injected each turn.
# chars / 4 ≈ tokens, so the default 4800 ≈ 1200 tokens.  Set 0 to disable.
_RECALL_CHAR_BUDGET: int = int(os.environ.get("MEMORY_RECALL_CHAR_BUDGET", "4800"))
# When True, memories with category="active_task" are pinned to the front of
# the recall block and bypass the never-categories filter and char budget.
# This ensures long-running task state is always in context.
_RECALL_ACTIVE_TASK_PIN: bool = os.environ.get(
    "MEMORY_ACTIVE_TASK_PIN", "on"
).strip().lower() not in ("off", "0", "false", "no")
# When True, the full durable-task protocol is prepended to every recall
# block so the model always knows how to manage multi-step work — without
# requiring an explicit skill invocation.  Set MEMORY_TASK_PROTOCOL=off to
# silence (e.g. in automated pipelines that manage the ledger externally).
_RECALL_TASK_PROTOCOL: bool = os.environ.get(
    "MEMORY_TASK_PROTOCOL", "on"
).strip().lower() not in ("off", "0", "false", "no")

# Full durable-task protocol injected every turn when _RECALL_TASK_PROTOCOL
# is True.  Front-loaded with the unconditional no-greeting rule so the
# model cannot misread the scope as "only during a running task".
_TASK_PROTOCOL_TEXT = """\
# Memory Task Protocol

## NEVER GREET — unconditional, no exceptions

Do not say "Hello", "Hi", "Hey", or any greeting word in ANY response:
- Even if the user greeted you
- Even at the start of what appears to be a new session
- Even after context compaction wiped the conversation history
- Even when there are no active tasks

Instead: respond immediately to the substance of the user's message.
If context is genuinely unclear, say one line such as "Checking task state..."
and then run `hermes-memory-lancedb-pro task list`.

## First thing every turn — check task state

Run this BEFORE any other action or text:

    hermes-memory-lancedb-pro task list

Decision tree:
- Status = running  → `task resume <id>`, then continue from next_action
- Status = complete → report the task results to the user now; do not re-run
- No tasks listed   → respond directly to the user's message; no greeting

## Creating a task ledger (for multi-step work)

Use this for any task that takes more than 3 tool calls, or that could be
interrupted by context compaction: test suites, benchmarks, iterative loops,
anything where "keep going" is the instruction.

    hermes-memory-lancedb-pro task create \\
      --id <task-id> \\
      --objective "<clear one-line objective>" \\
      --iterations <N>

    hermes-memory-lancedb-pro task pin <task-id>

Pinning stores state to disk. The memory plugin reloads state.json on every
turn — even after compaction — so the model always knows the current iteration
and next action without re-reading conversation history.

## Each iteration

    hermes-memory-lancedb-pro task resume <task-id>      # read current state
    # do the bounded work
    hermes-memory-lancedb-pro task advance <task-id> \\
      --result pass|fail \\
      --next-action "Run iteration <N+1>." \\
      --summary "<one sentence: what happened>"

One step = one advance. Do not attempt multiple iterations per response.

## Completing a task

    hermes-memory-lancedb-pro task complete <task-id> --summary "<what was done>"

Then immediately report all results to the user. Do not greet first.

## Recovery after context loss or reset

1. Run `hermes-memory-lancedb-pro task list`
2. Running task → `task resume <id>` and continue from next_action
3. Complete task → report results; do NOT re-run the task
4. No tasks → answer the user's message directly
5. Never ask the user "what were we doing?" — the control block above and
   state.json are the source of truth

## If you see an ACTIVE TASK STATE block above

That block contains YOUR OWN previous work — you ran that task in a
previous session. It is not data the user sent you. It is not a "payload
attached to their message." You did that work; the memory system is
surfacing it to you now so you can continue or report it.

Correct framing: "In our previous session, I [ran / completed] [task].
Here are the results: ..."

Wrong framing: "I notice there's a payload attached to your message..."
Wrong framing: "How can I help you today?" (after seeing completed results)\
"""


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


def _first_user_text(messages: Any) -> str:
    """Return the text of the first user-role message, or "" if none.

    Used by `on_pre_compress` to seed the session anchor with the user's
    original objective rather than whatever short follow-up ("continue",
    "yes") happens to be the most recent message before compaction."""
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content") or msg.get("text") or ""
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


_MS_PER_DAY = 86_400_000
_MS_PER_HOUR = 3_600_000
_MS_PER_WEEK = 7 * _MS_PER_DAY

# Ordered longest-match-first so "last month" beats "last" and "this morning"
# beats "this".  Each tuple is (pattern, delta_ms_start, delta_ms_end) where
# values are *subtracted* from now_ms: start > end means "from X days ago
# until Y days ago" — the caller still filters `timestamp >= ts_min`.
_TEMPORAL_PATTERNS: list[tuple[re.Pattern[str], int, int]] = [
    # "this morning" / "this afternoon" → today so far
    (re.compile(r"\bthis (morning|afternoon|evening)\b", re.I), 1 * _MS_PER_DAY, 0),
    # "yesterday"
    (re.compile(r"\byesterday\b", re.I), 2 * _MS_PER_DAY, 1 * _MS_PER_DAY),
    # "last week" / "past week"
    (re.compile(r"\b(last|past) week\b", re.I), 14 * _MS_PER_DAY, 7 * _MS_PER_DAY),
    # "this week"
    (re.compile(r"\bthis week\b", re.I), 7 * _MS_PER_DAY, 0),
    # "last month" / "past month"
    (re.compile(r"\b(last|past) month\b", re.I), 60 * _MS_PER_DAY, 30 * _MS_PER_DAY),
    # "this month"
    (re.compile(r"\bthis month\b", re.I), 30 * _MS_PER_DAY, 0),
    # "last year"
    (re.compile(r"\blast year\b", re.I), 730 * _MS_PER_DAY, 365 * _MS_PER_DAY),
    # "recently" / "lately" — loose 7-day window
    (re.compile(r"\b(recently|lately)\b", re.I), 7 * _MS_PER_DAY, 0),
    # "today"
    (re.compile(r"\btoday\b", re.I), 1 * _MS_PER_DAY, 0),
    # Named months: "in January", "last March", "back in April"
    (re.compile(
        r"\b(?:in|last|back in|during)\s+(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\b", re.I,
    ), 365 * _MS_PER_DAY, 0),   # search entire past year; month name boosts BM25 anyway
]


def _parse_temporal_intent(query: str, now_ms: int) -> tuple[int, int] | None:
    """Return (ts_min_ms, ts_max_ms) if the query contains a clear temporal
    reference, else None.

    ts_min_ms is the start of the relevant window (older boundary).
    ts_max_ms is the end of the relevant window (newer boundary, ≤ now_ms).
    The caller should post-filter results to memories whose timestamp is
    between ts_min_ms and ts_max_ms."""
    for pattern, delta_start, delta_end in _TEMPORAL_PATTERNS:
        if pattern.search(query):
            ts_min = now_ms - delta_start
            ts_max = now_ms - delta_end
            if ts_max < ts_min:
                ts_max = now_ms  # safety: never invert the range
            return (ts_min, ts_max)
    return None


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
    hermes-agent's prefetch_all skips injection entirely.

    active_task memories get a visually-distinct block so the model
    treats them as authoritative session state rather than recalled facts.
    Regular memories keep the bullet-point format."""
    if not results:
        return ""
    task_lines: list[str] = []
    mem_lines: list[str] = []
    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        cat = r.get("category") or "other"
        if cat == "active_task":
            task_lines.append("=== ACTIVE TASK STATE ===")
            task_lines.append(text)
            task_lines.append("=" * 25)
        else:
            score = next(
                (r[k] for k in ("_final_score", "_rrf_score", "score") if r.get(k) is not None),
                0.0,
            )
            trend = (r.get("_decay") or {}).get("freshness_trend", "")
            trend_tag = f" [{trend}]" if trend and trend != "stable" else ""
            mem_lines.append(f"- [{cat}] {text} (score={score:.2f}{trend_tag})")
    parts = [p for p in ["\n".join(task_lines), "\n".join(mem_lines)] if p]
    return "\n\n".join(parts) if parts else ""


def _apply_recall_guardrails(
    results: list[dict[str, Any]],
    never_categories: frozenset[str],
    char_budget: int,
    pin_active_tasks: bool,
) -> list[dict[str, Any]]:
    """Filter and reorder recall results per guardrail configuration.

    Order of operations:
    1. Split out ``active_task`` pinned memories (immune to all filters).
    2. Drop memories in ``never_categories`` from the remainder.
    3. Enforce the char budget (approximate token budget) on the remainder.
    4. Return pinned first, then budgeted rest.

    Pinned memories always appear at the top of the recall block so an
    active task control block is never crowded out by unrelated memories.
    """
    if pin_active_tasks:
        pinned = [r for r in results if r.get("category") == "active_task"]
        rest = [r for r in results if r.get("category") != "active_task"]
    else:
        pinned, rest = [], list(results)

    if never_categories:
        rest = [
            r for r in rest
            if (r.get("category") or "other") not in never_categories
        ]

    if char_budget > 0 and rest:
        budget = char_budget
        budgeted: list[dict[str, Any]] = []
        for r in rest:
            cost = len(r.get("text") or "") + 60  # ~60 chars overhead per formatted line
            if budget < cost and budgeted:
                logger.debug(
                    "lancedb_pro recall budget exhausted after %d items; %d dropped",
                    len(budgeted),
                    len(rest) - len(budgeted),
                )
                break
            budget -= cost
            budgeted.append(r)
        rest = budgeted

    return pinned + rest


def _stable_task_block(state: dict[str, Any]) -> str:
    """A running-task block built ONLY from immutable fields (task id,
    objective). Safe to place in the cached system prompt: it does not
    change on every ``task advance`` the way ``build_control_block`` does
    (current iteration, next action, recent summary). The model fetches
    live iteration state with ``task resume`` instead."""
    task_id = state.get("task_id", "unknown")
    objective = state.get("objective", "(none)")
    return (
        "ACTIVE TASK — do NOT greet.\n"
        f"Task ID:   {task_id}\n"
        f"Objective: {objective}\n"
        "Status:    running (you started this task in this conversation).\n"
        "For the current iteration and next action, run:\n"
        f"  hermes-memory-lancedb-pro task resume {task_id}\n"
        "Then continue the task — do NOT greet, do NOT restart it."
    )


def _refresh_active_task_memories(
    results: list[dict[str, Any]],
    *,
    stable: bool = False,
) -> list[dict[str, Any]]:
    """For pinned active_task memories that carry a state_path, reload the
    control block text from ``state.json`` on every recall.

    This keeps the injected task block current after ``advance_iteration``
    and makes it survive context compaction.

    ``stable``: when True the running-task block is rendered from immutable
    fields only (see ``_stable_task_block``) so it does NOT change between
    turns. The ``system_prompt_block`` path requires this — the system
    prompt is prompt-cache breakpoint 1, and mutating it every turn busts
    the cache (see the context-compression / caching guide). ``on_pre_compress``
    leaves ``stable`` False: its output feeds the compression summary, which
    is regenerated each compaction anyway, so the full live block is fine.
    """
    refreshed = []
    for r in results:
        if r.get("category") != "active_task":
            refreshed.append(r)
            continue
        try:
            meta_raw = r.get("metadata") or "{}"
            meta: dict[str, Any] = (
                json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
            )
            state_path = meta.get("state_path")
            if not state_path:
                refreshed.append(r)
                continue
            expanded = os.path.expanduser(str(state_path))
            if not os.path.exists(expanded):
                logger.debug(
                    "lancedb_pro active_task state_path missing: %s", expanded
                )
                refreshed.append(r)
                continue
            with open(expanded, encoding="utf-8") as fh:
                state = json.load(fh)

            # Completed tasks get a results-pending notice rather than
            # the iteration-advance control block.  Without this the model
            # sees "Status: complete / Next action: (none)" and defaults
            # to greeting.  The notice also instructs the model on HOW to
            # present the results — critical because the recall block can
            # appear as injected context that the model mistakes for
            # user-provided data ("payload attached to your message").
            if state.get("status") in ("complete", "completed"):
                obj = state.get("objective", "unknown task")
                summary = (state.get("completion_summary") or state.get("recent_summary") or "").strip()
                control_text = (
                    f"YOU completed this task in a previous session (not the user):\n"
                    f"Task: {obj}\n"
                    + (f"Summary: {summary}\n" if summary else "")
                    + "\n"
                    "ACTION: Present these results to the user as your OWN completed work.\n"
                    "Say something like: \"In our previous session I ran [task]. Here are the results: ...\"\n"
                    "Do NOT say the results are 'attached to your message' or 'a payload you sent'.\n"
                    "Do NOT greet. Do NOT re-run the task."
                )
            elif stable:
                control_text = _stable_task_block(state)
            else:
                control_text = _build_task_control_block(state)

            refreshed.append({**r, "text": control_text})
            logger.debug(
                "lancedb_pro refreshed active_task memory from %s (status=%s iter=%s)",
                expanded,
                state.get("status"),
                state.get("current_iteration"),
            )
        except Exception as exc:
            logger.debug("lancedb_pro active_task refresh failed: %s", exc)
            refreshed.append(r)
    return refreshed


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


def _anchor_belongs(metadata: dict[str, Any], conversation_id: str) -> bool:
    """True if an active_task memory should be visible to the conversation
    identified by `conversation_id`.

    A single hermes-agent gateway process serves many conversations from one
    shared, profile-scoped `MemoryStore`. Formal `task pin`s (carrying a
    `state_path`) are deliberate, global user actions and stay visible
    everywhere. Auto-anchors are implicit breadcrumbs and must be visible
    ONLY to the conversation that created them — otherwise one
    conversation's "you are mid-task" state bleeds into another.

    An empty `conversation_id` (no session id in play) matches everything —
    the single-conversation / CLI fallback."""
    if metadata.get("state_path"):
        return True
    if not conversation_id:
        return True
    return metadata.get("conversation_id", "") == conversation_id


def _auto_anchor_session_if_needed(
    user_content: str,
    session_id: str,
    conversation_id: str,
    store: MemoryStore,
) -> None:
    """Auto-create an active_task breadcrumb if this conversation has none.

    Without a pinned task, context compaction silently destroys all session
    state and the model resets to greeting.  This breadcrumb gives
    `system_prompt_block` something to surface after compaction so the model
    knows it is mid-session and must not greet.

    Idempotent — skips when a formal task pin exists, or when this
    conversation already has an auto-anchor. The anchor is keyed by
    `conversation_id` (stable across the context-compression session-id
    rotations within one conversation), NOT by the raw session id. A genuine
    `/reset` clears it via `_archive_auto_anchors`."""
    try:
        existing = store.list_memories(
            limit=20,
            category="active_task",
            include_archived=False,
        )
    except Exception as exc:
        logger.debug("lancedb_pro auto-anchor check failed: %s", exc)
        return

    # Formal task pin present — nothing to do.
    if any(
        isinstance(r.get("metadata"), dict) and r["metadata"].get("state_path")
        for r in existing
    ):
        return

    # This conversation already has an auto-anchor — keep it. (A different
    # conversation's anchor does not count: each conversation gets its own.)
    if any(
        isinstance(r.get("metadata"), dict)
        and r["metadata"].get("auto_anchor")
        and _anchor_belongs(r["metadata"], conversation_id)
        for r in existing
    ):
        return

    snippet = (user_content or "").strip()
    if len(snippet) > 200:
        snippet = snippet[:197] + "…"

    anchor_text = (
        "SESSION IN PROGRESS — do NOT greet.\n"
        + (f'User started with: "{snippet}"\n' if snippet else "")
        + "\nContinue from where you left off.\n"
        "Run `hermes-memory-lancedb-pro task list` to check for pinned tasks.\n"
        "If mid-task: resume it. Otherwise: answer the user's message directly."
    )

    meta: dict[str, Any] = {
        "auto_anchor": True,
        "priority": "must_include",
        "conversation_id": conversation_id,
    }
    if session_id:
        meta["source_session"] = session_id

    try:
        store.store(
            text=anchor_text,
            category="active_task",
            scope="agent",
            importance=0.9,
            metadata_extra=meta,
        )
        logger.debug(
            "lancedb_pro auto-anchored conversation %s",
            conversation_id or session_id or "global",
        )
    except Exception as exc:
        logger.debug("lancedb_pro auto-anchor write failed: %s", exc)


def _archive_auto_anchors(store: MemoryStore, conversation_id: str) -> int:
    """Archive this conversation's live auto-anchor(s); returns the count.

    Called on a genuine session reset so a brand-new conversation does not
    inherit the previous one's task breadcrumb. Only auto-anchors belonging
    to `conversation_id` are archived — a reset in one conversation must not
    disturb other conversations sharing the same store. An empty
    `conversation_id` archives every auto-anchor (single-conversation
    fallback). Formal `task pin`s (with a `state_path`) are left untouched."""
    archived = 0
    try:
        rows = store.list_memories(
            limit=20, category="active_task", include_archived=False,
        )
    except Exception as exc:
        logger.debug("lancedb_pro auto-anchor archive scan failed: %s", exc)
        return 0
    for r in rows:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        if not meta.get("auto_anchor") or meta.get("state_path"):
            continue
        if conversation_id and meta.get("conversation_id", "") != conversation_id:
            continue
        try:
            store.update(r["id"], metadata_extra={"state": _ARCHIVED_STATE})
            archived += 1
        except Exception as exc:
            logger.debug("lancedb_pro auto-anchor archive failed: %s", exc)
    return archived


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
            # Stable per-conversation id. Unlike `_session_id` (which the host
            # rotates on every context compression) this only changes on a
            # genuine reset — so it cleanly scopes auto-anchors to one
            # conversation even in a gateway serving many from one store.
            self._conversation_id: str = ""
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
            # Seed the conversation id. Subsequent context-compression session
            # rotations keep it; only a genuine /reset replaces it.
            self._conversation_id = session_id
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
            injected as anchors, deduplicated against the relevance results.

            When the query contains a clear temporal reference ("last week",
            "yesterday", "in January" …) the relevance results are
            post-filtered to memories whose timestamp falls inside the
            corresponding window.  Session anchors bypass this filter so
            task-framing memories are always present."""
            if not query or not query.strip():
                # No query — system_prompt_block assembling the
                # query-independent system prompt.  Return the protocol text
                # PLUS this conversation's active-task state from disk.
                parts: list[str] = []
                if _RECALL_TASK_PROTOCOL:
                    parts.append(_TASK_PROTOCOL_TEXT)
                try:
                    task_mems = self._store.list_memories(
                        limit=20,
                        category="active_task",
                        include_archived=False,
                    )
                    # Scope to this conversation — formal pins stay global,
                    # auto-anchors from other conversations are dropped.
                    task_mems = [
                        m for m in task_mems
                        if _anchor_belongs(m.get("metadata") or {}, self._conversation_id)
                    ]
                    # stable=True: this text lands in the cached system
                    # prompt, so the running-task block must not change on
                    # every `task advance`.
                    task_mems = _refresh_active_task_memories(task_mems, stable=True)
                    # Prefer formal task pins (state_path) over auto-anchors
                    # when both exist — the pin is always more authoritative.
                    formal_pins = [
                        m for m in task_mems
                        if (m.get("metadata") or {}).get("state_path")
                    ]
                    if formal_pins:
                        task_mems = formal_pins
                    if task_mems:
                        task_block = _format_recall(task_mems)
                        if task_block:
                            parts.append(task_block)
                except Exception as _exc:
                    logger.debug(
                        "lancedb_pro no-query active task inject failed: %s", _exc
                    )
                return "\n\n".join(p for p in parts if p)
            now_ms = int(time.time() * 1000)
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

            # Temporal post-filter — if the query has a clear time reference,
            # drop relevance results that fall outside the window.
            temporal_range = _parse_temporal_intent(query, now_ms)
            if temporal_range is not None and results:
                ts_min, ts_max = temporal_range
                filtered = [
                    r for r in results
                    if ts_min <= int(r.get("timestamp") or 0) <= ts_max
                ]
                # Keep unfiltered results if nothing survives the window
                # (avoids returning an empty context block on edge cases).
                if filtered:
                    results = filtered
                else:
                    logger.debug(
                        "lancedb_pro temporal filter (%d–%d ms) matched 0 of %d "
                        "results — keeping unfiltered",
                        ts_min, ts_max, len(results),
                    )

            # Session anchors — always append the 2 oldest (task framing) and
            # 2 most-recently-written session memories so context continuity
            # holds regardless of how many turns have passed.  Without the
            # "first" anchors, task framing from turn 1 falls out of the
            # recency window after turn 3 and is only recoverable by relevance
            # search, which fails when the current query is semantically
            # distant (e.g. "check slot 7" vs "stress test my memory").
            #
            # Anchors are filtered by noise and minimum length so trivial
            # exchanges ("Hello", "OK", single-word acks) are never re-injected
            # as session context — the classic cause of the greeting-replay bug
            # where the model echoes turn-1 "Hello" on every subsequent turn.
            if session_id:
                try:
                    existing_ids = {r.get("id") for r in results}
                    first_anchors = self._store.first_for_session(session_id, limit=2)
                    recent_anchors = self._store.recent_for_session(session_id, limit=2)
                    seen: set[str | None] = set(existing_ids)
                    extra_anchors = []
                    for m in first_anchors + recent_anchors:
                        mid = m.get("id")
                        if mid in seen:
                            continue
                        # Skip noise: too short or flagged by the decay noise filter.
                        text = (m.get("text") or "").strip()
                        if len(text) < 20 or _is_noise(text):
                            logger.debug(
                                "lancedb_pro: skipping noise/short anchor %s (%d chars)",
                                mid,
                                len(text),
                            )
                            seen.add(mid)
                            continue
                        extra_anchors.append(m)
                        seen.add(mid)
                    results = results + extra_anchors
                except Exception as e:
                    logger.debug("lancedb_pro session anchor lookup failed: %s", e)

            # Drop active_task memories from the query-dependent recall path.
            # They are owned by `system_prompt_block`, which injects the task
            # protocol + active-task state into the system prompt every turn.
            # Without this filter the auto-anchor (a recent session memory)
            # would also be pulled in here via `recent_for_session`, injecting
            # a second copy of the same block on every turn.
            results = [r for r in results if r.get("category") != "active_task"]

            # Recall guardrails — drop never-categories, enforce char/token
            # budget. (Active-task pinning is a no-op here now that the
            # category is filtered out above — see `system_prompt_block`.)
            results = _apply_recall_guardrails(
                results,
                _RECALL_NEVER_CATEGORIES,
                _RECALL_CHAR_BUDGET,
                _RECALL_ACTIVE_TASK_PIN,
            )

            logger.debug(
                "lancedb_pro recall: injecting %d items [%s] for session %s",
                len(results),
                ", ".join(r.get("category", "?") for r in results),
                session_id or "global",
            )

            if results and session_id:
                with self._pending_lock:
                    self._pending_used_ids[session_id] = [
                        r["id"] for r in results if r.get("id")
                    ]

            recall_block = _format_recall(results)
            reflection_block = self._reflection_block(session_id)
            # The durable-task protocol is NOT injected here — it lives in
            # `system_prompt_block()`, the authoritative per-turn system-prompt
            # hook. Injecting it via prefetch too would duplicate ~375 tokens
            # every turn and bust prompt caching (prefetch text changes turn
            # to turn; the system prompt is stable and cacheable).
            parts = [p for p in [reflection_block, recall_block] if p]
            return "\n\n".join(parts)

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
            """Query-dependent recall — the standard (`main`-branch) path.

            Returns the formatted recall block (relevant memories +
            reflection) for the user message position. On a host running
            the `feat/memory-provider-hooks` branch — which adds
            `before_prompt_build` — the host detects our `before_prompt_build`
            override and SKIPS this method to avoid double-injecting recall.
            On a `main`-branch host (no `before_prompt_build`) this is the
            recall injection point.

            The durable-task protocol and active-task state are NOT returned
            here — those belong to `system_prompt_block()`, which every host
            calls. An empty query therefore yields an empty string rather
            than falling through to the protocol branch of `_do_recall`."""
            if not query or not query.strip():
                return ""
            self._flush_pending_write()
            return self._do_recall(query, session_id or self._session_id)

        def system_prompt_block(self) -> str:
            """Query-independent text injected into the SYSTEM PROMPT.

            This is hermes-agent's authoritative memory hook: `MemoryManager.
            build_system_prompt()` calls it once per turn and concatenates the
            result into the system prompt itself.

            It is the correct home for the durable-task protocol and the
            active-task control block because:

              * the system prompt is re-assembled every turn but is NOT
                discarded by context compaction — so a "you are mid-task,
                do not greet" directive placed here survives compaction
                automatically, which `prefetch` context (user-message
                position) does not;
              * the model treats system-prompt text as ground truth rather
                than as recalled data it might mistake for a user payload;
              * the protocol text is stable, so keeping it here (instead of
                in the turn-varying `prefetch` block) is a prompt-cache hit.

            The system prompt is prompt-cache breakpoint 1 and must stay
            stable between turns, so the active-task block injected here uses
            the immutable-fields-only rendering (`_refresh_active_task_memories
            (stable=True)`): it does not change on every `task advance`. The
            model fetches live iteration state with `task resume`; the full
            live block also reaches the compression summary via
            `on_pre_compress`."""
            self._flush_pending_write()
            try:
                return self._do_recall("", self._session_id)
            except Exception as exc:
                logger.debug("lancedb_pro system_prompt_block failed: %s", exc)
                return _TASK_PROTOCOL_TEXT if _RECALL_TASK_PROTOCOL else ""

        def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
            """Called by the host right before context compression discards
            old messages. The return value is merged into the compression
            summary prompt, so anything returned here survives compaction
            inside the compressed context itself.

            Two jobs:
              1. Ensure a session recovery anchor exists — compaction is the
                 exact failure point for the greeting loop, so this is the
                 last safe moment to guarantee `system_prompt_block` will
                 have an active-task breadcrumb to surface afterwards.
              2. Return the current active-task control block so the host
                 folds it into the compressed summary as a second, redundant
                 copy of the task state."""
            # Join any in-flight sync_turn write first. That thread also
            # calls `_auto_anchor_session_if_needed`; flushing it here
            # serialises the two callers so they cannot both pass the
            # "no anchor exists" check and create duplicate anchors.
            self._flush_pending_write()
            try:
                first_user = _first_user_text(messages)
                _auto_anchor_session_if_needed(
                    first_user, self._session_id, self._conversation_id,
                    self._store,
                )
            except Exception as exc:
                logger.debug("lancedb_pro on_pre_compress anchor failed: %s", exc)
            try:
                task_mems = self._store.list_memories(
                    limit=20, category="active_task", include_archived=False,
                )
                task_mems = [
                    m for m in task_mems
                    if _anchor_belongs(m.get("metadata") or {}, self._conversation_id)
                ]
                task_mems = _refresh_active_task_memories(task_mems)
                formal_pins = [
                    m for m in task_mems
                    if (m.get("metadata") or {}).get("state_path")
                ]
                if formal_pins:
                    task_mems = formal_pins
                return _format_recall(task_mems) if task_mems else ""
            except Exception as exc:
                logger.debug("lancedb_pro on_pre_compress export failed: %s", exc)
                return ""

        def before_prompt_build(self, turn_state: dict[str, Any]) -> str:
            """Query-dependent recall — the `feat/memory-provider-hooks` path.

            This is a non-standard hook: it exists on hermes-agent's
            `feat/memory-provider-hooks` branch, not on `main`. When the host
            supports it, it is called once per turn after the user message is
            known and the result is appended to the system prompt; the host
            then SKIPS this provider's `prefetch` to avoid double-injecting
            recall. On a `main`-branch host the hook simply never fires — an
            unused method is harmless, so the same wheel runs unmodified on
            both branches (recall travels via `prefetch` instead).

            Like `prefetch`, this returns only the query-dependent recall
            block. The durable-task protocol and active-task state come from
            `system_prompt_block()`, which both host branches always call —
            so an empty query yields an empty string here rather than
            duplicating the protocol the system prompt already carries."""
            query = str(turn_state.get("query") or "")
            if not query.strip():
                return ""
            self._flush_pending_write()
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
            # Capture store, extractor and conversation id at dispatch time so
            # a concurrent initialize() / on_session_switch() call cannot swap
            # them out mid-write and redirect this turn's data elsewhere.
            _extractor = self._smart_extractor
            _store = self._store
            _conversation_id = self._conversation_id

            # Build a context string describing the source so the extraction
            # LLM knows what kind of data it's looking at.  Hindsight research
            # found this to be the single highest-impact extraction-quality
            # lever — "vague missions produce vague results."
            _extraction_context = (
                f"Hermes agent conversation turn, "
                f"session={effective_session_id or 'unknown'}, "
                f"scope=agent. Extract durable facts, preferences, entities, "
                f"events, and problem/solution pairs. Ignore greetings, "
                f"acknowledgements, and transient scaffolding."
            )

            def _do() -> None:
                if _extractor is not None:
                    try:
                        _extractor.extract_and_persist(
                            user_content=user_content,
                            assistant_content=assistant_content,
                            session_key=effective_session_id,
                            scope="agent",
                            context=_extraction_context,
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

                # Ensure a recovery anchor exists so system_prompt_block can
                # return meaningful context after context compaction — even
                # when the model hasn't explicitly run `task create` + `task pin`.
                try:
                    _auto_anchor_session_if_needed(
                        user_content, effective_session_id, _conversation_id,
                        _store,
                    )
                except Exception as _anchor_exc:
                    logger.debug(
                        "lancedb_pro auto-anchor failed: %s", _anchor_exc
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
            # Let the previous session's last write land before switching,
            # so a reset can reliably see (and archive) its auto-anchor.
            self._flush_pending_write()
            # A genuine reset (/new, /reset) begins a fresh conversation.
            # Archive THIS conversation's auto-anchor before the id changes —
            # otherwise `system_prompt_block` would surface a stale "you are
            # mid-task" breadcrumb on turn 1 of the new conversation. Scoped
            # to this conversation so a reset never disturbs others sharing
            # the store. Formal `task pin`s are left intact.
            if reset:
                n = _archive_auto_anchors(self._store, self._conversation_id)
                if n:
                    logger.debug(
                        "lancedb_pro session reset: archived %d auto-anchor(s)", n
                    )
                self._conversation_id = new_session_id
            # A non-reset switch (context compression, /branch, /resume) is
            # the SAME conversation continuing — `_conversation_id` is kept so
            # the auto-anchor survives the session-id rotation.
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
    ``~/.hermes/hermes-agent/plugins/memory/lancedb_pro/``. Registers a configured
    LanceDBProMemoryProvider with the host context.

    A `~/.hermes/hermes-agent/plugins/memory/lancedb_pro/__init__.py` shim needs only:

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
