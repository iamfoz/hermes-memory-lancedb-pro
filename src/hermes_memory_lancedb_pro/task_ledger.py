"""Durable task-state management for long-running agent tasks.

Task state lives outside the LLM context window, under ``<task root>/<task_id>/``
where the task root is profile-isolated (see ``_resolve_task_root``): it follows
``HERMES_HOME`` when set, falling back to ``~/.hermes``.  The runner re-reads
``state.json`` at the start of every iteration and updates it atomically after
each step so context compaction, model resets, and session restarts cannot
silently lose task progress.

Layout::

    <task root>/<task_id>/         # <task root> = $HERMES_HOME/workspace/tasks
        state.json      — objective, status, iteration counter, next_action
        results.jsonl   — per-iteration pass/fail log
        events.jsonl    — reset detections, retries, blockers
        log.md          — human-readable notes

Typical runner loop::

    state = load_state(task_id)
    prompt = build_control_block(state) + "\\n" + my_question
    response = call_model(prompt)
    if looks_like_reset(response):
        append_jsonl(task_id, "events.jsonl", {"event": "reset_detected"})
        # reconstruct and retry
    else:
        state = advance_iteration(task_id, result="pass", next_action="...")
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

# ---------------------------------------------------------------------------
# Task root directory
# ---------------------------------------------------------------------------
# Resolution order, highest priority first:
#   1. HERMES_TASK_ROOT  — explicit override of the task ledger location.
#   2. HERMES_HOME       — the per-profile home hermes-agent exports for each
#                          `hermes -p <name>` profile. Anchoring under it keeps
#                          each profile's task ledger isolated, matching the
#                          profile isolation the rest of the plugin already
#                          honours via the `hermes_home` initialize() kwarg.
#   3. ~/.hermes         — legacy single-profile default.
# The env vars are read at import time; a profile sets them before launch, and
# a `task` CLI subprocess spawned by the agent inherits them.


def _resolve_task_root() -> Path:
    task_root_env = os.environ.get("HERMES_TASK_ROOT", "").strip()
    if task_root_env:
        return Path(task_root_env).expanduser()
    hermes_home_env = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home_env:
        return Path(hermes_home_env).expanduser() / "workspace" / "tasks"
    return Path.home() / ".hermes" / "workspace" / "tasks"


TASK_ROOT: Path = _resolve_task_root()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_INVARIANTS: list[str] = [
    "Do not greet the user.",
    "Do not restart the conversation.",
    "Do not answer the original opening message.",
    "Before each iteration, reload state.json.",
    "After each iteration, update state.json atomically.",
    "If context is unclear, reload state.json and continue from next_action.",
]

# Short responses matching any of these patterns while a task is active
# are treated as a model-reset event.
_RESET_PATTERNS: tuple[str, ...] = (
    "hello",
    "hi there",
    "how can i help",
    "what can i help",
    "what would you like",
    "how may i assist",
    "👋",
    "good morning",
    "good afternoon",
    "good evening",
)

# At least one of these tokens must appear in a short response for it to
# be treated as legitimate task output.
_TASK_KEYWORDS: frozenset[str] = frozenset({
    "iteration",
    "state",
    "result",
    "task",
    "running",
    "complete",
    "fail",
    "objective",
    "action",
    "pass",
    "error",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _task_dir(task_id: str, root: Path | None = None) -> Path:
    return (root or TASK_ROOT) / task_id


def _state_path(task_id: str, root: Path | None = None) -> Path:
    return _task_dir(task_id, root) / "state.json"


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_task_id(task_id: str) -> str:
    """Reject task ids that could escape the task root.

    A task id becomes a directory name, so anything containing a path
    separator — or ``.`` / ``..`` — would let a write, or a GC ``shutil``
    call, operate outside the task root. Allows alphanumerics, dot,
    underscore and hyphen. Returns *task_id* unchanged when valid.
    """
    if not task_id or task_id in (".", "..") or not _TASK_ID_RE.match(task_id):
        raise ValueError(f"invalid task_id: {task_id!r}")
    return task_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* atomically via a temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, "updated_at": _now_iso()}
    with NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def create_task(
    task_id: str,
    objective: str,
    *,
    target_iterations: int | None = None,
    next_action: str = "Begin first iteration.",
    invariants: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Create a new task ledger directory and ``state.json``.

    Raises ``FileExistsError`` if a task with this *task_id* already exists.
    Returns the initial state dict.
    """
    _validate_task_id(task_id)
    sp = _state_path(task_id, root)
    if sp.exists():
        raise FileExistsError(f"Task {task_id!r} already exists at {sp}")
    state: dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "objective": objective,
        "created_at": _now_iso(),
        "current_iteration": 0,
        "target_iterations": target_iterations,
        "last_successful_iteration": None,
        "next_action": next_action,
        "invariants": invariants if invariants is not None else list(_DEFAULT_INVARIANTS),
        "recent_summary": "",
        "blockers": [],
        "files": {
            "results": "results.jsonl",
            "events": "events.jsonl",
            "log": "log.md",
        },
        **(extra or {}),
    }
    atomic_write_json(sp, state)
    return state


def load_state(task_id: str, root: Path | None = None) -> dict[str, Any]:
    """Load and return task state.

    Raises ``FileNotFoundError`` if the task does not exist.
    """
    sp = _state_path(task_id, root)
    with sp.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(
    task_id: str, state: dict[str, Any], root: Path | None = None
) -> None:
    """Atomically overwrite ``state.json`` with *state*."""
    atomic_write_json(_state_path(task_id, root), state)


def advance_iteration(
    task_id: str,
    *,
    result: str = "pass",
    next_action: str | None = None,
    summary: str | None = None,
    blockers: list[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Increment *current_iteration*, log the result, and save state atomically.

    *next_action* defaults to a generic "Run iteration N+1." string derived
    from the new counter value, or "All iterations complete." once
    *target_iterations* is reached.
    """
    state = load_state(task_id, root)
    current = int(state.get("current_iteration") or 0)
    state["last_successful_iteration"] = current
    state["current_iteration"] = current + 1

    if next_action is not None:
        state["next_action"] = next_action
    else:
        target = state.get("target_iterations")
        n = state["current_iteration"]
        if target is not None:
            state["next_action"] = (
                f"Run iteration {n + 1}." if n < target else "All iterations complete."
            )
        else:
            state["next_action"] = f"Run iteration {n + 1}."

    if summary is not None:
        state["recent_summary"] = summary
    if blockers is not None:
        state["blockers"] = blockers

    append_jsonl(
        task_id, "results.jsonl", {"iteration": current, "result": result}, root=root
    )
    save_state(task_id, state, root)
    return state


def complete_task(
    task_id: str,
    summary: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """Mark task as complete and write a completion event."""
    state = load_state(task_id, root)
    state["status"] = "complete"
    state["completed_at"] = _now_iso()
    state["next_action"] = "Task complete."
    if summary:
        state["recent_summary"] = summary
    save_state(task_id, state, root)
    append_jsonl(task_id, "events.jsonl", {"event": "complete"}, root=root)
    return state


def set_task_hold(
    task_id: str, hold: bool, root: Path | None = None
) -> dict[str, Any]:
    """Set or clear the GC-hold flag on a task.

    A held task (``gc_hold`` true) is exempt from all garbage collection —
    never archived, deleted, or reported as abandoned, regardless of age or
    status. Returns the updated state.
    """
    state = load_state(task_id, root)
    state["gc_hold"] = bool(hold)
    save_state(task_id, state, root)
    return state


def append_jsonl(
    task_id: str,
    filename: str,
    event: dict[str, Any],
    root: Path | None = None,
) -> None:
    """Append *event* as a timestamped JSONL line to *filename* in the task dir."""
    path = _task_dir(task_id, root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now_iso(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_control_block(state: dict[str, Any]) -> str:
    """Return the ACTIVE TASK CONTROL BLOCK string to prepend to each prompt.

    Including this in every iteration prompt means context compaction cannot
    silently lose the task objective or stopping condition.
    """
    task_id = state.get("task_id", "unknown")
    status = state.get("status", "unknown")
    objective = state.get("objective", "(none)")
    current = state.get("current_iteration", 0)
    target = state.get("target_iterations")
    last_ok = state.get("last_successful_iteration")
    next_action = state.get("next_action", "(none)")
    summary = (state.get("recent_summary") or "").strip()
    blockers: list[str] = state.get("blockers") or []
    invariants: list[str] = state.get("invariants") or []

    iter_info = f"{current}" + (f" of {target}" if target is not None else "")

    lines: list[str] = [
        "ACTIVE TASK CONTROL BLOCK",
        "",
        f"Task ID:          {task_id}",
        f"Status:           {status}",
        f"Objective:        {objective}",
        f"Current iter:     {iter_info}",
        f"Last OK iter:     {last_ok}",
        f"Next action:      {next_action}",
    ]
    if summary:
        lines += ["", f"Recent summary: {summary}"]
    if blockers:
        lines += ["", "Blockers:"] + [f"  - {b}" for b in blockers]
    if invariants:
        lines += ["", "Rules:"] + [f"  - {inv}" for inv in invariants]
    lines.append("")
    return "\n".join(lines)


def looks_like_reset(response: str, active_task: bool = True) -> bool:
    """Return True when *response* looks like a greeting/model-reset rather than task output.

    Only meaningful when *active_task* is True.  Short responses that match a
    greeting pattern *or* contain no task-relevant tokens are flagged.  Responses
    longer than 500 characters are never flagged — they clearly contain content.
    """
    if not active_task:
        return False
    text = response.strip().lower()
    if len(text) > 500:
        return False
    return any(p in text for p in _RESET_PATTERNS) or (
        len(text) < 200 and not any(kw in text for kw in _TASK_KEYWORDS)
    )


def list_tasks(root: Path | None = None) -> list[dict[str, Any]]:
    """Return task state dicts for all tasks under *root* (default: TASK_ROOT)."""
    base = root or TASK_ROOT
    if not base.is_dir():
        return []
    tasks = []
    for task_dir in sorted(base.iterdir()):
        sp = task_dir / "state.json"
        if sp.is_file():
            try:
                with sp.open("r", encoding="utf-8") as f:
                    tasks.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return tasks
