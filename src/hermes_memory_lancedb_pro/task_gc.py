"""Garbage collection and retention for the durable task ledger.

The task ledger (see :mod:`task_ledger`) writes a directory per task under the
task root and never removes it, so completed tasks accumulate without bound.
This module provides cooldown-gated GC that, by default, *archives* completed
tasks past a retention window — moving them under ``<root>/archive/`` so the
``results.jsonl`` / ``events.jsonl`` audit trail is preserved — and then
hard-deletes archived directories after a longer grace period. A ``delete``
mode hard-deletes outright instead.

It mirrors the cooldown-gated maintenance pattern used by the memory store
(see :mod:`memory_compactor`): a JSON state file records the last run, and a
provider wrapper gates auto-runs on an elapsed-time check.

Pure-Python — no LanceDB dependency. The provider layer is responsible for
cleaning up the matching ``active_task`` pin memory after a task is GC'd.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .task_ledger import TASK_ROOT, _validate_task_id

__all__ = [
    "TASK_GC_LOG_FILENAME",
    "TASK_GC_STATE_FILENAME",
    "TaskGCConfig",
    "TaskGCResult",
    "record_task_gc_run",
    "run_task_gc",
    "should_run_task_gc",
]

# State file consumed by should_run_task_gc / record_task_gc_run, and the
# append-only audit log of every GC action — both live in the task root.
TASK_GC_STATE_FILENAME = ".task-gc-state.json"
TASK_GC_LOG_FILENAME = ".task-gc-log.jsonl"

# Subdirectory under the task root that holds archived tasks. The main scan
# skips it; only the second stage ever hard-deletes from inside it.
_ARCHIVE_DIRNAME = "archive"

_DAY_MS = 24 * 60 * 60 * 1000


@dataclass
class TaskGCConfig:
    """Tuning for :func:`run_task_gc`."""

    retention_days: int = 30          # complete tasks older than this are GC'd
    mode: str = "archive"             # "archive" | "delete"
    archive_grace_days: int = 90      # 2nd-stage hard-delete of archives; 0 disables
    abandoned_running_days: int = 30  # running tasks idle this long are reported only
    dry_run: bool = False


@dataclass
class TaskGCResult:
    """Outcome of a :func:`run_task_gc` pass."""

    scanned: int = 0
    archived: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    purged_archive: list[str] = field(default_factory=list)
    abandoned_running: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    skipped_running: int = 0
    skipped_recent: int = 0
    skipped_unparseable: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cooldown gating (mirrors memory_compactor.should_run_compaction)
# ---------------------------------------------------------------------------


def should_run_task_gc(state_file: str, cooldown_hours: int) -> bool:
    """True when *cooldown_hours* have elapsed since the last recorded run.

    A missing or malformed state file is treated as "never run" → True.
    """
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return True
    last = state.get("last_run_at")
    if not isinstance(last, (int, float)):
        return True
    elapsed_ms = int(time.time() * 1000) - int(last)
    return elapsed_ms >= cooldown_hours * 60 * 60 * 1000


def record_task_gc_run(state_file: str) -> None:
    """Persist a ``last_run_at`` marker so :func:`should_run_task_gc` can gate."""
    parent = os.path.dirname(state_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"last_run_at": int(time.time() * 1000)}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _epoch_ms(iso: object) -> float | None:
    """Parse an ISO-8601 timestamp to epoch milliseconds, or None."""
    if not isinstance(iso, str) or not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp() * 1000


def _completion_ms(state: dict) -> float | None:
    """Best available completion timestamp: completed_at ?? updated_at ?? created_at."""
    for key in ("completed_at", "updated_at", "created_at"):
        ms = _epoch_ms(state.get(key))
        if ms is not None:
            return ms
    return None


def _read_state(state_path: Path) -> dict | None:
    """Read and JSON-parse a state.json, or None when missing/unreadable."""
    try:
        with state_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _archive_dest(archive_dir: Path, task_id: str) -> Path:
    """Destination under archive/, suffixed with an epoch on collision."""
    dest = archive_dir / task_id
    if dest.exists():
        dest = archive_dir / f"{task_id}__{int(time.time())}"
    return dest


def _log_action(base: Path, action: str, task_id: str) -> None:
    """Append one line to the append-only GC audit log. Best-effort."""
    line = json.dumps(
        {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "action": action,
            "task_id": task_id,
        },
        ensure_ascii=False,
    )
    try:
        with (base / TASK_GC_LOG_FILENAME).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _collect_task(
    task_dir: Path, base: Path, cfg: TaskGCConfig, task_id: str, result: TaskGCResult
) -> None:
    """Archive or delete one completed, past-retention task directory."""
    try:
        if cfg.mode == "delete":
            if not cfg.dry_run:
                shutil.rmtree(task_dir)
                _log_action(base, "deleted", task_id)
            result.deleted.append(task_id)
        else:
            if not cfg.dry_run:
                archive_dir = base / _ARCHIVE_DIRNAME
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(task_dir), str(_archive_dest(archive_dir, task_id)))
                _log_action(base, "archived", task_id)
            result.archived.append(task_id)
    except OSError as e:
        result.errors.append(f"{task_id}: {e}")


def _gc_task(
    task_dir: Path, base: Path, cfg: TaskGCConfig, now_ms: int, result: TaskGCResult
) -> None:
    """Classify one task directory and act on it (stage 1)."""
    result.scanned += 1
    state = _read_state(task_dir / "state.json")
    if state is None:
        result.skipped_unparseable += 1
        return

    task_id = task_dir.name
    try:
        _validate_task_id(task_id)
    except ValueError:
        result.errors.append(f"{task_id}: invalid task id, skipped")
        return

    if state.get("gc_hold"):
        result.held.append(task_id)
        return

    if state.get("status") != "complete":
        # Running (or unknown) — never mutate; flag only if long-idle.
        idle_ms = _epoch_ms(state.get("updated_at")) or _epoch_ms(
            state.get("created_at")
        )
        if idle_ms is not None and (
            now_ms - idle_ms
        ) >= cfg.abandoned_running_days * _DAY_MS:
            result.abandoned_running.append(task_id)
        result.skipped_running += 1
        return

    comp_ms = _completion_ms(state)
    if comp_ms is None or (now_ms - comp_ms) < cfg.retention_days * _DAY_MS:
        result.skipped_recent += 1
        return

    _collect_task(task_dir, base, cfg, task_id, result)


def _purge_archive(
    base: Path, cfg: TaskGCConfig, now_ms: int, result: TaskGCResult
) -> None:
    """Hard-delete archived task dirs past retention + grace (stage 2)."""
    archive_dir = base / _ARCHIVE_DIRNAME
    if not archive_dir.is_dir():
        return
    purge_ms = (cfg.retention_days + cfg.archive_grace_days) * _DAY_MS
    for arch in sorted(archive_dir.iterdir()):
        if not arch.is_dir():
            continue
        state = _read_state(arch / "state.json")
        if state is None:
            continue
        if state.get("gc_hold"):
            result.held.append(arch.name)
            continue
        comp_ms = _completion_ms(state)
        if comp_ms is None or (now_ms - comp_ms) < purge_ms:
            continue
        try:
            if not cfg.dry_run:
                shutil.rmtree(arch)
                _log_action(base, "purged", arch.name)
            result.purged_archive.append(arch.name)
        except OSError as e:
            result.errors.append(f"archive/{arch.name}: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_task_gc(
    root: Path | None = None, config: TaskGCConfig | None = None
) -> TaskGCResult:
    """Garbage-collect the task ledger under *root* (default :data:`TASK_ROOT`).

    Completed tasks past the retention window are archived (or deleted);
    running tasks are never touched (only reported if long-idle); held tasks
    (``gc_hold``) are exempt entirely. With ``archive_grace_days > 0`` a second
    stage hard-deletes archived directories past ``retention + grace``.
    """
    cfg = config or TaskGCConfig()
    base = root or TASK_ROOT
    result = TaskGCResult(dry_run=cfg.dry_run)
    if not base.is_dir():
        return result

    now_ms = int(time.time() * 1000)

    # Stage 1 — archive / delete completed tasks past retention.
    for task_dir in sorted(base.iterdir()):
        if task_dir.is_dir() and task_dir.name != _ARCHIVE_DIRNAME:
            _gc_task(task_dir, base, cfg, now_ms, result)

    # Stage 2 — hard-delete archived tasks past retention + grace.
    if cfg.archive_grace_days > 0:
        _purge_archive(base, cfg, now_ms, result)

    return result
