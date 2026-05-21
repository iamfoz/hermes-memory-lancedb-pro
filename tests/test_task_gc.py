"""Unit tests for task_gc — task-ledger garbage collection. No heavy deps."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from hermes_memory_lancedb_pro.task_gc import (
    TaskGCConfig,
    record_task_gc_run,
    run_task_gc,
    should_run_task_gc,
)


@pytest.fixture
def task_root(tmp_path):
    root = tmp_path / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _put_task(root, task_id, **fields):
    """Write a task directory's state.json directly, for precise control."""
    d = root / task_id
    d.mkdir(parents=True, exist_ok=True)
    state = {"task_id": task_id, "objective": f"obj {task_id}", **fields}
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


def _put_archived(root, task_id, **fields):
    """Write a directory directly under archive/, for stage-2 tests."""
    d = root / "archive" / task_id
    d.mkdir(parents=True, exist_ok=True)
    state = {"task_id": task_id, "objective": f"obj {task_id}", **fields}
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


class TestArchive:
    def test_completed_past_retention_is_archived(self, task_root):
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.archived == ["old"]
        assert not (task_root / "old").exists()
        assert (task_root / "archive" / "old" / "state.json").exists()

    def test_audit_files_preserved_in_archive(self, task_root):
        d = _put_task(task_root, "withlogs", status="complete", completed_at=_iso(40))
        (d / "results.jsonl").write_text('{"iteration": 0, "result": "pass"}\n')
        (d / "events.jsonl").write_text('{"event": "complete"}\n')
        run_task_gc(task_root, TaskGCConfig(retention_days=30))
        arch = task_root / "archive" / "withlogs"
        assert (arch / "results.jsonl").exists()
        assert (arch / "events.jsonl").exists()

    def test_recent_completed_is_skipped(self, task_root):
        _put_task(task_root, "fresh", status="complete", completed_at=_iso(5))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.archived == []
        assert result.skipped_recent == 1
        assert (task_root / "fresh").exists()

    def test_completed_without_timestamp_is_skipped(self, task_root):
        # No completion timestamp at all → fail-safe, never collected.
        _put_task(task_root, "notime", status="complete")
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.archived == []
        assert (task_root / "notime").exists()

    def test_running_task_never_touched(self, task_root):
        _put_task(task_root, "live", status="running", updated_at=_iso(2))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.archived == [] and result.deleted == []
        assert result.skipped_running == 1
        assert (task_root / "live").exists()

    def test_gc_log_written(self, task_root):
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        run_task_gc(task_root, TaskGCConfig(retention_days=30))
        log = task_root / ".task-gc-log.jsonl"
        assert log.exists()
        rec = json.loads(log.read_text().splitlines()[0])
        assert rec["action"] == "archived" and rec["task_id"] == "old"


class TestDeleteMode:
    def test_delete_removes_dir(self, task_root):
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30, mode="delete"))
        assert result.deleted == ["old"]
        assert not (task_root / "old").exists()
        assert not (task_root / "archive").exists()


class TestHold:
    def test_held_completed_task_is_skipped(self, task_root):
        _put_task(task_root, "keep", status="complete", completed_at=_iso(99),
                  gc_hold=True)
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.held == ["keep"]
        assert result.archived == []
        assert (task_root / "keep").exists()

    def test_held_archived_task_not_purged(self, task_root):
        arch = _put_archived(task_root, "keptarch", status="complete",
                             completed_at=_iso(400), gc_hold=True)
        result = run_task_gc(
            task_root, TaskGCConfig(retention_days=30, archive_grace_days=90)
        )
        assert "keptarch" in result.held
        assert arch.exists()


class TestArchivePurge:
    def test_old_archived_dir_purged(self, task_root):
        arch = _put_archived(task_root, "ancient", status="complete",
                             completed_at=_iso(400))
        result = run_task_gc(
            task_root, TaskGCConfig(retention_days=30, archive_grace_days=90)
        )
        assert result.purged_archive == ["ancient"]
        assert not arch.exists()

    def test_archive_grace_zero_disables_stage2(self, task_root):
        arch = _put_archived(task_root, "ancient", status="complete",
                             completed_at=_iso(400))
        result = run_task_gc(
            task_root, TaskGCConfig(retention_days=30, archive_grace_days=0)
        )
        assert result.purged_archive == []
        assert arch.exists()


class TestEdgeCases:
    def test_unparseable_state_skipped(self, task_root):
        d = task_root / "broken"
        d.mkdir()
        (d / "state.json").write_text("{not json")
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.skipped_unparseable == 1
        assert d.exists()

    def test_missing_root_returns_empty(self, tmp_path):
        result = run_task_gc(tmp_path / "nope", TaskGCConfig())
        assert result.scanned == 0 and result.archived == []

    def test_dry_run_changes_nothing(self, task_root):
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30, dry_run=True))
        assert result.archived == ["old"]
        assert result.dry_run is True
        assert (task_root / "old").exists()
        assert not (task_root / "archive").exists()
        assert not (task_root / ".task-gc-log.jsonl").exists()

    def test_abandoned_running_reported_not_deleted(self, task_root):
        _put_task(task_root, "stale", status="running", updated_at=_iso(60))
        result = run_task_gc(
            task_root, TaskGCConfig(retention_days=30, abandoned_running_days=30)
        )
        assert result.abandoned_running == ["stale"]
        assert result.skipped_running == 1
        assert (task_root / "stale").exists()

    def test_archive_dir_not_scanned_as_task(self, task_root):
        (task_root / "archive").mkdir()
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        result = run_task_gc(task_root, TaskGCConfig(retention_days=30))
        assert result.scanned == 1  # the archive/ dir itself is not counted

    def test_archive_name_collision_suffixed(self, task_root):
        _put_archived(task_root, "old", status="complete", completed_at=_iso(5))
        _put_task(task_root, "old", status="complete", completed_at=_iso(40))
        result = run_task_gc(
            task_root, TaskGCConfig(retention_days=30, archive_grace_days=0)
        )
        assert result.archived == ["old"]
        names = sorted(p.name for p in (task_root / "archive").iterdir())
        assert len(names) == 2  # original + the collision-suffixed move


class TestCooldown:
    def test_missing_state_file_runs(self, tmp_path):
        assert should_run_task_gc(str(tmp_path / "nope.json"), 168) is True

    def test_just_recorded_does_not_run(self, tmp_path):
        sf = str(tmp_path / "gc-state.json")
        record_task_gc_run(sf)
        assert should_run_task_gc(sf, 168) is False

    def test_elapsed_cooldown_runs(self, tmp_path):
        sf = tmp_path / "gc-state.json"
        sf.write_text(json.dumps({"last_run_at": 0}))  # epoch 0 → long elapsed
        assert should_run_task_gc(str(sf), 168) is True
