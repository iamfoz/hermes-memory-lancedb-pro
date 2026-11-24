"""Regression tests for task-ledger defects found in the 0.16.0 audit."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime, timedelta

import pytest

from hermes_memory_lancedb_pro import _cli
from hermes_memory_lancedb_pro import task_gc as gc
from hermes_memory_lancedb_pro import task_ledger as tl


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "tasks"
    monkeypatch.setattr(tl, "TASK_ROOT", r)
    return r


def cli(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hermes-memory-lancedb-pro", *argv])
    try:
        return _cli.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def _iso(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _completed(root, task_id: str, days_ago: float, **extra) -> None:
    tl.create_task(task_id, "objective", root=root)
    tl.complete_task(task_id, root=root)
    sp = tl._state_path(task_id, root)
    state = json.loads(sp.read_text())
    state.update(completed_at=_iso(days_ago), updated_at=_iso(days_ago), **extra)
    sp.write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# GC: grace period runs from the archive time
# ---------------------------------------------------------------------------


def test_gc_does_not_archive_and_purge_in_the_same_run(root):
    _completed(root, "old", days_ago=400)
    result = gc.run_task_gc(root=root)
    assert result.archived == ["old"] and result.purged_archive == []
    archived = root / "archive" / "old"
    assert (archived / "state.json").is_file()
    assert json.loads((archived / "state.json").read_text())["archived_at"]
    # A second run right away still keeps it: the grace period has just begun.
    assert gc.run_task_gc(root=root).purged_archive == []
    assert archived.is_dir()


def test_gc_purges_an_archive_once_grace_has_passed_since_archiving(root):
    _completed(root, "old", days_ago=400)
    gc.run_task_gc(root=root)
    sp = root / "archive" / "old" / "state.json"
    state = json.loads(sp.read_text())
    state["archived_at"] = _iso(91)
    sp.write_text(json.dumps(state))
    assert gc.run_task_gc(root=root).purged_archive == ["old"]


def test_gc_still_purges_archives_made_before_archived_at_existed(root):
    _completed(root, "legacy", days_ago=400)
    (root / "archive").mkdir()
    (root / "legacy").rename(root / "archive" / "legacy")
    assert gc.run_task_gc(root=root).purged_archive == ["legacy"]


# ---------------------------------------------------------------------------
# GC: locking, statuses, reserved id
# ---------------------------------------------------------------------------


@pytest.mark.skipif(tl.fcntl is None, reason="flock needs POSIX fcntl")
def test_gc_skips_a_task_that_is_being_written(root):
    _completed(root, "busy", days_ago=60)
    with tl.task_lock("busy", root):
        result = gc.run_task_gc(root=root)
        assert result.archived == [] and result.skipped_busy == 1
        state = tl.load_state("busy", root)
        state["gc_hold"] = True
        tl._write_state("busy", state, root)
    assert (root / "busy" / "state.json").is_file()
    assert not (root / "archive").exists()
    # Now held, so the next run leaves it alone too.
    assert gc.run_task_gc(root=root).held == ["busy"]


def test_writer_cannot_recreate_a_collected_task(root):
    _completed(root, "gone", days_ago=60)
    state = tl.load_state("gone", root)
    gc.run_task_gc(root=root)
    with pytest.raises(FileNotFoundError):
        tl._write_state("gone", state, root)
    with pytest.raises(FileNotFoundError):
        tl.mark_reported("gone", root=root)
    assert not (root / "gone").exists()


def test_gc_rechecks_state_under_the_lock(root, monkeypatch):
    _completed(root, "t", days_ago=60)
    real = gc._collect_task

    def hold_then_collect(task_dir, base, cfg, task_id, result):
        tl.set_task_hold(task_id, True, root=base)  # changed after the scan
        return real(task_dir, base, cfg, task_id, result)

    monkeypatch.setattr(gc, "_collect_task", hold_then_collect)
    assert gc.run_task_gc(root=root).archived == []
    assert (root / "t" / "state.json").is_file()


def test_gc_collects_completed_spelling_and_does_not_flag_parked_tasks(root):
    _completed(root, "done", days_ago=60, status="completed")
    for task_id, status in (("w", "waiting"), ("b", "blocked"), ("r", "running")):
        tl.create_task(task_id, "o", root=root)
        sp = tl._state_path(task_id, root)
        state = json.loads(sp.read_text())
        state.update(status=status, updated_at=_iso(60), created_at=_iso(60))
        sp.write_text(json.dumps(state))
    result = gc.run_task_gc(root=root)
    assert result.archived == ["done"]
    assert result.abandoned_running == ["r"]  # parked tasks are deliberate
    assert result.skipped_running == 3


def test_archive_is_a_reserved_task_id(root, monkeypatch):
    with pytest.raises(ValueError, match="reserved"):
        tl.create_task("archive", "o", root=root)
    assert cli(monkeypatch, "task", "create", "--id", "archive", "-q") == 2


# ---------------------------------------------------------------------------
# Corrupt ledgers: exit codes, never tracebacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["[]", '"x"', "null", "5", "{not json", ""])
@pytest.mark.parametrize("verb", [
    ("show",), ("resume",), ("advance",), ("complete",), ("hold",), ("unhold",),
    ("wait", "--reason", "r"), ("block", "--reason", "r"), ("unwait",), ("wake",),
    ("reported",),
], ids=lambda v: v[0])
def test_corrupt_state_exits_4_for_every_verb(root, monkeypatch, capsys, content, verb):
    (root / "bad").mkdir(parents=True)
    (root / "bad" / "state.json").write_text(content)
    name, *rest = verb
    assert cli(monkeypatch, "task", name, "bad", *rest) == 4
    assert "Traceback" not in capsys.readouterr().err
    assert (root / "bad" / "state.json").read_text() == content  # untouched


def test_list_survives_corrupt_and_incomplete_ledgers(root, monkeypatch, capsys):
    tl.create_task("ok", "o", root=root)
    for name, content in (("list-json", "[]"), ("null", "null"), ("torn", "{")):
        (root / name).mkdir()
        (root / name / "state.json").write_text(content)
    (root / "no-id").mkdir()
    (root / "no-id" / "state.json").write_text('{"status": "running"}')
    assert cli(monkeypatch, "task", "list") == 0
    out = capsys.readouterr().out
    assert "ok" in out and "no-id" in out
    assert cli(monkeypatch, "task", "list", "--json") == 0
    ids = [t["task_id"] for t in json.loads(capsys.readouterr().out)]
    assert ids == ["no-id", "ok"]


def test_to_skill_survives_a_corrupt_archived_ledger(root, tmp_path):
    from hermes_memory_lancedb_pro import task_skill

    (root / "archive" / "junk").mkdir(parents=True)
    (root / "archive" / "junk" / "state.json").write_text("[]")
    _completed(root, "wanted", days_ago=1)
    (root / "wanted").rename(root / "archive" / "wanted__123")
    dest = task_skill.scaffold_skill_from_task("wanted", root=root, out_dir=tmp_path / "skill")
    assert (dest / "SKILL.md").is_file()


@pytest.mark.parametrize("junk", ["", '{"timestamp": "2026-01-01", "iter', "5", "[1, 2]"])
def test_resume_tolerates_bad_results_lines(root, monkeypatch, capsys, junk):
    tl.create_task("t", "o", root=root)
    tl.advance_iteration("t", result="pass", root=root)
    tl.advance_iteration("t", result="fail", root=root)
    with (root / "t" / "results.jsonl").open("a") as f:
        f.write(junk + "\n")
    assert cli(monkeypatch, "task", "resume", "t") == 0
    assert "Results so far: 2 (pass=1, fail=1)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Status guards
# ---------------------------------------------------------------------------


def test_advance_on_a_completed_task_is_refused(root, monkeypatch):
    tl.create_task("t", "o", root=root)
    tl.advance_iteration("t", root=root)
    tl.complete_task("t", root=root)
    assert cli(monkeypatch, "task", "advance", "t", "-q") == 4
    state = tl.load_state("t", root)
    assert state["current_iteration"] == 1 and state["next_action"] == "Task complete."


def test_complete_clears_waiting_on_and_keeps_first_completion_time(root):
    tl.create_task("t", "o", root=root)
    tl.wait_task("t", "CI", root=root)
    first = tl.complete_task("t", root=root)
    assert first["waiting_on"] is None and first["status"] == "complete"
    time.sleep(1.1)
    again = tl.complete_task("t", summary="again", root=root)
    assert again["completed_at"] == first["completed_at"]
    assert again["recent_summary"] == "again"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_task_root_is_absolute_and_follows_the_given_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_TASK_ROOT", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-profile"))
    assert tl._resolve_task_root() == tmp_path / "launch-profile" / "workspace" / "tasks"
    assert tl._resolve_task_root(tmp_path / "other") == tmp_path / "other" / "workspace" / "tasks"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_TASK_ROOT", "rel/tasks")
    assert tl._resolve_task_root("ignored") == tmp_path / "rel" / "tasks"


def test_pin_stores_an_absolute_state_path_for_a_relative_task_root(monkeypatch, tmp_path):
    import argparse

    stored = {}

    class FakeStore:
        def list_memories(self, **kw):
            return []

        def store(self, **kw):
            stored.update(kw)
            return "id-1"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_cli, "_open_store", lambda args: FakeStore())
    assert cli(monkeypatch, "task", "create", "--id", "rel", "--task-root", "tasks", "-q") == 0
    ns = argparse.Namespace(task_id="rel", task_root="tasks", path=None, quiet=True)
    assert _cli._cmd_task_pin(ns) == 0
    assert stored["metadata_extra"]["state_path"] == str(tmp_path / "tasks" / "rel" / "state.json")


def test_auto_gc_uses_the_providers_hermes_home(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from hermes_memory_lancedb_pro import provider

    monkeypatch.delenv("HERMES_TASK_ROOT", raising=False)
    monkeypatch.setattr(provider, "_AUTO_TASK_GC_COOLDOWN_HOURS", 168)
    launch_root = tmp_path / "launch" / "workspace" / "tasks"
    monkeypatch.setattr(tl, "TASK_ROOT", launch_root)
    _completed(launch_root, "a-task", days_ago=60)
    other_home = tmp_path / "other"
    other_root = other_home / "workspace" / "tasks"
    _completed(other_root, "b-task", days_ago=60)

    provider._maybe_auto_task_gc(MagicMock(), str(other_home))
    assert (other_root / "archive" / "b-task").is_dir()
    assert (launch_root / "a-task").is_dir()  # the other profile's ledger is untouched
    assert (other_root / ".task-gc-state.json").is_file()


def test_skill_scaffold_defaults_to_the_profile_skills_dir(root, monkeypatch, tmp_path):
    from hermes_memory_lancedb_pro import task_skill

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    _completed(root, "t", days_ago=1)
    dest = task_skill.scaffold_skill_from_task("t", root=root)
    assert dest == tmp_path / "profile" / "skills" / "t"
