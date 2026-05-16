"""Unit tests for task_ledger — no heavy dependencies required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hermes_memory_lancedb_pro.task_ledger import (
    advance_iteration,
    append_jsonl,
    atomic_write_json,
    build_control_block,
    complete_task,
    create_task,
    list_tasks,
    load_state,
    looks_like_reset,
    save_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task_root(tmp_path):
    """A temporary task root directory."""
    return tmp_path / "tasks"


@pytest.fixture
def simple_task(task_root):
    """A pre-created running task with 5 target iterations."""
    return create_task(
        "test-task-001",
        objective="Run 5 test iterations.",
        target_iterations=5,
        root=task_root,
    )


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "out.json"
        atomic_write_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["key"] == "value"

    def test_adds_updated_at(self, tmp_path):
        path = tmp_path / "x.json"
        atomic_write_json(path, {"a": 1})
        data = json.loads(path.read_text())
        assert "updated_at" in data

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "x.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        data = json.loads(path.read_text())
        assert data["v"] == 2

    def test_source_dict_not_mutated(self, tmp_path):
        path = tmp_path / "x.json"
        src = {"k": "v"}
        atomic_write_json(path, src)
        assert "updated_at" not in src


# ---------------------------------------------------------------------------
# create_task / load_state / save_state
# ---------------------------------------------------------------------------


class TestCreateLoadSave:
    def test_create_returns_state_dict(self, task_root):
        state = create_task("t1", "Do something.", root=task_root)
        assert state["task_id"] == "t1"
        assert state["status"] == "running"
        assert state["objective"] == "Do something."
        assert state["current_iteration"] == 0

    def test_state_json_written(self, task_root):
        create_task("t2", "objective", root=task_root)
        sp = task_root / "t2" / "state.json"
        assert sp.exists()
        data = json.loads(sp.read_text())
        assert data["task_id"] == "t2"

    def test_create_raises_if_exists(self, task_root):
        create_task("dup", "first", root=task_root)
        with pytest.raises(FileExistsError):
            create_task("dup", "second", root=task_root)

    def test_load_state_roundtrip(self, task_root):
        create_task("t3", "obj", root=task_root)
        state = load_state("t3", root=task_root)
        assert state["task_id"] == "t3"
        assert state["objective"] == "obj"

    def test_load_state_missing_raises(self, task_root):
        with pytest.raises(FileNotFoundError):
            load_state("nonexistent", root=task_root)

    def test_save_state_updates(self, task_root):
        state = create_task("t4", "obj", root=task_root)
        state["status"] = "complete"
        save_state("t4", state, root=task_root)
        reloaded = load_state("t4", root=task_root)
        assert reloaded["status"] == "complete"

    def test_custom_target_iterations(self, task_root):
        state = create_task("t5", "obj", target_iterations=20, root=task_root)
        assert state["target_iterations"] == 20

    def test_custom_invariants(self, task_root):
        state = create_task("t6", "obj", invariants=["Only rule."], root=task_root)
        assert state["invariants"] == ["Only rule."]

    def test_default_invariants_present(self, task_root):
        state = create_task("t7", "obj", root=task_root)
        assert len(state["invariants"]) > 0
        assert any("context" in inv.lower() for inv in state["invariants"])

    def test_extra_fields_included(self, task_root):
        state = create_task("t8", "obj", extra={"custom_key": 42}, root=task_root)
        assert state["custom_key"] == 42


# ---------------------------------------------------------------------------
# advance_iteration
# ---------------------------------------------------------------------------


class TestAdvanceIteration:
    def test_increments_counter(self, task_root, simple_task):
        state = advance_iteration("test-task-001", root=task_root)
        assert state["current_iteration"] == 1
        assert state["last_successful_iteration"] == 0

    def test_next_action_auto_generated(self, task_root, simple_task):
        state = advance_iteration("test-task-001", root=task_root)
        assert "2" in state["next_action"]

    def test_next_action_complete_at_target(self, task_root):
        create_task("done-task", "obj", target_iterations=2, root=task_root)
        advance_iteration("done-task", root=task_root)  # → 1
        state = advance_iteration("done-task", root=task_root)  # → 2 (== target)
        assert "complete" in state["next_action"].lower()

    def test_custom_next_action(self, task_root, simple_task):
        state = advance_iteration(
            "test-task-001", next_action="Do the special thing.", root=task_root
        )
        assert state["next_action"] == "Do the special thing."

    def test_summary_updated(self, task_root, simple_task):
        state = advance_iteration("test-task-001", summary="All good so far.", root=task_root)
        assert state["recent_summary"] == "All good so far."

    def test_blockers_updated(self, task_root, simple_task):
        state = advance_iteration("test-task-001", blockers=["DB connection lost."], root=task_root)
        assert state["blockers"] == ["DB connection lost."]

    def test_results_jsonl_appended(self, task_root, simple_task):
        advance_iteration("test-task-001", result="fail", root=task_root)
        lines = (task_root / "test-task-001" / "results.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["result"] == "fail"
        assert record["iteration"] == 0

    def test_state_persisted(self, task_root, simple_task):
        advance_iteration("test-task-001", root=task_root)
        state = load_state("test-task-001", root=task_root)
        assert state["current_iteration"] == 1

    def test_multiple_advances(self, task_root, simple_task):
        for _ in range(3):
            advance_iteration("test-task-001", root=task_root)
        state = load_state("test-task-001", root=task_root)
        assert state["current_iteration"] == 3
        lines = (task_root / "test-task-001" / "results.jsonl").read_text().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------


class TestCompleteTask:
    def test_sets_status_complete(self, task_root, simple_task):
        state = complete_task("test-task-001", root=task_root)
        assert state["status"] == "complete"

    def test_next_action_complete_message(self, task_root, simple_task):
        state = complete_task("test-task-001", root=task_root)
        assert "complete" in state["next_action"].lower()

    def test_summary_stored(self, task_root, simple_task):
        state = complete_task("test-task-001", summary="All 5 passed.", root=task_root)
        assert state["recent_summary"] == "All 5 passed."

    def test_events_jsonl_written(self, task_root, simple_task):
        complete_task("test-task-001", root=task_root)
        lines = (task_root / "test-task-001" / "events.jsonl").read_text().splitlines()
        assert any(json.loads(ln).get("event") == "complete" for ln in lines)

    def test_state_persisted(self, task_root, simple_task):
        complete_task("test-task-001", root=task_root)
        state = load_state("test-task-001", root=task_root)
        assert state["status"] == "complete"


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------


class TestAppendJsonl:
    def test_creates_file_and_appends(self, task_root, simple_task):
        append_jsonl("test-task-001", "custom.jsonl", {"x": 1}, root=task_root)
        path = task_root / "test-task-001" / "custom.jsonl"
        assert path.exists()
        record = json.loads(path.read_text().strip())
        assert record["x"] == 1

    def test_multiple_appends_multiple_lines(self, task_root, simple_task):
        for i in range(3):
            append_jsonl("test-task-001", "ev.jsonl", {"i": i}, root=task_root)
        lines = (task_root / "test-task-001" / "ev.jsonl").read_text().splitlines()
        assert len(lines) == 3

    def test_timestamp_added(self, task_root, simple_task):
        append_jsonl("test-task-001", "ev.jsonl", {"event": "test"}, root=task_root)
        record = json.loads((task_root / "test-task-001" / "ev.jsonl").read_text())
        assert "timestamp" in record


# ---------------------------------------------------------------------------
# build_control_block
# ---------------------------------------------------------------------------


class TestBuildControlBlock:
    def test_contains_task_id(self, simple_task):
        block = build_control_block(simple_task)
        assert "test-task-001" in block

    def test_contains_objective(self, simple_task):
        block = build_control_block(simple_task)
        assert "Run 5 test iterations." in block

    def test_contains_iteration_info(self, simple_task):
        block = build_control_block(simple_task)
        assert "0 of 5" in block

    def test_contains_next_action(self, simple_task):
        block = build_control_block(simple_task)
        assert simple_task["next_action"] in block

    def test_contains_invariants(self, simple_task):
        block = build_control_block(simple_task)
        assert "Rules:" in block

    def test_summary_included_when_present(self):
        state = {
            "task_id": "x",
            "status": "running",
            "objective": "obj",
            "current_iteration": 1,
            "target_iterations": None,
            "last_successful_iteration": 0,
            "next_action": "continue",
            "recent_summary": "Three passed.",
            "blockers": [],
            "invariants": [],
        }
        block = build_control_block(state)
        assert "Three passed." in block

    def test_blockers_included_when_present(self):
        state = {
            "task_id": "x",
            "status": "running",
            "objective": "obj",
            "current_iteration": 1,
            "target_iterations": None,
            "last_successful_iteration": 0,
            "next_action": "continue",
            "recent_summary": "",
            "blockers": ["DB timeout."],
            "invariants": [],
        }
        block = build_control_block(state)
        assert "DB timeout." in block
        assert "Blockers:" in block

    def test_no_target_iterations(self):
        state = {
            "task_id": "x",
            "status": "running",
            "objective": "obj",
            "current_iteration": 3,
            "target_iterations": None,
            "last_successful_iteration": 2,
            "next_action": "go",
            "recent_summary": "",
            "blockers": [],
            "invariants": [],
        }
        block = build_control_block(state)
        assert "3" in block
        assert " of " not in block


# ---------------------------------------------------------------------------
# looks_like_reset
# ---------------------------------------------------------------------------


class TestLooksLikeReset:
    def test_hello_is_reset(self):
        assert looks_like_reset("Hello! How can I help you today?") is True

    def test_hi_there_is_reset(self):
        assert looks_like_reset("Hi there! 👋 What would you like to do?") is True

    def test_task_output_not_reset(self):
        assert looks_like_reset("Running iteration 12 of 50. Result: pass.") is False

    def test_long_response_not_reset(self):
        long_text = "Hello " + "x " * 300
        assert looks_like_reset(long_text) is False

    def test_inactive_task_never_reset(self):
        assert looks_like_reset("Hello! How can I help?", active_task=False) is False

    def test_empty_short_response_is_reset(self):
        assert looks_like_reset("OK.", active_task=True) is True

    def test_task_keyword_saves_short_response(self):
        assert looks_like_reset("Running task now.", active_task=True) is False

    def test_result_keyword_saves_short_response(self):
        assert looks_like_reset("Result: pass.", active_task=True) is False

    def test_500_char_boundary(self):
        # Exactly 501 chars — long enough to bypass length check
        text = "hello " + "a" * 495
        assert looks_like_reset(text) is False


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_empty_root_returns_empty(self, tmp_path):
        assert list_tasks(tmp_path) == []

    def test_nonexistent_root_returns_empty(self, tmp_path):
        assert list_tasks(tmp_path / "nonexistent") == []

    def test_returns_all_tasks(self, task_root):
        create_task("a", "obj a", root=task_root)
        create_task("b", "obj b", root=task_root)
        tasks = list_tasks(task_root)
        ids = [t["task_id"] for t in tasks]
        assert "a" in ids and "b" in ids

    def test_alphabetical_order(self, task_root):
        for tid in ("z-task", "a-task", "m-task"):
            create_task(tid, "obj", root=task_root)
        tasks = list_tasks(task_root)
        ids = [t["task_id"] for t in tasks]
        assert ids == sorted(ids)

    def test_skips_directories_without_state_json(self, task_root):
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "orphan-dir").mkdir()
        create_task("valid", "obj", root=task_root)
        tasks = list_tasks(task_root)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "valid"
