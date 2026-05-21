"""Provider-level tests for the task-GC wiring — pure-Python, no LanceDB.

`_maybe_auto_task_gc` and `_soft_delete_task_pin` are module-level helpers
testable directly with a mock store and monkeypatched module constants,
mirroring tests/test_auto_purge.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hermes_memory_lancedb_pro.provider import (
    _maybe_auto_task_gc,
    _soft_delete_task_pin,
)


class TestSoftDeleteTaskPin:
    def test_removes_matching_pin(self):
        store = MagicMock()
        store.list_memories.return_value = [
            {"id": "m1", "metadata": {"task_id": "wanted"}},
            {"id": "m2", "metadata": {"task_id": "other"}},
            {"id": "m3", "metadata": {}},
        ]
        _soft_delete_task_pin(store, "wanted")
        store.forget.assert_called_once_with("m1")

    def test_handles_json_string_metadata(self):
        store = MagicMock()
        store.list_memories.return_value = [
            {"id": "m1", "metadata": json.dumps({"task_id": "wanted"})},
        ]
        _soft_delete_task_pin(store, "wanted")
        store.forget.assert_called_once_with("m1")

    def test_no_match_forgets_nothing(self):
        store = MagicMock()
        store.list_memories.return_value = [
            {"id": "m2", "metadata": {"task_id": "other"}},
        ]
        _soft_delete_task_pin(store, "wanted")
        store.forget.assert_not_called()

    def test_list_failure_is_swallowed(self):
        store = MagicMock()
        store.list_memories.side_effect = RuntimeError("db locked")
        _soft_delete_task_pin(store, "wanted")  # must not raise
        store.forget.assert_not_called()

    def test_forget_failure_is_swallowed(self):
        store = MagicMock()
        store.list_memories.return_value = [
            {"id": "m1", "metadata": {"task_id": "wanted"}},
        ]
        store.forget.side_effect = RuntimeError("row gone")
        _soft_delete_task_pin(store, "wanted")  # must not raise


class TestMaybeAutoTaskGC:
    def test_disabled_when_cooldown_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "hermes_memory_lancedb_pro.provider._AUTO_TASK_GC_COOLDOWN_HOURS", 0
        )
        monkeypatch.setattr("hermes_memory_lancedb_pro.task_ledger.TASK_ROOT", tmp_path)
        _maybe_auto_task_gc(MagicMock())
        assert not (tmp_path / ".task-gc-state.json").exists()

    def test_records_run_when_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "hermes_memory_lancedb_pro.provider._AUTO_TASK_GC_COOLDOWN_HOURS", 168
        )
        monkeypatch.setattr("hermes_memory_lancedb_pro.task_ledger.TASK_ROOT", tmp_path)
        monkeypatch.setattr("hermes_memory_lancedb_pro.task_gc.TASK_ROOT", tmp_path)
        _maybe_auto_task_gc(MagicMock())
        assert (tmp_path / ".task-gc-state.json").exists()

    def test_exception_is_swallowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "hermes_memory_lancedb_pro.provider._AUTO_TASK_GC_COOLDOWN_HOURS", 168
        )
        monkeypatch.setattr("hermes_memory_lancedb_pro.task_ledger.TASK_ROOT", tmp_path)
        monkeypatch.setattr(
            "hermes_memory_lancedb_pro.task_gc.run_task_gc",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        _maybe_auto_task_gc(MagicMock())  # must not raise
