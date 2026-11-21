"""Task CLI behaviour the task heartbeat relies on.

The heartbeat's ``ledger-caps`` probe drives this CLI as a subprocess with
``HERMES_TASK_ROOT`` pointing at a scratch root and checks exact observable
behaviour. ``TestProbeContract`` does the same, so a change that would flip a
probe to false fails here first. The other tests call ``main()`` in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hermes_memory_lancedb_pro import _cli
from hermes_memory_lancedb_pro import task_ledger as _tl


@pytest.fixture
def task_root(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    monkeypatch.setattr(_tl, "TASK_ROOT", root)
    return root


def cli(monkeypatch, *argv: str) -> int:
    """Run ``main()`` in-process; return its exit code (argparse exits too)."""
    monkeypatch.setattr(sys, "argv", ["hermes-memory-lancedb-pro", *argv])
    try:
        return _cli.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def show(monkeypatch, capsys, task_id: str) -> dict:
    capsys.readouterr()
    assert cli(monkeypatch, "task", "show", task_id) == 0
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# waiting / blocked
# ---------------------------------------------------------------------------


class TestWaitBlockUnwait:
    def test_wait_records_waiting_on(self, task_root, monkeypatch, capsys):
        assert cli(monkeypatch, "task", "create", "--id", "t", "-q") == 0
        assert cli(monkeypatch, "task", "wait", "t", "--reason", "CI run",
                   "--ref", "PR#7", "--clears-when", "CI is green") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["status"] == "waiting"
        w = state["waiting_on"]
        assert (w["reason"], w["ref"], w["clears_when"]) == ("CI run", "PR#7", "CI is green")
        assert w["since"]  # ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(w["since"])

    def test_wait_optional_fields_default_to_null(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "wait", "t", "--reason", "r") == 0
        w = show(monkeypatch, capsys, "t")["waiting_on"]
        assert w["ref"] is None and w["clears_when"] is None

    def test_wait_requires_reason(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "wait", "t") == 2

    def test_block_then_unwait(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "block", "t", "--reason", "needs a key") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["status"] == "blocked"
        assert state["waiting_on"]["reason"] == "needs a key"
        assert cli(monkeypatch, "task", "unwait", "t") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["status"] == "running" and state["waiting_on"] is None

    def test_unwait_clears_waiting(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        cli(monkeypatch, "task", "wait", "t", "--reason", "r")
        assert cli(monkeypatch, "task", "unwait", "t") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["status"] == "running" and state["waiting_on"] is None

    def test_unwait_running_task_is_refused(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "unwait", "t") != 0
        assert _tl.load_state("t")["status"] == "running"

    def test_wait_on_completed_task_is_refused(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        cli(monkeypatch, "task", "complete", "t", "-q")
        assert cli(monkeypatch, "task", "wait", "t", "--reason", "r") != 0
        assert cli(monkeypatch, "task", "block", "t", "--reason", "r") != 0
        assert _tl.load_state("t")["status"] == "complete"

    def test_events_logged(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        cli(monkeypatch, "task", "wait", "t", "--reason", "r", "-q")
        cli(monkeypatch, "task", "unwait", "t", "-q")
        events = [json.loads(ln)["event"] for ln in
                  (task_root / "t" / "events.jsonl").read_text().splitlines()]
        assert events == ["waiting", "unwait"]

    def test_show_fills_waiting_on_for_older_ledgers(self, task_root, monkeypatch, capsys):
        _tl.create_task("old", "o")
        state = _tl.load_state("old")
        del state["waiting_on"]
        _tl.save_state("old", state)
        assert show(monkeypatch, capsys, "old")["waiting_on"] is None

    def test_control_block_shows_waiting_on(self, task_root):
        _tl.create_task("t", "o")
        state = _tl.wait_task("t", "upstream release", ref="v2", clears_when="tag exists")
        block = _tl.build_control_block(state)
        assert "Status:           waiting" in block
        assert "Waiting on:       upstream release" in block
        assert "v2" in block and "tag exists" in block


# ---------------------------------------------------------------------------
# The heartbeat's ledger-caps probe, verbatim (subprocess + HERMES_TASK_ROOT)
# ---------------------------------------------------------------------------


def _probe_cli(root, *argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HERMES_TASK_ROOT": str(root)}
    env.pop("HERMES_HOME", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_memory_lancedb_pro._cli", "task", *argv],
        check=False, env=env, capture_output=True, text=True, timeout=60,
    )


def _probe_show(root, task_id: str) -> dict:
    r = _probe_cli(root, "show", task_id)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestProbeContract:
    def test_status(self, tmp_path):
        root = tmp_path / "scratch"
        assert _probe_cli(root, "create", "--id", "probe", "-q").returncode == 0
        assert _probe_cli(root, "wait", "probe", "--reason", "probe").returncode == 0
        s = _probe_show(root, "probe")
        assert s["status"] == "waiting" and isinstance(s["waiting_on"], dict)
        assert set(s["waiting_on"]) == {"reason", "ref", "since", "clears_when"}
        assert _probe_cli(root, "unwait", "probe").returncode == 0
        assert _probe_cli(root, "block", "probe", "--reason", "probe").returncode == 0
        assert _probe_show(root, "probe")["status"] == "blocked"
        assert _probe_cli(root, "unwait", "probe").returncode == 0
        assert _probe_show(root, "probe")["status"] == "running"
