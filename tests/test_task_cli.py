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
        assert cli(monkeypatch, "task", "unwait", "t") == 4
        assert _tl.load_state("t")["status"] == "running"

    def test_wait_on_completed_task_is_refused(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        cli(monkeypatch, "task", "complete", "t", "-q")
        assert cli(monkeypatch, "task", "wait", "t", "--reason", "r") == 4
        assert cli(monkeypatch, "task", "block", "t", "--reason", "r") == 4
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
# wake_count / max_wakes
# ---------------------------------------------------------------------------


class TestWakes:
    def test_create_records_max_wakes(self, task_root, monkeypatch, capsys):
        assert cli(monkeypatch, "task", "create", "--id", "t", "--max-wakes", "3", "-q") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["max_wakes"] == 3 and state["wake_count"] == 0

    def test_max_wakes_defaults_to_null(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        state = show(monkeypatch, capsys, "t")
        assert state["max_wakes"] is None and state["wake_count"] == 0

    def test_negative_max_wakes_rejected(self, task_root, monkeypatch):
        assert cli(monkeypatch, "task", "create", "--id", "t", "--max-wakes", "-1", "-q") == 2
        assert not (task_root / "t").exists()

    def test_wake_counts_up(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "--max-wakes", "3", "-q")
        for _ in range(2):
            assert cli(monkeypatch, "task", "wake", "t", "-q") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["wake_count"] == 2 and state["max_wakes"] == 3
        assert state["last_woken_at"]
        assert state["status"] == "running"

    def test_wake_past_budget_still_counts_and_warns(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "--max-wakes", "1", "-q")
        cli(monkeypatch, "task", "wake", "t")
        assert "max_wakes reached" in capsys.readouterr().out
        assert cli(monkeypatch, "task", "wake", "t", "-q") == 0
        assert _tl.load_state("t")["wake_count"] == 2

    def test_wake_missing_task_fails(self, task_root, monkeypatch):
        assert cli(monkeypatch, "task", "wake", "nope") == 4

    def test_wake_older_ledger_starts_at_zero(self, task_root, monkeypatch, capsys):
        _tl.create_task("old", "o")
        state = _tl.load_state("old")
        del state["wake_count"], state["max_wakes"]
        _tl.save_state("old", state)
        assert show(monkeypatch, capsys, "old")["wake_count"] == 0
        cli(monkeypatch, "task", "wake", "old", "-q")
        state = show(monkeypatch, capsys, "old")
        assert state["wake_count"] == 1 and state["max_wakes"] is None


# ---------------------------------------------------------------------------
# list --json
# ---------------------------------------------------------------------------


class TestListJson:
    def test_empty_root_is_empty_array(self, task_root, monkeypatch, capsys):
        assert cli(monkeypatch, "task", "list", "--json") == 0
        assert json.loads(capsys.readouterr().out) == []

    def test_every_state_sorted_by_task_id(self, task_root, monkeypatch, capsys):
        for tid in ("beta", "alpha", "Gamma-2", "gamma.1"):
            _tl.create_task(tid, f"objective {tid}")
        _tl.complete_task("beta")
        capsys.readouterr()
        assert cli(monkeypatch, "task", "list", "--json") == 0
        tasks = json.loads(capsys.readouterr().out)
        ids = [t["task_id"] for t in tasks]
        assert ids == sorted(ids) and set(ids) == {"alpha", "beta", "Gamma-2", "gamma.1"}
        by_id = {t["task_id"]: t for t in tasks}
        assert by_id["beta"] == _tl.load_state("beta")
        assert by_id["alpha"]["objective"] == "objective alpha"

    def test_skips_archive_and_unreadable(self, task_root, monkeypatch, capsys):
        _tl.create_task("ok", "o")
        (task_root / "archive").mkdir()
        (task_root / "broken").mkdir()
        (task_root / "broken" / "state.json").write_text("{not json")
        capsys.readouterr()
        assert cli(monkeypatch, "task", "list", "--json") == 0
        assert [t["task_id"] for t in json.loads(capsys.readouterr().out)] == ["ok"]

    def test_plain_list_unchanged(self, task_root, monkeypatch, capsys):
        assert cli(monkeypatch, "task", "list") == 0
        assert capsys.readouterr().out.strip() == "No tasks found."


# ---------------------------------------------------------------------------
# advance --result fail
# ---------------------------------------------------------------------------


class TestAdvanceFail:
    def test_fail_keeps_last_ok(self, task_root, monkeypatch, capsys):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "advance", "t", "--result", "pass", "-q") == 0
        assert show(monkeypatch, capsys, "t")["last_successful_iteration"] == 1
        assert cli(monkeypatch, "task", "advance", "t", "--result", "fail", "-q") == 0
        state = show(monkeypatch, capsys, "t")
        assert state["last_successful_iteration"] == 1
        assert state["current_iteration"] == 2


# ---------------------------------------------------------------------------
# Exit codes: 2 = usage/config, 4 = not found/refused
# ---------------------------------------------------------------------------

_ON_MISSING = [
    ("show",), ("resume",), ("advance",), ("complete",), ("hold",), ("unhold",),
    ("wait", "--reason", "r"), ("block", "--reason", "r"), ("unwait",), ("wake",),
    ("to-skill",),
]


class TestExitCodes:
    @pytest.mark.parametrize("argv", _ON_MISSING, ids=lambda a: a[0])
    def test_missing_task_exits_4(self, task_root, monkeypatch, tmp_path, argv):
        verb, *rest = argv
        if verb == "to-skill":
            rest = ["--out", str(tmp_path / "skill")]
        assert cli(monkeypatch, "task", verb, "missing", *rest) == 4
        assert not (task_root / "missing").exists()

    def test_pin_missing_task_exits_4(self, task_root, monkeypatch, tmp_path):
        assert cli(monkeypatch, "task", "pin", "missing", "--path", str(tmp_path / "db")) == 4

    def test_create_existing_exits_4(self, task_root, monkeypatch):
        assert cli(monkeypatch, "task", "create", "--id", "t", "-q") == 0
        assert cli(monkeypatch, "task", "create", "--id", "t", "-q") == 4

    def test_refused_transition_exits_4(self, task_root, monkeypatch):
        cli(monkeypatch, "task", "create", "--id", "t", "-q")
        assert cli(monkeypatch, "task", "unwait", "t") == 4

    @pytest.mark.parametrize("argv", [
        ("show", "missing", "--no-such-flag"),
        ("list", "--no-such-flag"),
        ("no-such-verb",),
        ("wait", "t"),             # --reason is required
        ("create",),               # --id is required
        ("advance", "t", "--result", "maybe"),
        ("create", "--id", "t", "--max-wakes", "lots"),
    ])
    def test_usage_errors_exit_2(self, task_root, monkeypatch, argv):
        assert cli(monkeypatch, "task", *argv) == 2

    def test_no_task_command_exits_2(self, task_root, monkeypatch):
        assert cli(monkeypatch, "task") == 2

    @pytest.mark.parametrize("bad_id", ["../escape", "a/b", "..", "sp ace"])
    def test_invalid_task_id_exits_2(self, task_root, monkeypatch, bad_id):
        assert cli(monkeypatch, "task", "show", bad_id) == 2
        assert cli(monkeypatch, "task", "create", "--id", bad_id, "-q") == 2
        assert cli(monkeypatch, "task", "wake", bad_id) == 2

    def test_bad_gc_mode_env_exits_2(self, task_root, monkeypatch):
        monkeypatch.setenv("MEMORY_TASK_GC_MODE", "shred")
        assert cli(monkeypatch, "task", "gc", "--dry-run", "-q") == 2

    def test_success_is_0(self, task_root, monkeypatch):
        assert cli(monkeypatch, "task", "create", "--id", "t", "-q") == 0
        assert cli(monkeypatch, "task", "list") == 0

    def test_in_host_dispatch_returns_the_same_codes(self, task_root):
        # `hermes lancedb_pro task ...` goes through register_cli; Hermes
        # turns the handler's int return into the process exit code.
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
        _cli.register_cli(parser)
        args = parser.parse_args(["task", "show", "missing"])
        assert args.func(args) == 4
        with pytest.raises(SystemExit) as e:
            parser.parse_args(["task", "show", "missing", "--no-such-flag"])
        assert e.value.code == 2


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

    def test_wakes(self, tmp_path):
        root = tmp_path / "scratch"
        assert _probe_cli(root, "create", "--id", "probe", "--max-wakes", "3", "-q").returncode == 0
        for _ in range(2):
            assert _probe_cli(root, "wake", "probe").returncode == 0
        s = _probe_show(root, "probe")
        assert s["wake_count"] == 2 and s["max_wakes"] == 3

    def test_list_json(self, tmp_path):
        root = tmp_path / "scratch"
        assert _probe_cli(root, "create", "--id", "probe", "-q").returncode == 0
        r = _probe_cli(root, "list", "--json")
        assert r.returncode == 0, r.stderr
        tasks = json.loads(r.stdout)
        assert isinstance(tasks, list) and "probe" in [t["task_id"] for t in tasks]

    def test_fail_keeps_last_ok(self, tmp_path):
        root = tmp_path / "scratch"
        assert _probe_cli(root, "create", "--id", "probe", "-q").returncode == 0
        assert _probe_cli(root, "advance", "probe", "--result", "pass", "-q").returncode == 0
        assert _probe_show(root, "probe")["last_successful_iteration"] == 1
        assert _probe_cli(root, "advance", "probe", "--result", "fail", "-q").returncode == 0
        assert _probe_show(root, "probe")["last_successful_iteration"] == 1

    def test_exit_codes(self, tmp_path):
        root = tmp_path / "scratch"
        assert _probe_cli(root, "show", "no-such-task").returncode == 4
        assert _probe_cli(root, "show", "no-such-task", "--no-such-flag").returncode == 2
