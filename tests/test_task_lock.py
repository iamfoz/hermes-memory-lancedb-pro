"""Every task-ledger state write holds an exclusive flock on <task dir>/.lock.

Concurrent writers (the agent's CLI calls, an unattended scheduler, other
processes) must never lose each other's updates. The vacuity guard below
disables the lock and widens the read-modify-write window to show that the
concurrency tests would catch a missing lock.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time

import pytest

from hermes_memory_lancedb_pro import task_ledger as _tl

pytestmark = pytest.mark.skipif(_tl.fcntl is None, reason="flock needs POSIX fcntl")

THREADS = 8
PER_THREAD = 10


def _slow_reads(monkeypatch, delay: float = 0.005) -> None:
    """Widen the read-modify-write window so a missing lock loses writes."""
    real = _tl.load_state

    def slow(task_id, root=None):
        state = real(task_id, root)
        time.sleep(delay)
        return state

    monkeypatch.setattr(_tl, "load_state", slow)


def _hammer(fn) -> None:
    errors: list[BaseException] = []
    barrier = threading.Barrier(THREADS)

    def worker():
        barrier.wait()
        try:
            for _ in range(PER_THREAD):
                fn()
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "writer hung (deadlock?)"
    assert errors == []


def test_unlocked_writes_are_lost(tmp_path, monkeypatch):
    # Vacuity guard: the same workload without the lock must lose updates.
    _tl.create_task("t", "o", root=tmp_path)
    monkeypatch.setattr(_tl, "fcntl", None)
    _slow_reads(monkeypatch)
    _hammer(lambda: _tl.wake_task("t", root=tmp_path))
    assert _tl.load_state("t", root=tmp_path)["wake_count"] < THREADS * PER_THREAD


def test_concurrent_advances_lose_nothing(tmp_path, monkeypatch):
    _tl.create_task("t", "o", root=tmp_path)
    _slow_reads(monkeypatch)
    _hammer(lambda: _tl.advance_iteration("t", root=tmp_path))
    state = _tl.load_state("t", root=tmp_path)
    total = THREADS * PER_THREAD
    assert state["current_iteration"] == total
    assert state["last_successful_iteration"] == total
    results = (tmp_path / "t" / "results.jsonl").read_text().splitlines()
    assert sorted(json.loads(r)["iteration"] for r in results) == list(range(total))


def test_mixed_writers_lose_nothing(tmp_path, monkeypatch):
    _tl.create_task("t", "o", root=tmp_path)
    _slow_reads(monkeypatch)
    ops = [
        lambda: _tl.wake_task("t", root=tmp_path),
        lambda: _tl.advance_iteration("t", result="fail", root=tmp_path),
        lambda: _tl.set_task_hold("t", True, root=tmp_path),
    ]
    counter = iter(range(10**6))

    def next_op():  # each call runs the next op in turn: wake, advance, hold
        ops[next(counter) % len(ops)]()

    _hammer(next_op)
    state = _tl.load_state("t", root=tmp_path)
    n = THREADS * PER_THREAD
    expected_wakes = len([i for i in range(n) if i % 3 == 0])
    expected_advances = len([i for i in range(n) if i % 3 == 1])
    assert state["wake_count"] == expected_wakes
    assert state["current_iteration"] == expected_advances
    assert state["gc_hold"] is True


def test_writer_waits_for_the_lock(tmp_path):
    _tl.create_task("t", "o", root=tmp_path)
    done = threading.Event()

    def writer():
        _tl.wake_task("t", root=tmp_path)
        done.set()

    with _tl.task_lock("t", root=tmp_path):
        t = threading.Thread(target=writer)
        t.start()
        assert not done.wait(0.3), "write went ahead while the lock was held"
    assert done.wait(10)
    t.join()
    assert _tl.load_state("t", root=tmp_path)["wake_count"] == 1


def test_lock_file_location(tmp_path):
    _tl.create_task("t", "o", root=tmp_path)
    assert (tmp_path / "t" / ".lock").is_file()


def test_locking_never_creates_a_missing_task(tmp_path):
    with pytest.raises(FileNotFoundError):
        _tl.wake_task("missing", root=tmp_path)
    assert not (tmp_path / "missing").exists()


def test_concurrent_create_has_one_winner(tmp_path):
    outcomes: list[str] = []
    barrier = threading.Barrier(THREADS)

    def creator(i):
        barrier.wait()
        try:
            _tl.create_task("t", f"objective {i}", root=tmp_path)
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=creator, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert outcomes.count("created") == 1 and outcomes.count("exists") == THREADS - 1


def _process_worker(root: str, n: int) -> None:
    from pathlib import Path

    from hermes_memory_lancedb_pro import task_ledger as tl

    for _ in range(n):
        tl.advance_iteration("t", root=Path(root))


def test_concurrent_processes_lose_nothing(tmp_path):
    _tl.create_task("t", "o", root=tmp_path)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_process_worker, args=(str(tmp_path), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    assert _tl.load_state("t", root=tmp_path)["current_iteration"] == 100
