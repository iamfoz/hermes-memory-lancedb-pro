"""sync_turn write path: ordering, no orphaned writes, and drains that cover
every pending write (inline on the host's worker, or background on the main
thread)."""

from __future__ import annotations

import concurrent.futures
import contextvars
import hashlib
import logging
import shutil
import tempfile
import threading
import time

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import provider  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402


class StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text)
        return [self._one(t) for t in text]

    def _one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-sync-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


class RecordingExtractor:
    """Records each write; a write whose text is in `gates` blocks until that
    gate is set. Tracks how many writes ran at once."""

    has_llm = False

    def __init__(self, store, delay: float = 0.0):
        self.store = store
        self.delay = delay
        self.gates: dict[str, threading.Event] = {}
        self.started: list[str] = []
        self.finished: list[str] = []
        self.context_values: list[str] = []
        self.started_event = threading.Event()
        self._lock = threading.Lock()
        self._running = 0
        self.max_running = 0

    def extract_and_persist(self, *, user_content, assistant_content, session_key, scope, context):
        with self._lock:
            self._running += 1
            self.max_running = max(self.max_running, self._running)
            self.started.append(user_content)
        self.context_values.append(_PROFILE.get())
        self.started_event.set()
        try:
            gate = self.gates.get(user_content)
            if gate is not None:
                gate.wait(timeout=15)
            if self.delay:
                time.sleep(self.delay)
            self.store.store(text="EXTRACTED " + user_content, category="fact",
                             metadata_extra={"source_session": session_key})
        finally:
            with self._lock:
                self._running -= 1
                self.finished.append(user_content)


_PROFILE: contextvars.ContextVar[str] = contextvars.ContextVar("profile", default="default")


def _make_provider(store, monkeypatch, extractor):
    class FakeBase:
        pass

    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: FakeBase)
    monkeypatch.setattr(provider, "_spawn_warmup", lambda store: None)
    monkeypatch.setattr(provider, "_maybe_auto_purge", lambda store: None)
    monkeypatch.setattr(provider, "_maybe_auto_compact", lambda store: None)
    monkeypatch.setattr(provider, "_maybe_auto_task_gc", lambda *a: None)
    p = provider._build_provider_class()(
        store=store, smart_extractor=extractor, auto_smart_extraction=False,
    )
    p.initialize("sess-1")
    return p


def _release_later(event: threading.Event, delay: float) -> None:
    threading.Timer(delay, event.set).start()


def test_main_thread_writes_run_one_at_a_time_in_order(store, monkeypatch):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    ext.gates["turn one"] = threading.Event()

    p.sync_turn("turn one", "ok")
    assert ext.started_event.wait(5)
    t0 = time.monotonic()
    p.sync_turn("turn two", "ok")
    p.sync_turn("turn three", "ok")
    # The caller is not blocked behind the slow write ...
    assert time.monotonic() - t0 < 1.0
    time.sleep(0.3)
    # ... and later writes do not start while it is still running.
    assert ext.started == ["turn one"]

    ext.gates["turn one"].set()
    assert p._drain_pending_writes("test")
    assert ext.finished == ["turn one", "turn two", "turn three"]
    assert ext.max_running == 1


def test_shutdown_waits_for_every_pending_write_not_just_the_newest(store, monkeypatch):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    gate = ext.gates["slow first turn"] = threading.Event()

    p.sync_turn("slow first turn", "ok")
    assert ext.started_event.wait(5)
    p.sync_turn("quick second turn", "ok")
    _release_later(gate, 0.5)
    p.shutdown()

    assert ext.finished == ["slow first turn", "quick second turn"]
    texts = {r["text"] for r in store.list_memories(limit=20, category="fact")}
    assert texts == {"EXTRACTED slow first turn", "EXTRACTED quick second turn"}


def test_on_session_end_waits_for_every_pending_write(store, monkeypatch):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    gate = ext.gates["slow turn"] = threading.Event()

    p.sync_turn("slow turn", "ok")
    assert ext.started_event.wait(5)
    p.sync_turn("last turn", "ok")
    _release_later(gate, 0.4)
    p.on_session_end([])
    assert ext.finished == ["slow turn", "last turn"]


def test_sync_turn_off_the_main_thread_writes_inline(store, monkeypatch):
    """The host calls sync_turn on its single memory worker and treats return
    as done (flush_pending, shutdown drain). The write must be finished then."""
    ext = RecordingExtractor(store, delay=0.5)
    p = _make_provider(store, monkeypatch, ext)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(p.sync_turn, "remember the deploy key lives in vault", "ok").result(10)
        assert ext.finished == ["remember the deploy key lives in vault"]
        texts = [r["text"] for r in store.list_memories(limit=20, category="fact")]
        assert texts == ["EXTRACTED remember the deploy key lives in vault"]


def test_inline_write_waits_for_an_earlier_background_write(store, monkeypatch):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    gate = ext.gates["main-thread turn"] = threading.Event()

    p.sync_turn("main-thread turn", "ok")
    assert ext.started_event.wait(5)
    _release_later(gate, 0.3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(p.sync_turn, "worker turn", "ok").result(10)
    assert ext.finished == ["main-thread turn", "worker turn"]
    assert ext.max_running == 1


def test_background_write_runs_under_the_callers_contextvars(store, monkeypatch):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    token = _PROFILE.set("work-profile")
    try:
        p.sync_turn("profile scoped turn", "ok")
    finally:
        _PROFILE.reset(token)
    assert p._drain_pending_writes("test")
    assert ext.context_values == ["work-profile"]


def test_drain_logs_when_it_gives_up(store, monkeypatch, caplog):
    ext = RecordingExtractor(store)
    p = _make_provider(store, monkeypatch, ext)
    gate = ext.gates["wedged turn"] = threading.Event()
    monkeypatch.setattr(provider, "_WRITE_DRAIN_TIMEOUT_S", 0.2)

    p.sync_turn("wedged turn", "ok")
    assert ext.started_event.wait(5)
    with caplog.at_level(logging.WARNING, logger=provider.logger.name):
        p.on_session_end([])
    gate.set()
    assert any(
        "on_session_end: gave up" in r.getMessage() and "1 pending" in r.getMessage()
        for r in caplog.records
    )
    assert p._drain_pending_writes("test")


def test_a_failing_write_does_not_block_later_writes(store, monkeypatch):
    p = _make_provider(store, monkeypatch, None)
    calls: list[str] = []

    def flaky_raw(user_content, *_a, **_kw):
        calls.append(user_content)
        if len(calls) == 1:
            raise RuntimeError("disk full")

    monkeypatch.setattr(p, "_raw_sync_turn", flaky_raw)
    p.sync_turn("first turn fails", "ok")
    p.sync_turn("second turn lands", "ok")
    assert p._drain_pending_writes("test")
    assert calls == ["first turn fails", "second turn lands"]


def test_real_host_flush_and_shutdown_cover_a_slow_extraction(store, tmp_path, monkeypatch):
    """Driven through the host's MemoryManager: flush_pending() and
    shutdown_all() must not return before a slow write has landed."""
    mm_mod = pytest.importorskip("agent.memory_manager")
    if provider._self_check() != "real":
        pytest.skip("hermes-agent not on PYTHONPATH")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    monkeypatch.setattr(provider, "_spawn_warmup", lambda store: None)
    for fn in ("_maybe_auto_purge", "_maybe_auto_compact"):
        monkeypatch.setattr(provider, fn, lambda store: None)
    monkeypatch.setattr(provider, "_maybe_auto_task_gc", lambda *a: None)

    ext = RecordingExtractor(store, delay=1.0)
    p = provider.LanceDBProMemoryProvider(
        store=store, smart_extractor=ext, auto_smart_extraction=False,
    )
    mm = mm_mod.MemoryManager()
    mm.add_provider(p)
    mm.initialize_all("s1", hermes_home=str(tmp_path / "hh"), platform="cli")

    mm.sync_all("my production database lives on host db-prod-7", "noted", session_id="s1")
    assert mm.flush_pending(30) is True
    assert ext.finished == ["my production database lives on host db-prod-7"]

    mm.sync_all("the staging database lives on host db-stg-2", "noted", session_id="s1")
    # The host drains its worker for 5s at most, then calls provider.shutdown().
    monkeypatch.setattr(mm_mod, "_SYNC_DRAIN_TIMEOUT_S", 0.1)
    mm.shutdown_all()
    assert ext.finished[-1] == "the staging database lives on host db-stg-2"
    texts = {r["text"] for r in store.list_memories(limit=20, category="fact")}
    assert "EXTRACTED the staging database lives on host db-stg-2" in texts
