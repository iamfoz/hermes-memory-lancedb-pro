"""Embedder calls must never overlap: torch's MPS backend is not thread-safe.

On Apple Silicon the provider's warmup thread (first encode: model load +
Metal shader JIT) raced the first turn's prefetch encode on the same model,
and two concurrent encodes aborted the gateway in Metal ("A command encoder is
already encoding to this command buffer"). These tests use an embedder that,
like Metal, fails if a second call enters while one is running, so a missing
lock fails deterministically rather than only under unlucky timing.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
import time
import types

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import store as store_module  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

THREADS = 8
HOLD_S = 0.02  # how long each call stays "inside" the embedder


class OverlapDetector:
    """Shared across embedders: counts calls currently inside any of them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.overlaps = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            overlapping = self.active > 1
            if overlapping:
                self.overlaps += 1
        if overlapping:
            self.exit()
            # What Metal does, minus taking the process down.
            raise RuntimeError("A command encoder is already encoding to this command buffer")

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


class MetalLikeEmbedder:
    """Deterministic 768-d encoder that refuses concurrent use."""

    def __init__(self, detector: OverlapDetector) -> None:
        self.detector = detector
        self.calls = 0

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        self.detector.enter()
        try:
            time.sleep(HOLD_S)
            self.calls += 1
            if isinstance(text, str):
                return [float(len(text) % 7)] * VECTOR_DIM
            return [[float(len(t) % 7)] * VECTOR_DIM for t in text]
        finally:
            self.detector.exit()


def _store(embedder=None) -> MemoryStore:
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-encode-lock-"))
    if embedder is not None:
        s._embedder = embedder
    return s


def _run_together(*fns, rounds: int = 3) -> list[BaseException]:
    """Start every fn at the same instant (barrier), `rounds` times; collect errors."""
    errors: list[BaseException] = []
    for _ in range(rounds):
        barrier = threading.Barrier(len(fns))

        def call(fn, barrier=barrier):
            barrier.wait()
            try:
                fn()
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=call, args=(fn,)) for fn in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "embedder call hung (deadlock?)"
    return errors


def test_detector_catches_unlocked_concurrent_encodes():
    # Guard against a vacuous pass: calling the model directly (no store
    # lock) from several threads at once must trip the detector.
    detector = OverlapDetector()
    model = MetalLikeEmbedder(detector)
    errors = _run_together(*[lambda: model.encode("x")] * THREADS)
    assert detector.overlaps > 0 and errors


def test_concurrent_encode_calls_are_serialised():
    detector = OverlapDetector()
    s = _store(MetalLikeEmbedder(detector))
    results: list[list[float]] = []
    errors = _run_together(*[lambda: results.append(s.encode("hello"))] * THREADS)
    assert errors == [] and detector.overlaps == 0 and detector.max_active == 1
    assert len(results) == THREADS * 3 and all(len(v) == VECTOR_DIM for v in results)


def test_warmup_racing_first_prefetch_encode():
    # The production crash: provider.initialize() starts a warmup thread
    # while the first turn's prefetch thread encodes the query.
    detector = OverlapDetector()
    s = _store(MetalLikeEmbedder(detector))
    errors = _run_together(s.warmup, lambda: s.encode("first user turn"))
    assert errors == [] and detector.overlaps == 0


def test_encode_batch_and_encode_are_serialised():
    detector = OverlapDetector()
    s = _store(MetalLikeEmbedder(detector))
    errors = _run_together(
        lambda: s.encode_batch(["a", "bb", "ccc"]),
        lambda: s.encode("query"),
        lambda: s.encode_batch(["dddd"]),
        s.warmup,
    )
    assert errors == [] and detector.overlaps == 0


def test_two_stores_never_encode_at_once():
    # One gateway can hold a store per Hermes profile, each with its own
    # model, but the MPS stream is process-wide: the lock must be too.
    detector = OverlapDetector()
    a = _store(MetalLikeEmbedder(detector))
    b = _store(MetalLikeEmbedder(detector))
    assert a._encode_lock is b._encode_lock is store_module._ENCODE_LOCK
    errors = _run_together(*[lambda: a.encode("a"), lambda: b.encode("b")] * 4)
    assert errors == [] and detector.overlaps == 0


# ---------------------------------------------------------------------------
# Model loading and MEMORY_EMBEDDING_DEVICE
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a stand-in `sentence_transformers` module and record each
    construction's kwargs. The class shares the overlap detector so loading
    can be checked against concurrent encodes."""
    detector = OverlapDetector()
    constructed: list[dict] = []

    class FakeSentenceTransformer(MetalLikeEmbedder):
        fail_devices: set[str] = set()

        def __init__(self, name, device=None):
            if device in self.fail_devices:
                raise RuntimeError(f"device {device!r} is not available")
            detector.enter()  # weights moving onto the device
            try:
                time.sleep(HOLD_S)
            finally:
                detector.exit()
            super().__init__(detector)
            self.device = device or "cpu"
            constructed.append({"name": name, "device": device})

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return types.SimpleNamespace(cls=FakeSentenceTransformer, detector=detector,
                                 constructed=constructed)


def test_model_load_is_serialised_with_other_stores_encodes(fake_sentence_transformers):
    fake = fake_sentence_transformers
    loaded = _store(MetalLikeEmbedder(fake.detector))
    loading = _store()  # no embedder yet: its first encode loads the model
    errors = _run_together(*[lambda: loaded.encode("busy")] * 3,
                           lambda: loading.encode("first"), rounds=1)
    assert errors == [] and fake.detector.overlaps == 0
    assert len(fake.constructed) == 1


def test_device_override_is_passed_to_the_model(fake_sentence_transformers, monkeypatch):
    monkeypatch.setattr(store_module, "EMBEDDING_DEVICE", "cpu")
    s = _store()
    s.encode("x")
    assert fake_sentence_transformers.constructed == [
        {"name": s.embedding_model_name, "device": "cpu"}
    ]
    assert s._embedding_device_label() == "cpu"


def test_no_override_leaves_device_selection_automatic(fake_sentence_transformers, monkeypatch):
    monkeypatch.setattr(store_module, "EMBEDDING_DEVICE", None)
    s = _store()
    assert s._embedding_device_label() == "auto"  # not loaded yet: never loads for stats
    assert fake_sentence_transformers.constructed == []
    s.encode("x")
    assert fake_sentence_transformers.constructed[0]["device"] is None


def test_unusable_device_falls_back_to_automatic(fake_sentence_transformers, monkeypatch, caplog):
    fake = fake_sentence_transformers
    fake.cls.fail_devices = {"cuda"}
    monkeypatch.setattr(store_module, "EMBEDDING_DEVICE", "cuda")
    s = _store()
    with caplog.at_level(logging.WARNING, logger=store_module.logger.name):
        vector = s.encode("still works")
    assert len(vector) == VECTOR_DIM
    assert [c["device"] for c in fake.constructed] == [None]
    assert "MEMORY_EMBEDDING_DEVICE='cuda' could not be used" in caplog.text


def test_env_var_is_read(monkeypatch):
    import importlib

    monkeypatch.setenv("MEMORY_EMBEDDING_DEVICE", "  mps ")
    try:
        assert importlib.reload(store_module).EMBEDDING_DEVICE == "mps"
        monkeypatch.setenv("MEMORY_EMBEDDING_DEVICE", "")
        assert importlib.reload(store_module).EMBEDDING_DEVICE is None
    finally:
        monkeypatch.delenv("MEMORY_EMBEDDING_DEVICE", raising=False)
        importlib.reload(store_module)
