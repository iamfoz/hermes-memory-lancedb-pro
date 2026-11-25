"""Regression tests for defects found in review of the open-item fixes."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import store as store_module  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration


class StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False, **_):
        if isinstance(text, str):
            return self._one(text)
        return [self._one(t) for t in text]

    @staticmethod
    def _one(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        n = sum(f * f for f in floats) ** 0.5
        return [f / n for f in floats]


def _open(path: str) -> MemoryStore:
    s = MemoryStore(db_path=path)
    s._embedder = StubEmbedder()
    s._initialise()
    return s


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="hermes-review-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(db_path):
    return _open(db_path)


@pytest.mark.parametrize("session_id", [
    "plain_sid", "pct%x", "tg:123/abc", 'quo"te', "back\\slash", "café", "日本", "it's",
])
def test_session_anchors_match_any_session_id(store, session_id):
    store.store(text="written in this session", category="fact",
                metadata_extra={"source_session": session_id})
    store.store(text="another session", category="fact",
                metadata_extra={"source_session": "other"})
    recent = store.recent_for_session(session_id, strict=True)
    first, last = store.anchors_for_session(session_id, strict=True)
    assert [r["text"] for r in recent] == ["written in this session"]
    assert [r["text"] for r in first] == [r["text"] for r in last] == ["written in this session"]


@pytest.mark.skipif(store_module.fcntl is None, reason="flock needs POSIX fcntl")
def test_reembed_refuses_while_an_agent_holds_the_store(db_path):
    import os

    seed = _open(db_path)
    seed.store(text="a memory", category="fact")
    # Rows but no scheme marker: a database written before prefixes existed.
    os.remove(os.path.join(db_path, store_module.EMBEDDING_SCHEME_FILE))

    agent = _open(db_path)
    assert agent.hold_in_use_lock() is True
    assert agent.hold_in_use_lock() is True  # idempotent

    cli = _open(db_path)
    assert cli.reembed(dry_run=True)["status"] == "dry_run"
    with pytest.raises(store_module.StoreInUseError, match="in use by a running Hermes agent"):
        cli.reembed()
    # Nothing changed, and nothing is left pending.
    assert cli.reembed(dry_run=True)["status"] == "dry_run"
    assert not cli.reembed_interrupted()


@pytest.mark.skipif(store_module.fcntl is None, reason="flock needs POSIX fcntl")
def test_agent_cannot_take_the_store_during_a_conversion(db_path):
    converting = _open(db_path)
    with converting._exclusive_use():
        assert _open(db_path).hold_in_use_lock() is False
    assert _open(db_path).hold_in_use_lock() is True


def test_reembed_names_duplicate_ids_instead_of_failing_obscurely(store, monkeypatch):
    mem = store.store(text="a memory", category="fact")
    monkeypatch.setattr(
        store, "_id_text_rows",
        lambda: [{"id": mem, "text": "a memory"}, {"id": mem, "text": "a memory"}],
    )
    with pytest.raises(ValueError, match="stored more than once"):
        store._reembed_passes("plain", "task_prefix", 64, None)


def test_only_official_models_get_the_remote_code_retry(store):
    calls: list[dict] = []

    class NeedsRemoteCode:
        def __init__(self, name, **kwargs):
            calls.append(kwargs)
            if not kwargs.get("trust_remote_code"):
                raise ValueError(
                    "requires you to execute the configuration file in that repo; "
                    "set trust_remote_code=True")

    store.embedding_model_name = "someone/nomic-embed-text-finetune"
    with pytest.raises(ValueError):
        store._new_model(NeedsRemoteCode)
    assert all(not c.get("trust_remote_code") for c in calls)

    calls.clear()
    store.embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
    store._new_model(NeedsRemoteCode)
    assert calls[-1].get("trust_remote_code") is True


def test_stale_reflection_rows_are_deleted_in_one_batch(store, monkeypatch):
    from hermes_memory_lancedb_pro.reflection import store as rstore

    ids = [store.store(text=f"reflection row {i}", category="reflection", scope="agent")
           for i in range(12)]
    monkeypatch.setattr(rstore, "_is_stale_reflection_row", lambda *a, **k: True)
    batches: list[int] = []
    real = store._delete_ids
    monkeypatch.setattr(store, "_delete_ids", lambda chunk: (batches.append(len(chunk)), real(chunk))[1])
    singles: list[str] = []
    monkeypatch.setattr(store, "delete", lambda mem_id: singles.append(mem_id) or True)
    assert rstore.purge_stale_reflection_rows(store) == 12
    assert batches == [12] and singles == []
    assert not any(store.row_exists(i) for i in ids)


def test_hybrid_min_score_uses_the_zero_to_one_score(store):
    store.store(text="the staging database lives on host db-7", category="fact")
    hits = store.search("staging database host", limit=5, mode="hybrid", min_score=0.2)
    assert hits and all(h["score"] >= 0.2 for h in hits)
    assert store.search("staging database host", limit=5, mode="hybrid", min_score=1.01) == []


def test_lost_write_ticket_does_not_wedge_later_writes(store, monkeypatch):
    from hermes_memory_lancedb_pro import provider

    class FakeBase:
        pass

    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: FakeBase)
    monkeypatch.setattr(provider, "_spawn_warmup", lambda store: None)
    p = provider._build_provider_class()(store=store, auto_smart_extraction=False)
    p.initialize("sess-1")

    real_spawn = provider._spawn_context_thread
    state = {"raised": False}

    def interrupted(*args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(provider, "_spawn_context_thread", interrupted)
    with pytest.raises(KeyboardInterrupt):
        p.sync_turn("remember the staging host is db-7", "Noted.")

    done = threading.Event()
    worker = threading.Thread(
        target=lambda: (p.sync_turn("and the port is 5432", "Noted."), done.set()))
    worker.start()
    assert done.wait(15), "a later write is wedged behind the lost ticket"
    worker.join(5)
    assert p._write_done == p._write_issued


def test_merge_into_an_orphaned_archived_row_keeps_the_new_fact():
    from hermes_memory_lancedb_pro.smart_extractor import _is_archived_row, _row_similarity

    assert _is_archived_row({"metadata": {"state": "archived"}})
    assert _is_archived_row({"metadata": '{"state": "archived"}'})
    assert not _is_archived_row({"metadata": {"state": "confirmed"}})
    assert _row_similarity({"_distance": 0.25}) == pytest.approx(0.75)
    assert _row_similarity({"score": 0.4}) == pytest.approx(0.4)
    assert _row_similarity({}) is None  # unknown, never a perfect match
