"""Regression tests for defects found in the 0.16.0 audit."""

from __future__ import annotations

import hashlib
import shutil
import tempfile

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli  # noqa: E402
from hermes_memory_lancedb_pro.store import (  # noqa: E402
    ARCHIVED_STATE,
    VECTOR_DIM,
    MemoryStore,
)

pytestmark = pytest.mark.integration


class StubEmbedder:
    """Cheap deterministic embedder — no model download required."""

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text: str, normalize: bool) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        if normalize:
            n = sum(f * f for f in floats) ** 0.5
            if n > 0:
                floats = [f / n for f in floats]
        return floats


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-audit-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def _raw(store: MemoryStore, mem_id: str) -> dict:
    rows = store._table.search().where(f"id = '{mem_id}'").limit(1).to_list()
    return store._row_to_dict(rows[0])


# ---------------------------------------------------------------------------
# Smoke test must never delete memories it did not write
# ---------------------------------------------------------------------------


def test_smoke_run_leaves_user_memories_alone(store, monkeypatch, capsys):
    texts = [f"user memory number {i} about topic {i * 7}" for i in range(12)]
    ids = [store.store(text=t, category="fact", scope="global") for t in texts]
    monkeypatch.setattr(_cli, "MemoryStore", lambda db_path=None: store)
    assert _cli._run(store.db_path) == 0
    assert "0 failed" in capsys.readouterr().out
    assert all(store.has_id(i) for i in ids)
    remaining = store.list_memories(limit=100, include_archived=True)
    assert sorted(r["text"] for r in remaining) == sorted(texts)  # no smoke rows left


# ---------------------------------------------------------------------------
# increment_access_count via a superseded id
# ---------------------------------------------------------------------------


def test_access_count_on_superseded_id_does_not_resurrect_it(store):
    old = store.store(text="my favourite colour is blue", category="preference")
    new = store.update(old, text="my favourite colour is green")
    assert new and new != old
    assert store.increment_access_count(old, force=True) is True
    assert _raw(store, old)["metadata"]["state"] == ARCHIVED_STATE
    live = _raw(store, new)
    assert live["metadata"].get("state") != ARCHIVED_STATE
    assert live["metadata"]["access_count"] == 1
    assert [r["text"] for r in store.list_memories(limit=10)] == ["my favourite colour is green"]


# ---------------------------------------------------------------------------
# list_memories must not come back short when filtered rows precede live ones
# ---------------------------------------------------------------------------


def test_list_memories_finds_rows_behind_many_archived(store):
    mem = store.store(text="version 0", category="fact")
    for i in range(1, 70):
        mem = store.update(mem, text=f"version {i}")
    other = store.store(text="another live row", category="fact")
    listed = store.list_memories(limit=5)
    assert {r["id"] for r in listed} == {mem, other}


def test_list_memories_finds_later_session_rows(store):
    for i in range(130):
        store.store(text=f"session A row {i}", category="fact",
                    metadata_extra={"source_session": "A"})
    for i in range(5):
        store.store(text=f"session B row {i}", category="fact",
                    metadata_extra={"source_session": "B"})
    listed = store.list_memories(limit=5, session_id="B")
    assert len(listed) == 5 and all("session B" in r["text"] for r in listed)


def test_list_memories_offset_pages_do_not_overlap(store):
    ids = [store.store(text=f"row {i}", category="fact") for i in range(30)]
    for i in ids[:20:2]:
        store.update(i, text=f"edited {i}")
    page1 = store.list_memories(limit=10, offset=0)
    page2 = store.list_memories(limit=10, offset=10)
    page3 = store.list_memories(limit=10, offset=20)
    seen = [r["id"] for r in page1 + page2 + page3]
    assert len(seen) == 30 and len(set(seen)) == 30
