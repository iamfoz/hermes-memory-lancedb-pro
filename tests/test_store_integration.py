"""Integration tests for MemoryStore.

These exercise the LanceDB layer with a stubbed embedder so they don't need
the (large) sentence-transformers model. Mark them with `integration` so
hosts without LanceDB can skip them via `pytest -m "not integration"`.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro.store import (
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

    def _one(self, text: str, normalize: bool):
        # Hash → 32 bytes; tile to VECTOR_DIM and convert to floats in [-1, 1]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = []
        for i in range(VECTOR_DIM):
            b = digest[i % len(digest)]
            floats.append((b - 128) / 128.0)
        if normalize:
            n = sum(f * f for f in floats) ** 0.5
            if n > 0:
                floats = [f / n for f in floats]
        return floats


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp(prefix="hermes-test-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = StubEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestCrud:
    def test_store_and_get(self, store):
        mid = store.store(text="hello world", category="fact", importance=0.6)
        assert len(mid) == 36
        got = store.get_by_id(mid)
        assert got is not None
        assert got["text"] == "hello world"
        assert got["importance"] == pytest.approx(0.6)
        assert got["metadata"]["state"] == "confirmed"

    def test_store_validates_text(self, store):
        with pytest.raises(ValueError):
            store.store(text="")
        with pytest.raises(ValueError):
            store.store(text="   ")

    def test_store_clamps_importance(self, store):
        mid_high = store.store(text="x" * 30, importance=5.0)
        mid_low = store.store(text="y" * 30, importance=-5.0)
        assert store.get_by_id(mid_high)["importance"] == 1.0
        assert store.get_by_id(mid_low)["importance"] == 0.0

    def test_store_normalises_invalid_category(self, store):
        mid = store.store(text="hello world thirty chars", category="not_a_real_category")
        got = store.get_by_id(mid)
        assert got["category"] == "other"

    def test_store_many_bulk(self, store):
        ids = store.store_many([
            {"text": "first entry has enough content", "importance": 0.4},
            {"text": "second entry with some content", "importance": 0.6, "category": "fact"},
        ])
        assert len(ids) == 2
        assert all(len(i) == 36 for i in ids)
        for mid in ids:
            assert store.has_id(mid)

    def test_update_supersede(self, store):
        mid = store.store(text="original text content here")
        assert store.update(mid, text="updated text content here", tier="core") is True
        # Original is now archived
        assert store.has_id(mid) is False
        # Can still fetch the archived row by id (audit)
        archived = store.get_by_id(mid)
        assert archived["metadata"]["state"] == "archived"
        assert archived["metadata"]["superseded_by"] is not None
        new_id = archived["metadata"]["superseded_by"]
        new_row = store.get_by_id(new_id)
        assert new_row is not None
        assert new_row["text"] == "updated text content here"
        assert new_row["metadata"]["tier"] == "core"
        assert new_row["metadata"]["supersedes"] == mid
        assert new_row["metadata"]["access_count"] == 0

    def test_update_metadata_only(self, store):
        mid = store.store(text="immutable text content")
        assert store.update(mid, importance=0.95, tier="core") is True
        row = store.get_by_id(mid)
        assert row["importance"] == pytest.approx(0.95)
        assert row["metadata"]["tier"] == "core"
        # Top-level field should NOT be mirrored into metadata (avoids drift)
        assert "importance" not in row["metadata"]
        assert "category" not in row["metadata"]

    def test_update_unknown_id(self, store):
        assert store.update("not-a-real-id", text="anything") is False

    def test_forget(self, store):
        mid = store.store(text="will be deleted soon")
        assert store.forget(mid) is True
        assert store.get_by_id(mid) is None
        assert store.forget(mid) is False  # already gone

    def test_increment_access_count(self, store):
        mid = store.store(text="counts accesses correctly")
        assert store.increment_access_count(mid)
        assert store.increment_access_count(mid)
        meta = store.get_by_id(mid)["metadata"]
        assert meta["access_count"] == 2


class TestQueries:
    def test_check_ids_returns_active_only(self, store):
        ids = [
            store.store(text=f"entry number {i} content here padded out")
            for i in range(5)
        ]
        # Archive one via supersede
        store.update(ids[2], text="superseded by update text")
        confirmed = store.check_ids(ids + ["fake-id-not-real"])
        assert ids[0] in confirmed
        assert ids[1] in confirmed
        assert ids[2] not in confirmed   # archived
        assert ids[3] in confirmed
        assert ids[4] in confirmed
        assert "fake-id-not-real" not in confirmed

    def test_check_ids_empty(self, store):
        assert store.check_ids([]) == []

    def test_list_memories_filters_archived(self, store):
        active = store.store(text="kept active in the listing")
        archived = store.store(text="will be archived after update")
        store.update(archived, text="now superseded by new entry")

        listed_ids = [r["id"] for r in store.list_memories(limit=20)]
        assert active in listed_ids
        assert archived not in listed_ids

        # include_archived returns both
        listed_all = [r["id"] for r in store.list_memories(limit=20, include_archived=True)]
        assert archived in listed_all

    def test_list_memories_combines_filters(self, store):
        store.store(text="fact entry global scope content", category="fact", scope="global")
        store.store(text="other entry global scope content", category="other", scope="global")
        store.store(text="fact entry user scope content here", category="fact", scope="user")

        # AND of category+scope — used to silently drop one filter
        results = store.list_memories(category="fact", scope="global", limit=20)
        assert len(results) == 1
        assert results[0]["category"] == "fact"
        assert results[0]["scope"] == "global"

    def test_categories_alias_for_category(self, store):
        store.store(text="alias test entry content here", category="preference")
        store.store(text="other test entry content here", category="other")
        # `categories` (plural) is what Hermes core sometimes passes
        results = store.list_memories(categories="preference", limit=10)
        assert len(results) == 1
        assert results[0]["category"] == "preference"

    def test_search_returns_dicts_for_all_modes(self, store):
        store.store_many([
            {"text": f"alpha entry number {i} content", "importance": 0.5}
            for i in range(3)
        ])
        for mode in ("vector", "bm25", "hybrid"):
            results = store.search("alpha", limit=5, mode=mode)
            assert isinstance(results, list)
            assert all(isinstance(r, dict) for r in results), f"mode={mode} returned non-dict"

    def test_search_empty_query(self, store):
        store.store(text="some real content here for search")
        assert store.search("", limit=5) == []
        assert store.search("   ", limit=5) == []

    def test_stats_has_active_and_archived_counts(self, store):
        store.store(text="active entry one with content")
        b = store.store(text="will be archived shortly text")
        store.update(b, text="superseded entry text content")

        stats = store.stats()
        assert stats["total_memories"] >= 3  # 2 visible + 1 archived
        assert stats["active_memories"] >= 2
        assert stats["archived_memories"] >= 1
        assert stats["vector_dimensions"] == VECTOR_DIM


class TestSingleton:
    def test_singleton_keyed_by_path(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            s_a = MemoryStore.get_instance(db_path=a)
            s_a._embedder = StubEmbedder()
            s_b = MemoryStore.get_instance(db_path=b)
            s_b._embedder = StubEmbedder()
            assert s_a is not s_b
            assert s_a.db_path != s_b.db_path
            # Same path returns same instance
            s_a2 = MemoryStore.get_instance(db_path=a)
            assert s_a is s_a2
