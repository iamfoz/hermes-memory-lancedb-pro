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
        # update() returns the new ID after supersede (not bool)
        new_id = store.update(mid, text="updated text content here", tier="core")
        assert isinstance(new_id, str)
        assert new_id != mid
        # Original is now archived
        assert store.has_id(mid) is False
        # Can still fetch the archived row by id (audit) using follow_chain=False
        archived = store.get_by_id(mid, follow_chain=False)
        assert archived["metadata"]["state"] == "archived"
        assert archived["metadata"]["superseded_by"] == new_id
        new_row = store.get_by_id(new_id)
        assert new_row is not None
        assert new_row["text"] == "updated text content here"
        assert new_row["metadata"]["tier"] == "core"
        assert new_row["metadata"]["supersedes"] == mid
        assert new_row["metadata"]["access_count"] == 0
        # The README workflow chains has_id on the return value
        assert store.has_id(new_id)

    def test_update_metadata_only(self, store):
        mid = store.store(text="immutable text content")
        # Metadata-only update returns the same id (no supersede, no new row)
        ret = store.update(mid, importance=0.95, tier="core")
        assert ret == mid
        row = store.get_by_id(mid)
        assert row["importance"] == pytest.approx(0.95)
        assert row["metadata"]["tier"] == "core"
        # Top-level field should NOT be mirrored into metadata (avoids drift)
        assert "importance" not in row["metadata"]
        assert "category" not in row["metadata"]

    def test_update_unknown_id(self, store):
        assert store.update("not-a-real-id", text="anything") is None

    def test_forget(self, store):
        mid = store.store(text="will be deleted soon")
        assert store.forget(mid) is True
        assert store.get_by_id(mid) is None
        assert store.forget(mid) is False  # already gone

    def test_increment_access_count(self, store):
        mid = store.store(text="counts accesses correctly")
        # First access always lands (count was 0).
        assert store.increment_access_count(mid)
        # Second rapid access is throttled — that's the feedback-loop fix.
        # Pass throttle_seconds=0 to mimic the legacy "always increment" path
        # for this test's intent (verifying the counter reaches 2).
        assert store.increment_access_count(mid, throttle_seconds=0)
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

    def test_search_results_have_normalised_score(self, store):
        # Every result must carry a normalised `score` field in [0, 1]
        # regardless of which underlying mode produced it. The raw
        # mode-specific fields (`_rrf_score`, `_distance`, `_score`)
        # are still preserved for advanced callers.
        store.store_many([
            {"text": f"beta normalised score entry {i} content here", "importance": 0.5}
            for i in range(3)
        ])
        for mode in ("vector", "bm25", "hybrid"):
            results = store.search("beta", limit=5, mode=mode)
            assert results, f"no results for mode={mode}"
            for r in results:
                assert "score" in r, f"mode={mode} missing normalised score field"
                assert isinstance(r["score"], float), f"mode={mode} score not a float"
                assert 0.0 <= r["score"] <= 1.0, (
                    f"mode={mode} score out of bounds: {r['score']}"
                )

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


class TestCompaction:
    """Fragment compaction — each write creates a new on-disk fragment, and
    a store left uncompacted exhausts the file-descriptor limit on read."""

    def test_optimize_preserves_rows(self, store):
        ids = store.store_many([{"text": f"row number {i}"} for i in range(20)])
        assert store.optimize() is True
        # Compaction never changes the row count, and every row survives.
        assert len(store._table) == 20
        for mid in ids:
            assert store.get_by_id(mid) is not None

    def test_optimize_safe_on_empty_table(self, store):
        assert store.optimize() is True
        assert len(store._table) == 0

    def test_auto_optimize_fires_after_threshold(self, store, monkeypatch):
        import hermes_memory_lancedb_pro.store as store_mod

        monkeypatch.setattr(store_mod, "AUTO_OPTIMIZE_EVERY", 10)
        calls: list[int] = []
        real_optimize = store.optimize
        monkeypatch.setattr(
            store, "optimize", lambda: (calls.append(1), real_optimize())[1]
        )

        for i in range(25):
            store.store(text=f"auto compaction entry {i}")

        # 25 writes at threshold 10 -> compaction elected at write 10 and 20.
        assert len(calls) == 2
        assert len(store._table) == 25
        assert store._writes_since_optimize == 5

    def test_auto_optimize_disabled_when_zero(self, store, monkeypatch):
        import hermes_memory_lancedb_pro.store as store_mod

        monkeypatch.setattr(store_mod, "AUTO_OPTIMIZE_EVERY", 0)
        calls: list[int] = []
        monkeypatch.setattr(store, "optimize", lambda: calls.append(1))

        for i in range(15):
            store.store(text=f"no compaction entry {i}")

        assert calls == []
        assert len(store._table) == 15

    def test_concurrent_writes_with_compaction_keep_exact_count(
        self, store, monkeypatch
    ):
        """Compaction running mid-stream against concurrent writers must not
        drop or duplicate rows."""
        from concurrent.futures import ThreadPoolExecutor

        import hermes_memory_lancedb_pro.store as store_mod

        monkeypatch.setattr(store_mod, "AUTO_OPTIMIZE_EVERY", 16)

        def worker(tid: int) -> None:
            for i in range(40):
                store.store(text=f"thread {tid} entry {i}")

        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(worker, range(6)))

        assert len(store._table) == 6 * 40
