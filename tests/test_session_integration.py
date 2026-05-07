"""Integration tests for session-scoped recall and the access-count throttle.

Reuses the StubEmbedder pattern from test_store_integration.py."""

from __future__ import annotations

import hashlib
import shutil
import tempfile

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro.store import (
    ACCESS_COUNT_THROTTLE_SECONDS,
    VECTOR_DIM,
    MemoryStore,
)

pytestmark = pytest.mark.integration


class StubEmbedder:
    """Deterministic embedder — no model download required."""

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text: str, normalize: bool):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
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


# ---------------------------------------------------------------------------
# Session scoping
# ---------------------------------------------------------------------------

class TestSessionScopedRecall:
    def test_session_id_filters_to_same_session(self, store):
        a = store.store(
            text="memory from session A about widgets",
            metadata_extra={"source_session": "sess-A"},
        )
        b = store.store(
            text="memory from session B about widgets",
            metadata_extra={"source_session": "sess-B"},
        )

        # No session_id: both surface
        ids = {r["id"] for r in store.search("widgets", limit=5)}
        assert {a, b} <= ids

        # Scoped to sess-A: only A
        ids = {r["id"] for r in store.search("widgets", limit=5, session_id="sess-A")}
        assert a in ids
        assert b not in ids

        # Scoped to sess-B: only B
        ids = {r["id"] for r in store.search("widgets", limit=5, session_id="sess-B")}
        assert b in ids
        assert a not in ids

    def test_cross_session_flag_bypasses_filter(self, store):
        always = store.store(
            text="cross-session widget knowledge surfaces everywhere",
            metadata_extra={"source_session": "sess-OLD", "cross_session": True},
        )
        scoped = store.store(
            text="scoped widget knowledge from sess-OLD only",
            metadata_extra={"source_session": "sess-OLD"},
        )

        ids = {r["id"] for r in store.search("widget", limit=5, session_id="sess-NEW")}
        assert always in ids
        assert scoped not in ids

    def test_core_tier_bypasses_filter(self, store):
        core = store.store(
            text="core widget knowledge surfaces across sessions",
            tier="core",
            metadata_extra={"source_session": "sess-OLD"},
        )
        working = store.store(
            text="working widget knowledge stays in its session",
            tier="working",
            metadata_extra={"source_session": "sess-OLD"},
        )

        ids = {r["id"] for r in store.search("widget", limit=5, session_id="sess-NEW")}
        assert core in ids
        assert working not in ids

    def test_session_filter_applies_to_all_modes(self, store):
        a = store.store(
            text="alpha SESSION_TEST_TOKEN content",
            metadata_extra={"source_session": "sess-A"},
        )
        store.store(
            text="bravo SESSION_TEST_TOKEN content",
            metadata_extra={"source_session": "sess-B"},
        )

        for mode in ("vector", "bm25", "hybrid"):
            ids = {
                r["id"]
                for r in store.search(
                    "SESSION_TEST_TOKEN", limit=10, mode=mode, session_id="sess-A"
                )
            }
            assert a in ids, f"mode={mode} dropped same-session match"
            # The other-session id must not appear
            assert all(
                r["metadata"].get("source_session") != "sess-B"
                for r in store.search(
                    "SESSION_TEST_TOKEN", limit=10, mode=mode, session_id="sess-A"
                )
            ), f"mode={mode} leaked other-session results"

    def test_session_filter_in_list_memories(self, store):
        a = store.store(
            text="A memory content", metadata_extra={"source_session": "sess-A"}
        )
        b = store.store(
            text="B memory content", metadata_extra={"source_session": "sess-B"}
        )

        scoped = {m["id"] for m in store.list_memories(limit=20, session_id="sess-A")}
        assert a in scoped
        assert b not in scoped


# ---------------------------------------------------------------------------
# Access-count throttle
# ---------------------------------------------------------------------------

class TestAccessCountThrottle:
    def test_first_access_always_increments(self, store):
        mid = store.store(text="fresh memory not yet accessed")
        # Even though last_accessed_at == created_at (both = now), the first
        # increment should land — that's the bug we fixed.
        assert store.increment_access_count(mid) is True
        meta = store.get_by_id(mid)["metadata"]
        assert meta["access_count"] == 1

    def test_repeat_access_throttled(self, store):
        mid = store.store(text="memory that gets recalled twice quickly")
        store.increment_access_count(mid)  # 0 -> 1
        # A rapid second increment must be blocked by the throttle
        assert store.increment_access_count(mid) is False
        meta = store.get_by_id(mid)["metadata"]
        assert meta["access_count"] == 1

    def test_force_bypasses_throttle(self, store):
        mid = store.store(text="memory that gets force-credited")
        store.increment_access_count(mid)  # 0 -> 1
        # force=True must always go through
        assert store.increment_access_count(mid, force=True) is True
        meta = store.get_by_id(mid)["metadata"]
        assert meta["access_count"] == 2

    def test_short_throttle_arg_allows_subsequent(self, store):
        mid = store.store(text="short-throttle memory")
        store.increment_access_count(mid, throttle_seconds=0)  # first
        # With a 0-second cooldown, a subsequent call should land
        store.increment_access_count(mid, throttle_seconds=0)
        meta = store.get_by_id(mid)["metadata"]
        assert meta["access_count"] >= 2

    def test_mark_recall_used_bypasses_throttle(self, store):
        ids = [
            store.store(text=f"recall-credited memory {i} content here")
            for i in range(3)
        ]
        # First credit lands — count goes 0 -> 1 for all three
        assert store.mark_recall_used(ids) == 3
        # Immediately credit again — even though the throttle would normally
        # block, mark_recall_used uses force=True
        assert store.mark_recall_used(ids) == 3
        for mid in ids:
            assert store.get_by_id(mid)["metadata"]["access_count"] == 2

    def test_throttle_seconds_default_is_5min(self):
        # Sanity: confirm the env-var default is 5 minutes
        assert ACCESS_COUNT_THROTTLE_SECONDS == 300


# ---------------------------------------------------------------------------
# min_score gate
# ---------------------------------------------------------------------------

class TestMinScore:
    def test_min_score_drops_weak_matches_in_hybrid(self, store):
        # Two memories — both will surface but with low rrf scores
        store.store(text="alpha entry one with content")
        store.store(text="bravo entry two with content")

        # An impossibly high min_score should drop everything
        results = store.search(
            "completely unrelated query about quantum mechanics",
            limit=5,
            mode="hybrid",
            min_score=0.99,
        )
        assert results == []

    def test_min_score_zero_no_op(self, store):
        store.store(text="some real content here for search")
        # min_score=0 should not drop anything that would normally surface
        results = store.search("real content", limit=5, mode="hybrid", min_score=0.0)
        assert len(results) >= 1

    def test_min_score_in_bm25(self, store):
        store.store(text="lexical match alpha bravo charlie")
        results = store.search("alpha", limit=5, mode="bm25", min_score=99.0)
        assert results == []
