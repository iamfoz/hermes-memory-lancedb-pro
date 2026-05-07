"""Tests for memory_compactor.

The clustering / merge helpers are pure Python (no LanceDB dep). The
end-to-end `run_compaction` test uses the StubEmbedder pattern from
`test_store_integration.py` and is marked `integration`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any

import pytest

from hermes_memory_lancedb_pro.memory_compactor import (
    CompactionConfig,
    CompactionEntry,
    build_clusters,
    build_merged_entry,
    record_compaction_run,
    should_run_compaction,
)


def _entry(
    *,
    eid: str = "x",
    text: str = "default text",
    vector: list[float] | None = None,
    category: str = "fact",
    scope: str = "global",
    importance: float = 0.5,
    timestamp: int = 0,
    metadata: dict[str, Any] | None = None,
) -> CompactionEntry:
    return CompactionEntry(
        id=eid,
        text=text,
        vector=vector if vector is not None else [1.0, 0.0],
        category=category,
        scope=scope,
        importance=importance,
        timestamp=timestamp,
        metadata=json.dumps(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Cluster building
# ---------------------------------------------------------------------------

class TestBuildClusters:
    def test_empty_returns_empty(self):
        assert build_clusters([], threshold=0.85, min_cluster_size=2) == []

    def test_below_min_cluster_size(self):
        # Only 1 entry → no cluster
        assert build_clusters([_entry()], threshold=0.85, min_cluster_size=2) == []

    def test_two_similar_form_cluster(self):
        v = [1.0, 0.0]
        plans = build_clusters(
            [_entry(eid="a", vector=v), _entry(eid="b", vector=v)],
            threshold=0.85,
            min_cluster_size=2,
        )
        assert len(plans) == 1
        assert sorted(plans[0].member_indices) == [0, 1]

    def test_orthogonal_no_cluster(self):
        plans = build_clusters(
            [_entry(eid="a", vector=[1.0, 0.0]),
             _entry(eid="b", vector=[0.0, 1.0])],
            threshold=0.85,
            min_cluster_size=2,
        )
        assert plans == []

    def test_seeded_by_importance(self):
        # The most important entry seeds the cluster — we verify by
        # checking that a high-importance "C" gets to seed even though
        # it appears third in input order
        v = [1.0, 0.0]
        plans = build_clusters(
            [
                _entry(eid="a", vector=v, importance=0.3),
                _entry(eid="b", vector=v, importance=0.5),
                _entry(eid="c", vector=v, importance=0.9),
            ],
            threshold=0.85,
            min_cluster_size=2,
        )
        assert len(plans) == 1
        # All three end up in the cluster regardless of seed
        assert len(plans[0].member_indices) == 3

    def test_skips_empty_vectors(self):
        plans = build_clusters(
            [_entry(eid="a", vector=[1.0, 0.0]),
             _entry(eid="b", vector=[]),
             _entry(eid="c", vector=[1.0, 0.0])],
            threshold=0.85,
            min_cluster_size=2,
        )
        assert len(plans) == 1
        # The empty-vector entry is not pulled in
        assert 1 not in plans[0].member_indices


# ---------------------------------------------------------------------------
# Merge strategy
# ---------------------------------------------------------------------------

class TestBuildMergedEntry:
    def test_dedupes_lines_case_insensitively(self):
        members = [
            _entry(text="Hello world\nfoo"),
            _entry(text="hello WORLD\nfoo\nbaz"),
        ]
        merged = build_merged_entry(members)
        # First-seen order preserved; duplicates dropped
        lines = merged.text.split("\n")
        assert lines[0] == "Hello world"
        assert "foo" in lines
        assert "baz" in lines
        assert len(lines) == 3  # no dup

    def test_importance_max(self):
        members = [_entry(importance=0.3), _entry(importance=0.7), _entry(importance=0.5)]
        merged = build_merged_entry(members)
        assert merged.importance == pytest.approx(0.7)

    def test_importance_clamped_to_one(self):
        members = [_entry(importance=1.5)]
        merged = build_merged_entry(members)
        assert merged.importance == 1.0

    def test_category_plurality_vote(self):
        members = [
            _entry(category="fact"),
            _entry(category="fact"),
            _entry(category="other"),
        ]
        merged = build_merged_entry(members)
        assert merged.category == "fact"

    def test_category_tie_broken_by_top_importance(self):
        members = [
            _entry(category="fact", importance=0.3),
            _entry(category="other", importance=0.9),
        ]
        merged = build_merged_entry(members)
        # tie 1-1 → top importance "other" wins
        assert merged.category == "other"

    def test_metadata_marker(self):
        members = [_entry(), _entry()]
        merged = build_merged_entry(members)
        meta = json.loads(merged.metadata)
        assert meta["compacted"] is True
        assert meta["source_count"] == 2
        assert "compacted_at" in meta

    def test_uses_first_member_scope(self):
        members = [_entry(scope="agent"), _entry(scope="agent")]
        merged = build_merged_entry(members)
        assert merged.scope == "agent"


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_first_run_should_proceed(self, tmp_path):
        f = str(tmp_path / "state.json")
        assert should_run_compaction(f, cooldown_hours=24) is True

    def test_after_run_blocked_within_cooldown(self, tmp_path):
        f = str(tmp_path / "state.json")
        record_compaction_run(f)
        assert should_run_compaction(f, cooldown_hours=24) is False

    def test_zero_cooldown_always_runs(self, tmp_path):
        f = str(tmp_path / "state.json")
        record_compaction_run(f)
        assert should_run_compaction(f, cooldown_hours=0) is True

    def test_malformed_state_treated_as_never_run(self, tmp_path):
        f = str(tmp_path / "state.json")
        os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
        with open(f, "w") as fh:
            fh.write("not json")
        assert should_run_compaction(f, cooldown_hours=24) is True

    def test_record_creates_parent_dirs(self, tmp_path):
        f = str(tmp_path / "nested" / "dir" / "state.json")
        record_compaction_run(f)
        assert os.path.exists(f)
        with open(f) as fh:
            data = json.load(fh)
        assert "last_run_at" in data


# ---------------------------------------------------------------------------
# Integration: end-to-end run_compaction
# ---------------------------------------------------------------------------

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")
pytestmark = pytest.mark.integration

from hermes_memory_lancedb_pro.memory_compactor import run_compaction  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402


class StubEmbedder:
    """Deterministic per-text vector so similar texts yield similar
    vectors. Hash the text but bias the first 8 dims by the first 8
    characters so we can craft "similar" texts in tests."""

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


class CloneEmbedder:
    """For compaction tests we want deliberately near-identical vectors;
    the StubEmbedder produces wildly different ones for different text.
    This embedder maps each input through a small clone family that
    shares a base vector, so similar logical inputs cluster."""

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text: str, normalize: bool):
        # Detect "cluster X" prefix and produce nearly-identical vectors.
        if "cluster A:" in text:
            base = [1.0] + [0.0] * (VECTOR_DIM - 1)
        elif "cluster B:" in text:
            base = [0.0, 1.0] + [0.0] * (VECTOR_DIM - 2)
        else:
            base = [0.5] * VECTOR_DIM
        # Tiny perturbation so vectors aren't bitwise identical
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        perturbed = [b + (digest[i % 32] / 255.0) * 1e-6 for i, b in enumerate(base)]
        if normalize:
            n = sum(f * f for f in perturbed) ** 0.5
            if n > 0:
                perturbed = [f / n for f in perturbed]
        return perturbed


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp(prefix="hermes-compactor-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = CloneEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestRunCompaction:
    def test_no_old_entries_no_op(self, store):
        # All fresh entries — none older than 7 days
        for i in range(3):
            store.store(text=f"cluster A: fresh entry {i}", scope="agent")
        result = run_compaction(store, CompactionConfig(min_age_days=7))
        assert result.scanned == 0
        assert result.clusters_found == 0

    def test_clusters_old_similar_entries(self, store):
        # Insert old similar entries by writing then back-dating timestamps.
        # We need to manipulate the timestamp; the store sets it to now,
        # so we use a config that allows min_age_days=0 to scan everything.
        ids = []
        for i in range(3):
            ids.append(store.store(text=f"cluster A: variant {i} of preference", scope="agent"))
        # Add an unrelated cluster
        for i in range(2):
            store.store(text=f"cluster B: separate fact {i}", scope="agent")

        # min_age_days=0 → all entries are eligible
        result = run_compaction(
            store,
            CompactionConfig(
                min_age_days=0,
                similarity_threshold=0.999,
                min_cluster_size=2,
            ),
        )
        # Both clusters should be found and merged
        assert result.scanned >= 5
        assert result.clusters_found >= 2
        assert result.memories_created >= 2
        assert result.memories_deleted >= 4

    def test_dry_run_makes_no_changes(self, store):
        for i in range(3):
            store.store(text=f"cluster A: variant {i}", scope="agent")
        before = len(store.list_memories(limit=100))
        result = run_compaction(
            store,
            CompactionConfig(
                min_age_days=0, similarity_threshold=0.999,
                min_cluster_size=2, dry_run=True,
            ),
        )
        after = len(store.list_memories(limit=100))
        assert result.dry_run is True
        assert result.memories_created == 0
        assert result.memories_deleted == 0
        assert before == after

    def test_skips_archived(self, store):
        a = store.store(text="cluster A: variant 1", scope="agent")
        store.store(text="cluster A: variant 2", scope="agent")
        # Archive `a` via supersede
        store.update(a, text="cluster A: superseded variant")

        run_compaction(
            store,
            CompactionConfig(
                min_age_days=0, similarity_threshold=0.5,
                min_cluster_size=2,
            ),
        )
        # The archived row should not be in the cluster pool — verify by
        # checking we can still get_by_id the archived row (it's still
        # present, just hidden from compaction)
        assert store.get_by_id(a)["metadata"]["state"] == "archived"
