"""Unit tests for batch_dedup (no LanceDB required)."""

from __future__ import annotations

import math

from hermes_memory_lancedb_pro.batch_dedup import (
    ExtractionCostStats,
    batch_dedup,
    create_extraction_cost_stats,
)
from hermes_memory_lancedb_pro.decay import cosine_similarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(components: list[float]) -> list[float]:
    """Return a unit vector in the direction of *components*."""
    mag = math.sqrt(sum(x * x for x in components))
    return [x / mag for x in components]


# ---------------------------------------------------------------------------
# batch_dedup
# ---------------------------------------------------------------------------


class TestBatchDedupEdgeCases:
    def test_empty_input_returns_empty_result(self):
        result = batch_dedup([], [])
        assert result.surviving_indices == []
        assert result.duplicate_indices == []
        assert result.input_count == 0
        assert result.output_count == 0

    def test_single_candidate_survives(self):
        result = batch_dedup(["abstract"], [[0.1, 0.2, 0.3]])
        assert result.surviving_indices == [0]
        assert result.duplicate_indices == []
        assert result.input_count == 1
        assert result.output_count == 1


class TestBatchDedupTwoCandidates:
    def test_identical_vectors_second_is_duplicate(self):
        vec = [1.0, 0.0, 0.0]
        result = batch_dedup(["a", "b"], [vec, vec])
        assert result.surviving_indices == [0]
        assert result.duplicate_indices == [1]
        assert result.input_count == 2
        assert result.output_count == 1

    def test_orthogonal_vectors_both_survive(self):
        # [1,0] and [0,1] have cosine similarity 0 — both survive
        result = batch_dedup(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        assert result.surviving_indices == [0, 1]
        assert result.duplicate_indices == []
        assert result.input_count == 2
        assert result.output_count == 2


class TestBatchDedupThreeCandidates:
    def test_first_and_third_identical_second_different(self):
        # first == third (similarity=1.0), second is orthogonal to both
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        vec_c = [1.0, 0.0, 0.0]  # same as vec_a
        result = batch_dedup(["a", "b", "c"], [vec_a, vec_b, vec_c])
        assert 0 in result.surviving_indices, "first should survive"
        assert 1 in result.surviving_indices, "second should survive"
        assert 2 in result.duplicate_indices, "third should be dup of first"
        assert result.input_count == 3
        assert result.output_count == 2


class TestBatchDedupThreshold:
    def test_at_threshold_does_not_dedup(self):
        # The TS implementation uses strict > (not >=), so a pair whose
        # similarity equals the threshold must NOT be treated as a duplicate.
        # Because floating-point arithmetic makes "exactly equal" unreliable,
        # we verify the boundary property by using a threshold that is
        # numerically just above the computed similarity, so the pair is
        # guaranteed to sit at-or-below it and must survive.
        #
        # vec_a = [1, 0], vec_b = [0.85, sin(arccos(0.85))]
        # cosine_similarity returns 0.85 + ε (float rounding).
        # Setting threshold = that exact value means sim == threshold → survives
        # (strict > is False).
        theta = math.acos(0.85)
        vec_a = [1.0, 0.0]
        vec_b = [math.cos(theta), math.sin(theta)]

        actual_sim = cosine_similarity(vec_a, vec_b)  # e.g. 0.8500000000000001

        # Use the exact computed value as the threshold: sim == threshold → not dup
        result = batch_dedup(["a", "b"], [vec_a, vec_b], threshold=actual_sim)
        assert result.surviving_indices == [0, 1], (
            "sim == threshold should NOT dedup (strict >)"
        )
        assert result.duplicate_indices == []

    def test_above_threshold_deduped(self):
        # Similarity clearly above threshold → second is a duplicate
        vec = [1.0, 0.0, 0.0]
        result = batch_dedup(["a", "b"], [vec, vec], threshold=0.85)
        assert result.duplicate_indices == [1]

    def test_below_threshold_both_survive(self):
        # Orthogonal → similarity=0 → both survive regardless of threshold
        result = batch_dedup(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], threshold=0.85)
        assert result.surviving_indices == [0, 1]


class TestBatchDedupSkipBehavior:
    def test_vectors_of_different_lengths_skip_pair(self):
        # Different lengths → cosine_similarity returns 0 → no dedup
        result = batch_dedup(["a", "b"], [[1.0, 0.0], [1.0, 0.0, 0.0]])
        assert result.surviving_indices == [0, 1]
        assert result.duplicate_indices == []

    def test_none_vector_entry_skipped(self):
        # None vector → pair is skipped → both survive
        result = batch_dedup(["a", "b"], [None, [1.0, 0.0]])
        assert result.surviving_indices == [0, 1]
        assert result.duplicate_indices == []

    def test_empty_vector_entry_skipped(self):
        # Empty list vector → pair is skipped → both survive
        result = batch_dedup(["a", "b"], [[], [1.0, 0.0]])
        assert result.surviving_indices == [0, 1]
        assert result.duplicate_indices == []


# ---------------------------------------------------------------------------
# create_extraction_cost_stats
# ---------------------------------------------------------------------------


class TestCreateExtractionCostStats:
    def test_returns_fresh_stats_with_zero_defaults(self):
        stats = create_extraction_cost_stats()
        assert isinstance(stats, ExtractionCostStats)
        assert stats.batch_deduped == 0
        assert stats.duration_ms == 0.0
        assert stats.llm_calls == 0

    def test_each_call_returns_independent_instance(self):
        s1 = create_extraction_cost_stats()
        s2 = create_extraction_cost_stats()
        s1.batch_deduped = 5
        assert s2.batch_deduped == 0
