"""Unit tests for noise filter, MMR diversity filter, and scoring pipeline."""

from __future__ import annotations

import time

import pytest

from hermes_memory_lancedb_pro.decay import (
    MIN_TEXT_LEN,
    ScoringConfig,
    ScoringPipeline,
    cosine_similarity,
    is_noise,
    mmr_diversity_filter,
)


class TestNoiseFilter:
    @pytest.mark.parametrize("text", [
        "I don't have any information",
        "I don't recall that",
        "no relevant memories found",
        "Do you remember when we talked about",
        "Did I mention this before?",
        "Hello there",
        "fresh session",
        "HEARTBEAT 1234",
        "query -> none",
        "no explicit solution",
        "<<<EXTERNAL_UNTRUSTED_CONTENT begin",
    ])
    def test_known_noise(self, text: str):
        assert is_noise(text) is True

    @pytest.mark.parametrize("text", [
        "Martyn prefers concise responses with UK English.",
        "The new build pipeline ships on Thursdays after the canary green.",
        "User asked for a roll-out plan and we recommended phased deployment.",
    ])
    def test_real_content_is_not_noise(self, text: str):
        assert is_noise(text) is False

    def test_empty_is_noise(self):
        assert is_noise("") is True
        assert is_noise(None) is True  # type: ignore

    def test_short_text_is_noise(self):
        assert is_noise("a" * (MIN_TEXT_LEN - 1)) is True
        # Non-noise short content with enough length: passes
        long_enough = "a" * MIN_TEXT_LEN + " something to say"
        assert is_noise(long_enough) is False


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_mismatched_dims(self):
        assert cosine_similarity([1, 2, 3], [1, 2]) == 0.0


class TestMMR:
    def test_single_result_returned_unchanged(self):
        results = [({"id": "a", "text": "hello"}, 0.9)]
        out = mmr_diversity_filter(results)
        assert out == results

    def test_demotes_near_duplicate_text(self):
        results = [
            ({"id": "a", "text": "the quick brown fox jumps over the lazy dog"}, 0.9),
            ({"id": "b", "text": "the quick brown fox jumps over the lazy dog"}, 0.85),
            ({"id": "c", "text": "completely different content about pineapples"}, 0.7),
        ]
        out = mmr_diversity_filter(results, similarity_threshold=0.85)
        ranked_ids = [e["id"] for e, _ in out]
        # `a` stays first (not a near-duplicate of itself).
        # `c` should beat `b` after demotion despite lower raw score.
        assert ranked_ids[0] == "a"
        assert ranked_ids.index("c") < ranked_ids.index("b")

    def test_uses_vectors_when_available(self):
        results = [
            ({"id": "a", "text": "x", "vector": [1.0, 0.0]}, 0.9),
            ({"id": "b", "text": "y", "vector": [1.0, 0.0]}, 0.85),  # identical
            ({"id": "c", "text": "z", "vector": [0.0, 1.0]}, 0.7),   # orthogonal
        ]
        out = mmr_diversity_filter(results, similarity_threshold=0.9)
        ranked_ids = [e["id"] for e, _ in out]
        assert ranked_ids.index("c") < ranked_ids.index("b")


class TestScoringPipeline:
    def test_filters_below_hard_min(self):
        pipeline = ScoringPipeline(ScoringConfig(hard_min_score=0.5))
        results = [
            {"id": "low", "text": "x", "_fusion_score": 0.1},
            {"id": "high", "text": "x", "_fusion_score": 0.9},
        ]
        out = pipeline.apply_scoring(results)
        assert [e["id"] for e in out] == ["high"]

    def test_attaches_decay_and_final_score(self):
        pipeline = ScoringPipeline()
        results = [
            {"id": "x", "text": "real content here", "_fusion_score": 0.8,
             "importance": 0.7,
             "metadata": {"tier": "working", "confidence": 0.9, "access_count": 2,
                          "created_at": int(time.time() * 1000),
                          "last_accessed_at": int(time.time() * 1000)}},
        ]
        out = pipeline.apply_scoring(results)
        assert "_final_score" in out[0]
        assert "_decay" in out[0]
        assert out[0]["_decay"]["composite"] > 0

    def test_long_text_penalised(self):
        pipeline = ScoringPipeline(ScoringConfig(length_norm_anchor=100, hard_min_score=0.0))
        results = [
            {"id": "short", "text": "a" * 50, "_fusion_score": 0.5, "importance": 0.5,
             "metadata": {"tier": "working", "confidence": 0.8}},
            {"id": "long", "text": "a" * 5000, "_fusion_score": 0.5, "importance": 0.5,
             "metadata": {"tier": "working", "confidence": 0.8}},
        ]
        out = pipeline.apply_scoring(results)
        short_score = next(e["_score"] for e in out if e["id"] == "short")
        long_score = next(e["_score"] for e in out if e["id"] == "long")
        assert short_score > long_score

    def test_empty_results_short_circuits(self):
        pipeline = ScoringPipeline()
        assert pipeline.apply_scoring([]) == []
