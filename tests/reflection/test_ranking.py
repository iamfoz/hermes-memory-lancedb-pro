"""Tests for hermes_memory_lancedb_pro.reflection.ranking."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.reflection.ranking import (
    REFLECTION_FALLBACK_SCORE_FACTOR,
    ReflectionScoreInput,
    compute_reflection_logistic,
    compute_reflection_score,
    normalize_reflection_line_for_aggregation,
)

# ---------------------------------------------------------------------------
# compute_reflection_logistic
# ---------------------------------------------------------------------------

class TestComputeReflectionLogistic:
    def test_at_midpoint_returns_half(self):
        # When age == midpoint, the exponent is 0, so result == 0.5.
        result = compute_reflection_logistic(age_days=10.0, midpoint_days=10.0, k=0.5)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_age_zero_returns_very_high(self):
        # Age 0 vs a large midpoint → logistic close to 1.
        result = compute_reflection_logistic(age_days=0.0, midpoint_days=30.0, k=0.1)
        assert result > 0.9

    def test_age_much_greater_than_midpoint_returns_low(self):
        # Age far beyond midpoint → logistic close to 0.
        result = compute_reflection_logistic(age_days=200.0, midpoint_days=10.0, k=0.5)
        assert result < 0.01

    def test_negative_age_clamped_to_zero(self):
        # Negative age is clamped to 0, same result as age=0.
        result_neg = compute_reflection_logistic(age_days=-5.0, midpoint_days=10.0, k=0.1)
        result_zero = compute_reflection_logistic(age_days=0.0, midpoint_days=10.0, k=0.1)
        assert result_neg == pytest.approx(result_zero)

    def test_non_finite_age_treated_as_zero(self):
        result = compute_reflection_logistic(
            age_days=float("inf"), midpoint_days=10.0, k=0.1
        )
        result_zero = compute_reflection_logistic(age_days=0.0, midpoint_days=10.0, k=0.1)
        assert result == pytest.approx(result_zero)

    def test_zero_midpoint_defaults_to_one(self):
        result_zero_mid = compute_reflection_logistic(age_days=0.0, midpoint_days=0.0, k=0.1)
        result_one_mid = compute_reflection_logistic(age_days=0.0, midpoint_days=1.0, k=0.1)
        assert result_zero_mid == pytest.approx(result_one_mid)

    def test_zero_k_defaults_to_point_one(self):
        result_zero_k = compute_reflection_logistic(age_days=5.0, midpoint_days=10.0, k=0.0)
        result_default_k = compute_reflection_logistic(age_days=5.0, midpoint_days=10.0, k=0.1)
        assert result_zero_k == pytest.approx(result_default_k)

    def test_result_in_unit_interval(self):
        for age in [0.0, 5.0, 10.0, 50.0, 200.0]:
            result = compute_reflection_logistic(age, 30.0, 0.2)
            assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# compute_reflection_score
# ---------------------------------------------------------------------------

class TestComputeReflectionScore:
    def _base_input(self, **overrides) -> ReflectionScoreInput:
        defaults = dict(age_days=0.0, midpoint_days=30.0, k=0.1)
        defaults.update(overrides)
        return ReflectionScoreInput(**defaults)

    def test_default_quality_and_base_weight_gives_logistic(self):
        inp = self._base_input(age_days=30.0)
        logistic = compute_reflection_logistic(30.0, 30.0, 0.1)
        assert compute_reflection_score(inp) == pytest.approx(logistic * 1.0 * 1.0)

    def test_quality_halved(self):
        full = compute_reflection_score(self._base_input(quality=1.0))
        half = compute_reflection_score(self._base_input(quality=0.5))
        assert half == pytest.approx(full * 0.5, rel=1e-9)

    def test_base_weight_doubled(self):
        single = compute_reflection_score(self._base_input(base_weight=1.0))
        doubled = compute_reflection_score(self._base_input(base_weight=2.0))
        assert doubled == pytest.approx(single * 2.0, rel=1e-9)

    def test_used_fallback_applies_factor(self):
        without = compute_reflection_score(self._base_input(used_fallback=False))
        with_fb = compute_reflection_score(self._base_input(used_fallback=True))
        assert with_fb == pytest.approx(without * REFLECTION_FALLBACK_SCORE_FACTOR, rel=1e-9)

    def test_used_fallback_factor_value(self):
        assert pytest.approx(0.75) == REFLECTION_FALLBACK_SCORE_FACTOR

    def test_quality_clamped_above_one(self):
        clamped = compute_reflection_score(self._base_input(quality=5.0))
        at_one = compute_reflection_score(self._base_input(quality=1.0))
        assert clamped == pytest.approx(at_one)

    def test_quality_clamped_below_zero(self):
        result = compute_reflection_score(self._base_input(quality=-1.0))
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_invalid_base_weight_defaults_to_one(self):
        valid = compute_reflection_score(self._base_input(base_weight=1.0))
        zero_bw = compute_reflection_score(self._base_input(base_weight=0.0))
        assert zero_bw == pytest.approx(valid)


# ---------------------------------------------------------------------------
# normalize_reflection_line_for_aggregation
# ---------------------------------------------------------------------------

class TestNormalizeReflectionLineForAggregation:
    def test_lowercases(self):
        assert normalize_reflection_line_for_aggregation("Hello World") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_reflection_line_for_aggregation("  hi  ") == "hi"

    def test_collapses_internal_whitespace(self):
        assert normalize_reflection_line_for_aggregation("foo   bar\t\nbaz") == "foo bar baz"

    def test_empty_string(self):
        assert normalize_reflection_line_for_aggregation("") == ""

    def test_already_normalized(self):
        assert normalize_reflection_line_for_aggregation("already fine") == "already fine"

    def test_converts_non_string(self):
        # str() should be called on non-string input.
        result = normalize_reflection_line_for_aggregation(42)  # type: ignore[arg-type]
        assert result == "42"
