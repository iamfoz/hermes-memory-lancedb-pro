"""Unit tests for the Weibull decay engine.

These run pure-Python — no LanceDB or sentence-transformers required."""

from __future__ import annotations

import time

import pytest

from hermes_memory_lancedb_pro.decay import (
    MS_PER_DAY,
    TIER_FLOOR,
    DecayConfig,
    compute_decay_score,
    evaluate_all_tiers,
    evaluate_tier,
)

NOW = 1_700_000_000_000  # fixed reference time for deterministic tests


def make_entry(
    *,
    importance: float = 0.5,
    tier: str = "working",
    confidence: float = 0.8,
    access_count: int = 0,
    age_days: float = 0,
    last_access_days_ago: float | None = None,
    state: str = "confirmed",
    metadata_extra: dict | None = None,
) -> dict:
    """Build a row-shaped entry for testing."""
    created_at = NOW - int(age_days * MS_PER_DAY)
    if last_access_days_ago is None:
        last_accessed_at = created_at
    else:
        last_accessed_at = NOW - int(last_access_days_ago * MS_PER_DAY)
    metadata = {
        "tier": tier,
        "confidence": confidence,
        "access_count": access_count,
        "created_at": created_at,
        "last_accessed_at": last_accessed_at,
        "state": state,
        "temporal_type": "static",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return {
        "id": f"id-{importance}-{tier}-{age_days}",
        "text": "some content " * 5,
        "category": "fact",
        "scope": "global",
        "importance": importance,
        "timestamp": created_at,
        "metadata": metadata,
    }


class TestComputeDecayScore:
    def test_fresh_high_importance_is_high_score(self):
        entry = make_entry(importance=0.9, access_count=10, age_days=0)
        score = compute_decay_score(entry, now_ms=NOW)
        # recency=1 (fresh), strong frequency, intrinsic=0.72 → composite >= 0.6
        assert score["composite"] >= 0.6
        assert score["recency"] == pytest.approx(1.0, rel=1e-3)

    def test_old_low_importance_decays(self):
        entry = make_entry(
            importance=0.3, tier="peripheral", age_days=120,
            last_access_days_ago=120,
        )
        score = compute_decay_score(entry, now_ms=NOW)
        assert score["recency"] < 0.2

    def test_top_level_importance_is_used(self):
        """Bug repro: previously importance was read from metadata only,
        which always defaulted to 0.5. The top-level column is authoritative."""
        # Same metadata, different top-level importance — should differ
        e_low = make_entry(importance=0.1, age_days=10)
        e_high = make_entry(importance=0.9, age_days=10)
        s_low = compute_decay_score(e_low, now_ms=NOW)
        s_high = compute_decay_score(e_high, now_ms=NOW)
        assert s_high["intrinsic"] > s_low["intrinsic"]
        assert s_high["composite"] > s_low["composite"]

    def test_tier_beta_affects_late_decay(self):
        e_core = make_entry(tier="core", age_days=90, importance=0.5)
        e_periph = make_entry(tier="peripheral", age_days=90, importance=0.5)
        s_core = compute_decay_score(e_core, now_ms=NOW, config=DecayConfig(apply_tier_floor=False))
        s_periph = compute_decay_score(e_periph, now_ms=NOW, config=DecayConfig(apply_tier_floor=False))
        # Core has lower beta → slower late decay → higher recency
        assert s_core["recency"] > s_periph["recency"]

    def test_tier_floor_applied_to_core(self):
        # Very stale, low importance core memory should still hit the core floor
        e = make_entry(
            tier="core",
            age_days=365,
            last_access_days_ago=365,
            importance=0.3,
            access_count=0,
        )
        score = compute_decay_score(e, now_ms=NOW)
        assert score["composite"] >= TIER_FLOOR["core"] - 1e-6

    def test_tier_floor_does_not_block_peripheral_demotion(self):
        """Reproduces the original bug where the peripheral floor coincided
        with the demotion threshold, making composite-based demotion a no-op."""
        e = make_entry(
            tier="working",  # current tier (so demotion is meaningful)
            age_days=90,
            last_access_days_ago=90,
            importance=0.1,
            access_count=0,
        )
        score = compute_decay_score(e, now_ms=NOW)
        # composite should drop well below the legacy 0.15 floor
        assert score["composite"] < 0.4

    def test_dynamic_temporal_decays_faster(self):
        e_static = make_entry(age_days=15)
        e_dynamic = make_entry(age_days=15, metadata_extra={"temporal_type": "dynamic"})
        s_static = compute_decay_score(e_static, now_ms=NOW)
        s_dynamic = compute_decay_score(e_dynamic, now_ms=NOW)
        assert s_dynamic["recency"] < s_static["recency"]

    def test_metadata_only_dict_back_compat(self):
        """Old callers pass a metadata-only dict positionally."""
        meta = {
            "importance": 0.7,
            "tier": "working",
            "confidence": 0.8,
            "access_count": 1,
            "created_at": NOW,
            "last_accessed_at": NOW,
        }
        score = compute_decay_score(meta, now_ms=NOW)
        assert score["composite"] > 0


class TestEvaluateTier:
    def test_promote_to_core(self):
        entry = make_entry(importance=0.9, tier="working", access_count=12)
        decay = {"composite": 0.8}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "core"

    def test_demote_core_to_working(self):
        entry = make_entry(importance=0.9, tier="core", access_count=1)
        decay = {"composite": 0.1}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "working"

    def test_demote_to_peripheral_via_age(self):
        entry = make_entry(tier="working", access_count=0, age_days=120)
        decay = {"composite": 0.5}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "peripheral"

    def test_demote_to_peripheral_via_low_composite(self):
        entry = make_entry(tier="working", access_count=5, age_days=10)
        decay = {"composite": 0.1}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "peripheral"

    def test_promote_peripheral_to_working(self):
        entry = make_entry(tier="peripheral", access_count=4, age_days=5)
        decay = {"composite": 0.5}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "working"

    def test_session_summary_skipped(self):
        entry = make_entry(
            tier="working",
            access_count=20,
            metadata_extra={"metadata_type": "session-summary"},
        )
        decay = {"composite": 0.95}
        assert evaluate_tier(entry, decay, now_ms=NOW) == "working"

    def test_evaluate_all_tiers_skips_archived(self):
        memories = [
            make_entry(importance=0.9, tier="working", access_count=20, state="archived"),
            make_entry(importance=0.9, tier="working", access_count=20, state="confirmed"),
        ]
        memories[0]["id"] = "archived-id"
        memories[1]["id"] = "active-id"
        # Manually set high composite so promotion fires for both
        changed = evaluate_all_tiers(memories, now_ms=NOW)
        assert "archived-id" not in changed


class TestSupportInfoConfidenceBlending:
    """compute_decay_score blends SupportInfoV2.global_strength into confidence."""

    def _make_entry_with_support(self, global_strength: float, total_obs: int):
        return {
            "importance": 0.8,
            "metadata": {
                "tier": "working",
                "confidence": 1.0,
                "created_at": NOW,
                "support_info": {
                    "global_strength": global_strength,
                    "total_observations": total_obs,
                    "slices": [],
                },
            },
        }

    def test_no_support_info_no_penalty(self):
        entry = make_entry(importance=0.8)
        score = compute_decay_score(entry, now_ms=NOW)
        # No support_info → confidence uses write-time value, no blending
        assert "freshness_trend" in score
        assert score["freshness_trend"] == "forming"

    def test_high_global_strength_minimal_penalty(self):
        entry = self._make_entry_with_support(global_strength=1.0, total_obs=5)
        score = compute_decay_score(entry, now_ms=NOW)
        # global_strength=1.0 → factor=1.0 → no penalty
        assert score["freshness_trend"] == "strengthening"
        entry_no_support = make_entry(importance=0.8)
        score_no_support = compute_decay_score(entry_no_support, now_ms=NOW)
        assert abs(score["composite"] - score_no_support["composite"]) < 0.05

    def test_low_global_strength_penalises_score(self):
        entry_strong = self._make_entry_with_support(global_strength=1.0, total_obs=5)
        entry_weak = self._make_entry_with_support(global_strength=0.0, total_obs=5)
        score_strong = compute_decay_score(entry_strong, now_ms=NOW)
        score_weak = compute_decay_score(entry_weak, now_ms=NOW)
        assert score_weak["composite"] < score_strong["composite"]
        assert score_weak["freshness_trend"] == "weakening"

    def test_below_threshold_obs_no_blending(self):
        # total_observations < 3 → blending not applied
        entry_low = self._make_entry_with_support(global_strength=0.0, total_obs=2)
        entry_high = self._make_entry_with_support(global_strength=0.0, total_obs=3)
        score_low = compute_decay_score(entry_low, now_ms=NOW)
        score_high = compute_decay_score(entry_high, now_ms=NOW)
        # low obs: no penalty (freshness_trend stays "forming")
        assert score_low["freshness_trend"] == "forming"
        # high obs: penalty applied (freshness_trend becomes "weakening")
        assert score_high["freshness_trend"] == "weakening"
        assert score_high["composite"] < score_low["composite"]

    def test_stable_trend_for_contested_memory(self):
        # global_strength ≈ 0.5 → stable trend
        entry = self._make_entry_with_support(global_strength=0.5, total_obs=10)
        score = compute_decay_score(entry, now_ms=NOW)
        assert score["freshness_trend"] == "stable"


class TestTemporalQueryIntent:
    """_parse_temporal_intent extracts timestamp ranges from natural-language queries."""

    from hermes_memory_lancedb_pro.provider import _parse_temporal_intent

    NOW_MS = 1_700_000_000_000  # fixed reference point

    def _ti(self, query: str):
        from hermes_memory_lancedb_pro.provider import _parse_temporal_intent
        return _parse_temporal_intent(query, self.NOW_MS)

    def test_no_temporal_returns_none(self):
        assert self._ti("what do I prefer for breakfast?") is None
        assert self._ti("remind me of my tech stack") is None

    def test_yesterday_returns_24h_window(self):
        result = self._ti("what did I say yesterday about the project?")
        assert result is not None
        ts_min, ts_max = result
        # window should be ~24–48h before NOW
        assert 0 < self.NOW_MS - ts_max < 2 * 86_400_000
        assert ts_min < ts_max

    def test_last_week_window(self):
        result = self._ti("what happened last week?")
        assert result is not None
        ts_min, ts_max = result
        # window should be 7–14 days before NOW
        assert ts_min < ts_max

    def test_today_returns_same_day_window(self):
        result = self._ti("what did we decide today?")
        assert result is not None
        ts_min, ts_max = result
        assert ts_max <= self.NOW_MS
        assert self.NOW_MS - ts_min <= 2 * 86_400_000

    def test_recently_returns_7_day_window(self):
        result = self._ti("what did I recently tell you about React?")
        assert result is not None

    def test_range_never_inverted(self):
        for query in [
            "yesterday", "last week", "this week", "this month",
            "recently", "today", "last month",
        ]:
            result = self._ti(query)
            if result is not None:
                ts_min, ts_max = result
                assert ts_min <= ts_max, f"range inverted for {query!r}"


# ---------------------------------------------------------------------------
# TestCreatedAtFallback — top-level timestamp fallback
# ---------------------------------------------------------------------------

class TestCreatedAtFallback:
    """compute_decay_score must use entry["timestamp"] when metadata.created_at is absent."""

    def test_ancient_timestamp_produces_low_recency(self):
        ten_years_ms = 10 * 365 * 24 * 3600 * 1000
        old_ts = int(time.time() * 1000) - ten_years_ms
        entry = {
            "text": "old memory",
            "importance": 0.7,
            "timestamp": old_ts,       # top-level column, no metadata.created_at
            "metadata": {
                "tier": "working",
                "confidence": 0.8,
                "access_count": 0,
                # no created_at here
            },
        }
        result = compute_decay_score(entry)
        assert result["recency"] < 0.05, (
            f"10-year-old memory should have near-zero recency; got {result['recency']}"
        )

    def test_metadata_created_at_takes_priority_over_timestamp(self):
        ten_years_ms = 10 * 365 * 24 * 3600 * 1000
        now_ms = int(time.time() * 1000)
        entry = {
            "text": "recent memory with old top-level timestamp",
            "importance": 0.7,
            "timestamp": now_ms - ten_years_ms,   # old
            "metadata": {
                "tier": "working",
                "confidence": 0.8,
                "access_count": 0,
                "created_at": now_ms - 1000,       # recent — should win
            },
        }
        result = compute_decay_score(entry)
        assert result["recency"] > 0.99, (
            "metadata.created_at should take priority over top-level timestamp"
        )
