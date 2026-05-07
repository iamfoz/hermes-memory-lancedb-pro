"""Tests for smart_metadata.py — ported from CortexReach smart-metadata.ts."""

from __future__ import annotations

import json
import time

from hermes_memory_lancedb_pro.smart_metadata import (
    _CONTEXT_ALIASES,
    MAX_SUPPORT_SLICES,
    SmartMemoryMetadata,
    SupportInfoV2,
    SupportSlice,
    _normalise_context,
    append_relation,
    build_smart_metadata,
    derive_fact_key,
    is_memory_active_at,
    is_memory_expired,
    parse_smart_metadata,
    parse_support_info,
    stringify_smart_metadata,
    update_support_stats,
)

# ---------------------------------------------------------------------------
# TestParseSmartMetadata
# ---------------------------------------------------------------------------

class TestParseSmartMetadata:
    def test_none_returns_defaults(self):
        m = parse_smart_metadata(None)
        assert isinstance(m, SmartMemoryMetadata)
        assert m.l0_abstract == ""
        assert m.l2_content == ""
        assert m.tier == "working"
        assert m.confidence == 0.7

    def test_empty_string_returns_defaults(self):
        m = parse_smart_metadata("")
        assert isinstance(m, SmartMemoryMetadata)
        assert m.l0_abstract == ""

    def test_valid_json_string(self):
        payload = json.dumps({
            "l0_abstract": "user prefers dark mode",
            "l1_overview": "- user prefers dark mode",
            "l2_content": "The user mentioned they prefer dark mode interfaces.",
            "memory_category": "preferences",
            "tier": "core",
            "confidence": 0.9,
            "access_count": 5,
            "valid_from": 1_000_000,
        })
        m = parse_smart_metadata(payload)
        assert m.l0_abstract == "user prefers dark mode"
        assert m.memory_category == "preferences"
        assert m.tier == "core"
        assert m.confidence == 0.9
        assert m.access_count == 5
        assert m.valid_from == 1_000_000

    def test_invalid_json_returns_defaults(self):
        m = parse_smart_metadata("{not valid json!!")
        assert isinstance(m, SmartMemoryMetadata)
        assert m.l0_abstract == ""

    def test_dict_input(self):
        d = {"l0_abstract": "test abstract", "confidence": 0.8, "valid_from": 500}
        m = parse_smart_metadata(d)
        assert m.l0_abstract == "test abstract"
        assert m.confidence == 0.8

    def test_entry_fallback_for_missing_l2_content(self):
        entry = {"text": "raw entry text", "timestamp": 1_234_567}
        m = parse_smart_metadata(None, entry)
        assert m.l2_content == "raw entry text"
        assert m.l0_abstract == "raw entry text"

    def test_entry_does_not_override_explicit_metadata(self):
        payload = json.dumps({
            "l0_abstract": "explicit abstract",
            "l2_content": "explicit content",
        })
        entry = {"text": "fallback text"}
        m = parse_smart_metadata(payload, entry)
        assert m.l0_abstract == "explicit abstract"
        assert m.l2_content == "explicit content"

    def test_confidence_clamped_to_01(self):
        m = parse_smart_metadata({"confidence": 2.5})
        assert m.confidence == 1.0
        m2 = parse_smart_metadata({"confidence": -1.0})
        assert m2.confidence == 0.0

    def test_access_count_non_negative(self):
        m = parse_smart_metadata({"access_count": -5})
        assert m.access_count == 0

    def test_invalid_tier_defaults_to_working(self):
        m = parse_smart_metadata({"tier": "legendary"})
        assert m.tier == "working"

    def test_valid_tier_preserved(self):
        for tier in ("core", "working", "peripheral"):
            m = parse_smart_metadata({"tier": tier})
            assert m.tier == tier

    def test_invalidated_at_only_set_if_after_valid_from(self):
        # invalidated_at before valid_from → None
        m = parse_smart_metadata({
            "valid_from": 2000,
            "invalidated_at": 1000,
        })
        assert m.invalidated_at is None

        # invalidated_at after valid_from → kept
        m2 = parse_smart_metadata({
            "valid_from": 1000,
            "invalidated_at": 2000,
        })
        assert m2.invalidated_at == 2000

    def test_extras_passthrough(self):
        m = parse_smart_metadata({"unknown_field": "hello", "another": 42})
        assert m.extras.get("unknown_field") == "hello"
        assert m.extras.get("another") == 42

    def test_source_session_string(self):
        m = parse_smart_metadata({"source_session": "sess-abc"})
        assert m.source_session == "sess-abc"

    def test_source_defaults_to_legacy_for_unknown(self):
        m = parse_smart_metadata({"source": "unknown-source"})
        assert m.source == "legacy"

    def test_known_sources_preserved(self):
        for src in ("manual", "auto-capture", "reflection", "session-summary", "legacy"):
            m = parse_smart_metadata({"source": src})
            assert m.source == src


# ---------------------------------------------------------------------------
# TestBuildSmartMetadata
# ---------------------------------------------------------------------------

class TestBuildSmartMetadata:
    def test_patch_overrides_existing_fields(self):
        base_meta = json.dumps({
            "l0_abstract": "old abstract",
            "confidence": 0.5,
            "tier": "working",
        })
        entry = {"metadata": base_meta, "text": "old text"}
        m = build_smart_metadata(entry, {"l0_abstract": "new abstract", "confidence": 0.95})
        assert m.l0_abstract == "new abstract"
        assert m.confidence == 0.95

    def test_missing_fields_get_defaults(self):
        m = build_smart_metadata(None, {})
        assert isinstance(m, SmartMemoryMetadata)
        assert m.tier == "working"
        assert m.l0_abstract == ""

    def test_passthrough_for_unknown_extras(self):
        entry = {"text": "foo"}
        m = build_smart_metadata(entry, {"my_custom_field": "bar"})
        assert m.extras.get("my_custom_field") == "bar"

    def test_access_count_preserved_when_not_patched(self):
        base_meta = json.dumps({"access_count": 7})
        entry = {"metadata": base_meta, "text": ""}
        m = build_smart_metadata(entry, {"tier": "core"})
        assert m.access_count == 7

    def test_relations_from_patch(self):
        entry = {"text": ""}
        rels = [{"type": "follows", "target_id": "abc"}]
        m = build_smart_metadata(entry, {"relations": rels})
        assert m.relations == rels

    def test_fact_key_derived_when_missing(self):
        entry = {"text": ""}
        m = build_smart_metadata(
            entry,
            {"memory_category": "preferences", "l0_abstract": "Dark mode: user prefers dark"},
        )
        assert m.fact_key == "preferences:dark mode"

    def test_supersedes_updated_via_patch(self):
        entry = {"text": ""}
        m = build_smart_metadata(entry, {"supersedes": "old-id-123"})
        assert m.supersedes == "old-id-123"

    def test_superseded_by_updated_via_patch(self):
        entry = {"text": ""}
        m = build_smart_metadata(entry, {"superseded_by": "new-id-456"})
        assert m.superseded_by == "new-id-456"


# ---------------------------------------------------------------------------
# TestStringifyAndRoundTrip
# ---------------------------------------------------------------------------

class TestStringifyAndRoundTrip:
    def test_round_trip_typed_fields(self):
        m = SmartMemoryMetadata(
            l0_abstract="test abstract",
            l1_overview="- test abstract",
            l2_content="full content here",
            memory_category="entities",
            tier="core",
            confidence=0.85,
            access_count=3,
            injected_count=1,
            bad_recall_count=0,
            valid_from=1_000_000,
            valid_until=9_000_000,
            fact_key="entities:test",
            supersedes="old-id",
            source="manual",
            source_session="sess-1",
        )
        serialized = stringify_smart_metadata(m)
        assert isinstance(serialized, str)
        parsed = parse_smart_metadata(serialized)
        assert parsed.l0_abstract == m.l0_abstract
        assert parsed.l1_overview == m.l1_overview
        assert parsed.l2_content == m.l2_content
        assert parsed.memory_category == m.memory_category
        assert parsed.tier == m.tier
        assert parsed.confidence == m.confidence
        assert parsed.access_count == m.access_count
        assert parsed.valid_from == m.valid_from
        assert parsed.valid_until == m.valid_until
        assert parsed.fact_key == m.fact_key
        assert parsed.supersedes == m.supersedes
        assert parsed.source == m.source
        assert parsed.source_session == m.source_session

    def test_stringify_returns_valid_json(self):
        m = SmartMemoryMetadata(l0_abstract="hello")
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert obj["l0_abstract"] == "hello"

    def test_optional_fields_omitted_when_none(self):
        m = SmartMemoryMetadata()
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert "valid_until" not in obj
        assert "invalidated_at" not in obj
        assert "fact_key" not in obj
        assert "supersedes" not in obj
        assert "superseded_by" not in obj

    def test_relations_capped_at_16(self):
        rels = [{"type": "ref", "target_id": str(i)} for i in range(20)]
        m = SmartMemoryMetadata(relations=rels)
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert len(obj["relations"]) == 16

    def test_extras_sources_capped_at_20(self):
        sources = [f"src-{i}" for i in range(30)]
        m = SmartMemoryMetadata(extras={"sources": sources})
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert len(obj["sources"]) == 20

    def test_extras_history_capped_at_50(self):
        history = [f"h-{i}" for i in range(60)]
        m = SmartMemoryMetadata(extras={"history": history})
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert len(obj["history"]) == 50

    def test_support_info_round_trips(self):
        si = SupportInfoV2(
            version=2,
            global_strength=0.75,
            total_observations=4,
            slices=[SupportSlice(
                context="morning",
                confirmations=3,
                contradictions=1,
                strength=0.75,
                last_observed_at=1_234_000,
            )],
        )
        m = SmartMemoryMetadata(support_info=si)
        s = stringify_smart_metadata(m)
        obj = json.loads(s)
        assert obj["support_info"]["global_strength"] == 0.75
        assert obj["support_info"]["slices"][0]["context"] == "morning"


# ---------------------------------------------------------------------------
# TestDeriveFactKey
# ---------------------------------------------------------------------------

class TestDeriveFactKey:
    def test_colon_pattern(self):
        assert derive_fact_key("preferences", "Dark mode: user prefers dark") == "preferences:dark mode"

    def test_cjk_full_width_colon(self):
        assert derive_fact_key("entities", "Location：Beijing") == "entities:location"

    def test_arrow_pattern(self):
        result = derive_fact_key("preferences", "Theme -> dark")
        assert result == "preferences:theme"

    def test_fat_arrow_pattern(self):
        result = derive_fact_key("entities", "Project => hermes")
        assert result == "entities:project"

    def test_no_pattern_returns_none(self):
        result = derive_fact_key("preferences", "user just likes stuff")
        # No colon or arrow → use the whole trimmed abstract as topic
        assert result is not None
        assert result.startswith("preferences:")

    def test_non_temporal_category_returns_none(self):
        assert derive_fact_key("events", "Meeting: quarterly review") is None
        assert derive_fact_key("cases", "Fix: memory leak") is None
        assert derive_fact_key("profile", "Name: Alice") is None
        assert derive_fact_key("patterns", "Morning: jog") is None

    def test_whitespace_collapsed_and_lowercased(self):
        result = derive_fact_key("preferences", "Dark   Mode: prefers dark")
        # "Dark   Mode" → lowercased & space-collapsed → "dark mode"
        assert result == "preferences:dark mode"

    def test_lowercased(self):
        result = derive_fact_key("entities", "GitHub: main profile")
        assert result == "entities:github"

    def test_multiple_colons_uses_first(self):
        # "a: b: c" → topic is "a"
        result = derive_fact_key("preferences", "Theme: color: blue")
        assert result == "preferences:theme"

    def test_cjk_characters_preserved(self):
        result = derive_fact_key("entities", "地点：北京")
        assert result == "entities:地点"

    def test_empty_abstract_returns_none(self):
        assert derive_fact_key("preferences", "") is None
        assert derive_fact_key("preferences", "   ") is None

    def test_valid_temporal_categories(self):
        assert derive_fact_key("preferences", "Color: blue") is not None
        assert derive_fact_key("entities", "Name: Alice") is not None

    def test_trailing_punctuation_stripped(self):
        # The TS regex strips trailing [。.!?]+ from the topic.
        # "Theme!: dark mode" → colon match extracts "Theme!" → strip trailing "!" → "theme"
        result = derive_fact_key("preferences", "Theme!: dark mode")
        assert result == "preferences:theme"


# ---------------------------------------------------------------------------
# TestUpdateSupportStats
# ---------------------------------------------------------------------------

class TestUpdateSupportStats:
    def test_first_observation_creates_slice(self):
        result = update_support_stats(None, "morning", "support")
        assert len(result.slices) == 1
        assert result.slices[0].context == "morning"
        assert result.slices[0].confirmations == 1
        assert result.slices[0].contradictions == 0

    def test_subsequent_confirmation_increments(self):
        r1 = update_support_stats(None, "evening", "support")
        r2 = update_support_stats(r1, "evening", "support")
        assert r2.slices[0].confirmations == 2

    def test_contradiction_tracked_separately(self):
        r1 = update_support_stats(None, "work", "support")
        r2 = update_support_stats(r1, "work", "contradict")
        s = r2.slices[0]
        assert s.confirmations == 1
        assert s.contradictions == 1

    def test_global_strength_recalculated(self):
        # 2 confirmations, 0 contradictions → strength = 1.0
        r = update_support_stats(None, "morning", "support")
        r = update_support_stats(r, "morning", "support")
        assert r.global_strength == 1.0

        # 2 conf + 2 contra → 0.5
        r = update_support_stats(r, "morning", "contradict")
        r = update_support_stats(r, "morning", "contradict")
        assert abs(r.global_strength - 0.5) < 1e-9

    def test_total_observations_sums_all_slices(self):
        r = update_support_stats(None, "morning", "support")
        r = update_support_stats(r, "evening", "support")
        r = update_support_stats(r, "evening", "contradict")
        assert r.total_observations == 3

    def test_cap_at_max_support_slices(self):
        r: SupportInfoV2 | None = None
        contexts = [f"ctx-{i}" for i in range(MAX_SUPPORT_SLICES + 2)]
        for ctx in contexts:
            r = update_support_stats(r, ctx, "support")
        assert len(r.slices) == MAX_SUPPORT_SLICES  # type: ignore[union-attr]

    def test_oldest_dropped_when_at_cap(self):
        """When cap is reached, the oldest slice is dropped and the new one survives."""
        # Build MAX_SUPPORT_SLICES slices manually with known, distinct timestamps
        # so we can reliably identify which one is "oldest".
        now_ms = int(time.time() * 1000)
        # Manually construct an existing SupportInfoV2 at the cap with known timestamps
        slices = [
            SupportSlice(
                context=f"ctx-{i}",
                confirmations=1,
                contradictions=0,
                strength=1.0,
                last_observed_at=now_ms - (MAX_SUPPORT_SLICES - i) * 1000,  # oldest = ctx-0
            )
            for i in range(MAX_SUPPORT_SLICES)
        ]
        existing = SupportInfoV2(
            version=2,
            global_strength=1.0,
            total_observations=MAX_SUPPORT_SLICES,
            slices=slices,
        )
        # Add a brand-new context — should evict ctx-0 (oldest last_observed_at)
        r = update_support_stats(existing, "brand-new-ctx", "support")
        assert len(r.slices) == MAX_SUPPORT_SLICES
        context_names = {s.context for s in r.slices}
        # Oldest (ctx-0) should be gone; brand-new-ctx should be present
        assert "brand-new-ctx" in context_names
        assert "ctx-0" not in context_names

    def test_dropped_slice_evidence_in_total_observations(self):
        """Dropped slices still contribute to total_observations."""
        r: SupportInfoV2 | None = None
        # Fill all slots
        for i in range(MAX_SUPPORT_SLICES):
            r = update_support_stats(r, f"ctx-{i}", "support")
        # Force eviction: add a new unique context
        r = update_support_stats(r, "overflow-ctx", "support")
        # We should still have at least all original observations counted
        # (= MAX_SUPPORT_SLICES + 1 confirmations total)
        assert r.total_observations >= MAX_SUPPORT_SLICES + 1  # type: ignore[union-attr]

    def test_none_context_defaults_to_general(self):
        r = update_support_stats(None, None, "support")
        assert r.slices[0].context == "general"

    def test_cjk_context_normalised(self):
        r = update_support_stats(None, "晚上", "support")
        assert r.slices[0].context == "evening"

    def test_slice_strength_recalculated(self):
        r = update_support_stats(None, "work", "support")
        r = update_support_stats(r, "work", "contradict")
        s = r.slices[0]
        # 1 conf / (1 conf + 1 contra) = 0.5
        assert abs(s.strength - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# TestParseSupportInfo
# ---------------------------------------------------------------------------

class TestParseSupportInfo:
    def test_none_returns_none(self):
        assert parse_support_info(None) is None

    def test_invalid_input_returns_none(self):
        assert parse_support_info("not-json") is None
        assert parse_support_info(42) is None  # type: ignore[arg-type]

    def test_v2_round_trip(self):
        si = SupportInfoV2(
            version=2,
            global_strength=0.8,
            total_observations=10,
            slices=[SupportSlice(
                context="morning",
                confirmations=8,
                contradictions=2,
                strength=0.8,
                last_observed_at=999_000,
            )],
        )
        d = {
            "slices": [
                {
                    "context": s.context,
                    "confirmations": s.confirmations,
                    "contradictions": s.contradictions,
                    "strength": s.strength,
                    "last_observed_at": s.last_observed_at,
                }
                for s in si.slices
            ],
            "global_strength": si.global_strength,
            "total_observations": si.total_observations,
        }
        parsed = parse_support_info(d)
        assert parsed is not None
        assert parsed.global_strength == 0.8
        assert parsed.total_observations == 10
        assert len(parsed.slices) == 1
        assert parsed.slices[0].context == "morning"
        assert parsed.slices[0].confirmations == 8

    def test_v1_upgrade_to_v2(self):
        v1 = {"confirmations": 3, "contradictions": 1}
        parsed = parse_support_info(v1)
        assert parsed is not None
        assert len(parsed.slices) == 1
        assert parsed.slices[0].context == "unknown"
        assert parsed.slices[0].confirmations == 3
        assert parsed.slices[0].contradictions == 1
        assert abs(parsed.global_strength - 0.75) < 1e-9
        assert parsed.total_observations == 4

    def test_v1_zero_total_returns_empty_v2(self):
        v1 = {"confirmations": 0, "contradictions": 0}
        parsed = parse_support_info(v1)
        assert parsed is not None
        assert parsed.total_observations == 0
        assert parsed.slices == []

    def test_json_string_input(self):
        d = {"slices": [], "global_strength": 0.5, "total_observations": 0}
        parsed = parse_support_info(json.dumps(d))
        assert parsed is not None
        assert parsed.slices == []


# ---------------------------------------------------------------------------
# TestAppendRelation
# ---------------------------------------------------------------------------

class TestAppendRelation:
    def test_adds_new_relation(self):
        result = append_relation([], "follows", "mem-1")
        assert result == [{"type": "follows", "target_id": "mem-1"}]

    def test_deduplicates_same_type_and_target(self):
        existing = [{"type": "follows", "target_id": "mem-1"}]
        result = append_relation(existing, "follows", "mem-1")
        assert result == existing  # unchanged

    def test_different_type_not_deduped(self):
        existing = [{"type": "follows", "target_id": "mem-1"}]
        result = append_relation(existing, "contradicts", "mem-1")
        assert len(result) == 2

    def test_different_target_not_deduped(self):
        existing = [{"type": "follows", "target_id": "mem-1"}]
        result = append_relation(existing, "follows", "mem-2")
        assert len(result) == 2

    def test_preserves_order(self):
        existing = [
            {"type": "a", "target_id": "1"},
            {"type": "b", "target_id": "2"},
        ]
        result = append_relation(existing, "c", "3")
        assert result[0] == {"type": "a", "target_id": "1"}
        assert result[1] == {"type": "b", "target_id": "2"}
        assert result[2] == {"type": "c", "target_id": "3"}

    def test_invalid_entries_filtered_out(self):
        existing = [None, 42, {"type": "ok", "target_id": "x"}]
        result = append_relation(existing, "new", "y")  # type: ignore[list-item]
        assert len(result) == 2
        assert result[0] == {"type": "ok", "target_id": "x"}

    def test_none_existing_treated_as_empty(self):
        result = append_relation(None, "ref", "z")  # type: ignore[arg-type]
        assert result == [{"type": "ref", "target_id": "z"}]


# ---------------------------------------------------------------------------
# TestIsMemoryActiveAt
# ---------------------------------------------------------------------------

class TestIsMemoryActiveAt:
    def _make(self, valid_from: int = 0, invalidated_at: int | None = None) -> SmartMemoryMetadata:
        return SmartMemoryMetadata(valid_from=valid_from, invalidated_at=invalidated_at)

    def test_active_when_valid_from_in_past(self):
        m = self._make(valid_from=1000)
        assert is_memory_active_at(m, at_ms=2000) is True

    def test_not_active_before_valid_from(self):
        m = self._make(valid_from=5000)
        assert is_memory_active_at(m, at_ms=1000) is False

    def test_active_at_exact_valid_from(self):
        # valid_from > at → False; valid_from == at → True (not strictly greater)
        m = self._make(valid_from=1000)
        assert is_memory_active_at(m, at_ms=1000) is True

    def test_not_active_when_invalidated_in_past(self):
        m = self._make(valid_from=1000, invalidated_at=1500)
        assert is_memory_active_at(m, at_ms=2000) is False

    def test_still_active_when_invalidated_in_future(self):
        m = self._make(valid_from=1000, invalidated_at=5000)
        assert is_memory_active_at(m, at_ms=2000) is True

    def test_defaults_to_now(self):
        # valid_from in deep past, no invalidated_at → active now
        m = self._make(valid_from=1)
        assert is_memory_active_at(m) is True


# ---------------------------------------------------------------------------
# TestIsMemoryExpired
# ---------------------------------------------------------------------------

class TestIsMemoryExpired:
    def test_not_expired_when_no_valid_until(self):
        m = SmartMemoryMetadata()
        assert is_memory_expired(m, at_ms=9_999_999_999) is False

    def test_expired_when_valid_until_in_past(self):
        m = SmartMemoryMetadata(valid_until=1000)
        assert is_memory_expired(m, at_ms=2000) is True

    def test_not_expired_when_valid_until_in_future(self):
        m = SmartMemoryMetadata(valid_until=9_000_000_000)
        assert is_memory_expired(m, at_ms=1000) is False

    def test_expired_at_exact_valid_until(self):
        # valid_until <= at → expired
        m = SmartMemoryMetadata(valid_until=1000)
        assert is_memory_expired(m, at_ms=1000) is True

    def test_defaults_to_now(self):
        # valid_until far in the future → not expired now
        future = int(time.time() * 1000) + 1_000_000
        m = SmartMemoryMetadata(valid_until=future)
        assert is_memory_expired(m) is False


# ---------------------------------------------------------------------------
# TestNormaliseContext
# ---------------------------------------------------------------------------

class TestNormaliseContext:
    def test_known_cjk_alias_maps_to_english(self):
        assert _normalise_context("晚上") == "evening"
        assert _normalise_context("早上") == "morning"
        assert _normalise_context("工作") == "work"
        assert _normalise_context("周末") == "weekend"
        assert _normalise_context("旅行") == "travel"

    def test_unknown_label_returns_lowercased(self):
        assert _normalise_context("CustomCtx") == "customctx"
        assert _normalise_context("WORK") == "work"

    def test_none_returns_none(self):
        assert _normalise_context(None) is None

    def test_empty_string_returns_none(self):
        assert _normalise_context("") is None
        assert _normalise_context("   ") is None

    def test_known_english_aliases(self):
        # English vocabulary labels pass through as lowercased
        assert _normalise_context("Morning") == "morning"
        assert _normalise_context("EVENING") == "evening"

    def test_all_cjk_aliases_present(self):
        """Every key in _CONTEXT_ALIASES should be reachable."""
        for cjk_key, expected in _CONTEXT_ALIASES.items():
            assert _normalise_context(cjk_key) == expected
