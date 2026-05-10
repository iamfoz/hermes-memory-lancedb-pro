"""Tests for reflection.mapped_metadata."""

from __future__ import annotations

from hermes_memory_lancedb_pro.reflection.mapped_metadata import (
    build_reflection_mapped_metadata,
    get_reflection_mapped_decay_defaults,
)

_ALL_KINDS = ["user-model", "agent-model", "lesson", "decision"]

_COMMON = dict(
    event_id="refl-20231114221320-abcd1234",
    agent_id="agent-7",
    session_key="sk-abc",
    session_id="sess-123",
    run_at=1_700_000_000_000,
)


def _item(kind, ordinal=0, group_size=1, category=None, heading="Section"):
    """Build a minimal dict simulating ReflectionMappedMemoryItem."""
    # category from TS is stored on the item but build_reflection_mapped_metadata
    # derives mapped_category from kind via _KIND_TO_CATEGORY — the item's
    # category field is passed through as-is (used for category override tests)
    return {
        "mapped_kind": kind,
        "category": category or _expected_category(kind),
        "heading": heading,
        "ordinal": ordinal,
        "group_size": group_size,
    }


def _expected_category(kind):
    return {
        "user-model": "preference",
        "agent-model": "preference",
        "lesson": "fact",
        "decision": "decision",
    }[kind]


class TestGetReflectionMappedDecayDefaults:
    def test_all_kinds_return_dict_with_expected_keys(self):
        for kind in _ALL_KINDS:
            d = get_reflection_mapped_decay_defaults(kind)
            assert set(d.keys()) == {"midpoint_days", "k", "base_weight", "quality"}

    def test_all_kinds_are_distinct(self):
        defaults = [
            (kind, get_reflection_mapped_decay_defaults(kind))
            for kind in _ALL_KINDS
        ]
        # all midpoint_days values are distinct
        midpoints = [d["midpoint_days"] for _, d in defaults]
        assert len(set(midpoints)) == 4, f"Expected 4 distinct midpoints, got {midpoints}"

    def test_decision_midpoint_is_45(self):
        assert get_reflection_mapped_decay_defaults("decision")["midpoint_days"] == 45.0

    def test_user_model_midpoint_is_21(self):
        assert get_reflection_mapped_decay_defaults("user-model")["midpoint_days"] == 21.0

    def test_agent_model_midpoint_is_10(self):
        assert get_reflection_mapped_decay_defaults("agent-model")["midpoint_days"] == 10.0

    def test_lesson_midpoint_is_7(self):
        assert get_reflection_mapped_decay_defaults("lesson")["midpoint_days"] == 7.0

    def test_decision_k(self):
        assert get_reflection_mapped_decay_defaults("decision")["k"] == 0.25

    def test_user_model_k(self):
        assert get_reflection_mapped_decay_defaults("user-model")["k"] == 0.30

    def test_agent_model_k(self):
        assert get_reflection_mapped_decay_defaults("agent-model")["k"] == 0.35

    def test_lesson_k(self):
        assert get_reflection_mapped_decay_defaults("lesson")["k"] == 0.45

    def test_decision_quality_is_1(self):
        assert get_reflection_mapped_decay_defaults("decision")["quality"] == 1.0

    def test_lesson_quality_is_0_9(self):
        assert get_reflection_mapped_decay_defaults("lesson")["quality"] == 0.9


class TestBuildReflectionMappedMetadata:
    def _build(self, kind, **overrides):
        item = overrides.pop("mapped_item", _item(kind))
        return build_reflection_mapped_metadata(mapped_item=item, **{**_COMMON, **overrides})

    def test_type_field(self):
        md = self._build("lesson")
        assert md.type == "memory-reflection-mapped"

    def test_stage_field(self):
        md = self._build("lesson")
        assert md.stage == "reflect-store"

    def test_reflection_version(self):
        md = self._build("lesson")
        assert md.reflection_version == 4

    def test_decay_model_logistic(self):
        md = self._build("lesson")
        assert md.decay_model == "logistic"

    # --- kind → category mapping ---

    def test_user_model_category_preference(self):
        md = self._build("user-model")
        assert md.mapped_category == "preference"

    def test_agent_model_category_preference(self):
        md = self._build("agent-model")
        assert md.mapped_category == "preference"

    def test_lesson_category_fact(self):
        md = self._build("lesson")
        assert md.mapped_category == "fact"

    def test_decision_category_decision(self):
        md = self._build("decision")
        assert md.mapped_category == "decision"

    # --- decay values come from the defaults table ---

    def test_decay_values_populated_from_defaults(self):
        for kind in _ALL_KINDS:
            defaults = get_reflection_mapped_decay_defaults(kind)
            md = self._build(kind)
            assert md.decay_midpoint_days == defaults["midpoint_days"]
            assert md.decay_k == defaults["k"]
            assert md.base_weight == defaults["base_weight"]
            assert md.quality == defaults["quality"]

    # --- ordinal and group_size set from item ---

    def test_ordinal_set(self):
        item = _item("lesson", ordinal=3, group_size=10)
        md = build_reflection_mapped_metadata(mapped_item=item, **_COMMON)
        assert md.ordinal == 3

    def test_group_size_set(self):
        item = _item("lesson", ordinal=0, group_size=7)
        md = build_reflection_mapped_metadata(mapped_item=item, **_COMMON)
        assert md.group_size == 7

    # --- session/agent fields ---

    def test_event_id_populated(self):
        md = self._build("decision")
        assert md.event_id == _COMMON["event_id"]

    def test_agent_id_populated(self):
        md = self._build("decision")
        assert md.agent_id == "agent-7"

    def test_session_key_populated(self):
        md = self._build("decision")
        assert md.session_key == "sk-abc"

    def test_session_id_populated(self):
        md = self._build("decision")
        assert md.session_id == "sess-123"

    def test_stored_at_populated(self):
        md = self._build("decision")
        assert md.stored_at == _COMMON["run_at"]

    # --- optional fields ---

    def test_used_fallback_default_false(self):
        md = self._build("lesson")
        assert md.used_fallback is False

    def test_used_fallback_propagated(self):
        md = self._build("lesson", used_fallback=True)
        assert md.used_fallback is True

    def test_error_signals_default_empty(self):
        md = self._build("lesson")
        assert md.error_signals == []

    def test_error_signals_propagated(self):
        md = self._build(
            "lesson",
            tool_error_signals=[{"signatureHash": "feedface"}],
        )
        assert md.error_signals == ["feedface"]

    def test_section_from_heading(self):
        item = _item("lesson", heading="Decisions Made")
        md = build_reflection_mapped_metadata(mapped_item=item, **_COMMON)
        assert md.section == "Decisions Made"
