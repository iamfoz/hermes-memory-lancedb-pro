"""Tests for reflection.item_store."""

from __future__ import annotations

from hermes_memory_lancedb_pro.reflection.item_store import (
    REFLECTION_DERIVED_BASE_WEIGHT,
    REFLECTION_DERIVED_DECAY_K,
    REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS,
    REFLECTION_DERIVED_QUALITY,
    REFLECTION_INVARIANT_BASE_WEIGHT,
    REFLECTION_INVARIANT_DECAY_K,
    REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS,
    REFLECTION_INVARIANT_QUALITY,
    build_reflection_item_payloads,
    get_reflection_item_decay_defaults,
)


class TestGetReflectionItemDecayDefaults:
    def test_invariant_defaults(self):
        d = get_reflection_item_decay_defaults("invariant")
        assert d["midpoint_days"] == REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS
        assert d["k"] == REFLECTION_INVARIANT_DECAY_K
        assert d["base_weight"] == REFLECTION_INVARIANT_BASE_WEIGHT
        assert d["quality"] == REFLECTION_INVARIANT_QUALITY

    def test_derived_defaults(self):
        d = get_reflection_item_decay_defaults("derived")
        assert d["midpoint_days"] == REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS
        assert d["k"] == REFLECTION_DERIVED_DECAY_K
        assert d["base_weight"] == REFLECTION_DERIVED_BASE_WEIGHT
        assert d["quality"] == REFLECTION_DERIVED_QUALITY

    def test_invariant_differs_from_derived(self):
        inv = get_reflection_item_decay_defaults("invariant")
        der = get_reflection_item_decay_defaults("derived")
        assert inv["midpoint_days"] != der["midpoint_days"]
        assert inv["k"] != der["k"]
        assert inv["base_weight"] != der["base_weight"]
        assert inv["quality"] != der["quality"]

    def test_invariant_midpoint_is_45(self):
        d = get_reflection_item_decay_defaults("invariant")
        assert d["midpoint_days"] == 45.0

    def test_derived_midpoint_is_7(self):
        d = get_reflection_item_decay_defaults("derived")
        assert d["midpoint_days"] == 7.0

    def test_returns_dict_with_expected_keys(self):
        d = get_reflection_item_decay_defaults("invariant")
        assert set(d.keys()) == {"midpoint_days", "k", "base_weight", "quality"}


class TestBuildReflectionItemPayloads:
    _common = dict(
        event_id="refl-20231114221320-abcd1234",
        agent_id="agent-7",
        session_key="sk-abc",
        session_id="sess-123",
        run_at=1_700_000_000_000,
    )

    def _items(self, *specs):
        """Build plain dicts simulating ReflectionSliceItem."""
        result = []
        for i, (text, kind) in enumerate(specs):
            result.append({
                "text": text,
                "item_kind": kind,
                "section": "Invariants" if kind == "invariant" else "Derived",
                "ordinal": i,
                "group_size": len(specs),
            })
        return result

    def test_payload_count_equals_item_count(self):
        items = self._items(
            ("fact one", "invariant"),
            ("obs one", "derived"),
            ("obs two", "derived"),
        )
        payloads = build_reflection_item_payloads(items=items, **self._common)
        assert len(payloads) == 3

    def test_ordering_preserved(self):
        items = self._items(
            ("first", "invariant"),
            ("second", "derived"),
        )
        payloads = build_reflection_item_payloads(items=items, **self._common)
        assert payloads[0].text == "first"
        assert payloads[1].text == "second"

    def test_invariant_kind_tag(self):
        items = self._items(("fact", "invariant"))
        payload = build_reflection_item_payloads(items=items, **self._common)[0]
        assert payload.kind == "item-invariant"

    def test_derived_kind_tag(self):
        items = self._items(("obs", "derived"))
        payload = build_reflection_item_payloads(items=items, **self._common)[0]
        assert payload.kind == "item-derived"

    def test_metadata_type(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.type == "memory-reflection-item"

    def test_metadata_stage(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.stage == "reflect-store"

    def test_metadata_reflection_version(self):
        items = self._items(("x", "derived"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.reflection_version == 4

    def test_metadata_event_id_populated(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.event_id == self._common["event_id"]

    def test_metadata_session_fields(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.session_key == "sk-abc"
        assert md.session_id == "sess-123"
        assert md.agent_id == "agent-7"

    def test_metadata_stored_at(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.stored_at == self._common["run_at"]

    def test_invariant_decay_values(self):
        items = self._items(("fact", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.decay_midpoint_days == REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS
        assert md.decay_k == REFLECTION_INVARIANT_DECAY_K
        assert md.base_weight == REFLECTION_INVARIANT_BASE_WEIGHT
        assert md.quality == REFLECTION_INVARIANT_QUALITY

    def test_derived_decay_values(self):
        items = self._items(("obs", "derived"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.decay_midpoint_days == REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS
        assert md.decay_k == REFLECTION_DERIVED_DECAY_K
        assert md.base_weight == REFLECTION_DERIVED_BASE_WEIGHT
        assert md.quality == REFLECTION_DERIVED_QUALITY

    def test_decay_model_is_logistic(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.decay_model == "logistic"

    def test_resolved_fields_default_none(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(items=items, **self._common)[0].metadata
        assert md.resolved_at is None
        assert md.resolved_by is None
        assert md.resolution_note is None

    def test_used_fallback_propagated(self):
        items = self._items(("x", "invariant"))
        md = build_reflection_item_payloads(
            items=items, used_fallback=True, **self._common
        )[0].metadata
        assert md.used_fallback is True

    def test_error_signals_propagated(self):
        items = self._items(("x", "invariant"))
        signals = [{"signatureHash": "cafebabe"}]
        md = build_reflection_item_payloads(
            items=items, tool_error_signals=signals, **self._common
        )[0].metadata
        assert md.error_signals == ["cafebabe"]

    def test_empty_items_returns_empty_list(self):
        payloads = build_reflection_item_payloads(items=[], **self._common)
        assert payloads == []
