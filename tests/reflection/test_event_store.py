"""Tests for reflection.event_store."""

from __future__ import annotations

import re

from hermes_memory_lancedb_pro.reflection.event_store import (
    build_reflection_event_payload,
    create_reflection_event_id,
)

# Common params used across multiple tests
_PARAMS = dict(
    run_at=1_700_000_000_000,  # 2023-11-14T22:13:20.000Z
    session_key="sk-abc",
    session_id="sess-123",
    agent_id="agent-7",
    command="reflect",
)


class TestCreateReflectionEventId:
    def test_id_format_matches_regex(self):
        event_id = create_reflection_event_id(**_PARAMS)
        assert re.fullmatch(r"refl-\d{14}-[0-9a-f]{8}", event_id), (
            f"ID '{event_id}' does not match expected format refl-YYYYMMDDHHMM-XXXXXXXX"
        )

    def test_deterministic_same_params(self):
        id1 = create_reflection_event_id(**_PARAMS)
        id2 = create_reflection_event_id(**_PARAMS)
        assert id1 == id2

    def test_different_params_different_id(self):
        id1 = create_reflection_event_id(**_PARAMS)
        id2 = create_reflection_event_id(**{**_PARAMS, "command": "other-command"})
        assert id1 != id2

    def test_different_session_id_different_id(self):
        id1 = create_reflection_event_id(**_PARAMS)
        id2 = create_reflection_event_id(**{**_PARAMS, "session_id": "sess-999"})
        assert id1 != id2

    def test_date_prefix_encodes_timestamp(self):
        # 2023-11-14T22:13:20.000Z → 20231114221320
        event_id = create_reflection_event_id(**_PARAMS)
        date_part = event_id.split("-")[1]  # "refl-YYYYMMDDHHMM-digest" → index 1
        # date_part is 14 chars
        assert len(date_part) == 14
        # starts with 2023
        assert date_part.startswith("2023")


class TestBuildReflectionEventPayload:
    def _build(self, **overrides):
        params = {
            "scope": "test-scope",
            **_PARAMS,
            **overrides,
        }
        return build_reflection_event_payload(**params)

    def test_kind_is_event(self):
        payload = self._build()
        assert payload.kind == "event"

    def test_metadata_type(self):
        payload = self._build()
        assert payload.metadata.type == "memory-reflection-event"

    def test_metadata_reflection_version(self):
        payload = self._build()
        assert payload.metadata.reflection_version == 4

    def test_metadata_stage(self):
        payload = self._build()
        assert payload.metadata.stage == "reflect-store"

    def test_metadata_fields_populated(self):
        payload = self._build()
        md = payload.metadata
        assert md.session_key == "sk-abc"
        assert md.session_id == "sess-123"
        assert md.agent_id == "agent-7"
        assert md.command == "reflect"
        assert md.stored_at == _PARAMS["run_at"]

    def test_used_fallback_default_false(self):
        payload = self._build()
        assert payload.metadata.used_fallback is False

    def test_used_fallback_propagated(self):
        payload = self._build(used_fallback=True)
        assert payload.metadata.used_fallback is True

    def test_error_signals_default_empty(self):
        payload = self._build()
        assert payload.metadata.error_signals == []

    def test_error_signals_propagated(self):
        signals = [{"signatureHash": "aabbccdd"}, {"signatureHash": "11223344"}]
        payload = self._build(tool_error_signals=signals)
        assert payload.metadata.error_signals == ["aabbccdd", "11223344"]

    def test_error_signals_snake_case_key(self):
        signals = [{"signature_hash": "deadbeef"}]
        payload = self._build(tool_error_signals=signals)
        assert payload.metadata.error_signals == ["deadbeef"]

    def test_text_contains_scope(self):
        payload = self._build(scope="my-scope")
        assert "my-scope" in payload.text

    def test_text_contains_session_id(self):
        payload = self._build()
        assert "sess-123" in payload.text

    def test_text_contains_agent_id(self):
        payload = self._build()
        assert "agent-7" in payload.text

    def test_text_contains_command(self):
        payload = self._build()
        assert "reflect" in payload.text

    def test_event_id_in_text(self):
        payload = self._build()
        assert payload.metadata.event_id in payload.text

    def test_metadata_event_id_matches_deterministic_id(self):
        payload = self._build()
        expected_id = create_reflection_event_id(**_PARAMS)
        assert payload.metadata.event_id == expected_id

    def test_explicit_event_id_used(self):
        payload = self._build(event_id="custom-id-123")
        assert payload.metadata.event_id == "custom-id-123"
        assert "custom-id-123" in payload.text
