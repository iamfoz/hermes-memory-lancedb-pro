"""Pure-Python tests for the session-scoping helper."""

from __future__ import annotations

from hermes_memory_lancedb_pro._sql import match_session


class TestMatchSession:
    def test_no_session_id_matches_everything(self):
        # Empty session_id means "no scoping" — passthrough True
        assert match_session({"source_session": "abc"}, "") is True
        assert match_session({}, "") is True

    def test_same_session_matches(self):
        meta = {"source_session": "sess-1"}
        assert match_session(meta, "sess-1") is True

    def test_different_session_does_not_match(self):
        meta = {"source_session": "sess-A"}
        assert match_session(meta, "sess-B") is False

    def test_missing_source_session_does_not_match(self):
        meta = {"tier": "working", "access_count": 0}
        assert match_session(meta, "sess-1") is False

    def test_empty_source_session_does_not_match(self):
        meta = {"source_session": ""}
        assert match_session(meta, "sess-1") is False

    def test_cross_session_flag_overrides(self):
        meta = {"source_session": "sess-OTHER", "cross_session": True}
        assert match_session(meta, "sess-CURRENT") is True

    def test_core_tier_is_cross_session(self):
        meta = {"source_session": "sess-OTHER", "tier": "core"}
        assert match_session(meta, "sess-CURRENT") is True

    def test_working_tier_is_not_cross_session(self):
        meta = {"source_session": "sess-OTHER", "tier": "working"}
        assert match_session(meta, "sess-CURRENT") is False

    def test_peripheral_tier_is_not_cross_session(self):
        meta = {"source_session": "sess-OTHER", "tier": "peripheral"}
        assert match_session(meta, "sess-CURRENT") is False

    def test_json_string_metadata(self):
        # JSON string metadata is also accepted (LanceDB stores it as JSON text)
        meta = '{"source_session": "sess-1"}'
        assert match_session(meta, "sess-1") is True
        assert match_session(meta, "sess-2") is False

    def test_invalid_metadata_does_not_match(self):
        # Garbage metadata can never satisfy the session filter
        assert match_session("not json {", "sess-1") is False
        assert match_session(None, "sess-1") is False
