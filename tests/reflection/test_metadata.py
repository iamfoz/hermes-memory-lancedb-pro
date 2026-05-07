"""Tests for hermes_memory_lancedb_pro.reflection.metadata."""

from __future__ import annotations

import json

from hermes_memory_lancedb_pro.reflection.metadata import (
    get_display_category_tag,
    is_reflection_entry,
    parse_reflection_metadata,
)

# ---------------------------------------------------------------------------
# parse_reflection_metadata
# ---------------------------------------------------------------------------

class TestParseReflectionMetadata:
    def test_none_returns_empty(self):
        assert parse_reflection_metadata(None) == {}

    def test_empty_string_returns_empty(self):
        assert parse_reflection_metadata("") == {}

    def test_valid_json_object(self):
        raw = json.dumps({"type": "reflection", "scope": "agent"})
        result = parse_reflection_metadata(raw)
        assert result == {"type": "reflection", "scope": "agent"}

    def test_invalid_json_returns_empty(self):
        assert parse_reflection_metadata("not-json{") == {}

    def test_non_dict_json_returns_empty(self):
        # A JSON array is valid JSON but not a dict.
        assert parse_reflection_metadata(json.dumps([1, 2, 3])) == {}

    def test_json_null_returns_empty(self):
        assert parse_reflection_metadata("null") == {}

    def test_json_string_scalar_returns_empty(self):
        assert parse_reflection_metadata('"hello"') == {}


# ---------------------------------------------------------------------------
# is_reflection_entry
# ---------------------------------------------------------------------------

class TestIsReflectionEntry:
    def test_category_reflection_is_true(self):
        entry = {"category": "reflection", "scope": "agent"}
        assert is_reflection_entry(entry) is True

    def test_metadata_type_memory_reflection(self):
        entry = {
            "category": "note",
            "metadata": json.dumps({"type": "memory-reflection"}),
        }
        assert is_reflection_entry(entry) is True

    def test_metadata_type_memory_reflection_event(self):
        entry = {
            "category": "note",
            "metadata": json.dumps({"type": "memory-reflection-event"}),
        }
        assert is_reflection_entry(entry) is True

    def test_metadata_type_memory_reflection_item(self):
        entry = {
            "category": "note",
            "metadata": json.dumps({"type": "memory-reflection-item"}),
        }
        assert is_reflection_entry(entry) is True

    def test_metadata_type_reflection(self):
        entry = {
            "category": "note",
            "metadata": json.dumps({"type": "reflection"}),
        }
        assert is_reflection_entry(entry) is True

    def test_non_reflection_entry_is_false(self):
        entry = {"category": "fact", "metadata": json.dumps({"type": "fact"})}
        assert is_reflection_entry(entry) is False

    def test_normal_entry_no_metadata_is_false(self):
        entry = {"category": "memory"}
        assert is_reflection_entry(entry) is False

    def test_metadata_as_dict_true(self):
        entry = {"category": "note", "metadata": {"type": "memory-reflection"}}
        assert is_reflection_entry(entry) is True

    def test_metadata_as_dict_false(self):
        entry = {"category": "note", "metadata": {"type": "other"}}
        assert is_reflection_entry(entry) is False

    def test_metadata_as_json_string_true(self):
        entry = {
            "category": "note",
            "metadata": '{"type": "memory-reflection-item"}',
        }
        assert is_reflection_entry(entry) is True


# ---------------------------------------------------------------------------
# get_display_category_tag
# ---------------------------------------------------------------------------

class TestGetDisplayCategoryTag:
    def test_reflection_entry_returns_reflection_scope(self):
        entry = {"category": "reflection", "scope": "agent"}
        assert get_display_category_tag(entry) == "reflection:agent"

    def test_reflection_entry_with_metadata_type(self):
        entry = {
            "category": "note",
            "scope": "agent",
            "metadata": json.dumps({"type": "memory-reflection"}),
        }
        assert get_display_category_tag(entry) == "reflection:agent"

    def test_non_reflection_entry_returns_category_scope(self):
        entry = {"category": "fact", "scope": "global"}
        assert get_display_category_tag(entry) == "fact:global"

    def test_non_reflection_uses_category_not_reflection(self):
        entry = {"category": "memory", "scope": "local"}
        tag = get_display_category_tag(entry)
        assert tag == "memory:local"
        assert not tag.startswith("reflection:")
