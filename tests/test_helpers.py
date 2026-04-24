"""Unit tests for the small SQL helpers (no LanceDB required)."""

from __future__ import annotations

from hermes_memory_lancedb_pro._sql import (
    and_clauses,
    escape_sql,
    is_archived,
    parse_metadata,
)


class TestEscapeSql:
    def test_no_quotes_unchanged(self):
        assert escape_sql("hello") == "hello"

    def test_single_quote_doubled(self):
        assert escape_sql("O'Brien") == "O''Brien"

    def test_multiple_quotes(self):
        assert escape_sql("'a''b'") == "''a''''b''"

    def test_non_string_input(self):
        assert escape_sql(42) == "42"


class TestParseMetadata:
    def test_dict_passthrough(self):
        assert parse_metadata({"a": 1}) == {"a": 1}

    def test_json_string_parsed(self):
        assert parse_metadata('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_empty(self):
        assert parse_metadata("not json {{{") == {}

    def test_non_dict_json_returns_empty(self):
        # JSON-valid list — we want a dict, so it's rejected
        assert parse_metadata("[1, 2, 3]") == {}

    def test_none_returns_empty(self):
        assert parse_metadata(None) == {}


class TestIsArchived:
    def test_archived_state(self):
        assert is_archived('{"state": "archived"}') is True

    def test_dict_archived(self):
        assert is_archived({"state": "archived"}) is True

    def test_confirmed_state(self):
        assert is_archived('{"state": "confirmed"}') is False

    def test_missing_state(self):
        assert is_archived('{"foo": "bar"}') is False

    def test_invalid_metadata(self):
        assert is_archived("garbage") is False
        assert is_archived(None) is False


class TestAndClauses:
    def test_no_clauses(self):
        assert and_clauses() is None

    def test_all_none(self):
        assert and_clauses(None, None, "") is None

    def test_single_clause(self):
        assert and_clauses("a = 1") == "(a = 1)"

    def test_two_clauses(self):
        assert and_clauses("a = 1", "b = 2") == "(a = 1) AND (b = 2)"

    def test_drops_falsy(self):
        assert and_clauses("a = 1", None, "", "c = 3") == "(a = 1) AND (c = 3)"
