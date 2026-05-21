"""Tests for jmunch gateway detection."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro import jmunch
from hermes_memory_lancedb_pro.jmunch import (
    JMUNCH_MODE_ENV,
    is_jmunch_in_use,
    jmunch_mode_configured,
    jmunch_request_headers,
    record_response_headers,
)


@pytest.fixture(autouse=True)
def _reset_jmunch_state(monkeypatch):
    """Every test starts with no declaration and no observed gateway."""
    monkeypatch.delenv(JMUNCH_MODE_ENV, raising=False)
    monkeypatch.setattr(jmunch, "_state", {"observed": False})


class TestModeConfigured:
    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", " true "])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv(JMUNCH_MODE_ENV, value)
        assert jmunch_mode_configured() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "  "])
    def test_falsy_values_disable(self, monkeypatch, value):
        monkeypatch.setenv(JMUNCH_MODE_ENV, value)
        assert jmunch_mode_configured() is False

    def test_unset_is_disabled(self):
        assert jmunch_mode_configured() is False


class TestIsJmunchInUse:
    def test_false_by_default(self):
        assert is_jmunch_in_use() is False

    def test_true_when_declared(self, monkeypatch):
        monkeypatch.setenv(JMUNCH_MODE_ENV, "true")
        assert is_jmunch_in_use() is True

    def test_true_when_observed(self):
        record_response_headers({"X-Jmunch-Gateway": "fork"})
        assert is_jmunch_in_use() is True


class TestRecordResponseHeaders:
    def test_latches_on_gateway_header(self):
        record_response_headers({"X-Jmunch-Gateway": "fork"})
        assert is_jmunch_in_use() is True

    def test_case_insensitive_header_name(self):
        record_response_headers({"x-jmunch-gateway": "fork"})
        assert is_jmunch_in_use() is True

    def test_detects_regardless_of_header_value(self):
        # Detection keys on the header's presence, not its content.
        record_response_headers({"X-Jmunch-Gateway": "anything at all"})
        assert is_jmunch_in_use() is True

    def test_noop_when_header_absent(self):
        record_response_headers({"content-type": "application/json"})
        assert is_jmunch_in_use() is False

    @pytest.mark.parametrize("headers", [None, "not a mapping", 42])
    def test_safe_on_non_mapping(self, headers):
        record_response_headers(headers)  # must not raise
        assert is_jmunch_in_use() is False

    def test_observation_latches_permanently(self):
        record_response_headers({"X-Jmunch-Gateway": "fork"})
        # A later non-jmunch response must not un-latch the observation.
        record_response_headers({"content-type": "application/json"})
        assert is_jmunch_in_use() is True


class TestRequestHeaders:
    _EXPECTED = {"X-Jmunch-Inject": "false", "X-Jmunch-Handleify": "false"}

    def test_passthrough_headers_when_in_use(self, monkeypatch):
        monkeypatch.setenv(JMUNCH_MODE_ENV, "true")
        assert jmunch_request_headers() == self._EXPECTED

    def test_empty_when_not_in_use(self):
        assert jmunch_request_headers() == {}

    def test_returns_fresh_dict_each_call(self, monkeypatch):
        monkeypatch.setenv(JMUNCH_MODE_ENV, "true")
        first = jmunch_request_headers()
        first["X-Other"] = "1"
        assert jmunch_request_headers() == self._EXPECTED
