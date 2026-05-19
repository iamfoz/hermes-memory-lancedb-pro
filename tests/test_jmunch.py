"""Tests for jmunch proxy detection."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro import jmunch
from hermes_memory_lancedb_pro.jmunch import (
    detected_jmunch_endpoint,
    is_jmunch_in_use,
    is_jmunch_url,
)

_URL_ENV_VARS = (
    "MEMORY_EXTRACTION_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)


@pytest.fixture(autouse=True)
def _clear_url_env(monkeypatch):
    """Every test starts with no LLM base-URL env vars set."""
    for var in _URL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestIsJmunchUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:7879/v1",  # jmunch gateway default port
            "http://127.0.0.1:7883/v1",  # the README example
            "http://localhost:7888/v1",
            "http://[::1]:7882/v1",
            "127.0.0.1:7881",  # scheme is optional
            "https://127.0.0.1:7890",
            "http://127.0.0.1:7894/v1",  # top of the default range (base+span-1)
        ],
    )
    def test_detects_local_jmunch_ports(self, url):
        assert is_jmunch_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "http://127.0.0.1:8080/v1",
            "http://localhost:11434/v1",  # Ollama's default port
            "http://127.0.0.1:1234/v1",  # LM Studio's default port
            "http://127.0.0.1:7878/v1",  # the jmunch dashboard — not an LLM endpoint
            "http://127.0.0.1:7895/v1",  # one past the range (base+span)
            "http://192.168.1.5:7881/v1",  # jmunch port but not loopback
            "http://127.0.0.1/v1",  # no port
        ],
    )
    def test_rejects_non_jmunch_urls(self, url):
        assert is_jmunch_url(url) is False

    @pytest.mark.parametrize("url", [None, "", "   ", "not a url", "http://"])
    def test_rejects_empty_and_malformed(self, url):
        # Must classify, never raise.
        assert is_jmunch_url(url) is False

    def test_respects_configured_port_range(self, monkeypatch):
        monkeypatch.setattr(jmunch, "JMUNCH_PORT_BASE", 9000)
        monkeypatch.setattr(jmunch, "JMUNCH_PORT_SPAN", 4)
        assert is_jmunch_url("http://127.0.0.1:9000/v1") is True
        assert is_jmunch_url("http://127.0.0.1:9003/v1") is True
        assert is_jmunch_url("http://127.0.0.1:9004/v1") is False
        # The jmunch default range no longer matches.
        assert is_jmunch_url("http://127.0.0.1:7879/v1") is False


class TestDetectionFromEnv:
    def test_none_when_no_env(self):
        assert detected_jmunch_endpoint() is None
        assert is_jmunch_in_use() is False

    def test_detects_extraction_base_url(self, monkeypatch):
        monkeypatch.setenv("MEMORY_EXTRACTION_BASE_URL", "http://127.0.0.1:7881/v1")
        assert detected_jmunch_endpoint() == "http://127.0.0.1:7881/v1"
        assert is_jmunch_in_use() is True

    def test_detects_openai_base_url_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:7884/v1")
        assert detected_jmunch_endpoint() == "http://localhost:7884/v1"
        assert is_jmunch_in_use() is True

    def test_extraction_var_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("MEMORY_EXTRACTION_BASE_URL", "http://127.0.0.1:7882/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:7885/v1")
        assert detected_jmunch_endpoint() == "http://127.0.0.1:7882/v1"

    def test_cloud_endpoint_is_not_jmunch(self, monkeypatch):
        monkeypatch.setenv("MEMORY_EXTRACTION_BASE_URL", "https://api.openai.com/v1")
        assert detected_jmunch_endpoint() is None
        assert is_jmunch_in_use() is False
