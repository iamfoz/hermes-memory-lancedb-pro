"""Tests for the optional Hermes Agent MemoryProvider adapter.

These run without hermes-agent installed: the adapter is built lazily
and the class falls back to a stub that raises a clear ImportError on
instantiation. Once hermes-agent is on PYTHONPATH the real provider
class is constructed and exercised."""

from __future__ import annotations

import importlib

import pytest

from hermes_memory_lancedb_pro import provider


class TestStandaloneImport:
    """Importing the provider module must not require hermes-agent."""

    def test_import_succeeds(self):
        # We're running in an environment where hermes-agent isn't installed
        # so this should produce the stub provider
        assert provider.PROVIDER_NAME == "lancedb_pro"

    def test_self_check_reports_stub(self):
        # When agent.memory_provider isn't importable, _self_check should
        # report "stub". (When hermes-agent IS installed, "real".)
        assert provider._self_check() in {"stub", "real"}

    def test_stub_raises_clear_error(self):
        # If we're in the stub case, instantiation must raise ImportError
        # with a useful message instead of an obscure attribute error.
        if provider._self_check() != "stub":
            pytest.skip("hermes-agent appears to be installed; stub path skipped")
        with pytest.raises(ImportError, match="hermes-agent"):
            provider.LanceDBProMemoryProvider()

    def test_register_memory_provider_raises_when_stub(self):
        if provider._self_check() != "stub":
            pytest.skip("hermes-agent installed; stub path skipped")
        with pytest.raises(ImportError, match="hermes-agent"):
            provider.register_memory_provider()

    def test_format_recall_empty(self):
        assert provider._format_recall([]) == ""

    def test_format_recall_skips_empty_text(self):
        out = provider._format_recall([
            {"id": "1", "text": "", "category": "fact", "_final_score": 0.5},
        ])
        assert out == ""

    def test_format_recall_includes_score(self):
        out = provider._format_recall([
            {
                "id": "1",
                "text": "Martyn prefers UK English",
                "category": "preference",
                "_final_score": 0.83,
            },
        ])
        assert "preference" in out
        assert "Martyn" in out
        assert "0.83" in out


class TestProviderClassFactory:
    """When hermes-agent IS installed (CI with full deps), exercise the
    real provider class. Otherwise these tests are skipped."""

    def setup_method(self):
        if provider._self_check() != "real":
            pytest.skip("hermes-agent not on PYTHONPATH")

    def test_provider_name(self):
        # Re-instantiate with a tiny in-memory-ish test config
        importlib.reload(provider)
        # Real path: provider has tool schemas, name, is_available
        cls = provider.LanceDBProMemoryProvider
        assert hasattr(cls, "name")
