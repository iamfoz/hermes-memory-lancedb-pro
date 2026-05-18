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


# ---------------------------------------------------------------------------
# New-hooks integration (requires hermes-agent + LanceDB)
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402


class _StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text: str, normalize: bool):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        if normalize:
            n = sum(f * f for f in floats) ** 0.5
            if n > 0:
                floats = [f / n for f in floats]
        return floats


@pytest.fixture
def real_store():
    tmpdir = tempfile.mkdtemp(prefix="hermes-prov-hooks-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = _StubEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
class TestNewHookOverrides:
    """The plugin's recall + observation hooks, verified against a real
    LanceDB store without needing hermes-agent itself to be the version
    that calls them — we instantiate the provider and call directly."""

    def setup_method(self):
        if provider._self_check() != "real":
            pytest.skip("hermes-agent not on PYTHONPATH")

    def test_does_not_override_before_prompt_build(self):
        """The plugin must NOT override before_prompt_build: the host's
        prefetch_all() skips any provider that does, and the host never
        calls before_prompt_build — overriding it would silently disable
        our prefetch recall."""
        import agent.memory_provider as agent_mp  # noqa: PLC0415

        cls = provider.LanceDBProMemoryProvider
        assert "before_prompt_build" not in vars(cls)
        # The observation hooks are still overridden.
        assert cls.on_recall_used is not agent_mp.MemoryProvider.on_recall_used
        assert cls.on_tool_call_observed is not agent_mp.MemoryProvider.on_tool_call_observed

    def test_prefetch_returns_recalled_memory(self, real_store):
        """prefetch is the host's real recall path; it returns the
        recalled memory formatted for the user-message position."""
        real_store.store(
            text="user prefers UK English in writing",
            metadata_extra={"source_session": "sess-A"},
        )
        p = provider.LanceDBProMemoryProvider(store=real_store)

        prefetch_block = p.prefetch("UK English", session_id="sess-A")
        assert prefetch_block.strip()
        assert "UK English" in prefetch_block
        assert prefetch_block.startswith("- [")

    def test_prefetch_caches_pending_ids(self, real_store):
        """prefetch populates _pending_used_ids[session_id] for later
        credit by on_recall_used / sync_turn."""
        real_store.store(
            text="cacheable preference content",
            metadata_extra={"source_session": "sess-B"},
        )
        p = provider.LanceDBProMemoryProvider(store=real_store)
        out = p.prefetch("cacheable", session_id="sess-B")
        assert out  # not empty
        assert "sess-B" in p._pending_used_ids
        assert len(p._pending_used_ids["sess-B"]) >= 1

    def test_on_recall_used_credits_anchor_match(self, real_store):
        """Memories whose text appears in the response get credited via
        mark_recall_used; memories that don't are dropped silently."""
        used_id = real_store.store(
            text="user always answers in markdown lists",
            metadata_extra={"source_session": "sess-C"},
        )
        unused_id = real_store.store(
            text="completely unrelated factoid about quokkas",
            metadata_extra={"source_session": "sess-C"},
        )
        p = provider.LanceDBProMemoryProvider(store=real_store)
        # Simulate prefetch having populated _pending_used_ids
        p._pending_used_ids["sess-C"] = [used_id, unused_id]

        # Response contains an anchor of the used memory; nothing of unused
        response = "Sure — I'll keep my answer in markdown lists going forward."
        p.on_recall_used(response, session_id="sess-C")

        # _pending_used_ids should be drained for that session (so
        # sync_turn doesn't double-credit)
        assert "sess-C" not in p._pending_used_ids
        # Used memory got bumped
        used_meta = real_store.get_by_id(used_id)["metadata"]
        unused_meta = real_store.get_by_id(unused_id)["metadata"]
        assert used_meta["access_count"] >= 1
        # Unused memory was NOT credited
        assert unused_meta.get("access_count", 0) == 0

    def test_on_recall_used_with_empty_response_is_noop(self, real_store):
        mid = real_store.store(text="some memory", metadata_extra={"source_session": "x"})
        p = provider.LanceDBProMemoryProvider(store=real_store)
        p._pending_used_ids["x"] = [mid]
        p.on_recall_used("", session_id="x")
        # No crediting happened — count is still 0
        meta = real_store.get_by_id(mid)["metadata"]
        assert meta.get("access_count", 0) == 0
        # But _pending_used_ids was popped (empty response → drain anyway,
        # since we pop before checking content)
        assert "x" not in p._pending_used_ids

    def test_on_tool_call_observed_is_noop_but_callable(self, real_store):
        p = provider.LanceDBProMemoryProvider(store=real_store)
        # Should not raise for any input shape
        p.on_tool_call_observed("read_file", {"path": "/foo"}, "content")
        p.on_tool_call_observed("broken", {}, "Error: oops", success=False)
        p.on_tool_call_observed("noargs", {}, None)


@pytest.mark.integration
class TestNoDoubleCreditOnNewHost:
    """Verifies sync_turn doesn't re-credit memories that on_recall_used
    has already credited. This is the key correctness invariant for the
    new-host code path."""

    def setup_method(self):
        if provider._self_check() != "real":
            pytest.skip("hermes-agent not on PYTHONPATH")

    def test_sync_turn_after_on_recall_used_is_a_noop_for_credit(self, real_store):
        mid = real_store.store(
            text="I prefer Vim shortcuts in my IDE",
            metadata_extra={"source_session": "sess-D"},
        )
        p = provider.LanceDBProMemoryProvider(store=real_store)
        # Pretend the host called prefetch then on_recall_used
        p._pending_used_ids["sess-D"] = [mid]
        p.on_recall_used("I'll remember your Vim shortcuts preference.",
                         session_id="sess-D")
        count_after_on_recall_used = real_store.get_by_id(mid)["metadata"]["access_count"]

        # Now the host calls sync_turn — _pending_used_ids has been
        # drained, so the legacy crediting block is skipped.
        p.sync_turn("user msg", "assistant msg", session_id="sess-D")
        count_after_sync_turn = real_store.get_by_id(mid)["metadata"]["access_count"]

        # No double-credit — count is the same
        assert count_after_sync_turn == count_after_on_recall_used


@pytest.mark.integration
class TestOldHostStillWorks:
    """When running against an old hermes-agent that DOESN'T call the
    new hooks, prefetch + sync_turn must still credit recalls
    correctly (the legacy timing-based path)."""

    def setup_method(self):
        if provider._self_check() != "real":
            pytest.skip("hermes-agent not on PYTHONPATH")

    def test_legacy_flow_still_credits_via_sync_turn(self, real_store):
        mid = real_store.store(
            text="legacy recall credited via sync_turn",
            metadata_extra={"source_session": "sess-E"},
        )
        p = provider.LanceDBProMemoryProvider(store=real_store)
        # Old host: only prefetch + sync_turn fire
        p.prefetch("legacy", session_id="sess-E")
        # _pending_used_ids should be populated
        assert "sess-E" in p._pending_used_ids
        # Old host calls sync_turn (no on_recall_used in between)
        p.sync_turn("user msg", "assistant msg", session_id="sess-E")
        meta = real_store.get_by_id(mid)["metadata"]
        # mark_recall_used ran in sync_turn — the prefetched id was credited
        assert meta.get("access_count", 0) >= 1
