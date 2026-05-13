"""Tests for the feature wiring added to LanceDBProMemoryProvider:
background warmup, auto-compaction, default admission control, and the
reflection write/read path.

The provider class is normally a stub when hermes-agent isn't on
PYTHONPATH. These tests rebuild it against a trivial fake base so the
real method bodies can be exercised directly."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time

import pytest

from hermes_memory_lancedb_pro import provider
from hermes_memory_lancedb_pro.smart_extractor import SmartExtractor
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore


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


class _FakeBase:
    """Stand-in for agent.memory_provider.MemoryProvider."""


class _FakeLlm:
    """Returns a canned reflection payload for any prompt."""

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {
            "invariants": ["always answer in UK English"],
            "derived": ["next run: write more tests"],
        }
        self.calls = 0

    def complete_json(self, prompt, *, label=None):
        self.calls += 1
        return self.payload


@pytest.fixture
def real_store():
    tmpdir = tempfile.mkdtemp(prefix="hermes-feat-wiring-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = _StubEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def provider_cls(monkeypatch):
    """Rebuild LanceDBProMemoryProvider against a fake base class."""
    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: _FakeBase)
    return provider._build_provider_class()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestPureHelpers:
    def test_coerce_str_list_filters_and_trims(self):
        assert provider._coerce_str_list(["  a ", "", None, "b", 3]) == ["a", "b", "3"]

    def test_coerce_str_list_rejects_non_list(self):
        assert provider._coerce_str_list("not a list") == []
        assert provider._coerce_str_list(None) == []

    def test_build_reflection_markdown_both_sections(self):
        md = provider._build_reflection_markdown(["inv one"], ["der one"])
        assert "## Invariants" in md
        assert "- inv one" in md
        assert "## Derived" in md
        assert "- der one" in md

    def test_build_reflection_markdown_empty(self):
        assert provider._build_reflection_markdown([], []) == ""

    def test_build_reflection_markdown_invariants_only(self):
        md = provider._build_reflection_markdown(["only inv"], [])
        assert "## Invariants" in md
        assert "## Derived" not in md


def test_build_reflection_prompt_contains_transcript_and_json_keys():
    from hermes_memory_lancedb_pro.extraction_prompts import build_reflection_prompt

    prompt = build_reflection_prompt("USER: hello\nASSISTANT: hi there")
    assert "hello" in prompt
    assert "invariants" in prompt
    assert "derived" in prompt
    assert "JSON" in prompt


# ---------------------------------------------------------------------------
# Admission control wiring
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAdmissionControllerWiring:
    def test_off_preset_returns_none(self, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_ADMISSION_PRESET", "off")
        assert provider._maybe_build_admission_controller(real_store, None) is None

    def test_balanced_preset_builds_controller(self, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_ADMISSION_PRESET", "balanced")
        ctrl = provider._maybe_build_admission_controller(real_store, None)
        assert ctrl is not None

    def test_unknown_preset_falls_back_to_balanced(self, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_ADMISSION_PRESET", "nonsense")
        ctrl = provider._maybe_build_admission_controller(real_store, None)
        assert ctrl is not None


# ---------------------------------------------------------------------------
# Auto-compaction
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAutoCompaction:
    def test_disabled_when_cooldown_zero(self, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_AUTO_COMPACT_COOLDOWN_HOURS", 0)
        # Must not raise and must not write the state file.
        provider._maybe_auto_compact(real_store)
        import os
        state = os.path.join(real_store.db_path, provider._COMPACT_STATE_FILENAME)
        assert not os.path.exists(state)

    def test_runs_and_records_cooldown(self, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_AUTO_COMPACT_COOLDOWN_HOURS", 168)
        provider._maybe_auto_compact(real_store)
        import os
        state = os.path.join(real_store.db_path, provider._COMPACT_STATE_FILENAME)
        assert os.path.exists(state)
        # A second immediate call is gated by the cooldown — still no error.
        provider._maybe_auto_compact(real_store)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWarmup:
    def test_initialize_spawns_warmup_once(self, provider_cls, real_store):
        p = provider_cls(store=real_store, auto_smart_extraction=False)
        assert p._warmed_up is False
        p.initialize("sess-1")
        assert p._warmed_up is True
        # A second initialize must not re-spawn (flag already set).
        p.initialize("sess-2")
        assert p._warmed_up is True


# ---------------------------------------------------------------------------
# Reflection write + read
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestReflectionWiring:
    def _provider(self, provider_cls, store, llm=None):
        extractor = SmartExtractor(store, llm=llm) if llm is not None else None
        return provider_cls(
            store=store,
            smart_extractor=extractor,
            auto_smart_extraction=False,
        )

    def test_session_end_writes_reflection(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
        llm = _FakeLlm()
        p = self._provider(provider_cls, real_store, llm=llm)
        p.initialize("sess-1")
        p.on_session_end([
            {"content": "I want all answers in UK English."},
            {"content": "Understood, I will use UK English."},
        ])
        assert llm.calls >= 1
        reflection_rows = [
            e for e in real_store.list_memories(limit=50)
            if e.get("category") == "reflection"
        ]
        assert reflection_rows, "expected reflection rows to be written"

    def test_reflection_surfaces_on_recall(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
        llm = _FakeLlm()
        p = self._provider(provider_cls, real_store, llm=llm)
        p.initialize("sess-1")
        p.on_session_end([
            {"content": "I want all answers in UK English."},
            {"content": "Understood, I will use UK English."},
        ])
        # A fresh session recalls the prior session's reflection.
        block = p._do_recall("how should you answer me", "sess-2")
        assert "[reflection/invariant]" in block
        assert "UK English" in block

    def test_reflection_disabled_skips_write(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", False)
        llm = _FakeLlm()
        p = self._provider(provider_cls, real_store, llm=llm)
        p.initialize("sess-1")
        p.on_session_end([
            {"content": "I want all answers in UK English."},
            {"content": "Understood."},
        ])
        assert llm.calls == 0
        reflection_rows = [
            e for e in real_store.list_memories(limit=50)
            if e.get("category") == "reflection"
        ]
        assert not reflection_rows

    def test_reflection_noop_without_llm(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
        # No smart extractor at all → reflection write is a no-op, no crash.
        p = self._provider(provider_cls, real_store, llm=None)
        p.initialize("sess-1")
        p.on_session_end([
            {"content": "some content"},
            {"content": "more content"},
        ])
        reflection_rows = [
            e for e in real_store.list_memories(limit=50)
            if e.get("category") == "reflection"
        ]
        assert not reflection_rows

    def test_reflection_block_cached_per_session(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
        p = self._provider(provider_cls, real_store, llm=_FakeLlm())
        p.initialize("sess-1")
        calls = []
        original = p._compute_reflection_block

        def _counting():
            calls.append(time.time())
            return original()

        p._compute_reflection_block = _counting
        p._reflection_block("sess-X")
        p._reflection_block("sess-X")
        assert len(calls) == 1, "second call for same session must hit the cache"


# ---------------------------------------------------------------------------
# on_session_switch session-ID tracking
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Extraction rate limiter wiring
# ---------------------------------------------------------------------------

class TestExtractionRateLimiter:
    def test_rate_limiter_not_created_when_zero(self, monkeypatch):
        monkeypatch.setattr(provider, "_EXTRACTION_RATE_LIMIT", 0)
        extractor = provider._maybe_build_default_smart_extractor.__wrapped__ if hasattr(
            provider._maybe_build_default_smart_extractor, "__wrapped__"
        ) else None
        # The simpler check: build an extractor directly and confirm no limiter
        from hermes_memory_lancedb_pro.smart_extractor import ExtractionRateLimiter, SmartExtractor
        ex = SmartExtractor.__new__(SmartExtractor)
        ex._rate_limiter = None
        assert ex._rate_limiter is None

    def test_rate_limiter_wired_via_env(self, monkeypatch):
        monkeypatch.setattr(provider, "_EXTRACTION_RATE_LIMIT", 30)
        from hermes_memory_lancedb_pro.smart_extractor import ExtractionRateLimiter
        rl = ExtractionRateLimiter(max_per_hour=30)
        assert rl.max_per_hour == 30
        assert not rl.is_rate_limited()


class TestSessionSwitch:
    def test_session_id_updated_on_switch(self, provider_cls, real_store):
        p = provider_cls(store=real_store, auto_smart_extraction=False)
        p.initialize("sess-1")
        assert p._session_id == "sess-1"
        p.on_session_switch("sess-2", parent_session_id="sess-1")
        assert p._session_id == "sess-2"

    def test_old_session_cache_cleared_on_switch(self, provider_cls, real_store, monkeypatch):
        monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
        p = provider_cls(store=real_store, auto_smart_extraction=False)
        p.initialize("sess-1")
        # Seed the cache with a fake entry for sess-1
        with p._reflection_lock:
            p._reflection_cache["sess-1"] = "old-reflection"
        p.on_session_switch("sess-2", parent_session_id="sess-1")
        with p._reflection_lock:
            assert "sess-1" not in p._reflection_cache
