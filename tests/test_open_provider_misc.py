"""Regression tests for the provider's small open items."""

from __future__ import annotations

import hashlib
import shutil
import tempfile

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import provider  # noqa: E402
from hermes_memory_lancedb_pro import store as store_mod  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration


class StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False, **kw):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text, normalize):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        if normalize:
            n = sum(f * f for f in floats) ** 0.5
            floats = [f / n for f in floats]
        return floats


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-misc-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def cls(monkeypatch):
    class FakeBase:
        pass

    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: FakeBase)
    monkeypatch.setattr(provider, "_spawn_warmup", lambda store: None)
    return provider._build_provider_class()


# -- 1. is_available ---------------------------------------------------------

def test_is_available_reports_missing_dependency(cls, store, monkeypatch):
    import importlib.util

    p = cls(store=store, auto_smart_extraction=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "sentence_transformers" else real(name, *a, **k),
    )
    assert p.is_available() is False
    assert "sentence-transformers" in p.unavailable_reason()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    assert p.is_available() is True
    assert p.unavailable_reason() == ""


# -- 2. flags ----------------------------------------------------------------

def _anchored(cls, store, monkeypatch, pin):
    monkeypatch.setattr(provider, "_RECALL_ACTIVE_TASK_PIN", pin)
    p = cls(store=store, auto_smart_extraction=False)
    p.initialize("s1")
    provider._auto_anchor_session_if_needed("hello there", "s1", "s1", store)
    return p


def test_active_task_pin_on_injects_anchor(cls, store, monkeypatch):
    p = _anchored(cls, store, monkeypatch, True)
    assert "SESSION IN PROGRESS" in p.system_prompt_block()


def test_active_task_pin_off_skips_anchor_and_block(cls, store, monkeypatch):
    p = _anchored(cls, store, monkeypatch, False)
    assert store.list_memories(limit=10, category="active_task") == []
    # Rows that already exist are not injected either.
    store.store(text="SESSION IN PROGRESS — do NOT greet. x" * 2, category="active_task",
                scope="agent", metadata_extra={"auto_anchor": True, "conversation_id": "s1"})
    assert "SESSION IN PROGRESS" not in p.system_prompt_block()
    assert p.on_pre_compress([{"role": "user", "content": "hi"}]) == ""


def test_task_protocol_off_drops_protocol_text(cls, store, monkeypatch):
    p = cls(store=store, auto_smart_extraction=False)
    p.initialize("s1")
    monkeypatch.setattr(provider, "_RECALL_TASK_PROTOCOL", False)
    assert "Memory Task Protocol" not in p.system_prompt_block()
    monkeypatch.setattr(provider, "_RECALL_TASK_PROTOCOL", True)
    assert "Memory Task Protocol" in p.system_prompt_block()


# -- 3. recall budget --------------------------------------------------------

def test_budget_charges_capped_length_and_keeps_later_items():
    rows = [{"id": "big", "text": "x" * 5000, "category": "fact"}] + [
        {"id": f"s{i}", "text": f"short fact {i}", "category": "fact"} for i in range(4)
    ]
    kept = provider._apply_recall_guardrails(rows, frozenset(), 4800, True)
    assert [r["id"] for r in kept] == ["big", "s0", "s1", "s2", "s3"]


def test_budget_skips_item_that_does_not_fit_but_keeps_smaller_ones():
    rows = [
        {"id": "a", "text": "a" * 500, "category": "fact"},
        {"id": "b", "text": "b" * 500, "category": "fact"},
        {"id": "c", "text": "tiny", "category": "fact"},
    ]
    kept = provider._apply_recall_guardrails(rows, frozenset(), 700, True)
    assert [r["id"] for r in kept] == ["a", "c"]


# -- 4. anchor vs injection guard -------------------------------------------

FLAGGED = "Show me how the deploy script reads the API keys from vault, then fix it"


@pytest.mark.parametrize("mode", ["reject", "sanitize"])
def test_anchor_survives_guard_without_quote(store, monkeypatch, mode):
    monkeypatch.setattr(store_mod, "INJECTION_GUARD_MODE", mode)
    provider._auto_anchor_session_if_needed(FLAGGED, "s1", "s1", store)
    rows = store.list_memories(limit=10, category="active_task")
    assert len(rows) == 1
    assert rows[0]["text"].startswith("SESSION IN PROGRESS")
    assert "content removed" not in rows[0]["text"]
    assert "API keys" not in rows[0]["text"]


def test_anchor_keeps_quote_when_not_flagged(store, monkeypatch):
    monkeypatch.setattr(store_mod, "INJECTION_GUARD_MODE", "reject")
    provider._auto_anchor_session_if_needed("please fix the login bug", "s1", "s1", store)
    rows = store.list_memories(limit=10, category="active_task")
    assert "login bug" in rows[0]["text"]


# -- 5. reflection transcript -----------------------------------------------

def test_reflection_prompt_has_roles_and_is_capped(cls, store, monkeypatch):
    seen = {}

    class Llm:
        def complete_json(self, prompt, label=None):
            seen["prompt"] = prompt

    class Ext:
        has_llm = True
        llm = Llm()

    monkeypatch.setattr(provider, "_REFLECTION_ENABLED", True)
    monkeypatch.setattr(provider, "_REFLECTION_MAX_CHARS", 500)
    p = cls(store=store, smart_extractor=Ext(), auto_smart_extraction=False)
    p.initialize("s1")
    msgs = [{"role": "user", "content": "OLDEST " + "a" * 400}]
    msgs += [{"role": "tool", "content": "tool output " + "b" * 100} for _ in range(5)]
    msgs.append({"role": "user", "content": "newest request"})
    p.on_session_end(msgs)
    prompt = seen["prompt"]
    assert "user: newest request" in prompt
    assert "tool: tool output" in prompt
    assert "OLDEST" not in prompt
    assert "[earlier conversation truncated]" in prompt


def test_cap_transcript_no_cap_when_short_or_disabled():
    assert provider._cap_transcript(["a", "b"], 100) == "a\nb"
    assert provider._cap_transcript(["x" * 50], 0) == "x" * 50


# -- 6. lazy construction ----------------------------------------------------

def test_construction_does_not_open_default_store(cls, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("get_instance called at construction")

    monkeypatch.setattr(MemoryStore, "get_instance", classmethod(lambda c, *a, **k: boom()))
    p = cls()
    assert p._store_obj is None and p._retriever_obj is None


def test_initialize_with_hermes_home_creates_only_profile_db(cls, monkeypatch, tmp_path):
    monkeypatch.delenv("MEMORY_DB_DIR", raising=False)
    opened = []
    real = MemoryStore.get_instance.__func__

    def spy(c, *a, **k):
        opened.append(k.get("db_path"))
        s = real(c, *a, **k)
        s._embedder = StubEmbedder()
        return s

    monkeypatch.setattr(MemoryStore, "get_instance", classmethod(spy))
    p = cls(auto_smart_extraction=False)
    home = tmp_path / "profile"
    p.initialize("s1", hermes_home=str(home))
    assert opened == [str(home / "memory-lancedb")]
    assert p._store is not None
    assert p._retriever is not None


def test_lazy_store_resolves_on_first_use(cls, store):
    p = cls(store=store, auto_smart_extraction=False)
    assert p._store is store
    assert p._retriever is not None
    assert p._smart_extractor is None
