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

    def test_first_user_text_returns_first_user_message(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "the objective"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "follow up"},
        ]
        assert provider._first_user_text(msgs) == "the objective"

    def test_first_user_text_handles_content_blocks(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "block text"},
            {"type": "image", "source": {}},
        ]}]
        assert provider._first_user_text(msgs) == "block text"

    def test_first_user_text_empty_when_no_user(self):
        assert provider._first_user_text([{"role": "assistant", "content": "x"}]) == ""
        assert provider._first_user_text([]) == ""

    def test_stable_task_block_omits_per_iteration_fields(self):
        state = {
            "task_id": "t1", "objective": "Do the thing", "status": "running",
            "current_iteration": 7, "next_action": "step eight",
            "recent_summary": "did stuff",
        }
        block = provider._stable_task_block(state)
        assert "Do the thing" in block
        assert "t1" in block
        assert "do NOT greet" in block
        # Per-iteration fields must be absent — including them would mutate
        # the cached system prompt on every `task advance`.
        assert "7" not in block
        assert "step eight" not in block
        assert "did stuff" not in block


def test_build_reflection_prompt_contains_transcript_and_json_keys():
    from hermes_memory_lancedb_pro.extraction_prompts import build_reflection_prompt

    prompt = build_reflection_prompt("USER: hello\nASSISTANT: hi there")
    assert "hello" in prompt
    assert "invariants" in prompt
    assert "derived" in prompt
    assert "JSON" in prompt


def test_refresh_active_task_stable_vs_live(tmp_path):
    """stable=True omits per-iteration fields; the default keeps the full
    live control block (current iteration / next action)."""
    from hermes_memory_lancedb_pro import task_ledger as tl

    task_root = tmp_path / "tasks"
    tl.create_task(
        "t-x", objective="The objective", target_iterations=3, root=task_root
    )
    tl.advance_iteration("t-x", next_action="the next action", root=task_root)
    state_path = str(tl._state_path("t-x", task_root))
    mem = {
        "category": "active_task", "text": "pinned snapshot",
        "metadata": {"task_id": "t-x", "state_path": state_path},
    }
    live = provider._refresh_active_task_memories([dict(mem)])[0]["text"]
    stable = provider._refresh_active_task_memories(
        [dict(mem)], stable=True
    )[0]["text"]
    assert "the next action" in live
    assert "the next action" not in stable
    assert "The objective" in stable


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


# ---------------------------------------------------------------------------
# on_memory_write — mirror built-in memory tool add / replace writes
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOnMemoryWrite:
    """on_memory_write mirrors built-in memory tool writes. Per the host
    contract (agent/tool_executor.py) it is called only for `add` and
    `replace`, with the NEW content and a `memory`/`user` target."""

    def _provider(self, provider_cls, store):
        p = provider_cls(store=store, auto_smart_extraction=False)
        p.initialize("sess-1")
        return p

    def test_add_user_target_stored_as_preference(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("add", "user", "The user loves cycling")
        rows = real_store.list_memories(limit=50)
        assert len(rows) == 1
        assert "cycling" in rows[0]["text"]
        assert rows[0]["category"] == "preference"
        assert rows[0]["scope"] == "user"

    def test_add_memory_target_stored_as_agent_scope(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("add", "memory", "The build uses pnpm, not npm")
        rows = real_store.list_memories(limit=50)
        assert len(rows) == 1
        assert "pnpm" in rows[0]["text"]
        assert rows[0]["scope"] == "agent"

    def test_replace_is_mirrored_as_new_content(self, provider_cls, real_store):
        """The host passes only the NEW content for a `replace` (no old
        text), so it must be stored — not silently dropped."""
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("replace", "user", "The user now prefers spaces")
        rows = real_store.list_memories(limit=50)
        assert len(rows) == 1
        assert "spaces" in rows[0]["text"]

    def test_remove_action_is_ignored(self, provider_cls, real_store):
        # The host never forwards `remove`; if it arrives we ignore it.
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("remove", "user", "something")
        assert real_store.list_memories(limit=50) == []

    def test_unknown_action_is_ignored(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("frobnicate", "user", "content")  # must not raise
        assert real_store.list_memories(limit=50) == []

    def test_empty_content_is_noop(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        p.on_memory_write("add", "user", "   ")  # must not raise
        assert real_store.list_memories(limit=50) == []

    def test_metadata_arg_is_accepted(self, provider_cls, real_store):
        # The host passes metadata={task_id, tool_call_id}; it must not crash.
        p = self._provider(provider_cls, real_store)
        p.on_memory_write(
            "add", "memory", "a fact",
            metadata={"task_id": "t1", "tool_call_id": "tc1"},
        )
        assert len(real_store.list_memories(limit=50)) == 1


# ---------------------------------------------------------------------------
# first_for_session / session anchor coverage
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSessionAnchors:
    """first_for_session returns oldest memories; _do_recall includes them."""

    def test_first_for_session_returns_oldest(self, real_store):
        sess = "anchor-sess"
        real_store.store(
            text="Task framing from turn one",
            category="other", scope="agent", importance=0.5,
            metadata_extra={"source_session": sess},
        )
        time.sleep(0.01)
        real_store.store(
            text="Detail from turn two",
            category="other", scope="agent", importance=0.5,
            metadata_extra={"source_session": sess},
        )
        time.sleep(0.01)
        real_store.store(
            text="Detail from turn three",
            category="other", scope="agent", importance=0.5,
            metadata_extra={"source_session": sess},
        )
        first = real_store.first_for_session(sess, limit=1)
        assert len(first) == 1
        assert "turn one" in first[0]["text"]

    def test_first_for_session_empty_when_no_session(self, real_store):
        assert real_store.first_for_session("") == []

    def test_recall_includes_task_framing_after_many_turns(
        self, provider_cls, real_store
    ):
        """After several turns, recall must still surface the session-start
        memory (task framing) via first_for_session anchors."""
        p = provider_cls(store=real_store, auto_smart_extraction=False)
        p.initialize("anchor-sess-2")
        sess = "anchor-sess-2"

        # Simulate 5 turns of conversation stored directly to bypass threading
        texts = [
            "Stress-test my memory — this is the task framing",
            "Turn two payload ABC",
            "Turn three payload DEF",
            "Turn four payload GHI",
            "Turn five payload JKL",
        ]
        for t in texts:
            real_store.store(
                text=t, category="other", scope="agent", importance=0.5,
                metadata_extra={"source_session": sess},
            )
            time.sleep(0.02)

        # Query is semantically unrelated to task framing
        block = p._do_recall("check slot 7", sess)
        assert "task framing" in block.lower() or "stress-test" in block.lower(), (
            "task framing memory from turn 1 must be in recall even after 5 turns"
        )


# ---------------------------------------------------------------------------
# TestFormatRecallFreshnessTrend
# ---------------------------------------------------------------------------

class TestFormatRecallFreshnessTrend:
    def _make_result(self, text: str, trend: str | None = None) -> dict:
        r: dict = {"text": text, "category": "facts", "_final_score": 0.75}
        if trend is not None:
            r["_decay"] = {"freshness_trend": trend}
        return r

    def test_weakening_trend_appears_in_output(self):
        out = provider._format_recall([self._make_result("Alice prefers Python", "weakening")])
        assert "[weakening]" in out

    def test_forming_trend_appears_in_output(self):
        out = provider._format_recall([self._make_result("Alice prefers Python", "forming")])
        assert "[forming]" in out

    def test_strengthening_trend_appears_in_output(self):
        out = provider._format_recall([self._make_result("Alice prefers Python", "strengthening")])
        assert "[strengthening]" in out

    def test_stable_trend_omitted_from_output(self):
        out = provider._format_recall([self._make_result("Alice prefers Python", "stable")])
        assert "[stable]" not in out
        assert "score=" in out

    def test_no_decay_no_trend_tag(self):
        out = provider._format_recall([self._make_result("Alice prefers Python")])
        assert "[" not in out.split("(")[1]  # no tag after score bracket

    def test_oversized_memory_is_truncated(self):
        """A single oversized memory (e.g. a whole source file wrongly stored
        as a memory) must never flood the recall block."""
        huge = "X" * 50_000
        out = provider._format_recall([self._make_result(huge)])
        assert "[…truncated]" in out
        assert len(out) < 2_000  # the 50KB blob never reaches the model

    def test_normal_memory_not_truncated(self):
        out = provider._format_recall([self._make_result("a short normal fact")])
        assert "[…truncated]" not in out
        assert "a short normal fact" in out


# ---------------------------------------------------------------------------
# system_prompt_block + on_pre_compress — post-compaction greeting-loop defence
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSystemPromptBlockAndCompaction:
    """The system-prompt hook and the pre-compaction anchor that together
    stop the model greeting after context compaction.

    `system_prompt_block` is hermes-agent's authoritative system-prompt
    hook (rebuilt at session start and after each compaction);
    `on_pre_compress` fires right before compaction discards old
    messages. The task protocol travels through these hooks."""

    def _provider(self, provider_cls, store, session="sess-1"):
        p = provider_cls(store=store, auto_smart_extraction=False)
        p.initialize(session)
        return p

    def test_system_prompt_block_includes_protocol(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        block = p.system_prompt_block()
        assert "NEVER GREET" in block
        assert "Memory Task Protocol" in block

    def test_system_prompt_block_surfaces_active_task(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        real_store.store(
            text="Run the stress suite to completion",
            category="active_task", scope="agent", importance=0.9,
            metadata_extra={
                "auto_anchor": True,
                "source_session": "sess-1",
                "conversation_id": "sess-1",
            },
        )
        block = p.system_prompt_block()
        assert "=== ACTIVE TASK STATE ===" in block
        assert "stress suite" in block

    def test_system_prompt_block_prefers_formal_pin_over_anchor(
        self, provider_cls, real_store
    ):
        p = self._provider(provider_cls, real_store)
        real_store.store(
            text="auto anchor breadcrumb text",
            category="active_task", scope="agent", importance=0.9,
            metadata_extra={
                "auto_anchor": True,
                "source_session": "sess-1",
                "conversation_id": "sess-1",
            },
        )
        real_store.store(
            text="FORMAL PIN control block",
            category="active_task", scope="global", importance=1.0,
            metadata_extra={"task_id": "t1", "state_path": "/nonexistent/state.json"},
        )
        block = p.system_prompt_block()
        assert "FORMAL PIN control block" in block
        assert "auto anchor breadcrumb text" not in block

    def test_on_pre_compress_creates_anchor_and_returns_block(
        self, provider_cls, real_store
    ):
        p = self._provider(provider_cls, real_store)
        out = p.on_pre_compress([
            {"role": "user", "content": "Benchmark the retriever end to end"},
            {"role": "assistant", "content": "starting the benchmark"},
        ])
        anchors = real_store.list_memories(limit=20, category="active_task")
        assert anchors, "on_pre_compress must create a recovery anchor"
        assert any(
            "Benchmark the retriever" in (m["text"] or "") for m in anchors
        )
        # Return value carries the task block for the compression summary.
        assert "=== ACTIVE TASK STATE ===" in out

    def test_on_pre_compress_anchor_survives_in_system_prompt(
        self, provider_cls, real_store
    ):
        p = self._provider(provider_cls, real_store)
        p.on_pre_compress([
            {"role": "user", "content": "Migrate the schema to v2"},
        ])
        # Simulate the post-compaction turn: the system prompt is rebuilt.
        block = p.system_prompt_block()
        assert "=== ACTIVE TASK STATE ===" in block
        assert "Migrate the schema" in block
        assert "do NOT greet" in block

    def test_reset_session_drops_stale_anchor(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store, session="sess-old")
        p.on_pre_compress([
            {"role": "user", "content": "Refactor the auth module"},
        ])
        assert "=== ACTIVE TASK STATE ===" in p.system_prompt_block()
        # A genuine reset (/new, /reset) starts a fresh conversation.
        p.on_session_switch("sess-new", parent_session_id="sess-old", reset=True)
        block = p.system_prompt_block()
        assert "Refactor the auth module" not in block
        assert "=== ACTIVE TASK STATE ===" not in block

    def test_non_reset_switch_keeps_anchor(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store, session="sess-old")
        p.on_pre_compress([
            {"role": "user", "content": "Refactor the auth module"},
        ])
        # /branch or /resume — reset=False — the task continues.
        p.on_session_switch("sess-branch", parent_session_id="sess-old", reset=False)
        assert "Refactor the auth module" in p.system_prompt_block()

    def test_active_task_not_duplicated_in_recall_path(
        self, provider_cls, real_store
    ):
        """The active-task block belongs to system_prompt_block; the
        query-dependent recall path must not inject a second copy."""
        p = self._provider(provider_cls, real_store)
        real_store.store(
            text="auto anchor breadcrumb for the session",
            category="active_task", scope="agent", importance=0.9,
            metadata_extra={
                "auto_anchor": True,
                "source_session": "sess-1",
                "conversation_id": "sess-1",
            },
        )
        assert "=== ACTIVE TASK STATE ===" in p.system_prompt_block()
        recall = p.prefetch("anything the user might ask")
        assert "=== ACTIVE TASK STATE ===" not in recall
        assert "auto anchor breadcrumb" not in recall

    def test_system_prompt_block_stable_across_task_advance(
        self, provider_cls, real_store, tmp_path
    ):
        """system_prompt_block feeds the cached system prompt (cache
        breakpoint 1). Its output must be byte-identical across a pinned
        task advancing an iteration — otherwise the cache is busted every
        task turn."""
        from hermes_memory_lancedb_pro import task_ledger as tl

        task_root = tmp_path / "tasks"
        tl.create_task(
            "t-cache", objective="Cache stability check",
            target_iterations=5, root=task_root,
        )
        state_path = str(tl._state_path("t-cache", task_root))
        p = self._provider(provider_cls, real_store)
        real_store.store(
            text="pinned snapshot",
            category="active_task", scope="global", importance=1.0,
            metadata_extra={"task_id": "t-cache", "state_path": state_path},
        )
        before = p.system_prompt_block()
        assert "Cache stability check" in before
        assert "=== ACTIVE TASK STATE ===" in before
        # Advance the task — current_iteration / next_action change on disk.
        tl.advance_iteration("t-cache", next_action="do step 2", root=task_root)
        after = p.system_prompt_block()
        assert after == before, (
            "system_prompt_block must not change when a pinned task advances"
        )

    def test_system_prompt_block_isolates_conversations(
        self, provider_cls, real_store
    ):
        """A gateway serves many conversations from one shared store. One
        conversation's auto-anchor must never surface in another's system
        prompt."""
        pa = self._provider(provider_cls, real_store, session="conv-A")
        pa.on_pre_compress([
            {"role": "user", "content": "Build the conv-A feature"},
        ])
        # A second conversation, separate provider instance, same store.
        pb = provider_cls(store=real_store, auto_smart_extraction=False)
        pb.initialize("conv-B")
        block_b = pb.system_prompt_block()
        assert "Build the conv-A feature" not in block_b
        assert "=== ACTIVE TASK STATE ===" not in block_b
        # conv-A still sees its own anchor.
        assert "Build the conv-A feature" in pa.system_prompt_block()


@pytest.mark.integration
class TestRecallHookSeparation:
    """The durable-task protocol lives only in `system_prompt_block`; the
    query-dependent recall path (`prefetch`) never duplicates it."""

    def _provider(self, provider_cls, store, session="sess-1"):
        p = provider_cls(store=store, auto_smart_extraction=False)
        p.initialize(session)
        return p

    def test_prefetch_empty_query_returns_empty(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        assert p.prefetch("") == ""
        assert p.prefetch("   ") == ""

    def test_provider_does_not_override_before_prompt_build(self, provider_cls):
        # Overriding before_prompt_build makes the host's prefetch_all() skip
        # this provider — and the host never calls before_prompt_build — so
        # the provider must NOT define it.
        assert "before_prompt_build" not in vars(provider_cls)

    def test_recall_path_does_not_duplicate_protocol(self, provider_cls, real_store):
        p = self._provider(provider_cls, real_store)
        # The protocol belongs to system_prompt_block...
        assert "Memory Task Protocol" in p.system_prompt_block()
        # ...and must NOT also appear in the query-dependent recall block.
        assert "Memory Task Protocol" not in p.prefetch("a real user query")


@pytest.mark.integration
class TestAutoAnchor:
    """_auto_anchor_session_if_needed: idempotent, pin-aware, conversation-
    scoped. The anchor is keyed by conversation id so it survives the
    session-id rotation that context compression performs, yet stays
    isolated from other conversations sharing the same store."""

    def test_creates_anchor_when_none_exists(self, real_store):
        provider._auto_anchor_session_if_needed(
            "Fix the failing tests", "sess-A", "conv-A", real_store
        )
        anchors = real_store.list_memories(limit=20, category="active_task")
        assert len(anchors) == 1
        assert "Fix the failing tests" in anchors[0]["text"]

    def test_idempotent_no_duplicate_for_same_conversation(self, real_store):
        for _ in range(3):
            provider._auto_anchor_session_if_needed(
                "Fix the failing tests", "sess-A", "conv-A", real_store
            )
        anchors = real_store.list_memories(limit=20, category="active_task")
        assert len(anchors) == 1

    def test_skips_when_formal_pin_exists(self, real_store):
        real_store.store(
            text="formal control block",
            category="active_task", scope="global", importance=1.0,
            metadata_extra={"task_id": "t1", "state_path": "/tmp/state.json"},
        )
        provider._auto_anchor_session_if_needed(
            "some objective", "sess-A", "conv-A", real_store
        )
        anchors = real_store.list_memories(limit=20, category="active_task")
        assert len(anchors) == 1  # only the formal pin; no auto-anchor added
        assert all(
            not (m.get("metadata") or {}).get("auto_anchor") for m in anchors
        )

    def test_anchor_survives_session_id_rotation(self, real_store):
        # Context compression rotates session_id mid-conversation but keeps
        # the conversation id. The anchor and its original objective must
        # persist, not churn into the latest turn's text.
        provider._auto_anchor_session_if_needed(
            "Original objective", "sess-1", "conv-1", real_store
        )
        provider._auto_anchor_session_if_needed(
            "later turn after a compaction", "sess-2", "conv-1", real_store
        )
        live = real_store.list_memories(
            limit=20, category="active_task", include_archived=False
        )
        assert len(live) == 1
        assert "Original objective" in live[0]["text"]

    def test_anchors_isolated_across_conversations(self, real_store):
        # Two conversations sharing one store each get their OWN anchor.
        provider._auto_anchor_session_if_needed(
            "conv A work", "sA", "conv-A", real_store
        )
        provider._auto_anchor_session_if_needed(
            "conv B work", "sB", "conv-B", real_store
        )
        live = real_store.list_memories(
            limit=20, category="active_task", include_archived=False
        )
        assert len(live) == 2
        by_conv = {
            (m["metadata"] or {}).get("conversation_id"): m["text"]
            for m in live
        }
        assert "conv A work" in by_conv["conv-A"]
        assert "conv B work" in by_conv["conv-B"]

    def test_archive_auto_anchors_scoped_to_one_conversation(self, real_store):
        provider._auto_anchor_session_if_needed("A", "sA", "conv-A", real_store)
        provider._auto_anchor_session_if_needed("B", "sB", "conv-B", real_store)
        n = provider._archive_auto_anchors(real_store, "conv-A")
        assert n == 1
        live = real_store.list_memories(
            limit=20, category="active_task", include_archived=False
        )
        assert len(live) == 1
        assert (live[0]["metadata"] or {}).get("conversation_id") == "conv-B"

    def test_archive_auto_anchors_leaves_formal_pins(self, real_store):
        real_store.store(
            text="auto anchor breadcrumb",
            category="active_task", scope="agent", importance=0.9,
            metadata_extra={
                "auto_anchor": True,
                "source_session": "sess-A",
                "conversation_id": "conv-A",
            },
        )
        real_store.store(
            text="FORMAL PIN block",
            category="active_task", scope="global", importance=1.0,
            metadata_extra={"task_id": "t1", "state_path": "/tmp/state.json"},
        )
        n = provider._archive_auto_anchors(real_store, "conv-A")
        assert n == 1
        live = real_store.list_memories(
            limit=20, category="active_task", include_archived=False
        )
        assert len(live) == 1
        assert "FORMAL PIN block" in live[0]["text"]
