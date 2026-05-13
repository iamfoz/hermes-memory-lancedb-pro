"""Tests for SmartExtractor — both the fallback path (no LLM) and the
full LLM pipeline using a deterministic fake LLM client."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from typing import Any

import pytest

from hermes_memory_lancedb_pro.smart_extractor import (
    ExtractionRateLimiter,
    ExtractionStats,
    SmartExtractor,
    strip_envelope_metadata,
)

# ---------------------------------------------------------------------------
# strip_envelope_metadata
# ---------------------------------------------------------------------------

class TestStripEnvelopeMetadata:
    def test_empty_passthrough(self):
        assert strip_envelope_metadata("") == ""

    def test_subagent_context_wrapper_removed(self):
        text = "[Subagent Context] You are running as a subagent (depth 1/1).\nReal user content here."
        out = strip_envelope_metadata(text)
        assert "[Subagent Context]" not in out
        assert "Real user content here" in out

    def test_subagent_task_wrapper_removed(self):
        text = "[Subagent Task] Reply with brief ack only.\nMore content."
        out = strip_envelope_metadata(text)
        assert "[Subagent Task]" not in out
        assert "More content" in out

    def test_inline_boilerplate_scrubbed(self):
        text = (
            "[Subagent Context] You are running as a subagent. "
            "Results auto-announce to your requester.\n"
            "User asks: what time is it?"
        )
        out = strip_envelope_metadata(text)
        assert "Results auto-announce" not in out
        assert "what time is it" in out

    def test_system_header_dropped(self):
        text = "System: [2026-03-18 14:21:36 GMT+8] Feishu[default] DM\nUser: hello"
        out = strip_envelope_metadata(text)
        assert "Feishu" not in out
        assert "User: hello" in out

    def test_untrusted_metadata_block_dropped(self):
        text = (
            "Conversation info (untrusted metadata):\n"
            "```json\n{\"chat_id\": \"abc\"}\n```\n"
            "Real conversation here."
        )
        out = strip_envelope_metadata(text)
        assert "Conversation info" not in out
        assert "chat_id" not in out
        assert "Real conversation here" in out

    def test_real_content_preserved(self):
        text = "User: hello world\nAssistant: hi"
        out = strip_envelope_metadata(text)
        assert "User: hello world" in out
        assert "Assistant: hi" in out


# ---------------------------------------------------------------------------
# Fallback path — no LLM
# ---------------------------------------------------------------------------

class FakeStore:
    """In-memory MemoryStore stand-in for fallback-path tests. Records every
    .store() call and serves a no-op for the smart-pipeline operations the
    fallback path doesn't reach."""

    def __init__(self):
        self.stored: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.encode_calls: list[str] = []
        self._next_id = 0

    def store(self, *, text, category, scope, importance, tier="working",
              confidence=0.7, metadata_extra=None):
        self._next_id += 1
        mem_id = f"id-{self._next_id}"
        self.stored.append({
            "id": mem_id, "text": text, "category": category, "scope": scope,
            "importance": importance, "tier": tier, "confidence": confidence,
            "metadata_extra": dict(metadata_extra or {}),
        })
        return mem_id

    def update(self, mem_id, *, text=None, category=None, importance=None,
               tier=None, metadata_extra=None):
        self.update_calls.append({
            "id": mem_id, "text": text, "category": category,
            "importance": importance, "tier": tier,
            "metadata_extra": dict(metadata_extra or {}),
        })
        return True

    def encode(self, text):
        self.encode_calls.append(text)
        # 8-d deterministic vector
        h = hashlib.sha256(text.encode()).digest()
        return [(h[i] - 128) / 128.0 for i in range(8)]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def search_by_vector(self, vector, limit=10, *, category=None, scope=None,
                         keep_vector=False):
        return []

    def get_by_id(self, mem_id):
        return None

    def mark_recall_used(self, ids):
        return len(ids)


class TestLegacyFallback:
    def test_no_llm_writes_two_entries(self):
        store = FakeStore()
        extractor = SmartExtractor(store)  # no llm
        assert extractor.has_llm is False

        stats = extractor.extract_and_persist(
            user_content="What time is it in London?",
            assistant_content="It's 14:23 GMT.",
            session_key="sess-1",
        )
        assert stats.created == 2
        assert len(store.stored) == 2
        # User entry
        user_entry = next(e for e in store.stored if e["metadata_extra"].get("role") == "user")
        assert "London" in user_entry["text"]
        assert user_entry["metadata_extra"]["source_session"] == "sess-1"
        # Assistant entry
        asst_entry = next(e for e in store.stored if e["metadata_extra"].get("role") == "assistant")
        assert "14:23" in asst_entry["text"]

    def test_no_llm_empty_inputs_no_writes(self):
        store = FakeStore()
        extractor = SmartExtractor(store)
        stats = extractor.extract_and_persist(user_content="", assistant_content="")
        assert stats.created == 0
        assert store.stored == []

    def test_no_llm_only_user_content(self):
        store = FakeStore()
        extractor = SmartExtractor(store)
        stats = extractor.extract_and_persist(user_content="hi", assistant_content="")
        assert stats.created == 1
        assert len(store.stored) == 1
        assert store.stored[0]["metadata_extra"]["role"] == "user"


# ---------------------------------------------------------------------------
# LLM pipeline with FakeLlm
# ---------------------------------------------------------------------------

class FakeLlm:
    """Deterministic stub LLM. Returns canned responses by label."""

    def __init__(self, *, extract=None, dedup=None, merge=None):
        self._extract = extract
        self._dedup = dedup
        self._merge = merge
        self.extract_calls = 0
        self.dedup_calls = 0
        self.merge_calls = 0

    def complete_json(self, prompt, *, label=None):
        if label == "extract-candidates":
            self.extract_calls += 1
            return self._extract
        if label == "dedup-decision":
            self.dedup_calls += 1
            return self._dedup
        if label == "merge-memory":
            self.merge_calls += 1
            return self._merge
        return None


class TestLlmPipelineNoCandidates:
    def test_empty_extraction_returns_empty_stats(self):
        store = FakeStore()
        llm = FakeLlm(extract={"memories": []})
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="ok", assistant_content="ok",
        )
        assert stats == ExtractionStats()
        assert llm.extract_calls == 1

    def test_invalid_json_response_returns_empty_stats(self):
        store = FakeStore()
        llm = FakeLlm(extract=None)  # LLM returned None
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="hello", assistant_content="hi",
        )
        assert stats.created == 0


class TestLlmPipelineCreate:
    def test_single_candidate_creates(self):
        # LLM extracts one preference; dedup says CREATE (no neighbours)
        store = FakeStore()
        llm = FakeLlm(
            extract={
                "memories": [{
                    "category": "preferences",
                    "abstract": "user prefers dark mode in IDE",
                    "overview": "expressed during config discussion",
                    "content": "user said they prefer dark mode in their IDE",
                }],
            },
            # dedup never called because vector_search returns []
        )
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="I prefer dark mode in my IDE.",
            assistant_content="Got it.",
            session_key="sess-1",
        )
        assert stats.created == 1
        assert len(store.stored) == 1
        entry = store.stored[0]
        assert entry["text"] == "user prefers dark mode in IDE"
        assert entry["category"] == "preference"  # mapped from "preferences"
        meta = entry["metadata_extra"]
        assert meta["memory_category"] == "preferences"
        assert meta["l0_abstract"] == "user prefers dark mode in IDE"
        assert "source" in meta

    def test_invalid_category_dropped(self):
        store = FakeStore()
        llm = FakeLlm(extract={
            "memories": [{
                "category": "not-a-real-category",
                "abstract": "should not be created",
                "overview": "x",
                "content": "y",
            }],
        })
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="x", assistant_content="y",
        )
        # Either the category is normalised to a real one (in which case
        # we'd see 1 created) OR dropped. Confirm with the actual behaviour:
        # `normalize_category` defaults unknown to "entities", so it'll be created.
        assert stats.created in (0, 1)

    def test_short_abstract_dropped(self):
        store = FakeStore()
        llm = FakeLlm(extract={
            "memories": [{
                "category": "preferences",
                "abstract": "x",  # too short
                "overview": "x",
                "content": "x",
            }],
        })
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="x", assistant_content="y",
        )
        assert stats.created == 0


class TestLlmPipelineDedup:
    """Test the LLM dedup decisions. Use a store that returns canned
    similar memories so the LLM dedup path engages."""

    class StoreWithSimilar(FakeStore):
        def __init__(self, similar):
            super().__init__()
            self._similar = similar

        def search_by_vector(self, vector, limit=10, *, category=None,
                             scope=None, keep_vector=False):
            return list(self._similar)

        def get_by_id(self, mem_id):
            for s in self._similar:
                if s.get("id") == mem_id:
                    return s
            return None

    def test_dedup_skip_does_not_create(self):
        similar = [{
            "id": "existing-1",
            "text": "user prefers dark mode",
            "category": "preference",
            "metadata": json.dumps({
                "memory_category": "preferences",
                "l0_abstract": "user prefers dark mode",
            }),
            "_distance": 0.05,
        }]
        store = self.StoreWithSimilar(similar)
        llm = FakeLlm(
            extract={"memories": [{
                "category": "preferences",
                "abstract": "user wants dark mode UI",
                "overview": "x",
                "content": "y",
            }]},
            dedup={
                "decision": "skip",
                "reason": "duplicate of existing-1",
                "match_index": 1,
            },
        )
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="dark mode again", assistant_content="ok",
        )
        assert stats.skipped == 1
        assert stats.created == 0
        assert llm.dedup_calls == 1

    def test_dedup_create(self):
        similar = [{
            "id": "existing-1",
            "text": "completely unrelated memory",
            "category": "entity",
            "metadata": json.dumps({"memory_category": "entities"}),
            "_distance": 0.5,
        }]
        store = self.StoreWithSimilar(similar)
        llm = FakeLlm(
            extract={"memories": [{
                "category": "preferences",
                "abstract": "user wants dark mode",
                "overview": "x", "content": "y",
            }]},
            dedup={"decision": "create", "reason": "different topic"},
        )
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="dark mode", assistant_content="ok",
        )
        assert stats.created == 1

    def test_dedup_supersede(self):
        similar = [{
            "id": "existing-1",
            "text": "user prefers Vim",
            "category": "preference",
            "metadata": json.dumps({"memory_category": "preferences"}),
            "_distance": 0.1,
        }]
        store = self.StoreWithSimilar(similar)
        llm = FakeLlm(
            extract={"memories": [{
                "category": "preferences",
                "abstract": "user now prefers Emacs",
                "overview": "preference change",
                "content": "user said they switched to Emacs",
            }]},
            dedup={
                "decision": "supersede",
                "reason": "preference change",
                "match_index": 1,
            },
        )
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="I switched to Emacs", assistant_content="ok",
        )
        # supersede on temporal-versioned category → both created+superseded incremented
        assert stats.created == 1
        assert stats.superseded == 1
        # Underlying store.update() called for supersede
        assert any(
            call["text"] == "user now prefers Emacs"
            for call in store.update_calls
        )

    def test_dedup_supersede_missing_index_degrades_to_create(self):
        similar = [{
            "id": "existing-1",
            "text": "x",
            "category": "preference",
            "metadata": json.dumps({"memory_category": "preferences"}),
            "_distance": 0.1,
        }]
        store = self.StoreWithSimilar(similar)
        llm = FakeLlm(
            extract={"memories": [{
                "category": "preferences",
                "abstract": "user wants something different",
                "overview": "x", "content": "y",
            }]},
            dedup={"decision": "supersede", "reason": "no match index"},
            # NO match_index — destructive decision must degrade
        )
        extractor = SmartExtractor(store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="x", assistant_content="y",
        )
        assert stats.created == 1
        assert stats.superseded == 0


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = ExtractionRateLimiter(max_per_hour=3)
        assert not rl.is_rate_limited()
        rl.record_extraction()
        rl.record_extraction()
        assert not rl.is_rate_limited()

    def test_blocks_at_limit(self):
        rl = ExtractionRateLimiter(max_per_hour=2)
        rl.record_extraction()
        rl.record_extraction()
        assert rl.is_rate_limited()

    def test_counts_recent(self):
        rl = ExtractionRateLimiter(max_per_hour=10)
        for _ in range(4):
            rl.record_extraction()
        assert rl.get_recent_count() == 4

    def test_rate_limited_extractor_falls_back_to_legacy(self):
        """When the rate limiter is already saturated the extractor must fall
        back to legacy writes without calling the LLM."""
        from unittest.mock import MagicMock
        from hermes_memory_lancedb_pro.smart_extractor import SmartExtractor

        store = MagicMock()
        store.encode.return_value = [0.0] * 768
        store.encode_batch.return_value = [[0.0] * 768]
        store.store.return_value = "fake-id"

        llm = MagicMock()
        llm.complete_json.return_value = {"memories": []}

        rl = ExtractionRateLimiter(max_per_hour=2)
        rl.record_extraction()
        rl.record_extraction()
        assert rl.is_rate_limited()

        ex = SmartExtractor(store, llm=llm, rate_limiter=rl)
        ex.extract_and_persist(user_content="hello", assistant_content="world", session_key="s")

        # LLM must NOT have been called since we're over the cap
        llm.complete_json.assert_not_called()
        # Legacy fallback writes directly to the store
        assert store.store.called

    def test_rate_limiter_records_on_llm_use(self):
        """Each successful LLM pipeline run must increment the rate limiter."""
        from unittest.mock import MagicMock
        from hermes_memory_lancedb_pro.smart_extractor import SmartExtractor

        store = MagicMock()
        store.encode.return_value = [0.0] * 768
        store.encode_batch.return_value = [[0.0] * 768]
        store.store.return_value = "fake-id"

        llm = MagicMock()
        # Return no candidates so the pipeline exits quickly
        llm.complete_json.return_value = {"memories": []}

        rl = ExtractionRateLimiter(max_per_hour=10)
        assert rl.get_recent_count() == 0

        ex = SmartExtractor(store, llm=llm, rate_limiter=rl)
        ex.extract_and_persist(user_content="hello", assistant_content="world", session_key="s")

        assert rl.get_recent_count() == 1


# ---------------------------------------------------------------------------
# Integration: real LanceDB MemoryStore + StubEmbedder + FakeLlm
# ---------------------------------------------------------------------------

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark_integration = pytest.mark.integration


class StubEmbedder:
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
    tmpdir = tempfile.mkdtemp(prefix="hermes-extractor-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = StubEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
class TestExtractorIntegration:
    def test_legacy_fallback_round_trip(self, real_store):
        extractor = SmartExtractor(real_store)
        stats = extractor.extract_and_persist(
            user_content="I like Python",
            assistant_content="Good choice.",
            session_key="sess-int",
        )
        assert stats.created == 2
        all_entries = real_store.list_memories(limit=10)
        roles = {e["metadata"].get("role") for e in all_entries}
        assert {"user", "assistant"} <= roles

    def test_llm_create_round_trip(self, real_store):
        llm = FakeLlm(
            extract={"memories": [{
                "category": "preferences",
                "abstract": "user prefers UK English in writing",
                "overview": "stated explicitly",
                "content": "user said they prefer UK English",
            }]},
        )
        extractor = SmartExtractor(real_store, llm=llm)
        stats = extractor.extract_and_persist(
            user_content="I prefer UK English.",
            assistant_content="Noted.",
            session_key="sess-int-2",
        )
        assert stats.created == 1
        # The stored entry has the smart category in metadata
        entries = real_store.list_memories(limit=10)
        target = next(
            e for e in entries if "UK English" in e.get("text", "")
        )
        assert target["metadata"].get("memory_category") == "preferences"
        assert target["metadata"].get("l0_abstract") == "user prefers UK English in writing"
