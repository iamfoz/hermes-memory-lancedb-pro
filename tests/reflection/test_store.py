"""Tests for the reflection orchestrator (`reflection/store.py`).

Pure-Python: uses a `FakeAdapter` instead of the LanceDB-backed
`MemoryStoreReflectionAdapter`. The integration test that exercises the
`MemoryStore` adapter lives later, marked `integration`."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

from hermes_memory_lancedb_pro.reflection.store import (
    REFLECTION_CATEGORY,
    MemoryStoreReflectionAdapter,
    build_reflection_store_payloads,
    compute_derived_line_quality,
    is_owned_by_agent,
    load_agent_reflection_slices_from_entries,
    load_reflection_mapped_rows_from_entries,
    store_reflection_to_lancedb,
)

REFLECTION_MD = """## Invariants
- always answer in UK English
- prefer short responses

## Derived
- this run: ship the reflection layer
- next run: write tests for the orchestrator
"""

NOW = 1_700_000_000_000  # fixed for determinism


# ---------------------------------------------------------------------------
# Fake adapter
# ---------------------------------------------------------------------------

@dataclass
class _StoredRow:
    text: str
    vector: list[float]
    category: str
    scope: str
    importance: float
    metadata: str


class FakeAdapter:
    """In-memory ReflectionStoreAdapter for tests. Records writes and
    serves canned vector_search responses."""

    def __init__(self, *, dedupe_hits: list[dict] | None = None):
        self.embed_calls: list[str] = []
        self.search_calls: list[tuple[tuple[float, ...], int, str | None]] = []
        self.stored: list[_StoredRow] = []
        # Optional canned search results — one per call. If shorter than
        # call count, returns [] for subsequent calls.
        self._dedupe_hits = list(dedupe_hits or [])

    def embed_passage(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        # Trivial deterministic vector: hash → 8-d float
        h = hash(text)
        return [(h >> i) & 1 for i in range(8)]

    def vector_search(self, vector, limit=10, *, scope=None):
        self.search_calls.append((tuple(vector), limit, scope))
        if self._dedupe_hits:
            return [self._dedupe_hits.pop(0)]
        return []

    def store_entry(self, *, text, vector, category, scope, importance, metadata):
        self.stored.append(_StoredRow(
            text=text, vector=list(vector), category=category, scope=scope,
            importance=importance, metadata=metadata,
        ))
        return f"id-{len(self.stored)}"


# ---------------------------------------------------------------------------
# build_reflection_store_payloads
# ---------------------------------------------------------------------------

class TestBuildPayloads:
    def test_event_payload_always_first(self):
        eid, slices, payloads = build_reflection_store_payloads(
            reflection_text=REFLECTION_MD,
            session_key="sess-key", session_id="sess-1",
            agent_id="alpha", command="reflect", scope="agent",
            run_at=NOW,
        )
        assert payloads[0].kind == "event"
        assert eid.startswith("refl-")
        assert len(slices.invariants) == 2
        assert len(slices.derived) == 2

    def test_one_payload_per_item_plus_event_plus_legacy(self):
        _eid, _slices, payloads = build_reflection_store_payloads(
            reflection_text=REFLECTION_MD,
            session_key="sess-key", session_id="sess-1",
            agent_id="alpha", command="reflect", scope="agent",
            run_at=NOW,
        )
        kinds = [p.kind for p in payloads]
        # 1 event + 2 invariant items + 2 derived items + 1 combined-legacy
        assert kinds.count("event") == 1
        assert kinds.count("item-invariant") == 2
        assert kinds.count("item-derived") == 2
        assert kinds.count("combined-legacy") == 1

    def test_legacy_combined_disabled(self):
        _eid, _slices, payloads = build_reflection_store_payloads(
            reflection_text=REFLECTION_MD,
            session_key="sess-key", session_id="sess-1",
            agent_id="alpha", command="reflect", scope="agent",
            run_at=NOW,
            write_legacy_combined=False,
        )
        kinds = [p.kind for p in payloads]
        assert "combined-legacy" not in kinds

    def test_empty_reflection_yields_event_only(self):
        # No bullets — only event payload should be emitted (no items, no legacy)
        _eid, slices, payloads = build_reflection_store_payloads(
            reflection_text="## Invariants\n\n## Derived\n",
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent", run_at=NOW,
        )
        assert slices.invariants == []
        assert slices.derived == []
        assert [p.kind for p in payloads] == ["event"]

    def test_event_id_propagation(self):
        eid_in = "refl-202312011530-deadbeef"
        eid_out, _slices, payloads = build_reflection_store_payloads(
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent",
            run_at=NOW, event_id=eid_in,
        )
        assert eid_out == eid_in
        # Event payload metadata also references it
        assert payloads[0].metadata.get("event_id") in (eid_in, None) \
            or eid_in in str(payloads[0].metadata)

    def test_tool_error_signals_normalised(self):
        # Pass both dict shapes and an object with a snake_case attribute
        class _Sig:
            signature_hash = "from-attr"

        _eid, _slices, payloads = build_reflection_store_payloads(
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent",
            run_at=NOW,
            tool_error_signals=[
                {"signature_hash": "snake"},
                {"signatureHash": "camel"},
                _Sig(),
            ],
        )
        legacy = next(p for p in payloads if p.kind == "combined-legacy")
        signals = legacy.metadata.get("error_signals") or []
        assert "snake" in signals
        assert "camel" in signals
        assert "from-attr" in signals


# ---------------------------------------------------------------------------
# store_reflection_to_lancedb (with FakeAdapter)
# ---------------------------------------------------------------------------

class TestStoreReflectionToLanceDB:
    def test_writes_all_payloads(self):
        adapter = FakeAdapter()
        result = store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent", run_at=NOW,
        )
        assert result.stored is True
        # event + 2 inv + 2 der + legacy = 6 writes
        assert len(adapter.stored) == 6
        # All entries get the reflection category
        assert all(r.category == REFLECTION_CATEGORY for r in adapter.stored)
        # All inherit the requested scope
        assert all(r.scope == "agent" for r in adapter.stored)

    def test_combined_legacy_dedupes_on_high_similarity(self):
        # Adapter returns a "near-duplicate" hit (cosine distance 0.01 →
        # similarity 0.99 ≥ default threshold 0.97) for the legacy lookup
        adapter = FakeAdapter(dedupe_hits=[
            {"id": "existing-1", "_distance": 0.01},
        ])
        result = store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent", run_at=NOW,
        )
        kinds = [r.metadata for r in adapter.stored]
        # 5 writes: legacy was deduped
        assert len(adapter.stored) == 5
        assert "combined-legacy" not in [
            json.loads(m).get("type", "") for m in kinds
        ] + [
            "memory-reflection" if "Invariants:" in r.text else None
            for r in adapter.stored
        ]
        assert "combined-legacy" not in result.stored_kinds

    def test_below_threshold_does_not_dedupe(self):
        adapter = FakeAdapter(dedupe_hits=[
            {"id": "existing-1", "_distance": 0.5},  # sim 0.5 < 0.97
        ])
        store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent", run_at=NOW,
        )
        # 6 writes — legacy went through
        assert len(adapter.stored) == 6

    def test_embed_failures_are_logged_not_raised(self):
        class FlakeyAdapter(FakeAdapter):
            def embed_passage(self, text):
                if "Session Reflection" in text:
                    raise RuntimeError("simulated embed failure")
                return super().embed_passage(text)

        adapter = FlakeyAdapter()
        result = store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="a", command="r", scope="agent", run_at=NOW,
        )
        # legacy embed failed → 5 writes
        assert len(adapter.stored) == 5
        assert result.stored is True  # other kinds did land
        assert "combined-legacy" not in result.stored_kinds


# ---------------------------------------------------------------------------
# is_owned_by_agent — ownership rules
# ---------------------------------------------------------------------------

class TestIsOwnedByAgent:
    def test_derived_strict_match(self):
        meta = {"item_kind": "derived", "agent_id": "alpha"}
        assert is_owned_by_agent(meta, "alpha") is True
        assert is_owned_by_agent(meta, "beta") is False

    def test_derived_empty_owner_invisible(self):
        meta = {"item_kind": "derived", "agent_id": ""}
        assert is_owned_by_agent(meta, "alpha") is False

    def test_derived_missing_owner_invisible(self):
        meta = {"item_kind": "derived"}  # no agent_id
        assert is_owned_by_agent(meta, "alpha") is False

    def test_invariant_allows_main_fallback(self):
        meta = {"item_kind": "invariant", "agent_id": "main"}
        assert is_owned_by_agent(meta, "alpha") is True

    def test_invariant_strict_match(self):
        meta = {"item_kind": "invariant", "agent_id": "alpha"}
        assert is_owned_by_agent(meta, "alpha") is True
        # Same agent_id matches, "main" matches, anything else: false
        meta = {"item_kind": "invariant", "agent_id": "beta"}
        assert is_owned_by_agent(meta, "alpha") is False

    def test_legacy_no_item_kind_allows_empty_owner(self):
        meta = {"agent_id": ""}
        assert is_owned_by_agent(meta, "alpha") is True

    def test_legacy_main_owner_visible(self):
        meta = {"agent_id": "main"}
        assert is_owned_by_agent(meta, "alpha") is True

    def test_malformed_item_kind_fails_closed(self):
        # Non-string non-None item_kind → fail closed
        meta = {"item_kind": 42, "agent_id": "alpha"}
        assert is_owned_by_agent(meta, "alpha") is False
        meta = {"item_kind": ["bad"], "agent_id": "alpha"}
        assert is_owned_by_agent(meta, "alpha") is False


# ---------------------------------------------------------------------------
# load_agent_reflection_slices_from_entries
# ---------------------------------------------------------------------------

def _entry(*, eid, text, ts, metadata):
    return {
        "id": eid,
        "text": text,
        "timestamp": ts,
        "metadata": json.dumps(metadata),
    }


class TestLoadAgentReflectionSlices:
    def test_empty_returns_empty(self):
        result = load_agent_reflection_slices_from_entries(
            entries=[], agent_id="alpha", now_ms=NOW,
        )
        assert result.invariants == []
        assert result.derived == []

    def test_picks_owned_invariants(self):
        entries = [
            _entry(
                eid="i1", text="always answer in UK English", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "invariant",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            ),
            _entry(
                eid="d1", text="this run: ship reflection", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "derived",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            ),
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert "always answer in UK English" in result.invariants
        assert "this run: ship reflection" in result.derived

    def test_filters_out_other_agents_derived(self):
        entries = [
            _entry(
                eid="d1", text="this run: secret content", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "derived",
                    "agent_id": "beta",  # not me
                    "stored_at": NOW,
                },
            ),
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert result.derived == []

    def test_p1_full_suppression_when_all_resolved(self):
        # All item rows resolved AND no legacy rows → suppress everything
        entries = [
            _entry(
                eid="i1", text="resolved invariant", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "invariant",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                    "resolved_at": NOW,  # ← resolved
                    "resolved_by": "alpha",
                },
            ),
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert result.invariants == []
        assert result.derived == []

    def test_p2_resolved_lines_filtered_from_legacy_fallback(self):
        # Item row is resolved; legacy row contains only the resolved
        # text → legacy can't revive it
        entries = [
            _entry(
                eid="i1", text="prefer short responses", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "invariant",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                    "resolved_at": NOW,
                },
            ),
            _entry(
                eid="L1",
                text="reflection · agent · 2023-01-01\n...\nInvariants:\n- prefer short responses\n",
                ts=NOW,
                metadata={
                    "type": "memory-reflection",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                    "invariants": ["prefer short responses"],
                    "derived": [],
                },
            ),
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert result.invariants == []  # P1 suppression — legacy has no unique content

    def test_max_age_filtering(self):
        # Item stored 100 days ago, derive_max_age_ms = 30 days → filtered out
        old_ts = NOW - 100 * 86_400_000
        entries = [
            _entry(
                eid="d1", text="ancient derived line", ts=old_ts,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "derived",
                    "agent_id": "alpha",
                    "stored_at": old_ts,
                },
            ),
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
            derive_max_age_ms=30 * 86_400_000,
        )
        assert result.derived == []

    def test_returns_top_8_invariants(self):
        # Make 12 distinct invariants — should cap at 8
        entries = [
            _entry(
                eid=f"i{i}", text=f"invariant rule number {i}", ts=NOW,
                metadata={
                    "type": "memory-reflection-item",
                    "item_kind": "invariant",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            )
            for i in range(12)
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert len(result.invariants) == 8


# ---------------------------------------------------------------------------
# load_reflection_mapped_rows_from_entries
# ---------------------------------------------------------------------------

class TestLoadMappedRows:
    def test_groups_by_kind(self):
        entries = [
            _entry(
                eid="m1", text="user prefers UK English", ts=NOW,
                metadata={
                    "type": "memory-reflection-mapped",
                    "kind": "user-model",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            ),
            _entry(
                eid="m2", text="agent should ask before deleting", ts=NOW,
                metadata={
                    "type": "memory-reflection-mapped",
                    "kind": "agent-model",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            ),
        ]
        result = load_reflection_mapped_rows_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        assert "user prefers UK English" in result.user_model
        assert "agent should ask before deleting" in result.agent_model
        assert result.lesson == []
        assert result.decision == []

    def test_max_per_kind(self):
        entries = [
            _entry(
                eid=f"m{i}", text=f"lesson {i}", ts=NOW,
                metadata={
                    "type": "memory-reflection-mapped",
                    "kind": "lesson",
                    "agent_id": "alpha",
                    "stored_at": NOW,
                },
            )
            for i in range(15)
        ]
        result = load_reflection_mapped_rows_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW, max_per_kind=5,
        )
        assert len(result.lesson) == 5

    def test_filters_unowned(self):
        entries = [
            _entry(
                eid="m1", text="something", ts=NOW,
                metadata={
                    "type": "memory-reflection-mapped",
                    "kind": "lesson",
                    "agent_id": "beta",  # not the calling agent
                    "stored_at": NOW,
                },
            ),
        ]
        result = load_reflection_mapped_rows_from_entries(
            entries=entries, agent_id="alpha", now_ms=NOW,
        )
        # mapped rows allow main fallback when agent_id differs but isn't main
        # Our `is_owned_by_agent` for non-derived requires owner == agent_id OR "main";
        # "beta" matches neither → filtered out
        assert result.lesson == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestComputeDerivedLineQuality:
    def test_zero_lines_floor(self):
        assert compute_derived_line_quality(0) == 0.2

    def test_six_lines_max(self):
        assert compute_derived_line_quality(6) == pytest.approx(1.0)
        assert compute_derived_line_quality(20) == pytest.approx(1.0)

    def test_intermediate(self):
        # 0.55 + 3*0.075 = 0.775
        assert compute_derived_line_quality(3) == pytest.approx(0.775)


# ---------------------------------------------------------------------------
# Integration: MemoryStoreReflectionAdapter against a real LanceDB tmp dir
# ---------------------------------------------------------------------------

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")
import hashlib
import shutil
import tempfile

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
    tmpdir = tempfile.mkdtemp(prefix="hermes-reflection-")
    try:
        s = MemoryStore(db_path=tmpdir)
        s._initialise()
        s._embedder = StubEmbedder()
        yield s
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
class TestMemoryStoreAdapterIntegration:
    def test_round_trip(self, real_store):
        adapter = MemoryStoreReflectionAdapter(real_store)
        result = store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="alpha", command="reflect", scope="agent",
            run_at=int(time.time() * 1000),
        )
        assert result.stored is True
        # Stored 6 entries (event + 2 inv + 2 der + legacy)
        all_entries = real_store.list_memories(limit=20)
        reflection_entries = [
            e for e in all_entries
            if e["category"] == REFLECTION_CATEGORY
        ]
        assert len(reflection_entries) >= 6

    def test_load_back(self, real_store):
        adapter = MemoryStoreReflectionAdapter(real_store)
        store_reflection_to_lancedb(
            adapter,
            reflection_text=REFLECTION_MD,
            session_key="sk", session_id="sid",
            agent_id="alpha", command="reflect", scope="agent",
            run_at=int(time.time() * 1000),
        )
        # Load entries back through MemoryStore.list_memories and pass into
        # the load function — verifies the read-side parses what we wrote
        entries_raw = real_store.list_memories(limit=50)
        # MemoryStore.list_memories returns metadata as a dict; the loader
        # accepts either dict or JSON string, so re-encode for the test
        entries_for_load = [
            {**e, "metadata": json.dumps(e["metadata"])}
            for e in entries_raw
        ]
        result = load_agent_reflection_slices_from_entries(
            entries=entries_for_load, agent_id="alpha",
        )
        # The 4 lines in REFLECTION_MD should round-trip
        assert any("UK English" in inv for inv in result.invariants)
        assert any("ship the reflection" in d for d in result.derived)
