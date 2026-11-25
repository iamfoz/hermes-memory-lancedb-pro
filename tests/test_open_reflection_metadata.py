"""Regression tests for the reflection layer and smart-metadata findings
(reflection growth, support-stat evidence, fact_key)."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro.reflection import (  # noqa: E402
    MemoryStoreReflectionAdapter,
    purge_stale_reflection_rows,
    store_reflection_to_lancedb,
)
from hermes_memory_lancedb_pro.reflection.store import (  # noqa: E402
    DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS,
    REFLECTION_CATEGORY,
    build_reflection_store_payloads,
)
from hermes_memory_lancedb_pro.smart_extractor import (  # noqa: E402
    CandidateMemory,
    SmartExtractor,
)
from hermes_memory_lancedb_pro.smart_metadata import (  # noqa: E402
    MAX_SUPPORT_SLICES,
    derive_fact_key,
    update_support_stats,
)
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

DAY_MS = 86_400_000
NOW = int(time.time() * 1000)

REFLECTION_MD = """## Invariants
- always answer in UK English
- prefer short responses

## Derived
- this run: ship the reflection layer
- next run: write tests for the orchestrator
"""


class StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text, normalize_embeddings)
        return [self._one(t, normalize_embeddings) for t in text]

    def _one(self, text: str, normalize: bool) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        if normalize:
            n = sum(f * f for f in floats) ** 0.5
            if n > 0:
                floats = [f / n for f in floats]
        return floats


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-open-refl-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def _reflect(store, run_at, *, session="s1", md=REFLECTION_MD, **kw):
    return store_reflection_to_lancedb(
        MemoryStoreReflectionAdapter(store),
        reflection_text=md, session_key=session, session_id=session,
        agent_id="main", command="session-end", scope="agent",
        run_at=run_at, **kw,
    )


def _rows(store):
    out = []
    for r in store.list_memories(limit=200, category=REFLECTION_CATEGORY):
        out.append((r["metadata"].get("type"), r["metadata"].get("item_kind"), r))
    return out


def _seed(store, text, meta, ts):
    store.store_raw(
        text=text, vector=store.encode(text), category=REFLECTION_CATEGORY,
        scope="agent", importance=0.7, metadata=json.dumps(meta), timestamp=ts,
    )


# ---------------------------------------------------------------------------
# n=40 -- reflection rows must not accumulate forever
# ---------------------------------------------------------------------------


def test_session_write_stores_no_event_row(store):
    _reflect(store, NOW)
    types = [t for t, _, _ in _rows(store)]
    assert "memory-reflection-event" not in types
    assert types.count("memory-reflection-item") == 4
    assert types.count("memory-reflection") == 1


def test_derived_and_legacy_rows_get_valid_until_invariants_do_not(store):
    _reflect(store, NOW)
    expected = NOW + DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS
    for typ, kind, row in _rows(store):
        vu = row["metadata"].get("valid_until")
        if typ == "memory-reflection" or kind == "derived":
            assert vu == expected
        else:
            assert kind == "invariant"
            assert vu is None


def test_purge_expired_reclaims_old_derived_rows(store):
    # Written 40 days ago: derived rows expired 26 days ago.
    _reflect(store, NOW - 40 * DAY_MS)
    assert store.purge_expired(grace_period_days=0, dry_run=True) == 3
    assert store.purge_expired(grace_period_days=0) == 3
    left = [(t, k) for t, k, _ in _rows(store)]
    assert sorted(left) == [("memory-reflection-item", "invariant")] * 2


def test_payload_builder_can_skip_event():
    _, _, with_event = build_reflection_store_payloads(
        reflection_text=REFLECTION_MD, session_key="k", session_id="s",
        agent_id="a", command="c", scope="agent", run_at=NOW,
    )
    _, _, without = build_reflection_store_payloads(
        reflection_text=REFLECTION_MD, session_key="k", session_id="s",
        agent_id="a", command="c", scope="agent", run_at=NOW, write_event=False,
    )
    assert with_event[0].kind == "event"
    assert [p.kind for p in without] == [p.kind for p in with_event[1:]]


def test_purge_stale_removes_rows_written_by_older_versions(store):
    old = NOW - 30 * DAY_MS
    _seed(store, "reflection-event", {"type": "memory-reflection-event",
                                      "stored_at": NOW}, NOW)
    _seed(store, "old derived", {"type": "memory-reflection-item",
                                 "item_kind": "derived", "stored_at": old}, old)
    _seed(store, "fresh derived", {"type": "memory-reflection-item",
                                   "item_kind": "derived", "stored_at": NOW}, NOW)
    _seed(store, "old invariant", {"type": "memory-reflection-item",
                                   "item_kind": "invariant", "stored_at": old}, old)
    _seed(store, "old legacy, derived only",
          {"type": "memory-reflection", "stored_at": old,
           "invariants": [], "derived": ["x"]}, old)
    _seed(store, "old legacy with invariants",
          {"type": "memory-reflection", "stored_at": old,
           "invariants": ["always do y"], "derived": ["x"]}, old)

    assert purge_stale_reflection_rows(store, now_ms=NOW) == 3
    left = sorted(r["text"] for _, _, r in _rows(store))
    assert left == ["fresh derived", "old invariant", "old legacy with invariants"]


def test_session_write_sweeps_stale_rows(store):
    old = NOW - 30 * DAY_MS
    _seed(store, "reflection-event", {"type": "memory-reflection-event",
                                      "stored_at": old}, old)
    _seed(store, "old derived", {"type": "memory-reflection-item",
                                 "item_kind": "derived", "stored_at": old}, old)
    _reflect(store, NOW)
    texts = [r["text"] for _, _, r in _rows(store)]
    assert "reflection-event" not in texts
    assert "old derived" not in texts
    assert "always answer in UK English" in texts


def test_sweep_failure_does_not_fail_the_write():
    class Adapter:
        def __init__(self):
            self.rows = []

        def embed_passage(self, text):
            return [0.1] * 8

        def vector_search(self, vector, limit=10, *, scope=None):
            return []

        def store_entry(self, **kw):
            self.rows.append(kw)
            return "id"

        def purge_stale_reflection_rows(self, *, now_ms=None):
            raise RuntimeError("boom")

    a = Adapter()
    result = store_reflection_to_lancedb(
        a, reflection_text=REFLECTION_MD, session_key="k", session_id="s",
        agent_id="a", command="c", scope="agent", run_at=NOW,
    )
    assert result.stored and len(a.rows) == 5


# ---------------------------------------------------------------------------
# n=31 -- evidence from dropped slices must persist across updates
# ---------------------------------------------------------------------------


def test_dropped_slice_evidence_survives_later_updates():
    info = None
    for i in range(MAX_SUPPORT_SLICES + 1):
        info = update_support_stats(
            info, f"ctx{i}", "contradict" if i == 0 else "support",
        )
    assert len(info.slices) == MAX_SUPPORT_SLICES
    assert info.total_observations == MAX_SUPPORT_SLICES + 1
    assert info.global_strength == pytest.approx(MAX_SUPPORT_SLICES / (MAX_SUPPORT_SLICES + 1))

    after = update_support_stats(info, "ctx3", "support")
    assert after.total_observations == MAX_SUPPORT_SLICES + 2
    # The contradiction in the dropped slice still counts.
    assert after.global_strength == pytest.approx(
        (MAX_SUPPORT_SLICES + 1) / (MAX_SUPPORT_SLICES + 2)
    )
    again = update_support_stats(after, "ctx4", "support")
    assert again.total_observations == MAX_SUPPORT_SLICES + 3
    assert again.global_strength < 1.0


def test_support_stats_without_dropped_slices_unchanged():
    info = update_support_stats(None, "a", "support")
    info = update_support_stats(info, "b", "contradict")
    info = update_support_stats(info, "a", "support")
    assert info.total_observations == 3
    assert info.global_strength == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# n=32 -- fact_key derivation and wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "https://a.example.com/docs",
    "https://b.example.com",
    "Meeting at 10:30 daily",
])
def test_fact_key_ignores_colons_in_urls_and_times(text):
    key = derive_fact_key("entities", text)
    assert key == f"entities:{text.lower()}"


def test_fact_key_distinguishes_urls_that_share_a_scheme():
    assert derive_fact_key("entities", "https://a.example.com/docs") != derive_fact_key(
        "entities", "https://b.example.com"
    )


def test_fact_key_earlier_separator_wins():
    assert derive_fact_key("preferences", "A -> B: C") == "preferences:a"
    assert derive_fact_key("preferences", "Editor: vim -> emacs") == "preferences:editor"


def test_fact_key_colon_forms_still_work():
    assert derive_fact_key("preferences", "Dark mode: on") == "preferences:dark mode"
    assert derive_fact_key("entities", "Location：Beijing") == "entities:location"
    assert derive_fact_key("entities", "Note:") == "entities:note"


class _RecordingStore:
    def __init__(self):
        self.stored = []
        self.updated = []

    def store(self, **kw):
        self.stored.append(kw)
        return "id-1"

    def update(self, mem_id, **kw):
        self.updated.append(kw)
        return "id-2"


def test_extractor_writes_fact_key_for_temporal_categories():
    store = _RecordingStore()
    ex = SmartExtractor(store)
    cand = CandidateMemory(
        category="preferences", abstract="Editor: user prefers vim",
        overview="o", content="c",
    )
    ex._store_candidate(
        cand, vector=[], session_key="s", target_scope="global",
    )
    assert store.stored[0]["metadata_extra"]["fact_key"] == "preferences:editor"

    other = CandidateMemory(
        category="cases", abstract="Bug: crash on start", overview="o", content="c",
    )
    ex._store_candidate(
        other, vector=[], session_key="s", target_scope="global",
    )
    assert "fact_key" not in store.stored[1]["metadata_extra"]


def test_supersede_resets_fact_key_to_the_new_wording():
    store = _RecordingStore()
    ex = SmartExtractor(store)
    cand = CandidateMemory(
        category="preferences", abstract="Theme: user now prefers light",
        overview="o", content="c",
    )
    ex._handle_supersede(
        candidate=cand, vector=[], match_id="old", session_key="s",
        target_scope="global", scope_filter=["global"], admission_audit=None,
    )
    assert store.updated[0]["metadata_extra"]["fact_key"] == "preferences:theme"
