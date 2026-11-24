"""Regression tests for defects found in the 0.16.0 audit."""

from __future__ import annotations

import hashlib
import shutil
import tempfile

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli  # noqa: E402
from hermes_memory_lancedb_pro.store import (  # noqa: E402
    ARCHIVED_STATE,
    VECTOR_DIM,
    MemoryStore,
)

pytestmark = pytest.mark.integration


class StubEmbedder:
    """Cheap deterministic embedder — no model download required."""

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
    d = tempfile.mkdtemp(prefix="hermes-audit-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def _raw(store: MemoryStore, mem_id: str) -> dict:
    rows = store._table.search().where(f"id = '{mem_id}'").limit(1).to_list()
    return store._row_to_dict(rows[0])


# ---------------------------------------------------------------------------
# Smoke test must never delete memories it did not write
# ---------------------------------------------------------------------------


def test_smoke_run_leaves_user_memories_alone(store, monkeypatch, capsys):
    texts = [f"user memory number {i} about topic {i * 7}" for i in range(12)]
    ids = [store.store(text=t, category="fact", scope="global") for t in texts]
    monkeypatch.setattr(_cli, "MemoryStore", lambda db_path=None: store)
    assert _cli._run(store.db_path) == 0
    assert "0 failed" in capsys.readouterr().out
    assert all(store.has_id(i) for i in ids)
    remaining = store.list_memories(limit=100, include_archived=True)
    assert sorted(r["text"] for r in remaining) == sorted(texts)  # no smoke rows left


# ---------------------------------------------------------------------------
# increment_access_count via a superseded id
# ---------------------------------------------------------------------------


def test_access_count_on_superseded_id_does_not_resurrect_it(store):
    old = store.store(text="my favourite colour is blue", category="preference")
    new = store.update(old, text="my favourite colour is green")
    assert new and new != old
    assert store.increment_access_count(old, force=True) is True
    assert _raw(store, old)["metadata"]["state"] == ARCHIVED_STATE
    live = _raw(store, new)
    assert live["metadata"].get("state") != ARCHIVED_STATE
    assert live["metadata"]["access_count"] == 1
    assert [r["text"] for r in store.list_memories(limit=10)] == ["my favourite colour is green"]


# ---------------------------------------------------------------------------
# list_memories must not come back short when filtered rows precede live ones
# ---------------------------------------------------------------------------


def test_list_memories_finds_rows_behind_many_archived(store):
    mem = store.store(text="version 0", category="fact")
    for i in range(1, 70):
        mem = store.update(mem, text=f"version {i}")
    other = store.store(text="another live row", category="fact")
    listed = store.list_memories(limit=5)
    assert {r["id"] for r in listed} == {mem, other}


def test_list_memories_finds_later_session_rows(store):
    for i in range(130):
        store.store(text=f"session A row {i}", category="fact",
                    metadata_extra={"source_session": "A"})
    for i in range(5):
        store.store(text=f"session B row {i}", category="fact",
                    metadata_extra={"source_session": "B"})
    listed = store.list_memories(limit=5, session_id="B")
    assert len(listed) == 5 and all("session B" in r["text"] for r in listed)


def test_list_memories_offset_pages_do_not_overlap(store):
    ids = [store.store(text=f"row {i}", category="fact") for i in range(30)]
    for i in ids[:20:2]:
        store.update(i, text=f"edited {i}")
    page1 = store.list_memories(limit=10, offset=0)
    page2 = store.list_memories(limit=10, offset=10)
    page3 = store.list_memories(limit=10, offset=20)
    seen = [r["id"] for r in page1 + page2 + page3]
    assert len(seen) == 30 and len(set(seen)) == 30


# ---------------------------------------------------------------------------
# active_task lookups must see every live row, however many anchors exist
# ---------------------------------------------------------------------------


def _anchors(store, conversation_id=None):
    from hermes_memory_lancedb_pro import provider

    rows = provider._list_active_task_rows(store)
    return [
        r for r in rows
        if r["metadata"].get("auto_anchor")
        and (conversation_id is None or r["metadata"].get("conversation_id") == conversation_id)
    ]


def test_auto_anchor_stays_idempotent_past_twenty_conversations(store):
    from hermes_memory_lancedb_pro import provider

    for i in range(30):
        provider._auto_anchor_session_if_needed(f"hello {i}", f"s{i}", f"conv-{i}", store)
    assert len(_anchors(store)) == 30
    for _ in range(5):  # later turns of the newest conversation
        provider._auto_anchor_session_if_needed("hello again", "s-new", "conv-new", store)
    assert len(_anchors(store, "conv-new")) == 1
    assert len(_anchors(store)) == 31


def test_reset_archives_anchor_past_twenty_conversations(store):
    from hermes_memory_lancedb_pro import provider

    for i in range(30):
        provider._auto_anchor_session_if_needed(f"hello {i}", f"s{i}", f"conv-{i}", store)
    assert provider._archive_auto_anchors(store, "conv-29") == 1
    assert _anchors(store, "conv-29") == []
    assert len(_anchors(store)) == 29


def test_formal_pin_visible_behind_many_anchors(store, tmp_path):
    from hermes_memory_lancedb_pro import provider
    from hermes_memory_lancedb_pro import task_ledger as tl

    for i in range(30):
        provider._auto_anchor_session_if_needed(f"hello {i}", f"s{i}", f"conv-{i}", store)
    tl.create_task("bench", "Benchmark the parser", root=tmp_path)
    store.store(
        text="pin", category="active_task", scope="global", importance=1.0,
        metadata_extra={"task_id": "bench", "state_path": str(tl._state_path("bench", tmp_path))},
    )
    rows = provider._list_active_task_rows(store)
    assert any(r["metadata"].get("task_id") == "bench" for r in rows)
    # With a formal pin present no further anchor is written.
    provider._auto_anchor_session_if_needed("hi", "s-x", "conv-x", store)
    assert _anchors(store, "conv-x") == []


def test_stale_auto_anchors_are_archived(store, monkeypatch):
    import time

    from hermes_memory_lancedb_pro import provider

    provider._auto_anchor_session_if_needed("old chat", "s-old", "conv-old", store)
    provider._auto_anchor_session_if_needed("recent chat", "s-recent", "conv-recent", store)
    old = _anchors(store, "conv-old")[0]
    aged = int(time.time() * 1000) - 30 * 86_400_000
    store._table.update(where=f"id = '{old['id']}'", values={"timestamp": aged})

    provider._auto_anchor_session_if_needed("new chat", "s-new", "conv-new", store)
    assert _anchors(store, "conv-old") == []
    assert len(_anchors(store, "conv-recent")) == 1
    assert len(_anchors(store, "conv-new")) == 1

    monkeypatch.setattr(provider, "_AUTO_ANCHOR_TTL_DAYS", 0)
    recent = _anchors(store, "conv-recent")[0]
    store._table.update(where=f"id = '{recent['id']}'", values={"timestamp": aged})
    provider._auto_anchor_session_if_needed("another", "s-n2", "conv-n2", store)
    assert len(_anchors(store, "conv-recent")) == 1  # TTL disabled


# ---------------------------------------------------------------------------
# A malformed numeric env var must not break import
# ---------------------------------------------------------------------------


def test_env_helpers_fall_back_on_bad_values(monkeypatch, caplog):
    from hermes_memory_lancedb_pro._env import env_float, env_int

    monkeypatch.setenv("X_INT", "five")
    monkeypatch.setenv("X_FLOAT", "nan")
    monkeypatch.setenv("X_OK", " 7 ")
    monkeypatch.setenv("X_BLANK", "  ")
    with caplog.at_level("WARNING"):
        assert env_int("X_INT", 5) == 5
        assert env_float("X_FLOAT", 0.25) == 0.25
    assert "X_INT" in caplog.text and "X_FLOAT" in caplog.text
    assert env_int("X_OK", 1) == 7 and env_float("X_OK", 1.0) == 7.0
    assert env_int("X_BLANK", 3) == 3 and env_int("X_UNSET", 9) == 9


def test_provider_imports_with_malformed_env(monkeypatch):
    import importlib

    from hermes_memory_lancedb_pro import provider

    monkeypatch.setenv("MEMORY_PREFETCH_LIMIT", "lots")
    monkeypatch.setenv("MEMORY_RECALL_CHAR_BUDGET", "")
    try:
        reloaded = importlib.reload(provider)
        assert reloaded.DEFAULT_PREFETCH_LIMIT == 5
        assert reloaded._RECALL_CHAR_BUDGET == 4800
    finally:
        monkeypatch.delenv("MEMORY_PREFETCH_LIMIT")
        monkeypatch.delenv("MEMORY_RECALL_CHAR_BUDGET")
        importlib.reload(provider)


# ---------------------------------------------------------------------------
# One database path for the provider and the CLI
# ---------------------------------------------------------------------------


def test_resolve_db_path_precedence(monkeypatch, tmp_path):
    from hermes_memory_lancedb_pro.store import resolve_db_path

    monkeypatch.delenv("MEMORY_DB_DIR", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_db_path() == str(tmp_path / ".hermes" / "memory-lancedb")

    profile = tmp_path / ".hermes" / "profiles" / "work"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert resolve_db_path() == str(profile / "memory-lancedb")
    # The host's hermes_home kwarg wins over the env var.
    assert resolve_db_path(hermes_home=str(tmp_path / "other")) == str(
        tmp_path / "other" / "memory-lancedb")

    monkeypatch.setenv("MEMORY_DB_DIR", str(tmp_path / "custom"))
    assert resolve_db_path() == str(tmp_path / "custom")
    assert resolve_db_path(hermes_home=str(profile)) == str(tmp_path / "custom")
    assert resolve_db_path(str(tmp_path / "explicit")) == str(tmp_path / "explicit")


def test_cli_opens_the_profile_database(monkeypatch, tmp_path):
    import argparse

    profile = tmp_path / "profiles" / "work"
    monkeypatch.delenv("MEMORY_DB_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    opened: list[str] = []

    class FakeStore:
        def __init__(self, db_path):
            opened.append(db_path)

        def _initialise(self):
            pass

    monkeypatch.setattr(_cli, "MemoryStore", FakeStore)
    _cli._open_store(argparse.Namespace(path=None))
    _cli._open_store(argparse.Namespace(path=str(tmp_path / "x")))
    assert opened == [str(profile / "memory-lancedb"), str(tmp_path / "x")]


# ---------------------------------------------------------------------------
# task pin / unpin
# ---------------------------------------------------------------------------


def _task_pins(store, task_id):
    rows = store.list_memories(limit=100, category="active_task")
    return [r for r in rows if r["metadata"].get("task_id") == task_id]


def test_repinning_replaces_the_pin_and_unpin_removes_it(store, monkeypatch, tmp_path, capsys):
    import argparse

    from hermes_memory_lancedb_pro import task_ledger as tl

    monkeypatch.setattr(_cli, "_open_store", lambda args: store)
    tl.create_task("t", "objective", root=tmp_path)
    ns = argparse.Namespace(task_id="t", task_root=str(tmp_path), path=None, quiet=True)
    assert _cli._cmd_task_pin(ns) == 0
    assert _cli._cmd_task_pin(ns) == 0
    assert len(_task_pins(store, "t")) == 1

    ns.task_command = "unpin"
    assert _cli._cmd_task_dispatch(ns) == 0
    assert _task_pins(store, "t") == []
    assert _cli._cmd_task_dispatch(ns) == 4  # nothing left to unpin
    assert tl.load_state("t", root=tmp_path)["status"] == "running"  # ledger untouched


# ---------------------------------------------------------------------------
# The cached system-prompt block must not call a parked task "running"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["waiting", "blocked"])
def test_stable_block_for_parked_task(tmp_path, verb):
    from hermes_memory_lancedb_pro import provider
    from hermes_memory_lancedb_pro import task_ledger as tl

    tl.create_task("rel", "Ship release 1.2", root=tmp_path)
    if verb == "waiting":
        state = tl.wait_task("rel", "CI run", ref="PR#7", clears_when="CI is green", root=tmp_path)
    else:
        state = tl.block_task("rel", "need prod credentials", root=tmp_path)
    block = provider._stable_task_block(state)
    assert f"Status:    {verb}" in block and "running" not in block
    assert "do NOT continue the work" in block
    assert "task unwait rel" in block and "Do NOT greet" in block
    if verb == "waiting":
        assert "CI run" in block and "PR#7" in block and "CI is green" in block
    else:
        assert "need prod credentials" in block and "needs a person" in block


def test_stable_block_for_running_task_unchanged_in_shape(tmp_path):
    from hermes_memory_lancedb_pro import provider
    from hermes_memory_lancedb_pro import task_ledger as tl

    state = tl.create_task("t", "The objective", root=tmp_path)
    block = provider._stable_task_block(state)
    assert "Status:    running" in block and "task resume t" in block
    # Stable across advances: nothing per-iteration is rendered.
    assert provider._stable_task_block(tl.advance_iteration("t", root=tmp_path)) == block


def test_protocol_text_covers_new_task_verbs():
    from hermes_memory_lancedb_pro import provider

    text = provider._TASK_PROTOCOL_TEXT
    for needle in ("task reported", "task wait", "task block", "task unwait",
                   "Status = waiting", "Status = blocked"):
        assert needle in text
