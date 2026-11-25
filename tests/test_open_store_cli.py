"""Regression tests for store maintenance paging, orphaned history, recall
anchors and CLI input handling."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli  # noqa: E402
from hermes_memory_lancedb_pro import store as store_module  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration

DAY_MS = 86_400_000


class StubEmbedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
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
    d = tempfile.mkdtemp(prefix="hermes-open-store-")
    s = MemoryStore(db_path=d)
    s._embedder = StubEmbedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def _ns(**kw):
    kw.setdefault("path", None)
    kw.setdefault("quiet", False)
    return argparse.Namespace(**kw)


def _raw(store, text, *, meta=None, ts=None, scope="global"):
    return store.store_raw(
        text=text,
        vector=store.encode(text),
        category="fact",
        scope=scope,
        importance=0.5,
        metadata=json.dumps(meta or {}),
        timestamp=ts,
    )


def _archived_meta(days_ago=90, **extra):
    return {
        "state": "archived",
        "invalidated_at": int(time.time() * 1000) - days_ago * DAY_MS,
        **extra,
    }


def _corrupt(store):
    """Remove a data file so a full scan fails, like a lost fragment."""
    data = os.path.join(store.db_path, "memories.lance", "data")
    for name in os.listdir(data):
        os.remove(os.path.join(data, name))


# ---------------------------------------------------------------------------
# 1. maintenance scans really cover every row
# ---------------------------------------------------------------------------


def test_purge_and_stats_see_rows_past_the_scan_cap(store, monkeypatch):
    monkeypatch.setattr(store_module, "MAX_SCAN_ROWS", 20)
    monkeypatch.setattr(MemoryStore, "_SCAN_PAGE_ROWS", 7)
    for i in range(30):
        _raw(store, f"live {i}")
    for i in range(10):
        _raw(store, f"old {i}", meta=_archived_meta())
    for i in range(5):
        _raw(store, f"expired {i}", meta={"valid_until": int(time.time() * 1000) - 90 * DAY_MS})

    st = store.stats()
    assert st["active_memories"] + st["archived_memories"] == st["total_memories"] == 45
    assert st["archived_memories"] == 10
    assert "scan_error" not in st

    assert store.purge_archived(dry_run=True) == 10
    assert store.purge_archived() == 10
    assert store.purge_expired() == 5
    assert store.stats()["total_memories"] == 30


# ---------------------------------------------------------------------------
# 2. updates and deletes feed the auto-optimize counter
# ---------------------------------------------------------------------------


def test_updates_and_deletes_trigger_auto_optimize(store, monkeypatch):
    ids = [store.store(text=f"row {i}", category="fact", scope="global") for i in range(3)]
    calls = []
    monkeypatch.setattr(store_module, "AUTO_OPTIMIZE_EVERY", 4)
    store._writes_since_optimize = 0
    monkeypatch.setattr(store, "optimize", lambda: calls.append(1) or True)

    for _ in range(3):
        store.mark_recall_used([ids[0]])
    assert calls == []
    store.forget(ids[1])  # fourth write (a delete)
    assert calls == [1]


# ---------------------------------------------------------------------------
# 3. orphaned history stays addressable but is never mutated
# ---------------------------------------------------------------------------


def test_orphaned_archived_row_is_readable_but_not_revived(store):
    a = store.store(text="my plan is X", category="fact", scope="global")
    b = store.update(a, text="my plan is Y")
    assert store.forget(b)  # live successor gone, `a` keeps superseded_by -> b

    row = store.get_by_id(a)
    assert row is not None and row["id"] == a
    assert row["metadata"]["state"] == "archived"

    assert store.update(a, text="resurrected") is None
    assert store.update(a, importance=0.9) is None
    assert store.increment_access_count(a, force=True) is False
    after = store.get_by_id(a)
    assert after["metadata"]["state"] == "archived"
    assert after["text"] == "my plan is X"
    assert "access_count" not in after["metadata"] or after["metadata"]["access_count"] == 0
    assert store.list_memories(limit=10) == []

    assert store.forget(a) is True
    assert store.get_by_id(a) is None


def test_chain_still_resolves_to_live_successor(store):
    a = store.store(text="v1", category="fact", scope="global")
    b = store.update(a, text="v2")
    assert store.get_by_id(a)["id"] == b


# ---------------------------------------------------------------------------
# 4. session anchors: cheap and correct
# ---------------------------------------------------------------------------


def _session_meta(sess, **extra):
    return {"source_session": sess, "state": "confirmed", **extra}


def test_session_anchors_ignore_scan_cap_and_other_sessions(store, monkeypatch):
    now = int(time.time() * 1000)
    for i in range(30):
        _raw(store, f"other {i}", meta=_session_meta("OTHER"), ts=now - 1000 + i)
    for i in range(4):
        _raw(store, f"mine {i}", meta=_session_meta("S"), ts=now - 500 + i)
    _raw(store, "mine archived", meta=_archived_meta(source_session="S"), ts=now)
    _raw(store, "core note", meta={"tier": "core"}, ts=now - 5000)
    monkeypatch.setattr(store_module, "MAX_SCAN_ROWS", 10)

    first = [r["text"] for r in store.first_for_session("S", 2, strict=True)]
    recent = [r["text"] for r in store.recent_for_session("S", 2, strict=True)]
    assert first == ["mine 0", "mine 1"]
    assert recent == ["mine 3", "mine 2"]
    assert store.first_for_session("S", 1)[0]["text"] == "core note"
    assert store.first_for_session("") == []

    firsts, recents = store.anchors_for_session("S", 2, strict=True)
    assert [r["text"] for r in firsts] == first
    assert [r["text"] for r in recents] == recent


def test_session_anchors_do_not_load_vectors_or_scan_unfiltered(store, monkeypatch):
    _raw(store, "mine", meta=_session_meta("S"), ts=int(time.time() * 1000))
    seen = []
    real_search = store._table.search

    class Spy:
        def __init__(self, q):
            self.q = q

        def __getattr__(self, name):
            attr = getattr(self.q, name)
            if not callable(attr):
                return attr

            def call(*a, **k):
                seen.append((name, a))
                out = attr(*a, **k)
                return Spy(out) if hasattr(out, "to_list") else out

            return call

    monkeypatch.setattr(store._table, "search", lambda *a, **k: Spy(real_search(*a, **k)), raising=False)
    rows = store.recent_for_session("S", 2, strict=True)
    assert [r["text"] for r in rows] == ["mine"]
    names = [n for n, _ in seen]
    assert names.count("to_list") == 1
    selected = next(a[0] for n, a in seen if n == "select")
    assert "vector" not in selected
    where = next(a[0] for n, a in seen if n == "where")
    assert "source_session" in where


# ---------------------------------------------------------------------------
# 5. export --limit
# ---------------------------------------------------------------------------


def _export(store, capsys, **kw):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        rc = _cli._cmd_export(_ns(out="-", **kw), _store=store)
    finally:
        sys.stdout = old
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return rc, lines, capsys.readouterr().err


def test_export_limit_counts_active_rows_and_reports_truncation(store, capsys):
    for i in range(4):
        _raw(store, f"gone {i}", meta=_archived_meta())
    for i in range(5):
        _raw(store, f"live {i}")

    rc, lines, err = _export(store, capsys, limit=3)
    assert rc == 0 and len(lines) == 3
    assert "truncated" in err

    rc, lines, err = _export(store, capsys, limit=5, quiet=True)
    assert len(lines) == 5 and "truncated" not in err

    rc, lines, err = _export(store, capsys, limit=None)
    assert len(lines) == 5 and "truncated" not in err


def test_export_truncation_warning_survives_quiet(store, capsys):
    for i in range(3):
        _raw(store, f"live {i}")
    _, lines, err = _export(store, capsys, limit=2, quiet=True)
    assert len(lines) == 2 and "truncated" in err


def test_export_rejects_non_positive_limit(store, capsys):
    for bad in (0, -1):
        rc, lines, err = _export(store, capsys, limit=bad)
        assert rc == 2 and lines == [] and "--limit" in err


@pytest.mark.parametrize("bad", ["0", "-3", "x"])
def test_limit_flag_validated_in_both_parsers(bad, monkeypatch):
    parser = argparse.ArgumentParser()
    _cli.register_cli(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["export", "--limit", bad])
    monkeypatch.setattr(sys, "argv", ["prog", "export", "--limit", bad])
    with pytest.raises(SystemExit):
        _cli.main()


# ---------------------------------------------------------------------------
# 6. import / prompts do not traceback
# ---------------------------------------------------------------------------


def _good_row(store, text, **over):
    rec = {
        "id": "", "text": text, "category": "fact", "scope": "global",
        "importance": 0.5, "timestamp": 1000, "vector": store.encode(text),
        "metadata": "{}",
    }
    rec.update(over)
    return rec


def test_import_missing_file_creates_nothing(tmp_path, capsys):
    target = tmp_path / "newdb"
    rc = _cli._cmd_import(_ns(input=str(tmp_path / "nope.jsonl"), path=str(target)))
    assert rc == 1
    assert "cannot read" in capsys.readouterr().err
    assert not target.exists()


def test_import_skips_bad_rows_and_imports_the_rest(store, tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    lines = [
        json.dumps(_good_row(store, "first good")),
        "[1, 2]",
        json.dumps(_good_row(store, "null importance", importance=None)),
        json.dumps(_good_row(store, "bad ts", timestamp="abc")),
        json.dumps(_good_row(store, "list meta", metadata="[1]")),
        "not json",
        json.dumps(_good_row(store, "last good")),
    ]
    src.write_text("\n".join(lines) + "\n")
    rc = _cli._cmd_import(_ns(input=str(src)), _store=store)
    err = capsys.readouterr().err
    assert rc == 1  # rejected rows are reported through the exit code
    assert "rejected 3" in err
    texts = sorted(r["text"] for r in store.list_memories(limit=50))
    assert texts == ["first good", "last good", "list meta", "null importance"]


def test_prompts_without_stdin_abort_instead_of_crashing(store, monkeypatch, tmp_path, capsys):
    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert _cli._cmd_purge(_ns(grace_days=30, dry_run=False, yes=False), _store=store) == 1
    assert _cli._cmd_init(_ns(path=str(tmp_path / "i"), yes=False, memory_md=None)) == 1
    assert _cli._cmd_reset(_ns(path=str(tmp_path / "r"), yes=False)) == 1
    assert "Aborted" in capsys.readouterr().out


def test_plugin_cli_without_subcommand_prints_help(capsys):
    parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
    _cli.register_cli(parser)
    args = parser.parse_args([])
    assert args.func(args) == 0
    assert "usage" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# 7. unreadable stores and mistyped paths
# ---------------------------------------------------------------------------


def test_commands_report_an_unreadable_store(store, capsys):
    for i in range(3):
        _raw(store, f"row {i}")
    _corrupt(store)

    assert _cli._cmd_stats(_ns(json=False), _store=store) == 1
    assert "could not be read" in capsys.readouterr().err
    assert _cli._cmd_stats(_ns(json=True), _store=store) == 1
    capsys.readouterr()
    assert _cli._cmd_purge(_ns(grace_days=30, dry_run=False, yes=True), _store=store) == 1
    assert "could not be read" in capsys.readouterr().err
    compact = _ns(dry_run=False, yes=True, scope=None, min_age_days=0, similarity=0.9, max_scan=50)
    assert _cli._cmd_compact(compact, _store=store) == 1
    assert "could not be read" in capsys.readouterr().err
    assert _cli._cmd_doctor(_ns(), _store=store) == 1


@pytest.mark.parametrize("cmd", ["stats", "doctor", "search", "export", "purge", "compact"])
def test_commands_do_not_create_a_store_at_a_mistyped_path(cmd, tmp_path, capsys, monkeypatch):
    target = tmp_path / "typo" / "db"
    monkeypatch.setattr(sys, "argv", ["prog", cmd, "--path", str(target)]
                        + (["q"] if cmd == "search" else [])
                        + (["--yes"] if cmd in ("purge",) else []))
    assert _cli.main() == 1
    assert "no memory store" in capsys.readouterr().err
    assert not target.exists() and not target.parent.exists()


def test_export_to_file_not_written_for_missing_store(tmp_path):
    out = tmp_path / "backup.jsonl"
    rc = _cli._cmd_export(_ns(path=str(tmp_path / "typo"), out=str(out), limit=None))
    assert rc == 1 and not out.exists()


def test_existing_store_still_opens_read_only_commands(store, capsys):
    _raw(store, "hello")
    assert _cli._cmd_stats(_ns(json=False, path=store.db_path)) == 0
    assert "total_memories:    1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 8. doctor -q
# ---------------------------------------------------------------------------


def test_doctor_quiet_prints_only_findings(store, capsys):
    _raw(store, "healthy")
    assert _cli._cmd_doctor(_ns(quiet=True), _store=store) == 0
    assert capsys.readouterr().out == ""

    _raw(store, "orphan", meta={"supersedes": "missing-id"})
    assert _cli._cmd_doctor(_ns(quiet=True), _store=store) == 0
    out = capsys.readouterr().out
    assert "Orphan supersedes" in out
    assert "=== Hermes Memory Doctor ===" not in out
    assert "--- Counts ---" not in out

    assert _cli._cmd_doctor(_ns(quiet=False), _store=store) == 0
    assert "--- Counts ---" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 9. compact asks before merging and deleting
# ---------------------------------------------------------------------------


def _seed_duplicates(store):
    vec = store.encode("shared duplicate topic")
    old_ts = int(time.time() * 1000) - DAY_MS
    for i in range(2):
        store.store_raw(
            text=f"shared duplicate topic variant {i}", vector=vec,
            category="fact", scope="global", importance=0.5,
            metadata=json.dumps({"tier": "working", "state": "confirmed"}),
            timestamp=old_ts,
        )


def _compact_ns(**kw):
    base = dict(dry_run=False, yes=False, scope=None, min_age_days=0,
                similarity=0.88, max_scan=200)
    base.update(kw)
    return _ns(**base)


def test_compact_declined_changes_nothing(store, monkeypatch, capsys):
    _seed_duplicates(store)
    monkeypatch.setattr(builtins, "input", lambda _p="": "no")
    assert _cli._cmd_compact(_compact_ns(), _store=store) == 1
    out = capsys.readouterr().out
    assert "1 cluster(s)" in out and "Aborted" in out
    assert len(store.list_memories(limit=10)) == 2


def test_compact_without_stdin_aborts(store, monkeypatch):
    _seed_duplicates(store)

    def eof(_p=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    assert _cli._cmd_compact(_compact_ns(), _store=store) == 1
    assert len(store.list_memories(limit=10)) == 2


def test_compact_confirmed_or_yes_merges(store, monkeypatch):
    _seed_duplicates(store)
    monkeypatch.setattr(builtins, "input", lambda _p="": "yes")
    assert _cli._cmd_compact(_compact_ns(), _store=store) == 0
    assert len(store.list_memories(limit=10)) == 1

    monkeypatch.setattr(builtins, "input", lambda _p="": pytest.fail("prompted despite --yes"))
    _seed_duplicates(store)
    assert _cli._cmd_compact(_compact_ns(yes=True), _store=store) == 0


def test_compact_nothing_to_merge_does_not_prompt(store, monkeypatch):
    _raw(store, "lonely", ts=int(time.time() * 1000) - DAY_MS)
    monkeypatch.setattr(builtins, "input", lambda _p="": pytest.fail("prompted"))
    assert _cli._cmd_compact(_compact_ns(), _store=store) == 0


def test_compact_yes_flag_in_both_parsers(monkeypatch):
    parser = argparse.ArgumentParser()
    _cli.register_cli(parser)
    assert parser.parse_args(["compact", "-y"]).yes is True
    assert parser.parse_args(["compact", "--yes"]).yes is True
    assert parser.parse_args(["compact"]).yes is False

    seen = {}
    monkeypatch.setattr(_cli, "_cmd_compact", lambda a: seen.setdefault("yes", a.yes) and 0)
    monkeypatch.setattr(sys, "argv", ["prog", "compact", "--yes"])
    assert _cli.main() == 0 and seen["yes"] is True
