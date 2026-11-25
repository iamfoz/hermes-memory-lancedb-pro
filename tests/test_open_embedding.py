"""Embedding model loading and per-database task-prefix schemes.

nomic-embed-text expects "search_document: " on stored text and
"search_query: " on queries. Databases written before the prefixes were
used hold unprefixed vectors, so the scheme is recorded per database and
`reembed` converts a legacy one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli  # noqa: E402
from hermes_memory_lancedb_pro import store as store_module  # noqa: E402
from hermes_memory_lancedb_pro.store import (  # noqa: E402
    EMBEDDING_SCHEME_FILE,
    SCHEME_PLAIN,
    SCHEME_TASK_PREFIX,
    VECTOR_DIM,
    MemoryStore,
    model_uses_task_prefixes,
)

pytestmark = pytest.mark.integration

DOC = "search_document: "
QUERY = "search_query: "


def _vec(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
    n = sum(f * f for f in floats) ** 0.5
    return [f / n for f in floats]


class RecordingEmbedder:
    """Deterministic hash embedder that records the exact model input."""

    def __init__(self, fail_on_call: int | None = None):
        self.inputs: list[str] = []
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.on_encode = None

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        self.calls += 1
        if self.on_encode is not None:
            self.on_encode()
        if self.fail_on_call is not None and self.calls >= self.fail_on_call:
            raise RuntimeError("embedder exploded")
        if isinstance(text, str):
            self.inputs.append(text)
            return _vec(text)
        self.inputs.extend(text)
        return [_vec(t) for t in text]


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="hermes-embedding-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _open(path: str, embedder=None, **kwargs) -> MemoryStore:
    s = MemoryStore(db_path=path, **kwargs)
    s._embedder = embedder or RecordingEmbedder()
    s._initialise()
    return s


def _marker(path: str) -> dict | None:
    try:
        with open(os.path.join(path, EMBEDDING_SCHEME_FILE), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _make_legacy(path: str, texts: list[str]) -> list[str]:
    """A database as every earlier version left it: unprefixed vectors and
    no scheme marker."""
    s = _open(path)
    os.remove(os.path.join(path, EMBEDDING_SCHEME_FILE))
    ids = [
        s.store_raw(text=t, vector=_vec(t), category="fact", scope="global",
                    importance=0.5, metadata={"tier": "working", "n": i})
        for i, t in enumerate(texts)
    ]
    assert _marker(path) is None
    return ids


def _vectors(s: MemoryStore) -> dict[str, list[float]]:
    rows = s._table.search().limit(10_000).to_list()
    return {r["id"]: [float(x) for x in r["vector"]] for r in rows}


def _rows(s: MemoryStore) -> dict[str, dict]:
    rows = s._table.search().limit(10_000).to_list()
    return {r["id"]: {k: r[k] for k in ("text", "metadata", "category", "scope")}
            for r in rows}


def _close(a: list[float], b: list[float]) -> bool:
    return all(abs(x - y) < 1e-5 for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Model loading: trust_remote_code only when the installed stack needs it
# ---------------------------------------------------------------------------


class FakeST:
    """Stands in for SentenceTransformer; records every construction."""

    calls: list[tuple[str, dict]] = []
    needs_trust = False
    bad_device = None

    def __init__(self, name, **kwargs):
        FakeST.calls.append((name, dict(kwargs)))
        if FakeST.bad_device and kwargs.get("device") == FakeST.bad_device:
            raise RuntimeError("device not available")
        if FakeST.needs_trust and not kwargs.get("trust_remote_code"):
            raise ValueError(
                f"The repository for {name} contains custom code which must be "
                "executed to correctly load the model. Please pass the argument "
                "`trust_remote_code=True` to allow custom code to be run."
            )


@pytest.fixture
def fake_st(monkeypatch):
    FakeST.calls = []
    FakeST.needs_trust = False
    FakeST.bad_device = None
    monkeypatch.setattr(store_module, "EMBEDDING_DEVICE", None)
    return FakeST


def test_load_that_works_without_trust_remote_code_is_unchanged(fake_st):
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-embedding-"))
    s._construct_embedder(fake_st)
    assert fake_st.calls == [(s.embedding_model_name, {})]


def test_nomic_load_retries_with_trust_remote_code_when_required(fake_st):
    fake_st.needs_trust = True
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-embedding-"))
    assert isinstance(s._construct_embedder(fake_st), fake_st)
    assert fake_st.calls == [
        (s.embedding_model_name, {}),
        (s.embedding_model_name, {"trust_remote_code": True}),
    ]


def test_trust_remote_code_retry_keeps_the_device_override(fake_st, monkeypatch):
    fake_st.needs_trust = True
    monkeypatch.setattr(store_module, "EMBEDDING_DEVICE", "cpu")
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-embedding-"))
    s._construct_embedder(fake_st)
    assert fake_st.calls[-1] == (
        s.embedding_model_name, {"trust_remote_code": True, "device": "cpu"},
    )


def test_other_models_never_get_trust_remote_code_implicitly(fake_st):
    fake_st.needs_trust = True
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-embedding-"),
                    embedding_model="someone/custom-code-model")
    with pytest.raises(ValueError, match="trust_remote_code"):
        s._construct_embedder(fake_st)
    assert len(fake_st.calls) == 1


def test_unrelated_load_errors_are_not_retried(fake_st):
    fake_st.bad_device = "cuda:7"
    s = MemoryStore(db_path=tempfile.mkdtemp(prefix="hermes-embedding-"))
    with pytest.raises(RuntimeError):
        s._new_model(fake_st, device="cuda:7")
    assert len(fake_st.calls) == 1


def test_model_prefix_detection():
    assert model_uses_task_prefixes("nomic-ai/nomic-embed-text-v1.5")
    assert model_uses_task_prefixes("/models/Nomic-Embed-Text-v1")
    assert not model_uses_task_prefixes("nomic-ai/nomic-embed-code")
    assert not model_uses_task_prefixes("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Scheme selection per database
# ---------------------------------------------------------------------------


def test_new_database_records_and_uses_task_prefixes(db_path):
    emb = RecordingEmbedder()
    s = _open(db_path, emb)
    assert s.embedding_scheme == SCHEME_TASK_PREFIX
    assert _marker(db_path)["scheme"] == SCHEME_TASK_PREFIX
    s.encode("hello")
    s.encode("hello", kind="query")
    s.encode_batch(["a", "b"])
    s.encode_batch(["c"], kind="query")
    assert emb.inputs == [DOC + "hello", QUERY + "hello", DOC + "a", DOC + "b", QUERY + "c"]
    assert not [f for f in os.listdir(db_path) if ".tmp-" in f]


def test_legacy_database_keeps_working_without_prefixes(db_path):
    _make_legacy(db_path, ["the sky is blue", "grass is green"])
    emb = RecordingEmbedder()
    s = _open(db_path, emb)
    assert s.embedding_scheme == SCHEME_PLAIN
    assert _marker(db_path) is None  # opening never relabels a legacy store
    hits = s.search("the sky is blue", mode="vector", limit=1)
    assert hits[0]["text"] == "the sky is blue"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    s.store(text="water is wet", category="fact")
    assert emb.inputs == ["the sky is blue", "water is wet"]


def test_prefixed_database_query_matches_its_documents(db_path):
    """Vector search on a new store sends the query prefix and finds rows
    stored with the document prefix (via a stub that ignores the prefix
    when hashing, as the real model maps both into one space)."""
    class PrefixBlind(RecordingEmbedder):
        def encode(self, text, **kw):
            super().encode(text, **kw)  # records the input

            def strip(t):
                return t.removeprefix(DOC).removeprefix(QUERY)
            if isinstance(text, str):
                return _vec(strip(text))
            return [_vec(strip(t)) for t in text]

    emb = PrefixBlind()
    s = _open(db_path, emb)
    mid = s.store(text="my cat is called Tom", category="fact")
    hits = s.search("my cat is called Tom", mode="vector", limit=1)
    assert hits[0]["id"] == mid
    assert emb.inputs == [DOC + "my cat is called Tom", QUERY + "my cat is called Tom"]


def test_every_store_write_path_encodes_documents(db_path):
    emb = RecordingEmbedder()
    s = _open(db_path, emb)
    mid = s.store(text="one", category="fact")
    s.store_many([{"text": "two"}, {"text": "three"}])
    s.update(mid, text="one, revised")
    assert emb.inputs == [DOC + "one", DOC + "two", DOC + "three", DOC + "one, revised"]
    emb.inputs.clear()
    s.search("q1", mode="vector")
    s.search("q2", mode="hybrid")
    s.search("q3", mode="bm25")
    assert emb.inputs == [QUERY + "q1", QUERY + "q2"]


def test_non_nomic_model_gets_no_prefixes(db_path):
    emb = RecordingEmbedder()
    s = _open(db_path, emb, embedding_model="sentence-transformers/all-mpnet-base-v2")
    assert s.embedding_scheme == SCHEME_PLAIN
    assert s.preferred_embedding_scheme == SCHEME_PLAIN
    assert _marker(db_path)["scheme"] == SCHEME_PLAIN
    s.encode("x")
    s.encode("y", kind="query")
    assert emb.inputs == ["x", "y"]
    assert s.reembed()["status"] == "up_to_date"


def test_unknown_kind_is_rejected(db_path):
    s = _open(db_path)
    with pytest.raises(ValueError, match="kind"):
        s.encode("x", kind="passage")


def test_unwritable_marker_on_new_database_falls_back_to_plain(db_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only file system")
    monkeypatch.setattr(store_module.os, "replace", boom)
    s = _open(db_path)
    assert s.embedding_scheme == SCHEME_PLAIN
    assert _marker(db_path) is None
    assert not [f for f in os.listdir(db_path) if ".tmp-" in f]


def test_failed_marker_write_leaves_the_old_marker(db_path, monkeypatch):
    s = _open(db_path)
    before = _marker(db_path)
    monkeypatch.setattr(store_module.os, "replace",
                        lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        s._write_scheme_marker(SCHEME_PLAIN)
    assert _marker(db_path) == before
    assert not [f for f in os.listdir(db_path) if ".tmp-" in f]


# ---------------------------------------------------------------------------
# reembed
# ---------------------------------------------------------------------------


TEXTS = [f"legacy memory number {i}" for i in range(7)]


def test_reembed_converts_a_legacy_store(db_path):
    ids = _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    archived = s.update(ids[0], text="legacy memory zero, corrected")  # plain too
    before = _rows(s)

    dry = s.reembed(dry_run=True)
    assert dry["status"] == "dry_run" and dry["rows"] == len(TEXTS) + 1
    assert _marker(db_path) is None and s.embedding_scheme == SCHEME_PLAIN

    result = s.reembed(batch_size=3)
    assert result["status"] == "converted"
    assert result["reembedded"] == len(TEXTS) + 1
    assert _marker(db_path)["scheme"] == SCHEME_TASK_PREFIX
    assert "reembed_target" not in _marker(db_path)
    assert s.embedding_scheme == SCHEME_TASK_PREFIX
    # Every row, archived ones included, now carries the document prefix;
    # text and metadata are untouched.
    after = _rows(s)
    assert after == before
    for rid, vec in _vectors(s).items():
        assert _close(vec, _vec(DOC + after[rid]["text"])), rid
    assert archived in after
    assert s.reembed()["status"] == "up_to_date"


def test_reembed_failure_while_encoding_changes_nothing(db_path):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path, RecordingEmbedder(fail_on_call=2))
    before = _vectors(s)
    with pytest.raises(RuntimeError):
        s.reembed(batch_size=3)
    assert _vectors(s) == before
    assert _marker(db_path) is None
    assert s.embedding_scheme == SCHEME_PLAIN
    assert not s.reembed_interrupted()


def test_reembed_failed_commit_changes_nothing(db_path, monkeypatch):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    before = _vectors(s)

    def broken(ids, vectors):
        raise RuntimeError("commit conflict")
    monkeypatch.setattr(s, "_replace_vectors", broken)
    with pytest.raises(RuntimeError):
        s.reembed()
    assert _vectors(s) == before
    assert s.embedding_scheme == SCHEME_PLAIN
    assert not s.reembed_interrupted()


def test_interrupted_reembed_is_reported_and_repaired(db_path, capsys):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    # Killed between the vector commit and the final marker.
    s._write_scheme_marker(SCHEME_PLAIN, reembed_target=SCHEME_TASK_PREFIX)
    assert s.stats()["reembed_interrupted"] is True
    assert _cli._cmd_doctor(argparse.Namespace(), _store=s) == 0
    assert "did not finish" in capsys.readouterr().out
    assert s.reembed()["status"] == "converted"
    assert not s.reembed_interrupted()
    for rid, vec in _vectors(s).items():
        assert _close(vec, _vec(DOC + _rows(s)[rid]["text"]))


def test_reembed_catches_rows_written_while_it_runs(db_path):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    other = _open(db_path)  # e.g. the running agent, still on "plain"
    late: list[str] = []

    def write_once(done, total):
        if not late:
            late.append(other.store(text="written during reembed", category="fact"))
    s.reembed(batch_size=2, progress=write_once)
    vec = _vectors(s)[late[0]]
    assert _close(vec, _vec(DOC + "written during reembed"))


def test_running_store_picks_up_a_reembed_by_another_process(db_path):
    _make_legacy(db_path, TEXTS)
    emb = RecordingEmbedder()
    agent = _open(db_path, emb)
    assert agent.embedding_scheme == SCHEME_PLAIN
    _open(db_path).reembed()
    agent.encode("after", kind="query")
    assert emb.inputs[-1] == QUERY + "after"


def test_reembed_holds_locks_in_the_documented_order(db_path):
    _make_legacy(db_path, TEXTS)
    emb = RecordingEmbedder()
    s = _open(db_path, emb)
    seen = []

    def check():
        seen.append((s._write_lock._is_owned(), store_module._ENCODE_LOCK.locked()))
    emb.on_encode = check
    s.reembed(batch_size=4)
    assert seen and all(seen_item == (True, True) for seen_item in seen)

    # Concurrent encodes from another thread neither deadlock nor fail.
    errors = []

    def encode_loop():
        try:
            for _ in range(20):
                s.encode("concurrent", kind="query")
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)
    s._write_scheme_marker(SCHEME_PLAIN)
    emb.on_encode = None
    t = threading.Thread(target=encode_loop)
    t.start()
    s.reembed(batch_size=2)
    t.join(timeout=30)
    assert not t.is_alive() and not errors


def test_reembed_rebuilds_an_existing_vector_index(db_path, monkeypatch):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    built = []
    monkeypatch.setattr(s, "_vector_index_metric", lambda: "cosine")
    monkeypatch.setattr(s, "_build_vector_index", lambda: built.append(1))
    s.reembed()
    assert built == [1]


def test_reembed_of_empty_store_only_relabels(db_path):
    s = _open(db_path)
    s._write_scheme_marker(SCHEME_PLAIN)
    result = s.reembed()
    assert result["status"] == "converted" and result["reembedded"] == 0
    assert _marker(db_path)["scheme"] == SCHEME_TASK_PREFIX


# ---------------------------------------------------------------------------
# CLI: reembed, doctor, stats, export / import
# ---------------------------------------------------------------------------


def _reembed_args(**kw) -> argparse.Namespace:
    base = {"dry_run": False, "yes": False, "batch_size": 64, "quiet": False, "path": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_reembed_dry_run_and_confirmation(db_path, monkeypatch, capsys):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    before = _vectors(s)

    assert _cli._cmd_reembed(_reembed_args(dry_run=True), _store=s) == 0
    assert "[dry-run] would re-encode 7 memories" in capsys.readouterr().out
    assert _vectors(s) == before

    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert _cli._cmd_reembed(_reembed_args(), _store=s) == 1
    assert "Aborted." in capsys.readouterr().out
    assert _vectors(s) == before and s.embedding_scheme == SCHEME_PLAIN

    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert _cli._cmd_reembed(_reembed_args(batch_size=2), _store=s) == 0
    assert s.embedding_scheme == SCHEME_TASK_PREFIX
    assert "now 'task_prefix'" in capsys.readouterr().err

    assert _cli._cmd_reembed(_reembed_args(yes=True), _store=s) == 0
    assert "nothing to re-encode" in capsys.readouterr().err


def test_cli_reembed_rejects_bad_batch_size(db_path):
    assert _cli._cmd_reembed(_reembed_args(batch_size=0), _store=_open(db_path)) == 2


def test_cli_reembed_reports_failure_without_change(db_path, capsys):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path, RecordingEmbedder(fail_on_call=1))
    assert _cli._cmd_reembed(_reembed_args(yes=True), _store=s) == 1
    assert "No vectors were changed" in capsys.readouterr().err


def test_reembed_is_wired_into_the_standalone_cli(db_path, monkeypatch, capsys):
    _make_legacy(db_path, TEXTS)
    monkeypatch.setattr(sys, "argv",
                        ["hermes-memory", "reembed", "--dry-run", "--path", db_path])
    assert _cli.main() == 0
    assert "would re-encode 7 memories" in capsys.readouterr().out


def test_reembed_is_wired_into_the_host_cli(monkeypatch):
    parser = argparse.ArgumentParser()
    _cli.register_cli(parser)
    args = parser.parse_args(["reembed", "-y", "--batch-size", "8", "--path", "/x"])
    seen = []
    monkeypatch.setattr(_cli, "_cmd_reembed", lambda a: seen.append(a) or 0)
    assert args.func(args) == 0
    assert seen[0].yes and seen[0].batch_size == 8 and seen[0].path == "/x"


def test_doctor_and_stats_recommend_reembed_for_a_legacy_store(db_path, capsys):
    _make_legacy(db_path, TEXTS)
    s = _open(db_path)
    _cli._cmd_doctor(argparse.Namespace(), _store=s)
    out = capsys.readouterr().out
    assert "embedding_scheme:  plain (legacy: no task prefixes)" in out
    assert "hermes-memory reembed" in out
    _cli._cmd_stats(argparse.Namespace(json=False), _store=s)
    out = capsys.readouterr().out
    assert "embedding_scheme:  plain (legacy" in out and "reembed" in out
    _cli._cmd_stats(argparse.Namespace(json=True), _store=s)
    payload = json.loads(capsys.readouterr().out)
    assert payload["embedding_scheme"] == SCHEME_PLAIN
    assert payload["embedding_scheme_preferred"] == SCHEME_TASK_PREFIX


def test_doctor_and_stats_are_quiet_for_a_new_store(db_path, capsys):
    s = _open(db_path)
    s.store(text="fresh", category="fact")
    _cli._cmd_doctor(argparse.Namespace(), _store=s)
    _cli._cmd_stats(argparse.Namespace(json=False), _store=s)
    out = capsys.readouterr().out
    assert "embedding_scheme:  task_prefix" in out
    assert "reembed" not in out


def _export(s: MemoryStore, path: str) -> list[dict]:
    args = argparse.Namespace(out=path, limit=1000, include_archived=True,
                              salvage=False, quiet=True)
    assert _cli._cmd_export(args, _store=s) == 0
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _import(s: MemoryStore, path: str) -> int:
    args = argparse.Namespace(input=path, allow_existing=False, reembed=False, quiet=True)
    return _cli._cmd_import(args, _store=s)


def test_export_records_the_scheme(db_path, tmp_path):
    s = _open(db_path)
    s.store(text="exported", category="fact")
    (record,) = _export(s, str(tmp_path / "out.jsonl"))
    assert record["embedding_scheme"] == SCHEME_TASK_PREFIX


def test_import_reencodes_vectors_from_another_scheme(tmp_path):
    legacy = str(tmp_path / "legacy")
    _make_legacy(legacy, ["imported memory"])
    records = _export(_open(legacy), str(tmp_path / "out.jsonl"))
    # An export written before the field existed: all its vectors are plain.
    for r in records:
        r.pop("embedding_scheme")
    old_export = tmp_path / "old.jsonl"
    old_export.write_text("".join(json.dumps(r) + "\n" for r in records))

    fresh = _open(str(tmp_path / "fresh"))
    assert _import(fresh, str(old_export)) == 0
    (vec,) = _vectors(fresh).values()
    assert _close(vec, _vec(DOC + "imported memory"))


def test_import_keeps_vectors_of_the_same_scheme(tmp_path):
    src = _open(str(tmp_path / "src"))
    src.store(text="same scheme", category="fact")
    export = str(tmp_path / "out.jsonl")
    (record,) = _export(src, export)
    emb = RecordingEmbedder()
    dst = _open(str(tmp_path / "dst"), emb)
    assert _import(dst, export) == 0
    assert emb.inputs == []
    (vec,) = _vectors(dst).values()
    assert _close(vec, record["vector"])
