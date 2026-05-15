"""Tests for the hermes-memory-lancedb-pro CLI and register_cli plugin commands.

Integration tests use a temporary LanceDB store with StubEmbedder (same
pattern as test_store_integration.py).  Plain unit tests patch sys.argv and
call main() / register_cli() directly so they don't require LanceDB.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
from typing import Any

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro._cli import (
    _cmd_doctor,
    _cmd_export,
    _cmd_import,
    main,
    register_cli,
)
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# StubEmbedder (mirrors test_store_integration.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_factory():
    """Return a factory that creates temporary MemoryStore instances."""
    dirs: list[str] = []

    def _make() -> MemoryStore:
        d = tempfile.mkdtemp(prefix="hermes-cli-test-")
        dirs.append(d)
        s = MemoryStore(db_path=d)
        s._initialise()
        s._embedder = StubEmbedder()
        return s

    yield _make

    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(store_factory):
    """Single pre-built store for tests that only need one."""
    return store_factory()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Args:
    """Lightweight argparse.Namespace stand-in for unit tests."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _export_to_str(store: MemoryStore, **kwargs) -> str:
    """Run _cmd_export against *store*, return captured JSONL string.

    Passes the pre-built store directly to bypass the second-instantiation
    path, which fails with the LanceDB version in the test venv when
    ``list_tables()`` returns a ListTablesResponse object rather than a list.
    """
    buf = io.StringIO()
    args = _Args(
        path=store.db_path,
        quiet=True,
        out="-",
        limit=kwargs.get("limit", 100_000),
        include_archived=kwargs.get("include_archived", False),
    )
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = _cmd_export(args, _store=store)
    finally:
        sys.stdout = old_stdout
    assert rc == 0
    return buf.getvalue()


def _import_from_str(store: MemoryStore, jsonl: str, **kwargs) -> int:
    """Run _cmd_import against *store* with JSONL from a string."""
    args = _Args(
        path=store.db_path,
        quiet=True,
        input="-",
        reembed=kwargs.get("reembed", False),
        allow_existing=kwargs.get("allow_existing", False),
    )
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(jsonl)
    try:
        rc = _cmd_import(args, _store=store)
    finally:
        sys.stdin = old_stdin
    return rc


def _doctor_output(store: MemoryStore) -> str:
    """Run _cmd_doctor, capture stdout, return it as a string."""
    buf = io.StringIO()
    args = _Args(path=store.db_path, quiet=True)
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rc = _cmd_doctor(args, _store=store)
    finally:
        sys.stdout = old_stdout
    assert rc == 0
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TestExport
# ---------------------------------------------------------------------------


class TestExport:
    def test_round_trip_count_and_text(self, store_factory):
        """export → import reproduces the same count and texts."""
        src = store_factory()
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "memory about user preferences for dark mode",
            "fact about project deadline being in march",
        ]
        for txt in texts:
            src.store(text=txt, category="fact")

        jsonl = _export_to_str(src)
        lines = [ln for ln in jsonl.splitlines() if ln.strip()]
        assert len(lines) == len(texts)

        dst = store_factory()
        rc = _import_from_str(dst, jsonl)
        assert rc == 0

        # All texts should be importable and discoverable
        imported_rows = list(dst._scan_all(limit=1000))
        imported_texts = {r.get("text") for r in imported_rows}
        for txt in texts:
            assert txt in imported_texts

    def test_default_excludes_archived(self, store):
        """Default export excludes archived rows."""
        active_id = store.store(text="active memory content here for testing")
        archived_id = store.store(text="soon to be archived memory content")
        # Archive the second row by superseding it
        store.update(archived_id, text="replacement text content for the archived row")

        jsonl = _export_to_str(store)
        lines = [ln for ln in jsonl.splitlines() if ln.strip()]
        exported_ids = {json.loads(ln)["id"] for ln in lines}

        # active_id should appear (it's active)
        assert active_id in exported_ids
        # archived_id should NOT appear (it's archived)
        assert archived_id not in exported_ids

    def test_include_archived_exports_archived_rows(self, store):
        """--include-archived includes archived rows."""
        archived_id = store.store(text="will be archived shortly for this test")
        store.update(archived_id, text="replacement supersedes the previous entry")

        # Without flag
        jsonl_default = _export_to_str(store)
        ids_default = {json.loads(ln)["id"] for ln in jsonl_default.splitlines() if ln.strip()}
        assert archived_id not in ids_default

        # With flag
        jsonl_all = _export_to_str(store, include_archived=True)
        ids_all = {json.loads(ln)["id"] for ln in jsonl_all.splitlines() if ln.strip()}
        assert archived_id in ids_all

    def test_limit_truncates_output(self, store):
        """--limit N caps the number of exported rows."""
        for i in range(5):
            store.store(text=f"memory entry number {i} with padding content here")

        jsonl = _export_to_str(store, limit=3)
        lines = [ln for ln in jsonl.splitlines() if ln.strip()]
        assert len(lines) <= 3

    def test_exported_row_has_required_fields(self, store):
        """Every exported row contains the expected keys."""
        store.store(text="complete row with all expected fields for export check")
        jsonl = _export_to_str(store)
        for line in jsonl.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("id", "text", "category", "scope", "importance",
                          "timestamp", "vector", "metadata"):
                assert field in record, f"missing field: {field}"
            assert len(record["vector"]) == VECTOR_DIM


# ---------------------------------------------------------------------------
# TestDoctor
# ---------------------------------------------------------------------------


class TestDoctor:
    def test_empty_store_no_anomalies(self, store):
        """An empty store reports zero counts cleanly with no anomalies."""
        out = _doctor_output(store)
        assert "total_memories:    0" in out
        assert "No anomalies detected." in out
        assert "No recommendations." in out

    def test_orphan_supersede_flagged(self, store):
        """A row referencing a non-existent supersedes id is flagged."""
        # Manually insert a row with a bogus supersedes pointer
        fake_supersedes_id = "00000000-dead-beef-0000-000000000000"
        meta = {
            "tier": "working",
            "access_count": 0,
            "confidence": 0.8,
            "temporal_type": "static",
            "state": "confirmed",
            "source": "manual",
            "source_session": "",
            "cross_session": False,
            "created_at": 0,
            "last_accessed_at": 0,
            "injected_count": 0,
            "bad_recall_count": 0,
            "supersedes": fake_supersedes_id,
            "superseded_by": None,
            "valid_from": 0,
            "valid_until": None,
            "fact_key": None,
            "relations": [],
        }
        store.store_raw(
            text="orphan supersede test entry content",
            vector=StubEmbedder()._one("orphan supersede test entry content", True),
            category="fact",
            scope="global",
            importance=0.5,
            metadata=json.dumps(meta),
        )

        out = _doctor_output(store)
        assert "Orphan supersedes" in out

    def test_archived_ratio_recommendation(self, store):
        """When archived ratio > 30%, a purge recommendation is printed."""
        # Store 4 memories, archive 2 → 2/4 = 50% archived
        ids = []
        for i in range(4):
            ids.append(store.store(text=f"ratio test memory number {i} content here"))

        # Archive 2 by superseding them
        store.update(ids[0], text="replacement for ratio test memory zero content")
        store.update(ids[1], text="replacement for ratio test memory one content here")

        out = _doctor_output(store)
        assert "archived_ratio" in out
        # Recommendation should appear since > 30%
        assert "purge" in out.lower() or "reclaim" in out.lower()

    def test_counts_in_output(self, store):
        """Doctor reports correct active/archived counts."""
        store.store(text="single active memory for counts test content")
        out = _doctor_output(store)
        assert "active:" in out
        assert "archived:" in out


# ---------------------------------------------------------------------------
# TestCli  (argv-level tests for hermes-memory-lancedb-pro bootstrap CLI)
# ---------------------------------------------------------------------------


class TestCli:
    def test_top_level_help_lists_bootstrap_commands(self, monkeypatch, capsys):
        """hermes-memory-lancedb-pro --help lists install-plugin and uninstall-plugin."""
        monkeypatch.setattr(sys, "argv", ["hermes-memory-lancedb-pro", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        help_text = captured.out
        assert "install-plugin" in help_text
        assert "uninstall-plugin" in help_text

    def test_no_args_exits_zero(self, monkeypatch, capsys):
        """hermes-memory-lancedb-pro with no args prints help and exits 0."""
        monkeypatch.setattr(sys, "argv", ["hermes-memory-lancedb-pro"])
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # some help text

    def test_unknown_subcommand_exits_nonzero(self, monkeypatch):
        """hermes-memory-lancedb-pro bogus-cmd exits non-zero."""
        monkeypatch.setattr(sys, "argv", ["hermes-memory-lancedb-pro", "bogus-cmd"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# TestRegisterCli  (plugin CLI slot wired by hermes-agent)
# ---------------------------------------------------------------------------


class TestRegisterCli:
    def test_register_cli_adds_export_import_doctor(self, capsys):
        """register_cli wires export, import, doctor onto the given subparser."""
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb-pro")
        register_cli(parser)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        help_text = captured.out
        assert "export" in help_text
        assert "import" in help_text
        assert "doctor" in help_text

    def test_export_help_via_register_cli(self, capsys):
        """hermes lancedb-pro export --help exits 0."""
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb-pro")
        register_cli(parser)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["export", "--help"])
        assert exc_info.value.code == 0

    def test_register_cli_adds_to_existing_subparsers_group(self, capsys):
        """register_cli must extend an existing subparsers group, not add a nested one.

        This is the real hermes-agent scenario: the parser already has a subparsers
        group containing 'setup', 'status', 'off', 'reset'.  Our commands must appear
        alongside them, not hidden in a nested group that argparse ignores at dispatch.
        """
        import argparse

        # Simulate hermes-agent's memory parser with its built-in commands
        parser = argparse.ArgumentParser(prog="hermes memory")
        subs = parser.add_subparsers(dest="memory_command")
        subs.add_parser("setup")
        subs.add_parser("status")
        subs.add_parser("off")
        subs.add_parser("reset")

        register_cli(parser)

        # All original commands still parseable
        for cmd in ("setup", "status", "off", "reset"):
            args = parser.parse_args([cmd])
            assert args.memory_command == cmd

        # Our commands now parse successfully via the same group
        for cmd in ("doctor", "export", "import", "lancedb-reset"):
            args = parser.parse_args([cmd])
            assert args.memory_command == cmd, f"'{cmd}' should be a valid memory_command"

    def test_register_cli_commands_have_func_default(self):
        """Each command registered by register_cli must set args.func for dispatch."""
        import argparse

        parser = argparse.ArgumentParser(prog="hermes memory")
        subs = parser.add_subparsers(dest="memory_command")
        subs.add_parser("setup")

        register_cli(parser)

        for cmd in ("doctor", "export", "import", "lancedb-reset"):
            args = parser.parse_args([cmd])
            assert callable(getattr(args, "func", None)), (
                f"'{cmd}' subparser must set args.func for hermes-agent dispatch"
            )

    def test_lancedb_reset_not_reset(self):
        """Our DB-reset command must be 'lancedb-reset', not 'reset', to avoid collision."""
        import argparse

        parser = argparse.ArgumentParser(prog="hermes memory")
        subs = parser.add_subparsers(dest="memory_command")
        subs.add_parser("reset")  # hermes-agent built-in

        register_cli(parser)

        choices = subs.choices if hasattr(subs, "choices") else {}
        assert "lancedb-reset" in choices, "plugin must register 'lancedb-reset'"
        assert choices.get("reset") is not choices.get("lancedb-reset"), (
            "plugin must not overwrite hermes-agent's built-in 'reset'"
        )
