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
from pathlib import Path
from typing import Any

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("lancedb.pydantic")

from hermes_memory_lancedb_pro._cli import (
    _cmd_doctor,
    _cmd_export,
    _cmd_import,
    _cmd_install_plugin,
    _cmd_uninstall_plugin,
    _resolve_hermes_home,
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

    def test_export_creates_missing_parent_directories(self, store, tmp_path):
        """export -o into a not-yet-existing directory creates the path."""
        store.store(text="memory written to a freshly created backup directory")
        out_path = tmp_path / "rescue" / "nested" / "memory-backup.jsonl"
        args = _Args(
            path=store.db_path,
            quiet=True,
            out=str(out_path),
            limit=100_000,
            include_archived=False,
        )
        rc = _cmd_export(args, _store=store)
        assert rc == 0
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8").strip()

    def test_export_fails_loudly_when_scan_errors(self, store, tmp_path):
        """A failed table scan exits non-zero instead of writing an empty file."""
        def _boom(*_a, **_kw):
            raise RuntimeError("simulated missing fragment")

        store._scan_all = _boom
        out_path = tmp_path / "should-not-be-created.jsonl"
        args = _Args(
            path=store.db_path,
            quiet=True,
            out=str(out_path),
            limit=100_000,
            include_archived=False,
            salvage=False,
        )
        rc = _cmd_export(args, _store=store)
        assert rc == 1
        assert not out_path.exists()

    def test_export_salvage_recovers_rows_from_healthy_store(self, store, tmp_path):
        """--salvage exports every row when the dataset is intact."""
        for i in range(5):
            store.store(text=f"salvageable memory number {i} with ample text")
        out_path = tmp_path / "salvage.jsonl"
        args = _Args(
            path=store.db_path,
            quiet=True,
            out=str(out_path),
            limit=100_000,
            include_archived=False,
            salvage=True,
        )
        rc = _cmd_export(args, _store=store)
        assert rc == 0
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) == 5


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


class TestInstallPlugin:
    """install-plugin must write the shim where hermes-agent actually scans:
    `<get_hermes_home()>/plugins/<name>/` — `~/.hermes/plugins/lancedb_pro/`
    by default. NOT `~/.hermes/hermes-agent/...`, and NOT the `plugins/memory/`
    subdir (which is only for providers bundled inside hermes-agent)."""

    import argparse as _argparse

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        """Sandbox Path.home() so default-home resolution and stale-root
        cleanup never touch the real ~/.hermes."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        return tmp_path

    def _ns(self, hermes_home=None, force=False):
        return self._argparse.Namespace(
            hermes_home=(str(hermes_home) if hermes_home else None),
            force=force, quiet=True,
        )

    # ---- _resolve_hermes_home: must match the host's get_hermes_home() ----

    def test_resolve_home_default_is_dot_hermes(self, home):
        # hermes-agent's get_hermes_home() defaults to ~/.hermes — NOT
        # ~/.hermes/hermes-agent. A mismatch hides the plugin entirely.
        assert _resolve_hermes_home(None) == (home / ".hermes").resolve()

    def test_resolve_home_uses_env(self, home, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(home / "custom"))
        assert _resolve_hermes_home(None) == (home / "custom").resolve()

    def test_resolve_home_explicit_arg_wins(self, home, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(home / "env"))
        assert _resolve_hermes_home(str(home / "explicit")) == (
            home / "explicit"
        ).resolve()

    # ---- install / uninstall ---------------------------------------------

    def test_install_creates_flat_path_under_dot_hermes(self, home):
        assert _cmd_install_plugin(self._ns()) == 0
        flat = home / ".hermes" / "plugins" / "lancedb_pro"
        assert (flat / "__init__.py").exists()
        assert (flat / "cli.py").exists()
        assert (flat / "plugin.yaml").exists()
        # Not the bundled-only plugins/memory/ subdir, not the old wrong home.
        assert not (home / ".hermes" / "plugins" / "memory" / "lancedb_pro").exists()
        assert not (home / ".hermes" / "hermes-agent" / "plugins").exists()

    def test_install_shim_passes_host_discovery_textscan(self, home):
        """hermes-agent's _is_memory_provider_dir() text-scans __init__.py for
        'register_memory_provider' or 'MemoryProvider' — the shim must contain
        one or the host never recognises the plugin directory."""
        _cmd_install_plugin(self._ns())
        shim = (
            home / ".hermes" / "plugins" / "lancedb_pro" / "__init__.py"
        ).read_text()
        assert "register_memory_provider" in shim or "MemoryProvider" in shim

    def test_reinstall_up_to_date_is_noop(self, home):
        assert _cmd_install_plugin(self._ns()) == 0
        assert _cmd_install_plugin(self._ns()) == 0

    def test_reinstall_refreshes_a_stale_shim(self, home):
        _cmd_install_plugin(self._ns())
        init_path = home / ".hermes" / "plugins" / "lancedb_pro" / "__init__.py"
        init_path.write_text("# stale outdated shim\n")
        assert _cmd_install_plugin(self._ns()) == 0
        refreshed = init_path.read_text()
        assert "register_memory_provider" in refreshed or "MemoryProvider" in refreshed

    def test_install_migrates_from_legacy_memory_subdir(self, home):
        legacy = home / ".hermes" / "plugins" / "memory" / "lancedb_pro"
        legacy.mkdir(parents=True)
        (legacy / "__init__.py").write_text("# old shim")
        (legacy / "plugin.yaml").write_text("name: lancedb_pro\n")
        assert _cmd_install_plugin(self._ns()) == 0
        flat = home / ".hermes" / "plugins" / "lancedb_pro"
        assert "register_memory_provider" in (flat / "__init__.py").read_text()
        assert not legacy.exists()

    def test_install_cleans_stale_old_home_root(self, home):
        # A pre-0.11.41 installer left a shim under ~/.hermes/hermes-agent/.
        stale = home / ".hermes" / "hermes-agent" / "plugins" / "lancedb_pro"
        stale.mkdir(parents=True)
        (stale / "__init__.py").write_text("# stale wrong-home shim")
        assert _cmd_install_plugin(self._ns()) == 0
        assert (
            home / ".hermes" / "plugins" / "lancedb_pro" / "__init__.py"
        ).exists()
        assert not stale.exists()

    def test_uninstall_removes_flat_and_legacy(self, home):
        _cmd_install_plugin(self._ns())
        legacy = home / ".hermes" / "plugins" / "memory" / "lancedb_pro"
        legacy.mkdir(parents=True)
        (legacy / "stale.txt").write_text("x")
        assert _cmd_uninstall_plugin(self._ns()) == 0
        assert not (home / ".hermes" / "plugins" / "lancedb_pro").exists()
        assert not legacy.exists()

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

    def test_register_cli_uses_own_subparsers_group(self):
        """register_cli must call add_subparsers on the fresh parser it receives.

        Per the hermes memory plugin spec, hermes-agent passes a FRESH
        ArgumentParser for the provider's own namespace.  Commands appear as
        'hermes lancedb_pro <subcommand>', NOT inside 'hermes memory'.
        """
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
        register_cli(parser)

        # All five commands parse correctly in the provider's own namespace
        for cmd in ("init", "doctor", "export", "import", "reset"):
            args = parser.parse_args([cmd])
            assert getattr(args, "lancedb_pro_command", None) == cmd, (
                f"'{cmd}' should be a valid lancedb_pro_command"
            )

    def test_register_cli_top_level_func_for_dispatch(self):
        """register_cli must set func on the top-level parser for args.func(args) dispatch.

        The spec pattern is: subparser.set_defaults(func=dispatcher)
        A single dispatcher reads args.lancedb_pro_command to route.
        """
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
        register_cli(parser)

        # Top-level func must be callable even before a subcommand is specified
        args = parser.parse_args([])
        assert callable(getattr(args, "func", None)), (
            "register_cli must call subparser.set_defaults(func=...) on the parent parser"
        )

    def test_register_cli_reset_is_reset_not_lancedb_reset(self):
        """DB-reset command is 'reset' (not 'lancedb-reset') in the provider's own namespace.

        There is no collision: 'hermes lancedb_pro reset' and 'hermes memory reset'
        are separate namespaces.  The lancedb-reset workaround is not needed.
        """
        import argparse

        parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
        register_cli(parser)

        args = parser.parse_args(["reset"])
        assert args.lancedb_pro_command == "reset"
        assert callable(getattr(args, "func", None))
