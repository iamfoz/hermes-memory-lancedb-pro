"""Console-script entry points for the installed package."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import task_ledger as _tl
from ._sql import parse_metadata as _parse_metadata
from .store import (
    DEFAULT_DB_PATH,
    MAX_SCAN_ROWS,
    VECTOR_DIM,
    MemoryStore,
)

PLUGIN_NAME = "lancedb_pro"
# IMPORTANT: hermes-agent's `_is_memory_provider_dir()` gates user-installed
# plugins with a cheap TEXT SCAN of this __init__.py — it must contain the
# literal string "register_memory_provider" or "MemoryProvider" or the host
# will not recognise the directory as a memory provider at all (no provider,
# no `hermes lancedb_pro` CLI). Both `register` and `register_memory_provider`
# are re-exported so the host's loader finds an entry point either way.
PLUGIN_SHIM_CONTENT = '''\
"""Hermes plugin discovery shim for hermes-memory-lancedb-pro.

The heavy package (lancedb, sentence-transformers, ...) must be installed
into Hermes' own Python environment with `hermes-pip install
hermes-memory-lancedb-pro`; this shim only re-exports the plugin's
`register` / `register_memory_provider` entry points so hermes-agent's
plugin loader recognises and discovers this MemoryProvider. If the import
below fails, the package landed in the wrong environment — reinstall with
hermes-pip.

Regenerate with: hermes-memory-lancedb-pro install-plugin
"""
from hermes_memory_lancedb_pro.provider import register, register_memory_provider

__all__ = ["register", "register_memory_provider"]
'''

PLUGIN_CLI_CONTENT = '''\
"""Hermes plugin CLI shim for hermes-memory-lancedb-pro.

Exposes register_cli() so hermes-agent can wire the lancedb_pro commands
(init, doctor, export, import, reset) into `hermes lancedb_pro`.

Regenerate with: hermes-memory-lancedb-pro install-plugin
"""
from hermes_memory_lancedb_pro._cli import register_cli

__all__ = ["register_cli"]
'''

SMOKE_PREFIX = "SMOKE_TEST_"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _entry_text(r) -> str:
    if isinstance(r, tuple):
        r = r[0]
    return r.get("text", "") if isinstance(r, dict) else ""


def _entry_id(r) -> str:
    if isinstance(r, tuple):
        r = r[0]
    return r.get("id", "") if isinstance(r, dict) else ""


def _open_store(args: argparse.Namespace) -> MemoryStore:
    """Open (or create) a MemoryStore at the path given by ``args.path``."""
    db_path = getattr(args, "path", None) or DEFAULT_DB_PATH
    store = MemoryStore(db_path=db_path)
    store._initialise()
    return store


def _stderr(msg: str, quiet: bool = False) -> None:
    """Write *msg* to stderr unless ``--quiet`` was passed."""
    if not quiet:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Export subcommand
# ---------------------------------------------------------------------------


def _cmd_export(
    args: argparse.Namespace,
    _store: MemoryStore | None = None,
) -> int:
    """Export memories to JSONL format.

    Each output line is a JSON object with ``id``, ``text``, ``category``,
    ``scope``, ``importance``, ``timestamp``, ``vector`` (list of floats),
    and ``metadata`` (JSON string).

    ``_store`` is an optional pre-built store, used by tests to bypass the
    second-instantiation path which can fail with some LanceDB versions.
    """
    store = _store if _store is not None else _open_store(args)
    quiet = getattr(args, "quiet", False)
    out_path = getattr(args, "out", "-")
    limit = getattr(args, "limit", 100_000)
    include_archived = getattr(args, "include_archived", False)

    out_file = sys.stdout if out_path == "-" else open(out_path, "w", encoding="utf-8")  # noqa: SIM115

    try:
        count = 0
        for row in store._scan_all(limit=limit):
            meta_str = row.get("metadata", "{}")
            meta = _parse_metadata(meta_str)
            if not include_archived and meta.get("state") == "archived":
                continue

            # Normalise the vector to a plain Python list of floats.
            raw_vec = row.get("vector")
            if raw_vec is None:
                vector: list[float] = []
            elif hasattr(raw_vec, "tolist"):
                vector = raw_vec.tolist()
            else:
                vector = list(raw_vec)

            record: dict[str, Any] = {
                "id": row.get("id", ""),
                "text": row.get("text", ""),
                "category": row.get("category", "other"),
                "scope": row.get("scope", "global"),
                "importance": float(row.get("importance", 0.5)),
                "timestamp": int(row.get("timestamp", 0)),
                "vector": vector,
                # Keep metadata as a JSON *string* so the import path can
                # round-trip it without re-serialising the dict.
                "metadata": meta_str if isinstance(meta_str, str) else json.dumps(meta),
            }
            out_file.write(json.dumps(record) + "\n")
            count += 1

        destination = "stdout" if out_path == "-" else out_path
        _stderr(f"wrote {count} memories to {destination}", quiet)
        return 0
    finally:
        if out_path != "-":
            out_file.close()


# ---------------------------------------------------------------------------
# Import subcommand
# ---------------------------------------------------------------------------


def _cmd_import(
    args: argparse.Namespace,
    _store: MemoryStore | None = None,
) -> int:
    """Import memories from a JSONL file produced by ``export``.

    By default refuses to import a row whose ``id`` is already present
    (non-archived) in the target store. Pass ``--allow-existing`` to skip
    duplicates silently. Pass ``--reembed`` to re-encode ``text`` with the
    current embedder instead of using the stored vector.

    ``_store`` is an optional pre-built store, used by tests to bypass the
    second-instantiation path which can fail with some LanceDB versions.
    """
    store = _store if _store is not None else _open_store(args)
    quiet = getattr(args, "quiet", False)
    in_path = getattr(args, "input", "-")
    allow_existing = getattr(args, "allow_existing", False)
    reembed = getattr(args, "reembed", False)

    in_file = sys.stdin if in_path == "-" else open(in_path, encoding="utf-8")  # noqa: SIM115

    imported = skipped = re_embedded = 0
    try:
        for lineno, raw_line in enumerate(in_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _stderr(f"  warning: line {lineno} is not valid JSON ({exc}); skipping", quiet)
                continue

            source_id = record.get("id", "")

            # Duplicate check
            if source_id and store.has_id(source_id):
                if allow_existing:
                    skipped += 1
                    continue
                _stderr(
                    f"  error: id {source_id!r} already exists in target store. "
                    "Use --allow-existing to skip duplicates.",
                    quiet=False,
                )
                return 1

            text = record.get("text", "")
            if not text or not text.strip():
                _stderr(f"  warning: line {lineno} has empty text; skipping", quiet)
                continue

            vector: list[float] = record.get("vector") or []
            dim_mismatch = len(vector) != VECTOR_DIM

            if reembed or dim_mismatch:
                if dim_mismatch and not reembed:
                    _stderr(
                        f"  warning: line {lineno} vector dim {len(vector)} != {VECTOR_DIM}; "
                        "re-encoding",
                        quiet,
                    )
                vector = store.encode(text)
                re_embedded += 1

            # Preserve the source id via metadata_extra so traceability is
            # maintained without modifying MemoryStore.store_raw's signature.
            meta_str = record.get("metadata", "{}")
            if isinstance(meta_str, dict):
                meta_str = json.dumps(meta_str)
            try:
                meta_dict = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
            meta_dict["source_id"] = source_id
            final_meta_str = json.dumps(meta_dict)

            store.store_raw(
                text=text,
                vector=vector,
                category=record.get("category", "other"),
                scope=record.get("scope", "global"),
                importance=float(record.get("importance", 0.5)),
                metadata=final_meta_str,
                timestamp=record.get("timestamp"),
            )
            imported += 1

    finally:
        if in_path != "-":
            in_file.close()

    _stderr(
        f"imported {imported} memories (skipped {skipped} duplicates, "
        f"re-embedded {re_embedded})",
        quiet,
    )
    return 0


# ---------------------------------------------------------------------------
# Doctor subcommand
# ---------------------------------------------------------------------------

_DOCTOR_MAX_SCAN = MAX_SCAN_ROWS
_ORPHAN_EXAMPLE_LIMIT = 5
_OLD_PERIPHERAL_DAYS = 90
_COMPACTION_MIN_AGE_DAYS = 7


def _cmd_doctor(
    args: argparse.Namespace,
    _store: MemoryStore | None = None,
) -> int:
    """Print a diagnostic report for the memory store.

    Checks counts, category/tier breakdowns, and several anomaly classes
    (orphan supersede chains, stale peripheral memories, cross_session
    memories with zero access).

    ``_store`` is an optional pre-built store, used by tests to bypass the
    second-instantiation path which can fail with some LanceDB versions.
    """
    store = _store if _store is not None else _open_store(args)
    stats = store.stats()

    total = stats["total_memories"]
    active = stats["active_memories"]
    archived = stats["archived_memories"]
    archived_ratio = (archived / total * 100) if total else 0.0

    # ---- Header ----
    print("=== Hermes Memory Doctor ===")
    print(f"db_path:           {stats['db_path']}")
    print(f"embedding_model:   {stats['embedding_model']}")
    print(f"vector_dimensions: {stats['vector_dimensions']}")
    print(f"total_memories:    {total}")
    print()

    # ---- Counts ----
    print("--- Counts ---")
    print(f"active:   {active}")
    print(f"archived: {archived}")
    print(f"archived_ratio: {archived_ratio:.1f}%")
    if archived_ratio > 30.0:
        print(
            "  → run 'hermes-memory purge --grace-days 30' to reclaim space "
            "(or call MemoryStore.purge_archived(grace_period_days=30))"
        )
    print()

    # ---- Categories ----
    print("--- Categories ---")
    categories = stats.get("categories", {})
    for cat, cnt in sorted(categories.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {cnt}")
    if not categories:
        print("  (none)")
    print()

    # ---- Tiers ----
    print("--- Tiers ---")
    tiers = stats.get("tiers", {})
    for tier, cnt in sorted(tiers.items(), key=lambda kv: -kv[1]):
        print(f"  {tier}: {cnt}")
    if not tiers:
        print("  (none)")
    print()

    # ---- Full scan for anomaly checks ----
    now_ms = int(time.time() * 1000)
    old_peripheral_cutoff = now_ms - (_OLD_PERIPHERAL_DAYS * 86_400_000)
    compaction_cutoff = now_ms - (_COMPACTION_MIN_AGE_DAYS * 86_400_000)

    all_ids: set[str] = set()
    rows_by_id: dict[str, dict] = {}
    orphan_supersedes: list[str] = []   # ids where supersedes target missing
    orphan_superseded_by: list[str] = []  # ids where superseded_by target missing
    old_peripheral_ids: list[str] = []
    cross_session_zero_access: list[str] = []
    archived_older_than_grace: int = 0
    old_non_archived_count: int = 0

    # First pass — collect all ids
    for row in store._scan_all(limit=_DOCTOR_MAX_SCAN):
        rid = row.get("id")
        if rid:
            all_ids.add(rid)
            rows_by_id[rid] = row

    # Second pass — check anomalies
    for rid, row in rows_by_id.items():
        meta = _parse_metadata(row.get("metadata", "{}"))
        state = meta.get("state", "confirmed")
        tier = meta.get("tier", "working")
        access_count = int(meta.get("access_count", 0) or 0)
        ts = int(row.get("timestamp", 0) or 0)
        cross_session = bool(meta.get("cross_session", False))
        invalidated_at = int(meta.get("invalidated_at", 0) or 0)

        # Orphan: supersedes points at a non-existent id
        sup = meta.get("supersedes")
        if sup and sup not in all_ids:
            orphan_supersedes.append(rid)

        # Orphan: superseded_by points at a non-existent id
        sup_by = meta.get("superseded_by")
        if sup_by and sup_by not in all_ids:
            orphan_superseded_by.append(rid)

        if state != "archived":
            # Old peripheral with zero access
            if (
                tier == "peripheral"
                and access_count == 0
                and ts < old_peripheral_cutoff
            ):
                old_peripheral_ids.append(rid)

            # Cross-session with zero access
            if cross_session and access_count == 0:
                cross_session_zero_access.append(rid)

            # Compaction candidates: old non-archived rows
            if ts and ts < compaction_cutoff:
                old_non_archived_count += 1
        else:
            # Archived older than 30 days (purge recommendation)
            grace_ms = 30 * 86_400_000
            check_ts = invalidated_at if invalidated_at else ts
            if check_ts and (now_ms - check_ts) > grace_ms:
                archived_older_than_grace += 1

    # ---- Anomalies ----
    print("--- Anomalies ---")
    has_anomalies = False

    if orphan_supersedes:
        has_anomalies = True
        print(f"Orphan supersedes (supersedes target missing): {len(orphan_supersedes)}")
        for eid in orphan_supersedes[:_ORPHAN_EXAMPLE_LIMIT]:
            print(f"  {eid}")

    if orphan_superseded_by:
        has_anomalies = True
        print(f"Orphan superseded_by (target missing): {len(orphan_superseded_by)}")
        for eid in orphan_superseded_by[:_ORPHAN_EXAMPLE_LIMIT]:
            print(f"  {eid}")

    if old_peripheral_ids:
        has_anomalies = True
        print(
            f"Old peripheral memories (>{_OLD_PERIPHERAL_DAYS}d, 0 access): "
            f"{len(old_peripheral_ids)}"
        )

    if cross_session_zero_access:
        has_anomalies = True
        print(
            f"Cross-session memories with 0 access (possible premature promotion): "
            f"{len(cross_session_zero_access)}"
        )

    if not has_anomalies:
        print("  No anomalies detected.")
    print()

    # ---- Recommendations ----
    print("--- Recommendations ---")
    any_rec = False

    if archived_older_than_grace > 0:
        any_rec = True
        print(
            f"Run `purge_archived(grace_period_days=30)` — "
            f"{archived_older_than_grace} archived rows older than 30 days."
        )

    if old_non_archived_count > 0:
        any_rec = True
        print(
            f"Consider running `run_compaction()` — "
            f"{old_non_archived_count} old non-archived rows that may be "
            "near-duplicates."
        )

    if not any_rec:
        print("  No recommendations.")
    print()

    return 0


# ---------------------------------------------------------------------------
# Plugin CLI — register_cli() wires hermes lancedb_pro <subcommand>
# ---------------------------------------------------------------------------


def _dispatch_plugin_cli(args: argparse.Namespace) -> int:
    """Dispatch hermes lancedb_pro <subcommand> to the correct handler."""
    cmd = getattr(args, "lancedb_pro_command", None)
    if cmd is None:
        return 0
    if cmd == "task":
        return _cmd_task_dispatch(args)
    return {
        "init": _cmd_init,
        "export": _cmd_export,
        "import": _cmd_import,
        "doctor": _cmd_doctor,
        "reset": _cmd_reset,
    }.get(cmd, lambda _: 0)(args)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Register lancedb-pro subcommands with the hermes-agent CLI.

    hermes-agent discovers ``cli.py`` via ``discover_plugin_cli_commands()``
    and calls this with a **fresh** ArgumentParser for the provider's own
    namespace.  Commands appear as:

        hermes lancedb_pro init|doctor|export|import|reset

    Follows the hermes memory plugin CLI spec exactly:
    ``add_subparsers`` on the fresh parser + ``set_defaults(func=dispatcher)``
    at the top level for ``args.func(args)`` dispatch.
    """
    subs = subparser.add_subparsers(dest="lancedb_pro_command")

    p_init = subs.add_parser(
        "init",
        help="Initialise the memory store (seed from MEMORY.md if empty)",
        description=(
            "Open or create the memory database and optionally seed entries from "
            "MEMORY.md when the store is empty."
        ),
    )
    p_init.add_argument("--path", default=None, metavar="PATH",
                        help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_init.add_argument("--memory-md", dest="memory_md", default=None, metavar="PATH",
                        help="Seed file (default: $MEMORY_MD or ~/.hermes/memory/MEMORY.md)")
    p_init.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    p_init.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress non-essential output")

    p_doctor = subs.add_parser(
        "doctor",
        help="Print a diagnostic report for the memory store",
        description="Scan the store and report counts, anomalies, and recommendations.",
    )
    p_doctor.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_doctor.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    p_export = subs.add_parser(
        "export",
        help="Export memories to JSONL",
        description="Stream memory rows to JSONL (one JSON object per line).",
    )
    p_export.add_argument("--out", "-o", default="-", metavar="PATH",
                          help="Output file path (default: stdout)")
    p_export.add_argument("--include-archived", action="store_true",
                          help="Include archived rows (excluded by default)")
    p_export.add_argument("--limit", type=int, default=100_000, metavar="N",
                          help="Maximum rows to export (default: 100000)")
    p_export.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_export.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    p_import = subs.add_parser(
        "import",
        help="Import memories from JSONL",
        description="Read a JSONL file produced by 'export' and write rows into the store.",
    )
    p_import.add_argument("--in", dest="input", default="-", metavar="PATH",
                          help="Input file path (default: stdin)")
    p_import.add_argument("--reembed", action="store_true",
                          help="Re-encode text with the current embedder instead of stored vectors")
    p_import.add_argument("--allow-existing", action="store_true",
                          help="Skip rows whose id already exists rather than aborting")
    p_import.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_import.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    p_reset = subs.add_parser(
        "reset",
        help="Wipe and reinitialise the LanceDB memory database",
        description=(
            "Delete the LanceDB database directory and re-run init, "
            "seeding fresh entries from MEMORY.md."
        ),
    )
    p_reset.add_argument("--path", default=None, metavar="PATH",
                         help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_reset.add_argument("--memory-md", dest="memory_md", default=None, metavar="PATH",
                         help="Seed file (default: $MEMORY_MD or ~/.hermes/memory/MEMORY.md)")
    p_reset.add_argument("-y", "--yes", action="store_true",
                         help="Skip confirmation prompt")
    p_reset.add_argument("-q", "--quiet", action="store_true",
                         help="Suppress non-essential output")

    p_task = subs.add_parser(
        "task",
        help="Manage durable task ledgers for long-running agent work",
        description=(
            "Task ledgers keep objective, iteration counter, and next_action in "
            "state.json outside the LLM context window.  The runner re-reads "
            "state.json each iteration so context compaction cannot lose progress."
        ),
    )
    _add_task_subparsers(p_task, dest="task_command")
    p_task.set_defaults(func=_cmd_task_dispatch)

    subparser.set_defaults(func=_dispatch_plugin_cli)


# ---------------------------------------------------------------------------
# install-plugin / uninstall-plugin
# ---------------------------------------------------------------------------


def _resolve_hermes_home(explicit: str | None) -> Path:
    """Pick the hermes home dir, matching hermes-agent's own
    ``hermes_constants.get_hermes_home()``:

        explicit ``--hermes-home`` arg  >  $HERMES_HOME env  >  ~/.hermes

    The default MUST be ``~/.hermes`` (not ``~/.hermes/hermes-agent``):
    that is where the host scans ``plugins/<name>/`` for user-installed
    providers. A mismatch here installs the shim where the host never
    looks, so the plugin is silently undiscovered."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _host_hermes_home() -> Path | None:
    """The home dir hermes-agent itself will use, via its own
    ``hermes_constants.get_hermes_home()`` — or None when hermes-agent is
    not importable from this environment. Used only to warn on a mismatch."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return Path(get_hermes_home()).expanduser().resolve()
    except Exception:
        return None


def _packaged_plugin_yaml() -> Path:
    """Return the path to the plugin.yaml shipped inside the installed wheel."""
    return Path(__file__).resolve().parent / "plugin.yaml"


def _cleanup_stale_home_roots(current_home: Path, quiet: bool) -> None:
    """Remove lancedb_pro shims left under the OLD wrong default home
    (``~/.hermes/hermes-agent``) by pre-0.11.41 installers. The host scans
    ``~/.hermes/plugins/`` — anything under the old root is dead weight."""
    old_home = (Path.home() / ".hermes" / "hermes-agent").resolve()
    if old_home == current_home.resolve():
        return
    for stale in (
        old_home / "plugins" / PLUGIN_NAME,
        old_home / "plugins" / "memory" / PLUGIN_NAME,
    ):
        if stale.exists():
            try:
                shutil.rmtree(str(stale))
                if not quiet:
                    sys.stdout.write(
                        f"Removed stale install from the old default home: {stale}\n"
                    )
            except Exception as e:
                _stderr(f"Could not remove stale install {stale}: {e}", quiet=False)


def _warn_home_mismatch(installed_home: Path, quiet: bool) -> None:
    """If hermes-agent is importable here and resolves a DIFFERENT home dir
    than where we installed, warn loudly — the plugin would be undiscovered."""
    host_home = _host_hermes_home()
    if host_home is not None and host_home != installed_home.resolve():
        _stderr(
            f"WARNING: hermes-agent resolves its home to\n"
            f"  {host_home}\n"
            f"but the shim was installed under\n"
            f"  {installed_home}\n"
            f"The host will NOT discover the plugin. Re-run with:\n"
            f"  hermes-memory-lancedb-pro install-plugin --hermes-home {host_home}\n"
            f"or set $HERMES_HOME so both agree.",
            quiet=False,
        )


def _cmd_install_plugin(args: argparse.Namespace) -> int:
    """Create ``<hermes_home>/plugins/lancedb_pro/`` with the discovery shim,
    cli.py, and a copy of plugin.yaml so hermes-agent can find this provider.

    hermes-agent discovers user-installed memory providers at
    ``$HERMES_HOME/plugins/<name>/`` (flat — the ``plugins/memory/`` subdir is
    only for providers bundled inside hermes-agent itself). Auto-migrates from
    the ``plugins/memory/lancedb_pro/`` path that 0.11.1–0.11.37 installers
    wrongly used and which the host never scans."""
    hermes_home = _resolve_hermes_home(getattr(args, "hermes_home", None))
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    init_path = plugin_dir / "__init__.py"
    cli_path = plugin_dir / "cli.py"
    yaml_target = plugin_dir / "plugin.yaml"
    yaml_source = _packaged_plugin_yaml()
    quiet = bool(getattr(args, "quiet", False))

    if not yaml_source.exists():
        _stderr(f"plugin.yaml missing from installed package at {yaml_source}", quiet=False)
        return 1

    # --- Auto-migrate the wrong path (plugins/memory/<name>/ → plugins/<name>/) ---
    # 0.11.1–0.11.37 installed under plugins/memory/, which the host's
    # user-plugin scan (plugins/<name>/) never finds — so the plugin would
    # silently fail to load.
    legacy_dir = hermes_home / "plugins" / "memory" / PLUGIN_NAME
    migrated = False
    if legacy_dir.exists() and not plugin_dir.exists():
        if not quiet:
            sys.stdout.write(
                f"Migrating plugin to the path hermes-agent actually scans:\n"
                f"  {legacy_dir} → {plugin_dir}\n"
            )
        try:
            plugin_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(legacy_dir), str(plugin_dir))
            shutil.rmtree(str(legacy_dir), ignore_errors=True)
            migrated = True
            if not quiet:
                sys.stdout.write("  Migration complete. Updating files to latest version.\n")
        except Exception as e:
            _stderr(
                f"Migration failed ({e}); installing fresh to {plugin_dir}.",
                quiet=False,
            )
    elif legacy_dir.exists():
        if not quiet:
            sys.stdout.write(
                f"Note: stale plugin directory found at {legacy_dir}\n"
                f"  hermes-agent does not scan that path — run "
                f"`rm -rf {legacy_dir}` to clean it up.\n"
            )

    # Remove installs left under the old wrong default home (~/.hermes/hermes-agent).
    _cleanup_stale_home_roots(hermes_home, quiet)

    existing_files = [p for p in (init_path, cli_path, yaml_target) if p.exists()]
    # Detect a stale shim — package upgraded but the shim files not
    # regenerated. The shim MUST match the installed package (an outdated
    # __init__.py can fail the host's discovery text-scan), so a stale
    # install is auto-refreshed rather than refused as "already installed".
    stale = False
    if existing_files:
        try:
            stale = (
                not init_path.exists()
                or init_path.read_text(encoding="utf-8") != PLUGIN_SHIM_CONTENT
                or not cli_path.exists()
                or cli_path.read_text(encoding="utf-8") != PLUGIN_CLI_CONTENT
                or not yaml_target.exists()
                or yaml_target.read_bytes() != yaml_source.read_bytes()
            )
        except Exception:
            stale = True
    # A just-migrated directory carries the OLD shim — always refresh it.
    force = bool(getattr(args, "force", False)) or migrated or stale
    if existing_files and not force:
        if not quiet:
            sys.stdout.write(
                f"{PLUGIN_NAME} plugin already installed and up to date at "
                f"{plugin_dir}\n"
            )
        _warn_home_mismatch(hermes_home, quiet)
        return 0

    plugin_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text(PLUGIN_SHIM_CONTENT, encoding="utf-8")
    cli_path.write_text(PLUGIN_CLI_CONTENT, encoding="utf-8")
    shutil.copyfile(yaml_source, yaml_target)

    if not quiet:
        action = "Reinstalled" if existing_files else "Installed"
        sys.stdout.write(
            f"{action} {PLUGIN_NAME} plugin at {plugin_dir}\n"
            f"  - {init_path.name} (discovery shim)\n"
            f"  - {cli_path.name} (plugin CLI shim)\n"
            f"  - {yaml_target.name} (manifest)\n"
            f"Next: set `memory.provider: lancedb_pro` in config.yaml, restart\n"
            f"the Hermes gateway, then `hermes memory setup` (optional).\n"
        )
    _warn_home_mismatch(hermes_home, quiet)
    return 0


def _cmd_uninstall_plugin(args: argparse.Namespace) -> int:
    """Remove ``<hermes_home>/plugins/lancedb_pro/``. Only deletes files we
    install (``__init__.py``, ``cli.py``, ``plugin.yaml``) and then the dir if empty
    — refuses to delete a dir containing unknown files.

    Also removes the stale ``<hermes_home>/plugins/memory/lancedb_pro/`` path
    that 0.11.1–0.11.37 installers wrongly used."""
    hermes_home = _resolve_hermes_home(getattr(args, "hermes_home", None))
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    quiet = bool(getattr(args, "quiet", False))

    # Remove the stale plugins/memory/<name>/ path if it still exists.
    legacy_dir = hermes_home / "plugins" / "memory" / PLUGIN_NAME
    if legacy_dir.exists():
        try:
            shutil.rmtree(str(legacy_dir))
            if not quiet:
                sys.stdout.write(f"Removed stale plugin directory {legacy_dir}\n")
        except Exception as e:
            _stderr(f"Could not remove stale plugin dir {legacy_dir}: {e}", quiet=False)

    if not plugin_dir.exists():
        if not quiet:
            sys.stdout.write(f"{PLUGIN_NAME} plugin not installed at {plugin_dir}\n")
        return 0

    managed = {"__init__.py", "cli.py", "plugin.yaml"}
    removed: list[str] = []
    for name in managed:
        target = plugin_dir / name
        if target.exists():
            target.unlink()
            removed.append(name)

    # Remove __pycache__ if present — it's a build artefact we own.
    pycache = plugin_dir / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache, ignore_errors=True)

    remaining = [p.name for p in plugin_dir.iterdir()]
    if remaining:
        if not quiet:
            sys.stdout.write(
                f"Removed {', '.join(removed) or 'no managed files'} from {plugin_dir}\n"
                f"Directory not deleted — contains unmanaged files: {remaining}\n"
            )
        return 0

    plugin_dir.rmdir()
    if not quiet:
        sys.stdout.write(f"Uninstalled {PLUGIN_NAME} plugin from {plugin_dir}\n")
    return 0


# ---------------------------------------------------------------------------
# Init / reset commands (Python equivalents of memory_init.sh / memory_reset.sh)
# ---------------------------------------------------------------------------


def _parse_memory_md(path: str) -> list[dict]:
    """Parse MEMORY.md into seeding entries.  Best-effort heuristic classification."""
    entries = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"^§\s*$", content, flags=re.MULTILINE)
    for raw_section in sections:
        section = raw_section.strip()
        if not section:
            continue
        if section.startswith("*This is the persistent") or section.startswith("---"):
            continue
        if section.startswith("User:") and "UK English" in section:
            continue
        for para in (p.strip() for p in section.split("\n\n")):
            if len(para) < 20:
                continue
            tl = para.lower()
            if any(k in tl for k in ("prefer", "always", "never", "formatting")):
                category = "preference"
            elif any(k in tl for k in ("decision", "chosen", "selected")):
                category = "decision"
            elif any(k in tl for k in ("deadline", "due", "date")):
                category = "fact"
            elif any(k in tl for k in ("project", "active", "cron", "system")):
                category = "other"
            else:
                category = "fact"
            entries.append({
                "text": para,
                "category": category,
                "scope": "global",
                "importance": 0.7,
            })
    return entries


def _cmd_init(args: argparse.Namespace) -> int:
    """Initialise the memory store and optionally seed from MEMORY.md.

    Equivalent to ``scripts/memory_init.sh`` but runs in-process.
    """
    db_path = getattr(args, "path", None) or DEFAULT_DB_PATH
    memory_md = getattr(args, "memory_md", None) or os.path.expanduser(
        os.environ.get("MEMORY_MD", "~/.hermes/memory/MEMORY.md")
    )
    quiet = bool(getattr(args, "quiet", False))

    if not getattr(args, "yes", False):
        print(f"This will initialise the memory store at: {db_path}")
        answer = input('Type "yes" to proceed: ').strip().lower()
        if answer != "yes":
            print("Aborted.")
            return 1

    if not quiet:
        print("=== LanceDB Memory Initialisation ===")
        print(f"DB Path:  {db_path}")
        print(f"Memory:   {memory_md}")

    store = MemoryStore(db_path=db_path)
    try:
        store._initialise()
        store.stats()
        if not quiet:
            print("  ✓ Connection established")
    except Exception as e:
        if not quiet:
            print(f"  ⚠ DB corrupted ({e}) — auto-recovering...")
        shutil.rmtree(db_path, ignore_errors=True)
        store = MemoryStore(db_path=db_path)
        store._initialise()
        if not quiet:
            print("  ✓ Recovered (fresh DB created)")

    existing = store.list_memories(limit=1)
    if existing:
        total = store.stats().get("total_memories", 0)
        if not quiet:
            print(f"  ⚠ Database already has {total} entries")
            print("  Skipping seed. Run `hermes-memory-lancedb-pro reset` to start fresh.")
        return 0

    if not os.path.exists(memory_md):
        if not quiet:
            print(f"  ℹ No seed file at {memory_md}")
        return 0

    entries = _parse_memory_md(memory_md)
    if not entries:
        if not quiet:
            print("  ⚠ No entries found in MEMORY.md")
        return 0

    if not quiet:
        print(f"  Seeding {len(entries)} entries from MEMORY.md...")
    ids = store.store_many(entries)
    if not quiet:
        print(f"    {len(ids)}/{len(entries)} entries stored")

    if store.maybe_create_vector_index():
        if not quiet:
            print("  ✓ Vector index created")
    elif not quiet:
        print(f"  ℹ Vector index pending (need 256+ rows, have {len(ids)})")

    stats = store.stats()
    if not quiet:
        print(f"\n  ✓ Seeded {stats['total_memories']} memories")
        print(f"  Categories: {stats['categories']}")
        print("\n=== Memory system ready ===")
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    """Wipe the memory database and reinitialise from MEMORY.md.

    Equivalent to ``scripts/memory_reset.sh`` but runs in-process.
    """
    db_path = getattr(args, "path", None) or DEFAULT_DB_PATH
    quiet = bool(getattr(args, "quiet", False))

    if not getattr(args, "yes", False):
        print(f"This will WIPE ALL MEMORIES at: {db_path}")
        answer = input('Type "yes" to proceed: ').strip().lower()
        if answer != "yes":
            print("Aborted.")
            return 1

    if not quiet:
        print("=== LanceDB Memory Reset ===")

    if os.path.isdir(db_path):
        if not quiet:
            print(f"  Wiping: {db_path}")
        shutil.rmtree(db_path)
        if not quiet:
            print("  ✓ Database cleared")
    elif not quiet:
        print("  (No existing database found)")

    args.yes = True  # user already confirmed; don't prompt again in _cmd_init
    return _cmd_init(args)


# ---------------------------------------------------------------------------
# Task ledger subcommands
# ---------------------------------------------------------------------------


def _cmd_task_create(args: argparse.Namespace) -> int:
    """Create a new task ledger."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: --id is required", file=sys.stderr)
        return 1
    objective = getattr(args, "objective", None) or ""
    iterations = getattr(args, "iterations", None)
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    quiet = bool(getattr(args, "quiet", False))
    try:
        state = _tl.create_task(task_id, objective, target_iterations=iterations, root=root)
        if not quiet:
            state_path = _tl._state_path(task_id, root)
            print(f"Created task {task_id!r}")
            print(f"  State: {state_path}")
            print(f"  Objective: {state['objective']}")
            if state["target_iterations"] is not None:
                print(f"  Target iterations: {state['target_iterations']}")
        return 0
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_task_list(args: argparse.Namespace) -> int:
    """List all task ledgers."""
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    tasks = _tl.list_tasks(root)
    if not tasks:
        print("No tasks found.")
        return 0
    for t in tasks:
        status = t.get("status", "?")
        current = t.get("current_iteration", 0)
        target = t.get("target_iterations")
        iter_str = f"{current}/{target}" if target is not None else str(current)
        held = "  [held]" if t.get("gc_hold") else ""
        print(f"  {t['task_id']:<40} [{status}]  iter {iter_str}{held}")
    return 0


def _cmd_task_show(args: argparse.Namespace) -> int:
    """Show detailed task state."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    try:
        state = _tl.load_state(task_id, root)
        print(json.dumps(state, indent=2))
        return 0
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1


def _cmd_task_resume(args: argparse.Namespace) -> int:
    """Print the ACTIVE TASK CONTROL BLOCK for resuming a task."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    try:
        state = _tl.load_state(task_id, root)
        print(_tl.build_control_block(state))
        results_path = _tl._task_dir(task_id, root) / "results.jsonl"
        if results_path.exists():
            lines = results_path.read_text(encoding="utf-8").splitlines()
            passed = sum(1 for ln in lines if json.loads(ln).get("result") == "pass")
            failed = len(lines) - passed
            print(f"Results so far: {len(lines)} (pass={passed}, fail={failed})")
        return 0
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1


def _cmd_task_complete(args: argparse.Namespace) -> int:
    """Mark a task as complete."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    summary = getattr(args, "summary", "") or ""
    quiet = bool(getattr(args, "quiet", False))
    try:
        _tl.complete_task(task_id, summary=summary, root=root)
        if not quiet:
            print(f"Task {task_id!r} marked complete.")
        return 0
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1


def _cmd_task_pin(args: argparse.Namespace) -> int:
    """Pin the active task control block as a memory so it is always recalled.

    Stores the current control block text as a memory with category='active_task'.
    Active-task memories are always prepended to the recall block regardless of
    relevance score, so the task state survives context compaction.
    """
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    quiet = bool(getattr(args, "quiet", False))
    try:
        state = _tl.load_state(task_id, root)
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1

    control_block = _tl.build_control_block(state)
    state_path = str(_tl._state_path(task_id, root))
    store = _open_store(args)
    meta = {"task_id": task_id, "state_path": state_path, "priority": "must_include"}
    mem_id = store.store(
        text=control_block,
        category="active_task",
        scope="global",
        importance=1.0,
        metadata_extra=meta,
    )
    if not quiet:
        print(f"Pinned task {task_id!r} as memory {mem_id}")
        print("  Category: active_task (always recalled first)")
    return 0


def _cmd_task_advance(args: argparse.Namespace) -> int:
    """Record one completed iteration and advance the task counter.

    Increments ``current_iteration``, appends a result to ``results.jsonl``,
    and updates ``next_action`` in ``state.json``.  Because the memory plugin
    reloads ``state.json`` on every recall, the model will see the updated
    iteration count and next_action on the very next turn — no re-pin needed.
    """
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    result = getattr(args, "result", "pass") or "pass"
    next_action = getattr(args, "next_action", None) or None
    summary = getattr(args, "summary", None) or None
    quiet = bool(getattr(args, "quiet", False))
    try:
        state = _tl.advance_iteration(
            task_id,
            result=result,
            next_action=next_action,
            summary=summary,
            root=root,
        )
        if not quiet:
            current = state["current_iteration"]
            target = state.get("target_iterations")
            iter_str = f"{current}/{target}" if target is not None else str(current)
            print(f"Task {task_id!r}: iteration advanced to {iter_str}")
            print(f"  Next action: {state['next_action']}")
        return 0
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1


def _cmd_task_gc(args: argparse.Namespace) -> int:
    """Garbage-collect completed task ledgers past the retention window."""
    from . import task_gc as _tgc

    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    dry_run = bool(getattr(args, "dry_run", False))
    quiet = bool(getattr(args, "quiet", False))

    def _arg_or_env_int(val: int | None, env_name: str, default: int) -> int:
        if val is not None:
            return val
        raw = os.environ.get(env_name, "").strip()
        try:
            return int(raw) if raw else default
        except ValueError:
            return default

    cfg = _tgc.TaskGCConfig(
        dry_run=dry_run,
        mode=(getattr(args, "mode", None)
              or os.environ.get("MEMORY_TASK_GC_MODE", "archive").strip().lower()
              or "archive"),
        retention_days=_arg_or_env_int(
            getattr(args, "retention_days", None), "MEMORY_TASK_RETENTION_DAYS", 30),
        archive_grace_days=_arg_or_env_int(
            getattr(args, "archive_grace_days", None),
            "MEMORY_TASK_ARCHIVE_GRACE_DAYS", 90),
    )

    result = _tgc.run_task_gc(root=root, config=cfg)

    if not quiet:
        hdr = f"Task GC — {cfg.mode} mode"
        if dry_run:
            hdr += " — DRY RUN (no changes made)"
        print(hdr)
        print(f"  Scanned:           {result.scanned}")
        if result.archived:
            print(f"  Archived ({len(result.archived)}): {', '.join(result.archived)}")
        if result.deleted:
            print(f"  Deleted ({len(result.deleted)}): {', '.join(result.deleted)}")
        if result.purged_archive:
            print(f"  Purged from archive ({len(result.purged_archive)}): "
                  f"{', '.join(result.purged_archive)}")
        if result.held:
            print(f"  Held, skipped ({len(result.held)}): {', '.join(result.held)}")
        if result.abandoned_running:
            print(f"  Abandoned running, left in place "
                  f"({len(result.abandoned_running)}): "
                  f"{', '.join(result.abandoned_running)}")
        print(f"  Skipped — still running: {result.skipped_running}")
        print(f"  Skipped — too recent:    {result.skipped_recent}")
        if result.skipped_unparseable:
            print(f"  Skipped — unreadable:    {result.skipped_unparseable}")
        for err in result.errors:
            print(f"  error: {err}", file=sys.stderr)

    gc_ids = [*result.archived, *result.deleted]
    if gc_ids and not dry_run:
        try:
            from .provider import _soft_delete_task_pin
            store = _open_store(args)
            for tid in gc_ids:
                _soft_delete_task_pin(store, tid)
            if not quiet:
                print(f"  Cleaned up active_task pins for {len(gc_ids)} task(s).")
        except Exception as e:
            print(f"  warning: pin cleanup skipped ({e})", file=sys.stderr)
    return 0


def _cmd_task_hold(args: argparse.Namespace) -> int:
    """Set or clear the GC-hold flag on a task (hold / unhold)."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("error: task_id is required", file=sys.stderr)
        return 1
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    quiet = bool(getattr(args, "quiet", False))
    hold = getattr(args, "task_command", "hold") != "unhold"
    try:
        _tl.set_task_hold(task_id, hold, root=root)
        if not quiet:
            if hold:
                print(f"Task {task_id!r} held — exempt from garbage collection.")
            else:
                print(f"Task {task_id!r} released — eligible for garbage collection.")
        return 0
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1


def _cmd_task_to_skill(args: argparse.Namespace) -> int:
    """Scaffold a draft skill from a task, or list candidate tasks."""
    from . import task_skill as _tsk

    task_id = getattr(args, "task_id", None)
    root_str = getattr(args, "task_root", None)
    root = Path(root_str).expanduser() if root_str else None
    quiet = bool(getattr(args, "quiet", False))
    search = getattr(args, "search", None)
    list_mode = bool(getattr(args, "list", False)) or search is not None or not task_id

    if list_mode:
        candidates = _tsk.list_skill_candidates(root, search=search)
        if not candidates:
            if search:
                print(f"No completed tasks match {search!r}.")
            else:
                print("No completed tasks available to turn into a skill.")
            return 0
        if not quiet:
            suffix = f" matching {search!r}" if search else ""
            print(f"Completed tasks that could become a skill{suffix}:")
        for i, c in enumerate(candidates, 1):
            loc = "" if c["location"] == "live" else "  (archived)"
            held = "  [held]" if c["held"] else ""
            when = (c["completed_at"] or "")[:10]
            objective = c["objective"] or "(no objective)"
            print(f"  {i}. {c['task_id']}  —  {objective}  ({when}){loc}{held}")
        return 0

    out_str = getattr(args, "out", None)
    out_dir = Path(out_str).expanduser() if out_str else None
    force = bool(getattr(args, "force", False))
    try:
        dest = _tsk.scaffold_skill_from_task(
            task_id, root=root, out_dir=out_dir, force=force
        )
    except FileNotFoundError:
        print(f"error: task {task_id!r} not found", file=sys.stderr)
        return 1
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"Scaffolded a draft skill from task {task_id!r}:")
        print(f"  {dest / 'SKILL.md'}")
        print(f"  {dest / 'AGENTS.md'}")
        print("  DRAFT — review and rewrite the Protocol section before use.")
    return 0


def _cmd_task_dispatch(args: argparse.Namespace) -> int:
    """Dispatch task sub-subcommands."""
    sub = getattr(args, "task_command", None)
    if sub is None:
        print(
            "Usage: ... task <create|list|show|resume|advance|complete|pin|"
            "gc|hold|unhold|to-skill>"
        )
        return 0
    return {
        "create": _cmd_task_create,
        "list": _cmd_task_list,
        "show": _cmd_task_show,
        "resume": _cmd_task_resume,
        "advance": _cmd_task_advance,
        "complete": _cmd_task_complete,
        "pin": _cmd_task_pin,
        "gc": _cmd_task_gc,
        "hold": _cmd_task_hold,
        "unhold": _cmd_task_hold,
        "to-skill": _cmd_task_to_skill,
    }.get(sub, lambda _: 0)(args)


def _add_task_subparsers(parent: argparse.ArgumentParser, dest: str = "task_command") -> None:
    """Wire task sub-subcommands onto *parent*. Shared by main() and register_cli()."""
    tsubs = parent.add_subparsers(dest=dest)

    p_create = tsubs.add_parser("create", help="Create a new task ledger")
    p_create.add_argument("--id", dest="task_id", required=True, metavar="TASK_ID",
                          help="Unique task identifier")
    p_create.add_argument("--objective", default="", metavar="TEXT",
                          help="One-line task objective")
    p_create.add_argument("--iterations", type=int, default=None, metavar="N",
                          help="Target iteration count (optional)")
    p_create.add_argument("--task-root", default=None, metavar="PATH",
                          help="Task root dir (default: ~/.hermes/workspace/tasks)")
    p_create.add_argument("-q", "--quiet", action="store_true")

    tsubs.add_parser("list", help="List all task ledgers").add_argument(
        "--task-root", default=None, metavar="PATH"
    )

    p_show = tsubs.add_parser("show", help="Show task state as JSON")
    p_show.add_argument("task_id", metavar="TASK_ID")
    p_show.add_argument("--task-root", default=None, metavar="PATH")

    p_resume = tsubs.add_parser("resume", help="Print the task control block for re-orienting")
    p_resume.add_argument("task_id", metavar="TASK_ID")
    p_resume.add_argument("--task-root", default=None, metavar="PATH")

    p_advance = tsubs.add_parser(
        "advance",
        help="Record a completed iteration and increment the counter",
    )
    p_advance.add_argument("task_id", metavar="TASK_ID")
    p_advance.add_argument("--result", default="pass", choices=["pass", "fail"],
                           help="Iteration result (default: pass)")
    p_advance.add_argument("--next-action", dest="next_action", default=None, metavar="TEXT",
                           help="Override the auto-generated next_action string")
    p_advance.add_argument("--summary", default=None, metavar="TEXT",
                           help="Short summary of what happened in this iteration")
    p_advance.add_argument("--task-root", default=None, metavar="PATH")
    p_advance.add_argument("-q", "--quiet", action="store_true")

    p_complete = tsubs.add_parser("complete", help="Mark a task as complete")
    p_complete.add_argument("task_id", metavar="TASK_ID")
    p_complete.add_argument("--summary", default="", metavar="TEXT",
                            help="Short completion summary")
    p_complete.add_argument("--task-root", default=None, metavar="PATH")
    p_complete.add_argument("-q", "--quiet", action="store_true")

    p_pin = tsubs.add_parser(
        "pin",
        help="Store task control block as an always-recalled active_task memory",
    )
    p_pin.add_argument("task_id", metavar="TASK_ID")
    p_pin.add_argument("--path", default=None, metavar="PATH",
                       help="Memory DB dir (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_pin.add_argument("--task-root", default=None, metavar="PATH")
    p_pin.add_argument("-q", "--quiet", action="store_true")

    p_gc = tsubs.add_parser(
        "gc", help="Garbage-collect completed task ledgers past the retention window"
    )
    p_gc.add_argument("--dry-run", action="store_true",
                      help="Report what would be collected without changing anything")
    p_gc.add_argument("--mode", default=None, choices=["archive", "delete"],
                      help="archive (default) keeps an audit trail; delete removes outright")
    p_gc.add_argument("--retention-days", dest="retention_days", type=int, default=None,
                      metavar="N", help="Completed tasks older than N days are collected")
    p_gc.add_argument("--archive-grace-days", dest="archive_grace_days", type=int,
                      default=None, metavar="N",
                      help="Hard-delete archived dirs older than retention+N; 0 disables")
    p_gc.add_argument("--task-root", default=None, metavar="PATH")
    p_gc.add_argument("--path", default=None, metavar="PATH",
                      help="Memory DB dir, for active_task pin cleanup")
    p_gc.add_argument("-q", "--quiet", action="store_true")

    p_hold = tsubs.add_parser("hold", help="Exempt a task from garbage collection")
    p_hold.add_argument("task_id", metavar="TASK_ID")
    p_hold.add_argument("--task-root", default=None, metavar="PATH")
    p_hold.add_argument("-q", "--quiet", action="store_true")

    p_unhold = tsubs.add_parser("unhold", help="Release a GC hold on a task")
    p_unhold.add_argument("task_id", metavar="TASK_ID")
    p_unhold.add_argument("--task-root", default=None, metavar="PATH")
    p_unhold.add_argument("-q", "--quiet", action="store_true")

    p_skill = tsubs.add_parser(
        "to-skill",
        help="Scaffold a reusable skill from a task, or list candidate tasks",
    )
    p_skill.add_argument("task_id", metavar="TASK_ID", nargs="?", default=None,
                         help="Task to scaffold; omit to list candidate tasks")
    p_skill.add_argument("--list", action="store_true",
                         help="List completed tasks (live + archived) as candidates")
    p_skill.add_argument("--search", default=None, metavar="KEYWORDS",
                         help="List only candidate tasks matching these keywords")
    p_skill.add_argument("--out", default=None, metavar="DIR",
                         help="Output dir (default: ~/.hermes/skills/<task-id>/)")
    p_skill.add_argument("--force", action="store_true",
                         help="Overwrite an existing non-empty output dir")
    p_skill.add_argument("--task-root", default=None, metavar="PATH")
    p_skill.add_argument("-q", "--quiet", action="store_true")


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the ``hermes-memory-lancedb-pro`` CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes-memory-lancedb-pro",
        description="Hermes LanceDB memory CLI — manage the lancedb_pro plugin and memory store.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")

    # ---- init ----
    p_init = subparsers.add_parser(
        "init",
        help="Initialise the memory store (seed from MEMORY.md if empty)",
        description=(
            "Open or create the memory database and optionally seed entries from "
            "MEMORY.md when the store is empty."
        ),
    )
    p_init.add_argument("--path", default=None, metavar="PATH",
                        help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_init.add_argument("--memory-md", dest="memory_md", default=None, metavar="PATH",
                        help="Seed file (default: $MEMORY_MD or ~/.hermes/memory/MEMORY.md)")
    p_init.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    p_init.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress non-essential output")

    # ---- reset ----
    p_reset = subparsers.add_parser(
        "reset",
        help="Wipe the memory database and reinitialise from MEMORY.md",
        description=(
            "Delete the existing LanceDB database directory and run init, "
            "seeding fresh entries from MEMORY.md."
        ),
    )
    p_reset.add_argument("--path", default=None, metavar="PATH",
                         help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_reset.add_argument("--memory-md", dest="memory_md", default=None, metavar="PATH",
                         help="Seed file (default: $MEMORY_MD or ~/.hermes/memory/MEMORY.md)")
    p_reset.add_argument("-y", "--yes", action="store_true",
                         help="Skip confirmation prompt")
    p_reset.add_argument("-q", "--quiet", action="store_true",
                         help="Suppress non-essential output")

    # ---- task ----
    p_task = subparsers.add_parser(
        "task",
        help="Manage durable task ledgers for long-running agent work",
        description=(
            "Task ledgers keep objective, iteration counter, and next_action in "
            "state.json outside the LLM context window so context compaction "
            "cannot lose progress."
        ),
    )
    _add_task_subparsers(p_task, dest="task_command")
    p_task.set_defaults(func=_cmd_task_dispatch)

    # ---- doctor ----
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Print a diagnostic report for the memory store",
        description="Scan the store and report counts, anomalies, and recommendations.",
    )
    p_doctor.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_doctor.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    # ---- export ----
    p_export = subparsers.add_parser(
        "export",
        help="Export memories to JSONL",
        description="Stream memory rows to JSONL (one JSON object per line).",
    )
    p_export.add_argument("--out", "-o", default="-", metavar="PATH",
                          help="Output file path (default: stdout)")
    p_export.add_argument("--include-archived", action="store_true",
                          help="Include archived rows (excluded by default)")
    p_export.add_argument("--limit", type=int, default=100_000, metavar="N",
                          help="Maximum rows to export (default: 100000)")
    p_export.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_export.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    # ---- import ----
    p_import = subparsers.add_parser(
        "import",
        help="Import memories from JSONL",
        description="Read a JSONL file produced by 'export' and write rows into the store.",
    )
    p_import.add_argument("--in", dest="input", default="-", metavar="PATH",
                          help="Input file path (default: stdin)")
    p_import.add_argument("--reembed", action="store_true",
                          help="Re-encode text with the current embedder instead of stored vectors")
    p_import.add_argument("--allow-existing", action="store_true",
                          help="Skip rows whose id already exists rather than aborting")
    p_import.add_argument("--path", default=None, metavar="PATH",
                          help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)")
    p_import.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress non-essential output")

    # ---- install-plugin ----
    p_install = subparsers.add_parser(
        "install-plugin",
        help="Install the Hermes plugin shim",
        description=(
            "Create <hermes_home>/plugins/lancedb_pro/ with the discovery "
            "__init__.py, cli.py, and a copy of plugin.yaml so hermes-agent can "
            "find this provider. Then set memory.provider: lancedb_pro in "
            "config.yaml."
        ),
    )
    p_install.add_argument(
        "--hermes-home",
        default=None,
        metavar="PATH",
        help="Hermes home dir (default: $HERMES_HOME or ~/.hermes)",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing installation",
    )
    p_install.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                           help=argparse.SUPPRESS)

    # ---- uninstall-plugin ----
    p_uninstall = subparsers.add_parser(
        "uninstall-plugin",
        help="Remove the Hermes plugin shim",
        description=(
            "Remove <hermes_home>/plugins/lancedb_pro/. Only files this "
            "command installed (__init__.py, cli.py, plugin.yaml) are removed; "
            "the dir is left in place if it contains anything else."
        ),
    )
    p_uninstall.add_argument(
        "--hermes-home",
        default=None,
        metavar="PATH",
        help="Hermes home dir (default: $HERMES_HOME or ~/.hermes)",
    )
    p_uninstall.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                             help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        return 0

    dispatch = {
        "task": _cmd_task_dispatch,
        "init": _cmd_init,
        "reset": _cmd_reset,
        "doctor": _cmd_doctor,
        "export": _cmd_export,
        "import": _cmd_import,
        "install-plugin": _cmd_install_plugin,
        "uninstall-plugin": _cmd_uninstall_plugin,
    }
    return dispatch[args.subcommand](args)


# ---------------------------------------------------------------------------
# Smoke-test entry point (kept for backwards compat)
# ---------------------------------------------------------------------------


def smoke_main() -> int:
    """Run the same end-to-end smoke test as ``scripts/memory_smoke_test.py``,
    but driven from the installed package so users don't have to clone the
    source. Use ``--ephemeral`` to point at a tmp dir that is wiped on exit."""
    parser = argparse.ArgumentParser(prog="hermes-memory-smoke")
    parser.add_argument("--path", help="Custom DB directory")
    parser.add_argument("--ephemeral", action="store_true", help="Use a tmp dir")
    args = parser.parse_args()

    tmpdir = None
    db_path = args.path
    if args.ephemeral:
        tmpdir = tempfile.mkdtemp(prefix="hermes-smoke-")
        db_path = tmpdir

    try:
        return _run(db_path)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run(db_path) -> int:
    store = MemoryStore(db_path=db_path) if db_path else MemoryStore()
    store._initialise()

    passed = failed = 0

    def ok(msg: str) -> None:
        nonlocal passed
        passed += 1
        print(f"  PASS: {msg}")

    def fail(msg: str) -> None:
        nonlocal failed
        failed += 1
        print(f"  FAIL: {msg}")

    # Cleanup
    for mode in ("vector", "bm25"):
        try:
            existing = store.search(SMOKE_PREFIX, limit=50, mode=mode)
        except Exception:
            existing = []
        for r in existing:
            if _entry_text(r).startswith(SMOKE_PREFIX):
                store.forget(mem_id=_entry_id(r))

    print("=== TEST: Store ===")
    mem_id = store.store(
        text=f"{SMOKE_PREFIX}PRIMARY: end-to-end check.",
        category="fact",
        scope="global",
        importance=0.3,
    )
    if mem_id and len(mem_id) == 36:
        ok(f"store() → {mem_id}")
    else:
        fail(f"store() returned bad id: {mem_id}")
        return 1

    for mode in ("vector", "bm25", "hybrid"):
        results = store.search(f"{SMOKE_PREFIX}PRIMARY", limit=5, mode=mode)
        if any(_entry_text(r).startswith(SMOKE_PREFIX) for r in results):
            ok(f"search(mode={mode!r}) found entry")
        else:
            fail(f"search(mode={mode!r}) missed entry ({len(results)} results)")

    if store.update(mem_id, text=f"{SMOKE_PREFIX}UPDATED: superseded.", tier="core"):
        ok("update() supersede succeeded")
    else:
        fail("update() returned False for an existing id")

    if not store.has_id(mem_id):
        ok("has_id() correctly excludes archived original")
    else:
        fail("has_id() returned True for archived id")

    # Cleanup
    for r in store.search(SMOKE_PREFIX, limit=50, mode="vector"):
        store.forget(mem_id=_entry_id(r))

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
