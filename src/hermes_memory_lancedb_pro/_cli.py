"""Console-script entry points for the installed package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ._sql import parse_metadata as _parse_metadata
from .store import (
    DEFAULT_DB_PATH,
    MAX_SCAN_ROWS,
    VECTOR_DIM,
    MemoryStore,
)

PLUGIN_NAME = "lancedb_pro"
PLUGIN_SHIM_CONTENT = '''\
"""Hermes plugin discovery shim for hermes-memory-lancedb-pro.

The heavy package (lancedb, sentence-transformers, ...) must be installed
into Hermes' own Python environment with `hermes-pip install
hermes-memory-lancedb-pro`; this shim only re-exports `register` so
hermes-agent's plugin loader can discover it. If the import below fails,
the package landed in the wrong environment — reinstall with hermes-pip.

Regenerate with: hermes-memory install-plugin
"""
from hermes_memory_lancedb_pro.provider import register

__all__ = ["register"]
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
# install-plugin / uninstall-plugin
# ---------------------------------------------------------------------------


def _resolve_hermes_home(explicit: str | None) -> Path:
    """Pick the hermes profile dir: explicit arg > $HERMES_HOME > ~/.hermes."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".hermes"


def _packaged_plugin_yaml() -> Path:
    """Return the path to the plugin.yaml shipped inside the installed wheel."""
    return Path(__file__).resolve().parent / "plugin.yaml"


def _cmd_install_plugin(args: argparse.Namespace) -> int:
    """Create ``<hermes_home>/plugins/lancedb_pro/`` with the discovery shim
    and a copy of plugin.yaml so hermes-agent can find this provider."""
    hermes_home = _resolve_hermes_home(getattr(args, "hermes_home", None))
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    init_path = plugin_dir / "__init__.py"
    yaml_target = plugin_dir / "plugin.yaml"
    yaml_source = _packaged_plugin_yaml()

    if not yaml_source.exists():
        _stderr(f"plugin.yaml missing from installed package at {yaml_source}", quiet=False)
        return 1

    existing_files = [p for p in (init_path, yaml_target) if p.exists()]
    force = bool(getattr(args, "force", False))
    if existing_files and not force:
        _stderr(
            f"Plugin already installed at {plugin_dir}\n"
            f"Pass --force to overwrite, or remove with:\n"
            f"    hermes-memory uninstall-plugin",
            quiet=False,
        )
        return 1

    plugin_dir.mkdir(parents=True, exist_ok=True)
    init_path.write_text(PLUGIN_SHIM_CONTENT, encoding="utf-8")
    shutil.copyfile(yaml_source, yaml_target)

    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        action = "Reinstalled" if existing_files else "Installed"
        sys.stdout.write(
            f"{action} {PLUGIN_NAME} plugin at {plugin_dir}\n"
            f"  - {init_path.name} (discovery shim)\n"
            f"  - {yaml_target.name} (manifest)\n"
            f"Next: configure with `hermes memory setup` or set the\n"
            f"MEMORY_EXTRACTION_* env vars manually. See README for details.\n"
        )
    return 0


def _cmd_uninstall_plugin(args: argparse.Namespace) -> int:
    """Remove ``<hermes_home>/plugins/lancedb_pro/``. Only deletes files we
    install (``__init__.py``, ``plugin.yaml``) and then the dir if empty —
    refuses to delete a dir containing unknown files."""
    hermes_home = _resolve_hermes_home(getattr(args, "hermes_home", None))
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    quiet = bool(getattr(args, "quiet", False))

    if not plugin_dir.exists():
        if not quiet:
            sys.stdout.write(f"{PLUGIN_NAME} plugin not installed at {plugin_dir}\n")
        return 0

    managed = {"__init__.py", "plugin.yaml"}
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
# Top-level dispatcher
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the ``hermes-memory`` multi-command CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes-memory",
        description="Hermes memory admin CLI — export, import, and diagnose the LanceDB store.",
    )
    parser.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help="DB directory (default: $MEMORY_DB_DIR or ~/.hermes/memory-lancedb)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential stderr output",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")

    # ---- export ----
    p_export = subparsers.add_parser(
        "export",
        help="Export memories to JSONL",
        description="Stream memory rows to JSONL (one JSON object per line).",
    )
    p_export.add_argument(
        "--out",
        "-o",
        default="-",
        metavar="PATH",
        help="Output file path (default: stdout)",
    )
    p_export.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived rows (excluded by default)",
    )
    p_export.add_argument(
        "--limit",
        type=int,
        default=100_000,
        metavar="N",
        help="Maximum rows to export (default: 100000)",
    )
    p_export.add_argument("--path", default=argparse.SUPPRESS, metavar="PATH", help=argparse.SUPPRESS)
    p_export.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # ---- import ----
    p_import = subparsers.add_parser(
        "import",
        help="Import memories from JSONL",
        description="Read a JSONL file produced by 'export' and write rows into the store.",
    )
    p_import.add_argument(
        "--in",
        dest="input",
        default="-",
        metavar="PATH",
        help="Input file path (default: stdin)",
    )
    p_import.add_argument(
        "--reembed",
        action="store_true",
        help="Re-encode text with the current embedder instead of using stored vectors",
    )
    p_import.add_argument(
        "--allow-existing",
        action="store_true",
        help="Skip rows whose id already exists rather than aborting",
    )
    p_import.add_argument("--path", default=argparse.SUPPRESS, metavar="PATH", help=argparse.SUPPRESS)
    p_import.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # ---- doctor ----
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Print a diagnostic report",
        description="Scan the store and report counts, anomalies, and recommendations.",
    )
    p_doctor.add_argument("--path", default=argparse.SUPPRESS, metavar="PATH", help=argparse.SUPPRESS)
    p_doctor.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # ---- install-plugin ----
    p_install = subparsers.add_parser(
        "install-plugin",
        help="Install the Hermes discovery shim",
        description=(
            "Create <hermes_home>/plugins/lancedb_pro/ with the discovery "
            "__init__.py and a copy of plugin.yaml so hermes-agent can find "
            "this provider."
        ),
    )
    p_install.add_argument(
        "--hermes-home",
        default=None,
        metavar="PATH",
        help="Hermes profile dir (default: $HERMES_HOME or ~/.hermes)",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing shim",
    )
    p_install.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # ---- uninstall-plugin ----
    p_uninstall = subparsers.add_parser(
        "uninstall-plugin",
        help="Remove the Hermes discovery shim",
        description=(
            "Remove <hermes_home>/plugins/lancedb_pro/. Only files this "
            "command installed (__init__.py, plugin.yaml) are removed; the "
            "dir is left in place if it contains anything else."
        ),
    )
    p_uninstall.add_argument(
        "--hermes-home",
        default=None,
        metavar="PATH",
        help="Hermes profile dir (default: $HERMES_HOME or ~/.hermes)",
    )
    p_uninstall.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        return 0

    # Subparser --path / --quiet use argparse.SUPPRESS so they only land in
    # the namespace when the user explicitly provides them; absent that the
    # top-level value is preserved. Nothing further to merge here.

    dispatch = {
        "export": _cmd_export,
        "import": _cmd_import,
        "doctor": _cmd_doctor,
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
