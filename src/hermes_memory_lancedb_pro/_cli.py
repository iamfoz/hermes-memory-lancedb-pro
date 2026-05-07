"""Console-script entry points for the installed package."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile

from .store import MemoryStore

SMOKE_PREFIX = "SMOKE_TEST_"


def _entry_text(r) -> str:
    if isinstance(r, tuple):
        r = r[0]
    return r.get("text", "") if isinstance(r, dict) else ""


def _entry_id(r) -> str:
    if isinstance(r, tuple):
        r = r[0]
    return r.get("id", "") if isinstance(r, dict) else ""


def smoke_main() -> int:
    """Run the same end-to-end smoke test as `scripts/memory_smoke_test.py`,
    but driven from the installed package so users don't have to clone the
    source. Use --ephemeral to point at a tmp dir that is wiped on exit."""
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
    sys.exit(smoke_main())
