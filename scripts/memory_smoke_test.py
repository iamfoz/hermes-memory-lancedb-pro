#!/usr/bin/env python
"""End-to-end smoke test for hermes-memory-lancedb-pro.

Runs against a (possibly seeded) database and exercises:
  store · vector search · BM25 search · hybrid search · update · delete ·
  stats · list · has_id · check_ids · purge_archived

Usage:
    python scripts/memory_smoke_test.py             # default store
    python scripts/memory_smoke_test.py --path /tmp/foo  # custom DB
    python scripts/memory_smoke_test.py --ephemeral      # tmp dir, auto-clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

# Add src directory to path so we can import the package without installing
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

from hermes_memory_lancedb_pro.store import MemoryStore  # noqa: E402

SMOKE_PREFIX = "SMOKE_TEST_"


class TestRunner:
    """Tracks pass/fail counts. Replaces the previous global-shadowing
    scheme that caused the summary to always print 0/0."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  PASS: {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        print(f"  FAIL: {msg}")

    @property
    def total(self) -> int:
        return self.passed + self.failed


def _entry_text(result) -> str:
    """The new public API returns dicts uniformly. Tuple-unpacking is kept
    only as a defensive fallback for old callers."""
    if isinstance(result, tuple):
        result = result[0]
    return result.get("text", "") if isinstance(result, dict) else ""


def _entry_id(result) -> str:
    if isinstance(result, tuple):
        result = result[0]
    return result.get("id", "") if isinstance(result, dict) else ""


def get_store(db_path=None) -> MemoryStore:
    if db_path:
        os.makedirs(db_path, exist_ok=True)
        store = MemoryStore(db_path=db_path)
    else:
        store = MemoryStore()
    store._initialise()
    return store


def cleanup_smoke_entries(store: MemoryStore) -> None:
    """Remove any leftover smoke-test entries from prior runs."""
    for mode in ("vector", "bm25"):
        try:
            existing = store.search(SMOKE_PREFIX, limit=50, mode=mode)
        except Exception:
            existing = []
        for r in existing:
            text = _entry_text(r)
            if text.startswith(SMOKE_PREFIX):
                store.forget(mem_id=_entry_id(r))


def main() -> int:
    parser = argparse.ArgumentParser(description="hermes-memory-lancedb-pro smoke test")
    parser.add_argument("--path", help="Custom DB directory")
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Use a fresh tmp dir and remove it on exit",
    )
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
    runner = TestRunner()
    store = get_store(db_path)

    print("=== Setup: cleaning previous smoke tests ===")
    cleanup_smoke_entries(store)

    # 1. Store
    print("\n=== TEST 1: Store ===")
    mem_id = store.store(
        text=f"{SMOKE_PREFIX}PRIMARY: LanceDB memory system end-to-end verification.",
        category="fact", scope="global", importance=0.3,
    )
    if mem_id and len(mem_id) == 36:
        runner.ok(f"Stored with UUID: {mem_id}")
    else:
        runner.fail(f"Invalid ID: {mem_id}")
        return 1

    # 1b. Bulk store
    print("\n=== TEST 1b: store_many ===")
    bulk_ids = store.store_many([
        {"text": f"{SMOKE_PREFIX}BULK_1: bulk write check", "importance": 0.4},
        {"text": f"{SMOKE_PREFIX}BULK_2: bulk write check", "importance": 0.4},
    ])
    if len(bulk_ids) == 2:
        runner.ok(f"Bulk-stored {len(bulk_ids)} entries")
    else:
        runner.fail(f"Bulk store returned {len(bulk_ids)} ids")

    # 2. Vector search
    print("\n=== TEST 2: Search (vector) ===")
    results = store.search("LanceDB memory system end-to-end", limit=5, mode="vector")
    if any(_entry_text(r).startswith(SMOKE_PREFIX) for r in results):
        runner.ok("Vector search found test entry")
    else:
        runner.fail(f"Vector search missed test entry, got {len(results)} results")

    # 3. BM25 search
    print("\n=== TEST 3: Search (BM25) ===")
    results = store.search(f"{SMOKE_PREFIX}PRIMARY", limit=10, mode="bm25")
    if any(_entry_text(r).startswith(SMOKE_PREFIX) for r in results):
        runner.ok("BM25 search found test entry")
    else:
        runner.fail(f"BM25 search missed test entry, got {len(results)} results")

    # 4. Hybrid search
    print("\n=== TEST 4: Search (hybrid) ===")
    results = store.search(f"{SMOKE_PREFIX}PRIMARY verification", limit=5, mode="hybrid")
    if any(_entry_text(r).startswith(SMOKE_PREFIX) for r in results):
        runner.ok("Hybrid search found test entry")
    else:
        runner.fail("Hybrid search missed test entry")
    if all(isinstance(r, dict) for r in results):
        runner.ok("Hybrid search returns dicts (consistent with vector/bm25)")
    else:
        runner.fail("Hybrid search returned non-dict items — API inconsistency")

    # 5. Update (supersede)
    print("\n=== TEST 5: Update ===")
    try:
        store.update(
            mem_id=mem_id,
            text=f"{SMOKE_PREFIX}UPDATED: write+search+update chain confirmed.",
            tier="core",
        )
        verify = store.search(f"{SMOKE_PREFIX}UPDATED", limit=1, mode="vector")
        if verify and "UPDATED" in _entry_text(verify[0]):
            runner.ok("Update persisted correctly (supersede pattern)")
        else:
            runner.fail("Update did not persist")
    except Exception as e:
        runner.fail(f"Update raised: {e}")

    # 5b. has_id should now report False for the original (archived) id
    print("\n=== TEST 5b: has_id on archived id ===")
    if not store.has_id(mem_id):
        runner.ok("has_id() correctly excludes archived id")
    else:
        runner.fail("has_id() returned True for an archived id")

    # 6. Delete
    print("\n=== TEST 6: Delete ===")
    try:
        all_matches = store.search(SMOKE_PREFIX, limit=20, mode="vector")
        deleted_ids = set()
        for r in all_matches:
            text = _entry_text(r)
            if text.startswith(SMOKE_PREFIX):
                rid = _entry_id(r)
                store.forget(mem_id=rid)
                deleted_ids.add(rid)
        # Forget bulk ids that may not have surfaced in vector search
        for bid in bulk_ids:
            if bid not in deleted_ids:
                store.forget(mem_id=bid)
                deleted_ids.add(bid)

        remaining = store.search(SMOKE_PREFIX, limit=20, mode="vector")
        smoke_left = [r for r in remaining if _entry_text(r).startswith(SMOKE_PREFIX)]
        if not smoke_left:
            runner.ok(f"Delete confirmed — {len(deleted_ids)} entries removed")
        else:
            runner.fail(f"Delete failed — {len(smoke_left)} entries still visible")
    except Exception as e:
        runner.fail(f"Delete raised: {e}")

    # 7. Stats
    print("\n=== TEST 7: Stats ===")
    stats = store.stats()
    if "total_memories" in stats and "active_memories" in stats:
        runner.ok(
            f"Stats: total={stats['total_memories']} "
            f"active={stats['active_memories']} "
            f"archived={stats['archived_memories']} "
            f"categories={stats['categories']}"
        )
    else:
        runner.fail(f"Stats missing required keys: {stats}")

    # 8. List memories
    print("\n=== TEST 8: List memories ===")
    mems = store.list_memories(limit=5)
    if isinstance(mems, list) and len(mems) <= 5:
        runner.ok(f"List returned {len(mems)} entries (asked for 5)")
    else:
        runner.fail(f"List returned an unexpected shape: {type(mems).__name__}")

    # 9. check_ids batch
    print("\n=== TEST 9: check_ids batch ===")
    sample = store.list_memories(limit=3)
    sample_ids = [m["id"] for m in sample]
    confirmed = store.check_ids(sample_ids + ["00000000-nope-nope-nope-000000000000"])
    if all(s in confirmed for s in sample_ids):
        runner.ok(f"check_ids returned {len(confirmed)} active ids")
    else:
        runner.fail("check_ids missed an active id")

    # Summary
    print(f"\n{'=' * 40}")
    print(f"Results: {runner.passed}/{runner.total} passed, {runner.failed}/{runner.total} failed")
    if runner.failed == 0:
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED — see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
