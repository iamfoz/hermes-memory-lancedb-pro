#!/usr/bin/env python
"""End-to-end smoke test for hermes-memory-lancedb-pro.

Runs against the seeded database and exercises:
  store · vector search · BM25 search · hybrid search · update · delete · stats · list · has_id

Usage:
    python scripts/memory_smoke_test.py             # default store
    python scripts/memory_smoke_test.py --path /custom/db  # custom DB
"""
import sys, os, argparse

# Add src directory to path so we can import the package
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))
from hermes_memory_lancedb_pro.store import MemoryStore


def get_store(db_path=None):
    """Create and initialise a MemoryStore."""
    if db_path:
        os.makedirs(db_path, exist_ok=True)
        store = MemoryStore(db_path=db_path)
    else:
        store = MemoryStore()
    store._initialise()
    return store


def ok(msg):
    global passed
    passed += 1
    print(f"  PASS: {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  FAIL: {msg}")


def cleanup_smoke_entries(store):
    """Remove any leftover smoke test entries from prior runs."""
    existing = store.search("SMOKE_TEST_", limit=50, mode="vector")
    for r in existing:
        entry = r[0] if isinstance(r, tuple) else r
        if entry["text"].startswith("SMOKE_TEST_"):
            store.forget(mem_id=entry["id"])


# ── Main ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="hermes-memory-lancedb-pro smoke test")
    parser.add_argument("--path", help="Custom DB directory (default: ~/.hermes/memory-lancedb)")
    args = parser.parse_args()

    passed = 0
    failed = 0

    store = get_store(args.path)

    print("=== Setup: cleaning previous smoke tests ===")
    cleanup_smoke_entries(store)

    # ── TEST 1: Store ─────────────────────────────────
    print("\n=== TEST 1: Store ===")
    mem_id = store.store(
        text="SMOKE_TEST_PRIMARY: LanceDB memory system end-to-end verification.",
        category="fact", scope="global", importance=0.3,
    )
    if mem_id and len(mem_id) == 36:
        ok(f"Stored with UUID: {mem_id}")
    else:
        fail(f"Invalid ID: {mem_id}")
        sys.exit(1)

    # ── TEST 2: Search (vector) ───────────────────────
    print("\n=== TEST 2: Search (vector) ===")
    results = store.search("LanceDB memory system end-to-end", limit=5, mode="vector")
    found = any(r["text"].startswith("SMOKE_TEST_") for r in results)
    if found:
        ok("Vector search found test entry")
    else:
        fail(f"Vector search missed test entry, got {len(results)} results")

    # ── TEST 3: Search (BM25) ─────────────────────────
    print("\n=== TEST 3: Search (BM25) ===")
    results = store.search("SMOKE_TEST_PRIMARY", limit=10, mode="bm25")
    found = any(r["text"].startswith("SMOKE_TEST_") for r in results)
    if found:
        ok("BM25 search found test entry")
    else:
        fail(f"BM25 search missed test entry, got {len(results)} results")

    # ── TEST 4: Search (hybrid) ───────────────────────
    print("\n=== TEST 4: Search (hybrid) ===")
    results = store.search("SMOKE_TEST_PRIMARY verification", limit=5, mode="hybrid")
    found = any(
        (r[0] if isinstance(r, tuple) else r)["text"].startswith("SMOKE_TEST_")
        for r in results
    )
    if found:
        ok("Hybrid search found test entry")
    else:
        fail("Hybrid search missed test entry")

    # ── TEST 5: Update (supersede) ────────────────────
    print("\n=== TEST 5: Update ===")
    try:
        store.update(
            mem_id=mem_id,
            text="SMOKE_TEST_UPDATED: write+search+update chain confirmed.",
        )
        verify = store.search("SMOKE_TEST_UPDATED", limit=1, mode="vector")
        if verify and "SMOKE_TEST_UPDATED" in verify[0]["text"]:
            ok("Update persisted correctly (supersede pattern)")
        else:
            fail("Update did not persist")
    except Exception as e:
        fail(f"Update raised: {e}")

    # ── TEST 6: Delete ────────────────────────────────
    print("\n=== TEST 6: Delete ===")
    try:
        # Delete all smoke test entries
        all_matches = store.search("SMOKE_TEST_", limit=20, mode="vector")
        deleted_ids = set()
        for r in all_matches:
            entry = r if isinstance(r, dict) else r[0]
            if entry["text"].startswith("SMOKE_TEST_"):
                store.forget(mem_id=entry["id"])
                deleted_ids.add(entry["id"])

        # Verify they're gone
        remaining = store.search("SMOKE_TEST_", limit=20, mode="vector")
        smoke_left = [
            r for r in remaining
            if (r if isinstance(r, dict) else r[0])["text"].startswith("SMOKE_TEST_")
        ]
        if not smoke_left:
            ok(f"Delete confirmed — {len(deleted_ids)} entries removed")
        else:
            fail(f"Delete failed — {len(smoke_left)} entries still visible")
    except Exception as e:
        fail(f"Delete raised: {e}")

    # ── TEST 7: Stats ─────────────────────────────────
    print("\n=== TEST 7: Stats ===")
    stats = store.stats()
    if stats["total_memories"] > 0:
        ok(f"Stats: {stats['total_memories']} memories, categories: {stats['categories']}")
    else:
        fail("Stats returned zero memories")

    # ── TEST 8: List memories ─────────────────────────
    print("\n=== TEST 8: List memories ===")
    mems = store.list_memories(limit=5)
    if len(mems) == 5:
        ok("List returned 5 entries")
    else:
        fail(f"List returned {len(mems)} instead of 5")

    # ── TEST 9: has_id (existence check) ──────────────
    print("\n=== TEST 9: has_id (existence check) ===")
    if not store.has_id(mem_id):
        ok("has_id correctly returns False for deleted entry")
    else:
        fail("has_id returned True for deleted entry")

    # ── Summary ───────────────────────────────────────
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
