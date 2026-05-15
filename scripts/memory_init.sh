#!/usr/bin/env bash
# memory_init.sh — Initialise the LanceDB memory system from MEMORY.md
# Usage: ./memory_init.sh [--seed]   (seeds by default)
#
# Env vars:
#   HERMES_PYTHON   — Python interpreter (auto-detected if unset)
#   MEMORY_MD       — Path to MEMORY.md seed file
#   MEMORY_DB_DIR   — LanceDB data directory
#   PLUGIN_DIR      — Plugin install directory (used as Python import root)
set -euo pipefail

# Discover Python: explicit override → hermes venv → first python3 on PATH
if [[ -n "${HERMES_PYTHON:-}" ]]; then
    PYTHON="$HERMES_PYTHON"
else
    PYTHON="$(find "$HOME/.hermes" -path '*/hermes-agent/venv/bin/python' 2>/dev/null | head -1 || true)"
    if [[ -z "$PYTHON" ]]; then
        PYTHON="$(command -v python3 || command -v python || true)"
    fi
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: No Python interpreter found. Set HERMES_PYTHON." >&2
    exit 1
fi

MEMORY_MD="${MEMORY_MD:-$HOME/.hermes/memory/MEMORY.md}"
MEMORY_DB_DIR="${MEMORY_DB_DIR:-$HOME/.hermes/memory-lancedb}"
PLUGIN_DIR="${PLUGIN_DIR:-$HOME/.hermes/hermes-agent/plugins/memory/lancedb_pro}"

export MEMORY_MD MEMORY_DB_DIR PLUGIN_DIR

echo "=== LanceDB Memory Initialisation ==="
echo "DB Path:  $MEMORY_DB_DIR"
echo "Memory:   $MEMORY_MD"
echo "Plugin:   $PLUGIN_DIR"

mkdir -p "$MEMORY_DB_DIR"

"$PYTHON" - << 'PYEOF'
import os
import re
import shutil
import sys

# Try the installed package first; fall back to importing from PLUGIN_DIR.
plugin_dir = os.environ.get("PLUGIN_DIR", os.path.expanduser("~/.hermes/hermes-agent/plugins/memory/lancedb_pro"))
candidates = [
    os.path.join(plugin_dir, "src"),  # editable / source layout
    plugin_dir,                       # flattened install
]
for path in candidates:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from hermes_memory_lancedb_pro.store import MemoryStore  # noqa: E402
except ImportError:
    # Legacy flattened layout: store.py at PLUGIN_DIR root
    from store import MemoryStore  # type: ignore  # noqa: E402


def parse_memory_md(path: str):
    """Parse MEMORY.md into structured entries.

    Best-effort heuristic classification. Section delimiter: a line that is
    just `§`. Paragraphs shorter than 20 chars are skipped to avoid noise."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"^§\s*$", content, flags=re.MULTILINE)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if section.startswith("*This is the persistent") or section.startswith("---"):
            continue
        if section.startswith("User:") and "UK English" in section:
            continue

        for para in (p.strip() for p in section.split("\n\n")):
            if len(para) < 20:
                continue

            text_lower = para.lower()
            if any(kw in text_lower for kw in ("prefer", "always", "never", "formatting")):
                category = "preference"
            elif any(kw in text_lower for kw in ("decision", "chosen", "selected")):
                category = "decision"
            elif any(kw in text_lower for kw in ("deadline", "due", "date")):
                category = "fact"
            elif any(kw in text_lower for kw in ("project", "active", "cron", "system")):
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


def main():
    memory_md = os.environ.get(
        "MEMORY_MD", os.path.expanduser("~/.hermes/memory/MEMORY.md")
    )
    db_dir = os.path.expanduser(
        os.environ.get("MEMORY_DB_DIR", "~/.hermes/memory-lancedb")
    )

    # Honour MEMORY_DB_DIR — the previous version constructed a default
    # store and silently ignored the env var.
    store = MemoryStore(db_path=db_dir)
    try:
        store._initialise()
        store.stats()
        print("  ✓ Connection established")
    except Exception as e:
        print(f"  ⚠ DB corrupted ({e}) — auto-recovering...")
        shutil.rmtree(db_dir, ignore_errors=True)
        store = MemoryStore(db_path=db_dir)
        store._initialise()
        print("  ✓ Recovered (fresh DB created)")

    existing = store.list_memories(limit=1)
    if existing:
        total = store.stats().get("total_memories", 0)
        print(f"  ⚠ Database already has {total} entries")
        print("  Skipping seed. Run memory_reset.sh to start fresh.")
        return

    if not os.path.exists(memory_md):
        print(f"  ℹ No seed file at {memory_md}")
        return

    entries = parse_memory_md(memory_md)
    if not entries:
        print("  ⚠ No entries found in MEMORY.md")
        return

    print(f"  Seeding {len(entries)} entries from MEMORY.md...")
    # Bulk-write — much faster than one row per insert.
    ids = store.store_many(entries)
    print(f"    {len(ids)}/{len(entries)} entries stored")

    # Vector index needs ~256 rows of training data
    if store.maybe_create_vector_index():
        print("  ✓ Vector index created")
    else:
        print(f"  ℹ Vector index pending (need 256+ rows, have {len(ids)})")

    stats = store.stats()
    print(f"\n  ✓ Seeded {stats['total_memories']} memories")
    print(f"  Categories: {stats['categories']}")

    test_entries = store.search("test", limit=1)
    print(f"  ✓ Search operational ({len(test_entries)} results for 'test')")


if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "=== Memory system ready ==="
