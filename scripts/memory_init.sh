#!/usr/bin/env bash
# memory_init.sh — Initialise the LanceDB memory system from MEMORY.md
# Usage: ./memory_init.sh [--seed]  (seed by default)
set -euo pipefail

PYTHON="${HERMES_PYTHON:-$(find ~/.hermes -path '*/hermes-agent/venv/bin/python' 2>/dev/null | head -1)}"
MEMORY_MD="${MEMORY_MD:-$HOME/.hermes/memory/MEMORY.md}"
DB_DIR="${MEMORY_DB_DIR:-$HOME/.hermes/memory-lancedb}"
PLUGIN_DIR="${PLUGIN_DIR:-$HOME/.hermes/plugins/lancedb_pro}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Hermes Python not found. Set HERMES_PYTHON."
    exit 1
fi

echo "=== LanceDB Memory Initialisation ==="
echo "DB Path:  $DB_DIR"
echo "Memory:   $MEMORY_MD"

mkdir -p "$DB_DIR"

# Run initialisation in Python
"$PYTHON" << 'PYEOF'
import json
import os
import re
import sys
import uuid
import time

# Add plugin to path
plugin_dir = os.environ.get("PLUGIN_DIR", os.path.expanduser("~/.hermes/plugins/lancedb_pro"))
sys.path.insert(0, plugin_dir)

from store import MemoryStore

def parse_memory_md(path):
    """Parse MEMORY.md into structured entries."""
    entries = []
    with open(path, 'r') as f:
        content = f.read()
    
    # Split by section delimiter (lines starting with §)
    sections = re.split(r'^§\s*$', content, flags=re.MULTILINE)
    
    for section in sections:
        section = section.strip()
        # Skip headers, empty sections, and metadata
        if not section:
            continue
        if section.startswith('*This is the persistent') or section.startswith('---'):
            continue
        if section.startswith('User:') and 'UK English' in section:
            continue  # Skip user profile header
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            if len(para) < 20:
                continue
            
            # Auto-classify by content heuristics
            text_lower = para.lower()
            if any(kw in text_lower for kw in ['prefer', 'always', 'never', 'formatting']):
                category = "preference"
            elif any(kw in text_lower for kw in ['deadline', 'due', 'date']):
                category = "fact"
            elif any(kw in text_lower for kw in ['decision', 'chosen', 'selected']):
                category = "decision"
            elif any(kw in text_lower for kw in ['project', 'active', 'cron', 'system']):
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
    memory_md = os.environ.get("MEMORY_MD", os.path.expanduser("~/.hermes/memory/MEMORY.md"))
    
    # Initialise store — with corruption recovery
    db_dir = os.path.expanduser(os.environ.get("MEMORY_DB_DIR", "~/.hermes/memory-lancedb"))
    store = MemoryStore()

    try:
        store._initialise()
        # Quick health check — if stats fail, DB is corrupted
        stats = store.stats()
        print("  ✓ Connection established")
    except Exception as e:
        print(f"  ⚠ DB corrupted ({e}) — auto-recovering...")
        import shutil
        shutil.rmtree(db_dir, ignore_errors=True)
        store = MemoryStore()  # Fresh instance
        store._initialise()
        print("  ✓ Recovered (fresh DB created)")
    
    # Check if already seeded
    existing = store.list_memories(limit=1)
    if existing:
        print(f"  ⚠ Database already has {len(store.stats()['categories'])} entries")
        print("  Skipping seed. Run memory_reset.sh to start fresh.")
        return
    
    # Seed from MEMORY.md
    if os.path.exists(memory_md):
        entries = parse_memory_md(memory_md)
        if not entries:
            print("  ⚠ No entries found in MEMORY.md")
            return
        
        print(f"  Seeding {len(entries)} entries from MEMORY.md...")
        for i, entry in enumerate(entries, 1):
            mem_id = store.store(
                text=entry["text"],
                category=entry["category"],
                scope=entry["scope"],
                importance=entry["importance"],
            )
            if i % 10 == 0 or i == len(entries):
                print(f"    {i}/{len(entries)} entries stored")
        
        # Create vector index if enough data (needs 256+ rows for PQ training)
        total = len(entries)
        if total >= 256:
            try:
                store._table.create_index(vector_column_name="vector")
                print("  ✓ Vector index created")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"  ⚠ Vector index: {e}")
        else:
            print(f"  ℹ Vector index skipped (need 256+ rows, have {total})")
    
    # Verify
    stats = store.stats()
    print(f"\n  ✓ Seeded {stats['total_memories']} memories")
    print(f"  Categories: {stats['categories']}")
    
    # Quick smoke test
    test_entries = store.search("test", limit=1)
    print(f"  ✓ Search operational ({len(test_entries)} results for 'test')")

if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "=== Memory system ready ==="
