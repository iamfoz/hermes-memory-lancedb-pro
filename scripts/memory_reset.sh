#!/usr/bin/env bash
# memory_reset.sh — Wipe LanceDB memory database and reinitialise
# Usage: ./memory_reset.sh [--seed]  (re-seeds from MEMORY.md by default)
set -euo pipefail

DB_DIR="${MEMORY_DB_DIR:-$HOME/.hermes/memory-lancedb}"

echo "=== LanceDB Memory Reset ==="

if [ -d "$DB_DIR" ]; then
    echo "  Wiping: $DB_DIR"
    rm -rf "$DB_DIR"
    echo "  ✓ Database cleared"
else
    echo "  (No existing database found)"
fi

# Reinitialise
exec "$HOME/.hermes/scripts/memory_init.sh"
