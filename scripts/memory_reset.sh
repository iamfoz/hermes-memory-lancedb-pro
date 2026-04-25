#!/usr/bin/env bash
# memory_reset.sh — Wipe LanceDB memory database and reinitialise
# Usage: ./memory_reset.sh
set -euo pipefail

DB_DIR="${MEMORY_DB_DIR:-$HOME/.hermes/memory-lancedb}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SCRIPT="${MEMORY_INIT_SCRIPT:-$SCRIPT_DIR/memory_init.sh}"

echo "=== LanceDB Memory Reset ==="

if [[ -d "$DB_DIR" ]]; then
    echo "  Wiping: $DB_DIR"
    rm -rf "$DB_DIR"
    echo "  ✓ Database cleared"
else
    echo "  (No existing database found)"
fi

if [[ ! -x "$INIT_SCRIPT" ]]; then
    echo "ERROR: init script not found at $INIT_SCRIPT" >&2
    echo "Set MEMORY_INIT_SCRIPT to override." >&2
    exit 1
fi

exec "$INIT_SCRIPT"
