#!/usr/bin/env bash
# memory_reset.sh — Wipe LanceDB memory database and reinitialise
# Usage: ./memory_reset.sh [-y]
#
# Same safety rules as `hermes-memory-lancedb-pro reset`: the database path is
# $MEMORY_DB_DIR, else <hermes home>/memory-lancedb; a directory that holds no
# LanceDB table is never deleted; and you are asked to confirm unless -y.
set -euo pipefail

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
DB_DIR="${MEMORY_DB_DIR:-$HERMES_HOME_DIR/memory-lancedb}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SCRIPT="${MEMORY_INIT_SCRIPT:-$SCRIPT_DIR/memory_init.sh}"

echo "=== LanceDB Memory Reset ==="

if [[ -d "$DB_DIR" ]]; then
    if [[ -n "$(ls -A "$DB_DIR")" ]] && ! compgen -G "$DB_DIR/*.lance" > /dev/null; then
        echo "ERROR: $DB_DIR does not look like a memory database" >&2
        echo "(no LanceDB table inside); refusing to delete it." >&2
        exit 1
    fi
    if [[ "${1:-}" != "-y" && "${1:-}" != "--yes" ]]; then
        echo "This will WIPE ALL MEMORIES at: $DB_DIR"
        read -r -p 'Type "yes" to proceed: ' answer
        if [[ "$answer" != "yes" ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
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
