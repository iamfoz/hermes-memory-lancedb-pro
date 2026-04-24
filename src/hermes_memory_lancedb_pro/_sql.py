"""Small pure-Python helpers used by the LanceDB-backed store.

Kept separate from `store.py` so they remain importable (and testable)
without `lancedb` and `sentence_transformers` installed."""

from __future__ import annotations

import json
from typing import Any

ARCHIVED_STATE = "archived"


def escape_sql(val: Any) -> str:
    """Escape a value for safe inclusion in single-quoted SQL literals."""
    return str(val).replace("'", "''")


def parse_metadata(value: Any) -> dict[str, Any]:
    """Parse metadata into a dict; accept dict, JSON string, or junk."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {}


def is_archived(metadata: Any) -> bool:
    """True if the row's metadata declares it archived."""
    parsed = parse_metadata(metadata)
    return parsed.get("state") == ARCHIVED_STATE


def and_clauses(*clauses: str | None) -> str | None:
    """Join SQL clauses with AND, dropping None / empty strings."""
    parts = [c for c in clauses if c]
    if not parts:
        return None
    return " AND ".join(f"({c})" for c in parts)
