"""Small pure-Python helpers used by the LanceDB-backed store.

Kept separate from `store.py` so they remain importable (and testable)
without `lancedb` and `sentence_transformers` installed."""

from __future__ import annotations

import json
from typing import Any

ARCHIVED_STATE = "archived"

# Tiers whose memories are always available regardless of session_id filtering.
# Kept in sync with the constant of the same name in store.py — duplicated
# here so this helpers module stays lancedb-free.
CROSS_SESSION_TIERS = frozenset({"core"})


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


def match_session(metadata: Any, session_id: str) -> bool:
    """Return True if a memory should be visible to the given `session_id`.

    A memory matches when ANY of:
      - its `metadata.source_session` equals `session_id` (this session)
      - its `metadata.cross_session` is truthy (explicit opt-in)
      - its `metadata.tier` is in `CROSS_SESSION_TIERS` (e.g. "core" — long-term
        knowledge that should surface across all sessions)

    A memory whose `source_session` is empty / missing is treated as "global"
    and only matches when cross_session=True or tier indicates it's cross-session.
    The default new-row metadata sets `source_session=""` and `cross_session=False`,
    so callers that don't pass session_id at write time get the legacy "all
    memories visible everywhere" behaviour only if they ALSO don't pass
    session_id at read time."""
    if not session_id:
        return True
    parsed = parse_metadata(metadata)
    if parsed.get("cross_session"):
        return True
    if parsed.get("tier") in CROSS_SESSION_TIERS:
        return True
    src = parsed.get("source_session") or ""
    return bool(src) and src == session_id
