"""Reflection metadata helpers.

Ported from CortexReach reflection-metadata.ts.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "parse_reflection_metadata",
    "is_reflection_entry",
    "get_display_category_tag",
]

_REFLECTION_METADATA_TYPES = frozenset(
    {
        "reflection",
        "memory-reflection",
        "memory-reflection-event",
        "memory-reflection-item",
    }
)


def parse_reflection_metadata(raw: str | dict | None) -> dict[str, Any]:
    """Best-effort metadata coercion to a dict.

    Accepts:
      * already-parsed ``dict`` (returned as-is) — the common case when called
        on rows produced by ``MemoryStore._row_to_dict``;
      * JSON string (parsed via ``json.loads``);
      * ``None`` / empty / non-object JSON — yields ``{}``.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if parsed and isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _coerce_metadata(value: Any) -> dict[str, Any]:
    """Accept a metadata field that may be a dict or a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return parse_reflection_metadata(value)
    return {}


def is_reflection_entry(entry: dict[str, Any]) -> bool:
    """Return True if *entry* represents a reflection.

    A reflection is identified by either:
    - ``entry["category"] == "reflection"``
    - ``entry["metadata"]["type"]`` being one of the four reflection type strings.
    """
    if entry.get("category") == "reflection":
        return True
    metadata = _coerce_metadata(entry.get("metadata"))
    return metadata.get("type") in _REFLECTION_METADATA_TYPES


def get_display_category_tag(entry: dict[str, Any]) -> str:
    """Return a UI display tag for *entry*.

    Reflection entries get ``"reflection:{scope}"``; all others get
    ``"{category}:{scope}"`` following the legacy convention.
    """
    if not is_reflection_entry(entry):
        return f"{entry.get('category', '')}:{entry.get('scope', '')}"
    return f"reflection:{entry.get('scope', '')}"
