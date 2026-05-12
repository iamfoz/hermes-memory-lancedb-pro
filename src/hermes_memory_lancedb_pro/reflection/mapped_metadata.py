"""Reflection mapped-metadata payload builder.

Ported from CortexReach reflection-mapped-metadata.ts.
Builds the metadata for "mapped" reflection memories — user-model,
agent-model, lesson, and decision kinds. Pure data assembly — no I/O,
no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "ReflectionMappedKind",
    "ReflectionMappedCategory",
    "ReflectionMappedMetadata",
    "build_reflection_mapped_metadata",
    "get_reflection_mapped_decay_defaults",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ReflectionMappedKind = Literal["user-model", "agent-model", "lesson", "decision"]
ReflectionMappedCategory = Literal["preference", "fact", "decision"]

# ---------------------------------------------------------------------------
# Decay defaults table (mirrors TS REFLECTION_MAPPED_DECAY_DEFAULTS)
# ---------------------------------------------------------------------------

_DECAY_DEFAULTS: dict[str, dict[str, float]] = {
    "decision":    {"midpoint_days": 45.0, "k": 0.25, "base_weight": 1.1,  "quality": 1.0},
    "user-model":  {"midpoint_days": 21.0, "k": 0.30, "base_weight": 1.0,  "quality": 0.95},
    "agent-model": {"midpoint_days": 10.0, "k": 0.35, "base_weight": 0.95, "quality": 0.93},
    "lesson":      {"midpoint_days":  7.0, "k": 0.45, "base_weight": 0.9,  "quality": 0.9},
}

# kind → category mapping (mirrors TS buildReflectionMappedMetadata)
_KIND_TO_CATEGORY: dict[str, ReflectionMappedCategory] = {
    "user-model":  "preference",
    "agent-model": "preference",
    "lesson":      "fact",
    "decision":    "decision",
}

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReflectionMappedMetadata:
    type: str = "memory-reflection-mapped"
    reflection_version: int = 4
    stage: str = "reflect-store"
    event_id: str = ""
    mapped_kind: ReflectionMappedKind = "lesson"
    mapped_category: ReflectionMappedCategory = "fact"
    section: str = ""
    ordinal: int = 0
    group_size: int = 0
    agent_id: str = ""
    session_key: str = ""
    session_id: str = ""
    stored_at: int = 0
    used_fallback: bool = False
    error_signals: list[str] = field(default_factory=list)
    decay_model: str = "logistic"
    decay_midpoint_days: float = 0.0
    decay_k: float = 0.0
    base_weight: float = 0.0
    quality: float = 0.0
    source_reflection_path: str | None = None
    reflection_heading: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_reflection_mapped_decay_defaults(
    kind: ReflectionMappedKind,
) -> dict[str, float]:
    """Return decay defaults for the given *kind*.

    Returns ``{"midpoint_days", "k", "base_weight", "quality"}``.
    Matches the TS ``getReflectionMappedDecayDefaults`` function.
    """
    return dict(_DECAY_DEFAULTS[kind])


def build_reflection_mapped_metadata(
    *,
    mapped_item: Any,  # duck-typed ReflectionMappedMemoryItem
    event_id: str,
    agent_id: str,
    session_key: str,
    session_id: str,
    run_at: int,
    used_fallback: bool = False,
    tool_error_signals: list[dict] | None = None,
    source_reflection_path: str | None = None,
) -> ReflectionMappedMetadata:
    """Build a :class:`ReflectionMappedMetadata` for a single mapped item.

    *mapped_item* is duck-typed as :class:`~.slices.ReflectionMappedMemoryItem`
    (or a plain dict) with ``mapped_kind``, ``category``, ``heading``,
    ``ordinal``, and ``group_size`` attributes/keys.
    """
    if tool_error_signals is None:
        tool_error_signals = []

    error_signals = [
        sig.get("signature_hash") or sig.get("signatureHash", "")
        for sig in tool_error_signals
    ]

    # Support both attribute access (dataclass/object) and dict access.
    # Accept either `kind` (per `ReflectionMappedMemoryItem`) or `mapped_kind`
    # (per the older TS field name) so callers using either shape work.
    if isinstance(mapped_item, dict):
        mapped_kind: ReflectionMappedKind = (
            mapped_item.get("kind") or mapped_item.get("mapped_kind")
        )
        # `category` is read for input validation but not propagated;
        # the canonical mapped_category comes from _KIND_TO_CATEGORY below
        _ = mapped_item.get("category")
        heading: str = mapped_item.get("heading") or mapped_item.get("section") or ""
        ordinal: int = mapped_item.get("ordinal", 0)
        group_size: int = mapped_item.get("group_size", 0)
    else:
        mapped_kind = (
            getattr(mapped_item, "kind", None)
            or getattr(mapped_item, "mapped_kind", None)
        )
        _ = getattr(mapped_item, "category", None)
        heading = (
            getattr(mapped_item, "heading", None)
            or getattr(mapped_item, "section", "")
            or ""
        )
        ordinal = getattr(mapped_item, "ordinal", 0)
        group_size = getattr(mapped_item, "group_size", 0)

    defaults = get_reflection_mapped_decay_defaults(mapped_kind)

    return ReflectionMappedMetadata(
        type="memory-reflection-mapped",
        reflection_version=4,
        stage="reflect-store",
        event_id=event_id,
        mapped_kind=mapped_kind,
        mapped_category=_KIND_TO_CATEGORY[mapped_kind],
        section=heading,
        ordinal=ordinal,
        group_size=group_size,
        agent_id=agent_id,
        session_key=session_key,
        session_id=session_id,
        stored_at=run_at,
        used_fallback=used_fallback,
        error_signals=error_signals,
        decay_model="logistic",
        decay_midpoint_days=defaults["midpoint_days"],
        decay_k=defaults["k"],
        base_weight=defaults["base_weight"],
        quality=defaults["quality"],
        source_reflection_path=source_reflection_path,
    )
