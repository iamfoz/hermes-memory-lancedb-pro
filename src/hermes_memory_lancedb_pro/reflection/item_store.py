"""Reflection item-store payload builder.

Ported from CortexReach reflection-item-store.ts.
Builds per-item payloads for the individual invariant/derived lines that
come out of a reflection run. Pure data assembly — no I/O, no LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "ReflectionItemMetadata",
    "ReflectionItemPayload",
    "build_reflection_item_payloads",
    "get_reflection_item_decay_defaults",
    # decay constants (re-exported for callers)
    "REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS",
    "REFLECTION_INVARIANT_DECAY_K",
    "REFLECTION_INVARIANT_BASE_WEIGHT",
    "REFLECTION_INVARIANT_QUALITY",
    "REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS",
    "REFLECTION_DERIVED_DECAY_K",
    "REFLECTION_DERIVED_BASE_WEIGHT",
    "REFLECTION_DERIVED_QUALITY",
]

# ---------------------------------------------------------------------------
# Decay constants (mirroring TS exports)
# ---------------------------------------------------------------------------

REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS: int = 45
REFLECTION_INVARIANT_DECAY_K: float = 0.22
REFLECTION_INVARIANT_BASE_WEIGHT: float = 1.1
REFLECTION_INVARIANT_QUALITY: float = 1.0

REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS: int = 7
REFLECTION_DERIVED_DECAY_K: float = 0.65
REFLECTION_DERIVED_BASE_WEIGHT: float = 1.0
REFLECTION_DERIVED_QUALITY: float = 0.95

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

ReflectionItemKind = Literal["invariant", "derived"]


@dataclass
class ReflectionItemMetadata:
    type: str = "memory-reflection-item"
    reflection_version: int = 4
    stage: str = "reflect-store"
    event_id: str = ""
    item_kind: ReflectionItemKind = "derived"
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
    resolved_at: int | None = None
    resolved_by: str | None = None
    resolution_note: str | None = None


@dataclass
class ReflectionItemPayload:
    kind: str = ""
    text: str = ""
    metadata: ReflectionItemMetadata = field(default_factory=ReflectionItemMetadata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_reflection_item_decay_defaults(
    item_kind: ReflectionItemKind,
) -> dict[str, float]:
    """Return decay defaults for a given item kind.

    Returns ``{"midpoint_days", "k", "base_weight", "quality"}``.
    """
    if item_kind == "invariant":
        return {
            "midpoint_days": float(REFLECTION_INVARIANT_DECAY_MIDPOINT_DAYS),
            "k": REFLECTION_INVARIANT_DECAY_K,
            "base_weight": REFLECTION_INVARIANT_BASE_WEIGHT,
            "quality": REFLECTION_INVARIANT_QUALITY,
        }
    return {
        "midpoint_days": float(REFLECTION_DERIVED_DECAY_MIDPOINT_DAYS),
        "k": REFLECTION_DERIVED_DECAY_K,
        "base_weight": REFLECTION_DERIVED_BASE_WEIGHT,
        "quality": REFLECTION_DERIVED_QUALITY,
    }


def build_reflection_item_payloads(
    *,
    items: Sequence[Any],  # duck-typed ReflectionSliceItem: .text, .item_kind, .section, .ordinal, .group_size
    event_id: str,
    agent_id: str,
    session_key: str,
    session_id: str,
    run_at: int,
    used_fallback: bool = False,
    tool_error_signals: list[dict] | None = None,
    source_reflection_path: str | None = None,
) -> list[ReflectionItemPayload]:
    """Build one :class:`ReflectionItemPayload` per item in *items*.

    *items* is a sequence of duck-typed :class:`~.slices.ReflectionSliceItem`
    objects (or plain dicts) with at least ``text``, ``item_kind``,
    ``section``, ``ordinal``, and ``group_size`` attributes/keys.
    """
    if tool_error_signals is None:
        tool_error_signals = []

    error_signals = [
        sig.get("signature_hash") or sig.get("signatureHash", "")
        for sig in tool_error_signals
    ]

    payloads: list[ReflectionItemPayload] = []
    for item in items:
        # Support both attribute access (dataclass/object) and dict access
        if isinstance(item, dict):
            item_kind: ReflectionItemKind = item["item_kind"]
            text: str = item["text"]
            section: str = item.get("section", "")
            ordinal: int = item.get("ordinal", 0)
            group_size: int = item.get("group_size", 0)
        else:
            item_kind = item.item_kind
            text = item.text
            section = getattr(item, "section", "")
            ordinal = getattr(item, "ordinal", 0)
            group_size = getattr(item, "group_size", 0)

        defaults = get_reflection_item_decay_defaults(item_kind)

        metadata = ReflectionItemMetadata(
            type="memory-reflection-item",
            reflection_version=4,
            stage="reflect-store",
            event_id=event_id,
            item_kind=item_kind,
            section=section,
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

        kind_tag = "item-invariant" if item_kind == "invariant" else "item-derived"

        payloads.append(ReflectionItemPayload(
            kind=kind_tag,
            text=text,
            metadata=metadata,
        ))

    return payloads
