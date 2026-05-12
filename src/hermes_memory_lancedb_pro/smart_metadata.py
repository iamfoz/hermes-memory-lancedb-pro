"""Smart-memory metadata layer — L0/L1/L2 abstracts, fact-key derivation,
temporal versioning, and per-context support stats.

Ported from CortexReach smart-metadata.ts (v700-line revision).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# TS-spec temporal-versioned set (mirrors memory_categories.TEMPORAL_VERSIONED_CATEGORIES)
# ---------------------------------------------------------------------------
_TEMPORAL_VERSIONED_CATEGORIES: frozenset[str] = frozenset({"preferences", "entities"})

# ---------------------------------------------------------------------------
# Module-level regex (compile once)
# ---------------------------------------------------------------------------

# Matches "[topic]: description" or "[topic]：description" (CJK full-width colon)
_COLON_RE = re.compile(r"^(.{1,120}?)[：:]")
# Matches "[topic] -> description" or "[topic] => description"
_ARROW_RE = re.compile(r"^(.{1,120}?)(?:\s*->|\s*=>)")
# Strip trailing punctuation from topic
_TRAILING_PUNCT_RE = re.compile(r"[。.!?]+$")

# ---------------------------------------------------------------------------
# Context-normalisation alias map
# ---------------------------------------------------------------------------

_CONTEXT_ALIASES: dict[str, str] = {
    # Morning
    "早上": "morning",
    "上午": "morning",
    "早晨": "morning",
    # Afternoon
    "下午": "afternoon",
    # Evening
    "傍晚": "evening",
    "晚上": "evening",
    # Night
    "深夜": "night",
    "夜晚": "night",
    "凌晨": "night",
    # Weekday
    "工作日": "weekday",
    "平时": "weekday",
    # Weekend
    "周末": "weekend",
    "假日": "weekend",
    "休息日": "weekend",
    # Work
    "工作": "work",
    "上班": "work",
    "办公": "work",
    # Leisure
    "休闲": "leisure",
    "放松": "leisure",
    "休息": "leisure",
    # Summer
    "夏天": "summer",
    "夏季": "summer",
    # Winter
    "冬天": "winter",
    "冬季": "winter",
    # Travel
    "旅行": "travel",
    "出差": "travel",
    "旅游": "travel",
}

# Maximum number of context slices stored per memory
MAX_SUPPORT_SLICES: int = 8

# ---------------------------------------------------------------------------
# Array-size caps (mirrors TS stringifySmartMetadata caps)
# ---------------------------------------------------------------------------

_MAX_SOURCES = 20
_MAX_HISTORY = 50
_MAX_RELATIONS = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupportSlice:
    """Per-context evidence (e.g. context_label='evening')."""

    context: str
    confirmations: int = 0
    contradictions: int = 0
    strength: float = 0.5        # confirmations / (confirmations + contradictions)
    last_observed_at: int = 0    # epoch ms


@dataclass
class SupportInfoV2:
    """v2 format: per-context evidence + global aggregate."""

    version: int = 2
    global_strength: float = 0.0    # weighted avg across slices
    total_observations: int = 0
    slices: list[SupportSlice] = field(default_factory=list)


@dataclass
class SmartMemoryMetadata:
    l0_abstract: str = ""
    l1_overview: str = ""
    l2_content: str = ""
    memory_category: str = ""           # one of memory_categories.SmartCategory
    tier: str = "working"
    confidence: float = 0.5             # [0, 1]
    access_count: int = 0
    injected_count: int = 0
    bad_recall_count: int = 0
    valid_from: int = 0
    valid_until: int | None = None
    invalidated_at: int | None = None
    fact_key: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    relations: list[dict] = field(default_factory=list)  # {type, target_id}
    support_info: SupportInfoV2 | None = None
    cross_session: bool = False
    source: str = "manual"
    source_session: str = ""
    # Free-form passthrough for any extra fields the TS doesn't enumerate
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal normalisation helpers (mirrors TS private helpers)
# ---------------------------------------------------------------------------

def _clamp01(value: Any, fallback: float) -> float:
    """Clamp a value to [0, 1]; use fallback if not finite."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(n):
        return fallback
    return min(1.0, max(0.0, n))


def _clamp_count(value: Any, fallback: int = 0) -> int:
    """Clamp a value to a non-negative integer; use fallback if invalid."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(n) or n < 0:
        return fallback
    return int(n)


def _normalize_tier(value: Any) -> str:
    if value in ("core", "working", "peripheral"):
        return value
    return "working"


def _normalize_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _normalize_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_timestamp(value: Any, fallback: int) -> int:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(n) or n <= 0:
        return fallback
    return int(n)


def _normalize_optional_timestamp(value: Any) -> int | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or n <= 0:
        return None
    return int(n)


def _normalize_source(value: Any) -> str:
    if value in ("manual", "auto-capture", "reflection", "session-summary", "legacy"):
        return value
    return "legacy"


def _normalize_state(value: Any) -> str:
    if value in ("pending", "confirmed", "archived"):
        return value
    return "confirmed"


def _normalize_layer(value: Any) -> str:
    if value in ("durable", "working", "reflection", "archive"):
        return value
    return "working"


def _derive_default_layer(source: str, memory_category: str, state: str) -> str:
    if source in ("reflection", "session-summary"):
        return "reflection"
    if state == "archived":
        return "archive"
    if memory_category in ("profile", "preferences", "events"):
        return "durable"
    return "working"


def _default_overview(text: str) -> str:
    return f"- {text}"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalise_context(label: str | None) -> str | None:
    """Normalise a raw context label to a canonical context string.

    Returns None for None/empty input.  Looks up the CJK alias map; returns
    the canonical English form if matched, otherwise the lowercased input.
    """
    if not label or not label.strip():
        return None
    lower = label.strip().lower()
    return _CONTEXT_ALIASES.get(lower, lower)


def derive_fact_key(category: str, abstract: str) -> str | None:
    """Return a stable fact-key for temporal-versioned categories.

    Format: ``"<category>:<topic_lowercase_spaces_collapsed>"``

    Extracts topic from:
    - ``"[topic]: description"`` (ASCII or CJK colon)
    - ``"[topic] -> description"`` / ``"[topic] => description"``

    Returns None for non-temporal categories or unparseable abstracts.
    """
    if category not in _TEMPORAL_VERSIONED_CATEGORIES:
        return None

    trimmed = abstract.strip()
    if not trimmed:
        return None

    topic = trimmed
    colon_match = _COLON_RE.match(trimmed)
    arrow_match = _ARROW_RE.match(trimmed)
    if colon_match and colon_match.group(1):
        topic = colon_match.group(1)
    elif arrow_match and arrow_match.group(1):
        topic = arrow_match.group(1)

    normalized = (
        _TRAILING_PUNCT_RE.sub("", topic.lower().replace("\t", " "))
    )
    # collapse internal whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return f"{category}:{normalized}" if normalized else None


def is_memory_active_at(
    metadata: SmartMemoryMetadata,
    at_ms: int | None = None,
) -> bool:
    """True iff the memory is within its valid window and not invalidated.

    ``at_ms`` defaults to the current time (epoch ms).
    """
    at = at_ms if at_ms is not None else int(time.time() * 1000)
    if metadata.valid_from > at:
        return False
    return metadata.invalidated_at is None or metadata.invalidated_at > at


def is_memory_expired(
    metadata: SmartMemoryMetadata,
    at_ms: int | None = None,
) -> bool:
    """True iff ``valid_until`` is set and the memory has passed its expiry.

    Separate from :func:`is_memory_active_at` (which checks ``invalidated_at``
    from superseding).  Returns False when ``valid_until`` is not set
    (no expiry = permanent).
    """
    at = at_ms if at_ms is not None else int(time.time() * 1000)
    return metadata.valid_until is not None and metadata.valid_until <= at


def parse_smart_metadata(
    raw: str | dict | None,
    entry: dict | None = None,
) -> SmartMemoryMetadata:
    """Parse JSON-string, dict, or None into a :class:`SmartMemoryMetadata`.

    If *entry* is provided, missing fields are filled from the entry dict
    (e.g. ``entry["text"]`` becomes ``l2_content`` when absent).  Parse
    errors are caught defensively and result in sensible defaults.
    """
    parsed: dict[str, Any] = {}
    if raw is not None:
        if isinstance(raw, dict):
            parsed = dict(raw)
        elif isinstance(raw, str) and raw.strip():
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    parsed = obj
            except (json.JSONDecodeError, ValueError):
                parsed = {}

    if entry is None:
        entry = {}

    text: str = entry.get("text") or ""
    ts_raw = entry.get("timestamp")
    timestamp: int = (
        int(ts_raw)
        if isinstance(ts_raw, (int, float)) and math.isfinite(float(ts_raw))
        else int(time.time() * 1000)
    )

    memory_category: str = str(parsed.get("memory_category", "patterns"))

    l0 = _normalize_text(parsed.get("l0_abstract"), text)
    l2 = _normalize_text(parsed.get("l2_content"), text)
    valid_from = _normalize_timestamp(parsed.get("valid_from"), timestamp)
    invalidated_at_raw = _normalize_optional_timestamp(parsed.get("invalidated_at"))

    # Fallback source derived from legacy `type` field
    type_field = parsed.get("type", "")
    if type_field == "session-summary":
        fallback_source = "session-summary"
    elif type_field in ("memory-reflection", "memory-reflection-item"):
        fallback_source = "reflection"
    else:
        fallback_source = "legacy"

    source = _normalize_source(parsed.get("source", fallback_source))
    # Default to "confirmed" regardless of source — session-summary memories
    # must be queryable immediately; an "archived" default silently hid them.
    state = _normalize_state(parsed.get("state", "confirmed"))
    memory_layer = _normalize_layer(
        parsed.get("memory_layer")
        if parsed.get("memory_layer") is not None
        else _derive_default_layer(source, memory_category, state)
    )
    # Write the *normalised* state + memory_layer back through extras so they
    # round-trip via stringify_smart_metadata. They aren't first-class fields
    # on SmartMemoryMetadata (kept compact), but TS-native callers expect them.
    parsed["state"] = state
    parsed["memory_layer"] = memory_layer

    fact_key = _normalize_optional_string(parsed.get("fact_key"))
    if fact_key is None:
        fact_key = derive_fact_key(memory_category, l0)

    invalidated_at: int | None = (
        invalidated_at_raw
        if invalidated_at_raw is not None and invalidated_at_raw >= valid_from
        else None
    )

    # The typed fields below have their own dataclass attributes; skip them
    # when building the extras dict. State + memory_layer + the various
    # last_* timestamps are TS-native passthrough — keep them in extras so
    # they round-trip through stringify_smart_metadata.
    _typed_keys = {
        "l0_abstract", "l1_overview", "l2_content", "memory_category",
        "tier", "confidence", "access_count", "injected_count",
        "bad_recall_count", "valid_from", "valid_until", "invalidated_at",
        "fact_key", "supersedes", "superseded_by", "relations",
        "support_info", "cross_session", "source", "source_session",
    }
    extras: dict[str, Any] = {k: v for k, v in parsed.items() if k not in _typed_keys}

    return SmartMemoryMetadata(
        l0_abstract=l0,
        l1_overview=_normalize_text(parsed.get("l1_overview"), _default_overview(l0)),
        l2_content=l2,
        memory_category=memory_category,
        tier=_normalize_tier(parsed.get("tier")),
        confidence=_clamp01(parsed.get("confidence"), 0.7),
        access_count=_clamp_count(parsed.get("access_count"), 0),
        injected_count=_clamp_count(parsed.get("injected_count"), 0),
        bad_recall_count=_clamp_count(parsed.get("bad_recall_count"), 0),
        valid_from=valid_from,
        valid_until=_normalize_optional_timestamp(parsed.get("valid_until")),
        invalidated_at=invalidated_at,
        fact_key=fact_key,
        supersedes=_normalize_optional_string(parsed.get("supersedes")),
        superseded_by=_normalize_optional_string(parsed.get("superseded_by")),
        relations=list(parsed.get("relations") or []),
        support_info=parse_support_info(parsed.get("support_info")),
        cross_session=bool(parsed.get("cross_session", False)),
        source=source,
        source_session=(
            parsed.get("source_session")
            if isinstance(parsed.get("source_session"), str)
            else ""
        ),
        extras=extras,
    )


def build_smart_metadata(
    entry: dict | None,
    patch: dict,
) -> SmartMemoryMetadata:
    """Start from *entry*'s existing metadata (parsed) and apply *patch*.

    Use this when updating an existing memory.  Unknown keys in *patch* are
    forwarded to ``extras``.
    """
    if entry is None:
        entry = {}

    # Parse the base from the entry's existing metadata field
    base = parse_smart_metadata(entry.get("metadata"), entry)

    l0_abstract = _normalize_text(patch.get("l0_abstract"), base.l0_abstract)

    next_category = (
        str(patch["memory_category"])
        if isinstance(patch.get("memory_category"), str)
        else base.memory_category
    )

    next_source = (
        _normalize_source(patch["source"])
        if "source" in patch
        else base.source
    )

    next_state_raw = patch.get("state")
    next_state = (
        _normalize_state(next_state_raw)
        if next_state_raw is not None
        else _normalize_state(base.extras.get("state", "confirmed"))
    )

    next_layer = (
        _normalize_layer(patch["memory_layer"])
        if "memory_layer" in patch
        else _normalize_layer(base.extras.get("memory_layer", "working"))
    )

    valid_from = _normalize_timestamp(patch.get("valid_from"), base.valid_from)

    if "invalidated_at" not in patch:
        invalidated_at = base.invalidated_at
    else:
        raw_inv = _normalize_optional_timestamp(patch["invalidated_at"])
        invalidated_at = (
            raw_inv if raw_inv is not None and raw_inv >= valid_from else None
        )

    fact_key = _normalize_optional_string(patch.get("fact_key"))
    if fact_key is None:
        fact_key = base.fact_key
    if fact_key is None:
        fact_key = derive_fact_key(next_category, l0_abstract)

    supersedes = (
        base.supersedes
        if "supersedes" not in patch
        else _normalize_optional_string(patch["supersedes"])
    )
    superseded_by = (
        base.superseded_by
        if "superseded_by" not in patch
        else _normalize_optional_string(patch["superseded_by"])
    )
    source_session = (
        str(patch["source_session"])
        if isinstance(patch.get("source_session"), str)
        else base.source_session
    )

    valid_until = (
        base.valid_until
        if "valid_until" not in patch
        else _normalize_optional_timestamp(patch["valid_until"])
    )

    # Merge extras: start from base extras, layer patch's unknown keys on top.
    # State + memory_layer go into extras (TS-native passthrough) — using the
    # normalised values computed above.
    _typed_keys = {
        "l0_abstract", "l1_overview", "l2_content", "memory_category",
        "tier", "confidence", "access_count", "injected_count",
        "bad_recall_count", "valid_from", "valid_until", "invalidated_at",
        "fact_key", "supersedes", "superseded_by", "relations",
        "support_info", "cross_session", "source", "source_session",
    }
    merged_extras = dict(base.extras)
    for k, v in patch.items():
        if k not in _typed_keys:
            merged_extras[k] = v
    # Carry the normalised state + memory_layer through extras so the next
    # parse / stringify cycle preserves them.
    merged_extras["state"] = next_state
    merged_extras["memory_layer"] = next_layer

    return SmartMemoryMetadata(
        l0_abstract=l0_abstract,
        l1_overview=_normalize_text(patch.get("l1_overview"), base.l1_overview),
        l2_content=_normalize_text(patch.get("l2_content"), base.l2_content),
        memory_category=next_category,
        tier=_normalize_tier(patch.get("tier", base.tier)),
        confidence=_clamp01(patch.get("confidence", base.confidence), base.confidence),
        access_count=_clamp_count(patch.get("access_count", base.access_count), base.access_count),
        injected_count=_clamp_count(patch.get("injected_count", base.injected_count), base.injected_count),
        bad_recall_count=_clamp_count(patch.get("bad_recall_count", base.bad_recall_count), base.bad_recall_count),
        valid_from=valid_from,
        valid_until=valid_until,
        invalidated_at=invalidated_at,
        fact_key=fact_key,
        supersedes=supersedes,
        superseded_by=superseded_by,
        relations=list(patch.get("relations", base.relations) or []),
        support_info=parse_support_info(patch.get("support_info")) if "support_info" in patch else base.support_info,
        cross_session=bool(patch.get("cross_session", base.cross_session)),
        source=next_source,
        source_session=source_session,
        extras=merged_extras,
    )


def stringify_smart_metadata(metadata: SmartMemoryMetadata) -> str:
    """JSON-encode a :class:`SmartMemoryMetadata` using snake_case keys.

    Array fields are capped to prevent metadata bloat:
    - ``sources``: last 20
    - ``history``: last 50
    - ``relations``: first 16
    """
    d: dict[str, Any] = {
        "l0_abstract": metadata.l0_abstract,
        "l1_overview": metadata.l1_overview,
        "l2_content": metadata.l2_content,
        "memory_category": metadata.memory_category,
        "tier": metadata.tier,
        "confidence": metadata.confidence,
        "access_count": metadata.access_count,
        "injected_count": metadata.injected_count,
        "bad_recall_count": metadata.bad_recall_count,
        "valid_from": metadata.valid_from,
        "cross_session": metadata.cross_session,
        "source": metadata.source,
        "source_session": metadata.source_session,
    }
    if metadata.valid_until is not None:
        d["valid_until"] = metadata.valid_until
    if metadata.invalidated_at is not None:
        d["invalidated_at"] = metadata.invalidated_at
    if metadata.fact_key is not None:
        d["fact_key"] = metadata.fact_key
    if metadata.supersedes is not None:
        d["supersedes"] = metadata.supersedes
    if metadata.superseded_by is not None:
        d["superseded_by"] = metadata.superseded_by

    relations = list(metadata.relations)
    if len(relations) > _MAX_RELATIONS:
        relations = relations[:_MAX_RELATIONS]
    d["relations"] = relations

    if metadata.support_info is not None:
        si = metadata.support_info
        slices_out = []
        for s in si.slices:
            slices_out.append({
                "context": s.context,
                "confirmations": s.confirmations,
                "contradictions": s.contradictions,
                "strength": s.strength,
                "last_observed_at": s.last_observed_at,
            })
        d["support_info"] = {
            "version": si.version,
            "global_strength": si.global_strength,
            "total_observations": si.total_observations,
            "slices": slices_out,
        }

    # Passthrough extras (apply array caps to sources/history if present)
    for k, v in metadata.extras.items():
        if k == "sources" and isinstance(v, list) and len(v) > _MAX_SOURCES:
            d[k] = v[-_MAX_SOURCES:]
        elif k == "history" and isinstance(v, list) and len(v) > _MAX_HISTORY:
            d[k] = v[-_MAX_HISTORY:]
        else:
            d[k] = v

    return json.dumps(d)


# ---------------------------------------------------------------------------
# Relation helpers
# ---------------------------------------------------------------------------

def append_relation(
    existing: list[dict],
    relation_type: str,
    target_id: str,
) -> list[dict]:
    """Append a ``{type, target_id}`` relation, deduplicating by (type, target_id)."""
    rows: list[dict] = [
        item for item in (existing or [])
        if (
            isinstance(item, dict)
            and isinstance(item.get("type"), str)
            and isinstance(item.get("target_id"), str)
        )
    ]
    if any(r["type"] == relation_type and r["target_id"] == target_id for r in rows):
        return rows
    return rows + [{"type": relation_type, "target_id": target_id}]


# ---------------------------------------------------------------------------
# Support-info parsing and update
# ---------------------------------------------------------------------------

def parse_support_info(raw: dict | str | None) -> SupportInfoV2 | None:
    """Parse a possibly-stale v1 or v2 support_info representation.

    v1 had a flat structure (``confirmations`` / ``contradictions`` at the
    top level); these are upgraded to v2 by treating them as a single
    ``"unknown"`` context slice.

    Returns None for missing/invalid input.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return None
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(raw, dict):
        obj = raw
    else:
        return None

    # v2: has a slices array
    if isinstance(obj.get("slices"), list):
        slices: list[SupportSlice] = []
        for s in obj["slices"]:
            if not isinstance(s, dict) or not isinstance(s.get("context"), str):
                continue
            conf = s.get("confirmations", 0)
            contra = s.get("contradictions", 0)
            strength_raw = s.get("strength", 0.5)
            slices.append(SupportSlice(
                context=str(s["context"]),
                confirmations=conf if isinstance(conf, int) and conf >= 0 else 0,
                contradictions=contra if isinstance(contra, int) and contra >= 0 else 0,
                strength=(
                    float(strength_raw)
                    if isinstance(strength_raw, (int, float))
                    and 0.0 <= float(strength_raw) <= 1.0
                    else 0.5
                ),
                last_observed_at=(
                    int(s["last_observed_at"])
                    if isinstance(s.get("last_observed_at"), (int, float))
                    else int(time.time() * 1000)
                ),
            ))
        gs = obj.get("global_strength", 0.5)
        to = obj.get("total_observations", 0)
        return SupportInfoV2(
            version=2,
            global_strength=float(gs) if isinstance(gs, (int, float)) else 0.5,
            total_observations=int(to) if isinstance(to, (int, float)) else 0,
            slices=slices,
        )

    # v1 flat format: { confirmations, contradictions }
    conf_v1 = obj.get("confirmations", 0)
    contra_v1 = obj.get("contradictions", 0)
    conf_n = int(conf_v1) if isinstance(conf_v1, (int, float)) and conf_v1 >= 0 else 0
    contra_n = int(contra_v1) if isinstance(contra_v1, (int, float)) and contra_v1 >= 0 else 0
    total = conf_n + contra_n
    if total == 0:
        return SupportInfoV2(
            version=2,
            global_strength=0.5,
            total_observations=0,
            slices=[],
        )
    strength = conf_n / total
    now_ms = int(time.time() * 1000)
    return SupportInfoV2(
        version=2,
        global_strength=strength,
        total_observations=total,
        slices=[SupportSlice(
            context="unknown",
            confirmations=conf_n,
            contradictions=contra_n,
            strength=strength,
            last_observed_at=now_ms,
        )],
    )


def update_support_stats(
    existing: SupportInfoV2 | None,
    context_label: str | None,
    event: Literal["support", "contradict"],
) -> SupportInfoV2:
    """Increment confirmations or contradictions for a context slice.

    Creates the slice if it is missing.  Recalculates ``global_strength``
    and ``total_observations`` after the update.

    Slices are capped at :data:`MAX_SUPPORT_SLICES`.  When at the cap and a
    *new* context arrives, the slice with the oldest ``last_observed_at`` is
    dropped, but its evidence counts remain in ``total_observations``
    (slight drift in ``global_strength`` is an accepted trade-off).
    """
    ctx = _normalise_context(context_label) or "general"

    # Deep-copy existing slices
    if existing is None:
        base_slices: list[SupportSlice] = []
    else:
        base_slices = [
            SupportSlice(
                context=s.context,
                confirmations=s.confirmations,
                contradictions=s.contradictions,
                strength=s.strength,
                last_observed_at=s.last_observed_at,
            )
            for s in existing.slices
        ]

    now_ms = int(time.time() * 1000)

    # Find or create the context slice
    target_slice: SupportSlice | None = None
    for s in base_slices:
        if s.context == ctx:
            target_slice = s
            break

    if target_slice is None:
        target_slice = SupportSlice(
            context=ctx,
            confirmations=0,
            contradictions=0,
            strength=0.5,
            last_observed_at=now_ms,
        )
        base_slices.append(target_slice)

    # Update slice
    if event == "support":
        target_slice.confirmations += 1
    else:
        target_slice.contradictions += 1

    slice_total = target_slice.confirmations + target_slice.contradictions
    target_slice.strength = (
        target_slice.confirmations / slice_total if slice_total > 0 else 0.5
    )
    target_slice.last_observed_at = now_ms

    # Cap slices: sort newest-first, drop oldest beyond the cap
    slices = base_slices
    dropped_conf = 0
    dropped_contra = 0
    if len(slices) > MAX_SUPPORT_SLICES:
        slices = sorted(slices, key=lambda s: s.last_observed_at, reverse=True)
        dropped = slices[MAX_SUPPORT_SLICES:]
        for d in dropped:
            dropped_conf += d.confirmations
            dropped_contra += d.contradictions
        slices = slices[:MAX_SUPPORT_SLICES]

    # Recompute global_strength including evidence from dropped slices
    total_conf = dropped_conf
    total_contra = dropped_contra
    for s in slices:
        total_conf += s.confirmations
        total_contra += s.contradictions

    total_obs = total_conf + total_contra
    global_strength = total_conf / total_obs if total_obs > 0 else 0.5

    return SupportInfoV2(
        version=2,
        global_strength=global_strength,
        total_observations=total_obs,
        slices=slices,
    )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "MAX_SUPPORT_SLICES",
    "SmartMemoryMetadata",
    "SupportInfoV2",
    "SupportSlice",
    "_CONTEXT_ALIASES",
    "_normalise_context",
    "append_relation",
    "build_smart_metadata",
    "derive_fact_key",
    "is_memory_active_at",
    "is_memory_expired",
    "parse_smart_metadata",
    "parse_support_info",
    "stringify_smart_metadata",
    "update_support_stats",
]
