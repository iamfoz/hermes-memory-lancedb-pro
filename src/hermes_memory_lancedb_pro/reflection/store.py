"""Reflection storage orchestrator.

Ports CortexReach's `reflection-store.ts`. Responsibilities:

  Write path: parse reflection markdown → build event/item/legacy payloads
              → embed → dedupe (combined-legacy only) → write.

  Read path:  filter reflection rows by agent ownership + resolved-item
              suppression → rank by logistic decay → return top invariants
              (≤ 8) and derived (≤ 10) lines, plus mapped slices grouped
              by kind (user-model, agent-model, lesson, decision).

This module does NOT call an LLM. The LLM that *generates* the reflection
markdown is the caller's job — typically PR 3's smart_extractor.

Persistence is dependency-injected via `ReflectionStoreAdapter` so the
package stays usable without LanceDB at import time and so tests can
substitute fakes. The default adapter wraps `MemoryStore` and uses its
`store_raw` + `search_by_vector` low-level helpers (which bypass the
default metadata machinery — reflection rows have their own schema).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .event_store import (
    build_reflection_event_payload,
    create_reflection_event_id,
)
from .item_store import (
    build_reflection_item_payloads,
    get_reflection_item_decay_defaults,
)
from .mapped_metadata import (
    ReflectionMappedKind,
    get_reflection_mapped_decay_defaults,
)
from .metadata import parse_reflection_metadata
from .ranking import (
    ReflectionScoreInput,
    compute_reflection_score,
    normalize_reflection_line_for_aggregation,
)
from .slices import (
    ReflectionSliceItem,
    ReflectionSlices,
    extract_reflection_slices,
    sanitize_injectable_reflection_lines,
    sanitize_reflection_slice_lines,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — keep these in sync with the TS source
# ---------------------------------------------------------------------------

REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS = 3
REFLECTION_DERIVE_LOGISTIC_K = 1.2
REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT = 0.35

DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000
DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS = 60 * 24 * 60 * 60 * 1000

REFLECTION_CATEGORY = "reflection"

ReflectionStoreKind = Literal[
    "event", "item-invariant", "item-derived", "combined-legacy"
]


# ---------------------------------------------------------------------------
# Adapter — the only surface that touches LanceDB
# ---------------------------------------------------------------------------

@runtime_checkable
class ReflectionStoreAdapter(Protocol):
    """Three-method persistence Protocol for the reflection write/read path.
    Default impl wraps `MemoryStore`; tests can substitute a fake."""

    def embed_passage(self, text: str) -> list[float]: ...

    def vector_search(
        self,
        vector: Sequence[float],
        limit: int = 10,
        *,
        scope: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def store_entry(
        self,
        *,
        text: str,
        vector: Sequence[float],
        category: str,
        scope: str,
        importance: float,
        metadata: str,  # JSON-encoded string
    ) -> str: ...


class MemoryStoreReflectionAdapter:
    """Default adapter wrapping a `MemoryStore`. Uses the low-level
    `store_raw` + `search_by_vector` helpers to skip re-encoding and
    keep reflection metadata under our control."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def embed_passage(self, text: str) -> list[float]:
        return self._store.encode(text)

    def vector_search(
        self,
        vector: Sequence[float],
        limit: int = 10,
        *,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.search_by_vector(
            vector, limit=limit, scope=scope, keep_vector=False,
        )

    def store_entry(
        self,
        *,
        text: str,
        vector: Sequence[float],
        category: str,
        scope: str,
        importance: float,
        metadata: str,
    ) -> str:
        return self._store.store_raw(
            text=text, vector=vector, category=category, scope=scope,
            importance=importance, metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Payload + result types
# ---------------------------------------------------------------------------

@dataclass
class ReflectionStorePayload:
    text: str
    metadata: dict[str, Any]
    kind: ReflectionStoreKind


@dataclass
class ReflectionStoreResult:
    stored: bool
    event_id: str
    slices: ReflectionSlices
    stored_kinds: list[ReflectionStoreKind] = field(default_factory=list)


@dataclass
class ReflectionMappedSlices:
    user_model: list[str] = field(default_factory=list)
    agent_model: list[str] = field(default_factory=list)
    lesson: list[str] = field(default_factory=list)
    decision: list[str] = field(default_factory=list)


@dataclass
class LoadReflectionSlicesResult:
    invariants: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_error_signals(signals: Sequence[Any]) -> list[dict[str, str]]:
    """Accept either dicts (`{"signature_hash": ...}` or
    `{"signatureHash": ...}`) or objects with a `.signature_hash` /
    `.signatureHash` attribute. Returns a list of dicts in the snake_case
    form the sub-modules expect."""
    out: list[dict[str, str]] = []
    for s in signals or ():
        h = ""
        if isinstance(s, dict):
            h = s.get("signature_hash") or s.get("signatureHash") or ""
        else:
            h = getattr(s, "signature_hash", None) or getattr(s, "signatureHash", "")
        if h:
            out.append({"signature_hash": str(h)})
    return out


def _payload_metadata_dict(meta_obj: Any) -> dict[str, Any]:
    """Convert a payload metadata dataclass / dict to a JSON-friendly dict."""
    if hasattr(meta_obj, "__dataclass_fields__"):
        return asdict(meta_obj)
    if isinstance(meta_obj, dict):
        return dict(meta_obj)
    return {"raw": str(meta_obj)}


def _slice_items_to_item_store_input(
    slices: ReflectionSlices,
) -> list[dict[str, Any]]:
    """Convert ``slices.ReflectionSliceItem`` (text, kind) into the dict
    shape `build_reflection_item_payloads` expects (text, item_kind,
    section, ordinal, group_size)."""
    out: list[dict[str, Any]] = []
    inv_total = len(slices.invariants)
    for i, item in enumerate(slices.invariants):
        out.append({
            "text": item.text,
            "item_kind": "invariant",
            "section": "invariants",
            "ordinal": i + 1,
            "group_size": inv_total,
        })
    der_total = len(slices.derived)
    for i, item in enumerate(slices.derived):
        out.append({
            "text": item.text,
            "item_kind": "derived",
            "section": "derived",
            "ordinal": i + 1,
            "group_size": der_total,
        })
    return out


def _injectable_slices(reflection_text: str) -> ReflectionSlices:
    """`extract_reflection_slices` followed by strict injection
    sanitisation — equivalent to TS `extractInjectableReflectionSlices`."""
    raw = extract_reflection_slices(reflection_text)
    inv = [
        ReflectionSliceItem(text=s, kind="invariant")
        for s in sanitize_injectable_reflection_lines(
            [it.text for it in raw.invariants]
        )
    ]
    der = [
        ReflectionSliceItem(text=s, kind="derived")
        for s in sanitize_injectable_reflection_lines(
            [it.text for it in raw.derived]
        )
    ]
    return ReflectionSlices(invariants=inv, derived=der)


# ---------------------------------------------------------------------------
# Build payloads
# ---------------------------------------------------------------------------

def build_reflection_store_payloads(
    *,
    reflection_text: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    scope: str,
    tool_error_signals: Sequence[Any] = (),
    run_at: int,
    used_fallback: bool = False,
    event_id: str | None = None,
    source_reflection_path: str | None = None,
    write_legacy_combined: bool = True,
) -> tuple[str, ReflectionSlices, list[ReflectionStorePayload]]:
    """Build all payloads for a reflection run.

    Returns `(event_id, slices, payloads)`. Payloads include:
      - one `event` (always)
      - one `item-invariant` / `item-derived` per extracted line
      - one `combined-legacy` (only when `write_legacy_combined=True`
        and at least one line was extracted)

    The caller decides whether to actually persist them."""
    slices = _injectable_slices(reflection_text)
    eid = event_id or create_reflection_event_id(
        run_at=run_at, session_key=session_key, session_id=session_id,
        agent_id=agent_id, command=command,
    )
    error_signals = _normalize_error_signals(tool_error_signals)

    # Event payload — always emitted.
    event_payload_obj = build_reflection_event_payload(
        scope=scope, session_key=session_key, session_id=session_id,
        agent_id=agent_id, command=command, run_at=run_at,
        used_fallback=used_fallback, tool_error_signals=error_signals,
        event_id=eid, source_reflection_path=source_reflection_path,
    )
    payloads: list[ReflectionStorePayload] = [
        ReflectionStorePayload(
            text=event_payload_obj.text,
            metadata=_payload_metadata_dict(event_payload_obj.metadata),
            kind="event",
        )
    ]

    # One per item.
    item_payloads = build_reflection_item_payloads(
        items=_slice_items_to_item_store_input(slices),
        event_id=eid, agent_id=agent_id,
        session_key=session_key, session_id=session_id,
        run_at=run_at, used_fallback=used_fallback,
        tool_error_signals=error_signals,
        source_reflection_path=source_reflection_path,
    )
    for ip in item_payloads:
        item_kind = getattr(ip.metadata, "item_kind", None)
        kind: ReflectionStoreKind = (
            "item-invariant" if item_kind == "invariant" else "item-derived"
        )
        payloads.append(
            ReflectionStorePayload(
                text=ip.text,
                metadata=_payload_metadata_dict(ip.metadata),
                kind=kind,
            )
        )

    # Optional v3 combined-legacy payload.
    if write_legacy_combined and (slices.invariants or slices.derived):
        payloads.append(_build_legacy_combined_payload(
            slices=slices, scope=scope, session_key=session_key,
            session_id=session_id, agent_id=agent_id, command=command,
            error_signals=error_signals, run_at=run_at,
            used_fallback=used_fallback,
            source_reflection_path=source_reflection_path,
        ))

    return eid, slices, payloads


def _build_legacy_combined_payload(
    *,
    slices: ReflectionSlices,
    scope: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    error_signals: list[dict[str, str]],
    run_at: int,
    used_fallback: bool,
    source_reflection_path: str | None,
) -> ReflectionStorePayload:
    """Legacy `memory-reflection` (v3) payload: invariants + derived
    bundled into a single entry. Kept for backward compat with stores
    written by older versions; new writes still emit it (disable via
    `write_legacy_combined=False`)."""
    iso = datetime.fromtimestamp(run_at / 1000, tz=UTC).isoformat()
    date_ymd = iso.split("T", 1)[0]

    derive_quality = compute_derived_line_quality(len(slices.derived))
    derive_base_weight = (
        REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT if used_fallback else 1.0
    )

    invariant_lines = (
        [f"- {it.text}" for it in slices.invariants]
        if slices.invariants else ["- (none captured)"]
    )
    derived_lines = (
        [f"- {it.text}" for it in slices.derived]
        if slices.derived else ["- (none captured)"]
    )

    text = "\n".join([
        f"reflection · {scope} · {date_ymd}",
        f"Session Reflection ({iso})",
        f"Session Key: {session_key}",
        f"Session ID: {session_id}",
        "",
        "Invariants:",
        *invariant_lines,
        "",
        "Derived:",
        *derived_lines,
    ])

    metadata: dict[str, Any] = {
        "type": "memory-reflection",
        "stage": "reflect-store",
        "reflection_version": 3,
        "scope": scope,
        "session_key": session_key,
        "session_id": session_id,
        "agent_id": agent_id,
        "command": command,
        "stored_at": run_at,
        "invariants": [it.text for it in slices.invariants],
        "derived": [it.text for it in slices.derived],
        "used_fallback": used_fallback,
        "error_signals": [s["signature_hash"] for s in error_signals],
        "decay_model": "logistic",
        "decay_midpoint_days": REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS,
        "decay_k": REFLECTION_DERIVE_LOGISTIC_K,
        "derive_base_weight": derive_base_weight,
        "derive_quality": derive_quality,
        "derive_source": "fallback" if used_fallback else "normal",
    }
    if source_reflection_path:
        metadata["source_reflection_path"] = source_reflection_path

    return ReflectionStorePayload(text=text, metadata=metadata, kind="combined-legacy")


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def _resolve_reflection_importance(kind: ReflectionStoreKind) -> float:
    if kind == "event":
        return 0.55
    if kind == "item-invariant":
        return 0.82
    if kind == "item-derived":
        return 0.78
    return 0.75  # combined-legacy


def store_reflection_to_lancedb(
    adapter: ReflectionStoreAdapter,
    *,
    reflection_text: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    scope: str,
    tool_error_signals: Sequence[Any] = (),
    run_at: int,
    used_fallback: bool = False,
    event_id: str | None = None,
    source_reflection_path: str | None = None,
    write_legacy_combined: bool = True,
    dedupe_threshold: float = 0.97,
) -> ReflectionStoreResult:
    """End-to-end reflection write. Embeds each payload, dedupes the
    combined-legacy payload against existing entries (cosine ≥ threshold
    is considered a duplicate and skipped), and writes the rest."""
    eid, slices, payloads = build_reflection_store_payloads(
        reflection_text=reflection_text, session_key=session_key,
        session_id=session_id, agent_id=agent_id, command=command,
        scope=scope, tool_error_signals=tool_error_signals, run_at=run_at,
        used_fallback=used_fallback, event_id=event_id,
        source_reflection_path=source_reflection_path,
        write_legacy_combined=write_legacy_combined,
    )

    stored_kinds: list[ReflectionStoreKind] = []
    for payload in payloads:
        try:
            vector = adapter.embed_passage(payload.text)
        except Exception as e:
            logger.warning("reflection embed failed (kind=%s): %s", payload.kind, e)
            continue

        if payload.kind == "combined-legacy":
            try:
                existing = adapter.vector_search(vector, limit=1, scope=scope)
            except Exception as e:
                logger.warning("reflection dedup search failed: %s", e)
                existing = []
            if existing:
                # `_distance` is cosine distance (lower = more similar).
                # similarity = 1 - distance ≥ threshold → skip the write.
                # Use an explicit None check — `0.0` is falsy and represents a
                # *perfect* match, exactly the case we want to dedupe.
                raw_dist = existing[0].get("_distance")
                top_distance = float(raw_dist) if raw_dist is not None else 1.0
                if (1.0 - top_distance) >= dedupe_threshold:
                    continue

        try:
            adapter.store_entry(
                text=payload.text, vector=vector,
                category=REFLECTION_CATEGORY, scope=scope,
                importance=_resolve_reflection_importance(payload.kind),
                metadata=json.dumps(payload.metadata),
            )
            stored_kinds.append(payload.kind)
        except Exception as e:
            logger.warning("reflection store failed (kind=%s): %s", payload.kind, e)

    return ReflectionStoreResult(
        stored=bool(stored_kinds),
        event_id=eid,
        slices=slices,
        stored_kinds=stored_kinds,
    )


# ---------------------------------------------------------------------------
# Ownership guard
# ---------------------------------------------------------------------------

def is_owned_by_agent(metadata: dict[str, Any], agent_id: str) -> bool:
    """Decide whether a stored reflection row is visible to `agent_id`.

    Rules (from CortexReach spec):
      - `derived` items: strict ownership — must match exactly. Empty
        owner → invisible (prevents leak via missing-owner rows).
      - Malformed `item_kind` (string but not derived/invariant; or
        non-string non-None value) → fail-closed (invisible).
      - Otherwise (invariant, legacy, mapped, missing item_kind):
        empty owner is allowed (back-compat with pre-ownership rows);
        non-empty owner must match agent_id OR equal the literal "main"
        (legacy fallback path)."""
    owner_raw = metadata.get("agent_id")
    owner = owner_raw.strip() if isinstance(owner_raw, str) else ""
    item_kind = metadata.get("item_kind")

    if isinstance(item_kind, str):
        if item_kind == "derived":
            if not owner:
                return False
            return owner == agent_id
        # Non-derived string item_kind falls through to the legacy path
    elif item_kind is not None:
        # item_kind is neither string nor None → malformed → fail closed
        return False

    if not owner:
        return True
    return owner in (agent_id, "main")


# ---------------------------------------------------------------------------
# Loaders — rank reflection rows by logistic decay
# ---------------------------------------------------------------------------

def _read_positive_number(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(n) or n <= 0:
        return fallback
    return n


def _read_clamped_number(value: Any, fallback: float, lo: float, hi: float) -> float:
    try:
        n = float(value)
        if math.isnan(n):
            n = fallback
    except (TypeError, ValueError):
        n = fallback
    return max(lo, min(hi, n))


def _metadata_timestamp(metadata: dict[str, Any], fallback_ts: int) -> int:
    stored = metadata.get("stored_at")
    if isinstance(stored, (int, float)) and stored > 0:
        return int(stored)
    return int(fallback_ts) if fallback_ts else 0


def _to_string_array(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def compute_derived_line_quality(non_placeholder_line_count: int) -> float:
    """Quality scales with how many non-placeholder derived lines a row
    yielded. Floor 0.2, ceiling 1.0 at 6+ lines. Mirrors TS exactly:
    `0.55 + min(6, n) * 0.075` for n > 0."""
    n = max(0, int(non_placeholder_line_count))
    if n <= 0:
        return 0.2
    return min(1.0, 0.55 + min(6, n) * 0.075)


def _resolve_legacy_derive_base_weight(metadata: dict[str, Any]) -> float:
    explicit = metadata.get("derive_base_weight")
    try:
        v = float(explicit)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isnan(v) and v > 0:
        return max(0.1, min(1.2, v))
    if metadata.get("used_fallback") is True:
        return REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT
    return 1.0


@dataclass
class _WeightedLineCandidate:
    line: str
    timestamp: int
    midpoint_days: float
    k: float
    base_weight: float
    quality: float
    used_fallback: bool


def load_agent_reflection_slices_from_entries(
    *,
    entries: Sequence[dict[str, Any]],
    agent_id: str,
    now_ms: int | None = None,
    derive_max_age_ms: int | None = None,
    invariant_max_age_ms: int | None = None,
) -> LoadReflectionSlicesResult:
    """Filter reflection rows by ownership + resolved-item suppression,
    then rank into the top invariants (≤ 8) and derived lines (≤ 10).

    Implements the P1 / P2 fixes from the TS: when all item rows have
    been resolved AND legacy rows would only re-expose already-resolved
    content, suppress the entire section (prevents the reflection
    fallback path from reviving advice the user has already moved on
    from)."""
    now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
    derive_max = (
        DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS
        if derive_max_age_ms is None
        else max(0, int(derive_max_age_ms))
    )
    invariant_max = (
        None if invariant_max_age_ms is None
        else max(0, int(invariant_max_age_ms))
    )

    annotated = [
        (e, parse_reflection_metadata(e.get("metadata")))
        for e in entries
    ]
    reflection_rows = [
        (e, m) for (e, m) in annotated
        if m.get("type") in ("memory-reflection-item", "memory-reflection")
        and is_owned_by_agent(m, agent_id)
    ]
    reflection_rows.sort(
        key=lambda em: int(em[0].get("timestamp", 0) or 0),
        reverse=True,
    )
    reflection_rows = reflection_rows[:160]

    item_rows = [(e, m) for (e, m) in reflection_rows
                 if m.get("type") == "memory-reflection-item"]
    legacy_rows = [(e, m) for (e, m) in reflection_rows
                   if m.get("type") == "memory-reflection"]

    unresolved = [(e, m) for (e, m) in item_rows if m.get("resolved_at") is None]
    resolved = [(e, m) for (e, m) in item_rows if m.get("resolved_at") is not None]

    has_items = bool(item_rows)
    has_legacy = bool(legacy_rows)

    resolved_invariant_texts = {
        normalize_reflection_line_for_aggregation(s)
        for (e, m) in resolved
        if m.get("item_kind") == "invariant"
        for s in sanitize_injectable_reflection_lines([e.get("text", "")])
    }
    resolved_derived_texts = {
        normalize_reflection_line_for_aggregation(s)
        for (e, m) in resolved
        if m.get("item_kind") == "derived"
        for s in sanitize_injectable_reflection_lines([e.get("text", "")])
    }

    legacy_has_unique_invariant = any(
        any(
            normalize_reflection_line_for_aggregation(line) not in resolved_invariant_texts
            for line in sanitize_injectable_reflection_lines(
                _to_string_array(m.get("invariants"))
            )
        )
        for (_, m) in legacy_rows
    )
    legacy_has_unique_derived = any(
        any(
            normalize_reflection_line_for_aggregation(line) not in resolved_derived_texts
            for line in sanitize_injectable_reflection_lines(
                _to_string_array(m.get("derived"))
            )
        )
        for (_, m) in legacy_rows
    )

    # P1: full suppression
    should_suppress = (
        has_items
        and not unresolved
        and (
            not has_legacy
            or (not legacy_has_unique_invariant and not legacy_has_unique_derived)
        )
    )
    if should_suppress:
        return LoadReflectionSlicesResult()

    # P2: per-section legacy filtering — only let legacy rows through if
    # they have at least one line that isn't already resolved
    invariant_legacy = [
        (e, m) for (e, m) in legacy_rows
        if any(
            normalize_reflection_line_for_aggregation(line) not in resolved_invariant_texts
            for line in sanitize_injectable_reflection_lines(
                _to_string_array(m.get("invariants"))
            )
        )
    ]
    derived_legacy = [
        (e, m) for (e, m) in legacy_rows
        if any(
            normalize_reflection_line_for_aggregation(line) not in resolved_derived_texts
            for line in sanitize_injectable_reflection_lines(
                _to_string_array(m.get("derived"))
            )
        )
    ]

    inv_candidates = _build_invariant_candidates(
        unresolved, invariant_legacy, resolved_invariant_texts,
    )
    der_candidates = _build_derived_candidates(
        unresolved, derived_legacy, agent_id, resolved_derived_texts,
    )

    invariants = _rank_reflection_lines(
        inv_candidates, now=now, max_age_ms=invariant_max, limit=8,
    )
    derived = _rank_reflection_lines(
        der_candidates, now=now, max_age_ms=derive_max, limit=10,
    )
    return LoadReflectionSlicesResult(invariants=invariants, derived=derived)


def _build_invariant_candidates(
    item_rows: Sequence[tuple[dict, dict]],
    legacy_rows: Sequence[tuple[dict, dict]],
    resolved_texts: set[str],
) -> list[_WeightedLineCandidate]:
    out: list[_WeightedLineCandidate] = []
    for (entry, metadata) in item_rows:
        if metadata.get("item_kind") != "invariant":
            continue
        safe_lines = sanitize_injectable_reflection_lines([entry.get("text", "")])
        if not safe_lines:
            continue
        defaults = get_reflection_item_decay_defaults("invariant")
        ts = _metadata_timestamp(metadata, int(entry.get("timestamp", 0) or 0))
        for line in safe_lines:
            out.append(_WeightedLineCandidate(
                line=line, timestamp=ts,
                midpoint_days=_read_positive_number(metadata.get("decay_midpoint_days"), defaults["midpoint_days"]),
                k=_read_positive_number(metadata.get("decay_k"), defaults["k"]),
                base_weight=_read_positive_number(metadata.get("base_weight"), defaults["base_weight"]),
                quality=_read_clamped_number(metadata.get("quality"), defaults["quality"], 0.2, 1.0),
                used_fallback=metadata.get("used_fallback") is True,
            ))
    if out:
        return out

    # Legacy fallback
    for (entry, metadata) in legacy_rows:
        defaults = get_reflection_item_decay_defaults("invariant")
        ts = _metadata_timestamp(metadata, int(entry.get("timestamp", 0) or 0))
        lines = sanitize_injectable_reflection_lines(
            _to_string_array(metadata.get("invariants"))
        )
        for line in lines:
            if normalize_reflection_line_for_aggregation(line) in resolved_texts:
                continue
            out.append(_WeightedLineCandidate(
                line=line, timestamp=ts,
                midpoint_days=defaults["midpoint_days"],
                k=defaults["k"],
                base_weight=defaults["base_weight"],
                quality=defaults["quality"],
                used_fallback=metadata.get("used_fallback") is True,
            ))
    return out


def _build_derived_candidates(
    item_rows: Sequence[tuple[dict, dict]],
    legacy_rows: Sequence[tuple[dict, dict]],
    agent_id: str,
    resolved_texts: set[str],
) -> list[_WeightedLineCandidate]:
    out: list[_WeightedLineCandidate] = []
    for (entry, metadata) in item_rows:
        if metadata.get("item_kind") != "derived":
            continue
        safe_lines = sanitize_injectable_reflection_lines([entry.get("text", "")])
        if not safe_lines:
            continue
        defaults = get_reflection_item_decay_defaults("derived")
        ts = _metadata_timestamp(metadata, int(entry.get("timestamp", 0) or 0))
        for line in safe_lines:
            out.append(_WeightedLineCandidate(
                line=line, timestamp=ts,
                midpoint_days=_read_positive_number(metadata.get("decay_midpoint_days"), defaults["midpoint_days"]),
                k=_read_positive_number(metadata.get("decay_k"), defaults["k"]),
                base_weight=_read_positive_number(metadata.get("base_weight"), defaults["base_weight"]),
                quality=_read_clamped_number(metadata.get("quality"), defaults["quality"], 0.2, 1.0),
                used_fallback=metadata.get("used_fallback") is True,
            ))
    if out:
        return out

    # Legacy fallback for derived: row visible iff:
    #   - has no derived content (treat as a pure-invariant legacy row), OR
    #   - has derived AND owner equals agent_id (NOT "main" — main's
    #     derived must NEVER leak to subagents per the CortexReach fix).
    for (entry, metadata) in legacy_rows:
        derived = metadata.get("derived")
        has_derived = isinstance(derived, list) and bool(derived)
        if has_derived:
            owner_raw = metadata.get("agent_id")
            owner = owner_raw.strip() if isinstance(owner_raw, str) else ""
            if not owner or owner == "main" or owner != agent_id:
                continue

        ts = _metadata_timestamp(metadata, int(entry.get("timestamp", 0) or 0))
        lines = sanitize_injectable_reflection_lines(
            _to_string_array(metadata.get("derived"))
        )
        if not lines:
            continue
        defaults_md = REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS
        defaults_k = REFLECTION_DERIVE_LOGISTIC_K
        defaults_bw = _resolve_legacy_derive_base_weight(metadata)
        defaults_q = compute_derived_line_quality(len(lines))

        for line in lines:
            if normalize_reflection_line_for_aggregation(line) in resolved_texts:
                continue
            out.append(_WeightedLineCandidate(
                line=line, timestamp=ts,
                midpoint_days=_read_positive_number(metadata.get("decay_midpoint_days"), defaults_md),
                k=_read_positive_number(metadata.get("decay_k"), defaults_k),
                base_weight=_read_positive_number(metadata.get("derive_base_weight"), defaults_bw),
                quality=_read_clamped_number(metadata.get("derive_quality"), defaults_q, 0.2, 1.0),
                used_fallback=metadata.get("used_fallback") is True,
            ))
    return out


def _rank_reflection_lines(
    candidates: Sequence[_WeightedLineCandidate],
    *,
    now: int,
    max_age_ms: int | None,
    limit: int,
) -> list[str]:
    """Aggregate scores across duplicate normalised lines and return
    the top `limit`. Ties broken by latest timestamp, then lex order."""

    @dataclass
    class _Bucket:
        line: str
        score: float
        latest_ts: int

    scores: dict[str, _Bucket] = {}
    for c in candidates:
        ts = c.timestamp if c.timestamp else now
        if max_age_ms is not None and (now - ts) > max_age_ms:
            continue
        age_days = max(0.0, (now - ts) / 86_400_000)
        s = compute_reflection_score(ReflectionScoreInput(
            age_days=age_days, midpoint_days=c.midpoint_days, k=c.k,
            base_weight=c.base_weight, quality=c.quality,
            used_fallback=c.used_fallback,
        ))
        if math.isnan(s) or s <= 0:
            continue
        key = normalize_reflection_line_for_aggregation(c.line)
        if not key:
            continue
        cur = scores.get(key)
        if cur is None:
            scores[key] = _Bucket(line=c.line, score=s, latest_ts=ts)
        else:
            cur.score += s
            if ts > cur.latest_ts:
                cur.latest_ts = ts
                cur.line = c.line

    ordered = sorted(
        scores.values(),
        key=lambda b: (-b.score, -b.latest_ts, b.line),
    )
    return [b.line for b in ordered[:limit]]


# ---------------------------------------------------------------------------
# Mapped row loader (user-model / agent-model / lesson / decision)
# ---------------------------------------------------------------------------

def _parse_mapped_kind(value: Any) -> ReflectionMappedKind | None:
    if value in ("user-model", "agent-model", "lesson", "decision"):
        return value  # type: ignore[return-value]
    return None


def load_reflection_mapped_rows_from_entries(
    *,
    entries: Sequence[dict[str, Any]],
    agent_id: str,
    now_ms: int | None = None,
    max_age_ms: int | None = None,
    max_per_kind: int = 10,
) -> ReflectionMappedSlices:
    """Filter mapped reflection rows by ownership, score by logistic
    decay, and group by kind. Returns up to `max_per_kind` lines per
    kind, sorted by score desc."""
    now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
    max_age = (
        DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS if max_age_ms is None
        else max(0, int(max_age_ms))
    )
    cap = max(1, int(max_per_kind))

    @dataclass
    class _Group:
        text: str
        score: float
        latest_ts: int
        kind: ReflectionMappedKind

    grouped: dict[str, _Group] = {}

    for entry in entries:
        metadata = parse_reflection_metadata(entry.get("metadata"))
        if metadata.get("type") != "memory-reflection-mapped":
            continue
        if not is_owned_by_agent(metadata, agent_id):
            continue
        mapped_kind = _parse_mapped_kind(metadata.get("kind") or metadata.get("mapped_kind"))
        if not mapped_kind:
            continue

        lines = sanitize_reflection_slice_lines([entry.get("text", "")])
        if not lines:
            continue

        defaults = get_reflection_mapped_decay_defaults(mapped_kind)
        ts = _metadata_timestamp(metadata, int(entry.get("timestamp", 0) or 0))
        if (now - ts) > max_age:
            continue

        for line in lines:
            age_days = max(0.0, (now - ts) / 86_400_000)
            score = compute_reflection_score(ReflectionScoreInput(
                age_days=age_days,
                midpoint_days=_read_positive_number(metadata.get("decay_midpoint_days"), defaults["midpoint_days"]),
                k=_read_positive_number(metadata.get("decay_k"), defaults["k"]),
                base_weight=_read_positive_number(metadata.get("base_weight"), defaults["base_weight"]),
                quality=_read_clamped_number(metadata.get("quality"), defaults["quality"], 0.2, 1.0),
                used_fallback=metadata.get("used_fallback") is True,
            ))
            if math.isnan(score) or score <= 0:
                continue
            normalized = normalize_reflection_line_for_aggregation(line)
            if not normalized:
                continue
            key = f"{mapped_kind}::{normalized}"
            cur = grouped.get(key)
            if cur is None:
                grouped[key] = _Group(text=line, score=score, latest_ts=ts, kind=mapped_kind)
            else:
                cur.score += score
                if ts > cur.latest_ts:
                    cur.latest_ts = ts
                    cur.text = line

    def _by_kind(kind: ReflectionMappedKind) -> list[str]:
        rows = sorted(
            (g for g in grouped.values() if g.kind == kind),
            key=lambda g: (-g.score, -g.latest_ts, g.text),
        )
        return [g.text for g in rows[:cap]]

    return ReflectionMappedSlices(
        user_model=_by_kind("user-model"),
        agent_model=_by_kind("agent-model"),
        lesson=_by_kind("lesson"),
        decision=_by_kind("decision"),
    )


def get_reflection_derived_decay_defaults() -> dict[str, float]:
    return get_reflection_item_decay_defaults("derived")


def get_reflection_invariant_decay_defaults() -> dict[str, float]:
    return get_reflection_item_decay_defaults("invariant")


__all__ = [
    "DEFAULT_REFLECTION_DERIVED_MAX_AGE_MS",
    "DEFAULT_REFLECTION_MAPPED_MAX_AGE_MS",
    "REFLECTION_CATEGORY",
    "REFLECTION_DERIVE_FALLBACK_BASE_WEIGHT",
    "REFLECTION_DERIVE_LOGISTIC_K",
    "REFLECTION_DERIVE_LOGISTIC_MIDPOINT_DAYS",
    "LoadReflectionSlicesResult",
    "MemoryStoreReflectionAdapter",
    "ReflectionMappedSlices",
    "ReflectionStoreAdapter",
    "ReflectionStoreKind",
    "ReflectionStorePayload",
    "ReflectionStoreResult",
    "build_reflection_store_payloads",
    "compute_derived_line_quality",
    "get_reflection_derived_decay_defaults",
    "get_reflection_invariant_decay_defaults",
    "is_owned_by_agent",
    "load_agent_reflection_slices_from_entries",
    "load_reflection_mapped_rows_from_entries",
    "store_reflection_to_lancedb",
]
