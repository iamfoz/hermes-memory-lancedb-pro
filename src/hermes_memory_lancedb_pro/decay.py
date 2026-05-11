"""
Decay engine, tier manager, noise filter, and MMR diversity.

Ported from CortexReach memory-lancedb-pro architecture:
  - Weibull stretched-exponential decay with importance-modulated half-life
  - Three-tier management: core / working / peripheral
  - Noise filtering for denials, meta-questions, boilerplate, extractor artifacts
  - MMR (Maximal Marginal Relevance) diversity demotion (token-Jaccard fallback
    when vectors aren't on the entry)
  - Scoring pipeline: length-norm -> hardMinScore -> composite decay -> sort
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MS_PER_DAY: int = 86_400_000

# Tier names (kept in sync with store.MEMORY_TIERS)
TIER_CORE = "core"
TIER_WORKING = "working"
TIER_PERIPHERAL = "peripheral"

# Weibull beta per tier — higher beta = faster late-stage decay
TIER_BETA: dict[str, float] = {
    TIER_CORE: 0.8,
    TIER_WORKING: 1.0,
    TIER_PERIPHERAL: 1.3,
}

# Soft retention floor per tier, expressed as a fraction of the maximum
# achievable composite. Applied at the END of compute_decay_score so that
# a high-tier memory never drops below a recall-priority minimum, but still
# leaves headroom below the working-tier demotion threshold.
TIER_FLOOR: dict[str, float] = {
    TIER_CORE: 0.45,
    TIER_WORKING: 0.20,
    TIER_PERIPHERAL: 0.0,
}


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

DENIAL_PATTERNS = [
    r"i don'?t have (any )?(information|data|memory|record)",
    r"i'?m not sure about",
    r"i don'?t recall",
    r"i don'?t remember",
    r"it looks like i don'?t",
    r"i wasn'?t able to find",
    r"no (relevant )?memories found",
    r"i don'?t have access to",
]

META_QUESTION_PATTERNS = [
    r"\bdo you (remember|recall|know about)\b",
    r"\bcan you (remember|recall)\b",
    r"\bdid i (tell|mention|say|share)\b",
    r"\bhave i (told|mentioned|said)\b",
    r"\bwhat did i (tell|say|mention)\b",
]

BOILERPLATE_PATTERNS = [
    r"^(hi|hello|hey|good morning|good evening|greetings)",
    r"^fresh session",
    r"^new session",
    r"^HEARTBEAT",
]

DIAGNOSTIC_ARTIFACT_PATTERNS = [
    r"\bquery\s*->\s*(none|no explicit solution|unknown|not found)\b",
    r"\buser asked for\b.*\b(none|no explicit solution|unknown|not found)\b",
    r"\bno explicit solution\b",
]

ENVELOPE_NOISE_PATTERNS = [
    r"^<<<EXTERNAL_UNTRUSTED_CONTENT\b",
    r"^<<<END_EXTERNAL_UNTRUSTED_CONTENT\b",
    r"^Sender\s*\(untrusted metadata\):",
    r"^Conversation info\s*\(untrusted metadata\):",
    r"^Thread starter\s*\(untrusted, for context\):",
    r"^Forwarded message context\s*\(untrusted metadata\):",
    r"^\[Queued messages while agent was busy\]",
]

_NOISE_RE: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        DENIAL_PATTERNS
        + META_QUESTION_PATTERNS
        + BOILERPLATE_PATTERNS
        + DIAGNOSTIC_ARTIFACT_PATTERNS
        + ENVELOPE_NOISE_PATTERNS
    )
]

MIN_TEXT_LEN = 10


def is_noise(text: str) -> bool:
    """Return True if `text` is too short or matches a noise pattern."""
    if not text or len(text.strip()) < MIN_TEXT_LEN:
        return True
    return any(pattern.search(text) for pattern in _NOISE_RE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_metadata(value: Any) -> dict[str, Any]:
    """Best-effort metadata parsing — accept dict, JSON string, or junk."""
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


def _extract_importance(entry: dict[str, Any], metadata: dict[str, Any]) -> float:
    """`importance` lives on the top-level row; metadata is a fallback only."""
    val = entry.get("importance")
    if val is None:
        val = metadata.get("importance", 0.5)
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Weibull decay
# ---------------------------------------------------------------------------

@dataclass
class DecayConfig:
    """Weibull stretched-exponential decay configuration."""
    recency_half_life_days: float = 30.0
    recency_weight: float = 0.4
    frequency_weight: float = 0.3
    intrinsic_weight: float = 0.3
    stale_threshold: float = 0.3
    search_boost_min: float = 0.3
    importance_modulation: float = 1.5
    apply_tier_floor: bool = True


def compute_decay_score(
    entry: dict[str, Any],
    *,
    now_ms: int | None = None,
    config: DecayConfig | None = None,
) -> dict[str, float]:
    """
    Compute Weibull composite decay score for a memory entry.

    `entry` may be either a flat row (with `importance` at the top level and
    a parsed `metadata` dict) or a metadata-only dict; both are accepted for
    backwards compatibility.

    Formula:
        composite = recency_weight * recency
                  + frequency_weight * frequency
                  + intrinsic_weight * intrinsic

    Recency: Weibull stretched-exponential, half-life modulated by importance.
    Frequency: logarithmic saturation of access_count plus recent-access bonus.
    Intrinsic: importance * confidence.

    Tier modulates the Weibull beta (decay shape) and applies a soft floor
    via TIER_FLOOR (disable with `config.apply_tier_floor=False`).
    """
    if config is None:
        config = DecayConfig()
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    metadata = _coerce_metadata(entry.get("metadata", entry))
    # If the caller passed a bare metadata dict, treat it as both
    is_metadata_only = "metadata" not in entry

    tier = metadata.get("tier", TIER_WORKING)
    importance = _extract_importance(
        {} if is_metadata_only else entry,
        metadata,
    )
    confidence = float(metadata.get("confidence", 0.8) or 0.8)
    access_count = int(metadata.get("access_count", 0) or 0)
    created_at = int(metadata.get("created_at", now_ms) or now_ms)
    last_accessed_at = int(metadata.get("last_accessed_at", created_at) or created_at)
    temporal_type = metadata.get("temporal_type", "static")

    # Recency
    last_active = last_accessed_at if access_count > 0 else created_at
    days_since = max(0.0, (now_ms - last_active) / MS_PER_DAY)
    base_half_life = (
        config.recency_half_life_days / 3
        if temporal_type == "dynamic"
        else config.recency_half_life_days
    )
    effective_half_life = base_half_life * math.exp(
        config.importance_modulation * importance
    )
    lam = math.log(2) / effective_half_life
    beta = TIER_BETA.get(tier, TIER_BETA[TIER_WORKING])
    recency = math.exp(-lam * math.pow(days_since, beta))

    # Frequency
    base_freq = 1 - math.exp(-access_count / 5)
    if access_count > 1:
        access_span_days = max(1.0, (last_accessed_at - created_at) / MS_PER_DAY)
        avg_gap_days = access_span_days / max(access_count - 1, 1)
        recentness_bonus = math.exp(-avg_gap_days / 30)
        freq = base_freq * (0.5 + 0.5 * recentness_bonus)
    else:
        freq = base_freq

    # Intrinsic
    intrinsic = importance * confidence

    composite = (
        config.recency_weight * recency
        + config.frequency_weight * freq
        + config.intrinsic_weight * intrinsic
    )

    # Soft tier floor (fraction of the theoretical maximum, not scaled by a
    # weight — the legacy `floor * intrinsic_weight` trick collapsed the
    # peripheral floor onto the demotion threshold and prevented demotion).
    if config.apply_tier_floor:
        floor = TIER_FLOOR.get(tier, 0.0)
        if floor > 0.0 and composite < floor:
            composite = floor

    return {
        "recency": round(recency, 4),
        "frequency": round(freq, 4),
        "intrinsic": round(intrinsic, 4),
        "composite": round(composite, 4),
    }


# Backwards-compatible alias kept for existing callers that pass a metadata
# dict positionally. New code should pass a full row as a kwarg.
WeibullDecay = compute_decay_score


# ---------------------------------------------------------------------------
# Tier manager
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    core_access_threshold: int = 10
    core_composite_threshold: float = 0.7
    core_importance_threshold: float = 0.8
    peripheral_composite_threshold: float = 0.15
    peripheral_age_days: int = 60
    working_access_threshold: int = 3
    working_composite_threshold: float = 0.4
    core_demotion_composite_threshold: float = 0.2
    core_demotion_access_threshold: int = 2


def evaluate_tier(
    entry: dict[str, Any],
    decay_score: dict[str, float],
    *,
    config: TierConfig | None = None,
    now_ms: int | None = None,
) -> str:
    """
    Evaluate and return the appropriate tier for a memory.

    Promotion: peripheral -> working -> core
    Demotion:  core -> working -> peripheral

    Rules (CortexReach spec):
      - peripheral -> working: access >= 3 AND composite >= 0.4
      - working -> core:       access >= 10 AND composite >= 0.7 AND importance >= 0.8
      - working -> peripheral: composite < 0.15 OR (age > 60d AND access < 2)
      - core -> working:       composite < 0.2 AND access < 2

    `entry` may be a flat row or a metadata-only dict.
    """
    if config is None:
        config = TierConfig()
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    metadata = _coerce_metadata(entry.get("metadata", entry))
    is_metadata_only = "metadata" not in entry

    current_tier = metadata.get("tier", TIER_WORKING)
    composite = float(decay_score.get("composite", 0.0))
    importance = _extract_importance(
        {} if is_metadata_only else entry,
        metadata,
    )
    access_count = int(metadata.get("access_count", 0) or 0)
    created_at = int(metadata.get("created_at", 0) or 0)
    age_days = (now_ms - created_at) / MS_PER_DAY if created_at > 0 else 0.0

    # Skip tier evaluation for session-summary entries (CortexReach convention)
    if metadata.get("metadata_type") == "session-summary":
        return current_tier

    # Core promotion (highest bar)
    if (
        access_count >= config.core_access_threshold
        and composite >= config.core_composite_threshold
        and importance >= config.core_importance_threshold
    ):
        return TIER_CORE

    # Core demotion: very low composite + low access
    if (
        current_tier == TIER_CORE
        and composite < config.core_demotion_composite_threshold
        and access_count < config.core_demotion_access_threshold
    ):
        return TIER_WORKING

    # Peripheral demotion: low composite OR old + unused.
    # Explicitly excludes CORE-tier memories — core can only be demoted to
    # working (above), never directly to peripheral in a single step.
    if current_tier != TIER_CORE and (
        composite < config.peripheral_composite_threshold
        or (age_days > config.peripheral_age_days and access_count < 2)
    ):
        return TIER_PERIPHERAL

    # Working promotion from peripheral
    if (
        current_tier == TIER_PERIPHERAL
        and access_count >= config.working_access_threshold
        and composite >= config.working_composite_threshold
    ):
        return TIER_WORKING

    return current_tier


def evaluate_all_tiers(
    memories: Iterable[dict[str, Any]],
    *,
    now_ms: int | None = None,
    decay_config: DecayConfig | None = None,
    tier_config: TierConfig | None = None,
) -> dict[str, str]:
    """
    Evaluate tier for every memory. Returns a `{mem_id: new_tier}` dict
    containing only memories whose tier changed.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if decay_config is None:
        decay_config = DecayConfig()
    if tier_config is None:
        tier_config = TierConfig()

    changed: dict[str, str] = {}
    for mem in memories:
        metadata = _coerce_metadata(mem.get("metadata", {}))
        # Don't re-tier archived rows — they shouldn't be in the active pool
        if metadata.get("state") == "archived":
            continue

        decay = compute_decay_score(mem, now_ms=now_ms, config=decay_config)
        new_tier = evaluate_tier(mem, decay, config=tier_config, now_ms=now_ms)
        current_tier = metadata.get("tier", TIER_WORKING)

        if new_tier != current_tier:
            changed[mem["id"]] = new_tier

    return changed


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 if either vector is empty/zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _entry_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Best-effort similarity: cosine on vectors if both have them, else
    token Jaccard on text. Vectors are usually stripped before reaching MMR,
    so the Jaccard fallback is the common path."""
    va, vb = a.get("vector"), b.get("vector")
    if va and vb:
        return cosine_similarity(va, vb)
    return _jaccard(_tokens(a.get("text", "")), _tokens(b.get("text", "")))


def mmr_diversity_filter(
    results: list[tuple[dict[str, Any], float]],
    *,
    similarity_threshold: float = 0.85,
    diversity_lambda: float = 0.7,
) -> list[tuple[dict[str, Any], float]]:
    """
    Apply Maximal Marginal Relevance to demote near-duplicate results.

    Falls back to token-Jaccard similarity when entries don't carry a
    `vector` field (the typical case after `_row_to_dict` strips them).

    NOTE: does not remove similar items — demotes them by halving the score
    when above the similarity threshold (per CortexReach spec).

    Args:
        results: list of (entry, score) tuples — expected sorted by score desc
        similarity_threshold: cosine/Jaccard above which to penalise
        diversity_lambda: 0.5 = balanced, 1.0 = relevance only

    Returns:
        Re-ranked list of (entry, score) tuples.
    """
    if len(results) <= 1:
        return list(results)

    selected: list[tuple[dict[str, Any], float]] = [results[0]]

    for entry, score in results[1:]:
        max_sim = 0.0
        for prev_entry, _prev_score in selected:
            sim = _entry_similarity(entry, prev_entry)
            max_sim = max(max_sim, sim)

        mmr_score = diversity_lambda * score - (1 - diversity_lambda) * max_sim
        if max_sim > similarity_threshold:
            mmr_score *= 0.5
        selected.append((entry, mmr_score))

    selected.sort(key=lambda x: x[1], reverse=True)
    return selected


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------
# Chain: fusion -> rerank -> length norm -> hardMinScore -> decay -> noise -> MMR

@dataclass
class ScoringConfig:
    length_norm_anchor: int = 500
    hard_min_score: float = 0.05  # applied BEFORE decay
    fusion_weight: float = 0.7    # query-relevance contribution
    decay_weight: float = 0.3     # memory-health contribution


class ScoringPipeline:
    """
    Multi-stage scoring pipeline matching CortexReach architecture.

    Stage order (spec):
      1. Length normalisation
      2. Hard min score filter (BEFORE decay)
      3. Composite decay scoring (Weibull)
      4. Sort by final score

    The `hard_min_score` gate happens BEFORE decay so that semantically
    irrelevant results don't survive on tier-floor alone.
    """

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()
        # Legacy attribute names preserved for callers that introspect them
        self.length_norm_anchor = self.config.length_norm_anchor
        self.hard_min_score = self.config.hard_min_score

    def apply_scoring(
        self,
        results: list[dict[str, Any]],
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Apply the full scoring pipeline to fused search results."""
        if not results:
            return []
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        cfg = self.config

        # Stage 1: length normalisation
        for entry in results:
            base_score = float(entry.get("_fusion_score", entry.get("_rrf_score", 0.0)))
            text_len = len(entry.get("text", "") or "")
            if cfg.length_norm_anchor > 0 and text_len > cfg.length_norm_anchor:
                length_penalty = 1.0 / (
                    1.0 + 0.5 * math.log2(text_len / cfg.length_norm_anchor)
                )
                base_score *= length_penalty
            entry["_score"] = base_score

        # Stage 2: hard min score (BEFORE decay)
        results = [e for e in results if e.get("_score", 0.0) >= cfg.hard_min_score]
        if not results:
            return []

        # Stage 3: composite decay scoring
        decay_config = DecayConfig()
        for entry in results:
            decay = compute_decay_score(entry, now_ms=now_ms, config=decay_config)
            entry["_decay"] = decay
            entry["_final_score"] = (
                cfg.fusion_weight * entry.get("_score", 0.0)
                + cfg.decay_weight * decay["composite"]
            )

        # Stage 4: sort by final score
        results.sort(key=lambda e: e.get("_final_score", 0.0), reverse=True)
        return results
