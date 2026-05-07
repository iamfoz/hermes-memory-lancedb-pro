"""
Decay engine, tier manager, noise filter, and MMR diversity.

Ported from CortexReach memory-lancedb-pro architecture:
  - Weibull stretched-exponential decay with importance-modulated half-life
  - Three-tier management: core (0.9 floor), working (0.7), peripheral (0.5)
  - Noise filtering for denials, meta-questions, boilerplate, extractor artifacts
  - MMR (Maximal Marginal Relevance) diversity filtering
  - Scoring pipeline: hardMinScore -> composite decay -> noise filter -> MMR
"""

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Constants
MS_PER_DAY = 86_400_000

# --- Noise Filter Patterns ---

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

# Pre-compile patterns
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in (
    DENIAL_PATTERNS + META_QUESTION_PATTERNS + BOILERPLATE_PATTERNS +
    DIAGNOSTIC_ARTIFACT_PATTERNS + ENVELOPE_NOISE_PATTERNS
)]


# ============================================================================
# Noise Filter
# ============================================================================

def is_noise(text: str) -> bool:
    """Check if text is noise (denial, meta-question, boilerplate, artifact)."""
    if not text or len(text.strip()) < 10:
        return True
    for pattern in _NOISE_RE:
        if pattern.search(text):
            return True
    return False


# ============================================================================
# Weibull Decay Engine
# ============================================================================

class DecayConfig:
    """Weibull stretched-exponential decay configuration."""
    recency_half_life_days: float = 30
    recency_weight: float = 0.4
    frequency_weight: float = 0.3
    intrinsic_weight: float = 0.3
    stale_threshold: float = 0.3
    search_boost_min: float = 0.3
    importance_modulation: float = 1.5
    beta_core: float = 0.8
    beta_working: float = 1.0
    beta_peripheral: float = 1.3
    core_decay_floor: float = 0.9
    working_decay_floor: float = 0.7
    peripheral_decay_floor: float = 0.5


TIER_BETA = {
    "core": 0.8,
    "working": 1.0,
    "peripheral": 1.3,
}

TIER_FLOOR = {
    "core": 0.9,
    "working": 0.7,
    "peripheral": 0.5,
}


def compute_decay_score(
    metadata: Dict[str, Any],
    now_ms: Optional[int] = None,
    config: DecayConfig = None,
) -> Dict[str, float]:
    """
    Compute Weibull composite decay score for a memory entry.

    Formula: composite = recency_weight * recency
                        + frequency_weight * frequency
                        + intrinsic_weight * intrinsic

    Recency: Weibull stretched-exponential, half-life modulated by importance.
    Frequency: Logarithmic saturation of access_count + recent-access bonus.
    Intrinsic: importance * confidence.

    Tier modulates Weibull beta (decay shape) and floor (minimum retention).
    """
    if config is None:
        config = DecayConfig()

    if now_ms is None:
        now_ms = int(time.time() * 1000)

    tier = metadata.get("tier", "working")
    importance = metadata.get("importance", 0.5)
    confidence = metadata.get("confidence", 0.8)
    access_count = metadata.get("access_count", 0)
    created_at = metadata.get("created_at", now_ms)
    last_accessed_at = metadata.get("last_accessed_at", created_at)
    temporal_type = metadata.get("temporal_type", "static")

    # --- Recency: Weibull stretched-exponential decay ---
    last_active = last_accessed_at if access_count > 0 else created_at
    days_since = max(0, (now_ms - last_active) / MS_PER_DAY)

    # Dynamic memories decay faster (shorter base half-life)
    base_half_life = config.recency_half_life_days / 3 if temporal_type == "dynamic" else config.recency_half_life_days
    # Higher importance -> longer effective half-life (slower decay)
    effective_half_life = base_half_life * math.exp(config.importance_modulation * importance)
    lam = math.log(2) / effective_half_life
    beta = TIER_BETA.get(tier, config.beta_working)

    recency = math.exp(-lam * math.pow(days_since, beta))

    # --- Frequency: logarithmic saturation with time-weighted bonus ---
    base_freq = 1 - math.exp(-access_count / 5)
    if access_count > 1:
        access_span_days = max(1, (last_accessed_at - created_at) / MS_PER_DAY)
        avg_gap_days = access_span_days / max(access_count - 1, 1)
        recentness_bonus = math.exp(-avg_gap_days / 30)
        freq = base_freq * (0.5 + 0.5 * recentness_bonus)
    else:
        freq = base_freq

    # --- Intrinsic value ---
    intrinsic = importance * confidence

    # --- Composite score ---
    composite = (
        config.recency_weight * recency +
        config.frequency_weight * freq +
        config.intrinsic_weight * intrinsic
    )

    # Apply tier floor (minimum retention based on tier)
    floor = TIER_FLOOR.get(tier, config.working_decay_floor)
    composite = max(composite, floor * config.intrinsic_weight)

    return {
        "recency": round(recency, 4),
        "frequency": round(freq, 4),
        "intrinsic": round(intrinsic, 4),
        "composite": round(composite, 4),
    }


# ============================================================================
# Tier Manager
# ============================================================================

class TierConfig:
    core_access_threshold: int = 10
    core_composite_threshold: float = 0.7
    core_importance_threshold: float = 0.8
    peripheral_composite_threshold: float = 0.15
    peripheral_age_days: int = 60
    working_access_threshold: int = 3
    working_composite_threshold: float = 0.4


def evaluate_tier(
    metadata: Dict[str, Any],
    decay_score: Dict[str, float],
    config: TierConfig = None,
) -> str:
    """
    Evaluate and return the appropriate tier for a memory.

    Promotion: peripheral -> working -> core
    Demotion: core -> working -> peripheral

    Rules (from CortexReach spec):
      - peripheral -> working: access >= 3 AND composite >= 0.4
      - working -> core: access >= 10 AND composite >= 0.7 AND importance >= 0.8
      - working -> peripheral: composite < 0.15 OR (age > 60 days AND access < 2)
      - core -> working: very low composite AND low access
    """
    if config is None:
        config = TierConfig()

    current_tier = metadata.get("tier", "working")
    composite = decay_score.get("composite", 0)
    importance = metadata.get("importance", 0.5)
    access_count = metadata.get("access_count", 0)
    created_at = metadata.get("created_at", 0)
    now_ms = int(time.time() * 1000)
    age_days = (now_ms - created_at) / MS_PER_DAY if created_at > 0 else 0

    # Skip tier evaluation for session-summary entries
    if metadata.get("metadata_type") == "session-summary":
        return current_tier

    # Core promotion (highest bar)
    if (
        access_count >= config.core_access_threshold and
        composite >= config.core_composite_threshold and
        importance >= config.core_importance_threshold
    ):
        return "core"

    # Core demotion: very low composite + low access
    if current_tier == "core" and composite < 0.2 and access_count < 2:
        return "working"

    # Peripheral demotion: low composite OR old + unused
    if (
        composite < config.peripheral_composite_threshold or
        (age_days > config.peripheral_age_days and access_count < 2)
    ):
        return "peripheral"

    # Working promotion from peripheral
    if current_tier == "peripheral" and (
        access_count >= config.working_access_threshold and
        composite >= config.working_composite_threshold
    ):
        return "working"

    # Default: keep current tier
    return current_tier


def evaluate_all_tiers(
    memories: List[Dict[str, Any]],
    now_ms: Optional[int] = None,
    decay_config: DecayConfig = None,
    tier_config: TierConfig = None,
) -> Dict[str, str]:
    """
    Evaluate tier for all memories. Returns dict of {mem_id: new_tier}
    for memories whose tier changed.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if decay_config is None:
        decay_config = DecayConfig()
    if tier_config is None:
        tier_config = TierConfig()

    changed = {}
    for mem in memories:
        metadata = mem.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        decay = compute_decay_score(metadata, now_ms, decay_config)
        new_tier = evaluate_tier(metadata, decay, tier_config)
        current_tier = metadata.get("tier", "working")

        if new_tier != current_tier:
            changed[mem["id"]] = new_tier

    return changed


# ============================================================================
# MMR Diversity Filter
# ============================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_diversity_filter(
    results: List[Tuple[Dict[str, Any], float]],
    similarity_threshold: float = 0.85,
    diversity_lambda: float = 0.7,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Apply Maximal Marginal Relevance to filter near-duplicate results.

    NOTE: Does NOT remove similar items — demotes them by pushing them
    lower in the ranking (per CortexReach spec).

    Args:
        results: List of (entry, score) tuples
        similarity_threshold: Cosine similarity above which demotion occurs
        diversity_lambda: Lambda for MMR (0.5 = equal, 1.0 = relevance only)

    Returns:
        Filtered and re-ranked results
    """
    if len(results) <= 1:
        return results

    selected = [results[0]]
    selected_vectors = []
    if "vector" in results[0][0]:
        selected_vectors.append(results[0][0]["vector"])

    for entry, score in results[1:]:
        vector = entry.get("vector")
        if vector is None:
            selected.append((entry, score))
            continue

        # Find max similarity to already-selected results
        max_sim = 0.0
        for sv in selected_vectors:
            sim = cosine_similarity(vector, sv)
            max_sim = max(max_sim, sim)

        # Apply MMR: relevance * lambda - max_sim * (1 - lambda)
        mmr_score = diversity_lambda * score - (1 - diversity_lambda) * max_sim

        # Demote but don't drop near-duplicates
        if max_sim > similarity_threshold:
            mmr_score *= 0.5  # Penalty for similarity

        selected.append((entry, mmr_score))
        selected_vectors.append(vector)

    # Re-sort by MMR score
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected


# ============================================================================
# Scoring Pipeline (spec-compliant)
# ============================================================================
# Chain: fusion -> rerank -> length norm -> hardMinScore -> decay -> noise -> MMR
# ============================================================================

class ScoringPipeline:
    """
    Multi-stage scoring pipeline matching CortexReach architecture.

    Chain order (spec):
      1. Length normalisation
      2. Hard min score filter (BEFORE decay)
      3. Composite decay scoring (Weibull)
      4. Noise filter
      5. MMR diversity

    The `hardMinScore` gate happens BEFORE decay/time-decay calculations
    to filter out semantically irrelevant results early.
    """

    def __init__(self):
        self.length_norm_anchor = 500
        self.hard_min_score = 0.05  # Applied BEFORE decay — low threshold to let decay boost valid results
        self.vector_weight = 0.7
        self.bm25_weight = 0.3

    def apply_scoring(
        self,
        results: List[Dict[str, Any]],
        now_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply the full scoring pipeline to search results."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        if not results:
            return []

        # --- Stage 1: Length normalisation ---
        for entry in results:
            base_score = entry.get("_fusion_score", entry.get("_rrf_score", 0.0))
            text_len = len(entry.get("text", ""))
            if self.length_norm_anchor > 0 and text_len > self.length_norm_anchor:
                length_penalty = 1.0 / (1.0 + 0.5 * math.log2(text_len / self.length_norm_anchor))
                base_score *= length_penalty
            entry["_score"] = base_score

        # --- Stage 2: Hard min score (BEFORE decay — per spec) ---
        results = [e for e in results if e.get("_score", 0) >= self.hard_min_score]

        if not results:
            return []

        # --- Stage 3: Composite decay scoring ---
        decay_config = DecayConfig()
        for entry in results:
            metadata = entry.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            decay = compute_decay_score(metadata, now_ms, decay_config)
            entry["_decay"] = decay

            # Blend fusion score with decay composite
            # Fusion score captures query relevance; decay captures memory health
            entry["_final_score"] = (
                self.vector_weight * entry.get("_score", 0) +
                self.bm25_weight * decay["composite"]
            )

        # --- Stage 4: Sort by final score ---
        results.sort(key=lambda e: e.get("_final_score", 0), reverse=True)

        return results
