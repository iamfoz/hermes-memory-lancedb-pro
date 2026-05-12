"""Admission control — gate low-signal memories before they hit the store.

Replicates the CortexReach "AMAC" (Admission Memory Admission Control) gate.
Each candidate is scored on five features:

    utility    LLM-judged "is this worth keeping for future cross-session
               interactions?" (set to 0.5 with `utility_mode="off"` if you
               don't have an LLM client)
    confidence ROUGE-like F1 + token coverage between candidate and the
               source conversation — measures how grounded the candidate is
    novelty   1 − (max cosine similarity vs existing same-category memories)
    recency   exponential ramp-up since the last similar memory (encourages
               periodic refresh; punishes immediate restatement)
    typePrior per-category baseline confidence (profile/preferences high,
              events low)

A weighted sum produces a score; `< reject_threshold` → reject, else
`pass_to_dedup` (with an `add` / `update_or_merge` hint based on whether
the candidate is sufficiently novel).

Admission is opt-in. The Store doesn't call this; callers (e.g. the
smart_extractor in PR 3) explicitly invoke `AdmissionController.evaluate`
to decide whether to write each candidate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .memory_categories import MEMORY_CATEGORIES, CandidateMemory, SmartCategory

if TYPE_CHECKING:  # pragma: no cover
    from .store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client protocol — lightweight; matches what we'll wire up in PR 3.
# ---------------------------------------------------------------------------

class ExtractorLLM(Protocol):
    """Minimal LLM contract for admission utility scoring (and, in PR 3,
    smart extraction). Any object exposing `complete_json` works."""

    def complete_json(
        self,
        prompt: str,
        *,
        label: str | None = None,
    ) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AdmissionPreset = Literal["balanced", "conservative", "high-recall"]
UtilityMode = Literal["standalone", "off"]


@dataclass
class AdmissionWeights:
    utility: float = 0.1
    confidence: float = 0.1
    novelty: float = 0.1
    recency: float = 0.1
    type_prior: float = 0.6


@dataclass
class AdmissionTypePriors:
    profile: float = 0.95
    preferences: float = 0.9
    entities: float = 0.75
    events: float = 0.45
    cases: float = 0.8
    patterns: float = 0.85


@dataclass
class AdmissionRecencyConfig:
    half_life_days: int = 14


@dataclass
class AdmissionControlConfig:
    preset: AdmissionPreset = "balanced"
    enabled: bool = False
    utility_mode: UtilityMode = "standalone"
    weights: AdmissionWeights = field(default_factory=AdmissionWeights)
    reject_threshold: float = 0.45
    admit_threshold: float = 0.6
    novelty_candidate_pool_size: int = 8
    recency: AdmissionRecencyConfig = field(default_factory=AdmissionRecencyConfig)
    type_priors: AdmissionTypePriors = field(default_factory=AdmissionTypePriors)
    audit_metadata: bool = True
    persist_rejected_audits: bool = True
    rejected_audit_file_path: str | None = None


# Preset registry — values lifted from CortexReach `ADMISSION_CONTROL_PRESETS`.
def _preset_balanced() -> AdmissionControlConfig:
    return AdmissionControlConfig()  # the default ctor IS balanced


def _preset_conservative() -> AdmissionControlConfig:
    return AdmissionControlConfig(
        preset="conservative",
        weights=AdmissionWeights(
            utility=0.16, confidence=0.16, novelty=0.18,
            recency=0.08, type_prior=0.42,
        ),
        reject_threshold=0.52,
        admit_threshold=0.68,
        novelty_candidate_pool_size=10,
        recency=AdmissionRecencyConfig(half_life_days=10),
        type_priors=AdmissionTypePriors(
            profile=0.98, preferences=0.94, entities=0.78,
            events=0.28, cases=0.78, patterns=0.8,
        ),
    )


def _preset_high_recall() -> AdmissionControlConfig:
    return AdmissionControlConfig(
        preset="high-recall",
        weights=AdmissionWeights(
            utility=0.08, confidence=0.1, novelty=0.08,
            recency=0.14, type_prior=0.6,
        ),
        reject_threshold=0.34,
        admit_threshold=0.52,
        novelty_candidate_pool_size=6,
        recency=AdmissionRecencyConfig(half_life_days=21),
        type_priors=AdmissionTypePriors(
            profile=0.96, preferences=0.92, entities=0.8,
            events=0.58, cases=0.84, patterns=0.88,
        ),
    )


_PRESETS: dict[AdmissionPreset, callable] = {
    "balanced": _preset_balanced,
    "conservative": _preset_conservative,
    "high-recall": _preset_high_recall,
}


def get_preset(name: AdmissionPreset = "balanced") -> AdmissionControlConfig:
    """Return a fresh config instance for the named preset."""
    return _PRESETS.get(name, _preset_balanced)()


def normalize_weights(w: AdmissionWeights) -> AdmissionWeights:
    """Renormalise weights to sum to 1.0. Empty/zero weights → defaults."""
    total = w.utility + w.confidence + w.novelty + w.recency + w.type_prior
    if total <= 0:
        return AdmissionWeights()
    return AdmissionWeights(
        utility=w.utility / total,
        confidence=w.confidence / total,
        novelty=w.novelty / total,
        recency=w.recency / total,
        type_prior=w.type_prior / total,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _clamp01(x: Any, fallback: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    return min(1.0, max(0.0, v))


# Han ideographs character class (used by tokenizer)
_HAN_RE = re.compile(r"[一-鿿]")
# A "word" character: any Unicode letter or number
_WORD_RE = re.compile(r"[^\W_]", re.UNICODE)


def _tokenize_text(value: str) -> list[str]:
    """Tokenize a string into Latin words and per-character Han ideographs.
    Mirrors the TS `tokenizeText` behaviour (each Han char is its own token,
    Latin/digits are grouped into word tokens, everything else is a separator)."""
    tokens: list[str] = []
    current = ""
    for ch in value.lower().strip():
        if _HAN_RE.match(ch):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
            continue
        if _WORD_RE.match(ch):
            current += ch
            continue
        if current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Standard O(n·m) LCS length. n,m are bounded by candidate / span
    token counts so this is fine for our purposes."""
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = max(prev[j], row[j - 1])
    return dp[len(a)][len(b)]


def _rouge_like_f1(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(a)
    recall = lcs / len(b)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return (2 * precision * recall) / denom


_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]+")


def _split_support_spans(conversation_text: str) -> list[str]:
    """Break the conversation into candidate "support spans" — one per
    line, plus per-sentence breakdowns of each line. Used by
    `score_confidence_support` to find the best matching span."""
    spans: list[str] = []
    seen: set[str] = set()
    for line in conversation_text.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        if trimmed not in seen:
            spans.append(trimmed)
            seen.add(trimmed)
        for sentence in _SENTENCE_SPLIT_RE.split(trimmed):
            cand = sentence.strip()
            if len(cand) >= 4 and cand not in seen:
                spans.append(cand)
                seen.add(cand)
    return spans


# Cosine similarity on plain Python lists. We don't reuse decay's helper
# because admission control needs to cope with mismatched-length vectors
# (TS does: takes min length and uses that prefix).
def _cosine_similarity_safe(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = 0.0
    norm_l = 0.0
    norm_r = 0.0
    for i in range(size):
        lv = float(left[i] or 0.0)
        rv = float(right[i] or 0.0)
        dot += lv * rv
        norm_l += lv * lv
        norm_r += rv * rv
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_l) * math.sqrt(norm_r))


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------

@dataclass
class AdmissionFeatureScores:
    utility: float = 0.0
    confidence: float = 0.0
    novelty: float = 1.0
    recency: float = 1.0
    type_prior: float = 0.5


@dataclass
class ConfidenceSupportBreakdown:
    score: float
    best_support: float
    coverage: float
    unsupported_ratio: float


@dataclass
class NoveltyBreakdown:
    score: float
    max_similarity: float
    matched_ids: list[str] = field(default_factory=list)
    compared_ids: list[str] = field(default_factory=list)


def score_type_prior(
    category: SmartCategory,
    type_priors: AdmissionTypePriors,
) -> float:
    return _clamp01(getattr(type_priors, category, 0.5))


def score_confidence_support(
    candidate: CandidateMemory,
    conversation_text: str,
) -> ConfidenceSupportBreakdown:
    """How grounded is the candidate in the source conversation?"""
    candidate_text = f"{candidate.abstract}\n{candidate.content}".strip()
    candidate_tokens = _tokenize_text(candidate_text)
    if not candidate_tokens:
        return ConfidenceSupportBreakdown(
            score=0.0, best_support=0.0, coverage=0.0, unsupported_ratio=1.0,
        )

    spans = _split_support_spans(conversation_text)
    conversation_tokens = set(_tokenize_text(conversation_text))
    best_support = 0.0
    for span in spans:
        span_tokens = _tokenize_text(span)
        f1 = _rouge_like_f1(candidate_tokens, span_tokens)
        best_support = max(best_support, f1)

    unique_candidate_tokens = list(set(candidate_tokens))
    if unique_candidate_tokens:
        supported = sum(1 for t in unique_candidate_tokens if t in conversation_tokens)
        coverage = supported / len(unique_candidate_tokens)
    else:
        coverage = 0.0
    unsupported_ratio = 1 - coverage if unique_candidate_tokens else 1.0

    score = _clamp01(
        best_support * 0.7 + coverage * 0.3 - unsupported_ratio * 0.25
    )
    return ConfidenceSupportBreakdown(
        score=score,
        best_support=best_support,
        coverage=coverage,
        unsupported_ratio=unsupported_ratio,
    )


def score_novelty_from_matches(
    candidate_vector: Sequence[float],
    matches: Sequence[dict[str, Any]],
) -> NoveltyBreakdown:
    """Lower similarity to existing entries → higher novelty.
    `matches` are MemoryStore search rows (each must include `id` and
    `vector`)."""
    if not candidate_vector or not matches:
        return NoveltyBreakdown(score=1.0, max_similarity=0.0)

    max_sim = 0.0
    compared: list[str] = []
    matched: list[str] = []
    for match in matches:
        mid = match.get("id", "")
        if not mid:
            continue
        compared.append(mid)
        sim = max(0.0, _cosine_similarity_safe(
            candidate_vector, match.get("vector") or []
        ))
        max_sim = max(max_sim, sim)
        if sim >= 0.55:
            matched.append(mid)
    return NoveltyBreakdown(
        score=_clamp01(1 - max_sim, 1.0),
        max_similarity=max_sim,
        matched_ids=matched,
        compared_ids=compared,
    )


def score_recency_gap(
    now_ms: int,
    matches: Sequence[dict[str, Any]],
    half_life_days: int,
) -> float:
    """1 − exp(−λ·gap_days) where λ = ln 2 / half_life_days. No matches
    or `half_life_days <= 0` → 1.0 (no penalty). gap = 0 → 0 (immediate
    restatement is heavily penalised). Older gaps approach 1.0."""
    if not matches or half_life_days <= 0:
        return 1.0
    timestamps = [
        int(m.get("timestamp", 0) or 0)
        for m in matches
        if isinstance(m.get("timestamp"), (int, float))
    ]
    if not timestamps:
        return 1.0
    latest = max(timestamps)
    if latest <= 0:
        return 1.0
    gap_days = max(0.0, (now_ms - latest) / 86_400_000)
    if gap_days == 0:
        return 0.0
    lam = math.log(2) / half_life_days
    return _clamp01(1 - math.exp(-lam * gap_days), 1.0)


def _build_utility_prompt(candidate: CandidateMemory, conversation_text: str) -> str:
    """The exact prompt CortexReach uses for utility judgement.
    Keep this in sync with the TS source — even small wording changes
    can shift score distributions."""
    excerpt = (
        conversation_text[-3000:]
        if len(conversation_text) > 3000
        else conversation_text
    )
    return f"""Evaluate whether this candidate memory is worth keeping for future cross-session interactions.

Conversation excerpt:
{excerpt}

Candidate memory:
- Category: {candidate.category}
- Abstract: {candidate.abstract}
- Overview: {candidate.overview}
- Content: {candidate.content}

Score future usefulness on a 0.0-1.0 scale.

Use higher scores for durable preferences, profile facts, reusable procedures, and long-lived project/entity state.
Use lower scores for one-off chatter, low-signal situational remarks, thin restatements, and low-value transient details.

Return JSON only:
{{
  "utility": 0.0,
  "reason": "short explanation"
}}"""


def score_utility(
    llm: ExtractorLLM | None,
    mode: UtilityMode,
    candidate: CandidateMemory,
    conversation_text: str,
) -> tuple[float, str | None]:
    """Returns `(score, reason)`. With `mode='off'` or no LLM, returns the
    neutral 0.5 — admission still works, just without LLM input."""
    if mode == "off" or llm is None:
        return 0.5, "Utility scoring disabled" if mode == "off" else "No LLM client"

    try:
        response = llm.complete_json(
            _build_utility_prompt(candidate, conversation_text),
            label="admission-utility",
        )
    except Exception as e:
        logger.warning("Utility LLM call failed: %s", e)
        return 0.5, "Utility scoring failed"

    if not response:
        return 0.5, "Utility scoring unavailable"

    score = _clamp01(response.get("utility"), 0.5)
    reason_raw = response.get("reason")
    reason = reason_raw.strip() if isinstance(reason_raw, str) else None
    return score, reason


# ---------------------------------------------------------------------------
# Audit records
# ---------------------------------------------------------------------------

@dataclass
class AdmissionAuditRecord:
    version: str = "amac-v1"
    decision: Literal["reject", "pass_to_dedup"] = "pass_to_dedup"
    hint: Literal["add", "update_or_merge"] | None = None
    score: float = 0.0
    reason: str = ""
    utility_reason: str | None = None
    thresholds_reject: float = 0.0
    thresholds_admit: float = 0.0
    weights: AdmissionWeights = field(default_factory=AdmissionWeights)
    feature_scores: AdmissionFeatureScores = field(default_factory=AdmissionFeatureScores)
    matched_existing_memory_ids: list[str] = field(default_factory=list)
    compared_existing_memory_ids: list[str] = field(default_factory=list)
    max_similarity: float = 0.0
    evaluated_at: int = 0


@dataclass
class AdmissionEvaluation:
    decision: Literal["reject", "pass_to_dedup"]
    hint: Literal["add", "update_or_merge"] | None
    audit: AdmissionAuditRecord


@dataclass
class AdmissionRejectionAuditEntry:
    version: str = "amac-v1"
    rejected_at: int = 0
    session_key: str = ""
    target_scope: str = ""
    scope_filter: list[str] = field(default_factory=list)
    candidate: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    conversation_excerpt: str = ""


def resolve_rejected_audit_path(
    db_path: str,
    config: AdmissionControlConfig | None = None,
) -> str:
    """Where to persist rejection audit entries. Mirrors CortexReach default
    of `<db_path>/../admission-audit/rejections.jsonl`."""
    if config and config.rejected_audit_file_path:
        path = config.rejected_audit_file_path.strip()
        if path:
            return path
    parent = os.path.dirname(db_path.rstrip("/")) or "."
    return os.path.join(parent, "admission-audit", "rejections.jsonl")


# ---------------------------------------------------------------------------
# AdmissionController
# ---------------------------------------------------------------------------

def _build_reason(
    *,
    decision: str,
    hint: str | None,
    score: float,
    reject_threshold: float,
    max_similarity: float,
    utility_reason: str | None,
) -> str:
    score_t = f"{score:.3f}"
    sim_t = f"{max_similarity:.3f}"
    util = f" Utility: {utility_reason}" if utility_reason else ""
    if decision == "reject":
        return (
            f"Admission rejected ({score_t} < {reject_threshold:.3f}). "
            f"maxSimilarity={sim_t}.{util}"
        ).strip()
    hint_t = f" hint={hint};" if hint else ""
    return (
        f"Admission passed ({score_t});{hint_t} maxSimilarity={sim_t}.{util}"
    ).strip()


class AdmissionController:
    """Score candidate memories and decide whether they admit to the store.

    Construct with an ExtractorLLM (optional — utility scoring degrades to
    neutral 0.5 without one), a config (use `get_preset(...)` for sane
    defaults), and a Store reference for novelty-pool lookups."""

    def __init__(
        self,
        store: MemoryStore,
        config: AdmissionControlConfig | None = None,
        llm: ExtractorLLM | None = None,
    ):
        self._store = store
        self._llm = llm
        self.config = config or get_preset("balanced")

    def evaluate(
        self,
        candidate: CandidateMemory,
        conversation_text: str,
        scope_filter: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> AdmissionEvaluation:
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        scope_str = scope_filter[0] if scope_filter else None
        relevant_matches = self._load_relevant_matches(
            candidate, candidate.vector or [], scope_str,
        )

        # Five feature scores
        utility_score, utility_reason = score_utility(
            self._llm, self.config.utility_mode, candidate, conversation_text,
        )
        confidence = score_confidence_support(candidate, conversation_text)
        novelty = score_novelty_from_matches(
            candidate.vector or [], relevant_matches,
        )
        recency = score_recency_gap(
            now_ms, relevant_matches, self.config.recency.half_life_days,
        )
        type_prior = score_type_prior(candidate.category, self.config.type_priors)

        feature_scores = AdmissionFeatureScores(
            utility=utility_score,
            confidence=confidence.score,
            novelty=novelty.score,
            recency=recency,
            type_prior=type_prior,
        )

        w = self.config.weights
        score = (
            feature_scores.utility * w.utility
            + feature_scores.confidence * w.confidence
            + feature_scores.novelty * w.novelty
            + feature_scores.recency * w.recency
            + feature_scores.type_prior * w.type_prior
        )

        if score < self.config.reject_threshold:
            decision: Literal["reject", "pass_to_dedup"] = "reject"
            hint: Literal["add", "update_or_merge"] | None = None
        else:
            decision = "pass_to_dedup"
            hint = (
                "add"
                if score >= self.config.admit_threshold and novelty.max_similarity < 0.55
                else "update_or_merge"
            )

        reason = _build_reason(
            decision=decision,
            hint=hint,
            score=score,
            reject_threshold=self.config.reject_threshold,
            max_similarity=novelty.max_similarity,
            utility_reason=utility_reason,
        )

        audit = AdmissionAuditRecord(
            version="amac-v1",
            decision=decision,
            hint=hint,
            score=score,
            reason=reason,
            utility_reason=utility_reason,
            thresholds_reject=self.config.reject_threshold,
            thresholds_admit=self.config.admit_threshold,
            weights=w,
            feature_scores=feature_scores,
            matched_existing_memory_ids=novelty.matched_ids,
            compared_existing_memory_ids=novelty.compared_ids,
            max_similarity=novelty.max_similarity,
            evaluated_at=now_ms,
        )

        logger.debug(
            "admission: decision=%s hint=%s score=%.3f abstract=%s",
            decision, hint, score, candidate.abstract[:80],
        )
        return AdmissionEvaluation(decision=decision, hint=hint, audit=audit)

    def _load_relevant_matches(
        self,
        candidate: CandidateMemory,
        candidate_vector: Sequence[float],
        scope: str | None,
    ) -> list[dict[str, Any]]:
        """Pull existing same-category memories near the candidate vector
        for novelty / recency scoring. Falls back to all categories if no
        same-category neighbours exist."""
        if not candidate_vector:
            return []
        # Use the embedded query path. _vector_search needs text; we hand it
        # the candidate.abstract because the underlying search re-encodes
        # text rather than accepting a raw vector. This is wasteful — TS
        # passes the vector directly. We trade off here for API simplicity;
        # callers that already have an embedder can override by overriding
        # _load_relevant_matches in a subclass.
        try:
            raw = self._store._vector_search(
                candidate.abstract,
                self.config.novelty_candidate_pool_size,
                None,
                scope,
                keep_vector=True,
            )
        except Exception as e:
            logger.warning("admission novelty pool fetch failed: %s", e)
            return []

        # Prefer same memory_category neighbours; fall back to all
        same_cat: list[dict[str, Any]] = []
        for match in raw:
            md = match.get("metadata") or {}
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except json.JSONDecodeError:
                    md = {}
            if md.get("memory_category") == candidate.category:
                same_cat.append(match)
        return same_cat if same_cat else list(raw)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def append_rejection_audit(
    file_path: str,
    entry: AdmissionRejectionAuditEntry,
) -> None:
    """Append a JSONL line for a rejected admission. Creates parent dirs
    on first write. Best-effort — failures are swallowed (audit logs
    must not break the admission path)."""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), default=str))
            f.write("\n")
    except OSError as e:
        logger.warning("Failed to write admission audit: %s", e)


__all__ = [
    "MEMORY_CATEGORIES",
    "AdmissionAuditRecord",
    "AdmissionController",
    "AdmissionControlConfig",
    "AdmissionEvaluation",
    "AdmissionFeatureScores",
    "AdmissionRecencyConfig",
    "AdmissionRejectionAuditEntry",
    "AdmissionTypePriors",
    "AdmissionWeights",
    "ConfidenceSupportBreakdown",
    "ExtractorLLM",
    "NoveltyBreakdown",
    "append_rejection_audit",
    "get_preset",
    "normalize_weights",
    "resolve_rejected_audit_path",
    "score_confidence_support",
    "score_novelty_from_matches",
    "score_recency_gap",
    "score_type_prior",
    "score_utility",
]
