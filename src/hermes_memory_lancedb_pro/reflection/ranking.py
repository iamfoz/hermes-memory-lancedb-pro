"""Reflection ranking / scoring helpers.

Ported from CortexReach reflection-ranking.ts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = [
    "REFLECTION_FALLBACK_SCORE_FACTOR",
    "ReflectionScoreInput",
    "compute_reflection_logistic",
    "compute_reflection_score",
    "normalize_reflection_line_for_aggregation",
]

REFLECTION_FALLBACK_SCORE_FACTOR: float = 0.75

# Pre-compiled pattern for collapsing internal whitespace.
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


@dataclass
class ReflectionScoreInput:
    """Inputs for the reflection scoring formula."""

    age_days: float
    midpoint_days: float
    k: float
    quality: float = field(default=1.0)      # 0..1 confidence
    base_weight: float = field(default=1.0)  # importance multiplier
    used_fallback: bool = field(default=False)


def compute_reflection_logistic(
    age_days: float,
    midpoint_days: float,
    k: float,
) -> float:
    """Standard logistic decay: ``1 / (1 + exp(k * (age_days - midpoint_days)))``.

    Inputs are clamped / defaulted to safe values matching the TS implementation:
    - ``age_days`` must be finite and >= 0 (defaults to 0)
    - ``midpoint_days`` must be finite and > 0 (defaults to 1)
    - ``k`` must be finite and > 0 (defaults to 0.1)
    """
    safe_age = max(0.0, age_days) if math.isfinite(age_days) else 0.0
    safe_midpoint = midpoint_days if (math.isfinite(midpoint_days) and midpoint_days > 0) else 1.0
    safe_k = k if (math.isfinite(k) and k > 0) else 0.1
    return 1.0 / (1.0 + math.exp(safe_k * (safe_age - safe_midpoint)))


def compute_reflection_score(input: ReflectionScoreInput) -> float:  # noqa: A002
    """Compute a composite reflection score.

    ``logistic * base_weight * quality * fallback_factor``

    ``quality`` is clamped to ``[0, 1]``; ``base_weight`` must be > 0
    (defaults to 1.0).  When ``used_fallback`` is True the result is
    multiplied by ``REFLECTION_FALLBACK_SCORE_FACTOR`` (0.75).
    """
    logistic = compute_reflection_logistic(input.age_days, input.midpoint_days, input.k)

    base_weight = (
        input.base_weight
        if (math.isfinite(input.base_weight) and input.base_weight > 0)
        else 1.0
    )
    quality = (
        max(0.0, min(1.0, input.quality))
        if math.isfinite(input.quality)
        else 1.0
    )
    fallback_factor = REFLECTION_FALLBACK_SCORE_FACTOR if input.used_fallback else 1.0

    return logistic * base_weight * quality * fallback_factor


def normalize_reflection_line_for_aggregation(line: str) -> str:
    """Normalise a reflection line for de-duplication.

    Strip leading/trailing whitespace, collapse runs of internal whitespace
    to a single space, and lowercase.  Used to identify lines that differ
    only in formatting.
    """
    return _WHITESPACE_RE.sub(" ", str(line).strip()).lower()
