"""Batch-internal dedup using cosine similarity on embedded abstract vectors.

Before running expensive per-candidate LLM dedup calls, this module checks all
candidates against each other using cosine similarity on their embedded abstracts.
Candidates with similarity > threshold are marked as batch duplicates and skipped.

For n <= 5 candidates, O(n²) pairwise comparison is trivial.

No LanceDB dependency — pure Python.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .decay import cosine_similarity

__all__ = [
    "BatchDedupResult",
    "ExtractionCostStats",
    "batch_dedup",
    "create_extraction_cost_stats",
]


# ============================================================================
# Types
# ============================================================================


@dataclass
class BatchDedupResult:
    """Result of a batch-internal cosine dedup pass."""

    surviving_indices: list[int]
    """Indices of candidates that survived (not duplicates)."""

    duplicate_indices: list[int]
    """Indices of candidates marked as batch duplicates."""

    input_count: int
    """Number of candidates before dedup."""

    output_count: int
    """Number of candidates after dedup."""


@dataclass
class ExtractionCostStats:
    """Tracks cost metrics for an extraction run."""

    batch_deduped: int = 0
    """Candidates dropped by batch dedup."""

    duration_ms: float = 0.0
    """Total extraction wall time in ms."""

    llm_calls: int = 0
    """Count of LLM invocations."""


# ============================================================================
# Batch Dedup
# ============================================================================


def batch_dedup(
    abstracts: Sequence[str],
    vectors: Sequence[Sequence[float] | None],
    threshold: float = 0.85,
) -> BatchDedupResult:
    """Perform batch-internal cosine dedup on candidate abstracts.

    Args:
        abstracts: Sequence of abstract strings from extracted candidates.
        vectors: Parallel sequence of embedded vectors for each abstract.
            Entries may be None or empty to indicate a missing embedding.
        threshold: Cosine similarity threshold above which candidates are
            considered duplicates. Uses strict greater-than (> threshold),
            so a similarity exactly equal to the threshold does NOT dedup.

    Returns:
        BatchDedupResult with surviving and duplicate indices.
    """
    n = len(abstracts)
    if n <= 1:
        return BatchDedupResult(
            surviving_indices=[0] if n == 1 else [],
            duplicate_indices=[],
            input_count=n,
            output_count=n,
        )

    # Track which candidates are duplicates
    is_duplicate = [False] * n

    # Pairwise comparison: O(n²) but n is small in practice
    for i in range(n):
        if is_duplicate[i]:
            continue
        for j in range(i + 1, n):
            if is_duplicate[j]:
                continue
            vi = vectors[i] if i < len(vectors) else None
            vj = vectors[j] if j < len(vectors) else None
            if not vi or not vj:
                continue
            if len(vi) == 0 or len(vj) == 0:
                continue

            sim = cosine_similarity(list(vi), list(vj))
            if sim > threshold:
                # Mark the later candidate as duplicate of the earlier one
                is_duplicate[j] = True

    surviving_indices: list[int] = []
    duplicate_indices: list[int] = []

    for i in range(n):
        if is_duplicate[i]:
            duplicate_indices.append(i)
        else:
            surviving_indices.append(i)

    return BatchDedupResult(
        surviving_indices=surviving_indices,
        duplicate_indices=duplicate_indices,
        input_count=n,
        output_count=len(surviving_indices),
    )


def create_extraction_cost_stats() -> ExtractionCostStats:
    """Create a fresh ExtractionCostStats tracker."""
    return ExtractionCostStats()
