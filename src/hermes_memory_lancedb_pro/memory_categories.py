"""Smart-memory category schema (the 6-category system).

Distinct from `store.MEMORY_CATEGORIES` which is the legacy column-level
classification (`preference / fact / decision / entity / other / reflection`).
The smart categories are a richer, extraction-oriented taxonomy used by
`admission_control` and (eventually) `smart_extractor`. They live in
`metadata.memory_category` so they can coexist with the legacy column.

Pure Python — no lancedb, no embedder dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Smart memory categories — matches the CortexReach schema.
SmartCategory = Literal[
    "profile",      # who the user is (name, role, identity)
    "preferences",  # what they like / want
    "entities",     # named things (people, places, projects)
    "events",       # one-off occurrences with timestamps
    "cases",        # reusable solutions / procedures
    "patterns",     # recurring patterns / habits
]

MEMORY_CATEGORIES: tuple[SmartCategory, ...] = (
    "profile",
    "preferences",
    "entities",
    "events",
    "cases",
    "patterns",
)

# Categories that are time-versioned: a newer entry with the same `fact_key`
# supersedes the older one (the older row is marked invalidated). Lifted
# from the CortexReach TS source — applied in smart_extractor's supersede
# branch and in admission control's recency scoring.
TEMPORAL_VERSIONED_CATEGORIES: frozenset[SmartCategory] = frozenset({
    "preferences", "entities",
})

# Categories where merging two similar entries is always preferred over
# storing a duplicate. Profile is the canonical case: a single user has
# one profile, even when discussed across many sessions.
ALWAYS_MERGE_CATEGORIES: frozenset[SmartCategory] = frozenset({"profile"})

# Categories that *support* merging (the LLM dedup `MERGE` decision is
# valid for these; for other categories, MERGE degrades to CREATE).
MERGE_SUPPORTED_CATEGORIES: frozenset[SmartCategory] = frozenset({
    "preferences", "entities", "patterns",
})


def normalize_category(value: Any) -> SmartCategory:
    """Coerce arbitrary input to a valid SmartCategory; default `entities`."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in MEMORY_CATEGORIES:
            return v  # type: ignore[return-value]
        # Legacy → smart mapping (best-effort; admission control is the only
        # caller that cares so we don't need exhaustive coverage).
        legacy_map = {
            "preference": "preferences",
            "fact": "entities",
            "decision": "events",
            "entity": "entities",
            "reflection": "patterns",
            "other": "entities",
        }
        return legacy_map.get(v, "entities")  # type: ignore[return-value]
    return "entities"


@dataclass
class CandidateMemory:
    """A proposed new memory entry, before admission / dedup decisions.

    The shape matches what `smart_extractor` will produce in PR 3, but it's
    defined here so `admission_control` is usable today by any caller that
    can build a candidate (e.g. a user-driven write API)."""

    category: SmartCategory
    # L0: short distillation (~1 line) — used for embedding + search anchor
    abstract: str
    # L1: a sentence or two of context
    overview: str
    # L2: full text content
    content: str
    # Optional: pre-computed embedding (admission control needs one for novelty)
    vector: list[float] | None = None
    # Free-form supplementary metadata that the extractor / caller wants to
    # carry through (e.g. session_id, source attribution, fact_key).
    metadata_extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ALWAYS_MERGE_CATEGORIES",
    "MEMORY_CATEGORIES",
    "MERGE_SUPPORTED_CATEGORIES",
    "TEMPORAL_VERSIONED_CATEGORIES",
    "CandidateMemory",
    "SmartCategory",
    "normalize_category",
]
