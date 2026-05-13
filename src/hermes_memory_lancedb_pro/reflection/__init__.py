"""Reflection layer — porting CortexReach's reflection subsystem.

The reflection layer captures structured insights ("invariants", "derived",
"lessons", "decisions") that an upstream LLM has produced in markdown
form, then persists, ranks, and replays them. This package owns:

- metadata          — reflection-entry classification helpers
- ranking           — logistic-decay scoring for reflection lines
- retry             — transient-error classifier for the LLM that
                      generates reflection markdown (used by callers)
- event_store       — payload builder for one-per-run reflection events
- item_store        — payload builder for individual invariant/derived items
- mapped_metadata   — payload builder for mapped memories (user-model,
                      agent-model, lesson, decision)
- slices            — markdown parser + sanitiser (prompt-injection guards)
- store             — orchestrator: parse → dedupe → embed → store, plus
                      load + rank the active reflection slices on recall

This package does NOT call an LLM. The LLM that generates the reflection
markdown is the caller's responsibility (PR 3's smart_extractor will be
the typical caller). All persistence is dependency-injected via a small
`ReflectionStoreAdapter` Protocol so the package stays usable without
LanceDB at import time.
"""

from __future__ import annotations

from .event_store import (
    ReflectionEventMetadata,
    ReflectionEventPayload,
    build_reflection_event_payload,
    create_reflection_event_id,
)
from .item_store import (
    ReflectionItemMetadata,
    ReflectionItemPayload,
    build_reflection_item_payloads,
    get_reflection_item_decay_defaults,
)
from .mapped_metadata import (
    ReflectionMappedCategory,
    ReflectionMappedKind,
    ReflectionMappedMetadata,
    build_reflection_mapped_metadata,
    get_reflection_mapped_decay_defaults,
)
from .metadata import (
    get_display_category_tag,
    is_reflection_entry,
    parse_reflection_metadata,
)
from .ranking import (
    ReflectionScoreInput,
    compute_reflection_logistic,
    compute_reflection_score,
    normalize_reflection_line_for_aggregation,
)
from .retry import (
    RetryClassifierResult,
    classify_reflection_retry,
    compute_reflection_retry_delay_ms,
    is_reflection_non_retry_error,
    is_transient_reflection_upstream_error,
    run_with_reflection_transient_retry_once,
)
from .slices import (
    ReflectionGovernanceEntry,
    ReflectionMappedMemoryItem,
    ReflectionSliceItem,
    ReflectionSlices,
    extract_reflection_learning_governance_candidates,
    extract_reflection_lessons,
    extract_reflection_mapped_memory_items,
    extract_reflection_slices,
    extract_section_markdown,
    is_recall_used,
    is_unsafe_injectable_reflection_line,
    parse_section_bullets,
    sanitize_injectable_reflection_lines,
    sanitize_reflection_slice_lines,
)
from .store import (
    MemoryStoreReflectionAdapter,
    ReflectionMappedSlices,
    ReflectionStoreAdapter,
    build_reflection_store_payloads,
    is_owned_by_agent,
    load_agent_reflection_slices_from_entries,
    load_reflection_mapped_rows_from_entries,
    store_reflection_to_lancedb,
)

__all__ = [
    # event store
    "ReflectionEventMetadata",
    "ReflectionEventPayload",
    "build_reflection_event_payload",
    "create_reflection_event_id",
    # item store
    "ReflectionItemMetadata",
    "ReflectionItemPayload",
    "build_reflection_item_payloads",
    "get_reflection_item_decay_defaults",
    # mapped metadata
    "ReflectionMappedCategory",
    "ReflectionMappedKind",
    "ReflectionMappedMetadata",
    "build_reflection_mapped_metadata",
    "get_reflection_mapped_decay_defaults",
    # metadata helpers
    "get_display_category_tag",
    "is_reflection_entry",
    "parse_reflection_metadata",
    # ranking
    "ReflectionScoreInput",
    "compute_reflection_logistic",
    "compute_reflection_score",
    "normalize_reflection_line_for_aggregation",
    # retry
    "RetryClassifierResult",
    "classify_reflection_retry",
    "compute_reflection_retry_delay_ms",
    "is_reflection_non_retry_error",
    "is_transient_reflection_upstream_error",
    "run_with_reflection_transient_retry_once",
    # slices
    "ReflectionGovernanceEntry",
    "ReflectionMappedMemoryItem",
    "ReflectionSliceItem",
    "ReflectionSlices",
    "extract_reflection_lessons",
    "extract_reflection_learning_governance_candidates",
    "extract_reflection_mapped_memory_items",
    "extract_reflection_slices",
    "extract_section_markdown",
    "is_recall_used",
    "is_unsafe_injectable_reflection_line",
    "parse_section_bullets",
    "sanitize_injectable_reflection_lines",
    "sanitize_reflection_slice_lines",
    # store
    "MemoryStoreReflectionAdapter",
    "ReflectionMappedSlices",
    "ReflectionStoreAdapter",
    "build_reflection_store_payloads",
    "is_owned_by_agent",
    "load_agent_reflection_slices_from_entries",
    "load_reflection_mapped_rows_from_entries",
    "store_reflection_to_lancedb",
]
