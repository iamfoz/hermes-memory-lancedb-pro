"""hermes-memory-lancedb-pro — LanceDB-backed persistent memory for Hermes Agent.

Provides hybrid BM25+vector search with Weibull decay and tier management.

Top-level imports are lazy: importing the package does not pull in `lancedb`
or `sentence_transformers`, so unit tests for the pure-Python pieces (decay,
noise filter, MMR) run without the heavy dependencies. Touching one of the
LanceDB-backed names (e.g. `MemoryStore`) triggers the real import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.11.30"

# Pure-Python re-exports (safe — no heavy deps)
# Reflection layer (PR 2). Pure-Python; the storage adapter delegates to
# MemoryStore but only the orchestrator imports from `..store` lazily.
from . import reflection  # noqa: F401
from .task_ledger import (  # noqa: F401
    TASK_ROOT,
    advance_iteration,
    append_jsonl,
    atomic_write_json,
    build_control_block,
    complete_task,
    create_task,
    list_tasks,
    load_state,
    looks_like_reset,
    save_state,
)
from .batch_dedup import (  # noqa: F401
    BatchDedupResult,
    ExtractionCostStats,
    batch_dedup,
    create_extraction_cost_stats,
)
from .decay import (  # noqa: F401
    DecayConfig,
    ScoringConfig,
    ScoringPipeline,
    TierConfig,
    WeibullDecay,
    compute_decay_score,
    evaluate_all_tiers,
    evaluate_tier,
    is_noise,
    mmr_diversity_filter,
)

# Smart extractor + LLM client (PR 3). Both are import-safe without an LLM
# SDK installed — the LLM clients lazy-import `openai` / `anthropic` only on
# instantiation, and the extractor's pipeline branches on `llm is None`.
from .extraction_prompts import (  # noqa: F401
    build_dedup_prompt,
    build_extraction_prompt,
    build_merge_prompt,
)
from .llm_client import (  # noqa: F401
    AnthropicLlmClient,
    LlmClient,
    OpenAICompatibleLlmClient,
    create_llm_client_from_env,
)
from .memory_categories import (  # noqa: F401
    MEMORY_CATEGORIES as SMART_MEMORY_CATEGORIES,
)
from .memory_categories import (  # noqa: F401
    CandidateMemory,
    SmartCategory,
    normalize_category,
)
from .session_compressor import (  # noqa: F401
    CompressResult,
    ScoredText,
    compress_texts,
    estimate_conversation_value,
    score_text,
)
from .smart_extractor import (  # noqa: F401
    DedupResult,
    ExtractionRateLimiter,
    ExtractionStats,
    SmartExtractor,
    SmartExtractorConfig,
    strip_envelope_metadata,
)
from .smart_metadata import (  # noqa: F401
    SmartMemoryMetadata,
    SupportInfoV2,
    SupportSlice,
    build_smart_metadata,
    derive_fact_key,
    parse_smart_metadata,
    parse_support_info,
    stringify_smart_metadata,
    update_support_stats,
)
from .temporal_classifier import (  # noqa: F401
    TemporalType,
    classify_temporal,
    infer_expiry,
)

# Names provided by submodules that pull in lancedb / sentence-transformers.
# Resolved on first access via __getattr__ so `import hermes_memory_lancedb_pro`
# (or importing a pure-Python name from it) doesn't crash without those deps.
_LAZY_ATTRS = {
    "MemoryStore": ("store", "MemoryStore"),
    "MemorySchema": ("store", "MemorySchema"),
    "MemoryRetriever": ("retriever", "MemoryRetriever"),
    "HybridRetriever": ("retriever", "HybridRetriever"),
    "LanceDBProMemoryProvider": ("provider", "LanceDBProMemoryProvider"),
    "register_memory_provider": ("provider", "register_memory_provider"),
    # admission_control + memory_compactor reference MemoryStore lazily
    # (TYPE_CHECKING) so they're safe to import without lancedb.
    "AdmissionController": ("admission_control", "AdmissionController"),
    "AdmissionControlConfig": ("admission_control", "AdmissionControlConfig"),
    "ExtractorLLM": ("admission_control", "ExtractorLLM"),
    "get_admission_preset": ("admission_control", "get_preset"),
    "CompactionConfig": ("memory_compactor", "CompactionConfig"),
    "CompactionResult": ("memory_compactor", "CompactionResult"),
    "run_compaction": ("memory_compactor", "run_compaction"),
    "should_run_compaction": ("memory_compactor", "should_run_compaction"),
    "record_compaction_run": ("memory_compactor", "record_compaction_run"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        from importlib import import_module
        module_name, attr = _LAZY_ATTRS[name]
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


if TYPE_CHECKING:  # pragma: no cover — for IDEs / type checkers only
    from .provider import LanceDBProMemoryProvider, register_memory_provider
    from .retriever import HybridRetriever, MemoryRetriever
    from .store import MemorySchema, MemoryStore


__all__ = [
    # Core
    "MemoryStore",
    "MemorySchema",
    "MemoryRetriever",
    "HybridRetriever",
    "LanceDBProMemoryProvider",
    "register_memory_provider",
    # Decay / scoring
    "DecayConfig",
    "ScoringConfig",
    "ScoringPipeline",
    "TierConfig",
    "WeibullDecay",
    "compute_decay_score",
    "evaluate_all_tiers",
    "evaluate_tier",
    "is_noise",
    "mmr_diversity_filter",
    # Smart category schema
    "CandidateMemory",
    "SmartCategory",
    "SMART_MEMORY_CATEGORIES",
    "normalize_category",
    # Temporal classifier
    "TemporalType",
    "classify_temporal",
    "infer_expiry",
    # Session compressor
    "ScoredText",
    "CompressResult",
    "score_text",
    "compress_texts",
    "estimate_conversation_value",
    # Batch dedup
    "BatchDedupResult",
    "ExtractionCostStats",
    "batch_dedup",
    "create_extraction_cost_stats",
    # Admission control
    "AdmissionController",
    "AdmissionControlConfig",
    "ExtractorLLM",
    "get_admission_preset",
    # Memory compactor
    "CompactionConfig",
    "CompactionResult",
    "run_compaction",
    "should_run_compaction",
    "record_compaction_run",
    # Reflection layer (subpackage)
    "reflection",
    # Task ledger — durable task state outside the context window
    "TASK_ROOT",
    "advance_iteration",
    "append_jsonl",
    "atomic_write_json",
    "build_control_block",
    "complete_task",
    "create_task",
    "list_tasks",
    "load_state",
    "looks_like_reset",
    "save_state",
    # Smart extractor + LLM client
    "AnthropicLlmClient",
    "DedupResult",
    "ExtractionRateLimiter",
    "ExtractionStats",
    "LlmClient",
    "OpenAICompatibleLlmClient",
    "SmartExtractor",
    "SmartExtractorConfig",
    "build_dedup_prompt",
    "build_extraction_prompt",
    "build_merge_prompt",
    "create_llm_client_from_env",
    "strip_envelope_metadata",
    # Smart metadata (L0/L1/L2 + support stats + fact-key)
    "SmartMemoryMetadata",
    "SupportInfoV2",
    "SupportSlice",
    "build_smart_metadata",
    "derive_fact_key",
    "parse_smart_metadata",
    "parse_support_info",
    "stringify_smart_metadata",
    "update_support_stats",
]
