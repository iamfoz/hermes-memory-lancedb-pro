"""hermes-memory-lancedb-pro — LanceDB-backed persistent memory for Hermes Agent.

Provides hybrid BM25+vector search with Weibull decay and tier management.

Top-level imports are lazy: importing the package does not pull in `lancedb`
or `sentence_transformers`, so unit tests for the pure-Python pieces (decay,
noise filter, MMR) run without the heavy dependencies. Touching one of the
LanceDB-backed names (e.g. `MemoryStore`) triggers the real import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

# Pure-Python re-exports (safe — no heavy deps)
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

# Names provided by submodules that pull in lancedb / sentence-transformers.
# Resolved on first access via __getattr__ so `import hermes_memory_lancedb_pro`
# (or importing a pure-Python name from it) doesn't crash without those deps.
_LAZY_ATTRS = {
    "MemoryStore": ("store", "MemoryStore"),
    "MemorySchema": ("store", "MemorySchema"),
    "MemoryRetriever": ("retriever", "MemoryRetriever"),
    "HybridRetriever": ("retriever", "HybridRetriever"),
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
    from .retriever import HybridRetriever, MemoryRetriever
    from .store import MemorySchema, MemoryStore


__all__ = [
    "MemoryStore",
    "MemorySchema",
    "MemoryRetriever",
    "HybridRetriever",
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
]
