# Architecture

This document describes how `hermes-memory-lancedb-pro` stores and retrieves
memories, and the design decisions behind it.

## Components

```
┌────────────────┐     ┌──────────────┐     ┌────────────────┐
│  MemoryStore   │────▶│  LanceDB     │────▶│  File-backed   │
│  (store.py)    │     │  Table       │     │  dataset       │
└──────┬─────────┘     └──────────────┘     └────────────────┘
       │
       ├─▶ _vector_search   nomic-embed-text-v1.5, 768-d cosine
       ├─▶ _bm25_search     full-text index scoped to the `text` column
       └─▶ _hybrid_search   Reciprocal Rank Fusion (k = 60)

┌────────────────────┐
│  MemoryRetriever   │  fusion → length-normalise → hard min-score
│  (retriever.py)    │  → decay scoring → noise filter → rerank
└────────┬───────────┘  → MMR diversity → lifecycle hooks
         │
┌────────▼────────────────┐
│ LanceDBProMemoryProvider │  the hermes-agent MemoryProvider adapter:
│ (provider.py)            │  session scoping, prefetch, extraction,
└──────────────────────────┘  reflection, compaction, task ledger
```

- **`MemoryStore`** — owns the LanceDB table and all CRUD. Path-keyed singleton
  (`MemoryStore.get_instance(db_path=...)`).
- **`MemoryRetriever`** — the full read pipeline on top of the store.
- **`LanceDBProMemoryProvider`** — the adapter that plugs the store and
  retriever into hermes-agent's plugin system. Imports `agent.memory_provider`
  lazily, so the rest of the package is usable without hermes-agent.

## Storage model

Each memory is one row:

| Column | Type | Index |
|---|---|---|
| `id` | `str` (UUID) | — |
| `text` | `str` | full-text (BM25) |
| `vector` | `float[768]` | IVF_PQ (cosine) |
| `category` | `str` | — |
| `scope` | `str` | — |
| `importance` | `float` | — |
| `timestamp` | `int` (epoch ms) | — |
| `metadata` | `str` (JSON) | — |

`category` and `importance` are authoritative as **top-level columns** —
`metadata` never mirrors them, which avoids the two drifting apart on update.
Everything else (session id, tier, decay state, support stats, entities,
temporal type) lives in the `metadata` JSON blob.

## Retrieval pipeline

`MemoryRetriever.retrieve()` runs these stages in order:

1. **Hybrid search** — BM25 and vector results fused with Reciprocal Rank
   Fusion (`k = 60`). RRF is purely rank-based, so it combines lexical and
   semantic hits without needing a shared score space.
2. **Length normalisation** — dampens the bias toward very short or very long
   texts.
3. **Hard min-score gate** — drops results below `min_score` (only active when
   `min_score > 0`, so a score of exactly `0.0` is not discarded).
4. **Decay scoring** — the Weibull recency term is folded together with
   importance, access frequency, and an evidence-weighted confidence blend into
   a single composite score.
5. **Noise filter** — strips boilerplate and trivially short text.
6. **Rerank** — an optional cross-encoder pass (LangSearch or Google Discovery
   Engine); skipped silently when no reranker is configured.
7. **MMR diversity** — Maximal Marginal Relevance removes near-duplicate hits
   so the recall block is not three phrasings of one fact.
8. **Lifecycle hooks** — throttled access-count and tier updates.

## Weibull decay

Recency uses a stretched-exponential (Weibull) curve:

```
recency = exp(-(λ · t)^β)        λ = ln(2)^(1/β) / half_life
```

`β` is per-tier — core 0.8, working 1.0, peripheral 1.3 — so core memories
decay slowly with a long tail and peripheral memories drop off sharply. The
`λ` definition guarantees `recency = 0.5` at exactly each tier's half-life.
Entries classified as temporally dynamic (see the temporal classifier) decay on
an accelerated schedule.

## Tier management

Memories live in one of three tiers — `core`, `working`, `peripheral`. A
throttled evaluator (every `MEMORY_TIER_EVAL_FREQUENCY` retrievals) promotes
and demotes rows based on access patterns and age. Core-tier memories are always
cross-session.

## Key design decisions

1. **Supersede over in-place update.** An update archives the old row
   (`state = archived`) and writes a new row with a fresh UUID. This preserves a
   full audit trail and avoids vector drift. `get_by_id()` follows the
   `superseded_by` chain so callers always reach the live version.
2. **Application-layer archived filtering.** Archived rows are filtered in
   Python against an `ARCHIVED_STATE` constant rather than with a SQL `LIKE`,
   which sidesteps escape-character hazards.
3. **BM25 scoped to `text`.** The full-text index covers only the `text`
   column, so metadata and headers never pollute lexical search.
4. **Top-level columns are authoritative.** `importance` and `category` are
   real columns, never duplicated into `metadata`.
5. **Lazy heavy imports.** Importing the package does not pull in `lancedb` or
   `sentence-transformers`; the pure-Python pieces (decay, noise filter, MMR,
   classifiers) run in tests without the heavy dependencies.
6. **A portable `score` field.** Every search result carries a `score` in
   `[0, 1]` that is comparable across vector, BM25, and hybrid modes.
   Mode-specific raw fields are prefixed with `_` and are debug-only.

## Anti-stickiness

The defining failure mode of agent memory is *stickiness* — stale memories from
prior tasks bleeding into a fresh conversation. Three mechanisms counter it:

- **`session_id` scoping** — recall is restricted to the current session unless
  a memory is explicitly `cross_session` or core-tier.
- **`min_score`** — weak semantic matches are dropped before they reach the LLM.
- **Access-count throttle** — `increment_access_count` is throttled, so a single
  turn cannot inflate a memory's frequency score and lock it into the top of
  every future recall.

## Further reading

- [hooks.md](hooks.md) — the hermes-agent lifecycle hooks the provider implements.
- [configuration.md](configuration.md) — every environment variable.
- [usage.md](usage.md) — standalone-library recipes.
