# hermes-memory-lancedb-pro

> LanceDB-backed persistent memory for [Hermes Agent](https://github.com/nousresearch/hermes-agent) — hybrid BM25+vector search with Weibull decay and tier management.

## Overview

A production-grade memory store that gives Hermes Agent persistent, searchable recall across sessions. Supports:

- **Hybrid search** — RRF-fused BM25 (lexical) + cosine (semantic) retrieval, all modes return the same shape
- **Weibull composite decay** — importance fades based on recency, access frequency, and recency bias
- **Tier management** — core / working / peripheral memory tiers with automatic promotion / demotion
- **Supersede pattern** — updates archive old rows and create new ones (no in-place mutation)
- **Auto-recovery** — detects corrupted databases and re-seeds from MEMORY.md
- **Archived row filtering** — superseded/deleted entries excluded from queries by default
- **Bulk inserts** — `store_many()` encodes and writes in one batch
- **Per-path singleton** — `get_instance(db_path=...)` is keyed by path so multi-store callers work

### Schema

| Column       | Type             | Index           |
|--------------|------------------|-----------------|
| `id`         | `str` (UUID)     | —               |
| `text`       | `str`            | FTS (BM25)      |
| `vector`     | `float768[]`     | IVF_PQ (cosine) |
| `category`   | `str`            | —               |
| `scope`      | `str`            | —               |
| `importance` | `float`          | —               |
| `timestamp`  | `int` (epoch ms) | —               |
| `metadata`   | `str` (JSON)     | —               |

## Quick Start

### Prerequisites

- Python 3.11+
- Hermes Agent installed

### Install

```bash
# As a regular package (recommended)
pip install -e .

# Or drop into your Hermes plugins directory:
# ~/.hermes/plugins/lancedb_pro/
# (dependencies still installed via pip; see pyproject.toml)
```

### Initialise

```bash
# Create DB and seed from MEMORY.md
bash scripts/memory_init.sh

# Or wipe and re-seed from scratch
bash scripts/memory_reset.sh
```

### Run the smoke test

```bash
# Against the default store
python scripts/memory_smoke_test.py

# Against an ephemeral tmp dir (no setup needed)
python scripts/memory_smoke_test.py --ephemeral

# Or, after `pip install -e .`:
hermes-memory-smoke --ephemeral
```

## Python API

```python
from hermes_memory_lancedb_pro import MemoryStore, MemoryRetriever

store = MemoryStore.get_instance()  # path-keyed singleton

# Store
mem_id = store.store(
    text="Martyn prefers concise responses with UK English.",
    category="preference", scope="global", importance=0.9,
)

# Bulk store — one batched embed call, one write transaction
ids = store.store_many([
    {"text": "fact one with enough length to pass the noise filter",
     "importance": 0.6, "category": "fact"},
    {"text": "fact two with enough length to pass the noise filter",
     "importance": 0.5},
])

# Search — three modes, all return List[Dict[str, Any]]
results = store.search("concise responses", limit=5, mode="hybrid")  # default
results = store.search("concise responses", limit=5, mode="vector")
results = store.search("concise responses", limit=5, mode="bm25")

# Update (supersede pattern — archives old, creates new)
store.update(mem_id=mem_id, text="Updated text here.", tier="core")

# Or metadata-only — no re-embedding, no supersede
store.update(mem_id=mem_id, importance=0.95, tier="core")

# Delete
store.forget(mem_id=mem_id)

# Utilities
store.has_id(mem_id)                           # False for deleted/archived
store.check_ids(["id-1", "id-2", "id-3"])      # batch existence check
store.stats()                                  # totals, breakdowns, archived count
store.list_memories(limit=10)                  # active memories only
store.list_memories(limit=10, include_archived=True)
store.purge_archived(grace_period_days=30)     # GC superseded rows

# Full-pipeline retrieval (rerank + MMR + lifecycle hooks)
retriever = MemoryRetriever(store)
hits = retriever.retrieve("concise responses", limit=5)
```

## Architecture

```
┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│  MemoryStore  │────▶│  LanceDB     │────▶│  SQLite/     │
│  (store.py)   │     │  Table       │     │  File-backed │
└──────┬────────┘     └──────────────┘     └──────────────┘
       │
       ├─▶ _vector_search   (nomic-embed-text-v1.5, 768d cosine)
       ├─▶ _bm25_search     (FTS scoped to `text` column)
       └─▶ _hybrid_search   (RRF fusion, k=60)

┌──────────────────┐
│ MemoryRetriever  │  fusion → length norm → hard min → decay
│ (retriever.py)   │  → noise filter → rerank → MMR diversity
└──────────────────┘  → lifecycle hooks (access count, tier)
```

### Key design decisions

1. **`fts_columns=["text"]`** — BM25 index scoped to the text column only, preventing metadata/header pollution.
2. **Python-side archived filtering** — `ARCHIVED_STATE` constant; rows are filtered post-fetch to avoid SQL `LIKE` escape hell.
3. **Supersede over in-place update** — updates archive the old row (`"state": "archived"`) and create a new one with a fresh UUID. Preserves audit trail and avoids vector drift.
4. **RRF fusion (k=60)** — hybrid search combines BM25 rank and vector rank into a single score. Pure rank-based — no `_distance` math (BM25 hits don't have one).
5. **Top-level columns are authoritative** — `importance` and `category` live as table columns; metadata never mirrors them. Avoids drift between the two on update.
6. **Throttled tier evaluation** — full-store tier re-evaluation runs every `MEMORY_TIER_EVAL_FREQUENCY` retrievals (default 10) to keep search latency low.

## Configuration

Environment variables (all optional):

| Variable                       | Default                            | Description                                                       |
|--------------------------------|------------------------------------|-------------------------------------------------------------------|
| `MEMORY_DB_DIR`                | `~/.hermes/memory-lancedb`         | Database directory                                                |
| `MEMORY_MD`                    | `~/.hermes/memory/MEMORY.md`       | Seed file for `memory_init.sh`                                    |
| `HERMES_PYTHON`                | auto-detected                      | Python interpreter for the init script                            |
| `HF_TOKEN`                     | *(none)*                           | HuggingFace token (avoids rate limits on embedding model download) |
| `LANGSEARCH_API_KEY`           | *(none)*                           | Enables cross-encoder reranking in `MemoryRetriever`              |
| `MEMORY_MAX_SCAN_ROWS`         | `100000`                           | Cap on full-table scans in `stats` / `purge_archived`             |
| `MEMORY_TIER_EVAL_FREQUENCY`   | `10`                               | Retrievals between full tier re-evaluations (set 0 to disable)    |
| `MEMORY_TIER_EVAL_BATCH`       | `500`                              | Rows fetched per tier evaluation                                  |

## Scripts

| Script                          | Purpose                                                  |
|---------------------------------|----------------------------------------------------------|
| `scripts/memory_init.sh`        | Create DB, seed from MEMORY.md, auto-recover if corrupt  |
| `scripts/memory_reset.sh`       | Wipe DB + reinitialise (uses sibling `memory_init.sh`)   |
| `scripts/memory_smoke_test.py`  | End-to-end test suite (use `--ephemeral` for tmp DB)     |

## Tests

```bash
pip install -e ".[dev]"

# Unit tests only — fast, no LanceDB or model download
pytest -m "not integration"

# Full suite, including LanceDB-backed tests with a stub embedder
pytest
```

## License

MIT
