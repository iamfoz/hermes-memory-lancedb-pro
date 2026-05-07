# hermes-memory-lancedb-pro

> LanceDB-backed persistent memory for [Hermes Agent](https://github.com/nousresearch/hermes-agent) — hybrid BM25+vector search with Weibull decay and tier management.

## Overview

A production-grade memory store that gives Hermes Agent persistent, searchable recall across sessions. Supports:

- **Hybrid search** — RRF-fused BM25 (lexical) + cosine (semantic) retrieval
- **Weibull composite decay** — importance fades based on recency, access frequency, and recency bias
- **Tier management** — core / working / peripheral memory tiers with automatic promotion
- **Supersede pattern** — updates archive old rows and create new ones (no in-place mutation)
- **Auto-recovery** — detects corrupted databases and re-seeds from MEMORY.md
- **Archived row filtering** — superseded/deleted entries excluded from all queries

### Schema

| Column | Type | Index |
|---|---|---|
| `id` | `str` (UUID) | — |
| `text` | `str` | FTS (BM25) |
| `vector` | `float768[]` | IVF_PQ (cosine) |
| `category` | `str` | — |
| `scope` | `str` | — |
| `importance` | `float` | — |
| `timestamp` | `int` (epoch ms) | — |
| `metadata` | `str` (JSON) | — |

## Quick Start

### Prerequisites

- Python 3.11+
- Hermes Agent installed

### Install

```bash
# The plugin installs into your Hermes plugins directory:
# ~/.hermes/plugins/lancedb_pro/

# Dependencies are installed via the plugin.yaml pip_dependencies:
# lance, lancedb, numpy, sentence-transformers
```

### Initialise

```bash
# Create DB and seed from MEMORY.md (34 entries)
bash ~/.hermes/scripts/memory_init.sh

# Or wipe and re-seed from scratch
bash ~/.hermes/scripts/memory_reset.sh
```

### Run the smoke test

```bash
cd ~/.hermes/plugins/lancedb_pro
python scripts/memory_smoke_test.py
# → 9/9 passed
```

## Python API

```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/.hermes/plugins/lancedb_pro"))
from store import MemoryStore

store = MemoryStore()
store._initialise()

# Store a memory
mem_id = store.store(
    text="Martyn prefers concise responses with UK English.",
    category="preference", scope="global", importance=0.9,
)

# Search — three modes
results = store.search("concise responses", limit=5, mode="hybrid")  # default
results = store.search("concise responses", limit=5, mode="vector")
results = store.search("concise responses", limit=5, mode="bm25")

# Update (supersede pattern — archives old, creates new)
store.update(mem_id=mem_id, text="Updated text here.")

# Delete
store.forget(mem_id=mem_id)

# Utilities
store.has_id(mem_id)       # False for deleted/archived
store.stats()              # total_memories, categories
store.list_memories(limit=10)

# Tier / decay
store.apply_tiers()        # promote/demote based on access patterns
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MemoryStore  │────▶│  LanceDB     │────▶│  SQLite/     │
│  (store.py)   │     │  Table       │     │  File-backed │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       ├─▶ _vector_search  (nomic-embed-text-v1.5, 768d cosine)
       ├─▶ _bm25_search    (FTS on text column only)
       └─▶ _hybrid_search  (RRF fusion, k=61)
```

### Key Design Decisions

1. **`fts_columns=["text"]`** — BM25 index scoped to text column only, preventing metadata/header pollution
2. **Python-side archived filtering** — `ARCHIVED_STATE` constant used to filter post-fetch, avoiding SQL LIKE escape hell
3. **Supersede over in-place update** — updates archive the old row (`"state": "archived"`) and create a new one with a fresh UUID. This preserves audit trail and avoids vector drift.
4. **RRF fusion (k=61)** — hybrid search combines BM25 rank and vector distance into a single score

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_DIR` | `~/.hermes/memory-lancedb` | Database directory |
| `MEMORY_MD` | `~/.hermes/memory/MEMORY.md` | Seed file for initialisation |
| `HERMES_PYTHON` | auto-detected | Python interpreter |
| `HF_TOKEN` | *(none)* | HuggingFace token (avoids rate limits on embedding model download) |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/memory_init.sh` | Create DB, seed from MEMORY.md, auto-recover if corrupted |
| `scripts/memory_reset.sh` | Wipe DB + reinitialise |
| `scripts/memory_smoke_test.py` | 9-point E2E test suite |

## License

MIT
