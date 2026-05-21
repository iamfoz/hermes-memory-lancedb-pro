# Usage

This document covers the command-line interface and the standalone Python
library. For the Hermes Agent plugin path, see
[hermes-integration.md](hermes-integration.md).

## Command-line interface

The package installs three console scripts:

| Command | Purpose |
|---|---|
| `hermes-memory-lancedb-pro` | Main admin CLI. |
| `hermes-memory` | Alias of the above — identical. |
| `hermes-memory-smoke` | End-to-end smoke test. |

When the plugin is active inside hermes-agent, the same subcommands are also
reachable as `hermes lancedb_pro <command>`.

### Store administration

Every store-touching command accepts `--path PATH` (the database directory;
defaults to `MEMORY_DB_DIR`) and `-q` / `--quiet`.

```bash
# Initialise the store; seed from MEMORY.md when empty
hermes-memory-lancedb-pro init [--memory-md PATH] [-y]

# Wipe the database directory and re-run init
hermes-memory-lancedb-pro reset [-y]

# Diagnostic report: counts, tier/category breakdown, anomalies, advice
hermes-memory-lancedb-pro doctor

# Export rows to JSONL (one object per line)
hermes-memory-lancedb-pro export -o backup.jsonl [--include-archived] [--limit N]

# Import rows from JSONL
hermes-memory-lancedb-pro import --in backup.jsonl [--reembed] [--allow-existing]
```

`init` and `reset` prompt for confirmation before changing data; pass
`-y` / `--yes` to skip the prompt in scripts. `import --reembed` re-encodes each
row's text with the current embedding model instead of trusting stored vectors.

### Durable task ledger

A task ledger keeps multi-step task state (objective, iteration counter,
`next_action`) on disk, outside the LLM context window, so it survives context
compaction. Ledgers live under `~/.hermes/workspace/tasks/` by default
(`--task-root` to override).

```bash
hermes-memory-lancedb-pro task create --id TASK_ID [--objective TEXT] [--iterations N]
hermes-memory-lancedb-pro task list
hermes-memory-lancedb-pro task show TASK_ID
hermes-memory-lancedb-pro task resume TASK_ID          # prints the control block
hermes-memory-lancedb-pro task advance TASK_ID --result pass --next-action "..."
hermes-memory-lancedb-pro task complete TASK_ID [--summary TEXT]
hermes-memory-lancedb-pro task pin TASK_ID             # store as an always-recalled memory
```

`task pin` stores the control block as an `active_task` memory, which the
recall path always surfaces first.

### Smoke test

```bash
hermes-memory-smoke --ephemeral   # run against a throwaway tmp database
```

## Standalone Python library

The store, retriever, decay engine, classifiers, and extraction pipeline import
and run **without hermes-agent installed**. Top-level imports are lazy, so
importing the package does not pull in `lancedb` or `sentence-transformers`
until you touch a storage-backed name.

### Basic store operations

```python
from hermes_memory_lancedb_pro import MemoryStore

store = MemoryStore.get_instance()          # path-keyed singleton

mem_id = store.store(
    text="Martyn prefers concise responses in UK English.",
    category="preference", scope="global", importance=0.9,
)

# Bulk write — one batched embed call, one transaction.
# Extra metadata goes in `metadata_extra` (the `metadata` column is internal).
ids = store.store_many([
    {"text": "fact one, long enough to pass the noise filter",
     "category": "fact", "importance": 0.6,
     "metadata_extra": {"source": "import"}},
    {"text": "fact two, long enough to pass the noise filter",
     "importance": 0.5},
])

# Update — supersede pattern: archives the old row, writes a new one.
new_id = store.update(mem_id=mem_id, text="Updated text.", tier="core")

store.delete(mem_id=new_id)                 # delete by id (alias: forget())
store.stats()                               # totals, tier/category breakdown
store.purge_archived(grace_period_days=30)  # GC superseded rows
```

### Search and retrieval

```python
from hermes_memory_lancedb_pro import MemoryStore, MemoryRetriever

store = MemoryStore.get_instance()

# Direct search — three modes, all return list[dict]
results = store.search("concise responses", limit=5, mode="hybrid")  # default
results = store.search("concise responses", limit=5, mode="vector")
results = store.search("concise responses", limit=5, mode="bm25")

# Every result has a portable `score` in [0, 1]; use it across all modes.
for r in results:
    print(r["score"], r["text"])

# Full pipeline — rerank, MMR, lifecycle hooks. Recommended inside an agent.
retriever = MemoryRetriever(store)
hits = retriever.retrieve(
    "concise responses",
    limit=5,
    session_id="sess-2026-05-07_abc",   # scope out other sessions
    min_score=0.2,                      # drop weak matches
)

# After the LLM has actually used the recall, credit those memories:
store.mark_recall_used([h["id"] for h in hits])
```

### Smart extractor

With an LLM client, the extractor replaces "store the raw turn" with a
structured six-category pipeline. `create_llm_client_from_env()` auto-detects
the client from environment variables (see
[configuration.md](configuration.md#llm-extraction)) and returns `None` when
nothing is configured — in which case the extractor falls back to raw-turn
writes.

```python
from hermes_memory_lancedb_pro import (
    SmartExtractor, create_llm_client_from_env, MemoryStore,
)

store = MemoryStore.get_instance()
extractor = SmartExtractor(store, llm=create_llm_client_from_env())

stats = extractor.extract_and_persist(
    user_content="I switched from Vim to Emacs.",
    assistant_content="Got it — I'll remember that.",
    session_key="sess-abc",
    scope="agent",
)
# stats.created / merged / skipped / superseded / supported / rejected
```

Any object with a `complete_json(prompt, *, label) -> dict | None` method
satisfies the `LlmClient` protocol, which is useful for tests and custom
gateways.

### Other building blocks

These modules are independently usable; see their docstrings for detail:

- `classify_temporal`, `infer_expiry` — temporal classification.
- `compress_texts`, `estimate_conversation_value` — session compression.
- `batch_dedup` — pre-LLM cosine deduplication.
- `AdmissionController`, `get_admission_preset` — the AMAC-v1 admission gate.
- `run_compaction`, `CompactionConfig` — clustering and merging old memories.
- `hermes_memory_lancedb_pro.reflection` — the reflection layer.

The complete export list is in
[architecture.md](architecture.md) and the package `__all__`.
