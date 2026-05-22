# Configuration reference

Every tuning knob in `hermes-memory-lancedb-pro` is an environment variable.
All of them are **optional** — the defaults are production-ready. Set them in
`~/.hermes/.env` (for a Hermes install) or in the process environment (for
standalone use).

Truthy values for boolean-style variables are `1`, `true`, `yes`, `on`
(case-insensitive); their negatives are `0`, `false`, `no`, `off`.

## Store and database

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_DB_DIR` | `~/.hermes/memory-lancedb` | Database directory. |
| `MEMORY_MAX_SCAN_ROWS` | `100000` | Upper bound on full-table scans (`stats`, `purge_archived`). |
| `MEMORY_AUTO_OPTIMIZE_EVERY` | `256` | Fragment-creating writes between automatic on-disk compactions. Each write creates a LanceDB fragment; compaction merges them so a high-volume store never exhausts the file-descriptor limit. `0` disables. |
| `MEMORY_MAX_TEXT_CHARS` | `8000` | Upper bound on a single memory's `text`, in characters. `store()` and `store_many()` reject oversized text rather than passing it to the embedder (where very large inputs can exhaust GPU / MPS memory). |
| `MEMORY_FD_LIMIT` | `4096` | Soft open-file limit the store raises its process to on construction (best-effort, capped at the hard limit). Guards against file-descriptor exhaustion under sustained write load. `0` disables. |
| `MEMORY_ACCESS_COUNT_THROTTLE_S` | `300` | Minimum seconds between `access_count` increments for one memory. Breaks the recall-frequency feedback loop that produces "sticky" memories. |
| `MEMORY_CROSS_SESSION_PROMOTION_K` | `3` | A memory recalled across this many distinct sessions is auto-promoted to `cross_session=True`. `0` disables. |
| `MEMORY_INJECTION_GUARD` | `warn` | Prompt-injection guard at write time: `off` / `warn` / `reject` / `sanitize`. |

## Recall and retrieval

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_PREFETCH_LIMIT` | `5` | Memories recalled per turn via the hermes-agent adapter. |
| `MEMORY_MIN_RECALL_SCORE` | `0.0` | Default score floor for `MemoryRetriever.retrieve()`. Raise to ~`0.2` to drop weak matches. |
| `MEMORY_TIER_EVAL_FREQUENCY` | `10` | Retrievals between full tier re-evaluations. `0` disables. |
| `MEMORY_TIER_EVAL_BATCH` | `500` | Rows fetched per tier evaluation. |
| `MEMORY_NEVER_CATEGORIES` | `greeting,ephemeral_chat` | Comma-separated categories never injected into recall. |
| `MEMORY_RECALL_CHAR_BUDGET` | `4800` | Approximate character budget for the recall block (~1200 tokens). `0` disables. |
| `MEMORY_RECALL_MAX_ITEM_CHARS` | `600` | Hard per-memory character cap inside the recall block. |
| `MEMORY_ACTIVE_TASK_PIN` | `on` | Pin `active_task` memories to the front of recall, bypassing filters and the char budget. |
| `MEMORY_TASK_PROTOCOL` | `on` | Prepend the durable-task protocol text to every recall block. |

## Admission control

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_ADMISSION_PRESET` | `balanced` | AMAC-v1 admission gate: `balanced` / `conservative` / `high-recall` / `off`. An unrecognised value falls back to `balanced`. When jmunch is detected and this is left unset, the default is raised to `high-recall`. |

## Compaction and purge

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_AUTO_PURGE_COOLDOWN_HOURS` | `24` | Hours between automatic `purge_archived` runs at session end. `0` disables. |
| `MEMORY_PURGE_GRACE_DAYS` | `30` | Archived rows younger than this are spared during a purge. |
| `MEMORY_AUTO_COMPACT_COOLDOWN_HOURS` | `168` | Hours between automatic memory-compaction runs (weekly). `0` disables. |

## Task-ledger garbage collection

Completed task directories under the task root are archived (or deleted) once
past the retention window. See [usage.md](usage.md) for the `task gc` command.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_TASK_GC_COOLDOWN_HOURS` | `168` | Hours between automatic task-GC runs at session end. `0` disables auto-GC. |
| `MEMORY_TASK_RETENTION_DAYS` | `30` | Completed tasks older than this are archived (or deleted). |
| `MEMORY_TASK_GC_MODE` | `archive` | `archive` moves the task dir under `<task-root>/archive/`, keeping its audit trail; `delete` hard-deletes it. |
| `MEMORY_TASK_ARCHIVE_GRACE_DAYS` | `90` | Archived task dirs older than `retention + this` are hard-deleted; `0` disables that second stage. |

## Reflection

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_REFLECTION` | `on` | Capture session reflections at session end and replay ranked invariant/derived slices on recall. |
| `MEMORY_REFLECTION_SCAN_LIMIT` | `200` | Rows scanned when loading reflection slices for recall. |
| `MEMORY_REFLECTION_AGENT_ID` | `main` | Agent identity used for reflection ownership. Multi-agent hosts may instead pass `agent_id` to `initialize()`. |

## Session summary

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_SESSION_SUMMARY_MAX_CHARS` | `4000` | Character budget for the compressed session-end transcript. `0` disables session-summary writes. |
| `MEMORY_SESSION_SUMMARY_MIN_MESSAGES` | `2` | Minimum messages before a session summary is written. |

## LLM extraction

The smart extractor needs an LLM. It resolves a client in this order: the full
`MEMORY_EXTRACTION_*` triple → `MEMORY_EXTRACTION_PROVIDER=anthropic` →
`OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → none (raw-turn fallback).

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_EXTRACTION_API_KEY` | *(none)* | API key for a dedicated extraction LLM. |
| `MEMORY_EXTRACTION_BASE_URL` | *(none)* | OpenAI-compatible base URL for a custom/self-hosted endpoint. |
| `MEMORY_EXTRACTION_MODEL` | *(none)* | Model id for extraction. |
| `MEMORY_EXTRACTION_PROVIDER` | *(none)* | `openai` (default behaviour) or `anthropic` to route the override through the Anthropic SDK. |
| `MEMORY_EXTRACTION_RATE_LIMIT` | `0` | Maximum extraction LLM calls per hour. `0` disables the cap. |
| `OPENAI_API_KEY` | *(none)* | Fallback extraction LLM (OpenAI-compatible). |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for the OpenAI fallback. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for the OpenAI fallback. |
| `ANTHROPIC_API_KEY` | *(none)* | Fallback extraction LLM (Anthropic). |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model for the Anthropic fallback. |

The `openai` and `anthropic` SDKs are **not** required dependencies — they are
imported lazily, only when the matching variables resolve. Install whichever you
need: `hermes-pip install openai` or `hermes-pip install anthropic`.

## jmunch

See [jmunch.md](jmunch.md) for the full guide.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_JMUNCH_MODE` | *(none)* | Declare up front that a jmunch gateway is in the LLM path. Optional — jmunch is also auto-detected. |
| `MEMORY_JMUNCH_PREFETCH_LIMIT` | `12` | Recall limit used *instead of* `MEMORY_PREFETCH_LIMIT` while jmunch is in use. |
| `MEMORY_JMUNCH_MIN_RECALL_SCORE` | `0.0` | Recall score floor while jmunch is in use (only when `min_score` is not set explicitly). |

## Reranker credentials

The reranker is optional; without credentials, retrieval simply skips the
rerank stage.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_RERANKER` | `auto` | Reranker backend: `auto` / `langsearch` / `google` / `disabled`. In `auto`, if both backends are configured the reranker is disabled with a warning until you choose explicitly. |
| `MEMORY_GOOGLE_RANKING_MODEL` | `semantic-ranker-512@latest` | Google Discovery Engine ranking model. |
| `LANGSEARCH_API_KEY` | *(none)* | Enables the LangSearch cross-encoder reranker. |
| `GOOGLE_CLOUD_PROJECT` | *(none)* | Enables the Google Discovery Engine reranker. Also read as `GOOGLE_PROJECT_ID` / `GOOGLE_PROJECT`. Authentication uses Application Default Credentials — no API key. |

The Google reranker needs the `google-auth` library:
`hermes-pip install 'hermes-memory-lancedb-pro[google]'`.

## Hermes home and tasks

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_HOME` | `~/.hermes` | Hermes home directory. Resolution order: `--hermes-home` flag → `HERMES_HOME` → `~/.hermes`. |
| `HERMES_TASK_ROOT` | `~/.hermes/workspace/tasks` | Root directory for durable task ledgers. |
| `MEMORY_MD` | `~/.hermes/memory/MEMORY.md` | Seed file for `init`. |
| `HF_TOKEN` | *(none)* | HuggingFace token; avoids rate limits when first downloading the embedding model. |

## Non-configurable constants

These are fixed in the code and listed here for reference:

- Embedding model: `nomic-ai/nomic-embed-text-v1.5` (768-dimensional).
- A vector index is built once the store holds 256+ rows.
- Categories: `preference`, `fact`, `decision`, `entity`, `other`,
  `reflection`, `active_task`.
- Tiers: `core`, `working`, `peripheral`.
- Row states: `pending`, `confirmed`, `archived`.
