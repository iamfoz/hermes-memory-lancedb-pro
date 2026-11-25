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
| `MEMORY_DB_DIR` | `<hermes home>/memory-lancedb` | Database directory, for the agent and every CLI command alike. Unset, each Hermes profile uses its own `$HERMES_HOME/memory-lancedb` (`~/.hermes/memory-lancedb` for the default profile). Setting it makes every profile share that one database. |
| `MEMORY_AUTO_ANCHOR_TTL_DAYS` | `14` | Session auto-anchors older than this many days are archived when a new conversation starts. `0` disables. |
| `MEMORY_MAX_SCAN_ROWS` | `100000` | Upper bound on single-query row fetches such as session-anchor and listing scans. `stats`, `purge` and `export` read every row regardless. |
| `MEMORY_AUTO_OPTIMIZE_EVERY` | `256` | Fragment-creating writes between automatic on-disk compactions. Each write creates a LanceDB fragment; compaction merges them so a high-volume store never exhausts the file-descriptor limit. `0` disables. |
| `MEMORY_MAX_TEXT_CHARS` | `8000` | Upper bound on a single memory's `text`, in characters. `store()` and `store_many()` reject oversized text rather than passing it to the embedder (where very large inputs can exhaust GPU / MPS memory). |
| `MEMORY_FD_LIMIT` | `4096` | Soft open-file limit the store raises its process to on construction (best-effort, capped at the hard limit). Guards against file-descriptor exhaustion under sustained write load. `0` disables. |
| `MEMORY_ACCESS_COUNT_THROTTLE_S` | `300` | Minimum seconds between `access_count` increments for one memory. Breaks the recall-frequency feedback loop that produces "sticky" memories. |
| `MEMORY_CROSS_SESSION_PROMOTION_K` | `3` | A memory recalled across this many distinct sessions is auto-promoted to `cross_session=True`. `0` disables. |
| `MEMORY_INJECTION_GUARD` | `warn` | Prompt-injection guard at write time: `off` / `warn` / `reject` / `sanitize`. The automatic session anchor is always written; if the guard flags the user's first message, the anchor is stored without the quoted snippet. |
| `MEMORY_EMBEDDING_DEVICE` | *(auto)* | Device for the embedding model: `cpu`, `mps`, `cuda`, `cuda:1`, …. Unset lets sentence-transformers choose. An unusable device logs a warning and falls back to automatic selection. `cpu` is a fallback for Apple Silicon hosts where MPS misbehaves. All embedder calls are serialised process-wide regardless of device. |

## Recall and retrieval

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_PREFETCH_LIMIT` | `5` | Memories recalled per turn via the hermes-agent adapter. |
| `MEMORY_MIN_RECALL_SCORE` | `0.0` | Default score floor for `MemoryRetriever.retrieve()`. Each result's `_final_score` is 0.7 × relevance (cosine similarity to the query, with a bonus when BM25 also matches) + 0.3 × memory health (recency, access frequency, importance). The scale is absolute, so a floor drops memories that do not match the query at all. Memory health alone contributes up to 0.3, so a floor below that can still let an unrelated but healthy memory through; choose the value from the `_final_score` of queries that should match nothing with your embedding model. |
| `MEMORY_TIER_EVAL_FREQUENCY` | `10` | Retrievals between full tier re-evaluations. `0` disables. |
| `MEMORY_TIER_EVAL_BATCH` | `500` | Rows fetched per tier evaluation. |
| `MEMORY_NEVER_CATEGORIES` | `greeting,ephemeral_chat` | Comma-separated categories never injected into recall. |
| `MEMORY_RECALL_CHAR_BUDGET` | `4800` | Approximate character budget for the recall block (~1200 tokens). `0` disables. |
| `MEMORY_RECALL_MAX_ITEM_CHARS` | `600` | Hard per-memory character cap inside the recall block. |
| `MEMORY_ACTIVE_TASK_PIN` | `on` | Include the active-task state (pinned tasks and the automatic session anchor) in the system prompt and in the pre-compression summary, and write the automatic session anchor. `off` disables all three. |
| `MEMORY_TASK_PROTOCOL` | `on` | Include the durable-task protocol text in the system prompt block (not in every recall block). `off` omits it. |
| `MEMORY_ENFORCE_VALID_UNTIL` | `on` | Hide memories whose `valid_until` has passed from every recall path (search, retriever, session anchors, extractor dedup). Rows are kept — `doctor`, `export`, and `get_by_id` still see them. `off` restores record-only behaviour. |

## Admission control

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_ADMISSION_PRESET` | `balanced` | AMAC-v1 admission gate: `balanced` / `conservative` / `high-recall` / `off`. An unrecognised value falls back to `balanced`. When jmunch is detected and this is left unset, the default is raised to `high-recall`. |
| `MEMORY_ADMISSION_AUDIT` | `on` | Write rejected admission candidates, with a short conversation excerpt, to `<db parent>/admission-audit/rejections.jsonl` (rotated at 1 MB, mode 0600, outside the database directory so reset/purge do not remove it). Set `off`, `0`, `false`, `no` or `disabled` to stop writing it. Admission control itself is unaffected. |

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
| `MEMORY_REFLECTION_MAX_CHARS` | `12000` | Cap on the role-labelled transcript sent to the session-end reflection LLM. The most recent part is kept and the cut is marked '[earlier conversation truncated]'. `0` disables the cap. |

## LLM extraction

The smart extractor needs an LLM. It resolves a client in this order: the full
`MEMORY_EXTRACTION_*` triple → `MEMORY_EXTRACTION_PROVIDER=anthropic` →
`OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → none (raw-turn fallback).

Credentials and endpoint (`MEMORY_EXTRACTION_*`, `OPENAI_*`, `ANTHROPIC_*`) are read through the host's per-profile secret scope when running inside Hermes, so a multiplexed profile uses its own `.env`; outside a host they are read from the process environment.

Conversation text and stored memories are passed to the extraction, dedup, merge, utility and reflection prompts inside delimited `<untrusted_data>` blocks marked as data, not instructions, and extracted abstract/overview/content are capped at 300/2000/4000 characters. This reduces prompt-injection risk; it is not a guarantee. Transient LLM failures (timeout, connection reset, 429, 5xx) are retried once after a 1-3 s jittered delay; the SDK clients also retry internally.

| Variable | Default | Purpose |
|---|---|---|
| `MEMORY_EXTRACTION_API_KEY` | *(none)* | API key for a dedicated extraction LLM. |
| `MEMORY_EXTRACTION_BASE_URL` | *(none)* | OpenAI-compatible base URL for a custom/self-hosted endpoint. |
| `MEMORY_EXTRACTION_MODEL` | *(none)* | Model id for extraction. |
| `MEMORY_EXTRACTION_PROVIDER` | *(none)* | `openai` (default behaviour) or `anthropic` to route the override through the Anthropic SDK. |
| `MEMORY_EXTRACTION_RATE_LIMIT` | `0` | Maximum extraction passes per hour. Each pass makes one extraction call plus dedup, admission-utility and merge calls per candidate, so this is not a cap on LLM calls or spend. When the cap is reached, turns are stored as raw user/assistant memories until the window clears. `0` disables the cap. |
| `MEMORY_WRITE_DRAIN_TIMEOUT_S` | `60` | Seconds that session end, session switch, pre-compress and shutdown wait for pending turn writes (one extraction pass can make several LLM calls). A warning is logged if the wait runs out; the writes keep running. Recall (prefetch) waits at most 2 seconds. |
| `OPENAI_API_KEY` | *(none)* | Fallback extraction LLM (OpenAI-compatible). |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for the OpenAI fallback. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for the OpenAI fallback. |
| `ANTHROPIC_API_KEY` | *(none)* | Fallback extraction LLM (Anthropic). |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model for the Anthropic fallback. |

The `openai` and `anthropic` SDKs are **not** required dependencies — they are
imported lazily, only when the matching variables resolve. Inside Hermes both
normally come from Hermes' own environment: `openai` is a core Hermes
dependency, and `anthropic` is part of Hermes' `all` extra, which a standard
`hermes pm install` selects. Never `pip install` into Hermes' environment — PM
drops undeclared packages on the next rebuild. Standalone, install them into
your own environment as usual.

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

The Google reranker needs the `google-auth` library. Inside Hermes, add it
through Hermes' own `google` extra: `hermes pm install --extra google`.
Standalone: `pip install 'hermes-memory-lancedb-pro[google]'`. Without it the
Google backend disables itself with an actionable warning.

## Hermes home and tasks

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_HOME` | `~/.hermes` | Hermes home directory. Resolution order: `--hermes-home` flag → `HERMES_HOME` → `~/.hermes`. |
| `HERMES_TASK_ROOT` | `<hermes home>/workspace/tasks` | Root directory for durable task ledgers. Unset, each Hermes profile uses `$HERMES_HOME/workspace/tasks` (`~/.hermes/workspace/tasks` for the default profile). A relative path is made absolute. |
| `MEMORY_MD` | `<hermes home>/memories/MEMORY.md` | Seed file for `init`. Defaults to the host's built-in memory file; the legacy `~/.hermes/memory/MEMORY.md` is used when only that exists. |
| `HF_TOKEN` | *(none)* | HuggingFace token; avoids rate limits when first downloading the embedding model. |

## Non-configurable constants

These are fixed in the code and listed here for reference:

- Embedding model: `nomic-ai/nomic-embed-text-v1.5` (768-dimensional). New databases encode stored text with the `search_document: ` prefix and queries with `search_query: `; the scheme is recorded in `<db dir>/embedding_scheme.json`. A database without that file that already has rows is a legacy, unprefixed store (see `reembed`). If the installed transformers cannot load the model's custom NomicBert code without consent, it is loaded again with `trust_remote_code=True` (nomic-embed-text models only).
- A vector index (cosine metric) is built once the store holds 256+ rows. An L2 vector index left by older versions is rebuilt as cosine on first open.
- Categories: `preference`, `fact`, `decision`, `entity`, `other`,
  `reflection`, `active_task`.
- Tiers: `core`, `working`, `peripheral`.
- Row states: `pending`, `confirmed`, `archived`.
