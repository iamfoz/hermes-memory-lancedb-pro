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
- **Temporal classifier** — distinguishes static facts from time-sensitive ("tomorrow", "next week") and infers expiry timestamps
- **Session compressor** — scores conversation turns by signal value (tool calls / corrections / decisions > greetings) before extraction
- **Batch dedup** — pre-LLM cosine dedup within extraction batches
- **Admission control** — pluggable AMAC-v1 gate (utility / confidence / novelty / recency / type-prior scoring) with `balanced` / `conservative` / `high-recall` presets
- **Memory compactor** — clusters and merges semantically similar old entries into single consolidated memories

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

There are **two pieces** to a working install, and they live in different
places. People sometimes confuse them — the actual code is **always** the
pip-installed package; the path under `~/.hermes/plugins/` is just a small
discovery shim that imports from the package.

**1. Install the package itself** (this is where the code runs from):

```bash
# From a clone of this repo:
pip install -e .

# Or, once published, straight from PyPI:
pip install hermes-memory-lancedb-pro
```

**2. Add the discovery shim** so hermes-agent can find the provider:

```bash
mkdir -p ~/.hermes/plugins/lancedb_pro
cat > ~/.hermes/plugins/lancedb_pro/__init__.py <<'EOF'
from hermes_memory_lancedb_pro.provider import (
    LanceDBProMemoryProvider,
    register_memory_provider,
)
__all__ = ["LanceDBProMemoryProvider", "register_memory_provider"]
EOF
```

That's it. The shim is ~5 lines; all real logic lives in the pip package
and gets updated whenever you `pip install -U`.

#### Upgrading from pre-0.2 installs

Earlier releases told users to clone the repo *into*
`~/.hermes/plugins/lancedb_pro/`. If you did that, the directory still
contains a stale full copy of the source. Replace it with the shim:

```bash
# Back up first if you've made local edits
mv ~/.hermes/plugins/lancedb_pro ~/.hermes/plugins/lancedb_pro.old

# Install the current version as a pip package
pip install -U hermes-memory-lancedb-pro   # or `pip install -e .` from a clone

# Recreate the discovery shim (see step 2 above)
mkdir -p ~/.hermes/plugins/lancedb_pro
cat > ~/.hermes/plugins/lancedb_pro/__init__.py <<'EOF'
from hermes_memory_lancedb_pro.provider import (
    LanceDBProMemoryProvider,
    register_memory_provider,
)
__all__ = ["LanceDBProMemoryProvider", "register_memory_provider"]
EOF
```

Verify which copy is actually loading:

```bash
python -c "import hermes_memory_lancedb_pro as m; print(m.__version__, m.__file__)"
```

If `__file__` points inside `~/.hermes/plugins/lancedb_pro/` and the
version is old, the stale checkout is still shadowing the pip install —
remove or rename that directory, then recreate the shim above.

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

# Every result has a portable `score` field (no underscore, 0..1, higher
# is better) that's safe to use across all three search modes. Use this
# for filtering/ranking in your own code:
for r in results:
    print(r["id"], r["score"], r["text"])

# Mode-specific raw fields prefixed with `_` (e.g. `_rrf_score`,
# `_distance`, `_score`) are internal/debug-only and only set on the
# mode that produced them. Don't rely on them — use `score` instead.

# Session-scoped search — only memories from this session, plus
# memories explicitly marked cross_session or core-tier (see below).
results = store.search(
    "concise responses",
    limit=5,
    session_id="sess-2026-05-07_abc",
    min_score=0.2,   # drop weak matches that would otherwise pollute context
)

# Update (supersede pattern — archives old, creates new).
# Returns the *new* memory ID after a text-changing supersede,
# the *same* mem_id after a metadata-only update, or None if not found.
new_id = store.update(mem_id=mem_id, text="Updated text here.", tier="core")
assert store.has_id(new_id)

# Metadata-only — no re-embedding, no supersede; returns the same id.
same_id = store.update(mem_id=new_id, importance=0.95, tier="core")
assert same_id == new_id

# Delete
store.forget(mem_id=mem_id)

# Utilities
store.has_id(mem_id)                           # False for deleted/archived
store.check_ids(["id-1", "id-2", "id-3"])      # batch existence check
store.stats()                                  # totals, breakdowns, archived count
store.list_memories(limit=10)                  # active memories only
store.list_memories(limit=10, include_archived=True)
store.purge_archived(grace_period_days=30)     # GC superseded rows

# Full-pipeline retrieval (rerank + MMR + lifecycle hooks).
# Both filters apply here too — recommended for use inside an agent.
retriever = MemoryRetriever(store)
hits = retriever.retrieve(
    "concise responses",
    limit=5,
    session_id="sess-2026-05-07_abc",
    min_score=0.2,
)

# After the LLM has actually consumed the recall, credit the memories
# (bypasses the per-recall throttle so the count reflects real usage):
store.mark_recall_used([h["id"] for h in hits])
```

## Avoiding "sticky" memory recall

The most common failure mode for an LLM agent backed by a memory store is
recall **stickiness** — old memories from a previous task or session keep
getting injected into a fresh conversation, and the model conflates the
prior context with the current question. Three knobs in this plugin
control that:

1. **`session_id`** — the strongest lever. When `MemoryRetriever.retrieve()`
   is called with `session_id="X"`, results are restricted to memories
   created in that session (matched against `metadata.source_session`),
   plus memories that opt in to cross-session visibility:
   - **`metadata.cross_session = True`** — explicit opt-in for memories
     that should surface across all sessions (user preferences, profile
     facts).
   - **`tier == "core"`** — long-term knowledge that's been promoted by
     the tier evaluator. Always cross-session.
   Pass `session_id=None` (default) for the legacy "see everything"
   behaviour.

2. **`min_score`** — drop weak semantic / lexical matches before they
   reach the LLM. The default is permissive (env `MEMORY_MIN_RECALL_SCORE`,
   default 0.0); raise it to ~0.2 if you'd rather have an empty recall
   than tangentially related noise.

3. **Access-count throttle** — `increment_access_count()` is throttled
   to once every `MEMORY_ACCESS_COUNT_THROTTLE_S` seconds (default 300).
   Without this, every retrieve bumps every result's access_count,
   raising its decay-frequency score, raising its composite, raising
   its retrieval rank — a feedback loop that produces "permanently
   sticky" memories. Use `force=True` or `mark_recall_used()` for the
   "this memory was definitely used" signal.

If you're integrating with **hermes-agent**, the recommended path is
to use the bundled `LanceDBProMemoryProvider` adapter (see below) — it
plumbs `session_id` through `prefetch()`, tags writes with
`source_session`, and credits recalls only on `sync_turn()`. That
combination is what closes the stickiness loop.

## Hermes Agent integration

This package ships a `MemoryProvider` adapter that hermes-agent's plugin
system can discover and instantiate. Drop a one-line shim into your
`~/.hermes/plugins/lancedb_pro/__init__.py`:

```python
# ~/.hermes/plugins/lancedb_pro/__init__.py
from hermes_memory_lancedb_pro.provider import (
    LanceDBProMemoryProvider,
    register_memory_provider,
)

__all__ = ["LanceDBProMemoryProvider", "register_memory_provider"]
```

Then activate it in your hermes config:

```yaml
memory:
  provider: lancedb_pro
```

The adapter:

- forwards `session_id` from `prefetch(query, session_id=...)` into
  `MemoryRetriever.retrieve(...)` so memories from prior sessions don't
  bleed into the current conversation
- tags every `sync_turn` write with `metadata.source_session` so future
  recalls in this session can find them
- caches the prefetched ids and credits them via `mark_recall_used()`
  on `sync_turn` — only the memories the model actually saw get their
  access_count bumped, eliminating the recall feedback loop
- mirrors built-in `/memory` tool writes via `on_memory_write`, marking
  them `cross_session=True` so user-curated facts surface in every session

The adapter has a soft, lazy dependency on hermes-agent: the rest of
this package (MemoryStore, MemoryRetriever, decay, scoring) imports
without hermes-agent installed and is fully usable as a standalone
library. Importing `hermes_memory_lancedb_pro.provider` succeeds either
way; `LanceDBProMemoryProvider()` will raise a clear `ImportError` if
hermes-agent isn't on PYTHONPATH at instantiation time.

This package has **no** dependency on jmunch-mcp; the two are completely
orthogonal and can be installed together or separately without
interfering.

The provider opportunistically uses three new hooks
(`before_prompt_build`, `on_recall_used`, `on_tool_call_observed`) that
are introduced via [a parallel hermes-agent PR](https://github.com/iamfoz/hermes-agent/pull/1).
On hosts that have those hooks merged, memory recall is injected into
the **system prompt** rather than the user message — a more authoritative
position for the LLM, with cleaner separation between agent context and
user input. On hosts without the hooks, the plugin falls back to the
classic user-message injection path automatically. Same wheel works
against both versions.

## Write-side helpers (CortexReach feature parity)

These modules support extraction / admission / compaction pipelines without
imposing them. Each is opt-in — call them yourself when the workflow fits.

### Temporal classification

```python
from hermes_memory_lancedb_pro import classify_temporal, infer_expiry

classify_temporal("I have a meeting tomorrow")  # → "dynamic"
classify_temporal("My favourite colour is blue")  # → "static"

infer_expiry("see you tomorrow")    # → +24h epoch ms
infer_expiry("see you next week")   # → +7d epoch ms
infer_expiry("static fact")          # → None
```

Use the result as `metadata_extra={"temporal_type": ..., "valid_until": ...}`
on `store.store(...)` so the Weibull decay engine's `temporal_type=="dynamic"`
fast-decay path activates and the entry is automatically marked stale after
its expiry passes.

### Session compressor

When persisting a long conversation under a fixed extraction budget, score
each turn first and keep the high-signal ones:

```python
from hermes_memory_lancedb_pro import compress_texts, estimate_conversation_value

result = compress_texts(turn_texts, max_chars=4000)
# result.texts is the chronological subset; result.scored has per-turn scores
# Decisions / corrections / tool_calls score 0.85-1.0; greetings 0.1.
```

`estimate_conversation_value(turn_texts)` returns a 0.0-1.0 estimate you can
use to decide whether the conversation is worth extracting at all.

### Batch dedup (pre-LLM)

Before sending candidate memories to an LLM dedup judge, drop the obvious
near-duplicates by cosine on their abstracts:

```python
from hermes_memory_lancedb_pro import batch_dedup

result = batch_dedup(abstracts, vectors, threshold=0.85)
survivors = [candidates[i] for i in result.surviving_indices]
```

### Admission control

A scoring gate that decides whether a candidate memory is worth admitting,
modelled on the CortexReach AMAC-v1 spec. Optional LLM client for the
"utility" feature; without one, that feature falls back to a neutral 0.5.

```python
from hermes_memory_lancedb_pro import (
    AdmissionController,
    CandidateMemory,
    get_admission_preset,
)

controller = AdmissionController(
    store,
    config=get_admission_preset("balanced"),  # or "conservative" / "high-recall"
    llm=my_extractor_llm,  # optional — see PR 3 for built-in adapters
)

candidate = CandidateMemory(
    category="preferences",
    abstract="user prefers dark mode",
    overview="...",
    content="user said they prefer dark mode in the IDE",
    vector=store.encode("user prefers dark mode"),
)
verdict = controller.evaluate(candidate, conversation_text="...")
if verdict.decision == "pass_to_dedup":
    store.store(text=candidate.content, ...)  # or merge into existing
```

The audit record on every verdict (`verdict.audit`) gives you the per-feature
score breakdown for tuning, plus a structured rejection reason.

### Smart extractor (LLM-driven 6-category extraction with switchable fallback)

When an LLM client is available, the smart extractor replaces "store
the raw user / assistant turn" with a structured 6-category pipeline
(profile / preferences / entities / events / cases / patterns), per-line
L0/L1/L2 metadata, vector-pre-filter + LLM-driven dedup decisions
(create / merge / skip / supersede / support / contextualize / contradict),
and admission-control gating.

**The extractor is switchable**: when `llm=None`, `extract_and_persist`
falls back to writing the raw user + assistant turns as separate
memories — exactly the legacy shape the hermes-agent provider has
always produced.

#### Auto-detection — piggy-backs on whatever the agent uses

`create_llm_client_from_env()` checks env vars in this order:

1. `MEMORY_EXTRACTION_API_KEY` + `MEMORY_EXTRACTION_BASE_URL` +
   `MEMORY_EXTRACTION_MODEL` → dedicated cheap-extractor override
2. `MEMORY_EXTRACTION_PROVIDER=anthropic` + `MEMORY_EXTRACTION_API_KEY`
   → Anthropic SDK
3. `OPENAI_API_KEY` (with optional `OPENAI_BASE_URL` /
   `OPENAI_MODEL`) → OpenAI-compat SDK; this is what makes
   **jmunch → Qwen / Ollama / LM Studio / OpenRouter just work** —
   the same env vars that point hermes-agent at your jmunch proxy
   serve the extractor too.
4. `ANTHROPIC_API_KEY` → Anthropic SDK
5. None of the above → returns `None` → extractor stays in fallback

The `LanceDBProMemoryProvider` adapter calls this factory at construction
time. If the result is non-None, smart extraction is used on every
`sync_turn`; if None, the legacy shape is preserved.

```python
from hermes_memory_lancedb_pro import (
    SmartExtractor, create_llm_client_from_env, MemoryStore,
)

store = MemoryStore.get_instance()
llm = create_llm_client_from_env()  # may be None
extractor = SmartExtractor(store, llm=llm)

extractor.has_llm           # → False if llm is None — fallback path
stats = extractor.extract_and_persist(
    user_content="I switched from Vim to Emacs.",
    assistant_content="Got it — I'll remember that.",
    session_key="sess-abc",
    scope="agent",
)
# stats.created / merged / skipped / superseded / supported / rejected
```

The provider auto-wires this for you:

```python
from hermes_memory_lancedb_pro import LanceDBProMemoryProvider

# auto_smart_extraction=True (default) → extractor built from env vars
provider = LanceDBProMemoryProvider()

# Or pass an explicit extractor / disable auto-detection
provider = LanceDBProMemoryProvider(
    smart_extractor=my_extractor,
    auto_smart_extraction=False,
)
```

#### Optional dependencies

The `openai` and `anthropic` SDKs are **not** declared as required deps —
they're imported lazily and only when the corresponding env vars resolve.
A typical user installs one of:

```bash
pip install openai      # for OpenAI / OpenRouter / Ollama / jmunch / etc.
pip install anthropic   # for native Anthropic
```

Without either, `create_llm_client_from_env()` returns `None` and the
extractor stays in fallback mode. No errors, no breaking import.

#### Custom LLM clients

Any object with a `complete_json(prompt, *, label) -> dict | None` method
satisfies `LlmClient` / `ExtractorLLM`. Useful for unit tests
(`FakeLlm` returning canned JSON) or for routing through your own gateway:

```python
class MyGatewayLlm:
    def complete_json(self, prompt, *, label=None):
        # ... your own HTTP call, retry, telemetry ...
        return {"memories": [...]}

extractor = SmartExtractor(store, llm=MyGatewayLlm())
```

### Reflection layer

Captures structured insights from an LLM-produced reflection summary
(invariants, derived deltas, lessons, decisions) and replays them on
recall. Lives under `hermes_memory_lancedb_pro.reflection.*`. The LLM
that *generates* the markdown summary is the caller's job (PR 3's
smart_extractor will be the typical caller).

```python
from hermes_memory_lancedb_pro.reflection import (
    MemoryStoreReflectionAdapter,
    store_reflection_to_lancedb,
    load_agent_reflection_slices_from_entries,
    load_reflection_mapped_rows_from_entries,
)

# After an LLM has produced a reflection summary like:
reflection_md = """## Invariants
- always answer in UK English
- prefer short responses

## Derived
- this run: ship the reflection layer
- next run: write tests for the orchestrator
"""

adapter = MemoryStoreReflectionAdapter(store)
result = store_reflection_to_lancedb(
    adapter,
    reflection_text=reflection_md,
    session_key="sk", session_id="sess-1",
    agent_id="alpha", command="reflect", scope="agent",
    run_at=int(time.time() * 1000),
)
# Writes one event payload, one item-invariant per invariant line, one
# item-derived per derived line, plus one combined-legacy bundle. The
# combined-legacy is deduped against existing entries (cosine ≥ 0.97).

# Later — recall on a fresh turn:
entries = store.list_memories(limit=200)
slices = load_agent_reflection_slices_from_entries(
    entries=entries, agent_id="alpha",
)
# slices.invariants — top 8 invariants (logistic decay; midpoint 45 days)
# slices.derived    — top 10 derived lines (midpoint 7 days)
```

Key behaviours ported from CortexReach's spec:

- **Logistic decay** — invariants stay relevant for ~45 days, derived
  for ~7. Used-fallback content gets a 0.75× penalty.
- **Ownership guard** — `derived` items are strictly per-agent; an
  agent can never see another agent's derived insights. `invariant`,
  `mapped`, and legacy entries allow a "main"-agent fallback.
- **Resolved-item suppression** — once all item rows in a section are
  marked `resolved_at`, the section is suppressed *unless* legacy rows
  carry unique unresolved content (the "P1/P2 fix"). Prevents the
  reflection fallback path from reviving advice the user has already
  moved on from.
- **Mapped rows** — separate `user_model` / `agent_model` / `lesson` /
  `decision` slices loaded via `load_reflection_mapped_rows_from_entries`,
  each ranked + capped per kind.
- **Prompt-injection guard** — every reflection line passes
  `sanitize_injectable_reflection_lines` (drops "ignore previous",
  `<system>...</system>`, etc.) before it can reach the LLM.

### Memory compactor

Periodic cron that clusters similar old memories and merges them into one:

```python
from hermes_memory_lancedb_pro import (
    CompactionConfig,
    record_compaction_run,
    run_compaction,
    should_run_compaction,
)

state_file = "~/.hermes/memory-lancedb/.compaction-state.json"
if should_run_compaction(state_file, cooldown_hours=24):
    result = run_compaction(
        store,
        CompactionConfig(min_age_days=7, similarity_threshold=0.88, min_cluster_size=2),
    )
    record_compaction_run(state_file)
    print(f"merged {result.clusters_found} clusters → -{result.memories_deleted} +{result.memories_created}")
```

Set `dry_run=True` to see the plan without writing.

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
| `MEMORY_RERANKER`              | `auto`                             | Which reranker to use: `langsearch`, `google`, `disabled`, or `auto`. In `auto` mode the system picks whichever service is configured; if **both** are present it logs a warning and disables reranking until you set this explicitly. |
| `LANGSEARCH_API_KEY`           | *(none)*                           | Enables LangSearch cross-encoder reranking. In `auto` mode, set this **or** `GOOGLE_CLOUD_PROJECT` (not both). On persistent 401/403/429 the reranker is disabled for the session with one warning. |
| `GOOGLE_CLOUD_PROJECT`         | *(none)*                           | Enables Google Discovery Engine Ranking API. Also accepted as `GOOGLE_PROJECT_ID` or `GOOGLE_PROJECT`. Ensure the [Discovery Engine API](https://console.cloud.google.com/apis/library/discoveryengine.googleapis.com) is enabled for the project. Free tier: 1,000 queries/month; ~$0.001/query thereafter. **No API key needed** — authentication uses Application Default Credentials (see below). |
| `MEMORY_GOOGLE_RANKING_MODEL`  | `semantic-ranker-512@latest`       | Google ranking model name (advanced). |
| `MEMORY_MAX_SCAN_ROWS`         | `100000`                           | Cap on full-table scans in `stats` / `purge_archived`             |
| `MEMORY_TIER_EVAL_FREQUENCY`   | `10`                               | Retrievals between full tier re-evaluations (set 0 to disable)    |
| `MEMORY_TIER_EVAL_BATCH`       | `500`                              | Rows fetched per tier evaluation                                  |
| `MEMORY_ACCESS_COUNT_THROTTLE_S` | `300`                            | Min seconds between access_count increments for the same memory  |
| `MEMORY_MIN_RECALL_SCORE`      | `0.0`                              | Default `min_score` for `MemoryRetriever.retrieve()` (0 = permissive) |
| `MEMORY_PREFETCH_LIMIT`        | `5`                                | Default recall size when used via the hermes-agent adapter        |
| `MEMORY_CROSS_SESSION_PROMOTION_K` | `3`                            | A memory recalled across this many distinct session_ids gets auto-promoted to `cross_session=True` |
| `MEMORY_INJECTION_GUARD`       | `warn`                             | Prompt-injection guard mode at write time: `off` / `warn` / `reject` / `sanitize` |

### Google Ranking API — Authentication Setup

The Google Discovery Engine Ranking API uses **OAuth2 authentication** —
API keys are explicitly rejected. The plugin uses the `google-auth` library's
[Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)
(ADC) chain, which tries credential sources in this order:

1. **Service account JSON** — recommended for production/server deployments:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

2. **Developer workstation** — run once on your local machine:
   ```bash
   gcloud auth application-default login
   ```
   Credentials are cached at
   `~/.config/gcloud/application_default_credentials.json`.

3. **GCP Metadata Server** — no configuration needed when running inside
   GCP Compute Engine, Cloud Run, GKE, etc.

Install `google-auth` before enabling this backend:

```bash
pip install 'hermes-memory-lancedb-pro[google]'
# or: pip install google-auth
```

Then set `GOOGLE_CLOUD_PROJECT` (and optionally `MEMORY_RERANKER=google`)
in `~/.hermes/.env`. No API key is required.

### LLM extraction (Smart Extractor)

The LLM-driven Smart Extractor needs to know which provider to call. By
default it falls back to `OPENAI_API_KEY` then `ANTHROPIC_API_KEY`, but
you almost always want the dedicated `MEMORY_EXTRACTION_*` overrides so
the extractor doesn't accidentally hit a different model than the agent.

| Variable                     | Purpose                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------|
| `MEMORY_EXTRACTION_API_KEY`  | API key for the extraction provider. Required to enable LLM extraction.                  |
| `MEMORY_EXTRACTION_BASE_URL` | OpenAI-compatible base URL (e.g. `http://127.0.0.1:7883/v1` for jmunch).                 |
| `MEMORY_EXTRACTION_MODEL`    | Model id (e.g. `Qwen3.6`, `gpt-4o-mini`, `claude-3-5-haiku-latest`).                     |
| `MEMORY_EXTRACTION_PROVIDER` | `openai` (default, OpenAI-compatible) or `anthropic` (native Anthropic SDK).             |

If you're routing through a local OpenAI-compatible proxy like
[**jmunch**](https://github.com/) — which is the recommended setup
when hermes-agent is configured with a `custom_providers` entry — point
the extractor at the same proxy:

```bash
# ~/.hermes/.env
MEMORY_EXTRACTION_PROVIDER=openai
MEMORY_EXTRACTION_BASE_URL=http://127.0.0.1:7883/v1
MEMORY_EXTRACTION_MODEL=Qwen3.6
MEMORY_EXTRACTION_API_KEY=local        # jmunch usually ignores the value
```

Without `MEMORY_EXTRACTION_*` set the extractor falls back to whichever
of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` it finds first; if neither is
set, extraction is disabled and writes go in raw (the rest of the
pipeline still works — decay, dedup, search etc.).

## Scripts

| Script                          | Purpose                                                  |
|---------------------------------|----------------------------------------------------------|
| `scripts/memory_init.sh`        | Create DB, seed from MEMORY.md, auto-recover if corrupt  |
| `scripts/memory_reset.sh`       | Wipe DB + reinitialise (uses sibling `memory_init.sh`)   |
| `scripts/memory_smoke_test.py`  | End-to-end test suite (use `--ephemeral` for tmp DB)     |

After `pip install -e .` you also get two console entry points:

```bash
hermes-memory-smoke --ephemeral    # full E2E smoke test
hermes-memory --help               # admin CLI: export / import / doctor
hermes-memory export --out backup.jsonl
hermes-memory import --in backup.jsonl --reembed
hermes-memory doctor               # diagnostic report; recommends purge / compaction
```

`MemoryStore.warmup()` is also worth calling at agent boot so the
embedding-model load + JIT cost doesn't land on the user's first turn:

```python
from hermes_memory_lancedb_pro import MemoryStore
MemoryStore.get_instance().warmup()
```

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
