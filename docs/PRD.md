# Product Requirements Document — hermes-memory-lancedb-pro

| | |
|---|---|
| **Status** | Living document |
| **Version** | 1.0 |
| **Date** | 2026-07-08 |
| **Owner** | Martyn Forryan (@iamfoz) |
| **Product version covered** | 0.14.x |

---

## 1. Overview

`hermes-memory-lancedb-pro` is a LanceDB-backed **persistent memory provider
plugin** for [Hermes Agent](https://github.com/nousresearch/hermes-agent) that
also works as a **standalone Python library** with no hermes-agent dependency.
It gives an LLM agent durable, searchable recall across sessions: every
conversation turn is distilled into structured memories, scored for relevance
and freshness, and surfaced back into the prompt when — and only when — it
matters.

It is a Python re-implementation and extension of the TypeScript
`CortexReach/memory-lancedb-pro` (see `ACKNOWLEDGEMENTS`).

## 2. Problem statement

An LLM agent with a naive memory store fails in a predictable way — **sticky
memory bleed**: old memories from unrelated tasks keep getting injected into
fresh conversations, the model conflates them with the current question, and
recall quality collapses to the point where memory is worse than no memory.
Secondary failure modes this product must also solve:

1. **Context compaction amnesia** — long-running multi-step tasks silently
   lose their state when the host compacts the conversation window.
2. **Unbounded growth** — a memory store that only ever adds rows degrades in
   both quality (duplicates crowd out signal) and cost (scan/index size).
3. **Low-quality writes** — storing raw turns verbatim fills the store with
   greetings, acknowledgements, and noise.
4. **Trust** — memory is a prompt-injection vector (a poisoned "memory" is
   replayed into every future prompt) and a privacy surface (it must be
   local-first).

## 3. Goals

- **G1 — Recall that helps, never hurts.** Surfaced memories must be relevant
  to the *current* session and query; stale or foreign-session content must
  stay out of the prompt.
- **G2 — Durability across resets.** Task state and key session context
  survive context compaction, session restarts, and model resets.
- **G3 — Self-maintaining store.** Retention (tiers, decay, supersede,
  purge, compaction) runs automatically with sensible defaults; every
  automatic behaviour has a manual command and an env-var off switch.
- **G4 — Zero-config core, opt-in intelligence.** The store, retrieval, and
  decay pipeline run fully offline with no API keys. LLM extraction,
  admission control, reflection, and reranking activate only when configured.
- **G5 — First-class operability.** A human (or an agent) can inspect,
  search, back up, restore, repair, and clean the store from the CLI without
  writing Python.
- **G6 — Drop-in host integration.** Installing into hermes-agent is two
  steps (pip install + one config line); the plugin degrades gracefully on
  hosts missing newer hook APIs.

### Non-goals

- **Multi-user / server deployment.** This is a single-user, local,
  file-backed store. No network service, no auth layer.
- **Arbitrary embedding models.** The schema is pinned to
  `nomic-embed-text-v1.5` (768-d); changing models is a re-embed migration,
  not a config toggle (see Roadmap R6).
- **Being a general vector database.** LanceDB is the engine; this product is
  the memory *policy* layer on top of it.
- **Perfect recall.** The product optimises precision of injected context
  over exhaustive retrieval; `export` exists for exhaustive access.

## 4. Personas & primary use cases

| Persona | Description | Primary needs |
|---|---|---|
| **P1 — Hermes Agent user** | Runs hermes-agent daily with the plugin enabled. Doesn't read this repo. | Recall "just works"; no sticky bleed; setup is two steps; problems are diagnosable with `doctor`. |
| **P2 — Agent (the model itself)** | The LLM operating inside a session. | Durable task protocol it can follow; `task` CLI; pinned control blocks that survive compaction; recall blocks it can trust. |
| **P3 — Library integrator** | Builds a non-Hermes project needing hybrid-search memory. | Clean Python API (`MemoryStore`, `MemoryRetriever`, `SmartExtractor`); no forced heavy imports; no host coupling. |
| **P4 — Operator / power user** | Maintains a long-lived store; scripts backups and hygiene. | `stats`/`search`/`export`/`import`/`purge`/`compact`/`doctor` with `--json` and `--dry-run`; salvage path for corruption. |

Representative user stories:

- *As P1*, when I start a fresh conversation, memories from last week's
  unrelated task must not appear — but my durable preferences must.
- *As P2*, after the host compacts my context mid-task, my next turn must show
  me the task id, iteration, and next action so I continue instead of greeting.
- *As P3*, I can `pip install` the package and use the store offline without
  hermes-agent, torch downloads at import time, or API keys.
- *As P4*, before deleting anything I can `--dry-run` it; after a crash I can
  `export --salvage` whatever is still readable.

## 5. Functional requirements

Status legend: ✅ shipped (0.14.x) · 🔶 partial · ⬜ planned (see §8/§9).

### 5.1 Storage & schema

- **FR-1** ✅ Fixed 8-column LanceDB schema: `id` (UUID), `text`
  (BM25-indexed), `vector` (768-d cosine), `category`, `scope`, `importance`
  [0–1], `timestamp` (epoch ms), `metadata` (JSON string). `category` and
  `importance` are authoritative top-level columns, never mirrored into
  metadata. (`store.py`)
- **FR-2** ✅ Write API: `store`, `store_many`, `store_raw` (pre-computed
  vector), with text-size cap (`MEMORY_MAX_TEXT_CHARS`, default 8000) and
  importance clamping.
- **FR-3** ✅ **Supersede pattern**: `update(text=...)` archives the old row
  (`state=archived`, `superseded_by`, `invalidated_at`) and writes a new row
  (`supersedes`, reset counters) — full audit trail, no in-place vector
  drift. New row is written first so partial failure never hides data.
  `get_by_id` follows supersede chains (depth-capped).
- **FR-4** ✅ Prompt-injection **write guard** (`MEMORY_INJECTION_GUARD`:
  off/warn/reject/sanitize) blocks planting `<system>`-style payloads into
  memory.
- **FR-5** ✅ Path-keyed singleton store instances; lazy heavy imports
  (importing the package never pulls lancedb/torch); embedder loads on first
  encode under a double-checked lock, with off-thread `warmup()`.

### 5.2 Retrieval & ranking

- **FR-10** ✅ Three search modes — `vector` (cosine), `bm25` (FTS scoped to
  `text` only), `hybrid` (Reciprocal Rank Fusion, k=60) — all returning
  uniform row dicts with a normalised `score ∈ [0,1]`.
- **FR-11** ✅ 8-stage retriever pipeline: wide-net parallel vector+BM25 →
  BM25 ghost filter → vector-dominant fusion with BM25-confirmation bonus →
  entity-overlap boost → scoring pipeline (length-norm, hard floor, composite
  decay) → noise filter → optional cross-encoder rerank → MMR diversity →
  min-score gate → lifecycle hooks. (`retriever.py`, `decay.py`)
- **FR-12** ✅ **Weibull decay**: `recency = exp(-(λ·t)^β)`, per-tier β (core
  0.8 / working 1.0 / peripheral 1.3), recency exactly 0.5 at each tier's
  half-life; half-life importance-modulated and ÷3 for `temporal_type=dynamic`;
  composite = 0.4·recency + 0.3·frequency + 0.3·intrinsic with per-tier
  floors.
- **FR-13** ✅ Optional reranker backends (LangSearch cross-encoder, Google
  Discovery Engine via optional `google-auth` extra) selected by
  `MEMORY_RERANKER`, with per-backend kill-switches on auth/quota errors and
  fail-soft fallback to fusion ranking.
- **FR-14** ✅ Recall output is budgeted (`MEMORY_RECALL_CHAR_BUDGET`,
  per-item cap) and category-filterable (`MEMORY_NEVER_CATEGORIES`).

### 5.3 Anti-stickiness (session scoping)

- **FR-20** ✅ Recall visibility rule: a memory is visible to a session iff
  its `source_session` matches, OR it is explicitly `cross_session`, OR it is
  core-tier. (`_sql.match_session`)
- **FR-21** ✅ Access-count increments throttled (default 300 s) to break the
  recall→frequency→re-recall feedback loop; `mark_recall_used` credits only
  memories the response actually referenced (n-gram overlap), consuming a
  pending ledger to avoid double-credit.
- **FR-22** ✅ Auto-promotion to `cross_session` only after recall in K
  (default 3) distinct sessions.
- **FR-23** ✅ MMR near-duplicate demotion and noise filtering keep the
  injected block diverse and substantive.

### 5.4 Memory lifecycle & hygiene

- **FR-30** ✅ Three tiers (core/working/peripheral) with promotion/demotion
  rules driven by access count + composite score, evaluated in batches every
  N recalls; core never demotes straight to peripheral.
- **FR-31** ✅ `purge_archived(grace_period_days, dry_run)` hard-deletes
  archived rows past a grace window; auto-purge cooldown-gated at session
  end/shutdown; **manual `purge` subcommand with `--dry-run` and
  confirmation prompt**.
- **FR-32** ✅ Near-duplicate **memory compaction** (cosine ≥ 0.88 clusters
  of old rows merged into one, sources archived), auto-run on a weekly
  cooldown; **manual `compact` subcommand with `--dry-run` merge-plan
  preview**.
- **FR-33** ✅ LanceDB **fragment compaction** (`optimize()`) auto-triggered
  every N writes (default 256), single-flighted, to prevent FD exhaustion.
- **FR-34** ✅ Temporal classification (`static`/`dynamic`, EN + zh cues) and
  expiry inference feed decay and validity windows.

### 5.5 Extraction & admission (opt-in LLM)

- **FR-40** ✅ SmartExtractor pipeline: envelope stripping → LLM candidate
  extraction (≤10, 6-category taxonomy: profile/preferences/entities/events/
  cases/patterns) → batch cosine dedup → admission gate → per-candidate
  neighbour search → LLM dedup decision → one of 7 outcomes
  (CREATE/MERGE/SKIP/SUPERSEDE/SUPPORT/CONTEXTUALIZE/CONTRADICT).
- **FR-41** ✅ AMAC-v1 admission control scoring utility/confidence/novelty/
  recency/type-prior with presets (balanced/conservative/high-recall/off) and
  JSONL rejection audit.
- **FR-42** ✅ LLM client auto-detection from env (OpenAI-compatible triple →
  provider override → OpenAI → Anthropic), JSON-repair fallback, one-shot
  transient retry, hourly rate limit. **No LLM configured → legacy raw-turn
  fallback; the product remains fully functional.**
- **FR-43** ✅ Session-end **summary compression**: turns scored by
  information density (tool calls/corrections/decisions high; greetings low)
  and greedily fitted to a char budget as a single `session-summary` memory.
- **FR-44** ✅ **Reflection layer**: session-end LLM reflection produces
  invariant (durable) + derived (short-lived) insights, stored per-agent,
  ranked by logistic decay on recall, sanitised against injection, with
  resolved-item suppression.

### 5.6 Durable tasks

- **FR-50** ✅ Task ledger on disk (`state.json`, `results.jsonl`,
  `events.jsonl`, `log.md`) under `$HERMES_HOME/workspace/tasks/<id>/`,
  atomically written, re-read every turn — context compaction cannot lose
  progress.
- **FR-51** ✅ Full `task` CLI: create/list/show/resume/advance/complete/
  pin/hold/unhold/gc/to-skill, available in both CLI surfaces. `task pin`
  stores the control block as an always-recalled `active_task` memory.
- **FR-52** ✅ Task GC: archive→grace→delete retention with `gc_hold`
  exemption, cooldown-gated auto-run + manual `task gc --dry-run`.
- **FR-53** ✅ `task to-skill` scaffolds a draft SKILL.md/AGENTS.md from a
  completed task; agent-facing `durable-task` and `task-to-skill` skills ship
  in-repo and instruct the PATH-independent in-host CLI form.

### 5.7 Host integration

- **FR-60** ✅ 10 provider hooks (`plugin.yaml`): `system_prompt_block`
  (cache-stable, never raises), `prefetch` (the real recall path),
  `sync_turn` (non-blocking daemon-thread writes), `on_pre_compress`
  (recovery anchor), `on_memory_write` (mirrors host memory tool),
  `on_recall_used` + `on_tool_call_observed` (non-standard, inert on stock
  hosts), `on_session_switch`, `shutdown`, `on_session_end`.
  `before_prompt_build` deliberately not implemented (host-skip semantics).
- **FR-61** ✅ Three discovery paths: entry points (`hermes_agent.plugins` +
  legacy `hermes.plugins`), the `install-plugin` shim under
  `$HERMES_HOME/plugins/lancedb_pro/` (with auto-migration from two
  historical wrong paths and staleness auto-refresh), and
  `ctx.register_cli_command` for host-level CLI wiring.
- **FR-62** ✅ jmunch gateway compensation: env/header detection; recall
  widening; admission loosened to high-recall; pass-through headers on LLM
  calls. Zero code dependency on jmunch.

### 5.8 Operability (CLI)

- **FR-70** ✅ Console scripts `hermes-memory-lancedb-pro` and alias
  `hermes-memory`; in-host `hermes lancedb_pro <cmd>`; smoke entry point
  `hermes-memory-smoke [--ephemeral]`.
- **FR-71** ✅ `init` (seed from MEMORY.md, corrupt-DB auto-recovery),
  `reset`, `doctor` (counts, tier/category breakdown, orphan-chain and
  stale-row anomalies, recommendations **that reference real commands**),
  `export` (JSONL; strict-fail on corrupt store; `--salvage` recovery scan),
  `import` (`--reembed`, `--allow-existing`, source-id traceability).
- **FR-72** ✅ **`stats`** (with `--json`), **`search`** (all modes +
  session-scoping flags + `--json`, vectors stripped), **`purge`**
  (`--grace-days`, `--dry-run`, confirm-unless`-y`), **`compact`**
  (`--dry-run` plan preview, similarity/age/scan knobs) — registered in both
  CLI surfaces via one shared parser helper so they cannot drift.
- **FR-73** ✅ `--version` flag. Destructive commands (`init`, `reset`,
  `purge`) prompt for confirmation; `-y/--yes` for scripts; `-q/--quiet`
  everywhere.
- **FR-74** ✅ Corruption recovery: `salvage_scan` walks dataset versions
  newest-first, then fragment-by-fragment, read-only.

## 6. Non-functional requirements

- **NFR-1 — Local-first & private.** All persistent data lives in a local
  LanceDB directory (`~/.hermes/memory-lancedb` by default, profile-isolated
  via `HERMES_HOME`). No network calls unless an LLM/reranker is explicitly
  configured.
- **NFR-2 — Fail-soft.** No memory failure may break the host conversation:
  every hook, the extractor, the rerankers, and auto-maintenance
  log-and-continue. `system_prompt_block` never raises.
- **NFR-3 — Corruption-safe.** A corrupt store must never masquerade as an
  empty one (`export` strict-fails); salvage recovers what is readable;
  `init` auto-recovers.
- **NFR-4 — Thread-safe.** Singleton, embedder, optimize, sync-thread,
  pending-ledger, and reflection caches are lock-guarded; ledger writes are
  atomic (`os.replace`).
- **NFR-5 — Bounded cost.** Full scans capped (`MEMORY_MAX_SCAN_ROWS`);
  recall block char-budgeted; access bumps throttled; fragment compaction +
  FD-limit raise guard against resource exhaustion; embedder warmup happens
  off the first turn; vector index (IVF_PQ) built at ≥256 rows.
- **NFR-6 — Configurable, never required.** Every behaviour ships with a
  working default and an env-var override (~40 knobs; see
  `docs/configuration.md`). Zero mandatory configuration.
- **NFR-7 — Testable without heavy deps.** Pure-Python subsystems import
  without lancedb/torch; integration tests run on a real LanceDB with a
  deterministic stub embedder (no model download); LLM paths tested with
  fake clients. Current suite: ~1300 tests.
- **NFR-8 — Platform.** Python ≥ 3.11; primary dev target arm64-darwin
  (Apple Silicon); CI/Linux supported.
- **NFR-9 — Injection-resistant.** Write guard on stores, sanitiser on
  reflection recall lines, envelope stripping before extraction.

## 7. Current state (0.14.x)

All FRs marked ✅ above are shipped and tested. Subsystem inventory:
`store.py` (storage core), `retriever.py` (+ rerankers), `decay.py`
(scoring/tiers/MMR/noise), `provider.py` (hooks + auto-maintenance),
`smart_extractor.py` / `admission_control.py` / `memory_categories.py` /
`smart_metadata.py` / `extraction_prompts.py` / `llm_client.py` /
`batch_dedup.py` / `temporal_classifier.py` (extraction),
`session_compressor.py`, `memory_compactor.py`, `reflection/` (8 modules),
`task_ledger.py` / `task_gc.py` / `task_skill.py`, `jmunch.py`, `_sql.py`,
`_cli.py`. Docs: architecture, configuration, hermes-integration, hooks,
jmunch, usage, PRD (this document). Skills: `durable-task`, `task-to-skill`.

## 8. Gap analysis

Gaps identified in the 2026-07 review, with disposition:

| # | Gap | Severity | Disposition |
|---|---|---|---|
| GA-1 | `doctor` recommended `hermes-memory purge` and `run_compaction()` — neither existed as a command | Defect | **Fixed** — `purge` + `compact` subcommands added; doctor text updated |
| GA-2 | Archived-row purge and near-dup compaction reachable only via provider automation or Python | Gap | **Fixed** — manual `purge` / `compact` with `--dry-run` |
| GA-3 | No way to inspect recall from the terminal (debugging stickiness required Python) | Gap | **Fixed** — `search` subcommand incl. `--session-id` scoping |
| GA-4 | Store overview required the full doctor scan | Nice-to-have | **Fixed** — `stats` (`--json`) |
| GA-5 | No `--version` flag | Nice-to-have | **Fixed** |
| GA-6 | Standalone vs in-host CLI defined parsers twice — drift risk (this is how GA-1 happened) | Gap | **Fixed** for new commands (shared helper); pre-existing five commands still duplicated — see R1 |
| GA-7 | Agent-facing guidance used the PATH-dependent console script (exit 127 under some host PATHs) | Defect | **Fixed** in 0.14.x-unreleased (in-host `hermes lancedb_pro …` form) |
| GA-8 | `on_tool_call_observed` hook is a no-op placeholder (no entity extraction from tool calls) | Gap | Roadmap R3 |
| GA-9 | No machine-readable `doctor --json` for monitoring/cron | Nice-to-have | Roadmap R2 |
| GA-10 | Embedding model fixed; no migration story for model/dimension changes beyond `import --reembed` | Gap | Roadmap R6 |
| GA-11 | No scheduled-maintenance entry point (users must rely on session-end cooldowns or cron the CLI themselves) | Nice-to-have | Roadmap R4 |
| GA-12 | `HF_TOKEN` documented but consumed only by transitive deps — docs could mislead | Nice-to-have | Roadmap R2 |
| GA-13 | CONTRIBUTING/SECURITY/bug-report template said `doctor` reports the package version — it didn't | Defect | **Fixed** — `doctor` and `stats` now print `package_version` (JSON included) |
| GA-14 | `doctor -q/--quiet` is declared but never read — an inert flag (pre-existing) | Nice-to-have | Roadmap R2 (wire or retire) |
| GA-15 | `[Unreleased]`/`[0.14.3]`/`[0.14.2]` changelog headers had no link definitions — dead compare links | Nice-to-have | **Fixed** |
| GA-16 | No test coverage for the `_cmd_init`/`_cmd_reset`/`smoke_main`/task-CLI wrapper layer (module logic is tested; the argparse wrappers are not) | Gap | Roadmap R2 |
| GA-17 | `MemoryStore.optimize()` (fragment compaction) has no manual CLI trigger — automatic only | Nice-to-have | Roadmap R4 (`maintain` includes it) |

## 9. Roadmap

**Near term (0.15.x)**
- **R1** — De-duplicate the remaining parser definitions (`init`, `reset`,
  `doctor`, `export`, `import`) into shared helpers like
  `_add_store_admin_parsers`, eliminating the last CLI-drift surface.
- **R2** — Operability polish: `doctor --json` (machine-readable health for
  cron/monitoring); wire or retire the inert `doctor -q` flag (GA-14); test
  coverage for the CLI wrapper layer — `init`, `reset`, `smoke_main`, task
  dispatch (GA-16); documentation pass on env-var edge cases (`HF_TOKEN`).

**Mid term (0.16.x)**
- **R3** — Implement `on_tool_call_observed`: extract entities/facts from
  successful tool calls (file paths, project names, API objects) as
  peripheral-tier candidates.
- **R4** — `hermes-memory maintain` — one command running
  purge → compact → optimize → task gc with a single cooldown state, suitable
  for cron.
- **R5** — Backup rotation: `export --rotate N` writing timestamped snapshots.

**Long term**
- **R6** — Embedding-model migration: side-by-side re-embed with progress +
  rollback (new table, verify, swap), unlocking model upgrades.
- **R7** — Multi-agent memory namespaces beyond `scope` (per-agent stores
  with shared core tier) — extends the existing `MEMORY_REFLECTION_AGENT_ID`
  concept store-wide.
- **R8** — Optional encryption-at-rest for the LanceDB directory.

## 10. Success metrics

- **Recall precision:** ≥ 90 % of injected memories judged on-topic for the
  current session (target measured via `bad_recall_count` /
  `injected_count` telemetry already stored per row).
- **Stickiness:** zero foreign-session, non-core, non-cross-session memories
  in recall blocks (enforced by design; regression-tested).
- **Durability:** 100 % of pinned tasks resumable after simulated compaction
  (covered by provider tests).
- **Hygiene:** archived ratio stays < 30 % on long-lived stores with default
  automation (the `doctor` warning threshold).
- **Operability:** every destructive operation previewable (`--dry-run`) and
  every automated behaviour manually invocable — no Python required.
- **Quality bar:** test suite green (~1300 tests), ruff clean, on every
  release.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LanceDB API drift across versions (`list_tables` return types, `checkout_latest` availability) | Version-tolerant fallbacks in store; strict-fail export; salvage path |
| LLM extraction cost/latency creep | Rate limiter, session compression budget, admission control, full non-LLM fallback |
| Memory as prompt-injection vector | Write guard + reflection sanitiser + envelope stripping (NFR-9) |
| Host hook API evolution | Capability-sniffing registration (`register_cli_command` optional; non-standard hooks inert) |
| Index/FD exhaustion on write-heavy use | Auto-optimize cadence + FD-limit raise + EMFILE detection |
| Silent store corruption | Doctor anomaly scan; export strict-fail; salvage; init auto-recovery |

---

## Appendix A — CLI surface (0.14.x + unreleased)

```
hermes-memory[-lancedb-pro] [--version]
  init | reset | doctor | export | import
  stats [--json] | search QUERY [--mode|--limit|--category|--scope|--session-id|--min-score|--json]
  purge [--grace-days N] [--dry-run] [-y] | compact [--dry-run] [--scope]... [--min-age-days] [--similarity] [--max-scan]
  task create|list|show|resume|advance|complete|pin|gc|hold|unhold|to-skill
  install-plugin | uninstall-plugin
hermes-memory-smoke [--path | --ephemeral]
hermes lancedb_pro <everything above except install/uninstall-plugin>
```

## Appendix B — Configuration

The complete environment-variable reference (~40 knobs across store, recall,
admission, compaction/purge, task-GC, reflection, session-summary, LLM,
jmunch, reranker) lives in [configuration.md](configuration.md). Design
details in [architecture.md](architecture.md); hook semantics in
[hooks.md](hooks.md).
