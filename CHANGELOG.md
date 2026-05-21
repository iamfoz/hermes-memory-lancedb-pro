# Changelog

All notable changes to **hermes-memory-lancedb-pro** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is in the `0.y.z` series the public API may change between
minor versions; breaking changes are called out under **Changed** and
**Removed**.

---

## [0.14.0] — 2026-05-21

### Added
- **`task to-skill` candidate discovery.** `task to-skill --list` (and
  `--search "<keywords>"`) lists completed tasks — live *and* archived — that
  could be turned into a skill, so an older task can be found and selected
  without knowing its id. `task to-skill <id>` now resolves a task whether it
  is live or archived (including GC's collision-suffixed archive names).
- **`task-to-skill` skill** (`skills/task-to-skill/`) — teaches the agent the
  full user-initiated flow: scaffold a recent task directly, or surface
  candidate tasks for the user to choose (as client option-buttons where the
  client supports them, otherwise a numbered list, with a keyword-search
  escape hatch), then author a polished `SKILL.md` from the draft. The
  synthesis is done by the agent itself, not a plugin-side LLM call.

---

## [0.13.0] — 2026-05-21

### Added
- **Task-ledger garbage collection.** Completed task directories no longer
  accumulate without bound. A cooldown-gated GC runs at session end, and is
  available manually as `hermes-memory-lancedb-pro task gc` (with `--dry-run`):
  completed tasks older than `MEMORY_TASK_RETENTION_DAYS` (default 30) are
  archived under `<task-root>/archive/`, preserving their `results.jsonl` /
  `events.jsonl` audit trail; archived directories are hard-deleted after a
  further `MEMORY_TASK_ARCHIVE_GRACE_DAYS` (default 90).
  `MEMORY_TASK_GC_MODE=delete` hard-deletes outright instead. Every action is
  recorded to an append-only `<task-root>/.task-gc-log.jsonl`. Running tasks
  are never touched (only reported when long-idle), and the matching
  `active_task` pin memory is removed so its `state_path` cannot dangle.
  Auto-GC is gated by `MEMORY_TASK_GC_COOLDOWN_HOURS` (default 168; `0`
  disables).
- **Per-task GC holds** — `task hold <id>` exempts a task from garbage
  collection entirely; `task unhold <id>` releases it. `task list` shows a
  `[held]` marker.
- **`task to-skill <id>`** — scaffold a draft reusable skill (`SKILL.md` +
  `AGENTS.md`) from a task. The objective and invariants transfer directly and
  the Protocol section is seeded with the task's iteration history, ready for
  the author to refine.
- `complete_task` records an explicit `completed_at` timestamp.

### Security
- Task ids are now validated — `create_task` rejects ids containing a path
  separator or `.` / `..`, closing a directory-traversal vector now that GC
  performs filesystem operations on task directories.

---

## [0.12.3] — 2026-05-21

### Changed
- jmunch detection no longer version-gates on a jmunch release number. The
  gateway-side support it relies on — the `X-Jmunch-Gateway` header and the
  `X-Jmunch-Inject` / `X-Jmunch-Handleify` pass-through headers — currently
  lives in a fork of jmunch-mcp rather than a versioned upstream release, so
  detection now keys purely on the presence of the `X-Jmunch-Gateway` header.

### Removed
- The internal jmunch version parsing and minimum-version check, the
  "upgrade jmunch" log warning, and the `jmunch_supports_passthrough()` /
  `observed_jmunch_version()` helpers — all tied to a release-version model
  that does not apply to an unversioned fork.

---

## [0.12.2] — 2026-05-21

### Added
- `MEMORY_MAX_TEXT_CHARS` (default 8000) — an upper bound on a single
  memory's `text`. `store()` and `store_many()` now reject oversized text
  with a clear error instead of passing it to the embedder, where very large
  inputs could exhaust GPU / MPS memory. A memory entry is meant to be a
  distilled fact; long content should be summarised — the smart extractor's
  job — before it is stored.

---

## [0.12.1] — 2026-05-21

### Added
- `MemoryStore.delete(mem_id)` — an alias for `forget()`, added because
  `delete` is the conventional method name to reach for. Behaviour is
  identical: a hard delete of the memory's live row.

---

## [0.12.0] — 2026-05-20

### Added
- **jmunch gateway detection** (`hermes_memory_lancedb_pro.jmunch`).
  `is_jmunch_in_use()` reports whether a [jmunch](https://github.com/) gateway
  sits in the LLM path. Detection is two-stage and adds no code dependency on
  jmunch-mcp: it is confirmed passively from the `X-Jmunch-Gateway` response
  header that a pass-through-capable jmunch gateway stamps on every reply (any
  port; latched by `record_response_headers()`), and can be declared up front
  with `MEMORY_JMUNCH_MODE=true` so startup-time tuning is correct before the
  first response arrives.
- **Extractor pass-through in jmunch mode** — the memory extractor's LLM calls
  send `X-Jmunch-Inject: false` and `X-Jmunch-Handleify: false` when jmunch is
  in use, so the gateway relays those calls verbatim and the extractor sees
  full-fidelity tool content. Both headers are inert on non-jmunch endpoints.
- **jmunch-mode recall widening** — a higher prefetch limit
  (`MEMORY_JMUNCH_PREFETCH_LIMIT`, default 12), a permissive `min_score`
  (`MEMORY_JMUNCH_MIN_RECALL_SCORE`, default 0.0), and a `high-recall` admission
  default. A jmunch gateway lossily compresses the agent's history; the injected
  memory block is not handle-ified, so widening recall pushes task context back
  through that lossless channel. Resolved per recall, so jmunch confirmed
  mid-session takes effect from the next turn. Explicit settings are respected.
- `MemoryStore.optimize()` — public, best-effort LanceDB fragment compaction,
  plus `MEMORY_AUTO_OPTIMIZE_EVERY` (default 256; `0` disables) controlling how
  often the write path compacts automatically.

### Fixed
- **File-descriptor exhaustion under sustained write load.** Every write created
  a new on-disk LanceDB fragment and every read opens every fragment, so a store
  that had absorbed thousands of single-row writes exhausted `ulimit -n` and
  degraded catastrophically. The write path now funnels through a single helper
  that compacts small fragments automatically and raises an actionable error if
  the descriptor limit is still reached.

---

## [0.11.1] — 2026-05-19

### Fixed
- **Store stress-test bugs** — `update()` wrote to the wrong row after a
  supersede chain; a stale in-process table view returned zero rows after a
  subprocess CLI write (`_checkout_latest()` now refreshes before every read);
  `CompactionConfig` gained `min_age_hours` for immediate-compaction tests;
  tier SQL filter and BM25 `min_score` cleanup corrected.
- **Greeting-replay loop** — session anchors injected the first stored memory
  every turn with no noise filter, so an opening "Hello" was echoed back. The
  anchor path now applies the minimum-length and `is_noise()` guards already
  used on relevance results.
- **Active-task hooks** — attribution and recall formatting fixes; the task
  protocol is routed through real hooks; `before_prompt_build` is treated as a
  proper hook; three hook-interaction bugs resolved.
- **Anchor stability** — the session anchor stays stable across compaction
  session rotation; auto-anchors are scoped per conversation to prevent
  cross-bleed; the task ledger is profile-isolated.
- `on_memory_write` now matches the built-in memory tool contract.

---

## [0.11.0] — 2026-05-17

### Added
- **Hindsight recall scoring** — prompted by a best-practices review:
  an extraction `context` field, structured-JSON conversation input, named
  `entities` extraction, an entity-overlap retrieval boost, an evidence-weighted
  confidence blend in `compute_decay_score`, a `freshness_trend` tag
  (`forming` / `strengthening` / `weakening`) in the recall block, and a
  temporal-query post-filter that narrows results to a detected time window.
- **Durable task ledger** (`task_ledger` module) — task state persisted under
  `~/.hermes/workspace/tasks/<task_id>/` (`state.json`, `results.jsonl`,
  `events.jsonl`, `log.md`) so a multi-step task survives context compaction.
  Adds `create_task`, `load_state`, `save_state`, `advance_iteration`,
  `complete_task`, `build_control_block`, `looks_like_reset`, `list_tasks`.
- **Task CLI** — `task create|list|show|resume|advance|complete|pin` in both the
  standalone CLI and the `hermes lancedb_pro` plugin namespace.
- **Durable-task protocol** — the protocol is injected into every prompt via
  `before_prompt_build` (`MEMORY_TASK_PROTOCOL`, default `on`), with an
  installable `skills/durable-task/` skill.
- **Recall guardrails** — `MEMORY_NEVER_CATEGORIES`, `MEMORY_RECALL_CHAR_BUDGET`
  (default 4800), and `MEMORY_ACTIVE_TASK_PIN` keep pinned task state present
  and stop stale chatter from crowding the recall block.

---

## [0.10.1] — 2026-05-15

### Fixed
- **Plugin discovery** — the install path and `kind: memory` routing were
  corrected so `hermes memory setup` reliably finds the provider.
- **Plugin CLI** — `register_cli` rewritten to match the Hermes plugin-CLI spec
  (a fresh subparser for the `hermes lancedb_pro` namespace); `init`, `reset`,
  `doctor`, `export`, `import` exposed in both the standalone and plugin CLIs;
  `init` / `reset` gated behind a confirmation prompt (`-y` to skip).
- **Cold-start recall** — `prefetch` / `before_prompt_build` flush the pending
  background write before querying, closing the first-turn write/read race;
  first+recent session anchors keep task framing in recall past turn 4;
  `on_memory_write` implements `edit` / `delete` / `replace_all`.

---

## [0.10.0] — 2026-05-14

### Added
- **Background embedding-model warmup** — `initialize()` warms the embedding
  model in a daemon thread, so the 10–30 s cold-start cost lands during session
  boot rather than the user's first turn.
- **Automatic memory compaction** — cooldown-gated compaction runs at session
  end, clustering and merging near-duplicate old memories
  (`MEMORY_AUTO_COMPACT_COOLDOWN_HOURS`, default weekly).
- **Admission control by default** — the smart extractor is built with an
  AMAC-v1 gate; `MEMORY_ADMISSION_PRESET` selects
  `balanced` / `conservative` / `high-recall` / `off`.
- **Reflection layer wired into the lifecycle** — session-end reflections are
  generated and replayed into recall (`MEMORY_REFLECTION` and friends).

---

## [0.9.0] — 2026-05-12

### Added
- `install-plugin` / `uninstall-plugin` CLI commands create and remove the
  discovery shim automatically; upgrades become `pip install -U`.
- Session-summary memories — at session end the conversation is condensed into
  a structured memory entry, superseding any prior summary for the session.
- Entry-point registration (`hermes.plugins` group) so newer hermes-agent
  builds can discover the provider without the shim.

### Fixed
- Hermes plugin spec-compliance audit — 28 correctness bugs across four review
  sweeps, including the Weibull decay `λ`/`β` ordering, a `sync_turn` /
  `on_session_end` thread race, tool-call vs tool-result classification,
  compaction provenance, reflection metadata parsing, and the min-score gate.

---

## [0.8.0] — 2026-05-09

### Added
- **Google Discovery Engine reranker** — uses the Ranking API
  (`semantic-ranker-512@latest`), authenticated via Application Default
  Credentials. `MEMORY_RERANKER` (`auto` / `langsearch` / `google` / `disabled`)
  selects the backend; `MEMORY_GOOGLE_RANKING_MODEL` overrides the model.
- Auto-purge of archived rows at session end
  (`MEMORY_AUTO_PURGE_COOLDOWN_HOURS`, `MEMORY_PURGE_GRACE_DAYS`).
- `doctor` CLI command — database-health report with purge / compaction advice.

### Fixed
- Reranker API keys were read at import time, ignoring env vars set afterward;
  reading is deferred to `__init__`.
- Replaced the rejected `x-goog-api-key` header with OAuth2 ADC for the Google
  Ranking API.

---

## [0.7.1] — 2026-05-08

### Fixed
- Five post-deploy issues: a stale singleton after `forget`, an incorrect
  archived-row count in `stats`, the BM25 index not rebuilding after bulk
  inserts, tier evaluation skipping rows near a batch boundary, and
  `purge_archived` ignoring `grace_period_days` on first run.
- `score_text` signal inversions — tool-call texts scored below neutral and
  corrections scored as acknowledgements; comparisons corrected.

---

## [0.7.0] — 2026-05-07

### Added
- **New MemoryProvider hooks** (feature-detected, with graceful fallback):
  - `before_prompt_build` — injects recalled memories into the system prompt.
  - `on_recall_used` — credits memories only when the LLM consumed them.
  - `on_tool_call_observed` — feeds tool calls to the extractor for context.
- `MEMORY_INJECTION_GUARD` (`off` / `warn` / `reject` / `sanitize`) — a
  prompt-injection guard applied at write time.

---

## [0.6.0] — 2026-05-05

### Added
- Ops and reliability tooling: `export` / `import --reembed`, `warmup()` for
  eager model load, `check_ids()` batch existence, `mark_recall_used()` for
  explicit recall credit, and `MemoryStore.stats()`.
- `MEMORY_MAX_SCAN_ROWS`, `MEMORY_TIER_EVAL_BATCH`,
  `MEMORY_ACCESS_COUNT_THROTTLE_S`, and `MEMORY_MIN_RECALL_SCORE` env vars.

### Changed
- MMR diversity reduced from O(n³) to O(n²) via a lookup map.
- Tier evaluation throttled to every `MEMORY_TIER_EVAL_FREQUENCY` retrievals.
- Reranker initialisation is lazy; key validation deferred to first use.

---

## [0.5.0] — 2026-05-03

### Added
- **Smart Extractor** — an LLM-driven six-category extraction pipeline
  (profile / preferences / entities / events / cases / patterns) with per-line
  metadata and vector-pre-filter dedup decisions (create / merge / skip /
  supersede / support / contextualize / contradict).
- `create_llm_client_from_env()` — auto-detects the extraction LLM from
  `MEMORY_EXTRACTION_*`, then `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`; returns
  `None` (raw-turn fallback) when nothing is configured.
- A custom `LlmClient` protocol so any `complete_json` implementation can drive
  extraction.

---

## [0.4.0] — 2026-05-01

### Added
- **Reflection layer** (`hermes_memory_lancedb_pro.reflection`) — stores
  structured LLM-produced reflection summaries (invariants / derived / mapped
  rows) and replays them on recall. Logistic decay (invariants ~45 days,
  derived ~7), a per-agent ownership guard, resolved-item suppression, and a
  prompt-injection guard on every reflection write.

---

## [0.3.0] — 2026-04-29

### Added
- **Temporal classifier** — distinguishes static facts from time-sensitive
  phrases and infers expiry timestamps; dynamic entries decay faster.
- **Session compressor** — scores turns by signal value before extraction
  (decisions / corrections / tool calls high, greetings low).
- **Batch dedup** — a cosine-based pre-LLM near-duplicate filter.
- **Admission control** — an AMAC-v1 scoring gate (utility / confidence /
  novelty / recency / type-prior) with `balanced` / `conservative` /
  `high-recall` presets.
- **Memory compactor** — clusters and merges near-duplicate old memories into
  consolidated entries with cooldown tracking.

---

## [0.2.0] — 2026-04-27

### Added
- **Session-scoped recall** — a `session_id` parameter on
  `MemoryRetriever.retrieve()` / `MemoryStore.search()` restricts results to the
  current session; `cross_session=True` and `tier="core"` memories always
  surface.
- **Access-count throttle** — `increment_access_count` is throttled to once per
  `MEMORY_ACCESS_COUNT_THROTTLE_S` seconds, breaking the recall-frequency
  feedback loop that produces "sticky" memories.
- `MEMORY_CROSS_SESSION_PROMOTION_K` — auto-promotes a memory to
  `cross_session=True` once it is recalled across that many distinct sessions.

---

## [0.1.1] — 2026-04-25

### Fixed
- Security hardening — SQL-injection guards on all filter paths; archived-row
  filtering moved to the application layer to avoid `LIKE` escape issues.
- Performance — RRF score normalisation corrected; noise-filter threshold tuned.
- Added the initial test suite covering store CRUD, hybrid search, decay, tier
  evaluation, and the retriever pipeline.

---

## [0.1.0] — 2026-04-23

### Added
- Initial release — a LanceDB-backed persistent memory store for hermes-agent.
- Hybrid BM25 + cosine vector search with RRF fusion (k=60).
- Weibull stretched-exponential decay with per-tier `β`
  (core 0.8 / working 1.0 / peripheral 1.3).
- Core / working / peripheral tier management with promotion and demotion.
- The supersede pattern — updates archive the old row and write a new one,
  preserving a full audit trail with no vector drift.
- Auto-recovery — detects corrupted databases and re-seeds from `MEMORY.md`.
- A per-path `MemoryStore` singleton and the full `MemoryRetriever` pipeline.
- The `LanceDBProMemoryProvider` hermes-agent adapter and the
  `hermes-memory-smoke` end-to-end smoke test.

---

[0.14.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.12.3...v0.13.0
[0.12.3]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.12.2...v0.12.3
[0.12.2]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.10.1...v0.11.0
[0.10.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/releases/tag/v0.1.0
