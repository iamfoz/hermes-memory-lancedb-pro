# Changelog

All notable changes to hermes-memory-lancedb-pro are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.11.5] — 2026-05-20

### Fixed
- `plugin.yaml` now declares `kind: memory` so `PluginManager` routes the
  plugin to the memory manager instead of the generic standalone loader.
  Without this field the loader called `ctx.register()` on a context that
  lacked `register_memory_provider()`, crashing with an `AttributeError`.
- `_resolve_hermes_home()` default corrected from `~/.hermes` to
  `~/.hermes/hermes-agent/` to match the actual hermes-agent installation
  layout; `--hermes-home` help strings updated accordingly.
- `MemoryStore.get_by_id()` now follows the `superseded_by` chain when the
  requested row is archived, returning the current live version instead of
  the stale archived one. Callers that explicitly want the archived row can
  pass `follow_chain=False`. A depth limit of 32 guards against malformed
  chains.

---

## [0.11.4] — 2026-05-20

### Fixed
- `hermes-memory` console script re-added as an alias for
  `hermes-memory-lancedb-pro` so both names work after install; the old
  binary is no longer orphaned when upgrading from 0.11.2.
- `install-plugin` auto-migrates the old `plugins/lancedb_pro/` directory
  to the correct `plugins/memory/lancedb_pro/` on first run, so users
  upgrading from pre-0.11.1 don't need to manually move the directory.
- `uninstall-plugin` also removes the old `plugins/lancedb_pro/` directory
  when found, as part of the same migration cleanup.
- Stale remote branches (`claude/restructure-repo-branches-BiSQH`,
  `feat/spec-compliance`) that contained old commit messages with AI
  session URLs were overwritten to point to clean main history.

---

## [0.11.3] — 2026-05-20

### Fixed
- `get_config_schema()` trimmed from 11 fields to 3 (the LLM extraction
  key, base URL, and model). Per spec guidance, only fields the user must
  actively configure belong in the setup wizard; the remaining 8 tuning
  knobs (`MEMORY_PREFETCH_LIMIT`, `MEMORY_ADMISSION_PRESET`, etc.) are
  documented in the README and set via environment variables directly.
- `handle_tool_call(name, args)` stub added — the ABC may define it
  abstractly even for providers with no tools; the no-op prevents potential
  instantiation failure on strict ABC implementations.
- Module docstring corrected: plugin path was still showing the old
  `~/.hermes/plugins/lancedb_pro/`; now shows the correct
  `~/.hermes/plugins/memory/lancedb_pro/`.

---

## [0.11.2] — 2026-05-20

### Changed
- Bootstrap CLI renamed from `hermes-memory` to `hermes-memory-lancedb-pro` to
  make it clear the command is provider-specific, not a generic Hermes memory CLI.
- `export`, `import`, and `doctor` subcommands moved out of `hermes-memory-lancedb-pro`
  and into the Hermes plugin CLI slot as `hermes lancedb-pro export|import|doctor`,
  following the plugin CLI spec (`register_cli(subparser)` in `cli.py`).

### Added
- `register_cli(subparser)` in `_cli.py` — hermes-agent calls this to wire
  `hermes lancedb-pro` subcommands into the main CLI once the plugin is active.
- `PLUGIN_CLI_CONTENT` shim (`cli.py`) written to the plugin directory by
  `install-plugin` so hermes-agent can discover and load `register_cli`.

### Fixed
- `install-plugin` and `uninstall-plugin` now manage `cli.py` alongside
  `__init__.py` and `plugin.yaml`.

---

## [0.11.1] — 2026-05-20

### Fixed
- Plugin directory corrected from `plugins/lancedb_pro/` to
  `plugins/memory/lancedb_pro/` per the Hermes memory plugin spec; the old path
  caused `hermes memory setup` to not discover the provider.
- `plugin.yaml` version field was stale at `0.9.4`; now tracks the package version.

---

## [0.11.0] — 2026-05-20

### Added
- **Background embedding-model warmup** — `LanceDBProMemoryProvider.initialize()` warms the embedding model in a daemon thread, so the 10-30 s cold-start cost lands during session boot instead of the user's first turn. No caller action required (previously the README asked users to call `warmup()` themselves).
- **Automatic memory compaction** — the provider runs cooldown-gated compaction at session end, clustering and merging near-duplicate old memories. `MEMORY_AUTO_COMPACT_COOLDOWN_HOURS` (default 168 = weekly; 0 disables). Runs per-scope so a merge never spans scopes. Previously `run_compaction()` was never invoked by the agent integration.
- **Admission control enabled by default** — the smart extractor is now built with an AMAC-v1 `AdmissionController`. `MEMORY_ADMISSION_PRESET` selects `balanced` (default) / `conservative` / `high-recall` / `off`. Previously the extractor always ran without an admission gate.
- **Reflection layer wired into the agent lifecycle** — at session end the provider generates a structured reflection (durable *invariants* + short-lived *derived* insights) via the extractor's LLM and persists it; on recall, ranked reflection slices are prepended to the memory-context block. `MEMORY_REFLECTION` (default on), `MEMORY_REFLECTION_SCAN_LIMIT`, `MEMORY_REFLECTION_AGENT_ID`. Previously the reflection subsystem was completely unwired — nothing wrote or read reflections.
- `build_reflection_prompt()` in `extraction_prompts`; `SmartExtractor.llm` property exposing the configured client.
- `initialize()` accepts an `agent_id` kwarg for multi-agent reflection ownership.

### Fixed
- Install instructions corrected to `hermes-pip` — a plain `pip install` lands the package outside Hermes' Python environment, so the discovery shim's `import` fails silently and the plugin never loads.
- `MemoryStoreReflectionAdapter` was missing from `hermes_memory_lancedb_pro.reflection`'s exports despite the README documenting it as importable — added to the package `__all__`.

---

## [0.10.0] — 2026-05-19

### Added
- `hermes-memory install-plugin` / `uninstall-plugin` CLI commands create and remove the `~/.hermes/plugins/lancedb_pro/` discovery shim automatically; upgrades now only require `pip install -U`.
- Session-summary memories: at session end the provider condenses the conversation into a structured memory entry, superseding any prior summary for the same session key.
- Entry-point registration (`hermes.plugins` group) so newer hermes-agent builds can discover the provider without the shim.

### Fixed — spec-compliance audit (28 bugs, four independent review sweeps)
- **Weibull decay formula** — lambda was computed before beta, giving the wrong half-life shape for core/peripheral tiers. Corrected to `λ = ln(2)^(1/β) / half_life` so `recency = 0.5` at exactly `half_life` for every tier.
- **Thread safety** — race between `sync_turn` and `on_session_end` eliminated; join+create+start of the background sync thread is now serialised through a dedicated `_dispatch_lock`.
- **Session compressor tool classification** — tool-result texts were mis-classified as tool-call texts (shared indicator set). Split into `TOOL_CALL_INDICATORS` / `TOOL_RESULT_INDICATORS`; `score_text` now returns `reason="tool_result"` for result blocks.
- **Memory compactor provenance** — merged cluster entries silently dropped `tier` and `cross_session`. Now preserves the highest-rank tier (core > working > peripheral) and OR-combines `cross_session` flags.
- **Reflection layer metadata** — `parse_reflection_metadata` only accepted strings; MemoryStore rows carry parsed dicts, so every lookup silently returned `{}`. Now accepts `str | dict | None`.
- **Support tracking reset** — `parse_support_info` discarded a `SupportInfoV2` instance passed back from a prior parse, resetting history to `None` on each call. Added early return for the instance case.
- **Admission control double-encoding** — `_handle_merge` was `json.dumps()`-wrapping an already-serialisable audit dict, producing a JSON string inside a JSON object. Removed the redundant encoding.
- **Min-score gate** — the retriever applied the gate even when `min_score=0.0`, discarding results with a score of exactly zero. Gate now only activates when `min_score > 0`.
- **Tier SQL filter** — LIKE pattern only matched `"tier": "core"` (with space), missing the compact form `"tier":"core"`. Now ORs both patterns.
- **Cross-session promotion** — guard condition was inverted; entries were promoted on every access rather than only when the `MEMORY_CROSS_SESSION_PROMOTION_K` threshold was crossed.
- **normalize_category warning flood** — logged a warning for every LLM candidate in the hot extraction path. Added `warn: bool = True` parameter; extractor calls it with `warn=False`.
- **Content-block message handling** — `_extract_message_texts` didn't handle `content: list[{type, text}]` returned by multi-modal models; now joins text parts.
- **reflection/mapped_metadata field names** — loader checked `.mapped_kind` but the dataclass field is `.kind`; mappings were silently skipped.
- **reflection/store dedup distance** — `_distance` was assumed present; missing field caused a `TypeError`. Now defaults to `1.0` (maximum distance) when absent.
- **admission_control span splitting** — `_split_support_spans` used `pass` instead of `continue`, allowing empty/duplicate lines to accumulate.
- Various None-guard, type-coercion, and import fixes (`slices.py` Callable import, `save_config` path handling, schema column gaps).

---

## [0.9.4] — 2026-05-07

### Added
- Auto-purge of archived rows at session end (`MEMORY_AUTO_PURGE_COOLDOWN_HOURS`, `MEMORY_PURGE_GRACE_DAYS`). Keeps the database lean without manual intervention.
- `hermes-memory doctor` CLI command reports database health and recommends purge / compaction.

### Fixed
- Replaced x-goog-api-key header with OAuth2 Application Default Credentials for the Google Discovery Engine Ranking API (API keys are rejected by that service).

---

## [0.9.3] — 2026-05-07

### Fixed
- Google reranker API key was read at import time; environment variables set after import were ignored. Deferred to `__init__`.
- Auto-selection logic warns clearly when both `LANGSEARCH_API_KEY` and `GOOGLE_CLOUD_PROJECT` are set and disables reranking until the ambiguity is resolved via `MEMORY_RERANKER`.

---

## [0.9.0] — 2026-05-06

### Added
- **Google Discovery Engine reranker** — uses the Ranking API (`semantic-ranker-512@latest`). Authenticate via Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login`). Free tier: 1,000 queries/month.
- `MEMORY_RERANKER` env var (`auto` / `langsearch` / `google` / `disabled`) to explicitly select the reranker backend.
- `MEMORY_GOOGLE_RANKING_MODEL` override for the Google ranking model.

---

## [0.7.2] — 2026-05-05

### Fixed
- `score_text` signal inversions: tool-call texts were scored below neutral; corrections were scored as acknowledgements. Inverted comparisons corrected.

---

## [0.7.1] — 2026-05-05

### Fixed
- Five post-deploy issues: stale singleton after `forget`, incorrect archived-row count in `stats`, BM25 index not rebuilt after bulk inserts, tier eval skipping rows near the batch boundary, and `purge_archived` not respecting `grace_period_days` on first run.

---

## [0.7.0] — 2026-05-04

### Added
- **New MemoryProvider hooks** (feature-detected, falls back gracefully on older hermes-agent):
  - `before_prompt_build` — injects recalled memories into the **system prompt** for stronger authority.
  - `on_recall_used` — credits memories only when the LLM actually consumed them, closing the recall-frequency feedback loop.
  - `on_tool_call_observed` — propagates tool calls to the extractor for richer session context.
- `MEMORY_INJECTION_GUARD` mode (`off` / `warn` / `reject` / `sanitize`) guards against prompt-injection attacks at write time.

---

## [0.6.0] — 2026-05-02

### Added
- Ops and reliability improvements: `hermes-memory export` / `import --reembed`, `warmup()` for eager model load at boot, `check_ids()` batch existence, `mark_recall_used()` for explicit recall credit.
- `MEMORY_MAX_SCAN_ROWS`, `MEMORY_TIER_EVAL_BATCH`, `MEMORY_ACCESS_COUNT_THROTTLE_S`, `MEMORY_MIN_RECALL_SCORE` env vars.
- `MemoryStore.stats()` returns tier breakdown, archived count, and compaction recommendation.

### Changed
- MMR diversity O(n³) → O(n²) via Map lookup.
- Throttled tier evaluation (every `MEMORY_TIER_EVAL_FREQUENCY` retrievals, default 10) to keep search latency low.
- Lazy reranker initialisation; key validation deferred to first use.

---

## [0.5.0] — 2026-04-29

### Added
- **Smart Extractor** — LLM-driven 6-category extraction pipeline (profile / preferences / entities / events / cases / patterns) with per-line metadata and vector-pre-filter dedup decisions (create / merge / skip / supersede / support / contextualize / contradict).
- `create_llm_client_from_env()` factory checks `MEMORY_EXTRACTION_*` overrides first, then `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. Returns `None` (raw-turn fallback) when nothing is configured.
- `auto_smart_extraction` flag on `LanceDBProMemoryProvider` (default `True`).
- Custom `LlmClient` protocol (`complete_json(prompt, *, label) -> dict | None`) for tests and custom gateways.

---

## [0.4.0] — 2026-04-25

### Added
- **Reflection layer** (`hermes_memory_lancedb_pro.reflection`): stores structured LLM-produced reflection summaries (invariants / derived / mapped rows) and replays them on recall.
  - Logistic decay: invariants midpoint 45 days, derived 7 days.
  - Ownership guard: `derived` items are strictly per-agent.
  - Resolved-item suppression prevents stale advice from resurfacing.
  - `sanitize_injectable_reflection_lines` prompt-injection guard on all reflection writes.

---

## [0.3.0] — 2026-04-20

### Added
- **Temporal classifier** (`classify_temporal`, `infer_expiry`): distinguishes static facts from time-sensitive phrases and infers expiry timestamps. Dynamic entries get accelerated Weibull decay.
- **Session compressor** (`compress_texts`, `estimate_conversation_value`): scores turns by signal value before extraction. Decisions / corrections / tool calls 0.85–1.0; greetings 0.1.
- **Batch dedup** (`batch_dedup`): cosine-based pre-LLM near-duplicate filter within extraction batches.
- **Admission control** (`AdmissionController`, `CandidateMemory`): AMAC-v1 scoring gate (utility / confidence / novelty / recency / type-prior) with `balanced` / `conservative` / `high-recall` presets.
- **Memory compactor** (`run_compaction`, `CompactionConfig`): greedy cosine-cluster expansion on old memories; merges clusters into single consolidated entries with cooldown tracking.

---

## [0.2.0] — 2026-04-14

### Added
- **Session-scoped recall** — `session_id` parameter on `MemoryRetriever.retrieve()` and `MemoryStore.search()` restricts results to the current session; `cross_session=True` and `tier="core"` memories always surface.
- **Access-count throttle** — `increment_access_count` throttled to once per `MEMORY_ACCESS_COUNT_THROTTLE_S` seconds (default 300) to prevent recall-frequency feedback loops producing "sticky" memories.
- `MEMORY_CROSS_SESSION_PROMOTION_K`: auto-promotes a memory to `cross_session=True` once recalled across this many distinct sessions.

---

## [0.1.1] — 2026-04-10

### Fixed
- Security hardening: SQL injection guards on all filter paths; archived-row filtering moved to the application layer to avoid `LIKE` escape issues.
- Performance: RRF score normalisation corrected; noise-filter threshold tuned.
- Added initial test suite covering store CRUD, hybrid search, decay, tier evaluation, and retriever pipeline.

---

## [0.1.0] — 2026-04-07

### Added
- Initial release: LanceDB-backed persistent memory store for hermes-agent.
- Hybrid BM25 + cosine vector search (RRF fusion, k=60).
- Weibull stretched-exponential decay with per-tier β (core 0.8 / working 1.0 / peripheral 1.3).
- Core / working / peripheral tier management with promotion and demotion.
- Supersede pattern: updates archive the old row and create a new one (full audit trail, no vector drift).
- Auto-recovery: detects corrupted databases and re-seeds from `MEMORY.md`.
- Per-path singleton (`MemoryStore.get_instance(db_path=...)`).
- `MemoryRetriever` full pipeline: fusion → length normalisation → hard min → decay scoring → noise filter → rerank → MMR diversity → lifecycle hooks.
- `LanceDBProMemoryProvider` hermes-agent adapter: session tagging, prefetch, `mark_recall_used` credit loop.
- `hermes-memory-smoke` end-to-end smoke test CLI.
- LangSearch cross-encoder reranker (`LANGSEARCH_API_KEY`).

[0.11.4]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.3...v0.11.4
[0.11.3]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.2...v0.11.3
[0.11.2]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.9.4...v0.10.0
[0.9.4]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.9.0...v0.9.3
[0.9.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.7.2...v0.9.0
[0.7.2]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/releases/tag/v0.1.0
