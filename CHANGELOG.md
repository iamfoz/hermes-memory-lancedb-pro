# Changelog

All notable changes to hermes-memory-lancedb-pro are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.11.23] — 2026-05-21

### Fixed
- **Task protocol never reached the model** (root cause of v0.11.20–0.11.22
  ineffectiveness) — `_do_recall` bailed out with `return ""` whenever the
  query was empty.  `before_prompt_build` is called to assemble the
  *query-independent system prompt*, so it always passes an empty query;
  the early return discarded the protocol text before it was ever prepended.
  This is why injecting the full `SKILL.md` content in v0.11.22 had zero
  observable effect — it was dropped every turn.  The empty-query branch now
  returns `_TASK_PROTOCOL_TEXT`, so the protocol is injected into the system
  prompt regardless of whether there is anything to recall.

---

## [0.11.22] — 2026-05-21

### Fixed
- **`_TASK_PROTOCOL_TEXT` was too compact to be effective** — the 14-line
  summary injected in v0.11.20 was not sufficient to prevent Hello-loop
  failures (sessions still collapsed within 6 messages without the skill).
  Replaced with the full `SKILL.md` content: explicit trigger conditions,
  step-by-step commands with exact syntax, recovery-after-reset instructions,
  and the invariants list.  Empirically, the full text is what drives reliable
  multi-step behaviour; the compact version did not.

---

## [0.11.21] — 2026-05-21

### Fixed
- **`update()` writes to the wrong row after supersede** (phase 15) — when
  `get_by_id()` follows the supersede chain from an old (archived) ID to the
  current live row, the subsequent `table.update(where="id = old_id")` was
  still targeting the archived row instead of the live one.  Changed to use
  `existing["id"]` (the chain-resolved ID) in the WHERE clause and return
  value so `update(old_id, category="fact")` correctly updates the current
  version.
- **Stale table view after CLI writes** (phases 7, 17, 18, 25) — after a
  subprocess CLI command (`import`, `reset`, etc.) writes to the LanceDB path,
  the in-process `LanceTable` handle could remain pinned to an older dataset
  version, returning 0 results for newly imported rows.  Added
  `_checkout_latest()` which calls `table.checkout_latest()` (LanceDB ≥ 0.20)
  or falls back to re-opening the table.  Called automatically at the start of
  `_vector_search`, `_bm25_search`, and `list_memories`.
- **`CompactionConfig` rejects `min_age_hours`** (phase 20) — stress tests pass
  `min_age_hours=0` to force immediate compaction, but the dataclass only had
  `min_age_days`.  Added `min_age_hours: int | None = None`; when set it takes
  precedence over `min_age_days` (0 = compact everything regardless of age).

---

## [0.11.20] — 2026-05-21

### Changed
- **Durable task protocol now integral to the plugin** — the compact task
  protocol (create → pin → resume → advance → complete + context-reset recovery
  invariants) is injected into every prompt via `before_prompt_build`,
  unconditionally, without requiring the `/durable-task` skill to be invoked.
  This change comes directly from observational data: without the protocol text
  present, sessions collapse to a "Hello" loop within ~12 steps; with it always
  visible, sessions sustain 90+ steps / 171 messages reliably.
- Replaced the reactive single-line task nudge (v0.11.19) with the full
  `_TASK_PROTOCOL_TEXT` block — always prepended ahead of the reflection and
  recall sections, even on turns with no recall results.
- New env var `MEMORY_TASK_PROTOCOL` (default `on`) controls the injection;
  set to `off` to suppress (e.g. automated pipelines that manage the ledger
  externally). `MEMORY_TASK_NUDGE` is retired.

---

## [0.11.19] — 2026-05-20

### Fixed
- **`task pin` crash** — `metadata_extra` was serialised as a JSON string before
  being passed to `store.store()`, which calls `meta.update(extra)` internally
  and requires a `dict`.  Now passes a plain `dict` directly, eliminating the
  `ValueError: dictionary update sequence element #0 has length 1` crash
  reported during stress testing.

### Added
- **Task nudge** — when recall returns results but no `active_task` memory is
  present, a one-line reminder is appended to the recall block pointing the
  model to `hermes-memory-lancedb-pro task create` + `task pin` and `/durable-task`.
  Controlled by `MEMORY_TASK_NUDGE` env var (default `on`); set to `off` to
  silence it in automated pipelines that manage the ledger externally.
- **`/durable-task` invocation** added to `skills/durable-task/AGENTS.md` with
  mandatory language (`MUST`) for multi-step tasks.

---

## [0.11.18] — 2026-05-20

### Added
- **`task advance` CLI subcommand** — records a completed iteration, increments
  `current_iteration`, appends to `results.jsonl`, and updates `next_action` in
  `state.json`.  Because the memory plugin reloads `state.json` on every recall,
  the model sees the updated state on the very next turn with no re-pin required.
  Flags: `--result pass|fail`, `--next-action <text>`, `--summary <text>`.
- **`skills/durable-task/SKILL.md`** — installable Hermes skill that teaches the
  agent the durable task protocol: create ledger → pin → resume before each step
  → advance after each step → complete.  Also covers post-compaction recovery
  (`task list` → `task resume` → continue from `next_action`).

---

## [0.11.17] — 2026-05-20

### Fixed
- **Post-compaction task recovery** — added `_refresh_active_task_memories` which
  runs after guardrails on every `_do_recall` call.  For any `active_task` memory
  that carries a `state_path` in its metadata (written by `task pin`), the control
  block text is re-read from `state.json` on disk rather than serving the snapshot
  from when `pin` was run.  This means:
  - The injected iteration counter and `next_action` are always current.
  - After context compaction the model still receives the correct task state in
    `before_prompt_build`, because the task ledger lives on disk — not in the
    context window.
- **Recall injection debug log** — every recall now logs at DEBUG level the count
  and category list of items being injected (e.g. `active_task, fact, preference`).
  Enable with `MEMORY_LOG_LEVEL=DEBUG` or equivalent to observe what the model
  sees on each turn.

### Added
- `import json` at module level in `provider.py` (was used only via `_parse_metadata`
  previously; now required directly by `_refresh_active_task_memories`).

---

## [0.11.16] — 2026-05-20

### Fixed
- **Greeting-replay bug** — session anchors (`first_for_session` + `recent_for_session`)
  were injected into every turn without any noise filter.  If the first stored memory
  in a session was a greeting ("Hello", "Hi there", "👋 How can I help?"), it was
  anchored into the system-prompt recall block on the very next turn, causing the
  model to echo it.  The anchor loop now applies two guards before appending a
  candidate:
  1. **Minimum length** — texts shorter than 20 chars are skipped (catches bare
     "Hello", "OK", "Sure").
  2. **`is_noise()` filter** — texts matching `BOILERPLATE_PATTERNS` (which already
     covers `^(hi|hello|hey|good morning|greetings|…)`) are skipped.
  Both guards were already applied to the main relevance results via the retriever's
  noise pre-filter; the anchor fast-path had no equivalent check.

---

## [0.11.15] — 2026-05-20

### Added
- **`task_ledger` module** — durable task-state management outside the LLM context
  window. Task state lives in `~/.hermes/workspace/tasks/<task_id>/` with
  `state.json` (objective, status, iteration counter, next_action, blockers,
  invariants), `results.jsonl`, `events.jsonl`, and `log.md`. Key functions:
  `create_task`, `load_state`, `save_state`, `advance_iteration`, `complete_task`,
  `append_jsonl`, `build_control_block`, `looks_like_reset`, `list_tasks`.
- **`looks_like_reset(response, active_task)`** — detects model greeting/reset
  responses so the runner can reject them, log a `reset_detected` event, and
  retry from `state.json` rather than silently losing iteration progress.
- **`build_control_block(state)`** — formats the `ACTIVE TASK CONTROL BLOCK`
  string (task ID, status, objective, current/target iteration, next_action,
  blockers, invariants) for prepending to every iteration prompt. Context
  compaction cannot lose the task objective when the control block is always
  re-injected from `state.json` each turn.
- **`hermes-memory-lancedb-pro task <subcommand>`** — standalone CLI task group:
  `create`, `list`, `show`, `resume` (prints control block + pass/fail summary),
  `complete`, `pin` (stores control block as an `active_task` memory).
- **`hermes lancedb_pro task <subcommand>`** — same commands in the plugin CLI
  namespace via `register_cli`.
- **Recall guardrails in `before_prompt_build` / `prefetch`**:
  - `MEMORY_NEVER_CATEGORIES` (default: `greeting,ephemeral_chat`) — categories
    never injected regardless of score, preventing old greetings from surfacing.
  - `MEMORY_RECALL_CHAR_BUDGET` (default: `4800` ≈ 1200 tokens) — caps the total
    size of the injected recall block so memory cannot crowd out the active task
    state or cause early compaction.
  - `MEMORY_ACTIVE_TASK_PIN` (default: `on`) — memories with
    `category="active_task"` are always prepended to the recall block, bypass
    never-categories filtering and the char budget. Use `task pin` to store the
    current control block as a pinned active-task memory.

### Tests
- 53 new tests in `tests/test_task_ledger.py` covering all public functions of
  `task_ledger` including atomic write, iteration sequencing, reset detection,
  and boundary cases.

---

## [0.11.14] — 2026-05-20

### Added
- **`init` command in `register_cli`** — `hermes lancedb_pro init` was missing from
  the plugin CLI namespace despite being available in the standalone CLI. Added
  `p_init` subparser block and `"init": _cmd_init` to `_dispatch_plugin_cli`.
- **Confirmation gate for `init` and `reset`** — both commands now prompt
  `Type "yes" to proceed:` before making any changes to the database. Pass `-y` /
  `--yes` to skip the prompt for scripted or automated use. The `--yes` flag is
  added to the parsers in both `register_cli` (plugin CLI) and `main()` (standalone
  CLI).

### Tests
- `test_register_cli_uses_own_subparsers_group` extended to include `"init"` in
  the commands-parse check.

---

## [0.11.13] — 2026-05-20

### Fixed
- **`register_cli` rewritten to match hermes memory plugin spec exactly** — the
  0.11.12 implementation incorrectly tried to inject commands into the *existing*
  `hermes memory` subparsers group by scanning `parser._actions`. The spec says
  hermes-agent passes a **fresh** ArgumentParser for the provider's own namespace
  (`hermes lancedb_pro`) and `register_cli` should call `add_subparsers()` on it
  directly. Commands now appear at `hermes lancedb_pro doctor|export|import|reset`.
  The top-level `subparser.set_defaults(func=_dispatch_plugin_cli)` pattern from
  the spec is restored.
- **`reset` renamed back from `lancedb-reset`** — "lancedb-reset" was needed to
  avoid collision with `hermes memory reset`. In the provider's own namespace
  (`hermes lancedb_pro`), there is no such collision, so the simpler name is correct.
- **`PLUGIN_CLI_CONTENT` shim** comment corrected to `hermes lancedb_pro`.

### Tests
- `TestRegisterCli` refreshed: removed the two tests that verified the incorrect
  injection behaviour; replaced with `test_register_cli_uses_own_subparsers_group`
  (commands parse under `lancedb_pro_command` dest), `test_register_cli_top_level_func_for_dispatch`
  (spec pattern: func on parent parser), and `test_register_cli_reset_is_reset_not_lancedb_reset`.

---

## [0.11.12] — 2026-05-20

### Fixed
- **`register_cli` now injects into the existing subparsers group** — the previous
  implementation called `parser.add_subparsers()` on the parser hermes-agent passes,
  creating a *second* nested subparsers group. Argparse only dispatches through the
  first group, so `hermes memory doctor` resolved as an invalid choice instead of
  routing to the plugin. Fixed: `register_cli` now iterates `parser._actions` to find
  the pre-existing `_SubParsersAction` (the one that already holds setup/status/off/reset)
  and calls `add_parser()` on it directly, so our commands appear at the same level.
- **`lancedb-reset` instead of `reset`** — the plugin was registering a `reset` command
  that collided with hermes-agent's built-in `hermes memory reset`. Renamed to
  `lancedb-reset` to avoid the conflict.
- **Each subparser sets its own `func` default** — commands now set
  `p_xxx.set_defaults(func=_cmd_xxx)` so hermes-agent's standard `args.func(args)`
  dispatch pattern routes directly without going through an intermediate
  `_dispatch_plugin_cli` wrapper.
- **`PLUGIN_CLI_CONTENT` shim comment** corrected from `hermes lancedb-pro` to
  `hermes memory`.

### Tests
- `TestRegisterCli` extended with 3 new cases: commands extend an existing subparsers
  group (the real hermes-agent scenario), every command has a callable `args.func`
  default, and `lancedb-reset` is registered without overwriting the built-in `reset`.

---

## [0.11.11] — 2026-05-20

### Fixed
- **Plugin path corrected everywhere** — `memory_init.sh` and its embedded Python
  fallback both used the stale pre-0.11.1 path `~/.hermes/plugins/lancedb_pro`.
  Corrected to `~/.hermes/hermes-agent/plugins/memory/lancedb_pro` matching what
  `install-plugin` actually creates. `register()` docstring in `provider.py` updated.
- **`compute_decay_score` falls back to top-level `timestamp`** — entries written
  without `metadata.created_at` (e.g. by external tooling) silently defaulted to
  `now_ms` for recency, making them look brand-new regardless of age. Fixed: when
  `metadata.created_at` is absent, the top-level LanceDB `timestamp` column is used
  instead. `metadata.created_at` still takes priority when present.
- **Temporal classifier recognises `current` (adjective)** — "Current stock price",
  "my current address" were classified as static because the pattern only matched
  `\bcurrently\b`. Broadened to `\bcurrent(?:ly)?\b`.
- **Greeting detector handles "Hi there" / "Hello there"** — the GREETING_PATTERNS
  regex required the string to end immediately after the greeting word. "Hi there"
  fell through to `short_statement` (0.4) instead of `greeting` (0.1). Added
  optional `(\s+there)?` group.

### Added
- **`init`, `reset`, `doctor`, `export`, `import` subcommands in the standalone CLI** —
  `hermes-memory-lancedb-pro` now exposes all admin commands directly:
  - `init` — open/create the memory store and optionally seed from MEMORY.md
  - `reset` — wipe the DB directory and re-run init
  - `doctor` — diagnostic report (previously only via `hermes lancedb-pro doctor`)
  - `export` — JSONL export
  - `import` — JSONL import
  The shell scripts `memory_init.sh` / `memory_reset.sh` remain for environments
  without the package installed, but the Python implementations are now canonical.

### Tests
- `TestCreatedAtFallback` (2 tests): ancient top-level timestamp produces low recency;
  `metadata.created_at` takes priority over top-level timestamp.
- `TestCurrentPattern` (3 tests): "current" adjective, "currently", compound phrase.
- `TestGreetingHiThere` (3 tests): "Hi there", "Hello there!", "Hi" alone.

---

## [0.11.10] — 2026-05-20

### Added
- **`freshness_trend` in recall block** — `_format_recall` now appends a `[forming]` /
  `[strengthening]` / `[weakening]` tag to each recalled memory when the evidence-weighted
  trend is not "stable".  The agent sees at a glance which memories are contested or newly
  forming and can weight them accordingly.
- **`entities` typed field in `SmartMemoryMetadata`** — LLM-extracted entity names are now
  a first-class field (`entities: list[str]`) rather than landing in the opaque `extras`
  blob.  `parse_smart_metadata` validates and strips non-strings; `stringify_smart_metadata`
  emits the field when non-empty; `build_smart_metadata` unions base + patch entities so
  previously tagged names survive subsequent updates.
- **Entity-overlap boost in retriever** — `MemoryRetriever.retrieve` now calls
  `_apply_entity_boost` (step 2.5) immediately after vector-dominant fusion.  Any fused
  result whose stored entity list contains a term that appears verbatim in the query
  receives a multiplicative score boost (×1.2 per match, capped at ×1.6 for 3+ matches)
  before the scoring pipeline applies decay weights.  Solves the case where an entity name
  lives only in metadata rather than in the memory text.

### Tests
- `TestEntitiesTypedField` (9 tests): parse, extras isolation, filter non-strings, stringify
  emit/omit, build union merge, base-preserve, roundtrip.
- `TestFormatRecallFreshnessTrend` (5 tests): weakening/forming/strengthening present,
  stable omitted, no-decay no tag.
- `TestEntityOverlapBoost` (8 tests): single match boosts, no-match unchanged, missing
  entities key unchanged, multi > single, 3-match cap, case-insensitive, `_entity_matches`
  field set, empty list.

---

## [0.11.9] — 2026-05-20

### Added / Changed — prompted by Hindsight (vectorize.io) best-practices review

**Extraction context field** (`extraction_prompts.py`, `smart_extractor.py`):
`build_extraction_prompt` now accepts a `context` parameter — a short
description of the content source (e.g. "Hermes agent turn, session=s-123,
scope=agent"). Injected as `## Content Context` into the extraction prompt.
Hindsight research identifies this as the single highest-impact extraction
quality lever. `sync_turn` constructs and passes this context automatically.

**Structured conversation JSON** (`smart_extractor._format_conversation`):
Turn content is now passed to the extraction LLM as a JSON conversation array
with explicit `role` and `timestamp` fields rather than a flat string.
Preserves entity relationships, causal context, and temporal markers that the
flat format loses (e.g. "moved away from Redux *last quarter*" stays bound to
the migration fact rather than fragmenting).

**Named entity extraction** (`extraction_prompts.py`, `smart_extractor.py`):
The extraction prompt now requests an `entities` list (proper nouns: people,
projects, tools, organisations) alongside each memory. Parsed and stored in
`metadata.entities` for future graph-traversal retrieval. A narrative-unit
rule is also added: "keep causally interdependent facts as a single memory
rather than splitting them".

**Evidence-weighted confidence + freshness trend** (`decay.py`):
`compute_decay_score` now reads `metadata.support_info.global_strength`
(ratio of confirmations to total observations) and blends it into the
effective confidence used for the intrinsic score component. Requires ≥ 3
observations to avoid penalising newly-created memories. Formula:
`confidence × (0.4 + 0.6 × global_strength)` — a fully contradicted memory
(strength=0) is scored at 40% of its write-time confidence; a fully confirmed
one (strength=1) is unchanged. The score dict now also includes
`freshness_trend` ("forming" / "strengthening" / "stable" / "weakening").
Fixed a latent bug: `0.0 or 0.5` was silently coercing a zero global_strength
to 0.5, defeating the penalty.

**Temporal query intent post-filter** (`provider.py`):
`_do_recall` now detects temporal language in the query ("yesterday",
"last week", "this morning", "recently", named months, …) via
`_parse_temporal_intent` and post-filters relevance results to memories whose
`timestamp` falls in the corresponding window. Session anchors (task-framing)
bypass the filter so they are always present. When the filter would produce an
empty result set it falls back to the unfiltered list.

---

## [0.11.8] — 2026-05-20

### Fixed
- Task-framing memories from the start of a session are now always injected
  into recall via **first+recent session anchors**.  Previously only the 2
  most-recently-written memories were anchored; after 3 subsequent turns the
  task framing (e.g. "stress test my memory") was no longer in the anchor
  window.  Relevance search alone cannot recover it when the current query is
  semantically distant, so the agent appeared to "forget" its goal at turn 4.
  `_do_recall` now combines the 2 oldest *and* 2 newest session memories as
  anchors (deduplicated against each other and the relevance results).
- Added `MemoryStore.first_for_session()` — same shape as `recent_for_session`
  but returns oldest-first, giving callers direct access to session-start
  entries.
- `MemoryStore._load_embedder()` is now guarded by a threading lock (double-
  checked locking pattern) to prevent two concurrent threads (warmup + first
  sync_turn write) from each loading the embedding model independently.
- `_flush_pending_write` default timeout increased from 1 s to 2 s to give
  the previous turn's write a wider window to complete before recall runs.

---

## [0.11.7] — 2026-05-20

### Fixed
- `on_memory_write` now implements `edit` and `delete` actions and supports
  `replace_all=True` bulk mutation.  Previously both actions were stubs that
  logged a warning and returned, leaving the LanceDB store out of sync with
  hermes-agent's built-in memory after any `/memory edit` or `/memory delete`
  command.
- `edit`: BM25-searches for memories whose text contains the `target` string,
  then supersedes each match with `content` (the new text).  Without
  `replace_all=True` only the single best match is updated; with it every
  matching entry is superseded.
- `delete`: same BM25 lookup, then soft-archives each match (marks
  `state=archived`) so the rows are hidden from future recalls but preserved
  in the audit trail.
- `replace_all` key is no longer forwarded into the stored metadata on `add`
  writes (it was a no-op scalar that polluted the metadata blob).

---

## [0.11.6] — 2026-05-20

### Fixed
- Cold-start write/read race: `prefetch` and `before_prompt_build` now call
  `_flush_pending_write()` before querying, joining the previous `sync_turn`
  background thread (1 s timeout). On a brand-new install the embedding model
  takes 10–30 s to load on first use; without the flush the recall for the
  first several turns returned empty because the turn-N write hadn't finished
  before the turn-N+1 read. Task framing stored in turn 1 is now visible from
  turn 2 onwards.
- Recency anchors in `_do_recall`: the 2 most-recently-written session memories
  are appended to relevance results (deduplicated). Relevance-only retrieval
  misses earlier task framing when the current query is semantically distant
  (e.g. "check slot 7" scores low against "stress test my memory"), causing the
  agent to lose its goal after context compression.
- `_raw_sync_turn` no longer stores raw assistant responses. They are verbose
  agent-side text that created a feedback loop: an early greeting stored in the
  fallback path would survive the retrieval-time noise filter, get re-injected
  after context compression, and cause the agent to re-greet. Only user-side
  content is now stored, and only after passing the `is_noise` check.

---

## [0.11.5] — 2026-05-20

### Removed
- `hermes-memory` console script alias removed. It looked like a generic
  system-level command and shadowed unrelated tools. Use
  `hermes-memory-lancedb-pro` instead. If you have a stale system install
  run `pip uninstall hermes-memory-lancedb-pro` and reinstall via hermes-pip.

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
- Development branch renamed from `claude/restructure-repo-branches-BiSQH`
  to `fix/0.11.5` to remove tooling-generated names from the public
  branch list.

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
