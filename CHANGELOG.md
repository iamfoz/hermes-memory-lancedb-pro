# Changelog

All notable changes to **hermes-memory-lancedb-pro** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is in the `0.y.z` series the public API may change between
minor versions; breaking changes are called out under **Changed** and
**Removed**.

---

## [Unreleased]

---

## [0.17.1] — 2026-11-25

### Fixed
- **CLI advice named commands that are not on the agent's PATH.** `doctor`
  recommendations and recovery hints said `hermes-memory reembed`,
  `hermes-memory purge`, `hermes-memory compact` and
  `hermes-memory-lancedb-pro export`. They now give the in-host form,
  `hermes lancedb_pro <command>`, which works wherever Hermes does.
- The `reembed` prompt said it was "best run" with Hermes stopped; it refuses
  to run otherwise, and now says so.

---

## [0.17.0] — 2026-11-25

### Upgrade notes
- **Back up first:** `hermes lancedb_pro export --include-archived -o backup.jsonl`.
  Nothing in this release deletes or rewrites stored memories.
- **Scores read higher.** Vector scores are now true cosine similarity (a match
  that scored 0.4 now scores about 0.7). Result order is unchanged.
- **`MEMORY_MIN_RECALL_SCORE` is now an absolute floor.** If you set it above
  0 it filters more than before, and a message with no good match recalls
  nothing. Lower it if recall looks thin.
- **`MEMORY_DB_DIR` is now honoured by the agent**, as documented. Check it
  points where your memories are.
- **`MEMORY_ACTIVE_TASK_PIN` and `MEMORY_TASK_PROTOCOL` now take effect** if
  set to `off`.
- **`compact` asks for confirmation.** Add `-y` in scripts.
- **Existing databases are reported as a legacy embedding scheme** and keep
  working unchanged. `reembed` converts them; it is optional, and Hermes must
  be stopped while it runs.
- **Skills ship with the plugin** as `lancedb_pro:<name>`. Remove any copies of
  `durable-task` and `task-to-skill` from `~/.hermes/skills/`.
- **`lancedb` 0.25 or newer** is required.

### Fixed — data loss
- **`hermes-memory-smoke` deleted real memories.** Its final cleanup forgot
  every vector-search hit for the smoke prefix without checking the text;
  vector search has no relevance floor, so each run against a real store
  hard-deleted up to 50 unrelated memories while reporting success. Cleanup
  now removes only rows the test wrote.
- **`init` wiped the database on any error.** Any exception while opening the
  store (file-descriptor exhaustion, permissions, a LanceDB version mismatch)
  was treated as corruption and the directory deleted without asking. An
  unreadable database is now renamed to `<path>.unreadable-<timestamp>` and a
  fresh one created; nothing is deleted.
- **`reset` deleted whatever directory `--path` named.** It now refuses a
  non-empty directory that holds no LanceDB table.
- **Import duplicated or orphaned memories.** Rows were written under fresh
  ids, so importing a backup twice doubled every memory, `--allow-existing`
  skipped nothing, and supersede links pointed at ids that no longer existed.
  Rows keep their exported id, and the duplicate check covers archived rows
  and rows an older version imported under a new id.
- **`update(text="")` destroyed a memory.** The supersede path skipped the
  empty-text and `MEMORY_MAX_TEXT_CHARS` guards; it now validates before the
  old row is archived.
- **Counting an access through a superseded id un-archived the old version**
  and never counted the access on the live row.

### Fixed — recall
- **Task pins and session anchors disappeared on long-lived installs.** Every
  reader listed only the first 20 `active_task` rows, and one auto-anchor per
  past conversation accumulated forever. After about 20 conversations the
  current anchor and any formal `task pin` dropped out of the system prompt
  and the compression summary, a new anchor was written every turn, and
  `/reset` stopped archiving. All readers now see every live row, and
  auto-anchors older than `MEMORY_AUTO_ANCHOR_TTL_DAYS` (default 14) are
  archived.
- **`list_memories` came back short or empty** when many archived or
  other-session rows preceded the wanted ones.
- **The CLI and the agent used different databases under a profile.** The
  provider used `<hermes_home>/memory-lancedb`; the CLI used `$MEMORY_DB_DIR`
  or `~/.hermes/memory-lancedb`. `task pin` wrote where the agent never read,
  and `stats`, `doctor`, `reset` and `purge` acted on the wrong database. Both
  now resolve the path the same way: `--path`, then `MEMORY_DB_DIR`, then
  `<hermes home>/memory-lancedb`. **The provider now honours `MEMORY_DB_DIR`,
  as documented** — if you have it set, check it points where your memories
  are (a warning is logged when it leaves an existing profile database
  unused).
- **Deleted or corrected built-in memories kept being recalled.**
  `on_memory_write` ignored `remove` and stored a `replace` as a second row.
  `replace` now supersedes the mirrored copy and `remove` deletes it, earlier
  wordings included.
- **A waiting or blocked task was shown as "running".** The cached
  system-prompt block told the model to continue it. It now says what the
  task is parked on and not to continue; the task protocol text covers
  `waiting`, `blocked`, `task wait/block/unwait` and `task reported`, and the
  completed-task notice says how to mark results reported.
- **`/resume` and `/branch` showed the previous conversation's state.** Only a
  compression switch keeps the conversation id now.
- **Session "first memories" were the store's oldest global rows**, in every
  session. The session anchors now use only rows written in that session.
- Concurrent updates lost access counts and could leave several live versions
  of one memory; store read-modify-write paths are now serialised.

### Fixed — robustness and security
- **Untrusted text in the system prompt.** The auto-anchor quoted the first
  user message raw; a message containing newlines and the block delimiter
  could close the ACTIVE TASK block and plant instructions. It is reduced to
  one plain line and labelled as quoted, not an instruction.
- **A malformed numeric `MEMORY_*` setting crashed the plugin at import.** It
  now logs a warning and uses the default.
- **Subagent, cron and flush contexts wrote to the user's memory.** The
  provider reads the host's `agent_context` and only the primary agent
  writes. The host's `agent_identity` is accepted as the agent id.
- `doctor` reported a corrupt store as healthy and empty; it now reports the
  scan failure, points to `export --salvage`, and exits 1.
- `compact` rejects an out-of-range `--similarity` (0 merged everything into
  one row), reports how many clusters were actually merged, and exits 1 when
  some could not be.
- `init`'s default seed file is the host's `<hermes home>/memories/MEMORY.md`
  (the old `~/.hermes/memory/MEMORY.md` is still used when only that exists).

### Fixed — task ledger
- **Garbage collection archived and hard-deleted in the same run.** The
  archive grace period ran from the task's completion time, so a task
  completed more than retention + grace days ago lost its audit trail at
  once. Archiving now stamps `archived_at` and the grace period runs from it.
- **GC raced task writers.** It moved or deleted a task directory without
  taking the task lock; a writer then recreated a partial ledger. GC now takes
  the lock without waiting (a busy task is skipped until the next run) and
  re-reads the state under it; writers refuse to write into a directory that
  has gone.
- **A corrupt `state.json` raised a traceback from every verb** and broke
  `task list` for all tasks. Every verb now exits 4, listings skip the bad
  entry, and `task resume` tolerates bad `results.jsonl` lines.
- Advancing a completed task is refused; completing a task clears
  `waiting_on` and keeps the first completion time when repeated.
- GC collects status `completed` like `complete`, no longer reports a waiting
  or blocked task as abandoned, and `archive` is a reserved task id.
- The provider's automatic GC used the task root computed at import time, so
  a process serving several profiles collected another profile's ledger.
- A relative `--task-root` / `HERMES_TASK_ROOT` was stored as a relative path
  in the pin, which the gateway could not find, so the pin never refreshed.
  Task roots are now absolute. `task to-skill` writes to the profile's
  skills directory.

### Fixed — compaction
- **Compaction resurrected expired memories and destroyed task pins.** It
  clustered every non-archived row by similarity and wrote the merged row
  with fresh metadata: a row past its `valid_until` came back as a permanent
  one, a task pin lost `state_path` and `task_id`, and a session summary
  lost its type. Active-task and reflection rows, and rows with
  `valid_until`, `metadata_type` or `state_path`, are now left alone.
- **Compaction merged across scopes and sessions**, moving user-scoped text
  to the agent scope and producing rows no session could see. Clustering
  now stays within one scope and one session; the merged row keeps the
  session and the summed access counts.
- A cluster whose merged text exceeded `MEMORY_MAX_TEXT_CHARS` failed on
  every run; clusters stop growing at the limit.
- Overlapping compaction passes each wrote their own merged row. One pass
  runs per store at a time.
- The session summary exceeded the store's text limit by 15 characters when
  truncated and was dropped; it is capped to the limit.

### Fixed — retrieval and extraction
- **Expiry dates hid events before they happened.** Offsets were measured to
  the start of the referenced period (`next week` expired in 7 days). Each
  now reaches the end of the period: tomorrow 48h, next week 14d, this week
  7d, next month 62d, this month 31d, tonight/today 24h.
- **Core memories could never be demoted.** Tier decisions used the floored
  decay score, and the core floor is above the demotion threshold. They now
  use the unfloored score.
- **Tier evaluation only ever looked at the oldest rows.** It now walks the
  table one batch per evaluation.
- Results the reranker never saw kept their full score and overtook the
  reranked rows; they get the same penalty as unscored candidates.
- Entity boost matched names inside unrelated words (`Go` in `good`).
- **Valid LLM output was discarded.** JSON recovery took the first code fence
  in a reply, so an object containing a fenced code sample was reduced to the
  sample; a brace inside a string ended the object early.
- **A merge could rewrite an unrelated memory.** A merge, support or
  contextualize decision with a missing or non-integer `match_index` fell
  back to the nearest neighbour. It now becomes a create.
- **A failed extraction call silently dropped the turn.** It is now stored
  through the legacy fallback; a failed merge call keeps the new fact.
- Profile merge missed the existing profile row on any real store and
  created duplicates. The Anthropic client gets a 30s timeout.

### Fixed — reflection
- The reflection block read the oldest N rows and froze once more existed;
  it reads the newest. Reflection rows no longer appear as ordinary recall
  hits. The prompt now asks for phrasing the reflection filters accept
  (nearly every line used to be discarded), and nothing is written when no
  line survives. Section labels no longer trip the injection guard.

### Fixed — security and packaging
- **`save_config` could plant variables in `.env`.** A value with a line
  break added arbitrary lines; the file was rewritten in place and
  world-readable. Line breaks are refused and the file is written privately
  and atomically.
- The admission rejection log no longer stores conversation excerpts, is
  private to the user, and rotates at 1 MB. It is now documented.
- `scripts/memory_reset.sh` deleted any directory `MEMORY_DB_DIR` named; it
  now has the same guard and confirmation as `reset`.
- `lancedb` floor raised to 0.25.0: on 0.20–0.24 the full-text index needed
  an undeclared package and BM25/hybrid search failed.
- Docs: the durable-task skill covers the 0.16 task verbs; CLI help and
  configuration describe the profile-relative defaults; `SECURITY.md`
  supports 0.16.x.

### Fixed — vector scores and embeddings
- **Vector search used squared-L2 distance, but every score and threshold
  assumes cosine.** A memory with cosine similarity 0.5 scored 0.0, and
  `min_score` and dedup thresholds were twice as strict as documented. Vector
  queries now use exact cosine distance (refined past the index's compressed
  approximation) and new vector indexes are built with the cosine metric.
  Databases that already have an L2 vector index have it rebuilt once, with
  the cosine metric, in the background when the agent next starts (not on
  open, so start-up is not delayed and read-only commands never write);
  searches are correct in the meantime. Memories are not touched.
  If the rebuild fails, the old index is dropped and search stays exact.
- **Recall scored vector hits relative to the best hit of each query**, so the
  nearest memory always looked like a strong match and `min_score` /
  `MEMORY_MIN_RECALL_SCORE` could not reject a query that matched nothing.
  Relevance is now the absolute cosine similarity, so a score floor drops
  memories that do not match the query.
- **Hybrid `search()` `score` mixed two scales.** It came from whichever leg
  (vector or BM25) found the row and did not follow the result order. It is now
  the scaled RRF score (`_rrf_score * 30`, clamped to [0, 1]) and decreases
  with rank. Rows found by both legs also keep the BM25 `_score`.
- **Embeddings now use nomic's task prefixes.** nomic-embed-text was given raw
  text for both stored memories and queries, although it expects
  `search_document: ` and `search_query: ` and retrieves worse without them.
  New databases use the prefixes and record this in `embedding_scheme.json` in
  the database directory. Existing databases keep working exactly as before
  until you run the new `hermes-memory reembed`.
- **Loading nomic-embed-text no longer fails where transformers needs
  `trust_remote_code`.** The model is loaded as before first; only if that
  fails because the model's custom code needs consent is it retried with
  `trust_remote_code=True`, and only for nomic-embed-text models.
- **Export and import keep embedding schemes apart.** `export` writes each
  row's `embedding_scheme`. `import` re-encodes rows whose vectors were made
  under a different scheme than the target store's (exports from earlier
  versions count as unprefixed), so restoring an old backup into a new
  database no longer mixes the two kinds of vectors.

### Fixed — CLI and store
- **`purge`, `purge --include-expired` and `stats` only looked at the first
  100,000 rows** of a large store, so old archived or expired rows beyond that
  were never purged and the counts did not add up. They now read every row, in
  pages and without loading vectors.
- **Automatic table compaction only counted new memories**, so a store that
  mostly updates or deletes rows (recall bookkeeping, forgetting) grew data
  files and versions without limit. Updates and deletes now count too.
- **Looking up an old memory version whose newer version had been forgotten or
  compacted away returned 'not found'** although the archived row still
  existed. It now returns the archived row, so `forget` on it works. `update`
  and the access counter refuse such a row instead of reviving it.
- **Every recall turn did two full-table scans that loaded all vectors** to
  find the session's first and most recent memories. The session filter now
  runs inside the query, no vectors are read, and `anchors_for_session()`
  returns both lists from one scan.
- **`export` could silently drop rows.** `--limit` was applied before archived
  rows were filtered out, and the default capped output at 100,000 rows with no
  warning. `export` now writes everything by default, counts `--limit` against
  the rows it writes, rejects limits below 1, and warns on stderr when output
  was cut off.
- **`import` crashed with a traceback on a missing file or a malformed row**,
  leaving a partial import behind. It now skips bad rows with a warning,
  reports how many were rejected, exits 1 if any were, and a missing input file
  no longer creates an empty database. The confirmation prompts of `purge`,
  `init`, `reset`, `reembed` and `compact` print 'Aborted' instead of crashing
  when stdin is closed. `hermes lancedb_pro` with no subcommand prints help.
- **`stats`, `doctor`, `purge` and `compact` reported a healthy empty store
  (exit 0) when the dataset could not be read.** They now print the error,
  point to `export --salvage` and exit 1. `stats`, `doctor`, `search`,
  `export`, `purge` and `compact` no longer create a new empty database at a
  mistyped `--path`; they report 'no memory store at PATH'. `doctor` also flags
  a total/active/archived row-count mismatch.
- **`doctor -q` was accepted but ignored.** It now prints only findings and
  errors.

### Fixed — extraction and LLM calls
- **Transient LLM failures were never retried.** Both LLM adapters swallowed
  every error, so the "retry once on timeout, connection reset, 429 or 5xx"
  wrapper never ran. A transient failure is now retried once after a 1-3 s
  pause; 400, 401, 403 and quota errors are not retried. After the retry the
  caller still gets None and the client keeps the last error.
- **The extraction rate limiter could be bypassed by overlapping turns, and
  its log said "calls/hr".** Check and record are now one locked step, and the
  log says "extractions/hr". The cap counts extraction passes, each of which
  makes several LLM calls, not individual LLM calls.
- **Conversation and stored-memory text went raw into extraction, dedup,
  merge, utility and reflection prompts.** It is now wrapped in a delimited
  `untrusted_data` block that the model is told to treat as data, not
  instructions, and content cannot close the block. Extracted and merged
  abstract, overview and content are capped at 300, 2000 and 4000 characters.
- **LLM credentials ignored the host's per-profile secret scope.** When the
  host is importable, the extraction client reads its key, endpoint and model
  through the host's `get_secret`, so under the multiplexing gateway one
  profile's conversations are never sent with another profile's key. Without a
  host it still reads `os.environ`. `create_llm_client_from_env` also accepts an
  explicit `env` mapping.

### Fixed — provider
- **Turn writes were lost at exit or ran out of order.** `sync_turn` started
  its own untracked background thread, so the host's flush and shutdown drain
  did not wait for it: a slow LLM extraction was dropped at exit, and a write
  still running after 5 seconds was orphaned and ran alongside the next one.
  `sync_turn` now writes on the host's own memory worker, and uses a background
  thread only when called from the main thread. Writes run one at a time, in
  order.
- **Session end, a session reset and shutdown now wait for every pending
  turn write**, not only the most recent one, for up to 60 seconds (was 10 s
  for session end and 5 s for shutdown). Context compression and other
  session switches keep a 2-second wait, since they run inside a live turn. The new
  `MEMORY_WRITE_DRAIN_TIMEOUT_S` setting changes the limit. When it is reached,
  a warning names the hook and the number of writes still pending. Background
  turn writes also run under the caller's profile context (`HERMES_HOME` and
  secret scope), as the host requires.
- **The provider now reports itself unavailable, with the reason, when
  `lancedb` or `sentence-transformers` is not installed.** Before, it always
  reported available and then failed every write and recall silently.
- **`MEMORY_ACTIVE_TASK_PIN=off` did nothing.** It now turns off the
  active-task block in the system prompt and the pre-compression summary, and
  stops the session anchor from being written.
- **One oversized memory used up the whole recall character budget.** Items are
  now charged at their truncated length, and an item that does not fit is
  skipped instead of ending recall.
- **The session anchor was dropped or replaced by a placeholder** when
  `MEMORY_INJECTION_GUARD=reject` or `sanitize` flagged the user's first
  message. It is now still written, without the quoted snippet.
- **Long tool-heavy sessions overflowed the session-end reflection prompt** and
  silently skipped the reflection. The prompt now labels each message with its
  speaker role and is capped to the most recent `MEMORY_REFLECTION_MAX_CHARS`
  characters (default 12000). It also retries once on a transient LLM failure.
- **Loading the plugin opened or created a database at the default path.** The
  profile database is now opened only at `initialize()`.

### Fixed — reflection rows and fact keys
- **Every session reflection added a provenance 'event' row that nothing ever
  read, and derived reflection rows stayed in the table forever.** Event rows
  are no longer written, derived and combined rows expire after 14 days, and
  expired or never-read reflection rows (including ones from older versions)
  are removed at the end of each session.
- **Evidence from a dropped context was forgotten.** When a fact had more than
  8 contexts, evidence from the dropped context was counted once and then lost
  on the next update, so a contradiction could disappear from a memory's
  support strength. Dropped evidence now stays in `total_observations` and
  `global_strength`.
- **`derive_fact_key` split on the first colon even inside URLs, times and
  host:port**, so unrelated entities got the same key (for example
  `entities:https`), and a colon after an arrow beat the earlier arrow. A colon
  now separates a topic only when followed by whitespace, and the earlier of
  colon or arrow wins. Auto-captured preferences and entities now record their
  `fact_key` in metadata (not yet used for supersede lookups).

### Changed
- **Vector `score` and `_distance` change for every user.** `_distance` is now
  cosine distance (half the old squared-L2 value for unit vectors), and
  `score = 1 - _distance` is the true cosine similarity instead of
  `2·cos − 1` clamped at 0: a cos-0.7 match went from 0.4 to 0.7, and a cos-0.5
  match from 0.0 to 0.5. `search(mode='vector', min_score=t)` now keeps rows
  with cos ≥ t instead of cos ≥ (1+t)/2, so it returns more rows for the same
  `t`. Ordering is unchanged. Hybrid `search()` `score` now comes from RRF:
  about 0.98 for a row ranked first by both legs, about 0.49 for a row ranked
  first by one leg only. Ranking and the hybrid `min_score` (still compared
  with raw `_rrf_score`) are unchanged.
- **Recall `_final_score` drops for most results, and `min_score` /
  `MEMORY_MIN_RECALL_SCORE` is now an absolute floor.** The vector relevance
  term used to be 1.0 for the nearest hit of every query; it is now that hit's
  actual cosine similarity. A non-zero floor therefore filters more, and users
  with one may need to retune it (the default 0.0 is unchanged). Ranking also
  shifts: memory health (the 0.3 decay term) and the MMR near-duplicate penalty
  weigh relatively more; BM25-only hits no longer outrank the weakest vector
  hits; and the pre-decay `hard_min_score` of 0.05 now drops vector hits with
  cosine below 0.05.
- **Existing L2 or dot vector indexes are rebuilt once as cosine** by the
  agent's background warm-up (these exist when `init` seeded 256 or more
  rows). This costs one IVF_PQ training run and logs an INFO line. If training fails, the
  old index is dropped and searches are exact scans until
  `maybe_create_vector_index` builds a cosine one, which no longer retrains an
  existing cosine index on every call. Vector queries also add
  `refine_factor(5)`: with an index, the top `limit × 5` candidates are
  re-scored on the full vectors, a small extra read per query.
- **Downstream thresholds now see true cosine.** Reflection dedupe
  (`dedupe_threshold=0.97`) now skips writes with cos ≥ 0.97 (it needed
  0.985), the smart-extractor dedup prompt shows cosine similarity instead of
  `2·cos − 1`, and the CLI `search --mode vector` score column shows cosine
  similarity.
- **New databases store vectors with the `search_document: ` prefix** and send
  queries with `search_query: `, writing `embedding_scheme.json` in the
  database directory (this includes databases made by `init`, `reset` and
  tests in fresh directories). Existing databases with rows and no marker stay
  on the unprefixed scheme and are never relabelled on open.
  Importing a JSONL export into a store with a different scheme now re-encodes
  those rows, which needs the embedding model; the common case is restoring an
  old backup into a new database, which used to copy vectors without loading
  it. `encode()` and `encode_batch()` take a keyword-only `kind` (`'document'`
  by default, or `'query'`); `MemoryStore.stats()` returns `embedding_scheme`,
  `embedding_scheme_preferred` and `reembed_interrupted`, and the `doctor` and
  `stats` text output gains an `embedding_scheme:` line.
  Downgrade caveat: an older package version ignores the marker. If it writes
  to a database marked `task_prefix` it adds unprefixed rows that `reembed`
  will not detect; delete the marker and re-run `reembed` to fix that.
- **`export` writes all rows by default** instead of capping at 100,000.
  `--limit` counts written rows (non-archived unless `--include-archived`), and
  0 or negative values are rejected (exit 2). Export holds all scanned rows,
  vectors included, in memory before writing, so there is no longer a row cap.
- **`get_by_id(old_id)` can return an archived row** instead of None when the
  chain's next row is missing. `update()` returns None and
  `increment_access_count()` returns False for it.
- **Exit codes changed.** `stats`, `doctor`, `search`, `export`, `purge` and
  `compact` exit 1 with `no memory store at PATH` when the directory or
  `memories.lance` table is missing (they used to create an empty database and
  exit 0). `stats`, `purge`, `compact` and `doctor` exit 1 on a table that
  cannot be read; `MemoryStore.stats()` may carry an extra `scan_error` key.
  `import` exits 1 when any row was rejected, including lines that are not
  valid JSON, and a missing input file returns 1 before any database is
  created.
- **`compact` now prompts before merging and deleting.** Scripts that ran it
  non-interactively must add `--yes`; with no stdin it aborts with exit 1. A
  real run with zero clusters says 'nothing to merge' and returns 0 without
  writing. `doctor -q` now suppresses all informational sections, so a script
  that relied on its full output with `-q` sees nothing on a healthy store.
- **Purges delete in batches** of `id IN (...)` instead of one delete per row,
  and updates and deletes count towards the auto-optimize threshold, so
  `optimize()` runs more often on update- or delete-heavy stores.
  `purge_archived` and `purge_expired` gained an optional keyword `strict`, and
  `_scan_all` gained `columns` and accepts `limit=None`; defaults are unchanged.
- **Extraction LLM key, base URL and model are read through the host's
  `get_secret`** when `agent.secret_scope` is importable. Single-profile hosts
  see no change, because `get_secret` falls through to `os.environ`. Under the
  multiplexing gateway with no scope installed it fails closed: no client is
  built and extraction uses the legacy raw write instead of the launch
  profile's key. `create_llm_client_from_env(env=None)` takes an optional `env`
  mapping that, when given, is the only source consulted.
- **LLM retry and prompt behaviour.** A transient failure costs one extra call
  and a 1-3 s sleep inside the `sync_turn` worker thread; stacked on the SDKs'
  own internal retries, a hard outage can take longer per call than before. The
  shipped clients keep their last error per thread, so `get_last_error()`
  returns the calling thread's error. The extraction, dedup, merge, utility and
  reflection prompts changed wording (data blocks plus a notice), so utility
  score distributions may shift slightly. Extracted and merged text longer than
  300/2000/4000 characters (abstract/overview/content) is truncated before
  storing; rows on disk are untouched. A non-string abstract, overview or
  content in an LLM reply yields an empty field instead of an `AttributeError`.
  `SmartExtractor.extract_and_persist` checks for empty conversation text
  before the rate limit, so an empty conversation no longer takes the
  rate-limit fallback path. `get_preset()` reads `MEMORY_ADMISSION_AUDIT`.
- **Reflection storage.** Event rows (`memory-reflection-event`) are no longer
  written by `store_reflection_to_lancedb`; the pure payload builder still emits
  one by default (`write_event=True`). New derived and combined-legacy rows
  carry `metadata.valid_until = run_at + 14 days`; `purge --include-expired`
  removes them 30 days after that by default, and the provider's auto-purge does
  not call it. After every reflection write the default adapter deletes stale
  reflection rows: all event rows, derived items older than 14 days, and legacy
  combined rows older than 14 days that have no invariants or carry
  `valid_until`. This permanently deletes rows written by older versions.
  Invariant rows never expire; they are only read within the newest-rows scan
  window (`MEMORY_REFLECTION_SCAN_LIMIT`, default 200). Resolved-item
  suppression and mapped reflection rows from 0.4.0 are library-only helpers;
  the provider writes and reads invariant and derived rows only.
- **Support stats and fact keys.** `update_support_stats` returns different
  `total_observations` and `global_strength` for memories that had already
  dropped a slice (the stored totals were an undercount; the first later update
  restores them). `derive_fact_key` output changes for abstracts with a colon
  not followed by whitespace, and for abstracts with both `': '` and `'->'`; it
  is also derived at read time when no value is stored, so such rows get a
  different derived key on read. Auto-captured and auto-superseded preferences
  and entities now carry a `fact_key` in their metadata.
- **Turn writes under the host.** `sync_turn` now blocks the host's mem-sync
  worker for the whole extraction pass instead of returning at once, as the
  host's design expects. A single pass longer than 30 s now counts toward the
  host's stuck-call gate (`_BACKGROUND_CALL_STALL_S`). `on_pre_compress` and
  `on_session_switch` can wait up to 60 s for a slow extraction (was 2 s), and
  `shutdown` and `on_session_end` up to 60 s (were 5 s and 10 s), so exit can
  take that much longer while a write is pending. Main-thread `sync_turn`
  callers no longer wait up to 5 s for the previous write; writes queue in
  order. Data on disk is unchanged.
- **Recall and anchors.** `MEMORY_ACTIVE_TASK_PIN=off` removes the active-task
  block and stops auto-anchor writes; existing `active_task` rows stay on disk
  and are not injected while it is off. The recall budget counts truncated item
  length and skips an item that does not fit, so the same query can return more
  items. In reject and sanitize modes the auto-anchor is stored without the
  user's first-message quote. The reflection transcript is capped at the most
  recent 12000 characters with each line prefixed by its role, and a transient
  retry can add 1-3 s while the dispatch lock is held. The provider
  constructor no longer opens a database; a provider built with neither a store
  nor `initialize()` still falls back to the default-path store on first use,
  and the provider is unavailable (skipped by the host) when `lancedb` or
  `sentence_transformers` cannot be found.

### Fixed — found in review of the above
- Session anchors returned nothing for a session id containing a quote, a
  backslash or a non-ASCII character.
- `search --min-score` in hybrid mode compared against the raw fused rank
  (at most about 0.033), so any sensible threshold returned nothing. It is
  compared with the 0–1 `score`, which is also what `search` prints.
- The extractor's dedup step offered the nearest memories to the LLM however
  unrelated; `SIMILARITY_THRESHOLD` (0.55), declared but never applied, now
  filters that pool. A merge into a memory that vanished meanwhile keeps the
  new fact instead of dropping it.
- A lost write ticket (an interrupt between queuing a turn write and starting
  it) could block every later turn write; tickets always complete now.
- `/resume`, `/branch` and context compression no longer wait up to a minute
  for a turn's extraction to finish.
- Stale reflection rows are deleted in one batch rather than one by one.
- Reranker keys are read through the host's per-profile secret accessor, and
  a failed host secret lookup no longer falls back to another profile's
  environment.
- The automatic `trust_remote_code` retry applies only to the official nomic
  model repositories.

### Added
- **A `lancedb-memory` skill for the agent.** The provider gives the agent no
  memory tools, so until now nothing told it how this memory differs from
  the built-in one. The skill covers reading recall scores and trend tags,
  remembering and forgetting on purpose through the built-in `memory` tool,
  searching (and how each mode's score reads), checking health, passing `-y`
  to `compact`, and which commands are the operator's to run.
- **Skills ship with the plugin.** The three skills (`lancedb-memory`,
  `durable-task`, `task-to-skill`) are packaged and registered with Hermes
  when the plugin loads, as `lancedb_pro:<name>`; nothing has to be copied
  into `~/.hermes/skills/`. They moved from `skills/` to
  `src/hermes_memory_lancedb_pro/skills/`. Each now has a header with a
  description: without one Hermes showed the agent the install-path line
  as the description of the two existing skills.
- **`task unpin <id>`** removes a task's pin without touching its ledger, and
  `task pin` replaces an earlier pin for the same task instead of adding a
  duplicate.
- `MemoryStore.forget(id, purge_history=True)` also deletes every earlier
  version; `MemoryStore.row_exists(id)`; `store_raw(mem_id=...)`;
  `resolve_db_path()`; `MEMORY_AUTO_ANCHOR_TTL_DAYS`.
- **`hermes-memory reembed`** (also `hermes lancedb_pro reembed`) re-encodes
  every stored vector with nomic's task prefixes in one all-or-nothing commit
  and leaves text and metadata alone. It has `--dry-run`, `--batch-size` and a
  confirmation prompt (skip with `-y`). It needs the database to itself:
  stop Hermes first, or it refuses (a running agent holds the database, since
  a vector it encoded before the switch could otherwise be written after it).
  An interrupted run is detected and fixed by running it again. `doctor` and `stats` show the embedding scheme and
  recommend `reembed` for a legacy store.
- **`compact -y/--yes`.** `compact` asks for confirmation before merging and
  deleting, showing the number of clusters and memories affected; `-y/--yes`
  skips the prompt. It exits 0 with 'nothing to merge' when nothing qualifies.
  Automatic compaction in the provider is unchanged.
- **`MEMORY_ADMISSION_AUDIT`.** The rejection audit log
  (`admission-audit/rejections.jsonl`, which holds rejected candidate text and
  scores) can be turned off with `MEMORY_ADMISSION_AUDIT=off`
  (also `0`, `false`, `no`, `disabled`) without disabling admission control. On
  by default.
- **`MEMORY_WRITE_DRAIN_TIMEOUT_S`** (default 60) and
  **`MEMORY_REFLECTION_MAX_CHARS`** (default 12000), described above.
- **Continuous integration** runs on every push and pull request: ruff lint
  plus the pytest suite on Python 3.11 to 3.14. Runs on the same ref are
  cancelled when superseded.

---

## [0.16.0] — 2026-11-21

### Added
- **Task statuses `waiting` and `blocked`.** `task wait <id> --reason R
  [--ref REF] [--clears-when TEXT]` parks a task as `waiting` and records
  `waiting_on = {reason, ref, since, clears_when}`; `task block <id> --reason
  R` parks it as `blocked`; `task unwait <id>` returns either to `running` and
  clears `waiting_on`. `task show` reports both fields (filling `waiting_on:
  null` in for older ledgers), and the control block shows what a parked task
  is waiting on. Library: `wait_task`, `block_task`, `unwait_task`,
  `TaskStateError`.
- **`wake_count` and `max_wakes`.** `task create --max-wakes N` stores a wake
  budget (default `null`); `task wake <id>` adds 1 to `wake_count` (from 0)
  and sets `last_woken_at`. `task show` reports both. The budget is recorded,
  not enforced: the scheduler that wakes the task decides when to stop.
- **`task list --json`** prints a JSON array of every task's `state.json`
  object, sorted by `task_id` (`[]` when there are none).
- **Locked state writes.** Every ledger state write holds an exclusive
  `fcntl.flock` on `<task dir>/.lock` from the read of `state.json` through
  the atomic replace (and the matching `results.jsonl` / `events.jsonl`
  append), so concurrent writers — the agent, an unattended scheduler,
  other processes — no longer lose each other's updates. `task_lock()` is
  exported for code that edits `state.json` itself. It is not re-entrant,
  and locking a missing task raises `FileNotFoundError` rather than
  creating a directory. This release is the first with the lock: point a
  version check (such as a heartbeat's `ledger_flock_min_version`) at it.
- **`reported` flag.** New field `reported` (`false` at create); `task
  reported <id>` sets it to `true` (and `reported_at`) on a completed task,
  and is refused for any other status. A pinned completed task with
  `reported: true` no longer asks the model to present its results — the
  pin only says the task is done and was already reported, keeping the
  do-not-greet / do-not-re-run guard. `task list` shows `[complete,
  reported]`. Library: `mark_reported`.

### Changed
- **`task` commands exit `2` for usage/config errors and `4` for
  not-found/refused, instead of `1`.** Not found, already exists, and a
  state change the task does not allow are `4`; an unknown flag or command
  (argparse already used `2`), a missing required option, an invalid task
  id, a bad `--max-wakes` or `MEMORY_TASK_GC_MODE`, and `task` with no
  subcommand (which used to exit `0`) are `2`. Error paths that raised a
  traceback (an invalid task id) now report the error instead. Scripts that
  test for exit code `1` from `task` commands need updating; the other
  commands are unchanged.

### Fixed
- **`last_successful_iteration` is now only moved by a passing advance.**
  Every `task advance` used to set it to the counter's value *before* the
  advance, whatever the result — so a failing iteration was recorded as
  successful, and after one passing iteration it read `0`. A `--result pass`
  advance now sets it to the new `current_iteration` (one pass → `1`); a
  `--result fail` advance leaves it unchanged. `results.jsonl` entries are
  unchanged (`iteration` there is still the pre-advance counter).

---

## [0.15.1] — 2026-11-09

### Fixed
- **Gateway crash (SIGABRT / SIGSEGV) on Apple Silicon on the first turn.**
  torch's MPS backend is not thread-safe. `initialize()` starts the
  `lancedb-pro-warmup` thread, whose `warmup()` runs the first encode (model
  load + Metal shader JIT), while the first turn's prefetch thread encodes
  the query on the same model; two concurrent encodes abort in Metal with
  "A command encoder is already encoding to this command buffer". The store
  only locked model *loading*. Every embedder call — `encode()`,
  `encode_batch()`, `warmup()` and model construction — now takes one
  process-wide lock (`MemoryStore._encode_lock`). It is shared by all stores
  in the process because the MPS stream is process-wide and one gateway can
  hold a store per Hermes profile. Embedding throughput is unchanged: the
  model already ran one batch at a time.

### Added
- **`MEMORY_EMBEDDING_DEVICE`** — optional device override for the embedding
  model (`cpu`, `mps`, `cuda`, `cuda:1`, …). Unset keeps automatic selection.
  If the device cannot be used, a warning is logged and the model loads with
  automatic selection instead. `MEMORY_EMBEDDING_DEVICE=cpu` is a fallback
  for hosts where MPS misbehaves.
- `stats()` reports `embedding_device`, and `hermes lancedb_pro stats` /
  `doctor` print it (`auto` until the model is loaded; reading it never loads
  the model).

---

## [0.15.0] — 2026-11-03

### Changed — install method (action required on current Hermes)
- **Install with `hermes plugins install iamfoz/hermes-memory-lancedb-pro`.**
  Current Hermes builds its Python environment through its package manager
  (PM) from its own lock plus enabled plugins' declared dependencies, and
  rebuilds it on update. A package added with `hermes-pip` (since removed
  from Hermes) is not declared, so the rebuild dropped it, and the provider
  failed with `Memory provider 'lancedb_pro' loaded but no provider instance
  found` — the real `No module named 'hermes_memory_lancedb_pro'` is logged
  only at DEBUG — while `hermes memory status` reported the plugin
  `NOT installed`. The repo is now itself a Hermes plugin (root
  `plugin.yaml`, `__init__.py`, `cli.py`); PM builds the package and its
  dependencies into every environment generation. **Migrate:**
  `mv ~/.hermes/plugins/lancedb_pro ~/.hermes/plugins/lancedb_pro.old`, then
  `hermes plugins install iamfoz/hermes-memory-lancedb-pro`. The memory
  database and task ledgers are untouched.
- **Entry point moved to `hermes_agent.memory_providers`**, the group Hermes
  scans for pip-installed memory providers. It was registered under
  `hermes_agent.plugins`, the general plugin group, which the general plugin
  manager imports eagerly — the path upstream keeps memory providers out of.
  The legacy `hermes.plugins` group is kept.
- `install-plugin` now warns that its shim is not durable on package-managed
  Hermes and points at the plugin install; docs no longer reference
  `hermes-pip`.
- **Python 3.13 and 3.14 supported and tested**; the development pin
  (`.python-version`) moves from 3.11 to 3.14, matching the CPython 3.14.9
  that Hermes' package manager pins on every platform. `requires-python`
  stays `>=3.11`. Python 3.15 is not supported yet: Hermes itself requires
  `<3.15`, so the plugin cannot run there even though its dependencies
  already publish 3.15 wheels.

### Added
- **`valid_until` is enforced on recall.** Memories the smart extractor
  stamped with an expiry ("dentist appointment tomorrow" → +24h, "in 3
  days" → +72h) are now hidden from every recall path once it passes:
  `search` in all modes, `MemoryRetriever`, the session recency/first-turn
  anchors, and `search_by_vector` — the extractor's dedup neighbour search,
  so a fresh occurrence of an event is never merged into, or skipped
  because of, an expired row recall can no longer show. Rows are not
  deleted: `get_by_id`, `list_memories`, `export`, and `doctor` still see
  them, and `doctor` gains an *Expiry* section reporting the enforcement
  state and how many memories are expired. Missing, null, zero, or
  malformed `valid_until` values never hide a memory. Turn it off with
  `MEMORY_ENFORCE_VALID_UNTIL=off`.
- **`purge --include-expired`** — also hard-deletes memories whose
  `valid_until` passed more than `--grace-days` (default 30) ago, the rows
  recall already hides. The grace window runs from the expiry, mirroring how
  archived rows are aged from `invalidated_at`. Archived rows stay
  `purge_archived`'s job, so nothing is counted twice; missing or malformed
  `valid_until` values are never touched. Works with `--dry-run`, prompts
  for confirmation unless `-y`, and is available in both the standalone and
  in-host CLIs. New `MemoryStore.purge_expired(grace_period_days, *,
  dry_run)`.
- **`stats` subcommand** — store statistics (counts, categories, tiers)
  without the doctor's full anomaly scan; `--json` emits the raw
  `MemoryStore.stats()` dict for scripting.
- **`search` subcommand** — recall debugging from the terminal. Runs the same
  `MemoryStore.search` the provider uses (`--mode hybrid|vector|bm25`,
  `--category`, `--scope`, `--session-id`, `--min-score`); `--json` emits raw
  result rows with vectors stripped.
- **`purge` subcommand** — permanently delete archived rows past
  `--grace-days` (default 30). Prompts for confirmation unless `-y`;
  `--dry-run` reports the count without deleting.
  `MemoryStore.purge_archived` gained a matching `dry_run` keyword.
- **`compact` subcommand** — run the near-duplicate merge pass
  (`memory_compactor.run_compaction`) on demand with explicit knobs
  (`--min-age-days`, `--similarity`, `--max-scan`, `--scope`); `--dry-run`
  prints the merge plan without writing.
- **`--version` flag** on the standalone CLI, and a `package_version` line in
  the `doctor` header and `stats` output (JSON included) — CONTRIBUTING,
  SECURITY, and the bug-report template have always told users `doctor`
  reports the version; now it actually does.
- All four new subcommands are registered in both the standalone CLI and the
  in-host `hermes lancedb_pro …` form, via a shared parser helper so the two
  surfaces cannot drift.
- **`docs/PRD.md`** — product requirements document: goals, personas,
  functional/non-functional requirements, current-state inventory, gap
  analysis, and roadmap.

### Fixed
- **`hermes plugins doctor lancedb_pro` is clean.** The manifest's `hooks:`
  list (an alias for `provides_hooks`) named our `MemoryProvider` methods,
  but that field declares *general* plugin hooks registered with
  `ctx.register_hook()` and checked against Hermes' `VALID_HOOKS`. The doctor
  therefore reported nine "unknown hook" errors and ten "declared but
  registration did not add it" warnings. The host's memory manager calls
  provider methods directly and never reads that list, so it is removed; the
  doctor now reports OK.
- **The manifest version tracked the package version only by hand** (it is
  what `hermes plugins doctor` displays); a test now fails if `pyproject`,
  `__version__`, and either manifest disagree.
- **Temporal classification now actually affects decay.** The smart
  extractor recorded its static/dynamic verdict as `memory_temporal_type`,
  but decay reads `temporal_type` — which the store defaults to `"static"` on
  every row. Every extracted memory therefore decayed as a permanent fact,
  including "dentist appointment tomorrow". The extractor now writes both
  keys, and decay prefers `memory_temporal_type`, so memories already in a
  store pick up their recorded verdict with no migration. Dynamic memories
  now use the intended one-third half-life.
- **Temporal classifier: broader coverage, without misfiring on permanent
  facts.** Adds a curated subset of the unmerged
  `feat/temporal-classifier-expand` work: `this afternoon/evening/weekend`,
  `next <day>`, `N hours ago` / `a few days ago`, `deadline` and
  `due on/by/date/soon`, relative durations (`in 3 days`, `3天后`, `两周后`),
  and Chinese `即将` / `马上` / `截止` / `这个周末`, with matching expiries
  (offsets round up, so an event is never expired early). Relative durations
  are *weak* signals: a habitual marker (`usually`, `every week`, `通常`, …)
  vetoes them, so "I can usually finish a sprint in 10 days" stays static.
  Patterns from the branch that misclassified permanent facts were left
  out — bare dates and ISO dates (birthdays), ordinals ("the 5th edition"),
  `on <day>` ("I go to the gym on Monday"), bare `appointment` /
  `scheduled` / `expires`, bare 周末, and unnumbered 天后 (also "diva"). Ten
  such facts are pinned as regression tests.
- **`doctor` no longer recommends commands that don't exist.** It used to
  suggest `hermes-memory purge --grace-days 30` (which was not a subcommand)
  and calling `run_compaction()` (Python-only). Both recommendations now point
  at the real `purge` / `compact` subcommands.
- **Changelog compare links** — `[Unreleased]`, `[0.14.3]`, and `[0.14.2]`
  had no link definitions, so their section headers rendered as dead links.
- **Removed `kind: memory` from `plugin.yaml`.** hermes-agent's manifest
  taxonomy now accepts only `standalone`, `backend`, `exclusive`, `platform`,
  and `model-provider`; memory providers are auto-detected from the shim's
  `__init__.py` markers, and that detection only runs when `kind` is absent.
  The invalid `memory` value was coerced to `standalone`, skipping
  auto-detection and letting the general plugin manager eagerly import the
  provider as a standalone plugin — the path upstream keeps memory providers
  out of. Memory discovery kept working via its separate path, but the
  plugin was mis-classified. **Existing installs:** re-run
  `hermes-memory-lancedb-pro install-plugin` after upgrading — it detects the
  stale copied manifest and refreshes it. A regression test now fails if a
  `kind` key reappears.

### Changed
- **Trailing optional parameters are now keyword-only** on high-arity public
  APIs (ruff `PLR0917`). The first five parameters keep their positions;
  only the 6th onward must be named:
  - `MemoryStore.store(…, *, confidence, metadata_extra)`
  - `MemoryStore.update(…, *, metadata_extra)`
  - `MemoryStore.list_memories(…, *, include_archived, session_id, categories)`
  - `MemoryStore.search(…, *, session_id, min_score)`
  - `build_merge_prompt(*, …)` — fully keyword-only: its seven same-typed
    string arguments were easy to transpose silently.

  No call site in the package, tests, scripts, or docs passed those
  arguments positionally. External code that did will now get a
  `TypeError` — name the arguments to fix it.
- **Agent-facing task commands now use the in-host `hermes lancedb_pro task …`
  form.** The durable-task protocol injected by the provider every turn, the
  `durable-task` / `task-to-skill` skills, and the `task to-skill` scaffold
  breadcrumb previously told the agent to run the bare `hermes-memory-lancedb-pro
  task …` console script. When Hermes shells out with a PATH that excludes its
  own venv `bin/`, that script is not found and the command fails with exit 127.
  The in-host form is dispatched in-process by the host (wired via
  `register_cli` / `register_cli_command`), so it needs no PATH lookup and works
  regardless of how the host spawns subprocesses. The standalone
  `hermes-memory-lancedb-pro` console script is unchanged and remains the
  documented invocation for using the plugin outside Hermes.

---

## [0.14.3] — 2026-05-22

### Added
- **`hermes lancedb_pro …` host-level subcommand restored.** The provider's
  `register(ctx)` now also calls `ctx.register_cli_command(...)` (when the
  host exposes that API), re-attaching `init / doctor / export / import /
  reset / task` as host-level subcommands of `hermes`. Falls back silently
  on hosts without the API; the standalone `hermes-memory-lancedb-pro`
  script keeps working either way.

### Fixed
- **Entry-point group renamed to `hermes_agent.plugins`** to match the
  current hermes-agent plugin loader, so entry-point-based discovery works
  again without depending on the `~/.hermes/plugins/lancedb_pro/` shim.
  The old `hermes.plugins` group is also kept for compatibility with
  older hermes-agent versions.

---

## [0.14.2] — 2026-05-22

### Added
- **`export --salvage`** — a best-effort recovery scan for a corrupted store.
  A normal export aborts the moment one on-disk fragment file is missing;
  with `--salvage` the exporter walks the dataset's version history
  newest-first, and falls back to reading it fragment-by-fragment — skipping
  the unreadable ones — so as many memories as possible can still be rescued
  to JSONL.

### Fixed
- **`export` no longer reports a corrupt store as an empty one.** A failed
  full-table scan was swallowed (`_scan_all` returned an empty iterator), so
  `export` wrote an empty file and exited `0` — indistinguishable from a
  genuinely empty store. It now exits non-zero with a clear error and points
  to `--salvage`.
- **`export -o` creates missing parent directories.** Exporting to a path
  whose parent directory did not exist raised `FileNotFoundError`; the parent
  directory is now created automatically.

---

## [0.14.1] — 2026-05-22

### Added
- `MEMORY_FD_LIMIT` (default 4096) — on construction, `MemoryStore` raises its
  process's soft open-file limit toward the hard ceiling. LanceDB opens a
  descriptor per on-disk fragment, so a process started with a low soft limit
  (macOS shells default to 256) could exhaust descriptors under sustained
  concurrent load. The store now lifts the limit itself — no privileges, no
  shell or launchd configuration required. Best-effort and capped at the hard
  limit; set `0` to disable.

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

[Unreleased]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.17.1...HEAD
[0.17.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.15.1...v0.16.0
[0.15.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.14.3...v0.15.0
[0.14.3]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.14.2...v0.14.3
[0.14.2]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/iamfoz/hermes-memory-lancedb-pro/compare/v0.14.0...v0.14.1
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
