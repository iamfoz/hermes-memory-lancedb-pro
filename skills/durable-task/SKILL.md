# Skill: Durable Task Ledger

**Install location**: `~/.hermes/skills/durable-task/SKILL.md`

Use this skill for any task that takes more than 3 sequential steps or tool calls.
It ensures your progress survives context compaction, session restarts, and model
resets by keeping state in a file on disk rather than in the conversation.

---

## When to use this skill

Trigger this protocol whenever:

- The user asks to run a test suite, benchmark, stress test, or any iterative work
- The user asks you to "keep going", "run N iterations", or "repeat until done"
- You expect to make more than 3 tool calls to complete the task
- You are resuming a task after receiving a greeting or context reset

---

## Protocol

### Step 1 — Create the task ledger

Before touching anything else, create a task ledger:

```bash
hermes-memory-lancedb-pro task create \
  --id <task-id> \
  --objective "<clear one-line objective>" \
  --iterations <N>
```

Choose a task ID that is unique and descriptive, e.g. `stress-test-2026-05-20`.
If `--iterations` is unknown, omit it.

### Step 2 — Pin it to memory

```bash
hermes-memory-lancedb-pro task pin <task-id>
```

This stores the task state in the memory database. The memory plugin reloads
`state.json` on every turn, so the model always sees the current iteration and
next action — even after context compaction wipes the conversation history.

**You only need to pin once.** The pin does not need to be refreshed as the task
advances; the plugin reads `state.json` live.

### Step 3 — Before each iteration

Read the current state before doing any work:

```bash
hermes-memory-lancedb-pro task resume <task-id>
```

Confirm:
- The task ID and objective match what you expect
- `current_iteration` and `next_action` are correct
- There are no unresolved blockers

### Step 4 — Do the work

Execute one bounded step. Do not attempt multiple iterations in a single response.
One step = one `advance`.

### Step 5 — After each iteration

Record the result and advance the counter:

```bash
hermes-memory-lancedb-pro task advance <task-id> \
  --result pass \
  --next-action "Run iteration <N+1>." \
  --summary "<one sentence: what happened>"
```

Use `--result fail` if the step errored. Always set `--next-action` explicitly
so the next turn knows exactly what to do without re-reading the whole history.

### Step 6 — Check stopping condition

After `advance`, check whether the task is complete:

```bash
hermes-memory-lancedb-pro task show <task-id>
```

If `current_iteration >= target_iterations`, or the objective is met:

```bash
hermes-memory-lancedb-pro task complete <task-id> --summary "<what was done>"
```

Then report results to the user.

---

## Recovery after a reset or greeting

If you find yourself about to greet the user, or if context is unclear:

1. **Check for a running task first:**
   ```bash
   hermes-memory-lancedb-pro task list
   ```

2. **If a running task exists, resume it:**
   ```bash
   hermes-memory-lancedb-pro task resume <task-id>
   ```

3. **Continue from `next_action`.** Do not re-introduce yourself. Do not ask
   the user what you were doing. The state file is the source of truth.

4. **Log the reset event:**
   The `resume` output tells you exactly where you are. Proceed.

---

## Invariants

These rules apply for the entire lifetime of a running task:

- Do not greet the user.
- Do not restart the conversation.
- Do not ask the user what you were doing — read `state.json`.
- Before each iteration: confirm state with `task resume`.
- After each iteration: update state with `task advance`.
- If `task list` shows a running task: continue it, do not start a new one.
- If blockers appear: record them with `task advance --result fail` and report.

---

## Example: stress test (50 iterations)

```bash
# Setup (once)
hermes-memory-lancedb-pro task create \
  --id stress-test-$(date +%Y%m%d-%H%M) \
  --objective "Run 50 stress-test iterations against hermes-memory-lancedb-pro" \
  --iterations 50

hermes-memory-lancedb-pro task pin stress-test-<id>

# Each iteration
hermes-memory-lancedb-pro task resume stress-test-<id>
# ... do the work ...
hermes-memory-lancedb-pro task advance stress-test-<id> \
  --result pass \
  --summary "Iteration N: retrieval latency 42ms, 0 failures"

# Completion
hermes-memory-lancedb-pro task complete stress-test-<id> \
  --summary "All 50 iterations passed. Mean latency 44ms, 0 failures."
```

---

## Reference

| Command | Purpose |
|---------|---------|
| `task create --id <id> --objective <text> --iterations N` | Create task ledger |
| `task pin <id>` | Pin to memory (survives compaction) |
| `task resume <id>` | Print current state and control block |
| `task advance <id> --result pass\|fail --next-action <text>` | Record iteration |
| `task list` | List all tasks |
| `task show <id>` | Full state as JSON |
| `task complete <id> --summary <text>` | Mark done |
