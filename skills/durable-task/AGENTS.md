## Durable Task Protocol

For any task requiring more than 3 sequential steps, tool calls, or iterations,
use the durable task ledger before starting work.

### Starting a long-running task

```bash
hermes-memory-lancedb-pro task create \
  --id <descriptive-unique-id> \
  --objective "<clear one-line objective>" \
  --iterations <N>

hermes-memory-lancedb-pro task pin <task-id>
```

The pin writes the task state to the memory store. The plugin reloads `state.json`
from disk on every turn, so the model always sees the current iteration and
`next_action` — including after context compaction.

### Each iteration loop

```bash
hermes-memory-lancedb-pro task resume <task-id>   # read state before doing work
# ... do the work for this iteration ...
hermes-memory-lancedb-pro task advance <task-id> \
  --result pass \
  --next-action "Run iteration <N+1>." \
  --summary "<one sentence: what happened>"
```

### Completion

```bash
hermes-memory-lancedb-pro task complete <task-id> --summary "<what was done>"
```

### If context resets or you are about to greet the user

1. Run `hermes-memory-lancedb-pro task list` — if a task is running, do not greet.
2. Run `hermes-memory-lancedb-pro task resume <task-id>` to reload state.
3. Continue from `next_action`. The state file is the source of truth, not conversation history.

### Invariants (always true while a task is running)

- Do not greet the user.
- Do not ask the user what you were doing — read `state.json`.
- One iteration per response. Record the result before starting the next.
- If `task list` shows status `running`, continue that task.
