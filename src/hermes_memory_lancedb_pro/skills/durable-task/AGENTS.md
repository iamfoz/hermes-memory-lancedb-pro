## Durable Task Protocol

Load the `lancedb_pro:durable-task` skill when starting any multi-step task for the full
procedure guide. For any task requiring more than 3 sequential steps, tool
calls, or iterations, you MUST use the durable task ledger before starting work.

### Starting a long-running task

```bash
hermes lancedb_pro task create \
  --id <descriptive-unique-id> \
  --objective "<clear one-line objective>" \
  --iterations <N>

hermes lancedb_pro task pin <task-id>
```

The pin writes the task state to the memory store. The plugin reloads `state.json`
from disk on every turn, so the model always sees the current iteration and
`next_action` — including after context compaction.

### Each iteration loop

```bash
hermes lancedb_pro task resume <task-id>   # read state before doing work
# ... do the work for this iteration ...
hermes lancedb_pro task advance <task-id> \
  --result pass \
  --next-action "Run iteration <N+1>." \
  --summary "<one sentence: what happened>"
```

### Completion

```bash
hermes lancedb_pro task complete <task-id> --summary "<what was done>"
```

Report the results to the user, then record that so they are not repeated:

```bash
hermes lancedb_pro task reported <task-id>
```

### If the task cannot proceed

```bash
hermes lancedb_pro task wait <task-id> --reason "<what>"     # waiting on something external
hermes lancedb_pro task block <task-id> --reason "<what>"    # needs a person
hermes lancedb_pro task unwait <task-id>                     # back to running
```

Do not continue a waiting or blocked task until it is un-parked.

### If context resets or you are about to greet the user

1. Run `hermes lancedb_pro task list` — if a task is running, do not greet.
2. Run `hermes lancedb_pro task resume <task-id>` to reload state.
3. Continue from `next_action`. The state file is the source of truth, not conversation history.

### Invariants (always true while a task is running)

- Do not greet the user.
- Do not ask the user what you were doing — read `state.json`.
- One iteration per response. Record the result before starting the next.
- If `task list` shows status `running`, continue that task.
- If it shows `waiting` or `blocked`, do not continue it: check whether the
  wait has cleared, then `task unwait`, or tell the user what it waits on.
