## Task → Skill

Invoke this skill when the user asks to turn a task they ran into a reusable
skill. It is user-initiated. See `SKILL.md` for the full procedure.

### A recent task

```bash
hermes-memory-lancedb-pro task to-skill <task-id>
```
Then rewrite the draft's Protocol into a clean, reusable procedure.

### An older task — let the user pick

```bash
hermes-memory-lancedb-pro task to-skill --list            # or --search "<kw>"
```
Present the candidates as selectable options — client option buttons where
available (as with model selection / operation approval), otherwise a numbered
text list — always with a final `0 — none of these (search by keyword, or
stop)`. On a keyword choice, re-list with `--search`; then scaffold the pick
with `task to-skill <task-id>`.

### Authoring

- Write the Protocol from what you actually did, not just the draft's log.
- Generalise — strip one-off ids/paths; describe the reusable shape of the work.
- Imperative, concrete steps; keep only the invariants that always apply.
- Delete the scaffold's `<!-- ... -->` comments and "Source material" block.
- The scaffold is a draft — never present it unedited as a finished skill.
