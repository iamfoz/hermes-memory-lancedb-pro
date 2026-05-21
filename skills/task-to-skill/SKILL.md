# Skill: Task → Skill

**Install location**: `~/.hermes/skills/task-to-skill/SKILL.md`

Use this skill when the user asks to turn a task they ran into a reusable
skill. The durable-task ledger already recorded what was done; your job is to
distil it into a clean, reusable `SKILL.md` + `AGENTS.md` that a future agent
can follow. The plugin gives you a scaffold — **you** write the real skill.

---

## When to use this skill

Trigger this protocol when the user says something like:

- "Turn the task we just did into a skill."
- "Could you make a skill out of that?"
- "We ran a task a few days ago — I'd like to turn it into a skill."

This is **user-initiated** — only run it when the user asks. There are two
cases: a **recent** task you still remember (Flow A), and an **older** task
you need to find (Flow B).

---

## Flow A — a recent task

You ran it, so you know its task ID (or `task list` shows it).

1. Scaffold the draft:
   ```bash
   hermes-memory-lancedb-pro task to-skill <task-id>
   ```
   This writes a draft `SKILL.md` + `AGENTS.md` under
   `~/.hermes/skills/<task-id>/`.
2. Author the real skill — see **Authoring** below.

---

## Flow B — find an older task

The user has not named a specific task. Surface candidates and let them pick.

1. List candidate tasks (completed tasks, live and archived):
   ```bash
   hermes-memory-lancedb-pro task to-skill --list
   ```
2. **Present the list to the user as selectable options.** If the client
   supports option buttons — the same UI used for model selection and
   operation approval — present each candidate as a button; otherwise present
   a numbered text list. Always add a final option:

   > `0` — none of these (search by keyword, or stop)

3. Act on the user's choice:
   - A candidate → go to step 4.
   - `0` → ask for keywords, then list the narrowed set:
     ```bash
     hermes-memory-lancedb-pro task to-skill --search "<keywords>"
     ```
     Re-present the result (step 2). If the user wants to stop, stop.
4. Scaffold the chosen task:
   ```bash
   hermes-memory-lancedb-pro task to-skill <task-id>
   ```
5. Author the real skill — see **Authoring** below.

---

## Authoring — turning the draft into a real skill

The scaffold is only a starting point; **you are the author**. The draft holds
the task's objective, its invariants, and its raw iteration history. Rewrite it
into something a future agent can actually follow:

- **Write the Protocol from what actually happened.** You ran this task — use
  what you remember, not just the draft's iteration log. Capture the real
  steps, the order they must happen in, and any preconditions.
- **Generalise.** Strip out the specifics of this one run — exact ids, paths,
  one-off values. A skill is reusable: describe the *shape* of the work.
- **Be imperative and concrete.** Each step is a bounded action with the exact
  command or decision rule. Match the house style of
  `~/.hermes/skills/durable-task/SKILL.md`.
- **Keep the invariants** that genuinely always apply; drop ones that were
  specific to this task.
- **Write a tight `AGENTS.md`** — a 3–5 bullet quickstart pointing back to
  `SKILL.md`.
- **Delete the scaffolding** — the `<!-- ... -->` comments and the
  "Source material" block — once the Protocol is rewritten.

Then tell the user where the finished skill is and what it does.

---

## Invariants

- User-initiated only — run this when the user asks, not on your own.
- Never invent steps the task did not perform. If the draft's material is too
  thin to write a real Protocol, say so and ask the user to fill the gaps.
- The output is a *draft* until you have rewritten it — never present an
  unedited scaffold as a finished skill.

---

## Reference

| Command | Purpose |
|---------|---------|
| `task to-skill --list` | List completed tasks (live + archived) as skill candidates |
| `task to-skill --search "<kw>"` | List candidates whose id/objective/summary match keywords |
| `task to-skill <task-id>` | Scaffold a draft skill from a task (live or archived) |
