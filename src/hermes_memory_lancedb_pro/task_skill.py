"""Scaffold a reusable skill from a task ledger.

``task to-skill`` takes a task that did something useful and writes a *draft*
skill — ``SKILL.md`` + ``AGENTS.md`` in the format used under ``skills/`` —
that the author then refines. The task's objective and invariants transfer
directly; the Protocol section is a skeleton seeded with the task's iteration
history for the author to rewrite into reusable steps.

Pure-Python — no LLM and no heavy dependencies. The output is explicitly a
draft, not a finished skill.
"""

from __future__ import annotations

import json
from pathlib import Path

from .task_ledger import _task_dir, _validate_task_id, load_state

__all__ = ["scaffold_skill_from_task"]

_LOG_EXCERPT_LIMIT = 4000


def _count_results(task_id: str, root: Path | None) -> tuple[int, int, int]:
    """Return (total, passed, failed) iteration results from results.jsonl."""
    path = _task_dir(task_id, root) / "results.jsonl"
    total = passed = failed = 0
    try:
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if rec.get("result") == "pass":
                    passed += 1
                elif rec.get("result") == "fail":
                    failed += 1
    except (FileNotFoundError, OSError):
        pass
    return total, passed, failed


def _read_log(task_id: str, root: Path | None) -> str:
    """Return the task's log.md content, truncated, or '' when absent/empty."""
    try:
        text = (_task_dir(task_id, root) / "log.md").read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return ""
    if len(text) > _LOG_EXCERPT_LIMIT:
        text = text[:_LOG_EXCERPT_LIMIT] + "\n…(truncated)"
    return text


def _title(slug: str) -> str:
    return " ".join(w for w in slug.replace("_", " ").replace("-", " ").split()).title()


def _skill_md(state: dict, slug: str, results: tuple[int, int, int], log_text: str) -> str:
    objective = state.get("objective") or "(no objective recorded)"
    invariants = [str(i) for i in (state.get("invariants") or [])]
    total, passed, failed = results

    inv_block = (
        "\n".join(f"- {inv}" for inv in invariants)
        if invariants
        else "- (no invariants recorded on the source task — add the rules this skill must follow)"
    )
    log_block = (
        "\nNotes from the task's `log.md`:\n\n> "
        + log_text.replace("\n", "\n> ")
        + "\n"
        if log_text
        else ""
    )

    return f"""# Skill: {_title(slug) or slug}

**Install location**: `~/.hermes/skills/{slug}/SKILL.md`

> **DRAFT** — scaffolded from task `{state.get("task_id", slug)}` by
> `hermes-memory-lancedb-pro task to-skill`. The objective and invariants
> below transferred directly from the task; the **Protocol** section is a
> skeleton seeded with the task's iteration history. Review and rewrite it
> into a clean, reusable procedure before relying on this skill.

---

## When to use this skill

The source task's objective was:

> {objective}

Replace this with the concrete trigger conditions for the *reusable* skill —
the situations in which a future agent should invoke it.

---

## Protocol

<!-- Rewrite the steps below into a clean, reusable, numbered procedure.
     The "Source material" block is the raw history of the task this skill
     was derived from — use it to reconstruct the real steps, then delete it. -->

### Step 1 — ...

(describe the first step)

### Step 2 — ...

(describe the next step)

### Source material from task `{state.get("task_id", slug)}`

- Objective: {objective}
- Iterations recorded: {total} ({passed} pass, {failed} fail)
- Latest summary: {state.get("recent_summary") or "(none)"}
{log_block}
---

## Invariants

These rules applied throughout the source task and are carried over verbatim:

{inv_block}

---

## Example

<!-- Add a concrete, runnable example of using this skill. -->

---

## Reference

<!-- List the key commands or tools this skill relies on. -->
"""


def _agents_md(state: dict, slug: str) -> str:
    objective = state.get("objective") or "(no objective recorded)"
    invariants = [str(i) for i in (state.get("invariants") or [])]
    inv_block = (
        "\n".join(f"- {inv}" for inv in invariants)
        if invariants
        else "- (add the rules this skill must follow)"
    )
    return f"""## {_title(slug) or slug}

> **DRAFT** scaffolded by `task to-skill` — review before use. See `SKILL.md`
> for the full draft.

Invoke this skill when starting work like: {objective}

### Steps

<!-- Summarise the key steps in a few bullets. See SKILL.md for the full draft. -->

### Invariants

{inv_block}
"""


def scaffold_skill_from_task(
    task_id: str,
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Scaffold a draft skill (``SKILL.md`` + ``AGENTS.md``) from a task.

    Reads the task's ``state.json``, ``results.jsonl`` and ``log.md`` and
    writes a draft skill into *out_dir* (default ``~/.hermes/skills/<task_id>/``).
    The objective and invariants transfer directly; the Protocol is a skeleton
    seeded with the iteration history. Returns the skill directory.

    Raises ``FileNotFoundError`` if the task does not exist, ``ValueError`` for
    an invalid *task_id*, and ``FileExistsError`` if *out_dir* already contains
    files and *force* is not set.
    """
    slug = _validate_task_id(task_id)
    state = load_state(task_id, root)  # FileNotFoundError if the task is unknown

    dest = out_dir or (Path.home() / ".hermes" / "skills" / slug)
    if dest.exists() and any(dest.iterdir()) and not force:
        raise FileExistsError(
            f"{dest} already exists and is not empty — pass force=True to overwrite"
        )
    dest.mkdir(parents=True, exist_ok=True)

    results = _count_results(task_id, root)
    log_text = _read_log(task_id, root)
    (dest / "SKILL.md").write_text(_skill_md(state, slug, results, log_text), encoding="utf-8")
    (dest / "AGENTS.md").write_text(_agents_md(state, slug), encoding="utf-8")
    return dest
