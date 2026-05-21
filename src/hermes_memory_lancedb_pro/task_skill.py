"""Scaffold a reusable skill from a task ledger, and list skill candidates.

``task to-skill`` takes a task that did something useful and writes a *draft*
skill — ``SKILL.md`` + ``AGENTS.md`` in the format used under ``skills/`` —
that the agent then refines into a polished, reusable skill. The task's
objective and invariants transfer directly; the Protocol section is a skeleton
seeded with the task's iteration history.

``list_skill_candidates`` surfaces completed tasks — live and archived — so an
older task can be found and turned into a skill without knowing its id.

Pure-Python — no LLM and no heavy dependencies. The scaffold is explicitly a
draft; the polish comes from the agent rewriting it (see the ``task-to-skill``
skill under ``skills/``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .task_ledger import TASK_ROOT, _validate_task_id

__all__ = ["list_skill_candidates", "scaffold_skill_from_task"]

_LOG_EXCERPT_LIMIT = 4000
_ARCHIVE_DIRNAME = "archive"


def _resolve_task_dir(task_id: str, root: Path | None = None) -> Path | None:
    """Find a task's directory — live under the root, or under ``archive/``.

    Returns the directory Path, or None when no such task exists. Handles GC's
    collision-suffixed archive names (``<id>__<epoch>``) by matching the
    ``task_id`` field inside ``state.json``.
    """
    base = root or TASK_ROOT
    live = base / task_id
    if (live / "state.json").is_file():
        return live
    archive_dir = base / _ARCHIVE_DIRNAME
    exact = archive_dir / task_id
    if (exact / "state.json").is_file():
        return exact
    if archive_dir.is_dir():
        for d in sorted(archive_dir.iterdir()):
            sp = d / "state.json"
            if not sp.is_file():
                continue
            try:
                if json.loads(sp.read_text(encoding="utf-8")).get("task_id") == task_id:
                    return d
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _count_results(task_dir: Path) -> tuple[int, int, int]:
    """Return (total, passed, failed) iteration results from results.jsonl."""
    total = passed = failed = 0
    try:
        with (task_dir / "results.jsonl").open(encoding="utf-8") as f:
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


def _read_log(task_dir: Path) -> str:
    """Return the task's log.md content, truncated, or '' when absent/empty."""
    try:
        text = (task_dir / "log.md").read_text(encoding="utf-8").strip()
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

    Resolves *task_id* whether the task is live or archived, reads its
    ``state.json`` / ``results.jsonl`` / ``log.md``, and writes a draft skill
    into *out_dir* (default ``~/.hermes/skills/<task_id>/``). Returns the skill
    directory.

    Raises ``ValueError`` for an invalid *task_id*, ``FileNotFoundError`` when
    no such task exists, and ``FileExistsError`` if *out_dir* already contains
    files and *force* is not set.
    """
    slug = _validate_task_id(task_id)
    task_dir = _resolve_task_dir(task_id, root)
    if task_dir is None:
        raise FileNotFoundError(f"task {task_id!r} not found (live or archived)")
    try:
        state = json.loads((task_dir / "state.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise FileNotFoundError(f"task {task_id!r}: unreadable state.json: {e}") from e

    dest = out_dir or (Path.home() / ".hermes" / "skills" / slug)
    if dest.exists() and any(dest.iterdir()) and not force:
        raise FileExistsError(
            f"{dest} already exists and is not empty — pass force=True to overwrite"
        )
    dest.mkdir(parents=True, exist_ok=True)

    results = _count_results(task_dir)
    log_text = _read_log(task_dir)
    (dest / "SKILL.md").write_text(
        _skill_md(state, slug, results, log_text), encoding="utf-8"
    )
    (dest / "AGENTS.md").write_text(_agents_md(state, slug), encoding="utf-8")
    return dest


def _candidate(task_dir: Path, location: str) -> dict | None:
    """Build a candidate dict for a completed task, or None if not eligible."""
    try:
        state = json.loads((task_dir / "state.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict) or state.get("status") != "complete":
        return None
    return {
        "task_id": state.get("task_id", task_dir.name),
        "objective": state.get("objective", "") or "",
        "completed_at": state.get("completed_at") or state.get("updated_at") or "",
        "summary": state.get("recent_summary", "") or "",
        "location": location,
        "held": bool(state.get("gc_hold")),
    }


def list_skill_candidates(
    root: Path | None = None, *, search: str | None = None
) -> list[dict]:
    """List completed tasks — live and archived — that could become a skill.

    Each entry is a dict with ``task_id``, ``objective``, ``completed_at``,
    ``summary``, ``location`` (``"live"`` / ``"archived"``) and ``held``,
    ordered newest-completion-first. When *search* is given, only candidates
    where every whitespace-separated term (case-insensitive) appears in the
    task id, objective or summary are returned.
    """
    base = root or TASK_ROOT
    candidates: list[dict] = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name != _ARCHIVE_DIRNAME:
                c = _candidate(d, "live")
                if c:
                    candidates.append(c)
        archive_dir = base / _ARCHIVE_DIRNAME
        if archive_dir.is_dir():
            for d in sorted(archive_dir.iterdir()):
                if d.is_dir():
                    c = _candidate(d, "archived")
                    if c:
                        candidates.append(c)

    if search:
        terms = search.lower().split()
        candidates = [
            c
            for c in candidates
            if all(
                term in f"{c['task_id']} {c['objective']} {c['summary']}".lower()
                for term in terms
            )
        ]

    candidates.sort(key=lambda c: c["completed_at"], reverse=True)
    return candidates
