"""Unit tests for task_skill — scaffolding a skill from a task. No heavy deps."""

from __future__ import annotations

import json

import pytest

from hermes_memory_lancedb_pro.task_ledger import (
    advance_iteration,
    complete_task,
    create_task,
)
from hermes_memory_lancedb_pro.task_skill import (
    list_skill_candidates,
    scaffold_skill_from_task,
)


@pytest.fixture
def task_root(tmp_path):
    return tmp_path / "tasks"


class TestScaffold:
    def test_writes_both_files(self, task_root, tmp_path):
        create_task("demo", "Do the demo thing.", root=task_root)
        out = tmp_path / "skill-out"
        dest = scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        assert dest == out
        assert (out / "SKILL.md").exists()
        assert (out / "AGENTS.md").exists()

    def test_skill_md_has_expected_sections(self, task_root, tmp_path):
        create_task("demo", "Do the demo thing.", root=task_root)
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        text = (out / "SKILL.md").read_text()
        for heading in (
            "# Skill:",
            "Install location",
            "## When to use this skill",
            "## Protocol",
            "## Invariants",
        ):
            assert heading in text
        assert "Do the demo thing." in text  # objective carried over

    def test_agents_md_generated(self, task_root, tmp_path):
        create_task("demo", "obj", root=task_root)
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        assert "DRAFT" in (out / "AGENTS.md").read_text()

    def test_invariants_copied_verbatim(self, task_root, tmp_path):
        create_task(
            "demo", "obj", root=task_root,
            invariants=["Rule one is unique-xyz.", "Rule two."],
        )
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        text = (out / "SKILL.md").read_text()
        assert "Rule one is unique-xyz." in text
        assert "Rule two." in text

    def test_iteration_history_seeded(self, task_root, tmp_path):
        create_task("demo", "obj", root=task_root)
        advance_iteration("demo", result="pass", root=task_root)
        advance_iteration("demo", result="fail", root=task_root)
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        assert "2 (1 pass, 1 fail)" in (out / "SKILL.md").read_text()

    def test_refuses_overwrite_without_force(self, task_root, tmp_path):
        create_task("demo", "obj", root=task_root)
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        with pytest.raises(FileExistsError):
            scaffold_skill_from_task("demo", root=task_root, out_dir=out)

    def test_force_overwrites(self, task_root, tmp_path):
        create_task("demo", "obj", root=task_root)
        out = tmp_path / "s"
        scaffold_skill_from_task("demo", root=task_root, out_dir=out)
        scaffold_skill_from_task("demo", root=task_root, out_dir=out, force=True)
        assert (out / "SKILL.md").exists()

    def test_unknown_task_raises(self, task_root, tmp_path):
        with pytest.raises(FileNotFoundError):
            scaffold_skill_from_task("nope", root=task_root, out_dir=tmp_path / "s")

    def test_invalid_task_id_raises(self, task_root, tmp_path):
        with pytest.raises(ValueError):
            scaffold_skill_from_task("../evil", root=task_root, out_dir=tmp_path / "s")


def _put_archived(task_root, task_id, **fields):
    """Write an archived task's state.json directly under archive/."""
    d = task_root / "archive" / task_id
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "task_id": task_id,
        "objective": f"obj {task_id}",
        "status": "complete",
        **fields,
    }
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return d


class TestArchiveAware:
    def test_scaffolds_an_archived_task(self, task_root, tmp_path):
        _put_archived(task_root, "old-task", objective="Archived work")
        dest = scaffold_skill_from_task("old-task", root=task_root, out_dir=tmp_path / "s")
        assert "Archived work" in (dest / "SKILL.md").read_text()

    def test_resolves_collision_suffixed_archive(self, task_root, tmp_path):
        # GC names a collision archive <id>__<epoch>; resolution matches the
        # task_id field inside state.json.
        d = task_root / "archive" / "renamed__123456"
        d.mkdir(parents=True)
        (d / "state.json").write_text(
            json.dumps({"task_id": "renamed", "objective": "x", "status": "complete"})
        )
        dest = scaffold_skill_from_task("renamed", root=task_root, out_dir=tmp_path / "s")
        assert (dest / "SKILL.md").exists()


class TestListCandidates:
    def test_lists_live_completed_tasks(self, task_root):
        create_task("done1", "First done task", root=task_root)
        complete_task("done1", root=task_root)
        cands = list_skill_candidates(task_root)
        assert [c["task_id"] for c in cands] == ["done1"]
        assert cands[0]["location"] == "live"

    def test_excludes_running_tasks(self, task_root):
        create_task("running1", "Still going", root=task_root)
        assert list_skill_candidates(task_root) == []

    def test_includes_archived_tasks(self, task_root):
        create_task("live-done", "Live done", root=task_root)
        complete_task("live-done", root=task_root)
        _put_archived(task_root, "arch-done", objective="Archived done")
        cands = list_skill_candidates(task_root)
        assert {c["task_id"]: c["location"] for c in cands} == {
            "live-done": "live",
            "arch-done": "archived",
        }

    def test_search_filters(self, task_root):
        for tid, obj in [("a", "build a telegram bot"), ("b", "write documentation")]:
            create_task(tid, obj, root=task_root)
            complete_task(tid, root=task_root)
        cands = list_skill_candidates(task_root, search="telegram")
        assert [c["task_id"] for c in cands] == ["a"]

    def test_search_requires_all_terms(self, task_root):
        create_task("x", "build a telegram bot", root=task_root)
        complete_task("x", root=task_root)
        assert list_skill_candidates(task_root, search="telegram bot") != []
        assert list_skill_candidates(task_root, search="telegram widget") == []

    def test_empty_when_no_completed_tasks(self, task_root):
        assert list_skill_candidates(task_root) == []

    def test_newest_completion_first(self, task_root):
        _put_archived(task_root, "older", completed_at="2026-05-01T00:00:00+00:00")
        _put_archived(task_root, "newer", completed_at="2026-05-10T00:00:00+00:00")
        cands = list_skill_candidates(task_root)
        assert [c["task_id"] for c in cands] == ["newer", "older"]
