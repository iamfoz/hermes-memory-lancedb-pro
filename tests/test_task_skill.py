"""Unit tests for task_skill — scaffolding a skill from a task. No heavy deps."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.task_ledger import advance_iteration, create_task
from hermes_memory_lancedb_pro.task_skill import scaffold_skill_from_task


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
