"""The agent skills shipped with the plugin.

They are the agent's only description of how to use this memory system, so
they must be registered with the host, packaged, and must never name a
command or flag that does not exist.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

import pytest

from hermes_memory_lancedb_pro import _cli, provider

REPO = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO / "src" / "hermes_memory_lancedb_pro" / "skills"
EXPECTED = {"durable-task", "lancedb-memory", "task-to-skill"}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {opt for action in parser._actions for opt in action.option_strings}


@pytest.fixture(scope="module")
def in_host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
    _cli.register_cli(parser)
    return parser


def test_three_skills_are_bundled_with_names_and_descriptions():
    skills = {s["name"]: s for s in provider.bundled_skills()}
    assert set(skills) == EXPECTED
    for name, skill in skills.items():
        assert Path(skill["path"]).is_file()
        assert len(skill["description"]) > 40, name
        # The description is what the agent sees when choosing a skill.
        assert not skill["description"].startswith("**")


def test_skill_names_do_not_collide_with_a_builtin_hermes_command():
    # `/memory` is a built-in Hermes command.
    assert "memory" not in {s["name"] for s in provider.bundled_skills()}


def test_each_skill_has_an_agents_summary():
    for name in EXPECTED:
        assert (SKILLS_DIR / name / "AGENTS.md").is_file()


def test_skills_are_packaged():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = data["tool"]["setuptools"]["package-data"]["hermes_memory_lancedb_pro"]
    assert "skills/*/*.md" in globs


def test_register_hands_every_skill_to_the_host(monkeypatch):
    class FakeBase:
        pass

    class Ctx:
        def __init__(self):
            self.skills: list[tuple] = []
            self.provider = None

        def register_memory_provider(self, p):
            self.provider = p

        def register_skill(self, name, path, description="", frontmatter=None):
            self.skills.append((name, path, description, frontmatter))

    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: FakeBase)
    monkeypatch.setattr(provider, "LanceDBProMemoryProvider", object)
    ctx = Ctx()
    provider.register(ctx)
    assert ctx.provider is not None
    assert {name for name, *_ in ctx.skills} == EXPECTED
    for name, path, description, frontmatter in ctx.skills:
        assert isinstance(path, Path) and path.name == "SKILL.md" and path.is_file()
        assert description and frontmatter["name"] == name


def test_a_host_without_skill_support_or_a_failing_one_does_not_break_register(monkeypatch):
    class FakeBase:
        pass

    class OldCtx:
        provider = None

        def register_memory_provider(self, p):
            self.provider = p

    class BrokenCtx(OldCtx):
        def register_skill(self, *a, **k):
            raise ValueError("refused")

    monkeypatch.setattr(provider, "_load_memory_provider_base", lambda: FakeBase)
    monkeypatch.setattr(provider, "LanceDBProMemoryProvider", object)
    for ctx in (OldCtx(), BrokenCtx()):
        provider.register(ctx)
        assert ctx.provider is not None


def test_front_matter_parses_with_the_hosts_own_parser():
    skill_utils = pytest.importorskip("agent.skill_utils")
    for name in EXPECTED:
        frontmatter, body = skill_utils.parse_frontmatter((SKILLS_DIR / name / "SKILL.md").read_text())
        assert frontmatter["name"] == name and frontmatter["description"]
        assert body.lstrip().startswith("# Skill:")


_COMMAND = re.compile(r"hermes lancedb_pro ((?:[a-z][a-z-]*)(?: [a-z][a-z-]*)?)([^\n`|]*)")


@pytest.mark.parametrize("doc", sorted(SKILLS_DIR.glob("*/*.md")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_command_and_flag_a_skill_names_exists(doc, in_host_parser):
    top = _subparsers(in_host_parser)
    text = doc.read_text().replace("\\\n", " ")
    checked = 0
    for match in _COMMAND.finditer(text):
        words = match.group(1).split()
        if words[0] not in top:
            if words[0] in ("task", "search", "stats", "doctor", "compact"):
                pytest.fail(f"{doc}: unknown command {words[0]!r}")
            continue  # prose such as "hermes lancedb_pro <command>"
        parser = top[words[0]]
        if words[0] == "task":
            task_cmds = _subparsers(parser)
            if len(words) < 2 or words[1] not in task_cmds:
                continue  # "task …" in prose
            parser = task_cmds[words[1]]
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*|-[a-z])(?![\w-])", match.group(2)):
            assert flag in _flags(parser), f"{doc}: `{match.group(0).strip()}` uses unknown flag {flag}"
        checked += 1
    assert checked, f"{doc} names no commands"


def test_the_memory_skill_marks_destructive_commands_as_the_operators(in_host_parser):
    text = (SKILLS_DIR / "lancedb-memory" / "SKILL.md").read_text()
    operators = text.split("## Commands that are the operator's, not yours")[1].split("\n## ")[0]
    for command in ("reset", "init", "purge", "import", "reembed", "export"):
        assert f"`{command}`" in operators
    # Every top-level command is either taught or fenced off.
    for command in _subparsers(in_host_parser):
        assert f"`{command}" in text or f"lancedb_pro {command}" in text, command
    assert "compact -y" in text  # the agent cannot answer a confirmation prompt


def test_cli_messages_recommend_the_in_host_command():
    """Advice printed by the CLI must name a command that works inside
    Hermes. The `hermes-memory` / `hermes-memory-lancedb-pro` console scripts
    are not on the agent's PATH; only `install-plugin` / `uninstall-plugin`,
    which exist solely in the standalone script, may be named that way."""
    source = (REPO / "src" / "hermes_memory_lancedb_pro" / "_cli.py").read_text()
    stale = [
        m.group(0)
        for m in re.finditer(r"hermes-memory(?:-lancedb-pro)? ([a-z][a-z-]*)", source)
        if m.group(1) not in ("install-plugin", "uninstall-plugin")
    ]
    assert stale == []
