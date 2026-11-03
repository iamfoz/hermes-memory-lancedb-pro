"""The repo is itself a Hermes plugin (`hermes plugins install
iamfoz/hermes-memory-lancedb-pro`). Guard the pieces the host and its
package manager (PM) depend on, without needing hermes-agent installed."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGED_MANIFEST = ROOT / "src" / "hermes_memory_lancedb_pro" / "plugin.yaml"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_root_manifest_matches_packaged_manifest():
    # Two install paths ship the same manifest: git install (root) and the
    # legacy shim (copied from the wheel). They must never drift.
    assert (ROOT / "plugin.yaml").read_bytes() == PACKAGED_MANIFEST.read_bytes()


def test_root_manifest_names_the_memory_provider_dir():
    # `hermes plugins install` names the plugin directory from manifest `name`;
    # it must equal memory.provider.
    lines = (ROOT / "plugin.yaml").read_text(encoding="utf-8").splitlines()
    assert "name: lancedb_pro" in lines
    assert not any(ln.lstrip().startswith("kind:") for ln in lines)


def test_root_init_carries_auto_detection_markers():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    # hermes-agent text-scans the first 8 KB for either marker.
    head = source[:8192]
    assert "register_memory_provider" in head or "MemoryProvider" in head
    assert "from hermes_memory_lancedb_pro.provider import register, register_memory_provider" in source


def test_root_init_exposes_register_entry_points():
    spec = importlib.util.spec_from_file_location("_lancedb_pro_root_shim", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.register)
    assert callable(module.register_memory_provider)


def test_root_cli_shim_exposes_register_cli():
    spec = importlib.util.spec_from_file_location("_lancedb_pro_root_cli", ROOT / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.register_cli)


def test_pyproject_is_a_pm_installable_project():
    # PM installs a plugin dir as a project only when its pyproject has a
    # [project] table and a build backend (pm.plugin_declarations).
    data = _pyproject()
    assert data["project"]["name"] == "hermes-memory-lancedb-pro"
    assert "build-backend" in data["build-system"]
    assert data["project"]["dependencies"], "PM must see the runtime deps"


def test_entry_point_uses_memory_provider_group_not_general_plugins():
    groups = _pyproject()["project"]["entry-points"]
    assert groups["hermes_agent.memory_providers"]["lancedb_pro"] == (
        "hermes_memory_lancedb_pro.provider:register"
    )
    # The general group is imported eagerly by the general plugin manager.
    assert "hermes_agent.plugins" not in groups


def test_no_stray_root_python_modules():
    # The host loader imports EVERY top-level *.py in the plugin dir as a
    # submodule; only the two shims may live there.
    assert sorted(p.name for p in ROOT.glob("*.py")) == ["__init__.py", "cli.py"]
