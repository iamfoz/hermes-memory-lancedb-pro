"""Hermes plugin entry for hermes-memory-lancedb-pro (repo-root shim).

Install the whole repository as a Hermes plugin:

    hermes plugins install iamfoz/hermes-memory-lancedb-pro

Hermes' package manager (PM) then builds this repo's ``pyproject.toml`` into
its managed environment — the package *and* its dependencies (lancedb,
sentence-transformers, …) — and rebuilds them into every new environment
generation, so `hermes update` can no longer drop them. This file only
re-exports the provider entry points; the host's discovery text-scans it for
the ``register_memory_provider`` / ``MemoryProvider`` markers.

If the import below fails, the package is missing from Hermes' environment:
run ``hermes pm install`` (or reinstall the plugin) rather than pip.
"""
from hermes_memory_lancedb_pro.provider import register, register_memory_provider

__all__ = ["register", "register_memory_provider"]
