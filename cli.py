"""Hermes plugin CLI shim (repo-root install).

Exposes register_cli() so hermes-agent wires the lancedb_pro commands
(init, doctor, export, import, reset, stats, search, purge, compact, task)
into `hermes lancedb_pro`.
"""
from hermes_memory_lancedb_pro._cli import register_cli

__all__ = ["register_cli"]
