"""Tests for the automatic purge_archived feature.

_maybe_auto_purge is a module-level helper that can be tested directly
without needing hermes-agent installed.  All tests are pure-Python
(no LanceDB), using monkeypatching to control the cooldown check and
the purge call itself.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from hermes_memory_lancedb_pro.provider import _maybe_auto_purge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_store(db_path: str) -> MagicMock:
    """Return a MagicMock MemoryStore with a real db_path."""
    store = MagicMock()
    store.db_path = db_path
    store.purge_archived.return_value = 0
    return store


def _write_state(db_path: str, last_run_ms: int) -> None:
    """Write a purge state file recording last_run_ms."""
    with open(os.path.join(db_path, ".purge-state.json"), "w") as f:
        json.dump({"last_run_at": last_run_ms}, f)


# ---------------------------------------------------------------------------
# Cooldown / disabled
# ---------------------------------------------------------------------------

class TestAutoPurgeCooldown:
    def test_disabled_when_cooldown_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 0)
        store = _mock_store(str(tmp_path))
        _maybe_auto_purge(store)
        store.purge_archived.assert_not_called()

    def test_skipped_when_cooldown_not_elapsed(self, monkeypatch, tmp_path):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        # Write a state file timestamped "now" — cooldown hasn't elapsed
        import time
        _write_state(str(tmp_path), int(time.time() * 1000))
        store = _mock_store(str(tmp_path))
        _maybe_auto_purge(store)
        store.purge_archived.assert_not_called()

    def test_runs_when_state_file_missing(self, monkeypatch, tmp_path):
        """No state file → treat as never run → purge should execute."""
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        _maybe_auto_purge(store)
        store.purge_archived.assert_called_once_with(grace_period_days=30)

    def test_runs_when_cooldown_elapsed(self, monkeypatch, tmp_path):
        """State file older than cooldown → purge executes."""
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 1)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 7)
        # Timestamp 2 hours ago
        import time
        old_ms = int(time.time() * 1000) - 2 * 60 * 60 * 1000
        _write_state(str(tmp_path), old_ms)
        store = _mock_store(str(tmp_path))
        _maybe_auto_purge(store)
        store.purge_archived.assert_called_once_with(grace_period_days=7)


# ---------------------------------------------------------------------------
# State file written after successful run
# ---------------------------------------------------------------------------

class TestAutoPurgeStateFile:
    def test_state_file_written_after_purge(self, monkeypatch, tmp_path):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        store.purge_archived.return_value = 5

        _maybe_auto_purge(store)

        state_path = os.path.join(str(tmp_path), ".purge-state.json")
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        assert "last_run_at" in state
        import time
        # last_run_at should be within the last few seconds
        assert abs(int(time.time() * 1000) - state["last_run_at"]) < 5000

    def test_state_file_written_even_when_nothing_purged(self, monkeypatch, tmp_path):
        """State file updated even when purge_archived returns 0 (no rows removed).
        This prevents hammering a store with no archived rows every session."""
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        store.purge_archived.return_value = 0

        _maybe_auto_purge(store)

        assert os.path.exists(os.path.join(str(tmp_path), ".purge-state.json"))

    def test_state_file_not_written_on_purge_exception(self, monkeypatch, tmp_path):
        """If purge_archived raises, state file should NOT be updated so the
        next session retries."""
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        store.purge_archived.side_effect = RuntimeError("LanceDB locked")

        _maybe_auto_purge(store)  # must not raise

        assert not os.path.exists(os.path.join(str(tmp_path), ".purge-state.json"))


# ---------------------------------------------------------------------------
# Grace period forwarded correctly
# ---------------------------------------------------------------------------

class TestAutoPurgeGracePeriod:
    def test_grace_days_forwarded_to_purge(self, monkeypatch, tmp_path):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 90)
        store = _mock_store(str(tmp_path))
        _maybe_auto_purge(store)
        store.purge_archived.assert_called_once_with(grace_period_days=90)

    def test_default_grace_is_30(self, monkeypatch, tmp_path):
        import hermes_memory_lancedb_pro.provider as prov
        assert prov._AUTO_PURGE_GRACE_DAYS == 30

    def test_default_cooldown_is_24(self, monkeypatch, tmp_path):
        import hermes_memory_lancedb_pro.provider as prov
        assert prov._AUTO_PURGE_COOLDOWN_HOURS == 24


# ---------------------------------------------------------------------------
# Env var wiring (read at module import; test that the constants exist)
# ---------------------------------------------------------------------------

class TestAutoPurgeEnvVars:
    def test_cooldown_env_var_name(self):
        """MEMORY_AUTO_PURGE_COOLDOWN_HOURS controls the cooldown."""
        import hermes_memory_lancedb_pro.provider as prov
        # The constant is read at import; we just verify it exists and is an int.
        assert isinstance(prov._AUTO_PURGE_COOLDOWN_HOURS, int)

    def test_grace_env_var_name(self):
        """MEMORY_PURGE_GRACE_DAYS controls the grace period."""
        import hermes_memory_lancedb_pro.provider as prov
        assert isinstance(prov._AUTO_PURGE_GRACE_DAYS, int)


# ---------------------------------------------------------------------------
# Resilience — purge errors must not propagate to caller
# ---------------------------------------------------------------------------

class TestAutoPurgeResilience:
    def test_exception_in_purge_is_swallowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        store.purge_archived.side_effect = Exception("unexpected DB error")
        # Must not raise — shutdown() must always succeed
        _maybe_auto_purge(store)

    def test_exception_logged_as_warning(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_COOLDOWN_HOURS", 24)
        monkeypatch.setattr("hermes_memory_lancedb_pro.provider._AUTO_PURGE_GRACE_DAYS", 30)
        store = _mock_store(str(tmp_path))
        store.purge_archived.side_effect = Exception("disk full")
        import logging
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.provider"):
            _maybe_auto_purge(store)
        assert any("disk full" in r.message for r in caplog.records)
