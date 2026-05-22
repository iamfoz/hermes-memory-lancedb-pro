"""Unit tests for the open-file-limit raise. Pure-Python, no LanceDB.

`_raise_fd_limit` mutates the process's RLIMIT_NOFILE, so each test restores
the original limits in a finally block.
"""

from __future__ import annotations

import resource

from hermes_memory_lancedb_pro.store import MEMORY_FD_LIMIT, _raise_fd_limit


def _expected_target(hard: int) -> int:
    if hard == resource.RLIM_INFINITY:
        return MEMORY_FD_LIMIT
    return min(MEMORY_FD_LIMIT, hard)


class TestRaiseFdLimit:
    def test_raises_low_soft_limit(self):
        orig_soft, orig_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        low = 256 if orig_hard == resource.RLIM_INFINITY else min(256, orig_hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (low, orig_hard))
            _raise_fd_limit()
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            assert new_soft == _expected_target(orig_hard)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (orig_soft, orig_hard))

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setattr("hermes_memory_lancedb_pro.store.MEMORY_FD_LIMIT", 0)
        orig_soft, orig_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        low = 256 if orig_hard == resource.RLIM_INFINITY else min(256, orig_hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (low, orig_hard))
            _raise_fd_limit()
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            assert new_soft == low  # left untouched when disabled
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (orig_soft, orig_hard))

    def test_noop_when_already_sufficient(self):
        orig_soft, orig_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = _expected_target(orig_hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, orig_hard))
            _raise_fd_limit()  # soft already at target — must be a clean no-op
            new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            assert new_soft == target
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (orig_soft, orig_hard))

    def test_never_exceeds_hard_limit(self):
        orig_soft, orig_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if orig_hard == resource.RLIM_INFINITY:
            return  # no finite ceiling to check against
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, orig_hard), orig_hard))
            _raise_fd_limit()
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            assert new_soft <= new_hard
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (orig_soft, orig_hard))
