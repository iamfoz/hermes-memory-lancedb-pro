"""Tolerant parsing of numeric environment variables.

Settings are read at import time. A typo such as ``MEMORY_PREFETCH_LIMIT=five``
used to raise ``ValueError`` during import and take the whole plugin (and its
CLI) down; these helpers log a warning and fall back to the default instead.
"""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    """``int(os.environ[name])``, or *default* when unset, blank or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using the default %r", name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    """``float(os.environ[name])``, or *default* when unset, blank or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value):
        logger.warning("%s=%r is not a finite number; using the default %r", name, raw, default)
        return default
    return value
