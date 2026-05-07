"""Reflection retry classification and transient-error handling.

Ported from CortexReach reflection-retry.ts.

Classifies LLM / upstream failures as:
- *transient* (network resets, gateway errors, timeouts) — eligible for one retry.
- *non-retry* (auth, quota, billing, policy, context-length) — hard failures that
  must not be retried.
"""

from __future__ import annotations

import json
import random as _random_module
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "RetryClassifierResult",
    "is_transient_reflection_upstream_error",
    "is_reflection_non_retry_error",
    "classify_reflection_retry",
    "compute_reflection_retry_delay_ms",
    "run_with_reflection_transient_retry_once",
]

# ---------------------------------------------------------------------------
# Pattern lists — ported verbatim from reflection-retry.ts
# ---------------------------------------------------------------------------

_TRANSIENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"unexpected eof", re.IGNORECASE),
    re.compile(r"\beconnreset\b", re.IGNORECASE),
    re.compile(r"\beconnaborted\b", re.IGNORECASE),
    re.compile(r"\betimedout\b", re.IGNORECASE),
    re.compile(r"\bepipe\b", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
    re.compile(r"socket hang up", re.IGNORECASE),
    re.compile(r"socket (?:closed|disconnected)", re.IGNORECASE),
    re.compile(r"connection (?:closed|aborted|dropped)", re.IGNORECASE),
    re.compile(r"early close", re.IGNORECASE),
    re.compile(r"stream (?:ended|closed) unexpectedly", re.IGNORECASE),
    re.compile(r"temporar(?:y|ily).*unavailable", re.IGNORECASE),
    re.compile(r"upstream.*unavailable", re.IGNORECASE),
    re.compile(r"service unavailable", re.IGNORECASE),
    re.compile(r"bad gateway", re.IGNORECASE),
    re.compile(r"gateway timeout", re.IGNORECASE),
    re.compile(r"\b(?:http|status)\s*(?:502|503|504)\b", re.IGNORECASE),
    re.compile(r"\btimed out\b", re.IGNORECASE),
    re.compile(r"\btimeout\b", re.IGNORECASE),
    re.compile(r"\bund_err_(?:socket|headers_timeout|body_timeout)\b", re.IGNORECASE),
    re.compile(r"network error", re.IGNORECASE),
    re.compile(r"fetch failed", re.IGNORECASE),
]

_NON_RETRY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b401\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"invalid api key", re.IGNORECASE),
    re.compile(r"invalid[_ -]?token", re.IGNORECASE),
    re.compile(r"\bauth(?:entication)?_?unavailable\b", re.IGNORECASE),
    re.compile(r"insufficient (?:credit|credits|balance)", re.IGNORECASE),
    re.compile(r"\bbilling\b", re.IGNORECASE),
    re.compile(r"\bquota exceeded\b", re.IGNORECASE),
    re.compile(r"payment required", re.IGNORECASE),
    re.compile(r"model .*not found", re.IGNORECASE),
    re.compile(r"no such model", re.IGNORECASE),
    re.compile(r"unknown model", re.IGNORECASE),
    re.compile(r"context length", re.IGNORECASE),
    re.compile(r"context window", re.IGNORECASE),
    re.compile(r"request too large", re.IGNORECASE),
    re.compile(r"payload too large", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit", re.IGNORECASE),
    re.compile(r"prompt too long", re.IGNORECASE),
    re.compile(r"session expired", re.IGNORECASE),
    re.compile(r"invalid session", re.IGNORECASE),
    re.compile(r"refusal", re.IGNORECASE),
    re.compile(r"content policy", re.IGNORECASE),
    re.compile(r"safety policy", re.IGNORECASE),
    re.compile(r"content filter", re.IGNORECASE),
    re.compile(r"disallowed", re.IGNORECASE),
]

_MAX_NORMALIZED_LEN = 260


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_error_message(error: Any) -> str:
    """Convert an arbitrary error value to a string, matching TS ``toErrorMessage``."""
    if isinstance(error, BaseException):
        msg = f"{type(error).__name__}: {error}".strip()
        return msg or "Error"
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error)
    except (TypeError, ValueError):
        return str(error)


def _clip_single_line(text: str, max_len: int = _MAX_NORMALIZED_LEN) -> str:
    """Collapse whitespace to a single space and truncate with '…' if needed."""
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetryClassifierResult:
    """Result of ``classify_reflection_retry``."""

    decision: Literal["retry", "noop"]
    reason: str
    normalized: str  # the normalized error message used for matching


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_transient_reflection_upstream_error(err: Any) -> bool:
    """Return True if *err* looks like a transient upstream failure."""
    msg = _to_error_message(err)
    return any(p.search(msg) for p in _TRANSIENT_PATTERNS)


def is_reflection_non_retry_error(err: Any) -> bool:
    """Return True if *err* should never be retried (auth, quota, policy, etc.)."""
    msg = _to_error_message(err)
    return any(p.search(msg) for p in _NON_RETRY_PATTERNS)


def classify_reflection_retry(
    error: Any,
    *,
    used_fallback: bool = False,
) -> RetryClassifierResult:
    """Classify whether *error* warrants a single retry attempt.

    Returns ``decision="retry"`` only when the error is transient AND not a
    hard non-retry error AND no fallback has already been used.  In all other
    cases returns ``decision="noop"`` with an explanatory ``reason``.
    """
    normalized = _clip_single_line(_to_error_message(error), _MAX_NORMALIZED_LEN)

    if is_reflection_non_retry_error(error):
        return RetryClassifierResult(
            decision="noop",
            reason="non_retry_error",
            normalized=normalized,
        )
    if used_fallback:
        return RetryClassifierResult(
            decision="noop",
            reason="fallback_already_used",
            normalized=normalized,
        )
    if is_transient_reflection_upstream_error(error):
        return RetryClassifierResult(
            decision="retry",
            reason="transient_upstream_failure",
            normalized=normalized,
        )
    return RetryClassifierResult(
        decision="noop",
        reason="non_transient_error",
        normalized=normalized,
    )


def compute_reflection_retry_delay_ms(
    random_fn: Callable[[], float] | None = None,
) -> int:
    """Return a jittered retry delay in milliseconds.

    ``1000 + floor(r * 2000)`` where ``r`` is drawn from ``random_fn``
    (default: ``random.random``).  Pass a deterministic function for tests.
    """
    r = random_fn() if random_fn is not None else _random_module.random()
    # Clamp to [0, 1] just as the TS implementation does.
    r = max(0.0, min(1.0, r)) if (r == r) else 0.0  # noqa: PLR0124 (NaN guard)
    return 1000 + int(r * 2000)


def run_with_reflection_transient_retry_once(
    fn: Callable[[], Any],
    *,
    used_fallback: bool = False,
    on_log: Callable[[str], None] | None = None,
    sleep_ms: Callable[[int], None] | None = None,
) -> Any:
    """Call *fn()*, retrying once on a transient upstream error.

    Parameters
    ----------
    fn:
        Zero-argument callable to invoke (and possibly retry).
    used_fallback:
        When True the classifier will not allow a retry even for transient
        errors (the fallback path has already been exhausted).
    on_log:
        Optional callable that receives informational / warning log strings.
    sleep_ms:
        Optional callable ``(ms: int) -> None`` used instead of
        ``time.sleep`` so tests can inject a no-op.
    """
    _log = on_log if on_log is not None else lambda _msg: None

    try:
        return fn()
    except Exception as exc:
        result = classify_reflection_retry(exc, used_fallback=used_fallback)
        if result.decision != "retry":
            raise

        delay = compute_reflection_retry_delay_ms()
        _log(
            f"reflection: transient upstream failure detected; retrying once in "
            f"{delay}ms ({result.reason}). error={result.normalized}"
        )

        if sleep_ms is not None:
            sleep_ms(delay)
        else:
            time.sleep(delay / 1000.0)

        try:
            value = fn()
            _log("reflection: retry succeeded")
            return value
        except Exception as retry_exc:
            _log(
                f"reflection: retry exhausted. "
                f"error={_clip_single_line(_to_error_message(retry_exc), _MAX_NORMALIZED_LEN)}"
            )
            raise
