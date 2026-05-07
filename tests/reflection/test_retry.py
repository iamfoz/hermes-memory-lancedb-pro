"""Tests for hermes_memory_lancedb_pro.reflection.retry."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.reflection.retry import (
    RetryClassifierResult,
    classify_reflection_retry,
    compute_reflection_retry_delay_ms,
    is_reflection_non_retry_error,
    is_transient_reflection_upstream_error,
    run_with_reflection_transient_retry_once,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _err(msg: str) -> Exception:
    return Exception(msg)


# ---------------------------------------------------------------------------
# is_transient_reflection_upstream_error
# ---------------------------------------------------------------------------

TRANSIENT_CASES = [
    "unexpected EOF from server",
    "ECONNRESET during request",
    "econnaborted",
    "ETIMEDOUT",
    "write EPIPE",
    "Connection reset by peer",
    "socket hang up",
    "socket closed unexpectedly",
    "socket disconnected",
    "connection closed",
    "connection aborted",
    "connection dropped",
    "early close detected",
    "stream ended unexpectedly",
    "stream closed unexpectedly",
    "temporarily unavailable",
    "temporarily unavailable right now",
    "upstream service unavailable",
    "service unavailable",
    "bad gateway response",
    "gateway timeout",
    "http 502",
    "status 503",
    "HTTP 504",
    "request timed out",
    "timeout while waiting",
    "UND_ERR_SOCKET",
    "UND_ERR_HEADERS_TIMEOUT",
    "UND_ERR_BODY_TIMEOUT",
    "network error occurred",
    "fetch failed",
]


@pytest.mark.parametrize("msg", TRANSIENT_CASES)
def test_transient_pattern_matches(msg):
    assert is_transient_reflection_upstream_error(_err(msg)) is True


def test_non_transient_message_returns_false():
    assert is_transient_reflection_upstream_error(_err("model not found")) is False


# ---------------------------------------------------------------------------
# is_reflection_non_retry_error
# ---------------------------------------------------------------------------

NON_RETRY_CASES = [
    "status 401 unauthorized",
    "Unauthorized",
    "invalid api key provided",
    "invalid_token supplied",
    "invalid-token in header",
    "invalid token here",
    "authentication_unavailable",
    "auth_unavailable now",
    "insufficient credit",
    "insufficient credits",
    "insufficient balance",
    "billing issue detected",
    "quota exceeded",
    "payment required",
    "model gpt-5 not found",
    "no such model exists",
    "unknown model",
    "context length exceeded",
    "context window too large",
    "request too large",
    "payload too large",
    "too many tokens",
    "token limit reached",
    "prompt too long",
    "session expired",
    "invalid session",
    "refusal detected",
    "content policy violation",
    "safety policy triggered",
    "content filter blocked",
    "disallowed content",
]


@pytest.mark.parametrize("msg", NON_RETRY_CASES)
def test_non_retry_pattern_matches(msg):
    assert is_reflection_non_retry_error(_err(msg)) is True


def test_non_non_retry_message_returns_false():
    assert is_reflection_non_retry_error(_err("socket hang up")) is False


# ---------------------------------------------------------------------------
# classify_reflection_retry
# ---------------------------------------------------------------------------

class TestClassifyReflectionRetry:
    def test_transient_returns_retry(self):
        result = classify_reflection_retry(_err("socket hang up"))
        assert result.decision == "retry"
        assert result.reason == "transient_upstream_failure"

    def test_non_retry_error_returns_noop(self):
        result = classify_reflection_retry(_err("401 Unauthorized"))
        assert result.decision == "noop"
        assert result.reason == "non_retry_error"

    def test_transient_plus_non_retry_non_retry_wins(self):
        # A message matching both transient and non-retry → non-retry wins.
        result = classify_reflection_retry(_err("quota exceeded and service unavailable"))
        assert result.decision == "noop"
        assert result.reason == "non_retry_error"

    def test_used_fallback_true_returns_noop(self):
        result = classify_reflection_retry(_err("socket hang up"), used_fallback=True)
        assert result.decision == "noop"
        assert result.reason == "fallback_already_used"

    def test_non_transient_non_non_retry_returns_noop(self):
        result = classify_reflection_retry(_err("some unknown failure"))
        assert result.decision == "noop"
        assert result.reason == "non_transient_error"

    def test_normalized_field_present(self):
        result = classify_reflection_retry(_err("socket hang up"))
        assert "socket hang up" in result.normalized

    def test_normalized_truncated_at_260(self):
        long_msg = "ECONNRESET " + "x" * 300
        result = classify_reflection_retry(_err(long_msg))
        assert len(result.normalized) <= 263  # 260 + "..."

    def test_result_is_dataclass(self):
        result = classify_reflection_retry(_err("timeout"))
        assert isinstance(result, RetryClassifierResult)


# ---------------------------------------------------------------------------
# compute_reflection_retry_delay_ms
# ---------------------------------------------------------------------------

class TestComputeReflectionRetryDelayMs:
    def test_range_1000_to_3000(self):
        for _ in range(50):
            delay = compute_reflection_retry_delay_ms()
            assert 1000 <= delay <= 3000

    def test_deterministic_half_gives_2000(self):
        delay = compute_reflection_retry_delay_ms(random_fn=lambda: 0.5)
        assert delay == 2000

    def test_deterministic_zero_gives_1000(self):
        delay = compute_reflection_retry_delay_ms(random_fn=lambda: 0.0)
        assert delay == 1000

    def test_deterministic_one_gives_3000(self):
        delay = compute_reflection_retry_delay_ms(random_fn=lambda: 1.0)
        assert delay == 3000

    def test_returns_int(self):
        assert isinstance(compute_reflection_retry_delay_ms(random_fn=lambda: 0.5), int)


# ---------------------------------------------------------------------------
# run_with_reflection_transient_retry_once
# ---------------------------------------------------------------------------

class TestRunWithReflectionTransientRetryOnce:
    def test_success_on_first_call(self):
        result = run_with_reflection_transient_retry_once(
            lambda: 42,
            sleep_ms=lambda _ms: None,
        )
        assert result == 42

    def test_retries_once_on_transient_error(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise Exception("socket hang up")
            return "ok"

        result = run_with_reflection_transient_retry_once(fn, sleep_ms=lambda _ms: None)
        assert result == "ok"
        assert len(calls) == 2

    def test_reraises_on_non_retry_error(self):
        def fn():
            raise Exception("401 Unauthorized")

        with pytest.raises(Exception, match="401 Unauthorized"):
            run_with_reflection_transient_retry_once(fn, sleep_ms=lambda _ms: None)

    def test_reraises_after_two_transient_failures(self):
        """Second call also fails — must propagate that error."""
        def fn():
            raise Exception("ECONNRESET")

        with pytest.raises(Exception, match="ECONNRESET"):
            run_with_reflection_transient_retry_once(fn, sleep_ms=lambda _ms: None)

    def test_no_retry_when_used_fallback(self):
        calls = []

        def fn():
            calls.append(1)
            raise Exception("socket hang up")

        with pytest.raises(Exception, match="socket hang up"):
            run_with_reflection_transient_retry_once(
                fn,
                used_fallback=True,
                sleep_ms=lambda _ms: None,
            )
        # Must not have retried — only one call.
        assert len(calls) == 1

    def test_on_log_called_on_retry(self):
        logs = []

        def fn():
            if not logs:
                raise Exception("timeout")
            return "done"

        run_with_reflection_transient_retry_once(
            fn,
            on_log=logs.append,
            sleep_ms=lambda _ms: None,
        )
        assert any("transient" in msg.lower() for msg in logs)

    def test_sleep_ms_called_with_positive_delay(self):
        delays = []

        def fn():
            if not delays:
                raise Exception("fetch failed")
            return "done"

        run_with_reflection_transient_retry_once(
            fn,
            sleep_ms=delays.append,
        )
        assert len(delays) == 1
        assert delays[0] >= 1000
