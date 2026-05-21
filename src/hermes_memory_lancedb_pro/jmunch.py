"""Detection of a jmunch gateway in the LLM request path.

jmunch (https://github.com/iamfoz/jmunch-mcp) is a gateway that sits
between an app and an upstream OpenAI- or Anthropic-compatible LLM API. It
reduces tokens by "handle-ifying" fat tool results — replacing a large
payload with a short summary plus an opaque handle. That lossily
compresses the agent's conversation history, so this plugin wants to know
when jmunch is in the path: it tunes recall/admission to compensate (see
`provider.py`) and asks the gateway to pass memory-extraction calls
through untouched.

Detection is two-stage, and creates no code dependency on jmunch-mcp:

  * Passive confirmation — a pass-through-capable jmunch gateway stamps
    every response with an `X-Jmunch-Gateway` header.
    `record_response_headers()`, called by the LLM client after each call,
    latches that observation the first time the header is seen. It is
    authoritative and works on any port — but only from the first response
    onward.
  * Startup declaration — because the passive signal isn't available
    before the first call, the operator can set `MEMORY_JMUNCH_MODE=true`
    to declare jmunch up front, so the startup-time tuning (admission
    preset, and recall from turn one) is correct immediately.

`is_jmunch_in_use()` is true when either signal has fired.

The gateway-side support this module relies on — stamping `X-Jmunch-Gateway`
and honouring the `X-Jmunch-Inject` / `X-Jmunch-Handleify` request headers —
currently lives in a fork of jmunch-mcp and is not yet part of an upstream
release. Detection therefore keys on the *presence* of the `X-Jmunch-Gateway`
header, not on any version string it might carry.
"""

from __future__ import annotations

import threading
from os import environ
from typing import Any

__all__ = [
    "JMUNCH_MODE_ENV",
    "is_jmunch_in_use",
    "jmunch_mode_configured",
    "jmunch_request_headers",
    "record_response_headers",
]

# Env var by which an operator declares, at startup, that jmunch is in the
# LLM path — see the module docstring.
JMUNCH_MODE_ENV = "MEMORY_JMUNCH_MODE"

# Response header a pass-through-capable jmunch gateway stamps on every
# response (lower-cased here; lookups are case-insensitive). Its presence is
# the detection signal — any value it carries is not interpreted.
_GATEWAY_HEADER = "x-jmunch-gateway"

# Request headers that make jmunch a pure pass-through for a call: no verb
# injection, no handle-ification, so the memory extractor sees the raw
# tool content. Both are inert on any non-jmunch endpoint, and a jmunch
# gateway that does not recognise them simply ignores them.
_PASSTHROUGH_HEADERS: dict[str, str] = {
    "X-Jmunch-Inject": "false",
    "X-Jmunch-Handleify": "false",
}

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

# Latched observation of a jmunch gateway seen on the wire. A mutable holder
# (rather than a rebound module scalar) so the latching helper needs no
# `global`. Written from the LLM-call thread, read from the recall path —
# guarded by `_lock`. `observed` is a monotonic latch, so the worst a race
# could do is delay an observation by one turn.
_lock = threading.Lock()
_state: dict[str, bool] = {"observed": False}


def jmunch_mode_configured() -> bool:
    """True when the operator declared jmunch via `MEMORY_JMUNCH_MODE`.
    This is the only signal available before the first LLM response."""
    return environ.get(JMUNCH_MODE_ENV, "").strip().lower() in _TRUE_TOKENS


def is_jmunch_in_use() -> bool:
    """True when jmunch is known to be in the LLM path — declared via
    `MEMORY_JMUNCH_MODE`, or confirmed by an `X-Jmunch-Gateway` response
    header seen on an earlier call."""
    if jmunch_mode_configured():
        return True
    with _lock:
        return _state["observed"]


def jmunch_request_headers() -> dict[str, str]:
    """Headers to attach to an LLM call when jmunch is in use: they tell
    the gateway to pass the request through verbatim — no verb injection,
    no handle-ification — so the memory extractor sees full-fidelity tool
    content. An empty dict when jmunch is not in use, so callers can splat
    the result unconditionally."""
    return dict(_PASSTHROUGH_HEADERS) if is_jmunch_in_use() else {}


def record_response_headers(headers: Any) -> None:
    """Inspect an LLM response's headers for `X-Jmunch-Gateway` and latch
    the observation. Call it after every LLM call with any headers mapping
    (a dict or an httpx.Headers); a no-op when the header is absent, so it
    is safe and transparent on non-jmunch endpoints. The header's presence
    is the signal — its value is not interpreted."""
    if _lookup_header(headers, _GATEWAY_HEADER) is None:
        return
    with _lock:
        _state["observed"] = True


def _lookup_header(headers: Any, name: str) -> Any:
    """Case-insensitive header lookup over a dict- or httpx.Headers-like
    mapping. Returns None when not found or `headers` is not a mapping."""
    if headers is None:
        return None
    try:
        items = list(headers.items())
    except (AttributeError, TypeError):
        return None
    lname = name.lower()
    for key, value in items:
        if str(key).lower() == lname:
            return value
    return None
