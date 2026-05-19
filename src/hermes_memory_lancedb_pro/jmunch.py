"""Best-effort detection of a local jmunch proxy.

jmunch is a local OpenAI-compatible LLM proxy that fronts small-context
local models (Qwen, Ollama, LM Studio, OpenRouter, ...). When the plugin
is talking to one of those models the usable context window is a fraction
of a cloud model's, so callers want to know — in order to budget the
prompts they build and the memory context they ask the host agent to
inject.

Detection is deliberately a *soft*, observational check: it pattern-matches
the configured endpoint URL and creates no code dependency on jmunch-mcp.
hermes-memory and jmunch-mcp remain completely independent packages (see
the README) — this module never imports, calls, or inspects jmunch itself.

Limitations: the plugin can only see the LLM endpoints it is configured
with (`MEMORY_EXTRACTION_BASE_URL` and the OpenAI-SDK base-URL env vars).
If the host agent routes through jmunch but the plugin's extractor is
pointed elsewhere — or extraction is disabled entirely — detection cannot
see it. Treat a True result as a strong hint, not a guarantee.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

__all__ = [
    "JMUNCH_PORT_BASE",
    "JMUNCH_PORT_SPAN",
    "detected_jmunch_endpoint",
    "is_jmunch_in_use",
    "is_jmunch_url",
]

# jmunch binds the loopback interface. Each proxy instance claims the next
# free port starting at JMUNCH_PORT_BASE, so a host running several proxies
# exposes a contiguous range (7881, 7882, 7883, ...). Both knobs are
# env-overridable for non-default deployments.
JMUNCH_PORT_BASE: int = int(os.environ.get("JMUNCH_PORT_BASE", "7881"))
# How many ports above the base count as jmunch. Bounded on purpose: an
# unbounded range would misdetect any unrelated high-port localhost service.
JMUNCH_PORT_SPAN: int = int(os.environ.get("JMUNCH_PORT_SPAN", "16"))

# Hostnames that mean "this machine". jmunch only ever binds loopback.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Env vars that can point an OpenAI-compatible client at a base URL. The
# plugin's own extractor var comes first; the rest are what the host agent
# (and the OpenAI SDK) commonly set.
_CANDIDATE_URL_ENV_VARS = (
    "MEMORY_EXTRACTION_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)


def is_jmunch_url(url: str | None) -> bool:
    """True when `url` points at what looks like a local jmunch proxy: a
    loopback host on a port in the jmunch range ``[BASE, BASE + SPAN)``.

    Accepts URLs with or without a scheme (``127.0.0.1:7881`` works as well
    as ``http://127.0.0.1:7881/v1``). Returns False for anything it can't
    confidently classify rather than raising."""
    if not url:
        return False
    raw = url.strip()
    if not raw:
        return False
    # urlparse only populates hostname/port when a scheme is present.
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        # Malformed authority (e.g. a non-numeric port) — not classifiable.
        return False
    if host not in _LOOPBACK_HOSTS or port is None:
        return False
    return JMUNCH_PORT_BASE <= port < JMUNCH_PORT_BASE + JMUNCH_PORT_SPAN


def detected_jmunch_endpoint() -> str | None:
    """Return the first configured LLM base URL that looks like a local
    jmunch proxy, or None when none do.

    Checks the plugin's own extraction endpoint first, then the OpenAI-SDK
    base-URL env vars the host agent commonly sets — see
    `_CANDIDATE_URL_ENV_VARS`."""
    for var in _CANDIDATE_URL_ENV_VARS:
        url = os.environ.get(var)
        if is_jmunch_url(url):
            return url
    return None


def is_jmunch_in_use() -> bool:
    """Best-effort: True when a configured LLM endpoint is a local jmunch
    proxy. A jmunch extraction endpoint strongly implies the host agent is
    on the same small-context proxy. See the module docstring for the
    detection's limits."""
    return detected_jmunch_endpoint() is not None
