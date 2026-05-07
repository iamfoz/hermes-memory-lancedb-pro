"""Reflection event-store payload builder.

Ported from CortexReach reflection-event-store.ts.
Builds a single per-run "reflection event" payload that records the
provenance of a reflection run: which session, agent, command, and when.
Pure data assembly — no I/O, no LLM.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

__all__ = [
    "ReflectionEventMetadata",
    "ReflectionEventPayload",
    "build_reflection_event_payload",
    "create_reflection_event_id",
]

REFLECTION_SCHEMA_VERSION: int = 4


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReflectionEventMetadata:
    type: str = "memory-reflection-event"
    reflection_version: int = REFLECTION_SCHEMA_VERSION
    stage: str = "reflect-store"
    event_id: str = ""
    session_key: str = ""
    session_id: str = ""
    agent_id: str = ""
    command: str = ""
    stored_at: int = 0
    used_fallback: bool = False
    error_signals: list[str] = field(default_factory=list)
    source_reflection_path: str | None = None


@dataclass
class ReflectionEventPayload:
    kind: str = "event"
    text: str = ""
    metadata: ReflectionEventMetadata = field(default_factory=ReflectionEventMetadata)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_reflection_event_id(
    *,
    run_at: int,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
) -> str:
    """Return a deterministic event ID of the form ``refl-YYYYMMDDHHMM-XXXXXXXX``.

    The 14-character date prefix is produced by the TS expression::

        new Date(safeRunAt).toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)

    For a timestamp like ``2024-01-23T14:05:00.000Z`` this strips all
    ``-``, ``:`` ``.`` ``T`` and ``Z`` characters to get
    ``20240123140500000`` then slices the first 14 chars →
    ``20240123140500``.  The digest is the first 8 hex chars of SHA-1 over
    the string ``runAt|sessionKey|sessionId|agentId|command``.
    """
    # NaN guard: only treat run_at as valid when it's truthy and not NaN
    safe_run_at = max(0, int(run_at)) if run_at and not math.isnan(float(run_at)) else 0
    dt = datetime.fromtimestamp(safe_run_at / 1000.0, tz=UTC)
    iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    # strip all -, :, ., T, Z — mirrors TS /[-:.TZ]/g replace
    cleaned = iso.replace("-", "").replace(":", "").replace(".", "").replace("T", "").replace("Z", "")
    date_part = cleaned[:14]

    raw = f"{safe_run_at}|{session_key}|{session_id}|{agent_id}|{command}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"refl-{date_part}-{digest}"


def build_reflection_event_payload(
    *,
    scope: str,
    session_key: str,
    session_id: str,
    agent_id: str,
    command: str,
    run_at: int,
    used_fallback: bool = False,
    tool_error_signals: list[dict] | None = None,
    event_id: str | None = None,
    source_reflection_path: str | None = None,
) -> ReflectionEventPayload:
    """Build a :class:`ReflectionEventPayload` for a single reflection run.

    Parameters mirror the TS ``BuildReflectionEventPayloadParams`` interface.
    ``tool_error_signals`` is a list of dicts with a ``signature_hash`` (or
    ``signatureHash``) key.
    """
    if tool_error_signals is None:
        tool_error_signals = []

    resolved_event_id = event_id or create_reflection_event_id(
        run_at=run_at,
        session_key=session_key,
        session_id=session_id,
        agent_id=agent_id,
        command=command,
    )

    # Extract signature hashes — accept both snake_case and camelCase keys
    error_signals = [
        sig.get("signature_hash") or sig.get("signatureHash", "")
        for sig in tool_error_signals
    ]

    metadata = ReflectionEventMetadata(
        type="memory-reflection-event",
        reflection_version=REFLECTION_SCHEMA_VERSION,
        stage="reflect-store",
        event_id=resolved_event_id,
        session_key=session_key,
        session_id=session_id,
        agent_id=agent_id,
        command=command,
        stored_at=run_at,
        used_fallback=used_fallback,
        error_signals=error_signals,
        source_reflection_path=source_reflection_path,
    )

    # Mirrors TS text join:
    #   `reflection-event · ${scope}`
    #   `eventId=${eventId}`
    #   `session=${sessionId}`
    #   `agent=${agentId}`
    #   `command=${command}`
    #   `usedFallback=${usedFallback ? "true" : "false"}`
    text = "\n".join([
        f"reflection-event · {scope}",
        f"eventId={resolved_event_id}",
        f"session={session_id}",
        f"agent={agent_id}",
        f"command={command}",
        f"usedFallback={'true' if used_fallback else 'false'}",
    ])

    return ReflectionEventPayload(
        kind="event",
        text=text,
        metadata=metadata,
    )
