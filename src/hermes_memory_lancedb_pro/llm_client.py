"""LLM client for memory extraction — OpenAI-compatible and Anthropic adapters.

Ported from CortexReach ``src/llm-client.ts`` (424 lines).  Adds an Anthropic
adapter and env-var auto-detection that were not present in the TS source.

Usage::

    from hermes_memory_lancedb_pro.llm_client import create_llm_client_from_env

    client = create_llm_client_from_env()
    if client:
        result = client.complete_json("Extract facts as JSON …")
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "LlmClient",
    "OpenAICompatibleLlmClient",
    "AnthropicLlmClient",
    "create_llm_client_from_env",
    "extract_json_from_response",
    "repair_common_json",
    # backward-compat alias
    "ExtractorLLM",
]

_SYSTEM_PROMPT = "You are a memory extraction assistant. Always respond with valid JSON only."


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class LlmClient(Protocol):
    """Public contract satisfied by every concrete adapter in this module."""

    def complete_json(
        self,
        prompt: str,
        *,
        label: str | None = None,
    ) -> dict[str, Any] | None: ...

    def get_last_error(self) -> str | None: ...


# backward-compat alias — admission_control.py imports ExtractorLLM which only
# requires complete_json; LlmClient is a strict superset so it satisfies that
# Protocol too.
ExtractorLLM = LlmClient


# ---------------------------------------------------------------------------
# JSON utilities (ported verbatim from TS extractJsonFromResponse /
# repairCommonJson)
# ---------------------------------------------------------------------------

def _next_non_whitespace(text: str, start: int) -> str | None:
    """Return the first non-whitespace character at or after *start*, or None."""
    for i in range(start, len(text)):
        ch = text[i]
        if not ch.isspace():
            return ch
    return None


def _extract_json_from_response(text: str) -> str | None:
    """Extract a JSON object from *text*.

    Try ``\\`\\`\\`json … \\`\\`\\``` fences first; fall back to brace-matching from
    the first ``{``.  Returns ``None`` when no balanced JSON object is found.
    """
    # Try markdown fence first (with or without "json" language tag)
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    first_brace = text.find("{")
    if first_brace == -1:
        return None

    depth = 0
    last_brace = -1
    for i in range(first_brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_brace = i
                break

    if last_brace == -1:
        return None
    return text[first_brace : last_brace + 1]


def _repair_common_json(text: str) -> str:
    """Best-effort repair for common LLM JSON issues:

    - unescaped quotes inside string values
    - raw newlines / tabs inside strings
    - trailing commas before ``}`` or ``]``

    Ported line-for-line from the 60-line TS implementation in
    ``repairCommonJson``.
    """
    result = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            result.append(ch)
            escaped = False
            continue

        if in_string:
            if ch == "\\":
                result.append(ch)
                escaped = True
                continue

            if ch == '"':
                next_ch = _next_non_whitespace(text, i + 1)
                if next_ch is None or next_ch in (",", "}", "]", ":"):
                    result.append(ch)
                    in_string = False
                else:
                    result.append('\\"')
                continue

            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                result.append("\\r")
                continue
            if ch == "\t":
                result.append("\\t")
                continue

            result.append(ch)
            continue

        if ch == '"':
            result.append(ch)
            in_string = True
            continue

        if ch == ",":
            next_ch = _next_non_whitespace(text, i + 1)
            if next_ch in ("}", "]"):
                continue  # drop trailing comma

        result.append(ch)

    return "".join(result)


def _preview_text(value: str, max_len: int = 200) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _parse_json_with_repair(
    raw: str,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run extract → parse → repair-and-parse pipeline.

    Returns ``(parsed_dict, error_message)``.  On success, ``error_message``
    is ``None``; on failure, ``parsed_dict`` is ``None``.
    """
    json_str = _extract_json_from_response(raw)
    if not json_str:
        err = (
            f"hermes-memory-lancedb-pro: llm-client [{label}] "
            f"no JSON object found (chars={len(raw)}, "
            f"preview={_preview_text(raw)!r})"
        )
        logger.warning("llm_client [%s]: no JSON object found: %s", label, _preview_text(raw, 200))
        return None, err

    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as first_err:
        repaired = _repair_common_json(json_str)
        if repaired != json_str:
            try:
                parsed = json.loads(repaired)
                logger.warning(
                    "llm_client [%s]: recovered malformed JSON via repair (chars=%d)",
                    label,
                    len(json_str),
                )
                return parsed, None
            except json.JSONDecodeError as repair_err:
                err = (
                    f"hermes-memory-lancedb-pro: llm-client [{label}] "
                    f"JSON.parse failed: {first_err}; repair failed: {repair_err} "
                    f"(chars={len(json_str)}, preview={_preview_text(json_str)!r})"
                )
                logger.warning("llm_client [%s]: repair failed: %s", label, _preview_text(json_str, 200))
                return None, err
        err = (
            f"hermes-memory-lancedb-pro: llm-client [{label}] "
            f"JSON.parse failed: {first_err} "
            f"(chars={len(json_str)}, preview={_preview_text(json_str)!r})"
        )
        logger.warning("llm_client [%s]: JSON.parse failed: %s", label, _preview_text(json_str, 200))
        return None, err


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
# ---------------------------------------------------------------------------

class OpenAICompatibleLlmClient:
    """LLM adapter that speaks the OpenAI Chat Completions API.

    Works with any OpenAI-compatible endpoint (OpenAI, Azure, Ollama, LM
    Studio, vLLM, …).  The ``openai`` package is imported lazily so that the
    module can be imported even when the SDK is not installed — the
    ``ImportError`` is raised only when this class is *instantiated*.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30,
    ) -> None:
        try:
            import openai as _openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use OpenAICompatibleLlmClient. "
                "Install it with: pip install openai"
            ) from exc

        self._client = _openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.base_url = base_url
        self._last_error: str | None = None

    def complete_json(
        self,
        prompt: str,
        *,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Send *prompt* and return a parsed JSON dict, or ``None`` on failure."""
        effective_label = label or "generic"
        self._last_error = None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            self._last_error = (
                f"hermes-memory-lancedb-pro: llm-client [{effective_label}] "
                f"request failed for model {self.model}: {exc}"
            )
            logger.warning("llm_client [%s]: request failed: %s", effective_label, exc)
            return None

        if not raw:
            self._last_error = (
                f"hermes-memory-lancedb-pro: llm-client [{effective_label}] "
                f"empty response content from model {self.model}"
            )
            logger.warning("llm_client [%s]: empty response", effective_label)
            return None

        parsed, err = _parse_json_with_repair(raw, effective_label)
        if err:
            self._last_error = err
        return parsed

    def get_last_error(self) -> str | None:
        """Return the error message from the most recent failed call, or ``None``."""
        return self._last_error


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicLlmClient:
    """LLM adapter for the Anthropic Messages API.

    The ``anthropic`` package is imported lazily so the module can be imported
    without the SDK installed — the ``ImportError`` is raised only on
    instantiation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic as _anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicLlmClient. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = _anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._max_tokens = max_tokens
        self._last_error: str | None = None

    def complete_json(
        self,
        prompt: str,
        *,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Send *prompt* and return a parsed JSON dict, or ``None`` on failure."""
        effective_label = label or "generic"
        self._last_error = None
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = response.content[0].text
        except Exception as exc:  # noqa: BLE001
            self._last_error = (
                f"hermes-memory-lancedb-pro: llm-client [{effective_label}] "
                f"request failed for model {self.model}: {exc}"
            )
            logger.warning("llm_client [%s]: request failed: %s", effective_label, exc)
            return None

        if not raw:
            self._last_error = (
                f"hermes-memory-lancedb-pro: llm-client [{effective_label}] "
                f"empty response content from model {self.model}"
            )
            logger.warning("llm_client [%s]: empty response", effective_label)
            return None

        parsed, err = _parse_json_with_repair(raw, effective_label)
        if err:
            self._last_error = err
        return parsed

    def get_last_error(self) -> str | None:
        """Return the error message from the most recent failed call, or ``None``."""
        return self._last_error


# ---------------------------------------------------------------------------
# Env-var auto-detection factory
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def create_llm_client_from_env() -> LlmClient | None:
    """Detect an LLM provider from environment variables and return a client.

    Priority order:

    1. ``MEMORY_EXTRACTION_API_KEY`` + ``MEMORY_EXTRACTION_BASE_URL`` +
       ``MEMORY_EXTRACTION_MODEL`` → :class:`OpenAICompatibleLlmClient`
       (dedicated cheap-extractor override).
    2. ``MEMORY_EXTRACTION_PROVIDER=anthropic`` + ``MEMORY_EXTRACTION_API_KEY``
       → :class:`AnthropicLlmClient`.
    3. ``OPENAI_API_KEY`` → :class:`OpenAICompatibleLlmClient` with
       ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``) and
       ``OPENAI_MODEL`` (default ``gpt-4o-mini``).
    4. ``ANTHROPIC_API_KEY`` → :class:`AnthropicLlmClient` with default model.
    5. Returns ``None`` if no env vars are set or the chosen SDK is not
       installed (``ImportError`` from instantiation is caught).
    """
    mem_api_key = os.environ.get("MEMORY_EXTRACTION_API_KEY", "").strip()
    mem_base_url = os.environ.get("MEMORY_EXTRACTION_BASE_URL", "").strip()
    mem_model = os.environ.get("MEMORY_EXTRACTION_MODEL", "").strip()
    mem_provider = os.environ.get("MEMORY_EXTRACTION_PROVIDER", "").strip().lower()

    # 1. Dedicated cheap-extractor override (OpenAI-compatible endpoint)
    if mem_api_key and mem_base_url and mem_model:
        try:
            return OpenAICompatibleLlmClient(
                api_key=mem_api_key,
                base_url=mem_base_url,
                model=mem_model,
            )
        except ImportError:
            logger.warning(
                "llm_client: MEMORY_EXTRACTION_* env vars set but 'openai' is not installed"
            )
            return None

    # 2. Anthropic via MEMORY_EXTRACTION_PROVIDER
    if mem_provider == "anthropic" and mem_api_key:
        model = mem_model or os.environ.get("ANTHROPIC_MODEL", "").strip() or _DEFAULT_ANTHROPIC_MODEL
        try:
            return AnthropicLlmClient(api_key=mem_api_key, model=model)
        except ImportError:
            logger.warning(
                "llm_client: MEMORY_EXTRACTION_PROVIDER=anthropic but 'anthropic' is not installed"
            )
            return None

    # 3. OPENAI_API_KEY
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or _DEFAULT_OPENAI_BASE_URL
        model = os.environ.get("OPENAI_MODEL", "").strip() or _DEFAULT_OPENAI_MODEL
        try:
            return OpenAICompatibleLlmClient(
                api_key=openai_key,
                base_url=base_url,
                model=model,
            )
        except ImportError:
            logger.warning("llm_client: OPENAI_API_KEY set but 'openai' is not installed")
            return None

    # 4. ANTHROPIC_API_KEY
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        model = os.environ.get("ANTHROPIC_MODEL", "").strip() or _DEFAULT_ANTHROPIC_MODEL
        try:
            return AnthropicLlmClient(api_key=anthropic_key, model=model)
        except ImportError:
            logger.warning("llm_client: ANTHROPIC_API_KEY set but 'anthropic' is not installed")
            return None

    return None


# ---------------------------------------------------------------------------
# Public aliases for the JSON utilities (exposed for testing)
# ---------------------------------------------------------------------------

extract_json_from_response = _extract_json_from_response
repair_common_json = _repair_common_json
