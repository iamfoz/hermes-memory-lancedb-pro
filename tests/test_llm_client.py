"""Tests for hermes_memory_lancedb_pro.llm_client.

SDK-call tests mock at the SDK level (not HTTP).  When openai / anthropic are
not installed the SDK-call tests are skipped cleanly via pytest.mark.skipif.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_memory_lancedb_pro.llm_client import (
    AnthropicLlmClient,
    OpenAICompatibleLlmClient,
    create_llm_client_from_env,
    extract_json_from_response,
    repair_common_json,
)

# ---------------------------------------------------------------------------
# Availability flags — used by skipif markers
# ---------------------------------------------------------------------------

_OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None
_ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None


# ===========================================================================
# TestExtractJsonFromResponse
# ===========================================================================

class TestExtractJsonFromResponse:
    def test_plain_json_returned_unchanged(self):
        text = '{"key": "value"}'
        assert extract_json_from_response(text) == text

    def test_markdown_fenced_json_extracted(self):
        text = '```json\n{"a": 1}\n```'
        assert extract_json_from_response(text) == '{"a": 1}'

    def test_markdown_fence_without_language_tag(self):
        text = "```\n{\"x\": true}\n```"
        assert extract_json_from_response(text) == '{"x": true}'

    def test_brace_matching_with_surrounding_text(self):
        text = "Here is the result:\n\n{\"score\": 0.9, \"reason\": \"good\"}\n\nDone."
        result = extract_json_from_response(text)
        assert result == '{"score": 0.9, "reason": "good"}'

    def test_no_brace_returns_none(self):
        assert extract_json_from_response("No JSON here at all.") is None

    def test_unbalanced_braces_returns_none(self):
        # Extra open brace never closed — no balanced run exists
        result = extract_json_from_response('{"key": "value"')
        assert result is None

    def test_first_balanced_run_extracted(self):
        # Two JSON objects in sequence — only the first balanced run is returned
        text = '{"a": 1} {"b": 2}'
        result = extract_json_from_response(text)
        assert result == '{"a": 1}'

    def test_nested_objects(self):
        text = '{"outer": {"inner": 42}}'
        result = extract_json_from_response(text)
        assert result == text

    def test_fence_takes_priority_over_brace_match(self):
        # When there's a fence, it's used even if there's bare JSON before it
        text = '{"bare": 1}\n```json\n{"fenced": 2}\n```'
        result = extract_json_from_response(text)
        assert result == '{"fenced": 2}'


# ===========================================================================
# TestRepairCommonJson
# ===========================================================================

class TestRepairCommonJson:
    def test_trailing_comma_before_brace_removed(self):
        text = '{"a": 1,}'
        result = repair_common_json(text)
        assert json.loads(result) == {"a": 1}

    def test_trailing_comma_before_bracket_removed(self):
        text = '{"list": [1, 2, 3,]}'
        result = repair_common_json(text)
        assert json.loads(result) == {"list": [1, 2, 3]}

    def test_unescaped_quote_inside_string_escaped(self):
        # Value contains an unescaped double-quote mid-string
        # Input:  {"msg": "say "hello" please"}
        # After repair the interior quote should be escaped
        text = '{"msg": "say "hello" please"}'
        repaired = repair_common_json(text)
        # Must parse without error
        parsed = json.loads(repaired)
        assert "msg" in parsed
        assert "hello" in parsed["msg"]

    def test_raw_newline_inside_string_escaped(self):
        text = '{"line": "first\nsecond"}'
        repaired = repair_common_json(text)
        parsed = json.loads(repaired)
        assert parsed["line"] == "first\nsecond"

    def test_raw_tab_inside_string_escaped(self):
        text = '{"col": "a\tb"}'
        repaired = repair_common_json(text)
        parsed = json.loads(repaired)
        assert parsed["col"] == "a\tb"

    def test_already_valid_json_unchanged_semantically(self):
        text = '{"ok": true, "n": 42}'
        repaired = repair_common_json(text)
        assert json.loads(repaired) == {"ok": True, "n": 42}

    def test_backslash_escaped_characters_preserved(self):
        # A properly escaped quote inside a string must survive repair
        text = r'{"escaped": "she said \"hi\""}'
        repaired = repair_common_json(text)
        parsed = json.loads(repaired)
        assert parsed["escaped"] == 'she said "hi"'

    def test_multiple_trailing_commas(self):
        text = '{"a": 1, "b": [1, 2,], "c": {"d": 3,},}'
        repaired = repair_common_json(text)
        parsed = json.loads(repaired)
        assert parsed == {"a": 1, "b": [1, 2], "c": {"d": 3}}


# ===========================================================================
# Helpers for SDK mocking
# ===========================================================================

def _make_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_anthropic_response(content: str) -> MagicMock:
    block = MagicMock()
    block.text = content
    resp = MagicMock()
    resp.content = [block]
    return resp


# ===========================================================================
# TestOpenAIClient
# ===========================================================================

class TestOpenAIClient:
    def test_import_error_when_openai_not_installed(self, monkeypatch):
        """Instantiation raises ImportError with a clear message when openai missing."""
        # Temporarily hide the openai module regardless of whether it's installed
        saved = sys.modules.get("openai", None)
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="openai"):
                OpenAICompatibleLlmClient(
                    api_key="sk-test",
                    base_url="https://api.openai.com/v1",
                    model="gpt-4o-mini",
                )
        finally:
            if saved is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = saved

    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="openai not installed")
    def test_successful_json_completion(self, monkeypatch):
        mock_response = _make_openai_response('{"result": "ok"}')
        mock_create = MagicMock(return_value=mock_response)

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create = mock_create
            client = OpenAICompatibleLlmClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            )
            # Replace the internal client with our mock
            client._client = MockOpenAI.return_value
            result = client.complete_json("Give me JSON")

        assert result == {"result": "ok"}
        assert client.get_last_error() is None

    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="openai not installed")
    def test_network_error_returns_none(self, monkeypatch):
        mock_create = MagicMock(side_effect=ConnectionError("connection refused"))

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create = mock_create
            client = OpenAICompatibleLlmClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            )
            client._client = MockOpenAI.return_value
            result = client.complete_json("prompt", label="test-net")

        assert result is None
        assert client.get_last_error() is not None
        assert "connection refused" in client.get_last_error()

    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="openai not installed")
    def test_malformed_json_repaired(self, monkeypatch):
        # Trailing comma — needs repair
        mock_response = _make_openai_response('{"a": 1,}')
        mock_create = MagicMock(return_value=mock_response)

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create = mock_create
            client = OpenAICompatibleLlmClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            )
            client._client = MockOpenAI.return_value
            result = client.complete_json("prompt")

        assert result == {"a": 1}

    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="openai not installed")
    def test_irreparably_malformed_json_returns_none(self, monkeypatch):
        mock_response = _make_openai_response("not json at all !!!!")
        mock_create = MagicMock(return_value=mock_response)

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create = mock_create
            client = OpenAICompatibleLlmClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            )
            client._client = MockOpenAI.return_value
            result = client.complete_json("prompt")

        assert result is None
        assert client.get_last_error() is not None

    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="openai not installed")
    def test_markdown_fenced_response_parsed(self, monkeypatch):
        mock_response = _make_openai_response('```json\n{"value": 99}\n```')
        mock_create = MagicMock(return_value=mock_response)

        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create = mock_create
            client = OpenAICompatibleLlmClient(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
            )
            client._client = MockOpenAI.return_value
            result = client.complete_json("prompt")

        assert result == {"value": 99}


# ===========================================================================
# TestAnthropicClient
# ===========================================================================

class TestAnthropicClient:
    def test_import_error_when_anthropic_not_installed(self):
        """Instantiation raises ImportError with a clear message when anthropic missing."""
        saved = sys.modules.get("anthropic", None)
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicLlmClient(api_key="sk-ant-test")
        finally:
            if saved is None:
                del sys.modules["anthropic"]
            else:
                sys.modules["anthropic"] = saved

    @pytest.mark.skipif(not _ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    def test_successful_json_completion(self):
        mock_response = _make_anthropic_response('{"status": "done"}')
        mock_create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create = mock_create
            client = AnthropicLlmClient(api_key="sk-ant-test")
            client._client = MockAnthropic.return_value
            result = client.complete_json("Give me JSON")

        assert result == {"status": "done"}
        assert client.get_last_error() is None

    @pytest.mark.skipif(not _ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    def test_network_error_returns_none(self):
        mock_create = MagicMock(side_effect=RuntimeError("timeout"))

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create = mock_create
            client = AnthropicLlmClient(api_key="sk-ant-test")
            client._client = MockAnthropic.return_value
            result = client.complete_json("prompt", label="test-ant")

        assert result is None
        assert "timeout" in client.get_last_error()

    @pytest.mark.skipif(not _ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    def test_malformed_json_repaired(self):
        mock_response = _make_anthropic_response('{"x": [1, 2,]}')
        mock_create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create = mock_create
            client = AnthropicLlmClient(api_key="sk-ant-test")
            client._client = MockAnthropic.return_value
            result = client.complete_json("prompt")

        assert result == {"x": [1, 2]}

    @pytest.mark.skipif(not _ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    def test_irreparably_malformed_json_returns_none(self):
        mock_response = _make_anthropic_response("sorry, I cannot do that")
        mock_create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create = mock_create
            client = AnthropicLlmClient(api_key="sk-ant-test")
            client._client = MockAnthropic.return_value
            result = client.complete_json("prompt")

        assert result is None
        assert client.get_last_error() is not None

    @pytest.mark.skipif(not _ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    def test_markdown_fenced_response_parsed(self):
        mock_response = _make_anthropic_response("```\n{\"items\": [1,2,3]}\n```")
        mock_create = MagicMock(return_value=mock_response)

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create = mock_create
            client = AnthropicLlmClient(api_key="sk-ant-test")
            client._client = MockAnthropic.return_value
            result = client.complete_json("prompt")

        assert result == {"items": [1, 2, 3]}


# ===========================================================================
# TestCreateLlmClientFromEnv
# ===========================================================================

# We run env-detection tests with both SDKs hidden so the tests exercise the
# env-detection logic itself rather than actual SDK instantiation.

def _hide_both_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make openai and anthropic appear unimportable for this test."""
    # Only hide if not already None (to avoid masking a real None marker from
    # a previous test bleeding through). We set them to None which triggers
    # the ImportError path inside the adapters.
    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]


def _stub_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a minimal stub for the openai module so instantiation succeeds."""
    stub = types.ModuleType("openai")
    mock_client_instance = MagicMock()
    stub.OpenAI = MagicMock(return_value=mock_client_instance)
    monkeypatch.setitem(sys.modules, "openai", stub)
    return stub


def _stub_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a minimal stub for the anthropic module so instantiation succeeds."""
    stub = types.ModuleType("anthropic")
    mock_client_instance = MagicMock()
    stub.Anthropic = MagicMock(return_value=mock_client_instance)
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    return stub


class TestCreateLlmClientFromEnv:
    def _clear_all_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "MEMORY_EXTRACTION_API_KEY",
            "MEMORY_EXTRACTION_BASE_URL",
            "MEMORY_EXTRACTION_MODEL",
            "MEMORY_EXTRACTION_PROVIDER",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_no_env_vars_returns_none(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        assert create_llm_client_from_env() is None

    def test_memory_extraction_openai_override(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_API_KEY", "mem-key")
        monkeypatch.setenv("MEMORY_EXTRACTION_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "qwen2:7b")
        _stub_openai(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, OpenAICompatibleLlmClient)
        assert client.model == "qwen2:7b"
        assert client.base_url == "http://localhost:8080/v1"

    def test_memory_extraction_provider_anthropic(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_PROVIDER", "anthropic")
        monkeypatch.setenv("MEMORY_EXTRACTION_API_KEY", "ant-key")
        _stub_anthropic(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, AnthropicLlmClient)

    def test_memory_extraction_provider_anthropic_with_model(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_PROVIDER", "anthropic")
        monkeypatch.setenv("MEMORY_EXTRACTION_API_KEY", "ant-key")
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-opus-4-5-20251001")
        _stub_anthropic(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, AnthropicLlmClient)
        assert client.model == "claude-opus-4-5-20251001"

    def test_openai_api_key_only(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        _stub_openai(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, OpenAICompatibleLlmClient)
        assert client.base_url == "https://api.openai.com/v1"
        assert client.model == "gpt-4o-mini"

    def test_openai_api_key_with_custom_base_url(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        _stub_openai(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, OpenAICompatibleLlmClient)
        assert client.base_url == "http://localhost:11434/v1"

    def test_anthropic_api_key_only(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
        _stub_anthropic(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, AnthropicLlmClient)

    def test_openai_takes_priority_over_anthropic(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
        _stub_openai(monkeypatch)
        _stub_anthropic(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, OpenAICompatibleLlmClient)

    def test_memory_extraction_takes_priority_over_openai(self, monkeypatch):
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_API_KEY", "mem-key")
        monkeypatch.setenv("MEMORY_EXTRACTION_BASE_URL", "http://proxy/v1")
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "local-model")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        _stub_openai(monkeypatch)

        client = create_llm_client_from_env()
        assert isinstance(client, OpenAICompatibleLlmClient)
        assert client.base_url == "http://proxy/v1"
        assert client.model == "local-model"

    def test_missing_sdk_returns_none(self, monkeypatch):
        """When the selected SDK is not installed, factory returns None."""
        self._clear_all_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        # Hide openai so the ImportError path is taken
        monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]
        # Also hide anthropic so fallback also fails
        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]

        result = create_llm_client_from_env()
        assert result is None
