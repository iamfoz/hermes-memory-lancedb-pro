"""A failing extraction model must be visible and diagnosable.

A failed extraction call no longer loses the turn (it is stored unprocessed),
so nothing looks wrong: memories keep arriving, never distilled.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import types

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli, llm_client  # noqa: E402
from hermes_memory_lancedb_pro.smart_extractor import SmartExtractor  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration


class Embedder:
    def encode(self, text, normalize_embeddings=False, show_progress_bar=False, **_):
        def one(t):
            d = hashlib.sha256(t.encode("utf-8")).digest()
            f = [(d[i % len(d)] - 128) / 128.0 for i in range(VECTOR_DIM)]
            n = sum(x * x for x in f) ** 0.5
            return [x / n for x in f]

        return one(text) if isinstance(text, str) else [one(t) for t in text]


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-extract-")
    s = MemoryStore(db_path=d)
    s._embedder = Embedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


class Llm:
    model = "test-model"

    def __init__(self, reply, error=None):
        self.reply, self.error = reply, error

    def complete_json(self, prompt, *, label=None):
        return self.reply

    def get_last_error(self):
        return self.error

    def last_error_is_transient(self):
        return False


TURN = dict(user_content="Remember that the staging database lives on host db-7.",
            assistant_content="Noted: staging is on db-7.")


def _doctor(store, capsys) -> tuple[int, str]:
    capsys.readouterr()
    rc = _cli._cmd_doctor(argparse.Namespace(path=None, quiet=False), _store=store)
    return rc, capsys.readouterr().out


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_timeout_and_max_tokens_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("MEMORY_EXTRACTION_TIMEOUT", raising=False)
    monkeypatch.delenv("MEMORY_EXTRACTION_MAX_TOKENS", raising=False)
    assert llm_client.extraction_timeout_s() == 30.0
    assert llm_client.extraction_max_tokens() == 4096
    monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT", "90")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_TOKENS", "8000")
    assert llm_client.extraction_timeout_s() == 90.0
    assert llm_client.extraction_max_tokens() == 8000


@pytest.mark.parametrize("bad", ["soon", "0", "-5", ""])
def test_bad_settings_fall_back_to_the_defaults(monkeypatch, bad):
    monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT", bad)
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_TOKENS", bad)
    assert llm_client.extraction_timeout_s() == 30.0
    assert llm_client.extraction_max_tokens() == 4096


def test_the_openai_client_is_built_with_the_configured_timeout(monkeypatch):
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT", "90")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_TOKENS", "8000")
    client = llm_client.OpenAICompatibleLlmClient(api_key="k", base_url="http://x/v1", model="m")
    assert built["timeout"] == 90.0 and client.timeout == 90.0 and client.max_tokens == 8000


# ---------------------------------------------------------------------------
# Empty replies from reasoning models
# ---------------------------------------------------------------------------


def _response(content, finish, reasoning=None):
    message = types.SimpleNamespace(content=content, reasoning_content=reasoning)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)])


def test_an_empty_reply_from_a_reasoning_model_says_why():
    reason = llm_client._empty_reply_reason(_response("", "length", "thinking..."), "some/model")
    assert "finish_reason=length" in reason and "reasoning" in reason
    assert "MEMORY_EXTRACTION_MAX_TOKENS" in reason and "non-reasoning model" in reason


def test_a_plain_empty_reply_does_not_blame_reasoning():
    reason = llm_client._empty_reply_reason(_response("", "stop"), "some/model")
    assert "finish_reason=stop" in reason and "reasoning" not in reason
    assert "empty response content" in llm_client._empty_reply_reason(object(), "m")


# ---------------------------------------------------------------------------
# Status recorded and reported
# ---------------------------------------------------------------------------


def test_a_failed_extraction_is_recorded_and_the_turn_is_still_stored(store):
    extractor = SmartExtractor(store, llm=Llm(None, "request failed: Timeout"))
    extractor.extract_and_persist(**TURN)
    status = llm_client.read_extraction_status(store.db_path)
    assert status["consecutive_failures"] == 1 and "Timeout" in status["last_error"]
    assert status["model"] == "test-model" and "last_success_at" not in status
    assert store.list_memories(limit=10)  # stored unprocessed, not lost


def test_a_success_clears_the_failure_count(store):
    SmartExtractor(store, llm=Llm(None, "boom")).extract_and_persist(**TURN)
    SmartExtractor(store, llm=Llm({"memories": []})).extract_and_persist(**TURN)
    status = llm_client.read_extraction_status(store.db_path)
    assert status["consecutive_failures"] == 0 and status["last_success_at"]


def test_doctor_reports_failing_extraction_and_recommends_the_test(store, capsys):
    rc, out = _doctor(store, capsys)
    assert rc == 0 and "no extraction calls recorded" in out
    assert "extraction-test" not in out

    for _ in range(3):
        llm_client.record_extraction_outcome(
            store.db_path, ok=False, model="some/model",
            error="empty response content from model some/model (finish_reason=length)")
    rc, out = _doctor(store, capsys)
    assert "Extraction is FAILING" in out and "consecutive failures: 3" in out
    assert "finish_reason=length" in out and "last success:         never" in out
    assert "hermes lancedb_pro extraction-test" in out

    llm_client.record_extraction_outcome(store.db_path, ok=True, model="some/model")
    rc, out = _doctor(store, capsys)
    assert "Extraction is FAILING" not in out and "consecutive failures: 0" in out
    assert "hermes lancedb_pro extraction-test" not in out


def test_one_or_two_failures_are_reported_but_not_called_failing(store, capsys):
    llm_client.record_extraction_outcome(store.db_path, ok=False, error="blip")
    _, out = _doctor(store, capsys)
    assert "consecutive failures: 1" in out and "FAILING" not in out


def test_a_corrupt_status_file_is_ignored(store, capsys):
    import os

    with open(os.path.join(store.db_path, llm_client.EXTRACTION_STATUS_FILE), "w") as f:
        f.write("{not json")
    assert llm_client.read_extraction_status(store.db_path) == {}
    rc, out = _doctor(store, capsys)
    assert rc == 0 and "no extraction calls recorded" in out


# ---------------------------------------------------------------------------
# extraction-test
# ---------------------------------------------------------------------------


def _run_test(monkeypatch, capsys, client) -> tuple[int, str]:
    monkeypatch.setattr(llm_client, "create_llm_client_from_env", lambda env=None: client)
    capsys.readouterr()
    rc = _cli._cmd_extraction_test(argparse.Namespace())
    return rc, capsys.readouterr().out


def test_extraction_test_reports_a_working_model(monkeypatch, capsys):
    client = Llm({"memories": [{"category": "entities", "abstract": "Staging database is on db-7"}]})
    client.timeout, client.max_tokens, client.base_url = 90.0, 4096, "http://router/v1"
    rc, out = _run_test(monkeypatch, capsys, client)
    assert rc == 0 and "OK in" in out and "1 memory candidate(s)" in out
    assert "test-model" in out and "http://router/v1" in out and "90.0s" in out
    assert "Staging database is on db-7" in out


def test_extraction_test_reports_the_error(monkeypatch, capsys):
    rc, out = _run_test(monkeypatch, capsys, Llm(None, "empty response content (finish_reason=length)"))
    assert rc == 1 and "FAILED" in out and "finish_reason=length" in out


def test_extraction_test_without_a_model_says_so(monkeypatch, capsys):
    rc, out = _run_test(monkeypatch, capsys, None)
    assert rc == 1 and "No extraction model is configured" in out


def test_extraction_test_survives_a_client_that_raises(monkeypatch, capsys):
    class Raises(Llm):
        def complete_json(self, prompt, *, label=None):
            raise RuntimeError("socket closed")

    rc, out = _run_test(monkeypatch, capsys, Raises(None))
    assert rc == 1 and "socket closed" in out


def test_extraction_test_is_in_both_parsers():
    host = argparse.ArgumentParser()
    _cli.register_cli(host)
    assert host.parse_args(["extraction-test"]).lancedb_pro_command == "extraction-test"
