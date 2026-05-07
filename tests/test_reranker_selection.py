"""Tests for multi-backend reranker selection and the Google ranking backend.

Because API keys are now read in MemoryRetriever.__init__ (not at module
import time), tests control the env by setting env vars before
instantiating the retriever — no module reloading required.
"""
from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from hermes_memory_lancedb_pro.retriever import MemoryRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retriever(monkeypatch, env: dict) -> MemoryRetriever:
    """Set env vars and return a freshly-constructed retriever."""
    # Clear all reranker-related vars so tests don't bleed into each other
    for k in (
        "LANGSEARCH_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_PROJECT_ID", "GOOGLE_PROJECT", "MEMORY_RERANKER",
        "MEMORY_GOOGLE_RANKING_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return MemoryRetriever(MagicMock())


# ---------------------------------------------------------------------------
# _resolve_reranker static method — tested directly for clarity
# ---------------------------------------------------------------------------

class TestResolveReranker:
    """Unit tests for the pure selection logic (no env, no HTTP)."""

    def _r(self, *, ls="", gp="", setting="auto") -> str:
        return MemoryRetriever._resolve_reranker(
            langsearch_key=ls,
            google_project=gp,
            setting=setting,
        )

    def test_no_keys_returns_empty(self):
        assert self._r() == ""

    def test_langsearch_only(self):
        assert self._r(ls="ls-key") == "langsearch"

    def test_google_project_only(self):
        assert self._r(gp="my-project") == "google"

    def test_google_without_project_returns_empty(self):
        assert self._r() == ""

    def test_both_configured_auto_returns_empty_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            result = self._r(ls="ls-key", gp="proj", setting="auto")
        assert result == ""
        assert "MEMORY_RERANKER" in caplog.text

    def test_explicit_langsearch_with_key(self):
        assert self._r(ls="ls-key", setting="langsearch") == "langsearch"

    def test_explicit_langsearch_without_key_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            result = self._r(setting="langsearch")
        assert result == ""
        assert "LANGSEARCH_API_KEY" in caplog.text

    def test_explicit_google_with_project(self):
        assert self._r(gp="proj", setting="google") == "google"

    def test_explicit_google_missing_project_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            result = self._r(setting="google")
        assert result == ""
        assert "GOOGLE_CLOUD_PROJECT" in caplog.text

    def test_disabled_setting_always_returns_empty(self):
        assert self._r(ls="ls-key", gp="proj", setting="disabled") == ""

    def test_explicit_wins_over_both_configured(self):
        assert self._r(ls="ls", gp="proj", setting="langsearch") == "langsearch"
        assert self._r(ls="ls", gp="proj", setting="google") == "google"


# ---------------------------------------------------------------------------
# MemoryRetriever.__init__ reads env vars — tests confirm dotenv timing fix
# ---------------------------------------------------------------------------

class TestEnvReadAtInit:
    """Verify that key reading is deferred to __init__, not import time.

    These tests set env vars before constructing the retriever (mimicking
    dotenv loading before provider instantiation) and check that the correct
    backend is selected.
    """

    def test_langsearch_key_in_env_selects_langsearch(self, monkeypatch):
        r = _retriever(monkeypatch, {"LANGSEARCH_API_KEY": "ls-test"})
        assert r._active_reranker == "langsearch"
        assert r._langsearch_api_key == "ls-test"

    def test_google_project_in_env_selects_google(self, monkeypatch):
        r = _retriever(monkeypatch, {"GOOGLE_CLOUD_PROJECT": "my-project"})
        assert r._active_reranker == "google"
        assert r._google_cloud_project == "my-project"

    def test_google_project_id_alias(self, monkeypatch):
        r = _retriever(monkeypatch, {"GOOGLE_PROJECT_ID": "alias-project"})
        assert r._active_reranker == "google"
        assert r._google_cloud_project == "alias-project"

    def test_no_keys_gives_no_reranker(self, monkeypatch):
        r = _retriever(monkeypatch, {})
        assert r._active_reranker == ""

    def test_both_configured_no_explicit_choice_warns_and_disables(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            r = _retriever(monkeypatch, {
                "LANGSEARCH_API_KEY": "ls-test",
                "GOOGLE_CLOUD_PROJECT": "my-project",
            })
        assert r._active_reranker == ""
        assert "MEMORY_RERANKER" in caplog.text

    def test_explicit_memory_reranker_overrides_auto(self, monkeypatch):
        r = _retriever(monkeypatch, {
            "LANGSEARCH_API_KEY": "ls-test",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "MEMORY_RERANKER": "google",
        })
        assert r._active_reranker == "google"

    def test_keys_absent_at_import_but_set_before_init(self, monkeypatch):
        """The core regression test: if GOOGLE_CLOUD_PROJECT is in .env and
        dotenv loads it into os.environ before MemoryRetriever is instantiated
        (but after the module is imported), the retriever must still pick it up."""
        # Simulate: module was imported with empty env (dotenv not loaded yet).
        # Then dotenv fires and populates the vars.
        # Then MemoryRetriever() is constructed — must see the key.
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project-from-dotenv")
        r = MemoryRetriever(MagicMock())
        assert r._active_reranker == "google"
        assert r._google_cloud_project == "my-project-from-dotenv"


# ---------------------------------------------------------------------------
# _apply_rerank_scores (shared blend logic — backend-agnostic)
# ---------------------------------------------------------------------------

class TestApplyRerankScores:
    def _candidates(self, n: int) -> list[dict]:
        return [
            {"id": str(i), "text": f"candidate {i}", "_final_score": 0.5 - i * 0.05}
            for i in range(n)
        ]

    def test_reorders_by_rerank_score(self):
        c = self._candidates(3)
        out = MemoryRetriever._apply_rerank_scores(list(c), list(c), 3, {0: 0.3, 1: 0.5, 2: 0.9})
        assert out[0]["id"] == "2"

    def test_unscored_candidates_penalised(self):
        c = self._candidates(3)
        out = MemoryRetriever._apply_rerank_scores(c, list(c), 3, {0: 0.8})
        scored = next(e for e in out if e["id"] == "0")
        unscored = next(e for e in out if e["id"] == "1")
        assert scored["_final_score"] > unscored["_final_score"]
        assert unscored["_rerank_score"] == 0.0

    def test_tail_preserved_unchanged(self):
        c = self._candidates(2)
        tail = {"id": "tail", "text": "tail", "_final_score": 0.99}
        results = list(c) + [tail]
        out = MemoryRetriever._apply_rerank_scores(c, results, 2, {0: 0.5, 1: 0.8})
        assert out[-1]["id"] == "tail"
        assert out[-1]["_final_score"] == 0.99

    def test_blend_weights_70_30(self):
        c = [{"id": "0", "text": "x", "_final_score": 1.0}]
        out = MemoryRetriever._apply_rerank_scores(c, list(c), 1, {0: 0.6})
        # 0.7 * 0.6 + 0.3 * 1.0 = 0.72
        assert out[0]["_final_score"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# Google auth token acquisition (_get_google_auth_token)
# ---------------------------------------------------------------------------

class TestGoogleAuthToken:
    """Unit tests for the ADC-based token acquisition method."""

    def _retriever_with_google(self, monkeypatch) -> MemoryRetriever:
        return _retriever(monkeypatch, {"GOOGLE_CLOUD_PROJECT": "test-project"})

    def _mock_google_modules(self, mock_auth=None, mock_transport=None):
        """Return a sys.modules patch dict for google.auth + transport.

        Python's ``import google.auth`` statement resolves ``google.auth`` as
        an *attribute* on the ``google`` package object, not as a standalone
        local variable.  We must therefore wire the attribute chain so that
        ``google.auth.default(...)`` (accessed via ``google_mod.auth``) and
        ``google.auth.transport.requests.Request()`` resolve to our mocks.
        """
        google_mod = MagicMock()
        google_auth = mock_auth if mock_auth is not None else MagicMock()
        google_transport = MagicMock()
        google_transport_requests = mock_transport if mock_transport is not None else MagicMock()
        # Wire attribute chain: google.auth → google_auth, etc.
        google_mod.auth = google_auth
        google_auth.transport = google_transport
        google_transport.requests = google_transport_requests
        return {
            "google": google_mod,
            "google.auth": google_auth,
            "google.auth.transport": google_transport,
            "google.auth.transport.requests": google_transport_requests,
        }

    def test_missing_google_auth_library_trips_disable(self, monkeypatch, caplog):
        r = self._retriever_with_google(monkeypatch)
        with patch.dict(sys.modules, {"google.auth": None, "google.auth.transport.requests": None}):
            with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
                token = r._get_google_auth_token()
        assert token is None
        assert r._google_disabled is True
        assert "google-auth" in caplog.text

    def test_adc_credentials_not_found_trips_disable(self, monkeypatch, caplog):
        r = self._retriever_with_google(monkeypatch)
        mock_auth = MagicMock()
        mock_auth.default.side_effect = Exception("Could not find credentials")
        with patch.dict(sys.modules, self._mock_google_modules(mock_auth=mock_auth)):
            with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
                token = r._get_google_auth_token()
        assert token is None
        assert r._google_disabled is True
        assert "GOOGLE_APPLICATION_CREDENTIALS" in caplog.text

    def test_valid_credentials_returns_token(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "ya29.good-token"
        mock_auth = MagicMock()
        mock_auth.default.return_value = (mock_creds, "test-project")
        with patch.dict(sys.modules, self._mock_google_modules(mock_auth=mock_auth)):
            token = r._get_google_auth_token()
        assert token == "ya29.good-token"
        assert r._google_disabled is False

    def test_expired_credentials_are_refreshed(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        mock_creds = MagicMock()
        mock_creds.valid = False  # expired
        mock_creds.token = "ya29.refreshed-token"
        mock_auth = MagicMock()
        mock_auth.default.return_value = (mock_creds, "test-project")
        with patch.dict(sys.modules, self._mock_google_modules(mock_auth=mock_auth)):
            token = r._get_google_auth_token()
        mock_creds.refresh.assert_called_once()
        assert token == "ya29.refreshed-token"

    def test_token_refresh_failure_trips_disable(self, monkeypatch, caplog):
        r = self._retriever_with_google(monkeypatch)
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.refresh.side_effect = Exception("Token refresh failed")
        mock_auth = MagicMock()
        mock_auth.default.return_value = (mock_creds, "test-project")
        with patch.dict(sys.modules, self._mock_google_modules(mock_auth=mock_auth)):
            with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
                token = r._get_google_auth_token()
        assert token is None
        assert r._google_disabled is True

    def test_cached_credentials_not_re_fetched(self, monkeypatch):
        """Second call reuses cached creds; google.auth.default not called again."""
        r = self._retriever_with_google(monkeypatch)
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.token = "ya29.cached-token"
        mock_auth = MagicMock()
        mock_auth.default.return_value = (mock_creds, "test-project")
        with patch.dict(sys.modules, self._mock_google_modules(mock_auth=mock_auth)):
            r._get_google_auth_token()
            r._get_google_auth_token()
        assert mock_auth.default.call_count == 1  # only fetched once


# ---------------------------------------------------------------------------
# Google backend: HTTP interaction
# ---------------------------------------------------------------------------

class TestGoogleRerankBackend:
    def _retriever_with_google(self, monkeypatch) -> MemoryRetriever:
        return _retriever(monkeypatch, {"GOOGLE_CLOUD_PROJECT": "test-project"})

    def _results(self, n: int) -> list[dict]:
        return [
            {"id": str(i), "text": f"result {i} " * 10, "_final_score": 0.9 - i * 0.1}
            for i in range(n)
        ]

    def test_reorders_by_google_score(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        results = self._results(4)
        # Reverse ranking: candidate 3 gets highest score
        reversed_records = [
            {"id": str(3 - i), "content": "", "relevanceScore": 0.9 - i * 0.1}
            for i in range(4)
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": reversed_records}
        r._google_session = MagicMock()
        r._google_session.headers = {}
        r._google_session.post.return_value = mock_resp

        with patch.object(r, "_get_google_auth_token", return_value="ya29.test-token"):
            out = r._rerank_google("query", results, top_n=4)
        assert out[0]["id"] == "3"

    def test_401_trips_disable_flag(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        results = self._results(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        r._google_session = MagicMock()
        r._google_session.headers = {}
        r._google_session.post.return_value = mock_resp

        with patch.object(r, "_get_google_auth_token", return_value="ya29.test-token"):
            out = r._rerank_google("query", results)
        assert r._google_disabled is True
        assert out == results

    def test_404_trips_disable_with_helpful_warning(self, monkeypatch, caplog):
        r = self._retriever_with_google(monkeypatch)
        results = self._results(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        r._google_session = MagicMock()
        r._google_session.headers = {}
        r._google_session.post.return_value = mock_resp

        with patch.object(r, "_get_google_auth_token", return_value="ya29.test-token"):
            with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
                r._rerank_google("query", results, top_n=3)

        assert r._google_disabled is True
        assert "Discovery Engine" in caplog.text

    def test_disabled_flag_short_circuits(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        r._google_disabled = True
        r._google_session = MagicMock()
        r._google_session.headers = {}
        r._rerank_google("query", self._results(3))
        r._google_session.post.assert_not_called()
        assert r._google_disabled is True

    def test_single_result_skipped(self, monkeypatch):
        r = self._retriever_with_google(monkeypatch)
        r._google_session = MagicMock()
        r._google_session.headers = {}
        with patch.object(r, "_get_google_auth_token", return_value="ya29.test-token"):
            r._rerank_google("query", self._results(1))
        r._google_session.post.assert_not_called()

    def test_none_token_short_circuits(self, monkeypatch):
        """If _get_google_auth_token returns None, no HTTP call is made."""
        r = self._retriever_with_google(monkeypatch)
        r._google_session = MagicMock()
        r._google_session.headers = {}
        with patch.object(r, "_get_google_auth_token", return_value=None):
            out = r._rerank_google("query", self._results(3))
        r._google_session.post.assert_not_called()
        assert out == self._results(3)

    def test_uses_oauth_bearer_token(self, monkeypatch):
        """Bearer token is set in the Authorization header, not x-goog-api-key."""
        r = _retriever(monkeypatch, {"GOOGLE_CLOUD_PROJECT": "test-project"})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [
            {"id": str(i), "relevanceScore": 0.5} for i in range(3)
        ]}
        real_headers: dict = {}
        with patch("requests.Session") as mock_cls, \
             patch.object(r, "_get_google_auth_token", return_value="ya29.my-token"):
            mock_session = MagicMock()
            mock_session.headers = real_headers
            mock_session.post.return_value = mock_resp
            mock_cls.return_value = mock_session
            r._google_session = None  # force lazy creation
            r._rerank_google("query", self._results(3), top_n=3)
        assert real_headers.get("Authorization") == "Bearer ya29.my-token"
        assert "x-goog-api-key" not in real_headers

    def test_project_used_in_url(self, monkeypatch):
        r = _retriever(monkeypatch, {"GOOGLE_CLOUD_PROJECT": "correct-project-123"})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [
            {"id": "0", "relevanceScore": 0.9},
            {"id": "1", "relevanceScore": 0.5},
        ]}
        r._google_session = MagicMock()
        r._google_session.headers = {}
        r._google_session.post.return_value = mock_resp
        with patch.object(r, "_get_google_auth_token", return_value="ya29.test-token"):
            r._rerank_google("query", self._results(2), top_n=2)

        call_url = r._google_session.post.call_args[0][0]
        assert "correct-project-123" in call_url
