"""Tests for multi-backend reranker selection and the Google ranking backend.

The _resolve_reranker() function is called at module import time, so tests
that need to vary env vars must reload the module or patch the module-level
ACTIVE_RERANKER constant.  We use monkeypatch + importlib.reload() for the
selection tests, and unittest.mock for the HTTP-level backend tests.
"""
from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: reload retriever with a custom env
# ---------------------------------------------------------------------------

def _reload_retriever(monkeypatch, env: dict) -> types.ModuleType:
    """Patch os.environ and force-reload retriever, returning the fresh module."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Clear keys not in env (so previous test state doesn't leak)
    for k in ("LANGSEARCH_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT",
               "GOOGLE_PROJECT_ID", "MEMORY_RERANKER"):
        if k not in env:
            monkeypatch.delenv(k, raising=False)

    mod_name = "hermes_memory_lancedb_pro.retriever"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    import hermes_memory_lancedb_pro.retriever as mod
    return mod


# ---------------------------------------------------------------------------
# Auto-selection logic
# ---------------------------------------------------------------------------

class TestRerankerAutoSelection:
    def test_no_keys_gives_empty(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {})
        assert mod.ACTIVE_RERANKER == ""

    def test_only_langsearch_key_selects_langsearch(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {"LANGSEARCH_API_KEY": "ls-test"})
        assert mod.ACTIVE_RERANKER == "langsearch"

    def test_only_google_keys_selects_google(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {
            "GOOGLE_API_KEY": "AIzaTest",
            "GOOGLE_CLOUD_PROJECT": "my-project",
        })
        assert mod.ACTIVE_RERANKER == "google"

    def test_google_key_without_project_gives_empty(self, monkeypatch):
        # GOOGLE_API_KEY alone isn't enough — project ID is in the URL
        mod = _reload_retriever(monkeypatch, {"GOOGLE_API_KEY": "AIzaTest"})
        assert mod.ACTIVE_RERANKER == ""

    def test_both_keys_auto_gives_empty_and_warns(self, monkeypatch, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            mod = _reload_retriever(monkeypatch, {
                "LANGSEARCH_API_KEY": "ls-test",
                "GOOGLE_API_KEY": "AIzaTest",
                "GOOGLE_CLOUD_PROJECT": "my-project",
            })
        assert mod.ACTIVE_RERANKER == ""
        assert "MEMORY_RERANKER" in caplog.text

    def test_explicit_langsearch_wins_when_both_set(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {
            "LANGSEARCH_API_KEY": "ls-test",
            "GOOGLE_API_KEY": "AIzaTest",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "MEMORY_RERANKER": "langsearch",
        })
        assert mod.ACTIVE_RERANKER == "langsearch"

    def test_explicit_google_wins_when_both_set(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {
            "LANGSEARCH_API_KEY": "ls-test",
            "GOOGLE_API_KEY": "AIzaTest",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "MEMORY_RERANKER": "google",
        })
        assert mod.ACTIVE_RERANKER == "google"

    def test_explicit_disabled_suppresses_all(self, monkeypatch):
        mod = _reload_retriever(monkeypatch, {
            "LANGSEARCH_API_KEY": "ls-test",
            "MEMORY_RERANKER": "disabled",
        })
        assert mod.ACTIVE_RERANKER == ""

    def test_explicit_langsearch_without_key_warns(self, monkeypatch, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            mod = _reload_retriever(monkeypatch, {"MEMORY_RERANKER": "langsearch"})
        assert mod.ACTIVE_RERANKER == ""
        assert "LANGSEARCH_API_KEY" in caplog.text

    def test_explicit_google_without_project_warns(self, monkeypatch, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"):
            mod = _reload_retriever(monkeypatch, {
                "GOOGLE_API_KEY": "AIzaTest",
                "MEMORY_RERANKER": "google",
                # no GOOGLE_CLOUD_PROJECT
            })
        assert mod.ACTIVE_RERANKER == ""
        assert "GOOGLE_CLOUD_PROJECT" in caplog.text

    def test_google_project_id_alias_accepted(self, monkeypatch):
        # GOOGLE_PROJECT_ID is an accepted alias for GOOGLE_CLOUD_PROJECT
        mod = _reload_retriever(monkeypatch, {
            "GOOGLE_API_KEY": "AIzaTest",
            "GOOGLE_PROJECT_ID": "my-project-via-alias",
        })
        assert mod.ACTIVE_RERANKER == "google"
        assert mod.GOOGLE_CLOUD_PROJECT == "my-project-via-alias"


# ---------------------------------------------------------------------------
# _apply_rerank_scores (shared blend logic — backend-agnostic)
# ---------------------------------------------------------------------------

class TestApplyRerankScores:
    """Tests for the static blend helper without any HTTP."""

    from hermes_memory_lancedb_pro.retriever import MemoryRetriever

    def _make_candidates(self, n: int) -> list[dict]:
        return [
            {"id": str(i), "text": f"candidate {i}", "_final_score": 0.5 - i * 0.05}
            for i in range(n)
        ]

    def test_all_scored_reorders_by_rerank(self):
        from hermes_memory_lancedb_pro.retriever import MemoryRetriever
        candidates = self._make_candidates(3)
        results = list(candidates)
        # Give candidate 2 the highest rerank score — it should bubble up
        score_map = {0: 0.3, 1: 0.5, 2: 0.9}
        out = MemoryRetriever._apply_rerank_scores(candidates, results, 3, score_map)
        assert out[0]["id"] == "2"

    def test_unscored_candidates_are_penalised(self):
        from hermes_memory_lancedb_pro.retriever import MemoryRetriever
        candidates = self._make_candidates(3)
        results = list(candidates)
        # Only candidate 0 is scored; 1 and 2 are dropped by reranker
        score_map = {0: 0.8}
        MemoryRetriever._apply_rerank_scores(candidates, results, 3, score_map)
        scored = next(e for e in results if e["id"] == "0")
        unscored = next(e for e in results if e["id"] == "1")
        assert scored["_final_score"] > unscored["_final_score"]
        assert unscored.get("_rerank_score", -1) == 0.0

    def test_tail_beyond_candidate_count_untouched(self):
        from hermes_memory_lancedb_pro.retriever import MemoryRetriever
        candidates = self._make_candidates(2)
        tail = {"id": "tail", "text": "tail entry", "_final_score": 0.99}
        results = list(candidates) + [tail]
        score_map = {0: 0.5, 1: 0.8}
        out = MemoryRetriever._apply_rerank_scores(candidates, results, 2, score_map)
        # tail must still be last and unmutated
        assert out[-1]["id"] == "tail"
        assert out[-1]["_final_score"] == 0.99

    def test_blend_weights(self):
        from hermes_memory_lancedb_pro.retriever import MemoryRetriever
        candidates = [{"id": "0", "text": "x", "_final_score": 1.0}]
        results = list(candidates)
        score_map = {0: 0.6}
        MemoryRetriever._apply_rerank_scores(candidates, results, 1, score_map)
        # 0.7 * 0.6 + 0.3 * 1.0 = 0.42 + 0.30 = 0.72
        assert results[0]["_final_score"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# Google backend: HTTP interaction
# ---------------------------------------------------------------------------

class TestGoogleRerankBackend:
    """Exercises _rerank_google by mocking the HTTP session."""

    def _make_retriever(self, store=None):
        """Return a MemoryRetriever with a fake store."""
        from hermes_memory_lancedb_pro.retriever import MemoryRetriever
        fake_store = MagicMock()
        r = MemoryRetriever(fake_store)
        return r

    def _make_results(self, n: int) -> list[dict]:
        return [
            {"id": str(i), "text": f"text for result {i} " * 5, "_final_score": 0.9 - i * 0.1}
            for i in range(n)
        ]

    def _google_response(self, n: int) -> dict:
        """Fake Google ranking response ordered by descending relevance."""
        return {
            "records": [
                {"id": str(i), "content": f"text {i}", "relevanceScore": 0.9 - i * 0.05}
                for i in range(n)
            ]
        }

    def test_reorders_by_google_score(self):
        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        results = self._make_results(4)
        # Reverse the order in the mock response so candidate 3 is ranked first
        reversed_records = [
            {"id": str(3 - i), "content": "", "relevanceScore": 0.9 - i * 0.1}
            for i in range(4)
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": reversed_records}

        with patch.object(
            ret_mod, "ACTIVE_RERANKER", "google"
        ), patch.object(
            ret_mod, "GOOGLE_CLOUD_PROJECT", "test-project"
        ):
            retriever._google_session = MagicMock()
            retriever._google_session.post.return_value = mock_resp
            out = retriever._rerank_google("query", results, top_n=4)

        # Candidate 3 should now be first
        assert out[0]["id"] == "3"

    def test_401_trips_disable_flag(self):
        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        results = self._make_results(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch.object(ret_mod, "GOOGLE_CLOUD_PROJECT", "test-project"):
            retriever._google_session = MagicMock()
            retriever._google_session.post.return_value = mock_resp
            out = retriever._rerank_google("query", results, top_n=3)

        assert retriever._google_disabled is True
        # Results returned unchanged
        assert out == results

    def test_404_trips_disable_flag_with_helpful_warning(self, caplog):
        import logging

        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        results = self._make_results(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with caplog.at_level(logging.WARNING, logger="hermes_memory_lancedb_pro.retriever"), \
             patch.object(ret_mod, "GOOGLE_CLOUD_PROJECT", "test-project"):
            retriever._google_session = MagicMock()
            retriever._google_session.post.return_value = mock_resp
            retriever._rerank_google("query", results, top_n=3)

        assert retriever._google_disabled is True
        assert "Discovery Engine" in caplog.text

    def test_disabled_flag_short_circuits(self):
        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        retriever._google_disabled = True
        results = self._make_results(3)
        # Session should never be touched
        retriever._google_session = MagicMock()
        out = retriever._rerank_google("query", results)
        retriever._google_session.post.assert_not_called()
        assert out is results

    def test_single_result_skipped(self):
        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        results = self._make_results(1)
        retriever._google_session = MagicMock()
        out = retriever._rerank_google("query", results)
        retriever._google_session.post.assert_not_called()
        assert out == results

    def test_uses_x_goog_api_key_header(self):
        from hermes_memory_lancedb_pro import retriever as ret_mod
        retriever = self._make_retriever()
        results = self._make_results(3)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._google_response(3)

        with patch.object(ret_mod, "GOOGLE_API_KEY", "AIza-test-key"), \
             patch.object(ret_mod, "GOOGLE_CLOUD_PROJECT", "test-project"):
            # Force session creation
            retriever._google_session = None
            with patch("requests.Session") as mock_session_cls:
                mock_session = MagicMock()
                mock_session.post.return_value = mock_resp
                mock_session_cls.return_value = mock_session
                retriever._rerank_google("query", results, top_n=3)
            mock_session.headers.update.assert_called()
            call_kwargs = mock_session.headers.update.call_args[0][0]
            assert call_kwargs.get("x-goog-api-key") == "AIza-test-key"
