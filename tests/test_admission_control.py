"""Tests for admission_control. Pure-Python (no LanceDB needed for the
scoring helpers); the AdmissionController integration test uses a stub
LLM and doesn't need a real LanceDB either."""

from __future__ import annotations

import json

import pytest

from hermes_memory_lancedb_pro.admission_control import (
    AdmissionControlConfig,
    AdmissionRejectionAuditEntry,
    AdmissionTypePriors,
    AdmissionWeights,
    _build_reason,
    _cosine_similarity_safe,
    _lcs_length,
    _rouge_like_f1,
    _split_support_spans,
    _tokenize_text,
    append_rejection_audit,
    get_preset,
    normalize_weights,
    resolve_rejected_audit_path,
    score_confidence_support,
    score_novelty_from_matches,
    score_recency_gap,
    score_type_prior,
    score_utility,
)
from hermes_memory_lancedb_pro.memory_categories import CandidateMemory

# ---------------------------------------------------------------------------
# Tokeniser / LCS / ROUGE
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_lowercases_and_word_splits(self):
        assert _tokenize_text("Hello World!") == ["hello", "world"]

    def test_han_chars_are_per_char_tokens(self):
        # CJK chars are individual tokens; the tokeniser switches modes
        # when it hits one
        assert _tokenize_text("你好world") == ["你", "好", "world"]

    def test_mixed_separators(self):
        assert _tokenize_text("foo,bar.baz") == ["foo", "bar", "baz"]

    def test_empty(self):
        assert _tokenize_text("") == []
        assert _tokenize_text("   ") == []

    def test_strips_outer_whitespace(self):
        assert _tokenize_text("  a b  ") == ["a", "b"]


class TestLcs:
    def test_disjoint_returns_zero(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_full_match(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_subsequence(self):
        # "a c" is a subsequence of "a b c"
        assert _lcs_length(["a", "c"], ["a", "b", "c"]) == 2

    def test_empty(self):
        assert _lcs_length([], ["a"]) == 0
        assert _lcs_length(["a"], []) == 0


class TestRougeF1:
    def test_disjoint_zero(self):
        assert _rouge_like_f1(["a"], ["b"]) == 0.0

    def test_full_match_one(self):
        assert _rouge_like_f1(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_partial(self):
        # LCS=2, |a|=3, |b|=2 → P=2/3, R=1, F1 = 2*0.667*1/1.667 = 0.8
        assert _rouge_like_f1(["a", "b", "c"], ["a", "b"]) == pytest.approx(0.8, abs=1e-3)


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self):
        assert _cosine_similarity_safe([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity_safe([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_similarity_safe([0, 0], [1, 1]) == 0.0

    def test_mismatched_lengths_truncates(self):
        # TS uses min length and proceeds — verify same behaviour
        assert _cosine_similarity_safe([1, 0, 99], [1, 0]) == pytest.approx(1.0)

    def test_empty(self):
        assert _cosine_similarity_safe([], [1]) == 0.0


# ---------------------------------------------------------------------------
# Support span splitter
# ---------------------------------------------------------------------------

class TestSplitSupportSpans:
    def test_lines_become_spans(self):
        text = "line one\nline two\nline three"
        spans = _split_support_spans(text)
        assert "line one" in spans
        assert "line two" in spans
        assert "line three" in spans

    def test_sentences_within_lines(self):
        text = "First sentence. Second sentence! Third?"
        spans = _split_support_spans(text)
        assert any("First sentence" in s for s in spans)
        assert any("Second sentence" in s for s in spans)
        assert any("Third" in s for s in spans)

    def test_short_sentences_below_4_chars_skipped(self):
        text = "ok!"  # 3 chars after stripping the !
        spans = _split_support_spans(text)
        # Whole line still included (3 chars line is fine)
        # but per-sentence "ok" should be excluded (<4 chars)
        assert spans == ["ok!"]


# ---------------------------------------------------------------------------
# Feature scorers
# ---------------------------------------------------------------------------

def _candidate(category="preferences", abstract="user prefers dark mode UI",
               overview=None, content=None, vector=None) -> CandidateMemory:
    return CandidateMemory(
        category=category,
        abstract=abstract,
        overview=overview or abstract,
        content=content or abstract,
        vector=vector,
    )


class TestScoreTypePrior:
    def test_default_priors(self):
        priors = AdmissionTypePriors()
        assert score_type_prior("profile", priors) == pytest.approx(0.95)
        assert score_type_prior("events", priors) == pytest.approx(0.45)

    def test_clamped(self):
        priors = AdmissionTypePriors(profile=2.5)
        assert score_type_prior("profile", priors) == 1.0


class TestScoreConfidenceSupport:
    def test_well_supported(self):
        cand = _candidate(abstract="user prefers dark mode")
        # Conversation literally contains the abstract
        score = score_confidence_support(cand, "user prefers dark mode UI in IDE")
        assert score.score > 0.5
        assert score.coverage > 0.5
        assert score.unsupported_ratio < 0.5

    def test_unsupported(self):
        cand = _candidate(abstract="alpha beta gamma delta epsilon")
        score = score_confidence_support(cand, "completely unrelated content here")
        assert score.score < 0.3
        assert score.unsupported_ratio > 0.5

    def test_empty_candidate(self):
        cand = _candidate(abstract="", content="")
        score = score_confidence_support(cand, "any conversation")
        assert score.score == 0.0
        assert score.unsupported_ratio == 1.0


class TestScoreNovelty:
    def test_no_matches_full_novelty(self):
        n = score_novelty_from_matches([1.0, 0.0], [])
        assert n.score == 1.0
        assert n.max_similarity == 0.0

    def test_zero_vector_full_novelty(self):
        n = score_novelty_from_matches([], [{"id": "x", "vector": [1, 0]}])
        assert n.score == 1.0

    def test_identical_match_low_novelty(self):
        n = score_novelty_from_matches(
            [1.0, 0.0],
            [{"id": "x", "vector": [1.0, 0.0]}],
        )
        assert n.score == pytest.approx(0.0)
        assert n.max_similarity == pytest.approx(1.0)
        assert "x" in n.matched_ids

    def test_threshold_for_matched_ids(self):
        # similarity 0.5 < 0.55 threshold → not in matched_ids (but compared)
        n = score_novelty_from_matches(
            [1.0, 0.0],
            [{"id": "near", "vector": [0.5, 0.866]}],  # 60° angle, sim ~0.5
        )
        assert "near" in n.compared_ids
        assert "near" not in n.matched_ids


class TestScoreRecency:
    def test_no_matches(self):
        assert score_recency_gap(1_000_000, [], 14) == 1.0

    def test_zero_half_life(self):
        assert score_recency_gap(1_000_000, [{"timestamp": 0}], 0) == 1.0

    def test_immediate_restatement(self):
        now = 1_700_000_000_000
        # gap = 0 → score = 0 (heavy penalty)
        assert score_recency_gap(now, [{"timestamp": now}], 14) == 0.0

    def test_long_gap_approaches_one(self):
        now = 1_700_000_000_000
        old = now - 365 * 86_400_000
        # 365 days at 14-day half-life → exp(-26 ln 2) ≈ 0 → score ≈ 1
        assert score_recency_gap(now, [{"timestamp": old}], 14) == pytest.approx(1.0, abs=1e-3)

    def test_one_half_life_gives_half(self):
        now = 1_700_000_000_000
        old = now - 14 * 86_400_000
        # gap = half_life → score = 1 - 0.5 = 0.5
        assert score_recency_gap(now, [{"timestamp": old}], 14) == pytest.approx(0.5, abs=1e-3)


class TestScoreUtility:
    def test_off_returns_neutral(self):
        cand = _candidate()
        score, reason = score_utility(None, "off", cand, "any text")
        assert score == 0.5
        assert "disabled" in reason.lower()

    def test_no_llm_returns_neutral(self):
        cand = _candidate()
        score, reason = score_utility(None, "standalone", cand, "any text")
        assert score == 0.5
        assert "no llm" in reason.lower()

    def test_stub_llm_used(self):
        class StubLLM:
            def complete_json(self, prompt, *, label=None):
                return {"utility": 0.85, "reason": "good preference"}

        cand = _candidate()
        score, reason = score_utility(StubLLM(), "standalone", cand, "any text")
        assert score == pytest.approx(0.85)
        assert reason == "good preference"

    def test_llm_failure_falls_back(self):
        class FailingLLM:
            def complete_json(self, prompt, *, label=None):
                raise RuntimeError("boom")

        cand = _candidate()
        score, reason = score_utility(FailingLLM(), "standalone", cand, "any text")
        assert score == 0.5
        assert "failed" in reason.lower()


# ---------------------------------------------------------------------------
# Config presets / weights
# ---------------------------------------------------------------------------

class TestPresets:
    def test_balanced_default(self):
        c = get_preset("balanced")
        assert c.preset == "balanced"
        assert c.reject_threshold == pytest.approx(0.45)

    def test_conservative_higher_thresholds(self):
        c = get_preset("conservative")
        assert c.reject_threshold > 0.45
        assert c.admit_threshold > 0.6
        # Conservative penalises events more
        assert c.type_priors.events < 0.45

    def test_high_recall_lower_thresholds(self):
        c = get_preset("high-recall")
        assert c.reject_threshold < 0.45
        assert c.admit_threshold < 0.6


class TestNormalizeWeights:
    def test_renormalises_to_one(self):
        w = AdmissionWeights(utility=0.2, confidence=0.2, novelty=0.2,
                             recency=0.2, type_prior=0.2)
        out = normalize_weights(w)
        s = out.utility + out.confidence + out.novelty + out.recency + out.type_prior
        assert s == pytest.approx(1.0)

    def test_zero_weights_falls_back_to_default(self):
        w = AdmissionWeights(utility=0, confidence=0, novelty=0, recency=0, type_prior=0)
        out = normalize_weights(w)
        # default has type_prior=0.6
        assert out.type_prior == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Reason builder + audit helpers
# ---------------------------------------------------------------------------

class TestBuildReason:
    def test_reject_message(self):
        r = _build_reason(
            decision="reject", hint=None, score=0.30,
            reject_threshold=0.45, max_similarity=0.1,
            utility_reason=None,
        )
        assert "rejected" in r
        assert "0.300" in r
        assert "0.450" in r

    def test_pass_with_hint(self):
        r = _build_reason(
            decision="pass_to_dedup", hint="add", score=0.7,
            reject_threshold=0.45, max_similarity=0.2,
            utility_reason="durable",
        )
        assert "passed" in r
        assert "hint=add" in r
        assert "Utility: durable" in r


class TestResolveAuditPath:
    def test_default_layout(self, tmp_path):
        db_path = str(tmp_path / "memory-lancedb")
        path = resolve_rejected_audit_path(db_path, None)
        assert path.endswith("admission-audit/rejections.jsonl")

    def test_explicit_override(self, tmp_path):
        cfg = AdmissionControlConfig(
            rejected_audit_file_path=str(tmp_path / "custom.jsonl")
        )
        path = resolve_rejected_audit_path("/somewhere", cfg)
        assert path == str(tmp_path / "custom.jsonl")


class TestAppendRejectionAudit:
    def test_writes_jsonl(self, tmp_path):
        path = str(tmp_path / "audit" / "rejections.jsonl")
        entry = AdmissionRejectionAuditEntry(
            rejected_at=1_700_000_000_000,
            session_key="sess-1",
            target_scope="agent",
            scope_filter=["agent"],
            candidate={"category": "events", "abstract": "x"},
            audit={"score": 0.1},
            conversation_excerpt="hello",
        )
        append_rejection_audit(path, entry)
        # Append a second entry
        append_rejection_audit(path, entry)

        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert len(lines) == 2
        # Each line must be valid JSON with the expected keys
        parsed = json.loads(lines[0])
        assert parsed["session_key"] == "sess-1"
        assert parsed["candidate"]["category"] == "events"
