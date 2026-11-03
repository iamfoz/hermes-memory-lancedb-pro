"""
Tests for hermes_memory_lancedb_pro.extraction_prompts
"""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.extraction_prompts import (
    build_dedup_prompt,
    build_extraction_prompt,
    build_merge_prompt,
)

# ---------------------------------------------------------------------------
# TestExtractionPrompt
# ---------------------------------------------------------------------------


class TestExtractionPrompt:
    def test_conversation_text_appears(self) -> None:
        prompt = build_extraction_prompt("Alice said hello to Bob.")
        assert "Alice said hello to Bob." in prompt

    def test_user_parameter_interpolated(self) -> None:
        prompt = build_extraction_prompt("some text", user="charlie")
        assert "User: charlie" in prompt

    def test_user_parameter_default(self) -> None:
        prompt = build_extraction_prompt("some text")
        assert "User: user" in prompt

    @pytest.mark.parametrize(
        "category",
        ["profile", "preferences", "entities", "events", "cases", "patterns"],
    )
    def test_all_six_categories_mentioned(self, category: str) -> None:
        prompt = build_extraction_prompt("irrelevant")
        assert category in prompt

    def test_memories_keyword_in_json_contract(self) -> None:
        prompt = build_extraction_prompt("irrelevant")
        assert '"memories"' in prompt

    def test_few_shot_abstract_present(self) -> None:
        prompt = build_extraction_prompt("irrelevant")
        assert '"abstract"' in prompt

    def test_few_shot_overview_present(self) -> None:
        prompt = build_extraction_prompt("irrelevant")
        assert '"overview"' in prompt

    def test_few_shot_content_present(self) -> None:
        prompt = build_extraction_prompt("irrelevant")
        assert '"content"' in prompt


# ---------------------------------------------------------------------------
# TestDedupPrompt
# ---------------------------------------------------------------------------


class TestDedupPrompt:
    _EXISTING = [
        {
            "abstract": "User prefers dark mode",
            "overview": "## Preference Domain\n- Topic: UI theme",
            "category": "preferences",
            "score": 0.912,
        },
        {
            "text": "User dislikes Comic Sans",
            "category": "preferences",
            "score": 0.801,
        },
    ]

    @pytest.mark.parametrize(
        "decision",
        ["CREATE", "MERGE", "SKIP", "SUPPORT", "CONTEXTUALIZE", "CONTRADICT", "SUPERSEDE"],
    )
    def test_all_seven_decisions_listed(self, decision: str) -> None:
        prompt = build_dedup_prompt("a", "b", "c", self._EXISTING)
        assert decision in prompt

    def test_existing_memory_text_appears(self) -> None:
        prompt = build_dedup_prompt("a", "b", "c", self._EXISTING)
        # First memory uses 'abstract' key
        assert "User prefers dark mode" in prompt
        # Second memory uses 'text' key
        assert "User dislikes Comic Sans" in prompt

    def test_existing_memory_category_appears(self) -> None:
        prompt = build_dedup_prompt("a", "b", "c", self._EXISTING)
        assert "[preferences]" in prompt

    def test_candidate_abstract_appears(self) -> None:
        prompt = build_dedup_prompt("my abstract", "my overview", "my content", self._EXISTING)
        assert "my abstract" in prompt

    def test_candidate_overview_appears(self) -> None:
        prompt = build_dedup_prompt("my abstract", "my overview", "my content", self._EXISTING)
        assert "my overview" in prompt

    def test_candidate_content_appears(self) -> None:
        prompt = build_dedup_prompt("my abstract", "my overview", "my content", self._EXISTING)
        assert "my content" in prompt

    def test_match_index_mentioned(self) -> None:
        prompt = build_dedup_prompt("a", "b", "c", self._EXISTING)
        assert "match_index" in prompt

    def test_empty_existing_memories(self) -> None:
        # Should not raise — empty sequence is valid
        prompt = build_dedup_prompt("a", "b", "c", [])
        assert "match_index" in prompt

    def test_memory_numbering_in_output(self) -> None:
        prompt = build_dedup_prompt("a", "b", "c", self._EXISTING)
        assert "1. [preferences]" in prompt
        assert "2. [preferences]" in prompt


# ---------------------------------------------------------------------------
# TestMergePrompt
# ---------------------------------------------------------------------------


def _merge_prompt(**overrides: str) -> str:
    """build_merge_prompt is keyword-only — seven same-typed strings are easy to
    transpose silently, so every caller must name them."""
    kwargs = dict(
        existing_abstract="old abstract",
        existing_overview="old overview",
        existing_content="old content",
        new_abstract="new abstract",
        new_overview="new overview",
        new_content="new content",
        category="preferences",
    )
    kwargs.update(overrides)
    return build_merge_prompt(**kwargs)


class TestMergePrompt:
    def test_positional_call_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            build_merge_prompt("ea", "eo", "ec", "na", "no", "nc", "cases")  # type: ignore[misc]

    def test_existing_abstract_present(self) -> None:
        prompt = _merge_prompt()
        assert "old abstract" in prompt

    def test_existing_overview_present(self) -> None:
        prompt = _merge_prompt()
        assert "old overview" in prompt

    def test_existing_content_present(self) -> None:
        prompt = _merge_prompt()
        assert "old content" in prompt

    def test_new_abstract_present(self) -> None:
        prompt = _merge_prompt()
        assert "new abstract" in prompt

    def test_new_overview_present(self) -> None:
        prompt = _merge_prompt()
        assert "new overview" in prompt

    def test_new_content_present(self) -> None:
        prompt = _merge_prompt()
        assert "new content" in prompt

    def test_category_appears(self) -> None:
        prompt = _merge_prompt(category="entities")
        assert "entities" in prompt

    def test_output_json_contract_abstract(self) -> None:
        prompt = _merge_prompt(category="cases")
        assert '"abstract"' in prompt

    def test_output_json_contract_overview(self) -> None:
        prompt = _merge_prompt(category="cases")
        assert '"overview"' in prompt

    def test_output_json_contract_content(self) -> None:
        prompt = _merge_prompt(category="cases")
        assert '"content"' in prompt
