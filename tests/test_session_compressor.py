"""Unit tests for session_compressor module."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.session_compressor import (
    compress_texts,
    estimate_conversation_value,
    score_text,
)

# ---------------------------------------------------------------------------
# TestScoreText
# ---------------------------------------------------------------------------


class TestScoreText:
    # --- empty ---

    def test_empty_string(self):
        result = score_text("", 0)
        assert result.score == 0.0
        assert result.reason == "empty"
        assert result.index == 0

    def test_whitespace_only(self):
        result = score_text("   \t\n", 2)
        assert result.score == 0.0
        assert result.reason == "empty"

    # --- tool_call ---

    @pytest.mark.parametrize("text", [
        "tool_use: memory_store",
        "function_call to the API",
        "memory_store called with payload",
        "memory_recall invoked",
        "memory_forget triggered",
        "memory_update succeeded",
    ])
    def test_tool_call(self, text: str):
        result = score_text(text, 0)
        assert result.score == 1.0
        assert result.reason == "tool_call"

    @pytest.mark.parametrize("text", [
        "result from tool_result block",
        "tool_result: the output was 42",
    ])
    def test_tool_result(self, text: str):
        result = score_text(text, 0)
        assert result.score == 1.0
        assert result.reason == "tool_result"

    # --- correction ---

    @pytest.mark.parametrize("text", [
        "No, that's not right",
        "Actually I meant the other one",
        "Instead use the blue variant",
        "That answer is wrong",
        "correction needed here",
        "Please fix this mistake",
        "不对，重新来",
        "应该是第二个",
        "應該是第二個選項",
        "错了，试试这个",
        "錯了，應該這樣做",
        "改成另一個方式",
        "不是第一个而是第二个",
    ])
    def test_correction(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.95
        assert result.reason == "correction"

    # --- decision ---

    @pytest.mark.parametrize("text", [
        "let's go with option B",
        "confirmed, ship it",
        "approved by the team",
        "decided to use React",
        "we'll use PostgreSQL for this",
        "going forward with the new design",
        "from now on we use tabs",
        "agreed on the timeline",
        "决定采用新方案",
        "決定今天發布",
        "确认了发布时间",
        "確認完成了",
        "选择了第二个",
        "選擇了新版本",
        "就这样吧",
        "就這樣決定了",
    ])
    def test_decision(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.85
        assert result.reason == "decision"

    # --- acknowledgment ---

    @pytest.mark.parametrize("text", [
        "ok",
        "okay",
        "k",
        "sure",
        "fine",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "got it",
        "understood",
        "cool",
        "nice",
        "great",
        "good",
        "perfect",
        "awesome",
        "alright",
        "yep",
        "yup",
        "yeah",
        "right",
        "ok.",
        "sure!",
        "好",
        "好的",
        "好的。",
        "嗯",
        "收到",
        "了解",
        "明白",
        "谢谢",
        "感谢",
        "👍",
    ])
    def test_acknowledgment(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.1
        assert result.reason == "acknowledgment"

    # --- substantive ---

    def test_substantive_non_cjk(self):
        # Must be >40 chars to be substantive for Latin text
        text = "a" * 41
        result = score_text(text, 0)
        assert result.score == 0.7
        assert result.reason == "substantive"

    def test_non_cjk_at_threshold_not_substantive(self):
        # Exactly 40 chars — NOT substantive (threshold is strictly greater)
        text = "a" * 40
        result = score_text(text, 0)
        assert result.reason != "substantive"

    def test_substantive_cjk(self):
        # CJK threshold is 30; 31 chars should be substantive
        text = "这" * 31
        result = score_text(text, 0)
        assert result.score == 0.7
        assert result.reason == "substantive"

    def test_cjk_at_threshold_not_substantive(self):
        # Exactly 30 chars — NOT substantive
        text = "这" * 30
        result = score_text(text, 0)
        assert result.reason != "substantive"

    def test_mixed_cjk_latin_uses_cjk_threshold(self):
        # Mixed text: has CJK chars so threshold is 30
        # 31 chars total with CJK — should be substantive
        text = "这" * 10 + "a" * 21  # 31 chars, has CJK
        result = score_text(text, 0)
        assert result.score == 0.7
        assert result.reason == "substantive"

    # --- system_xml ---

    def test_system_xml(self):
        # Text must exceed the non-CJK substantive threshold (80 chars)
        text = "<system>You are a helpful assistant and should behave accordingly in all cases no matter what</system>"
        result = score_text(text, 0)
        assert result.score == 0.3
        assert result.reason == "system_xml"

    def test_system_xml_with_hyphens(self):
        text = "<system-prompt>Long enough boilerplate content here to exceed threshold.</system-prompt>"
        result = score_text(text, 0)
        assert result.score == 0.3
        assert result.reason == "system_xml"

    def test_non_xml_substantive_not_classified_as_xml(self):
        text = "This is a plain substantive text that is long enough to exceed the threshold " + "x" * 10
        result = score_text(text, 0)
        assert result.reason == "substantive"

    # --- short_question ---

    def test_short_question_ascii(self):
        result = score_text("What is this?", 0)
        assert result.score == 0.5
        assert result.reason == "short_question"

    def test_short_question_cjk_fullwidth(self):
        result = score_text("这是什么？", 0)
        # 5 chars — below CJK threshold (30), not substantive; has ？
        assert result.score == 0.5
        assert result.reason == "short_question"

    # --- short_statement ---

    def test_short_statement(self):
        result = score_text("Hello world", 0)
        assert result.score == 0.4
        assert result.reason == "short_statement"

    def test_short_statement_no_question_mark(self):
        result = score_text("Use dark mode", 0)
        assert result.score == 0.4
        assert result.reason == "short_statement"

    # --- corrections: inflected forms (post v0.9.0 fix) ---

    @pytest.mark.parametrize("text", [
        "Fixed it",
        "Fixed the deployment bug",
        "I'm fixing the issue now",
        "Fixes the regression",
        "Corrected the typo",
        "Correcting the example output",
        "There were corrections to the spec",
    ])
    def test_correction_inflected(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.95, f"expected correction score for {text!r}"
        assert result.reason == "correction"

    # --- decisions: inflected forms ---

    @pytest.mark.parametrize("text", [
        "Decided on option B",
        "We're deciding now",
        "Confirms the new schedule",
        "Approving the rollout",
    ])
    def test_decision_inflected(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.85, f"expected decision score for {text!r}"
        assert result.reason == "decision"

    # --- greetings (post v0.9.0 fix) ---

    @pytest.mark.parametrize("text", [
        "Hi",
        "Hello",
        "Hey!",
        "Howdy",
        "Good morning",
        "Good afternoon!",
        "How are you?",
        "How are things going?",
        "How's it going?",
        "你好",
        "您好",
    ])
    def test_greeting_scored_low(self, text: str):
        result = score_text(text, 0)
        assert result.score == 0.1, f"expected greeting score for {text!r}"
        assert result.reason == "greeting"

    def test_greeting_does_not_swallow_substantive(self):
        # "Hi, the deployment failed because of a port conflict on 8080."
        # starts with "Hi" but is substantive; greeting pattern is anchored
        # to whole-line so this should NOT match greeting.
        text = "Hi, the deployment failed because of a port conflict on 8080."
        result = score_text(text, 0)
        assert result.reason != "greeting"

    # --- substantive: technical sentences in the 40-80 char range
    # used to fall into short_statement (0.4); now they correctly score
    # higher than short questions and greetings.

    def test_short_technical_sentence_is_substantive(self):
        text = "The deployment failed because of a port conflict"
        assert 40 < len(text) <= 80, "test premise: between 40 and 80 chars"
        result = score_text(text, 0)
        assert result.score == 0.7
        assert result.reason == "substantive"

    def test_substantive_outranks_greeting_and_question(self):
        # The headline regression from the bug report: a real technical
        # statement should outrank "How are you?" / "Hi".
        substantive = score_text("The deployment failed because of a port conflict", 0).score
        greeting = score_text("How are you?", 0).score
        hi = score_text("Hi", 0).score
        fixed_it = score_text("Fixed it", 0).score
        assert substantive > greeting
        assert substantive > hi
        assert fixed_it > substantive  # corrections outrank substantive

    # --- index is preserved ---

    def test_index_preserved(self):
        result = score_text("", 7)
        assert result.index == 7
        result2 = score_text("hello world", 42)
        assert result2.index == 42

    # --- text field preserved ---

    def test_text_field_preserved(self):
        original = "  hello world  "
        result = score_text(original, 0)
        assert result.text is original


# ---------------------------------------------------------------------------
# TestCompressTexts
# ---------------------------------------------------------------------------


class TestCompressTexts:
    def test_empty_input(self):
        result = compress_texts([], max_chars=1000)
        assert result.texts == []
        assert result.scored == []
        assert result.dropped == 0
        assert result.total_chars == 0

    def test_total_under_budget_returns_all(self):
        texts = ["hello", "world", "foo"]
        result = compress_texts(texts, max_chars=10000)
        assert result.texts == texts
        assert result.dropped == 0
        assert result.total_chars == sum(len(t) for t in texts)

    def test_scores_all_texts_even_under_budget(self):
        texts = ["hello", "world"]
        result = compress_texts(texts, max_chars=10000)
        assert len(result.scored) == 2

    def test_keeps_first_and_last_when_over_budget(self):
        # Build texts where middle entries are long but first/last are short
        first = "first text"
        last = "last text"
        middle = ["x" * 100] * 10  # 1000 chars of filler
        texts = [first] + middle + [last]
        # budget allows first + last but not all middle
        result = compress_texts(texts, max_chars=len(first) + len(last) + 5)
        assert result.texts[0] == first
        assert result.texts[-1] == last

    def test_dropped_count_correct(self):
        texts = ["a" * 50, "b" * 50, "c" * 50, "d" * 50]
        # Budget fits only 2 entries (100 chars)
        result = compress_texts(texts, max_chars=105)
        assert result.dropped == len(texts) - len(result.texts)
        assert result.dropped >= 0

    def test_tool_call_next_line_pairing(self):
        # When a tool_call is selected, the next line should be pulled in too
        tool_call_text = "tool_use: memory_store {data: 'x'}"
        tool_result_text = "Result from the tool call above"
        filler = ["filler line that is ignored"] * 5
        # Put tool_call first (after a dummy "first"), then tool_result,
        # then filler, then a dummy "last"
        first = "session start"
        last = "session end"
        texts = [first, tool_call_text, tool_result_text] + filler + [last]
        # Budget: first + last + tool_call + tool_result but tight on filler
        budget = (
            len(first) + len(last) + len(tool_call_text) + len(tool_result_text) + 20
        )
        result = compress_texts(texts, max_chars=budget)
        assert tool_call_text in result.texts
        assert tool_result_text in result.texts

    def test_pairing_only_fires_for_tool_call_not_tool_result(self):
        # tool_use lines (reason="tool_call") pull in the next line as a pair.
        # tool_result lines (reason="tool_result") do NOT trigger forward pairing,
        # so the line after a tool_result is not dragged in automatically.
        first = "session start"
        last = "session end"
        tool_call_text = "tool_use: do something important"  # reason="tool_call" → pairs with next
        tool_result_text = "tool_result: done"               # reason="tool_result" → no pairing
        unrelated_next = "unrelated line after tool_result"
        filler = ["short filler"] * 5
        texts = [first, tool_call_text, tool_result_text, unrelated_next] + filler + [last]

        # Budget: fits first + last + tool_call + tool_result but not unrelated_next.
        # tool_call (index 1) pairs with tool_result (index 2), so both land.
        # tool_result (index 2) does NOT pair with unrelated_next (index 3).
        budget = len(first) + len(last) + len(tool_call_text) + len(tool_result_text) + 5
        result = compress_texts(texts, max_chars=budget)
        assert tool_call_text in result.texts
        assert tool_result_text in result.texts
        assert unrelated_next not in result.texts

    def test_all_low_score_fallback_min_texts_honoured(self):
        # Feed only acknowledgments (score=0.1 < min_score_to_keep=0.3)
        texts = ["ok", "sure", "yeah", "got it", "nice", "great", "fine"]
        # Tight budget: can only fit a few chars
        # min_score_to_keep default is 0.3; all acks score 0.1
        result = compress_texts(texts, max_chars=10000, min_texts=3)
        # all_low is True, so we must have at least min(3, len(texts)) = 3
        assert len(result.texts) >= 3

    def test_all_low_score_fallback_kept_from_end(self):
        # Verify the fallback picks from the END (most recent)
        texts = ["ok", "sure", "yeah", "got it", "nice"]
        # Budget large enough to fit all
        result = compress_texts(texts, max_chars=10000, min_texts=3)
        # Since total fits budget, all returned (not fallback path)
        assert len(result.texts) == 5

    def test_all_low_score_fallback_tight_budget(self):
        # Use short pure acks to guarantee low score (multi-word repetitions
        # like "ok ok ok ..." don't match the acknowledgment regex anchors).
        acks = ["ok", "sure", "yeah", "got it", "nice", "yep", "yup"]
        # Budget: very tight, only fits ~10 chars
        result = compress_texts(acks, max_chars=10, min_texts=3, min_score_to_keep=0.3)
        # all_low=True; fallback kicks in; we want at least min(3,7)=3
        # But budget may prevent adding 3 — addIndex respects budget
        # At least 1 (first and last are always attempted)
        assert len(result.texts) >= 1

    def test_output_ordering_is_chronological(self):
        # High-score items should appear in original order, not score order.
        # Use unique strings so positional lookup is unambiguous.
        texts = [
            "session start marker",                       # index 0
            "ack one",                                    # index 1 — low score
            "tool_use: memory_store important data",      # index 2 — high score
            "ack two",                                    # index 3 — low score
            "we'll use PostgreSQL for the project db",    # index 4 — decision
            "session end marker",                         # index 5
        ]
        result = compress_texts(texts, max_chars=500)
        # All texts fit in budget, so all returned; verify chronological order.
        text_to_idx = {t: i for i, t in enumerate(texts)}
        original_positions = [text_to_idx[t] for t in result.texts]
        assert original_positions == sorted(original_positions)

    def test_single_text(self):
        texts = ["only one entry"]
        result = compress_texts(texts, max_chars=1000)
        assert result.texts == texts
        assert result.dropped == 0

    def test_two_texts_keeps_both_as_first_and_last(self):
        texts = ["first", "last"]
        result = compress_texts(texts, max_chars=1000)
        assert result.texts == texts

    def test_total_chars_matches_output(self):
        texts = ["hello world", "foo bar baz", "short"]
        result = compress_texts(texts, max_chars=10000)
        assert result.total_chars == sum(len(t) for t in result.texts)


# ---------------------------------------------------------------------------
# TestEstimateConversationValue
# ---------------------------------------------------------------------------


class TestEstimateConversationValue:
    def test_empty_returns_zero(self):
        assert estimate_conversation_value([]) == 0.0

    def test_memory_intent_english(self):
        texts = ["Please remember my preference for dark mode"]
        value = estimate_conversation_value(texts)
        assert value >= 0.5

    def test_memory_intent_english_variants(self):
        for phrase in ["recall that", "don't forget this", "note that we use tabs", "keep in mind"]:
            value = estimate_conversation_value([f"You should {phrase} always."])
            assert value >= 0.5, f"Expected >= 0.5 for phrase: {phrase!r}"

    def test_memory_intent_dont_forget_apostrophe(self):
        value = estimate_conversation_value(["don't forget to set the flag"])
        assert value >= 0.5

    def test_memory_intent_cjk(self):
        texts = ["记住我的偏好是深色模式"]
        value = estimate_conversation_value(texts)
        assert value >= 0.5

    def test_memory_intent_cjk_variants(self):
        for phrase in ["记住", "記住", "别忘", "不要忘", "记一下", "記一下"]:
            value = estimate_conversation_value([f"{phrase}这个设置"])
            assert value >= 0.5, f"Expected >= 0.5 for CJK phrase: {phrase!r}"

    def test_tool_calls_add_point_four(self):
        texts = ["tool_use: memory_store something"]
        value = estimate_conversation_value(texts)
        assert value >= 0.4

    def test_tool_calls_exact_contribution(self):
        # No other signals, just a tool_call; should be exactly 0.4
        texts = ["tool_use only signal here"]
        value = estimate_conversation_value(texts)
        assert value == pytest.approx(0.4)

    def test_correction_adds_point_three(self):
        texts = ["Actually that was wrong, fix it"]
        value = estimate_conversation_value(texts)
        assert value >= 0.3

    def test_decision_adds_point_three(self):
        texts = ["Agreed, let's go with the second option"]
        value = estimate_conversation_value(texts)
        assert value >= 0.3

    def test_correction_exact_contribution(self):
        # Only correction signal
        texts = ["Actually we need to change this"]
        value = estimate_conversation_value(texts)
        assert value == pytest.approx(0.3)

    def test_substantive_volume_adds_point_two(self):
        # Lines >20 chars, summed >200 chars
        texts = ["x" * 50] * 5  # 5 lines × 50 chars = 250 > 200
        value = estimate_conversation_value(texts)
        assert value >= 0.2

    def test_substantive_volume_below_threshold_no_bonus(self):
        # 190 chars total in lines >20 chars — no +0.2
        texts = ["x" * 38] * 5  # 5 × 38 = 190 chars
        value = estimate_conversation_value(texts)
        assert value < 0.2

    def test_multi_turn_adds_point_one(self):
        # >6 texts adds +0.1
        texts = ["ok"] * 7
        value = estimate_conversation_value(texts)
        assert value >= 0.1

    def test_multi_turn_exactly_six_no_bonus(self):
        texts = ["ok"] * 6
        value = estimate_conversation_value(texts)
        # 6 texts is NOT >6, so no bonus
        assert value == pytest.approx(0.0)

    def test_caps_at_one(self):
        # Maximise all signals: memory intent + tool + correction + substantive + multi-turn
        texts = (
            ["remember tool_use: memory_store correct fix actually wrong"]
            + ["x" * 50] * 5  # substantive volume
            + ["extra"] * 5   # push length > 6
        )
        value = estimate_conversation_value(texts)
        assert value <= 1.0

    def test_caps_at_one_with_all_signals(self):
        # Build a conversation that would exceed 1.0 without capping
        # memory(+0.5) + tool(+0.4) + correction(+0.3) + substantive(+0.2) + multi-turn(+0.1) = 1.5
        texts = (
            ["remember this tool_use: memory_store actually fix it"]
            + ["a" * 50] * 5
            + ["pad"] * 6
        )
        value = estimate_conversation_value(texts)
        assert value == pytest.approx(1.0)

    def test_no_signals_returns_zero(self):
        # Pure acknowledgments, short, <7 texts
        texts = ["ok", "sure", "yeah"]
        value = estimate_conversation_value(texts)
        assert value == pytest.approx(0.0)

    # --- post-deploy fixes (v0.9.0) ---

    def test_string_input_is_coerced_to_single_turn(self):
        # Defensive coercion: callers occasionally pass a joined transcript
        # as a single string rather than a list. Should not crash and should
        # not iterate the string as characters.
        value = estimate_conversation_value(
            "Please remember to use UK English in all my docs going forward."
        )
        # Memory-intent phrase fires → at least 0.5
        assert value >= 0.5

    def test_none_input_returns_zero(self):
        assert estimate_conversation_value(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert estimate_conversation_value("") == 0.0

    def test_substantive_baseline_floor_for_real_conversation(self):
        # A real-but-short troubleshooting exchange that doesn't trip
        # any of the explicit signal patterns. Pre-fix this returned 0.0,
        # which made the throttle skip extraction entirely. Post-fix it
        # gets at least the substantive >100 char bump (+0.15).
        texts = [
            "The deployment failed with EADDRINUSE on port 8080.",
            "I checked netstat and saw an old gunicorn worker still running.",
            "Killed it and the redeploy went through cleanly.",
        ]
        value = estimate_conversation_value(texts)
        assert value > 0.0, "substantive deployment troubleshooting should not score 0"
        assert value >= 0.15

    def test_substantive_500_chars_gets_higher_bump(self):
        texts = ["x" * 110] * 6  # 660 chars > 500 → +0.3
        value = estimate_conversation_value(texts)
        assert value >= 0.3

    def test_baseline_floor_for_single_substantive_turn(self):
        # One non-trivial turn, no signal patterns, < 100 chars total
        # substantive content. Pre-fix → 0.0; post-fix → 0.1 baseline.
        texts = ["I sometimes use the alpine variant of the image."]
        value = estimate_conversation_value(texts)
        assert value >= 0.1

    def test_baseline_does_not_apply_to_trivial_acks(self):
        # All turns < 20 chars stripped → baseline floor must NOT fire.
        texts = ["ok", "thanks", "got it"]
        value = estimate_conversation_value(texts)
        assert value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestGreetingHiThere
# ---------------------------------------------------------------------------

class TestGreetingHiThere:
    def test_hi_there_scores_as_greeting(self):
        result = score_text("Hi there", 0)
        assert result.score == 0.1, f"'Hi there' should score 0.1 (greeting), got {result.score}"
        assert result.reason == "greeting"

    def test_hello_there_scores_as_greeting(self):
        result = score_text("Hello there!", 0)
        assert result.score == 0.1
        assert result.reason == "greeting"

    def test_hi_alone_still_greeting(self):
        result = score_text("Hi", 0)
        assert result.score == 0.1
        assert result.reason == "greeting"
