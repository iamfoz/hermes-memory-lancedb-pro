"""Tests for temporal_classifier — ported from cortexreach-memory-lancedb-pro TS spec."""

from __future__ import annotations

import pytest

from hermes_memory_lancedb_pro.temporal_classifier import classify_temporal, infer_expiry

# Fixed base time for deterministic expiry assertions
NOW_MS = 1_700_000_000_000

_MS_PER_HOUR = 60 * 60 * 1000
_MS_PER_DAY = 24 * _MS_PER_HOUR


# ---------------------------------------------------------------------------
# classify_temporal — dynamic English keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I have a meeting today",
        "I saw her yesterday",
        "Let's do it tomorrow",
        "I recently moved to a new city",
        "I am currently working on a project",
        "Right now I feel tired",
        "I'll finish it this week",
        "We'll ship this month",
        "Last week we had a sprint",
        "Next week is the deadline",
        "This morning I went for a run",
        "Let's catch up tonight",
        "I'll handle it later",
    ],
)
def test_dynamic_en_keywords(text: str) -> None:
    assert classify_temporal(text) == "dynamic"


# ---------------------------------------------------------------------------
# classify_temporal — dynamic Chinese keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "我今天很忙",       # 今天
        "她昨天来过",       # 昨天
        "明天见",           # 明天
        "最近工作很多",     # 最近
        "正在处理中",       # 正在
        "刚才说过了",       # 刚才
        "刚刚到家",         # 刚刚
        "这周很忙",         # 这周
        "这个月计划多",     # 这个月
        "上周开会了",       # 上周
        "下周出差",         # 下周
        "目前状态良好",     # 目前
        "现在出发",         # 现在
        "今晚有活动",       # 今晚
        "今早吃了早饭",     # 今早
        "稍后回复你",       # 稍后
        "待会见",           # 待会
    ],
)
def test_dynamic_zh_keywords(text: str) -> None:
    assert classify_temporal(text) == "dynamic"


# ---------------------------------------------------------------------------
# classify_temporal — static English keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "My favorite color is blue",
        "I prefer tea over coffee",
        "I always wake up at 7am",
        "My name is Alice",
        "I was born in 1990",
        "I graduated from MIT",
        "I live in Seattle",
        "I work at Google",
        "My job is software engineer",
        "Her profession is medicine",
        "My hobby is painting",
        "I am allergic to peanuts",
    ],
)
def test_static_en_keywords(text: str) -> None:
    assert classify_temporal(text) == "static"


# ---------------------------------------------------------------------------
# classify_temporal — static Chinese keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "我喜欢编程",       # 喜欢
        "我偏好安静",       # 偏好
        "我一直住这里",     # 一直
        "我的名字叫小明",   # 名字
        "我叫做小华",       # 叫做
        "我出生在北京",     # 出生
        "我毕业于清华",     # 毕业
        "我住在上海",       # 住在
        "我工作很忙",       # 工作
        "我的职业是医生",   # 职业
        "我的爱好是跑步",   # 爱好
        "我对花粉过敏",     # 过敏
    ],
)
def test_static_zh_keywords(text: str) -> None:
    assert classify_temporal(text) == "static"


# ---------------------------------------------------------------------------
# classify_temporal — precedence and default
# ---------------------------------------------------------------------------


def test_both_present_dynamic_wins() -> None:
    """When both dynamic and static keywords appear, dynamic wins."""
    assert classify_temporal("My favorite restaurant is full today") == "dynamic"


def test_both_present_zh_dynamic_wins() -> None:
    """Chinese: dynamic keyword overrides static keyword."""
    assert classify_temporal("我喜欢今天的天气") == "dynamic"


def test_neither_present_defaults_static() -> None:
    """When neither dynamic nor static keywords are found, default is static."""
    assert classify_temporal("The sky is blue and the grass is green") == "static"


def test_empty_string_defaults_static() -> None:
    assert classify_temporal("") == "static"


def test_word_boundary_no_false_positive_later() -> None:
    """'later' inside 'collateral' must not trigger dynamic."""
    assert classify_temporal("This is collateral damage") == "static"


# ---------------------------------------------------------------------------
# infer_expiry — English phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_offset_ms",
    [
        ("Let's meet tomorrow", 24 * _MS_PER_HOUR),
        ("day after tomorrow is free", 48 * _MS_PER_HOUR),
        ("next week we'll finish", 7 * _MS_PER_DAY),
        ("this week is busy", 3 * _MS_PER_DAY),
        ("next month we'll launch", 30 * _MS_PER_DAY),
        ("this month is packed", 15 * _MS_PER_DAY),
        ("let's do it tonight", 12 * _MS_PER_HOUR),
        ("we spoke today", 18 * _MS_PER_HOUR),
    ],
)
def test_infer_expiry_en(text: str, expected_offset_ms: int) -> None:
    result = infer_expiry(text, now_ms=NOW_MS)
    assert result == NOW_MS + expected_offset_ms


# ---------------------------------------------------------------------------
# infer_expiry — Chinese phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_offset_ms",
    [
        ("明天见", 24 * _MS_PER_HOUR),
        ("后天有空", 48 * _MS_PER_HOUR),
        ("下周开会", 7 * _MS_PER_DAY),
        ("这周很忙", 3 * _MS_PER_DAY),
        ("下个月发布", 30 * _MS_PER_DAY),
        ("这个月计划多", 15 * _MS_PER_DAY),
        ("今晚有活动", 12 * _MS_PER_HOUR),
        ("今天开心", 18 * _MS_PER_HOUR),
    ],
)
def test_infer_expiry_zh(text: str, expected_offset_ms: int) -> None:
    result = infer_expiry(text, now_ms=NOW_MS)
    assert result == NOW_MS + expected_offset_ms


# ---------------------------------------------------------------------------
# infer_expiry — no match returns None
# ---------------------------------------------------------------------------


def test_infer_expiry_no_temporal_phrase() -> None:
    assert infer_expiry("The weather is nice", now_ms=NOW_MS) is None


def test_infer_expiry_empty_string() -> None:
    assert infer_expiry("", now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# infer_expiry — now_ms is respected (not wall clock)
# ---------------------------------------------------------------------------


def test_infer_expiry_uses_provided_now_ms() -> None:
    """Offset must be added to provided now_ms, not the real clock."""
    custom_now = 1_600_000_000_000
    result = infer_expiry("tomorrow", now_ms=custom_now)
    assert result == custom_now + 24 * _MS_PER_HOUR


def test_infer_expiry_different_now_ms_values() -> None:
    offset = 7 * _MS_PER_DAY
    for base in (0, 1_000_000, 1_700_000_000_000, 9_999_999_999_999):
        assert infer_expiry("next week", now_ms=base) == base + offset


# ---------------------------------------------------------------------------
# infer_expiry — ordering: day-after-tomorrow before tomorrow
# ---------------------------------------------------------------------------


def test_day_after_tomorrow_beats_tomorrow_rule() -> None:
    """'day after tomorrow' must match the 48h rule, not fall through to 24h."""
    result = infer_expiry("day after tomorrow", now_ms=NOW_MS)
    assert result == NOW_MS + 48 * _MS_PER_HOUR


def test_zh_day_after_tomorrow_beats_tomorrow_rule() -> None:
    result = infer_expiry("后天明天", now_ms=NOW_MS)
    # 后天 appears first in rules (48h), so that should win
    assert result == NOW_MS + 48 * _MS_PER_HOUR
