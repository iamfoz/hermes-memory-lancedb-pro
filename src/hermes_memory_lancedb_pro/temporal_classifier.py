"""
Temporal Classifier
Classifies memory text as static (permanent fact) or dynamic (time-sensitive).
Infers expiry timestamps from temporal expressions.
"""

from __future__ import annotations

import re
import time
from typing import Literal

# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------

TemporalType = Literal["static", "dynamic"]

# ---------------------------------------------------------------------------
# Dynamic patterns — time-sensitive indicators.
# Uses word-boundary regexes for EN to avoid substring false positives
# (e.g. "later" matching "collateral").
#
# Philosophy: prefer false-negative over false-positive here.  Classifying a
# genuinely dynamic memory as static is low-cost (it sticks around longer than
# needed).  Classifying a static fact as dynamic risks expiring it prematurely.
# So we require temporal qualifiers on ambiguous words (e.g. day names need a
# prefix like "next" / "on" / "last" rather than catching bare "Monday" which
# is often a recurring-schedule fact: "I work Monday to Friday").
# ---------------------------------------------------------------------------

_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

_DYNAMIC_PATTERNS_EN: list[re.Pattern[str]] = [
    # --- originally present ---
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bcurrently\b", re.IGNORECASE),
    re.compile(r"\bright now\b", re.IGNORECASE),
    re.compile(r"\bthis week\b", re.IGNORECASE),
    re.compile(r"\bthis month\b", re.IGNORECASE),
    re.compile(r"\blast week\b", re.IGNORECASE),
    re.compile(r"\bnext week\b", re.IGNORECASE),
    re.compile(r"\bthis morning\b", re.IGNORECASE),
    re.compile(r"\btonight\b", re.IGNORECASE),
    re.compile(r"\blater\b", re.IGNORECASE),

    # --- time of day ---
    re.compile(r"\bthis (afternoon|evening)\b", re.IGNORECASE),
    re.compile(r"\bthis weekend\b", re.IGNORECASE),

    # --- qualified day names (next/last/on/this + day) ---
    # Bare day names are intentionally excluded — "I work Monday to Friday"
    # is a static recurring schedule, not a specific time-bound event.
    re.compile(
        rf"\b(next|last|on|this)\s+({_DAYS})\b",
        re.IGNORECASE,
    ),

    # --- relative future: "in 3 days", "in 2 hours", "in a week" ---
    re.compile(r"\bin \d+ (days?|hours?|minutes?|weeks?)\b", re.IGNORECASE),
    re.compile(r"\bin an? (hour|day|week)\b", re.IGNORECASE),

    # --- relative past: "3 hours ago", "a few days ago" ---
    re.compile(r"\b\d+ (days?|hours?|minutes?) ago\b", re.IGNORECASE),
    re.compile(r"\ba few (days?|hours?) ago\b", re.IGNORECASE),

    # --- scheduling / deadline language ---
    re.compile(r"\bappointment\b", re.IGNORECASE),
    re.compile(r"\bdeadline\b", re.IGNORECASE),
    re.compile(r"\bdue (date|on|by|soon)\b", re.IGNORECASE),
    re.compile(r"\bexpires?\b", re.IGNORECASE),
    re.compile(r"\bscheduled\b", re.IGNORECASE),

    # --- explicit time-of-day: "at HH:MM am/pm" (colon required to avoid
    #     matching habitual patterns like "I always wake up at 7am") ---
    re.compile(r"\bat \d{1,2}:\d{2}\s*(am|pm)\b", re.IGNORECASE),

    # --- explicit dates: "May 7", "March 15th", "Jan. 3" ---
    # Month names (avoiding bare "May" which is also a modal verb —
    # require it to be followed by a digit).
    re.compile(
        r"\b(january|february|march|april|june|july|august"
        r"|september|october|november|december)\s+\d{1,2}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bMay\s+\d", re.IGNORECASE),
    re.compile(
        r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{1,2}\b",
        re.IGNORECASE,
    ),

    # --- ISO 8601 date: "2026-05-07" ---
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),

    # --- ordinal day-of-month: "the 15th", "on the 3rd" ---
    re.compile(r"\bthe \d{1,2}(st|nd|rd|th)\b", re.IGNORECASE),
]

_DYNAMIC_KEYWORDS_ZH: list[str] = [
    # originally present
    "今天", "昨天", "明天", "最近", "正在", "刚才", "刚刚",
    "这周", "这个月", "上周", "下周", "目前", "现在",
    "今晚", "今早", "稍后", "待会",
    # additions
    "今下午", "今天下午", "这个周末", "周末",
    "待会儿", "即将", "马上", "截止", "截止日期", "约好了",
    # variable-offset suffixes: "3天后", "2周后", "4小时后"
    "天后", "周后", "小时后",
]

# ---------------------------------------------------------------------------
# Static patterns — permanent / recurring fact indicators.
# ---------------------------------------------------------------------------

_STATIC_PATTERNS_EN: list[re.Pattern[str]] = [
    # --- originally present ---
    re.compile(r"\bfavorite\b", re.IGNORECASE),
    re.compile(r"\bprefer\b", re.IGNORECASE),
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bname is\b", re.IGNORECASE),
    re.compile(r"\bborn\b", re.IGNORECASE),
    re.compile(r"\bgraduated\b", re.IGNORECASE),
    re.compile(r"\blive in\b", re.IGNORECASE),
    re.compile(r"\bwork at\b", re.IGNORECASE),
    re.compile(r"\bjob\b", re.IGNORECASE),
    re.compile(r"\bprofession\b", re.IGNORECASE),
    re.compile(r"\bhobby\b", re.IGNORECASE),
    re.compile(r"\ballergic\b", re.IGNORECASE),

    # --- UK/variant spelling ---
    re.compile(r"\bfavourite\b", re.IGNORECASE),

    # --- habitual / recurring ---
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\busually\b", re.IGNORECASE),
    re.compile(r"\btypically\b", re.IGNORECASE),
    re.compile(r"\bgenerally\b", re.IGNORECASE),

    # --- permanent relationships ---
    re.compile(
        r"\bmy (wife|husband|partner|son|daughter"
        r"|brother|sister|mother|father|mom|dad|spouse)\b",
        re.IGNORECASE,
    ),

    # --- dietary / medical ---
    re.compile(r"\bvegetarian\b", re.IGNORECASE),
    re.compile(r"\bvegan\b", re.IGNORECASE),
    re.compile(r"\bgluten.free\b", re.IGNORECASE),
    re.compile(r"\blactose.intolerant\b", re.IGNORECASE),
    re.compile(r"\bdiabetic\b", re.IGNORECASE),

    # --- tools / skills (ongoing use) ---
    re.compile(r"\bI (use|code in|work with|write in)\b", re.IGNORECASE),
    re.compile(r"\bmy (editor|IDE|setup|stack)\b", re.IGNORECASE),
]

_STATIC_KEYWORDS_ZH: list[str] = [
    # originally present
    "喜欢", "偏好", "一直", "名字", "叫做", "出生",
    "毕业", "住在", "工作", "职业", "爱好", "过敏",
    # additions
    "妻子", "老婆", "丈夫", "老公", "儿子", "女儿",
    "兄弟", "姐妹", "父母", "通常", "从不", "一般",
    "素食", "纯素", "乳糖不耐",
]

# ---------------------------------------------------------------------------
# Expiry rules: fixed offsets.
# Order matters — more specific patterns must come before more general ones
# (e.g. "day after tomorrow" before "tomorrow").
# ---------------------------------------------------------------------------

_MS_PER_HOUR = 60 * 60 * 1000
_MS_PER_DAY = 24 * _MS_PER_HOUR

_EXPIRY_RULES: list[tuple[list[re.Pattern[str]], int]] = [
    # 后天 / day after tomorrow → +48h
    (
        [re.compile(r"后天"), re.compile(r"day after tomorrow", re.IGNORECASE)],
        48 * _MS_PER_HOUR,
    ),
    # 明天 / tomorrow → +24h
    (
        [re.compile(r"明天"), re.compile(r"\btomorrow\b", re.IGNORECASE)],
        24 * _MS_PER_HOUR,
    ),
    # next DAYNAME → +7 days (approximate; avoids computing exact calendar offset)
    (
        [re.compile(rf"\bnext\s+({_DAYS})\b", re.IGNORECASE)],
        7 * _MS_PER_DAY,
    ),
    # 下周 / next week → +7d
    (
        [re.compile(r"下周"), re.compile(r"\bnext week\b", re.IGNORECASE)],
        7 * _MS_PER_DAY,
    ),
    # on DAYNAME / this DAYNAME → +4 days (midpoint within the current week)
    (
        [re.compile(rf"\b(on|this)\s+({_DAYS})\b", re.IGNORECASE)],
        4 * _MS_PER_DAY,
    ),
    # 这周 / this week → +3d
    (
        [re.compile(r"这周"), re.compile(r"\bthis week\b", re.IGNORECASE)],
        3 * _MS_PER_DAY,
    ),
    # this weekend → +4d
    (
        [re.compile(r"这个周末"), re.compile(r"\bthis weekend\b", re.IGNORECASE)],
        4 * _MS_PER_DAY,
    ),
    # 下个月 / next month → +30d
    (
        [re.compile(r"下个月"), re.compile(r"\bnext month\b", re.IGNORECASE)],
        30 * _MS_PER_DAY,
    ),
    # 这个月 / this month → +15d
    (
        [re.compile(r"这个月"), re.compile(r"\bthis month\b", re.IGNORECASE)],
        15 * _MS_PER_DAY,
    ),
    # 今晚 / tonight → +12h
    (
        [re.compile(r"今晚"), re.compile(r"\btonight\b", re.IGNORECASE)],
        12 * _MS_PER_HOUR,
    ),
    # 今天 / today / this morning / this afternoon / this evening → +18h
    (
        [
            re.compile(r"今天"), re.compile(r"今早"), re.compile(r"今下午"),
            re.compile(r"\btoday\b", re.IGNORECASE),
            re.compile(r"\bthis (morning|afternoon|evening)\b", re.IGNORECASE),
        ],
        18 * _MS_PER_HOUR,
    ),
]

# ---------------------------------------------------------------------------
# Variable-offset expiry patterns: "in N days", "in N hours", "in N weeks".
# Each tuple is (compiled_pattern, ms_per_unit).  The pattern must have
# exactly one capture group containing the integer multiplier.
# ---------------------------------------------------------------------------

_VARIABLE_EXPIRY_RULES: list[tuple[re.Pattern[str], int]] = [
    # "in 3 days" / "in 3 day"
    (re.compile(r"\bin (\d+) days?\b", re.IGNORECASE), _MS_PER_DAY),
    # "in 2 weeks"
    (re.compile(r"\bin (\d+) weeks?\b", re.IGNORECASE), 7 * _MS_PER_DAY),
    # "in 4 hours"
    (re.compile(r"\bin (\d+) hours?\b", re.IGNORECASE), _MS_PER_HOUR),
    # Chinese variants: "3天后", "2周后", "4小时后"
    (re.compile(r"(\d+)天后"), _MS_PER_DAY),
    (re.compile(r"(\d+)周后"), 7 * _MS_PER_DAY),
    (re.compile(r"(\d+)小时后"), _MS_PER_HOUR),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["TemporalType", "classify_temporal", "infer_expiry"]


def classify_temporal(text: str) -> TemporalType:
    """Classify memory text as static (permanent fact) or dynamic (time-sensitive).

    Rule-based: keywords → classification.  Default: ``"static"`` (safer —
    keeps a memory around longer rather than expiring it prematurely).

    If **both** dynamic and static patterns match, ``"dynamic"`` wins because
    time-sensitive information takes priority (e.g. "My favourite meeting is
    tomorrow" — the fact that it's tomorrow matters more than the preference).
    """
    has_dynamic = any(p.search(text) for p in _DYNAMIC_PATTERNS_EN) or any(
        kw in text for kw in _DYNAMIC_KEYWORDS_ZH
    )
    has_static = any(p.search(text) for p in _STATIC_PATTERNS_EN) or any(
        kw in text for kw in _STATIC_KEYWORDS_ZH
    )

    if has_dynamic:
        return "dynamic"
    if has_static:
        return "static"
    return "static"


def infer_expiry(text: str, now_ms: int | None = None) -> int | None:
    """Infer expiry timestamp (epoch-ms) from temporal expressions in text.

    Returns ``None`` if no recognised temporal expression is found.

    Checks variable-multiplier rules first (``"in N days"`` etc.), then
    fixed-offset rules (``"tomorrow"``, ``"next week"`` etc.) in specificity
    order.

    Args:
        text:   memory text to inspect.
        now_ms: base timestamp in milliseconds (default: current wall-clock).
    """
    base_time = now_ms if now_ms is not None else int(time.time() * 1000)

    # 1. Variable-offset rules ("in 3 days", "2周后", …)
    for pattern, ms_per_unit in _VARIABLE_EXPIRY_RULES:
        m = pattern.search(text)
        if m:
            try:
                n = int(m.group(1))
            except (IndexError, ValueError):
                continue
            return base_time + n * ms_per_unit

    # 2. Fixed-offset rules (most specific first)
    for patterns, offset_ms in _EXPIRY_RULES:
        for pattern in patterns:
            if pattern.search(text):
                return base_time + offset_ms

    return None
