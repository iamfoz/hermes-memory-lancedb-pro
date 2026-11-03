"""
Temporal Classifier
Classifies memory text as static (permanent fact) or dynamic (time-sensitive).
Infers expiry timestamps from temporal expressions.

Design rule: prefer a false negative over a false positive. Misclassifying a
time-bound memory as static costs little (it lingers a while longer).
Misclassifying a permanent fact as dynamic makes it decay three times faster
and may stamp it with an expiry. So:

* **Strong** dynamic signals (``tomorrow``, ``next Friday``, ``deadline``)
  always classify as dynamic.
* **Weak** dynamic signals — bare relative durations such as ``in 10 days`` or
  ``3天后`` — only count when the text has no habitual marker
  (``usually``, ``every week``, ``通常`` …). "I can usually finish a sprint in
  10 days" is a permanent fact about the user, not an event.
* Deliberately **not** matched, because they fire on permanent facts: bare
  calendar dates and ISO dates (birthdays), ordinals ("the 5th edition"),
  ``on <day>`` ("I go to the gym on Monday"), and bare ``appointment`` /
  ``scheduled`` / ``expires`` ("monthly dentist appointment").
"""

from __future__ import annotations

import re
import time
from typing import Literal

# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------

TemporalType = Literal["static", "dynamic"]

_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Chinese numerals that commonly precede a duration ("三天后", "两周后").
_ZH_NUM = r"[0-9一二两三四五六七八九十几]+"

# ---------------------------------------------------------------------------
# Strong dynamic patterns — always time-sensitive.
# Word-boundary regexes for EN avoid substring false positives
# (e.g. "later" matching "collateral").
# ---------------------------------------------------------------------------

_DYNAMIC_PATTERNS_EN: list[re.Pattern[str]] = [
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bcurrent(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bright now\b", re.IGNORECASE),
    re.compile(r"\bthis week\b", re.IGNORECASE),
    re.compile(r"\bthis month\b", re.IGNORECASE),
    re.compile(r"\blast week\b", re.IGNORECASE),
    re.compile(r"\bnext week\b", re.IGNORECASE),
    re.compile(r"\bthis morning\b", re.IGNORECASE),
    re.compile(r"\btonight\b", re.IGNORECASE),
    re.compile(r"\blater\b", re.IGNORECASE),
    # Time of day / weekend within the current week.
    re.compile(r"\bthis (?:afternoon|evening|weekend)\b", re.IGNORECASE),
    # "next Friday" — always one specific upcoming day. ("on Monday" and
    # "last Friday" are excluded: both are common in recurring schedules.)
    re.compile(rf"\bnext (?:{_DAYS})\b", re.IGNORECASE),
    # Relative past: "3 hours ago", "a few days ago".
    re.compile(r"\b\d+ (?:minutes?|hours?|days?|weeks?) ago\b", re.IGNORECASE),
    re.compile(r"\ba few (?:minutes|hours|days|weeks) ago\b", re.IGNORECASE),
    # Deadlines. "due" requires a qualifier ("due on", "due by", "due date").
    re.compile(r"\bdeadline\b", re.IGNORECASE),
    re.compile(r"\bdue (?:date|on|by|soon)\b", re.IGNORECASE),
]

_DYNAMIC_KEYWORDS_ZH: list[str] = [
    "今天", "昨天", "明天", "最近", "正在", "刚才", "刚刚",
    "这周", "这个月", "上周", "下周", "目前", "现在",
    "今晚", "今早", "稍后", "待会",
    # Additions. Bare 周末 ("weekend") is excluded — "我周末喜欢打球"
    # ("I like playing ball on weekends") is habitual.
    "这个周末", "即将", "马上", "截止",
]

# ---------------------------------------------------------------------------
# Weak dynamic patterns — relative durations. Dynamic only when no habitual
# marker is present (see module docstring).
# ---------------------------------------------------------------------------

_WEAK_DYNAMIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bin \d+ (?:minutes?|hours?|days?|weeks?)\b", re.IGNORECASE),
    re.compile(r"\bin an? (?:hour|day|week)\b", re.IGNORECASE),
    # "3天后" / "两周后" / "4小时后". A numeral is required, so 天后 in its
    # other sense ("diva", as in 天后王菲) never matches.
    re.compile(rf"{_ZH_NUM}\s*(?:天|周|个?星期|个?小时|分钟)后"),
]

_HABITUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:always|usually|typically|generally|normally|often|never)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:every|each) (?:day|night|morning|evening|week|month|year|time)\b",
        re.IGNORECASE,
    ),
    re.compile(r"通常|一般|总是|从不|每(?:天|晚|周|个?星期|个?月|年|次)"),
]

# ---------------------------------------------------------------------------
# Static patterns — permanent fact indicators. Dynamic signals still win a
# tie; these document intent and feed the habitual veto above.
# ---------------------------------------------------------------------------

_STATIC_PATTERNS_EN: list[re.Pattern[str]] = [
    re.compile(r"\bfavou?rite\b", re.IGNORECASE),
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
    re.compile(r"\b(?:usually|typically|generally|never)\b", re.IGNORECASE),
    re.compile(
        r"\bmy (?:wife|husband|partner|spouse|son|daughter"
        r"|brother|sister|mother|father|mom|mum|dad)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:vegetarian|vegan|gluten[- ]free|lactose[- ]intolerant|diabetic)\b",
        re.IGNORECASE,
    ),
]

_STATIC_KEYWORDS_ZH: list[str] = [
    "喜欢", "偏好", "一直", "名字", "叫做", "出生",
    "毕业", "住在", "工作", "职业", "爱好", "过敏",
    "通常", "从不", "妻子", "丈夫", "儿子", "女儿", "父母",
    "素食", "纯素", "乳糖不耐",
]

# ---------------------------------------------------------------------------
# Expiry rules: ordered list of (patterns, offset_ms) pairs.
# Order matters — more specific patterns (day after tomorrow) must come
# before more general ones (tomorrow) to match correctly. Offsets round UP:
# expiring an event late is harmless, expiring it early loses it.
# ---------------------------------------------------------------------------

_MS_PER_HOUR = 60 * 60 * 1000
_MS_PER_DAY = 24 * _MS_PER_HOUR
_MS_PER_MINUTE = 60 * 1000

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
    # next <day> → +7d (the latest that day can be)
    (
        [re.compile(rf"\bnext (?:{_DAYS})\b", re.IGNORECASE)],
        7 * _MS_PER_DAY,
    ),
    # 下周 / next week → +7d
    (
        [re.compile(r"下周"), re.compile(r"\bnext week\b", re.IGNORECASE)],
        7 * _MS_PER_DAY,
    ),
    # 这个周末 / this weekend → +7d (said on a Monday, Sunday is 6 days out)
    (
        [re.compile(r"这个周末"), re.compile(r"\bthis weekend\b", re.IGNORECASE)],
        7 * _MS_PER_DAY,
    ),
    # 这周 / this week → +3d
    (
        [re.compile(r"这周"), re.compile(r"\bthis week\b", re.IGNORECASE)],
        3 * _MS_PER_DAY,
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
    # 今天 / today / this afternoon / this evening → +18h
    (
        [
            re.compile(r"今天"),
            re.compile(r"\btoday\b", re.IGNORECASE),
            re.compile(r"\bthis (?:afternoon|evening)\b", re.IGNORECASE),
        ],
        18 * _MS_PER_HOUR,
    ),
]

# Variable-offset expiry: (pattern with one integer capture group, ms per
# unit). Weak signals — skipped when the text is habitual.
_VARIABLE_EXPIRY_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\bin (\d+) weeks?\b", re.IGNORECASE), 7 * _MS_PER_DAY),
    (re.compile(r"\bin (\d+) days?\b", re.IGNORECASE), _MS_PER_DAY),
    (re.compile(r"\bin (\d+) hours?\b", re.IGNORECASE), _MS_PER_HOUR),
    (re.compile(r"\bin (\d+) minutes?\b", re.IGNORECASE), _MS_PER_MINUTE),
    (re.compile(r"(\d+)\s*(?:周|个?星期)后"), 7 * _MS_PER_DAY),
    (re.compile(r"(\d+)\s*天后"), _MS_PER_DAY),
    (re.compile(r"(\d+)\s*个?小时后"), _MS_PER_HOUR),
    (re.compile(r"(\d+)\s*分钟后"), _MS_PER_MINUTE),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["TemporalType", "classify_temporal", "infer_expiry"]


def _is_habitual(text: str) -> bool:
    return any(p.search(text) for p in _HABITUAL_PATTERNS)


def classify_temporal(text: str) -> TemporalType:
    """Classify memory text as static (permanent fact) or dynamic (time-sensitive).

    Rule-based: keywords -> classification. Default: "static" (safer default).
    A strong dynamic signal always wins, even alongside static keywords
    (time-sensitive info takes priority). A weak signal (relative duration)
    counts only when the text is not habitual.
    """
    if any(p.search(text) for p in _DYNAMIC_PATTERNS_EN) or any(
        kw in text for kw in _DYNAMIC_KEYWORDS_ZH
    ):
        return "dynamic"
    if any(p.search(text) for p in _WEAK_DYNAMIC_PATTERNS) and not _is_habitual(text):
        return "dynamic"
    # Static keywords or no signal at all → "static" (avoids premature decay).
    return "static"


def infer_expiry(text: str, now_ms: int | None = None) -> int | None:
    """Infer expiry timestamp (epoch-ms) from temporal expressions in text.

    Returns None if no recognised temporal expression is found. Fixed-offset
    rules run first, in specificity order; variable-offset rules ("in 3
    days", "2周后") apply only to non-habitual text.

    Args:
        text:   memory text to inspect
        now_ms: base timestamp in milliseconds (default: current wall-clock time)
    """
    base_time = now_ms if now_ms is not None else int(time.time() * 1000)

    for patterns, offset_ms in _EXPIRY_RULES:
        for pattern in patterns:
            if pattern.search(text):
                return base_time + offset_ms

    if not _is_habitual(text):
        for pattern, ms_per_unit in _VARIABLE_EXPIRY_RULES:
            m = pattern.search(text)
            if m:
                return base_time + int(m.group(1)) * ms_per_unit

    return None
