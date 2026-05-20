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
]

_DYNAMIC_KEYWORDS_ZH: list[str] = [
    "今天", "昨天", "明天", "最近", "正在", "刚才", "刚刚",
    "这周", "这个月", "上周", "下周", "目前", "现在",
    "今晚", "今早", "稍后", "待会",
]

# ---------------------------------------------------------------------------
# Static patterns — permanent fact indicators.
# ---------------------------------------------------------------------------

_STATIC_PATTERNS_EN: list[re.Pattern[str]] = [
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
]

_STATIC_KEYWORDS_ZH: list[str] = [
    "喜欢", "偏好", "一直", "名字", "叫做", "出生",
    "毕业", "住在", "工作", "职业", "爱好", "过敏",
]

# ---------------------------------------------------------------------------
# Expiry rules: ordered list of (patterns, offset_ms) pairs.
# Order matters — more specific patterns (day after tomorrow) must come
# before more general ones (tomorrow) to match correctly.
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
    # 下周 / next week → +7d
    (
        [re.compile(r"下周"), re.compile(r"\bnext week\b", re.IGNORECASE)],
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
    # 今天 / today → +18h
    (
        [re.compile(r"今天"), re.compile(r"\btoday\b", re.IGNORECASE)],
        18 * _MS_PER_HOUR,
    ),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["TemporalType", "classify_temporal", "infer_expiry"]


def classify_temporal(text: str) -> TemporalType:
    """Classify memory text as static (permanent fact) or dynamic (time-sensitive).

    Rule-based: keywords -> classification. Default: "static" (safer default).
    If BOTH dynamic and static keywords match, "dynamic" wins (time-sensitive
    info takes priority).
    """
    has_dynamic = any(p.search(text) for p in _DYNAMIC_PATTERNS_EN) or any(
        kw in text for kw in _DYNAMIC_KEYWORDS_ZH
    )
    has_static = any(p.search(text) for p in _STATIC_PATTERNS_EN) or any(
        kw in text for kw in _STATIC_KEYWORDS_ZH
    )

    # If BOTH match → "dynamic" wins (time-sensitive info takes priority)
    if has_dynamic:
        return "dynamic"
    # If only static matches → static
    if has_static:
        return "static"
    # If NEITHER match → "static" (safer default, avoids premature expiry)
    return "static"


def infer_expiry(text: str, now_ms: int | None = None) -> int | None:
    """Infer expiry timestamp (epoch-ms) from temporal expressions in text.

    Returns None if no recognised temporal expression is found.

    Args:
        text:   memory text to inspect
        now_ms: base timestamp in milliseconds (default: current wall-clock time)
    """
    base_time = now_ms if now_ms is not None else int(time.time() * 1000)

    for patterns, offset_ms in _EXPIRY_RULES:
        for pattern in patterns:
            if pattern.search(text):
                return base_time + offset_ms

    return None
