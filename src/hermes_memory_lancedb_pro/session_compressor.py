"""
Session Compressor

Scores and compresses conversation texts before memory extraction.
Prioritises high-signal content (tool calls, corrections, decisions) over
low-signal content (greetings, acknowledgments) so that the fixed extraction
budget captures the most important parts of a conversation.

Ported from CortexReach session-compressor.ts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ScoredText",
    "CompressResult",
    "TOOL_CALL_INDICATORS",
    "CORRECTION_INDICATORS",
    "DECISION_INDICATORS",
    "ACKNOWLEDGMENT_PATTERNS",
    "score_text",
    "compress_texts",
    "estimate_conversation_value",
]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ScoredText:
    """Score record for a single text segment."""

    index: int
    text: str
    score: float
    reason: str


@dataclass
class CompressResult:
    """Result of compressing a list of text segments."""

    texts: list[str]
    scored: list[ScoredText]
    dropped: int
    total_chars: int


# ---------------------------------------------------------------------------
# Indicator patterns (compiled at module level)
# ---------------------------------------------------------------------------

TOOL_CALL_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"\btool_use\b", re.IGNORECASE),
    re.compile(r"\btool_result\b", re.IGNORECASE),
    re.compile(r"\bfunction_call\b", re.IGNORECASE),
    re.compile(r"\b(memory_store|memory_recall|memory_forget|memory_update)\b", re.IGNORECASE),
]

CORRECTION_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"^no[,.\s]", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\binstead\b", re.IGNORECASE),
    re.compile(r"\bwrong\b", re.IGNORECASE),
    re.compile(r"\bcorrect(ion)?\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
    re.compile(r"不对"),
    re.compile(r"应该是"),
    re.compile(r"應該是"),
    re.compile(r"错了"),
    re.compile(r"錯了"),
    re.compile(r"改成"),
    re.compile(r"不是.*而是"),
]

DECISION_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"\blet'?s go with\b", re.IGNORECASE),
    re.compile(r"\bconfirmed?\b", re.IGNORECASE),
    re.compile(r"\bapproved?\b", re.IGNORECASE),
    re.compile(r"\bdecided?\b", re.IGNORECASE),
    re.compile(r"\bwe'?ll use\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bfrom now on\b", re.IGNORECASE),
    re.compile(r"\bagreed\b", re.IGNORECASE),
    re.compile(r"决定"),
    re.compile(r"決定"),
    re.compile(r"确认"),
    re.compile(r"確認"),
    re.compile(r"选择了"),
    re.compile(r"選擇了"),
    re.compile(r"就这样"),
    re.compile(r"就這樣"),
]

ACKNOWLEDGMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(ok|okay|k|sure|fine|thanks|thank you|thx|ty|got it|understood|cool|nice"
        r"|great|good|perfect|awesome|alright|yep|yup|yeah|right)\s*[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^好的?\s*[。！]?$"),
    re.compile(r"^嗯\s*[。]?$"),
    re.compile(r"^收到\s*[。！]?$"),
    re.compile(r"^了解\s*[。！]?$"),
    re.compile(r"^明白\s*[。！]?$"),
    re.compile(r"^谢谢\s*[。！]?$"),
    re.compile(r"^感谢\s*[。！]?$"),
    re.compile(r"^👍\s*$"),
]

# CJK character detection (matches the TS hasCJK pattern extended per spec)
_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ가-힯]")

# System-XML boilerplate detection (port of TS two-regex check)
_XML_OPEN_RE = re.compile(r"^<[a-z-]+>")
_XML_CLOSE_RE = re.compile(r"</[a-z-]+>\s*$")

# Memory intent patterns for estimate_conversation_value
_MEMORY_INTENT_EN = re.compile(
    r"\b(remember|recall|don'?t forget|note that|keep in mind)\b",
    re.IGNORECASE,
)
_MEMORY_INTENT_CJK = re.compile(r"(记住|記住|别忘|不要忘|记一下|記一下)")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_text(text: str, index: int) -> ScoredText:
    """Score a single text segment by its information density."""
    trimmed = text.strip()

    # Empty / whitespace-only
    if len(trimmed) == 0:
        return ScoredText(index=index, text=text, score=0.0, reason="empty")

    # Tool call indicators → highest value
    if any(p.search(trimmed) for p in TOOL_CALL_INDICATORS):
        return ScoredText(index=index, text=text, score=1.0, reason="tool_call")

    # Corrections → very high value
    if any(p.search(trimmed) for p in CORRECTION_INDICATORS):
        return ScoredText(index=index, text=text, score=0.95, reason="correction")

    # Decisions / confirmations → high value
    if any(p.search(trimmed) for p in DECISION_INDICATORS):
        return ScoredText(index=index, text=text, score=0.85, reason="decision")

    # Acknowledgments → very low value
    if any(p.search(trimmed) for p in ACKNOWLEDGMENT_PATTERNS):
        return ScoredText(index=index, text=text, score=0.1, reason="acknowledgment")

    # Substantive content vs short questions.
    # CJK characters carry ~2-3x more meaning per character, so use a lower
    # threshold (same approach as adaptive-retrieval).
    has_cjk = bool(_CJK_RE.search(trimmed))
    substantive_min_length = 30 if has_cjk else 80
    if len(trimmed) > substantive_min_length:
        # Check for boilerplate (XML tags, system messages)
        if _XML_OPEN_RE.match(trimmed) and _XML_CLOSE_RE.search(trimmed):
            return ScoredText(index=index, text=text, score=0.3, reason="system_xml")
        return ScoredText(index=index, text=text, score=0.7, reason="substantive")

    # Short questions
    if "?" in trimmed or "？" in trimmed:
        return ScoredText(index=index, text=text, score=0.5, reason="short_question")

    # Short but not a question and not an acknowledgment
    return ScoredText(index=index, text=text, score=0.4, reason="short_statement")


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

_DEFAULT_MIN_TEXTS = 3


def compress_texts(
    texts: list[str],
    max_chars: int,
    *,
    min_texts: int = _DEFAULT_MIN_TEXTS,
    min_score_to_keep: float = 0.3,
) -> CompressResult:
    """
    Compress an array of text segments to fit within a character budget.

    Strategy:
    1. Score all texts.
    2. If total chars <= budget, return all.
    3. Always include first and last text (session boundaries).
    4. Sort remaining by score descending (stable: tie → ascending index).
    5. Greedily select until budget exhausted.
    6. If a tool_call line is added, also try to add the next line as a
       paired result.  Pairing only fires for tool_call, NOT tool_result.
    7. All-low-score fallback: if every score < min_score_to_keep, ensure at
       least min_texts entries (added from the end of the list).
    8. Re-sort selected by original index (chronological).
    """
    if not texts:
        return CompressResult(texts=[], scored=[], dropped=0, total_chars=0)

    # Score everything
    scored = [score_text(t, i) for i, t in enumerate(texts)]

    # Total chars of all texts
    all_chars = sum(len(t) for t in texts)

    # If already within budget, return all
    if all_chars <= max_chars:
        return CompressResult(
            texts=list(texts),
            scored=scored,
            dropped=0,
            total_chars=all_chars,
        )

    # Build selected set starting with first and last
    selected_indices: set[int] = set()
    used_chars = 0

    def add_index(idx: int) -> bool:
        nonlocal used_chars
        if idx in selected_indices or idx < 0 or idx >= len(texts):
            return False
        length = len(texts[idx])
        if used_chars + length > max_chars:
            # Hard cap: even the first/last text cannot exceed budget
            return False
        selected_indices.add(idx)
        used_chars += length
        return True

    # Always keep first and last
    add_index(0)
    if len(texts) > 1:
        add_index(len(texts) - 1)

    # Build candidate list excluding first/last, sorted by score desc
    # (stable: tie → ascending index)
    last_idx = len(texts) - 1
    candidates = sorted(
        (s for s in scored if s.index != 0 and s.index != last_idx),
        key=lambda s: (-s.score, s.index),
    )

    # Identify paired indices (tool_call at i → result at i+1).
    # Only pair from a tool_call line, NOT from tool_result — a result line
    # should not pull in the next unrelated line as its "partner".
    paired_with: dict[int, int] = {}
    for s in scored:
        if (
            s.reason == "tool_call"
            and s.index + 1 < len(texts)
            and s.index not in paired_with
            and s.index + 1 not in paired_with
        ):
            paired_with[s.index] = s.index + 1
            paired_with[s.index + 1] = s.index

    # Greedily add candidates
    for candidate in candidates:
        if used_chars >= max_chars:
            break
        added = add_index(candidate.index)
        if added:
            # If this is part of a pair, try to add the partner
            partner = paired_with.get(candidate.index)
            if partner is not None:
                add_index(partner)

    # All-low-score fallback: if everything scored below threshold, ensure
    # we keep at least min_texts (the last N by original order)
    all_low = all(s.score < min_score_to_keep for s in scored)
    if all_low and len(selected_indices) < min(min_texts, len(texts)):
        for i in range(len(texts) - 1, -1, -1):
            if len(selected_indices) >= min(min_texts, len(texts)):
                break
            add_index(i)

    # Re-sort selected by original index to preserve chronological order
    sorted_indices = sorted(selected_indices)
    result_texts = [texts[i] for i in sorted_indices]
    total_chars = sum(len(t) for t in result_texts)

    return CompressResult(
        texts=result_texts,
        scored=scored,
        dropped=len(texts) - len(sorted_indices),
        total_chars=total_chars,
    )


# ---------------------------------------------------------------------------
# Conversation Value Estimation (for Adaptive Throttling)
# ---------------------------------------------------------------------------


def estimate_conversation_value(texts: list[str] | str | None) -> float:
    """
    Estimate the overall value of a conversation for memory extraction.
    Returns a number between 0.0 and 1.0.

    Used by the adaptive extraction throttle to skip low-value conversations.

    Accepts a list of text turns; for convenience also accepts a single
    string (treated as one turn) or None (treated as empty). Substantive
    conversations always return at least a small positive baseline so the
    throttle distinguishes them from empty/trivial input.
    """
    # Defensive input coercion: callers in the wild sometimes pass a single
    # string (a joined transcript) rather than a list of turns. Treat that
    # as one turn rather than iterating its characters.
    if texts is None:
        return 0.0
    if isinstance(texts, str):
        texts = [texts] if texts.strip() else []
    if not texts:
        return 0.0

    value = 0.0
    joined = " ".join(texts)

    # Has explicit memory intent? (e.g. "remember this", "记住") +0.5
    # These should NEVER be skipped by the low-value gate.
    if _MEMORY_INTENT_EN.search(joined) or _MEMORY_INTENT_CJK.search(joined):
        value += 0.5

    # Has tool calls? +0.4
    if any(p.search(joined) for p in TOOL_CALL_INDICATORS):
        value += 0.4

    # Has corrections or decisions? +0.3
    has_correction_or_decision = any(
        p.search(joined) for p in CORRECTION_INDICATORS
    ) or any(p.search(joined) for p in DECISION_INDICATORS)
    if has_correction_or_decision:
        value += 0.3

    # Substantive content scoring: graduated rather than a single 200-char
    # cliff. A typical troubleshooting exchange (300-600 chars) should
    # comfortably clear the throttle floor; the previous threshold left
    # such conversations stuck at 0.0 if they didn't trigger any of the
    # pattern-based bumps.
    substantive_chars = sum(
        len(t) for t in texts if len(t.strip()) > 20
    )
    if substantive_chars > 500:
        value += 0.3
    elif substantive_chars > 200:
        value += 0.2
    elif substantive_chars > 100:
        value += 0.15

    # Has multi-turn exchanges (>6 texts)? +0.1
    if len(texts) > 6:
        value += 0.1

    # Baseline floor for any non-trivial conversation: anything with at
    # least one substantive turn (> 20 chars stripped) gets a small
    # positive value so the throttle treats it as "worth considering",
    # even if no specific intent/tool/correction pattern fires.
    if value == 0.0 and any(len(t.strip()) > 20 for t in texts):
        value = 0.1

    return min(value, 1.0)
