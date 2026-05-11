"""
Markdown parser + sanitiser for LLM-generated reflection output.

Ported verbatim from CortexReach's reflection-slices.ts (375 lines).
Pure Python — no I/O, no LLM calls.

Public API
----------
Types:
    ReflectionSliceItem
    ReflectionSlices
    ReflectionMappedMemoryItem
    ReflectionGovernanceEntry

Extraction:
    extract_section_markdown
    parse_section_bullets
    extract_reflection_slices
    extract_reflection_mapped_memory_items
    extract_reflection_lessons
    extract_reflection_learning_governance_candidates

Sanitisation:
    sanitize_reflection_slice_lines
    is_unsafe_injectable_reflection_line
    sanitize_injectable_reflection_lines

Heuristics:
    is_recall_used
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

__all__ = [
    # types
    "ReflectionSliceItem",
    "ReflectionSlices",
    "ReflectionMappedMemoryItem",
    "ReflectionGovernanceEntry",
    # extraction
    "extract_section_markdown",
    "parse_section_bullets",
    "extract_reflection_slices",
    "extract_reflection_mapped_memory_items",
    "extract_reflection_lessons",
    "extract_reflection_learning_governance_candidates",
    # sanitisation
    "sanitize_reflection_slice_lines",
    "is_unsafe_injectable_reflection_line",
    "sanitize_injectable_reflection_lines",
    # heuristics
    "is_recall_used",
]


@dataclass
class ReflectionSliceItem:
    text: str
    kind: Literal["invariant", "derived"]


@dataclass
class ReflectionSlices:
    invariants: list[ReflectionSliceItem]
    derived: list[ReflectionSliceItem]


@dataclass
class ReflectionMappedMemoryItem:
    text: str
    kind: Literal["user-model", "agent-model", "lesson", "decision"]
    ordinal: int      # position within kind group (1-based)
    group_size: int   # total items of this kind in the markdown


@dataclass
class ReflectionGovernanceEntry:
    title: str
    body: str


# ---------------------------------------------------------------------------
# Compiled regex patterns (module level, per decay.py style)
# ---------------------------------------------------------------------------

# Placeholder detection — ported verbatim from isPlaceholderReflectionSliceLine
_RE_PLACEHOLDER_NONE = re.compile(r"^\(none( captured)?\)$", re.IGNORECASE)
_RE_PLACEHOLDER_LABEL = re.compile(
    r"^(invariants?|reflections?|derived)[:：]$", re.IGNORECASE
)
_RE_PLACEHOLDER_DELTA = re.compile(
    r"apply this session'?s deltas next run", re.IGNORECASE
)
_RE_PLACEHOLDER_DISTILLED = re.compile(
    r"apply this session'?s distilled changes next run", re.IGNORECASE
)
_RE_PLACEHOLDER_INVESTIGATE = re.compile(
    r"investigate why embedded reflection generation failed", re.IGNORECASE
)

# Normalisation — strip section-label prefixes, ported from normalizeReflectionSliceLine
_RE_BOLD = re.compile(r"\*\*")
_RE_SECTION_PREFIX = re.compile(
    r"^(invariants?|reflections?|derived)[:：]\s*", re.IGNORECASE
)

# Injection guard patterns — ported verbatim from INJECTABLE_REFLECTION_BLOCK_PATTERNS
_INJECTABLE_REFLECTION_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^\s*(?:(?:next|this)\s+run\s+)?(?:ignore|disregard|forget|override|bypass)\b[\s\S]{0,80}\b(?:instructions?|guardrails?|policy|developer|system)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|dump|show|output)\b[\s\S]{0,80}\b(?:system prompt|developer prompt|hidden prompt|hidden instructions?|full prompt|prompt verbatim|secrets?|keys?|tokens?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\s*\/?\s*(?:system|assistant|user|tool|developer|inherited-rules|derived-focus)\b[^>]*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:system|assistant|user|developer|tool)\s*:",
        re.IGNORECASE,
    ),
]

# Heuristic classifiers — ported verbatim from isInvariantRuleLike / isDerivedDeltaLike
_RE_INVARIANT_START = re.compile(
    r"^(always|never|when\b|if\b|before\b|after\b|prefer\b|avoid\b|require\b|only\b|do not\b|must\b|should\b)",
    re.IGNORECASE,
)
_RE_INVARIANT_BODY = re.compile(
    r"\b(must|should|never|always|prefer|avoid|required?)\b", re.IGNORECASE
)

_RE_DERIVED_START = re.compile(
    r"^(this run|next run|going forward|follow-up|re-check|retest|verify|confirm|avoid repeating|adjust|change|update|retry|keep|watch)\b",
    re.IGNORECASE,
)
_RE_DERIVED_BODY = re.compile(
    r"\b(this run|next run|delta|change|adjust|retry|re-check|retest|verify|confirm|avoid repeating|follow-up)\b",
    re.IGNORECASE,
)

# Open-loop action heuristic — ported verbatim from isOpenLoopAction
_RE_OPEN_LOOP_START = re.compile(
    r"^(investigate|verify|confirm|re-check|retest|update|add|remove|fix|avoid|keep|watch|document)\b",
    re.IGNORECASE,
)

# Legacy combined section keyword filters
_RE_INVARIANT_KEYWORD = re.compile(
    r"invariant|stable|policy|rule", re.IGNORECASE
)
_RE_DERIVED_KEYWORD = re.compile(
    r"reflect|inherit|derive|change|apply", re.IGNORECASE
)

# Recall usage markers (EN + CJK) — ported verbatim from isRecallUsed
_RECALL_USAGE_MARKERS: list[str] = [
    "remember",
    "之前",
    "记得",
    "记得",
    "according to",
    "based on what",
    "as you mentioned",
    "如前所述",
    "如您所說",
    "如您所说的",
    "我記得",
    "我记得",
    "之前你說",
    "之前你说",
    "之前提到",
    "之前提到的",
    "根据之前",
    "依据之前",
    "按照之前",
    "照您之前",
    "照你说的",
    "from previous",
    "earlier you",
    "in the memory",
    "the memory mentioned",
    "the memories show",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_reflection_slice_line(line: str) -> str:
    """Strip bold markers and leading section-label prefix. Port of normalizeReflectionSliceLine."""
    return _RE_SECTION_PREFIX.sub("", _RE_BOLD.sub("", line)).strip()


def _is_placeholder_reflection_slice_line(line: str) -> bool:
    """Return True if line is a known placeholder. Port of isPlaceholderReflectionSliceLine."""
    normalized = _normalize_reflection_slice_line(line)
    if not normalized:
        return True
    return bool(
        _RE_PLACEHOLDER_NONE.match(normalized)
        or _RE_PLACEHOLDER_LABEL.match(normalized)
        or _RE_PLACEHOLDER_DELTA.search(normalized)
        or _RE_PLACEHOLDER_DISTILLED.search(normalized)
        or _RE_PLACEHOLDER_INVESTIGATE.search(normalized)
    )


def _is_invariant_rule_like(line: str) -> bool:
    """Port of isInvariantRuleLike."""
    return bool(
        _RE_INVARIANT_START.match(line)
        or _RE_INVARIANT_BODY.search(line)
    )


def _is_derived_delta_like(line: str) -> bool:
    """Port of isDerivedDeltaLike."""
    return bool(
        _RE_DERIVED_START.match(line)
        or _RE_DERIVED_BODY.search(line)
    )


def _is_open_loop_action(line: str) -> bool:
    """Port of isOpenLoopAction."""
    return bool(_RE_OPEN_LOOP_START.match(line))


# ---------------------------------------------------------------------------
# Public sanitisation API
# ---------------------------------------------------------------------------

def sanitize_reflection_slice_lines(lines: Sequence[str]) -> list[str]:
    """
    Drop placeholder lines and strip bold/prefix markers.

    Port of sanitizeReflectionSliceLines. Placeholder comparison is
    done on the normalised form; output preserves original casing only
    after stripping bold and section-label prefixes (per the TS source).
    """
    result: list[str] = []
    for raw in lines:
        normalized = _normalize_reflection_slice_line(raw)
        if not _is_placeholder_reflection_slice_line(normalized):
            result.append(normalized)
    return result


def is_unsafe_injectable_reflection_line(line: str) -> bool:
    """
    Return True when line matches a prompt-injection pattern.

    Port of isUnsafeInjectableReflectionLine. Empty/blank lines
    (after normalisation) are also considered unsafe.
    """
    normalized = _normalize_reflection_slice_line(line)
    if not normalized:
        return True
    return any(pat.search(normalized) for pat in _INJECTABLE_REFLECTION_BLOCK_PATTERNS)


def sanitize_injectable_reflection_lines(lines: Sequence[str]) -> list[str]:
    """
    Strict sanitisation: drop unsafe lines entirely (NOT just trim).

    Port of sanitizeInjectableReflectionLines. Applies placeholder
    sanitisation first, then drops prompt-injection lines.
    """
    return [
        line
        for line in sanitize_reflection_slice_lines(lines)
        if not is_unsafe_injectable_reflection_line(line)
    ]


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_section_markdown(md: str, heading: str) -> str:
    """
    Return the body text under ## <heading>.

    Port of extractSectionMarkdown. Comparison is case-insensitive on the
    trimmed line. Returns empty string if heading is not found.
    """
    lines = re.split(r"\r?\n", md)
    heading_needle = f"## {heading}".lower()
    in_section = False
    collected: list[str] = []

    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("## "):
            if in_section and lower != heading_needle:
                break
            in_section = lower == heading_needle
            continue
        if not in_section:
            continue
        collected.append(raw)

    return "\n".join(collected).strip()


def parse_section_bullets(md: str, heading: str) -> list[str]:
    """
    Extract `- item` and `* item` bullet lines under the named section.

    Port of parseSectionBullets. Bullet markers recognised: `-` and `*`
    (matching the TS source exactly).
    """
    body = extract_section_markdown(md, heading)
    lines = re.split(r"\r?\n", body)
    collected: list[str] = []

    for raw in lines:
        line = raw.strip()
        if line.startswith("- ") or line.startswith("* "):
            normalized = line[2:].strip()
            if normalized:
                collected.append(normalized)

    return collected


# ---------------------------------------------------------------------------
# Reflection slices
# ---------------------------------------------------------------------------

def _extract_reflection_slices_with_sanitizer(
    reflection_text: str,
    sanitize_lines: Callable[[Sequence[str]], list[str]],
) -> ReflectionSlices:
    """Internal port of extractReflectionSlicesWithSanitizer."""

    invariant_section = parse_section_bullets(reflection_text, "Invariants")
    derived_section = parse_section_bullets(reflection_text, "Derived")
    merged_section = parse_section_bullets(reflection_text, "Invariants & Reflections")

    invariants_primary = [
        line for line in sanitize_lines(invariant_section)
        if _is_invariant_rule_like(line)
    ]
    derived_primary = [
        line for line in sanitize_lines(derived_section)
        if _is_derived_delta_like(line)
    ]

    invariant_lines_legacy = [
        line
        for line in sanitize_lines([s for s in merged_section if _RE_INVARIANT_KEYWORD.search(s)])
        if _is_invariant_rule_like(line)
    ]
    reflection_lines_legacy = [
        line
        for line in sanitize_lines([s for s in merged_section if _RE_DERIVED_KEYWORD.search(s)])
        if _is_derived_delta_like(line)
    ]
    open_loop_lines = [
        line
        for line in sanitize_lines(parse_section_bullets(reflection_text, "Open loops / next actions"))
        if _is_open_loop_action(line) and _is_derived_delta_like(line)
    ]
    durable_decision_lines = [
        line
        for line in sanitize_lines(parse_section_bullets(reflection_text, "Decisions (durable)"))
        if _is_invariant_rule_like(line)
    ]

    if invariants_primary:
        invariants = invariants_primary
    elif invariant_lines_legacy:
        invariants = invariant_lines_legacy
    else:
        invariants = durable_decision_lines

    derived = derived_primary or reflection_lines_legacy + open_loop_lines

    # Top-N truncation happens AFTER sanitisation
    invariant_items = [
        ReflectionSliceItem(text=t, kind="invariant")
        for t in invariants[:8]
    ]
    derived_items = [
        ReflectionSliceItem(text=t, kind="derived")
        for t in derived[:10]
    ]

    return ReflectionSlices(invariants=invariant_items, derived=derived_items)


def extract_reflection_slices(md: str) -> ReflectionSlices:
    """
    Parse `## Invariants` (top 8) and `## Derived` (top 10) sections.

    Sanitises + classifies lines via rule-like / delta-like heuristics.
    Port of extractReflectionSlices.
    """
    return _extract_reflection_slices_with_sanitizer(md, sanitize_reflection_slice_lines)


# ---------------------------------------------------------------------------
# Mapped memory items
# ---------------------------------------------------------------------------

# Section headings and kind mapping — ported verbatim from the TS mappedSections array
_MAPPED_SECTIONS: list[tuple[str, Literal["user-model", "agent-model", "lesson", "decision"]]] = [
    ("User model deltas (about the human)", "user-model"),
    ("Agent model deltas (about the assistant/system)", "agent-model"),
    ("Lessons & pitfalls (symptom / cause / fix / prevention)", "lesson"),
    ("Decisions (durable)", "decision"),
]


def _extract_reflection_mapped_memory_items_with_sanitizer(
    reflection_text: str,
    sanitize_lines: Callable[[Sequence[str]], list[str]],
) -> list[ReflectionMappedMemoryItem]:
    """Internal port of extractReflectionMappedMemoryItemsWithSanitizer."""
    result: list[ReflectionMappedMemoryItem] = []

    for heading, kind in _MAPPED_SECTIONS:
        lines = sanitize_lines(parse_section_bullets(reflection_text, heading))
        group_size = len(lines)
        for ordinal_0, text in enumerate(lines):
            result.append(
                ReflectionMappedMemoryItem(
                    text=text,
                    kind=kind,
                    ordinal=ordinal_0 + 1,   # 1-based per spec
                    group_size=group_size,
                )
            )

    return result


def extract_reflection_mapped_memory_items(md: str) -> list[ReflectionMappedMemoryItem]:
    """
    Parse all four mapped sections and return items with ordinal + group_size.

    Port of extractReflectionMappedMemoryItems.
    Sections: User model / Agent model / Lessons / Decisions.
    """
    return _extract_reflection_mapped_memory_items_with_sanitizer(
        md, sanitize_reflection_slice_lines
    )


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------

def extract_reflection_lessons(reflection_text: str) -> list[str]:
    """
    Return sanitised bullet lines from the Lessons section.

    Port of extractReflectionLessons. Uses the full section heading:
    'Lessons & pitfalls (symptom / cause / fix / prevention)'.
    """
    return sanitize_reflection_slice_lines(
        parse_section_bullets(
            reflection_text,
            "Lessons & pitfalls (symptom / cause / fix / prevention)",
        )
    )


# ---------------------------------------------------------------------------
# Learning governance candidates
# ---------------------------------------------------------------------------

def _parse_governance_entry_block(block: str) -> ReflectionGovernanceEntry | None:
    """
    Parse a single ### Title … body block into a ReflectionGovernanceEntry.

    Port of parseReflectionGovernanceEntry — adapted for the Python public
    type (title + body) rather than the TS structured entry.
    """
    # Strip the ### heading line to get the title
    title_match = re.match(r"^###\s+(.+?)[\r\n]", block)
    if not title_match:
        # Try single-line block
        title_match = re.match(r"^###\s+(.+)$", block.strip())
    if not title_match:
        return None

    title = title_match.group(1).strip()
    # Body is everything after the heading line
    body = re.sub(r"^###\s+[^\n]*\n?", "", block, count=1).strip()
    return ReflectionGovernanceEntry(title=title, body=body)


def extract_reflection_learning_governance_candidates(
    reflection_text: str,
) -> list[ReflectionGovernanceEntry]:
    """
    Parse `### Title` blocks under the Learning Governance section.

    Port of extractReflectionLearningGovernanceCandidates. Looks under:
    '## Learning governance candidates (.learnings / promotion / skill extraction)'

    Body is everything until the next `### ` or end-of-section.
    Falls back to returning a single entry wrapping bullet lines when no
    `### Entry` blocks are found (matching TS fallback behaviour).
    """
    section_heading = (
        "Learning governance candidates (.learnings / promotion / skill extraction)"
    )
    section = extract_section_markdown(reflection_text, section_heading)
    if not section:
        return []

    # Split on ### Title boundaries (lookahead keeps the delimiter)
    raw_blocks = re.split(r"(?=^###\s+)", section, flags=re.MULTILINE)
    blocks = [b.strip() for b in raw_blocks if b.strip()]

    # Filter to blocks that actually start with ###
    entry_blocks = [b for b in blocks if b.startswith("###")]

    parsed: list[ReflectionGovernanceEntry] = []
    for block in entry_blocks:
        entry = _parse_governance_entry_block(block)
        if entry is not None:
            parsed.append(entry)

    if parsed:
        return parsed

    # Fallback: treat sanitised bullets as a single entry body
    fallback_bullets = sanitize_reflection_slice_lines(
        parse_section_bullets(reflection_text, section_heading)
    )
    if not fallback_bullets:
        return []

    body = "\n".join(f"- {line}" for line in fallback_bullets)
    return [
        ReflectionGovernanceEntry(
            title="Reflection learning governance candidates",
            body=body,
        )
    ]


# ---------------------------------------------------------------------------
# Recall heuristic
# ---------------------------------------------------------------------------

def is_recall_used(response: str, injected_ids: Sequence[str]) -> bool:
    """
    Heuristic: return True when the response shows evidence of using recalled memories.

    Port of isRecallUsed. Checks for:
    - Any injected memory ID mentioned in the response
    - EN + CJK recall marker phrases

    Short responses (<= 24 chars) are treated as not using recall.
    """
    if not response or len(response) <= 24:
        return False
    if not injected_ids:
        return False

    response_lower = response.lower()

    # Check whether any injected memory ID appears literally in the response.
    if any(mem_id in response_lower for mem_id in injected_ids):
        return True

    return any(marker.lower() in response_lower for marker in _RECALL_USAGE_MARKERS)
