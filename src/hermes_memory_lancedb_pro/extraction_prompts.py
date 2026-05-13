"""
Prompt templates for intelligent memory extraction.

Three mandatory prompts:
- build_extraction_prompt: 6-category L0/L1/L2 extraction with few-shot
- build_dedup_prompt: CREATE/MERGE/SKIP dedup decision
- build_merge_prompt: Memory merge with three-level structure

Ported verbatim from CortexReach extraction-prompts.ts.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "build_extraction_prompt",
    "build_dedup_prompt",
    "build_merge_prompt",
    "build_reflection_prompt",
]


def _format_existing_memories(existing_memories: Sequence[dict]) -> str:
    """Format a sequence of memory dicts into the numbered list the LLM expects.

    Mirrors the inline formatting in smart-extractor.ts ``llmDedupDecision``:
        `${i + 1}. [${category}] ${abstract}\\n   Overview: ${overview}\\n   Score: ${score}`

    Fields read from each dict (all optional with sensible defaults):
        - ``abstract`` / ``l0_abstract`` — L0 abstract text
        - ``text`` — fallback when no abstract field is present
        - ``overview`` / ``l1_overview`` — L1 overview text
        - ``category`` / ``memory_category`` — category label
        - ``score`` — similarity score (float)
    """
    lines: list[str] = []
    for i, mem in enumerate(existing_memories, start=1):
        abstract = (
            mem.get("abstract")
            or mem.get("l0_abstract")
            or mem.get("text", "")
        )
        overview = mem.get("overview") or mem.get("l1_overview") or ""
        category = mem.get("category") or mem.get("memory_category") or ""
        score = mem.get("score")
        score_str = f"{float(score):.3f}" if score is not None else "0.000"
        lines.append(
            f"{i}. [{category}] {abstract}\n   Overview: {overview}\n   Score: {score_str}"
        )
    return "\n".join(lines)


def build_extraction_prompt(conversation_text: str, user: str = "user") -> str:
    """Return the extraction prompt with 6-category decision table and few-shot examples.

    Args:
        conversation_text: The recent conversation to analyse.
        user: The user identifier to interpolate into the prompt.

    Returns:
        The fully-rendered extraction prompt string.
    """
    return f"""Analyze the following session context and extract memories worth long-term preservation.

User: {user}

Target Output Language: auto (detect from recent messages)

## Recent Conversation
{conversation_text}

# Memory Extraction Criteria

## What is worth remembering?
- Personalized information: Information specific to this user, not general domain knowledge
- Long-term validity: Information that will still be useful in future sessions
- Specific and clear: Has concrete details, not vague generalizations

## What is NOT worth remembering?
- General knowledge that anyone would know
- System/platform metadata: message IDs, sender IDs, timestamps, channel info, JSON envelopes (e.g. "System: [timestamp] Feishu...", "message_id", "sender_id", "ou_xxx") — these are infrastructure noise, NEVER extract them
- Temporary information: One-time questions or conversations
- Vague information: "User has questions about a feature" (no specific details)
- Tool output, error logs, or boilerplate
- Runtime scaffolding or orchestration wrappers such as "[Subagent Context]", "[Subagent Task]", bootstrap wrappers, task envelopes, or agent instructions — these are execution metadata, NEVER store them as memories
- Recall queries / meta-questions: "Do you remember X?", "你还记得X吗?", "你知道我喜欢什么吗" — these are retrieval requests, NOT new information to store
- Degraded or incomplete references: If the user mentions something vaguely ("that thing I said"), do NOT invent details or create a hollow memory

# Memory Classification

## Core Decision Logic

| Question | Answer | Category |
|----------|--------|----------|
| Who is the user? | Identity, attributes | profile |
| What does the user prefer? | Preferences, habits | preferences |
| What is this thing? | Person, project, organization | entities |
| What happened? | Decision, milestone | events |
| How was it solved? | Problem + solution | cases |
| What is the process? | Reusable steps | patterns |

## Precise Definition

**profile** - User identity (static attributes). Test: "User is..."
**preferences** - User preferences (tendencies). Test: "User prefers/likes..."
**entities** - Continuously existing nouns. Test: "XXX's state is..."
**events** - Things that happened. Test: "XXX did/completed..."
**cases** - Problem + solution pairs. Test: Contains "problem -> solution"
**patterns** - Reusable processes. Test: Can be used in "similar situations"

## Common Confusion
- "Plan to do X" -> events (action, not entity)
- "Project X status: Y" -> entities (describes entity)
- "User prefers X" -> preferences (not profile)
- "Encountered problem A, used solution B" -> cases (not events)
- "General process for handling certain problems" -> patterns (not cases)

# Three-Level Structure

Each memory contains three levels:

**abstract (L0)**: One-liner index
- Merge types (preferences/entities/profile/patterns): `[Merge key]: [Description]`
- Independent types (events/cases): Specific description

**overview (L1)**: Structured Markdown summary with category-specific headings

**content (L2)**: Full narrative with background and details

# Few-shot Examples

## profile
```json
{{
  "category": "profile",
  "abstract": "User basic info: AI development engineer, 3 years LLM experience",
  "overview": "## Background\\n- Occupation: AI development engineer\\n- Experience: 3 years LLM development\\n- Tech stack: Python, LangChain",
  "content": "User is an AI development engineer with 3 years of LLM application development experience."
}}
```

## preferences
```json
{{
  "category": "preferences",
  "abstract": "Python code style: No type hints, concise and direct",
  "overview": "## Preference Domain\\n- Language: Python\\n- Topic: Code style\\n\\n## Details\\n- No type hints\\n- Concise function comments\\n- Direct implementation",
  "content": "User prefers Python code without type hints, with concise function comments."
}}
```

## cases
```json
{{
  "category": "cases",
  "abstract": "LanceDB BigInt numeric handling issue",
  "overview": "## Problem\\nLanceDB 0.26+ returns BigInt for numeric columns\\n\\n## Solution\\nCoerce values with Number(...) before arithmetic",
  "content": "When LanceDB returns BigInt values, wrap them with Number() before doing arithmetic operations."
}}
```

# Output Format

Return JSON:
{{
  "memories": [
    {{
      "category": "profile|preferences|entities|events|cases|patterns",
      "abstract": "One-line index",
      "overview": "Structured Markdown summary",
      "content": "Full narrative"
    }}
  ]
}}

Notes:
- Output language should match the dominant language in the conversation
- Only extract truly valuable personalized information
- If nothing worth recording, return {{"memories": []}}
- Maximum 5 memories per extraction
- Preferences should be aggregated by topic"""


def build_dedup_prompt(
    abstract: str,
    overview: str,
    content: str,
    existing_memories: Sequence[dict],
) -> str:
    """Return the 7-decision dedup prompt.

    Args:
        abstract: L0 abstract of the candidate memory.
        overview: L1 overview of the candidate memory.
        content: L2 content of the candidate memory.
        existing_memories: Sequence of existing memory dicts to compare against.
            Each dict may contain: ``text``, ``abstract``, ``l0_abstract``,
            ``overview``, ``l1_overview``, ``category``, ``memory_category``,
            ``score``.

    Returns:
        The fully-rendered dedup prompt string.
    """
    existing_formatted = _format_existing_memories(existing_memories)
    return f"""Determine how to handle this candidate memory.

**Candidate Memory**:
Abstract: {abstract}
Overview: {overview}
Content: {content}

**Existing Similar Memories**:
{existing_formatted}

Please decide:
- SKIP: Candidate memory duplicates existing memories, no need to save. Also SKIP if the candidate contains LESS information than an existing memory on the same topic (information degradation — e.g., candidate says "programming language preference" but existing memory already says "programming language preference: Python, TypeScript")
- CREATE: This is completely new information not covered by any existing memory, should be created
- MERGE: Candidate memory adds genuinely NEW details to an existing memory and should be merged
- SUPERSEDE: Candidate states that the same mutable fact has changed over time. Keep the old memory as historical but no longer current, and create a new current memory.
- SUPPORT: Candidate reinforces/confirms an existing memory in a specific context (e.g. "still prefers tea in the evening")
- CONTEXTUALIZE: Candidate adds a situational nuance to an existing memory (e.g. existing: "likes coffee", candidate: "prefers tea at night" — different context, same topic)
- CONTRADICT: Candidate directly contradicts an existing memory in a specific context (e.g. existing: "runs on weekends", candidate: "stopped running on weekends")

IMPORTANT:
- "events" and "cases" categories are independent records — they do NOT support MERGE/SUPERSEDE/SUPPORT/CONTEXTUALIZE/CONTRADICT. For these categories, only use SKIP or CREATE.
- If the candidate appears to be derived from a recall question (e.g., "Do you remember X?" / "你记得X吗？") and an existing memory already covers topic X with equal or more detail, you MUST choose SKIP.
- A candidate with less information than an existing memory on the same topic should NEVER be CREATED or MERGED — always SKIP.
- For "preferences" and "entities", use SUPERSEDE when the candidate replaces the current truth instead of adding detail or context. Example: existing "Preferred editor: VS Code", candidate "Preferred editor: Zed".
- For SUPPORT/CONTEXTUALIZE/CONTRADICT, you MUST provide a context_label from this vocabulary: general, morning, evening, night, weekday, weekend, work, leisure, summer, winter, travel.

Return JSON format:
{{
  "decision": "skip|create|merge|supersede|support|contextualize|contradict",
  "match_index": 1,
  "reason": "Decision reason",
  "context_label": "evening"
}}

- If decision is "merge"/"supersede"/"support"/"contextualize"/"contradict", set "match_index" to the number of the existing memory (1-based).
- Only include "context_label" for support/contextualize/contradict decisions."""


def build_merge_prompt(
    existing_abstract: str,
    existing_overview: str,
    existing_content: str,
    new_abstract: str,
    new_overview: str,
    new_content: str,
    category: str,
) -> str:
    """Return the merge synthesis prompt.

    Args:
        existing_abstract: L0 abstract of the existing memory.
        existing_overview: L1 overview of the existing memory.
        existing_content: L2 content of the existing memory.
        new_abstract: L0 abstract of the new information.
        new_overview: L1 overview of the new information.
        new_content: L2 content of the new information.
        category: Memory category (e.g. "profile", "preferences", …).

    Returns:
        The fully-rendered merge prompt string.
    """
    return f"""Merge the following memory into a single coherent record with all three levels.

** Category **: {category}

** Existing Memory:**
    Abstract: {existing_abstract}
  Overview:
{existing_overview}
  Content:
{existing_content}

** New Information:**
    Abstract: {new_abstract}
  Overview:
{new_overview}
  Content:
{new_content}

  Requirements:
  - Remove duplicate information
    - Keep the most up - to - date details
      - Maintain a coherent narrative
        - Keep code identifiers / URIs / model names unchanged when they are proper nouns

Return JSON:
  {{
    "abstract": "Merged one-line abstract",
      "overview": "Merged structured Markdown overview",
        "content": "Merged full content"
  }} """


def build_reflection_prompt(conversation_text: str) -> str:
    """Return the session-reflection prompt.

    Asks the LLM to distil a completed session into two buckets:
    ``invariants`` (durable truths, weeks-long relevance) and
    ``derived`` (this-run takeaways and next-run actions, days-long
    relevance). The reflection layer ranks each bucket with its own
    logistic-decay curve, so the split matters.

    Args:
        conversation_text: The full session transcript to reflect on.

    Returns:
        The fully-rendered reflection prompt string.
    """
    return f"""Review this completed assistant session and distil a concise reflection.

## Session Transcript
{conversation_text}

# Task

Produce two lists:

- **invariants** — durable truths about the user or the working
  relationship that will still matter weeks from now: standing
  preferences, hard constraints, settled decisions. Stable, not transient.
- **derived** — this-session takeaways and concrete next-run actions.
  Short-lived and specific to recent work.

# Rules

- Each entry is ONE short sentence. No markdown, no bullet characters.
- At most 6 invariants, at most 8 derived.
- Only include what the transcript actually supports — never invent.
- If a list has nothing worth keeping, return it empty.
- Skip infrastructure noise, tool logs, and one-off chatter.

Return ONLY this JSON object:
{{
  "invariants": ["...", "..."],
  "derived": ["...", "..."]
}}"""
