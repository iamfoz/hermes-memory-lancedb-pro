"""Pin the keyword-only boundary on high-arity public APIs.

Trailing optional parameters are keyword-only so call sites stay readable and
same-typed arguments cannot be silently transposed (ruff PLR0917).
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro.extraction_prompts import build_merge_prompt
from hermes_memory_lancedb_pro.store import MemoryStore

KW = inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    ("method", "keyword_only"),
    [
        ("store", {"confidence", "metadata_extra"}),
        ("update", {"metadata_extra"}),
        ("list_memories", {"include_archived", "session_id", "categories"}),
        ("search", {"session_id", "min_score"}),
    ],
)
def test_memory_store_keyword_only_params(method, keyword_only):
    params = inspect.signature(getattr(MemoryStore, method)).parameters
    actual = {name for name, p in params.items() if p.kind is KW}
    assert actual == keyword_only


def test_build_merge_prompt_is_fully_keyword_only():
    params = inspect.signature(build_merge_prompt).parameters.values()
    assert all(p.kind is KW for p in params)
