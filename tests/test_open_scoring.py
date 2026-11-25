"""Vector distance metric and score scale.

Covers: cosine distance on every vector query and index (including the
one-time rebuild of an L2 index left by older versions), the hybrid `score`
field following the RRF ranking, and the retriever's score floor being able
to reject a query that matches nothing.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import warnings

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro.retriever import MemoryRetriever  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration


class BagOfWordsEmbedder:
    """One dimension per distinct word, so cosine similarity is exact and
    predictable: two 4-word texts sharing 2 words have cosine 0.5."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text)
        return [self._one(t) for t in text]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * VECTOR_DIM
        for word in text.lower().split():
            # The store adds nomic's task prefix to new databases; these
            # tests are about the distance metric, so ignore it.
            if word in ("search_document:", "search_query:"):
                continue
            idx = self.vocab.setdefault(word, len(self.vocab))
            vec[idx % VECTOR_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _open(path: str, embedder: BagOfWordsEmbedder) -> MemoryStore:
    s = MemoryStore(db_path=path)
    s._embedder = embedder
    s._initialise()
    return s


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="hermes-scoring-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def embedder():
    return BagOfWordsEmbedder()


@pytest.fixture
def store(db_path, embedder):
    return _open(db_path, embedder)


QUERY = "alpha beta gamma delta"
HALF = "alpha beta zeta eta"  # cosine 0.5 to QUERY


def _fill_for_index(s: MemoryStore, n: int = 300) -> None:
    s.store_many([
        {"text": f"row filler{i}", "category": "fact"} for i in range(n)
    ])


# ---------------------------------------------------------------------------
# (a) cosine distance on every vector query
# ---------------------------------------------------------------------------


def test_vector_search_distance_is_cosine(store):
    store.store(text=QUERY, category="fact")
    store.store(text=HALF, category="fact")
    rows = {r["text"]: r for r in store.search(QUERY, limit=5, mode="vector")}
    assert rows[QUERY]["_distance"] == pytest.approx(0.0, abs=1e-5)
    # Squared L2 would report 2 - 2*0.5 = 1.0 and score 0.0.
    assert rows[HALF]["_distance"] == pytest.approx(0.5, abs=1e-5)
    assert rows[HALF]["score"] == pytest.approx(0.5, abs=1e-5)


def test_vector_min_score_uses_cosine_similarity(store):
    store.store(text=QUERY, category="fact")
    store.store(text=HALF, category="fact")
    texts = [r["text"] for r in store.search(QUERY, limit=5, mode="vector", min_score=0.4)]
    assert HALF in texts


def test_search_by_vector_distance_is_cosine(store, embedder):
    store.store(text=HALF, category="fact")
    rows = store.search_by_vector(embedder.encode(QUERY), limit=1)
    assert rows[0]["_distance"] == pytest.approx(0.5, abs=1e-5)


def test_new_vector_index_uses_cosine_and_exact_distances(store, embedder):
    store.store(text=HALF, category="fact")
    _fill_for_index(store)
    assert store.maybe_create_vector_index() is True
    assert store._vector_index_metric() == "cosine"
    # A second call must not retrain (create_index replaces by default).
    assert store.maybe_create_vector_index() is False
    rows = store.search_by_vector(embedder.encode(QUERY), limit=1)
    assert rows[0]["text"] == HALF
    # Refined, not the PQ approximation.
    assert rows[0]["_distance"] == pytest.approx(0.5, abs=1e-4)


def test_existing_l2_index_is_rebuilt_as_cosine_on_open(db_path, embedder):
    s = _open(db_path, embedder)
    s.store(text=HALF, category="fact")
    _fill_for_index(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # What older versions built: LanceDB's default (L2) metric.
        s._table.create_index(vector_column_name="vector")
    assert s._vector_index_metric() == "l2"
    count = s._table.count_rows()

    reopened = _open(db_path, embedder)
    # Opening does not retrain (that would block start-up and make read-only
    # commands write); queries are already correct against the old index.
    assert reopened._vector_index_metric() == "l2"
    rows = reopened.search_by_vector(embedder.encode(QUERY), limit=1)
    assert rows[0]["_distance"] == pytest.approx(0.5, abs=1e-4)

    reopened.warmup()  # the background step does the rebuild
    assert reopened._vector_index_metric() == "cosine"
    assert reopened._table.count_rows() == count
    rows = reopened.search_by_vector(embedder.encode(QUERY), limit=1)
    assert rows[0]["text"] == HALF
    assert rows[0]["_distance"] == pytest.approx(0.5, abs=1e-4)


def test_l2_index_is_dropped_when_rebuild_fails(db_path, embedder, monkeypatch):
    s = _open(db_path, embedder)
    s.store(text=HALF, category="fact")
    _fill_for_index(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s._table.create_index(vector_column_name="vector")
    count = s._table.count_rows()

    def boom(self):
        raise RuntimeError("not enough rows to train")

    # Patch the class this test instantiates (other tests reload the module).
    monkeypatch.setattr(MemoryStore, "_build_vector_index", boom)
    reopened = _open(db_path, embedder)
    reopened.warmup()
    assert reopened._vector_index_metric() is None
    assert reopened._table.count_rows() == count
    rows = reopened.search_by_vector(embedder.encode(QUERY), limit=1)
    assert rows[0]["_distance"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# (c) hybrid `score` follows the RRF ranking
# ---------------------------------------------------------------------------


def test_hybrid_score_is_scaled_rrf_and_monotonic(store):
    store.store(text=QUERY, category="fact")
    store.store(text=HALF, category="fact")
    store.store(text="unrelated words here now", category="fact")
    rows = store.search(QUERY, limit=5, mode="hybrid")
    assert len(rows) >= 2
    for r in rows:
        assert r["score"] == pytest.approx(min(1.0, r["_rrf_score"] * 30.0))
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    # HALF is found by both legs: it keeps the BM25 leg's raw signal too.
    half = next(r for r in rows if r["text"] == HALF)
    assert "_distance" in half and "_score" in half
    assert half["score"] > 0.9


# ---------------------------------------------------------------------------
# (b) the recall score floor can reject a query that matches nothing
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever(store, monkeypatch):
    monkeypatch.setenv("MEMORY_RERANKER", "disabled")
    return MemoryRetriever(store)


def test_unrelated_query_is_rejected_by_score_floor(store, retriever):
    store.store(text=QUERY, category="fact")
    store.store(text="zeta eta theta iota", category="fact")
    # No shared word: cosine 0 to every memory, no BM25 hit. Per-query
    # min-max normalisation used to give the nearest one fusion 0.5-1.0.
    out = retriever.retrieve("kappa lambda mu nu", limit=5, source="cli", min_score=0.2)
    assert out == []


def test_single_unrelated_memory_is_rejected(store, retriever):
    store.store(text=QUERY, category="fact")
    out = retriever.retrieve("kappa lambda mu nu", limit=5, source="cli", min_score=0.2)
    assert out == []


def test_relevant_query_still_passes_floor(store, retriever):
    store.store(text=QUERY, category="fact")
    store.store(text="zeta eta theta iota", category="fact")
    out = retriever.retrieve(QUERY, limit=5, source="cli", min_score=0.2)
    assert [r["text"] for r in out] == [QUERY]


def test_vector_fusion_score_is_absolute_cosine_similarity(retriever):
    vector = [
        {"id": "a", "text": "a", "_distance": 0.6},
        {"id": "b", "text": "b", "_distance": 0.9},
    ]
    fused = {e["id"]: e for e in retriever._vector_dominant_fusion(vector, [])}
    assert fused["a"]["_fusion_score"] == pytest.approx(0.4)
    assert fused["b"]["_fusion_score"] == pytest.approx(0.1)
    # Ordering is unchanged by the switch from min-max normalisation.
    assert fused["a"]["_fusion_score"] > fused["b"]["_fusion_score"]
