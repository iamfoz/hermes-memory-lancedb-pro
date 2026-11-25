"""Compaction must consider the whole store, and doctor must only recommend
it when there is something to merge."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import tempfile

import pytest

pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import _cli  # noqa: E402
from hermes_memory_lancedb_pro import memory_compactor as mc  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark = pytest.mark.integration


class Embedder:
    """Hash vectors, except that texts starting with "dup" share one vector."""

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False, **_):
        if isinstance(text, str):
            return self._one(text)
        return [self._one(t) for t in text]

    @staticmethod
    def _one(text: str) -> list[float]:
        body = text.split(": ", 1)[-1]  # ignore an embedding task prefix
        key = "dup" if body.startswith("dup") else body
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        floats = [(digest[i % len(digest)] - 128) / 128.0 for i in range(VECTOR_DIM)]
        n = sum(f * f for f in floats) ** 0.5
        return [f / n for f in floats]


@pytest.fixture
def store():
    d = tempfile.mkdtemp(prefix="hermes-compact-")
    s = MemoryStore(db_path=d)
    s._embedder = Embedder()
    s._initialise()
    yield s
    shutil.rmtree(d, ignore_errors=True)


def _now_config(**kw):
    return mc.CompactionConfig(min_age_days=0, **kw)


def test_duplicates_beyond_the_first_200_rows_are_found(store):
    store.store_many([{"text": f"distinct memory {i}", "category": "fact", "scope": "agent"}
                      for i in range(260)])
    a = store.store(text="dup: the staging host is db-7", category="fact", scope="agent")
    b = store.store(text="dup: staging database host db-7", category="fact", scope="agent")
    dry = mc.run_compaction(store, _now_config(dry_run=True))
    assert dry.scanned == 262 and dry.clusters_found == 1
    result = mc.run_compaction(store, _now_config())
    assert result.memories_created == 1 and result.memories_deleted == 2
    assert not store.row_exists(a) and not store.row_exists(b)


def test_the_cap_applies_to_eligible_rows_not_raw_rows(store):
    # 40 rows compaction must skip come first; the cap must not be spent on them.
    store.store_many([{"text": f"pin {i}", "category": "active_task", "scope": "agent"}
                      for i in range(40)])
    store.store(text="dup: one", category="fact", scope="agent")
    store.store(text="dup: two", category="fact", scope="agent")
    dry = mc.run_compaction(store, _now_config(dry_run=True, max_memories_to_scan=10))
    assert dry.scanned == 2 and dry.clusters_found == 1


def test_default_cap_is_well_above_the_old_200():
    assert mc.CompactionConfig().max_memories_to_scan >= 2000


def test_fast_and_pure_similarity_agree(monkeypatch):
    rng = random.Random(7)
    vectors = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(12)]
    vectors[3] = []            # a row with no vector
    vectors[5] = [0.0] * 16    # a zero vector
    vectors[8] = list(vectors[1])
    fast = mc._similarity_rows(vectors)
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *a, **k):
        if name == "numpy":
            raise ImportError("numpy unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    pure = mc._similarity_rows(vectors)
    monkeypatch.undo()
    for i in range(len(vectors)):
        assert list(fast(i)) == pytest.approx(list(pure(i)), abs=1e-9)
    assert fast(1)[8] == pytest.approx(1.0)
    assert fast(3)[1] == 0.0 and fast(5)[1] == 0.0


def test_mixed_dimensions_fall_back_and_do_not_cluster():
    sims = mc._similarity_rows([[1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0]])
    assert sims(0)[2] == pytest.approx(1.0) and sims(0)[1] == 0.0


def _doctor(store, capsys) -> str:
    capsys.readouterr()
    assert _cli._cmd_doctor(argparse.Namespace(path=None, quiet=False), _store=store) == 0
    return capsys.readouterr().out


def test_doctor_recommends_compact_only_when_clusters_exist(store, capsys, monkeypatch):
    # Old rows alone are not a reason: this used to print their count as
    # possible duplicates, and kept printing it after every compaction.
    store.store_many([{"text": f"distinct memory {i}", "category": "fact", "scope": "agent"}
                      for i in range(20)])
    real_config = mc.CompactionConfig

    def no_age_floor(**kw):  # the rows in this test were written just now
        return real_config(min_age_days=0, **kw)

    monkeypatch.setattr(mc, "CompactionConfig", no_age_floor)
    out = _doctor(store, capsys)
    assert "compact --dry-run" not in out

    store.store(text="dup: one", category="fact", scope="agent")
    store.store(text="dup: two", category="fact", scope="agent")
    out = _doctor(store, capsys)
    assert "hermes lancedb_pro compact --dry-run" in out
    assert "1 cluster(s) of near-duplicate memories (2 rows)" in out

    mc.run_compaction(store, real_config(min_age_days=0))
    assert "compact --dry-run" not in _doctor(store, capsys)


def test_doctor_survives_a_failing_cluster_check(store, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(mc, "run_compaction", boom)
    assert "compact --dry-run" not in _doctor(store, capsys)
