"""`valid_until` enforcement on recall.

The smart extractor infers `valid_until` (epoch ms) from phrases like
"tomorrow" or "in 3 days". Every recall path must hide rows past it; admin
paths must still see them; MEMORY_ENFORCE_VALID_UNTIL=off restores the
record-only behaviour.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import time

import pytest

from hermes_memory_lancedb_pro._sql import is_expired

NOW = 1_700_000_000_000
HOUR = 3_600_000


# ---------------------------------------------------------------------------
# Pure helper — no lancedb required
# ---------------------------------------------------------------------------


class TestIsExpired:
    @pytest.mark.parametrize(
        "valid_until",
        [NOW - 1, NOW, NOW - 30 * 24 * HOUR, str(NOW - HOUR), float(NOW - HOUR)],
    )
    def test_past_or_now_is_expired(self, valid_until):
        assert is_expired({"valid_until": valid_until}, NOW) is True

    @pytest.mark.parametrize(
        "metadata",
        [
            {"valid_until": NOW + 1},
            {"valid_until": None},
            {"valid_until": 0},
            {"valid_until": -5},
            {"valid_until": True},
            {"valid_until": "soon"},
            {"valid_until": [NOW]},
            {},
            "not json",
            None,
        ],
    )
    def test_future_missing_or_malformed_is_not_expired(self, metadata):
        # A malformed timestamp must never hide a memory.
        assert is_expired(metadata, NOW) is False

    def test_accepts_json_string_metadata(self):
        assert is_expired(f'{{"valid_until": {NOW - 1}}}', NOW) is True


# ---------------------------------------------------------------------------
# Store / retriever integration
# ---------------------------------------------------------------------------

lancedb = pytest.importorskip("lancedb")

from hermes_memory_lancedb_pro import store as store_module  # noqa: E402
from hermes_memory_lancedb_pro._cli import _cmd_doctor  # noqa: E402
from hermes_memory_lancedb_pro.retriever import MemoryRetriever  # noqa: E402
from hermes_memory_lancedb_pro.store import VECTOR_DIM, MemoryStore  # noqa: E402

pytestmark_integration = pytest.mark.integration

SESSION = "sess-expiry"
EXPIRED = "dentist appointment for the zebra project"
LIVE_FUTURE = "dentist appointment for the walrus project"
PERMANENT = "dentist appointment preference for the falcon project"


class StubEmbedder:
    """Deterministic embedder; texts sharing a word overlap in vector space."""

    def encode(self, text, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(text, str):
            return self._one(text)
        return [self._one(t) for t in text]

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * VECTOR_DIM
        for word in text.lower().split():
            h = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(0, VECTOR_DIM, 8):
                vec[i + (h[i % 32] % 8)] += 1.0
        n = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / n for v in vec]


@pytest.fixture
def seeded():
    """A store holding one expired, one future-dated, and one permanent row."""
    d = tempfile.mkdtemp(prefix="hermes-valid-until-")
    s = MemoryStore(db_path=d)
    s._initialise()
    s._embedder = StubEmbedder()
    now = int(time.time() * 1000)
    ids = {
        "expired": s.store(
            text=EXPIRED, category="fact",
            metadata_extra={"valid_until": now - HOUR, "source_session": SESSION},
        ),
        "future": s.store(
            text=LIVE_FUTURE, category="fact",
            metadata_extra={"valid_until": now + 24 * HOUR, "source_session": SESSION},
        ),
        "permanent": s.store(
            text=PERMANENT, category="fact",
            metadata_extra={"source_session": SESSION},
        ),
    }
    s.optimize()  # fold new rows into the FTS index for bm25 / hybrid
    yield s, ids
    shutil.rmtree(d, ignore_errors=True)


def _texts(rows) -> set[str]:
    return {r.get("text") for r in rows}


@pytestmark_integration
class TestRecallHidesExpired:
    @pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
    def test_search_excludes_expired(self, seeded, mode):
        s, _ = seeded
        found = _texts(s.search("dentist appointment project", limit=10, mode=mode))
        assert EXPIRED not in found
        assert LIVE_FUTURE in found
        assert PERMANENT in found

    def test_session_scoped_search_excludes_expired(self, seeded):
        s, _ = seeded
        found = _texts(s.search(
            "dentist appointment project", limit=10, mode="vector", session_id=SESSION,
        ))
        assert EXPIRED not in found
        assert {LIVE_FUTURE, PERMANENT} <= found

    def test_retriever_excludes_expired(self, seeded):
        s, _ = seeded
        found = _texts(MemoryRetriever(s).retrieve(
            "dentist appointment project", limit=10, source="test",
        ))
        assert EXPIRED not in found
        assert found, "retriever should still return the live rows"

    def test_search_by_vector_excludes_expired(self, seeded):
        # Extractor dedup path: an expired row must not absorb a new memory.
        s, _ = seeded
        found = _texts(s.search_by_vector(s.encode(EXPIRED), limit=10))
        assert EXPIRED not in found
        assert LIVE_FUTURE in found

    def test_session_anchors_exclude_expired(self, seeded):
        s, _ = seeded
        recent = _texts(s.recent_for_session(SESSION, limit=10))
        first = _texts(s.first_for_session(SESSION, limit=10))
        for anchors in (recent, first):
            assert EXPIRED not in anchors
            assert {LIVE_FUTURE, PERMANENT} <= anchors


@pytestmark_integration
class TestAdminPathsStillSeeExpired:
    def test_get_by_id_returns_expired_row(self, seeded):
        s, ids = seeded
        row = s.get_by_id(ids["expired"])
        assert row is not None and row["text"] == EXPIRED

    def test_list_memories_includes_expired_row(self, seeded):
        s, _ = seeded
        assert EXPIRED in _texts(s.list_memories(limit=50))

    def test_doctor_reports_expired_count(self, seeded):
        s, ids = seeded
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            assert _cmd_doctor(type("A", (), {"path": s.db_path, "quiet": True})(), _store=s) == 0
        finally:
            sys.stdout = old
        out = buf.getvalue()
        assert "valid_until enforcement: on" in out
        assert "expired memories (hidden from recall): 1" in out
        assert ids["expired"] in out


@pytestmark_integration
class TestEnforcementSwitch:
    def test_off_switch_restores_expired_rows(self, seeded, monkeypatch):
        s, _ = seeded
        monkeypatch.setattr(store_module, "ENFORCE_VALID_UNTIL", False)
        assert EXPIRED in _texts(s.search("dentist appointment project", limit=10, mode="vector"))
        assert EXPIRED in _texts(s.search_by_vector(s.encode(EXPIRED), limit=10))
        assert EXPIRED in _texts(s.recent_for_session(SESSION, limit=10))

    def test_archived_rows_hidden_even_when_switch_off(self, seeded, monkeypatch):
        s, ids = seeded
        monkeypatch.setattr(store_module, "ENFORCE_VALID_UNTIL", False)
        s.update(ids["permanent"], text="dentist appointment preference revised falcon")
        found = _texts(s.search("dentist appointment project", limit=10, mode="vector"))
        assert PERMANENT not in found  # archived by the supersede


@pytestmark_integration
class TestEndToEndExpiry:
    """Real SmartExtractor → real store → recall, across a clock jump."""

    class _Llm:
        def __init__(self, content: str) -> None:
            self.content = content

        def complete_json(self, prompt, *, label=""):
            if label.startswith("extract"):
                return {"memories": [{
                    "category": "events", "abstract": self.content,
                    "overview": "", "content": self.content,
                }]}
            return {"decision": "create"}

    def test_event_recalled_until_expiry_then_hidden(self, monkeypatch):
        from hermes_memory_lancedb_pro.smart_extractor import SmartExtractor

        d = tempfile.mkdtemp(prefix="hermes-valid-until-e2e-")
        try:
            s = MemoryStore(db_path=d)
            s._initialise()
            s._embedder = StubEmbedder()
            text = "Quarterly review with the otter team tomorrow"
            SmartExtractor(s, llm=self._Llm(text)).extract_and_persist(
                user_content=text, assistant_content="Noted.", session_key=SESSION,
            )
            query = "otter team quarterly review"
            assert text in _texts(s.search(query, limit=5, mode="vector"))

            real_time = time.time
            monkeypatch.setattr(time, "time", lambda: real_time() + 2 * 86_400)
            assert text not in _texts(s.search(query, limit=5, mode="vector"))
            assert s.list_memories(limit=10), "row still exists for admin paths"
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# purge --include-expired
# ---------------------------------------------------------------------------

DAY = 24 * HOUR


class _PurgeArgs:
    def __init__(self, **kw):
        self.grace_days, self.dry_run, self.yes, self.quiet = 30, False, True, False
        self.include_expired, self.path = False, None
        self.__dict__.update(kw)


@pytest.fixture
def purge_store():
    """Rows covering every purge-relevant case, keyed by role."""
    d = tempfile.mkdtemp(prefix="hermes-purge-expired-")
    s = MemoryStore(db_path=d)
    s._initialise()
    s._embedder = StubEmbedder()
    now = int(time.time() * 1000)
    meta = {
        "long_expired": {"valid_until": now - 40 * DAY},
        "just_expired": {"valid_until": now - HOUR},           # inside grace
        "future": {"valid_until": now + DAY},
        "permanent": {},
        "malformed": {"valid_until": "not-a-timestamp"},
    }
    ids = {k: s.store(text=f"purge fixture {k}", category="fact", metadata_extra=v)
           for k, v in meta.items()}
    # Archived long ago AND long expired: purge_archived's job, counted once.
    ids["archived_expired"] = s.store_raw(
        text="purge fixture archived_expired", vector=s.encode("archived expired"),
        category="fact", scope="global", importance=0.4,
        metadata=json.dumps({"state": "archived", "invalidated_at": now - 40 * DAY,
                             "valid_until": now - 40 * DAY}),
    )
    yield s, ids
    shutil.rmtree(d, ignore_errors=True)


def _remaining(s) -> set[str]:
    return {r["id"] for r in s._scan_all()}


@pytestmark_integration
class TestPurgeExpired:
    def test_store_purge_expired_respects_grace(self, purge_store):
        s, ids = purge_store
        assert s.purge_expired(grace_period_days=30) == 1
        left = _remaining(s)
        assert ids["long_expired"] not in left
        for keep in ("just_expired", "future", "permanent", "malformed", "archived_expired"):
            assert ids[keep] in left, keep

    def test_store_purge_expired_dry_run_deletes_nothing(self, purge_store):
        s, ids = purge_store
        before = _remaining(s)
        assert s.purge_expired(grace_period_days=30, dry_run=True) == 1
        assert _remaining(s) == before

    def test_zero_grace_removes_everything_expired_but_not_malformed(self, purge_store):
        s, ids = purge_store
        assert s.purge_expired(grace_period_days=0) == 2
        left = _remaining(s)
        assert ids["long_expired"] not in left and ids["just_expired"] not in left
        assert ids["malformed"] in left

    def test_cli_default_leaves_expired_rows(self, purge_store, capsys):
        s, ids = purge_store
        from hermes_memory_lancedb_pro._cli import _cmd_purge

        assert _cmd_purge(_PurgeArgs(), _store=s) == 0
        left = _remaining(s)
        assert ids["long_expired"] in left            # untouched without the flag
        assert ids["archived_expired"] not in left    # normal archived purge
        assert "expired more than" not in capsys.readouterr().err

    def test_cli_include_expired_reports_each_kind_once(self, purge_store, capsys):
        s, ids = purge_store
        from hermes_memory_lancedb_pro._cli import _cmd_purge

        assert _cmd_purge(_PurgeArgs(include_expired=True), _store=s) == 0
        err = capsys.readouterr().err
        assert "purged 1 archived memories older than 30 days" in err
        assert "purged 1 memories expired more than 30 days ago" in err
        left = _remaining(s)
        assert ids["long_expired"] not in left and ids["archived_expired"] not in left
        assert {ids["just_expired"], ids["future"], ids["permanent"], ids["malformed"]} <= left

    def test_cli_include_expired_dry_run(self, purge_store, capsys):
        s, _ = purge_store
        from hermes_memory_lancedb_pro._cli import _cmd_purge

        before = _remaining(s)
        assert _cmd_purge(_PurgeArgs(include_expired=True, dry_run=True), _store=s) == 0
        assert "[dry-run] would purge 1 memories expired" in capsys.readouterr().err
        assert _remaining(s) == before

    def test_cli_notes_when_enforcement_off(self, purge_store, capsys, monkeypatch):
        s, _ = purge_store
        from hermes_memory_lancedb_pro._cli import _cmd_purge

        monkeypatch.setattr(store_module, "ENFORCE_VALID_UNTIL", False)
        _cmd_purge(_PurgeArgs(include_expired=True, dry_run=True), _store=s)
        assert "MEMORY_ENFORCE_VALID_UNTIL is off" in capsys.readouterr().err


def test_include_expired_flag_parses_on_both_cli_surfaces(monkeypatch):
    import argparse

    from hermes_memory_lancedb_pro import _cli

    parser = argparse.ArgumentParser(prog="hermes lancedb_pro")
    _cli.register_cli(parser)
    args = parser.parse_args(["purge", "--include-expired", "--grace-days", "7", "--dry-run"])
    assert args.include_expired is True and args.grace_days == 7

    seen = {}
    monkeypatch.setattr(_cli, "_cmd_purge", lambda a: seen.setdefault("a", a) and 0)
    monkeypatch.setattr(sys, "argv", ["hermes-memory", "purge", "--include-expired", "--dry-run"])
    _cli.main()
    assert seen["a"].include_expired is True
