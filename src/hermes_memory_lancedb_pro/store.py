"""
LanceDB-backed memory store with nomic-embed-text embeddings.

Schema:
  id: UUID string
  text: str (BM25-indexed)
  vector: float768[] (cosine-indexed)
  category: str (preference, fact, decision, entity, other, reflection)
  scope: str (global, agent, project, user, custom)
  importance: float [0.0-1.0]
  timestamp: int (epoch ms)
  metadata: str (JSON: tier, access_count, confidence, temporal_type, ...)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb
from lancedb.pydantic import LanceModel, Vector

# `sentence_transformers` is only needed when we actually have to encode text.
# Importing it eagerly pulls in torch (hundreds of MB) and prevents tests with
# stub embedders from running on lightweight hosts.
if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

from ._sql import (
    ARCHIVED_STATE,
)
from ._sql import (
    and_clauses as _and_clauses,
)
from ._sql import (
    escape_sql as _escape_sql,
)
from ._sql import (
    is_archived as _is_archived,
)
from ._sql import (
    parse_metadata as _parse_metadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH: str = os.path.expanduser(
    os.environ.get("MEMORY_DB_DIR", "~/.hermes/memory-lancedb")
)
DEFAULT_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM: int = 768

# Reasonable upper bound for full-table scans inside the API. Configurable
# via env var so users with very large stores can lift it.
MAX_SCAN_ROWS: int = int(os.environ.get("MEMORY_MAX_SCAN_ROWS", "100000"))

# Heuristic: when filtering archived rows post-fetch, fetch this many
# multiples of `limit` to leave headroom.
SEARCH_OVERFETCH_MULTIPLIER: int = 3

# Index training threshold — LanceDB IVF_PQ needs ~256 rows minimum
VECTOR_INDEX_MIN_ROWS: int = 256

MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other", "reflection"]
MEMORY_TIERS = ["core", "working", "peripheral"]
MEMORY_STATES = ["pending", "confirmed", "archived"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class MemorySchema(LanceModel):
    """LanceDB table schema for memory entries."""
    id: str
    text: str
    vector: Vector(dim=VECTOR_DIM)
    category: str
    scope: str
    importance: float
    timestamp: int
    metadata: str  # JSON string


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """LanceDB-backed memory store with hybrid retrieval support."""

    # One instance per (resolved) db_path. The previous implementation used
    # a single class-level `_instance` that ignored db_path arguments, which
    # silently returned the wrong store when callers used different paths.
    _instances: dict[str, MemoryStore] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.embedding_model_name = embedding_model
        self._embedder: SentenceTransformer | None = None
        self._db = None
        self._table = None
        self._initialised = False

    # ----- lifecycle -----

    @classmethod
    def get_instance(
        cls,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> MemoryStore:
        """Return a path-keyed singleton, initialised on first use."""
        key = str(Path(db_path).expanduser().resolve())
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(db_path=db_path, embedding_model=embedding_model)
                inst._initialise()
                cls._instances[key] = inst
            return inst

    def _initialise(self):
        """Connect to LanceDB and ensure the memories table exists.
        The embedder is loaded lazily on first encode call."""
        if self._initialised:
            return
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.db_path)
        self._ensure_table()
        self._initialised = True

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # heavy import
            logger.debug("Loading embedding model: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)

    def _ensure_table(self):
        """Open or create the memories table; create the FTS index on `text`."""
        table_name = "memories"
        # `list_tables()` is the supported API; `table_names()` is deprecated
        # in modern LanceDB but still present as a back-compat shim.
        try:
            existing_tables = self._db.list_tables()
        except AttributeError:
            existing_tables = self._db.table_names()
        if table_name in existing_tables:
            self._table = self._db.open_table(table_name)
            return

        self._table = self._db.create_table(table_name, schema=MemorySchema)
        try:
            self._table.create_fts_index("text")
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("Failed to create FTS index on 'text': %s", e)

    def maybe_create_vector_index(self) -> bool:
        """Create the IVF_PQ vector index when there are enough rows to train.
        Returns True if the index was just created, False otherwise."""
        try:
            row_count = len(self._table)
        except Exception:
            return False
        if row_count < VECTOR_INDEX_MIN_ROWS:
            return False
        try:
            self._table.create_index(vector_column_name="vector")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "exists" in msg:
                return False
            logger.warning("Vector index creation failed: %s", e)
            return False

    # ----- embedding -----

    @property
    def embedder(self) -> SentenceTransformer:
        self._load_embedder()
        return self._embedder

    @staticmethod
    def _to_list(vec) -> list[float]:
        """Normalise an embedder return value to a Python list."""
        if isinstance(vec, list):
            return vec
        return vec.tolist()  # numpy.ndarray, torch.Tensor, etc.

    def encode(self, text: str) -> list[float]:
        """Encode text into a single normalised 768-d vector."""
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return self._to_list(vector)

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode many texts in one shot. Use this for bulk inserts."""
        vectors = self.embedder.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [self._to_list(v) for v in vectors]

    # ----- internals -----

    def _build_metadata(
        self,
        *,
        tier: str,
        confidence: float,
        now_ms: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = {
            "tier": tier,
            "access_count": 0,
            "confidence": confidence,
            "temporal_type": "static",
            "state": "confirmed",
            "source": "manual",
            "source_session": "",
            "created_at": now_ms,
            "last_accessed_at": now_ms,
            "injected_count": 0,
            "bad_recall_count": 0,
            "supersedes": None,
            "superseded_by": None,
            "valid_from": now_ms,
            "valid_until": None,
            "fact_key": None,
            "relations": [],
        }
        if extra:
            meta.update(extra)
        return meta

    @staticmethod
    def _normalise_inputs(
        category: str,
        tier: str,
        importance: float,
        confidence: float,
    ):
        if category not in MEMORY_CATEGORIES:
            category = "other"
        if tier not in MEMORY_TIERS:
            tier = "working"
        importance = max(0.0, min(1.0, float(importance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        return category, tier, importance, confidence

    # ----- CRUD: write -----

    def store(
        self,
        text: str,
        category: str = "other",
        scope: str = "global",
        importance: float = 0.5,
        tier: str = "working",
        confidence: float = 0.8,
        metadata_extra: dict[str, Any] | None = None,
    ) -> str:
        """Store a new memory entry. Returns the new memory ID."""
        if not text or not text.strip():
            raise ValueError("MemoryStore.store: `text` must be non-empty")

        category, tier, importance, confidence = self._normalise_inputs(
            category, tier, importance, confidence
        )

        mem_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        metadata = self._build_metadata(
            tier=tier, confidence=confidence, now_ms=now_ms, extra=metadata_extra
        )

        vector = self.encode(text)
        self._table.add([
            MemorySchema(
                id=mem_id,
                text=text,
                vector=vector,
                category=category,
                scope=scope,
                importance=importance,
                timestamp=now_ms,
                metadata=json.dumps(metadata),
            )
        ])
        return mem_id

    def store_many(
        self,
        entries: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Bulk-store entries. Each entry is a dict of `store()` kwargs.

        Encodes all texts in one batch and writes the rows in one transaction —
        much faster than calling `store()` in a loop."""
        if not entries:
            return []

        prepared: list[MemorySchema] = []
        ids: list[str] = []
        texts = [str(e["text"]) for e in entries]
        vectors = self.encode_batch(texts)

        now_ms = int(time.time() * 1000)
        for entry, vector in zip(entries, vectors, strict=True):
            text = entry["text"]
            if not text or not str(text).strip():
                raise ValueError("store_many: every entry must have non-empty `text`")

            category, tier, importance, confidence = self._normalise_inputs(
                entry.get("category", "other"),
                entry.get("tier", "working"),
                entry.get("importance", 0.5),
                entry.get("confidence", 0.8),
            )
            scope = entry.get("scope", "global")
            extra = entry.get("metadata_extra")
            mem_id = str(uuid.uuid4())
            metadata = self._build_metadata(
                tier=tier, confidence=confidence, now_ms=now_ms, extra=extra
            )
            prepared.append(
                MemorySchema(
                    id=mem_id,
                    text=text,
                    vector=vector,
                    category=category,
                    scope=scope,
                    importance=importance,
                    timestamp=now_ms,
                    metadata=json.dumps(metadata),
                )
            )
            ids.append(mem_id)

        self._table.add(prepared)
        return ids

    def update(
        self,
        mem_id: str,
        text: str | None = None,
        importance: float | None = None,
        category: str | None = None,
        tier: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing memory.

        If `text` is provided this uses the supersede pattern: the existing
        row is marked archived and a new row with a fresh UUID replaces it.
        Otherwise the existing row's columns and metadata are updated in-place.

        Returns True if the memory was found and updated."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False

        metadata = _parse_metadata(existing["metadata"])
        now_ms = int(time.time() * 1000)

        if text is not None:
            return self._supersede(
                existing=existing,
                old_metadata=metadata,
                text=text,
                importance=importance,
                category=category,
                tier=tier,
                metadata_extra=metadata_extra,
                now_ms=now_ms,
            )

        # Metadata-only update path
        update_values: dict[str, Any] = {}

        if tier is not None:
            if tier not in MEMORY_TIERS:
                tier = "working"
            metadata["tier"] = tier
        if metadata_extra:
            metadata.update(metadata_extra)
        metadata["last_modified_at"] = now_ms

        # Top-level columns are authoritative — write to columns AND keep
        # metadata clean of duplicate column values.
        update_values["metadata"] = json.dumps(metadata)
        if importance is not None:
            update_values["importance"] = max(0.0, min(1.0, float(importance)))
        if category is not None:
            if category not in MEMORY_CATEGORIES:
                category = "other"
            update_values["category"] = category

        self._table.update(
            where=f"id = '{_escape_sql(mem_id)}'",
            values=update_values,
        )
        return True

    def _supersede(
        self,
        *,
        existing: dict[str, Any],
        old_metadata: dict[str, Any],
        text: str,
        importance: float | None,
        category: str | None,
        tier: str | None,
        metadata_extra: dict[str, Any] | None,
        now_ms: int,
    ) -> bool:
        old_id = existing["id"]
        new_id = str(uuid.uuid4())

        # 1. Archive the old row (still readable for audit, hidden from queries)
        archived_meta = dict(old_metadata)
        archived_meta["state"] = ARCHIVED_STATE
        archived_meta["superseded_by"] = new_id
        archived_meta["invalidated_at"] = now_ms

        # 2. Fresh metadata for the new row — start from the old metadata so
        # custom fields (fact_key, source, relations…) carry forward, but
        # reset lifecycle counters and link back to the predecessor.
        new_meta = dict(old_metadata)
        new_meta.pop("invalidated_at", None)
        new_meta.pop("superseded_by", None)
        new_meta["state"] = "confirmed"
        new_meta["supersedes"] = old_id
        new_meta["created_at"] = now_ms
        new_meta["last_accessed_at"] = now_ms
        new_meta["last_modified_at"] = now_ms
        new_meta["access_count"] = 0
        new_meta["injected_count"] = 0
        new_meta["bad_recall_count"] = 0

        # Apply caller overrides
        new_tier = tier if tier in MEMORY_TIERS else new_meta.get("tier", "working")
        new_meta["tier"] = new_tier
        if metadata_extra:
            new_meta.update(metadata_extra)

        # Resolve top-level fields
        new_category = (
            category if (category is not None and category in MEMORY_CATEGORIES)
            else existing["category"]
        )
        new_importance = (
            max(0.0, min(1.0, float(importance)))
            if importance is not None
            else float(existing["importance"])
        )

        vector = self.encode(text)

        # Atomic-ish: do the write of the new row first so a partial failure
        # leaves the old row visible (preferred over a missing row).
        self._table.add([
            MemorySchema(
                id=new_id,
                text=text,
                vector=vector,
                category=new_category,
                scope=existing["scope"],
                importance=new_importance,
                timestamp=now_ms,
                metadata=json.dumps(new_meta),
            )
        ])

        self._table.update(
            where=f"id = '{_escape_sql(old_id)}'",
            values={"metadata": json.dumps(archived_meta)},
        )
        return True

    def forget(self, mem_id: str) -> bool:
        """Hard-delete a memory by ID. Returns True if the memory existed."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        self._table.delete(f"id = '{_escape_sql(mem_id)}'")
        return True

    def increment_access_count(self, mem_id: str) -> bool:
        """Increment access_count and update last_accessed_at."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        metadata = _parse_metadata(existing["metadata"])
        metadata["access_count"] = int(metadata.get("access_count", 0) or 0) + 1
        metadata["last_accessed_at"] = int(time.time() * 1000)
        self._table.update(
            where=f"id = '{_escape_sql(mem_id)}'",
            values={"metadata": json.dumps(metadata)},
        )
        return True

    # ----- CRUD: read -----

    def get_by_id(self, mem_id: str) -> dict[str, Any] | None:
        """Retrieve a memory by ID (regardless of archive state)."""
        results = (
            self._table.search()
            .where(f"id = '{_escape_sql(mem_id)}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return None
        return self._row_to_dict(results[0])

    def has_id(self, mem_id: str) -> bool:
        """True if the ID exists and is not archived. Used for BM25 ghost
        protection — see also `check_ids` for batch lookups."""
        results = (
            self._table.search()
            .where(f"id = '{_escape_sql(mem_id)}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return False
        return not _is_archived(results[0].get("metadata", ""))

    def check_ids(self, mem_ids: Sequence[str]) -> list[str]:
        """Return the subset of `mem_ids` that exist and are not archived.

        Always passes an explicit limit covering every requested id — a
        previous bug omitted `.limit()` and silently returned LanceDB's
        default page size, falsely flagging the rest as ghosts."""
        if not mem_ids:
            return []
        # de-dup but keep original input intact so callers aren't surprised
        unique_ids = list(dict.fromkeys(mem_ids))
        in_clause = ",".join(f"'{_escape_sql(mid)}'" for mid in unique_ids)
        results = (
            self._table.search()
            .where(f"id IN ({in_clause})")
            .limit(max(len(unique_ids), 1))
            .to_list()
        )
        return [
            r["id"] for r in results
            if not _is_archived(r.get("metadata", ""))
        ]

    def list_memories(
        self,
        limit: int = 20,
        category: str | None = None,
        tier: str | None = None,
        scope: str | None = None,
        offset: int = 0,
        include_archived: bool = False,
        # Hermes core may pass `categories` instead of `category`
        categories: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memories with optional filters.

        Filters are AND-combined into a single WHERE clause — chaining
        `.where()` on a LanceQueryBuilder replaces the previous predicate
        in many LanceDB versions, so we must build the clause ourselves."""
        effective_category = category if category is not None else categories
        clauses: list[str] = []
        if effective_category:
            clauses.append(f"category = '{_escape_sql(effective_category)}'")
        if scope:
            clauses.append(f"scope = '{_escape_sql(scope)}'")
        if tier:
            clauses.append(
                f"metadata LIKE '%\"tier\": \"{_escape_sql(tier)}\"%'"
            )

        # Over-fetch when we'll filter archived rows post-query
        fetch_limit = limit if include_archived else limit + offset + max(limit, 16) * 2

        query = self._table.search()
        where_clause = _and_clauses(*clauses)
        if where_clause:
            query = query.where(where_clause)
        results = query.limit(fetch_limit).to_list()

        rows = [self._row_to_dict(r) for r in results]
        if not include_archived:
            rows = [r for r in rows if r["metadata"].get("state") != ARCHIVED_STATE]
        return rows[offset : offset + limit]

    def search(
        self,
        query_text: str,
        limit: int = 10,
        mode: str = "hybrid",  # "vector" | "bm25" | "hybrid"
        category: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories using vector, BM25, or hybrid mode.

        All three modes return `List[Dict[str, Any]]` — the previous version
        returned tuples for hybrid mode, breaking polymorphic callers."""
        if not query_text or not query_text.strip():
            return []
        if mode == "vector":
            return self._vector_search(query_text, limit, category, scope)
        if mode == "bm25":
            return self._bm25_search(query_text, limit, category, scope)
        return self._hybrid_search(query_text, limit, category, scope)

    # ----- search modes -----

    def _filters_clause(
        self,
        category: str | None,
        scope: str | None,
    ) -> str | None:
        clauses: list[str] = []
        if category:
            clauses.append(f"category = '{_escape_sql(category)}'")
        if scope:
            clauses.append(f"scope = '{_escape_sql(scope)}'")
        return _and_clauses(*clauses)

    def _vector_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
        keep_vector: bool = False,
    ) -> list[dict[str, Any]]:
        vector = self.encode(query_text)
        search = self._table.search(vector, vector_column_name="vector")
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        results = search.limit(limit * SEARCH_OVERFETCH_MULTIPLIER).to_list()
        return [
            self._row_to_dict(r, distance=r.get("_distance"), keep_vector=keep_vector)
            for r in results
            if not _is_archived(r.get("metadata", ""))
        ][:limit]

    def _bm25_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
        keep_vector: bool = False,
    ) -> list[dict[str, Any]]:
        # Scope FTS to the `text` column only — LanceDB's FTS otherwise indexes
        # all string columns, polluting results with id/category/metadata.
        search = self._table.search(
            query_text, query_type="fts", fts_columns=["text"]
        )
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        results = search.limit(limit * SEARCH_OVERFETCH_MULTIPLIER).to_list()
        return [
            self._row_to_dict(r, score=r.get("_score"), keep_vector=keep_vector)
            for r in results
            if not _is_archived(r.get("metadata", ""))
        ][:limit]

    def _hybrid_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of vector + BM25 results.

        Always returns plain dicts (with `_rrf_score` set) for parity with
        the other search modes."""
        candidate = max(limit * 2, 10)
        vector_results = self._vector_search(query_text, candidate, category, scope)
        bm25_results = self._bm25_search(query_text, candidate, category, scope)

        # Rank dicts (1-based) — used to compute RRF without O(n²) scans
        v_rank = {r["id"]: i + 1 for i, r in enumerate(vector_results)}
        b_rank = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}

        rrf_k = 60
        merged: dict[str, dict[str, Any]] = {}
        for entry in vector_results:
            mid = entry["id"]
            entry["_rrf_score"] = (
                1.0 / (rrf_k + v_rank[mid])
                + (1.0 / (rrf_k + b_rank[mid]) if mid in b_rank else 0.0)
            )
            merged[mid] = entry

        for entry in bm25_results:
            mid = entry["id"]
            if mid in merged:
                continue
            entry["_rrf_score"] = (
                (1.0 / (rrf_k + v_rank[mid]) if mid in v_rank else 0.0)
                + 1.0 / (rrf_k + b_rank[mid])
            )
            merged[mid] = entry

        ranked = sorted(
            merged.values(),
            key=lambda e: e.get("_rrf_score", 0.0),
            reverse=True,
        )
        return ranked[:limit]

    # ----- bulk ops -----

    def stats(self) -> dict[str, Any]:
        """Return memory store statistics. Counts include archived rows;
        use `purge_archived` to remove them. The category/tier breakdowns
        scan up to MAX_SCAN_ROWS rows; very large stores may want a sampling
        strategy instead."""
        try:
            count = len(self._table)
        except Exception:
            count = 0

        cat_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        active_count = 0
        archived_count = 0

        for row in self._scan_all(limit=MAX_SCAN_ROWS):
            cat = row.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            meta = _parse_metadata(row.get("metadata", "{}"))
            tier = meta.get("tier", "working")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if meta.get("state") == ARCHIVED_STATE:
                archived_count += 1
            else:
                active_count += 1

        return {
            "total_memories": count,
            "active_memories": active_count,
            "archived_memories": archived_count,
            "db_path": self.db_path,
            "embedding_model": self.embedding_model_name,
            "vector_dimensions": VECTOR_DIM,
            "categories": cat_counts,
            "tiers": tier_counts,
        }

    def purge_archived(self, grace_period_days: int = 30) -> int:
        """Permanently delete archived memories older than grace_period_days.

        Returns the number of entries deleted. Iterates pages so stores
        larger than MAX_SCAN_ROWS are still cleaned up correctly."""
        cutoff_ms = int(time.time() * 1000) - (grace_period_days * 24 * 60 * 60 * 1000)
        purged = 0
        for row in self._scan_all(limit=MAX_SCAN_ROWS):
            meta = _parse_metadata(row.get("metadata", "{}"))
            if meta.get("state") != ARCHIVED_STATE:
                continue
            invalidated = meta.get("invalidated_at", 0) or 0
            if invalidated and invalidated < cutoff_ms:
                mem_id = row.get("id")
                if mem_id:
                    self._table.delete(f"id = '{_escape_sql(mem_id)}'")
                    purged += 1

        if purged:
            logger.info(
                "Purged %d archived memories older than %d days",
                purged, grace_period_days,
            )
        return purged

    def _scan_all(self, limit: int = MAX_SCAN_ROWS) -> Iterable[dict[str, Any]]:
        """Iterate every row in the table up to `limit`. Yields raw rows."""
        try:
            results = self._table.search().limit(limit).to_list()
        except Exception as e:
            logger.error("Full table scan failed: %s", e)
            return iter(())
        return iter(results)

    # ----- formatters -----

    @staticmethod
    def _row_to_dict(
        row: dict[str, Any],
        distance: float | None = None,
        score: float | None = None,
        keep_vector: bool = False,
    ) -> dict[str, Any]:
        """Convert a LanceDB row to a clean dictionary.

        Vectors are dropped by default — they're 768-dim floats that bloat
        downstream pipelines. Pass `keep_vector=True` (e.g. from MMR) to
        preserve them."""
        meta = row["metadata"]
        result: dict[str, Any] = {
            "id": row["id"],
            "text": row["text"],
            "category": row["category"],
            "scope": row["scope"],
            "importance": row["importance"],
            "timestamp": row["timestamp"],
            "metadata": _parse_metadata(meta) if isinstance(meta, str) else meta,
        }
        if keep_vector and "vector" in row:
            result["vector"] = row["vector"]
        if distance is not None:
            result["_distance"] = distance
        if score is not None:
            result["_score"] = score
        if "_rrf_score" in row:
            result["_rrf_score"] = row["_rrf_score"]
        return result
