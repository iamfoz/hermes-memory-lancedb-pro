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
  metadata: str (JSON: tier, access_count, confidence, temporal_type, etc.)
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import lancedb
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer

# Constants
DEFAULT_DB_PATH = os.path.expanduser("~/.hermes/memory-lancedb")
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM = 768

# SQL LIKE pattern to exclude archived (superseded) memories.
# The metadata JSON contains '"state": "archived"' when a row has been superseded.
# We use a module-level constant to avoid escape-hell in .where() calls.
ARCHIVED_STATE = '"state": "archived"'

# Memory categories
MEMORY_CATEGORIES = [
    "preference", "fact", "decision", "entity", "other", "reflection",
]

# Memory tiers
MEMORY_TIERS = ["core", "working", "peripheral"]

# Memory states
MEMORY_STATES = ["pending", "confirmed", "archived"]


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


class MemoryStore:
    """LanceDB-backed memory store with hybrid retrieval support."""

    _instance: Optional["MemoryStore"] = None

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.db_path = str(Path(db_path).resolve())
        self.embedding_model_name = embedding_model
        self._embedder: Optional[SentenceTransformer] = None
        self._db: Optional[lancedb.DB] = None
        self._table = None

    @classmethod
    def get_instance(
        cls,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "MemoryStore":
        if cls._instance is None:
            cls._instance = cls(db_path=db_path, embedding_model=embedding_model)
            cls._instance._initialise()
        return cls._instance

    def _initialise(self):
        """Initialise LanceDB connection and create table if needed."""
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.db_path)
        self._ensure_table()
        # Lazy-load embedder on first use
        self._load_embedder()

    def _load_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)

    def _ensure_table(self):
        """Create the memories table if it doesn't exist."""
        table_name = "memories"
        if table_name in self._db.table_names():
            self._table = self._db.open_table(table_name)
        else:
            self._table = self._db.create_table(table_name, schema=MemorySchema)
            # Create BM25/FTS index on text column
            self._table.create_fts_index("text")
            # Vector index is created lazily after data is loaded (empty tables can't train)

    @property
    def embedder(self) -> SentenceTransformer:
        self._load_embedder()
        return self._embedder

    def encode(self, text: str) -> list[float]:
        """Encode text to 768-dim vector."""
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts efficiently."""
        vectors = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    # --- CRUD Operations ---

    def store(
        self,
        text: str,
        category: str = "other",
        scope: str = "global",
        importance: float = 0.5,
        tier: str = "working",
        confidence: float = 0.8,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new memory entry. Returns the memory ID."""
        if category not in MEMORY_CATEGORIES:
            category = "other"
        if tier not in MEMORY_TIERS:
            tier = "working"

        mem_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)

        metadata = {
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
            **(metadata_extra or {}),
        }

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

    def get_by_id(self, mem_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID."""
        results = self._table.search().where(f"id = '{mem_id}'").limit(1).to_list()
        if not results:
            return None
        return self._row_to_dict(results[0])

    def update(
        self,
        mem_id: str,
        text: Optional[str] = None,
        importance: Optional[float] = None,
        category: Optional[str] = None,
        tier: Optional[str] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory. Returns True if found and updated."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False

        metadata = existing["metadata"] if isinstance(existing["metadata"], dict) else json.loads(existing["metadata"])

        if text is not None:
            # Re-encode vector when text changes
            vector = self.encode(text)
            # Supersede pattern: create new version
            metadata["supersedes"] = mem_id
            metadata["state"] = "archived"
            metadata["invalidated_at"] = int(time.time() * 1000)

            # Update old entry with supersede info
            self._table.update(
                where=f"id = '{mem_id}'",
                values={
                    "metadata": json.dumps(metadata),
                },
            )

            # Create new version
            new_id = str(uuid.uuid4())
            metadata["id"] = new_id
            metadata["supersedes"] = mem_id
            metadata["state"] = "confirmed"
            metadata["invalidated_at"] = None
            metadata["access_count"] = 0
            metadata["created_at"] = int(time.time() * 1000)

            category = category or existing["category"]
            importance = importance or existing["importance"]
            tier = tier or metadata.get("tier", "working")

            self._table.add([
                MemorySchema(
                    id=new_id,
                    text=text,
                    vector=vector,
                    category=category,
                    scope=existing["scope"],
                    importance=importance,
                    timestamp=metadata["created_at"],
                    metadata=json.dumps(metadata),
                )
            ])
        else:
            # Metadata-only update (importance, category, tier)
            if importance is not None:
                metadata["importance"] = importance
            if tier is not None:
                metadata["tier"] = tier
            if category is not None:
                metadata["category"] = category
            if metadata_extra:
                metadata.update(metadata_extra)

            update_values = {"metadata": json.dumps(metadata)}
            if importance is not None:
                update_values["importance"] = importance
            if category is not None:
                update_values["category"] = category

            self._table.update(
                where=f"id = '{mem_id}'",
                values=update_values,
            )

        return True

    def forget(self, mem_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        self._table.delete(f"id = '{mem_id}'")
        return True

    def list_memories(
        self,
        limit: int = 20,
        category: Optional[str] = None,
        tier: Optional[str] = None,
        scope: Optional[str] = None,
        offset: int = 0,
        categories: Optional[str] = None,  # Hermes core may pass 'categories' instead of 'category'
    ) -> List[Dict[str, Any]]:
        """List memories with optional filters."""
        query = self._table.search()

        # Support both 'category' and 'categories' param names (Hermes core uses 'categories')
        effective_category = category if category is not None else categories
        if effective_category:
            query = query.where(f"category = '{effective_category}'")
        if scope:
            query = query.where(f"scope = '{scope}'")
        if tier:
            query = query.where(f"metadata LIKE '%\"tier\": \"{tier}\"%'")

        results = query.limit(limit).offset(offset).to_list()
        return [self._row_to_dict(r) for r in results]

    def search(
        self,
        query_text: str,
        limit: int = 10,
        mode: str = "hybrid",  # "vector", "bm25", "hybrid"
        category: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories using vector, BM25, or hybrid mode."""
        if mode == "vector":
            return self._vector_search(query_text, limit, category, scope)
        elif mode == "bm25":
            return self._bm25_search(query_text, limit, category, scope)
        else:
            return self._hybrid_search(query_text, limit, category, scope)

    def _vector_search(
        self, query_text: str, limit: int, category: Optional[str], scope: Optional[str]
    ) -> List[Dict[str, Any]]:
        vector = self.encode(query_text)
        search = self._table.search(vector, vector_column_name="vector")
        if category:
            search = search.where(f"category = '{category}'")
        if scope:
            search = search.where(f"scope = '{scope}'")
        # Fetch extra to account for archived rows we'll filter out
        results = search.limit(limit * 3).to_list()
        return [
            self._row_to_dict(r, distance=r.get("_distance"))
            for r in results
            if ARCHIVED_STATE not in r.get("metadata", "")
        ][:limit]

    def _bm25_search(
        self, query_text: str, limit: int, category: Optional[str], scope: Optional[str]
    ) -> List[Dict[str, Any]]:
        # Scope FTS to the `text` column only — LanceDB FTS indexes all string
        # columns by default, which pollutes results with id/category/metadata.
        search = self._table.search(query_text, query_type="fts", fts_columns=["text"])
        if category:
            search = search.where(f"category = '{category}'")
        if scope:
            search = search.where(f"scope = '{scope}'")
        # Fetch extra to account for archived rows we'll filter out
        results = search.limit(limit * 3).to_list()
        return [
            self._row_to_dict(r)
            for r in results
            if ARCHIVED_STATE not in r.get("metadata", "")
        ][:limit]

    def _hybrid_search(
        self, query_text: str, limit: int, category: Optional[str], scope: Optional[str]
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """Return list of (entry, vector_score, bm25_score) tuples."""
        vector_results = self._vector_search(query_text, limit * 2, category, scope)
        bm25_results = self._bm25_search(query_text, limit * 2, category, scope)

        # Build lookup maps
        vector_map = {r["id"]: r for r in vector_results}
        bm25_map = {r["id"]: r for r in bm25_results}
        all_ids = set(vector_map.keys()) | set(bm25_map.keys())

        results = []
        for mid in all_ids:
            entry = vector_map.get(mid) or bm25_map.get(mid)
            v_score = vector_map.get(mid, {}).get("_distance", 0.0)
            b_score = bm25_map.get(mid, {}).get("_distance", 0.0)
            results.append((entry, v_score, b_score))

        # RRF fusion ranking
        rrf_k = 60  # standard RRF constant
        for i, (entry, vs, bs) in enumerate(results):
            # Lower distance = higher score, so invert
            v_rank = sum(1 for _, v, _ in results if v <= vs) + 1
            b_rank = sum(1 for _, _, b in results if b <= bs) + 1
            entry["_rrf_score"] = 1.0 / (rrf_k + v_rank) + 1.0 / (rrf_k + b_rank)

        results.sort(key=lambda x: x[0].get("_rrf_score", 0), reverse=True)
        return [(r[0], r[1], r[2]) for r in results[:limit]]

    def has_id(self, mem_id: str) -> bool:
        """Check if a memory ID exists in the store (excluding archived rows).
        Used for BM25 ghost protection."""
        results = (
            self._table.search()
            .where(f"id = '{mem_id}'")
            .limit(1)
            .to_list()
        )
        if not results:
            return False
        # Filter out archived/superseded rows
        return ARCHIVED_STATE not in results[0].get("metadata", "")

    def increment_access_count(self, mem_id: str) -> bool:
        """Increment access count and update last_accessed_at."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        metadata = existing["metadata"] if isinstance(existing["metadata"], dict) else json.loads(existing["metadata"])
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        metadata["last_accessed_at"] = int(time.time() * 1000)
        self._table.update(
            where=f"id = '{mem_id}'",
            values={"metadata": json.dumps(metadata)},
        )
        return True

    def stats(self) -> Dict[str, Any]:
        """Return memory store statistics."""
        count = len(self._table)

        # Category breakdown
        all_results = self._table.search().limit(10000).to_list()
        cat_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {}
        for row in all_results:
            cat = row.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            meta = json.loads(row.get("metadata", "{}"))
            tier = meta.get("tier", "working")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_memories": count,
            "db_path": self.db_path,
            "embedding_model": self.embedding_model_name,
            "vector_dimensions": VECTOR_DIM,
            "categories": cat_counts,
            "tiers": tier_counts,
        }

    def purge_archived(self, grace_period_days: int = 30) -> int:
        """
        Permanently delete archived memories older than grace_period_days.

        Returns number of entries purged.

        Archived entries are old versions of memories that were superseded
        by updates. They are excluded from search results but still occupy
        database space. This method cleans them up after a configurable
        grace period.
        """
        cutoff_ms = int(time.time() * 1000) - (grace_period_days * 24 * 60 * 60 * 1000)

        all_results = self._table.search().limit(10000).to_list()
        purged = 0
        for row in all_results:
            meta = json.loads(row.get("metadata", "{}"))
            if meta.get("state") == "archived":
                invalidated = meta.get("invalidated_at", 0)
                if invalidated and invalidated < cutoff_ms:
                    mem_id = row.get("id", "")
                    if mem_id:
                        self._table.delete(f"id = '{mem_id}'")
                        purged += 1

        if purged:
            logger.info(
                "Purged %d archived memories older than %d days",
                purged, grace_period_days,
            )
        return purged

    @staticmethod
    def _row_to_dict(row: Dict[str, Any], distance: float = None) -> Dict[str, Any]:
        """Convert a LanceDB row to a clean dictionary."""
        result = {
            "id": row["id"],
            "text": row["text"],
            "category": row["category"],
            "scope": row["scope"],
            "importance": row["importance"],
            "timestamp": row["timestamp"],
            "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
        }
        if distance is not None:
            result["_distance"] = distance
        if "_rrf_score" in row:
            result["_rrf_score"] = row["_rrf_score"]
        return result
