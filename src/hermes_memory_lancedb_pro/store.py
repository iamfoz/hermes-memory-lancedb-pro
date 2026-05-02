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
    match_session as _match_session,
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

# Tiers whose memories are always available regardless of session_id filtering.
# Core memories represent long-term knowledge (user preferences, profile facts);
# scoping them to a single session would defeat the point of persistent memory.
CROSS_SESSION_TIERS = frozenset({"core"})

# Minimum gap between successive access_count increments for the same memory.
# Without this throttle, every retrieve() bumps every result's access_count,
# which feeds into the decay/frequency score, which raises the memory's
# composite score, which makes it more likely to be retrieved next time —
# a runaway feedback loop that produces "sticky" memories that surface on
# every turn regardless of relevance. 5 minutes is long enough to break the
# loop within a single conversation while still credit-scoring genuine
# repeated use.
ACCESS_COUNT_THROTTLE_SECONDS: int = int(
    os.environ.get("MEMORY_ACCESS_COUNT_THROTTLE_S", "300")
)

# Auto-promote a memory to cross_session=True after it's been recalled in
# this many distinct session_ids. Closes a loop in the session-scoping
# design: memories that are useful across many conversations earn the
# cross-session flag without manual intervention. 0 disables auto-promotion.
CROSS_SESSION_PROMOTION_THRESHOLD: int = int(
    os.environ.get("MEMORY_CROSS_SESSION_PROMOTION_K", "3")
)

# Prompt-injection guard mode — applied at every write site that takes
# free-form text. Stops a malicious tool result or pasted snippet from
# planting `<system>...</system>` / "ignore previous instructions" / etc.
# into the memory store, where it would later be injected into a system
# prompt and influence the agent. Modes:
#   "off"      — no check (legacy behaviour; not recommended)
#   "warn"     — log a warning, allow the write (default)
#   "reject"   — raise ValueError
#   "sanitize" — replace the offending content with a placeholder
INJECTION_GUARD_MODE: str = os.environ.get(
    "MEMORY_INJECTION_GUARD", "warn",
).lower().strip()


def _check_injection_guard(text: str, *, where: str) -> str:
    """Apply the prompt-injection guard. Returns the (possibly sanitised)
    text, or raises ValueError when mode == 'reject'.

    `where` is a short label for log lines (e.g. "store", "supersede").

    The guard reuses the same patterns as `reflection.slices` since
    duplicating the regex list would let the two implementations drift."""
    if not text or INJECTION_GUARD_MODE in ("off", ""):
        return text

    # Lazy import to avoid coupling store.py module load to reflection/.
    try:
        from .reflection.slices import is_unsafe_injectable_reflection_line
    except ImportError:
        return text

    # The reflection helper checks per line. For free-form memory text
    # (potentially multi-line) we check each line and decide based on
    # the worst case across them.
    lines = text.split("\n")
    flagged: list[int] = []
    for i, line in enumerate(lines):
        if line.strip() and is_unsafe_injectable_reflection_line(line):
            flagged.append(i)

    if not flagged:
        return text

    preview = text[:120].replace("\n", "\\n")
    if INJECTION_GUARD_MODE == "reject":
        raise ValueError(
            f"injection guard ({where}): refused to store text matching "
            f"a prompt-injection pattern (preview={preview!r})"
        )
    if INJECTION_GUARD_MODE == "sanitize":
        cleaned = [
            "[content removed: prompt-injection guard]" if i in flagged else line
            for i, line in enumerate(lines)
        ]
        logger.warning(
            "injection guard (%s): sanitised %d line(s) before write (preview=%r)",
            where, len(flagged), preview,
        )
        return "\n".join(cleaned)
    # "warn" mode — log and pass through
    logger.warning(
        "injection guard (%s): %d suspicious line(s) in text being stored (preview=%r)",
        where, len(flagged), preview,
    )
    return text


def _append_relation(
    existing: Any,
    *,
    relation_type: str,
    target_id: str,
) -> list[dict[str, str]]:
    """Append `{type, target_id}` to a relations array, dedup on the
    (type, target_id) pair. Used by the supersede path so that lineage
    chains are walkable from either direction.

    Mirrors the smaller helper in `smart_metadata.append_relation` but
    without the dependency cycle (store.py can't import smart_metadata
    at module load — `_handle_supersede` calls smart_metadata back the
    other way)."""
    out: list[dict[str, str]] = []
    if isinstance(existing, list):
        for r in existing:
            if isinstance(r, dict):
                out.append({"type": str(r.get("type", "")), "target_id": str(r.get("target_id", ""))})
    seen = {(r["type"], r["target_id"]) for r in out}
    if (relation_type, target_id) not in seen:
        out.append({"type": relation_type, "target_id": target_id})
    return out


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

    def warmup(self) -> None:
        """Pre-load the embedding model + run a throwaway encode.

        First-time users pay a 10-30 s cold-start cost (model download +
        torch JIT) on the very first `encode()` call. Hermes-agent should
        call this on session init so that latency lands during boot
        rather than the user's first turn. Idempotent; safe to call
        repeatedly."""
        self._initialise()
        self._load_embedder()
        try:
            self._embedder.encode(  # type: ignore[union-attr]
                "warmup", normalize_embeddings=True,
            )
        except Exception as e:
            logger.warning("MemoryStore.warmup encode failed: %s", e)

    def _ensure_table(self):
        """Open or create the memories table; create the FTS index on `text`.

        Tolerates a TOCTOU race between `list_tables()` and `create_table()`
        — when two stores connect to the same path concurrently (common in
        tests, possible in multi-process setups), the second one can see
        an empty list and then collide on create. Treat "already exists"
        as "open instead"."""
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

        try:
            self._table = self._db.create_table(table_name, schema=MemorySchema)
        except Exception as e:
            if "already exists" in str(e).lower():
                self._table = self._db.open_table(table_name)
                return
            raise

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
            # When True, this memory is exempt from session_id filtering
            # (used for global preferences, profile facts, etc. that should
            # surface across all sessions). Set via metadata_extra at write
            # time; core-tier memories are also cross-session by default.
            "cross_session": False,
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
        text = _check_injection_guard(text, where="store")

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
            text = _check_injection_guard(text, where="store_many")

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

    def store_raw(
        self,
        *,
        text: str,
        vector: Sequence[float],
        category: str,
        scope: str,
        importance: float,
        metadata: str,
        timestamp: int | None = None,
    ) -> str:
        """Low-level write: caller supplies the encoded vector AND the
        full metadata JSON string. Bypasses `_build_metadata` so callers
        with their own metadata schemas (reflection, audit logs, etc.)
        aren't forced through the default-field machinery.

        Returns the new memory id. Use this when you need:
          - a pre-computed vector (saves a re-encode)
          - a metadata JSON that diverges from the standard schema (e.g.
            reflection items, with their own decay / quality / item_kind
            fields)

        For ordinary writes, prefer `store()` / `store_many()`."""
        if not text or not str(text).strip():
            raise ValueError("MemoryStore.store_raw: `text` must be non-empty")
        if not vector:
            raise ValueError("MemoryStore.store_raw: `vector` must be non-empty")
        text = _check_injection_guard(text, where="store_raw")

        mem_id = str(uuid.uuid4())
        ts = int(time.time() * 1000) if timestamp is None else int(timestamp)
        importance = max(0.0, min(1.0, float(importance)))

        self._table.add([
            MemorySchema(
                id=mem_id,
                text=text,
                vector=list(vector),
                category=category,
                scope=scope,
                importance=importance,
                timestamp=ts,
                metadata=metadata,
            )
        ])
        return mem_id

    def search_by_vector(
        self,
        vector: Sequence[float],
        limit: int = 10,
        *,
        category: str | None = None,
        scope: str | None = None,
        keep_vector: bool = False,
    ) -> list[dict[str, Any]]:
        """Search by a pre-computed vector — avoids re-encoding when the
        caller already has the embedding (e.g. reflection dedup, where
        the vector was just computed for the write)."""
        if not vector:
            return []
        search = self._table.search(list(vector), vector_column_name="vector")
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        results = search.limit(max(limit * SEARCH_OVERFETCH_MULTIPLIER, limit)).to_list()
        return [
            self._row_to_dict(r, distance=r.get("_distance"), keep_vector=keep_vector)
            for r in results
            if not _is_archived(r.get("metadata", ""))
        ][:limit]

    def update(
        self,
        mem_id: str,
        text: str | None = None,
        importance: float | None = None,
        category: str | None = None,
        tier: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Update an existing memory.

        If `text` is provided this uses the supersede pattern: the existing
        row is marked archived and a new row with a fresh UUID replaces it.
        Otherwise the existing row's columns and metadata are updated in-place.

        Returns:
            - the **new** memory ID after a supersede (text was changed)
            - the **same** `mem_id` after a metadata-only update
            - `None` if the memory wasn't found

        The non-None return is always truthy, so existing callers that
        used the result as a boolean (`if store.update(...)`) keep working.
        Callers chaining the result (e.g. `new_id = store.update(id, text=...)`)
        now get the actual new ID."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return None

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
        # Metadata-only update keeps the same id
        return mem_id

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
    ) -> str:
        text = _check_injection_guard(text, where="supersede")
        old_id = existing["id"]
        new_id = str(uuid.uuid4())

        # 1. Archive the old row (still readable for audit, hidden from queries).
        # Also append a `superseded_by` relation to the relations chain so
        # downstream tooling can walk the lineage.
        archived_meta = dict(old_metadata)
        archived_meta["state"] = ARCHIVED_STATE
        archived_meta["superseded_by"] = new_id
        archived_meta["invalidated_at"] = now_ms
        archived_meta["relations"] = _append_relation(
            archived_meta.get("relations"),
            relation_type="superseded_by",
            target_id=new_id,
        )

        # 2. Fresh metadata for the new row — start from the old metadata so
        # custom fields (fact_key, source, relations…) carry forward, but
        # reset lifecycle counters and link back to the predecessor via both
        # the `supersedes` field AND a relations entry.
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
        new_meta["relations"] = _append_relation(
            new_meta.get("relations"),
            relation_type="supersedes",
            target_id=old_id,
        )

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
        return new_id

    def forget(self, mem_id: str) -> bool:
        """Hard-delete a memory by ID. Returns True if the memory existed."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        self._table.delete(f"id = '{_escape_sql(mem_id)}'")
        return True

    def increment_access_count(
        self,
        mem_id: str,
        *,
        force: bool = False,
        throttle_seconds: int | None = None,
    ) -> bool:
        """Increment access_count and update last_accessed_at.

        Throttled by default: a memory's access_count won't increment more
        than once per `ACCESS_COUNT_THROTTLE_SECONDS` (default 5 min). This
        breaks the recall feedback loop where every retrieve() bumped the
        count, which raised the decay-frequency score, which made the same
        memory more retrievable next time, ad infinitum.

        Pass `force=True` to bypass the throttle (e.g. for a definitive
        "memory was actually used" signal from the agent). Pass
        `throttle_seconds=N` to override the default cooldown."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False

        metadata = _parse_metadata(existing["metadata"])
        now_ms = int(time.time() * 1000)
        cooldown_ms = (
            ACCESS_COUNT_THROTTLE_SECONDS if throttle_seconds is None else max(0, int(throttle_seconds))
        ) * 1000

        # Skip the throttle on the very first access (count==0). Without this,
        # a freshly-stored memory whose `last_accessed_at == created_at` would
        # have its first recall blocked, and access_count would never tick up.
        current_count = int(metadata.get("access_count", 0) or 0)
        if not force and cooldown_ms > 0 and current_count > 0:
            last = int(metadata.get("last_accessed_at", 0) or 0)
            if last and (now_ms - last) < cooldown_ms:
                return False

        metadata["access_count"] = current_count + 1
        metadata["last_accessed_at"] = now_ms
        self._table.update(
            where=f"id = '{_escape_sql(mem_id)}'",
            values={"metadata": json.dumps(metadata)},
        )
        return True

    def mark_recall_used(
        self,
        mem_ids: Sequence[str],
        *,
        session_id: str | None = None,
    ) -> int:
        """Definitively mark memories as having been used (injected into
        a prompt and meaningfully referenced). Bypasses the access-count
        throttle. Returns the number of memories actually updated.

        Batched: one IN-clause read + one update per id (the per-id
        update is necessary because each row's metadata diverges).

        When `session_id` is provided, also tracks the recall against
        the row's `cross_session_recalls` set. Once a memory has been
        recalled in `CROSS_SESSION_PROMOTION_THRESHOLD` distinct sessions
        within the recent window, it's auto-promoted to
        `cross_session=True` — closing a feedback loop in the design.
        Memories that are useful across many conversations earn the
        cross-session flag without manual intervention."""
        if not mem_ids:
            return 0

        unique_ids = list(dict.fromkeys(mem_ids))
        in_clause = ",".join(f"'{_escape_sql(mid)}'" for mid in unique_ids)
        try:
            rows = (
                self._table.search()
                .where(f"id IN ({in_clause})")
                .limit(max(len(unique_ids), 1))
                .to_list()
            )
        except Exception as e:
            logger.warning("mark_recall_used batch fetch failed: %s", e)
            return 0

        now_ms = int(time.time() * 1000)
        updated = 0
        for row in rows:
            mem_id = row.get("id")
            if not mem_id:
                continue
            metadata = _parse_metadata(row.get("metadata"))
            metadata["access_count"] = int(metadata.get("access_count", 0) or 0) + 1
            metadata["last_accessed_at"] = now_ms

            # Cross-session promotion ledger
            if session_id and not metadata.get("cross_session"):
                ledger = list(metadata.get("cross_session_recalls") or [])
                if session_id not in ledger:
                    ledger.append(session_id)
                    # Cap the ledger to the most recent K entries; we only
                    # need to know "K distinct sessions" not the full history.
                    if len(ledger) > CROSS_SESSION_PROMOTION_THRESHOLD * 2:
                        ledger = ledger[-CROSS_SESSION_PROMOTION_THRESHOLD * 2:]
                    metadata["cross_session_recalls"] = ledger
                    if len(ledger) >= CROSS_SESSION_PROMOTION_THRESHOLD:
                        metadata["cross_session"] = True
                        metadata["cross_session_promoted_at"] = now_ms
                        logger.debug(
                            "auto-promoted %s to cross_session after %d sessions",
                            mem_id, len(ledger),
                        )

            try:
                self._table.update(
                    where=f"id = '{_escape_sql(mem_id)}'",
                    values={"metadata": json.dumps(metadata)},
                )
                updated += 1
            except Exception as e:
                logger.warning("mark_recall_used update %s failed: %s", mem_id, e)
        return updated

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
        session_id: str | None = None,
        # Hermes core may pass `categories` instead of `category`
        categories: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memories with optional filters.

        Filters are AND-combined into a single WHERE clause — chaining
        `.where()` on a LanceQueryBuilder replaces the previous predicate
        in many LanceDB versions, so we must build the clause ourselves.

        When `session_id` is set, only memories from that session (or
        explicitly cross-session memories — see `_match_session`) are
        returned. Pass `session_id=None` to disable session scoping (the
        default, for full backwards compat)."""
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

        # Over-fetch when we'll filter archived/session rows post-query
        post_filter = (not include_archived) or (session_id is not None)
        fetch_limit = (
            limit + offset + max(limit, 16) * 2 if post_filter else limit
        )

        query = self._table.search()
        where_clause = _and_clauses(*clauses)
        if where_clause:
            query = query.where(where_clause)
        results = query.limit(fetch_limit).to_list()

        rows = [self._row_to_dict(r) for r in results]
        if not include_archived:
            rows = [r for r in rows if r["metadata"].get("state") != ARCHIVED_STATE]
        if session_id is not None:
            rows = [r for r in rows if _match_session(r["metadata"], session_id)]
        return rows[offset : offset + limit]

    def search(
        self,
        query_text: str,
        limit: int = 10,
        mode: str = "hybrid",  # "vector" | "bm25" | "hybrid"
        category: str | None = None,
        scope: str | None = None,
        session_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories using vector, BM25, or hybrid mode.

        All three modes return `List[Dict[str, Any]]` — the previous version
        returned tuples for hybrid mode, breaking polymorphic callers.

        `session_id` (optional): when set, results are restricted to memories
        whose `metadata.source_session` matches, plus memories explicitly
        marked `cross_session` or living in a cross-session tier (core).
        Without this filter, retrieving from a fresh conversation surfaces
        memories from earlier sessions and produces "sticky" recall — the
        agent gets confused about what the user is currently asking.

        `min_score` (optional): drops results whose relevance is below the
        threshold. Semantics depend on the mode:
          - vector: keep iff `(1 - _distance) >= min_score`  (cosine)
          - bm25:   keep iff `_score >= min_score`           (FTS rank)
          - hybrid: keep iff `_rrf_score >= min_score`        (fused rank)
        """
        if not query_text or not query_text.strip():
            return []
        if mode == "vector":
            return self._vector_search(
                query_text, limit, category, scope,
                session_id=session_id, min_score=min_score,
            )
        if mode == "bm25":
            return self._bm25_search(
                query_text, limit, category, scope,
                session_id=session_id, min_score=min_score,
            )
        return self._hybrid_search(
            query_text, limit, category, scope,
            session_id=session_id, min_score=min_score,
        )

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

    @staticmethod
    def _post_filter(
        rows: list[dict[str, Any]],
        *,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        """Apply non-archived + session_id filters in Python.

        Archived filtering is always applied (callers that want archived
        rows go through different paths). Session filtering is applied
        only when `session_id` is provided."""
        out = []
        for row in rows:
            meta = row.get("metadata", "")
            if _is_archived(meta):
                continue
            if session_id is not None and not _match_session(meta, session_id):
                continue
            out.append(row)
        return out

    def _vector_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
        keep_vector: bool = False,
        *,
        session_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        vector = self.encode(query_text)
        search = self._table.search(vector, vector_column_name="vector")
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        # Over-fetch more aggressively when post-filtering by session, since
        # session-matching rows may be sparse in the top-N from LanceDB.
        overfetch = SEARCH_OVERFETCH_MULTIPLIER * (3 if session_id is not None else 1)
        results = search.limit(max(limit * overfetch, limit)).to_list()
        results = self._post_filter(results, session_id=session_id)

        rows = [
            self._row_to_dict(r, distance=r.get("_distance"), keep_vector=keep_vector)
            for r in results
        ]
        if min_score is not None:
            rows = [
                r for r in rows
                if (1.0 - float(r.get("_distance") or 0.0)) >= min_score
            ]
        return rows[:limit]

    def _bm25_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
        keep_vector: bool = False,
        *,
        session_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        # Scope FTS to the `text` column only — LanceDB's FTS otherwise indexes
        # all string columns, polluting results with id/category/metadata.
        search = self._table.search(
            query_text, query_type="fts", fts_columns=["text"]
        )
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        overfetch = SEARCH_OVERFETCH_MULTIPLIER * (3 if session_id is not None else 1)
        results = search.limit(max(limit * overfetch, limit)).to_list()
        results = self._post_filter(results, session_id=session_id)

        rows = [
            self._row_to_dict(r, score=r.get("_score"), keep_vector=keep_vector)
            for r in results
        ]
        if min_score is not None:
            rows = [r for r in rows if float(r.get("_score") or 0.0) >= min_score]
        return rows[:limit]

    def _hybrid_search(
        self,
        query_text: str,
        limit: int,
        category: str | None,
        scope: str | None,
        *,
        session_id: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of vector + BM25 results.

        Always returns plain dicts (with `_rrf_score` set) for parity with
        the other search modes."""
        candidate = max(limit * 2, 10)
        vector_results = self._vector_search(
            query_text, candidate, category, scope,
            session_id=session_id,
        )
        bm25_results = self._bm25_search(
            query_text, candidate, category, scope,
            session_id=session_id,
        )

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
        if min_score is not None:
            ranked = [
                r for r in ranked
                if float(r.get("_rrf_score") or 0.0) >= min_score
            ]
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

        Every result gets a normalised `score` field (higher = better,
        clamped to [0, 1] for vector / hybrid modes; raw for BM25). The
        mode-specific raw signals (`_distance` for vector, `_score` for
        BM25, `_rrf_score` for hybrid) are also preserved so advanced
        callers can introspect them.

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

        # Mode-specific raw signals
        if distance is not None:
            result["_distance"] = distance
        if score is not None:
            result["_score"] = score
        if "_rrf_score" in row:
            result["_rrf_score"] = row["_rrf_score"]

        # Normalised user-facing `score` field — always higher-is-better,
        # always present. Set whichever raw signal is most informative for
        # this row. Hybrid takes precedence (most informative); then
        # vector (cosine 1-distance); then BM25 raw.
        if "_rrf_score" in result:
            # RRF scores are typically 0..0.033 (1/(60+1)+1/(60+1)). Scale
            # by 30 so a top RRF hit lands near 1.0; clamp to [0, 1].
            result["score"] = max(0.0, min(1.0, float(result["_rrf_score"]) * 30.0))
        elif distance is not None:
            # Cosine distance: 0 = identical, ~1 = orthogonal. Normalised
            # vectors give distances in [0, 2]; for relevant matches it's
            # almost always 0..1.
            result["score"] = max(0.0, min(1.0, 1.0 - float(distance)))
        elif score is not None:
            # BM25 raw — no clean normalisation. Scale by 10 (typical
            # max for short FTS queries) and clamp; advanced callers
            # should use `_score` directly.
            result["score"] = max(0.0, min(1.0, float(score) / 10.0))
        else:
            result["score"] = 0.0
        return result
