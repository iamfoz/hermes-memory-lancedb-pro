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

import contextlib
import functools
import json
import logging
import os
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:  # POSIX only; without it the in-use lock is a no-op.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

import lancedb
from lancedb.pydantic import LanceModel, Vector

# `sentence_transformers` is only needed when we actually have to encode text.
# Importing it eagerly pulls in torch (hundreds of MB) and prevents tests with
# stub embedders from running on lightweight hosts.
if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

from ._env import env_int
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
    is_expired as _is_expired,
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

def _session_anchor_match(metadata: dict[str, Any], session_id: str, strict: bool) -> bool:
    """Session filter for `first_for_session` / `recent_for_session`.

    `strict` keeps only rows written IN this session. The default also
    admits cross-session and core-tier rows (`match_session`), which made
    the "first memories of this session" the two oldest global rows of the
    whole store, in every session, forever."""
    if strict:
        return metadata.get("source_session") == session_id
    return _match_session(metadata, session_id)


IN_USE_LOCK_FILE = ".in-use.lock"


class StoreInUseError(RuntimeError):
    """The database is held by a running agent and cannot be converted."""


def _serialised(method):
    """Run a MemoryStore method under the store's write lock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)

    return wrapper


def resolve_db_path(explicit: str | None = None, hermes_home: str | None = None) -> str:
    """The memory database directory, resolved the same way everywhere.

    Highest priority first:
      1. *explicit* — a ``--path`` argument or a caller-supplied path.
      2. ``MEMORY_DB_DIR`` — the documented override.
      3. ``<hermes home>/memory-lancedb`` — *hermes_home* as passed by the
         host, else ``$HERMES_HOME`` (set per ``hermes -p <profile>``), so
         each profile keeps its own database.
      4. ``~/.hermes/memory-lancedb``.

    The provider and every CLI command call this, so the CLI always works on
    the database the running agent uses.
    """
    if explicit and str(explicit).strip():
        return os.path.expanduser(str(explicit))
    env_dir = os.environ.get("MEMORY_DB_DIR", "").strip()
    if env_dir:
        return os.path.expanduser(env_dir)
    home = (str(hermes_home).strip() if hermes_home else "") or os.environ.get(
        "HERMES_HOME", ""
    ).strip()
    if home:
        return os.path.join(os.path.expanduser(home), "memory-lancedb")
    return os.path.expanduser("~/.hermes/memory-lancedb")


DEFAULT_DB_PATH: str = resolve_db_path()
DEFAULT_EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM: int = 768

# Optional device override for the embedding model: "cpu", "mps", "cuda",
# "cuda:1", ... Unset leaves the choice to sentence-transformers (MPS on Apple
# Silicon, CUDA when present, else CPU). If the requested device cannot be
# used, loading logs a warning and falls back to automatic selection rather
# than failing every turn. Read at load time, so it applies per process.
EMBEDDING_DEVICE: str | None = os.environ.get("MEMORY_EMBEDDING_DEVICE", "").strip() or None

# How stored text and queries are turned into embedder input, recorded per
# database in EMBEDDING_SCHEME_FILE. nomic-embed-text is trained with task
# prefixes ("search_document: " on stored text, "search_query: " on queries)
# and retrieves noticeably worse without them, but every database written
# before this was recorded holds unprefixed vectors. Mixing the two in one
# table would compare vectors from different input distributions, so the
# scheme is a property of the database: one with rows and no marker is
# "plain" (legacy) until `hermes lancedb_pro reembed` converts it; a new database
# starts on "task_prefix".
EMBEDDING_SCHEME_FILE: str = "embedding_scheme.json"
SCHEME_PLAIN: str = "plain"
SCHEME_TASK_PREFIX: str = "task_prefix"
EMBEDDING_SCHEMES = (SCHEME_PLAIN, SCHEME_TASK_PREFIX)
EMBED_KINDS = ("document", "query")
_NOMIC_TASK_PREFIXES: dict[str, str] = {
    "document": "search_document: ",
    "query": "search_query: ",
}
# Rows per embedder call in `reembed`, and how many catch-up passes it makes
# for rows written by another process while it ran.
REEMBED_BATCH_SIZE: int = 64
_REEMBED_MAX_PASSES: int = 5


# Models whose bundled custom code may be loaded automatically when the
# installed transformers requires it.
_TRUSTED_REMOTE_CODE_MODELS = frozenset({
    "nomic-ai/nomic-embed-text-v1",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v2-moe",
})


def model_uses_task_prefixes(model_name: str | None) -> bool:
    """True for models trained with nomic's search_document/search_query
    prefixes (nomic-embed-text v1, v1.5, v2). Other models, including
    nomic-embed-code, which uses different instructions, get raw text."""
    return "nomic-embed-text" in str(model_name or "").lower()


def _needs_trust_remote_code(exc: Exception) -> bool:
    """True when a model load failed because its repository ships custom
    modelling code that transformers will not run without consent."""
    return "trust_remote_code" in str(exc)


# One embedder operation at a time, process-wide. torch's MPS backend is not
# thread-safe: two threads encoding concurrently abort in Metal ("A command
# encoder is already encoding to this command buffer", SIGABRT/SIGSEGV) — e.g.
# the provider's warmup thread and the first turn's prefetch thread. The MPS
# stream is shared by every model in the process, and one gateway can hold a
# MemoryStore per Hermes profile, so this is module-level rather than per
# store. Model loading takes it too: moving weights onto the device uses the
# same stream. Lock order everywhere: a store's _embedder_lock, then this one.
_ENCODE_LOCK = threading.Lock()

# Reasonable upper bound for full-table scans inside the API. Configurable
# via env var so users with very large stores can lift it.
MAX_SCAN_ROWS: int = env_int("MEMORY_MAX_SCAN_ROWS", 100000)

# Heuristic: when filtering archived rows post-fetch, fetch this many
# multiples of `limit` to leave headroom.
SEARCH_OVERFETCH_MULTIPLIER: int = 3

# Index training threshold — LanceDB IVF_PQ needs ~256 rows minimum
VECTOR_INDEX_MIN_ROWS: int = 256

# Every vector query and vector index uses cosine distance (0 = same
# direction, 1 = orthogonal, 2 = opposite), which is what `score = 1 -
# _distance`, `min_score` and every dedup threshold assume. LanceDB defaults
# to squared L2, which for unit vectors is `2 - 2*cos`: twice the cosine
# distance, so a cos-0.5 match used to score 0.0.
VECTOR_DISTANCE_TYPE: str = "cosine"

# With an IVF_PQ index, `_distance` is a compressed (PQ) approximation that
# can be off by 0.1 or more, which is enough to flip a dedup threshold.
# Refining re-scores `limit * factor` candidates on the full vectors, so the
# returned distances are exact. Without an index it is a no-op.
VECTOR_REFINE_FACTOR: int = 5

# Auto-compaction threshold. Every `_table.add()` writes a new on-disk data
# fragment, and every read has to open every fragment of the table's current
# version. A store that has absorbed thousands of single-row writes therefore
# opens thousands of files per query, which exhausts the process
# file-descriptor limit (`ulimit -n`) and degrades catastrophically — inserts
# and searches slow to a crawl and eventually time out.
#
# After this many fragment-creating writes the store runs `optimize()`, which
# compacts the small fragments into a few large ones (and folds newly-written
# rows into the FTS / vector indexes). Set to 0 to disable auto-compaction.
AUTO_OPTIMIZE_EVERY: int = env_int("MEMORY_AUTO_OPTIMIZE_EVERY", 256)

# Hard upper bound on a single memory's `text`, in characters. A memory entry
# is meant to be a distilled fact, not a document: beyond this size the
# embedding model cannot represent the text in one vector anyway, and very
# large inputs can exhaust GPU / MPS memory. `store()` and `store_many()`
# reject oversized text rather than letting it reach the embedder. Raise this
# if you genuinely need larger entries — though the smart extractor is the
# better place to summarise long content.
MAX_TEXT_CHARS: int = env_int("MEMORY_MAX_TEXT_CHARS", 8000)

# Soft open-file (file-descriptor) limit the store ensures for its process.
# LanceDB opens a descriptor per on-disk fragment, so under sustained write
# load a process started with a low soft limit (macOS shells default to 256)
# can exhaust descriptors. `MemoryStore` raises the soft limit toward the hard
# ceiling on construction — no privileges needed. Set 0 to disable.
MEMORY_FD_LIMIT: int = env_int("MEMORY_FD_LIMIT", 4096)

# "active_task" backs the durable-task pin: `task pin`, the recall-guardrail
# active-task pinning, `_format_recall`, `_refresh_active_task_memories` and the
# session auto-anchor all read/write this category. It MUST be a recognised
# category — otherwise `store()` silently coerces it to "other" and every
# `list_memories(category="active_task")` lookup returns nothing, which
# silently disables task-state recovery across context compaction.
MEMORY_CATEGORIES = [
    "preference", "fact", "decision", "entity", "other", "reflection", "active_task",
]
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
ACCESS_COUNT_THROTTLE_SECONDS: int = env_int("MEMORY_ACCESS_COUNT_THROTTLE_S", 300)

# Auto-promote a memory to cross_session=True after it's been recalled in
# this many distinct session_ids. Closes a loop in the session-scoping
# design: memories that are useful across many conversations earn the
# cross-session flag without manual intervention. 0 disables auto-promotion.
CROSS_SESSION_PROMOTION_THRESHOLD: int = env_int("MEMORY_CROSS_SESSION_PROMOTION_K", 3)

# Hide memories whose metadata `valid_until` (epoch ms, inferred by the smart
# extractor from phrases like "tomorrow" / "in 3 days") has passed from every
# recall path: search (all modes), the retriever, session anchors, and the
# extractor's dedup neighbour search. Expired rows are not deleted — admin
# paths (get_by_id, list_memories, export, doctor) still see them. Set
# MEMORY_ENFORCE_VALID_UNTIL=off to restore the old record-only behaviour.
ENFORCE_VALID_UNTIL: bool = os.environ.get(
    "MEMORY_ENFORCE_VALID_UNTIL", "on"
).strip().lower() not in ("off", "0", "false", "no")


def _rrf_to_score(rrf: Any) -> float:
    """Map an RRF score onto [0, 1]. RRF scores are 0..~0.033 (1/61 + 1/61
    for a row ranked first by both legs); scaling by 30 puts that near 1.0."""
    return max(0.0, min(1.0, float(rrf or 0.0) * 30.0))


def _hidden_from_recall(metadata: Any, now_ms: int) -> bool:
    """True if a row must never be surfaced by recall: archived, or past its
    ``valid_until`` while enforcement is on. Reads the module-level switch at
    call time so tests (and embedders) can toggle it."""
    if _is_archived(metadata):
        return True
    return ENFORCE_VALID_UNTIL and _is_expired(metadata, now_ms)

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


def _raise_if_fd_exhaustion(exc: Exception) -> None:
    """Re-raise *exc* as a clear OSError when it signals file-descriptor
    exhaustion (EMFILE / errno 24 / "too many open files").

    LanceDB + PyTorch each open many file handles; on macOS the default
    ulimit -n is only 256, which can be exhausted under sustained load
    (~2 000+ store operations with concurrent reads).

    Callers: wrap LanceDB I/O in try/except and call this before re-raising
    so users get an actionable message instead of a cryptic LanceError."""
    msg = str(exc).lower()
    if "too many open files" in msg or "os error 24" in msg or "emfile" in msg:
        raise OSError(
            "MemoryStore: OS file-descriptor limit exhausted — memory operations "
            "will fail until the limit is raised.  Fix: run  `ulimit -n 4096`  "
            "in your shell before starting hermes-agent, or add it to "
            "~/.zshrc / ~/.bashrc.  "
            f"Original error: {exc}"
        ) from exc


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
        from .reflection.slices import (
            _normalize_reflection_slice_line,
            is_unsafe_injectable_reflection_line,
        )
    except ImportError:
        return text

    # The reflection helper checks per line. For free-form memory text
    # (potentially multi-line) we check each line and decide based on
    # the worst case across them.
    lines = text.split("\n")
    flagged: list[int] = []
    for i, line in enumerate(lines):
        # A bare reflection section label ("Invariants:", "Derived:")
        # normalises to an empty line, which the reflection helper counts
        # as unsafe — so every reflection write was flagged as suspicious.
        # Skip only those; any line with content left is still checked
        # against the injection patterns.
        if (
            line.strip()
            and _normalize_reflection_slice_line(line)
            and is_unsafe_injectable_reflection_line(line)
        ):
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
# File-descriptor limit
# ---------------------------------------------------------------------------


def _raise_fd_limit() -> None:
    """Best-effort raise of this process's soft open-file limit.

    LanceDB opens a descriptor per on-disk fragment, so under sustained write
    load a process started with a low soft limit (macOS shells default to 256)
    can exhaust descriptors. Raising the soft limit toward the hard ceiling
    needs no privileges. Controlled by ``MEMORY_FD_LIMIT`` (default 4096; set
    ``0`` to disable). Best-effort: any failure is logged at debug and ignored.
    """
    if MEMORY_FD_LIMIT <= 0:
        return
    try:
        import resource
    except ImportError:  # non-Unix platform — no rlimits
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    target = MEMORY_FD_LIMIT
    if hard != resource.RLIM_INFINITY and hard > 0:
        target = min(target, hard)
    if target <= soft:
        return  # already sufficient
    # Some platforms (notably macOS) reject an unlimited hard value here, so
    # retry with the hard limit clamped to the target as a fallback.
    for new_hard in (hard, target):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, new_hard))
            logger.info("Raised open-file soft limit %d -> %d", soft, target)
            return
        except (ValueError, OSError):
            continue
    logger.debug("Could not raise the open-file limit above %d", soft)


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
        _raise_fd_limit()
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.embedding_model_name = embedding_model
        self._embedder: SentenceTransformer | None = None
        self._embedder_lock = threading.Lock()
        # Embedding scheme of this database (see EMBEDDING_SCHEME_FILE), set
        # by _initialise. `_scheme_stamp` identifies the marker file version
        # it was read from, so a `reembed` run by another process (the CLI
        # while the agent runs) is picked up on the next encode.
        self._scheme: str | None = None
        self._scheme_stamp: tuple[int, int, int] | None = None
        # Serialises read-modify-write sequences on a row's metadata (update,
        # supersede, access counting, forget). Without it concurrent callers
        # — the provider's prefetch, sync and extraction threads share one
        # store — lost counter updates, un-archived superseded rows, and
        # produced several live versions of one memory. Re-entrant because
        # these methods call each other. Always taken BEFORE the embedder
        # locks, never after.
        self._write_lock = threading.RLock()
        # Serialises every embedder call (encode, warmup, model load). torch's
        # MPS backend is not thread-safe, so concurrent encodes abort in Metal.
        # Shared by all stores in the process — see _ENCODE_LOCK.
        self._encode_lock = _ENCODE_LOCK
        self._db = None
        self._table = None
        self._initialised = False
        # Fragment-compaction bookkeeping — see AUTO_OPTIMIZE_EVERY. The
        # counter tracks fragment-creating writes; `_optimize_counter_lock`
        # guards it, while `_optimize_run_lock` single-flights the compaction
        # so only one runs at a time even under heavy concurrent writes.
        self._writes_since_optimize = 0
        self._optimize_counter_lock = threading.Lock()
        self._optimize_run_lock = threading.Lock()

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
        self._init_embedding_scheme()
        self._initialised = True

    def _load_embedder(self):
        if self._embedder is not None:
            return
        with self._embedder_lock:
            if self._embedder is None:  # double-checked under lock
                from sentence_transformers import SentenceTransformer  # heavy import
                logger.debug("Loading embedding model: %s", self.embedding_model_name)
                # Loading moves the weights onto the device; on MPS that uses
                # the process-wide stream, so serialise it with encodes.
                with self._encode_lock:
                    self._embedder = self._construct_embedder(SentenceTransformer)

    def _construct_embedder(self, sentence_transformer_cls):
        """Build the model on ``EMBEDDING_DEVICE`` when set. A device that
        cannot be used (typo, CUDA absent, MPS unavailable) falls back to
        automatic selection with a warning: the override is a safety valve,
        and failing every turn would be worse than ignoring it."""
        device = EMBEDDING_DEVICE
        if not device:
            return self._new_model(sentence_transformer_cls)
        try:
            return self._new_model(sentence_transformer_cls, device=device)
        except Exception as e:
            logger.warning(
                "MEMORY_EMBEDDING_DEVICE=%r could not be used (%s); falling back "
                "to automatic device selection", device, e,
            )
            return self._new_model(sentence_transformer_cls)

    def _new_model(self, sentence_transformer_cls, **kwargs):
        """Construct the model, retrying with ``trust_remote_code=True`` for
        nomic-embed-text when the installed transformers needs it.

        nomic-embed-text v1/v1.5 ship custom NomicBert modelling code. Some
        transformers / sentence-transformers releases load them natively,
        others refuse without ``trust_remote_code=True``. Trying without it
        first keeps a load that works today unchanged; the retry is limited
        to nomic models so a user-chosen model never runs remote code
        implicitly."""
        try:
            return sentence_transformer_cls(self.embedding_model_name, **kwargs)
        except Exception as e:
            # Exact official repositories only: a third-party or local model
            # whose name merely contains "nomic-embed-text" must never have
            # its custom code run without the user asking for it.
            if not (str(self.embedding_model_name).lower() in _TRUSTED_REMOTE_CODE_MODELS
                    and _needs_trust_remote_code(e)):
                raise
            logger.info(
                "Loading %s needs its custom model code; retrying with "
                "trust_remote_code=True", self.embedding_model_name,
            )
            return sentence_transformer_cls(
                self.embedding_model_name, trust_remote_code=True, **kwargs,
            )

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
            # The warmup thread races the first turn's prefetch encode; on
            # MPS two concurrent encodes abort the process. Serialise.
            with self._encode_lock:
                self._embedder.encode(  # type: ignore[union-attr]
                    "warmup", normalize_embeddings=True,
                )
        except Exception as e:
            logger.warning("MemoryStore.warmup encode failed: %s", e)
        # Background housekeeping that must stay off the start-up path.
        try:
            self._initialise()
            self._migrate_vector_index_metric()
        except Exception as e:
            logger.warning("MemoryStore.warmup index migration failed: %s", e)

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
            # The vector-index metric migration is NOT done here: retraining
            # an index takes seconds on a large store and this runs inside
            # the provider's synchronous start-up, and in read-only CLI
            # commands. `warmup()` does it, in the background.
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

    def _vector_index_metric(self) -> str | None:
        """Distance metric of the index on the `vector` column, lower-cased
        ("l2", "cosine", "dot"); "" when an index exists but its metric
        cannot be read; None when there is no vector index."""
        try:
            indices = self._table.list_indices()
        except Exception:
            return None
        for idx in indices:
            if "vector" not in list(getattr(idx, "columns", None) or []):
                continue
            metric = None
            with contextlib.suppress(Exception):
                metric = self._table.index_stats(idx.name).distance_type
            if not metric:
                details = getattr(idx, "index_details", None) or {}
                metric = details.get("metric_type")
            return str(metric or "").lower()
        return None

    def _build_vector_index(self) -> None:
        # The metric-first form is the call every supported lancedb accepts
        # (>= 0.25); newer releases flag it as deprecated, which is noise here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._table.create_index(
                metric=VECTOR_DISTANCE_TYPE, vector_column_name="vector", replace=True,
            )

    def _migrate_vector_index_metric(self) -> None:
        """Rebuild, once, a vector index built with another metric.

        Databases indexed before cosine was set explicitly carry an L2
        IVF_PQ index. A cosine query against it still returns correct
        distances, but LanceDB falls back to a brute-force scan and logs a
        warning on every query. The index holds no data of its own, so
        replacing it loses nothing. If the rebuild fails (for example the
        table has shrunk below what IVF_PQ can train on), the stale index is
        dropped so queries stay exact and quiet; `maybe_create_vector_index`
        builds a cosine one later."""
        metric = self._vector_index_metric()
        if not metric or metric == VECTOR_DISTANCE_TYPE:
            return
        logger.info(
            "Rebuilding the %s vector index with the %s metric",
            metric, VECTOR_DISTANCE_TYPE,
        )
        try:
            self._build_vector_index()
            return
        except Exception as e:
            logger.warning("Rebuilding the %s vector index failed: %s", metric, e)
        try:
            # Another process may have rebuilt it meanwhile: look again
            # before dropping, so a fresh cosine index is never removed.
            self._checkout_latest()
            if self._vector_index_metric() in (VECTOR_DISTANCE_TYPE, None, ""):
                return
            for idx in self._table.list_indices():
                if "vector" in list(getattr(idx, "columns", None) or []):
                    self._table.drop_index(idx.name)
        except Exception as e:
            logger.warning("Dropping the %s vector index failed: %s", metric, e)

    def maybe_create_vector_index(self) -> bool:
        """Create the cosine IVF_PQ vector index when there are enough rows
        to train, or rebuild one that uses another metric. Returns True if
        an index was just built, False otherwise."""
        metric = self._vector_index_metric()
        if metric in (VECTOR_DISTANCE_TYPE, ""):
            # Already right, or unreadable: never rebuild on every call
            # (create_index replaces by default).
            return False
        try:
            row_count = len(self._table)
        except Exception:
            return False
        if row_count < VECTOR_INDEX_MIN_ROWS:
            return False
        try:
            self._build_vector_index()
            return True
        except Exception as e:
            logger.warning("Vector index creation failed: %s", e)
            return False

    def _vector_query(self, vector: Sequence[float]):
        """Start a vector query whose `_distance` is exact cosine distance."""
        return (
            self._table.search(list(vector), vector_column_name="vector")
            .distance_type(VECTOR_DISTANCE_TYPE)
            .refine_factor(VECTOR_REFINE_FACTOR)
        )

    # ----- maintenance -----

    def _table_add(self, rows: list[MemorySchema]) -> None:
        """Append `rows` to the table, then run cooldown-gated fragment
        compaction.

        Every append funnels through here (updates and deletes through
        `_table_update` / `_table_delete`) so a single counter sees all
        fragment-creating writes. Keeping `self._table.add` out of the
        individual CRUD methods means the compaction policy — and the
        file-descriptor-exhaustion guard — live in exactly one place.
        See `optimize`."""
        try:
            self._table.add(rows)
        except Exception as _e:
            _raise_if_fd_exhaustion(_e)
            raise
        self._maybe_optimize()

    def _table_update(self, *, where: str, values: dict[str, Any]) -> None:
        """`table.update` that also feeds the auto-optimize counter.

        Updates rewrite the row into a new data file and a new table version,
        so they accumulate fragments exactly like appends do — recall
        bookkeeping alone is one update per recalled memory per turn."""
        try:
            self._table.update(where=where, values=values)
        except Exception as _e:
            _raise_if_fd_exhaustion(_e)
            raise
        self._maybe_optimize()

    def _table_delete(self, where: str) -> None:
        """`table.delete` that also feeds the auto-optimize counter (deletes
        leave deletion files and a new version behind)."""
        try:
            self._table.delete(where)
        except Exception as _e:
            _raise_if_fd_exhaustion(_e)
            raise
        self._maybe_optimize()

    def _maybe_optimize(self) -> None:
        """Run `optimize()` once `AUTO_OPTIMIZE_EVERY` fragment-creating
        writes have accumulated.

        The counter is incremented and reset under `_optimize_counter_lock`
        so exactly one writer per cycle is elected to compact. The compaction
        itself runs *without* that lock held — concurrent writers keep
        flowing — and is single-flighted by a non-blocking acquire of
        `_optimize_run_lock`: if a previous compaction is still in progress
        this cycle is skipped, and the next threshold crossing retries."""
        if AUTO_OPTIMIZE_EVERY <= 0:
            return
        with self._optimize_counter_lock:
            self._writes_since_optimize += 1
            if self._writes_since_optimize < AUTO_OPTIMIZE_EVERY:
                return
            self._writes_since_optimize = 0
        if not self._optimize_run_lock.acquire(blocking=False):
            return
        try:
            self.optimize()
        finally:
            self._optimize_run_lock.release()

    def optimize(self) -> bool:
        """Compact on-disk fragments and refresh indexes.

        Each individual write creates its own small LanceDB fragment. Left
        unchecked, a high-volume store accumulates thousands of fragment
        files and every query has to open all of them at once, exhausting
        the file-descriptor limit and grinding throughput to a halt.
        `optimize()` merges those fragments into a few large files, prunes
        table versions older than LanceDB's default retention window, and
        folds newly-written rows into the FTS / vector indexes.

        The row count is never changed. Safe to call concurrently with reads
        and writes — LanceDB resolves the compaction commit against
        concurrent appends. Best-effort: any failure is logged and swallowed
        so a maintenance hiccup never breaks the write path. Returns True
        when compaction succeeded.

        Called automatically by the write path (see `AUTO_OPTIMIZE_EVERY`);
        also safe to call manually, e.g. from a scheduled maintenance job."""
        if self._table is None:
            return False
        try:
            self._table.optimize()
            return True
        except Exception as e:
            logger.warning(
                "MemoryStore.optimize (fragment compaction) failed: %s", e
            )
            return False

    # ----- embedding -----

    @property
    def embedder(self) -> SentenceTransformer:
        self._load_embedder()
        return self._embedder

    def _embedding_device_label(self) -> str:
        """The device the loaded model runs on, else the configured override,
        else "auto". Never loads the model (stats must stay cheap)."""
        device = getattr(self._embedder, "device", None) if self._embedder is not None else None
        if device is not None:
            return str(device)
        return EMBEDDING_DEVICE or "auto"

    @staticmethod
    def _to_list(vec) -> list[float]:
        """Normalise an embedder return value to a Python list."""
        if isinstance(vec, list):
            return vec
        return vec.tolist()  # numpy.ndarray, torch.Tensor, etc.

    # ----- embedding scheme -----

    @property
    def scheme_marker_path(self) -> str:
        return os.path.join(self.db_path, EMBEDDING_SCHEME_FILE)

    @property
    def preferred_embedding_scheme(self) -> str:
        """The scheme a new database gets, and `reembed` converts to."""
        if model_uses_task_prefixes(self.embedding_model_name):
            return SCHEME_TASK_PREFIX
        return SCHEME_PLAIN

    def _marker_stamp(self) -> tuple[int, int, int] | None:
        try:
            st = os.stat(self.scheme_marker_path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def read_scheme_marker(self) -> dict[str, Any] | None:
        """The parsed marker file, or None when it is missing or unreadable."""
        try:
            with open(self.scheme_marker_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("scheme") not in EMBEDDING_SCHEMES:
            return None
        return data

    def _write_scheme_marker(self, scheme: str, **extra: Any) -> None:
        """Atomically replace the marker: readers see the old file or the
        new one, never a torn write."""
        payload = {
            "scheme": scheme,
            "model": self.embedding_model_name,
            "updated_at": int(time.time() * 1000),
            **extra,
        }
        tmp = f"{self.scheme_marker_path}.tmp-{os.getpid()}-{threading.get_ident()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.scheme_marker_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        self._scheme = scheme
        self._scheme_stamp = self._marker_stamp()

    def _init_embedding_scheme(self) -> None:
        """Decide this database's scheme: the marker's when there is one;
        "plain" for a legacy database (rows, no marker), whose vectors were
        all made without prefixes; otherwise (empty) the preferred scheme,
        recorded so later opens agree."""
        stamp = self._marker_stamp()
        marker = self.read_scheme_marker() if stamp is not None else None
        if marker is not None:
            self._scheme, self._scheme_stamp = marker["scheme"], stamp
            return
        try:
            empty = self._table.count_rows() == 0
        except Exception:
            empty = False  # unknown: assume existing vectors, change nothing
        if not empty:
            self._scheme = SCHEME_PLAIN
            return
        try:
            self._write_scheme_marker(self.preferred_embedding_scheme)
        except OSError as e:
            # Unrecorded, a later open would see rows and no marker and
            # treat them as plain, so plain is the only consistent choice.
            logger.warning(
                "Could not record the embedding scheme in %s (%s); using %r",
                self.scheme_marker_path, e, SCHEME_PLAIN,
            )
            self._scheme = SCHEME_PLAIN

    @property
    def embedding_scheme(self) -> str:
        """This database's current scheme, re-read when the marker changed
        on disk (another process ran `reembed`)."""
        stamp = self._marker_stamp()
        if stamp is not None and stamp != self._scheme_stamp:
            marker = self.read_scheme_marker()
            if marker is not None:
                self._scheme, self._scheme_stamp = marker["scheme"], stamp
        return self._scheme or SCHEME_PLAIN

    def _embedder_input(
        self, texts: list[str], kind: str, scheme: str | None = None,
    ) -> list[str]:
        if kind not in EMBED_KINDS:
            raise ValueError(f"encode: kind must be one of {EMBED_KINDS}, got {kind!r}")
        scheme = scheme or self.embedding_scheme
        if scheme != SCHEME_TASK_PREFIX or not model_uses_task_prefixes(
            self.embedding_model_name
        ):
            return texts
        prefix = _NOMIC_TASK_PREFIXES[kind]
        return [prefix + t for t in texts]

    def _encode_many(self, texts: Sequence[str], kind: str, scheme: str | None = None):
        """Raw embedder output (one row per text) under this store's scheme,
        or `scheme` when given (`reembed`)."""
        inputs = self._embedder_input(list(texts), kind, scheme)
        # Resolve (and if needed load) the model BEFORE taking the encode
        # lock — loading takes _embedder_lock then the encode lock, and the
        # order must be the same everywhere.
        model = self.embedder
        with self._encode_lock:
            return model.encode(
                inputs,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def encode(self, text: str, *, kind: str = "document") -> list[float]:
        """Encode text into a single normalised 768-d vector. Thread-safe:
        embedder calls are serialised process-wide (see _ENCODE_LOCK).

        `kind` is "document" for text that is stored, or compared with
        stored text (dedup), and "query" for a search query; with the
        "task_prefix" scheme it picks nomic's prefix."""
        (text,) = self._embedder_input([text], kind)
        model = self.embedder
        with self._encode_lock:
            vector = model.encode(text, normalize_embeddings=True)
        return self._to_list(vector)

    def encode_batch(
        self, texts: Sequence[str], *, kind: str = "document",
    ) -> list[list[float]]:
        """Encode many texts in one shot. Use this for bulk inserts.
        Thread-safe, like :meth:`encode`; `kind` as for :meth:`encode`."""
        return [self._to_list(v) for v in self._encode_many(texts, kind)]

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
        *,
        confidence: float = 0.8,
        metadata_extra: dict[str, Any] | None = None,
    ) -> str:
        """Store a new memory entry. Returns the new memory ID."""
        if not text or not text.strip():
            raise ValueError("MemoryStore.store: `text` must be non-empty")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(
                f"MemoryStore.store: `text` is {len(text)} chars, exceeding the "
                f"{MAX_TEXT_CHARS}-char limit (MEMORY_MAX_TEXT_CHARS). A memory "
                f"entry should be a distilled fact, not a document — summarise "
                f"or split it before storing."
            )
        text = _check_injection_guard(text, where="store")

        category, tier, importance, confidence = self._normalise_inputs(
            category, tier, importance, confidence
        )

        mem_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        metadata = self._build_metadata(
            tier=tier, confidence=confidence, now_ms=now_ms, extra=metadata_extra
        )

        vector = self.encode(text, kind="document")
        self._table_add([
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
        for idx, txt in enumerate(texts):
            if len(txt) > MAX_TEXT_CHARS:
                raise ValueError(
                    f"store_many: entry {idx} `text` is {len(txt)} chars, "
                    f"exceeding the {MAX_TEXT_CHARS}-char limit "
                    f"(MEMORY_MAX_TEXT_CHARS) — summarise or split it."
                )
        vectors = self.encode_batch(texts, kind="document")

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

        self._table_add(prepared)
        return ids

    def row_exists(self, mem_id: str) -> bool:
        """True if a row with exactly this id is stored — live OR archived.

        Unlike `has_id()` (live rows only) and `get_by_id()` (follows the
        supersede chain), this answers "is this id taken?"."""
        if not mem_id:
            return False
        self._checkout_latest()
        rows = (
            self._table.search()
            .where(f"id = '{_escape_sql(str(mem_id))}'")
            .limit(1)
            .to_list()
        )
        return bool(rows)

    def store_raw(
        self,
        *,
        text: str,
        vector: Sequence[float],
        category: str,
        scope: str,
        importance: float,
        metadata: str | dict[str, Any],
        timestamp: int | None = None,
        mem_id: str | None = None,
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

        Pass `mem_id` to keep an existing id (restoring an export, so
        supersede links between rows stay valid). Raises `ValueError` if a
        row with that id already exists, archived or not.

        For ordinary writes, prefer `store()` / `store_many()`."""
        if not text or not str(text).strip():
            raise ValueError("MemoryStore.store_raw: `text` must be non-empty")
        if not vector:
            raise ValueError("MemoryStore.store_raw: `vector` must be non-empty")
        text = _check_injection_guard(text, where="store_raw")
        # Accept a pre-built dict as well as a JSON string so callers don't
        # have to json.dumps() themselves (and to match store()'s metadata_extra API).
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        if mem_id is None:
            mem_id = str(uuid.uuid4())
        else:
            mem_id = str(mem_id).strip()
            if not mem_id:
                raise ValueError("MemoryStore.store_raw: `mem_id` must be non-empty")
            if self.row_exists(mem_id):
                raise ValueError(f"MemoryStore.store_raw: id {mem_id!r} already exists")
        ts = int(time.time() * 1000) if timestamp is None else int(timestamp)
        importance = max(0.0, min(1.0, float(importance)))

        self._table_add([
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
        search = self._vector_query(vector)
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        results = search.limit(max(limit * SEARCH_OVERFETCH_MULTIPLIER, limit)).to_list()
        # Expired rows are excluded too: the smart extractor uses this for
        # dedup, and matching an expired "dentist tomorrow" would merge or
        # skip a fresh occurrence into a row recall can no longer surface.
        now_ms = int(time.time() * 1000)
        return [
            self._row_to_dict(r, distance=r.get("_distance"), keep_vector=keep_vector)
            for r in results
            if not _hidden_from_recall(r.get("metadata", ""), now_ms)
        ][:limit]

    @_serialised
    def update(
        self,
        mem_id: str,
        text: str | None = None,
        importance: float | None = None,
        category: str | None = None,
        tier: str | None = None,
        *,
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
        if text is not None:
            # Same guards as store(): validate BEFORE the old row is archived,
            # so a bad replacement can never destroy the existing memory.
            if not str(text).strip():
                raise ValueError("MemoryStore.update: `text` must be non-empty")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(
                    f"MemoryStore.update: `text` is {len(text)} chars, exceeding the "
                    f"{MAX_TEXT_CHARS}-char limit (MEMORY_MAX_TEXT_CHARS)."
                )

        existing = self.get_by_id(mem_id)
        if existing is None:
            return None
        if _is_archived(existing["metadata"]):
            # The chain led to an archived row: its successor is gone. Editing
            # or superseding it would resurrect a stale version, so treat it
            # as not updatable (it can still be read or forgotten).
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

        # Metadata-only update path.
        # NOTE: get_by_id follows the supersede chain, so existing["id"] may
        # differ from mem_id (the old archived row). Always use existing["id"]
        # as the target so the update lands on the live row.
        effective_id = existing["id"]
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

        self._table_update(
            where=f"id = '{_escape_sql(effective_id)}'",
            values=update_values,
        )
        return effective_id

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

        vector = self.encode(text, kind="document")

        # Atomic-ish: do the write of the new row first so a partial failure
        # leaves the old row visible (preferred over a missing row).
        self._table_add([
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

        self._table_update(
            where=f"id = '{_escape_sql(old_id)}'",
            values={"metadata": json.dumps(archived_meta)},
        )
        return new_id

    @_serialised
    def forget(self, mem_id: str, *, purge_history: bool = False) -> bool:
        """Hard-delete a memory by ID. Returns True if the memory existed.

        By default only the live version is deleted; earlier versions it
        superseded stay archived until the archive purge removes them (the
        compactor relies on this). Pass ``purge_history=True`` when a user
        asks for a memory to be removed: it also deletes every earlier
        wording, so none of them stays on disk or in an export."""
        existing = self.get_by_id(mem_id)
        if existing is None:
            return False
        if not purge_history:
            # get_by_id follows the supersede chain; delete the resolved live
            # row rather than an already-archived predecessor.
            self._table_delete(f"id = '{_escape_sql(existing['id'])}'")
            return True
        # get_by_id follows the supersede chain forward to the live row;
        # walk `supersedes` back from there to collect the whole lineage.
        lineage: list[str] = [existing["id"]]
        seen: set[str] = {existing["id"]}
        row: dict[str, Any] | None = existing
        while row is not None and len(lineage) < 1000:
            prev_id = (row.get("metadata") or {}).get("supersedes")
            if not prev_id or prev_id in seen:
                break
            seen.add(prev_id)
            lineage.append(prev_id)
            row = self.get_by_id(prev_id, follow_chain=False)
        if mem_id not in seen:
            lineage.append(mem_id)
        id_list = ", ".join(f"'{_escape_sql(i)}'" for i in lineage)
        self._table_delete(f"id IN ({id_list})")
        return True

    def delete(self, mem_id: str) -> bool:
        """Delete a memory by ID — alias for `forget()`.

        Provided because `delete` is the conventional name to reach for;
        the behaviour is identical to `forget()` — a hard delete of the
        live row, following the supersede chain to the current version."""
        return self.forget(mem_id)

    @_serialised
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
        if existing is None or _is_archived(existing["metadata"]):
            # Archived with no live successor: nothing to count against.
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
        # get_by_id follows the supersede chain, so `existing` may be the live
        # successor of `mem_id`. Write to the row we read: writing its metadata
        # onto the archived `mem_id` row would un-archive the stale version.
        self._table_update(
            where=f"id = '{_escape_sql(existing['id'])}'",
            values={"metadata": json.dumps(metadata)},
        )
        return True

    @_serialised
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

            # Cross-session promotion ledger. Threshold <= 0 disables both the
            # ledger maintenance and the promotion (per docstring on the env var).
            if (
                CROSS_SESSION_PROMOTION_THRESHOLD > 0
                and session_id
                and not metadata.get("cross_session")
            ):
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
                self._table_update(
                    where=f"id = '{_escape_sql(mem_id)}'",
                    values={"metadata": json.dumps(metadata)},
                )
                updated += 1
            except Exception as e:
                logger.warning("mark_recall_used update %s failed: %s", mem_id, e)
        return updated

    # ----- CRUD: read -----

    def get_by_id(self, mem_id: str, *, follow_chain: bool = True) -> dict[str, Any] | None:
        """Retrieve a memory by ID (regardless of archive state).

        If *follow_chain* is True (the default) and the requested row is
        archived with a ``superseded_by`` pointer, the method follows that
        chain until it reaches the current live version, guarding against
        cycles with a depth limit of 32.

        When the next hop is missing (the successor was deleted by `forget`
        or the compactor) the last row found is returned — an archived row
        whose `state` says so — so the history stays addressable. Callers
        that would change a row (`update`, `increment_access_count`) refuse
        such a row rather than touch a stale version.
        """
        self._checkout_latest()
        seen: set[str] = set()
        current_id = mem_id
        last_row: dict[str, Any] | None = None
        while True:
            results = (
                self._table.search()
                .where(f"id = '{_escape_sql(current_id)}'")
                .limit(1)
                .to_list()
            )
            if not results:
                return last_row
            row = self._row_to_dict(results[0])
            last_row = row
            if not follow_chain:
                return row
            meta = row.get("metadata") or {}
            next_id: str | None = meta.get("superseded_by") if meta.get("state") == ARCHIVED_STATE else None
            if not next_id or next_id in seen or len(seen) >= 32:
                return row
            seen.add(current_id)
            current_id = next_id

    def has_id(self, mem_id: str) -> bool:
        """True if the ID exists and is not archived. Used for BM25 ghost
        protection — see also `check_ids` for batch lookups."""
        self._checkout_latest()
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
        self._checkout_latest()
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
        *,
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
            # Match both Python json.dumps spacing ("tier": "x") and compact
            # serialisation ("tier":"x") so legacy or external-tool data isn't
            # silently excluded.
            safe_tier = _escape_sql(tier)
            clauses.append(
                f"(metadata LIKE '%\"tier\": \"{safe_tier}\"%' OR "
                f"metadata LIKE '%\"tier\":\"{safe_tier}\"%')"
            )

        # Archived and other-session rows are filtered in Python after the
        # query, so a fixed over-fetch can come back short (or empty) when
        # many such rows precede the wanted ones — e.g. a memory edited many
        # times, or one large earlier session. Widen the window until enough
        # rows survive the filter or the table is exhausted.
        post_filter = (not include_archived) or (session_id is not None)
        wanted = max(0, limit) + max(0, offset)
        fetch_limit = wanted + max(limit, 16) * 2 if post_filter else wanted
        where_clause = _and_clauses(*clauses)

        self._checkout_latest()
        while True:
            query = self._table.search()
            if where_clause:
                query = query.where(where_clause)
            results = query.limit(fetch_limit).to_list()

            rows = [self._row_to_dict(r) for r in results]
            if not include_archived:
                rows = [r for r in rows if r["metadata"].get("state") != ARCHIVED_STATE]
            if session_id is not None:
                rows = [r for r in rows if _match_session(r["metadata"], session_id)]
            exhausted = len(results) < fetch_limit
            if len(rows) >= wanted or exhausted or fetch_limit >= MAX_SCAN_ROWS:
                break
            fetch_limit = min(fetch_limit * 4, MAX_SCAN_ROWS)
        return rows[offset : offset + limit]

    # Columns the recall anchors need; the 768-d vector is never read.
    _ANCHOR_COLUMNS = (
        "id", "text", "category", "scope", "importance", "timestamp", "metadata",
    )

    @staticmethod
    def _like_any(key: str, value_json: str) -> str:
        """LIKE test for `"key": value` under both json.dumps spacings."""
        # LIKE treats a backslash as its escape character, and JSON text
        # contains backslashes for quotes, backslashes and non-ASCII
        # characters. Double them so the pattern matches the stored text
        # literally; otherwise such a session id matched nothing.
        v = _escape_sql(value_json.replace("\\", "\\\\"))
        k = _escape_sql(json.dumps(key).replace("\\", "\\\\"))
        return f"(metadata LIKE '%{k}: {v}%' OR metadata LIKE '%{k}:{v}%')"

    def _session_anchor_rows(
        self, session_id: str, strict: bool
    ) -> list[dict[str, Any]]:
        """Recallable rows for `session_id`, unordered, read in one scan.

        The session test is pushed into the query as a LIKE prefilter and
        then re-checked in Python (LIKE wildcards in an id only ever
        over-match), so the cost no longer grows with the whole table, and
        no vector column is loaded. Returns [] on a scan error."""
        sess = self._like_any("source_session", json.dumps(session_id))
        if strict:
            session_clause = sess
        else:
            session_clause = " OR ".join(
                [sess, self._like_any("cross_session", "true")]
                + [self._like_any("tier", json.dumps(t)) for t in sorted(CROSS_SESSION_TIERS)]
            )
        where = _and_clauses(
            "timestamp > 0",
            session_clause,
            "NOT " + self._like_any("state", json.dumps(ARCHIVED_STATE)),
        )
        try:
            raw = (
                self._table.search()
                .where(where)
                .select(list(self._ANCHOR_COLUMNS))
                .limit(MAX_SCAN_ROWS)
                .to_list()
            )
        except Exception:
            return []
        now_ms = int(time.time() * 1000)
        out = []
        for r in raw:
            d = self._row_to_dict(r)
            if _hidden_from_recall(d["metadata"], now_ms):
                continue
            if not _session_anchor_match(d["metadata"], session_id, strict):
                continue
            out.append(d)
        return out

    def anchors_for_session(
        self, session_id: str, limit: int = 2, *, strict: bool = False
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """`(first_for_session, recent_for_session)` from a single scan."""
        if not session_id:
            return [], []
        rows = self._session_anchor_rows(session_id, strict)
        rows.sort(key=lambda x: x.get("timestamp", 0))
        return rows[:limit], rows[::-1][:limit]

    def recent_for_session(
        self, session_id: str, limit: int = 2, *, strict: bool = False
    ) -> list[dict[str, Any]]:
        """Return the *limit* most recently written recallable (non-archived,
        non-expired) memories for *session_id*, ordered newest-first by
        timestamp.

        Used by the recall path as a recency anchor so the agent keeps sight
        of the latest instruction even when the current query is semantically
        distant from earlier task framing."""
        return self.anchors_for_session(session_id, limit, strict=strict)[1]

    def first_for_session(
        self, session_id: str, limit: int = 2, *, strict: bool = False
    ) -> list[dict[str, Any]]:
        """Return the *limit* oldest recallable (non-archived, non-expired)
        memories for *session_id*, ordered oldest-first by timestamp.

        Used alongside `recent_for_session` to pin task-framing memories
        (written at the start of a session) so they are always injected even
        after many subsequent turns push them out of the recency window."""
        return self.anchors_for_session(session_id, limit, strict=strict)[0]

    def search(
        self,
        query_text: str,
        limit: int = 10,
        mode: str = "hybrid",  # "vector" | "bm25" | "hybrid"
        category: str | None = None,
        scope: str | None = None,
        *,
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
          - hybrid: keep iff `score >= min_score`             (the 0-1 `score`
            derived from the fused rank; the raw `_rrf_score` tops out near
            0.033, so a threshold on it could never be met by a sane value)

        Every result carries a `score` in [0, 1], higher is better: cosine
        similarity `1 - _distance` (vector), `_score / 10` (bm25), or
        `_rrf_score * 30` (hybrid, so it follows the hybrid ranking).
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
        """Apply non-archived, non-expired, and session_id filters in Python.

        Archived and expired (``valid_until`` passed) filtering is always
        applied — callers that want those rows go through admin paths.
        Session filtering is applied only when `session_id` is provided."""
        now_ms = int(time.time() * 1000)
        out = []
        for row in rows:
            meta = row.get("metadata", "")
            if _hidden_from_recall(meta, now_ms):
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
        self._checkout_latest()
        vector = self.encode(query_text, kind="query")
        search = self._vector_query(vector)
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        # Over-fetch more aggressively when post-filtering by session, since
        # session-matching rows may be sparse in the top-N from LanceDB.
        overfetch = SEARCH_OVERFETCH_MULTIPLIER * (3 if session_id is not None else 1)
        try:
            results = search.limit(max(limit * overfetch, limit)).to_list()
        except Exception as _e:
            _raise_if_fd_exhaustion(_e)
            raise
        results = self._post_filter(results, session_id=session_id)

        rows = [
            self._row_to_dict(r, distance=r.get("_distance"), keep_vector=keep_vector)
            for r in results
        ]
        if min_score is not None:
            rows = [
                r for r in rows
                if (d := r.get("_distance")) is not None and (1.0 - float(d)) >= min_score
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
        self._checkout_latest()
        # Scope FTS to the `text` column only — LanceDB's FTS otherwise indexes
        # all string columns, polluting results with id/category/metadata.
        search = self._table.search(
            query_text, query_type="fts", fts_columns=["text"]
        )
        where = self._filters_clause(category, scope)
        if where:
            search = search.where(where)
        overfetch = SEARCH_OVERFETCH_MULTIPLIER * (3 if session_id is not None else 1)
        try:
            results = search.limit(max(limit * overfetch, limit)).to_list()
        except Exception as _e:
            _raise_if_fd_exhaustion(_e)
            raise
        results = self._post_filter(results, session_id=session_id)

        if min_score is not None:
            results = [r for r in results if float(r.get("_score") or 0.0) >= min_score]
        rows = [
            self._row_to_dict(r, score=r.get("_score"), keep_vector=keep_vector)
            for r in results
        ]
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
        the other search modes. `score` is derived from `_rrf_score`, so it
        follows the returned order."""
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
        bm25_map = {r["id"]: r for r in bm25_results}
        merged: dict[str, dict[str, Any]] = {}
        for entry in vector_results:
            mid = entry["id"]
            entry["_rrf_score"] = (
                1.0 / (rrf_k + v_rank[mid])
                + (1.0 / (rrf_k + b_rank[mid]) if mid in b_rank else 0.0)
            )
            # Keep the BM25 leg's raw signal on rows found by both legs.
            if mid in bm25_map and "_score" in bm25_map[mid]:
                entry["_score"] = bm25_map[mid]["_score"]
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

        # The legs' `_row_to_dict` set `score` from their own signal (cosine
        # or BM25), which mixed two scales and did not follow this ranking.
        for entry in merged.values():
            entry["score"] = _rrf_to_score(entry["_rrf_score"])

        ranked = sorted(
            merged.values(),
            key=lambda e: e.get("_rrf_score", 0.0),
            reverse=True,
        )
        if min_score is not None:
            ranked = [
                r for r in ranked
                if _rrf_to_score(float(r.get("_rrf_score") or 0.0)) >= min_score
            ]
        return ranked[:limit]

    # ----- bulk ops -----

    def stats(self) -> dict[str, Any]:
        """Return memory store statistics. Counts include archived rows;
        use `purge_archived` to remove them. The breakdowns read every row
        (metadata only, in pages).

        When the table cannot be read, `scan_error` holds the message and
        the active / archived counts are not meaningful (they stay 0) —
        callers must not present that as a healthy empty store."""
        scan_error: str | None = None
        try:
            count = len(self._table)
        except Exception as e:
            count = 0
            scan_error = str(e)

        cat_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        active_count = 0
        archived_count = 0

        try:
            rows = list(self._scan_all(
                limit=None, strict=True, columns=("category", "metadata"),
            ))
        except Exception as e:
            rows = []
            scan_error = scan_error or str(e)
        for row in rows:
            cat = row.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            meta = _parse_metadata(row.get("metadata", "{}"))
            tier = meta.get("tier", "working")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if meta.get("state") == ARCHIVED_STATE:
                archived_count += 1
            else:
                active_count += 1

        result = {
            "total_memories": count,
            "active_memories": active_count,
            "archived_memories": archived_count,
            "db_path": self.db_path,
            "embedding_model": self.embedding_model_name,
            "embedding_device": self._embedding_device_label(),
            "embedding_scheme": self.embedding_scheme,
            "embedding_scheme_preferred": self.preferred_embedding_scheme,
            "reembed_interrupted": self.reembed_interrupted(),
            "vector_dimensions": VECTOR_DIM,
            "categories": cat_counts,
            "tiers": tier_counts,
        }
        if scan_error is not None:
            result["scan_error"] = scan_error
        return result

    _PURGE_BATCH = 500

    def _delete_ids(self, ids: list[str]) -> None:
        """Delete rows by id in batches (one table version per batch)."""
        for i in range(0, len(ids), self._PURGE_BATCH):
            chunk = ids[i : i + self._PURGE_BATCH]
            in_list = ", ".join(f"'{_escape_sql(m)}'" for m in chunk)
            self._table_delete(f"id IN ({in_list})")

    def purge_archived(
        self,
        grace_period_days: int = 30,
        *,
        dry_run: bool = False,
        strict: bool = False,
    ) -> int:
        """Permanently delete archived memories older than grace_period_days.

        Returns the number of entries deleted. With ``dry_run=True`` nothing
        is deleted — the return value is the count a real run would remove.
        Every row is examined (in pages, without vectors), however large the
        store. With ``strict=True`` an unreadable table raises instead of
        reporting 0."""
        cutoff_ms = int(time.time() * 1000) - (grace_period_days * 24 * 60 * 60 * 1000)
        to_delete: list[str] = []
        for row in self._scan_all(
            limit=None, strict=strict, columns=("id", "metadata")
        ):
            meta = _parse_metadata(row.get("metadata", "{}"))
            if meta.get("state") != ARCHIVED_STATE:
                continue
            invalidated = meta.get("invalidated_at", 0) or 0
            if invalidated and invalidated < cutoff_ms:
                mem_id = row.get("id")
                if mem_id:
                    to_delete.append(mem_id)

        purged = len(to_delete)
        if purged and not dry_run:
            self._delete_ids(to_delete)
            logger.info(
                "Purged %d archived memories older than %d days",
                purged, grace_period_days,
            )
        return purged

    def purge_expired(
        self,
        grace_period_days: int = 30,
        *,
        dry_run: bool = False,
        strict: bool = False,
    ) -> int:
        """Permanently delete memories whose ``valid_until`` passed more than
        *grace_period_days* ago.

        The grace window mirrors :meth:`purge_archived` — measured from the
        expiry, not the write — so an event is kept on disk for a while after
        recall stops showing it. Archived rows are skipped (they belong to
        ``purge_archived``, so nothing is counted twice), and rows with a
        missing or malformed ``valid_until`` are never touched. Returns the
        number deleted, or with ``dry_run=True`` the number a real run would
        delete."""
        cutoff_ms = int(time.time() * 1000) - (grace_period_days * 24 * 60 * 60 * 1000)
        to_delete: list[str] = []
        for row in self._scan_all(
            limit=None, strict=strict, columns=("id", "metadata")
        ):
            meta = _parse_metadata(row.get("metadata", "{}"))
            if meta.get("state") == ARCHIVED_STATE or not _is_expired(meta, cutoff_ms):
                continue
            mem_id = row.get("id")
            if mem_id:
                to_delete.append(mem_id)

        purged = len(to_delete)
        if purged and not dry_run:
            self._delete_ids(to_delete)
            logger.info(
                "Purged %d memories expired more than %d days ago",
                purged, grace_period_days,
            )
        return purged

    # -- in-use lock ---------------------------------------------------------
    # A running agent encodes a vector and may write it seconds later (after
    # LLM dedup calls). If the embedding scheme changed in between, that row
    # would carry a vector of the old scheme in a store marked as new, and
    # no later pass could tell. So `reembed` needs the store to itself: the
    # agent holds a shared flock on IN_USE_LOCK_FILE for as long as it runs,
    # and `reembed` takes it exclusively or refuses.

    def hold_in_use_lock(self) -> bool:
        """Mark this database as in use by a running agent (shared lock,
        kept until the process exits). Returns False when `reembed` is
        converting it right now, or locking is unavailable."""
        if fcntl is None or getattr(self, "_in_use_fd", None) is not None:
            return getattr(self, "_in_use_fd", None) is not None
        try:
            os.makedirs(self.db_path, exist_ok=True)
            fd = os.open(os.path.join(self.db_path, IN_USE_LOCK_FILE),
                         os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._in_use_fd = fd
        return True

    @contextlib.contextmanager
    def _exclusive_use(self) -> Iterator[None]:
        """Hold the database exclusively, or raise `StoreInUseError`."""
        if fcntl is None:
            yield
            return
        os.makedirs(self.db_path, exist_ok=True)
        fd = os.open(os.path.join(self.db_path, IN_USE_LOCK_FILE),
                     os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise StoreInUseError(
                    "the memory database is in use by a running Hermes agent; "
                    "stop Hermes, run reembed, then start it again"
                ) from e
            yield
        finally:
            os.close(fd)

    def reembed_interrupted(self) -> bool:
        """True when a `reembed` started converting and did not finish."""
        marker = self.read_scheme_marker()
        return bool(marker and marker.get("reembed_target"))

    def reembed(
        self,
        *,
        batch_size: int = REEMBED_BATCH_SIZE,
        dry_run: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Re-encode every row's vector with the preferred embedding scheme.

        All-or-nothing: every vector is computed first, then all of them are
        written in one table commit (only the `vector` column; text and
        metadata are untouched), and only then does the marker name the new
        scheme. A failure or interruption before the commit changes nothing;
        one between the commit and the marker leaves `reembed_target` in the
        marker, which `doctor` reports, and running `reembed` again repairs
        it. Rows another process writes meanwhile are caught by further
        passes. Returns a summary dict; `status` is "up_to_date",
        "dry_run", "converted" or "incomplete" (rows kept arriving; run it
        again)."""
        if batch_size < 1:
            raise ValueError("reembed: batch_size must be >= 1")
        target = self.preferred_embedding_scheme
        with self._write_lock:
            self._checkout_latest()
            current = self.embedding_scheme
            interrupted = self.reembed_interrupted()
            result: dict[str, Any] = {
                "from": current, "to": target, "rows": self._table.count_rows(),
                "reembedded": 0, "interrupted_before": interrupted,
            }
            if current == target and not interrupted:
                result["status"] = "up_to_date"
                return result
            if dry_run:
                result["status"] = "dry_run"
                return result

            had_vector_index = self._vector_index_metric() is not None
            with self._exclusive_use():
                result["reembedded"], complete = self._reembed_passes(
                    current, target, batch_size, progress,
                )
            result["status"] = "converted" if complete else "incomplete"
        if had_vector_index and result["reembedded"]:
            # The IVF_PQ codebook was trained on the old vectors.
            try:
                self._build_vector_index()
            except Exception as e:
                logger.warning("reembed: rebuilding the vector index failed: %s", e)
        if result["reembedded"]:
            self.optimize()
        return result

    def _reembed_passes(
        self,
        current: str,
        target: str,
        batch_size: int,
        progress: Callable[[int, int], None] | None,
    ) -> tuple[int, bool]:
        """Convert every row, then any row that appeared meanwhile. Returns
        (rows re-encoded, whether a final pass found nothing left)."""
        done: set[str] = set()
        flipped = False
        for _ in range(_REEMBED_MAX_PASSES):
            self._checkout_latest()
            rows = [r for r in self._id_text_rows() if r["id"] not in done]
            if not rows:
                break
            ids = [r["id"] for r in rows]
            duplicated = sorted({i for i in ids if ids.count(i) > 1}) if len(set(ids)) != len(ids) else []
            if duplicated:
                # The vector write is keyed on id and would be rejected as
                # ambiguous, after all the encoding work. Say what is wrong.
                raise ValueError(
                    f"{len(duplicated)} id(s) are stored more than once "
                    f"(e.g. {duplicated[0]}); remove the duplicate rows, then run "
                    "reembed again"
                )
            vectors = self._reembed_vectors(
                [r["text"] for r in rows], target, batch_size, progress,
            )
            if not flipped:
                self._write_scheme_marker(current, reembed_target=target)
            try:
                self._replace_vectors(ids, vectors)
            except Exception:
                if not flipped:
                    # The commit failed, so nothing changed: drop the
                    # pending flag. (An interrupt keeps it: the commit may
                    # have landed.)
                    self._write_scheme_marker(current)
                raise
            if not flipped:
                # From here on this and every other process encodes new
                # rows and queries with the target scheme.
                self._write_scheme_marker(target)
                flipped = True
            done.update(ids)
        else:
            # Rows kept arriving: keep the pending flag so doctor says to
            # run it again rather than claiming a clean conversion.
            self._write_scheme_marker(target, reembed_target=target)
            return len(done), False
        if not flipped:
            # Empty table: nothing to convert, only the label changes.
            self._write_scheme_marker(target)
        return len(done), True

    def _id_text_rows(self) -> list[dict[str, Any]]:
        """Every row's id and text, without vectors (strict scan)."""
        n = max(self._table.count_rows(), 1)
        return self._table.search().select(["id", "text"]).limit(n).to_list()

    def _reembed_vectors(
        self,
        texts: list[str],
        scheme: str,
        batch_size: int,
        progress: Callable[[int, int], None] | None,
    ):
        import numpy as np

        chunks = []
        for start in range(0, len(texts), batch_size):
            out = np.asarray(
                self._encode_many(texts[start:start + batch_size], "document", scheme),
                dtype=np.float32,
            )
            if out.ndim != 2 or out.shape[1] != VECTOR_DIM:
                raise ValueError(
                    f"reembed: the embedder returned vectors of shape {out.shape}, "
                    f"the table stores {VECTOR_DIM}-d vectors; nothing was changed"
                )
            chunks.append(out)
            if progress is not None:
                progress(min(start + batch_size, len(texts)), len(texts))
        return np.concatenate(chunks)

    def _replace_vectors(self, ids: list[str], vectors) -> None:
        """Overwrite the `vector` of rows `ids` in a single commit."""
        import pyarrow as pa

        vector_type = self._table.schema.field("vector").type
        flat = pa.array(vectors.reshape(-1), type=pa.float32())
        source = pa.table({
            "id": pa.array(ids, type=pa.string()),
            "vector": pa.FixedSizeListArray.from_arrays(flat, VECTOR_DIM).cast(vector_type),
        })
        self._table.merge_insert("id").when_matched_update_all().execute(source)

    def _checkout_latest(self) -> None:
        """Advance the table handle to the latest dataset version.

        When another process (e.g. the CLI import command) has written to the
        same LanceDB path, the in-process table handle may be pinned to an
        older version and miss the new rows entirely.  Calling this before any
        read brings the handle up to date.  Falls back to re-opening the table
        for older LanceDB versions that lack `checkout_latest()`."""
        try:
            self._table.checkout_latest()
        except AttributeError:
            # Older lancedb without checkout_latest — re-open for a fresh handle.
            import contextlib
            with contextlib.suppress(Exception):
                self._table = self._db.open_table("memories")
        except Exception:
            pass

    _SCAN_PAGE_ROWS = 5000

    def _scan_all(
        self,
        limit: int | None = MAX_SCAN_ROWS,
        *,
        strict: bool = False,
        columns: Sequence[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        """Iterate rows of the table, up to `limit` (None = every row).
        Yields raw rows.

        Reads in pages so a large store is never materialised in one query,
        and `columns` (e.g. without `vector`) keeps the 768-d vectors out of
        memory when only metadata is needed.

        Fail-soft by default — a scan error is logged and an empty iterator
        returned, so `stats` / `purge_archived` degrade gracefully. Pass
        `strict=True` to re-raise instead: callers that must not mistake a
        corrupt store (a missing fragment file) for an empty one — `export`,
        `doctor` — need the failure surfaced rather than swallowed."""
        results: list[dict[str, Any]] = []
        try:
            while limit is None or len(results) < limit:
                page = self._SCAN_PAGE_ROWS
                if limit is not None:
                    page = min(page, limit - len(results))
                query = self._table.search()
                if columns:
                    query = query.select(list(columns))
                rows = query.offset(len(results)).limit(page).to_list()
                results.extend(rows)
                if len(rows) < page:
                    break
        except Exception as e:
            logger.error("Full table scan failed: %s", e)
            if strict:
                raise
            return iter(())
        return iter(results)

    def salvage_scan(
        self, limit: int = MAX_SCAN_ROWS
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Best-effort recovery scan for a corrupted dataset.

        A normal scan aborts entirely the moment one fragment file is
        missing. This walks the table's version history newest-first and
        returns the first version that scans cleanly. If no version is
        intact and the optional `pylance` package is available, it falls
        back to reading the dataset fragment-by-fragment, skipping the
        unreadable ones. Read-only — the table handle is restored to the
        latest version before returning.

        Returns `(rows, report)`. `report` keys: `mode` (`version`,
        `fragment` or `none`), `version`, `rows`, `skipped_fragments`,
        `failed_versions`, `note`."""
        report: dict[str, Any] = {
            "mode": "none", "version": None, "rows": 0,
            "skipped_fragments": [], "failed_versions": [], "note": "",
        }
        try:
            versions = [v["version"] for v in self._table.list_versions()]
        except Exception as e:
            logger.warning("salvage: could not list table versions: %s", e)
            versions = []

        # Strategy 1 — newest table version that scans cleanly.
        try:
            for ver in sorted(versions, reverse=True):
                try:
                    self._table.checkout(ver)
                    rows = self._table.search().limit(limit).to_list()
                except Exception as e:
                    logger.warning("salvage: version %s unreadable: %s", ver, e)
                    report["failed_versions"].append(ver)
                    continue
                report.update(mode="version", version=ver, rows=len(rows))
                return rows, report
        finally:
            import contextlib
            with contextlib.suppress(Exception):
                self._table.checkout_latest()

        # Strategy 2 — fragment-by-fragment (needs the optional pylance pkg).
        return self._salvage_fragments(limit, report), report

    def _salvage_fragments(
        self, limit: int, report: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Read the dataset fragment-by-fragment, skipping unreadable ones.

        Used by `salvage_scan` when no whole version scans cleanly. Needs
        the optional `pylance` package; without it, records a note in
        `report` and returns an empty list."""
        try:
            dataset = self._table.to_lance()
        except Exception as e:
            report["note"] = (
                "fragment-level salvage unavailable — install the optional "
                f"'pylance' package (pip install pylance) to enable it ({e})"
            )
            logger.warning("salvage: fragment scan unavailable: %s", e)
            return []
        rows: list[dict[str, Any]] = []
        for frag in dataset.get_fragments():
            try:
                frag_rows = frag.to_table().to_pylist()
            except Exception as e:
                logger.warning(
                    "salvage: fragment %s unreadable: %s", frag.fragment_id, e
                )
                report["skipped_fragments"].append(str(frag.fragment_id))
                continue
            rows.extend(frag_rows)
            if len(rows) >= limit:
                rows = rows[:limit]
                break
        report.update(mode="fragment", rows=len(rows))
        return rows

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
        parsed_meta = _parse_metadata(meta) if isinstance(meta, str) else meta
        result: dict[str, Any] = {
            "id": row["id"],
            "text": row["text"],
            "category": row["category"],
            "scope": row["scope"],
            "importance": row["importance"],
            "timestamp": row["timestamp"],
            "metadata": parsed_meta,
            # Convenience top-level key so callers don't have to dig into
            # metadata — mirrors how `category`/`scope`/`importance` are
            # exposed directly even though tier lives in the metadata JSON.
            "tier": parsed_meta.get("tier", "working"),
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
            result["score"] = _rrf_to_score(result["_rrf_score"])
        elif distance is not None:
            # Cosine distance (see VECTOR_DISTANCE_TYPE): 0 = identical,
            # 1 = orthogonal, 2 = opposite, so this is cosine similarity
            # with negative values clamped to 0.
            result["score"] = max(0.0, min(1.0, 1.0 - float(distance)))
        elif score is not None:
            # BM25 raw — no clean normalisation. Scale by 10 (typical
            # max for short FTS queries) and clamp; advanced callers
            # should use `_score` directly.
            result["score"] = max(0.0, min(1.0, float(score) / 10.0))
        else:
            result["score"] = 0.0
        return result
