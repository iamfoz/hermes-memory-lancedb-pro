"""Memory Compactor — progressive summarisation by clustering similar memories.

Identifies clusters of semantically similar memories older than `min_age_days`
and merges each cluster into a single, higher-quality entry. Reduces noise
in the long-tail recall path: months of "I prefer X" / "I like X" / "X is
good" entries collapse into one consolidated preference.

Algorithm (matches CortexReach memory-compactor.ts):
    1. Fetch memories older than `min_age_days` (with vectors).
    2. Greedy cluster expansion seeded by importance — most important entry
       in an unassigned set seeds a new cluster, sweeping in any other
       unassigned entry whose cosine similarity ≥ threshold.
    3. For each cluster of size ≥ `min_cluster_size`:
        - text:       deduplicated lines joined with newline
        - importance: max across cluster (never downgrade)
        - category:   plurality vote (legacy column)
        - scope:      first member's scope (callers must pre-group by scope)
        - metadata:   { compacted: True, source_count: N, compacted_at: ms }
    4. Delete source entries, store merged entry.

The compactor is opt-in. Run it on a cron / start-up cooldown — see
`should_run_compaction` and `record_compaction_run`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._sql import is_archived as _is_archived
from ._sql import parse_metadata as _parse_metadata
from .decay import cosine_similarity

if TYPE_CHECKING:  # pragma: no cover
    from .store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / types
# ---------------------------------------------------------------------------

@dataclass
class CompactionConfig:
    """Knobs for `run_compaction`. All fields have working defaults."""
    min_age_days: int = 7
    # min_age_hours takes precedence when set (0 = compact immediately).
    # Provided for callers that need sub-day granularity.
    min_age_hours: int | None = None
    similarity_threshold: float = 0.88
    min_cluster_size: int = 2
    max_memories_to_scan: int = 200
    dry_run: bool = False
    cooldown_hours: int = 24


@dataclass
class CompactionEntry:
    """A row pulled from the store for compaction. Mirrors a MemoryStore
    row but flattened — vectors are required for similarity clustering."""
    id: str
    text: str
    vector: list[float]
    category: str
    scope: str
    importance: float
    timestamp: int
    metadata: str  # JSON-encoded; same convention as the underlying table


@dataclass
class ClusterMerged:
    text: str
    importance: float
    category: str
    scope: str
    metadata: str  # JSON-encoded merge marker


@dataclass
class ClusterPlan:
    member_indices: list[int]
    merged: ClusterMerged


@dataclass
class CompactionResult:
    scanned: int = 0
    clusters_found: int = 0
    memories_deleted: int = 0
    memories_created: int = 0
    dry_run: bool = False
    plans: list[ClusterPlan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def build_clusters(
    entries: Sequence[CompactionEntry],
    threshold: float,
    min_cluster_size: int,
    max_merged_chars: int | None = None,
) -> list[ClusterPlan]:
    """Greedy cluster expansion seeded by importance descending.

    The most important entry in the unassigned set seeds each cluster and
    pulls in any other unassigned entry with cosine similarity ≥ threshold.
    Only clusters of size ≥ `min_cluster_size` are returned."""
    n = len(entries)
    if n < min_cluster_size:
        return []

    # Sort indices by importance desc; ties broken by original order
    order = sorted(range(n), key=lambda i: (-entries[i].importance, i))
    assigned = [False] * n
    plans: list[ClusterPlan] = []

    for seed_idx in order:
        if assigned[seed_idx]:
            continue
        seed_vec = entries[seed_idx].vector
        if not seed_vec:
            continue

        cluster = [seed_idx]
        assigned[seed_idx] = True
        # Upper bound on the merged text (member texts joined by newlines).
        # A cluster that would not fit the store's text limit could never be
        # written and was rediscovered, and failed, on every run.
        merged_chars = len(entries[seed_idx].text)

        for j in range(n):
            if assigned[j]:
                continue
            other_vec = entries[j].vector
            if not other_vec:
                continue
            if cosine_similarity(seed_vec, other_vec) >= threshold:
                extra = len(entries[j].text) + 1
                if max_merged_chars is not None and merged_chars + extra > max_merged_chars:
                    continue  # left unassigned: it may seed or join another cluster
                merged_chars += extra
                cluster.append(j)
                assigned[j] = True

        if len(cluster) >= min_cluster_size:
            members = [entries[i] for i in cluster]
            plans.append(
                ClusterPlan(
                    member_indices=cluster,
                    merged=build_merged_entry(members),
                )
            )

    return plans


_TIER_RANK = {"core": 2, "working": 1, "peripheral": 0}


def _merge_cluster_provenance(
    members: Sequence[CompactionEntry],
) -> tuple[str, bool]:
    """Combine tier + cross_session across cluster members: pick the
    highest-rank tier (core > working > peripheral) and OR cross_session
    so a merged entry never silently downgrades existing provenance."""
    best_tier = "working"
    best_rank = -1
    any_cross_session = False
    for m in members:
        try:
            meta = json.loads(m.metadata) if isinstance(m.metadata, str) else (m.metadata or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            meta = {}
        member_tier = meta.get("tier", "working")
        r = _TIER_RANK.get(member_tier, 1)
        if r > best_rank:
            best_rank = r
            best_tier = member_tier
        if meta.get("cross_session"):
            any_cross_session = True
    return best_tier, any_cross_session


def _member_meta(member: CompactionEntry) -> dict:
    try:
        meta = json.loads(member.metadata) if isinstance(member.metadata, str) else member.metadata
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _as_int(value: object) -> int:
    try:
        return int(value or 0)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def build_merged_entry(members: Sequence[CompactionEntry]) -> ClusterMerged:
    """Merge a cluster into a single proposed entry.

    Text: dedupe lines case-insensitively, preserve first-seen order, join
    with newline. Importance: max (never downgrade). Category: plurality
    vote, ties broken by highest-importance member. Scope: first member's
    (caller is responsible for grouping by scope)."""
    seen: set[str] = set()
    lines: list[str] = []
    for m in members:
        for raw_line in m.text.split("\n"):
            trimmed = raw_line.strip()
            if not trimmed:
                continue
            key = trimmed.lower()
            if key not in seen:
                seen.add(key)
                lines.append(trimmed)
    text = "\n".join(lines)

    importance = min(1.0, max(m.importance for m in members))

    # Category: plurality vote, tiebreak by max-importance member's category
    counts: dict[str, int] = {}
    for m in members:
        counts[m.category] = counts.get(m.category, 0) + 1
    best_count = 0
    category = members[0].category
    for cat, c in counts.items():
        if c > best_count:
            best_count = c
            category = cat
    # Tiebreak: if multiple categories share best_count, prefer the most
    # important member's category
    tied = [cat for cat, c in counts.items() if c == best_count]
    if len(tied) > 1:
        top = max(members, key=lambda m: m.importance)
        if top.category in tied:
            category = top.category

    best_tier, any_cross_session = _merge_cluster_provenance(members)
    merged_meta: dict[str, object] = {
        "compacted": True,
        "source_count": len(members),
        "compacted_at": int(time.time() * 1000),
        "tier": best_tier,
        "cross_session": any_cross_session,
        "source": "compactor",
    }
    # Members are grouped by session before clustering, so they share one
    # source_session. Keep it: a merged row with none is hidden from every
    # session-filtered recall. Usage counters carry over as totals.
    metas = [_member_meta(m) for m in members]
    sessions = {str(meta.get("source_session") or "") for meta in metas}
    if len(sessions) == 1 and "" not in sessions:
        merged_meta["source_session"] = sessions.pop()
    for counter in ("access_count", "bad_recall_count"):
        total = sum(_as_int(meta.get(counter)) for meta in metas)
        if total:
            merged_meta[counter] = total
    metadata = json.dumps(merged_meta)

    return ClusterMerged(
        text=text,
        importance=importance,
        category=category,
        scope=members[0].scope,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_compaction(
    store: MemoryStore,
    config: CompactionConfig | None = None,
    scopes: Sequence[str] | None = None,
) -> CompactionResult:
    """Run a single compaction pass over memories in the given scopes.

    `scopes=None` scans all scopes. Set `config.dry_run=True` to report a
    plan without writing changes (useful for tuning the threshold)."""
    cfg = config or CompactionConfig()
    if cfg.min_age_hours is not None:
        cutoff_ms = int(time.time() * 1000) - cfg.min_age_hours * 3_600_000
    else:
        cutoff_ms = int(time.time() * 1000) - cfg.min_age_days * 86_400_000

    entries = _fetch_for_compaction(
        store, cutoff_ms, scopes, cfg.max_memories_to_scan
    )
    valid = [e for e in entries if e.vector]

    if not valid:
        return CompactionResult(scanned=0, clusters_found=0, dry_run=cfg.dry_run)

    # Cluster only within one scope AND one session. Clustering across them
    # moved user-scoped text into the agent scope (or the reverse) and
    # produced rows belonging to no session.
    from .store import MAX_TEXT_CHARS  # noqa: PLC0415 - store imports lancedb; keep this module light

    groups: dict[tuple[str, str], list[int]] = {}
    for idx, entry in enumerate(valid):
        session = str(_member_meta(entry).get("source_session") or "")
        groups.setdefault((entry.scope, session), []).append(idx)
    plans: list[ClusterPlan] = []
    for indices in groups.values():
        group_plans = build_clusters(
            [valid[i] for i in indices],
            cfg.similarity_threshold,
            cfg.min_cluster_size,
            max_merged_chars=MAX_TEXT_CHARS,
        )
        for plan in group_plans:
            plans.append(ClusterPlan(
                member_indices=[indices[i] for i in plan.member_indices],
                merged=plan.merged,
            ))

    if cfg.dry_run:
        logger.info(
            "memory-compactor [dry-run]: scanned=%d clusters=%d",
            len(valid), len(plans),
        )
        return CompactionResult(
            scanned=len(valid),
            clusters_found=len(plans),
            dry_run=True,
            plans=plans,
        )

    deleted = 0
    created = 0
    for plan in plans:
        members = [valid[i] for i in plan.member_indices]
        try:
            # Build the merged entry. We use store.store() which encodes its
            # own vector — we can't pass our pre-merged vector through the
            # public API, so the embedder runs once per cluster (acceptable
            # since clusters are rare and merge text is short).
            merged_meta = json.loads(plan.merged.metadata)
            store.store(
                text=plan.merged.text,
                category=plan.merged.category,
                scope=plan.merged.scope,
                importance=plan.merged.importance,
                metadata_extra=merged_meta,
            )
            created += 1
        except Exception as e:
            logger.warning(
                "memory-compactor: failed to write merged entry of %d members: %s",
                len(members), e,
            )
            continue

        for m in members:
            try:
                if store.forget(m.id):
                    deleted += 1
            except Exception as e:
                logger.warning(
                    "memory-compactor: failed to delete source %s: %s", m.id, e,
                )

    logger.info(
        "memory-compactor: scanned=%d clusters=%d deleted=%d created=%d",
        len(valid), len(plans), deleted, created,
    )
    return CompactionResult(
        scanned=len(valid),
        clusters_found=len(plans),
        memories_deleted=deleted,
        memories_created=created,
        dry_run=False,
        plans=plans,
    )


# Categories whose rows carry structure a text merge would destroy.
_NEVER_COMPACT_CATEGORIES = frozenset({"active_task", "reflection"})


def _mergeable(category: str, meta_raw: object) -> bool:
    """False for rows compaction must leave alone.

    - ``active_task`` pins and anchors: merging dropped ``state_path`` and
      ``task_id``, turning a task pin into a plain fact (or fusing two tasks).
    - rows with a ``valid_until``: the merged row has none, so an expired
      memory hidden from recall came back as a permanent one.
    - rows with a ``metadata_type`` (session summaries, reflection items):
      the type, and the decay exemption that goes with it, was lost.
    """
    if category in _NEVER_COMPACT_CATEGORIES:
        return False
    try:
        meta = _parse_metadata(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except Exception:
        return True
    if not isinstance(meta, dict):
        return True
    return not (
        meta.get("valid_until") or meta.get("metadata_type") or meta.get("state_path")
    )


def _fetch_for_compaction(
    store: MemoryStore,
    cutoff_ms: int,
    scopes: Sequence[str] | None,
    limit: int,
) -> list[CompactionEntry]:
    """Pull rows older than `cutoff_ms` (and not archived) with their
    vectors intact. Hitting `_scan_all` is cheap on stores up to
    MAX_SCAN_ROWS; very large stores should override this helper."""
    out: list[CompactionEntry] = []
    scope_set = set(scopes) if scopes else None
    for row in store._scan_all(limit=limit):
        ts = int(row.get("timestamp", 0) or 0)
        if ts > cutoff_ms:
            continue
        meta_raw = row.get("metadata", "")
        if _is_archived(meta_raw):
            continue
        if not _mergeable(row.get("category", ""), meta_raw):
            continue
        scope = row.get("scope", "")
        if scope_set is not None and scope not in scope_set:
            continue
        vec = row.get("vector")
        if not vec:
            continue
        # Make sure metadata is a JSON string (the table stores it that way;
        # _scan_all returns raw rows so it should already be a string, but
        # be defensive).
        if isinstance(meta_raw, dict):
            meta_str = json.dumps(meta_raw)
        else:
            meta_str = str(meta_raw or "{}")
            # Validate it parses; fall back to {}
            try:
                _parse_metadata(meta_str)
            except Exception:
                meta_str = "{}"
        out.append(CompactionEntry(
            id=row["id"],
            text=row["text"],
            vector=list(vec),
            category=row["category"],
            scope=scope,
            importance=float(row.get("importance", 0.0) or 0.0),
            timestamp=ts,
            metadata=meta_str,
        ))
    return out


# ---------------------------------------------------------------------------
# Cooldown helpers
# ---------------------------------------------------------------------------

def should_run_compaction(state_file: str, cooldown_hours: int) -> bool:
    """True if `cooldown_hours` have elapsed since the last recorded run.
    Missing / malformed state file → True (treat as "never run")."""
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return True
    last = state.get("last_run_at")
    if not isinstance(last, (int, float)):
        return True
    elapsed_ms = int(time.time() * 1000) - int(last)
    return elapsed_ms >= cooldown_hours * 60 * 60 * 1000


def record_compaction_run(state_file: str) -> None:
    """Persist a `last_run_at` marker for `should_run_compaction`."""
    # os.path.dirname returns "" for bare filenames; makedirs("") raises.
    if parent := os.path.dirname(state_file):
        os.makedirs(parent, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump({"last_run_at": int(time.time() * 1000)}, f)


__all__ = [
    "ClusterMerged",
    "ClusterPlan",
    "CompactionConfig",
    "CompactionEntry",
    "CompactionResult",
    "build_clusters",
    "build_merged_entry",
    "record_compaction_run",
    "run_compaction",
    "should_run_compaction",
]
