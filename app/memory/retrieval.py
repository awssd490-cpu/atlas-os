"""Retrieval pipeline — composable, multi-stage memory retrieval.

Architecture
============

Every retrieval request flows through a series of stages.  Each stage
receives a ``RetrievalContext`` and returns a (possibly modified)
context.  Stages are independent, testable, and composable.

Phases
------

1. **Execution** — ``SearchExecutor`` fetches matching memories from the
   repository.  This is the only stage that touches the database.

2. **Filter** — zero or more stages remove candidates that don't match
   (namespace, type, state, tag, source, owner, content, temporal,
   importance).  These run on the in-memory candidate list.

3. **Enrich** — ``RelationshipExpander`` traverses the graph to find
   related memories not captured by the initial query.

4. **Rank** — ``ImportanceRanker`` scores each candidate by importance,
   recency, and frequency; ``TopKTruncation`` keeps the top N.

5. **Deduplicate** — ``Deduplicator`` removes any duplicate IDs
   introduced by expansion.

ADR-014: https://github.com/atlas/memory/adr/retrieval-pipeline-stages
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Protocol

from app.memory.interfaces import MemoryGraph, MemoryQuery
from app.memory.memory import Memory, MemoryState
from app.storage.interfaces import PaginationParams


# ---------------------------------------------------------------------------
# Query & context types
# ---------------------------------------------------------------------------


class RetrievalQuery:
    """Input parameters for a retrieval pipeline invocation.

    Every field is optional.  An empty query returns all active memories
    up to *limit*.
    """

    def __init__(
        self,
        *,
        memory_types: list[str] | None = None,
        namespaces: list[str] | None = None,
        states: list[MemoryState] | None = None,
        tags: list[str] | None = None,
        content_search: str | None = None,
        sources: list[str] | None = None,
        owners: list[str] | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        correlation_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        accessed_after: str | None = None,
        expand_relationships: bool = False,
        relationship_max_depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> None:
        self.memory_types = memory_types
        self.namespaces = namespaces
        self.states = states
        self.tags = tags
        self.content_search = content_search
        self.sources = sources
        self.owners = owners
        self.min_importance = min_importance
        self.max_importance = max_importance
        self.correlation_id = correlation_id
        self.created_after = created_after
        self.created_before = created_before
        self.accessed_after = accessed_after
        self.expand_relationships = expand_relationships
        self.relationship_max_depth = relationship_max_depth
        self.relationship_types = relationship_types
        self.limit = limit


class RetrievalContext:
    """Mutable pipeline state flowing through all stages.

    Attributes:
        query: The original query (immutable after pipeline start).
        candidates: In-memory candidate list — stages add, remove,
            reorder, or score items in place.
        aborted: When set, the pipeline halts after the current stage.
        metadata: Accumulated telemetry and stage-specific data.
        scores: Per-memory-ID score assigned by the ranker stage.
    """

    def __init__(self, query: RetrievalQuery) -> None:
        self.query = query
        self.candidates: list[Memory] = []
        self.aborted = False
        self.metadata: dict[str, Any] = {}
        self.scores: dict[str, float] = {}

    def add_stage_meta(self, stage: str, key: str, value: Any) -> None:
        """Record a metadata value under *stage*."""
        if stage not in self.metadata:
            self.metadata[stage] = {}
        self.metadata[stage][key] = value


class RetrievalResult:
    """Final output of a pipeline invocation."""

    def __init__(
        self,
        memories: list[Memory],
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.memories = memories
        self.total = total
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "total": self.total,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Stage protocol
# ---------------------------------------------------------------------------


class RetrievalStage(Protocol):
    """A single stage in the retrieval pipeline.

    Receives a ``RetrievalContext`` and returns it (possibly modified).
    Stages are **async callables** — implement ``__call__`` or use a
    plain async function with the right signature.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        """Process the context and return it."""
        ...


# ---------------------------------------------------------------------------
# Pipeline executor
# ---------------------------------------------------------------------------


class RetrievalPipeline:
    """Orchestrates a sequence of ``RetrievalStage`` calls.

    Stages are run in order.  If any stage sets ``ctx.aborted = True``
    the pipeline halts immediately.
    """

    def __init__(self, stages: list[RetrievalStage]) -> None:
        assert len(stages) > 0, "pipeline must have at least one stage"
        self._stages = list(stages)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run the query through every stage and return the result."""
        ctx = RetrievalContext(query=query)

        for i, stage in enumerate(self._stages):
            stage_name = getattr(stage, "__class__", type(stage)).__name__
            if isinstance(stage_name, type):
                stage_name = stage_name.__name__
            if hasattr(stage, "__class__"):
                stage_name = type(stage).__name__

            t0 = time.monotonic()
            ctx = await stage(ctx)
            elapsed = (time.monotonic() - t0) * 1000

            ctx.add_stage_meta(stage_name, "elapsed_ms", round(elapsed, 2))
            ctx.add_stage_meta(
                stage_name,
                "candidates_after",
                len(ctx.candidates),
            )

            if ctx.aborted:
                ctx.add_stage_meta("pipeline", "aborted_at", stage_name)
                break

        return RetrievalResult(
            memories=ctx.candidates,
            total=len(ctx.candidates),
            metadata=ctx.metadata,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _query_to_memory_query(query: RetrievalQuery) -> MemoryQuery:
    """Convert a ``RetrievalQuery`` to a ``MemoryQuery`` for repo search."""
    return MemoryQuery(
        memory_types=query.memory_types,
        namespaces=query.namespaces,
        states=query.states,
        tags=query.tags,
        content_search=query.content_search,
        sources=query.sources,
        owners=query.owners,
        min_importance=query.min_importance,
        max_importance=query.max_importance,
        correlation_id=query.correlation_id,
        created_after=query.created_after,
        created_before=query.created_before,
        accessed_after=query.accessed_after,
    )


# ===================================================================
# Pipeline stages
# ===================================================================


# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------


class SearchExecutor:
    """Fetch matching memories from the repository.

    Converts the ``RetrievalQuery`` to a ``MemoryQuery`` and runs it
    against ``MemoryRepository.search()``.  Populates ``ctx.candidates``.

    This is typically the **first** stage in the pipeline.
    """

    def __init__(self, repository: Any) -> None:  # MemoryRepository
        self._repo = repository

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        mq = _query_to_memory_query(ctx.query)
        pagination = PaginationParams(limit=ctx.query.limit or 100)

        # Default to active-only if no state filter was provided
        if not ctx.query.states:
            mq.states = [MemoryState.ACTIVE]

        memories = await self._repo.search(mq, limit=pagination.limit, offset=pagination.offset)
        ctx.candidates = memories
        ctx.add_stage_meta("SearchExecutor", "query_matched", len(memories))
        return ctx


# -------------------------------------------------------------------
# Filters
# -------------------------------------------------------------------


class NamespaceFilter:
    """Filter candidates to one or more namespaces.

    Only active when ``ctx.query.namespaces`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        namespaces = ctx.query.namespaces
        if not namespaces or not ctx.candidates:
            return ctx

        ns_set = set(namespaces)
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if m.namespace in ns_set]
        ctx.add_stage_meta(
            "NamespaceFilter",
            "removed",
            before - len(ctx.candidates),
        )
        return ctx


class TypeFilter:
    """Filter candidates to one or more memory types.

    Only active when ``ctx.query.memory_types`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        types = ctx.query.memory_types
        if not types or not ctx.candidates:
            return ctx

        type_set = set(types)
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if m.memory_type in type_set]
        ctx.add_stage_meta("TypeFilter", "removed", before - len(ctx.candidates))
        return ctx


class StateFilter:
    """Filter candidates to one or more lifecycle states.

    Only active when ``ctx.query.states`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        states = ctx.query.states
        if not states or not ctx.candidates:
            return ctx

        state_set = {s.value if isinstance(s, MemoryState) else s for s in states}
        before = len(ctx.candidates)
        ctx.candidates = [
            m for m in ctx.candidates if (m.state.value if isinstance(m.state, MemoryState) else m.state) in state_set
        ]
        ctx.add_stage_meta("StateFilter", "removed", before - len(ctx.candidates))
        return ctx


class TagFilter:
    """Filter candidates by tags.

    Only active when ``ctx.query.tags`` is set.  A candidate matches if
    it has **any** of the requested tags.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        tags = ctx.query.tags
        if not tags or not ctx.candidates:
            return ctx

        tag_set = set(tags)
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if tag_set & set(m.tags)]
        ctx.add_stage_meta("TagFilter", "removed", before - len(ctx.candidates))
        return ctx


class SourceFilter:
    """Filter candidates by source.

    Only active when ``ctx.query.sources`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        sources = ctx.query.sources
        if not sources or not ctx.candidates:
            return ctx

        src_set = set(sources)
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if m.source in src_set]
        ctx.add_stage_meta("SourceFilter", "removed", before - len(ctx.candidates))
        return ctx


class OwnerFilter:
    """Filter candidates by owner.

    Only active when ``ctx.query.owners`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        owners = ctx.query.owners
        if not owners or not ctx.candidates:
            return ctx

        owner_set = set(owners)
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if m.owner in owner_set]
        ctx.add_stage_meta("OwnerFilter", "removed", before - len(ctx.candidates))
        return ctx


class ContentSearch:
    """Filter candidates by content substring match (case-insensitive).

    Only active when ``ctx.query.content_search`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        needle = ctx.query.content_search
        if not needle or not ctx.candidates:
            return ctx

        needle_lower = needle.lower()
        before = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if needle_lower in m.content.lower()]
        ctx.add_stage_meta("ContentSearch", "removed", before - len(ctx.candidates))
        return ctx


class TemporalFilter:
    """Filter candidates by creation-time range.

    Only active when ``ctx.query.created_after`` or
    ``ctx.query.created_before`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.candidates:
            return ctx
        after = ctx.query.created_after
        before = ctx.query.created_before
        if not after and not before:
            return ctx

        def _in_range(m: Memory) -> bool:
            if m.created_at is None:
                return False
            ts = m.created_at.isoformat()
            if after and ts < after:
                return False
            if before and ts > before:
                return False
            return True

        orig = len(ctx.candidates)
        ctx.candidates = [m for m in ctx.candidates if _in_range(m)]
        ctx.add_stage_meta("TemporalFilter", "removed", orig - len(ctx.candidates))
        return ctx


class ImportanceFilter:
    """Filter candidates by importance range.

    Only active when ``ctx.query.min_importance`` or
    ``ctx.query.max_importance`` is set.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.candidates:
            return ctx
        lo = ctx.query.min_importance
        hi = ctx.query.max_importance
        if lo is None and hi is None:
            return ctx

        before = len(ctx.candidates)
        ctx.candidates = [
            m
            for m in ctx.candidates
            if (lo is None or m.importance >= lo) and (hi is None or m.importance <= hi)
        ]
        ctx.add_stage_meta("ImportanceFilter", "removed", before - len(ctx.candidates))
        return ctx


# -------------------------------------------------------------------
# Ranking
# -------------------------------------------------------------------


class ImportanceRanker:
    """Score and sort candidates by importance, recency, and frequency.

    The composite score is::

        score = importance * recency_multiplier * frequency_multiplier

    Each multiplier is drawn from the policy engine's
    ``ImportanceScorer``.  Scores are stored in ``ctx.scores`` and the
    candidate list is sorted descending.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.candidates:
            return ctx

        scores: dict[str, float] = {}
        for mem in ctx.candidates:
            base = mem.importance
            recency = self._recency_score(mem)
            frequency = self._frequency_score(mem)
            score = base * recency * frequency
            score = min(1.0, max(0.0, score))
            scores[mem.id.value] = score

        ctx.scores = scores
        ctx.candidates.sort(key=lambda m: scores.get(m.id.value, 0.0), reverse=True)
        ctx.add_stage_meta("ImportanceRanker", "top_score", max(scores.values()) if scores else 0.0)
        ctx.add_stage_meta("ImportanceRanker", "bottom_score", min(scores.values()) if scores else 0.0)
        return ctx

    @staticmethod
    def _recency_score(memory: Memory) -> float:
        """Score recency — recently accessed memories score higher.

        Returns a multiplier in [1.0, 1.5].
        """
        if memory.accessed_at is None:
            return 1.0
        age_hours = (_now_utc() - memory.accessed_at).total_seconds() / 3600.0
        if age_hours < 1:
            return 1.5
        if age_hours < 24:
            return 1.3
        if age_hours < 168:
            return 1.1
        return 1.0

    @staticmethod
    def _frequency_score(memory: Memory) -> float:
        """Score access frequency — often-accessed memories score higher.

        Returns a multiplier in [1.0, 1.3].
        """
        if memory.access_count == 0:
            return 1.0
        if memory.access_count > 50:
            return 1.3
        if memory.access_count > 10:
            return 1.15
        return 1.05


# -------------------------------------------------------------------
# Enrichment
# -------------------------------------------------------------------


class RelationshipExpander:
    """Traverse the graph and append related memories.

    Uses the ``MemoryGraph`` to find neighbours of each candidate and
    appends any that aren't already in the candidate list.

    Only active when ``ctx.query.expand_relationships`` is ``True``.
    """

    def __init__(self, graph: MemoryGraph | None = None) -> None:
        self._graph = graph

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.query.expand_relationships or self._graph is None or not ctx.candidates:
            return ctx

        depth = ctx.query.relationship_max_depth or 1
        rel_types = ctx.query.relationship_types

        existing_ids: set[str] = {m.id.value for m in ctx.candidates}
        added: list[Memory] = []

        for mem in ctx.candidates:
            related = await self._graph.get_related(
                mem.id.value,
                rel_type=None,  # all relationship types
                direction="both",
                max_depth=depth,
            )
            for related_mem in related:
                rid = related_mem.id.value
                if rid not in existing_ids:
                    existing_ids.add(rid)
                    added.append(related_mem)

        ctx.candidates.extend(added)
        ctx.add_stage_meta("RelationshipExpander", "added", len(added))
        return ctx


# -------------------------------------------------------------------
# Deduplication
# -------------------------------------------------------------------


class Deduplicator:
    """Remove duplicate memories by ID, keeping the first occurrence.

    Useful after ``RelationshipExpander`` which may yield memories
    already in the candidate list.
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        if not ctx.candidates:
            return ctx

        seen: set[str] = set()
        deduped: list[Memory] = []
        for mem in ctx.candidates:
            mid = mem.id.value
            if mid not in seen:
                seen.add(mid)
                deduped.append(mem)

        before = len(ctx.candidates)
        ctx.candidates = deduped
        ctx.add_stage_meta("Deduplicator", "removed", before - len(ctx.candidates))
        return ctx


# -------------------------------------------------------------------
# Truncation
# -------------------------------------------------------------------


class TopKTruncation:
    """Keep only the top *k* candidates.

    The *k* comes from ``ctx.query.limit``.  When a ranker has run
    before this stage, the top-k are the *k* highest-scoring memories.

    When no ranker has run, the candidates are truncated by their
    current list order (which defaults to DB insertion order, so a
    ranker should generally precede this stage).
    """

    async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
        k = ctx.query.limit or 100
        if len(ctx.candidates) <= k:
            ctx.add_stage_meta("TopKTruncation", "truncated", 0)
            return ctx

        ctx.add_stage_meta("TopKTruncation", "truncated", len(ctx.candidates) - k)
        ctx.candidates = ctx.candidates[:k]
        return ctx


# -------------------------------------------------------------------
# Default pipeline assembly
# -------------------------------------------------------------------


def default_pipeline(
    repository: Any,
    graph: MemoryGraph | None = None,
) -> RetrievalPipeline:
    """Build the default retrieval pipeline.

    Stage order::

        1. SearchExecutor
        2. NamespaceFilter
        3. TypeFilter
        4. StateFilter
        5. TagFilter
        6. SourceFilter
        7. OwnerFilter
        8. ContentSearch
        9. TemporalFilter
        10. ImportanceFilter
        11. ImportanceRanker
        12. RelationshipExpander
        13. Deduplicator
        14. TopKTruncation
    """
    return RetrievalPipeline([
        SearchExecutor(repository),
        NamespaceFilter(),
        TypeFilter(),
        StateFilter(),
        TagFilter(),
        SourceFilter(),
        OwnerFilter(),
        ContentSearch(),
        TemporalFilter(),
        ImportanceFilter(),
        ImportanceRanker(),
        RelationshipExpander(graph),
        Deduplicator(),
        TopKTruncation(),
    ])
