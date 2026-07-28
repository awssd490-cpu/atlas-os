"""Memory compression — reduce memory growth while preserving information.

Architecture
============

Compression operates in four phases:

1. **Selection** — ``CompressionPolicy`` decides which memories to
   compress, how many, and what strategies to use.

2. **Strategy** — One or more ``CompressionStrategy`` implementations
   process the selected memories.  Each strategy is a standalone async
   callable.

3. **Execution** — ``CompressionService`` orchestrates selection,
   pre-snapshot, strategy execution, persistence of results, and
   cleanup of originals.

4. **Event** — ``MemoriesCompressed`` is published after each run.

Strategies
----------

All strategies are deterministic (no LLM dependency).  They operate on
in-memory lists of ``Memory`` objects and return ``StrategyResult``.

- ``DedupStrategy`` — removes exact content duplicates (same content,
  same namespace), keeping the highest-importance copy.
- ``MergeRelatedStrategy`` — combines related memories (via the
  ``MemoryGraph``) that share a relationship type and namespace into a
  single merged Memory, preserving provenance.
- ``TruncationStrategy`` — truncates memory content to a maximum length
  for stale or low-importance memories.
- ``ArchiveLowValueStrategy`` — transitions memories below an importance
  threshold to ``ARCHIVED`` state (complementary to GC).

Adding a strategy
-----------------

Subclass ``CompressionStrategy`` and implement ``__call__``.  Register
it in the policy's strategy map.
"""

from __future__ import annotations

import abc
import time
from typing import Any

from app.core.interfaces import EventBus, Logger, TelemetryService
from app.memory.events import MemoriesCompressed
from app.memory.interfaces import (
    CompressionResult,
    MemoryCompressor,
    MemoryGraph,
    MemoryQuery,
)
from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.manager import MemoryRepository
from app.memory.snapshots import SnapshotRepository, SnapshotService
from app.storage.interfaces import FilterCondition, FilterOperator, SQLConnection


# ---------------------------------------------------------------------------
# Strategy result
# ---------------------------------------------------------------------------


class StrategyResult:
    """Result of a single compression strategy."""

    __slots__ = ("kept", "removed_ids", "strategy_name")

    def __init__(
        self,
        *,
        kept: list[Memory],
        removed_ids: list[str],
        strategy_name: str = "",
    ) -> None:
        self.kept = kept
        self.removed_ids = removed_ids
        self.strategy_name = strategy_name

    @property
    def removed_count(self) -> int:
        return len(self.removed_ids)


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class CompressionStrategy(abc.ABC):
    """A single compression strategy.

    Implement ``__call__`` which receives a list of candidates and
    returns a ``StrategyResult``.
    """

    @abc.abstractmethod
    async def __call__(
        self,
        memories: list[Memory],
        *,
        graph: MemoryGraph | None = None,
    ) -> StrategyResult:
        ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class DedupStrategy(CompressionStrategy):
    """Remove exact content duplicates within the same namespace.

    Keeps the highest-importance copy.  All other duplicates are
    reported as removed.
    """

    async def __call__(
        self,
        memories: list[Memory],
        *,
        graph: MemoryGraph | None = None,
    ) -> StrategyResult:
        seen: dict[tuple[str, str], Memory] = {}  # (namespace, content) -> best
        removed: list[str] = []

        for mem in memories:
            key = (mem.namespace, mem.content)
            existing = seen.get(key)
            if existing is None:
                seen[key] = mem
            else:
                # Keep the one with higher importance
                if mem.importance > existing.importance:
                    seen[key] = mem
                    removed.append(existing.id.value)
                else:
                    removed.append(mem.id.value)

        return StrategyResult(
            kept=list(seen.values()),
            removed_ids=removed,
            strategy_name="dedup",
        )


class MergeRelatedStrategy(CompressionStrategy):
    """Combine related memories into a single merged memory.

    Groups memories by ``(namespace, relationship_type)`` via the
    ``MemoryGraph``.  Each group is merged into one memory that:

    - Concatenates content with separators
    - Takes the highest importance, max confidence
    - Merges tags (deduplicated)
    - Preserves provenance (original IDs) in metadata

    Only merges groups with 2+ memories.
    """

    async def __call__(
        self,
        memories: list[Memory],
        *,
        graph: MemoryGraph | None = None,
    ) -> StrategyResult:
        if graph is None or not memories:
            return StrategyResult(
                kept=memories,
                removed_ids=[],
                strategy_name="merge_related",
            )

        mem_by_id: dict[str, Memory] = {m.id.value: m for m in memories}
        processed: set[str] = set()
        merged: list[Memory] = []
        removed: list[str] = []
        seen_ids: set[str] = set()

        for mem in memories:
            mid = mem.id.value
            if mid in processed or mid in seen_ids:
                continue

            # Find related memories of the same namespace
            related = await graph.get_related(mid, direction="both", max_depth=1)
            group_ids = [mid]
            for rel in related:
                rid = rel.id.value
                if rid in mem_by_id and rid not in processed and rid not in seen_ids:
                    # Only merge if same namespace
                    if rel.namespace == mem.namespace:
                        group_ids.append(rid)

            if len(group_ids) < 2:
                merged.append(mem)
                processed.add(mid)
                continue

            # Merge the group
            group_mems = [mem_by_id[gid] for gid in group_ids if gid in mem_by_id]
            merged_mem = self._merge_group(group_mems)
            merged.append(merged_mem)
            processed.add(mid)
            seen_ids.update(gid for gid in group_ids if gid != mid)

        # Add memories that were not part of any merge
        for mem in memories:
            if mem.id.value not in processed and mem.id.value not in seen_ids:
                if mem.id.value not in {m.id.value for m in merged}:
                    merged.append(mem)

        removed = [m.id.value for m in memories if m.id.value not in {mm.id.value for mm in merged}]

        return StrategyResult(
            kept=merged,
            removed_ids=removed,
            strategy_name="merge_related",
        )

    @staticmethod
    def _merge_group(memories: list[Memory]) -> Memory:
        """Merge a list of memories into one."""
        best = max(memories, key=lambda m: m.importance)
        all_tags: list[str] = []
        content_parts: list[str] = []
        max_confidence = 0.0
        provenance: list[str] = []

        for m in memories:
            if m.content:
                content_parts.append(m.content)
            all_tags.extend(m.tags)
            max_confidence = max(max_confidence, m.confidence)
            provenance.append(m.id.value)

        merged = Memory(
            memory_type=best.memory_type,
            namespace=best.namespace,
            content="\n---\n".join(content_parts) if content_parts else best.content,
            importance=best.importance,
            confidence=max_confidence,
            source=best.source,
            owner=best.owner,
            tags=list(set(all_tags)),
            metadata={
                "compressed": True,
                "compression_strategy": "merge_related",
                "provenance": provenance,
            },
        )
        return merged


class TruncationStrategy(CompressionStrategy):
    """Truncate memory content to a maximum length.

    Memories whose content exceeds *max_length* characters are truncated
    and an indicator is appended.

    This strategy returns a result where **all** memories are "kept"
    (the truncated versions), and none are removed.  The caller decides
    whether to persist the truncated versions.
    """

    def __init__(self, max_length: int = 500) -> None:
        self._max_length = max_length

    async def __call__(
        self,
        memories: list[Memory],
        *,
        graph: MemoryGraph | None = None,
    ) -> StrategyResult:
        kept: list[Memory] = []
        for mem in memories:
            if len(mem.content) > self._max_length:
                mem.content = mem.content[:self._max_length] + "... [truncated]"
                mem.metadata["compressed"] = True
                mem.metadata["compression_strategy"] = "truncate"
            kept.append(mem)

        return StrategyResult(
            kept=kept,
            removed_ids=[],
            strategy_name="truncate",
        )


class ArchiveLowValueStrategy(CompressionStrategy):
    """Transition low-importance memories to ARCHIVED state.

    Uses ``MemoryRepository.transition_state`` via the repository
    directly.  These are removed from active retrieval but preserved.
    """

    def __init__(
        self,
        archive_threshold: float = 0.3,
        max_age_days: float = 30.0,
    ) -> None:
        self._archive_threshold = archive_threshold
        self._max_age_days = max_age_days

    async def __call__(
        self,
        memories: list[Memory],
        *,
        graph: MemoryGraph | None = None,
    ) -> StrategyResult:
        removed: list[str] = []
        kept: list[Memory] = []

        for mem in memories:
            if mem.state != MemoryState.ACTIVE:
                kept.append(mem)
                continue
            if mem.importance >= self._archive_threshold:
                kept.append(mem)
                continue

            age_days = mem.age_seconds / 86400.0
            if age_days < self._max_age_days:
                kept.append(mem)
                continue

            # Archive this memory
            mem.transition_to(MemoryState.ARCHIVED)
            removed.append(mem.id.value)
            # Kept as archived — not deleted
            kept.append(mem)

        return StrategyResult(
            kept=kept,
            removed_ids=removed,
            strategy_name="archive_low_value",
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class CompressionPolicy:
    """Configuration for a compression run.

    Controls which strategies run, their thresholds, and the target
    compression ratio.
    """

    def __init__(
        self,
        *,
        enabled_strategies: list[str] | None = None,
        max_memories_per_run: int = 500,
        archive_threshold: float = 0.3,
        max_age_days: float = 30.0,
        content_truncation_length: int = 500,
        take_snapshot_before: bool = True,
        namespace_filters: list[str] | None = None,
        type_filters: list[str] | None = None,
    ) -> None:
        self.enabled_strategies = enabled_strategies or [
            "dedup",
            "truncate",
            "archive_low_value",
        ]
        self.max_memories_per_run = max_memories_per_run
        self.archive_threshold = archive_threshold
        self.max_age_days = max_age_days
        self.content_truncation_length = content_truncation_length
        self.take_snapshot_before = take_snapshot_before
        self.namespace_filters = namespace_filters
        self.type_filters = type_filters


# ---------------------------------------------------------------------------
# Compressor implementation (implements MemoryCompressor ABC)
# ---------------------------------------------------------------------------


class MemoryCompressorImpl(MemoryCompressor):
    """Core compressor — applies a strategy to a list of memories.

    This is the low-level interface from ``interfaces.py``.  It takes a
    pre-selected list and returns a ``CompressionResult``.
    """

    _strategy_registry: dict[str, type[CompressionStrategy]] = {
        "dedup": DedupStrategy,
        "merge_related": MergeRelatedStrategy,
        "truncate": TruncationStrategy,
        "archive_low_value": ArchiveLowValueStrategy,
    }

    def __init__(self, graph: MemoryGraph | None = None) -> None:
        self._graph = graph

    async def compress(
        self,
        memories: list[Memory],
        *,
        target_count: int | None = None,
        strategy: str = "dedup",
    ) -> CompressionResult:
        strategy_cls = self._strategy_registry.get(strategy)
        if strategy_cls is None:
            raise ValueError(f"Unknown compression strategy: {strategy!r}")

        if strategy == "truncate":
            strat = strategy_cls(max_length=500)
        elif strategy == "archive_low_value":
            strat = strategy_cls(archive_threshold=0.3, max_age_days=30.0)
        else:
            strat = strategy_cls()

        original_count = len(memories)
        result = await strat(memories, graph=self._graph)

        if target_count is not None and len(result.kept) > target_count:
            # Sort by importance descending and keep top-k
            result.kept.sort(key=lambda m: m.importance, reverse=True)
            removed_extra = [m.id.value for m in result.kept[target_count:]]
            result.kept = result.kept[:target_count]
            result.removed_ids.extend(removed_extra)

        return CompressionResult(
            compressed=result.kept,
            original_count=original_count,
            compressed_count=len(result.kept),
            strategy=strategy,
        )

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: type[CompressionStrategy]) -> None:
        """Register a custom strategy for use by name."""
        cls._strategy_registry[name] = strategy_cls


# ---------------------------------------------------------------------------
# CompressionService (high-level orchestration)
# ---------------------------------------------------------------------------


class CompressionService:
    """High-level compression orchestrator.

    Integrates candidate selection, pre-snapshot, strategy execution,
    persistence, and event emission.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        connection: SQLConnection,
        graph: MemoryGraph | None = None,
        snapshot_service: SnapshotService | None = None,
        event_bus: EventBus | None = None,
        telemetry: TelemetryService | None = None,
        logger: Logger | None = None,
        policy: CompressionPolicy | None = None,
    ) -> None:
        self._repo = repository
        self._conn = connection
        self._graph = graph
        self._snapshot_service = snapshot_service
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._logger = logger
        self._policy = policy or CompressionPolicy()

        self._compressor = MemoryCompressorImpl(graph=graph)

    async def run(self) -> CompressionResult:
        """Run a full compression cycle.

        1. Select candidates (policy-driven)
        2. Optionally snapshot before destructive operations
        3. Execute each enabled strategy sequentially
        4. Persist compressed memories (update kept, delete removed)
        5. Emit ``MemoriesCompressed``
        """
        start = time.monotonic()
        policy = self._policy

        # 1. Select candidates
        candidates = await self._select_candidates()
        if not candidates:
            if self._logger:
                self._logger.info("Compression skipped — no candidates")
            return CompressionResult(compressed=[], original_count=0, compressed_count=0, strategy="none")

        original_count = len(candidates)

        # 2. Pre-snapshot
        snapshot_id = ""
        if policy.take_snapshot_before and self._snapshot_service is not None:
            snap = await self._snapshot_service.create_snapshot(
                label=f"pre-compression-{original_count}mem",
            )
            snapshot_id = snap.snapshot_id
            if self._logger:
                self._logger.info(
                    "Pre-compression snapshot created | id={sid} memories={n}",
                    sid=snapshot_id,
                    n=original_count,
                )

        # 3. Execute strategies sequentially
        working_set: list[Memory] = candidates
        final_strategy = "none"

        for strategy_name in policy.enabled_strategies:
            if not working_set:
                break
            try:
                result = await self._compressor.compress(
                    working_set,
                    strategy=strategy_name,
                )
                working_set = result.compressed
                final_strategy = strategy_name
                if self._logger:
                    self._logger.info(
                        "Compression strategy={s} | before={b} after={a} ratio={r}",
                        s=strategy_name,
                        b=result.original_count,
                        a=result.compressed_count,
                        r=round(result.ratio, 3),
                    )
                if result.removed_ids:
                    await self._persist_removed(result.removed_ids)
            except Exception as exc:
                if self._logger:
                    self._logger.error(
                        "Compression strategy {s} failed | error={e}",
                        s=strategy_name,
                        e=str(exc),
                    )

        compressed_count = len(working_set)

        # 4. Persist compressed result
        await self._persist_working_set(working_set)

        elapsed = (time.monotonic() - start) * 1000
        ratio = compressed_count / original_count if original_count > 0 else 1.0

        # 5. Emit event
        await self._emit(
            MemoriesCompressed(
                original_count=original_count,
                compressed_count=compressed_count,
                strategy=final_strategy,
                ratio=ratio,
                snapshot_id=snapshot_id,
            ),
        )

        if self._telemetry:
            self._telemetry.record_module_lifecycle(
                "memory", "compression_run", elapsed, True,
            )

        if self._logger:
            self._logger.info(
                "Compression complete | original={orig} compressed={comp} strategy={strat} ratio={r} elapsed_ms={ms}",
                orig=original_count,
                comp=compressed_count,
                strat=final_strategy,
                r=round(ratio, 3),
                ms=round(elapsed, 2),
            )

        return CompressionResult(
            compressed=working_set,
            original_count=original_count,
            compressed_count=compressed_count,
            strategy=final_strategy,
        )

    async def compress_memories(
        self,
        memories: list[Memory],
        *,
        strategy: str = "dedup",
        target_count: int | None = None,
    ) -> CompressionResult:
        """Direct compression of a specific memory list (no selection).

        Useful for compressing a subset of memories without running
        the full selection pipeline.
        """
        return await self._compressor.compress(
            memories,
            target_count=target_count,
            strategy=strategy,
        )

    async def compress(
        self,
        memories: list[Memory],
        *,
        target_count: int | None = None,
        strategy: str = "dedup",
    ) -> CompressionResult:
        """Implement ``MemoryCompressor.compress``.

        Delegates to the inner ``MemoryCompressorImpl``.
        """
        return await self._compressor.compress(
            memories,
            target_count=target_count,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _select_candidates(self) -> list[Memory]:
        """Select candidate memories via the repository."""
        query = MemoryQuery(states=[MemoryState.ACTIVE])
        if self._policy.namespace_filters:
            query.namespaces = self._policy.namespace_filters
        if self._policy.type_filters:
            query.memory_types = self._policy.type_filters

        candidates = await self._repo.search(query, limit=self._policy.max_memories_per_run)
        return candidates

    async def _persist_removed(self, removed_ids: list[str]) -> None:
        """Hard-delete removed memories from the database."""
        for mid in removed_ids:
            await self._repo.delete(MemoryId(mid))

    async def _persist_working_set(self, working_set: list[Memory]) -> None:
        """Update all memories in the working set.

        For each memory in *working_set*, if it has a matching row in
        the DB, update it.  If it's a new merged memory (no existing
        row), insert it.

        Then delete any kept memories that were NOT in the working set
        (they were deduplicated/merged into others).  Actually, the
        removed ones are already handled by ``_persist_removed``.
        Memories that still exist but are part of the compressed set
        get updated.
        """
        for mem in working_set:
            existing = await self._repo.get(mem.id)
            if existing is not None:
                await self._repo.update(mem)
            else:
                # Newly created memory (e.g. from merge) — add it
                await self._repo.add(mem)

    async def _emit(self, event: Any) -> None:
        if self._event_bus:
            await self._event_bus.publish(event)
