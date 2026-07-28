"""MemoryManager — the primary facade for the memory system.

All memory operations go through this class.  It wraps the repository
with caching, event emission, telemetry, and policy enforcement.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from app.core.interfaces import EventBus
from app.core.interfaces import Logger, TelemetryService
from app.memory.events import (
    MemoryAccessed,
    MemoryCreated,
    MemoryStateChanged,
    MemoryUpdated,
)
from app.memory.interfaces import (
    GCResult,
    MemoryGarbageCollector,
    MemoryGraph,
    MemoryQuery,
    MemorySearchService,
    MemoryService,
    MemorySnapshot,
    MemorySnapshotService,
    Page,
    PaginationParams,
    SortField,
)
from app.memory.memory import Memory, MemoryId, MemoryState
from app.memory.policies import (
    DEFAULT_TYPE_POLICIES,
    ImportanceScorer,
    MemoryTypePolicy,
    RetentionPolicy,
)
from app.memory.snapshots import SnapshotService
from app.storage.interfaces import (
    CacheService,
    FilterCondition,
    FilterOperator,
    SQLConnection,
)


class MemoryRepository:
    """SQLite-backed repository for Memory objects.

    Handles all SQL operations for the memories table.  No business logic.
    """

    def __init__(self, connection: SQLConnection) -> None:
        self._conn = connection

    async def add(self, memory: Memory) -> Memory:
        row = memory.to_dict()
        columns = ", ".join(row.keys())
        placeholders = ", ".join([f":{k}" for k in row])
        sql = f"INSERT INTO memories ({columns}) VALUES ({placeholders})"
        await self._conn.execute(sql, row)
        return memory

    async def get(self, memory_id: MemoryId) -> Memory | None:
        row = await self._conn.fetchone(
            "SELECT * FROM memories WHERE id = :id",
            {"id": memory_id.value},
        )
        return Memory.from_dict(row) if row else None

    async def update(self, memory: Memory) -> Memory:
        row = memory.to_dict()
        set_clause = ", ".join([f"{k} = :{k}" for k in row if k != "id"])
        sql = f"UPDATE memories SET {set_clause} WHERE id = :id"
        row["id"] = memory.id.value
        await self._conn.execute(sql, row)
        return memory

    async def delete(self, memory_id: MemoryId) -> bool:
        await self._conn.execute(
            "DELETE FROM memories WHERE id = :id",
            {"id": memory_id.value},
        )
        return True

    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
        filters: list[FilterCondition] | None = None,
    ) -> Page[Memory]:
        pagination = pagination or PaginationParams()
        where_clauses: list[str] = []
        params: dict[str, Any] = {}

        if filters:
            for fc in filters:
                clause, param_name = self._build_filter(fc)
                if clause:
                    where_clauses.append(clause)
                    if param_name and fc.value is not None:
                        params[param_name] = fc.value

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        count_sql = f"SELECT COUNT(*) as cnt FROM memories WHERE {where_sql}"
        count_row = await self._conn.fetchone(count_sql, params)
        total = count_row["cnt"] if count_row else 0

        order_clause = ""
        if sort:
            parts = []
            for sf in sort:
                direction = "ASC" if sf.order.value == "asc" else "DESC"
                parts.append(f"{sf.field} {direction}")
            order_clause = " ORDER BY " + ", ".join(parts)

        query_sql = f"SELECT * FROM memories WHERE {where_sql}{order_clause} LIMIT :limit OFFSET :offset"
        query_params = {**params, "limit": pagination.limit, "offset": pagination.offset}
        rows = await self._conn.fetchall(query_sql, query_params)
        items = [Memory.from_dict(r) for r in rows]
        return Page(items=items, total=total, offset=pagination.offset, limit=pagination.limit)

    async def count(self, state: str | None = None) -> int:
        if state:
            row = await self._conn.fetchone(
                "SELECT COUNT(*) as cnt FROM memories WHERE state = :state",
                {"state": state},
            )
        else:
            row = await self._conn.fetchone("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"] if row else 0

    async def count_by_namespace(self, namespace: str) -> int:
        row = await self._conn.fetchone(
            "SELECT COUNT(*) as cnt FROM memories WHERE namespace = :ns AND state = 'active'",
            {"ns": namespace},
        )
        return row["cnt"] if row else 0

    async def find_by_state(
        self,
        state: MemoryState,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        rows = await self._conn.fetchall(
            "SELECT * FROM memories WHERE state = :state ORDER BY importance DESC LIMIT :limit OFFSET :offset",
            {"state": state.value, "limit": limit, "offset": offset},
        )
        return [Memory.from_dict(r) for r in rows]

    async def find_expired(self, limit: int = 500) -> list[Memory]:
        """Find memories whose TTL has expired (only active ones)."""
        rows = await self._conn.fetchall(
            """
            SELECT * FROM memories
            WHERE state IN ('active', 'archived')
              AND ttl IS NOT NULL
              AND ttl > 0
              AND (julianday('now') - julianday(created_at)) * 86400.0 > ttl
            LIMIT :limit
            """,
            {"limit": limit},
        )
        return [Memory.from_dict(r) for r in rows]

    async def find_below_importance(
        self,
        threshold: float,
        *,
        state: MemoryState = MemoryState.ACTIVE,
        limit: int = 500,
    ) -> list[Memory]:
        rows = await self._conn.fetchall(
            "SELECT * FROM memories WHERE state = :state AND importance < :threshold LIMIT :limit",
            {"state": state.value, "threshold": threshold, "limit": limit},
        )
        return [Memory.from_dict(r) for r in rows]

    async def search(
        self,
        query: MemoryQuery,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if query.memory_types:
            placeholders = ", ".join([f":mt_{i}" for i in range(len(query.memory_types))])
            conditions.append(f"memory_type IN ({placeholders})")
            for i, t in enumerate(query.memory_types):
                params[f"mt_{i}"] = t
        if query.namespaces:
            placeholders = ", ".join([f":ns_{i}" for i in range(len(query.namespaces))])
            conditions.append(f"namespace IN ({placeholders})")
            for i, ns in enumerate(query.namespaces):
                params[f"ns_{i}"] = ns
        if query.states:
            placeholders = ", ".join([f":st_{i}" for i in range(len(query.states))])
            conditions.append(f"state IN ({placeholders})")
            for i, s in enumerate(query.states):
                params[f"st_{i}"] = s.value
        if query.tags:
            for i, tag in enumerate(query.tags):
                conditions.append(f"tags LIKE :tag_{i}")
                params[f"tag_{i}"] = f"%{tag}%"
        if query.content_search:
            conditions.append("content LIKE :content_search")
            params["content_search"] = f"%{query.content_search}%"
        if query.sources:
            placeholders = ", ".join([f":src_{i}" for i in range(len(query.sources))])
            conditions.append(f"source IN ({placeholders})")
            for i, s in enumerate(query.sources):
                params[f"src_{i}"] = s
        if query.owners:
            placeholders = ", ".join([f":own_{i}" for i in range(len(query.owners))])
            conditions.append(f"owner IN ({placeholders})")
            for i, o in enumerate(query.owners):
                params[f"own_{i}"] = o
        if query.min_importance is not None:
            conditions.append("importance >= :min_imp")
            params["min_imp"] = query.min_importance
        if query.max_importance is not None:
            conditions.append("importance <= :max_imp")
            params["max_imp"] = query.max_importance
        if query.correlation_id:
            conditions.append("correlation_id = :corr")
            params["corr"] = query.correlation_id
        if query.created_after:
            conditions.append("created_at >= :cafter")
            params["cafter"] = query.created_after
        if query.created_before:
            conditions.append("created_at <= :cbefore")
            params["cbefore"] = query.created_before
        if query.only_expired:
            conditions.append("ttl IS NOT NULL AND ttl > 0")
            conditions.append(
                "(julianday('now') - julianday(created_at)) * 86400.0 > ttl"
            )

        where_sql = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM memories WHERE {where_sql} ORDER BY importance DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
        rows = await self._conn.fetchall(sql, params)
        return [Memory.from_dict(r) for r in rows]

    @staticmethod
    def _build_filter(fc: FilterCondition) -> tuple[str | None, str | None]:
        param = f"p_{fc.field.replace('.', '_')}"
        mapping = {
            "eq": (f"{fc.field} = :{param}", param),
            "ne": (f"{fc.field} != :{param}", param),
            "gt": (f"{fc.field} > :{param}", param),
            "gte": (f"{fc.field} >= :{param}", param),
            "lt": (f"{fc.field} < :{param}", param),
            "lte": (f"{fc.field} <= :{param}", param),
            "like": (f"{fc.field} LIKE :{param}", param),
            "in": (f"{fc.field} IN (:{param})", param),
        }
        if fc.operator.value in mapping:
            return mapping[fc.operator.value]
        if fc.operator.value == "is_null":
            return (f"{fc.field} IS NULL", None)
        if fc.operator.value == "not_null":
            return (f"{fc.field} IS NOT NULL", None)
        return (None, None)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager(MemoryService, MemorySearchService, MemoryGarbageCollector, MemorySnapshotService):
    """Central facade for all memory operations.

    Implements MemoryService, MemorySearchService, MemoryGarbageCollector,
    and MemorySnapshotService.  Composite rather than separate classes
    because they share the same repository, cache, and event bus.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        cache: CacheService | None = None,
        event_bus: EventBus | None = None,
        telemetry: TelemetryService | None = None,
        logger: Logger | None = None,
        policies: dict[str, MemoryTypePolicy] | None = None,
        retention: RetentionPolicy | None = None,
        graph: MemoryGraph | None = None,
        snapshot_service: SnapshotService | None = None,
    ) -> None:
        self._repo = repository
        self._cache = cache
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._logger = logger
        self._type_policies = policies or DEFAULT_TYPE_POLICIES
        self._retention = retention or RetentionPolicy()
        self._graph = graph
        self._snapshot_service = snapshot_service

    @property
    def graph(self) -> MemoryGraph | None:
        """Access the memory graph for relationship queries."""
        return self._graph

    @property
    def snapshots(self) -> SnapshotService | None:
        """Access the snapshot service for checkpoint/restore."""
        return self._snapshot_service

    # ------------------------------------------------------------------
    # MemoryService implementation
    # ------------------------------------------------------------------

    async def create(self, memory: Memory) -> Memory:
        start = time.monotonic()
        policy = self._type_policies.get(memory.memory_type)
        if policy:
            if memory.ttl is None and policy.ttl is not None:
                memory.ttl = policy.ttl
            if memory.importance == 0.5:
                memory.importance = policy.default_importance
        await self._repo.add(memory)
        await self._emit(MemoryCreated(
            memory_id=memory.id.value,
            memory_type=memory.memory_type,
            namespace=memory.namespace,
            importance=memory.importance,
        ))
        await self._cache_set(f"memory:{memory.id.value}", memory)
        self._record("memory.create", start)
        return memory

    async def get(self, memory_id: MemoryId) -> Memory | None:
        start = time.monotonic()
        cached = await self._cache_get(f"memory:{memory_id.value}")
        if cached is not None:
            self._record("memory.get.cache_hit", start)
            return cached
        memory = await self._repo.get(memory_id)
        if memory:
            memory.touch()
            await self._repo.update(memory)
            await self._cache_set(f"memory:{memory_id.value}", memory)
            await self._emit(MemoryAccessed(memory_id=memory_id.value))
        self._record("memory.get", start)
        return memory

    async def update(self, memory: Memory) -> Memory:
        start = time.monotonic()
        old = await self._repo.get(memory.id)
        changes: dict[str, Any] = {}
        if old:
            for field in ("content", "importance", "tags", "metadata", "owner", "source"):
                old_val = getattr(old, field, None)
                new_val = getattr(memory, field, None)
                if old_val != new_val:
                    changes[field] = new_val
        memory.version += 1
        memory.updated_at = _now_utc()
        await self._repo.update(memory)
        await self._cache_set(f"memory:{memory.id.value}", memory)
        if changes:
            await self._emit(MemoryUpdated(
                memory_id=memory.id.value,
                version=memory.version,
                changes=changes,
            ))
        self._record("memory.update", start)
        return memory

    async def delete(self, memory_id: MemoryId) -> bool:
        start = time.monotonic()
        await self._cache_delete(f"memory:{memory_id.value}")
        result = await self._repo.delete(memory_id)
        self._record("memory.delete", start)
        return result

    async def transition_state(
        self,
        memory_id: MemoryId,
        target: MemoryState,
        reason: str = "",
    ) -> Memory | None:
        start = time.monotonic()
        memory = await self._repo.get(memory_id)
        if memory is None:
            return None
        from_state = memory.state.value
        memory.transition_to(target)
        await self._repo.update(memory)
        await self._cache_set(f"memory:{memory_id.value}", memory)
        await self._emit(MemoryStateChanged(
            memory_id=memory_id.value,
            from_state=from_state,
            to_state=target.value,
            reason=reason,
        ))
        self._record("memory.transition_state", start)
        return memory

    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
    ) -> Page[Memory]:
        start = time.monotonic()
        result = await self._repo.list(
            pagination=pagination,
            sort=sort,
            filters=[FilterCondition("state", FilterOperator.EQ, MemoryState.ACTIVE.value)],
        )
        self._record("memory.list", start)
        return result

    async def count(self, state: MemoryState | None = None) -> int:
        return await self._repo.count(state=state.value if state else None)

    # ------------------------------------------------------------------
    # MemorySearchService implementation
    # ------------------------------------------------------------------

    async def search(
        self,
        query: MemoryQuery,
        *,
        pagination: PaginationParams | None = None,
    ) -> Page[Memory]:
        start = time.monotonic()
        pagination = pagination or PaginationParams()
        items = await self._repo.search(
            query,
            limit=pagination.limit,
            offset=pagination.offset,
        )
        # Estimate total with a separate count
        total = len(items)  # best-effort for pagination
        self._record("memory.search", start)
        return Page(items=items, total=total, offset=pagination.offset, limit=pagination.limit)

    async def search_by_importance(
        self,
        *,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        q = MemoryQuery(
            namespaces=[namespace] if namespace else None,
            min_importance=min_importance,
            states=[MemoryState.ACTIVE],
        )
        return await self._repo.search(q, limit=limit)

    async def search_by_tag(
        self,
        tag: str,
        *,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        q = MemoryQuery(
            tags=[tag],
            namespaces=[namespace] if namespace else None,
            states=[MemoryState.ACTIVE],
        )
        return await self._repo.search(q, limit=limit)

    async def search_temporal(
        self,
        *,
        after: str | None = None,
        before: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        q = MemoryQuery(
            namespaces=[namespace] if namespace else None,
            created_after=after,
            created_before=before,
            states=[MemoryState.ACTIVE],
        )
        return await self._repo.search(q, limit=limit)

    # ------------------------------------------------------------------
    # MemoryGarbageCollector implementation
    # ------------------------------------------------------------------

    async def collect(self) -> GCResult:
        start = time.monotonic()
        result = GCResult()
        processed_ids: set[str] = set()

        # Phase 1: Archive low-importance active memories
        low_imp = await self._repo.find_below_importance(
            self._retention.archive_threshold,
            state=MemoryState.ACTIVE,
        )
        for mem in low_imp:
            if mem.id.value in processed_ids:
                continue
            if self._retention.should_archive(mem):
                mem.transition_to(MemoryState.ARCHIVED)
                await self._repo.update(mem)
                processed_ids.add(mem.id.value)
                result.archived += 1

        # Phase 2: Forget archived low-importance + expired memories
        archived_low = await self._repo.find_below_importance(
            self._retention.archive_threshold * 0.5,
            state=MemoryState.ARCHIVED,
        )
        for mem in archived_low:
            if mem.id.value in processed_ids:
                continue
            mem.transition_to(MemoryState.FORGOTTEN)
            await self._repo.update(mem)
            processed_ids.add(mem.id.value)
            result.forgotten += 1

        expired = await self._repo.find_expired()
        for mem in expired:
            if mem.id.value in processed_ids:
                continue
            if mem.state != MemoryState.FORGOTTEN:
                mem.transition_to(MemoryState.FORGOTTEN)
                await self._repo.update(mem)
                processed_ids.add(mem.id.value)
                result.forgotten += 1

        # Phase 3: Delete expired forgotten memories
        forgotten_list = await self._repo.find_by_state(MemoryState.FORGOTTEN)
        for mem in forgotten_list:
            if mem.id.value in processed_ids:
                continue
            if self._retention.should_delete(mem):
                await self._repo.delete(mem.id)
                processed_ids.add(mem.id.value)
                result.deleted += 1

        elapsed = (time.monotonic() - start) * 1000
        if self._telemetry:
            self._telemetry.record_module_lifecycle(
                "memory", "gc_collect", elapsed, True
            )
        if self._logger:
            self._logger.info(
                "GC sweep complete | archived={archived} forgotten={forgotten} deleted={deleted}",
                archived=result.archived,
                forgotten=result.forgotten,
                deleted=result.deleted,
            )
        return result

    async def count_candidates(self) -> int:
        count = 0
        # Active below archive threshold
        low_imp = await self._repo.find_below_importance(
            self._retention.archive_threshold,
            state=MemoryState.ACTIVE,
        )
        count += len(low_imp)
        # Expired
        expired = await self._repo.find_expired()
        count += len(expired)
        # Forgotten past grace period
        forgotten_list = await self._repo.find_by_state(MemoryState.FORGOTTEN)
        for mem in forgotten_list:
            if self._retention.should_delete(mem):
                count += 1
        return count

    # ------------------------------------------------------------------
    # MemorySnapshotService implementation
    # ------------------------------------------------------------------

    async def create_snapshot(self, label: str = "") -> MemorySnapshot:
        if self._snapshot_service is None:
            raise RuntimeError("Snapshot service not configured")
        return await self._snapshot_service.create_snapshot(label=label)

    async def list_snapshots(self) -> list[dict[str, Any]]:
        if self._snapshot_service is None:
            return []
        return await self._snapshot_service.list_snapshots()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cache_set(self, key: str, memory: Memory) -> None:
        if self._cache:
            await self._cache.set(key, memory.to_dict(), ttl=300.0)

    async def _cache_get(self, key: str) -> Memory | None:
        if self._cache:
            data = await self._cache.get(key)
            if data is not None and isinstance(data, dict):
                return Memory.from_dict(data)
        return None

    async def _cache_delete(self, key: str) -> None:
        if self._cache:
            await self._cache.delete(key)

    async def _emit(self, event: Any) -> None:
        if self._event_bus:
            await self._event_bus.publish(event)

    def _record(self, operation: str, start: float) -> None:
        if self._telemetry:
            elapsed = (time.monotonic() - start) * 1000
            self._telemetry.record_event_metrics(operation, elapsed, True)


def _now_utc():
    return datetime.now(timezone.utc)


class _MemorySnapshot:
    """Placeholder for Phase 3.5 snapshot service."""
    pass
