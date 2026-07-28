"""EventRepository — domain repository for persisted event records.

Demonstrates how a domain repository extends the base CRUD repository.
This is the reading-side counterpart to the EventStore (which handles
the writing/replay side).
"""

from __future__ import annotations

from typing import Any

from app.storage.interfaces import (
    FilterCondition,
    FilterOperator,
    Page,
    PaginationParams,
    SortField,
    SortOrder,
)
from app.storage.repository.sqlite import SqliteRepository


class EventRecord:
    """Domain model for a persisted event (read-only view)."""

    def __init__(
        self,
        *,
        id: str,
        event_type: str,
        source: str,
        correlation_id: str,
        timestamp: str,
        payload: str,
    ) -> None:
        self.id = id
        self.event_type = event_type
        self.source = source
        self.correlation_id = correlation_id
        self.timestamp = timestamp
        self.payload = payload


class EventRepository(SqliteRepository[EventRecord, str]):
    """Repository for querying persisted event records.

    Provides domain-specific query methods on top of base CRUD.
    """

    @property
    def _table(self) -> str:
        return "event_store"

    @property
    def _id_field(self) -> str:
        return "id"

    def _model_to_row(self, model: EventRecord) -> dict[str, Any]:
        return {
            "id": model.id,
            "event_type": model.event_type,
            "source": model.source,
            "correlation_id": model.correlation_id,
            "timestamp": model.timestamp,
            "payload": model.payload,
        }

    def _row_to_model(self, row: dict[str, Any]) -> EventRecord:
        return EventRecord(
            id=row["id"],
            event_type=row["event_type"],
            source=row["source"],
            correlation_id=row["correlation_id"],
            timestamp=row["timestamp"],
            payload=row["payload"],
        )

    # ------------------------------------------------------------------
    # Overrides — event_store has no deleted_at column
    # ------------------------------------------------------------------

    async def get(self, id: str) -> EventRecord | None:
        """Retrieve by primary key without soft-delete filter."""
        row = await self._connection.fetchone(
            f"SELECT * FROM event_store WHERE {self._id_field} = :id",
            {"id": id},
        )
        return self._row_to_model(row) if row else None

    async def count(
        self, filters: list[FilterCondition] | None = None
    ) -> int:
        """Count without soft-delete filter."""
        sql = "SELECT COUNT(*) as cnt FROM event_store"
        params: dict[str, Any] = {}
        if filters:
            where_clauses = []
            for fc in filters:
                clause, param_name = self._build_filter(fc)
                if clause:
                    where_clauses.append(clause)
                    if param_name and fc.value is not None:
                        params[param_name] = fc.value
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
        row = await self._connection.fetchone(sql, params)
        return row["cnt"] if row else 0

    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
        filters: list[FilterCondition] | None = None,
    ) -> Page[EventRecord]:
        """Paginated listing without soft-delete filtering."""
        # Bypass soft-delete logic in base — use raw connection queries
        pagination = pagination or PaginationParams()
        params: dict[str, Any] = {}

        count_sql = "SELECT COUNT(*) as cnt FROM event_store"
        query_sql = "SELECT * FROM event_store"
        where_clauses: list[str] = []

        if filters:
            for fc in filters:
                clause, param_name = self._build_filter(fc)
                if clause:
                    where_clauses.append(clause)
                    if param_name and fc.value is not None:
                        params[param_name] = fc.value

        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
            count_sql += where_sql
            query_sql += where_sql

        count_row = await self._connection.fetchone(count_sql, params)
        total = count_row["cnt"] if count_row else 0

        order_clause = ""
        if sort:
            order_parts = []
            for sf in sort:
                direction = "ASC" if sf.order.value == "asc" else "DESC"
                order_parts.append(f"{sf.field} {direction}")
            order_clause = " ORDER BY " + ", ".join(order_parts)

        query_sql += order_clause
        query_sql += " LIMIT :limit OFFSET :offset"
        query_params = {**params, "limit": pagination.limit, "offset": pagination.offset}

        rows = await self._connection.fetchall(query_sql, query_params)
        items = [self._row_to_model(r) for r in rows]
        return Page(items=items, total=total, offset=pagination.offset, limit=pagination.limit)

    # ------------------------------------------------------------------
    # Domain-specific query methods
    # ------------------------------------------------------------------

    async def find_by_type(
        self,
        event_type: str,
        *,
        pagination: PaginationParams | None = None,
    ) -> Page[EventRecord]:
        """Find all events of a specific type."""
        return await self.list(
            pagination=pagination,
            filters=[FilterCondition("event_type", FilterOperator.EQ, event_type)],
        )

    async def find_by_source(
        self,
        source: str,
        *,
        pagination: PaginationParams | None = None,
    ) -> Page[EventRecord]:
        """Find all events from a specific source."""
        return await self.list(
            pagination=pagination,
            filters=[FilterCondition("source", FilterOperator.EQ, source)],
        )

    async def find_by_correlation(
        self,
        correlation_id: str,
        *,
        pagination: PaginationParams | None = None,
    ) -> Page[EventRecord]:
        """Find all events with a specific correlation ID."""
        return await self.list(
            pagination=pagination,
            sort=[SortField("timestamp")],
            filters=[
                FilterCondition(
                    "correlation_id", FilterOperator.EQ, correlation_id
                )
            ],
        )

    async def find_recent(
        self,
        limit: int = 20,
    ) -> Page[EventRecord]:
        """Return the most recent events."""
        return await self.list(
            pagination=PaginationParams(offset=0, limit=limit),
            sort=[SortField("timestamp", order=SortOrder.DESC)],
        )
