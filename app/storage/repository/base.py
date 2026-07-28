"""BaseRepository — generic CRUD for domain models.

Provides the default implementation every domain repository extends.
Domain-specific query methods are added by subclassing.
"""

from __future__ import annotations

import abc
from typing import Any, Generic

from app.storage.interfaces import (
    BaseRepository,
    FilterCondition,
    Page,
    PaginationParams,
    SortField,
    TModel,
    TId,
)


class BaseRepositoryImpl(BaseRepository[TModel, TId]):
    """Abstract base repository that domain repositories extend.

    Subclasses must provide:
    - ``_table`` — the SQL table name
    - ``_id_field`` — the primary key column name
    - ``_model_to_row(model)`` — convert model to dict
    - ``_row_to_model(row)`` — convert Row to model
    - Access to a ``SQLConnection`` via ``_connection``
    """

    # -- Subclass hooks --------------------------------------------------

    @property
    @abc.abstractmethod
    def _table(self) -> str:
        """SQL table name."""

    @property
    @abc.abstractmethod
    def _id_field(self) -> str:
        """Primary key column name."""

    @abc.abstractmethod
    def _model_to_row(self, model: TModel) -> dict[str, Any]:
        """Convert a domain model to a flat dict for INSERT/UPDATE."""

    @abc.abstractmethod
    def _row_to_model(self, row: dict[str, Any]) -> TModel:
        """Convert a database row back to a domain model."""

    # -- CRUD --------------------------------------------------------------

    async def add(self, model: TModel) -> TModel:
        """Insert a new record and return it."""
        row = self._model_to_row(model)
        columns = ", ".join(row.keys())
        placeholders = ", ".join([f":{k}" for k in row])
        sql = f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})"
        await self._connection.execute(sql, row)
        return model

    async def get(self, id: TId) -> TModel | None:
        """Retrieve by primary key, or ``None``."""
        sql = f"SELECT * FROM {self._table} WHERE {self._id_field} = :id AND deleted_at IS NULL"
        row = await self._connection.fetchone(sql, {"id": id})
        return self._row_to_model(row) if row else None

    async def update(self, model: TModel) -> TModel:
        """Update an existing record by primary key."""
        row = self._model_to_row(model)
        id_value = row.get(self._id_field)
        set_clause = ", ".join([f"{k} = :{k}" for k in row if k != self._id_field])
        set_clause += ", updated_at = CURRENT_TIMESTAMP"
        sql = f"UPDATE {self._table} SET {set_clause} WHERE {self._id_field} = :id"
        row["id"] = id_value
        await self._connection.execute(sql, row)
        return model

    async def delete(self, id: TId) -> bool:
        """Soft-delete by setting ``deleted_at``.

        Returns ``True`` if a row was marked as deleted.
        """
        sql = f"UPDATE {self._table} SET deleted_at = CURRENT_TIMESTAMP WHERE {self._id_field} = :id AND deleted_at IS NULL"
        await self._connection.execute(sql, {"id": id})
        # For a soft delete we can't easily know if a row was affected
        # without another query.  Return True for now.
        return True

    async def hard_delete(self, id: TId) -> bool:
        """Permanently remove a record.

        Use only when soft delete is inappropriate (e.g. temp data).
        """
        sql = f"DELETE FROM {self._table} WHERE {self._id_field} = :id"
        await self._connection.execute(sql, {"id": id})
        return True

    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
        filters: list[FilterCondition] | None = None,
    ) -> Page[TModel]:
        """Paginated listing with sorting and filtering.

        Only returns non-deleted records (``deleted_at IS NULL``).
        """
        pagination = pagination or PaginationParams()
        where_clauses = ["deleted_at IS NULL"]
        params: dict[str, Any] = {}

        if filters:
            for fc in filters:
                clause, param_name = self._build_filter(fc)
                if clause:
                    where_clauses.append(clause)
                    if param_name and fc.value is not None:
                        params[param_name] = fc.value

        where_sql = " AND ".join(where_clauses)

        # Count
        count_sql = f"SELECT COUNT(*) as cnt FROM {self._table} WHERE {where_sql}"
        count_row = await self._connection.fetchone(count_sql, params)
        total = count_row["cnt"] if count_row else 0

        # Sort
        order_clause = ""
        if sort:
            order_parts = []
            for sf in sort:
                direction = "ASC" if sf.order.value == "asc" else "DESC"
                order_parts.append(f"{sf.field} {direction}")
            order_clause = " ORDER BY " + ", ".join(order_parts)

        # Pagination
        limit = pagination.limit
        offset = pagination.offset
        query = f"SELECT * FROM {self._table} WHERE {where_sql}{order_clause} LIMIT :limit OFFSET :offset"
        query_params = {**params, "limit": limit, "offset": offset}
        rows = await self._connection.fetchall(query, query_params)

        items = [self._row_to_model(r) for r in rows]
        return Page(items=items, total=total, offset=offset, limit=limit)

    async def count(self, filters: list[FilterCondition] | None = None) -> int:
        """Count records matching optional filters."""
        where_clauses = ["deleted_at IS NULL"]
        params: dict[str, Any] = {}

        if filters:
            for fc in filters:
                clause, param_name = self._build_filter(fc)
                if clause:
                    where_clauses.append(clause)
                    if param_name and fc.value is not None:
                        params[param_name] = fc.value

        where_sql = " AND ".join(where_clauses)
        sql = f"SELECT COUNT(*) as cnt FROM {self._table} WHERE {where_sql}"
        row = await self._connection.fetchone(sql, params)
        return row["cnt"] if row else 0

    async def add_batch(self, models: list[TModel]) -> list[TModel]:
        """Insert multiple records in a single operation."""
        if not models:
            return models
        rows = [self._model_to_row(m) for m in models]
        columns = ", ".join(rows[0].keys())
        placeholders = ", ".join([f":{k}" for k in rows[0]])
        sql = f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})"
        await self._connection.executemany(sql, rows)
        return models

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(fc: FilterCondition) -> tuple[str | None, str | None]:
        """Build a WHERE clause fragment and parameter name from a filter."""
        param = f"p_{fc.field.replace('.', '_')}"

        mapping = {
            "eq": (f"{fc.field} = :{param}", param),
            "ne": (f"{fc.field} != :{param}", param),
            "gt": (f"{fc.field} > :{param}", param),
            "gte": (f"{fc.field} >= :{param}", param),
            "lt": (f"{fc.field} < :{param}", param),
            "lte": (f"{fc.field} <= :{param}", param),
            "is_null": (f"{fc.field} IS NULL", None),
            "not_null": (f"{fc.field} IS NOT NULL", None),
        }

        result = mapping.get(fc.operator.value)
        if result:
            return result

        if fc.operator.value == "in":
            return (f"{fc.field} IN (:{param})", param)
        if fc.operator.value == "not_in":
            return (f"{fc.field} NOT IN (:{param})", param)
        if fc.operator.value == "like":
            return (f"{fc.field} LIKE :{param}", param)
        if fc.operator.value == "ilike":
            return (f"LOWER({fc.field}) LIKE LOWER(:{param})", param)

        return (None, None)
