"""Tests for SqliteRepository (BaseRepositoryImpl).

Uses a concrete ``WidgetRepo`` and a ``widgets`` table in an in-memory
SQLite database to exercise all repository operations.

Verifies:
- add and get by id
- update modifies fields
- delete (soft) sets deleted_at
- list with pagination
- list with sorting
- list with filtering
- count
- add_batch
- get returns None for missing/soft-deleted
"""

from __future__ import annotations

from typing import Any

import pytest

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.repository.base import BaseRepositoryImpl
from app.storage.repository.sqlite import SqliteRepository
from app.storage.interfaces import (
    FilterCondition,
    FilterOperator,
    Page,
    PaginationParams,
    SortField,
    SortOrder,
    TModel,
    TId,
)


# ---------------------------------------------------------------------------
# Test domain model
# ---------------------------------------------------------------------------


class Widget:
    """Simple domain model for testing."""

    def __init__(self, *, id: str, name: str, price: float, owner: str | None = None):
        self.id = id
        self.name = name
        self.price = price
        self.owner = owner

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Widget):
            return NotImplemented
        return self.id == other.id and self.name == other.name and self.price == other.price


# ---------------------------------------------------------------------------
# Test repository
# ---------------------------------------------------------------------------


class WidgetRepo(SqliteRepository[Widget, str]):
    @property
    def _table(self) -> str:
        return "widgets"

    @property
    def _id_field(self) -> str:
        return "id"

    def _model_to_row(self, model: Widget) -> dict[str, Any]:
        return {
            "id": model.id,
            "name": model.name,
            "price": model.price,
            "owner": model.owner,
        }

    def _row_to_model(self, row: dict[str, Any]) -> Widget:
        return Widget(
            id=row["id"],
            name=row["name"],
            price=row["price"],
            owner=row.get("owner"),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def connection() -> SQLiteConnection:
    """In-memory SQLite connection with a widgets table."""
    c = SQLiteConnection(":memory:")
    await c.execute(
        """
        CREATE TABLE widgets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL DEFAULT 0.0,
            owner TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT,
            deleted_at TEXT
        )
        """
    )
    yield c
    await c.close()


@pytest.fixture
def repo(connection: SQLiteConnection) -> WidgetRepo:
    return WidgetRepo(connection=connection)


# ---------------------------------------------------------------------------
# Add / Get
# ---------------------------------------------------------------------------


class TestAddAndGet:
    async def test_add_and_get(self, repo: WidgetRepo) -> None:
        w = Widget(id="w1", name="Gadget", price=9.99)
        await repo.add(w)
        result = await repo.get("w1")
        assert result is not None
        assert result.name == "Gadget"
        assert result.price == 9.99

    async def test_get_nonexistent(self, repo: WidgetRepo) -> None:
        result = await repo.get("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    async def test_update(self, repo: WidgetRepo) -> None:
        w = Widget(id="w2", name="Old", price=1.0)
        await repo.add(w)
        w.name = "New"
        w.price = 2.0
        await repo.update(w)
        result = await repo.get("w2")
        assert result is not None
        assert result.name == "New"
        assert result.price == 2.0


# ---------------------------------------------------------------------------
# Delete (soft)
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_soft_delete_marks_deleted(self, repo: WidgetRepo) -> None:
        w = Widget(id="w3", name="Temp", price=5.0)
        await repo.add(w)
        await repo.delete("w3")
        # Soft-deleted records should not appear in get()
        result = await repo.get("w3")
        assert result is None

    async def test_soft_delete_still_in_database(self, connection: SQLiteConnection, repo: WidgetRepo) -> None:
        w = Widget(id="w4", name="Hidden", price=5.0)
        await repo.add(w)
        await repo.delete("w4")
        # Direct query should still find the row (with deleted_at set)
        row = await connection.fetchone("SELECT * FROM widgets WHERE id = :id", {"id": "w4"})
        assert row is not None
        assert row["deleted_at"] is not None


# ---------------------------------------------------------------------------
# List — pagination
# ---------------------------------------------------------------------------


class TestList:
    async def test_list_returns_all_not_deleted(self, repo: WidgetRepo) -> None:
        for i in range(5):
            await repo.add(Widget(id=f"l{i}", name=f"Item{i}", price=float(i)))
        page = await repo.list()
        assert page.total == 5
        assert len(page.items) == 5

    async def test_list_with_pagination(self, repo: WidgetRepo) -> None:
        for i in range(10):
            await repo.add(Widget(id=f"p{i}", name=f"Item{i}", price=float(i)))
        page = await repo.list(pagination=PaginationParams(offset=0, limit=3))
        assert len(page.items) == 3
        assert page.total == 10
        assert page.has_next is True

    async def test_list_last_page(self, repo: WidgetRepo) -> None:
        for i in range(5):
            await repo.add(Widget(id=f"lp{i}", name=f"Item{i}", price=float(i)))
        page = await repo.list(pagination=PaginationParams(offset=3, limit=3))
        assert len(page.items) == 2  # items 3 and 4
        assert page.has_next is False

    async def test_list_excludes_deleted(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="keep", name="Keep", price=1.0))
        await repo.add(Widget(id="remove", name="Remove", price=2.0))
        await repo.delete("remove")
        page = await repo.list()
        assert page.total == 1
        assert page.items[0].id == "keep"


class TestListSort:
    async def test_sort_ascending(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="b", name="Beta", price=2.0))
        await repo.add(Widget(id="a", name="Alpha", price=1.0))
        page = await repo.list(sort=[SortField("price", SortOrder.ASC)])
        assert page.items[0].price == 1.0
        assert page.items[1].price == 2.0

    async def test_sort_descending(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="a", name="Alpha", price=1.0))
        await repo.add(Widget(id="b", name="Beta", price=2.0))
        page = await repo.list(sort=[SortField("price", SortOrder.DESC)])
        assert page.items[0].price == 2.0
        assert page.items[1].price == 1.0


class TestListFilter:
    async def test_filter_eq(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="a", name="Alpha", price=1.0))
        await repo.add(Widget(id="b", name="Beta", price=2.0))
        page = await repo.list(filters=[FilterCondition("price", FilterOperator.EQ, 1.0)])
        assert page.total == 1
        assert page.items[0].id == "a"

    async def test_filter_gt(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="a", name="Alpha", price=1.0))
        await repo.add(Widget(id="b", name="Beta", price=5.0))
        await repo.add(Widget(id="c", name="Gamma", price=10.0))
        page = await repo.list(filters=[FilterCondition("price", FilterOperator.GT, 4.0)])
        assert page.total == 2

    async def test_filter_is_null(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="a", name="Alpha", price=1.0, owner="bob"))
        await repo.add(Widget(id="b", name="Beta", price=2.0))
        page = await repo.list(filters=[FilterCondition("owner", FilterOperator.IS_NULL)])
        assert page.total == 1
        assert page.items[0].id == "b"


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestCount:
    async def test_count_all(self, repo: WidgetRepo) -> None:
        for i in range(5):
            await repo.add(Widget(id=f"c{i}", name=f"Item{i}", price=float(i)))
        assert await repo.count() == 5

    async def test_count_with_filter(self, repo: WidgetRepo) -> None:
        await repo.add(Widget(id="a", name="Alpha", price=1.0))
        await repo.add(Widget(id="b", name="Beta", price=100.0))
        cnt = await repo.count(filters=[FilterCondition("price", FilterOperator.GT, 50.0)])
        assert cnt == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    async def test_add_batch(self, repo: WidgetRepo) -> None:
        widgets = [
            Widget(id="b1", name="Batch1", price=1.0),
            Widget(id="b2", name="Batch2", price=2.0),
        ]
        await repo.add_batch(widgets)
        assert await repo.count() == 2
