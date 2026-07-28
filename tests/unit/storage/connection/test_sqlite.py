"""Tests for SQLiteConnection.

Verifies:
- Connection opens (including in-memory)
- execute/fetchone/fetchall for CRUD
- Parameterized queries (both positional and named)
- executemany
- execute_script for DDL batches
- close and is_closed
- Error raises ConnectionError_ on bad SQL
"""

from __future__ import annotations

import pytest

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.errors import ConnectionError_


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def conn() -> SQLiteConnection:
    """In-memory SQLite connection for testing."""
    c = SQLiteConnection(":memory:")
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_open_and_close(self) -> None:
        c = SQLiteConnection(":memory:")
        assert c.is_closed is False
        await c.execute("CREATE TABLE t (x INTEGER)")
        await c.close()
        assert c.is_closed is True

    async def test_double_close_safe(self) -> None:
        c = SQLiteConnection(":memory:")
        await c.close()
        await c.close()  # should not raise


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------


class TestDDL:
    async def test_create_table(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        # Verify the table exists
        rows = await conn.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    async def test_insert_and_fetchone(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        await conn.execute("INSERT INTO t (id, val) VALUES (?, ?)", [1, "hello"])
        row = await conn.fetchone("SELECT * FROM t WHERE id = ?", [1])
        assert row is not None
        assert row["val"] == "hello"

    async def test_insert_and_fetchall(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        await conn.execute("INSERT INTO t (id, val) VALUES (?, ?)", [1, "a"])
        await conn.execute("INSERT INTO t (id, val) VALUES (?, ?)", [2, "b"])
        rows = await conn.fetchall("SELECT * FROM t ORDER BY id")
        assert len(rows) == 2
        assert rows[0]["val"] == "a"

    async def test_fetchone_no_result(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        row = await conn.fetchone("SELECT * FROM t WHERE id = ?", [99])
        assert row is None

    async def test_fetchall_empty(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        rows = await conn.fetchall("SELECT * FROM t")
        assert rows == []

    async def test_named_params(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        await conn.execute("INSERT INTO t (id, val) VALUES (:id, :val)", {"id": 1, "val": "named"})
        row = await conn.fetchone("SELECT * FROM t WHERE id = :id", {"id": 1})
        assert row is not None
        assert row["val"] == "named"


# ---------------------------------------------------------------------------
# executemany
# ---------------------------------------------------------------------------


class TestExecutemany:
    async def test_insert_many(self, conn: SQLiteConnection) -> None:
        await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        await conn.executemany(
            "INSERT INTO t (id, val) VALUES (?, ?)",
            [[1, "a"], [2, "b"], [3, "c"]],
        )
        rows = await conn.fetchall("SELECT * FROM t ORDER BY id")
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------


class TestExecuteScript:
    async def test_batch_ddl(self, conn: SQLiteConnection) -> None:
        await conn.execute_script(
            """
            CREATE TABLE a (x INTEGER);
            CREATE TABLE b (y TEXT);
            """
        )
        tables = await conn.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in tables}
        assert "a" in names
        assert "b" in names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_bad_sql_raises(self, conn: SQLiteConnection) -> None:
        with pytest.raises(ConnectionError_):
            await conn.execute("INVALID SQL STATEMENT")
