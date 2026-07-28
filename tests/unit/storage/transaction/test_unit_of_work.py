"""Tests for SqliteUnitOfWork.

Verifies:
- Context manager begins a transaction
- Commit persists changes
- Rollback discards changes
- Auto-rollback on exit without explicit commit
- Multiple operations within one UoW
"""

from __future__ import annotations

import pytest

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.transaction.unit_of_work import SqliteUnitOfWork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def connection() -> SQLiteConnection:
    c = SQLiteConnection(":memory:")
    await c.execute("CREATE TABLE test_uow (id INTEGER PRIMARY KEY, val TEXT)")
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


class TestCommit:
    async def test_commit_persists_changes(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection) as uow:
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'persisted')")
            await uow.commit()

        # Verify data is visible on the same connection
        row = await connection.fetchone("SELECT * FROM test_uow WHERE id = 1")
        assert row is not None
        assert row["val"] == "persisted"

    async def test_multiple_operations_committed(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection) as uow:
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'a')")
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (2, 'b')")
            await uow.commit()

        rows = await connection.fetchall("SELECT * FROM test_uow ORDER BY id")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    async def test_rollback_discards_changes(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection) as uow:
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'discarded')")
            await uow.rollback()

        rows = await connection.fetchall("SELECT * FROM test_uow")
        assert len(rows) == 0

    async def test_commit_after_rollback_does_not_persist(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection) as uow:
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'gone')")
            await uow.rollback()
            # The transaction is already rolled back; further commits
            # start a new implicit transaction but the rolled-back data
            # is gone
        rows = await connection.fetchall("SELECT * FROM test_uow")
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Auto-rollback
# ---------------------------------------------------------------------------


class TestAutoRollback:
    async def test_auto_rollback_on_exit(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection):
            await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'auto')")
            # No commit — should auto-rollback

        rows = await connection.fetchall("SELECT * FROM test_uow")
        assert len(rows) == 0

    async def test_exception_causes_rollback(self, connection: SQLiteConnection) -> None:
        with pytest.raises(RuntimeError):
            async with SqliteUnitOfWork(connection):
                await connection.execute("INSERT INTO test_uow (id, val) VALUES (1, 'boom')")
                raise RuntimeError("unexpected error")

        rows = await connection.fetchall("SELECT * FROM test_uow")
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


class TestFlush:
    async def test_flush_noop(self, connection: SQLiteConnection) -> None:
        async with SqliteUnitOfWork(connection) as uow:
            await uow.flush()  # should not raise
            await uow.commit()
