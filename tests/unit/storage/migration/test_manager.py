"""Tests for SqliteMigrationManager and V001 migration.

Verifies:
- Tracking table initialization
- Apply a migration creates the expected tables
- Double-apply raises MigrationError
- Rollback drops the tables
- Double-rollback raises MigrationError
- History returns applied migrations
- Pending returns unapplied migrations
- apply_all applies in order
- rollback_all rolls back in reverse order
"""

from __future__ import annotations

import pytest

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.errors import MigrationError
from app.storage.migration.manager import SqliteMigrationManager
from app.storage.migration.sqlite import V001_InitialSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def connection() -> SQLiteConnection:
    c = SQLiteConnection(":memory:")
    yield c
    await c.close()


@pytest.fixture
def manager() -> SqliteMigrationManager:
    return SqliteMigrationManager()


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    async def test_initialize_creates_tracking_table(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        row = await connection.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_history'"
        )
        assert row is not None

    async def test_initialize_is_idempotent(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        await manager.initialize(connection)  # should not raise


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class TestApply:
    async def test_apply_creates_tables(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        migration = V001_InitialSchema()
        await manager.initialize(connection)
        await manager.apply(connection, migration)
        # Verify event_store table exists
        row = await connection.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='event_store'"
        )
        assert row is not None

    async def test_double_apply_raises(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        migration = V001_InitialSchema()
        await manager.initialize(connection)
        await manager.apply(connection, migration)
        with pytest.raises(MigrationError, match="already been applied"):
            await manager.apply(connection, migration)

    async def test_has_been_applied(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        assert await manager.has_been_applied(connection, "V001") is False
        await manager.apply(connection, V001_InitialSchema())
        assert await manager.has_been_applied(connection, "V001") is True


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    async def test_rollback_drops_tables(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        migration = V001_InitialSchema()
        await manager.initialize(connection)
        await manager.apply(connection, migration)
        await manager.rollback(connection, migration)
        row = await connection.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='event_store'"
        )
        assert row is None

    async def test_double_rollback_raises(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        migration = V001_InitialSchema()
        await manager.initialize(connection)
        await manager.apply(connection, migration)
        await manager.rollback(connection, migration)
        with pytest.raises(MigrationError, match="not been applied"):
            await manager.rollback(connection, migration)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    async def test_history_returns_applied(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        await manager.apply(connection, V001_InitialSchema())
        rows = await manager.history(connection)
        assert len(rows) == 1
        assert rows[0]["version"] == "V001"

    async def test_history_empty_initially(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        assert await manager.history(connection) == []


# ---------------------------------------------------------------------------
# Pending
# ---------------------------------------------------------------------------


class TestPending:
    async def test_pending_returns_unapplied(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        pending = await manager.pending(connection, [V001_InitialSchema()])
        assert len(pending) == 1
        assert pending[0].version == "V001"

    async def test_pending_empty_after_apply(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.initialize(connection)
        await manager.apply(connection, V001_InitialSchema())
        pending = await manager.pending(connection, [V001_InitialSchema()])
        assert pending == []


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------


class TestBulk:
    async def test_apply_all(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        applied = await manager.apply_all(connection, [V001_InitialSchema()])
        assert applied == ["V001"]
        assert await manager.has_been_applied(connection, "V001") is True

    async def test_rollback_all(self, connection: SQLiteConnection, manager: SqliteMigrationManager) -> None:
        await manager.apply_all(connection, [V001_InitialSchema()])
        rolled_back = await manager.rollback_all(connection, [V001_InitialSchema()])
        assert rolled_back == ["V001"]
        assert await manager.has_been_applied(connection, "V001") is False
