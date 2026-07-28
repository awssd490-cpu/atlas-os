"""Migration manager — orchestrates schema migrations.

Maintains a ``migration_history`` tracking table.  Migrations are
applied in version order; rollbacks are applied in reverse version order.
"""

from __future__ import annotations

from typing import Any

from app.storage.errors import MigrationError
from app.storage.interfaces import Migration, MigrationManager, SQLConnection


class SqliteMigrationManager(MigrationManager):
    """Migration manager backed by SQLite.

    Creates a ``migration_history`` table on initialization and tracks
    every applied migration by version and checksum.
    """

    async def initialize(self, connection: SQLConnection) -> None:
        """Create the migration tracking table if it doesn't exist."""
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_history (
                version     TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
                checksum    TEXT NOT NULL DEFAULT ''
            )
            """
        )

    async def apply(self, connection: SQLConnection, migration: Migration) -> None:
        """Apply a migration and record it in the history.

        Raises :class:`MigrationError` if the migration has already been
        applied.
        """
        if await self.has_been_applied(connection, migration.version):
            raise MigrationError(
                f"Migration '{migration.version}' has already been applied",
                details={"version": migration.version},
            )

        await migration.up(connection)
        await connection.execute(
            "INSERT INTO migration_history (version, description) VALUES (:version, :desc)",
            {"version": migration.version, "desc": migration.__class__.__name__},
        )

    async def rollback(self, connection: SQLConnection, migration: Migration) -> None:
        """Rollback a migration and remove it from the history.

        Raises :class:`MigrationError` if the migration has never been
        applied.
        """
        if not await self.has_been_applied(connection, migration.version):
            raise MigrationError(
                f"Migration '{migration.version}' has not been applied",
                details={"version": migration.version},
            )

        await migration.down(connection)
        await connection.execute(
            "DELETE FROM migration_history WHERE version = :version",
            {"version": migration.version},
        )

    async def has_been_applied(self, connection: SQLConnection, version: str) -> bool:
        """Return True when the migration version exists in history."""
        row = await connection.fetchone(
            "SELECT 1 FROM migration_history WHERE version = :version",
            {"version": version},
        )
        return row is not None

    async def history(self, connection: SQLConnection) -> list[dict[str, Any]]:
        """Return all applied migrations, ordered by version."""
        rows = await connection.fetchall(
            "SELECT * FROM migration_history ORDER BY version"
        )
        return [dict(r) for r in rows]

    async def pending(
        self,
        connection: SQLConnection,
        migrations: list[Migration],
    ) -> list[Migration]:
        """Return migrations that have not been applied, sorted by version."""
        await self.initialize(connection)
        pending_list: list[Migration] = []
        for m in sorted(migrations, key=lambda x: x.version):
            if not await self.has_been_applied(connection, m.version):
                pending_list.append(m)
        return pending_list

    async def apply_all(
        self,
        connection: SQLConnection,
        migrations: list[Migration],
    ) -> list[str]:
        """Apply all pending migrations in version order.

        Returns a list of version identifiers that were applied.
        """
        applied: list[str] = []
        for m in await self.pending(connection, migrations):
            await self.apply(connection, m)
            applied.append(m.version)
        return applied

    async def rollback_all(
        self,
        connection: SQLConnection,
        migrations: list[Migration],
    ) -> list[str]:
        """Rollback all applied migrations in reverse version order.

        Returns a list of version identifiers that were rolled back.
        """
        rolled_back: list[str] = []
        sorted_migrations = sorted(migrations, key=lambda x: x.version, reverse=True)
        for m in sorted_migrations:
            if await self.has_been_applied(connection, m.version):
                await self.rollback(connection, m)
                rolled_back.append(m.version)
        return rolled_back
