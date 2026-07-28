"""SQLite-specific migrations for ATLAS.

Each migration is a class that implements the ``Migration`` ABC with
``up(conn)`` and ``down(conn)`` methods.
"""

from __future__ import annotations

from app.storage.interfaces import Migration, SQLConnection


class V001_InitialSchema(Migration):
    """Foundation schema: event store, configuration store, and metadata tables."""

    @property
    def version(self) -> str:
        return "V001"

    async def up(self, connection: SQLConnection) -> None:
        """Create the initial set of tables."""
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS event_store (
                id              TEXT PRIMARY KEY,
                event_type      TEXT NOT NULL,
                version         INTEGER NOT NULL DEFAULT 1,
                source          TEXT NOT NULL,
                correlation_id  TEXT NOT NULL,
                target          TEXT NOT NULL DEFAULT '*',
                timestamp       TEXT NOT NULL,
                payload         TEXT NOT NULL,
                metadata        TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_type ON event_store(event_type)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_correlation ON event_store(correlation_id)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON event_store(source)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON event_store(timestamp)"
        )

    async def down(self, connection: SQLConnection) -> None:
        """Drop the event_store table and its indexes."""
        await connection.execute("DROP INDEX IF EXISTS idx_timestamp")
        await connection.execute("DROP INDEX IF EXISTS idx_source")
        await connection.execute("DROP INDEX IF EXISTS idx_correlation")
        await connection.execute("DROP INDEX IF EXISTS idx_event_type")
        await connection.execute("DROP TABLE IF EXISTS event_store")
