"""V002: Memory system schema.

Creates the memories, memory_relationships, and memory_snapshots tables
used by the Memory Engine.
"""

from __future__ import annotations

from app.storage.interfaces import Migration, SQLConnection


class V002_MemorySchema(Migration):
    """Foundation schema for the memory system.

    Tables:
        memories — core memory records with all metadata fields
        memory_relationships — directed edges between memories
        memory_snapshots — named point-in-time captures
    """

    @property
    def version(self) -> str:
        return "V002"

    async def up(self, connection: SQLConnection) -> None:
        """Create memory tables."""
        # Core memories table
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                memory_type     TEXT NOT NULL DEFAULT 'short_term',
                namespace       TEXT NOT NULL DEFAULT 'default',
                content         TEXT NOT NULL DEFAULT '',
                importance      REAL NOT NULL DEFAULT 0.5,
                confidence      REAL NOT NULL DEFAULT 1.0,
                ttl             REAL,
                state           TEXT NOT NULL DEFAULT 'active',
                source          TEXT NOT NULL DEFAULT 'manual',
                owner           TEXT NOT NULL DEFAULT 'system',
                tags            TEXT NOT NULL DEFAULT '',
                metadata        TEXT NOT NULL DEFAULT '{}',
                correlation_id  TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
                accessed_at     TEXT NOT NULL DEFAULT (datetime('now')),
                archived_at     TEXT,
                forgotten_at    TEXT,
                deleted_at      TEXT,
                access_count    INTEGER NOT NULL DEFAULT 0,
                version         INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        # Indexes for common query patterns
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_ns ON memories(namespace)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_ns_state ON memories(namespace, state)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_corr ON memories(correlation_id)"
        )

        # Memory relationships table
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                rel_type    TEXT NOT NULL,
                properties  TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            )
            """
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_source ON memory_relationships(source_id)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_target ON memory_relationships(target_id)"
        )
        await connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_mr_type ON memory_relationships(rel_type)"
        )

        # Memory snapshots table
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_snapshots (
                id          TEXT PRIMARY KEY,
                label       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                data        TEXT NOT NULL
            )
            """
        )

    async def down(self, connection: SQLConnection) -> None:
        """Drop memory tables."""
        await connection.execute("DROP TABLE IF EXISTS memory_snapshots")
        await connection.execute("DROP TABLE IF EXISTS memory_relationships")
        await connection.execute("DROP TABLE IF EXISTS memories")
