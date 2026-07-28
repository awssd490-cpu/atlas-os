"""SQLiteStorageEngine — manages SQLite connections.

Provides a pool of :class:`SQLiteConnection` instances.  Currently the
pool size is 1 (SQLite is single-writer; multiple connections would just
contend on the WAL).
"""

from __future__ import annotations

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.engine.base import BaseStorageEngine
from app.storage.interfaces import SQLConnection


class SQLiteStorageEngine(BaseStorageEngine):
    """SQLite-backed storage engine.

    Usage::

        engine = SQLiteStorageEngine(path="data/atlas.db")
        await engine.connect()
        conn = await engine.connection()
        await conn.execute("CREATE TABLE ...")
        await engine.disconnect()
    """

    def __init__(self, path: str = "data/atlas.db") -> None:
        self._path = path
        self._connection: SQLiteConnection | None = None

    async def connect(self) -> None:
        """Open the SQLite connection (no pool — single connection)."""
        if self._connection is not None:
            return  # already connected
        self._connection = SQLiteConnection(self._path)
        # Force the connection to open by executing a trivial query
        await self._connection.execute("SELECT 1")

    async def disconnect(self) -> None:
        """Close the connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def connection(self) -> SQLConnection:
        """Return the shared connection.

        Raises :class:`ConnectionError_` if not yet connected.
        """
        if self._connection is None:
            from app.storage.errors import ConnectionError_

            raise ConnectionError_(
                "Storage engine not connected",
                details={"path": self._path},
            )
        return self._connection

    @property
    def path(self) -> str:
        return self._path
