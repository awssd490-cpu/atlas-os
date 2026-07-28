"""StorageEngine ABC.

Every backend engine implements this interface to manage connection
lifecycle and health.
"""

from __future__ import annotations

import abc

from app.storage.interfaces import SQLConnection, StorageEngine


class BaseStorageEngine(StorageEngine):
    """ABC with default ``is_healthy`` that checks connectivity."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the backing connection(s)."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close all connections and release resources."""

    @abc.abstractmethod
    async def connection(self) -> SQLConnection:
        """Return a usable connection."""

    async def is_healthy(self) -> bool:
        """Check health by executing a trivial query.

        Returns ``False`` on any error rather than raising.
        """
        try:
            conn = await self.connection()
            result = await conn.fetchone("SELECT 1 AS ok")
            return result is not None and result["ok"] == 1
        except Exception:
            return False
