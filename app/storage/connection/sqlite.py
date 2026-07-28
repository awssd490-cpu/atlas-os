"""SQLite connection implementation.

Wraps the stdlib ``sqlite3`` module and dispatches all I/O through
``asyncio.to_thread()`` so the event loop is never blocked.

SQLite connections are single-writer by design (serialized at the
connection level).  Using a thread pool adds no contention beyond what
SQLite itself imposes, while keeping the event loop free.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

from app.storage.errors import ConnectionError_
from app.storage.interfaces import Row, SQLConnection


class SQLiteConnection(SQLConnection):
    """Async SQLite connection backed by stdlib ``sqlite3``.

    Usage::

        conn = SQLiteConnection(":memory:")
        await conn.execute("CREATE TABLE t (x)")
        row = await conn.fetchone("SELECT * FROM t WHERE x = ?", [1])
        await conn.close()
    """

    def __init__(
        self,
        path: str,
        *,
        timeout: float = 5.0,
        journal_mode: str = "WAL",
    ) -> None:
        self._path = path
        self._timeout = timeout
        self._journal_mode = journal_mode
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_open(self) -> sqlite3.Connection:
        """Return the underlying connection, opening it lazily if needed."""
        if self._conn is None:
            try:
                self._conn = await asyncio.to_thread(self._open_sync)
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"Failed to open SQLite database at '{self._path}'",
                    details={"path": self._path, "error": str(exc)},
                ) from exc
        return self._conn

    def _open_sync(self) -> sqlite3.Connection:
        """Synchronous connection opener (runs in thread pool)."""
        conn = sqlite3.connect(
            self._path,
            timeout=self._timeout,
            check_same_thread=False,  # we manage threading via lock
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode={self._journal_mode}")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # SQLConnection interface
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: dict[str, Any] | list | None = None) -> None:
        """Execute a SQL statement without returning rows."""
        async with self._lock:
            conn = await self._ensure_open()
            try:
                await asyncio.to_thread(self._execute_sync, conn, sql, params)
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"SQL execute failed: {sql[:100]}",
                    details={"sql": sql[:200], "error": str(exc)},
                ) from exc

    async def fetchone(
        self,
        sql: str,
        params: dict[str, Any] | list | None = None,
    ) -> Row | None:
        """Execute a query and return the first row, or ``None``."""
        async with self._lock:
            conn = await self._ensure_open()
            try:
                row = await asyncio.to_thread(self._fetchone_sync, conn, sql, params)
                return Row(row) if row is not None else None
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"SQL fetchone failed: {sql[:100]}",
                    details={"sql": sql[:200], "error": str(exc)},
                ) from exc

    async def fetchall(
        self,
        sql: str,
        params: dict[str, Any] | list | None = None,
    ) -> list[Row]:
        """Execute a query and return all matching rows."""
        async with self._lock:
            conn = await self._ensure_open()
            try:
                rows = await asyncio.to_thread(self._fetchall_sync, conn, sql, params)
                return [Row(r) for r in rows]
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"SQL fetchall failed: {sql[:100]}",
                    details={"sql": sql[:200], "error": str(exc)},
                ) from exc

    async def executemany(self, sql: str, params: list[dict[str, Any] | list]) -> None:
        """Execute the same statement for every parameter set."""
        async with self._lock:
            conn = await self._ensure_open()
            try:
                await asyncio.to_thread(self._executemany_sync, conn, sql, params)
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"SQL executemany failed: {sql[:100]}",
                    details={"sql": sql[:200], "error": str(exc)},
                ) from exc

    async def execute_script(self, sql: str) -> None:
        """Execute a multi-statement script (DDL batches).

        Uses ``executescript`` which handles multiple statements separated
        by semicolons.  This is NOT safe for user-supplied SQL.
        """
        async with self._lock:
            conn = await self._ensure_open()
            try:
                await asyncio.to_thread(self._script_sync, conn, sql)
            except sqlite3.Error as exc:
                raise ConnectionError_(
                    f"SQL execute_script failed: {sql[:100]}",
                    details={"sql": sql[:200], "error": str(exc)},
                ) from exc

    async def close(self) -> None:
        """Close the connection."""
        if self._closed:
            return
        self._closed = True
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None

    @property
    def is_closed(self) -> bool:
        """Return ``True`` after the connection has been closed."""
        return self._closed

    # ------------------------------------------------------------------
    # Synchronous helpers (run in thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_sync(
        conn: sqlite3.Connection,
        sql: str,
        params: dict[str, Any] | list | None,
    ) -> None:
        conn.execute(sql, params or ())

    @staticmethod
    def _fetchone_sync(
        conn: sqlite3.Connection,
        sql: str,
        params: dict[str, Any] | list | None,
    ) -> Any:
        return conn.execute(sql, params or ()).fetchone()

    @staticmethod
    def _fetchall_sync(
        conn: sqlite3.Connection,
        sql: str,
        params: dict[str, Any] | list | None,
    ) -> list[Any]:
        return conn.execute(sql, params or ()).fetchall()

    @staticmethod
    def _executemany_sync(
        conn: sqlite3.Connection,
        sql: str,
        params: list[dict[str, Any] | list],
    ) -> None:
        conn.executemany(sql, params)

    @staticmethod
    def _script_sync(conn: sqlite3.Connection, sql: str) -> None:
        conn.executescript(sql)
