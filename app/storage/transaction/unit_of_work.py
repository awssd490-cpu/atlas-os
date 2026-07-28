"""Unit of Work implementation.

A UnitOfWork wraps a connection and provides transaction commit/rollback
semantics.  Multiple repository operations share one UoW, guaranteeing
atomicity.
"""

from __future__ import annotations

from typing import Any

from app.storage.interfaces import SQLConnection, UnitOfWork, UnitOfWorkFactory


class SqliteUnitOfWork(UnitOfWork):
    """Unit of Work backed by a SQLite connection.

    Usage::

        async with uow:
            repo = WidgetRepo(uow.connection)
            await repo.add(widget)
            await uow.commit()
    """

    def __init__(self, connection: SQLConnection) -> None:
        self._connection = connection
        self._committed = False
        self._rolled_back = False

    async def commit(self) -> None:
        """Persist all changes made within this UoW."""
        await self._connection.execute("COMMIT")
        self._committed = True

    async def rollback(self) -> None:
        """Discard all changes made within this UoW."""
        await self._connection.execute("ROLLBACK")
        self._rolled_back = True

    async def flush(self) -> None:
        """Emit pending writes without committing.

        SQLite doesn't have a flush concept distinct from transaction
        state.  This is a no-op for SQLite but exists for API
        compatibility with PostgreSQL-level UoWs.
        """

    @property
    def connection(self) -> SQLConnection:
        """Return the underlying connection for repositories to use."""
        return self._connection

    @property
    def is_done(self) -> bool:
        """Return True if commit or rollback has been called."""
        return self._committed or self._rolled_back

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SqliteUnitOfWork":
        """Begin a transaction."""
        await self._connection.execute("BEGIN")
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Auto-rollback on exit if not explicitly committed."""
        if not self._committed and not self._rolled_back:
            await self.rollback()
        # Release any resources


class SqliteUnitOfWorkFactory(UnitOfWorkFactory):
    """Creates SqliteUnitOfWork instances around a shared engine."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def create(self) -> UnitOfWork:
        """Create a new UoW backed by a connection from the engine."""
        conn = await self._engine.connection()
        return SqliteUnitOfWork(connection=conn)
