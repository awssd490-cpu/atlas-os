"""SQLite-specific repository mixin and helpers.

Provides ``SqliteRepository`` — a concrete ``BaseRepositoryImpl`` that
uses a ``SQLiteConnection`` as its transport.
"""

from __future__ import annotations

from typing import Any, Generic

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.repository.base import BaseRepositoryImpl, TModel, TId


class SqliteRepository(BaseRepositoryImpl[TModel, TId], Generic[TModel, TId]):
    """A repository backed by an SQLite connection.

    Example::

        class MyRepo(SqliteRepository[MyModel, str]):
            @property
            def _table(self) -> str: return "my_models"
            @property
            def _id_field(self) -> str: return "id"

            def _model_to_row(self, m): return {...}
            def _row_to_model(self, r): return MyModel(...)
    """

    def __init__(self, connection: SQLiteConnection) -> None:
        self._connection = connection
