"""Tests for SQLiteStorageEngine.

Verifies:
- Connect/disconnect lifecycle
- ``connection()`` returns a usable connection
- ``connection()`` raises before connect
- ``is_healthy`` returns True when connected
- ``is_healthy`` returns False when not connected
"""

from __future__ import annotations

import pytest

from app.storage.errors import ConnectionError_
from app.storage.engine.sqlite import SQLiteStorageEngine


class TestLifecycle:
    async def test_connect_disconnect(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        assert engine.path == ":memory:"
        await engine.connect()
        await engine.disconnect()

    async def test_double_connect_safe(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        await engine.connect()
        await engine.connect()  # should not raise
        await engine.disconnect()


class TestConnection:
    async def test_connection_returns_usable(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        await engine.connect()
        conn = await engine.connection()
        row = await conn.fetchone("SELECT 1 AS val")
        assert row is not None
        assert row["val"] == 1
        await engine.disconnect()

    async def test_connection_before_connect_raises(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        with pytest.raises(ConnectionError_, match="not connected"):
            await engine.connection()


class TestHealth:
    async def test_healthy_when_connected(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        await engine.connect()
        assert await engine.is_healthy() is True
        await engine.disconnect()

    async def test_unhealthy_when_disconnected(self) -> None:
        engine = SQLiteStorageEngine(":memory:")
        assert await engine.is_healthy() is False
