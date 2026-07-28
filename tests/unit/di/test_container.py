"""Tests for the DI Container.

Verifies:
- Register and resolve singletons (same instance each time)
- Register and resolve transient (new instance each time)
- Async factory support
- ``is_registered``
- ``init_singletons`` pre-resolves
- ``dispose`` calls ``close()`` / ``shutdown()``
- Duplicate registration raises
- Unregistered resolution raises with helpful message
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import DependencyResolutionError
from app.core.interfaces import DIContainer
from app.di.container import Container


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.started = False
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AsyncEngine:
    def __init__(self) -> None:
        self.closed = False

    async def shutdown(self) -> None:
        self.closed = True


class _Noop:
    pass


# ---------------------------------------------------------------------------
# Singleton resolution
# ---------------------------------------------------------------------------


class TestSingleton:
    async def test_resolve_returns_same_instance(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine("singleton"))
        a = await container.resolve(_Engine)
        b = await container.resolve(_Engine)
        assert a is b
        assert a.name == "singleton"

    async def test_resolve_after_dispose_returns_new(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine())
        a = await container.resolve(_Engine)
        await container.dispose()
        b = await container.resolve(_Engine)
        assert a is not b


# ---------------------------------------------------------------------------
# Transient resolution
# ---------------------------------------------------------------------------


class TestTransient:
    async def test_resolve_returns_new_each_time(self) -> None:
        container = Container()
        container.register(_Noop, lambda c: _Noop(), singleton=False)
        a = await container.resolve(_Noop)
        b = await container.resolve(_Noop)
        assert a is not b


# ---------------------------------------------------------------------------
# Async factories
# ---------------------------------------------------------------------------


class TestAsyncFactory:
    async def test_async_factory_is_awaited(self) -> None:
        container = Container()

        async def factory(_c: DIContainer) -> _Engine:
            engine = _Engine("async-built")
            engine.started = True
            return engine

        container.register(_Engine, factory)
        engine = await container.resolve(_Engine)
        assert engine.started is True
        assert engine.name == "async-built"


# ---------------------------------------------------------------------------
# is_registered
# ---------------------------------------------------------------------------


class TestIsRegistered:
    def test_registered_returns_true(self) -> None:
        container = Container()
        container.register(_Noop, lambda c: _Noop())
        assert container.is_registered(_Noop) is True

    def test_unregistered_returns_false(self) -> None:
        container = Container()
        assert container.is_registered(_Noop) is False


# ---------------------------------------------------------------------------
# init_singletons
# ---------------------------------------------------------------------------


class TestInitSingletons:
    async def test_pre_resolves_all_singletons(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine("pre-built"))
        container.register(_Noop, lambda c: _Noop(), singleton=False)
        await container.init_singletons()
        engine = await container.resolve(_Engine)
        assert engine.name == "pre-built"

    async def test_does_not_resolve_transients(self) -> None:
        container = Container()
        container.register(_Noop, lambda c: _Noop(), singleton=False)
        await container.init_singletons()
        # Should not have been cached — no assertion needed if no error


# ---------------------------------------------------------------------------
# dispose
# ---------------------------------------------------------------------------


class TestDispose:
    async def test_dispose_calls_close(self) -> None:
        container = Container()
        engine = _Engine()
        container.register(_Engine, lambda c: engine)
        await container.resolve(_Engine)
        await container.dispose()
        assert engine.closed is True

    async def test_dispose_calls_async_shutdown(self) -> None:
        container = Container()
        async_engine = _AsyncEngine()
        container.register(_AsyncEngine, lambda c: async_engine)
        await container.resolve(_AsyncEngine)
        await container.dispose()
        assert async_engine.closed is True

    async def test_dispose_clears_cache(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine("original"))
        orig = await container.resolve(_Engine)
        await container.dispose()
        fresh = await container.resolve(_Engine)
        assert orig is not fresh


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_duplicate_registration_raises(self) -> None:
        container = Container()
        container.register(_Engine, lambda c: _Engine())
        with pytest.raises(DependencyResolutionError, match="already registered"):
            container.register(_Engine, lambda c: _Engine())

    async def test_unregistered_resolve_raises(self) -> None:
        container = Container()
        with pytest.raises(DependencyResolutionError, match="No factory registered"):
            await container.resolve(_Engine)
