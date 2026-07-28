"""Dependency-injection container for ATLAS.

**Responsibility:** manage service factories (sync and async), singleton
caching, and graceful disposal.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.errors import DependencyResolutionError
from app.core.interfaces import DIContainer

T = type


class Container(DIContainer):
    """Lightweight, explicit DI container.

    Usage::

        container = Container()
        container.register(ConfigService, lambda c: PydanticConfigService())
        container.register(EventBus, lambda c: InProcessEventBus(), singleton=True)

        config = await container.resolve(ConfigService)
    """

    def __init__(self) -> None:
        self._factories: dict[type, _Registration] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        interface: type[T],
        factory: Callable[["DIContainer"], T | Awaitable[T]],
        *,
        singleton: bool = True,
    ) -> None:
        """Register *factory* for *interface*.

        Args:
            interface: The type to register.
            factory: Callable(container) -> instance.  May be sync or async.
            singleton: If ``True`` (default), the factory runs once and the
                result is cached for all future resolutions.
        """
        if interface in self._factories:
            raise DependencyResolutionError(
                f"Interface '{interface.__name__}' is already registered",
                details={"interface": interface.__name__},
            )
        self._factories[interface] = _Registration(factory=factory, singleton=singleton)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve(self, interface: type[T]) -> T:
        """Resolve an instance of *interface*.

        Raises :class:`DependencyResolutionError` when *interface* has not
        been registered.
        """
        reg = self._factories.get(interface)
        if reg is None:
            raise DependencyResolutionError(
                f"No factory registered for '{interface.__name__}'",
                details={
                    "interface": interface.__name__,
                    "registered": [t.__name__ for t in self._factories],
                },
            )

        # Singleton — resolve once, cache forever
        if reg.singleton and reg._cached is not None:
            return reg._cached  # type: ignore[return-value]

        instance = await self._call_factory(reg.factory)

        if reg.singleton:
            reg._cached = instance

        return instance  # type: ignore[return-value]

    def is_registered(self, interface: type[T]) -> bool:
        """Whether *interface* has a registered factory."""
        return interface in self._factories

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_singletons(self) -> None:
        """Pre-resolve all singletons to surface boot errors early.

        This is useful during kernel startup to catch wiring errors before
        the server starts accepting traffic.
        """
        for interface, reg in self._factories.items():
            if reg.singleton and reg._cached is None:
                instance = await self._call_factory(reg.factory)
                reg._cached = instance

    async def dispose(self) -> None:
        """Dispose all singleton instances that implement ``close()`` or
        ``async shutdown()``."""
        for interface, reg in self._factories.items():
            if reg._cached is not None:
                await self._maybe_dispose(reg._cached)
                reg._cached = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_factory(self, factory: Callable[..., Any]) -> Any:
        """Call *factory*, handling both sync and async functions."""
        result = factory(self)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _maybe_dispose(self, instance: Any) -> None:
        """Call ``shutdown()`` (async) or ``close()`` (sync) if they exist."""
        if hasattr(instance, "shutdown"):
            maybe = instance.shutdown()
            if inspect.isawaitable(maybe):
                await maybe
        elif hasattr(instance, "close"):
            maybe = instance.close()
            if inspect.isawaitable(maybe):
                await maybe


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _Registration:
    """Holds a factory function and its singleton state."""

    __slots__ = ("factory", "singleton", "_cached")

    def __init__(
        self,
        factory: Callable[..., Any],
        singleton: bool,
    ) -> None:
        self.factory = factory
        self.singleton = singleton
        self._cached: Any = None
