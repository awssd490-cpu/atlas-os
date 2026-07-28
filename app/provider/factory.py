"""Provider factory — constructs provider instances with dependency injection.

The factory handles configuration validation, lifecycle, and construction
of provider instances.  Callers request a provider by name and receive a
fully configured instance.
"""

from __future__ import annotations

from typing import Any

from app.provider.errors import ProviderError, ProviderNotFoundError
from app.provider.provider import Provider
from app.provider.registry import ProviderRegistry


class ProviderFactory:
    """Constructs provider instances with configuration injection.

    Usage::

        factory = ProviderFactory(registry)
        provider = await factory.create("claude", config={...})
        await provider.initialize()
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry
        # Internal registry of (name → callable) constructors
        self._constructors: dict[str, type[Provider]] = {}

    @property
    def registry(self) -> ProviderRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Constructor registration
    # ------------------------------------------------------------------

    def register_constructor(self, name: str, cls: type[Provider]) -> None:
        """Register a provider class for construction.

        Args:
            name: The provider name (e.g. ``"claude"``).
            cls: The provider class (must implement ``Provider``).

        Raises:
            ValueError: If *cls* does not implement ``Provider``.
        """
        if not issubclass(cls, Provider):
            raise ValueError(f"{cls.__name__} does not implement the Provider ABC")
        self._constructors[name] = cls

    def unregister_constructor(self, name: str) -> None:
        """Remove a registered constructor."""
        self._constructors.pop(name, None)

    def list_constructors(self) -> list[str]:
        """Return all registered constructor names."""
        return list(self._constructors.keys())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    async def create(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        register: bool = True,
        set_default: bool = False,
    ) -> Provider:
        """Create and optionally register a provider instance.

        Args:
            name: The provider name.
            config: Configuration to pass to the provider constructor.
            register: If ``True``, register the instance in the registry.
            set_default: If ``True``, set as the default provider.

        Returns:
            A fully constructed ``Provider`` instance (not yet initialized).

        Raises:
            ProviderNotFoundError: If no constructor is registered for *name*.
        """
        cls = self._constructors.get(name)
        if cls is None:
            raise ProviderNotFoundError(
                name=name,
                details={"available_constructors": list(self._constructors.keys())},
            )

        instance = cls(**(config or {}))
        if register:
            self._registry.register(name, instance, default=set_default)
        return instance

    async def create_and_initialize(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        register: bool = True,
        set_default: bool = False,
    ) -> Provider:
        """Create, optionally register, and initialize a provider.

        Calls ``initialize()`` on the instance before returning.
        """
        instance = await self.create(
            name,
            config=config,
            register=register,
            set_default=set_default,
        )
        await instance.initialize()
        return instance

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def initialize_all(self) -> None:
        """Initialize every registered provider."""
        for name in self._registry.list_providers():
            provider = self._registry.lookup(name)
            await provider.initialize()

    async def shutdown_all(self) -> None:
        """Shutdown every registered provider."""
        errors: list[tuple[str, Exception]] = []
        for name in self._registry.list_providers():
            try:
                provider = self._registry.lookup(name)
                await provider.shutdown()
            except Exception as exc:
                errors.append((name, exc))
        if errors:
            msg = "; ".join(f"{n}: {e}" for n, e in errors)
            raise ProviderError(f"Shutdown errors: {msg}")

    async def create_default(
        self,
        config: dict[str, Any] | None = None,
    ) -> Provider:
        """Create and initialize the default provider using its constructor.

        The *name* is taken from the registry's ``default_name``.
        """
        name = self._registry.default_name
        if not name:
            raise ProviderNotFoundError(
                name="(default)",
                details={"message": "No default provider name configured"},
            )
        return await self.create_and_initialize(name, config=config, register=True)
