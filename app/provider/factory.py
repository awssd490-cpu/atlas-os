"""Provider factory — constructs provider instances with dependency injection.

The factory handles configuration validation, lifecycle, and construction
of provider instances.  Callers request a provider by name and receive a
fully configured instance.
"""

from __future__ import annotations

from typing import Any

from app.provider.config import ConfigBuilder, DictConfigSource, ProviderConfig, validate_config
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
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_provider_config(name: str, provider_config: ProviderConfig) -> None:
        """Validate a ``ProviderConfig``, raising on failure."""
        errors: list[str] = []

        if not provider_config.credentials.has_key:
            errors.append("api_key is required")

        temp = provider_config.generation.temperature
        if temp < 0.0 or temp > 2.0:
            errors.append(f"temperature must be in [0.0, 2.0], got {temp}")

        mt = provider_config.generation.max_tokens
        if mt < 1:
            errors.append(f"max_tokens must be >= 1, got {mt}")

        r = provider_config.retry.max_retries
        if r < 0 or r > 20:
            errors.append(f"max_retries must be in [0, 20], got {r}")

        if errors:
            raise ValueError(
                f"Provider {name!r} configuration invalid: "
                f"{'; '.join(errors)}"
            )

    def build_config(self, name: str, config: dict[str, Any] | None = None) -> ProviderConfig:
        """Build a validated ``ProviderConfig`` for *name*.

        Merges the provided *config* dictionary with any environment
        variables (via ``EnvConfigSource``).

        Args:
            name: The provider name.
            config: Optional configuration dictionary.

        Returns:
            A validated ``ProviderConfig``.

        Raises:
            ValueError: If validation fails.
        """
        builder = ConfigBuilder()
        if config:
            builder.add_source(DictConfigSource(config, priority=10))
        provider_config = builder.build(name=name)

        self._validate_provider_config(name, provider_config)

        return provider_config

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    async def create(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        provider_config: ProviderConfig | None = None,
        register: bool = True,
        set_default: bool = False,
    ) -> Provider:
        """Create and optionally register a provider instance.

        Args:
            name: The provider name.
            config: Raw configuration dictionary (alternative to *provider_config*).
            provider_config: Pre-built ``ProviderConfig`` (alternative to *config*).
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

        resolved = provider_config or self.build_config(name, config)
        instance = cls(config=resolved)
        if register:
            self._registry.register(name, instance, default=set_default)
        return instance

    async def create_and_initialize(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        provider_config: ProviderConfig | None = None,
        register: bool = True,
        set_default: bool = False,
    ) -> Provider:
        """Create, optionally register, and initialize a provider.

        Calls ``initialize()`` on the instance before returning.
        """
        instance = await self.create(
            name,
            config=config,
            provider_config=provider_config,
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
        *,
        provider_config: ProviderConfig | None = None,
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
        return await self.create_and_initialize(
            name, config=config, provider_config=provider_config, register=True,
        )
