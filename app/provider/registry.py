"""Provider registry — central directory of all available providers.

Providers register themselves here by name.  The registry supports
discovery, lookup, and capability-based filtering.
"""

from __future__ import annotations

from typing import Any

from app.provider.errors import DuplicateProviderError, ProviderNotFoundError
from app.provider.provider import Provider


class ProviderRegistry:
    """Central registry of all available providers.

    Thread-safe for concurrent reads.  Registration is typically done
    once at startup.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._default: str = ""

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, provider: Provider, *, default: bool = False) -> None:
        """Register *provider* under *name*.

        Args:
            name: Canonical name (e.g. ``"claude"``, ``"gpt"``).
            provider: The provider instance.
            default: If ``True``, set as the default provider.

        Raises:
            DuplicateProviderError: If *name* is already registered.
        """
        if name in self._providers:
            raise DuplicateProviderError(
                name=name,
                details={"existing_type": type(self._providers[name]).__name__},
            )
        self._providers[name] = provider
        if default or not self._default:
            self._default = name

    def unregister(self, name: str) -> None:
        """Remove a provider from the registry.

        Args:
            name: The provider name to remove.

        Raises:
            ProviderNotFoundError: If *name* is not registered.
        """
        if name not in self._providers:
            raise ProviderNotFoundError(name=name)
        del self._providers[name]
        if self._default == name:
            # Reassign default to the next registered provider, if any
            self._default = next(iter(self._providers)) if self._providers else ""

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> Provider:
        """Look up a provider by name.

        Args:
            name: The provider name.

        Returns:
            The ``Provider`` instance.

        Raises:
            ProviderNotFoundError: If *name* is not registered.
        """
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderNotFoundError(
                name=name,
                details={"available": list(self._providers.keys())},
            )
        return provider

    def default_provider(self) -> Provider:
        """Return the default provider.

        Raises:
            ProviderNotFoundError: If no providers are registered.
        """
        if not self._default:
            raise ProviderNotFoundError(name="(default)")
        return self._providers[self._default]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers.keys())

    def find_by_capability(self, capability: str) -> list[tuple[str, Provider]]:
        """Find all providers that support *capability*.

        Returns:
            List of ``(name, provider)`` tuples.
        """
        return [
            (name, p)
            for name, p in self._providers.items()
            if p.supports_capability(capability)
        ]

    def count(self) -> int:
        """Return the number of registered providers."""
        return len(self._providers)

    @property
    def default_name(self) -> str:
        """Return the default provider name."""
        return self._default

    @default_name.setter
    def default_name(self, name: str) -> None:
        """Set the default provider by name.

        Raises:
            ProviderNotFoundError: If *name* is not registered.
        """
        if name not in self._providers:
            raise ProviderNotFoundError(
                name=name,
                details={"available": list(self._providers.keys())},
            )
        self._default = name
