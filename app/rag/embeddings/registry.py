"""Embedding provider registry.

A global registry that maps provider names to ``EmbeddingProvider``
subclasses (not instances).  Applications register their provider
types at startup, then instantiate them via the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.embeddings.base import EmbeddingProvider

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_providers: dict[str, type[EmbeddingProvider]] = {}


def register_provider(
    name: str,
    provider_cls: type[EmbeddingProvider],
) -> None:
    """Register an embedding provider class.

    Args:
        name: Unique provider name (e.g. ``"openai"``).
        provider_cls: The provider class (not an instance).

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _providers:
        raise ValueError(f"Provider {name!r} is already registered")
    _providers[name] = provider_cls


def get_provider(name: str) -> type[EmbeddingProvider]:
    """Look up a registered provider class by name.

    Args:
        name: The provider name.

    Returns:
        The registered provider class.

    Raises:
        UnsupportedEmbeddingProvider: If the provider is not registered.
    """
    from app.rag.embeddings.errors import UnsupportedEmbeddingProvider

    try:
        return _providers[name]
    except KeyError:
        raise UnsupportedEmbeddingProvider(name) from None


def list_providers() -> list[str]:
    """Return the names of all registered providers."""
    return list(_providers)


def clear_providers() -> None:
    """Remove all registered providers (used in tests)."""
    _providers.clear()
