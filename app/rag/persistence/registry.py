"""Persistence backend registry.

A global registry that maps backend names to ``PersistenceBackend``
subclasses (not instances).  Applications register their backend
types at startup, then instantiate them via the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.persistence.base import PersistenceBackend

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_backends: dict[str, type[PersistenceBackend]] = {}


def register(
    name: str,
    backend_cls: type[PersistenceBackend],
) -> None:
    """Register a persistence backend class.

    Args:
        name: Unique backend name (e.g. ``"json"``, ``"sqlite"``).
        backend_cls: The backend class (not an instance).

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _backends:
        raise ValueError(f"Persistence backend {name!r} is already registered")
    _backends[name] = backend_cls


def unregister(name: str) -> None:
    """Unregister a previously registered persistence backend class.

    Args:
        name: The backend name to unregister.

    Raises:
        PersistenceNotFound: If the backend is not registered.
    """
    from app.rag.persistence.errors import PersistenceNotFound

    if name not in _backends:
        raise PersistenceNotFound(name)
    del _backends[name]


def get(name: str) -> type[PersistenceBackend]:
    """Look up a registered persistence backend class by name.

    Args:
        name: The backend name.

    Returns:
        The registered backend class.

    Raises:
        PersistenceNotFound: If the backend is not registered.
    """
    from app.rag.persistence.errors import PersistenceNotFound

    try:
        return _backends[name]
    except KeyError:
        raise PersistenceNotFound(name) from None


def list_backends() -> list[str]:
    """Return the names of all registered persistence backends."""
    return list(_backends)


def clear_backends() -> None:
    """Remove all registered persistence backends (used in tests)."""
    _backends.clear()
