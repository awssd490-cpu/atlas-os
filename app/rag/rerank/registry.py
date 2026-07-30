"""Reranker provider registry.

A global registry that maps reranker names to ``Reranker`` subclasses
(not instances).  Applications register their reranker types at
startup, then instantiate them via the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.rerank.base import Reranker

# ---------------------------------------------------------------------------
# Reranker registry
# ---------------------------------------------------------------------------

_rerankers: dict[str, type[Reranker]] = {}


def register_reranker(
    name: str,
    reranker_cls: type[Reranker],
) -> None:
    """Register a reranker class.

    Args:
        name: Unique reranker name (e.g. ``"cross_encoder"``).
        reranker_cls: The reranker class (not an instance).

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _rerankers:
        raise ValueError(f"Reranker {name!r} is already registered")
    _rerankers[name] = reranker_cls


def get_reranker(name: str) -> type[Reranker]:
    """Look up a registered reranker class by name.

    Args:
        name: The reranker name.

    Returns:
        The registered reranker class.

    Raises:
        RerankerNotFound: If the reranker is not registered.
    """
    from app.rag.rerank.errors import RerankerNotFound

    try:
        return _rerankers[name]
    except KeyError:
        raise RerankerNotFound(name) from None


def list_rerankers() -> list[str]:
    """Return the names of all registered rerankers."""
    return list(_rerankers)


def clear_rerankers() -> None:
    """Remove all registered rerankers (used in tests)."""
    _rerankers.clear()
