"""Evaluation runner registry.

A global registry that maps runner names to ``EvaluationRunner``
subclasses (not instances).  Applications register their runner
types at startup, then instantiate them via the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.evaluation.base import EvaluationRunner

# ---------------------------------------------------------------------------
# Runner registry
# ---------------------------------------------------------------------------

_runners: dict[str, type[EvaluationRunner]] = {}


def register(
    name: str,
    runner_cls: type[EvaluationRunner],
) -> None:
    """Register an evaluation runner class.

    Args:
        name: Unique runner name (e.g. ``"retrieval"``, ``"reranking"``).
        runner_cls: The runner class (not an instance).

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _runners:
        raise ValueError(f"Evaluation runner {name!r} is already registered")
    _runners[name] = runner_cls


def unregister(name: str) -> None:
    """Unregister a previously registered evaluation runner class.

    Args:
        name: The runner name to unregister.

    Raises:
        EvaluationNotFound: If the runner is not registered.
    """
    from app.rag.evaluation.errors import EvaluationNotFound

    if name not in _runners:
        raise EvaluationNotFound(name)
    del _runners[name]


def get(name: str) -> type[EvaluationRunner]:
    """Look up a registered evaluation runner class by name.

    Args:
        name: The runner name.

    Returns:
        The registered runner class.

    Raises:
        EvaluationNotFound: If the runner is not registered.
    """
    from app.rag.evaluation.errors import EvaluationNotFound

    try:
        return _runners[name]
    except KeyError:
        raise EvaluationNotFound(name) from None


def list_runners() -> list[str]:
    """Return the names of all registered evaluation runners."""
    return list(_runners)


def clear_runners() -> None:
    """Remove all registered evaluation runners (used in tests)."""
    _runners.clear()
