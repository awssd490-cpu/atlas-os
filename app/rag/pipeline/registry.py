"""Pipeline registry.

A global registry that maps pipeline names to ``KnowledgePipeline``
subclasses (not instances).  Applications register their pipeline
types at startup, then instantiate them via the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.pipeline.base import KnowledgePipeline

# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------

_pipelines: dict[str, type[KnowledgePipeline]] = {}


def register(
    name: str,
    pipeline_cls: type[KnowledgePipeline],
) -> None:
    """Register a pipeline class.

    Args:
        name: Unique pipeline name (e.g. ``"standard"``).
        pipeline_cls: The pipeline class (not an instance).

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _pipelines:
        raise ValueError(f"Pipeline {name!r} is already registered")
    _pipelines[name] = pipeline_cls


def unregister(name: str) -> None:
    """Unregister a previously registered pipeline class.

    Args:
        name: The pipeline name to unregister.

    Raises:
        PipelineNotFound: If the pipeline is not registered.
    """
    from app.rag.pipeline.errors import PipelineNotFound

    if name not in _pipelines:
        raise PipelineNotFound(name)
    del _pipelines[name]


def get(name: str) -> type[KnowledgePipeline]:
    """Look up a registered pipeline class by name.

    Args:
        name: The pipeline name.

    Returns:
        The registered pipeline class.

    Raises:
        PipelineNotFound: If the pipeline is not registered.
    """
    from app.rag.pipeline.errors import PipelineNotFound

    try:
        return _pipelines[name]
    except KeyError:
        raise PipelineNotFound(name) from None


def list_pipelines() -> list[str]:
    """Return the names of all registered pipelines."""
    return list(_pipelines)


def clear_pipelines() -> None:
    """Remove all registered pipelines (used in tests)."""
    _pipelines.clear()
