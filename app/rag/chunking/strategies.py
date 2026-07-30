"""Built-in chunking strategies.

This module defines the available strategy names and provides the
registration machinery.  Actual algorithm implementations are added
in a later checkpoint — for now only stub functions exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.rag.chunking.base import ChunkResult
from app.rag.chunking.config import ChunkingConfig

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_strategies: dict[str, Callable[[str, ChunkingConfig], ChunkResult]] = {}


def register_strategy(
    name: str,
    func: Callable[[str, ChunkingConfig], ChunkResult],
) -> None:
    """Register a chunking strategy implementation.

    Args:
        name: Unique strategy name (e.g. ``"fixed_size"``).
        func: The strategy function.

    Raises:
        ValueError: If *name* is already registered.
    """
    if name in _strategies:
        raise ValueError(f"Strategy {name!r} is already registered")
    _strategies[name] = func


def get_strategy(name: str) -> Callable[[str, ChunkingConfig], ChunkResult]:
    """Look up a registered strategy by name.

    Args:
        name: The strategy name.

    Returns:
        The registered strategy function.

    Raises:
        UnsupportedStrategyError: If the strategy is not registered.
    """
    from app.rag.chunking.errors import UnsupportedStrategyError

    try:
        return _strategies[name]
    except KeyError:
        raise UnsupportedStrategyError(name) from None


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return list(_strategies)


def clear_strategies() -> None:
    """Remove all registered strategies (used in tests)."""
    _strategies.clear()


# ---------------------------------------------------------------------------
# Available strategy names (constants)
# ---------------------------------------------------------------------------

STRATEGY_FIXED_SIZE = "fixed_size"
STRATEGY_RECURSIVE = "recursive"
STRATEGY_SENTENCE = "sentence"

# ---------------------------------------------------------------------------
# Stub implementations (placeholder — real algorithms added later)
# ---------------------------------------------------------------------------


def _stub_chunk(text: str, config: ChunkingConfig) -> ChunkResult:
    """Stub chunker that returns a single chunk of the entire text.

    This is a temporary stand-in.  Real implementations will be added
    in a later checkpoint.
    """
    from app.rag.chunking.metadata import ChunkMetadata
    from app.rag.models import KnowledgeChunk

    chunk = KnowledgeChunk(
        chunk_id="",
        document_id="",
        content=text.strip() if config.strip_whitespace else text,
        index=0,
    )
    meta = ChunkMetadata(
        start_char=0,
        end_char=len(text),
        index=0,
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    return ChunkResult(
        chunks=(chunk,),
        metadata=(meta,),
        config=config,
        total_chunks=1,
        original_length=len(text),
    )


# Register the stub under all known strategy names so the engine can
# resolve them without error.  These are replaced when the real
# implementations land.
register_strategy(STRATEGY_FIXED_SIZE, _stub_chunk)
register_strategy(STRATEGY_RECURSIVE, _stub_chunk)
register_strategy(STRATEGY_SENTENCE, _stub_chunk)
