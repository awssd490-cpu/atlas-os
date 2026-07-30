"""ChunkingEngine — the public entry point for document chunking.

The engine owns the chunking lifecycle:
  1. Validate config.
  2. Resolve the strategy implementation.
  3. Execute the strategy.
  4. Wrap the result.

It does NOT store, register, or persist chunks — it only produces them.
"""

from __future__ import annotations

from typing import Any

from app.rag.chunking.base import ChunkResult
from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.errors import ChunkingEngineError, UnsupportedStrategyError
from app.rag.chunking.strategies import (
    STRATEGY_FIXED_SIZE,
    STRATEGY_RECURSIVE,
    STRATEGY_SENTENCE,
    _stub_chunk,
    get_strategy,
    list_strategies,
    register_strategy,
)
from app.rag.models import KnowledgeChunk


class ChunkingEngine:
    """Skeleton chunking engine.

    Usage::

        engine = ChunkingEngine()
        result = engine.chunk("Some long document text...")
        for chunk in result.chunks:
            print(chunk.content)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialise the engine.

        Args:
            config: Default configuration used when ``chunk()`` is called
                without an explicit config.  If omitted, a default
                ``ChunkingConfig()`` is used.
        """
        self._config = config or ChunkingConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ChunkingConfig:
        """Return the engine's default configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        config: ChunkingConfig | None = None,
        *,
        document_id: str = "",
    ) -> ChunkResult:
        """Chunk a document text into ``KnowledgeChunk`` objects.

        Args:
            text: The document text to chunk.
            config: Optional per-call configuration.  Falls back to the
                engine's default config if omitted.
            document_id: Optional document ID to assign to every
                produced chunk.

        Returns:
            A ``ChunkResult`` with the produced chunks and metadata.

        Raises:
            ChunkingEngineError: On any chunking failure.
        """
        resolved_config = config or self._config

        try:
            resolved_config.validate()
        except Exception as exc:
            raise ChunkingEngineError(
                "Invalid chunking configuration",
                details={"strategy": resolved_config.strategy},
            ) from exc

        try:
            strategy = get_strategy(resolved_config.strategy)
        except UnsupportedStrategyError:
            available = list_strategies()
            raise ChunkingEngineError(
                f"Unsupported strategy: {resolved_config.strategy!r}. "
                f"Available strategies: {available}",
                details={"requested": resolved_config.strategy, "available": available},
            ) from None

        try:
            result = strategy(text, resolved_config)
        except Exception as exc:
            raise ChunkingEngineError(
                f"Strategy {resolved_config.strategy!r} failed",
                details={"strategy": resolved_config.strategy},
            ) from exc

        # Assign document_id to every chunk if provided
        if document_id:
            chunks = tuple(
                chunk
                if chunk.document_id
                else KnowledgeChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    content=chunk.content,
                    index=chunk.index,
                    metadata=chunk.metadata,
                )
                for chunk in result.chunks
            )
            return ChunkResult(
                chunks=chunks,
                metadata=result.metadata,
                config=result.config,
                total_chunks=result.total_chunks,
                original_length=result.original_length,
            )

        return result

    # ------------------------------------------------------------------
    # Strategy management
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        name: str,
        strategy: Any,
    ) -> None:
        """Register an additional strategy at runtime.

        Args:
            name: Strategy name.
            strategy: Callable following the ``ChunkingStrategy`` protocol.

        Raises:
            ValueError: If the name is already registered.
        """
        register_strategy(name, strategy)

    def available_strategies(self) -> list[str]:
        """Return the names of all currently registered strategies."""
        return list_strategies()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the engine to its initial state.

        Resets the default config and re-registers the built-in stub
        strategies.  Any dynamically registered strategies are cleared.
        """
        from app.rag.chunking.strategies import clear_strategies

        self._config = ChunkingConfig()
        clear_strategies()
        # Re-register built-in stubs
        register_strategy(STRATEGY_FIXED_SIZE, _stub_chunk)
        register_strategy(STRATEGY_RECURSIVE, _stub_chunk)
        register_strategy(STRATEGY_SENTENCE, _stub_chunk)
