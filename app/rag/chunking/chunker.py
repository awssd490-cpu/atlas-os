"""ChunkingEngine — the public entry point for document chunking.

The engine owns the chunking lifecycle:
  1. Validate configuration.
  2. Resolve the strategy implementation.
  3. Execute the strategy.
  4. Post-process chunks (assign document IDs and chunk IDs).
  5. Return ``ChunkResult``.

It does NOT store, register, or persist chunks — it only produces them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from app.rag.chunking.base import ChunkResult
from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.errors import ChunkingEngineError, UnsupportedStrategyError
from app.rag.chunking.metadata import ChunkMetadata
from app.rag.chunking.strategies import (
    _register_builtins,
    get_strategy,
    list_strategies,
    register_strategy,
)
from app.rag.models import KnowledgeChunk


def _make_chunk_id(strategy: str, document_id: str, index: int) -> str:
    """Generate a deterministic chunk ID.

    The ID encodes the strategy name, optional document ID, and chunk
    position so it is reproducible given the same inputs.
    """
    if document_id:
        return f"{strategy}:{document_id}:{index}"
    return f"{strategy}:{index}"


class ChunkingEngine:
    """Entry point for chunking documents.

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
            strategy_fn = get_strategy(resolved_config.strategy)
        except UnsupportedStrategyError:
            available = list_strategies()
            raise ChunkingEngineError(
                f"Unsupported strategy: {resolved_config.strategy!r}. "
                f"Available strategies: {available}",
                details={"requested": resolved_config.strategy, "available": available},
            ) from None

        try:
            result = strategy_fn(text, resolved_config)
        except Exception as exc:
            raise ChunkingEngineError(
                f"Strategy {resolved_config.strategy!r} failed",
                details={"strategy": resolved_config.strategy},
            ) from exc

        # Post-process: assign deterministic chunk IDs and document IDs,
        # and ensure metadata is consistent.
        new_chunks: list[KnowledgeChunk] = []
        new_metadatas: list[ChunkMetadata] = []

        for i, chunk in enumerate(result.chunks):
            cid = _make_chunk_id(resolved_config.strategy, document_id, i)
            did = document_id or chunk.document_id

            # Update metadata with engine-assigned fields
            meta = (
                result.metadata[i]
                if i < len(result.metadata)
                else ChunkMetadata()
            )
            meta = dataclasses.replace(
                meta,
                document_id=did,
                chunk_index=i,
            )

            new_chunks.append(
                KnowledgeChunk(
                    chunk_id=cid,
                    document_id=did,
                    content=chunk.content,
                    index=i,
                    metadata=meta,
                )
            )
            new_metadatas.append(meta)

        return ChunkResult(
            chunks=tuple(new_chunks),
            metadata=tuple(new_metadatas),
            config=result.config,
            total_chunks=len(new_chunks),
            original_length=result.original_length,
        )

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

        Resets the default config to factory defaults and restores the
        built-in strategy set.  Any dynamically registered strategies
        are cleared.
        """
        self._config = ChunkingConfig()
        _register_builtins()
