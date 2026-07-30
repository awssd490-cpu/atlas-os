"""Chunking configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for the chunking engine.

    Attributes:
        strategy: The chunking strategy name (e.g. ``"fixed_size"``,
            ``"recursive"``, ``"sentence"``).  Default ``"fixed_size"``.
        chunk_size: Maximum number of characters per chunk.  Default 512.
        chunk_overlap: Number of characters to overlap between consecutive
            chunks.  Default 64.
        min_chunk_size: Minimum number of characters a chunk must contain.
            Default 32.
        separator: Separator string used by some strategies (e.g.
            ``"\\n\\n"`` for paragraph splitting).  Default ``""``.
        secondary_separators: Ordered list of fallback separators used by
            recursive strategies.  Default ``[]``.
        max_chunks: Hard limit on the number of chunks produced per
            document.  ``0`` means no limit.  Default 0.
        strip_whitespace: Whether to strip leading/trailing whitespace
            from each chunk.  Default ``True``.
        strategy_params: Additional strategy-specific keyword arguments
            forwarded to the strategy implementation.
    """

    strategy: str = "fixed_size"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 32
    separator: str = ""
    secondary_separators: tuple[str, ...] = ()
    max_chunks: int = 0
    strip_whitespace: bool = True
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ChunkingConfigError: If any value is out of range or invalid.
        """
        from app.rag.chunking.errors import ChunkingConfigError

        if self.chunk_size < 1:
            raise ChunkingConfigError(
                "chunk_size must be at least 1",
                details={"chunk_size": self.chunk_size},
            )
        if self.chunk_overlap < 0:
            raise ChunkingConfigError(
                "chunk_overlap must be non-negative",
                details={"chunk_overlap": self.chunk_overlap},
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingConfigError(
                "chunk_overlap must be less than chunk_size",
                details={"chunk_overlap": self.chunk_overlap, "chunk_size": self.chunk_size},
            )
        if self.min_chunk_size < 1:
            raise ChunkingConfigError(
                "min_chunk_size must be at least 1",
                details={"min_chunk_size": self.min_chunk_size},
            )
        if self.min_chunk_size > self.chunk_size:
            raise ChunkingConfigError(
                "min_chunk_size must not exceed chunk_size",
                details={"min_chunk_size": self.min_chunk_size, "chunk_size": self.chunk_size},
            )
