"""Built-in chunking strategies.

This module defines the available strategy names, registration machinery,
and the concrete strategy implementations:

- whole_document — single chunk for the entire document
- fixed_size — fixed-size character chunks with configurable overlap
- sentence — split on sentence boundaries (``.``, ``!``, ``?``)
- paragraph — split on blank lines
- sliding_window — fixed-size sliding window with configurable stride
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from app.rag.chunking.base import ChunkResult
from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.errors import ChunkingConfigError
from app.rag.chunking.metadata import ChunkMetadata
from app.rag.models import KnowledgeChunk

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

STRATEGY_WHOLE_DOCUMENT = "whole_document"
STRATEGY_FIXED_SIZE = "fixed_size"
STRATEGY_SENTENCE = "sentence"
STRATEGY_PARAGRAPH = "paragraph"
STRATEGY_SLIDING_WINDOW = "sliding_window"
STRATEGY_RECURSIVE = "recursive"  # reserved for future use

# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def _make_metadata(
    text: str,
    start: int,
    end: int,
    index: int,
    config: ChunkingConfig,
) -> ChunkMetadata:
    """Build a :class:`ChunkMetadata` for the span ``text[start:end]``."""
    chunk_text = text[start:end]

    # word count: split on any whitespace
    word_count = len(chunk_text.split()) if chunk_text.strip() else 0
    # line count: number of newlines + 1 if non-empty
    line_count = chunk_text.count("\n") + 1 if chunk_text else 0

    return ChunkMetadata(
        chunk_index=index,
        character_start=start,
        character_end=end,
        word_count=word_count,
        line_count=line_count,
        strategy=config.strategy,
        created_at=time.time(),
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )


def _apply_strip(content: str, config: ChunkingConfig) -> str:
    """Strip whitespace from *content* if the config requests it."""
    return content.strip() if config.strip_whitespace else content


# ---------------------------------------------------------------------------
# whole_document
# ---------------------------------------------------------------------------


def _whole_document_strategy(text: str, config: ChunkingConfig) -> ChunkResult:
    """Produce a single chunk containing the entire document."""
    original_length = len(text)

    if not text:
        return ChunkResult(config=config, original_length=0)

    content = _apply_strip(text, config)
    if not content:
        return ChunkResult(config=config, original_length=original_length)

    meta = _make_metadata(text, 0, len(text), 0, config)
    chunk = KnowledgeChunk(content=content, index=0, metadata=meta)

    return ChunkResult(
        chunks=(chunk,),
        metadata=(meta,),
        config=config,
        total_chunks=1,
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# fixed_size
# ---------------------------------------------------------------------------


def _fixed_size_strategy(text: str, config: ChunkingConfig) -> ChunkResult:
    """Split text into fixed-size character chunks with overlap.

    The chunk advance step is ``chunk_size - chunk_overlap``.
    Empty chunks (after stripping) are silently skipped.
    """
    original_length = len(text)
    if not text:
        return ChunkResult(config=config, original_length=0)

    chunk_size = config.chunk_size
    step = chunk_size - config.chunk_overlap
    if step <= 0:
        raise ChunkingConfigError(
            "chunk_overlap must be less than chunk_size for fixed_size",
            details={"chunk_size": chunk_size, "chunk_overlap": config.chunk_overlap},
        )

    chunks: list[KnowledgeChunk] = []
    metadatas: list[ChunkMetadata] = []
    pos = 0
    index = 0
    text_len = len(text)

    while pos < text_len:
        end = min(pos + chunk_size, text_len)
        content = _apply_strip(text[pos:end], config)
        if content:
            meta = _make_metadata(text, pos, end, index, config)
            chunks.append(KnowledgeChunk(content=content, index=index, metadata=meta))
            metadatas.append(meta)
            index += 1
        if end == text_len:
            break
        pos += step

    return ChunkResult(
        chunks=tuple(chunks),
        metadata=tuple(metadatas),
        config=config,
        total_chunks=len(chunks),
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# sentence
# ---------------------------------------------------------------------------

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _sentence_strategy(text: str, config: ChunkingConfig) -> ChunkResult:
    """Split text on sentence boundaries (``.``, ``!``, ``?``).

    The delimiter is the whitespace following the punctuation, which is
    consumed during splitting.  Empty sentences and sentences that become
    empty after stripping are skipped.
    """
    original_length = len(text)
    if not text:
        return ChunkResult(config=config, original_length=0)

    raw_parts = _SENTENCE_PATTERN.split(text)
    chunks: list[KnowledgeChunk] = []
    metadatas: list[ChunkMetadata] = []
    index = 0
    search_pos = 0

    for part in raw_parts:
        if not part:
            continue
        content = _apply_strip(part, config)
        if not content:
            continue

        # Locate this part in the original text starting from search_pos.
        # The regex delimiter may have consumed variable-length whitespace,
        # so we use find() rather than tracking a running offset.
        start = text.find(part, search_pos)
        if start < 0:
            # Fallback for pathological case — should not happen
            start = search_pos
        end = start + len(part)

        meta = _make_metadata(text, start, end, index, config)
        chunks.append(KnowledgeChunk(content=content, index=index, metadata=meta))
        metadatas.append(meta)
        index += 1
        search_pos = end

    return ChunkResult(
        chunks=tuple(chunks),
        metadata=tuple(metadatas),
        config=config,
        total_chunks=len(chunks),
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# paragraph
# ---------------------------------------------------------------------------

_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


def _paragraph_strategy(text: str, config: ChunkingConfig) -> ChunkResult:
    """Split text on blank lines (two newlines with optional whitespace).

    Empty paragraphs and paragraphs that become empty after stripping
    are skipped.
    """
    original_length = len(text)
    if not text:
        return ChunkResult(config=config, original_length=0)

    raw_parts = _PARAGRAPH_PATTERN.split(text)
    chunks: list[KnowledgeChunk] = []
    metadatas: list[ChunkMetadata] = []
    index = 0
    search_pos = 0

    for part in raw_parts:
        if not part:
            continue
        content = _apply_strip(part, config)
        if not content:
            continue

        # Locate this part in the original text.  The blank-line delimiter
        # can vary in length (``\n\n``, ``\n \n``, ``\n\n\n``, etc.), so
        # we locate it with find() rather than a running offset.
        start = text.find(part, search_pos)
        if start < 0:
            start = search_pos
        end = start + len(part)

        meta = _make_metadata(text, start, end, index, config)
        chunks.append(KnowledgeChunk(content=content, index=index, metadata=meta))
        metadatas.append(meta)
        index += 1
        search_pos = end

    return ChunkResult(
        chunks=tuple(chunks),
        metadata=tuple(metadatas),
        config=config,
        total_chunks=len(chunks),
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# sliding_window
# ---------------------------------------------------------------------------


def _sliding_window_strategy(text: str, config: ChunkingConfig) -> ChunkResult:
    """Traverse text with a fixed-size sliding window.

    Uses ``window_size`` and ``stride`` from config.  The window never
    extends beyond the document bounds.  Empty windows (after stripping)
    are skipped.
    """
    original_length = len(text)
    if not text:
        return ChunkResult(config=config, original_length=0)

    window_size = config.window_size
    stride = config.stride

    chunks: list[KnowledgeChunk] = []
    metadatas: list[ChunkMetadata] = []
    pos = 0
    index = 0
    text_len = len(text)

    while pos < text_len:
        end = min(pos + window_size, text_len)
        content = _apply_strip(text[pos:end], config)
        if content:
            meta = _make_metadata(text, pos, end, index, config)
            chunks.append(KnowledgeChunk(content=content, index=index, metadata=meta))
            metadatas.append(meta)
            index += 1
        if end == text_len:
            break
        pos += stride

    return ChunkResult(
        chunks=tuple(chunks),
        metadata=tuple(metadatas),
        config=config,
        total_chunks=len(chunks),
        original_length=original_length,
    )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    """Register all built-in strategy implementations.

    This is called once at module load and may be re-called after
    ``clear_strategies()`` to restore defaults.
    """
    _strategies.clear()
    register_strategy(STRATEGY_WHOLE_DOCUMENT, _whole_document_strategy)
    register_strategy(STRATEGY_FIXED_SIZE, _fixed_size_strategy)
    register_strategy(STRATEGY_SENTENCE, _sentence_strategy)
    register_strategy(STRATEGY_PARAGRAPH, _paragraph_strategy)
    register_strategy(STRATEGY_SLIDING_WINDOW, _sliding_window_strategy)


# Auto-register builtins at module load
_register_builtins()
