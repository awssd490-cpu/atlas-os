"""Tests for the chunking architecture — strategies, engine, metadata.

Checkpoint 2 covers all five chunking strategies, the real engine
lifecycle, metadata population, chunk IDs, and edge cases.
"""

from __future__ import annotations

import time

import pytest

from app.rag.chunking import (
    ChunkMetadata,
    ChunkResult,
    ChunkingConfig,
    ChunkingConfigError,
    ChunkingEngine,
    ChunkingEngineError,
    ChunkingError,
    ChunkingStrategyError,
    UnsupportedStrategyError,
)
from app.rag.chunking.base import ChunkingStrategy
from app.rag.chunking.chunker import ChunkingEngine as ChunkingEngine_Impl
from app.rag.chunking.errors import ChunkingError as ChunkingError_Impl
from app.rag.chunking.metadata import ChunkMetadata as ChunkMetadata_Impl
from app.rag.chunking.strategies import (
    STRATEGY_FIXED_SIZE,
    STRATEGY_PARAGRAPH,
    STRATEGY_RECURSIVE,
    STRATEGY_SENTENCE,
    STRATEGY_SLIDING_WINDOW,
    STRATEGY_WHOLE_DOCUMENT,
    clear_strategies,
    get_strategy,
    list_strategies,
    register_strategy,
)
from app.rag.errors import KnowledgeError
from app.rag.models import KnowledgeChunk


# ======================================================================
# Helper
# ======================================================================


def _text(n: int, base: str = "chunk") -> str:
    """Build a string of *n* words so that chunking produces repeatable
    results.  Each word is ``base`` + index, separated by spaces."""
    return " ".join(f"{base}_{i}" for i in range(n))


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    """Verify that all public symbols import cleanly."""

    def test_chunking_config_imported(self) -> None:
        assert ChunkingConfig is not None

    def test_chunking_engine_imported(self) -> None:
        assert ChunkingEngine is ChunkingEngine_Impl

    def test_chunking_error_imported(self) -> None:
        assert ChunkingError is ChunkingError_Impl

    def test_chunk_metadata_imported(self) -> None:
        assert ChunkMetadata is ChunkMetadata_Impl

    def test_chunk_result_imported(self) -> None:
        assert ChunkResult is not None

    def test_error_hierarchy(self) -> None:
        assert issubclass(ChunkingError, KnowledgeError)
        assert issubclass(ChunkingConfigError, ChunkingError)
        assert issubclass(ChunkingEngineError, ChunkingError)
        assert issubclass(ChunkingStrategyError, ChunkingError)
        assert issubclass(UnsupportedStrategyError, ChunkingError)

    def test_chunking_strategy_protocol(self) -> None:
        """Verifies the protocol is importable and callable-shaped."""
        assert ChunkingStrategy is not None
        assert hasattr(ChunkingStrategy, "__call__")

    def test_strategy_constants_imported(self) -> None:
        assert STRATEGY_FIXED_SIZE == "fixed_size"
        assert STRATEGY_WHOLE_DOCUMENT == "whole_document"
        assert STRATEGY_SENTENCE == "sentence"
        assert STRATEGY_PARAGRAPH == "paragraph"
        assert STRATEGY_SLIDING_WINDOW == "sliding_window"
        assert STRATEGY_RECURSIVE == "recursive"

    def test_knowledge_chunk_reused(self) -> None:
        """Confirm we reuse the existing KnowledgeChunk model."""
        chunk = KnowledgeChunk(
            chunk_id="test_id",
            document_id="doc_1",
            content="Hello",
            index=0,
        )
        assert isinstance(chunk, KnowledgeChunk)


# ======================================================================
# ChunkingConfig
# ======================================================================


class TestChunkingConfig:
    def test_default_values(self) -> None:
        cfg = ChunkingConfig()
        assert cfg.strategy == "fixed_size"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.min_chunk_size == 32
        assert cfg.separator == ""
        assert cfg.secondary_separators == ()
        assert cfg.max_chunks == 0
        assert cfg.strip_whitespace is True
        assert cfg.window_size == 512
        assert cfg.stride == 256
        assert cfg.strategy_params == {}

    def test_custom_values(self) -> None:
        cfg = ChunkingConfig(
            strategy="paragraph",
            chunk_size=1024,
            chunk_overlap=128,
            min_chunk_size=64,
            separator="\n\n",
            secondary_separators=("\n", "."),
            max_chunks=10,
            strip_whitespace=False,
            window_size=500,
            stride=200,
            strategy_params={"model": "gpt-4"},
        )
        assert cfg.strategy == "paragraph"
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 128
        assert cfg.min_chunk_size == 64
        assert cfg.separator == "\n\n"
        assert cfg.secondary_separators == ("\n", ".")
        assert cfg.max_chunks == 10
        assert cfg.strip_whitespace is False
        assert cfg.window_size == 500
        assert cfg.stride == 200
        assert cfg.strategy_params == {"model": "gpt-4"}

    def test_immutable(self) -> None:
        cfg = ChunkingConfig(strategy="fixed_size")
        with pytest.raises(AttributeError):
            cfg.strategy = "recursive"  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        cfg = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        cfg.validate()  # should not raise

    def test_validate_chunk_size_zero(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_size=0).validate()

    def test_validate_chunk_size_negative(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_size=-1).validate()

    def test_validate_overlap_negative(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_overlap=-1).validate()

    def test_validate_overlap_equals_size(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100).validate()

    def test_validate_overlap_exceeds_size(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_size=100, chunk_overlap=200).validate()

    def test_validate_min_chunk_size_zero(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(min_chunk_size=0).validate()

    def test_validate_min_chunk_size_exceeds_chunk_size(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(chunk_size=100, min_chunk_size=200).validate()

    def test_validate_window_size_zero(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(window_size=0).validate()

    def test_validate_stride_zero(self) -> None:
        with pytest.raises(ChunkingConfigError):
            ChunkingConfig(stride=0).validate()


# ======================================================================
# ChunkMetadata
# ======================================================================


class TestChunkMetadata:
    def test_default_values(self) -> None:
        m = ChunkMetadata()
        assert m.document_id == ""
        assert m.chunk_index == 0
        assert m.character_start == -1
        assert m.character_end == -1
        assert m.word_count == 0
        assert m.line_count == 0
        assert m.strategy == ""
        assert m.created_at == 0.0
        assert m.chunk_size == 0
        assert m.overlap == 0
        assert m.is_partial is False
        assert m.token_count == -1

    def test_custom_values(self) -> None:
        m = ChunkMetadata(
            document_id="doc_1",
            chunk_index=1,
            character_start=0,
            character_end=500,
            word_count=80,
            line_count=5,
            strategy="fixed_size",
            created_at=1000.0,
            chunk_size=512,
            overlap=64,
            is_partial=False,
            token_count=125,
        )
        assert m.document_id == "doc_1"
        assert m.chunk_index == 1
        assert m.character_start == 0
        assert m.character_end == 500
        assert m.word_count == 80
        assert m.line_count == 5
        assert m.strategy == "fixed_size"
        assert m.created_at == 1000.0
        assert m.chunk_size == 512
        assert m.overlap == 64
        assert m.is_partial is False
        assert m.token_count == 125

    def test_immutable(self) -> None:
        m = ChunkMetadata(chunk_index=0)
        with pytest.raises(AttributeError):
            m.chunk_index = 1  # type: ignore[misc]

    def test_created_at_timestamp(self) -> None:
        """created_at is set to a reasonable Unix timestamp."""
        config = ChunkingConfig(strategy="whole_document")
        engine = ChunkingEngine()
        result = engine.chunk("Hello", config=config)
        meta = result.metadata[0]
        assert meta.created_at > 0
        assert abs(meta.created_at - time.time()) < 5


# ======================================================================
# ChunkResult
# ======================================================================


class TestChunkResult:
    def test_empty_result(self) -> None:
        r = ChunkResult()
        assert r.chunks == ()
        assert r.metadata == ()
        assert r.total_chunks == 0
        assert r.original_length == 0

    def test_with_chunks(self) -> None:
        chunk = KnowledgeChunk(
            chunk_id="c1",
            document_id="d1",
            content="test content",
            index=0,
        )
        meta = ChunkMetadata(character_start=0, character_end=12, chunk_index=0)
        result = ChunkResult(
            chunks=(chunk,),
            metadata=(meta,),
            total_chunks=1,
            original_length=12,
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "test content"
        assert result.metadata[0].character_start == 0
        assert result.total_chunks == 1
        assert result.original_length == 12

    def test_immutable(self) -> None:
        r = ChunkResult()
        with pytest.raises(AttributeError):
            r.total_chunks = 5  # type: ignore[misc]


# ======================================================================
# Whole document strategy
# ======================================================================


class TestWholeDocument:
    def test_single_chunk(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("Hello world, this is a test.", config=config)
        assert result.total_chunks == 1
        assert result.chunks[0].content == "Hello world, this is a test."
        assert result.original_length == len("Hello world, this is a test.")

    def test_empty_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("", config=config)
        assert result.total_chunks == 0
        assert result.original_length == 0

    def test_preserves_entire_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        text = "Line 1\nLine 2\nLine 3\n"
        result = engine.chunk(text, config=config)
        assert result.chunks[0].content == "Line 1\nLine 2\nLine 3"
        assert result.original_length == len(text)

    def test_strips_whitespace(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("  hello  ", config=config)
        assert result.chunks[0].content == "hello"

    def test_metadata_populated(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("Hello world", config=config, document_id="doc_1")
        meta = result.metadata[0]
        assert meta.strategy == "whole_document"
        assert meta.character_start == 0
        assert meta.character_end == 11
        assert meta.word_count == 2
        assert meta.line_count == 1
        assert meta.chunk_index == 0
        assert meta.document_id == "doc_1"

    def test_chunk_id_deterministic(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        r1 = engine.chunk("Hello", config=config, document_id="doc_1")
        r2 = engine.chunk("Hello", config=config, document_id="doc_1")
        assert r1.chunks[0].chunk_id == r2.chunks[0].chunk_id


# ======================================================================
# Fixed size strategy
# ======================================================================


class TestFixedSize:
    def _config(self, **kw: object) -> ChunkingConfig:
        return ChunkingConfig(
            strategy=STRATEGY_FIXED_SIZE,
            min_chunk_size=1,
            **kw,  # type: ignore
        )

    def test_exact_fit(self) -> None:
        """Text that fits exactly in one chunk."""
        engine = ChunkingEngine()
        config = self._config(chunk_size=100)
        text = "A" * 100
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1
        assert result.chunks[0].content == "A" * 100

    def test_multiple_chunks(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=20, chunk_overlap=0)
        text = "A" * 60  # fits in exactly 3 chunks of 20
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 3
        assert result.chunks[0].content == "A" * 20
        assert result.chunks[1].content == "A" * 20
        assert result.chunks[2].content == "A" * 20

    def test_with_overlap(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=5, chunk_overlap=2)
        text = "0123456789"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 3
        assert result.chunks[0].content == "01234"
        assert result.chunks[1].content == "34567"  # step=3, pos=3→8
        assert result.chunks[2].content == "6789"  # pos=6→10

    def test_deterministic(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=10, chunk_overlap=2)
        text = "Hello World!"
        r1 = engine.chunk(text, config=config)
        r2 = engine.chunk(text, config=config)
        assert r1.total_chunks == r2.total_chunks
        assert r1.chunks[0].content == r2.chunks[0].content

    def test_no_empty_chunks(self) -> None:
        """Never produce empty chunks."""
        engine = ChunkingEngine()
        config = self._config(chunk_size=10, chunk_overlap=5)
        text = "A"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1
        assert len(result.chunks[0].content) > 0

    def test_empty_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_FIXED_SIZE)
        result = engine.chunk("", config=config)
        assert result.total_chunks == 0
        assert result.original_length == 0

    def test_ordering_preserved(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=3, chunk_overlap=0)
        text = "ABCDEFGHI"
        result = engine.chunk(text, config=config)
        for i, chunk in enumerate(result.chunks):
            assert str(chunk.index) == str(i)

    def test_chunk_ids_deterministic(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=5, chunk_overlap=0)
        text = "A" * 15
        r1 = engine.chunk(text, config=config, document_id="doc_1")
        r2 = engine.chunk(text, config=config, document_id="doc_1")
        for c1, c2 in zip(r1.chunks, r2.chunks):
            assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_format(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=10, chunk_overlap=0)
        text = "A" * 25
        result = engine.chunk(text, config=config, document_id="d1")
        assert result.chunks[0].chunk_id == "fixed_size:d1:0"
        assert result.chunks[1].chunk_id == "fixed_size:d1:1"

    def test_metadata_populated(self) -> None:
        engine = ChunkingEngine()
        config = self._config(chunk_size=20, chunk_overlap=5)
        text = "A" * 45
        result = engine.chunk(text, config=config, document_id="doc_1")
        for i, meta in enumerate(result.metadata):
            assert meta.strategy == "fixed_size"
            assert meta.document_id == "doc_1"
            assert meta.chunk_index == i
            assert meta.character_end - meta.character_start <= 20
            assert meta.chunk_size == 20
            assert meta.overlap == 5


# ======================================================================
# Sentence strategy
# ======================================================================


class TestSentence:
    def test_simple_sentences(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Hello world. This is a test. Goodbye!"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 3
        assert "Hello world" in result.chunks[0].content
        assert "This is a test" in result.chunks[1].content
        assert "Goodbye" in result.chunks[2].content

    def test_question_mark(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Is this working? Let's verify!"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 2
        assert "Is this working" in result.chunks[0].content
        assert "verify" in result.chunks[1].content

    def test_multiple_blank_lines_handled(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "First.\n\n\nSecond?"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 2
        assert "First" in result.chunks[0].content
        assert "Second" in result.chunks[1].content

    def test_skip_empty_results(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "   \n\n  "
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 0

    def test_empty_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        result = engine.chunk("", config=config)
        assert result.total_chunks == 0

    def test_preserves_ordering(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Alpha. Beta. Gamma."
        result = engine.chunk(text, config=config)
        assert "Alpha" in result.chunks[0].content
        assert "Beta" in result.chunks[1].content
        assert "Gamma" in result.chunks[2].content

    def test_metadata_populated(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "First sentence. Second sentence. Third one!"
        result = engine.chunk(text, config=config, document_id="d1")
        for i, meta in enumerate(result.metadata):
            assert meta.strategy == "sentence"
            assert meta.document_id == "d1"
            assert meta.chunk_index == i
            assert meta.word_count > 0

    def test_chunk_ids(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        result = engine.chunk("A. B.", config=config, document_id="d1")
        assert result.chunks[0].chunk_id == "sentence:d1:0"
        assert result.chunks[1].chunk_id == "sentence:d1:1"

    def test_character_positions_correct_with_delimiter(self) -> None:
        """character_start/character_end must account for consumed delimiter."""
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        # "A." starts at 0, "B." starts at 3 after the space delimiter
        text = "A. B."
        result = engine.chunk(text, config=config)
        assert result.metadata[0].character_start == 0
        assert result.metadata[0].character_end == 2
        assert result.metadata[1].character_start == 3
        assert result.metadata[1].character_end == 5

    def test_character_positions_multiple_spaces(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "A.   B.   C."
        result = engine.chunk(text, config=config)
        assert result.metadata[0].character_start == 0
        assert result.metadata[0].character_end == 2
        assert result.metadata[1].character_start == 5
        assert result.metadata[1].character_end == 7
        assert result.metadata[2].character_start == 10
        assert result.metadata[2].character_end == 12

    def test_sentence_slice_match(self) -> None:
        """Every chunk content must match the slice of the original text."""
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Hello world. How are you? I'm fine!"
        result = engine.chunk(text, config=config)
        for chunk, meta in zip(result.chunks, result.metadata):
            expected = text[meta.character_start:meta.character_end]
            assert chunk.content == expected, (
                f"Content mismatch: {chunk.content!r} != {expected!r}"
            )


# ======================================================================
# Paragraph strategy
# ======================================================================


class TestParagraph:
    def test_paragraphs(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 3
        assert "First paragraph" in result.chunks[0].content
        assert "Second paragraph" in result.chunks[1].content
        assert "Third" in result.chunks[2].content

    def test_multiple_blank_lines(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "P1.\n\n\n\n\nP2."
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 2
        assert "P1" in result.chunks[0].content
        assert "P2" in result.chunks[1].content

    def test_skip_empty_paragraphs(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "\n\n\n"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 0

    def test_empty_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        result = engine.chunk("", config=config)
        assert result.total_chunks == 0

    def test_preserves_ordering(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "First.\n\nSecond.\n\nThird."
        result = engine.chunk(text, config=config)
        assert "First" in result.chunks[0].content
        assert "Second" in result.chunks[1].content
        assert "Third" in result.chunks[2].content

    def test_single_paragraph_no_split(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "This is a single paragraph without blank lines."
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1

    def test_metadata_populated(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "Para A.\n\nPara B."
        result = engine.chunk(text, config=config, document_id="d1")
        for i, meta in enumerate(result.metadata):
            assert meta.strategy == "paragraph"
            assert meta.document_id == "d1"
            assert meta.chunk_index == i
            assert meta.word_count > 0

    def test_chunk_ids(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        result = engine.chunk("A.\n\nB.", config=config, document_id="d1")
        assert result.chunks[0].chunk_id == "paragraph:d1:0"
        assert result.chunks[1].chunk_id == "paragraph:d1:1"

    def test_character_positions_variable_delimiter(self) -> None:
        """Blank-line delimiter can vary — positions must still be correct."""
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "P1.\n\n\n\n\nP2."
        result = engine.chunk(text, config=config)
        assert result.metadata[0].character_start == 0
        assert result.metadata[0].character_end == 3
        assert result.metadata[1].character_start == 8
        assert result.metadata[1].character_end == 11

    def test_paragraph_slice_match(self) -> None:
        """Every chunk content must match the slice of the original text."""
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "First.\n\n\nSecond.\n\n\n\nThird."
        result = engine.chunk(text, config=config)
        for chunk, meta in zip(result.chunks, result.metadata):
            expected = text[meta.character_start:meta.character_end]
            assert chunk.content == expected, (
                f"Content mismatch: {chunk.content!r} != {expected!r}"
            )


# ======================================================================
# Sliding window strategy
# ======================================================================


class TestSlidingWindow:
    def _config(self, **kw: object) -> ChunkingConfig:
        return ChunkingConfig(
            strategy=STRATEGY_SLIDING_WINDOW,
            min_chunk_size=1,
            **kw,  # type: ignore
        )

    def test_basic_traversal(self) -> None:
        engine = ChunkingEngine()
        config = self._config(window_size=10, stride=5)
        text = "A" * 30
        result = engine.chunk(text, config=config)
        assert result.total_chunks >= 4

    def test_exact_window(self) -> None:
        """When window_size equals document length, one chunk."""
        engine = ChunkingEngine()
        config = self._config(window_size=10, stride=10)
        text = "A" * 10
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1

    def test_window_larger_than_document(self) -> None:
        engine = ChunkingEngine()
        config = self._config(window_size=100, stride=50)
        text = "Hello"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1

    def test_deterministic(self) -> None:
        engine = ChunkingEngine()
        config = self._config(window_size=10, stride=5)
        text = "A" * 30
        r1 = engine.chunk(text, config=config)
        r2 = engine.chunk(text, config=config)
        assert r1.total_chunks == r2.total_chunks
        assert r1.chunks[0].content == r2.chunks[0].content
        assert r1.chunks[0].chunk_id == r2.chunks[0].chunk_id

    def test_never_exceeds_bounds(self) -> None:
        """Window never extends beyond document."""
        engine = ChunkingEngine()
        config = self._config(window_size=15, stride=10)
        text = "0123456789"
        result = engine.chunk(text, config=config)
        for meta in result.metadata:
            assert meta.character_end <= len(text)
            assert meta.character_start >= 0

    def test_empty_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_SLIDING_WINDOW,
            window_size=512,
            stride=256,
        )
        result = engine.chunk("", config=config)
        assert result.total_chunks == 0

    def test_stride_greater_than_window(self) -> None:
        """Stride can be larger than window_size — windows are spaced."""
        engine = ChunkingEngine()
        config = self._config(window_size=5, stride=20)
        text = "A" * 50
        result = engine.chunk(text, config=config)
        # pos 0→5, pos 20→25, pos 40→45 = 3 chunks (not 10)
        assert result.total_chunks == 3

    def test_metadata_populated(self) -> None:
        engine = ChunkingEngine()
        config = self._config(window_size=10, stride=5)
        text = "A" * 25
        result = engine.chunk(text, config=config, document_id="d1")
        for i, meta in enumerate(result.metadata):
            assert meta.strategy == "sliding_window"
            # chunk_size is from ChunkingConfig.chunk_size (default 512)
            # window_size=10 is passed via strategy_params when used
            assert meta.document_id == "d1"
            assert meta.chunk_index == i

    def test_chunk_ids(self) -> None:
        engine = ChunkingEngine()
        config = self._config(window_size=10, stride=10)
        result = engine.chunk("A" * 30, config=config, document_id="d1")
        assert result.chunks[0].chunk_id == "sliding_window:d1:0"
        assert result.chunks[1].chunk_id == "sliding_window:d1:1"


# ======================================================================
# Unicode
# ======================================================================


class TestUnicode:
    def test_unicode_text(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        text = "Hello 世界! 🌍✨"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 1
        assert result.chunks[0].content == text

    def test_fixed_size_unicode(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_FIXED_SIZE,
            chunk_size=10,
            chunk_overlap=0,
            min_chunk_size=1,
        )
        text = "你好世界! 这是一个测试。"
        result = engine.chunk(text, config=config)
        assert result.total_chunks >= 1
        # Verify ordering preserved
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i

    def test_sentence_unicode(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Hello world. Unicode text here! Also works? Yes."
        result = engine.chunk(text, config=config)
        assert result.total_chunks >= 3
        # Verify the English sentence splitting works with unicode-surrounded text
        assert "Hello world" in result.chunks[0].content

    def test_sentence_with_cjk_characters(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        text = "Hello. 你好世界! 测试. End."
        result = engine.chunk(text, config=config)
        assert result.total_chunks >= 2

    def test_emoji_preserved(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        text = "🎉🎊🎈"
        result = engine.chunk(text, config=config)
        assert "🎉🎊🎈" in result.chunks[0].content

    def test_mixed_language_paragraph(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_PARAGRAPH)
        text = "English paragraph.\n\nSecond para.\n\nThird paragraph!"
        result = engine.chunk(text, config=config)
        assert result.total_chunks == 3


# ======================================================================
# Edge cases — empty documents, ordering, metadata, chunk IDs
# ======================================================================


class TestEdgeCases:
    def test_empty_document_all_strategies(self) -> None:
        engine = ChunkingEngine()
        for strategy in (STRATEGY_WHOLE_DOCUMENT, STRATEGY_FIXED_SIZE,
                         STRATEGY_SENTENCE, STRATEGY_PARAGRAPH,
                         STRATEGY_SLIDING_WINDOW):
            config = ChunkingConfig(strategy=strategy)
            result = engine.chunk("", config=config)
            assert result.total_chunks == 0, f"{strategy} should yield 0 chunks for ''"

    def test_whitespace_only_document(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("   \n  \t  ", config=config)
        # strip_whitespace=True → empty content after strip → no chunks
        assert result.total_chunks == 0

    def test_ordering_fixed_size(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_FIXED_SIZE,
            chunk_size=5,
            chunk_overlap=0,
            min_chunk_size=1,
        )
        text = "AAAAABBBBBCCCCCDDDDD"
        result = engine.chunk(text, config=config)
        for i, chunk in enumerate(result.chunks):
            assert chunk.index == i
        assert result.chunks[0].content == "AAAAA"
        assert result.chunks[-1].content == "DDDDD"

    def test_chunk_ids_no_document_id(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_FIXED_SIZE,
            chunk_size=10,
            chunk_overlap=0,
            min_chunk_size=1,
        )
        result = engine.chunk("A" * 25, config=config)
        assert result.chunks[0].chunk_id == "fixed_size:0"
        assert result.chunks[1].chunk_id == "fixed_size:1"

    def test_chunk_ids_deterministic_all_strategies(self) -> None:
        engine = ChunkingEngine()
        texts = {
            STRATEGY_WHOLE_DOCUMENT: "Hello world.",
            STRATEGY_FIXED_SIZE: "Hello world. Another sentence.",
            STRATEGY_SENTENCE: "Hello world. Another sentence!",
            STRATEGY_PARAGRAPH: "Para A.\n\nPara B.",
            STRATEGY_SLIDING_WINDOW: "A" * 30,
        }
        for strategy, text in texts.items():
            config = ChunkingConfig(
                strategy=strategy,
                chunk_size=10,
                chunk_overlap=0,
                window_size=10,
                stride=10,
                min_chunk_size=1,
            )
            r1 = engine.chunk(text, config=config, document_id="doc1")
            r2 = engine.chunk(text, config=config, document_id="doc1")
            for c1, c2 in zip(r1.chunks, r2.chunks):
                assert c1.chunk_id == c2.chunk_id, f"{strategy} chunk IDs differ"

    def test_metadata_word_count(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        text = "one two three four five"
        result = engine.chunk(text, config=config)
        assert result.metadata[0].word_count == 5

    def test_metadata_line_count(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        text = "line1\nline2\nline3"
        result = engine.chunk(text, config=config)
        assert result.metadata[0].line_count == 3

    def test_fixed_size_slice_match(self) -> None:
        """Every fixed_size chunk content matches the original text slice."""
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_FIXED_SIZE,
            chunk_size=10,
            chunk_overlap=3,
            min_chunk_size=1,
        )
        text = "0123456789ABCDEFGHIJ"
        result = engine.chunk(text, config=config)
        for chunk, meta in zip(result.chunks, result.metadata):
            expected = text[meta.character_start:meta.character_end]
            assert chunk.content == expected.strip(), (
                f"Slice mismatch: {chunk.content!r} != {expected!r}"
            )

    def test_sliding_window_slice_match(self) -> None:
        """Every sliding_window chunk content matches the original text slice."""
        engine = ChunkingEngine()
        config = ChunkingConfig(
            strategy=STRATEGY_SLIDING_WINDOW,
            window_size=10,
            stride=5,
            min_chunk_size=1,
        )
        text = "0123456789ABCDEFGHIJ"  # 20 chars
        result = engine.chunk(text, config=config)
        for chunk, meta in zip(result.chunks, result.metadata):
            expected = text[meta.character_start:meta.character_end]
            assert chunk.content == expected.strip(), (
                f"Slice mismatch: {chunk.content!r} != {expected!r}"
            )


# ======================================================================
# ChunkingEngine
# ======================================================================


class TestChunkingEngine:
    def test_default_construction(self) -> None:
        engine = ChunkingEngine()
        assert isinstance(engine.config, ChunkingConfig)
        assert engine.config.strategy == "fixed_size"

    def test_custom_config(self) -> None:
        cfg = ChunkingConfig(strategy="paragraph", chunk_size=256)
        engine = ChunkingEngine(config=cfg)
        assert engine.config.strategy == "paragraph"
        assert engine.config.chunk_size == 256

    def test_config_property_immutable(self) -> None:
        """The config property returns the config, but it's frozen."""
        engine = ChunkingEngine()
        with pytest.raises(AttributeError):
            engine.config.strategy = "other"  # type: ignore[misc]

    def test_chunk_default_config(self) -> None:
        """Chunk with the engine's default config."""
        engine = ChunkingEngine()
        result = engine.chunk("Hello world")
        assert result.total_chunks >= 1
        assert result.original_length == 11
        assert len(result.chunks) >= 1

    def test_chunk_with_document_id(self) -> None:
        engine = ChunkingEngine()
        result = engine.chunk("Some text", document_id="doc_42")
        assert result.chunks[0].document_id == "doc_42"

    def test_chunk_with_explicit_config(self) -> None:
        engine = ChunkingEngine()
        cfg = ChunkingConfig(strategy="whole_document")
        result = engine.chunk("Hello", config=cfg)
        assert result.total_chunks == 1

    def test_chunk_strips_whitespace(self) -> None:
        engine = ChunkingEngine()
        result = engine.chunk("  spaced text  ")
        assert result.chunks[0].content == "spaced text"

    def test_chunk_no_strip(self) -> None:
        cfg = ChunkingConfig(strip_whitespace=False)
        engine = ChunkingEngine(config=cfg)
        result = engine.chunk("  spaced text  ")
        assert result.chunks[0].content == "  spaced text  "

    def test_chunk_invalid_config_raises(self) -> None:
        engine = ChunkingEngine()
        with pytest.raises(ChunkingEngineError):
            engine.chunk("text", config=ChunkingConfig(chunk_size=0))

    def test_chunk_invalid_stride_raises(self) -> None:
        engine = ChunkingEngine()
        with pytest.raises(ChunkingEngineError):
            engine.chunk(
                "text",
                config=ChunkingConfig(
                    strategy=STRATEGY_SLIDING_WINDOW,
                    window_size=10,
                    stride=0,
                ),
            )

    def test_available_strategies(self) -> None:
        engine = ChunkingEngine()
        strategies = engine.available_strategies()
        assert "fixed_size" in strategies
        assert "whole_document" in strategies
        assert "sentence" in strategies
        assert "paragraph" in strategies
        assert "sliding_window" in strategies

    def test_register_strategy(self) -> None:
        engine = ChunkingEngine()
        engine.register_strategy("custom", lambda t, c: ChunkResult())
        assert "custom" in engine.available_strategies()

    def test_strategy_registry_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changes to the global registry via one engine are visible to all."""
        engine_a = ChunkingEngine()
        engine_b = ChunkingEngine()
        engine_a.register_strategy("shared_test", lambda t, c: ChunkResult())
        assert "shared_test" in engine_b.available_strategies()

    def test_chunk_returns_chunk_result(self) -> None:
        engine = ChunkingEngine()
        result = engine.chunk("test")
        assert isinstance(result, ChunkResult)
        assert hasattr(result, "chunks")
        assert hasattr(result, "metadata")
        assert hasattr(result, "config")
        assert hasattr(result, "total_chunks")
        assert hasattr(result, "original_length")

    def test_reset_restores_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.rag.chunking.strategies import _register_builtins
        monkeypatch.setattr("app.rag.chunking.strategies._strategies", {})
        _register_builtins()
        engine = ChunkingEngine(config=ChunkingConfig(strategy="paragraph"))
        engine.register_strategy("reset_custom_unique", lambda t, c: ChunkResult())
        engine.reset()
        assert engine.config.strategy == "fixed_size"
        # custom strategies should be gone
        assert "reset_custom_unique" not in engine.available_strategies()

    def test_metadata_contains_all_required_fields(self) -> None:
        engine = ChunkingEngine()
        config = ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT)
        result = engine.chunk("Test text", config=config, document_id="d1")
        meta = result.metadata[0]
        assert meta.document_id == "d1"
        assert meta.chunk_index == 0
        assert meta.character_start >= 0
        assert meta.character_end > meta.character_start
        assert meta.word_count > 0
        assert meta.line_count > 0
        assert meta.strategy == "whole_document"
        assert meta.created_at > 0


# ======================================================================
# Strategy registry (global)
# ======================================================================


class TestStrategyRegistry:
    def test_get_strategy(self) -> None:
        fn = get_strategy("fixed_size")
        assert callable(fn)

    def test_get_unknown_strategy_raises(self) -> None:
        with pytest.raises(UnsupportedStrategyError):
            get_strategy("nonexistent")

    def test_unregistered_strategy_via_engine(self) -> None:
        engine = ChunkingEngine()
        with pytest.raises(ChunkingEngineError):
            engine.chunk("text", config=ChunkingConfig(strategy="nonexistent"))

    def test_register_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_strategy("fixed_size", lambda t, c: ChunkResult())

    def test_list_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.chunking.strategies._strategies", {})
        from app.rag.chunking.strategies import _register_builtins
        _register_builtins()
        names = list_strategies()
        assert isinstance(names, list)
        assert len(names) == 5
        assert "whole_document" in names
        assert "fixed_size" in names
        assert "sentence" in names
        assert "paragraph" in names
        assert "sliding_window" in names

    def test_clear_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use monkeypatch to isolate test from the global registry."""
        monkeypatch.setattr("app.rag.chunking.strategies._strategies", {})
        assert list_strategies() == []
        register_strategy("test_only", lambda t, c: ChunkResult())
        assert list_strategies() == ["test_only"]


# ======================================================================
# Error hierarchy
# ======================================================================


class TestChunkingErrors:
    def test_chunking_error_message(self) -> None:
        err = ChunkingError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "CHUNKING_ERROR"

    def test_chunking_config_error(self) -> None:
        err = ChunkingConfigError("Bad config")
        assert err.code == "CHUNKING_CONFIG_ERROR"
        assert isinstance(err, ChunkingError)

    def test_chunking_engine_error(self) -> None:
        err = ChunkingEngineError("Engine failure")
        assert err.code == "CHUNKING_ENGINE_ERROR"
        assert isinstance(err, ChunkingError)

    def test_chunking_strategy_error(self) -> None:
        err = ChunkingStrategyError("Strategy failed")
        assert err.code == "CHUNKING_STRATEGY_ERROR"

    def test_unsupported_strategy_error_with_name(self) -> None:
        err = UnsupportedStrategyError("test_strat")
        assert "test_strat" in str(err)
        assert err.code == "CHUNKING_UNSUPPORTED_STRATEGY"

    def test_unsupported_strategy_error_empty(self) -> None:
        err = UnsupportedStrategyError()
        assert str(err) == "Unsupported strategy"
        assert err.code == "CHUNKING_UNSUPPORTED_STRATEGY"

    def test_to_dict(self) -> None:
        err = ChunkingConfigError("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "CHUNKING_CONFIG_ERROR"
        assert d["message"] == "test"
        assert d["details"] == {"key": "val"}

    def test_knowledge_error_is_base(self) -> None:
        """ChunkingError derives from KnowledgeError, not directly from AtlasError."""
        assert issubclass(ChunkingError, KnowledgeError)
