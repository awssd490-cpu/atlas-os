"""Basic unit tests for the chunking architecture.

Checkpoint 1 — verifies imports, configuration, object construction,
and error hierarchy.  Does NOT test chunking algorithms.
"""

from __future__ import annotations

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
    STRATEGY_RECURSIVE,
    STRATEGY_SENTENCE,
    clear_strategies,
    get_strategy,
    list_strategies,
    register_strategy,
)
from app.rag.errors import KnowledgeError
from app.rag.models import KnowledgeChunk


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
        # It's a Protocol, so we can't instantiate it directly,
        # but we can check it defines __call__
        assert hasattr(ChunkingStrategy, "__call__")

    def test_strategy_constants(self) -> None:
        assert STRATEGY_FIXED_SIZE == "fixed_size"
        assert STRATEGY_RECURSIVE == "recursive"
        assert STRATEGY_SENTENCE == "sentence"

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
        assert cfg.strategy_params == {}

    def test_custom_values(self) -> None:
        cfg = ChunkingConfig(
            strategy="recursive",
            chunk_size=1024,
            chunk_overlap=128,
            min_chunk_size=64,
            separator="\n\n",
            secondary_separators=("\n", "."),
            max_chunks=10,
            strip_whitespace=False,
            strategy_params={"model": "gpt-4"},
        )
        assert cfg.strategy == "recursive"
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 128
        assert cfg.min_chunk_size == 64
        assert cfg.separator == "\n\n"
        assert cfg.secondary_separators == ("\n", ".")
        assert cfg.max_chunks == 10
        assert cfg.strip_whitespace is False
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


# ======================================================================
# ChunkMetadata
# ======================================================================


class TestChunkMetadata:
    def test_default_values(self) -> None:
        m = ChunkMetadata()
        assert m.start_char == -1
        assert m.end_char == -1
        assert m.index == 0
        assert m.strategy == ""
        assert m.chunk_size == 0
        assert m.overlap == 0
        assert m.is_partial is False
        assert m.token_count == -1

    def test_custom_values(self) -> None:
        m = ChunkMetadata(
            start_char=0,
            end_char=500,
            index=1,
            strategy="fixed_size",
            chunk_size=512,
            overlap=64,
            is_partial=False,
            token_count=125,
        )
        assert m.start_char == 0
        assert m.end_char == 500
        assert m.index == 1
        assert m.strategy == "fixed_size"
        assert m.chunk_size == 512
        assert m.overlap == 64
        assert m.is_partial is False
        assert m.token_count == 125

    def test_immutable(self) -> None:
        m = ChunkMetadata(index=0)
        with pytest.raises(AttributeError):
            m.index = 1  # type: ignore[misc]


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
        meta = ChunkMetadata(start_char=0, end_char=12, index=0)
        result = ChunkResult(
            chunks=(chunk,),
            metadata=(meta,),
            total_chunks=1,
            original_length=12,
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "test content"
        assert result.metadata[0].start_char == 0
        assert result.total_chunks == 1
        assert result.original_length == 12

    def test_immutable(self) -> None:
        r = ChunkResult()
        with pytest.raises(AttributeError):
            r.total_chunks = 5  # type: ignore[misc]


# ======================================================================
# ChunkingEngine
# ======================================================================


class TestChunkingEngine:
    def test_default_construction(self) -> None:
        engine = ChunkingEngine()
        assert isinstance(engine.config, ChunkingConfig)
        assert engine.config.strategy == "fixed_size"

    def test_custom_config(self) -> None:
        cfg = ChunkingConfig(strategy="recursive", chunk_size=256)
        engine = ChunkingEngine(config=cfg)
        assert engine.config.strategy == "recursive"
        assert engine.config.chunk_size == 256

    def test_config_property_immutable(self) -> None:
        """The config property returns the config, but it's frozen."""
        engine = ChunkingEngine()
        with pytest.raises(AttributeError):
            engine.config.strategy = "other"  # type: ignore[misc]

    def test_chunk_default_config(self) -> None:
        """Chunk with the engine's default config — returns stub result."""
        engine = ChunkingEngine()
        result = engine.chunk("Hello world")
        assert result.total_chunks == 1
        assert result.original_length == 11
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Hello world"

    def test_chunk_with_document_id(self) -> None:
        engine = ChunkingEngine()
        result = engine.chunk("Some text", document_id="doc_42")
        assert result.chunks[0].document_id == "doc_42"

    def test_chunk_with_explicit_config(self) -> None:
        engine = ChunkingEngine()
        cfg = ChunkingConfig(strategy="fixed_size", chunk_size=100)
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

    def test_available_strategies(self) -> None:
        engine = ChunkingEngine()
        strategies = engine.available_strategies()
        assert "fixed_size" in strategies
        assert "recursive" in strategies
        assert "sentence" in strategies

    def test_register_strategy(self) -> None:
        engine = ChunkingEngine()
        # Register a custom strategy
        engine.register_strategy("custom", lambda t, c: ChunkResult())
        assert "custom" in engine.available_strategies()

    def test_strategy_registry_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changes to the global registry via one engine are visible to all."""
        engine_a = ChunkingEngine()
        engine_b = ChunkingEngine()
        engine_a.register_strategy("shared_test", lambda t, c: ChunkResult())
        assert "shared_test" in engine_b.available_strategies()


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

    def test_list_strategies(self) -> None:
        names = list_strategies()
        assert isinstance(names, list)
        assert len(names) >= 3

    def test_clear_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use monkeypatch to isolate test from the global registry."""
        monkeypatch.setattr("app.rag.chunking.strategies._strategies", {})
        assert list_strategies() == []
        # Re-register for isolation
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
