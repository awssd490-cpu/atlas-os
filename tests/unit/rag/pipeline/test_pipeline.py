"""Tests for the knowledge pipeline architecture."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.pipeline import (
    InvalidPipelineConfiguration,
    KnowledgePipeline,
    PipelineConfig,
    PipelineError,
    PipelineNotFound,
    PipelineResult,
    PipelineStats,
    clear_pipelines,
    get,
    list_pipelines,
    register,
    unregister,
)
from app.rag.pipeline.base import KnowledgePipeline as KnowledgePipeline_Impl
from app.rag.pipeline.config import PipelineConfig as PipelineConfig_Impl
from app.rag.pipeline.errors import PipelineError as PipelineError_Impl
from app.rag.pipeline.errors import PipelineNotFound as PipelineNotFound_Impl
from app.rag.pipeline.models import PipelineResult as PipelineResult_Impl
from app.rag.pipeline.models import PipelineStats as PipelineStats_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_pipeline_config_imported(self) -> None:
        assert PipelineConfig is PipelineConfig_Impl

    def test_pipeline_error_imported(self) -> None:
        assert PipelineError is PipelineError_Impl

    def test_pipeline_not_found_imported(self) -> None:
        assert PipelineNotFound is PipelineNotFound_Impl

    def test_knowledge_pipeline_imported(self) -> None:
        assert KnowledgePipeline is KnowledgePipeline_Impl

    def test_pipeline_stats_imported(self) -> None:
        assert PipelineStats is PipelineStats_Impl

    def test_pipeline_result_imported(self) -> None:
        assert PipelineResult is PipelineResult_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(PipelineError, KnowledgeError)
        assert issubclass(InvalidPipelineConfiguration, PipelineError)
        assert issubclass(PipelineNotFound, PipelineError)

    def test_registry_functions_imported(self) -> None:
        assert callable(register)
        assert callable(unregister)
        assert callable(get)
        assert callable(list_pipelines)
        assert callable(clear_pipelines)


# ======================================================================
# PipelineConfig
# ======================================================================


class TestPipelineConfig:
    def test_default_values(self) -> None:
        cfg = PipelineConfig()
        assert cfg.auto_embed is True
        assert cfg.auto_index is True
        assert cfg.auto_rerank is True
        assert cfg.batch_size == 32

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(
            auto_embed=False,
            auto_index=False,
            auto_rerank=False,
            batch_size=64,
        )
        assert cfg.auto_embed is False
        assert cfg.auto_index is False
        assert cfg.auto_rerank is False
        assert cfg.batch_size == 64

    def test_immutable(self) -> None:
        cfg = PipelineConfig()
        with pytest.raises(AttributeError):
            cfg.batch_size = 64  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        PipelineConfig(batch_size=1).validate()
        PipelineConfig(batch_size=100).validate()

    def test_validate_batch_size_zero(self) -> None:
        with pytest.raises(InvalidPipelineConfiguration):
            PipelineConfig(batch_size=0).validate()

    def test_validate_batch_size_negative(self) -> None:
        with pytest.raises(InvalidPipelineConfiguration):
            PipelineConfig(batch_size=-1).validate()


# ======================================================================
# PipelineStats
# ======================================================================


class TestPipelineStats:
    def test_default_values(self) -> None:
        s = PipelineStats()
        assert s.documents == 0
        assert s.chunks == 0
        assert s.vectors == 0
        assert s.searches == 0

    def test_custom_values(self) -> None:
        s = PipelineStats(documents=10, chunks=50, vectors=50, searches=100)
        assert s.documents == 10
        assert s.chunks == 50
        assert s.vectors == 50
        assert s.searches == 100

    def test_immutable(self) -> None:
        s = PipelineStats()
        with pytest.raises(AttributeError):
            s.documents = 5  # type: ignore[misc]


# ======================================================================
# PipelineResult
# ======================================================================


class TestPipelineResult:
    def test_default_values(self) -> None:
        r = PipelineResult()
        assert r.context == ""
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = PipelineResult(context="result text", metadata={"elapsed_ms": 12.5})
        assert r.context == "result text"
        assert r.metadata == {"elapsed_ms": 12.5}

    def test_immutable(self) -> None:
        r = PipelineResult()
        with pytest.raises(AttributeError):
            r.context = "new"  # type: ignore[misc]


# ======================================================================
# KnowledgePipeline ABC + Registry
# ======================================================================


class TestKnowledgePipeline:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            KnowledgePipeline()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(KnowledgePipeline, "ingest")
        assert hasattr(KnowledgePipeline, "search")
        assert hasattr(KnowledgePipeline, "clear")
        assert hasattr(KnowledgePipeline, "stats")

    def test_default_config(self) -> None:
        class MinimalPipeline(KnowledgePipeline):
            async def ingest(
                self, documents: list[Any], **kwargs: Any
            ) -> int:
                return 0

            async def search(
                self, query: str, **kwargs: Any
            ) -> PipelineResult:
                return PipelineResult()

            async def clear(self, **kwargs: Any) -> None:
                pass

            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        pipeline = MinimalPipeline()
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.config.batch_size == 32

    def test_custom_config(self) -> None:
        class MinimalPipeline(KnowledgePipeline):
            async def ingest(
                self, documents: list[Any], **kwargs: Any
            ) -> int:
                return 0

            async def search(
                self, query: str, **kwargs: Any
            ) -> PipelineResult:
                return PipelineResult()

            async def clear(self, **kwargs: Any) -> None:
                pass

            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        config = PipelineConfig(batch_size=16)
        pipeline = MinimalPipeline(config=config)
        assert pipeline.config.batch_size == 16


class TestPipelineRegistry:
    def test_register_and_get(self) -> None:
        class FakePipeline(KnowledgePipeline):
            async def ingest(
                self, documents: list[Any], **kwargs: Any
            ) -> int:
                return 0

            async def search(
                self, query: str, **kwargs: Any
            ) -> PipelineResult:
                return PipelineResult()

            async def clear(self, **kwargs: Any) -> None:
                pass

            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        register("fake", FakePipeline)
        assert get("fake") is FakePipeline
        clear_pipelines()

    def test_register_duplicate_raises(self) -> None:
        class P1(KnowledgePipeline):
            async def ingest(self, documents: list[Any], **kwargs: Any) -> int:
                return 0
            async def search(self, query: str, **kwargs: Any) -> PipelineResult:
                return PipelineResult()
            async def clear(self, **kwargs: Any) -> None:
                pass
            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        class P2(KnowledgePipeline):
            async def ingest(self, documents: list[Any], **kwargs: Any) -> int:
                return 0
            async def search(self, query: str, **kwargs: Any) -> PipelineResult:
                return PipelineResult()
            async def clear(self, **kwargs: Any) -> None:
                pass
            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        register("dup", P1)
        with pytest.raises(ValueError, match="already registered"):
            register("dup", P2)
        clear_pipelines()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(PipelineNotFound):
            get("nonexistent")

    def test_unregister(self) -> None:
        class P(KnowledgePipeline):
            async def ingest(self, documents: list[Any], **kwargs: Any) -> int:
                return 0
            async def search(self, query: str, **kwargs: Any) -> PipelineResult:
                return PipelineResult()
            async def clear(self, **kwargs: Any) -> None:
                pass
            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        register("to_remove", P)
        unregister("to_remove")
        assert "to_remove" not in list_pipelines()
        clear_pipelines()

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(PipelineNotFound):
            unregister("nonexistent")

    def test_list_pipelines(self) -> None:
        class P(KnowledgePipeline):
            async def ingest(self, documents: list[Any], **kwargs: Any) -> int:
                return 0
            async def search(self, query: str, **kwargs: Any) -> PipelineResult:
                return PipelineResult()
            async def clear(self, **kwargs: Any) -> None:
                pass
            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        register("a", P)
        register("b", P)
        names = list_pipelines()
        assert "a" in names
        assert "b" in names
        clear_pipelines()

    def test_clear_pipelines(self) -> None:
        class P(KnowledgePipeline):
            async def ingest(self, documents: list[Any], **kwargs: Any) -> int:
                return 0
            async def search(self, query: str, **kwargs: Any) -> PipelineResult:
                return PipelineResult()
            async def clear(self, **kwargs: Any) -> None:
                pass
            async def stats(self, **kwargs: Any) -> PipelineStats:
                return PipelineStats()

        register("x", P)
        clear_pipelines()
        assert list_pipelines() == []


# ======================================================================
# Error hierarchy
# ======================================================================


class TestPipelineErrors:
    def test_pipeline_error_message(self) -> None:
        err = PipelineError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "PIPELINE_ERROR"

    def test_invalid_configuration_error(self) -> None:
        err = InvalidPipelineConfiguration("Bad config")
        assert err.code == "INVALID_PIPELINE_CONFIGURATION"

    def test_pipeline_not_found_with_name(self) -> None:
        err = PipelineNotFound("my_pipeline")
        assert "my_pipeline" in str(err)

    def test_pipeline_not_found_empty(self) -> None:
        err = PipelineNotFound()
        assert str(err) == "Pipeline not found"

    def test_to_dict(self) -> None:
        err = InvalidPipelineConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_PIPELINE_CONFIGURATION"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(PipelineError, KnowledgeError)
