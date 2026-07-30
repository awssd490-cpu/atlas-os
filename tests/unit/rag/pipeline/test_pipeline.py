"""Tests for the knowledge pipeline architecture and DefaultKnowledgePipeline."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.pipeline import (
    DefaultKnowledgePipeline,
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

    def test_default_pipeline_imported(self) -> None:
        assert DefaultKnowledgePipeline is not None
        assert issubclass(DefaultKnowledgePipeline, KnowledgePipeline)

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


# ======================================================================
# DefaultKnowledgePipeline — ingestion tests
# ======================================================================


def _make_loader(
    docs: list[Any],
) -> Any:
    """Build a loader callable that returns *docs*."""
    def loader(path: str) -> list[Any]:
        return docs
    return loader


@pytest.fixture
def single_doc() -> list[Any]:
    from app.rag.models import KnowledgeDocument
    return [
        KnowledgeDocument(
            document_id="doc_1",
            title="Test Document",
            content="Paris is the capital of France. London is the capital of the UK.",
        ),
    ]


@pytest.fixture
def multi_docs() -> list[Any]:
    from app.rag.models import KnowledgeDocument
    return [
        KnowledgeDocument(document_id="d1", title="Doc 1", content="Paris is the capital of France."),
        KnowledgeDocument(document_id="d2", title="Doc 2", content="London is the capital of the UK."),
        KnowledgeDocument(document_id="d3", title="Doc 3", content="Berlin is the capital of Germany."),
    ]


@pytest.fixture
def empty_doc() -> list[Any]:
    from app.rag.models import KnowledgeDocument
    return [KnowledgeDocument(document_id="empty", title="Empty", content="")]


@pytest.fixture
def unicode_doc() -> list[Any]:
    from app.rag.models import KnowledgeDocument
    return [
        KnowledgeDocument(
            document_id="unicode_1",
            title="Unicode",
            content="Paris est la capitale de la France. 东京は日本の首都です。",
        ),
    ]


@pytest.fixture
def pipeline(single_doc: list[Any]) -> DefaultKnowledgePipeline:
    from app.rag.chunking import ChunkingEngine, ChunkingConfig
    from app.rag.knowledge_base import KnowledgeBase

    chunk_config = ChunkingConfig(
        strategy="whole_document",
        min_chunk_size=1,
    )
    return DefaultKnowledgePipeline(
        loader=_make_loader(single_doc),
        chunker=ChunkingEngine(config=chunk_config),
        knowledge_base=KnowledgeBase(),
        config=PipelineConfig(auto_embed=False, auto_index=False),
    )


class TestDefaultPipelineConstruction:
    def test_required_args(self) -> None:
        from app.rag.chunking import ChunkingEngine
        from app.rag.knowledge_base import KnowledgeBase

        p = DefaultKnowledgePipeline(
            loader=_make_loader([]),
            chunker=ChunkingEngine(),
            knowledge_base=KnowledgeBase(),
        )
        assert isinstance(p, KnowledgePipeline)
        assert p.loader is not None
        assert p.chunker is not None
        assert p.knowledge_base is not None
        assert p.embedding_provider is None
        assert p.vector_store is None

    def test_subclass_of_knowledge_pipeline(self) -> None:
        assert issubclass(DefaultKnowledgePipeline, KnowledgePipeline)

    def test_default_config(self) -> None:
        from app.rag.chunking import ChunkingEngine
        from app.rag.knowledge_base import KnowledgeBase

        p = DefaultKnowledgePipeline(
            loader=_make_loader([]),
            chunker=ChunkingEngine(),
            knowledge_base=KnowledgeBase(),
        )
        assert p.config.auto_embed is True
        assert p.config.batch_size == 32

    def test_custom_config(self) -> None:
        from app.rag.chunking import ChunkingEngine
        from app.rag.knowledge_base import KnowledgeBase

        cfg = PipelineConfig(auto_embed=False, batch_size=16)
        p = DefaultKnowledgePipeline(
            loader=_make_loader([]),
            chunker=ChunkingEngine(),
            knowledge_base=KnowledgeBase(),
            config=cfg,
        )
        assert p.config.batch_size == 16


class TestDefaultPipelineSingleDocument:
    """Ingest a single document and verify everything."""

    @pytest.fixture(autouse=True)
    async def setup(self, pipeline: DefaultKnowledgePipeline) -> None:
        await pipeline.clear()

    async def test_ingest_single(self, pipeline: DefaultKnowledgePipeline) -> None:
        count = await pipeline.ingest("/fake/path")
        assert count == 1
        assert pipeline.knowledge_base.count() == 1

    async def test_ingest_documents_single(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        result = await pipeline.ingest_documents(single_doc)
        assert result.metadata["documents_ingested"] == 1
        assert result.metadata["chunks_created"] > 0

    async def test_stats_after_ingest(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await pipeline.ingest_documents(single_doc)
        s = await pipeline.stats()
        assert s.documents == 1
        assert s.chunks > 0

    async def test_metadata_fields(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        result = await pipeline.ingest_documents(single_doc)
        meta = result.metadata
        assert "documents_ingested" in meta
        assert "chunks_created" in meta
        assert "vectors_created" in meta
        assert "elapsed_time" in meta
        assert "embedding_enabled" in meta
        assert "indexing_enabled" in meta
        assert meta["embedding_enabled"] is False
        assert meta["indexing_enabled"] is False
        assert meta["elapsed_time"] >= 0


class TestDefaultPipelineMultipleDocuments:
    @pytest.fixture(autouse=True)
    async def setup(self, pipeline: DefaultKnowledgePipeline) -> None:
        await pipeline.clear()

    async def test_ingest_multiple(
        self, pipeline: DefaultKnowledgePipeline, multi_docs: list[Any]
    ) -> None:
        result = await pipeline.ingest_documents(multi_docs)
        assert result.metadata["documents_ingested"] == 3
        assert pipeline.knowledge_base.count() == 3

    async def test_stats_accumulates(
        self, pipeline: DefaultKnowledgePipeline, multi_docs: list[Any]
    ) -> None:
        await pipeline.ingest_documents(multi_docs)
        s = await pipeline.stats()
        assert s.documents == 3
        assert s.chunks > 0
        assert s.searches == 0

    async def test_deterministic_ordering(
        self, pipeline: DefaultKnowledgePipeline, multi_docs: list[Any]
    ) -> None:
        """Ingesting the same documents twice yields the same chunk IDs."""
        # Override pipeline with a loader for multi_docs
        pipe = DefaultKnowledgePipeline(
            loader=_make_loader(multi_docs),
            chunker=pipeline.chunker,
            knowledge_base=pipeline.knowledge_base,
            config=PipelineConfig(auto_embed=False, auto_index=False),
        )
        await pipe.clear()
        await pipe.ingest("/fake")
        docs1 = [d.document_id for d in pipe.knowledge_base.list_documents()]

        # Clear and re-ingest
        await pipe.clear()
        result = await pipe.ingest_documents(multi_docs)
        assert result.metadata["documents_ingested"] == 3
        docs2 = [d.document_id for d in pipe.knowledge_base.list_documents()]
        assert docs1 == docs2


class TestDefaultPipelineEmptyDocument:
    @pytest.fixture(autouse=True)
    async def setup(self, pipeline: DefaultKnowledgePipeline) -> None:
        await pipeline.clear()

    async def test_empty_document(
        self, pipeline: DefaultKnowledgePipeline, empty_doc: list[Any]
    ) -> None:
        result = await pipeline.ingest_documents(empty_doc)
        # Empty content still produces 0 chunks
        assert result.metadata["documents_ingested"] == 1
        assert result.metadata["chunks_created"] == 0


class TestDefaultPipelineUnicode:
    @pytest.fixture(autouse=True)
    async def setup(self, pipeline: DefaultKnowledgePipeline) -> None:
        await pipeline.clear()

    async def test_unicode_ingest(
        self, pipeline: DefaultKnowledgePipeline, unicode_doc: list[Any]
    ) -> None:
        result = await pipeline.ingest_documents(unicode_doc)
        assert result.metadata["documents_ingested"] == 1
        assert result.metadata["chunks_created"] > 0

    async def test_unicode_content_preserved(
        self, pipeline: DefaultKnowledgePipeline, unicode_doc: list[Any]
    ) -> None:
        await pipeline.ingest_documents(unicode_doc)
        doc = pipeline.knowledge_base.get("unicode_1")
        assert doc is not None
        assert "东京" in doc.content


class TestDefaultPipelineDuplicates:
    """Duplicate documents are silently skipped."""

    @pytest.fixture(autouse=True)
    async def setup(self, pipeline: DefaultKnowledgePipeline) -> None:
        await pipeline.clear()

    async def test_duplicate_skipped(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await pipeline.ingest_documents(single_doc)
        result = await pipeline.ingest_documents(single_doc)
        # Second ingest should skip the duplicate
        assert result.metadata["documents_ingested"] == 0

    async def test_stats_after_duplicate(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await pipeline.ingest_documents(single_doc)
        await pipeline.ingest_documents(single_doc)
        s = await pipeline.stats()
        assert s.documents == 1  # Only 1 unique doc


class TestDefaultPipelineClear:
    async def test_clear_resets_all(
        self, pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await pipeline.ingest_documents(single_doc)
        assert pipeline.knowledge_base.count() > 0
        await pipeline.clear()
        assert pipeline.knowledge_base.count() == 0
        s = await pipeline.stats()
        assert s.documents == 0
        assert s.chunks == 0
        assert s.vectors == 0
        assert s.searches == 0

    async def test_clear_with_vector_store(
        self, single_doc: list[Any]
    ) -> None:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.vectorstore import MemoryVectorStore

        vs = MemoryVectorStore()
        pipe = DefaultKnowledgePipeline(
            loader=_make_loader(single_doc),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            vector_store=vs,
            config=PipelineConfig(auto_embed=False, auto_index=False),
        )
        await pipe.ingest_documents(single_doc)
        assert vs.count() == 0  # No embedding, nothing indexed
        await pipe.clear()


class TestDefaultPipelineAutoEmbed:
    """Ingestion with auto_embed=True generates vectors."""

    @pytest.fixture
    def emb_pipeline(self, single_doc: list[Any]) -> DefaultKnowledgePipeline:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider

        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)

        return DefaultKnowledgePipeline(
            loader=_make_loader(single_doc),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            config=PipelineConfig(auto_embed=True, auto_index=False),
        )

    async def test_auto_embed_creates_vectors(
        self, emb_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await emb_pipeline.clear()
        result = await emb_pipeline.ingest_documents(single_doc)
        assert result.metadata["embedding_enabled"] is True
        assert result.metadata["vectors_created"] > 0

    async def test_auto_embed_stats(
        self, emb_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await emb_pipeline.clear()
        await emb_pipeline.ingest_documents(single_doc)
        s = await emb_pipeline.stats()
        assert s.vectors > 0


class TestDefaultPipelineAutoIndex:
    """Ingestion with auto_index=True inserts vectors into the store."""

    @pytest.fixture
    def index_pipeline(self, single_doc: list[Any]) -> DefaultKnowledgePipeline:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.vectorstore import MemoryVectorStore

        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)

        return DefaultKnowledgePipeline(
            loader=_make_loader(single_doc),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            vector_store=MemoryVectorStore(),
            config=PipelineConfig(auto_embed=True, auto_index=True),
        )

    async def test_auto_index_adds_vectors(
        self, index_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await index_pipeline.clear()
        await index_pipeline.ingest_documents(single_doc)
        assert index_pipeline.vector_store is not None
        assert index_pipeline.vector_store.count() > 0

    async def test_auto_index_stats(
        self, index_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await index_pipeline.clear()
        result = await index_pipeline.ingest_documents(single_doc)
        assert result.metadata["indexing_enabled"] is True
        assert result.metadata["vectors_created"] > 0

    async def test_indexing_enabled_metadata(
        self, index_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await index_pipeline.clear()
        result = await index_pipeline.ingest_documents(single_doc)
        assert result.metadata["embedding_enabled"] is True
        assert result.metadata["indexing_enabled"] is True

    async def test_clear_clears_vector_store(
        self, index_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await index_pipeline.clear()
        await index_pipeline.ingest_documents(single_doc)
        assert index_pipeline.vector_store is not None
        assert index_pipeline.vector_store.count() > 0
        await index_pipeline.clear()
        assert index_pipeline.vector_store.count() == 0


class TestDefaultPipelineDisabledEmbed:
    """Ingestion with auto_embed=False does not generate vectors."""

    @pytest.fixture
    def no_emb_pipeline(self, single_doc: list[Any]) -> DefaultKnowledgePipeline:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider

        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)

        return DefaultKnowledgePipeline(
            loader=_make_loader(single_doc),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            config=PipelineConfig(auto_embed=False, auto_index=False),
        )

    async def test_no_vectors_created(
        self, no_emb_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await no_emb_pipeline.clear()
        result = await no_emb_pipeline.ingest_documents(single_doc)
        assert result.metadata["embedding_enabled"] is False
        assert result.metadata["vectors_created"] == 0


class TestDefaultPipelineDisabledIndex:
    """Ingestion with auto_index=False does not insert into vector store."""

    @pytest.fixture
    def no_idx_pipeline(self, single_doc: list[Any]) -> DefaultKnowledgePipeline:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.vectorstore import MemoryVectorStore

        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)

        return DefaultKnowledgePipeline(
            loader=_make_loader(single_doc),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            vector_store=MemoryVectorStore(),
            config=PipelineConfig(auto_embed=True, auto_index=False),
        )

    async def test_no_vectors_in_store(
        self, no_idx_pipeline: DefaultKnowledgePipeline, single_doc: list[Any]
    ) -> None:
        await no_idx_pipeline.clear()
        result = await no_idx_pipeline.ingest_documents(single_doc)
        assert result.metadata["embedding_enabled"] is True
        assert result.metadata["indexing_enabled"] is False
        assert no_idx_pipeline.vector_store is not None
        # Vectors are created but not indexed into vector store
        assert no_idx_pipeline.vector_store.count() == 0


class TestDefaultPipelineBatchSize:
    """Respects batch_size setting for embedding batches."""

    async def test_batch_size_respected(self) -> None:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.models import KnowledgeDocument

        # Create enough docs to require multiple batches (batch_size=2)
        docs = [
            KnowledgeDocument(
                document_id=f"d{i}",
                title=f"Doc {i}",
                content=f"This is document number {i} with enough text to get chunked.",
            )
            for i in range(5)
        ]

        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)

        pipe = DefaultKnowledgePipeline(
            loader=_make_loader(docs),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            config=PipelineConfig(auto_embed=True, auto_index=False, batch_size=2),
        )
        await pipe.clear()
        result = await pipe.ingest_documents(docs)
        assert result.metadata["documents_ingested"] == 5
        assert result.metadata["vectors_created"] > 0


class TestDefaultPipelineDeterministic:
    """Repeated ingestion of identical documents produces identical results."""

    async def test_stats_deterministic(self) -> None:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument

        docs = [
            KnowledgeDocument(document_id="a", title="A", content="Hello world."),
            KnowledgeDocument(document_id="b", title="B", content="Foo bar baz."),
        ]

        cfg = PipelineConfig(auto_embed=False, auto_index=False)
        pipe = DefaultKnowledgePipeline(
            loader=_make_loader(docs),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            config=cfg,
        )
        await pipe.clear()
        r1 = await pipe.ingest_documents(docs)
        await pipe.clear()
        r2 = await pipe.ingest_documents(docs)

        assert r1.metadata["documents_ingested"] == r2.metadata["documents_ingested"]
        assert r1.metadata["chunks_created"] == r2.metadata["chunks_created"]


# ======================================================================
# DefaultKnowledgePipeline — search tests
# ======================================================================


class _SearchTestPipeline:
    """Helper to build a pipeline pre-loaded with documents for search tests."""

    @staticmethod
    async def create(
        docs: list[Any],
        *,
        embedding_provider: Any = None,
        vector_store: Any = None,
        reranker: Any = None,
        config: PipelineConfig | None = None,
        chunk_strategy: str = "whole_document",
    ) -> DefaultKnowledgePipeline:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase

        chunk_config = ChunkingConfig(strategy=chunk_strategy, min_chunk_size=1)
        kb = KnowledgeBase(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            reranker=reranker,
        )
        cfg = config or PipelineConfig(auto_embed=False, auto_index=False)

        pipe = DefaultKnowledgePipeline(
            loader=_make_loader(docs),
            chunker=ChunkingEngine(config=chunk_config),
            knowledge_base=kb,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            config=cfg,
        )
        await pipe.clear()
        await pipe.ingest_documents(docs)
        return pipe


@pytest.fixture
def search_docs() -> list[Any]:
    from app.rag.models import KnowledgeDocument
    return [
        KnowledgeDocument(
            document_id="d1", title="Paris",
            content="Paris is the capital of France.",
        ),
        KnowledgeDocument(
            document_id="d2", title="London",
            content="London is the capital of the UK.",
        ),
        KnowledgeDocument(
            document_id="d3", title="Berlin",
            content="Berlin is the capital of Germany.",
        ),
        KnowledgeDocument(
            document_id="d4", title="Tokyo",
            content="Tokyo is the capital of Japan.",
        ),
    ]


class TestDefaultPipelineKeywordSearch:
    """Keyword-only search through the existing retriever."""

    async def test_keyword_search_returns_results(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("capital France")
        assert result.context != ""
        assert "Paris" in result.context
        assert "capital" in result.context

    async def test_keyword_search_metadata(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("capital")
        meta = result.metadata
        assert "query" in meta
        assert "retrieval_mode" in meta
        assert "reranking_enabled" in meta
        assert "chunks_returned" in meta
        assert "elapsed_time" in meta
        assert meta["retrieval_mode"] == "keyword"
        assert meta["reranking_enabled"] is False
        assert meta["chunks_returned"] > 0

    async def test_stats_updated(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        await pipe.search("capital")
        s = await pipe.stats()
        assert s.searches == 1
        assert s.documents == len(search_docs)

    async def test_repeated_searches_increment(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        await pipe.search("capital")
        await pipe.search("France")
        await pipe.search("Japan")
        s = await pipe.stats()
        assert s.searches == 3

    async def test_relevance_ranking(
        self, search_docs: list[Any]
    ) -> None:
        """Search for France should rank Paris highest."""
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("France")
        assert "Paris" in result.context
        assert result.metadata["chunks_returned"] > 0

    async def test_deterministic_ordering(
        self, search_docs: list[Any]
    ) -> None:
        """Same query on same data yields same results."""
        pipe = await _SearchTestPipeline.create(search_docs)
        r1 = await pipe.search("capital")
        await pipe.clear()
        await pipe.ingest_documents(search_docs)
        r2 = await pipe.search("capital")
        assert r1.context == r2.context
        assert r1.metadata["chunks_returned"] == r2.metadata["chunks_returned"]

    async def test_search_with_max_chunks(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("capital", max_chunks=2)
        # With max_chunks=2, at most 2 results
        assert result.metadata["chunks_returned"] <= 2

    async def test_search_with_min_score(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        # High min_score should filter out low-relevance results
        result = await pipe.search("capital", min_score=100.0)
        assert result.metadata["chunks_returned"] == 0


class TestDefaultPipelineSearchEmpty:
    """Search behavior with empty or minimal knowledge base."""

    async def test_empty_knowledge_base(self) -> None:
        from app.rag.chunking import ChunkingEngine, ChunkingConfig
        from app.rag.knowledge_base import KnowledgeBase

        pipe = DefaultKnowledgePipeline(
            loader=_make_loader([]),
            chunker=ChunkingEngine(config=ChunkingConfig(strategy="whole_document", min_chunk_size=1)),
            knowledge_base=KnowledgeBase(),
            config=PipelineConfig(auto_embed=False, auto_index=False),
        )
        await pipe.clear()
        result = await pipe.search("anything")
        assert result.context == ""
        assert result.metadata["chunks_returned"] == 0

    async def test_empty_query(self, search_docs: list[Any]) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("")
        assert result.context == ""
        assert result.metadata == {}

    async def test_whitespace_query(self, search_docs: list[Any]) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("   ")
        assert result.context == ""


class TestDefaultPipelineSearchUnicode:
    """Search with unicode queries and content."""

    async def test_unicode_query(self) -> None:
        from app.rag.models import KnowledgeDocument

        docs = [
            KnowledgeDocument(
                document_id="u1",
                title="Japanese Capital",
                content="东京是日本的首都。",
            ),
        ]
        pipe = await _SearchTestPipeline.create(docs)
        result = await pipe.search("东京")
        assert result.context != ""
        assert "东京" in result.context

    async def test_unicode_content_searchable(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        # Ensure unicode documents are still findable with ascii queries
        result = await pipe.search("Japan")
        assert result.context != ""
        assert "Tokyo" in result.context


class TestDefaultPipelineHybridSearch:
    """Hybrid (keyword + semantic) search."""

    async def test_hybrid_search_returns_results(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.vectorstore import MemoryVectorStore

        docs = [
            KnowledgeDocument(document_id="h1", title="Paris", content="Paris is the capital of France."),
            KnowledgeDocument(document_id="h2", title="London", content="London is the capital of the UK."),
            KnowledgeDocument(document_id="h3", title="Tokyo", content="Tokyo is the capital of Japan."),
        ]
        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)
        vs = MemoryVectorStore()

        pipe = await _SearchTestPipeline.create(
            docs,
            embedding_provider=provider,
            vector_store=vs,
            config=PipelineConfig(auto_embed=True, auto_index=True),
        )
        result = await pipe.search("capital France")
        assert result.context != ""
        assert result.metadata["retrieval_mode"] == "hybrid"

    async def test_hybrid_metadata(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.vectorstore import MemoryVectorStore

        docs = [
            KnowledgeDocument(document_id="h1", title="Paris", content="Paris is a city in France."),
        ]
        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)
        vs = MemoryVectorStore()

        pipe = await _SearchTestPipeline.create(
            docs,
            embedding_provider=provider,
            vector_store=vs,
            config=PipelineConfig(auto_embed=True, auto_index=True),
        )
        result = await pipe.search("Paris")
        assert result.metadata["retrieval_mode"] == "hybrid"
        assert "chunks_returned" in result.metadata

    async def test_hybrid_stats_update(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
        from app.rag.vectorstore import MemoryVectorStore

        docs = [
            KnowledgeDocument(document_id="h1", title="Paris", content="Paris is the capital of France."),
        ]
        emb_cfg = EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_cfg)
        vs = MemoryVectorStore()

        pipe = await _SearchTestPipeline.create(
            docs,
            embedding_provider=provider,
            vector_store=vs,
            config=PipelineConfig(auto_embed=True, auto_index=True),
        )
        await pipe.search("Paris")
        s = await pipe.stats()
        assert s.searches == 1


class TestDefaultPipelineSearchWithReranker:
    """Search with an active reranker."""

    async def test_reranked_search(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.rerank import DefaultReranker, RerankConfig

        docs = [
            KnowledgeDocument(document_id="r1", title="Paris",
                              content="Paris is the capital of France. It is known for the Eiffel Tower."),
            KnowledgeDocument(document_id="r2", title="London",
                              content="London is the capital of the UK."),
        ]
        reranker = DefaultReranker()
        pipe = await _SearchTestPipeline.create(docs, reranker=reranker)
        result = await pipe.search("Paris Eiffel Tower")
        assert result.context != ""
        assert result.metadata["reranking_enabled"] is True

    async def test_reranked_metadata(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.rerank import DefaultReranker

        docs = [
            KnowledgeDocument(document_id="r1", title="Paris",
                              content="Paris is the capital of France."),
        ]
        reranker = DefaultReranker()
        pipe = await _SearchTestPipeline.create(docs, reranker=reranker)
        result = await pipe.search("capital")
        meta = result.metadata
        assert meta["reranking_enabled"] is True
        assert meta["retrieval_mode"] == "keyword"

    async def test_reranked_stats_update(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.rerank import DefaultReranker

        docs = [
            KnowledgeDocument(document_id="r1", title="Paris",
                              content="Paris is the capital of France."),
        ]
        reranker = DefaultReranker()
        pipe = await _SearchTestPipeline.create(docs, reranker=reranker)
        await pipe.search("capital")
        s = await pipe.stats()
        assert s.searches == 1


class TestDefaultPipelineSearchDisabledReranker:
    """Search with a disabled reranker."""

    async def test_disabled_reranker_passthrough(self) -> None:
        from app.rag.models import KnowledgeDocument
        from app.rag.rerank import DefaultReranker, RerankConfig

        docs = [
            KnowledgeDocument(document_id="r1", title="Paris",
                              content="Paris is the capital of France."),
        ]
        config = RerankConfig(enabled=False)
        reranker = DefaultReranker(config=config)
        pipe = await _SearchTestPipeline.create(docs, reranker=reranker)
        result = await pipe.search("capital")
        assert result.metadata["reranking_enabled"] is False
        assert result.context != ""


class TestDefaultPipelineSearchUnifiedMetadata:
    """Unified metadata across search scenarios."""

    async def test_metadata_fields_present(
        self, search_docs: list[Any]
    ) -> None:
        pipe = await _SearchTestPipeline.create(search_docs)
        result = await pipe.search("capital")
        meta = result.metadata
        assert "query" in meta
        assert "retrieval_mode" in meta
        assert "reranking_enabled" in meta
        assert "chunks_returned" in meta
        assert "elapsed_time" in meta
        assert isinstance(meta["elapsed_time"], float)
        assert meta["elapsed_time"] >= 0
        assert meta["query"] == "capital"
