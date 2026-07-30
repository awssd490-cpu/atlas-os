"""Tests for the persistence architecture and JsonPersistenceBackend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.rag.persistence import (
    InvalidPersistenceConfiguration,
    JsonPersistenceBackend,
    PersistenceBackend,
    PersistenceConfig,
    PersistenceError,
    PersistenceNotFound,
    PersistenceResult,
    PersistenceStats,
    clear_backends,
    get,
    list_backends,
    register,
    unregister,
)
from app.rag.persistence.base import PersistenceBackend as PersistenceBackend_Impl
from app.rag.persistence.config import PersistenceConfig as PersistenceConfig_Impl
from app.rag.persistence.errors import PersistenceError as PersistenceError_Impl
from app.rag.persistence.errors import PersistenceNotFound as PersistenceNotFound_Impl
from app.rag.persistence.models import PersistenceResult as PersistenceResult_Impl
from app.rag.persistence.models import PersistenceStats as PersistenceStats_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_persistence_config_imported(self) -> None:
        assert PersistenceConfig is PersistenceConfig_Impl

    def test_persistence_error_imported(self) -> None:
        assert PersistenceError is PersistenceError_Impl

    def test_persistence_not_found_imported(self) -> None:
        assert PersistenceNotFound is PersistenceNotFound_Impl

    def test_persistence_backend_imported(self) -> None:
        assert PersistenceBackend is PersistenceBackend_Impl

    def test_persistence_stats_imported(self) -> None:
        assert PersistenceStats is PersistenceStats_Impl

    def test_persistence_result_imported(self) -> None:
        assert PersistenceResult is PersistenceResult_Impl

    def test_json_backend_imported(self) -> None:
        assert JsonPersistenceBackend is not None
        assert issubclass(JsonPersistenceBackend, PersistenceBackend)

    def test_error_hierarchy(self) -> None:
        assert issubclass(PersistenceError, KnowledgeError)
        assert issubclass(InvalidPersistenceConfiguration, PersistenceError)
        assert issubclass(PersistenceNotFound, PersistenceError)

    def test_registry_functions_imported(self) -> None:
        assert callable(register)
        assert callable(unregister)
        assert callable(get)
        assert callable(list_backends)
        assert callable(clear_backends)


# ======================================================================
# PersistenceConfig
# ======================================================================


class TestPersistenceConfig:
    def test_default_values(self) -> None:
        cfg = PersistenceConfig()
        assert cfg.compress is True
        assert cfg.overwrite is False
        assert cfg.include_embeddings is True
        assert cfg.include_vectors is True

    def test_custom_values(self) -> None:
        cfg = PersistenceConfig(
            compress=False,
            overwrite=True,
            include_embeddings=False,
            include_vectors=False,
        )
        assert cfg.compress is False
        assert cfg.overwrite is True
        assert cfg.include_embeddings is False
        assert cfg.include_vectors is False

    def test_immutable(self) -> None:
        cfg = PersistenceConfig()
        with pytest.raises(AttributeError):
            cfg.compress = False  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        PersistenceConfig().validate()
        PersistenceConfig(
            compress=False, overwrite=True,
            include_embeddings=False, include_vectors=False,
        ).validate()


# ======================================================================
# PersistenceStats
# ======================================================================


class TestPersistenceStats:
    def test_default_values(self) -> None:
        s = PersistenceStats()
        assert s.documents == 0
        assert s.chunks == 0
        assert s.embeddings == 0
        assert s.vectors == 0
        assert s.size_bytes == 0

    def test_custom_values(self) -> None:
        s = PersistenceStats(
            documents=10, chunks=50, embeddings=50,
            vectors=25, size_bytes=102400,
        )
        assert s.documents == 10
        assert s.chunks == 50
        assert s.embeddings == 50
        assert s.vectors == 25
        assert s.size_bytes == 102400

    def test_immutable(self) -> None:
        s = PersistenceStats()
        with pytest.raises(AttributeError):
            s.documents = 5  # type: ignore[misc]


# ======================================================================
# PersistenceResult
# ======================================================================


class TestPersistenceResult:
    def test_default_values(self) -> None:
        r = PersistenceResult()
        assert r.success is False
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = PersistenceResult(
            success=True,
            metadata={"elapsed_ms": 12.5, "bytes_written": 4096},
        )
        assert r.success is True
        assert r.metadata == {"elapsed_ms": 12.5, "bytes_written": 4096}

    def test_immutable(self) -> None:
        r = PersistenceResult()
        with pytest.raises(AttributeError):
            r.success = True  # type: ignore[misc]


# ======================================================================
# PersistenceBackend ABC + Registry
# ======================================================================


class TestPersistenceBackend:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            PersistenceBackend()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(PersistenceBackend, "save")
        assert hasattr(PersistenceBackend, "load")
        assert hasattr(PersistenceBackend, "exists")
        assert hasattr(PersistenceBackend, "delete")
        assert hasattr(PersistenceBackend, "stats")

    def test_default_config(self) -> None:
        class MinimalBackend(PersistenceBackend):
            async def save(
                self, path: str, data: object, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def load(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def exists(
                self, path: str, **kwargs: object
            ) -> bool:
                return False

            async def delete(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def stats(
                self, path: str, **kwargs: object
            ) -> PersistenceStats:
                return PersistenceStats()

        backend = MinimalBackend()
        assert isinstance(backend.config, PersistenceConfig)
        assert backend.config.compress is True

    def test_custom_config(self) -> None:
        class MinimalBackend(PersistenceBackend):
            async def save(
                self, path: str, data: object, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def load(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def exists(
                self, path: str, **kwargs: object
            ) -> bool:
                return False

            async def delete(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()

            async def stats(
                self, path: str, **kwargs: object
            ) -> PersistenceStats:
                return PersistenceStats()

        config = PersistenceConfig(compress=False)
        backend = MinimalBackend(config=config)
        assert backend.config.compress is False


class TestPersistenceRegistry:
    def test_register_and_get(self) -> None:
        class FakeBackend(PersistenceBackend):
            async def save(
                self, path: str, data: object, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()
            async def load(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()
            async def exists(
                self, path: str, **kwargs: object
            ) -> bool:
                return False
            async def delete(
                self, path: str, **kwargs: object
            ) -> PersistenceResult:
                return PersistenceResult()
            async def stats(
                self, path: str, **kwargs: object
            ) -> PersistenceStats:
                return PersistenceStats()

        register("fake", FakeBackend)
        assert get("fake") is FakeBackend
        clear_backends()

    def test_register_duplicate_raises(self) -> None:
        class B1(PersistenceBackend):
            async def save(self, path: str, data: object, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def load(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def exists(self, path: str, **kwargs: object) -> bool:
                return False
            async def delete(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def stats(self, path: str, **kwargs: object) -> PersistenceStats:
                return PersistenceStats()

        class B2(PersistenceBackend):
            async def save(self, path: str, data: object, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def load(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def exists(self, path: str, **kwargs: object) -> bool:
                return False
            async def delete(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def stats(self, path: str, **kwargs: object) -> PersistenceStats:
                return PersistenceStats()

        register("dup", B1)
        with pytest.raises(ValueError, match="already registered"):
            register("dup", B2)
        clear_backends()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(PersistenceNotFound):
            get("nonexistent")

    def test_unregister(self) -> None:
        class B(PersistenceBackend):
            async def save(self, path: str, data: object, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def load(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def exists(self, path: str, **kwargs: object) -> bool:
                return False
            async def delete(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def stats(self, path: str, **kwargs: object) -> PersistenceStats:
                return PersistenceStats()

        register("to_remove", B)
        unregister("to_remove")
        assert "to_remove" not in list_backends()
        clear_backends()

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(PersistenceNotFound):
            unregister("nonexistent")

    def test_list_backends(self) -> None:
        class B(PersistenceBackend):
            async def save(self, path: str, data: object, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def load(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def exists(self, path: str, **kwargs: object) -> bool:
                return False
            async def delete(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def stats(self, path: str, **kwargs: object) -> PersistenceStats:
                return PersistenceStats()

        register("a", B)
        register("b", B)
        names = list_backends()
        assert "a" in names
        assert "b" in names
        clear_backends()

    def test_clear_backends(self) -> None:
        class B(PersistenceBackend):
            async def save(self, path: str, data: object, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def load(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def exists(self, path: str, **kwargs: object) -> bool:
                return False
            async def delete(self, path: str, **kwargs: object) -> PersistenceResult:
                return PersistenceResult()
            async def stats(self, path: str, **kwargs: object) -> PersistenceStats:
                return PersistenceStats()

        register("x", B)
        clear_backends()
        assert list_backends() == []


# ======================================================================
# Error hierarchy
# ======================================================================


class TestPersistenceErrors:
    def test_persistence_error_message(self) -> None:
        err = PersistenceError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "PERSISTENCE_ERROR"

    def test_invalid_configuration_error(self) -> None:
        err = InvalidPersistenceConfiguration("Bad config")
        assert err.code == "INVALID_PERSISTENCE_CONFIGURATION"

    def test_persistence_not_found_with_name(self) -> None:
        err = PersistenceNotFound("json_backend")
        assert "json_backend" in str(err)

    def test_persistence_not_found_empty(self) -> None:
        err = PersistenceNotFound()
        assert str(err) == "Persistence backend not found"

    def test_to_dict(self) -> None:
        err = InvalidPersistenceConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_PERSISTENCE_CONFIGURATION"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(PersistenceError, KnowledgeError)


# ======================================================================
# JsonPersistenceBackend — save tests
# ======================================================================


@pytest.fixture
def tmp_path() -> str:
    """Return a temporary file path for JSON output."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def empty_kb() -> Any:
    from app.rag.knowledge_base import KnowledgeBase
    return KnowledgeBase()


@pytest.fixture
def populated_kb() -> Any:
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.models import KnowledgeDocument, KnowledgeChunk, KnowledgeMetadata
    kb = KnowledgeBase()
    doc = KnowledgeDocument(
        document_id="doc_1",
        title="Test Document",
        content="Paris is the capital of France. London is the capital of the UK.",
        chunks=(
            KnowledgeChunk(
                chunk_id="doc_1:0",
                document_id="doc_1",
                content="Paris is the capital of France.",
                index=0,
                metadata=KnowledgeMetadata(source="test", tags=("geography",)),
            ),
            KnowledgeChunk(
                chunk_id="doc_1:1",
                document_id="doc_1",
                content="London is the capital of the UK.",
                index=1,
                metadata=KnowledgeMetadata(source="test", tags=("geography",)),
            ),
        ),
        metadata=KnowledgeMetadata(source="test"),
    )
    kb.register(doc)
    return kb


@pytest.fixture
def kb_with_embeddings() -> Any:
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.models import KnowledgeDocument, KnowledgeChunk, KnowledgeMetadata
    from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
    from app.rag.embeddings.models import EmbeddingVector

    provider = DeterministicEmbeddingProvider(
        EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
    )
    kb = KnowledgeBase(
        embedding_provider=provider,
    )
    doc = KnowledgeDocument(
        document_id="emb_doc",
        title="Embedded Document",
        content="Paris is the capital of France.",
        chunks=(
            KnowledgeChunk(
                chunk_id="emb_doc:0",
                document_id="emb_doc",
                content="Paris is the capital of France.",
                index=0,
            ),
        ),
    )
    kb.register(doc)
    # Manually inject embedding vector
    import asyncio
    result = asyncio.run(provider.embed_batch(["Paris is the capital of France."]))
    kb._embeddings["emb_doc:0"] = result.embeddings[0]
    return kb


@pytest.fixture
def kb_with_vectors() -> Any:
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.models import KnowledgeDocument, KnowledgeChunk
    from app.rag.embeddings import EmbeddingConfig, DeterministicEmbeddingProvider
    from app.rag.embeddings.models import EmbeddingVector
    from app.rag.vectorstore import MemoryVectorStore

    provider = DeterministicEmbeddingProvider(
        EmbeddingConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
    )
    vs = MemoryVectorStore()
    kb = KnowledgeBase(
        embedding_provider=provider,
        vector_store=vs,
    )
    doc = KnowledgeDocument(
        document_id="vec_doc",
        title="Vector Document",
        content="Berlin is the capital of Germany.",
        chunks=(
            KnowledgeChunk(
                chunk_id="vec_doc:0",
                document_id="vec_doc",
                content="Berlin is the capital of Germany.",
                index=0,
            ),
        ),
    )
    kb.register(doc)
    # Manually inject embedding and vector
    import asyncio
    result = asyncio.run(provider.embed_batch(["Berlin is the capital of Germany."]))
    kb._embeddings["vec_doc:0"] = result.embeddings[0]
    vs.add("vec_doc:0", result.embeddings[0].vector)
    return kb


# ======================================================================
# JsonPersistenceBackend — construction
# ======================================================================


class TestJsonBackendConstruction:
    def test_subclass_of_persistence_backend(self) -> None:
        assert issubclass(JsonPersistenceBackend, PersistenceBackend)

    def test_default_config(self) -> None:
        backend = JsonPersistenceBackend()
        assert backend.config.compress is True
        assert backend.config.overwrite is False

    def test_custom_config(self) -> None:
        config = PersistenceConfig(overwrite=True, compress=False)
        backend = JsonPersistenceBackend(config=config)
        assert backend.config.overwrite is True
        assert backend.config.compress is False


# ======================================================================
# JsonPersistenceBackend — save to empty knowledge base
# ======================================================================


class TestJsonBackendSaveEmpty:
    async def test_save_empty_kb(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, empty_kb)
        assert result.success is True
        assert result.metadata["documents"] == 0
        assert result.metadata["chunks"] == 0

    async def test_save_empty_kb_file_created(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, empty_kb)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

    async def test_save_empty_kb_json_structure(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, empty_kb)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["documents"] == []
        assert data["chunks"] == []
        assert "embeddings" in data
        assert "vectors" in data
        assert "metadata" in data

    async def test_save_empty_kb_metadata(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, empty_kb)
        meta = result.metadata
        assert "path" in meta
        assert "size_bytes" in meta
        assert meta["size_bytes"] > 0
        assert "elapsed_time" in meta
        assert meta["elapsed_time"] >= 0


# ======================================================================
# JsonPersistenceBackend — save populated knowledge base
# ======================================================================


class TestJsonBackendSavePopulated:
    async def test_save_populated_kb(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, populated_kb)
        assert result.success is True
        assert result.metadata["documents"] == 1
        assert result.metadata["chunks"] == 2

    async def test_save_populated_kb_json(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["documents"]) == 1
        assert len(data["chunks"]) == 2
        assert data["documents"][0]["document_id"] == "doc_1"
        assert data["chunks"][0]["chunk_id"] == "doc_1:0"
        assert data["chunks"][1]["chunk_id"] == "doc_1:1"

    async def test_save_populated_kb_content(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        chunk = data["chunks"][0]
        assert chunk["content"] == "Paris is the capital of France."
        assert chunk["index"] == 0

    async def test_save_populated_kb_metadata(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, populated_kb)
        meta = result.metadata
        assert meta["documents"] == 1
        assert meta["chunks"] == 2


# ======================================================================
# JsonPersistenceBackend — unicode
# ======================================================================


class TestJsonBackendUnicode:
    async def test_save_unicode(self, tmp_path: str) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk, KnowledgeMetadata

        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="uni",
            title="Unicode Test",
            content="东京是日本的首都。Paris est la capitale de la France.",
            chunks=(
                KnowledgeChunk(
                    chunk_id="uni:0",
                    document_id="uni",
                    content="东京是日本的首都。",
                    index=0,
                ),
            ),
        )
        kb.register(doc)
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, kb)
        assert result.success is True

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "东京" in data["chunks"][0]["content"]

    async def test_unicode_stats(self, tmp_path: str) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk

        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="u2",
            title="日本語",
            content="日本語のテスト",
            chunks=(
                KnowledgeChunk(chunk_id="u2:0", document_id="u2", content="日本語のテスト", index=0),
            ),
        )
        kb.register(doc)
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        result = await backend.save(tmp_path, kb)
        assert result.metadata["documents"] == 1
        assert result.metadata["chunks"] == 1


# ======================================================================
# JsonPersistenceBackend — overwrite protection
# ======================================================================


class TestJsonBackendOverwrite:
    async def test_overwrite_protection(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend()  # overwrite=False
        with open(tmp_path, "w") as f:
            f.write("existing")
        with pytest.raises(PersistenceError) as exc:
            await backend.save(tmp_path, empty_kb)
        assert "already exists" in str(exc.value)

    async def test_overwrite_allowed(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        # Write once, then overwrite
        result1 = await backend.save(tmp_path, populated_kb)
        assert result1.success is True
        result2 = await backend.save(tmp_path, populated_kb)
        assert result2.success is True

    async def test_overwrite_allowed_file_updated(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        # Write again with different data
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument

        kb2 = KnowledgeBase()
        doc2 = KnowledgeDocument(document_id="new", title="New", content="New content.")
        # Manually add without chunking
        from app.rag.models import KnowledgeChunk
        doc2_with_chunks = KnowledgeDocument(
            document_id="new", title="New", content="New content.",
            chunks=(KnowledgeChunk(chunk_id="new:0", document_id="new", content="New content.", index=0),),
        )
        kb2.register(doc2_with_chunks)

        await backend.save(tmp_path, kb2)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["documents"][0]["document_id"] == "new"


# ======================================================================
# JsonPersistenceBackend — deterministic output
# ======================================================================


class TestJsonBackendDeterministic:
    async def test_deterministic_output(self, tmp_path: str, populated_kb: Any) -> None:
        cfg = PersistenceConfig(overwrite=True)
        backend = JsonPersistenceBackend(config=cfg)
        await backend.save(tmp_path, populated_kb)
        with open(tmp_path, encoding="utf-8") as f:
            first = json.load(f)

        # Save again with overwrite
        await backend.save(tmp_path, populated_kb)
        with open(tmp_path, encoding="utf-8") as f:
            second = json.load(f)

        # Documents and chunks should be identical (timestamps will differ)
        assert first["version"] == second["version"]
        assert first["documents"] == second["documents"]
        assert first["chunks"] == second["chunks"]
        assert first["metadata"]["document_count"] == second["metadata"]["document_count"]
        assert first["metadata"]["chunk_count"] == second["metadata"]["chunk_count"]

    async def test_sorted_keys(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        # Python dicts preserve insertion order, but json.dumps(sort_keys=True)
        # should have sorted top-level keys
        keys = list(data.keys())
        # version comes first, then documents, chunks, etc (alphabetically)
        assert keys == sorted(keys)


# ======================================================================
# JsonPersistenceBackend — stats
# ======================================================================


class TestJsonBackendStats:
    async def test_stats_after_save(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        stats = await backend.stats(tmp_path)
        assert stats.documents == 1
        assert stats.chunks == 2
        assert stats.size_bytes > 0

    async def test_stats_empty_kb(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, empty_kb)
        stats = await backend.stats(tmp_path)
        assert stats.documents == 0
        assert stats.chunks == 0
        assert stats.size_bytes > 0

    async def test_stats_before_save_raises(self, tmp_path: str) -> None:
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError):
            await backend.stats(tmp_path)


# ======================================================================
# JsonPersistenceBackend — exists / delete
# ======================================================================


class TestJsonBackendExists:
    async def test_exists_after_save(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        assert await backend.exists(tmp_path) is False
        await backend.save(tmp_path, populated_kb)
        assert await backend.exists(tmp_path) is True

    async def test_delete(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        result = await backend.delete(tmp_path)
        assert result.success is True
        assert await backend.exists(tmp_path) is False

    async def test_delete_not_found(self, tmp_path: str) -> None:
        backend = JsonPersistenceBackend()
        result = await backend.delete(tmp_path)
        assert result.success is False
        assert result.metadata["reason"] == "not_found"


# ======================================================================
# JsonPersistenceBackend — non-KnowledgeBase data
# ======================================================================


class TestJsonBackendInvalidData:
    async def test_save_non_kb_raises(self, tmp_path: str) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        with pytest.raises(PersistenceError, match="KnowledgeBase instance"):
            await backend.save(tmp_path, {"not": "a knowledge base"})

    async def test_save_none_raises(self, tmp_path: str) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        with pytest.raises(PersistenceError, match="KnowledgeBase instance"):
            await backend.save(tmp_path, None)  # type: ignore[arg-type]


# ======================================================================
# JsonPersistenceBackend — embeddings configuration
# ======================================================================


class TestJsonBackendEmbeddings:
    async def test_include_embeddings(self, tmp_path: str, kb_with_embeddings: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=True),
        )
        result = await backend.save(tmp_path, kb_with_embeddings)
        assert result.metadata["embeddings"] == 1

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["embeddings"]) == 1
        assert data["embeddings"][0]["chunk_id"] == "emb_doc:0"
        assert len(data["embeddings"][0]["vector"]) == 4

    async def test_exclude_embeddings(self, tmp_path: str, kb_with_embeddings: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=False),
        )
        result = await backend.save(tmp_path, kb_with_embeddings)
        assert result.metadata["embeddings"] == 0

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "embeddings" not in data


# ======================================================================
# JsonPersistenceBackend — vectors configuration
# ======================================================================


class TestJsonBackendVectors:
    async def test_include_vectors(self, tmp_path: str, kb_with_vectors: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_vectors=True),
        )
        result = await backend.save(tmp_path, kb_with_vectors)
        assert result.metadata["vectors"] == 1

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["vectors"]) == 1
        assert data["vectors"][0]["chunk_id"] == "vec_doc:0"

    async def test_exclude_vectors(self, tmp_path: str, kb_with_vectors: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_vectors=False),
        )
        result = await backend.save(tmp_path, kb_with_vectors)
        assert result.metadata["vectors"] == 0

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "vectors" not in data


# ======================================================================
# JsonPersistenceBackend — load tests
# ======================================================================


class TestJsonBackendLoad:
    """Basic load functionality."""

    async def test_load_empty(self, tmp_path: str, empty_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, empty_kb)
        result = await backend.load(tmp_path)
        assert result.success is True
        kb = result.metadata["knowledge_base"]
        assert kb.count() == 0

    async def test_load_populated(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        result = await backend.load(tmp_path)
        assert result.success is True
        kb = result.metadata["knowledge_base"]
        assert kb.count() == 1
        doc = kb.get("doc_1")
        assert doc is not None
        assert doc.title == "Test Document"
        assert len(doc.chunks) == 2

    async def test_load_content_preserved(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        result = await backend.load(tmp_path)
        kb = result.metadata["knowledge_base"]
        doc = kb.get("doc_1")
        assert doc is not None
        assert doc.content == "Paris is the capital of France. London is the capital of the UK."
        assert doc.chunks[0].content == "Paris is the capital of France."
        assert doc.chunks[1].content == "London is the capital of the UK."

    async def test_load_metadata(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        result = await backend.load(tmp_path)
        meta = result.metadata
        assert meta["documents"] == 1
        assert meta["chunks"] == 2
        assert meta["size_bytes"] > 0

    async def test_load_file_not_found(self, tmp_path: str) -> None:
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load("/nonexistent/path.json")
        assert "does not exist" in str(exc.value)

    async def test_load_empty_string_path(self) -> None:
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError):
            await backend.load("")


class TestJsonBackendLoadUnicode:
    """Load preserves unicode content."""

    async def test_load_unicode(self, tmp_path: str) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk

        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="u1",
            title="日本語",
            content="东京是日本的首都。",
            chunks=(
                KnowledgeChunk(chunk_id="u1:0", document_id="u1", content="东京是日本的首都。", index=0),
            ),
        )
        kb.register(doc)
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, kb)
        result = await backend.load(tmp_path)
        kb2 = result.metadata["knowledge_base"]
        loaded = kb2.get("u1")
        assert loaded is not None
        assert "东京" in loaded.content
        assert "东京" in loaded.chunks[0].content


class TestJsonBackendLoadValidation:
    """Load validates file content."""

    async def test_invalid_json(self, tmp_path: str) -> None:
        with open(tmp_path, "w") as f:
            f.write("not json")
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "Failed to parse" in str(exc.value)

    async def test_unsupported_version(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": 999, "documents": [], "chunks": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "Unsupported" in str(exc.value)
        assert "999" in str(exc.value)

    async def test_missing_version(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"documents": [], "chunks": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "version" in str(exc.value)

    async def test_missing_documents(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": 1, "chunks": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "documents" in str(exc.value)

    async def test_missing_chunks(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": 1, "documents": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "chunks" in str(exc.value)

    async def test_version_not_int(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": "one", "documents": [], "chunks": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "Unsupported" in str(exc.value)

    async def test_documents_not_list(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": 1, "documents": "not a list", "chunks": []}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "documents" in str(exc.value).lower()

    async def test_chunks_not_list(self, tmp_path: str) -> None:
        import json
        with open(tmp_path, "w") as f:
            json.dump({"version": 1, "documents": [], "chunks": "not a list"}, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "chunks" in str(exc.value).lower()

    async def test_root_not_object(self, tmp_path: str) -> None:
        with open(tmp_path, "w") as f:
            f.write("[]")
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "object" in str(exc.value).lower()


class TestJsonBackendLoadDuplicates:
    """Load rejects duplicate document_ids and chunk_ids."""

    async def test_duplicate_document_id(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [
                {"document_id": "dup", "title": "A", "content": "A"},
                {"document_id": "dup", "title": "B", "content": "B"},
            ],
            "chunks": [],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "Duplicate" in str(exc.value)
        assert "dup" in str(exc.value)

    async def test_duplicate_chunk_id(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [],
            "chunks": [
                {"chunk_id": "c1", "document_id": "d1", "content": "A", "index": 0},
                {"chunk_id": "c1", "document_id": "d2", "content": "B", "index": 1},
            ],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "Duplicate" in str(exc.value)
        assert "c1" in str(exc.value)


class TestJsonBackendLoadMissingFields:
    """Load validates required fields on chunks and documents."""

    async def test_chunk_missing_chunk_id(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [],
            "chunks": [{"document_id": "d1", "content": "A", "index": 0}],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "chunk_id" in str(exc.value)

    async def test_chunk_missing_document_id(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [],
            "chunks": [{"chunk_id": "c1", "content": "A", "index": 0}],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "document_id" in str(exc.value)

    async def test_document_missing_document_id(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [{"title": "No ID", "content": "Missing ID"}],
            "chunks": [],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "document_id" in str(exc.value)


class TestJsonBackendLoadCorruptedReferences:
    """Load rejects embeddings/vectors referencing unknown chunks."""

    async def test_embedding_unknown_chunk(self, tmp_path: str, populated_kb: Any) -> None:
        import json
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        # Inject a bad embedding reference
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        data["embeddings"] = [{"chunk_id": "nonexistent", "vector": [0.1], "dimensions": 1}]
        with open(tmp_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "unknown chunk" in str(exc.value)

    async def test_vector_unknown_chunk(self, tmp_path: str, populated_kb: Any) -> None:
        import json
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        # Inject a bad vector reference
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        data["vectors"] = [{"chunk_id": "nonexistent", "vector": [0.1]}]
        with open(tmp_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(PersistenceError) as exc:
            await backend.load(tmp_path)
        assert "unknown chunk" in str(exc.value)

    async def test_chunk_orphan_without_document(self, tmp_path: str) -> None:
        import json
        payload = {
            "version": 1,
            "documents": [],
            "chunks": [{"chunk_id": "orphan", "document_id": "missing_doc", "content": "X", "index": 0}],
        }
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        backend = JsonPersistenceBackend()
        # Orphan chunks are gracefully ignored (they simply won't be
        # attached to any document in the KB)
        result = await backend.load(tmp_path)
        assert result.success is True


class TestJsonBackendLoadRestoreEmbeddings:
    """Load restores embeddings when present."""

    async def test_embeddings_restored(self, tmp_path: str, kb_with_embeddings: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=True),
        )
        await backend.save(tmp_path, kb_with_embeddings)
        result = await backend.load(tmp_path)
        kb = result.metadata["knowledge_base"]
        vec = kb.get_embedding("emb_doc:0")
        assert vec is not None
        assert len(vec.vector) == 4
        assert vec.provider == "det"

    async def test_embeddings_not_present(self, tmp_path: str, populated_kb: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=True),
        )
        await backend.save(tmp_path, populated_kb)
        result = await backend.load(tmp_path)
        kb = result.metadata["knowledge_base"]
        # No embeddings were saved for populated_kb
        assert kb.list_embeddings() == []

    async def test_embeddings_not_in_file(self, tmp_path: str, kb_with_embeddings: Any) -> None:
        """Load succeeds even when embeddings were excluded from save."""
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=False),
        )
        await backend.save(tmp_path, kb_with_embeddings)
        result = await backend.load(tmp_path)
        assert result.success is True


class TestJsonBackendLoadRestoreVectors:
    """Load restores vector store when present."""

    async def test_vectors_restored(self, tmp_path: str, kb_with_vectors: Any) -> None:
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_vectors=True),
        )
        await backend.save(tmp_path, kb_with_vectors)
        result = await backend.load(tmp_path)
        kb = result.metadata["knowledge_base"]
        vs = kb.vector_store
        assert vs is not None
        assert vs.count() == 1
        assert vs.contains("vec_doc:0")

    async def test_vectors_not_in_file(self, tmp_path: str, kb_with_vectors: Any) -> None:
        """Load succeeds even when vectors were excluded from save."""
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_vectors=False),
        )
        await backend.save(tmp_path, kb_with_vectors)
        result = await backend.load(tmp_path)
        kb = result.metadata["knowledge_base"]
        assert kb.vector_store is None


class TestJsonBackendLoadRoundTrip:
    """Save → load → save produces identical output."""

    async def test_round_trip_empty(self, tmp_path: str, empty_kb: Any) -> None:
        import json
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, empty_kb)
        r = await backend.load(tmp_path)
        kb2 = r.metadata["knowledge_base"]
        # Re-save to same path (overwrite=True)
        await backend.save(tmp_path, kb2)

        # Save was round-tripped — contents should be equivalent
        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["documents"] == []
        assert data["metadata"]["document_count"] == 0

    async def test_round_trip_populated(
        self, tmp_path: str, populated_kb: Any
    ) -> None:
        import json
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)
        r = await backend.load(tmp_path)
        kb2 = r.metadata["knowledge_base"]
        # Load → save should preserve content
        await backend.save(tmp_path, kb2)

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["documents"]) == 1
        assert len(data["chunks"]) == 2
        assert data["documents"][0]["document_id"] == "doc_1"

    async def test_round_trip_with_embeddings(
        self, tmp_path: str, kb_with_embeddings: Any
    ) -> None:
        import json
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True, include_embeddings=True),
        )
        await backend.save(tmp_path, kb_with_embeddings)
        r = await backend.load(tmp_path)
        kb2 = r.metadata["knowledge_base"]
        await backend.save(tmp_path, kb2)

        with open(tmp_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert len(data["documents"]) == 1
        assert len(data.get("embeddings", [])) == 1

    async def test_deterministic_round_trip(self, tmp_path: str, populated_kb: Any) -> None:
        """Loading from the same file twice yields equivalent state."""
        backend = JsonPersistenceBackend(
            config=PersistenceConfig(overwrite=True),
        )
        await backend.save(tmp_path, populated_kb)

        r1 = await backend.load(tmp_path)
        r2 = await backend.load(tmp_path)

        kb1 = r1.metadata["knowledge_base"]
        kb2 = r2.metadata["knowledge_base"]

        assert kb1.count() == kb2.count()
        assert kb1.get("doc_1").title == kb2.get("doc_1").title
        assert len(kb1.get("doc_1").chunks) == len(kb2.get("doc_1").chunks)
