"""Tests for the persistence architecture."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.persistence import (
    InvalidPersistenceConfiguration,
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
