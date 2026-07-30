"""Architecture tests for the embedding layer.

Checkpoint 1 — verifies imports, configuration, provider interface,
registry, model construction, error hierarchy, and immutability.
"""

from __future__ import annotations

import pytest

from app.rag.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingVector,
    InvalidEmbeddingConfiguration,
    UnsupportedEmbeddingProvider,
    clear_providers,
    get_provider,
    list_providers,
    register_provider,
)
from app.rag.embeddings.base import EmbeddingProvider as EmbeddingProvider_Impl
from app.rag.embeddings.config import EmbeddingConfig as EmbeddingConfig_Impl
from app.rag.embeddings.errors import EmbeddingError as EmbeddingError_Impl
from app.rag.embeddings.models import EmbeddingResult as EmbeddingResult_Impl
from app.rag.embeddings.models import EmbeddingVector as EmbeddingVector_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    """Verify that all public symbols import cleanly."""

    def test_embedding_config_imported(self) -> None:
        assert EmbeddingConfig is EmbeddingConfig_Impl

    def test_embedding_error_imported(self) -> None:
        assert EmbeddingError is EmbeddingError_Impl

    def test_embedding_provider_imported(self) -> None:
        assert EmbeddingProvider is EmbeddingProvider_Impl

    def test_embedding_result_imported(self) -> None:
        assert EmbeddingResult is EmbeddingResult_Impl

    def test_embedding_vector_imported(self) -> None:
        assert EmbeddingVector is EmbeddingVector_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(EmbeddingError, KnowledgeError)
        assert issubclass(InvalidEmbeddingConfiguration, EmbeddingError)
        assert issubclass(EmbeddingProviderError, EmbeddingError)
        assert issubclass(UnsupportedEmbeddingProvider, EmbeddingError)

    def test_registry_functions_imported(self) -> None:
        assert callable(register_provider)
        assert callable(get_provider)
        assert callable(list_providers)
        assert callable(clear_providers)


# ======================================================================
# EmbeddingConfig
# ======================================================================


class TestEmbeddingConfig:
    def test_default_values(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.provider_name == "openai"
        assert cfg.dimensions == 768
        assert cfg.batch_size == 32
        assert cfg.normalize_embeddings is True
        assert cfg.timeout == 30.0

    def test_custom_values(self) -> None:
        cfg = EmbeddingConfig(
            provider_name="ollama",
            dimensions=384,
            batch_size=16,
            normalize_embeddings=False,
            timeout=60.0,
        )
        assert cfg.provider_name == "ollama"
        assert cfg.dimensions == 384
        assert cfg.batch_size == 16
        assert cfg.normalize_embeddings is False
        assert cfg.timeout == 60.0

    def test_immutable(self) -> None:
        cfg = EmbeddingConfig()
        with pytest.raises(AttributeError):
            cfg.dimensions = 512  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        cfg = EmbeddingConfig(dimensions=384, batch_size=10, timeout=5.0)
        cfg.validate()  # should not raise

    def test_validate_dimensions_zero(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(dimensions=0).validate()

    def test_validate_dimensions_negative(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(dimensions=-1).validate()

    def test_validate_batch_size_zero(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(batch_size=0).validate()

    def test_validate_batch_size_negative(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(batch_size=-1).validate()

    def test_validate_timeout_zero(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(timeout=0).validate()

    def test_validate_timeout_negative(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(timeout=-1.0).validate()

    def test_validate_provider_name_empty(self) -> None:
        with pytest.raises(InvalidEmbeddingConfiguration):
            EmbeddingConfig(provider_name="").validate()


# ======================================================================
# EmbeddingVector
# ======================================================================


class TestEmbeddingVector:
    def test_default_values(self) -> None:
        v = EmbeddingVector()
        assert v.vector == ()
        assert v.dimensions == 0
        assert v.provider == ""
        assert v.created_at == 0.0
        assert v.metadata == {}

    def test_custom_values(self) -> None:
        v = EmbeddingVector(
            vector=(0.1, 0.2, 0.3),
            dimensions=3,
            provider="test_provider",
            created_at=1000.0,
            metadata={"text_length": 10},
        )
        assert v.vector == (0.1, 0.2, 0.3)
        assert v.dimensions == 3
        assert v.provider == "test_provider"
        assert v.created_at == 1000.0
        assert v.metadata == {"text_length": 10}

    def test_vector_is_tuple(self) -> None:
        """Vector must be a tuple, not a list."""
        v = EmbeddingVector(vector=(0.5,))
        assert isinstance(v.vector, tuple)

    def test_immutable(self) -> None:
        v = EmbeddingVector(dimensions=128)
        with pytest.raises(AttributeError):
            v.dimensions = 256  # type: ignore[misc]


# ======================================================================
# EmbeddingResult
# ======================================================================


class TestEmbeddingResult:
    def test_default_values(self) -> None:
        r = EmbeddingResult()
        assert r.embeddings == ()
        assert r.provider == ""
        assert r.config is None
        assert r.total_texts == 0
        assert r.elapsed_ms == 0.0

    def test_with_embeddings(self) -> None:
        vec = EmbeddingVector(vector=(0.1,), dimensions=1, provider="test")
        result = EmbeddingResult(
            embeddings=(vec,),
            provider="test",
            total_texts=1,
            elapsed_ms=5.2,
        )
        assert len(result.embeddings) == 1
        assert result.embeddings[0].dimensions == 1
        assert result.total_texts == 1
        assert result.elapsed_ms == 5.2

    def test_immutable(self) -> None:
        r = EmbeddingResult()
        with pytest.raises(AttributeError):
            r.total_texts = 5  # type: ignore[misc]


# ======================================================================
# EmbeddingProvider (abstract)
# ======================================================================


class TestEmbeddingProvider:
    def test_is_abstract(self) -> None:
        """EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        """Verify abstract methods exist on the class."""
        assert hasattr(EmbeddingProvider, "embed")
        assert hasattr(EmbeddingProvider, "embed_batch")

    def test_concrete_subclass(self) -> None:
        """A minimal concrete subclass can be created."""
        class TestProvider(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "test"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        config = EmbeddingConfig(provider_name="test", dimensions=64)
        provider = TestProvider(config)
        assert provider.name == "test"
        assert provider.config.dimensions == 64

    def test_config_property(self) -> None:
        """The config property returns the config passed at init."""
        class TestProvider(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "test"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        cfg = EmbeddingConfig(dimensions=128)
        provider = TestProvider(cfg)
        assert provider.config.dimensions == 128
        # Config should be immutable
        with pytest.raises(AttributeError):
            provider.config.dimensions = 256  # type: ignore[misc]

    def test_config_validation(self) -> None:
        """Provider subclasses can validate config in __init__."""
        class ValidatingProvider(EmbeddingProvider):
            def __init__(self, config: EmbeddingConfig) -> None:
                config.validate()
                super().__init__(config)

            @property
            def name(self) -> str:
                return "validating"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        with pytest.raises(InvalidEmbeddingConfiguration):
            ValidatingProvider(EmbeddingConfig(dimensions=0))

    def test_embed_is_async(self) -> None:
        """embed() returns a coroutine when called."""
        class AsyncProvider(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "async_test"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        import asyncio
        provider = AsyncProvider(EmbeddingConfig())
        result = asyncio.run(provider.embed("hello"))
        assert isinstance(result, EmbeddingResult)


# ======================================================================
# Registry
# ======================================================================


class TestEmbeddingRegistry:
    def test_register_and_get(self) -> None:
        class FakeProvider(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "fake"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        register_provider("fake", FakeProvider)
        cls = get_provider("fake")
        assert cls is FakeProvider
        clear_providers()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(UnsupportedEmbeddingProvider):
            get_provider("nonexistent")

    def test_register_duplicate_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.embeddings.registry._providers", {})

        class P1(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "dup"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        class P2(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "dup"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        register_provider("dup", P1)
        with pytest.raises(ValueError, match="already registered"):
            register_provider("dup", P2)

    def test_list_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.embeddings.registry._providers", {})

        class P(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "p"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        assert list_providers() == []
        register_provider("a", P)
        register_provider("b", P)
        assert set(list_providers()) == {"a", "b"}

    def test_clear_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.embeddings.registry._providers", {})

        class P(EmbeddingProvider):
            @property
            def name(self) -> str:
                return "p"

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult()

            async def embed_batch(self, texts: list[str]) -> EmbeddingResult:  # type: ignore[override]
                return EmbeddingResult()

        register_provider("p", P)
        assert list_providers() == ["p"]
        clear_providers()
        assert list_providers() == []


# ======================================================================
# Error hierarchy
# ======================================================================


class TestEmbeddingErrors:
    def test_embedding_error_message(self) -> None:
        err = EmbeddingError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "EMBEDDING_ERROR"

    def test_invalid_configuration_error(self) -> None:
        err = InvalidEmbeddingConfiguration("Bad config")
        assert err.code == "INVALID_EMBEDDING_CONFIGURATION"
        assert isinstance(err, EmbeddingError)

    def test_provider_error(self) -> None:
        err = EmbeddingProviderError("Provider failed")
        assert err.code == "EMBEDDING_PROVIDER_ERROR"
        assert isinstance(err, EmbeddingError)

    def test_unsupported_provider_error_with_name(self) -> None:
        err = UnsupportedEmbeddingProvider("ollama")
        assert "ollama" in str(err)
        assert err.code == "UNSUPPORTED_EMBEDDING_PROVIDER"

    def test_unsupported_provider_error_empty(self) -> None:
        err = UnsupportedEmbeddingProvider()
        assert str(err) == "Unsupported provider"
        assert err.code == "UNSUPPORTED_EMBEDDING_PROVIDER"

    def test_to_dict(self) -> None:
        err = InvalidEmbeddingConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_EMBEDDING_CONFIGURATION"
        assert d["message"] == "test"
        assert d["details"] == {"key": "val"}

    def test_knowledge_error_is_base(self) -> None:
        """EmbeddingError derives from KnowledgeError, not directly from AtlasError."""
        assert issubclass(EmbeddingError, KnowledgeError)
