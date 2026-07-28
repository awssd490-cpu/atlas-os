"""Tests for ProviderFactory."""

from __future__ import annotations

import pytest

from app.provider.errors import ProviderNotFoundError
from app.provider.factory import ProviderFactory
from app.provider.registry import ProviderRegistry

from tests.unit.provider.test_provider import _EchoProvider

_TEST_CONFIG = {"api_key": "test-key-12345"}


class TestProviderFactory:
    def test_register_constructor(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        assert "echo" in factory.list_constructors()

    def test_register_invalid_class(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)

        class NotAProvider:
            pass

        with pytest.raises(ValueError, match="does not implement"):
            factory.register_constructor("bad", NotAProvider)  # type: ignore[arg-type]

    def test_unregister_constructor(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        factory.unregister_constructor("echo")
        assert "echo" not in factory.list_constructors()

    async def test_create(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        provider = await factory.create("echo", config=_TEST_CONFIG, register=False)
        assert provider is not None
        assert reg.count() == 0  # not registered

    async def test_create_with_register(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        provider = await factory.create("echo", config=_TEST_CONFIG, register=True)
        assert reg.count() == 1
        assert reg.lookup("echo") is provider

    async def test_create_set_default(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        await factory.create("echo", config=_TEST_CONFIG, register=True, set_default=True)
        assert reg.default_provider() is not None

    async def test_create_not_found(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        with pytest.raises(ProviderNotFoundError):
            await factory.create("unknown", config=_TEST_CONFIG)

    async def test_create_and_initialize(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        provider = await factory.create_and_initialize("echo", config=_TEST_CONFIG, register=True)
        assert provider._initialized is True

    async def test_initialize_all(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        await factory.create("echo", config=_TEST_CONFIG, register=False)
        await factory.initialize_all()

    async def test_shutdown_all(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        p = await factory.create("echo", config=_TEST_CONFIG, register=True)
        await factory.shutdown_all()
        assert p._shutdown is True

    async def test_list_constructors_empty(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        assert factory.list_constructors() == []

    async def test_create_with_provider_config(self) -> None:
        from app.provider.config import ProviderConfig

        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        pc = ProviderConfig(name="echo", credentials=ProviderConfig.from_dict({"api_key": "k"}).credentials)
        provider = await factory.create("echo", provider_config=pc, register=False)
        assert provider is not None

    async def test_config_validation_failure(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        with pytest.raises(ValueError, match="configuration invalid"):
            await factory.create("echo", config={}, register=False)
