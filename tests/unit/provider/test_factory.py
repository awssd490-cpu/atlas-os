"""Tests for ProviderFactory."""

from __future__ import annotations

import pytest

from app.provider.errors import ProviderNotFoundError
from app.provider.factory import ProviderFactory
from app.provider.registry import ProviderRegistry

from tests.unit.provider.test_provider import _EchoProvider


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
        provider = await factory.create("echo", register=False)
        assert provider is not None
        assert reg.count() == 0  # not registered

    async def test_create_with_register(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        provider = await factory.create("echo", register=True)
        assert reg.count() == 1
        assert reg.lookup("echo") is provider

    async def test_create_set_default(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        await factory.create("echo", register=True, set_default=True)
        assert reg.default_provider() is not None

    async def test_create_not_found(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        with pytest.raises(ProviderNotFoundError):
            await factory.create("unknown")

    async def test_create_and_initialize(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        provider = await factory.create_and_initialize("echo", register=True)
        assert provider._initialized is True

    async def test_initialize_all(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        await factory.create("echo", register=False)
        # Should not raise
        await factory.initialize_all()

    async def test_shutdown_all(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        p = await factory.create("echo", register=True)
        await factory.shutdown_all()
        assert p._shutdown is True

    async def test_list_constructors_empty(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        assert factory.list_constructors() == []

    @property
    def test_create_default(self) -> None:
        reg = ProviderRegistry()
        factory = ProviderFactory(reg)
        factory.register_constructor("echo", _EchoProvider)
        reg.register("echo", _EchoProvider(), default=True)
        # create_default would need an async test
