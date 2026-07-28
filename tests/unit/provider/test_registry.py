"""Tests for ProviderRegistry."""

from __future__ import annotations

import pytest

from app.provider.errors import DuplicateProviderError, ProviderNotFoundError
from app.provider.provider import Provider
from app.provider.registry import ProviderRegistry

from tests.unit.provider.test_provider import _EchoProvider


class TestProviderRegistry:
    def test_empty_registry(self) -> None:
        reg = ProviderRegistry()
        assert reg.count() == 0
        assert reg.list_providers() == []

    def test_register(self) -> None:
        reg = ProviderRegistry()
        provider = _EchoProvider()
        reg.register("echo", provider)
        assert reg.count() == 1
        assert "echo" in reg.list_providers()

    def test_register_duplicate(self) -> None:
        reg = ProviderRegistry()
        reg.register("echo", _EchoProvider())
        with pytest.raises(DuplicateProviderError):
            reg.register("echo", _EchoProvider())

    def test_lookup(self) -> None:
        reg = ProviderRegistry()
        provider = _EchoProvider()
        reg.register("echo", provider)
        found = reg.lookup("echo")
        assert found is provider

    def test_lookup_not_found(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            reg.lookup("nonexistent")

    def test_unregister(self) -> None:
        reg = ProviderRegistry()
        reg.register("echo", _EchoProvider())
        reg.unregister("echo")
        assert reg.count() == 0

    def test_unregister_not_found(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            reg.unregister("missing")

    def test_default_provider(self) -> None:
        reg = ProviderRegistry()
        p1 = _EchoProvider()
        reg.register("p1", p1)
        assert reg.default_provider() is p1

    def test_default_with_multiple(self) -> None:
        reg = ProviderRegistry()
        p1 = _EchoProvider()
        p2 = _EchoProvider()
        reg.register("p1", p1)
        reg.register("p2", p2, default=True)
        assert reg.default_provider() is p2

    def test_default_name_setter(self) -> None:
        reg = ProviderRegistry()
        reg.register("a", _EchoProvider())
        reg.register("b", _EchoProvider())
        reg.default_name = "b"
        assert reg.default_name == "b"

    def test_default_name_setter_invalid(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            reg.default_name = "missing"

    def test_default_not_found_when_empty(self) -> None:
        reg = ProviderRegistry()
        with pytest.raises(ProviderNotFoundError):
            reg.default_provider()

    def test_find_by_capability(self) -> None:
        reg = ProviderRegistry()
        reg.register("basic", _EchoProvider())
        results = reg.find_by_capability("temperature")
        assert len(results) == 1
        assert results[0][0] == "basic"

    def test_find_by_capability_no_match(self) -> None:
        reg = ProviderRegistry()
        reg.register("basic", _EchoProvider())
        results = reg.find_by_capability("vision")
        assert len(results) == 0

    def test_unregister_updates_default(self) -> None:
        reg = ProviderRegistry()
        reg.register("a", _EchoProvider())
        reg.register("b", _EchoProvider(), default=True)
        reg.unregister("b")
        # Default should switch to 'a'
        assert reg.default_name == "a"
