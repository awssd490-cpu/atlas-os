"""Tests for InMemoryCapabilityRegistry.

Verifies:
- Register and query capabilities
- Multiple providers per capability
- ``get()`` returns first-registered
- ``get_all()`` returns all in order
- ``find_by_provider``
- ``is_registered``
"""

from __future__ import annotations

from app.core.manifest import CapabilityDeclaration
from app.modules.capabilities import InMemoryCapabilityRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def registry() -> InMemoryCapabilityRegistry:
    return InMemoryCapabilityRegistry()


CAP_A = CapabilityDeclaration(name="storage.sql", version="1.0", description="SQL storage")
CAP_B = CapabilityDeclaration(name="memory.vector", version="2.0", description="Vector search")
CAP_C = CapabilityDeclaration(name="storage.sql", version="1.1", description="Alternative SQL storage")


class TestRegisterAndGet:
    def test_register_and_get(self) -> None:
        reg = registry()
        reg.register(CAP_A, "module_a")
        entry = reg.get("storage.sql")
        assert entry is not None
        assert entry.capability.name == "storage.sql"
        assert entry.provider == "module_a"

    def test_get_nonexistent(self) -> None:
        reg = registry()
        assert reg.get("nonexistent") is None

    def test_get_returns_first_registered(self) -> None:
        reg = registry()
        reg.register(CAP_A, "module_a")
        reg.register(CAP_C, "module_c")
        entry = reg.get("storage.sql")
        assert entry is not None
        assert entry.provider == "module_a"  # first wins


class TestGetAll:
    def test_returns_all_providers(self) -> None:
        reg = registry()
        reg.register(CAP_A, "module_a")
        reg.register(CAP_C, "module_c")
        entries = reg.get_all("storage.sql")
        assert len(entries) == 2

    def test_empty_when_none(self) -> None:
        reg = registry()
        assert reg.get_all("nonexistent") == []


class TestFindByProvider:
    def test_returns_module_capabilities(self) -> None:
        reg = registry()
        reg.register(CAP_A, "module_a")
        reg.register(CAP_B, "module_a")
        entries = reg.find_by_provider("module_a")
        assert len(entries) == 2
        assert {e.capability.name for e in entries} == {"storage.sql", "memory.vector"}

    def test_empty_for_unknown_module(self) -> None:
        reg = registry()
        assert reg.find_by_provider("unknown") == []


class TestIsRegistered:
    def test_true_when_registered(self) -> None:
        reg = registry()
        reg.register(CAP_A, "module_a")
        assert reg.is_registered("storage.sql") is True

    def test_false_when_not_registered(self) -> None:
        reg = registry()
        assert reg.is_registered("nonexistent") is False
