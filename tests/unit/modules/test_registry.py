"""Tests for InMemoryModuleRegistry.

Verifies:
- Register and query modules
- Duplicate registration raises ValueError
- ``get()`` on unknown raises ``ModuleNotFoundError_``
- ``all()`` returns registered modules
- ``update_state`` transitions
- ``boot_order`` topological sort
- Missing dependency detection
- Cycle detection
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.errors import ModuleDependencyError, ModuleNotFoundError_
from app.core.interfaces import Module, ModuleState
from app.core.manifest import ModuleManifest
from app.modules.registry import InMemoryModuleRegistry


# ---------------------------------------------------------------------------
# Test module — factory function
# ---------------------------------------------------------------------------


class _TestModule(Module):
    """Helper module used across tests."""

    def __init__(self, manifest: ModuleManifest) -> None:
        super().__init__()
        self._manifest = manifest

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest


def make_module(name: str, deps: list[str] | None = None) -> Module:
    """Create a test module with the given name and dependencies."""
    manifest = ModuleManifest(
        name=name,
        version="1.0.0",
        description=f"Module {name}",
        dependencies=deps or [],
    )
    return _TestModule(manifest=manifest)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> InMemoryModuleRegistry:
    return InMemoryModuleRegistry()


# ---------------------------------------------------------------------------
# Register / query
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_adds_module(self, registry: InMemoryModuleRegistry) -> None:
        mod = make_module("mod_test")
        registry.register(mod)
        assert registry.count() == 1

    def test_duplicate_name_raises(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(make_module("mod_dup"))


class TestGet:
    def test_returns_module(self, registry: InMemoryModuleRegistry) -> None:
        mod = make_module("mod_alpha")
        registry.register(mod)
        assert registry.get("mod_alpha") is mod

    def test_missing_raises(self, registry: InMemoryModuleRegistry) -> None:
        with pytest.raises(ModuleNotFoundError_, match="not found"):
            registry.get("mod_unknown")


class TestAll:
    def test_returns_all_with_states(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_a"))
        registry.register(make_module("mod_b"))
        all_modules = registry.all()
        assert len(all_modules) == 2
        names = [m[0] for m in all_modules]
        assert "mod_a" in names
        assert "mod_b" in names

    def test_initial_state_is_registered(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_init"))
        entry = registry.all()[0]
        assert entry[1] == ModuleState.REGISTERED

    def test_empty_when_no_modules(self) -> None:
        registry = InMemoryModuleRegistry()
        assert registry.all() == []


class TestCount:
    def test_count_zero(self) -> None:
        registry = InMemoryModuleRegistry()
        assert registry.count() == 0

    def test_count_after_register(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_first"))
        registry.register(make_module("mod_second"))
        assert registry.count() == 2


class TestUpdateState:
    def test_state_transitions(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_state"))
        registry.update_state("mod_state", ModuleState.ACTIVE)
        all_mods = registry.all()
        assert all_mods[0][1] == ModuleState.ACTIVE

    def test_update_unknown_raises(self, registry: InMemoryModuleRegistry) -> None:
        with pytest.raises(ModuleNotFoundError_):
            registry.update_state("mod_unknown", ModuleState.ACTIVE)


# ---------------------------------------------------------------------------
# Boot order (topological sort)
# ---------------------------------------------------------------------------


class TestBootOrder:
    def test_single_module(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_standalone"))
        order = registry.boot_order()
        assert len(order) == 1
        assert order[0].name == "mod_standalone"

    def test_independent_modules(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_first"))
        registry.register(make_module("mod_second"))
        order = registry.boot_order()
        names = [m.name for m in order]
        assert "mod_first" in names
        assert "mod_second" in names

    def test_dependency_order(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_b", deps=["mod_a"]))
        registry.register(make_module("mod_a"))
        order = registry.boot_order()
        names = [m.name for m in order]
        assert names.index("mod_a") < names.index("mod_b")

    def test_chain_dependency(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_c", deps=["mod_b"]))
        registry.register(make_module("mod_b", deps=["mod_a"]))
        registry.register(make_module("mod_a"))
        order = registry.boot_order()
        names = [m.name for m in order]
        assert names.index("mod_a") < names.index("mod_b") < names.index("mod_c")

    def test_missing_dependency_raises(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_main", deps=["mod_missing"]))
        with pytest.raises(ModuleDependencyError, match="not registered"):
            registry.boot_order()

    def test_cycle_detected(self, registry: InMemoryModuleRegistry) -> None:
        registry.register(make_module("mod_x", deps=["mod_y"]))
        registry.register(make_module("mod_y", deps=["mod_x"]))
        with pytest.raises(ModuleDependencyError, match="Circular dependency"):
            registry.boot_order()
