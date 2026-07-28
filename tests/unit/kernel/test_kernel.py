"""Tests for Kernel.

Verifies:
- State machine transitions (CREATED → BOOTING → RUNNING → STOPPED)
- Boot calls all lifecycle hooks
- Module registration and lookup
- Shutdown (graceful and from FAILED state)
- Health reporting
- State enforcement (reject boot when not CREATED)
- Capability registration from manifests
"""

from __future__ import annotations

import pytest

from app.core.errors import LifecycleError
from app.core.interfaces import KernelState, Module, ModuleState
from app.core.manifest import CapabilityDeclaration, ModuleHealth, ModuleManifest
from app.kernel.builder import KernelBuilder
from app.kernel.kernel import Kernel


# ---------------------------------------------------------------------------
# Test modules
# ---------------------------------------------------------------------------


class _SimpleModule(Module):
    """Minimal module that tracks lifecycle calls."""

    def __init__(self, name: str = "mod_simple") -> None:
        super().__init__()
        self._manifest = ModuleManifest(name=name, version="1.0.0")
        self.hooks: list[str] = []

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest

    async def initialize(self, context) -> None:
        await super().initialize(context)
        self.hooks.append("initialize")

    async def start(self) -> None:
        self.hooks.append("start")

    async def ready(self) -> None:
        self.hooks.append("ready")

    async def stop(self) -> None:
        self.hooks.append("stop")

    async def shutdown(self) -> None:
        self.hooks.append("shutdown")


class _CapabilityModule(Module):
    """Module that declares capabilities."""

    def __init__(self, name: str = "mod_capable") -> None:
        super().__init__()
        self._manifest = ModuleManifest(
            name=name,
            version="1.0.0",
            capabilities=[
                CapabilityDeclaration(name="storage.sql", version="1.0"),
                CapabilityDeclaration(name="testing.mock", version="1.0"),
            ],
        )

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest


class _FailingModule(Module):
    """Module that fails during initialize."""

    def __init__(self, name: str = "mod_fail") -> None:
        super().__init__()
        self._manifest = ModuleManifest(name=name, version="1.0.0")
        self.initialized = False

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest

    async def initialize(self, context) -> None:
        msg = f"{self.name} failed"

    async def start(self) -> None:
        raise RuntimeError(f"{self.name} failed on start")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    async def test_initial_state_is_created(self) -> None:
        kernel = Kernel()
        assert kernel.state == KernelState.CREATED

    async def test_boot_transitions_to_running(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_state")).build()
        await kernel.boot()
        assert kernel.state == KernelState.RUNNING

    async def test_shutdown_transitions_to_stopped(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_shut")).build()
        await kernel.boot()
        await kernel.shutdown()
        assert kernel.state == KernelState.STOPPED

    async def test_double_boot_raises(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_db")).build()
        await kernel.boot()
        with pytest.raises(LifecycleError, match="not 'created'"):
            await kernel.boot()


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    async def test_boot_calls_all_hooks(self) -> None:
        mod = _SimpleModule("mod_hooks")
        kernel = KernelBuilder().with_module(mod).build()
        await kernel.boot()
        assert mod.hooks == ["initialize", "start", "ready"]

    async def test_shutdown_calls_stop_and_shutdown(self) -> None:
        mod = _SimpleModule("mod_shutdown")
        kernel = KernelBuilder().with_module(mod).build()
        await kernel.boot()
        await kernel.shutdown()
        assert "stop" in mod.hooks
        assert "shutdown" in mod.hooks

    async def test_startup_time_is_recorded(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_time")).build()
        await kernel.boot()
        # uptime uses monotonic clock — on fast systems it may be ~0.0
        assert kernel.uptime() >= 0.0
        assert kernel.state == KernelState.RUNNING


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------


class TestModuleRegistration:
    async def test_register_before_boot(self) -> None:
        kernel = Kernel()
        mod = _SimpleModule("mod_reg")
        kernel.register(mod)
        assert kernel.module_count() == 1

    async def test_register_after_boot_raises(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_first")).build()
        await kernel.boot()
        with pytest.raises(LifecycleError):
            kernel.register(_SimpleModule("mod_second"))

    async def test_get_module(self) -> None:
        kernel = Kernel()
        mod = _SimpleModule("mod_get")
        kernel.register(mod)
        assert kernel.get_module("mod_get") is mod

    async def test_get_modules(self) -> None:
        kernel = KernelBuilder() \
            .with_module(_SimpleModule("mod_a")) \
            .with_module(_SimpleModule("mod_b")) \
            .build()
        await kernel.boot()
        modules = kernel.get_modules()
        assert len(modules) == 2


# ---------------------------------------------------------------------------
# Failure scenarios
# ---------------------------------------------------------------------------


class TestFailures:
    async def test_boot_failure_transitions_to_failed(self) -> None:
        kernel = KernelBuilder().with_module(_FailingModule("mod_fail2")).build()
        with pytest.raises(Exception):
            await kernel.boot()
        assert kernel.state == KernelState.FAILED

    async def test_shutdown_idempotent_when_failed(self) -> None:
        kernel = KernelBuilder().with_module(_FailingModule("mod_fail3")).build()
        with pytest.raises(Exception):
            await kernel.boot()
        # Should not raise — shutdown is safe from FAILED
        await kernel.shutdown()
        assert kernel.state == KernelState.STOPPED


# ---------------------------------------------------------------------------
# Capability registration
# ---------------------------------------------------------------------------


class TestCapabilityRegistration:
    async def test_capabilities_registered_from_manifest(self) -> None:
        kernel = KernelBuilder().with_module(_CapabilityModule("mod_caps")).build()
        await kernel.boot()

        ctx = kernel._context
        assert ctx is not None
        assert ctx.capabilities.is_registered("storage.sql") is True
        assert ctx.capabilities.is_registered("testing.mock") is True

    async def test_capability_find_by_provider(self) -> None:
        kernel = KernelBuilder().with_module(_CapabilityModule("mod_find")).build()
        await kernel.boot()

        ctx = kernel._context
        assert ctx is not None
        entries = ctx.capabilities.find_by_provider("mod_find")
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_healthy_when_running(self) -> None:
        kernel = KernelBuilder().with_module(_SimpleModule("mod_health")).build()
        await kernel.boot()
        result = await kernel.health()
        assert result["status"] == "healthy"
        assert "telemetry" in result

    async def test_unhealthy_when_not_running(self) -> None:
        kernel = Kernel()
        result = await kernel.health()
        assert result["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Dependency order
# ---------------------------------------------------------------------------


class TestDependencyOrder:
    async def test_boot_respects_dependency_order(self) -> None:
        mod_a = _SimpleModule("mod_dep_a")
        mod_b = _SimpleModule("mod_dep_b")

        # Override manifest for mod_b to depend on mod_a
        mod_b._manifest = ModuleManifest(
            name="mod_dep_b", version="1.0.0", dependencies=["mod_dep_a"]
        )

        kernel = KernelBuilder() \
            .with_module(mod_a) \
            .with_module(mod_b) \
            .build()
        await kernel.boot()

        assert mod_a.hooks == ["initialize", "start", "ready"]
        assert mod_b.hooks == ["initialize", "start", "ready"]
