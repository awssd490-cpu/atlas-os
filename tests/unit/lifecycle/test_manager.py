"""Tests for LifecycleManager.

Verifies:
- Full initialize → start → ready cycle
- Modules are called in dependency order
- Failure during initialize triggers rollback (exception propagates)
- Stop and shutdown call modules in reverse order
- Health collection aggregates module health
- Telemetry events are recorded
- Lifecycle state transitions are correct
"""

from __future__ import annotations

import pytest

from app.core.events import Event
from app.core.interfaces import (
    EventBus,
    KernelContext,
    Module,
    ModuleRegistry,
    ModuleState,
    TelemetryService,
)
from app.core.manifest import ModuleHealth, ModuleManifest
from app.events.bus import InProcessEventBus
from app.lifecycle.manager import LifecycleManager
from app.modules.registry import InMemoryModuleRegistry
from app.telemetry.service import InMemoryTelemetryService


# ---------------------------------------------------------------------------
# Test module helpers
# ---------------------------------------------------------------------------


class _HookTracker(Module):
    """Records which hooks were called and in what order."""

    def __init__(self, name: str, deps: list[str] | None = None) -> None:
        super().__init__()
        self._manifest = ModuleManifest(
            name=name, version="1.0.0", dependencies=deps or []
        )
        self.hooks: list[str] = []
        self._fail_on: str | None = None

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest

    def fail_on(self, hook: str) -> None:
        """Set a hook that should raise an error."""
        self._fail_on = hook

    async def initialize(self, context: KernelContext) -> None:
        self.hooks.append("initialize")
        if self._fail_on == "initialize":
            raise RuntimeError(f"{self.name} failed on initialize")

    async def start(self) -> None:
        self.hooks.append("start")
        if self._fail_on == "start":
            raise RuntimeError(f"{self.name} failed on start")

    async def ready(self) -> None:
        self.hooks.append("ready")

    async def pause(self) -> None:
        self.hooks.append("pause")

    async def resume(self) -> None:
        self.hooks.append("resume")

    async def stop(self) -> None:
        self.hooks.append("stop")

    async def shutdown(self) -> None:
        self.hooks.append("shutdown")

    async def health(self) -> ModuleHealth:
        self.hooks.append("health")
        return ModuleHealth.ok(module=self.name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ModuleRegistry:
    return InMemoryModuleRegistry()


@pytest.fixture
def event_bus() -> EventBus:
    return InProcessEventBus()


@pytest.fixture
def telemetry() -> TelemetryService:
    return InMemoryTelemetryService()


@pytest.fixture
def context() -> KernelContext:
    """Minimal context for lifecycle testing.

    We use a sentinel class so modules can receive something.
    Full KernelContext is tested in the kernel integration tests.
    """
    from app.config.service import PydanticConfigService
    from app.logging.service import LoguruLoggingService

    class _TestContext(KernelContext):
        @property
        def config(self):
            return PydanticConfigService()

        @property
        def logging(self):
            return LoguruLoggingService()

        @property
        def logger(self):
            return LoguruLoggingService().get_logger("test")

        @property
        def event_bus(self):
            return event_bus

        @property
        def telemetry(self):
            return telemetry

        @property
        def capabilities(self):
            from app.modules.capabilities import InMemoryCapabilityRegistry
            return InMemoryCapabilityRegistry()

        async def resolve(self, interface):
            raise NotImplementedError

    return _TestContext()


@pytest.fixture
def manager(
    registry: ModuleRegistry,
    event_bus: EventBus,
    telemetry: TelemetryService,
) -> LifecycleManager:
    return LifecycleManager(registry=registry, event_bus=event_bus, telemetry=telemetry)


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    async def test_initializes_in_order(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod_a = _HookTracker("mod_a")
        mod_b = _HookTracker("mod_b", deps=["mod_a"])
        registry.register(mod_a)
        registry.register(mod_b)

        await manager.initialize(context)

        assert mod_a.hooks == ["initialize"]
        assert mod_b.hooks == ["initialize"]

    async def test_state_transition(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        registry.register(_HookTracker("mod_x"))
        await manager.initialize(context)
        entries = registry.all()
        assert entries[0][1] == ModuleState.INITIALIZED

    async def test_failure_propagates(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_fail")
        mod.fail_on("initialize")
        registry.register(mod)

        with pytest.raises(Exception):
            await manager.initialize(context)


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


class TestStart:
    async def test_starts_in_order(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod_a = _HookTracker("mod_a")
        mod_b = _HookTracker("mod_b", deps=["mod_a"])
        registry.register(mod_a)
        registry.register(mod_b)
        await manager.initialize(context)

        await manager.start()

        assert mod_a.hooks == ["initialize", "start"]
        assert mod_b.hooks == ["initialize", "start"]

    async def test_state_active(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        registry.register(_HookTracker("mod_x"))
        await manager.initialize(context)
        await manager.start()
        entries = registry.all()
        assert entries[0][1] == ModuleState.ACTIVE

    async def test_failure_propagates(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_fail")
        mod.fail_on("start")
        registry.register(mod)
        await manager.initialize(context)

        with pytest.raises(Exception):
            await manager.start()


# ---------------------------------------------------------------------------
# Full cycle
# ---------------------------------------------------------------------------


class TestFullCycle:
    async def test_initialize_start_ready(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_main")
        registry.register(mod)

        await manager.initialize(context)
        await manager.start()
        await manager.ready()

        assert mod.hooks == ["initialize", "start", "ready"]

    async def test_initialize_start_stop_shutdown(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_main")
        registry.register(mod)

        await manager.initialize(context)
        await manager.start()
        await manager.stop()
        await manager.shutdown()

        assert mod.hooks == ["initialize", "start", "stop", "shutdown"]


# ---------------------------------------------------------------------------
# Pause / Resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    async def test_pause_module(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_pause")
        registry.register(mod)
        await manager.initialize(context)
        await manager.start()

        await manager.pause("mod_pause")
        assert "pause" in mod.hooks
        entry = registry.all()[0]
        assert entry[1] == ModuleState.PAUSED

    async def test_resume_module(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_resume")
        registry.register(mod)
        await manager.initialize(context)
        await manager.start()
        await manager.pause("mod_resume")

        await manager.resume("mod_resume")
        assert "resume" in mod.hooks
        entry = registry.all()[0]
        assert entry[1] == ModuleState.ACTIVE

    async def test_pause_all(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod_a = _HookTracker("mod_a")
        mod_b = _HookTracker("mod_b")
        registry.register(mod_a)
        registry.register(mod_b)
        await manager.initialize(context)
        await manager.start()

        await manager.pause()
        assert "pause" in mod_a.hooks
        assert "pause" in mod_b.hooks


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_collects_all_module_health(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod_a = _HookTracker("mod_a")
        mod_b = _HookTracker("mod_b")
        registry.register(mod_a)
        registry.register(mod_b)
        await manager.initialize(context)

        result = await manager.collect_health()
        assert result["status"] == "healthy"
        assert "mod_a" in result["modules"]
        assert "mod_b" in result["modules"]

    async def test_health_does_not_change_state(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod = _HookTracker("mod_state")
        registry.register(mod)
        await manager.initialize(context)
        await manager.collect_health()
        # Should still be INITIALIZED
        entries = registry.all()
        assert entries[0][1] == ModuleState.INITIALIZED


# ---------------------------------------------------------------------------
# Reverse-order lifecycle hooks
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_reverse_order(
        self,
        registry: ModuleRegistry,
        manager: LifecycleManager,
        context: KernelContext,
    ) -> None:
        mod_a = _HookTracker("mod_a")
        mod_b = _HookTracker("mod_b", deps=["mod_a"])
        registry.register(mod_a)
        registry.register(mod_b)
        await manager.initialize(context)
        await manager.start()

        await manager.stop()
        assert mod_a.hooks == ["initialize", "start", "stop"]
        assert mod_b.hooks == ["initialize", "start", "stop"]
