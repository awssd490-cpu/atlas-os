"""Tests for MemoryModule.

Verifies:
- Manifest declares memory.store, memory.search, memory.state capabilities
- Initialize and start creates manager and runs migration
- Health returns healthy after start
- Shutdown cleans up
- Full lifecycle
"""

from __future__ import annotations

import pytest

from app.core.interfaces import KernelContext
from app.memory.module import MemoryModule


# ---------------------------------------------------------------------------
# Test context builder
# ---------------------------------------------------------------------------


def _make_context(tmp_path: str) -> KernelContext:
    """Build a minimal KernelContext for testing the memory module."""
    from app.config.service import PydanticConfigService
    from app.config.settings import AtlasSettings
    from app.logging.service import LoguruLoggingService
    from app.events.bus import InProcessEventBus
    from app.telemetry.service import InMemoryTelemetryService
    from app.modules.capabilities import InMemoryCapabilityRegistry

    settings = AtlasSettings(
        storage={"sqlite_path": f"{tmp_path}/memory_test.db"},
        memory={
            "archive_threshold": 0.2,
            "grace_period_seconds": 3600.0,
        },
    )
    config = PydanticConfigService(settings=settings)
    logging_service = LoguruLoggingService()
    logger = logging_service.get_logger("test")

    class _TestContext(KernelContext):
        @property
        def config(self):
            return config

        @property
        def logging(self):
            return logging_service

        @property
        def logger(self):
            return logger

        @property
        def event_bus(self):
            return InProcessEventBus()

        @property
        def telemetry(self):
            return InMemoryTelemetryService()

        @property
        def capabilities(self):
            return InMemoryCapabilityRegistry()

        async def resolve(self, interface):
            raise NotImplementedError

    return _TestContext()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_name(self) -> None:
        mod = MemoryModule()
        assert mod.manifest.name == "memory"
        assert mod.manifest.version == "1.0.0"

    def test_manifest_depends_on_storage(self) -> None:
        mod = MemoryModule()
        assert "storage" in mod.manifest.dependencies

    def test_manifest_declares_memory_capabilities(self) -> None:
        mod = MemoryModule()
        cap_names = {c.name for c in mod.manifest.capabilities}
        assert "memory.store" in cap_names
        assert "memory.search" in cap_names
        assert "memory.state" in cap_names

    def test_manifest_declares_at_least_three_capabilities(self) -> None:
        mod = MemoryModule()
        assert len(mod.manifest.capabilities) >= 3


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_initialize_and_start(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("memory_mod"))
        mod = MemoryModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()

        assert mod.manager is not None

        await mod.stop()
        await mod.shutdown()

    async def test_health_after_start(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("memory_health"))
        mod = MemoryModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()

        health = await mod.health()
        assert health.status.value == "healthy"

        await mod.shutdown()

    async def test_health_before_initialize(self) -> None:
        mod = MemoryModule()
        health = await mod.health()
        assert health.status.value == "unhealthy"

    async def test_full_lifecycle(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("memory_full"))
        mod = MemoryModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()

        # Verify manager works
        mgr = mod.manager
        assert mgr is not None

        # Create a memory through the manager
        from app.memory.memory import Memory
        m = Memory(content="lifecycle test", importance=0.9)
        await mgr.create(m)
        assert await mgr.count() == 1

        health = await mod.health()
        assert health.status.value == "healthy"

        await mod.stop()
        await mod.shutdown()

    async def test_shutdown_cleanup(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("memory_shutdown"))
        mod = MemoryModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()
        await mod.shutdown()

        # Manager should be None after shutdown
        assert mod.manager is None
