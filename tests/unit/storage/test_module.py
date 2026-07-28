"""Tests for StorageModule.

Verifies:
- Module registers capabilities: storage.sql, storage.cache, storage.event_store, storage.vector, storage.graph, storage.object
- Module initializes and starts (connect + migrate)
- Health returns healthy after start
- Health returns unhealthy before start
- Shutdown disconnects engine
- Full lifecycle without errors
"""

from __future__ import annotations

import pytest

from app.core.interfaces import KernelContext
from app.storage.module import StorageModule


# ---------------------------------------------------------------------------
# Test context builder
# ---------------------------------------------------------------------------


def _make_context(tmp_path: str) -> KernelContext:
    """Build a minimal KernelContext for testing the storage module."""
    from app.config.service import PydanticConfigService
    from app.config.settings import AtlasSettings
    from app.logging.service import LoguruLoggingService
    from app.events.bus import InProcessEventBus
    from app.telemetry.service import InMemoryTelemetryService
    from app.modules.capabilities import InMemoryCapabilityRegistry

    settings = AtlasSettings(
        storage={"sqlite_path": f"{tmp_path}/test.db", "object_store_path": tmp_path}
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
# Manifest & capabilities
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_name(self) -> None:
        mod = StorageModule()
        assert mod.manifest.name == "storage"
        assert mod.manifest.version == "1.0.0"

    def test_manifest_declares_storage_capabilities(self) -> None:
        mod = StorageModule()
        cap_names = {c.name for c in mod.manifest.capabilities}
        assert "storage.sql" in cap_names
        assert "storage.cache" in cap_names
        assert "storage.event_store" in cap_names
        assert "storage.vector" in cap_names
        assert "storage.graph" in cap_names
        assert "storage.object" in cap_names

    def test_manifest_declares_at_least_six_capabilities(self) -> None:
        mod = StorageModule()
        assert len(mod.manifest.capabilities) >= 6


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_initialize_and_start(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("storage_test"))
        mod = StorageModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        assert mod.engine is not None
        assert mod.cache is not None
        assert mod.vector_store is not None
        assert mod.graph_store is not None
        assert mod.object_store is not None

        await mod.start()
        assert mod.event_store is not None

        await mod.stop()
        await mod.shutdown()

    async def test_health_after_start(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("storage_health"))
        mod = StorageModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()

        health = await mod.health()
        assert health.status.value == "healthy"

        await mod.shutdown()

    async def test_health_before_initialize(self) -> None:
        mod = StorageModule()
        health = await mod.health()
        assert health.status.value == "unhealthy"

    async def test_shutdown_disconnects_engine(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = str(tmp_path_factory.mktemp("storage_shutdown"))
        mod = StorageModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()
        await mod.shutdown()

        assert mod.engine is not None
        healthy = await mod.engine.is_healthy()
        assert healthy is False

    async def test_full_lifecycle(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Full boot -> stop -> shutdown cycle."""
        tmp = str(tmp_path_factory.mktemp("storage_full"))
        mod = StorageModule()
        ctx = _make_context(tmp)

        await mod.initialize(ctx)
        await mod.start()
        health = await mod.health()
        assert health.status.value == "healthy"
        await mod.stop()
        await mod.shutdown()
