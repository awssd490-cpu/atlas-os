"""StorageModule — the kernel module that provides storage services.

Integrates the full storage stack with the ATLAS kernel:

- **Capability registration**: declares every storage backend as a
  capability (``storage.sql``, ``storage.cache``, ``storage.event_store``,
  ``storage.vector``, ``storage.graph``, ``storage.object``) so consumers
  discover storage by capability identifier, never by implementation.
- **Health integration**: reports engine health through the standard
  ``Module.health()`` hook, which the kernel aggregates into the
  ``/health`` endpoint.
- **Telemetry integration**: records storage lifecycle timings and errors
  through the kernel ``TelemetryService``.
- **Service provision**: creates all storage services and exposes them
  as properties. The kernel (or application) registers them in the DI
  container after the module initializes.
"""

from __future__ import annotations

import time

from app.core.interfaces import KernelContext, Module
from app.core.manifest import (
    CapabilityDeclaration,
    ModuleHealth,
    ModuleManifest,
)
from app.storage.cache.memory import MemoryCache
from app.storage.engine.sqlite import SQLiteStorageEngine
from app.storage.event_store.service import SqliteEventStore
from app.storage.graph.memory import InMemoryGraphStore
from app.storage.interfaces import (
    CacheService,
    EventStore,
    GraphStore,
    ObjectStore,
    StorageEngine,
    VectorStore,
)
from app.storage.migration.manager import SqliteMigrationManager
from app.storage.migration.sqlite import V001_InitialSchema
from app.storage.object_store.local import LocalFileObjectStore
from app.storage.vector.memory import InMemoryVectorStore


class StorageModule(Module):
    """Kernel module providing the Phase 2 storage foundation.

    Lifecycle:
        - ``initialize(ctx)`` — read config, create services
        - ``start()`` — connect engine, run migrations, create event store
        - ``health()`` — report engine + cache health
        - ``stop()`` — flush pending writes
        - ``shutdown()`` — disconnect engine
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest = ModuleManifest(
            name="storage",
            version="1.0.0",
            description="Persistence & memory foundation: SQL, cache, events, vectors, graph, objects",
            capabilities=[
                CapabilityDeclaration(
                    name="storage.sql",
                    version="1.0",
                    description="SQL persistence (SQLite backend)",
                ),
                CapabilityDeclaration(
                    name="storage.cache",
                    version="1.0",
                    description="Best-effort TTL cache (in-memory backend)",
                ),
                CapabilityDeclaration(
                    name="storage.event_store",
                    version="1.0",
                    description="Append-only event persistence and replay",
                ),
                CapabilityDeclaration(
                    name="storage.vector",
                    version="1.0",
                    description="Vector similarity search (in-memory backend)",
                ),
                CapabilityDeclaration(
                    name="storage.graph",
                    version="1.0",
                    description="Knowledge graph storage (in-memory backend)",
                ),
                CapabilityDeclaration(
                    name="storage.object",
                    version="1.0",
                    description="Binary object storage (local filesystem backend)",
                ),
            ],
        )
        self._engine: SQLiteStorageEngine | None = None
        self._cache: MemoryCache | None = None
        self._event_store: SqliteEventStore | None = None
        self._vector_store: InMemoryVectorStore | None = None
        self._graph_store: InMemoryGraphStore | None = None
        self._object_store: LocalFileObjectStore | None = None
        self._migrations_applied: list[str] = []

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest

    # ------------------------------------------------------------------
    # Service properties (exposed for DI registration by kernel/app)
    # ------------------------------------------------------------------

    @property
    def engine(self) -> SQLiteStorageEngine | None:
        return self._engine

    @property
    def cache(self) -> MemoryCache | None:
        return self._cache

    @property
    def event_store(self) -> SqliteEventStore | None:
        return self._event_store

    @property
    def vector_store(self) -> InMemoryVectorStore | None:
        return self._vector_store

    @property
    def graph_store(self) -> InMemoryGraphStore | None:
        return self._graph_store

    @property
    def object_store(self) -> LocalFileObjectStore | None:
        return self._object_store

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, context: KernelContext) -> None:
        """Create storage services from config."""
        await super().initialize(context)

        sqlite_path = context.config.get("storage.sqlite_path", default="data/atlas.db")
        cache_ttl = context.config.get("storage.cache_ttl_default", default=300)
        cache_max = context.config.get("storage.cache_max_size", default=10_000)
        object_path = context.config.get("storage.object_store_path", default="data/objects")

        if sqlite_path != ":memory:":
            from pathlib import Path

            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        self._engine = SQLiteStorageEngine(path=sqlite_path)
        self._cache = MemoryCache(default_ttl=float(cache_ttl), max_size=int(cache_max))
        self._vector_store = InMemoryVectorStore()
        self._graph_store = InMemoryGraphStore()
        self._object_store = LocalFileObjectStore(base_path=object_path)

        context.logger.info(
            "Storage module initialized | sqlite={path}",
            path=sqlite_path,
        )

    async def start(self) -> None:
        """Connect the engine, run migrations, and create the event store."""
        assert self._context is not None, "initialize() must run before start()"
        assert self._engine is not None

        start_time = time.monotonic()

        await self._engine.connect()

        conn = await self._engine.connection()
        manager = SqliteMigrationManager()
        self._migrations_applied = await manager.apply_all(conn, [V001_InitialSchema()])

        self._event_store = SqliteEventStore(connection=conn)

        duration_ms = (time.monotonic() - start_time) * 1000
        self._context.telemetry.record_module_lifecycle(
            "storage", "storage_connect_and_migrate", duration_ms, True
        )

        self._context.logger.info(
            "Storage module started | migrations={migrations} | duration_ms={ms}",
            migrations=self._migrations_applied,
            ms=round(duration_ms, 2),
        )

    async def health(self) -> ModuleHealth:
        """Report storage health: engine connectivity + cache stats."""
        if self._engine is None:
            return ModuleHealth.unhealthy(reason="engine not initialized")

        engine_healthy = await self._engine.is_healthy()
        cache_stats = self._cache.stats() if self._cache else {}
        event_count = 0
        if self._event_store is not None:
            try:
                event_count = await self._event_store.count()
            except Exception:
                pass

        if not engine_healthy:
            return ModuleHealth.unhealthy(
                engine="unreachable",
                cache=cache_stats,
            )

        return ModuleHealth.ok(
            engine="connected",
            sqlite_path=self._engine.path,
            migrations=self._migrations_applied,
            cache=cache_stats,
            events_stored=event_count,
        )

    async def stop(self) -> None:
        """Cease active work (no background tasks in Phase 2)."""

    async def shutdown(self) -> None:
        """Disconnect the engine and clear caches."""
        if self._cache is not None:
            await self._cache.clear()
        if self._engine is not None:
            await self._engine.disconnect()
        if self._context is not None:
            self._context.logger.info("Storage module shut down")