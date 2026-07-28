"""MemoryModule — kernel module providing the ATLAS memory system.

Integrates with Phase 1:
- Module lifecycle: initialize, start, health, shutdown
- Capability registry: declares memory.store, memory.search, memory.state
- Telemetry: records memory operations
- Event bus: publishes memory lifecycle events

Integrates with Phase 2:
- SQLiteConnection: persists memories via MemoryRepository
- CacheService: caches frequent retrievals
- Migration system: runs V002 schema
"""

from __future__ import annotations

import time
from typing import Any

from app.core.interfaces import KernelContext, Module
from app.core.manifest import CapabilityDeclaration, ModuleHealth, ModuleManifest
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.memory import MemoryState
from app.memory.migrations import V002_MemorySchema
from app.memory.policies import DEFAULT_TYPE_POLICIES, RetentionPolicy
from app.storage.cache.memory import MemoryCache
from app.storage.connection.sqlite import SQLiteConnection
from app.storage.interfaces import CacheService
from app.storage.migration.manager import SqliteMigrationManager


class MemoryModule(Module):
    """Kernel module providing the ATLAS memory system.

    Lifecycle:
        - ``initialize(ctx)`` — create repository and manager from config
        - ``start()`` — run V002 migration, start manager
        - ``health()`` — report memory counts, cache stats
        - ``shutdown()`` — clean up
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest = ModuleManifest(
            name="memory",
            version="1.0.0",
            description="Knowledge & Memory Engine: organize, retrieve, prioritize, evolve, forget",
            dependencies=["storage"],
            capabilities=[
                CapabilityDeclaration(
                    name="memory.store",
                    version="1.0",
                    description="Memory CRUD and lifecycle management",
                ),
                CapabilityDeclaration(
                    name="memory.search",
                    version="1.0",
                    description="Memory search, filtering, importance ranking",
                ),
                CapabilityDeclaration(
                    name="memory.state",
                    version="1.0",
                    description="Memory state transitions (archive, forget, delete)",
                ),
            ],
        )
        self._repository: MemoryRepository | None = None
        self._manager: MemoryManager | None = None
        self._connection: SQLiteConnection | None = None
        self._cache: CacheService | None = None

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest

    @property
    def manager(self) -> MemoryManager | None:
        return self._manager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, context: KernelContext) -> None:
        """Create the memory repository and manager from config."""
        await super().initialize(context)
        context.logger.info("Memory module initialized")

    async def start(self) -> None:
        """Run V002 migration and create the manager."""
        assert self._context is not None, "initialize() must run before start()"

        start_time = time.monotonic()

        # Open a dedicated SQLite connection for memory
        sqlite_path = self._context.config.get(
            "storage.sqlite_path", default="data/atlas.db"
        )
        self._connection = SQLiteConnection(sqlite_path)
        await self._connection.execute("SELECT 1")  # ensure open

        # Run V002 migration
        migration_manager = SqliteMigrationManager()
        await migration_manager.apply_all(self._connection, [V002_MemorySchema()])

        # Create repository
        self._repository = MemoryRepository(connection=self._connection)

        # Create cache (dedicated in-memory cache for memories)
        cache_ttl = self._context.config.get(
            "storage.cache_ttl_default", default=300
        )
        cache_max = self._context.config.get(
            "storage.cache_max_size", default=10_000
        )
        self._cache = MemoryCache(
            default_ttl=float(cache_ttl), max_size=int(cache_max)
        )

        # Build retention policy from config
        retention = RetentionPolicy(
            archive_threshold=self._context.config.get(
                "memory.archive_threshold", default=0.2
            ),
            importance_decay_rate=self._context.config.get(
                "memory.importance_decay_rate", default=0.1
            ),
            grace_period_seconds=self._context.config.get(
                "memory.grace_period_seconds", default=604800.0
            ),
            enable_auto_archive=self._context.config.get(
                "memory.enable_auto_archive", default=True
            ),
        )

        self._manager = MemoryManager(
            repository=self._repository,
            cache=self._cache,
            event_bus=self._context.event_bus,
            telemetry=self._context.telemetry,
            logger=self._context.logger,
            policies=DEFAULT_TYPE_POLICIES,
            retention=retention,
        )

        duration_ms = (time.monotonic() - start_time) * 1000
        self._context.telemetry.record_module_lifecycle(
            "memory", "memory_start_and_migrate", duration_ms, True
        )

        self._context.logger.info(
            "Memory module started | duration_ms={ms}",
            ms=round(duration_ms, 2),
        )

    async def health(self) -> ModuleHealth:
        """Report memory system health: counts, GC candidates, cache stats."""
        if self._manager is None:
            return ModuleHealth.unhealthy(reason="memory manager not initialized")

        try:
            active_count = await self._manager.count(MemoryState.ACTIVE)
            archived_count = await self._manager.count(MemoryState.ARCHIVED)
            gc_candidates = await self._manager.count_candidates()
            cache_stats = self._cache.stats() if self._cache else {}

            details: dict[str, Any] = {
                "memories_active": active_count,
                "memories_archived": archived_count,
                "gc_candidates": gc_candidates,
                "type_policies_loaded": len(DEFAULT_TYPE_POLICIES),
                "cache": cache_stats,
            }

            return ModuleHealth.ok(**details)
        except Exception as exc:
            return ModuleHealth.unhealthy(reason=str(exc))

    async def stop(self) -> None:
        """Cease active work."""
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._cache is not None:
            await self._cache.clear()
        self._manager = None
        self._repository = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        if self._context is not None:
            self._context.logger.info("Memory module shut down")
