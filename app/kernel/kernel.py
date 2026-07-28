"""AtlasKernel — the central composition root.

**Responsibility:** compose all core services, register modules, drive
the lifecycle state machine, and expose diagnostics.

The kernel enforces a state machine:

    CREATED ─→ BOOTING ─→ RUNNING ─→ SHUTTING_DOWN ─→ STOPPED
       │                     │                              │
       └───→ FAILED ←────────┘                              │
                             └──────────────────────────────┘
"""

from __future__ import annotations

import time
from typing import Any

from app.config.service import PydanticConfigService
from app.config.settings import AtlasSettings
from app.core.events import (
    KernelBooted,
    KernelBooting,
    KernelShuttingDown,
    KernelStopped,
    ModuleRegistered,
)
from app.core.interfaces import (
    ConfigService,
    EventBus,
    KernelContext,
    KernelState,
    Logger,
    LoggingService,
    Module,
    ModuleRegistry,
    ModuleState,
    TelemetryService,
)
from app.di.container import Container
from app.events.bus import InProcessEventBus
from app.kernel.context import KernelContextImpl
from app.lifecycle.manager import LifecycleManager
from app.logging.service import LoguruLoggingService
from app.modules.capabilities import InMemoryCapabilityRegistry
from app.modules.registry import InMemoryModuleRegistry
from app.telemetry.service import InMemoryTelemetryService


class Kernel:
    """The ATLAS microkernel.

    Usage::

        kernel = Kernel()
        kernel.register(MyModule())
        await kernel.boot()
        # ... server runs ...
        await kernel.shutdown()
    """

    def __init__(self, settings: AtlasSettings | None = None) -> None:
        self._state = KernelState.CREATED
        self._boot_start: float = 0.0
        self._boot_end: float = 0.0
        self._start_time: float = 0.0  # time.monotonic() at boot

        # Create core services
        self._config_service: ConfigService = PydanticConfigService(settings=settings)
        self._logging_service: LoggingService = LoguruLoggingService()
        self._event_bus: EventBus = InProcessEventBus()
        self._telemetry: TelemetryService = InMemoryTelemetryService()
        self._container: Container = Container()
        self._capability_registry = InMemoryCapabilityRegistry()
        self._module_registry: ModuleRegistry = InMemoryModuleRegistry()
        self._lifecycle_manager = LifecycleManager(
            registry=self._module_registry,
            event_bus=self._event_bus,
            telemetry=self._telemetry,
        )

        # Register core services in the DI container
        self._register_core_services()

        # No context yet — built during boot()
        self._context: KernelContext | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> KernelState:
        return self._state

    @property
    def config(self) -> ConfigService:
        return self._config_service

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def container(self) -> Container:
        return self._container

    @property
    def logging(self) -> LoggingService:
        return self._logging_service

    @property
    def telemetry(self) -> TelemetryService:
        return self._telemetry

    # ------------------------------------------------------------------
    # Module lifecycle
    # ------------------------------------------------------------------

    def register(self, module: Module) -> None:
        """Register a module.

        Raises :class:`ValueError` on duplicate name.
        """
        self._assert_state(KernelState.CREATED)
        self._module_registry.register(module)

        # Subscribe to the event bus if the module has handlers in its manifest
        # (Modules subscribe manually in start() — this is for event-forwarding
        # integration later)
        self._emit_sync(ModuleRegistered(module_name=module.name, module_version=module.manifest.version))

    def get_module(self, name: str) -> Module:
        """Return registered module by name.

        Raises :class:`ModuleNotFoundError_`.
        """
        return self._module_registry.get(name)

    def get_modules(self) -> list[tuple[str, ModuleState, Module]]:
        return self._module_registry.all()

    def module_count(self) -> int:
        return self._module_registry.count()

    # ------------------------------------------------------------------
    # Boot / Shutdown
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Boot the kernel and all registered modules.

        State flow: CREATED → BOOTING → RUNNING (or FAILED).
        """
        self._assert_state(KernelState.CREATED)
        self._state = KernelState.BOOTING
        self._boot_start = time.monotonic()
        self._start_time = time.monotonic()

        await self._emit_async(KernelBooting(module_count=self.module_count()))

        # Configure logging first so we can log during boot
        self._logging_service.configure()
        logger = self._logging_service.get_logger("kernel")
        logger.info(
            "Kernel boot starting | modules={module_count}",
            module_count=self.module_count(),
        )

        try:
            # Register capability declarations from module manifests
            for entry in self._module_registry.all():
                name, _state, module = entry
                for cap in module.manifest.capabilities:
                    self._capability_registry.register(cap, name)

            # Build the KernelContext
            self._context = KernelContextImpl(
                config=self._config_service,
                logging_service=self._logging_service,
                logger=self._logging_service.get_logger("system"),
                event_bus=self._event_bus,
                telemetry=self._telemetry,
                capabilities=self._capability_registry,
                container=self._container,
            )

            # Initialize all modules (context injection)
            await self._lifecycle_manager.initialize(self._context)

            # Start all modules
            await self._lifecycle_manager.start()

            # Ready barrier (post-start cross-module capability discovery)
            await self._lifecycle_manager.ready()

            self._state = KernelState.RUNNING
            self._boot_end = time.monotonic()
            duration_ms = (self._boot_end - self._boot_start) * 1000
            self._telemetry.set_startup_duration(duration_ms)

            await self._emit_async(
                KernelBooted(duration_ms=duration_ms, module_count=self.module_count())
            )
            logger.info(
                "Kernel boot complete | duration_ms={duration_ms} | modules={count}",
                duration_ms=duration_ms,
                count=self.module_count(),
            )

        except Exception:
            was_failure = True
            self._state = KernelState.FAILED
            logger.exception("Kernel boot failed — shutting down initialized modules")
            # Attempt to shut down anything that was initialized
            await self._shutdown_internal(was_failure=was_failure)
            raise

    async def shutdown(self) -> None:
        """Shut down the kernel gracefully.

        Safe to call regardless of current state (idempotent for STOPPED).
        A call to shutdown from ``FAILED`` acknowledges the failure and
        transitions to ``STOPPED``.
        """
        if self._state == KernelState.STOPPED:
            return
        # From FAILED, an explicit shutdown acknowledges the failure
        await self._shutdown_internal(was_failure=False)

    async def _shutdown_internal(self, was_failure: bool = False) -> None:
        """Internal shutdown orchestration.

        Args:
            was_failure: When ``True`` the shutdown is happening after a
                boot failure; the final state is NOT changed to STOPPED
                (it stays as FAILED).
        """
        if not was_failure:
            self._state = KernelState.SHUTTING_DOWN
        await self._emit_async(KernelShuttingDown())

        logger = self._logging_service.get_logger("kernel")
        logger.info("Kernel shutting down")

        try:
            # Stop modules (cease work, but keep resources)
            await self._lifecycle_manager.stop()

            # Shutdown modules (release resources)
            await self._lifecycle_manager.shutdown()

            # Dispose DI container singletons
            await self._container.dispose()

        except Exception:
            logger.exception("Error during kernel shutdown")

        if not was_failure:
            self._state = KernelState.STOPPED
        await self._emit_async(KernelStopped())
        logger.info("Kernel shutdown complete")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Aggregate health of the kernel and all modules."""
        base: dict[str, Any] = {
            "status": "healthy" if self._state == KernelState.RUNNING else "unhealthy",
            "kernel_state": self._state.value,
            "uptime_seconds": self.uptime(),
        }

        if self._state == KernelState.RUNNING:
            module_health = await self._lifecycle_manager.collect_health()
            base["modules"] = module_health["modules"]
            if module_health["status"] != "healthy":
                base["status"] = module_health["status"]

        telemetry_snapshot = self._telemetry.snapshot()
        telemetry_snapshot.uptime_seconds = self.uptime()

        base["telemetry"] = telemetry_snapshot.to_dict()
        return base

    def uptime(self) -> float:
        """Seconds since kernel booted (0.0 before boot)."""
        if self._start_time == 0.0:
            return 0.0
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_core_services(self) -> None:
        """Register kernel services in the DI container so modules can
        resolve them if needed."""
        self._container.register(ConfigService, lambda c: self._config_service)
        self._container.register(EventBus, lambda c: self._event_bus)
        self._container.register(LoggingService, lambda c: self._logging_service)
        self._container.register(TelemetryService, lambda c: self._telemetry)

    def _assert_state(self, required: KernelState) -> None:
        """Assert the kernel is in *required* state."""
        if self._state != required:
            from app.core.errors import LifecycleError

            raise LifecycleError(
                f"Kernel state '{self._state.value}' is not '{required.value}'",
                details={"current": self._state.value, "required": required.value},
            )

    def _emit_sync(self, event: Any) -> None:
        """Fire an event without awaiting.

        Used during registration where we're in sync context.
        The event bus handles publish() in the background when awaited.
        The caller should ensure this is eventually flushed.
        """
        # Sync emission in registration phase — captured and sent later
        # by boot().  We store it temporarily.
        self._pending_events = getattr(self, "_pending_events", [])
        self._pending_events.append(event)

    async def _emit_async(self, event: Any) -> None:
        """Publish a kernel lifecycle event to the bus."""
        # Flush any pending events first
        if hasattr(self, "_pending_events") and self._pending_events:
            for ev in self._pending_events:
                await self._event_bus.publish(ev)
            self._pending_events.clear()

        await self._event_bus.publish(event)
