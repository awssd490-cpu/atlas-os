"""Core ATLAS interfaces.

Every kernel-level abstraction lives here.  Implementations reside in their
respective subpackages; ``interfaces.py`` imports nothing from those packages
so the contract layer stays dependency-free.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from app.core.manifest import (
        CapabilityDeclaration,
        ModuleHealth,
        ModuleManifest,
    )
    from app.core.events import Event

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


class _Missing:
    """Sentinel for 'no default supplied' in config lookups."""

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


# ---------------------------------------------------------------------------
# Kernel state
# ---------------------------------------------------------------------------


class KernelState(enum.Enum):
    CREATED = "created"
    BOOTING = "booting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Module lifecycle
# ---------------------------------------------------------------------------


class ModuleState(enum.Enum):
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# KernelContext — the only surface modules touch
# ---------------------------------------------------------------------------


class KernelContext(abc.ABC):
    """Context injected into every module on :meth:`Module.initialize`.

    Modules access *all* kernel services through this object rather than
    through the DI container directly.  This prevents the service-locator
    anti-pattern and makes dependencies explicit.
    """

    @property
    @abc.abstractmethod
    def config(self) -> "ConfigService":
        """Typed configuration access."""

    @property
    @abc.abstractmethod
    def logging(self) -> "LoggingService":
        """Structured logging facade."""

    @property
    @abc.abstractmethod
    def logger(self) -> "Logger":
        """A logger pre-bound to this module's name."""

    @property
    @abc.abstractmethod
    def event_bus(self) -> "EventBus":
        """Internal event bus."""

    @property
    @abc.abstractmethod
    def telemetry(self) -> "TelemetryService":
        """Kernel telemetry recorder."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> "CapabilityRegistry":
        """Capability registry for discovering modules by what they do."""

    @abc.abstractmethod
    async def resolve(self, interface: type[T]) -> T:
        """Resolve a service from the DI container.

        This is the *only* way modules should obtain service instances.
        Prefer properties above over raw resolution when possible.
        """


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class Module(abc.ABC):
    """Base class every ATLAS module must implement.

    Lifecycle (called by the kernel in order):

    .. code-block::

        register
            ↓
        initialize(ctx)     ← allocate resources, bind the context
            ↓
        start()             ← begin work, register event handlers
            ↓
        ready()             ← barrier: *all* modules started before this fires
            ↓
        active ──→ pause() → resume() → active  (optional cycles)
            ↓
        stop()              ← cease work
            ↓
        shutdown()          ← release resources

    Only :attr:`manifest` is abstract.  Every other hook has a no-op default
    so simple modules avoid ceremony.
    """

    def __init__(self) -> None:
        self._state = ModuleState.REGISTERED
        self._context: KernelContext | None = None

    # -- Identity -----------------------------------------------------------

    @property
    @abc.abstractmethod
    def manifest(self) -> ModuleManifest:
        """Static declaration returned to the kernel on registration.

        This is the ONLY abstract property.  All other hooks have no-op
        defaults.
        """

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def state(self) -> ModuleState:
        return self._state

    # -- Lifecycle hooks (all optional) -------------------------------------

    async def initialize(self, context: KernelContext) -> None:
        """Allocate resources and store the context.

        Called once, before :meth:`start`.  This is the right place to
        open connections, parse config relevant to the module, and store
        *context* for later use.

        Args:
            context: :class:`KernelContext` for access to kernel services.
        """
        self._context = context

    async def start(self) -> None:
        """Begin active work and register event handlers.

        Called after :meth:`initialize`.  At this point the module may
        start background tasks and subscribe to events on the bus.
        """

    async def ready(self) -> None:
        """Post-start barrier.

        Called *after* every module has been started.  Cross-module
        capability lookups are safe here.  Use this to verify external
        dependencies are reachable, or to initialise coordination state.
        """

    async def pause(self) -> None:
        """Suspend active work gracefully.

        The module should stop processing new events but keep resources
        allocated.  A subsequent :meth:`resume` may be called.
        """

    async def resume(self) -> None:
        """Resume active work after :meth:`pause`."""

    async def stop(self) -> None:
        """Cease active work.

        Called before :meth:`shutdown`.  The module should stop event
        processing but keep resources allocated for a clean shutdown.
        """

    async def shutdown(self) -> None:
        """Release all resources.

        The final lifecycle call.  After this the module is unusable.
        """

    async def health(self) -> "ModuleHealth":
        """Return the module's current health status.

        The default implementation returns ``HEALTHY`` with no details.
        Override to report component-level health (e.g. database
        connectivity, queue depth).
        """
        from app.core.manifest import ModuleHealth

        return ModuleHealth.ok()


# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------


CapabilityName = str  # dotted, e.g. "storage.sql" | "memory.vector_search"


class CapabilityEntry(Protocol):
    """A resolved capability with its provider module name."""

    capability: "CapabilityDeclaration"
    provider: str  # module name


class CapabilityRegistry(abc.ABC):
    """Lets modules discover one another by declared capability.

    Multiple modules may register the same capability name.  ``get()``
    returns the first-registered provider; ``get_all()`` returns all.
    """

    @abc.abstractmethod
    def register(self, capability: "CapabilityDeclaration", provider: str) -> None:
        """Declare that *provider* (a module name) offers *capability*."""

    @abc.abstractmethod
    def get(self, name: CapabilityName) -> "CapabilityEntry | None":
        """Return the first-registered provider of *name*, or ``None``."""

    @abc.abstractmethod
    def get_all(self, name: CapabilityName) -> list["CapabilityEntry"]:
        """Return every provider registered for *name* (empty list if none)."""

    @abc.abstractmethod
    def find_by_provider(self, module_name: str) -> list["CapabilityEntry"]:
        """Return all capabilities registered for a given module."""

    @abc.abstractmethod
    def is_registered(self, name: CapabilityName) -> bool:
        """Return ``True`` when at least one provider for *name* exists."""


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

E = TypeVar("E", bound="Event")


class EventHandler(Protocol[E]):
    """Protocol for an async event-handler callback."""

    async def __call__(self, event: E) -> None: ...


class EventBus(abc.ABC):
    """Decoupled publish/subscribe channel for typed events."""

    @abc.abstractmethod
    async def publish(self, event: Event) -> None:
        """Fire-and-forget publish: handler failures are logged but isolated.

        All handlers run concurrently.  A failing handler never prevents
        other handlers from running.
        """

    @abc.abstractmethod
    def subscribe(
        self,
        event_type: type[Event],
        handler: EventHandler[Any],
    ) -> None:
        """Register *handler* for all events of *event_type* (or subtypes).

        Handlers run in registration order.  Idempotent per
        ``(event_type, handler)`` pair.
        """

    @abc.abstractmethod
    def unsubscribe(self, event_type: type[Event], handler: EventHandler[Any]) -> None:
        """Remove a previously registered handler.

        Raises :class:`ValueError` when not registered.
        """

    @abc.abstractmethod
    async def emit_and_wait(
        self,
        event: Event,
        *,
        timeout: float | None = None,
    ) -> list[BaseException | None]:
        """Publish and await all handlers.

        Unlike :meth:`publish`, this **propagates** exceptions.  Useful
        for command-style events where the caller needs to know handlers
        succeeded.

        Returns a parallel list: ``None`` per successful handler,
        :class:`BaseException` per failed one.
        """

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """Snapshot of event-bus metrics for the telemetry system."""


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


Factory = Callable[["DIContainer"], T | Awaitable[T]]


class DIContainer(abc.ABC):
    """Central registry for service factories and singletons."""

    @abc.abstractmethod
    def register(
        self,
        interface: type[T],
        factory: Factory[T],
        *,
        singleton: bool = True,
    ) -> None:
        """Register *factory* for *interface*.

        Args:
            interface: Abstract or concrete type to register.
            factory: Callable(container) -> instance; may be sync or async.
            singleton: If ``True`` (default), the factory is called once
                and then cached.  Otherwise a new instance per ``resolve``.
        """

    @abc.abstractmethod
    async def resolve(self, interface: type[T]) -> T:
        """Return an instance of *interface*.

        Raises :class:`DependencyResolutionError` if not registered.
        """

    @abc.abstractmethod
    def is_registered(self, interface: type[T]) -> bool:
        """Whether *interface* has a registered factory."""

    @abc.abstractmethod
    async def init_singletons(self) -> None:
        """Pre-resolve all singletons so boot errors surface early."""

    @abc.abstractmethod
    async def dispose(self) -> None:
        """Dispose singleton instances that implement ``close()`` or
        ``shutdown()``."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigService(abc.ABC):
    """Typed, validated configuration provider."""

    @abc.abstractmethod
    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Return the value at a dotted *key* (e.g. ``"database.host"``).

        Raises :class:`ConfigurationError` when *key* is absent and no
        *default* is given.
        """

    @abc.abstractmethod
    def get_section(self, prefix: str) -> dict[str, Any]:
        """Return values whose keys start with *prefix*.

        Returns a flattened dict keyed on the suffix after *prefix*.
        """

    @abc.abstractmethod
    def dump(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        """Return the entire config as a plain dict.

        ``SecretStr`` fields are masked when *mask_secrets* is ``True``.
        """


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@runtime_checkable
class Logger(Protocol):
    """Minimal structured-logger protocol.

    Matches the subset of Loguru used inside ATLAS so the implementation
    is swappable.
    """

    def debug(self, _message: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, _message: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, _message: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, _message: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, _message: str, *args: Any, **kwargs: Any) -> None: ...
    def bind(self, **kwargs: Any) -> "Logger": ...


class LoggingService(abc.ABC):
    """Structured-logging facade that controls Loguru configuration."""

    @abc.abstractmethod
    def get_logger(self, module: str) -> Logger:
        """Return a logger pre-bound to the given *module* name."""

    @abc.abstractmethod
    def configure(self) -> None:
        """(Re)configure sinks from the current configuration."""


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TelemetrySnapshot:
    """Immutable snapshot of kernel telemetry counters.

    Created by :meth:`TelemetryService.snapshot()`.
    """

    def __init__(
        self,
        *,
        uptime_seconds: float = 0.0,
        total_events_emitted: int = 0,
        events_by_type: dict[str, int] | None = None,
        total_errors: int = 0,
        errors_by_category: dict[str, int] | None = None,
        module_load_times: dict[str, float] | None = None,
        module_states: dict[str, str] | None = None,
        startup_duration_ms: float = 0.0,
    ) -> None:
        self.uptime_seconds = uptime_seconds
        self.total_events_emitted = total_events_emitted
        self.events_by_type = (events_by_type or {}).copy()
        self.total_errors = total_errors
        self.errors_by_category = (errors_by_category or {}).copy()
        self.module_load_times = (module_load_times or {}).copy()
        self.module_states = (module_states or {}).copy()
        self.startup_duration_ms = startup_duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_events_emitted": self.total_events_emitted,
            "events_by_type": self.events_by_type,
            "total_errors": self.total_errors,
            "errors_by_category": self.errors_by_category,
            "module_load_times": self.module_load_times,
            "module_states": self.module_states,
            "startup_duration_ms": self.startup_duration_ms,
        }


class TelemetryService(abc.ABC):
    """Lightweight kernel-level metrics recorder.

    Present from Phase 1 so every component records startup / error /
    throughput metrics without waiting for a dedicated Telemetry Engine
    (Phase 8).  The counters are held in memory and exported through the
    health and system-info endpoints.
    """

    @abc.abstractmethod
    def record_event_metrics(self, event_type: str, duration_ms: float, success: bool) -> None:
        """Record a single event-publish cycle."""

    @abc.abstractmethod
    def record_module_lifecycle(
        self,
        module_name: str,
        lifecycle_hook: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        """Record the duration of one module-lifecycle hook call."""

    @abc.abstractmethod
    def record_error(self, category: str, error_type: str, *, count: int = 1) -> None:
        """Increment an error counter."""

    @abc.abstractmethod
    def set_startup_duration(self, duration_ms: float) -> None:
        """Record the full kernel-startup duration."""

    @abc.abstractmethod
    def snapshot(self) -> TelemetrySnapshot:
        """Return a point-in-time copy of all counters."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all counters (useful in tests)."""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthCheck(abc.ABC):
    """Interface for a single health check the kernel runs periodically."""

    @abc.abstractmethod
    async def check(self) -> "HealthResult":
        """Run the check and return a result."""


@runtime_checkable
class HealthResult(Protocol):
    """Minimal health-check result protocol."""

    healthy: bool
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------


class ModuleRegistry(abc.ABC):
    """Tracks all registered modules and their runtime state."""

    @abc.abstractmethod
    def register(self, module: Module) -> None:
        """Register *module*.  Raises on duplicate names."""

    @abc.abstractmethod
    def get(self, name: str) -> Module:
        """Get module by name.  Raises :class:`ModuleNotFoundError_`."""

    @abc.abstractmethod
    def all(self) -> list[tuple[str, ModuleState, Module]]:
        """All registered modules with name, state, and instance."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return the number of registered modules."""

    @abc.abstractmethod
    def boot_order(self) -> list[Module]:
        """Topologically sorted list respecting dependency declarations.

        Performs Kahn's algorithm.  Raises on cycles or missing deps.
        """

    @abc.abstractmethod
    def update_state(self, name: str, state: ModuleState) -> None:
        """Update the runtime state of a registered module."""
