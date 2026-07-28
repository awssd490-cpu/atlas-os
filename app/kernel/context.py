"""KernelContext implementation — the sole surface modules interact with.

**Responsibility:** provide scoped access to kernel services (config,
logging, event bus, telemetry, capability registry, DI resolution)
without exposing the DI container directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.interfaces import (
    CapabilityRegistry,
    ConfigService,
    EventBus,
    KernelContext,
    Logger,
    LoggingService,
    TelemetryService,
)

if TYPE_CHECKING:
    from app.di.container import Container


class KernelContextImpl(KernelContext):
    """Concrete KernelContext injected into every module on ``initialize()``.

    Modules access all kernel services through this object.  The reference
    is stored on the module instance itself so it's available for the
    entire module lifetime.
    """

    def __init__(
        self,
        config: ConfigService,
        logging_service: LoggingService,
        logger: Logger,
        event_bus: EventBus,
        telemetry: TelemetryService,
        capabilities: CapabilityRegistry,
        container: "Container",
    ) -> None:
        self._config = config
        self._logging_service = logging_service
        self._logger = logger
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._capabilities = capabilities
        self._container = container

    @property
    def config(self) -> ConfigService:
        return self._config

    @property
    def logging(self) -> LoggingService:
        return self._logging_service

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def telemetry(self) -> TelemetryService:
        return self._telemetry

    @property
    def capabilities(self) -> CapabilityRegistry:
        return self._capabilities

    async def resolve(self, interface: type) -> Any:
        return await self._container.resolve(interface)
