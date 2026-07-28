"""Lifecycle manager — orchestrates the ATLAS module lifecycle.

**Responsibility:** drive each registered module through its lifecycle
hooks (initialize → start → ready → pause/resume → stop → shutdown) in
dependency order, with telemetry recording, event emission, and rollback
on failure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.events import ModuleFailed, ModuleLifecycleEvent
from app.core.interfaces import (
    EventBus,
    KernelContext,
    Module,
    ModuleRegistry,
    ModuleState,
    TelemetryService,
)

# ---------------------------------------------------------------------------
# Lifecycle hook names (used for telemetry and events)
# ---------------------------------------------------------------------------

HOOK_INITIALIZE = "initialize"
HOOK_START = "start"
HOOK_READY = "ready"
HOOK_PAUSE = "pause"
HOOK_RESUME = "resume"
HOOK_STOP = "stop"
HOOK_SHUTDOWN = "shutdown"
HOOK_HEALTH = "health"


class LifecycleManager:
    """Orchestrates the module lifecycle.

    The lifecycle is:

        REGISTERED
            ↓ initialize()
        INITIALIZED
            ↓ start()
        ACTIVE ──→ pause() → resume() ──→ ACTIVE
            ↓ stop()
        STOPPED
            ↓ shutdown()
        SHUTDOWN
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        event_bus: EventBus,
        telemetry: TelemetryService,
        shutdown_timeout: float = 30.0,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._shutdown_timeout = shutdown_timeout

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(self, context: KernelContext) -> None:
        """Call ``initialize(context)`` on every module in boot order.

        Modules that fail are recorded and the kernel will handle rollback.
        """
        order = self._registry.boot_order()
        for module in order:
            await self._run_hook(module, HOOK_INITIALIZE, context=context)
            self._registry.update_state(module.name, ModuleState.INITIALIZED)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Call ``start()`` on every initialized module in boot order."""
        order = self._registry.boot_order()
        for module in order:
            await self._run_hook(module, HOOK_START)
            self._registry.update_state(module.name, ModuleState.ACTIVE)

    # ------------------------------------------------------------------
    # Ready (post-start barrier)
    # ------------------------------------------------------------------

    async def ready(self) -> None:
        """Call ``ready()`` on every active module in boot order.

        This is the cross-module capability-discovery safe point.
        """
        order = self._registry.boot_order()
        for module in order:
            await self._run_hook(module, HOOK_READY)

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    async def pause(self, module_name: str | None = None) -> None:
        """Pause one or all modules.

        Args:
            module_name: If provided, pause only this module.  Otherwise
                pause all active modules.
        """
        if module_name:
            module = self._registry.get(module_name)
            await self._run_hook(module, HOOK_PAUSE)
            self._registry.update_state(module.name, ModuleState.PAUSED)
        else:
            for entry in self._registry.all():
                name, state, module = entry
                if state == ModuleState.ACTIVE:
                    await self._run_hook(module, HOOK_PAUSE)
                    self._registry.update_state(name, ModuleState.PAUSED)

    async def resume(self, module_name: str | None = None) -> None:
        """Resume one or all paused modules."""
        if module_name:
            module = self._registry.get(module_name)
            await self._run_hook(module, HOOK_RESUME)
            self._registry.update_state(module.name, ModuleState.ACTIVE)
        else:
            for entry in self._registry.all():
                name, state, module = entry
                if state == ModuleState.PAUSED:
                    await self._run_hook(module, HOOK_RESUME)
                    self._registry.update_state(name, ModuleState.ACTIVE)

    # ------------------------------------------------------------------
    # Stop (all modules in reverse order)
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Call ``stop()`` on every module in **reverse** boot order."""
        order = list(reversed(self._registry.boot_order()))
        for module in order:
            if self._registry.all() and any(
                m[1] in (ModuleState.ACTIVE, ModuleState.PAUSED) and m[2] is module
                for m in self._registry.all()
            ):
                await self._run_hook(module, HOOK_STOP)
                self._registry.update_state(module.name, ModuleState.STOPPED)

    # ------------------------------------------------------------------
    # Shutdown (all modules in reverse order, with timeout)
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Shut down all modules in **reverse** boot order.

        Each module's ``shutdown()`` has ``_shutdown_timeout`` seconds
        before it is cancelled.
        """
        order = list(reversed(self._registry.boot_order()))
        for module in order:
            try:
                await asyncio.wait_for(
                    self._run_hook(module, HOOK_SHUTDOWN),
                    timeout=self._shutdown_timeout,
                )
            except asyncio.TimeoutError:
                self._telemetry.record_error(
                    "lifecycle", f"shutdown_timeout:{module.name}"
                )
            self._registry.update_state(module.name, ModuleState.SHUTDOWN)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def collect_health(self) -> dict[str, Any]:
        """Collect health from all modules.

        Returns a dict mapping module names to their ``ModuleHealth``
        dict, plus a combined status.
        """
        health_results: dict[str, Any] = {}
        overall = "healthy"

        for entry in self._registry.all():
            name, state, module = entry
            try:
                result = await self._run_hook(module, HOOK_HEALTH)
                # The hook returns None for no-op default, but health()
                # has a default that returns ModuleHealth.ok()
                health_results[name] = {
                    "state": state.value,
                    "health": result.to_dict() if hasattr(result, "to_dict") else {"status": str(result)},
                }
                if hasattr(result, "status") and result.status == "unhealthy":
                    overall = "unhealthy"
                elif hasattr(result, "status") and result.status == "degraded" and overall != "unhealthy":
                    overall = "degraded"
            except Exception as exc:
                health_results[name] = {
                    "state": state.value,
                    "health": {"status": "unhealthy", "error": str(exc)},
                }
                overall = "unhealthy"

        return {"status": overall, "modules": health_results}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_hook(
        self,
        module: Module,
        hook: str,
        **kwargs: Any,
    ) -> Any:
        """Run a single lifecycle hook with telemetry and event emission.

        Returns the result of the hook, or raises on failure (depending
        on hook semantics).
        """
        start = time.monotonic()
        try:
            if hook == HOOK_INITIALIZE:
                result = await module.initialize(**kwargs)
            elif hook == HOOK_START:
                result = await module.start()
            elif hook == HOOK_READY:
                result = await module.ready()
            elif hook == HOOK_PAUSE:
                result = await module.pause()
            elif hook == HOOK_RESUME:
                result = await module.resume()
            elif hook == HOOK_STOP:
                result = await module.stop()
            elif hook == HOOK_SHUTDOWN:
                result = await module.shutdown()
            elif hook == HOOK_HEALTH:
                result = await module.health()
            else:
                raise ValueError(f"Unknown lifecycle hook: {hook}")

            duration_ms = (time.monotonic() - start) * 1000
            self._telemetry.record_module_lifecycle(module.name, hook, duration_ms, True)

            await self._event_bus.publish(
                ModuleLifecycleEvent(
                    module_name=module.name,
                    lifecycle_hook=hook,
                    success=True,
                    duration_ms=duration_ms,
                )
            )
            return result

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            self._telemetry.record_module_lifecycle(module.name, hook, duration_ms, False)
            self._telemetry.record_error("lifecycle", f"{module.name}.{hook}")

            await self._event_bus.publish(
                ModuleFailed(
                    module_name=module.name,
                    lifecycle_hook=hook,
                    error_message=str(exc),
                )
            )
            await self._event_bus.publish(
                ModuleLifecycleEvent(
                    module_name=module.name,
                    lifecycle_hook=hook,
                    success=False,
                    duration_ms=duration_ms,
                )
            )

            # For start/initialize hooks, propagate to trigger rollback
            if hook in (HOOK_INITIALIZE, HOOK_START):
                raise

            # For non-essential hooks (health, pause, resume), swallow
            return None
