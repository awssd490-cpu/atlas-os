"""System introspection endpoints.

Exposes module registry, event bus stats, and configuration for
observability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_kernel
from app.api.schemas import (
    ConfigResponse,
    EventStatsResponse,
    ModuleInfo,
    ModuleListResponse,
)
from app.kernel.kernel import Kernel

router = APIRouter(tags=["system"])


@router.get("/system/modules", response_model=ModuleListResponse)
async def list_modules(kernel: Kernel = Depends(get_kernel)) -> ModuleListResponse:
    """Return all registered modules with their metadata."""
    modules: list[ModuleInfo] = []
    for name, state, module in kernel.get_modules():
        modules.append(
            ModuleInfo(
                name=name,
                version=module.manifest.version,
                status=state.value,
                capabilities=[c.name for c in module.manifest.capabilities],
                dependencies=list(module.manifest.dependencies),
            )
        )
    return ModuleListResponse(modules=modules)


@router.get("/system/events", response_model=EventStatsResponse)
async def event_stats(kernel: Kernel = Depends(get_kernel)) -> EventStatsResponse:
    """Return event bus statistics."""
    stats = kernel.event_bus.stats()
    return EventStatsResponse(
        total_events_emitted=stats.get("total_events_emitted", 0),
        registered_event_types=stats.get("registered_event_types", 0),
        total_handlers=stats.get("total_handlers", 0),
        total_errors=stats.get("total_errors", 0),
        events_by_type=stats.get("events_by_type", {}),
    )


@router.get("/system/config", response_model=ConfigResponse)
async def system_config(kernel: Kernel = Depends(get_kernel)) -> ConfigResponse:
    """Return the current configuration with secrets masked."""
    return ConfigResponse(config=kernel.config.dump(mask_secrets=True))
