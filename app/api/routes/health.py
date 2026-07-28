"""Health and readiness endpoints.

Route handlers are thin — they read from the kernel and delegate to
response schemas.  No business logic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.api.deps import get_kernel
from app.api.schemas import HealthResponse, LiveResponse, ReadyResponse
from app.kernel.kernel import Kernel

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(kernel: Kernel = Depends(get_kernel)) -> HealthResponse:
    """Return aggregated health of the kernel and all modules."""
    result = await kernel.health()
    return HealthResponse(
        status=result.get("status", "unhealthy"),
        uptime_seconds=kernel.uptime(),
        timestamp=datetime.now(timezone.utc),
        kernel_state=kernel.state.value,
        modules=result.get("modules"),
        telemetry=result.get("telemetry"),
    )


@router.get("/health/ready", response_model=ReadyResponse)
async def ready(kernel: Kernel = Depends(get_kernel)) -> ReadyResponse:
    """Return readiness — is the kernel ready to accept traffic?"""
    modules_info: dict[str, str] = {}
    for name, state, _mod in kernel.get_modules():
        modules_info[name] = state.value

    return ReadyResponse(
        ready=kernel.state.value == "running",
        modules={
            "total": kernel.module_count(),
            "states": modules_info,
        },
    )


@router.get("/health/live", response_model=LiveResponse)
async def live(kernel: Kernel = Depends(get_kernel)) -> LiveResponse:
    """Return liveness — is the kernel process alive?"""
    return LiveResponse(
        alive=kernel.state.value not in ("stopped", "failed"),
        kernel_state=kernel.state.value,
    )
