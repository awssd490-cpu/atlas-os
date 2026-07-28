"""API response schemas for the ATLAS HTTP interface.

These are plain Pydantic models — no business logic, no domain imports.
Routes only transform kernel responses into these shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status (healthy/degraded/unhealthy)")
    version: str = Field(default="0.1.0", description="ATLAS version")
    uptime_seconds: float = Field(default=0.0, description="Seconds since kernel boot")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    kernel_state: str | None = None
    modules: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None


class ReadyResponse(BaseModel):
    ready: bool
    modules: dict[str, Any]


class LiveResponse(BaseModel):
    alive: bool
    kernel_state: str


class ModuleInfo(BaseModel):
    name: str
    version: str
    status: str
    capabilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class ModuleListResponse(BaseModel):
    modules: list[ModuleInfo]


class EventStatsResponse(BaseModel):
    total_events_emitted: int
    registered_event_types: int
    total_handlers: int
    total_errors: int
    events_by_type: dict[str, int] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    config: dict[str, Any]


class ErrorResponse(BaseModel):
    error: dict[str, Any]
