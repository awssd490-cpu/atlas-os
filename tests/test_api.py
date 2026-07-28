"""Integration tests for the ATLAS HTTP API.

Tests the full HTTP stack against a booted kernel.  Validates:
- Health endpoint returns healthy status
- Ready endpoint returns ready
- Live endpoint returns alive
- System endpoints return module list, event stats, config
- 404s for unknown routes
- Response schemas match expected contracts
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    async def test_health_returns_healthy(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == "0.1.0"
        assert body["uptime_seconds"] >= 0.0
        assert body["kernel_state"] == "running"

    async def test_health_includes_telemetry(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "telemetry" in body
        assert "uptime_seconds" in body["telemetry"]

    async def test_health_includes_modules(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "modules" in body
        assert "mod_test" in body["modules"]


class TestReadyEndpoint:
    async def test_ready_when_booted(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["modules"]["total"] >= 1

    async def test_ready_includes_module_states(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert "states" in body["modules"]
        assert body["modules"]["states"]["mod_test"] == "active"


class TestLiveEndpoint:
    async def test_live_when_running(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health/live")
        assert response.status_code == 200
        body = response.json()
        assert body["alive"] is True
        assert body["kernel_state"] == "running"


class TestSystemEndpoints:
    async def test_list_modules(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/system/modules")
        assert response.status_code == 200
        body = response.json()
        assert len(body["modules"]) >= 1
        mod = body["modules"][0]
        assert mod["name"] == "mod_test"
        assert mod["version"] == "1.0.0"
        assert mod["status"] == "active"

    async def test_event_stats(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/system/events")
        assert response.status_code == 200
        body = response.json()
        assert "total_events_emitted" in body
        assert "registered_event_types" in body
        assert "events_by_type" in body

    async def test_system_config(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/system/config")
        assert response.status_code == 200
        body = response.json()
        assert "config" in body
        config_body = body["config"]
        assert config_body["app"]["name"] == "atlas-test"
        assert config_body["app"]["environment"] == "testing"

    async def test_config_masks_secrets(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/system/config")
        assert response.status_code == 200
        body = response.json()
        database = body["config"]["database"]
        assert database["password"] == "*****"


class TestNotFound:
    async def test_unknown_route_returns_404(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/nonexistent")
        assert response.status_code == 404
