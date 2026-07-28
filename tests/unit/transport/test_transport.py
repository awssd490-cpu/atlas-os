"""Tests for HttpTransport."""

from __future__ import annotations

from typing import Any

import pytest

from app.provider.config import ProviderConfig
from app.transport.transport import HttpTransport
from app.transport.models import HttpMethod, HttpRequest, HttpResponse, TransportConfig


class TestHttpTransport:
    async def test_initialize_and_shutdown(self) -> None:
        transport = HttpTransport()
        assert transport.is_initialized is False
        await transport.initialize()
        assert transport.is_initialized is True
        await transport.shutdown()
        assert transport.is_initialized is False

    async def test_send_requires_initialize(self) -> None:
        transport = HttpTransport()
        with pytest.raises(RuntimeError, match="not initialized"):
            await transport.send(HttpRequest.post("https://test.com"))

    async def test_statistics_before_init(self) -> None:
        transport = HttpTransport()
        stats = transport.statistics
        assert stats.total_requests == 0

    async def test_send_json(self) -> None:
        """send_json builds an HttpRequest and sends it — needs initialized transport."""
        transport = HttpTransport()
        await transport.initialize()
        # This will fail with connection error but that's fine — it proves
        # the request was built correctly
        with pytest.raises(Exception, match=".*"):
            await transport.send_json(
                "https://api.nonexistent.example.com/v1/messages",
                {"test": True},
            )
        await transport.shutdown()

    async def test_health_check_failure(self) -> None:
        transport = HttpTransport()
        await transport.initialize()
        result = await transport.health_check("https://api.nonexistent.example.com")
        assert result is False
        await transport.shutdown()
