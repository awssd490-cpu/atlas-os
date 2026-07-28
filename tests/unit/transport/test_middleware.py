"""Tests for transport middleware."""

from __future__ import annotations

from typing import Any

import pytest

from app.provider.config import ProviderConfig
from app.transport.middleware import (
    AuthMiddleware,
    CoreSender,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RetryMiddleware,
)
from app.transport.models import HttpHeaders, HttpRequest, HttpResponse


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _MockClient:
    """Mock httpx client that returns a canned response."""

    def __init__(self, status: int = 200, text: str = "ok", fail: bool = False) -> None:
        self._status = status
        self._text = text
        self._fail = fail
        self.last_url = ""
        self.last_headers: dict[str, str] = {}
        self.last_json: Any = None

    async def post(self, url: str, **kwargs: Any) -> Any:
        self.last_url = url
        self.last_headers = kwargs.get("headers", {})
        self.last_json = kwargs.get("json")
        if self._fail:
            raise ConnectionError("mock failure")
        return _MockResponse(self._status, self._text)

    async def get(self, url: str, **kwargs: Any) -> Any:
        self.last_url = url
        self.last_headers = kwargs.get("headers", {})
        if self._fail:
            raise ConnectionError("mock failure")
        return _MockResponse(self._status, self._text)

    async def put(self, url: str, **kwargs: Any) -> Any:
        return await self.post(url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> Any:
        return await self.post(url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> Any:
        return await self.get(url, **kwargs)

    async def aclose(self) -> None:
        pass


class _MockResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status_code = status
        self.content = text.encode()
        self.text = text
        self.headers = {"content-type": "text/plain"}
        self.elapsed = type("_", (), {"total_seconds": lambda self: 0.1})()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=type("_", (), {"url": ""})(),
                response=self,
            )


# ---------------------------------------------------------------------------
# CoreSender
# ---------------------------------------------------------------------------


class TestCoreSender:
    async def test_sends_post(self) -> None:
        client = _MockClient()
        sender = CoreSender(client)
        chain = MiddlewareChain()
        chain.set_core(sender)
        response = await chain.send(HttpRequest.post("https://api.test"))
        assert response.status_code == 200
        assert response.text == "ok"


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------


class TestRetryMiddleware:
    async def test_passes_through_on_success(self) -> None:
        client = _MockClient()
        chain = MiddlewareChain()
        chain.add(RetryMiddleware(max_retries=2))
        chain.set_core(CoreSender(client))
        response = await chain.send(HttpRequest.post("https://api.test"))
        assert response.status_code == 200

    async def test_retries_on_failure(self) -> None:
        client = _MockClient(fail=True)
        chain = MiddlewareChain()
        chain.add(RetryMiddleware(max_retries=1, base_delay=0.01))
        chain.set_core(CoreSender(client))
        with pytest.raises(ConnectionError):
            await chain.send(HttpRequest.post("https://api.test"))


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    async def test_adds_auth_header(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test-key"})
        client = _MockClient()
        chain = MiddlewareChain()
        chain.add(AuthMiddleware(config))
        chain.set_core(CoreSender(client))
        await chain.send(HttpRequest.post("https://api.test"))
        auth_header = client.last_headers.get("Authorization", "")
        assert "Bearer" in auth_header
        assert "sk-test-key" in auth_header


# ---------------------------------------------------------------------------
# MiddlewareChain
# ---------------------------------------------------------------------------


class TestMiddlewareChain:
    async def test_build_and_send(self) -> None:
        client = _MockClient()
        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.set_core(CoreSender(client))
        response = await chain.send(HttpRequest.post("https://api.test"))
        assert response.status_code == 200

    async def test_no_core_raises(self) -> None:
        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        with pytest.raises(RuntimeError, match="Core sender"):
            chain.build()


# ---------------------------------------------------------------------------
# MetricsMiddleware
# ---------------------------------------------------------------------------


class TestMetricsMiddleware:
    async def test_tracks_requests(self) -> None:
        client = _MockClient()
        metrics = MetricsMiddleware()
        chain = MiddlewareChain()
        chain.add(metrics)
        chain.set_core(CoreSender(client))
        await chain.send(HttpRequest.post("https://api.test"))
        await chain.send(HttpRequest.post("https://api.test"))
        stats = metrics.statistics
        assert stats.total_requests == 2
