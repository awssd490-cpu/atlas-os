"""Tests for transport error translation."""

from __future__ import annotations

import pytest

from app.provider.errors import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
)
from app.transport.errors import TransportError, translate_error


class TestTranslateError:
    def test_httpx_timeout(self) -> None:
        import httpx
        exc = httpx.TimeoutException("Connection timed out")
        result = translate_error(exc)
        assert isinstance(result, TimeoutError)

    def test_httpx_connect_error(self) -> None:
        import httpx
        exc = httpx.ConnectError("DNS resolution failed")
        result = translate_error(exc)
        assert isinstance(result, ProviderUnavailableError)

    def test_httpx_remote_protocol_error(self) -> None:
        import httpx
        exc = httpx.RemoteProtocolError("Connection reset")
        result = translate_error(exc)
        assert isinstance(result, ProviderUnavailableError)

    def test_httpx_http_status_401_translated(self) -> None:
        """401 should become AuthenticationError."""
        import httpx

        class _MockResp:
            status_code = 401
            content = b""
            text = ""
            headers = {}

        exc = httpx.HTTPStatusError(
            "Unauthorized",
            request=type("_", (), {"url": "https://api.test"})(),
            response=_MockResp(),
        )
        result = translate_error(exc)
        assert isinstance(result, AuthenticationError)

    def test_httpx_http_status_429_translated(self) -> None:
        """429 should become RateLimitError."""
        import httpx

        class _MockResp:
            status_code = 429
            content = b""
            text = ""
            headers = {}

        exc = httpx.HTTPStatusError(
            "Too Many Requests",
            request=type("_", (), {"url": "https://api.test"})(),
            response=_MockResp(),
        )
        result = translate_error(exc)
        assert isinstance(result, RateLimitError)

    def test_httpx_http_status_500_translated(self) -> None:
        """500 should become ProviderUnavailableError."""
        import httpx

        class _MockResp:
            status_code = 500
            content = b""
            text = ""
            headers = {}

        exc = httpx.HTTPStatusError(
            "Server Error",
            request=type("_", (), {"url": "https://api.test"})(),
            response=_MockResp(),
        )
        result = translate_error(exc)
        assert isinstance(result, ProviderUnavailableError)

    def test_generic_exception(self) -> None:
        result = translate_error(RuntimeError("something broke"))
        assert isinstance(result, ProviderError)

    def test_transport_error_code(self) -> None:
        err = TransportError("transport failure")
        assert err.code == "TRANSPORT_ERROR"

    def test_context_added(self) -> None:
        result = translate_error(RuntimeError("fail"), context="send_message")
        assert "send_message" in str(result)
