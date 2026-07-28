"""Tests for transport domain models."""

from __future__ import annotations

import json

from app.transport.models import (
    HttpAuth,
    HttpHeaders,
    HttpMethod,
    HttpRequest,
    HttpResponse,
    TransportConfig,
    TransportResult,
    TransportStatistics,
)


class TestHttpMethod:
    def test_values(self) -> None:
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.DELETE.value == "DELETE"


class TestHttpHeaders:
    def test_of(self) -> None:
        h = HttpHeaders.of(Authorization="Bearer x", ContentType="json")
        assert h.get("Authorization") == "Bearer x"

    def test_as_dict(self) -> None:
        h = HttpHeaders.of(key="val")
        assert h.as_dict == {"key": "val"}

    def test_empty(self) -> None:
        h = HttpHeaders()
        assert h.as_dict == {}

    def test_immutable(self) -> None:
        h = HttpHeaders.of(a="b")
        with pytest.raises(AttributeError):
            h.entries = {}  # type: ignore[misc]


class TestHttpAuth:
    def test_bearer(self) -> None:
        a = HttpAuth(type="bearer", token="sk-test")
        assert a.type == "bearer"
        assert a.token == "sk-test"


class TestHttpRequest:
    def test_post(self) -> None:
        req = HttpRequest.post("https://api.example.com", json_body={"key": "val"})
        assert req.method == HttpMethod.POST
        assert req.url == "https://api.example.com"
        assert req.json_body == {"key": "val"}

    def test_get(self) -> None:
        req = HttpRequest.get("https://api.example.com/ping")
        assert req.method == HttpMethod.GET


class TestHttpResponse:
    def test_success(self) -> None:
        resp = HttpResponse(status_code=200, text='{"ok": true}')
        assert resp.is_success is True
        assert resp.is_client_error is False
        assert resp.json() == {"ok": True}

    def test_client_error(self) -> None:
        resp = HttpResponse(status_code=404)
        assert resp.is_client_error is True
        assert resp.is_success is False

    def test_server_error(self) -> None:
        resp = HttpResponse(status_code=500)
        assert resp.is_server_error is True

    def test_json_body(self) -> None:
        resp = HttpResponse(status_code=200, body=b'{"a": 1}')
        assert resp.json() == {"a": 1}


class TestTransportConfig:
    def test_defaults(self) -> None:
        c = TransportConfig()
        assert c.request_timeout == 60.0


class TestTransportStatistics:
    def test_defaults(self) -> None:
        s = TransportStatistics()
        assert s.total_requests == 0


class TestTransportResult:
    def test_create(self) -> None:
        r = TransportResult(response=HttpResponse(status_code=200), retries=1)
        assert r.retries == 1


import pytest
