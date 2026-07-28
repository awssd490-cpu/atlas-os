"""Transport domain models.

Every model in this module is immutable and provider-agnostic.
The transport knows nothing about LLM concepts — only HTTP.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class HttpMethod(str, enum.Enum):
    """HTTP method for transport requests."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass(frozen=True)
class HttpHeaders:
    """Immutable HTTP headers container."""

    entries: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)

    @property
    def as_dict(self) -> dict[str, str]:
        return dict(self.entries)

    @classmethod
    def of(cls, **headers: str) -> "HttpHeaders":
        return cls(entries=dict(headers))


@dataclass(frozen=True)
class HttpAuth:
    """Authentication configuration for a transport request."""

    type: str = "bearer"  # "bearer", "api_key", "custom", "none"
    credentials_key: str = ""  # header name or query param name
    token: str = ""


@dataclass(frozen=True)
class HttpRequest:
    """A provider-agnostic HTTP request."""

    method: HttpMethod = HttpMethod.POST
    url: str = ""
    headers: HttpHeaders = field(default_factory=HttpHeaders)
    json_body: dict[str, Any] | list | None = None
    data: bytes | None = None
    query_params: dict[str, str] = field(default_factory=dict)
    auth: HttpAuth = field(default_factory=HttpAuth)
    stream: bool = False

    @classmethod
    def post(cls, url: str, **kwargs: Any) -> "HttpRequest":
        return cls(method=HttpMethod.POST, url=url, **kwargs)

    @classmethod
    def get(cls, url: str, **kwargs: Any) -> "HttpRequest":
        return cls(method=HttpMethod.GET, url=url, **kwargs)


@dataclass(frozen=True)
class HttpResponse:
    """A provider-agnostic HTTP response."""

    status_code: int = 0
    headers: HttpHeaders = field(default_factory=HttpHeaders)
    body: bytes = b""
    text: str = ""
    elapsed_seconds: float = 0.0

    def json(self) -> Any:
        """Parse body as JSON."""
        import json
        return json.loads(self.text) if self.text else json.loads(self.body)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        return 500 <= self.status_code < 600


@dataclass(frozen=True)
class TransportConfig:
    """Configuration for the HTTP transport layer.

    Consumed from ``ProviderConfig`` by the transport factory.
    """

    base_url: str = ""
    api_path: str = "/v1/messages"
    stream_path: str = "/v1/messages/stream"
    request_timeout: float = 60.0
    connect_timeout: float = 10.0
    stream_timeout: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_backoff: float = 2.0


@dataclass(frozen=True)
class TransportStatistics:
    """Statistics for a transport session."""

    total_requests: int = 0
    total_retries: int = 0
    total_errors: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class TransportResult:
    """The result of a transport ``send()`` call."""

    response: HttpResponse = field(default_factory=HttpResponse)
    retries: int = 0
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
