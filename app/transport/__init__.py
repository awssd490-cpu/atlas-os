"""Universal HTTP Transport Layer.

All provider HTTP communication flows through this layer.
Provider adapters must NEVER call httpx directly.

The transport knows nothing about Claude, GPT, Gemini, Ollama,
or Atlas prompts — it simply sends HTTP requests and returns
responses.
"""

from __future__ import annotations

from app.transport.transport import HttpTransport
from app.transport.models import (
    HttpRequest,
    HttpResponse,
    HttpMethod,
    HttpHeaders,
    HttpAuth,
    TransportConfig,
    TransportResult,
    TransportStatistics,
)
from app.transport.middleware import (
    Middleware,
    RetryMiddleware,
    AuthMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
)
from app.transport.errors import TransportError, translate_error

__all__ = [
    "HttpTransport",
    "HttpRequest",
    "HttpResponse",
    "HttpMethod",
    "HttpHeaders",
    "HttpAuth",
    "TransportConfig",
    "TransportResult",
    "TransportStatistics",
    "Middleware",
    "RetryMiddleware",
    "AuthMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "TransportError",
    "translate_error",
]
