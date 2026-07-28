"""Transport error translation — maps HTTP/httpx exceptions to Atlas errors.

The transport layer catches network-level exceptions and translates them
into the provider error hierarchy so that provider adapters and callers
only ever see ``ProviderError`` subclasses.
"""

from __future__ import annotations

from typing import Any

from app.provider.errors import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    StreamingError,
    TimeoutError,
)


class TransportError(ProviderError):
    """Base class for transport-level errors."""

    def __init__(self, message: str, *, code: str = "TRANSPORT_ERROR", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code=code, details=details)


def translate_error(error: Exception, context: str = "") -> ProviderError:
    """Translate an HTTP/httpx exception into an Atlas ``ProviderError``.

    Args:
        error: The exception to translate.
        context: Optional context string (e.g. the operation being performed).

    Returns:
        A ``ProviderError`` subclass appropriate to the failure mode.
    """
    message = str(error) or error.__class__.__name__
    if context:
        message = f"{context}: {message}"

    import httpx

    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(message=message)

    if isinstance(error, httpx.ConnectError):
        return ProviderUnavailableError(message=message)

    if isinstance(error, httpx.RemoteProtocolError):
        return ProviderUnavailableError(message=message)

    if isinstance(error, httpx.TransportError):
        return TransportError(message=message)

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 401 or status == 403:
            return AuthenticationError(message=message)
        if status == 429:
            return RateLimitError(message=message)
        if status >= 500:
            return ProviderUnavailableError(message=message)
        return ProviderError(message=message)

    return ProviderError(message=message)
