"""Provider error hierarchy.

Every provider-related error derives from ``ProviderError`` so callers
can catch all provider issues distinctly from platform errors.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ProviderError(AtlasError):
    """Base class for all provider errors."""

    def __init__(self, message: str, *, code: str = "PROVIDER_ERROR", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code=code, details=details)


class AuthenticationError(ProviderError):
    """Raised when provider API key or credentials are invalid."""

    def __init__(self, message: str = "Provider authentication failed", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_AUTHENTICATION_ERROR", details=details)


class RateLimitError(ProviderError):
    """Raised when the provider rate limit is exceeded."""

    def __init__(self, message: str = "Provider rate limit exceeded", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_RATE_LIMIT_ERROR", details=details)


class TimeoutError(ProviderError):
    """Raised when a provider request times out."""

    def __init__(self, message: str = "Provider request timed out", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_TIMEOUT_ERROR", details=details)


class InvalidRequestError(ProviderError):
    """Raised when the request to a provider is malformed."""

    def __init__(self, message: str = "Invalid provider request", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_INVALID_REQUEST_ERROR", details=details)


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unreachable or down."""

    def __init__(self, message: str = "Provider unavailable", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_UNAVAILABLE_ERROR", details=details)


class StreamingError(ProviderError):
    """Raised when streaming fails mid-response."""

    def __init__(self, message: str = "Streaming error", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_STREAMING_ERROR", details=details)


class TokenLimitError(ProviderError):
    """Raised when the request exceeds the provider's token limit."""

    def __init__(self, message: str = "Token limit exceeded", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="PROVIDER_TOKEN_LIMIT_ERROR", details=details)


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""

    def __init__(self, name: str = "", *, details: dict[str, Any] | None = None) -> None:
        msg = f"Provider {name!r} not found" if name else "Provider not found"
        super().__init__(msg, code="PROVIDER_NOT_FOUND", details=details)


class DuplicateProviderError(ProviderError):
    """Raised when a provider is registered under an already-used name."""

    def __init__(self, name: str = "", *, details: dict[str, Any] | None = None) -> None:
        msg = f"Provider {name!r} is already registered" if name else "Duplicate provider registration"
        super().__init__(msg, code="PROVIDER_DUPLICATE", details=details)


class CapabilityNotSupportedError(ProviderError):
    """Raised when the requested capability is not supported by the provider."""

    def __init__(self, capability: str = "", *, details: dict[str, Any] | None = None) -> None:
        msg = f"Capability {capability!r} not supported" if capability else "Capability not supported"
        super().__init__(msg, code="PROVIDER_CAPABILITY_NOT_SUPPORTED", details=details)
