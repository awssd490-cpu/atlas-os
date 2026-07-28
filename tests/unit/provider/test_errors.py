"""Tests for provider error hierarchy."""

from __future__ import annotations

import pytest

from app.provider.errors import (
    AuthenticationError,
    CapabilityNotSupportedError,
    DuplicateProviderError,
    InvalidRequestError,
    ProviderError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    StreamingError,
    TimeoutError,
    TokenLimitError,
)
from app.core.errors import AtlasError


class TestProviderErrorHierarchy:
    def test_provider_error_is_atlas_error(self) -> None:
        err = ProviderError("test")
        assert isinstance(err, AtlasError)

    def test_authentication_error(self) -> None:
        err = AuthenticationError()
        assert err.code == "PROVIDER_AUTHENTICATION_ERROR"

    def test_rate_limit_error(self) -> None:
        err = RateLimitError()
        assert err.code == "PROVIDER_RATE_LIMIT_ERROR"

    def test_timeout_error(self) -> None:
        err = TimeoutError()
        assert err.code == "PROVIDER_TIMEOUT_ERROR"

    def test_invalid_request_error(self) -> None:
        err = InvalidRequestError()
        assert err.code == "PROVIDER_INVALID_REQUEST_ERROR"

    def test_unavailable_error(self) -> None:
        err = ProviderUnavailableError()
        assert err.code == "PROVIDER_UNAVAILABLE_ERROR"

    def test_streaming_error(self) -> None:
        err = StreamingError()
        assert err.code == "PROVIDER_STREAMING_ERROR"

    def test_token_limit_error(self) -> None:
        err = TokenLimitError()
        assert err.code == "PROVIDER_TOKEN_LIMIT_ERROR"

    def test_not_found_error(self) -> None:
        err = ProviderNotFoundError(name="test-provider")
        assert "test-provider" in str(err)
        assert err.code == "PROVIDER_NOT_FOUND"

    def test_duplicate_error(self) -> None:
        err = DuplicateProviderError(name="dup")
        assert "dup" in str(err)

    def test_capability_not_supported(self) -> None:
        err = CapabilityNotSupportedError(capability="vision")
        assert "vision" in str(err)

    def test_to_dict(self) -> None:
        err = RateLimitError(details={"retry_after": 30})
        d = err.to_dict()
        assert d["code"] == "PROVIDER_RATE_LIMIT_ERROR"
        assert d["details"]["retry_after"] == 30
