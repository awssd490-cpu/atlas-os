"""Provider interface — the contract every LLM provider must implement.

The core never knows whether it is talking to Claude, GPT, Gemini, Ollama,
or any other provider — everything is simply a ``Provider``.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from app.provider.errors import (
    AuthenticationError,
    CapabilityNotSupportedError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
    StreamingError,
    TimeoutError,
    TokenLimitError,
)
from app.provider.models import (
    ProviderCapability,
    ProviderInfo,
    ProviderMetadata,
    ProviderRequest,
    ProviderResponse,
    StreamingChunk,
)


class Provider(abc.ABC):
    """Abstract interface that every LLM provider must implement.

    All methods are async.  Implementations are responsible for
    translating ``ProviderRequest`` into their native format and
    translating responses back into ``ProviderResponse``.
    """

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Prepare the provider for use.

        Called once at startup.  Implementations should validate
        configuration, open connections, or prepare resources.
        """
        ...

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Release provider resources.

        Called once at shutdown.  Implementations should close
        connections and clean up.
        """
        ...

    @abc.abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request and receive a complete (non-streaming) response.

        Args:
            request: The provider-agnostic request.

        Returns:
            A ``ProviderResponse`` with the generated content.

        Raises:
            AuthenticationError: Invalid API key or credentials.
            RateLimitError: Rate limit exceeded.
            TimeoutError: Request timed out.
            InvalidRequestError: Malformed request.
            ProviderUnavailableError: Provider is unreachable.
            TokenLimitError: Request exceeds token limits.
        """
        ...

    @abc.abstractmethod
    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        """Send a request and stream the response.

        Args:
            request: The provider-agnostic request.

        Yields:
            ``StreamingChunk`` objects as they arrive.

        Raises:
            AuthenticationError: Invalid API key or credentials.
            RateLimitError: Rate limit exceeded.
            TimeoutError: Request timed out.
            InvalidRequestError: Malformed request.
            ProviderUnavailableError: Provider is unreachable.
            StreamingError: Streaming failed mid-response.
        """
        ...

    @abc.abstractmethod
    async def count_tokens(self, request: ProviderRequest) -> int:
        """Estimate or compute the token count for *request*.

        Args:
            request: The request to count tokens for.

        Returns:
            The estimated token count.
        """
        ...

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def provider_info(self) -> ProviderInfo:
        """Return the provider's metadata and capability declaration."""
        ...

    async def health_check(self) -> bool:
        """Return ``True`` when the provider is functional.

        Default implementation calls ``count_tokens`` on a minimal request.
        Override for provider-specific health checks.
        """
        try:
            req = ProviderRequest(messages=[], system="ping")
            await self.count_tokens(req)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------

    def supports_capability(self, name: str) -> bool:
        """Return ``True`` if the provider supports the given capability."""
        return self.provider_info.has_capability(name)

    def assert_capability(self, name: str) -> None:
        """Raise ``CapabilityNotSupportedError`` if the capability is missing."""
        if not self.supports_capability(name):
            raise CapabilityNotSupportedError(
                capability=name,
                details={"provider": self.provider_info.metadata.name},
            )

    # ------------------------------------------------------------------
    # Common capabilities
    # ------------------------------------------------------------------

    @property
    def supports_streaming(self) -> bool:
        return self.supports_capability("streaming")

    @property
    def supports_tool_calling(self) -> bool:
        return self.supports_capability("tool_calling")

    @property
    def supports_vision(self) -> bool:
        return self.supports_capability("vision")

    @property
    def supports_audio(self) -> bool:
        return self.supports_capability("audio")

    @property
    def supports_json_mode(self) -> bool:
        return self.supports_capability("json_mode")

    @property
    def supports_embeddings(self) -> bool:
        return self.supports_capability("embeddings")
