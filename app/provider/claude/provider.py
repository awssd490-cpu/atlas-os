"""ClaudeProvider — the reference provider implementation.

All networking goes through ``HttpTransport``.
No httpx calls, no environment variable parsing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.provider.claude.mapper import ClaudeRequestMapper, ClaudeResponseMapper
from app.provider.config import ProviderConfig
from app.provider.models import (
    Capabilities,
    ProviderCapability,
    ProviderInfo,
    ProviderMetadata,
    ProviderRequest,
    ProviderResponse,
    StreamingChunk,
)
from app.provider.provider import Provider
from app.transport.models import HttpMethod, HttpRequest
from app.transport.transport import HttpTransport


class ClaudeProvider(Provider):
    """Provider implementation for Anthropic's Claude API.

    Uses ``HttpTransport`` for all networking.
    Uses ``ProviderConfig`` for all configuration.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._transport: HttpTransport | None = None
        self._req_mapper = ClaudeRequestMapper()
        self._resp_mapper = ClaudeResponseMapper()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create and initialize the HTTP transport."""
        transport = HttpTransport(config=self._config)
        await transport.initialize()
        self._transport = transport

    async def shutdown(self) -> None:
        """Shutdown the HTTP transport."""
        if self._transport is not None:
            await self._transport.shutdown()
            self._transport = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Send a request to Claude and return the response.

        All HTTP communication goes through ``HttpTransport``.
        """
        if self._transport is None:
            raise RuntimeError("ClaudeProvider not initialized — call initialize() first")

        body = self._req_mapper.to_dict(request)
        url = self._config.endpoint.request_url
        response = await self._transport.send_json(url, body)

        if not response.is_success:
            self._raise_for_status(response)

        data = response.json()
        return self._resp_mapper.to_response(data)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        """Stream a response from Claude.

        Yields ``StreamingChunk`` objects parsed from SSE events.
        """
        if self._transport is None:
            raise RuntimeError("ClaudeProvider not initialized — call initialize() first")

        return self._stream_internal(request)

    async def _stream_internal(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        """Internal streaming coroutine."""
        body = self._req_mapper.to_dict(request)
        body["stream"] = True
        url = self._config.endpoint.stream_url

        http_req = HttpRequest(
            method=HttpMethod.POST,
            url=url,
            json_body=body,
        )

        buffer = b""
        async for raw_chunk in self._transport.stream(http_req):
            buffer += raw_chunk
            # Parse SSE events from buffer
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.startswith(b"data: "):
                    event_data = line[6:].decode("utf-8").strip()
                    if event_data == "[DONE]":
                        return
                    try:
                        payload = json.loads(event_data)
                        event_type = payload.get("type", "")
                        chunk = self._resp_mapper.to_chunk(event_type, payload)
                        if chunk is not None:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def count_tokens(self, request: ProviderRequest) -> int:
        """Estimate token count via a lightweight request.

        Delegates to the transport for the actual HTTP call.
        """
        body = self._req_mapper.to_dict(request)
        body["max_tokens"] = 1  # minimal for counting
        url = self._config.endpoint.base_url.rstrip("/") + "/v1/messages/count_tokens"

        try:
            if self._transport is None:
                return 0
            response = await self._transport.send_json(url, body)
            if response.is_success:
                data = response.json()
                return data.get("input_tokens", 0)
        except Exception:
            pass

        # Fallback: rough estimate
        total = len(request.system)
        for msg in request.messages:
            total += len(msg.content)
        return max(1, total // 4)

    # ------------------------------------------------------------------
    # Provider info
    # ------------------------------------------------------------------

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            metadata=ProviderMetadata(
                name="claude",
                version="1.0.0",
                description="Anthropic Claude API (Messages API)",
                website="https://anthropic.com",
                documentation="https://docs.anthropic.com/en/api/messages",
            ),
            capabilities=[
                ProviderCapability(name=Capabilities.STREAMING, description="Server-sent events streaming"),
                ProviderCapability(name=Capabilities.SYSTEM_PROMPTS, description="System prompt support"),
                ProviderCapability(name=Capabilities.TEMPERATURE, description="Temperature control"),
                ProviderCapability(name=Capabilities.STOP_SEQUENCES, description="Custom stop sequences"),
                ProviderCapability(name=Capabilities.JSON_MODE, description="JSON mode via system prompt"),
            ],
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the Claude API is reachable."""
        if self._transport is None:
            return False
        url = self._config.endpoint.base_url.rstrip("/") + "/v1/messages"
        return await self._transport.health_check(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        """Raise the appropriate ProviderError for a non-success response."""
        from app.provider.errors import (
            AuthenticationError,
            ProviderUnavailableError,
            RateLimitError,
        )

        status = response.status_code
        body = response.text[:500] if response.text else ""
        detail = f"Claude API returned {status}: {body}"

        if status == 401 or status == 403:
            raise AuthenticationError(message=detail)
        if status == 429:
            raise RateLimitError(message=detail)
        if status >= 500:
            raise ProviderUnavailableError(message=detail)
        raise RuntimeError(detail)
