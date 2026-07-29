"""OpenAICompatibleProvider — Universal OpenAI-Compatible Provider.

Communicates with ANY server implementing the OpenAI Chat Completions API.

Examples:
    - OpenAI:        https://api.openai.com/v1
    - OpenRouter:    https://openrouter.ai/api/v1
    - Groq:          https://api.groq.com/openai/v1
    - Together AI:   https://api.together.xyz/v1
    - Fireworks AI:  https://api.fireworks.ai/inference/v1
    - DeepInfra:     https://api.deepinfra.com/v1/openai
    - Nebius:        https://api.nebius.ai/v1
    - SambaNova:     https://api.sambanova.ai/v1
    - LM Studio:     http://localhost:1234/v1
    - LocalAI:       http://localhost:8080/v1
    - vLLM:          http://localhost:8000/v1
    - LiteLLM:       http://localhost:4000/v1
    - Ollama:        http://localhost:11434/v1

All networking goes through ``HttpTransport``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

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
from app.provider.openai.mapper import OpenAIRequestMapper, OpenAIResponseMapper
from app.provider.openai.models import DEFAULT_API_PATH, DEFAULT_STREAM_PATH
from app.provider.provider import Provider
from app.transport.models import HttpMethod, HttpRequest
from app.transport.transport import HttpTransport


class OpenAICompatibleProvider(Provider):
    """Provider implementation for any OpenAI-compatible API.

    Uses ``HttpTransport`` for all networking.
    Uses ``ProviderConfig`` for all configuration.

    The same provider works with OpenAI, OpenRouter, Groq, Together AI,
    Fireworks AI, DeepInfra, and many others — just change the base URL.

    No code changes required between providers.
    Only configuration changes.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._transport: HttpTransport | None = None
        self._req_mapper = OpenAIRequestMapper()
        self._resp_mapper = OpenAIResponseMapper()

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
        """Send a request to an OpenAI-compatible API and return the response.

        All HTTP communication goes through ``HttpTransport``.
        """
        if self._transport is None:
            raise RuntimeError(
                "OpenAICompatibleProvider not initialized — call initialize() first"
            )

        body = self._build_request_body(request)
        url = self._request_url
        response = await self._transport.send_json(url, body)

        if not response.is_success:
            self._raise_for_status(response)

        data = response.json()
        return self._resp_mapper.to_response(data)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        """Stream a response from an OpenAI-compatible API.

        Yields ``StreamingChunk`` objects parsed from SSE ``data:`` events.
        The HTTP transport handles all networking; this method only
        parses the SSE stream.
        """
        if self._transport is None:
            raise RuntimeError(
                "OpenAICompatibleProvider not initialized — call initialize() first"
            )

        return self._stream_internal(request)

    async def _stream_internal(
        self, request: ProviderRequest
    ) -> AsyncIterator[StreamingChunk]:
        """Internal streaming coroutine.

        Parses the SSE stream from an OpenAI-compatible API.

        Format::

            data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}

            data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}

            data: [DONE]
        """
        body = self._build_request_body(request)
        body["stream"] = True
        url = self._request_url

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
                    if not event_data:
                        continue
                    try:
                        payload = json.loads(event_data)
                        chunk = self._resp_mapper.to_chunk(payload)
                        if chunk is not None:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    async def count_tokens(self, request: ProviderRequest) -> int:
        """Estimate token count.

        OpenAI-compatible APIs don't generally have a count_tokens endpoint,
        so we use a rough character-based estimate.
        """
        total = len(request.system) if request.system else 0
        for msg in request.messages:
            total += len(msg.content) if msg.content else 0
        return max(1, total // 4)

    # ------------------------------------------------------------------
    # Provider info
    # ------------------------------------------------------------------

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            metadata=ProviderMetadata(
                name="openai-compatible",
                version="1.0.0",
                description="Universal OpenAI-Compatible provider",
                website="https://platform.openai.com/docs/api-reference/chat",
                documentation="https://platform.openai.com/docs/api-reference/chat",
            ),
            capabilities=[
                ProviderCapability(
                    name=Capabilities.STREAMING,
                    description="Server-sent events streaming",
                ),
                ProviderCapability(
                    name=Capabilities.JSON_MODE,
                    description="JSON mode via response_format",
                ),
                ProviderCapability(
                    name=Capabilities.SYSTEM_PROMPTS,
                    description="System prompt support",
                ),
                ProviderCapability(
                    name=Capabilities.TEMPERATURE,
                    description="Temperature control",
                ),
                ProviderCapability(
                    name=Capabilities.STOP_SEQUENCES,
                    description="Custom stop sequences",
                ),
                ProviderCapability(
                    name=Capabilities.FUNCTION_CALLING,
                    description="Function/tool calling (parse only)",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the API endpoint is reachable.

        Sends a lightweight request to the model list endpoint.
        """
        if self._transport is None:
            return False
        base = self._config.endpoint.base_url.rstrip("/")
        url = f"{base}/models"
        return await self._transport.health_check(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _request_url(self) -> str:
        """Build the request URL from the endpoint configuration.

        For OpenAI-compatible APIs, if the configured ``api_path`` is
        still the Claude default (``/v1/messages``), we default to
        ``/chat/completions`` instead.  This allows users to omit
        ``api_path`` when setting ``base_url``.
        """
        base = self._config.endpoint.base_url.rstrip("/")
        if not base:
            return ""

        path = self._config.endpoint.api_path
        # If api_path is unset or still the Claude default, use OpenAI default
        if not path or path in ("/v1/messages", "/v1/messages/stream"):
            path = DEFAULT_API_PATH
        return f"{base}{path}"

    def _build_request_body(self, request: ProviderRequest) -> dict[str, Any]:
        """Build the request body using generation defaults from config."""
        gen = self._config.generation
        return self._req_mapper.to_dict(
            request,
            model=self._config.model or gen.model,
            frequency_penalty=gen.frequency_penalty,
            presence_penalty=gen.presence_penalty,
        )

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        """Raise the appropriate ProviderError for a non-success response."""
        from app.provider.errors import (
            AuthenticationError,
            InvalidRequestError,
            ProviderUnavailableError,
            RateLimitError,
        )

        status = response.status_code
        body = response.text[:500] if response.text else ""
        detail = f"OpenAI-compatible API returned {status}: {body}"

        if status == 400:
            # Try to extract more specific OpenAI error
            try:
                data = response.json()
                error = data.get("error", {})
                err_code = error.get("code", "")
                err_message = error.get("message", "")
                if err_code or err_message:
                    detail = f"OpenAI-compatible API error [{err_code}]: {err_message}"
            except Exception:
                pass
            raise InvalidRequestError(message=detail)
        if status == 401 or status == 403:
            raise AuthenticationError(message=detail)
        if status == 429:
            raise RateLimitError(message=detail)
        if status >= 500:
            raise ProviderUnavailableError(message=detail)
        raise RuntimeError(detail)
