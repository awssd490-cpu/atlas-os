"""HttpTransport — the core HTTP communication layer.

All provider HTTP communication flows through ``HttpTransport``.
Provider adapters must NEVER call httpx directly.

The transport knows nothing about Claude, GPT, Gemini, Ollama,
or Atlas prompts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.provider.config import ProviderConfig
from app.transport.errors import translate_error
from app.transport.middleware import (
    AuthMiddleware,
    CoreSender,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareChain,
    RetryMiddleware,
)
from app.transport.models import (
    HttpAuth,
    HttpMethod,
    HttpRequest,
    HttpResponse,
    TransportConfig,
    TransportStatistics,
)


class HttpTransport:
    """Provider-independent HTTP transport.

    Manages HTTP clients, middleware, and sends all provider requests.
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        transport_config: TransportConfig | None = None,
    ) -> None:
        self._config = config
        self._tconf = transport_config or TransportConfig()
        self._client: Any = None
        self._chain: MiddlewareChain | None = None
        self._metrics: MetricsMiddleware | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the HTTP client and build the middleware chain."""
        import httpx

        tconf = self._tconf
        limits = httpx.Limits(
            max_keepalive_connections=10,
            max_connections=50,
        )

        timeouts = httpx.Timeout(
            timeout=tconf.request_timeout,
            connect=tconf.connect_timeout,
        )

        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=timeouts,
            follow_redirects=True,
        )

        # Build middleware chain
        self._metrics = MetricsMiddleware()

        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.add(RetryMiddleware(
            max_retries=self._config.retry.max_retries if self._config else tconf.max_retries,
            base_delay=self._config.retry.base_delay_seconds if self._config else tconf.retry_base_delay,
            max_delay=self._config.retry.max_delay_seconds if self._config else tconf.retry_max_delay,
            backoff=self._config.retry.backoff_multiplier if self._config else tconf.retry_backoff,
        ))
        if self._config:
            chain.add(AuthMiddleware(self._config))
        chain.add(self._metrics)
        chain.set_core(CoreSender(self._client, config=self._config))
        self._chain = chain

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(self, request: HttpRequest) -> HttpResponse:
        """Send an HTTP request and return the response.

        Args:
            request: The provider-agnostic HTTP request.

        Returns:
            The HTTP response.

        Raises:
            ProviderError subclasses on failure.
        """
        if self._chain is None:
            raise RuntimeError("Transport not initialized — call initialize() first")

        try:
            return await self._chain.send(request)
        except Exception as exc:
            translated = translate_error(exc, context=request.url)
            raise translated from exc

    async def send_json(
        self,
        url: str,
        json_body: dict[str, Any] | list,
        *,
        method: HttpMethod = HttpMethod.POST,
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
    ) -> HttpResponse:
        """Convenience method — send a JSON request.

        Builds an ``HttpRequest`` from parameters and sends it.
        """
        from app.transport.models import HttpHeaders

        req = HttpRequest(
            method=method,
            url=url,
            headers=HttpHeaders.of(**(headers or {})),
            json_body=json_body,
            query_params=query_params or {},
        )
        return await self.send(req)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        request: HttpRequest,
    ) -> AsyncIterator[bytes]:
        """Stream response body bytes from an HTTP request.

        Args:
            request: The streaming HTTP request.

        Yields:
            Raw bytes chunks from the response body.

        Raises:
            ProviderError subclasses on failure.
        """
        if self._client is None:
            raise RuntimeError("Transport not initialized — call initialize() first")

        import httpx

        headers = request.headers.as_dict if request.headers else {}
        auth_obj = httpx.BearerToken(request.auth.token) if request.auth and request.auth.type == "bearer" and request.auth.token else None

        try:
            async with self._client.stream(
                request.method.value.lower(),
                request.url,
                headers=headers,
                json=request.json_body,
                auth=auth_obj,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except Exception as exc:
            raise translate_error(exc, context=request.url) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self, url: str) -> bool:
        """Check if the remote endpoint is reachable."""
        try:
            req = HttpRequest.get(url)
            response = await self.send(req)
            return response.is_success
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> TransportStatistics:
        """Return accumulated transport statistics."""
        if self._metrics is None:
            return TransportStatistics()
        return self._metrics.statistics

    @property
    def is_initialized(self) -> bool:
        return self._client is not None
