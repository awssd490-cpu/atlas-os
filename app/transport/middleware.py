"""Transport middleware pipeline.

Middleware wraps the core HTTP request/response cycle.
Each middleware intercepts requests and can modify them,
add logging, handle retries, inject authentication, etc.
Middleware is async and composable.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

from app.provider.config import ProviderConfig
from app.transport.errors import RateLimitError, translate_error
from app.transport.models import HttpHeaders, HttpRequest, HttpResponse


logger = logging.getLogger("atlas.transport")


# ---------------------------------------------------------------------------
# Middleware protocol
# ---------------------------------------------------------------------------


class Middleware(abc.ABC):
    """Abstract middleware that wraps the request/response cycle.

    Implementations override ``send()`` to intercept requests and
    optionally call ``next_middleware.send()`` to continue the chain.
    """

    def __init__(self) -> None:
        self._next: Middleware | None = None

    def set_next(self, middleware: "Middleware") -> None:
        self._next = middleware

    @abc.abstractmethod
    async def send(self, request: HttpRequest) -> HttpResponse:
        """Send *request* through the middleware chain.

        Args:
            request: The outgoing HTTP request.

        Returns:
            The HTTP response.

        Raises:
            ProviderError subclasses on failure.
        """
        ...


# ---------------------------------------------------------------------------
# Core sender — always last in the chain
# ---------------------------------------------------------------------------


class CoreSender(Middleware):
    """The terminal middleware that actually sends HTTP requests.

    Always sits at the end of the middleware chain.
    """

    def __init__(self, client: Any, config: ProviderConfig | None = None) -> None:
        super().__init__()
        self._client = client
        self._config = config

    async def send(self, request: HttpRequest) -> HttpResponse:
        import httpx
        from app.transport.models import HttpHeaders as HH

        url = request.url
        headers = request.headers.as_dict if request.headers else {}
        params = request.query_params or None

        auth_obj = None
        if request.auth and request.auth.type == "bearer":
            auth_obj = httpx.BearerToken(request.auth.token) if request.auth.token else None

        timeout_seconds = self._config.timeout.request_timeout_seconds if self._config else 60.0

        try:
            t0 = time.monotonic()

            method_map = {
                "GET": self._client.get,
                "POST": self._client.post,
                "PUT": self._client.put,
                "PATCH": self._client.patch,
                "DELETE": self._client.delete,
            }
            sender = method_map.get(request.method.value, self._client.post)
            kw: dict[str, Any] = {
                "url": url, "headers": headers, "params": params,
                "auth": auth_obj, "timeout": timeout_seconds,
            }
            if request.method.value in ("POST", "PUT", "PATCH"):
                kw["json"] = request.json_body
                kw["data"] = request.data

            resp = await sender(**kw)
            elapsed = time.monotonic() - t0
            resp.raise_for_status()

            return HttpResponse(
                status_code=resp.status_code,
                headers=HH.of(**(dict(resp.headers))),
                body=resp.content,
                text=resp.text,
                elapsed_seconds=elapsed,
            )

        except httpx.HTTPStatusError as exc:
            elapsed = time.monotonic() - t0 if 't0' in dir() else 0.0
            status = exc.response.status_code
            return HttpResponse(
                status_code=status,
                headers=HH.of(**(dict(exc.response.headers))),
                body=exc.response.content,
                text=exc.response.text,
                elapsed_seconds=elapsed,
            )


# ---------------------------------------------------------------------------
# Built-in middleware
# ---------------------------------------------------------------------------


class RetryMiddleware(Middleware):
    """Retry requests on transient failures with exponential backoff.

    Wraps the downstream middleware chain.  Intercepts the response
    and retries on retryable status codes or exceptions.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff: float = 2.0,
    ) -> None:
        super().__init__()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._backoff = backoff

    async def send(self, request: HttpRequest) -> HttpResponse:
        last_error: Exception | None = None
        import asyncio
        import random

        for attempt in range(self._max_retries + 1):
            try:
                if self._next:
                    response = await self._next.send(request)
                else:
                    from app.transport.models import HttpResponse as HR
                    response = HR()

                if attempt > 0:
                    logger.debug("Retry attempt %d succeeded", attempt)

                return response

            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries and self._is_retryable(exc):
                    delay = min(
                        self._base_delay * (self._backoff ** attempt) + random.uniform(0, 0.5),
                        self._max_delay,
                    )
                    logger.debug("Retrying in %.2fs (attempt %d/%d)", delay, attempt + 1, self._max_retries)
                    await asyncio.sleep(delay)
                else:
                    raise

        # Should not reach here
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the exception is worth retrying."""
        from app.provider.errors import RateLimitError, TimeoutError, ProviderUnavailableError
        return isinstance(exc, (RateLimitError, TimeoutError, ProviderUnavailableError))


class AuthMiddleware(Middleware):
    """Injects authentication into outgoing requests.

    Reads credentials from the ``ProviderConfig`` and applies
    the configured auth strategy.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__()
        self._config = config

    async def send(self, request: HttpRequest) -> HttpResponse:
        import httpx

        if not self._next:
            from app.transport.models import HttpResponse as HR
            return HR()

        creds = self._config.credentials
        if creds.has_key:
            # Inject Authorization header
            new_headers = dict(request.headers.entries if request.headers else {})
            new_headers["Authorization"] = f"Bearer {creds.api_key}"
            if creds.organization_id:
                new_headers["anthropic-organization"] = creds.organization_id
            request = HttpRequest(
                method=request.method,
                url=request.url,
                headers=HttpHeaders.of(**new_headers),
                json_body=request.json_body,
                data=request.data,
                query_params=request.query_params,
                auth=request.auth,
                stream=request.stream,
            )

        return await self._next.send(request)


class LoggingMiddleware(Middleware):
    """Logs request/response summaries.

    Logs at DEBUG level by default.
    """

    def __init__(self, log_level: str = "debug") -> None:
        super().__init__()
        self._log_level = log_level

    async def send(self, request: HttpRequest) -> HttpResponse:
        if not self._next:
            from app.transport.models import HttpResponse as HR
            return HR()

        t0 = time.monotonic()
        method = request.method.value
        url = request.url

        try:
            response = await self._next.send(request)
            elapsed = (time.monotonic() - t0) * 1000
            log_msg = f"{method} {url} → {response.status_code} ({elapsed:.0f}ms)"
            self._log(log_msg)
            return response
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            self._log(f"{method} {url} → ERROR {type(exc).__name__} ({elapsed:.0f}ms)", level="warning")
            raise

    def _log(self, message: str, level: str = "debug") -> None:
        if level == "warning":
            logger.warning(message)
        else:
            logger.debug(message)


class MetricsMiddleware(Middleware):
    """Records transport metrics (request counts, timing, errors).

    Statistics are accumulated in a ``TransportStatistics`` object.
    """

    def __init__(self) -> None:
        super().__init__()
        self._requests = 0
        self._retries = 0
        self._errors = 0

    @property
    def statistics(self) -> Any:
        from app.transport.models import TransportStatistics
        return TransportStatistics(
            total_requests=self._requests,
            total_retries=self._retries,
            total_errors=self._errors,
        )

    async def send(self, request: HttpRequest) -> HttpResponse:
        if not self._next:
            from app.transport.models import HttpResponse as HR
            return HR()

        self._requests += 1
        try:
            response = await self._next.send(request)
            return response
        except Exception as exc:
            self._errors += 1
            raise


# ---------------------------------------------------------------------------
# Middleware chain builder
# ---------------------------------------------------------------------------


class MiddlewareChain:
    """Builds and manages a middleware chain.

    Usage::

        chain = MiddlewareChain()
        chain.add(LoggingMiddleware())
        chain.add(RetryMiddleware(max_retries=3))
        chain.add(AuthMiddleware(config))
        chain.set_core(CoreSender(client, config))
        response = await chain.send(request)
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []
        self._core: CoreSender | None = None

    def add(self, middleware: Middleware) -> "MiddlewareChain":
        """Add a middleware to the chain.

        Middleware added first runs first (outermost).
        """
        self._middleware.append(middleware)
        return self

    def set_core(self, core: CoreSender) -> None:
        """Set the core sender (always runs last)."""
        self._core = core

    def build(self) -> Middleware:
        """Link all middleware together and return the entry point.

        The first middleware added becomes the outermost/last chain link.
        The core sender is linked to the end of the chain.
        """
        if not self._middleware:
            if self._core:
                return self._core
            raise RuntimeError("No middleware or core sender configured")

        if self._core is None:
            raise RuntimeError("Core sender not set — call set_core() first")

        # Chain: mw[0] → mw[1] → ... → mw[n-1] → core
        chain = list(self._middleware)
        previous = self._core
        for mw in reversed(chain):
            mw.set_next(previous)
            previous = mw

        return chain[0]

    async def send(self, request: HttpRequest) -> HttpResponse:
        entry = self.build()
        return await entry.send(request)
