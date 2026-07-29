"""Agent Runtime — the reasoning loop orchestrator.

The ``AgentRuntime`` owns the reasoning loop.

It coordinates three existing systems:
1. **Provider Runtime** — generates responses (no looping, no tool execution)
2. **Tool Runtime** — executes tools (provider-agnostic)
3. **Integration Layer** — converts between provider messages and tool calls/results

Ownership boundaries:
- The agent runtime loops; the provider never loops.
- The agent runtime coordinates; the tool runtime executes.
- The agent runtime never contains provider-specific logic.
- The agent runtime never executes tools directly.
"""

from __future__ import annotations

import time
from typing import Any

from app.agent.config import AgentConfig
from app.agent.errors import (
    IterationLimitExceeded,
    ProviderExecutionError,
    ToolCallLimitExceeded,
)
from app.provider.models import ProviderMessage, ProviderRequest, ProviderResponse
from app.provider.provider import Provider
from app.tools.integration import execute_tool_calls, extract_tool_calls
from app.tools.runtime import ToolRuntime


class AgentRuntime:
    """Orchestrates the reasoning loop between a provider and tools.

    The runtime sends messages to the provider, extracts tool calls from
    responses, executes them via ``ToolRuntime``, formats the results
    back into provider messages, and repeats until the provider returns
    a final response (one without tool calls).

    Usage::

        runtime = AgentRuntime(provider, tool_runtime)
        response = await runtime.run(messages)
        print(response.content)
    """

    def __init__(
        self,
        provider: Provider,
        tool_runtime: ToolRuntime,
        config: AgentConfig | None = None,
    ) -> None:
        self._provider = provider
        self._tool_runtime = tool_runtime
        self._config = config or AgentConfig.default()

        # Internal state (reset per run)
        self._iteration_count: int = 0
        self._tool_calls_executed: int = 0
        self._provider_requests: int = 0
        self._start_time: float = 0.0
        self._elapsed_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[ProviderMessage],
        *,
        config: AgentConfig | None = None,
        system: str = "",
        **request_kwargs: Any,
    ) -> ProviderResponse:
        """Run the reasoning loop.

        Sends *messages* (and optional *system* prompt) to the provider,
        then iterates: extract tool calls → execute tools → send results
        back to the provider — until the provider returns a response with
        no tool calls.

        The caller's ``request_kwargs`` dict is never mutated.

        Args:
            messages: The initial conversation messages.
            config: Optional per-run configuration override.
            system: Optional system prompt.
            **request_kwargs: Additional keyword arguments forwarded to
                ``ProviderRequest`` (e.g. ``max_tokens``, ``temperature``).

        Returns:
            The final ``ProviderResponse`` from the provider.

        Raises:
            IterationLimitExceeded: If the iteration limit is reached.
            ToolCallLimitExceeded: If the tool call limit is reached.
            ProviderExecutionError: If the provider raises an exception.
            AgentError: For other agent-level errors.
        """
        resolved_config = config or self._config
        active_messages = list(messages)

        # Never mutate caller-owned objects — copy kwargs
        kwargs = dict(request_kwargs)

        # Reset per-run state
        self._iteration_count = 0
        self._tool_calls_executed = 0
        self._provider_requests = 0
        self._start_time = time.monotonic()
        self._elapsed_time = 0.0

        try:
            return await self._loop(
                active_messages,
                resolved_config,
                system,
                kwargs,
            )
        finally:
            self._elapsed_time = time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Internal state properties
    # ------------------------------------------------------------------

    @property
    def iteration_count(self) -> int:
        """Number of reasoning iterations completed in the last run."""
        return self._iteration_count

    @property
    def tool_calls_executed(self) -> int:
        """Number of tool calls executed in the last run."""
        return self._tool_calls_executed

    @property
    def provider_requests(self) -> int:
        """Number of provider requests made in the last run."""
        return self._provider_requests

    @property
    def elapsed_time(self) -> float:
        """Wall-clock time of the last run in seconds."""
        return self._elapsed_time

    @property
    def provider(self) -> Provider:
        """The underlying provider instance."""
        return self._provider

    @property
    def tool_runtime(self) -> ToolRuntime:
        """The underlying tool runtime."""
        return self._tool_runtime

    @property
    def config(self) -> AgentConfig:
        """The agent configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _provider_type(self) -> str:
        """Derive the provider type from the provider's own metadata.

        Provider identity is obtained from ``provider_info.metadata.name``.
        Recognised values: ``"claude"``, otherwise ``"openai"`` (default).
        This avoids storing provider-specific strings in ``AgentConfig``.
        """
        name = self._provider.provider_info.metadata.name.lower()
        if "claude" in name:
            return "claude"
        return "openai"

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(
        self,
        messages: list[ProviderMessage],
        config: AgentConfig,
        system: str,
        kwargs: dict[str, Any],
    ) -> ProviderResponse:
        """Core reasoning loop."""
        last_response: ProviderResponse | None = None

        for iteration in range(config.max_iterations):
            self._iteration_count = iteration + 1

            # 1. Call the provider
            response = await self._call_provider(messages, system, kwargs)
            last_response = response

            # 2. Extract tool calls
            tool_calls = extract_tool_calls(response)

            # 3. No tool calls → final response
            if not tool_calls:
                return response

            # 4. Check tool call limit before executing
            if self._tool_calls_executed + len(tool_calls) > config.max_tool_calls:
                raise ToolCallLimitExceeded(
                    max_tool_calls=config.max_tool_calls,
                    details={
                        "iteration": iteration + 1,
                        "tool_calls_executed": self._tool_calls_executed,
                        "pending_tool_calls": len(tool_calls),
                    },
                )

            # 5. Append assistant message with tool calls to conversation
            assistant_msg = ProviderMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)

            # 6. Execute tool calls and format results
            tool_messages = await execute_tool_calls(
                tool_calls,
                self._tool_runtime,
                provider_type=self._provider_type,
            )

            # 7. Append tool results to conversation
            messages.extend(tool_messages)
            self._tool_calls_executed += len(tool_calls)

        # Iteration limit reached — delegate to helper
        return self._handle_iteration_limit(config, last_response)

    def _handle_iteration_limit(
        self,
        config: AgentConfig,
        last_response: ProviderResponse | None,
    ) -> ProviderResponse:
        """Handle the iteration-limit condition.

        When ``raise_on_iteration_limit`` is ``True`` (the default),
        an exception is always raised.

        When it is ``False``:
        - If ``return_partial_response`` is ``True`` AND a real provider
          response exists, that response is returned with metadata flags.
        - Otherwise an exception is still raised.

        This method consolidates three code paths that previously existed
        inline in the loop.
        """
        if config.raise_on_iteration_limit:
            raise IterationLimitExceeded(
                max_iterations=config.max_iterations,
                details={
                    "iteration_count": self._iteration_count,
                    "tool_calls_executed": self._tool_calls_executed,
                },
            )

        if config.return_partial_response and last_response is not None:
            # Preserve the real provider output, augment with metadata
            meta = dict(last_response.metadata)
            meta.update({
                "partial": True,
                "iteration_count": self._iteration_count,
                "tool_calls_executed": self._tool_calls_executed,
            })
            return ProviderResponse(
                content=last_response.content,
                message=last_response.message,
                stop_reason=last_response.stop_reason,
                usage=last_response.usage,
                tool_calls=last_response.tool_calls,
                metadata=meta,
            )

        raise IterationLimitExceeded(
            max_iterations=config.max_iterations,
            details={
                "iteration_count": self._iteration_count,
                "tool_calls_executed": self._tool_calls_executed,
                "reason": "return_partial_response is False or no last_response",
            },
        )

    async def _call_provider(
        self,
        messages: list[ProviderMessage],
        system: str,
        kwargs: dict[str, Any],
    ) -> ProviderResponse:
        """Make a single provider request.

        Assumes *kwargs* is a local copy owned by the runtime (already
        copied in ``run()``), so mutating it is safe.
        """
        self._provider_requests += 1

        # Pop 'tools' from the local copy — safe because we own kwargs
        tools_list = kwargs.pop("tools", [])

        request = ProviderRequest(
            messages=messages,
            system=system,
            tools=tools_list,
            **kwargs,
        )

        try:
            return await self._provider.generate(request)
        except Exception as exc:
            raise ProviderExecutionError(
                message=f"Provider execution failed at iteration {self._iteration_count}: {exc}",
                original_exception=exc if isinstance(exc, Exception) else None,
                details={
                    "iteration": self._iteration_count,
                    "provider_requests": self._provider_requests,
                },
            ) from exc
