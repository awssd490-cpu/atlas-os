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
from collections.abc import AsyncIterator
from typing import Any

from app.agent.config import AgentConfig
from app.agent.errors import (
    IterationLimitExceeded,
    ProviderExecutionError,
    ToolCallLimitExceeded,
)
from app.agent.events import (
    AgentCompletedEvent,
    AgentEvent,
    AgentEventDispatcher,
    AgentFailedEvent,
    AgentStartedEvent,
    IterationCompletedEvent,
    IterationStartedEvent,
    MemoryInjectionCompletedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
    PlanCompletedEvent,
    PlanCreatedEvent,
    PlanStepCompletedEvent,
    PlanStepFailedEvent,
    PlanStepStartedEvent,
    ProviderChunkReceivedEvent,
    ProviderRequestStartedEvent,
    ProviderResponseReceivedEvent,
    ProviderStreamCompletedEvent,
    ProviderStreamFailedEvent,
    ProviderStreamStartedEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionStartedEvent,
)
from app.agent.memory import MemoryContextBuilder
from app.agent.plan import Plan, PlanResult, PlanStep, StepStatus
from app.agent.planner import PlanningEngine
from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    Role,
    StopReason,
    StreamingChunk,
)
from app.provider.streaming import ProviderStreamResult, aggregate_stream
from app.provider.provider import Provider
from app.tools.integration import execute_tool_calls, extract_tool_calls
from app.tools.runtime import ToolRuntime


class AgentRuntime:
    """Orchestrates the reasoning loop between a provider and tools.

    The runtime sends messages to the provider, extracts tool calls from
    responses, executes them via ``ToolRuntime``, formats the results
    back into provider messages, and repeats until the provider returns
    a final response (one without tool calls).

    The runtime optionally accepts an ``AgentEventDispatcher`` for
    observing execution in real time.  It also supports async iteration
    via ``stream()``.

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
        dispatcher: AgentEventDispatcher | None = None,
        memory: MemorySearchService | None = None,
    ) -> None:
        self._provider = provider
        self._tool_runtime = tool_runtime
        self._config = config or AgentConfig.default()
        self._dispatcher = dispatcher or AgentEventDispatcher()
        self._memory_builder = MemoryContextBuilder(memory)
        self._planner = PlanningEngine()

        # Internal state (reset per run)
        self._iteration_count: int = 0
        self._tool_calls_executed: int = 0
        self._provider_requests: int = 0
        self._start_time: float = 0.0
        self._elapsed_time: float = 0.0
        self._last_stream_response: ProviderResponse | None = None

    # ------------------------------------------------------------------
    # Public API — non-streaming
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[ProviderMessage],
        *,
        config: AgentConfig | None = None,
        system: str = "",
        **request_kwargs: Any,
    ) -> ProviderResponse:
        """Run the reasoning loop (non-streaming).

        Sends *messages* (and optional *system* prompt) to the provider,
        then iterates: extract tool calls → execute tools → send results
        back to the provider — until the provider returns a response with
        no tool calls.

        The caller's ``request_kwargs`` dict is never mutated.

        If a dispatcher was provided at construction time, events are
        emitted during execution.

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
            await self._emit(AgentStartedEvent())
            return await self._loop(active_messages, resolved_config, system, kwargs)
        except Exception as exc:
            await self._emit(
                AgentFailedEvent(
                    error=str(exc),
                    error_type=type(exc).__name__,
                    iteration=self._iteration_count,
                )
            )
            raise
        finally:
            self._elapsed_time = time.monotonic() - self._start_time
            self._last_stream_response = None

    # ------------------------------------------------------------------
    # Public API — streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[ProviderMessage],
        *,
        config: AgentConfig | None = None,
        system: str = "",
        **request_kwargs: Any,
    ) -> AsyncIterator[AgentEvent | ProviderResponse]:
        """Run the reasoning loop as an async event stream.

        Yields ``AgentEvent`` objects as the loop progresses, then
        yields the final ``ProviderResponse``.

        Args:
            messages: The initial conversation messages.
            config: Optional per-run configuration override.
            system: Optional system prompt.
            **request_kwargs: Additional keyword arguments forwarded to
                ``ProviderRequest``.

        Yields:
            ``AgentEvent`` objects during execution, then the final
            ``ProviderResponse`` at completion.
        """
        resolved_config = config or self._config
        active_messages = list(messages)
        kwargs = dict(request_kwargs)

        self._iteration_count = 0
        self._tool_calls_executed = 0
        self._provider_requests = 0
        self._start_time = time.monotonic()
        self._elapsed_time = 0.0
        self._last_stream_response = None

        try:
            async for event in self._stream_inner(
                active_messages, resolved_config, system, kwargs,
            ):
                yield event
        except Exception as exc:
            yield AgentFailedEvent(
                error=str(exc),
                error_type=type(exc).__name__,
                iteration=self._iteration_count,
            )
            raise
        finally:
            self._elapsed_time = time.monotonic() - self._start_time

    async def _stream_inner(
        self,
        messages: list[ProviderMessage],
        config: AgentConfig,
        system: str,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[AgentEvent | ProviderResponse]:
        """Internal streaming coroutine.

        Yields every event plus the final response.
        """
        yield AgentStartedEvent()
        await self._emit(AgentStartedEvent())

        last_response: ProviderResponse | None = None

        # Initialize plan
        plan = self._create_and_emit_plan(config)

        for iteration in range(config.max_iterations):
            self._iteration_count = iteration + 1

            # Start plan step
            plan = self._start_plan_step(plan, config)

            event = IterationStartedEvent(iteration=self._iteration_count)
            yield event
            await self._emit(event)

            # 1. Retrieve memories and inject context
            memory_context = await self._retrieve_and_inject_memories(
                messages, system, config, True,  # streaming mode — yields events
            )
            if memory_context is not None:
                augmented_messages, augmented_system = memory_context
                # Only yield events in streaming mode; injection events
                # are already yielded by _retrieve_and_inject_memories
            else:
                augmented_messages = messages
                augmented_system = system

            # 2. Call the provider (streaming if enabled)
            event = ProviderRequestStartedEvent(
                iteration=self._iteration_count,
                message_count=len(augmented_messages),
            )
            yield event
            await self._emit(event)

            stream_result = await self._call_provider_stream(
                augmented_messages, augmented_system, kwargs,
            )

            if hasattr(stream_result, "__aiter__"):
                # Streaming mode — stream_result is an AsyncIterator
                from app.provider.streaming import aggregate_stream

                raw_stream = stream_result

                await self._emit(
                    ProviderStreamStartedEvent(
                        iteration=self._iteration_count,
                        stream_type="provider",
                        message_count=len(augmented_messages),
                    )
                )

                chunk_index = 0
                aggregated = await aggregate_stream(raw_stream)

                content = aggregated.content
                response = ProviderResponse(
                    content=content,
                    message=ProviderMessage(role=Role.ASSISTANT, content=content),
                    stop_reason=StopReason.STOP,
                    tool_calls=aggregated.tool_calls,
                )

                await self._emit(
                    ProviderStreamCompletedEvent(
                        iteration=self._iteration_count,
                        total_chunks=chunk_index,
                        total_content_length=len(content),
                    )
                )

                if aggregated.tool_calls:
                    response = ProviderResponse(
                        content=content,
                        message=ProviderMessage(
                            role=Role.ASSISTANT,
                            content=content,
                            tool_calls=aggregated.tool_calls,
                        ),
                        stop_reason=StopReason.TOOL_CALL,
                        tool_calls=aggregated.tool_calls,
                    )
            else:
                # Non-streaming mode
                response = stream_result

            last_response = response

            event = ProviderResponseReceivedEvent(
                iteration=self._iteration_count,
                content_length=len(response.content),
                tool_call_count=len(response.tool_calls),
            )
            yield event
            await self._emit(event)

            # 3. Extract tool calls
            tool_calls = extract_tool_calls(response)

            # 4. No tool calls → final response
            if not tool_calls:
                # Complete the current plan step
                plan = self._complete_plan_step(plan, config)
                # Finish the plan
                plan_result = self._finish_plan(plan, config)
                self._emit_plan_completed(plan_result)

                event = AgentCompletedEvent(
                    iteration=self._iteration_count,
                    total_iterations=self._iteration_count,
                    total_tool_calls=self._tool_calls_executed,
                    total_provider_requests=self._provider_requests,
                )
                yield event
                await self._emit(event)
                self._last_stream_response = response
                yield response
                return

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
            for tc in tool_calls:
                event = ToolExecutionStartedEvent(
                    iteration=self._iteration_count,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                )
                yield event
                await self._emit(event)

            tool_messages = await execute_tool_calls(
                tool_calls,
                self._tool_runtime,
                provider_type=self._provider_type,
            )

            for i, msg in enumerate(tool_messages):
                tc = tool_calls[i] if i < len(tool_calls) else None
                event = ToolExecutionFinishedEvent(
                    iteration=self._iteration_count,
                    tool_name=tc.name if tc else "",
                    tool_call_id=tc.id if tc else "",
                    success=(msg.content != ""),
                )
                yield event
                await self._emit(event)

            # 7. Append tool results to conversation
            messages.extend(tool_messages)
            self._tool_calls_executed += len(tool_calls)

            # Complete the current plan step
            plan = self._complete_plan_step(plan, config)

            event = IterationCompletedEvent(
                iteration=self._iteration_count,
                tool_calls_in_iteration=len(tool_calls),
            )
            yield event
            await self._emit(event)

        # Iteration limit reached
        result = self._handle_iteration_limit(config, last_response)
        if config.planning_enabled:
            plan_result = self._planner.finish_plan(plan, failed=False)
            self._emit_plan_completed(plan_result)

        event = AgentCompletedEvent(
            iteration=self._iteration_count,
            total_iterations=self._iteration_count,
            total_tool_calls=self._tool_calls_executed,
            total_provider_requests=self._provider_requests,
        )
        yield event
        await self._emit(event)

        self._last_stream_response = result
        yield result

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

    @property
    def dispatcher(self) -> AgentEventDispatcher:
        """The event dispatcher."""
        return self._dispatcher

    @property
    def current_plan(self) -> Plan | None:
        """The current execution plan, if planning is enabled.

        Returns ``None`` if planning is disabled or no run has started.
        """
        if not self._config.planning_enabled:
            return None
        return self._planner.current_plan

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

    async def _emit(self, event: AgentEvent) -> None:
        """Emit an event to the registered dispatcher, if any."""
        if self._dispatcher is not None:
            await self._dispatcher.emit(event)

    async def _retrieve_and_inject_memories(
        self,
        messages: list[ProviderMessage],
        system: str,
        config: AgentConfig,
        streaming: bool = False,
    ) -> tuple[list[ProviderMessage], str] | None:
        """Retrieve relevant memories and augment the request context.

        If memory is disabled or no memory service is configured, returns
        ``None`` and the caller should use the original ``messages`` and
        ``system`` unchanged.

        In streaming mode (``streaming=True``), memory-related events
        are yielded by this method — but since we are a coroutine called
        from within an async generator, events are emitted via the
        dispatcher instead.

        Args:
            messages: The current conversation messages.
            system: The current system prompt.
            config: The agent config.
            streaming: Whether we are in streaming mode.

        Returns:
            ``(augmented_messages, augmented_system)`` if memories were
            injected, or ``None`` if no injection was performed.
        """
        if not config.memory_enabled:
            return None

        memory_service = self._memory_builder
        if memory_service is None:
            return None

        # Build query from the last user message
        query_text = ""
        for msg in reversed(messages):
            role_str = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            if role_str == "user":
                query_text = msg.content
                break

        await self._emit(
            MemoryRetrievalStartedEvent(
                iteration=self._iteration_count,
                query=query_text,
            )
        )

        memories = await memory_service.retrieve(
            query=query_text,
            config=config,
        )

        await self._emit(
            MemoryRetrievalCompletedEvent(
                iteration=self._iteration_count,
                memory_count=len(memories),
                query=query_text,
            )
        )

        if not memories:
            return None

        # Format memories and inject
        formatted = memory_service.format_as_text(memories)
        _inject_method = config.inject_memory_as
        method = config.inject_memory_as

        if method == "system":
            # Prepend to system prompt
            augmented_system = f"{system}\n\n{formatted}" if system else formatted
            augmented_messages = list(messages)

        elif method == "assistant":
            # Append as an assistant message
            augmented_system = system
            augmented_messages = list(messages)
            augmented_messages.append(
                ProviderMessage(
                    role="assistant",
                    content=f"*Context from memory:*\n{formatted}",
                )
            )

        elif method == "hidden_context":
            # Store in request metadata (handled by provider)
            augmented_system = system
            augmented_messages = list(messages)

        else:
            return None

        await self._emit(
            MemoryInjectionCompletedEvent(
                iteration=self._iteration_count,
                method=method,
                metadata={"memory_count": len(memories)},
            )
        )

        return augmented_messages, augmented_system

    # ------------------------------------------------------------------
    # Plan helpers
    # ------------------------------------------------------------------

    def _create_and_emit_plan(self, config: AgentConfig) -> Plan | None:
        """Create a plan and emit PlanCreatedEvent.

        Returns ``None`` if planning is disabled.
        """
        if not config.planning_enabled:
            return None

        title = f"Agent reasoning ({config.max_iterations} iterations max)"
        steps = tuple(
            PlanStep(
                id=str(__import__("uuid").uuid4())[:8],
                title=f"Iteration {i + 1}",
                status=StepStatus.PENDING,
            )
            for i in range(min(config.max_iterations, config.max_plan_steps))
        )
        # Insert as list
        plan = self._planner.create_plan(
            title=title,
            steps=list(steps),
        )
        self._emit_async(
            PlanCreatedEvent(
                iteration=self._iteration_count,
                plan_id=plan.id,
                step_count=plan.step_count,
            )
        )
        return plan

    def _start_plan_step(
        self,
        plan: Plan | None,
        config: AgentConfig,
    ) -> Plan | None:
        """Start the next plan step and emit PlanStepStartedEvent.

        Returns the updated plan (or ``None`` if planning is disabled).
        """
        if plan is None or not config.planning_enabled:
            return None

        next_step = self._planner.get_next_step(plan)
        if next_step is None:
            return plan

        updated = self._planner.start_step(plan, next_step.id)
        self._emit_async(
            PlanStepStartedEvent(
                iteration=self._iteration_count,
                step_id=next_step.id,
                step_title=next_step.title,
            )
        )
        return updated

    def _complete_plan_step(
        self,
        plan: Plan | None,
        config: AgentConfig,
    ) -> Plan | None:
        """Complete the current plan step and emit PlanStepCompletedEvent.

        Returns the updated plan (or ``None`` if planning is disabled).
        """
        if plan is None or not config.planning_enabled:
            return None
        if self._planner.current_step_id is None:
            return plan

        step_id = self._planner.current_step_id
        step = next((s for s in plan.steps if s.id == step_id), None)
        updated = self._planner.complete_step(plan, step_id)
        self._emit_async(
            PlanStepCompletedEvent(
                iteration=self._iteration_count,
                step_id=step_id,
                step_title=step.title if step else "",
            )
        )
        return updated

    def _finish_plan(
        self,
        plan: Plan | None,
        config: AgentConfig,
        *,
        failed: bool = False,
    ) -> PlanResult | None:
        """Finish the plan and return the result.

        Returns ``None`` if planning is disabled.
        """
        if plan is None or not config.planning_enabled:
            return None
        return self._planner.finish_plan(plan, failed=failed)

    def _emit_plan_completed(self, plan_result: PlanResult | None) -> None:
        """Emit PlanCompletedEvent for the given result."""
        if plan_result is None:
            return
        self._emit_async(
            PlanCompletedEvent(
                iteration=self._iteration_count,
                plan_id=plan_result.plan.id,
                completed=plan_result.completed,
                total_steps=plan_result.total_steps,
                completed_steps=plan_result.completed_steps,
                failed_steps=plan_result.failed_steps,
            )
        )

    def _emit_async(self, event: AgentEvent) -> None:
        """Fire-and-forget emit via the dispatcher.

        This is a last-resort helper for synchronous methods that need
        to emit events without awaiting.  Prefer ``await _emit()`` in
        async contexts.
        """
        import asyncio
        if self._dispatcher is not None:
            # Must await in the event loop — schedule it
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._dispatcher.emit(event))
            except RuntimeError:
                pass

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
        """Core reasoning loop (non-streaming).

        Emits events via the dispatcher but does not yield them.
        """
        last_response: ProviderResponse | None = None

        # Initialize plan
        plan = self._create_and_emit_plan(config)

        for iteration in range(config.max_iterations):
            self._iteration_count = iteration + 1

            # Start plan step
            plan = self._start_plan_step(plan, config)

            await self._emit(IterationStartedEvent(iteration=self._iteration_count))

            # 1. Retrieve memories and inject context
            memory_context = await self._retrieve_and_inject_memories(
                messages, system, config, False,  # non-streaming — no yield
            )
            if memory_context is not None:
                augmented_messages, augmented_system = memory_context
            else:
                augmented_messages = messages
                augmented_system = system

            # 2. Call the provider
            await self._emit(
                ProviderRequestStartedEvent(
                    iteration=self._iteration_count,
                    message_count=len(augmented_messages),
                )
            )
            response = await self._call_provider(augmented_messages, augmented_system, kwargs)
            last_response = response

            await self._emit(
                ProviderResponseReceivedEvent(
                    iteration=self._iteration_count,
                    content_length=len(response.content),
                    tool_call_count=len(response.tool_calls),
                )
            )

            # 3. Extract tool calls
            tool_calls = extract_tool_calls(response)

            # 4. No tool calls → final response
            if not tool_calls:
                # Complete the current plan step
                plan = self._complete_plan_step(plan, config)
                # Finish the plan
                plan_result = self._finish_plan(plan, config)
                self._emit_plan_completed(plan_result)

                await self._emit(
                    AgentCompletedEvent(
                        iteration=self._iteration_count,
                        total_iterations=self._iteration_count,
                        total_tool_calls=self._tool_calls_executed,
                        total_provider_requests=self._provider_requests,
                    )
                )
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
            for tc in tool_calls:
                await self._emit(
                    ToolExecutionStartedEvent(
                        iteration=self._iteration_count,
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                    )
                )

            tool_messages = await execute_tool_calls(
                tool_calls,
                self._tool_runtime,
                provider_type=self._provider_type,
            )

            for i, msg in enumerate(tool_messages):
                tc = tool_calls[i] if i < len(tool_calls) else None
                await self._emit(
                    ToolExecutionFinishedEvent(
                        iteration=self._iteration_count,
                        tool_name=tc.name if tc else "",
                        tool_call_id=tc.id if tc else "",
                        success=(msg.content != ""),
                    )
                )

            # 7. Append tool results to conversation
            messages.extend(tool_messages)
            self._tool_calls_executed += len(tool_calls)

            # Complete the current plan step
            plan = self._complete_plan_step(plan, config)

            await self._emit(
                IterationCompletedEvent(
                    iteration=self._iteration_count,
                    tool_calls_in_iteration=len(tool_calls),
                )
            )

        # Iteration limit reached — delegate to helper
        result = self._handle_iteration_limit(config, last_response)
        # Finish the plan as completed (partial response is still a completion)
        if config.planning_enabled:
            plan_result = self._planner.finish_plan(plan, failed=False)
            self._emit_plan_completed(plan_result)

        await self._emit(
            AgentCompletedEvent(
                iteration=self._iteration_count,
                total_iterations=self._iteration_count,
                total_tool_calls=self._tool_calls_executed,
                total_provider_requests=self._provider_requests,
            )
        )
        return result

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
        """Make a single provider request (non-streaming).

        Assumes *kwargs* is a local copy owned by the runtime (already
        copied in ``run()``), so mutating it is safe.
        """
        self._provider_requests += 1

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

    async def _call_provider_stream(
        self,
        messages: list[ProviderMessage],
        system: str,
        kwargs: dict[str, Any],
    ) -> Any:
        """Make a streaming provider request.

        Returns an ``AsyncIterator[StreamingChunk]`` if streaming was
        started, or a ``ProviderResponse`` if the provider doesn't
        support streaming (fallback to ``generate()``).

        Assumes *kwargs* is a local copy owned by the runtime.
        """
        self._provider_requests += 1

        tools_list = kwargs.pop("tools", [])

        request = ProviderRequest(
            messages=messages,
            system=system,
            tools=tools_list,
            **kwargs,
        )

        try:
            # Use the provider's stream method
            raw_stream = self._provider.stream(request)
            return raw_stream
        except Exception:
            # Fallback to non-streaming
            try:
                response = await self._provider.generate(request)
                return response
            except Exception as exc:
                raise ProviderExecutionError(
                    message=f"Provider execution failed at iteration {self._iteration_count}: {exc}",
                    original_exception=exc if isinstance(exc, Exception) else None,
                    details={
                        "iteration": self._iteration_count,
                        "provider_requests": self._provider_requests,
                    },
                ) from exc
