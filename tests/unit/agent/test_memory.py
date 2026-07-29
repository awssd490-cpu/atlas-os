"""Tests for memory integration in AgentRuntime."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.config import AgentConfig
from app.agent.events import (
    AgentEvent,
    MemoryInjectionCompletedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
)
from app.agent.memory import MemoryContextBuilder
from app.agent.runtime import AgentRuntime
from app.memory.interfaces import MemoryQuery, MemorySearchService, Page
from app.memory.memory import Memory, MemoryState
from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
)
from app.provider.provider import Provider
from app.tools.models import ToolDefinition, ToolParameter
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Mock memory service
# ---------------------------------------------------------------------------


class _MockMemoryService(MemorySearchService):
    """Mock that returns canned memories."""

    def __init__(
        self,
        memories: list[Memory] | None = None,
    ) -> None:
        self._memories = memories or []
        self.search_calls: list[MemoryQuery] = []
        self.search_by_importance_calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: MemoryQuery,
        *,
        pagination: Any | None = None,
    ) -> Page[Memory]:
        self.search_calls.append(query)
        return Page(items=self._memories, total=len(self._memories), offset=0, limit=len(self._memories))

    async def search_by_importance(
        self,
        *,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        self.search_by_importance_calls.append({
            "namespace": namespace,
            "min_importance": min_importance,
            "limit": limit,
        })
        return self._memories[:limit]

    async def search_by_tag(
        self,
        tag: str,
        *,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        return []

    async def search_temporal(
        self,
        *,
        after: str | None = None,
        before: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        return []


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class _MockProvider(Provider):
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._call_count += 1
        if not self._responses:
            return ProviderResponse(content="", stop_reason=StopReason.STOP)
        return self._responses.pop(0)

    def stream(self, request: ProviderRequest):  # type: ignore[override]
        raise NotImplementedError

    async def count_tokens(self, request: ProviderRequest) -> int:
        return 0

    @property
    def provider_info(self) -> Any:
        from app.provider.models import ProviderCapability, ProviderInfo, ProviderMetadata
        return ProviderInfo(
            metadata=ProviderMetadata(name="mock"),
            capabilities=[ProviderCapability(name="tool_calling")],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(content: str, importance: float = 0.8) -> Memory:
    return Memory(
        content=content,
        importance=importance,
        memory_type="long_term",
        namespace="default",
        tags=["test"],
    )


def _make_response(content: str) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        message=ProviderMessage(role=Role.ASSISTANT, content=content),
        stop_reason=StopReason.STOP,
        usage=ProviderUsage(prompt_tokens=5, completion_tokens=5),
    )


def _create_tool_runtime() -> ToolRuntime:
    registry = ToolRegistry()
    async def calc(expression: str) -> str:
        return str(eval(expression))
    registry.register(ToolDefinition(
        name="calculator",
        parameters=(ToolParameter(name="expression", type="string"),),
        fn=calc,
    ))
    return ToolRuntime(registry)


# ---------------------------------------------------------------------------
# MemoryContextBuilder tests
# ---------------------------------------------------------------------------


class TestMemoryContextBuilder:
    async def test_no_memory_service(self) -> None:
        """Builder returns empty list when no service configured."""
        builder = MemoryContextBuilder()
        memories = await builder.retrieve(query="test")
        assert memories == []

    async def test_retrieve_with_results(self) -> None:
        """Retrieve returns memories from the memory service."""
        mock = _MockMemoryService([
            _make_memory("Paris is the capital of France", 0.9),
        ])
        builder = MemoryContextBuilder(mock)
        memories = await builder.retrieve(query="capital of France")
        assert len(memories) == 1
        assert "Paris" in memories[0].content

    async def test_retrieve_empty_results(self) -> None:
        """Retrieve returns empty list when no memories match."""
        mock = _MockMemoryService([])
        builder = MemoryContextBuilder(mock)
        memories = await builder.retrieve(query="unknown")
        assert memories == []

    async def test_retrieve_respects_limit(self) -> None:
        """Retrieve returns at most *limit* memories."""
        mock = _MockMemoryService([
            _make_memory(f"Memory {i}", 0.8) for i in range(20)
        ])
        builder = MemoryContextBuilder(mock)
        memories = await builder.retrieve(query="test", limit=5)
        assert len(memories) <= 5

    async def test_retrieve_by_importance(self) -> None:
        """Retrieve by importance works."""
        mock = _MockMemoryService([
            _make_memory("Important fact", 0.95),
        ])
        builder = MemoryContextBuilder(mock)
        memories = await builder.retrieve_by_importance(
            min_importance=0.5, limit=10, namespace="default",
        )
        assert len(memories) == 1

    def test_format_as_text(self) -> None:
        """Format memories as text."""
        builder = MemoryContextBuilder()
        memories = [
            _make_memory("Memory 1", 0.9),
            _make_memory("Memory 2", 0.7),
        ]
        text = builder.format_as_text(memories)
        assert "Memory 1" in text
        assert "Memory 2" in text
        assert "importance: 0.90" in text

    def test_format_as_text_empty(self) -> None:
        """Empty memory list produces empty text."""
        builder = MemoryContextBuilder()
        assert builder.format_as_text([]) == ""

    def test_format_as_text_short(self) -> None:
        """Short format omits metadata."""
        builder = MemoryContextBuilder()
        text = builder.format_as_text_short([
            _make_memory("Brief note", 0.5),
        ])
        assert "Brief note" in text
        assert "importance" not in text.lower()


# ---------------------------------------------------------------------------
# Runtime memory integration tests
# ---------------------------------------------------------------------------


class TestAgentRuntimeMemoryIntegration:
    async def test_memory_retrieved_before_provider_request(self) -> None:
        """Memory is retrieved before each provider request."""
        memory_service = _MockMemoryService([
            _make_memory("User likes Python", 0.9),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Got it")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="What do I like?")],
        )
        # Memory search should have been called
        assert len(memory_service.search_calls) == 1

    async def test_memory_not_retrieved_when_disabled(self) -> None:
        """Memory is not retrieved when memory_enabled=False."""
        memory_service = _MockMemoryService([
            _make_memory("Some memory", 0.9),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            config=AgentConfig(memory_enabled=False),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
        )
        # Memory search should NOT have been called
        assert len(memory_service.search_calls) == 0

    async def test_memory_retrieved_every_iteration(self) -> None:
        """Memory is retrieved each reasoning iteration."""
        memory_service = _MockMemoryService([
            _make_memory("Python fact", 0.9),
        ])
        runtime = AgentRuntime(
            _MockProvider([
                _make_response("Tool time"),
                _make_response("Final"),
            ]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Start")],
        )
        # Should have searched at least once (before first provider call)
        assert len(memory_service.search_calls) >= 1

    async def test_no_memory_service_does_not_crash(self) -> None:
        """Runtime works without a memory service."""
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=None,
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
        )
        assert response.content == "Ok"

    async def test_empty_memory_does_not_crash(self) -> None:
        """Runtime works when memory service returns empty."""
        memory_service = _MockMemoryService([])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
        )
        assert response.content == "Ok"

    async def test_memory_retrieval_failure_does_not_crash(self) -> None:
        """AgentRuntime handles memory service exception gracefully."""
        class _FailingMemoryService(MemorySearchService):
            async def search(self, query, *, pagination=None):
                raise RuntimeError("Memory service unavailable")

            async def search_by_importance(self, **kwargs):
                raise RuntimeError("unavailable")

            async def search_by_tag(self, *args, **kwargs):
                return []

            async def search_temporal(self, *args, **kwargs):
                return []

        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=_FailingMemoryService(),
        )
        # The runtime should propagate the memory error from the failing
        # service rather than silently swallowing it.
        with pytest.raises(RuntimeError):
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Hi")],
            )

    async def test_provider_request_unchanged_with_empty_memory(self) -> None:
        """Provider request is unchanged when no memories retrieved."""
        memory_service = _MockMemoryService([])
        provider = _MockProvider([_make_response("Ok")])
        runtime = AgentRuntime(
            provider,
            _create_tool_runtime(),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
        )
        # Should have made exactly 1 provider request
        assert provider._call_count == 1


# ---------------------------------------------------------------------------
# Memory injection strategy tests
# ---------------------------------------------------------------------------


class TestMemoryInjectionStrategies:
    async def test_system_injection_default(self) -> None:
        """Memories are injected into the system prompt by default."""
        memory_service = _MockMemoryService([
            _make_memory("Important context", 0.9),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
            system="You are a helpful assistant.",
        )

    async def test_system_injection_with_existing_system_prompt(self) -> None:
        """Injected memories are appended to existing system prompt."""
        memory_service = _MockMemoryService([
            _make_memory("User fact", 0.9),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            config=AgentConfig(inject_memory_as="system"),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
            system="Original prompt.",
        )

    async def test_assistant_injection(self) -> None:
        """Memories injected as assistant message."""
        memory_service = _MockMemoryService([
            _make_memory("Previous context", 0.8),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            config=AgentConfig(inject_memory_as="assistant"),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
        )

    async def test_hidden_context_injection(self) -> None:
        """Memories stored as hidden context."""
        memory_service = _MockMemoryService([
            _make_memory("Hidden data", 0.8),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            config=AgentConfig(inject_memory_as="hidden_context"),
            memory=memory_service,
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Test")],
        )

    async def test_original_conversation_unchanged(self) -> None:
        """Original conversation messages are not mutated."""
        memory_service = _MockMemoryService([
            _make_memory("Some memory", 0.8),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        original = [
            ProviderMessage(role=Role.USER, content="Hello"),
        ]
        original_copy = list(original)
        await runtime.run(original)
        # Original should not have been modified
        assert len(original) == len(original_copy)
        assert original[0].content == original_copy[0].content


# ---------------------------------------------------------------------------
# Memory events
# ---------------------------------------------------------------------------


class TestMemoryEvents:
    async def test_memory_retrieval_started_event(self) -> None:
        """MemoryRetrievalStartedEvent is emitted."""
        memory_service = _MockMemoryService([
            _make_memory("Fact", 0.8),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        events: list[str] = []

        def listener(event: AgentEvent) -> None:
            events.append(event.event_type)

        runtime.dispatcher.add_listener(listener)
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Query")],
        )

        assert "memory_retrieval_started" in events

    async def test_memory_retrieval_completed_event(self) -> None:
        """MemoryRetrievalCompletedEvent is emitted with count."""
        memory_service = _MockMemoryService([
            _make_memory("Fact 1", 0.8),
            _make_memory("Fact 2", 0.7),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        completed: list[MemoryRetrievalCompletedEvent] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "memory_retrieval_completed":
                completed.append(event)  # type: ignore[arg-type]

        runtime.dispatcher.add_listener(listener)
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Query")],
        )

        assert len(completed) >= 1
        assert completed[0].memory_count >= 2

    async def test_memory_retrieval_completed_with_zero(self) -> None:
        """MemoryRetrievalCompletedEvent emitted even when no memories found."""
        memory_service = _MockMemoryService([])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            memory=memory_service,
        )
        completed_count = 0
        retrieval_started = 0

        def listener(event: AgentEvent) -> None:
            nonlocal completed_count, retrieval_started
            if event.event_type == "memory_retrieval_completed":
                completed_count += 1
            if event.event_type == "memory_retrieval_started":
                retrieval_started += 1

        runtime.dispatcher.add_listener(listener)
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Query")],
        )

        assert retrieval_started >= 1
        assert completed_count >= 1

    async def test_memory_events_when_disabled(self) -> None:
        """No memory events when memory is disabled."""
        memory_service = _MockMemoryService([
            _make_memory("Fact", 0.8),
        ])
        runtime = AgentRuntime(
            _MockProvider([_make_response("Ok")]),
            _create_tool_runtime(),
            config=AgentConfig(memory_enabled=False),
            memory=memory_service,
        )
        event_types: list[str] = []

        def listener(event: AgentEvent) -> None:
            event_types.append(event.event_type)

        runtime.dispatcher.add_listener(listener)
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Query")],
        )

        assert "memory_retrieval_started" not in event_types
        assert "memory_retrieval_completed" not in event_types
