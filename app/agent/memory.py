"""Memory integration layer for the Agent Runtime.

Provides ``MemoryContextBuilder`` — a thin orchestration helper that
calls the existing ``MemorySearchService`` to retrieve relevant memories
and formats them for injection into provider requests.

This module does NOT:
- Store memories
- Rank using embeddings
- Persist data
- Know anything about providers

It is an orchestration helper only.
"""

from __future__ import annotations

from typing import Any

from app.agent.config import AgentConfig
from app.memory.interfaces import MemoryQuery, MemorySearchService
from app.memory.memory import Memory, MemoryState


class MemoryContextBuilder:
    """Retrieves relevant memories and formats them for provider injection.

    Uses the existing ``MemorySearchService`` interface for all retrieval.
    The provider never knows this layer exists.

    Usage::

        builder = MemoryContextBuilder(memory_service)
        context = await builder.retrieve(query_text, limit=5)
    """

    def __init__(self, memory_service: MemorySearchService | None = None) -> None:
        self._memory_service = memory_service

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str = "",
        *,
        limit: int = 10,
        min_importance: float = 0.0,
        config: AgentConfig | None = None,
    ) -> list[Memory]:
        """Retrieve relevant memories for *query*.

        Delegates to the existing ``MemorySearchService`` using content
        search and importance filtering.  No new retrieval logic.

        Args:
            query: The text to search for in existing memories.
            limit: Maximum number of memories to return.
            min_importance: Minimum importance score threshold.
            config: Optional ``AgentConfig`` override for memory settings.

        Returns:
            A list of matching ``Memory`` objects, ordered by importance
            descending.  Empty list if memory is disabled or no service.
        """
        if self._memory_service is None:
            return []

        resolved_limit = config.memory_limit if config else limit
        resolved_min = config.minimum_memory_score if config else min_importance

        # Use the existing MemorySearchService.search() via MemoryQuery
        memory_query = MemoryQuery(
            content_search=query if query else None,
            min_importance=resolved_min if resolved_min > 0.0 else None,
            states=[MemoryState.ACTIVE],
        )

        result = await self._memory_service.search(
            memory_query,
            pagination=None,
        )

        memories = list(result.items) if hasattr(result, "items") else list(result)

        # Apply limit
        return memories[:resolved_limit]

    async def retrieve_by_importance(
        self,
        *,
        limit: int = 10,
        min_importance: float = 0.0,
        namespace: str = "default",
    ) -> list[Memory]:
        """Retrieve the highest-importance memories from a namespace.

        Uses ``MemorySearchService.search_by_importance()``.

        Args:
            limit: Maximum memories to return.
            min_importance: Minimum importance threshold.
            namespace: Namespace to search.

        Returns:
            A list of memories ordered by importance descending.
        """
        if self._memory_service is None:
            return []

        return await self._memory_service.search_by_importance(
            namespace=namespace,
            min_importance=min_importance,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_as_text(self, memories: list[Memory]) -> str:
        """Format memories as plain text for system-prompt injection.

        Each memory is rendered as a line with its content, importance,
        and tags.

        Args:
            memories: The memories to format.

        Returns:
            A plain-text string ready for injection.
        """
        if not memories:
            return ""

        lines: list[str] = ["Relevant context:"]
        for m in memories:
            parts = [m.content]
            if m.tags:
                parts.append(f"[tags: {', '.join(m.tags)}]")
            parts.append(f"(importance: {m.importance:.2f})")
            lines.append("- " + " ".join(parts))

        return "\n".join(lines)

    def format_as_text_short(self, memories: list[Memory]) -> str:
        """Format memories as short text lines.

        Each memory is a single line with just its content.

        Args:
            memories: The memories to format.

        Returns:
            A compact text string.
        """
        if not memories:
            return ""

        lines: list[str] = ["Previous context:"]
        for m in memories:
            lines.append(f"- {m.content}")

        return "\n".join(lines)
