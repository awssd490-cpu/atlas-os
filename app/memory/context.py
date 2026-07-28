"""Context Builder — memory-to-context orchestration layer.

The Context Builder is the bridge between Phase 3 memory subsystems
and the LLM.  It composes retrieval, compression, and snapshots into
a structured, model-agnostic ``ContextPackage``.

Every stage is independently replaceable.  No provider-specific
formatting exists in this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.memory.budget import BudgetConfig, BudgetResult, TokenBudgetManager
from app.memory.compression import CompressionService
from app.memory.tokens import TokenEstimator
from app.memory.interfaces import MemoryGraph
from app.memory.manager import MemoryManager
from app.memory.memory import Memory, MemoryState
from app.memory.retrieval import (
    RetrievalPipeline,
    RetrievalQuery,
    RetrievalResult,
    default_pipeline,
)
from app.memory.selection import MemorySelectionEngine, SelectionConfig, SelectionResult
from app.memory.snapshots import SnapshotService


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextSource:
    """Provenance — where a piece of context came from."""

    source_type: str  # "memory", "relationship", "snapshot"
    memory_id: str = ""
    namespace: str = ""
    memory_type: str = ""
    importance: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class ContextSection:
    """A named section in the context package.

    Each section has a type, a list of memory entries, and metadata.
    Sections are ordered by priority (highest first).
    """

    section_type: str  # "user_query", "relevant_memories", "related_memories",
                       # "long_term_facts", "conversation_history",
                       # "working_memory", "retrieved_metadata"
    label: str = ""
    memories: list[Memory] = field(default_factory=list)
    sources: list[ContextSource] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    @property
    def memory_count(self) -> int:
        return len(self.memories)


@dataclass(frozen=True)
class ContextStatistics:
    """Aggregated statistics for a context assembly run."""

    total_memories: int = 0
    total_sections: int = 0
    total_tokens: int = 0
    retrieval_ms: float = 0.0
    compression_ms: float = 0.0
    assembly_ms: float = 0.0
    estimated_tokens: int = 0
    sources_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextPackage:
    """The final, model-agnostic context output.

    This is what gets handed to a Provider for LLM formatting.
    It contains everything the LLM needs and nothing else.
    """

    request_id: str = ""
    sections: list[ContextSection] = field(default_factory=list)
    statistics: ContextStatistics = field(default_factory=ContextStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_memories(self) -> int:
        return sum(s.memory_count for s in self.sections)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.sections)

    @property
    def section_types(self) -> list[str]:
        return [s.section_type for s in self.sections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sections": [
                {
                    "section_type": s.section_type,
                    "label": s.label,
                    "memory_count": s.memory_count,
                    "token_count": s.token_count,
                    "sources": [{"source_type": src.source_type, "memory_id": src.memory_id}
                                for src in s.sources],
                }
                for s in self.sections
            ],
            "statistics": {
                "total_memories": self.statistics.total_memories,
                "total_sections": self.statistics.total_sections,
                "total_tokens": self.statistics.total_tokens,
                "retrieval_ms": self.statistics.retrieval_ms,
                "compression_ms": self.statistics.compression_ms,
                "assembly_ms": self.statistics.assembly_ms,
                "estimated_tokens": self.statistics.estimated_tokens,
                "sources_breakdown": dict(self.statistics.sources_breakdown),
            },
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ContextBuilderConfig:
    """Configuration for the Context Builder."""

    max_tokens: int = 4096
    max_memories_per_section: int = 50
    enable_relationship_expansion: bool = True
    enable_compression: bool = True
    enable_snapshot_awareness: bool = False
    enable_selection: bool = True
    section_order: list[str] = field(default_factory=lambda: [
        "user_query",
        "working_memory",
        "relevant_memories",
        "related_memories",
        "long_term_facts",
        "conversation_history",
        "retrieved_metadata",
    ])
    default_retrieval_limit: int = 100
    working_memory_namespace: str = "working"
    long_term_namespaces: list[str] = field(default_factory=lambda: ["long_term", "semantic"])
    conversation_namespace: str = "conversation"


# ---------------------------------------------------------------------------
# ContextBuilder (primary interface)
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Primary interface for context assembly.

    Composes Phase 3 subsystems into a structured ContextPackage.
    Every stage is independently replaceable.

    Usage::

        builder = ContextBuilder(memory_manager, config=...)
        package = await builder.build(request)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        pipeline: RetrievalPipeline | None = None,
        config: ContextBuilderConfig | None = None,
        selection_engine: MemorySelectionEngine | None = None,
        budget_manager: TokenBudgetManager | None = None,
    ) -> None:
        self._memory_manager = memory_manager
        self._pipeline = pipeline
        self._config = config or ContextBuilderConfig()
        self._selection_engine = selection_engine
        self._budget_manager = budget_manager

    @property
    def config(self) -> ContextBuilderConfig:
        return self._config

    @property
    def selection_engine(self) -> MemorySelectionEngine | None:
        """Access the memory selection engine."""
        return self._selection_engine

    @property
    def budget_manager(self) -> TokenBudgetManager | None:
        """Access the token budget manager."""
        return self._budget_manager

    async def build(
        self,
        *,
        request_id: str = "",
        query: RetrievalQuery | None = None,
        user_content: str = "",
        namespaces: list[str] | None = None,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextPackage:
        """Assemble a context package from the given parameters.

        Args:
            request_id: Optional tracing identifier.
            query: Pre-built RetrievalQuery (overrides individual params).
            user_content: Raw user input text.
            namespaces: Filter to these namespaces.
            memory_types: Filter to these memory types.
            tags: Filter to these tags.
            max_tokens: Override the default token budget.
            metadata: Extra metadata to include in the package.

        Returns:
            A fully assembled ContextPackage.
        """
        start_time = time.monotonic()
        cfg = self._config

        resolved_query = query or RetrievalQuery(
            namespaces=namespaces,
            memory_types=memory_types,
            tags=tags,
            expand_relationships=cfg.enable_relationship_expansion,
            limit=cfg.default_retrieval_limit,
        )

        sections: list[ContextSection] = []
        total_retrieval_ms = 0.0
        total_compression_ms = 0.0

        # ------------------------------------------------------------------
        # Phase 1: User query section
        # ------------------------------------------------------------------
        if user_content:
            query_memory = Memory(
                content=user_content,
                memory_type="query",
                namespace="query",
                importance=1.0,
                source="user",
            )
            sections.append(ContextSection(
                section_type="user_query",
                label="User Input",
                memories=[query_memory],
                sources=[ContextSource(source_type="query", memory_id=query_memory.id.value)],
                token_count=TokenEstimator.estimate_memory(query_memory),
            ))

        # ------------------------------------------------------------------
        # Phase 2: Working memory
        # ------------------------------------------------------------------
        if cfg.working_memory_namespace:
            working_result = await self._retrieve_with_metadata(
                RetrievalQuery(
                    namespaces=[cfg.working_memory_namespace],
                    limit=cfg.max_memories_per_section,
                ),
            )
            total_retrieval_ms += working_result["elapsed_ms"]
            if working_result["result"].memories:
                sections.append(self._build_section(
                    "working_memory",
                    "Working Memory",
                    working_result["result"],
                ))

        # ------------------------------------------------------------------
        # Phase 3: Primary retrieval (relevant memories)
        # ------------------------------------------------------------------
        primary_result = await self._retrieve_with_metadata(resolved_query)
        total_retrieval_ms += primary_result["elapsed_ms"]
        if primary_result["result"].memories:
            sections.append(self._build_section(
                "relevant_memories",
                "Relevant Memories",
                primary_result["result"],
            ))

        # ------------------------------------------------------------------
        # Phase 4: Long-term facts
        # ------------------------------------------------------------------
        if cfg.long_term_namespaces:
            lt_query = RetrievalQuery(
                namespaces=cfg.long_term_namespaces,
                limit=cfg.max_memories_per_section,
                expand_relationships=cfg.enable_relationship_expansion,
                min_importance=0.5,
            )
            lt_result = await self._retrieve_with_metadata(lt_query)
            total_retrieval_ms += lt_result["elapsed_ms"]
            if lt_result["result"].memories:
                sections.append(self._build_section(
                    "long_term_facts",
                    "Long-Term Facts",
                    lt_result["result"],
                ))

        # ------------------------------------------------------------------
        # Phase 5: Conversation history
        # ------------------------------------------------------------------
        if cfg.conversation_namespace:
            conv_query = RetrievalQuery(
                namespaces=[cfg.conversation_namespace],
                limit=cfg.max_memories_per_section,
            )
            conv_result = await self._retrieve_with_metadata(conv_query)
            total_retrieval_ms += conv_result["elapsed_ms"]
            if conv_result["result"].memories:
                sections.append(self._build_section(
                    "conversation_history",
                    "Conversation History",
                    conv_result["result"],
                ))

        # ------------------------------------------------------------------
        # Phase 6: Related memories (from relationship expansion)
        # ------------------------------------------------------------------
        if cfg.enable_relationship_expansion:
            # Use the graph to find related memories for the primary results
            graph = self._memory_manager.graph
            primary_memories = primary_result["result"].memories
            related_result = await self._expand_related(primary_memories, graph)
            total_retrieval_ms += related_result.get("elapsed_ms", 0)
            if related_result["memories"]:
                sections.append(self._build_section_from_list(
                    "related_memories",
                    "Related Memories",
                    related_result["memories"],
                ))

        # ------------------------------------------------------------------
        # Phase 7: Apply memory selection to each section
        # ------------------------------------------------------------------
        if cfg.enable_selection and self._selection_engine is not None:
            selected_sections: list[ContextSection] = []
            for section in sections:
                if not section.memories:
                    selected_sections.append(section)
                    continue
                result = await self._selection_engine.select(
                    section.memories,
                    max_memories=cfg.max_memories_per_section,
                )
                if result.selected:
                    selected_sections.append(ContextSection(
                        section_type=section.section_type,
                        label=section.label,
                        memories=result.selected,
                        sources=section.sources,
                        metadata={
                            **section.metadata,
                            "selection_rejected": len(result.rejected),
                            "selection_stats": result.statistics,
                        },
                        token_count=TokenEstimator.estimate_memories(result.selected),
                    ))
                else:
                    selected_sections.append(section)
            sections = selected_sections

        # ------------------------------------------------------------------
        # Phase 8: Apply compression to relevant sections
        # ------------------------------------------------------------------
        if cfg.enable_compression:
            compressor = self._memory_manager.compressor
            if compressor is not None:
                compressed_sections: list[ContextSection] = []
                for section in sections:
                    if section.memories:
                        try:
                            comp_result = await compressor.compress(
                                section.memories,
                                strategy="dedup",
                                target_count=cfg.max_memories_per_section,
                            )
                            if comp_result.compressed_count < len(section.memories):
                                compressed_sections.append(ContextSection(
                                    section_type=section.section_type,
                                    label=section.label + " (compressed)",
                                    memories=comp_result.compressed,
                                    metadata={
                                        **section.metadata,
                                        "compression_ratio": comp_result.ratio,
                                        "original_count": comp_result.original_count,
                                    },
                                    token_count=TokenEstimator.estimate_memories(
                                        comp_result.compressed,
                                    ),
                                ))
                                total_compression_ms += 50  # estimated
                            else:
                                compressed_sections.append(section)
                        except Exception:
                            compressed_sections.append(section)
                    else:
                        compressed_sections.append(section)
                sections = compressed_sections

        # ------------------------------------------------------------------
        # Assembly: order sections and apply token budget
        # ------------------------------------------------------------------
        assembly_start = time.monotonic()

        ordered = self._order_sections(sections)
        budget = max_tokens or cfg.max_tokens

        # Use TokenBudgetManager if available, otherwise fall back to inline
        if self._budget_manager is not None:
            # Build a temporary package for the budget manager
            temp_stats = ContextStatistics(
                total_memories=sum(s.memory_count for s in ordered),
                total_sections=len(ordered),
                total_tokens=sum(s.token_count for s in ordered),
            )
            temp_package = ContextPackage(
                request_id=request_id or "",
                sections=ordered,
                statistics=temp_stats,
            )
            budget_result = await self._budget_manager.optimise(
                temp_package,
                target_budget=budget,
            )
            final_package = budget_result.package
            trimmed = final_package.sections
            used_tokens = final_package.total_tokens
            budget_decisions = budget_result.decisions
        else:
            trimmed, used_tokens = self._apply_token_budget(ordered, budget)
            budget_decisions = []

        assembly_ms = (time.monotonic() - assembly_start) * 1000

        # ------------------------------------------------------------------
        # Build statistics
        # ------------------------------------------------------------------
        source_breakdown: dict[str, int] = {}
        for section in trimmed:
            for src in section.sources:
                st = src.source_type
                source_breakdown[st] = source_breakdown.get(st, 0) + 1

        stats = ContextStatistics(
            total_memories=sum(s.memory_count for s in trimmed),
            total_sections=len(trimmed),
            total_tokens=used_tokens,
            retrieval_ms=round(total_retrieval_ms, 2),
            compression_ms=round(total_compression_ms, 2),
            assembly_ms=round(assembly_ms, 2),
            estimated_tokens=used_tokens,
            sources_breakdown=source_breakdown,
        )

        total_elapsed = (time.monotonic() - start_time) * 1000

        meta: dict[str, Any] = {
            **(metadata or {}),
            "total_elapsed_ms": round(total_elapsed, 2),
            "config_max_tokens": budget,
        }
        if budget_decisions:
            meta["budget_decisions"] = [
                {"memory_id": d.memory_id, "section_type": d.section_type, "reason": d.reason}
                for d in budget_decisions[:20]  # limit to avoid huge metadata
            ]

        return ContextPackage(
            request_id=request_id or "",
            sections=trimmed,
            statistics=stats,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retrieve_with_metadata(
        self,
        query: RetrievalQuery,
    ) -> dict[str, Any]:
        """Run the pipeline and return results + timing."""
        t0 = time.monotonic()
        pipeline = self._resolve_pipeline()
        result = await pipeline.retrieve(query)
        elapsed = (time.monotonic() - t0) * 1000
        return {"result": result, "elapsed_ms": elapsed}

    def _resolve_pipeline(self) -> RetrievalPipeline:
        """Return the configured pipeline or build the default one."""
        if self._pipeline is not None:
            return self._pipeline
        repo = getattr(self._memory_manager, "_repo", None)
        graph = self._memory_manager.graph
        if repo is None:
            raise RuntimeError("MemoryManager has no repository available")
        return default_pipeline(repo, graph=graph)

    def _build_section(
        self,
        section_type: str,
        label: str,
        result: RetrievalResult,
    ) -> ContextSection:
        """Build a ContextSection from a RetrievalResult."""
        sources = [
            ContextSource(
                source_type=section_type,
                memory_id=m.id.value,
                namespace=m.namespace,
                memory_type=m.memory_type,
                importance=m.importance,
                score=result.metadata.get("ImportanceRanker", {}).get("top_score", 0.0),
            )
            for m in result.memories
        ]
        return ContextSection(
            section_type=section_type,
            label=label,
            memories=result.memories,
            sources=sources,
            metadata=result.metadata,
            token_count=TokenEstimator.estimate_memories(result.memories),
        )

    def _build_section_from_list(
        self,
        section_type: str,
        label: str,
        memories: list[Memory],
    ) -> ContextSection:
        """Build a ContextSection from a raw memory list."""
        sources = [
            ContextSource(
                source_type=section_type,
                memory_id=m.id.value,
                namespace=m.namespace,
                memory_type=m.memory_type,
                importance=m.importance,
            )
            for m in memories
        ]
        return ContextSection(
            section_type=section_type,
            label=label,
            memories=memories,
            sources=sources,
            token_count=TokenEstimator.estimate_memories(memories),
        )

    async def _expand_related(
        self,
        memories: list[Memory],
        graph: MemoryGraph | None,
    ) -> dict[str, Any]:
        """Expand from a list of memories to find related ones."""
        if graph is None or not memories:
            return {"memories": [], "elapsed_ms": 0.0}

        t0 = time.monotonic()
        seen: set[str] = {m.id.value for m in memories}
        related: list[Memory] = []

        for mem in memories:
            neighbours = await graph.get_related(
                mem.id.value,
                direction="both",
                max_depth=1,
            )
            for nb in neighbours:
                if nb.id.value not in seen:
                    seen.add(nb.id.value)
                    related.append(nb)

        elapsed = (time.monotonic() - t0) * 1000
        return {"memories": related, "elapsed_ms": elapsed}

    def _order_sections(self, sections: list[ContextSection]) -> list[ContextSection]:
        """Sort sections according to the configured section_order."""
        order_map = {st: i for i, st in enumerate(self._config.section_order)}
        return sorted(
            sections,
            key=lambda s: order_map.get(s.section_type, 999),
        )

    @staticmethod
    def _apply_token_budget(
        sections: list[ContextSection],
        budget: int,
    ) -> tuple[list[ContextSection], int]:
        """Trim sections to fit within *budget* tokens.

        Trims from the *end* (lowest-priority sections first), and within
        a section trims memories from the end.
        """
        if not sections:
            return [], 0

        result: list[ContextSection] = []
        used = 0

        for section in sections:
            new_memories: list[Memory] = []
            new_sources: list[ContextSource] = []
            section_used = 0

            for mem, src in zip(section.memories, section.sources or []):
                mem_tokens = TokenEstimator.estimate_memory(mem)
                if used + section_used + mem_tokens > budget:
                    break
                new_memories.append(mem)
                new_sources.append(src) if src else new_sources.append(
                    ContextSource(source_type=section.section_type, memory_id=mem.id.value)
                )
                section_used += mem_tokens

            if new_memories:
                result.append(ContextSection(
                    section_type=section.section_type,
                    label=section.label,
                    memories=new_memories,
                    sources=new_sources,
                    metadata=section.metadata,
                    token_count=section_used,
                ))
                used += section_used

            if used >= budget:
                break

        return result, used


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
