"""Token Budget Manager — optimises context to fit within a token budget.

The Context Builder orchestrates.  The Token Budget Manager decides
how many tokens each section gets and which memories survive.

All budget decisions are delegable to this subsystem, keeping the
Context Builder focused on orchestration rather than trimming.

Every component is independently testable and replaceable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.memory.memory import Memory
from app.memory.tokens import TokenEstimator


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetAllocation:
    """How many tokens were allocated to a section."""

    section_type: str = ""
    requested_tokens: int = 0
    allocated_tokens: int = 0
    trimmed_memories: int = 0
    preserved_memory_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BudgetDecision:
    """Why a particular trimming decision was made."""

    memory_id: str = ""
    section_type: str = ""
    reason: str = ""
    token_savings: int = 0


@dataclass(frozen=True)
class BudgetStatistics:
    """Aggregated statistics for a budget optimisation run."""

    original_tokens: int = 0
    final_tokens: int = 0
    tokens_removed: int = 0
    memories_removed: int = 0
    sections_trimmed: int = 0
    allocations: list[BudgetAllocation] = field(default_factory=list)
    emergency_mode_used: bool = False


@dataclass(frozen=True)
class BudgetResult:
    """The output of a budget optimisation run."""

    package: Any = field(default_factory=lambda: type("_", (), {"sections": [], "total_tokens": 0, "total_memories": 0, "request_id": "", "statistics": type("_", (), {"total_memories": 0, "total_sections": 0, "total_tokens": 0, "retrieval_ms": 0.0, "compression_ms": 0.0, "assembly_ms": 0.0, "estimated_tokens": 0, "sources_breakdown": {}}), "metadata": {}, "to_dict": lambda: {}})())
    statistics: BudgetStatistics = field(default_factory=BudgetStatistics)
    decisions: list[BudgetDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.statistics.original_tokens,
            "final_tokens": self.statistics.final_tokens,
            "tokens_removed": self.statistics.tokens_removed,
            "memories_removed": self.statistics.memories_removed,
            "sections_trimmed": self.statistics.sections_trimmed,
            "emergency_mode": self.statistics.emergency_mode_used,
            "decision_count": len(self.decisions),
        }


# ---------------------------------------------------------------------------
# Allocation strategy protocol
# ---------------------------------------------------------------------------


class AllocationStrategy(Protocol):
    """Determines per-section token allocation."""

    def allocate(
        self,
        sections: list,
        total_budget: int,
        *,
        preserved_section_types: set[str] | None = None,
    ) -> dict[str, int]:
        ...


# ---------------------------------------------------------------------------
# Built-in allocation strategies
# ---------------------------------------------------------------------------


class PriorityFirstAllocation:
    """Allocate budget to sections in priority order."""

    def allocate(
        self,
        sections: list,
        total_budget: int,
        *,
        preserved_section_types: set[str] | None = None,
    ) -> dict[str, int]:
        allocation: dict[str, int] = {}
        remaining = total_budget
        preserved = preserved_section_types or set()

        for section in sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            if st in preserved and section.token_count <= remaining:
                allocation[st] = section.token_count
                remaining -= section.token_count

        for section in sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            if st in allocation:
                continue
            allocated = min(section.token_count, remaining)
            allocation[st] = allocated
            remaining -= allocated
            if remaining <= 0:
                break

        return allocation


class WeightedAllocation:
    """Allocate budget proportional to section importance weights."""

    def __init__(self, section_weights: dict[str, float] | None = None) -> None:
        self._section_weights = section_weights or {
            "user_query": 5.0,
            "working_memory": 3.0,
            "relevant_memories": 4.0,
            "related_memories": 2.0,
            "long_term_facts": 3.0,
            "conversation_history": 2.0,
            "retrieved_metadata": 1.0,
        }

    def allocate(
        self,
        sections: list,
        total_budget: int,
        *,
        preserved_section_types: set[str] | None = None,
    ) -> dict[str, int]:
        preserved = preserved_section_types or set()
        allocation: dict[str, int] = {}
        remaining = total_budget

        for section in sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            if st in preserved and section.token_count <= remaining:
                allocation[st] = section.token_count
                remaining -= section.token_count

        remaining_sections = [s for s in sections if (
            s.section_type if hasattr(s, "section_type") else ""
        ) not in allocation]
        total_weight = sum(
            self._section_weights.get(s.section_type if hasattr(s, "section_type") else "", 1.0)
            for s in remaining_sections
        )

        if total_weight <= 0 or not remaining_sections:
            for section in remaining_sections:
                st = section.section_type if hasattr(section, "section_type") else ""
                allocation[st] = min(section.token_count, remaining)
            return allocation

        for section in remaining_sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            weight = self._section_weights.get(st, 1.0)
            proportion = weight / total_weight
            allocated = min(section.token_count, int(remaining * proportion))
            allocation[st] = allocated
            remaining -= allocated

        return allocation


class ProportionalAllocation:
    """Allocate budget proportional to each section's size relative to total."""

    def allocate(
        self,
        sections: list,
        total_budget: int,
        *,
        preserved_section_types: set[str] | None = None,
    ) -> dict[str, int]:
        preserved = preserved_section_types or set()
        allocation: dict[str, int] = {}
        remaining = total_budget

        for section in sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            if st in preserved and section.token_count <= remaining:
                allocation[st] = section.token_count
                remaining -= section.token_count

        remaining_sections = [s for s in sections if (
            s.section_type if hasattr(s, "section_type") else ""
        ) not in allocation]
        total_requested = sum(s.token_count for s in remaining_sections)
        if total_requested <= 0:
            return allocation

        for section in remaining_sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            proportion = section.token_count / total_requested if total_requested > 0 else 0
            allocated = min(section.token_count, int(remaining * proportion))
            allocation[st] = allocated

        return allocation


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Configuration for the Token Budget Manager."""

    default_budget: int = 4096
    emergency_budget: int = 1024
    allocation_strategy: str = "priority_first"
    section_weights: dict[str, float] = field(default_factory=lambda: {
        "user_query": 5.0,
        "working_memory": 3.0,
        "relevant_memories": 4.0,
        "related_memories": 2.0,
        "long_term_facts": 3.0,
        "conversation_history": 2.0,
        "retrieved_metadata": 1.0,
    })
    preserved_section_types: set[str] = field(default_factory=lambda: {"user_query"})
    min_allocation_per_section: int = 0
    enable_emergency_mode: bool = True
    remove_zero_allocation_sections: bool = False
    memory_overhead_tokens: int = 40


# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------


class TokenBudgetManager:
    """Owns every decision related to token allocation and budget optimisation."""

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()

    @property
    def config(self) -> BudgetConfig:
        return self._config

    async def optimise(
        self,
        package: Any,
        *,
        target_budget: int | None = None,
    ) -> BudgetResult:
        """Optimise *package* to fit within *target_budget*."""
        cfg = self._config
        budget = target_budget or cfg.default_budget
        original_tokens = getattr(package, "total_tokens", 0)

        if original_tokens <= budget:
            return BudgetResult(
                package=package,
                statistics=BudgetStatistics(
                    original_tokens=original_tokens,
                    final_tokens=original_tokens,
                    tokens_removed=0,
                    memories_removed=0,
                    sections_trimmed=0,
                ),
            )

        # Step 1: Allocate budget per section
        strategy = self._build_strategy()
        allocations = strategy.allocate(
            package.sections,
            budget,
            preserved_section_types=cfg.preserved_section_types,
        )

        # Step 2: Trim each section to its allocation
        optimised_sections: list = []
        decisions: list[BudgetDecision] = []
        total_allocated = 0
        total_trimmed_memories = 0
        trimmed_sections = 0
        allocations_out: list[BudgetAllocation] = []

        from app.memory.context import ContextSection

        for section in package.sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            available = allocations.get(st, 0)

            if available <= 0 and cfg.remove_zero_allocation_sections:
                trimmed_sections += 1
                decisions.append(BudgetDecision(
                    section_type=st,
                    reason="zero allocation — section removed",
                    token_savings=section.token_count if hasattr(section, "token_count") else 0,
                ))
                continue

            trimmed_section, removed_count, section_decisions = self._trim_section(
                section, available,
            )
            total_trimmed_memories += removed_count
            decisions.extend(section_decisions)
            if removed_count > 0:
                trimmed_sections += 1

            if hasattr(trimmed_section, "memories") and trimmed_section.memories:
                optimised_sections.append(trimmed_section)
                total_allocated += trimmed_section.token_count if hasattr(trimmed_section, "token_count") else 0

            allocations_out.append(BudgetAllocation(
                section_type=st,
                requested_tokens=section.token_count if hasattr(section, "token_count") else 0,
                allocated_tokens=trimmed_section.token_count if hasattr(trimmed_section, "token_count") and trimmed_section.memories else 0,
                trimmed_memories=removed_count,
                preserved_memory_ids=[m.id.value for m in trimmed_section.memories] if hasattr(trimmed_section, "memories") else [],
            ))

        # Step 3: Emergency mode
        emergency_used = False
        if total_allocated > budget and cfg.enable_emergency_mode:
            emergency_sections: list = []
            for section in optimised_sections:
                st = section.section_type if hasattr(section, "section_type") else ""
                if st in cfg.preserved_section_types:
                    emergency_sections.append(section)
                else:
                    emergency_sections.append(self._force_empty_section(section))
            total_allocated = sum(s.token_count for s in emergency_sections)
            optimised_sections = emergency_sections
            emergency_used = True
            decisions.append(BudgetDecision(
                section_type="__emergency__",
                reason="emergency budget activated — non-preserved sections emptied",
                token_savings=original_tokens - total_allocated,
            ))

        # Step 4: Build result
        stats = BudgetStatistics(
            original_tokens=original_tokens,
            final_tokens=total_allocated,
            tokens_removed=original_tokens - total_allocated,
            memories_removed=total_trimmed_memories,
            sections_trimmed=trimmed_sections,
            allocations=allocations_out,
            emergency_mode_used=emergency_used,
        )

        from app.memory.context import ContextPackage

        optimised_package = ContextPackage(
            request_id=getattr(package, "request_id", ""),
            sections=optimised_sections,
            statistics=getattr(package, "statistics", None),
            metadata={
                **(getattr(package, "metadata", {})),
                "budget_optimised": True,
                "budget_original_tokens": original_tokens,
                "budget_final_tokens": total_allocated,
                "budget_target": budget,
            },
        )

        return BudgetResult(
            package=optimised_package,
            statistics=stats,
            decisions=decisions,
            metadata={"target_budget": budget},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_strategy(self) -> AllocationStrategy:
        strat = self._config.allocation_strategy
        if strat == "weighted":
            return WeightedAllocation(section_weights=self._config.section_weights)
        if strat == "proportional":
            return ProportionalAllocation()
        return PriorityFirstAllocation()

    def _trim_section(
        self,
        section: Any,
        budget: int,
    ) -> tuple[Any, int, list[BudgetDecision]]:
        """Trim memories from *section* to fit *budget* tokens."""
        from app.memory.context import ContextSection, ContextSource

        decisions: list[BudgetDecision] = []
        trimmed_memories: list[Memory] = []
        trimmed_sources: list = []
        used = 0

        section_memories = section.memories if hasattr(section, "memories") else []
        section_sources = section.sources if hasattr(section, "sources") else []
        st = section.section_type if hasattr(section, "section_type") else ""

        for mem, src in zip(section_memories, section_sources or []):
            mem_tokens = TokenEstimator.estimate_memory(mem)
            if used + mem_tokens > budget:
                decisions.append(BudgetDecision(
                    memory_id=mem.id.value,
                    section_type=st,
                    reason=f"token budget exceeded ({used + mem_tokens} > {budget})",
                    token_savings=mem_tokens,
                ))
                continue
            trimmed_memories.append(mem)
            trimmed_sources.append(src)
            used += mem_tokens

        removed_count = len(section_memories) - len(trimmed_memories)

        new_section = ContextSection(
            section_type=st,
            label=section.label if hasattr(section, "label") else "",
            memories=trimmed_memories,
            sources=trimmed_sources,
            metadata=section.metadata if hasattr(section, "metadata") else {},
            token_count=used,
        )

        return new_section, removed_count, decisions

    @staticmethod
    def _force_empty_section(section: Any) -> Any:
        """Return a version of *section* with all memories removed."""
        from app.memory.context import ContextSection

        return ContextSection(
            section_type=section.section_type if hasattr(section, "section_type") else "",
            label=section.label if hasattr(section, "label") else "",
            memories=[],
            sources=[],
            metadata=section.metadata if hasattr(section, "metadata") else {},
            token_count=0,
        )
