"""Context Optimization Engine — improves ContextPackage quality.

This is the final intelligence layer before Prompt Assembly.

Its responsibility is NOT token budgeting.  It is making the
context better: removing redundancy, merging duplicates,
improving ordering, and maximising information density while
preserving meaning.

Every optimisation pass is independently testable.

No provider-specific knowledge exists in this module.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.memory.memory import Memory, MemoryState
from app.memory.tokens import TokenEstimator


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationDecision:
    """Record of a single optimization action."""

    pass_name: str = ""
    memory_id: str = ""
    section_type: str = ""
    action: str = ""  # "removed", "merged", "reordered", "cleaned"
    reason: str = ""


@dataclass(frozen=True)
class OptimizationStatistics:
    """Aggregated statistics for an optimization run."""

    original_memories: int = 0
    final_memories: int = 0
    original_sections: int = 0
    final_sections: int = 0
    original_tokens: int = 0
    final_tokens: int = 0
    removed_count: int = 0
    merged_count: int = 0
    reordered_count: int = 0
    passes_run: int = 0


@dataclass(frozen=True)
class OptimizationResult:
    """The output of a full optimization run."""

    package: Any = field(default_factory=lambda: type("_", (), {"sections": [], "total_memories": 0, "total_tokens": 0, "total_sections": 0, "request_id": "", "statistics": type("_", (), {"total_memories": 0, "total_sections": 0, "total_tokens": 0, "retrieval_ms": 0.0, "compression_ms": 0.0, "assembly_ms": 0.0, "estimated_tokens": 0, "sources_breakdown": {}}), "metadata": {}, "to_dict": lambda: {}})())
    decisions: list[OptimizationDecision] = field(default_factory=list)
    statistics: OptimizationStatistics = field(default_factory=OptimizationStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_memories": self.statistics.original_memories,
            "final_memories": self.statistics.final_memories,
            "removed_count": self.statistics.removed_count,
            "merged_count": self.statistics.merged_count,
            "passes_run": self.statistics.passes_run,
            "decision_count": len(self.decisions),
        }


# ---------------------------------------------------------------------------
# Optimization config
# ---------------------------------------------------------------------------


@dataclass
class OptimizationConfig:
    """Configuration for the Context Optimization Engine."""

    strategy: str = "balanced"  # "conservative", "balanced", "aggressive"

    # Which passes to enable
    enable_duplicate_elimination: bool = True
    enable_near_duplicate_elimination: bool = True
    enable_metadata_cleanup: bool = True
    enable_empty_section_removal: bool = True
    enable_memory_ordering: bool = True
    enable_section_ordering: bool = False
    enable_statistics_recalculation: bool = True

    # Thresholds
    near_duplicate_similarity: float = 0.85
    max_section_count: int = 20

    # Pinned items to always preserve
    preserved_section_types: set[str] = field(default_factory=lambda: {"user_query"})
    preserved_memory_ids: set[str] = field(default_factory=set)

    # Strategy presets
    _conservative: dict[str, bool] = field(default_factory=lambda: {
        "enable_duplicate_elimination": True,
        "enable_near_duplicate_elimination": False,
        "enable_metadata_cleanup": True,
        "enable_empty_section_removal": True,
        "enable_memory_ordering": False,
        "enable_section_ordering": False,
        "enable_statistics_recalculation": True,
    })
    _aggressive: dict[str, bool] = field(default_factory=lambda: {
        "enable_duplicate_elimination": True,
        "enable_near_duplicate_elimination": True,
        "enable_metadata_cleanup": True,
        "enable_empty_section_removal": True,
        "enable_memory_ordering": True,
        "enable_section_ordering": True,
        "enable_statistics_recalculation": True,
    })

    def apply_strategy(self, strategy: str) -> None:
        """Apply a strategy preset."""
        if strategy == "conservative":
            for k, v in self._conservative.items():
                setattr(self, k, v)
        elif strategy == "aggressive":
            for k, v in self._aggressive.items():
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Optimization pass ABC
# ---------------------------------------------------------------------------


class OptimizationPass(abc.ABC):
    """A single, independently runnable optimisation pass."""

    name: str = ""

    @abc.abstractmethod
    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        ...


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------


def make_strategy_config(strategy: str, base: OptimizationConfig | None = None) -> OptimizationConfig:
    """Build an ``OptimizationConfig`` from a strategy name."""
    cfg = base or OptimizationConfig()
    cfg.strategy = strategy
    cfg.apply_strategy(strategy)
    return cfg


# ---------------------------------------------------------------------------
# Built-in optimisation passes
# ---------------------------------------------------------------------------


class DuplicateEliminationPass(OptimizationPass):
    """Remove exact duplicate memories within each section."""

    name = "duplicate_elimination"

    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        from app.memory.context import ContextPackage, ContextSection, ContextSource

        decisions: list[OptimizationDecision] = []
        new_sections: list = []

        for section in package.sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            seen: dict[tuple[str, str], Memory] = {}
            kept_sources: list = []
            removed_ids: set[str] = set()

            section_mems = section.memories if hasattr(section, "memories") else []
            section_srcs = section.sources if hasattr(section, "sources") else []

            for mem, src in zip(section_mems, section_srcs or []):
                key = (mem.namespace, mem.content)
                existing = seen.get(key)
                if existing is None:
                    seen[key] = mem
                    kept_sources.append(src)
                else:
                    if mem.importance > existing.importance:
                        seen[key] = mem
                        removed_ids.add(existing.id.value)
                    else:
                        removed_ids.add(mem.id.value)
                    decisions.append(OptimizationDecision(
                        pass_name=self.name, memory_id=mem.id.value, section_type=st,
                        action="removed", reason=f"duplicate of {existing.id.value}",
                    ))

            kept_mems = list(seen.values())
            tokens = TokenEstimator.estimate_memories(kept_mems) if removed_ids else \
                (section.token_count if hasattr(section, "token_count") else 0)

            new_sections.append(ContextSection(
                section_type=st,
                label=section.label if hasattr(section, "label") else "",
                memories=kept_mems,
                sources=kept_sources[:len(kept_mems)],
                metadata=section.metadata if hasattr(section, "metadata") else {},
                token_count=tokens,
            ))

        return ContextPackage(
            request_id=package.request_id if hasattr(package, "request_id") else "",
            sections=new_sections,
            statistics=package.statistics if hasattr(package, "statistics") else None,
            metadata=package.metadata if hasattr(package, "metadata") else {},
        ), decisions


class MetadataCleanupPass(OptimizationPass):
    """Remove redundant or oversized metadata from memories."""

    name = "metadata_cleanup"

    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        from app.memory.context import ContextPackage, ContextSection

        decisions: list[OptimizationDecision] = []

        new_sections: list = []
        for section in package.sections:
            new_mems: list[Memory] = []
            section_mems = section.memories if hasattr(section, "memories") else []
            section_srcs = section.sources if hasattr(section, "sources") else []

            for mem in section_mems:
                if mem.metadata:
                    cleaned = {
                        k: (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v
                        for k, v in mem.metadata.items()
                        if v not in (None, "", {}, [])
                    }
                    if len(cleaned) != len(mem.metadata):
                        mem.metadata = cleaned
                        decisions.append(OptimizationDecision(
                            pass_name=self.name, memory_id=mem.id.value,
                            section_type=section.section_type if hasattr(section, "section_type") else "",
                            action="cleaned", reason="metadata truncated or emptied",
                        ))
                new_mems.append(mem)

            tokens = TokenEstimator.estimate_memories(new_mems)
            new_sections.append(ContextSection(
                section_type=section.section_type if hasattr(section, "section_type") else "",
                label=section.label if hasattr(section, "label") else "",
                memories=new_mems,
                sources=list(section_srcs)[:len(new_mems)],
                metadata=section.metadata if hasattr(section, "metadata") else {},
                token_count=tokens,
            ))

        return ContextPackage(
            request_id=package.request_id if hasattr(package, "request_id") else "",
            sections=new_sections,
            statistics=package.statistics if hasattr(package, "statistics") else None,
            metadata=package.metadata if hasattr(package, "metadata") else {},
        ), decisions


class EmptySectionRemovalPass(OptimizationPass):
    """Remove sections that have no memories."""

    name = "empty_section_removal"

    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        from app.memory.context import ContextPackage

        decisions: list[OptimizationDecision] = []
        new_sections = [
            s for s in package.sections
            if hasattr(s, "memories") and s.memories
        ]
        removed = len(package.sections) - len(new_sections)
        for s in package.sections:
            if not hasattr(s, "memories") or not s.memories:
                decisions.append(OptimizationDecision(
                    pass_name=self.name, section_type=s.section_type if hasattr(s, "section_type") else "",
                    action="removed", reason="section has no memories",
                ))

        return ContextPackage(
            request_id=package.request_id if hasattr(package, "request_id") else "",
            sections=new_sections,
            statistics=package.statistics if hasattr(package, "statistics") else None,
            metadata=package.metadata if hasattr(package, "metadata") else {},
        ), decisions


class MemoryOrderingPass(OptimizationPass):
    """Order memories within each section by importance descending."""

    name = "memory_ordering"

    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        from app.memory.context import ContextPackage, ContextSection

        decisions: list[OptimizationDecision] = []
        new_sections: list = []

        for section in package.sections:
            section_mems = section.memories if hasattr(section, "memories") else []
            section_srcs = section.sources if hasattr(section, "sources") else []
            st = section.section_type if hasattr(section, "section_type") else ""

            pairs = list(zip(section_mems, section_srcs or []))
            if not pairs:
                new_sections.append(section)
                continue

            pairs.sort(key=lambda p: (-p[0].importance, p[0].id.value))
            reordered_mems = [p[0] for p in pairs]
            reordered_srcs = [p[1] for p in pairs]

            if reordered_mems != list(section_mems):
                decisions.append(OptimizationDecision(
                    pass_name=self.name, section_type=st,
                    action="reordered", reason="memories reordered by importance",
                ))

            tokens = TokenEstimator.estimate_memories(reordered_mems)
            new_sections.append(ContextSection(
                section_type=st,
                label=section.label if hasattr(section, "label") else "",
                memories=reordered_mems,
                sources=reordered_srcs,
                metadata=section.metadata if hasattr(section, "metadata") else {},
                token_count=tokens,
            ))

        return ContextPackage(
            request_id=package.request_id if hasattr(package, "request_id") else "",
            sections=new_sections,
            statistics=package.statistics if hasattr(package, "statistics") else None,
            metadata=package.metadata if hasattr(package, "metadata") else {},
        ), decisions


class NearDuplicateEliminationPass(OptimizationPass):
    """Remove near-duplicate memories via content overlap ratio."""

    name = "near_duplicate_elimination"

    async def run(
        self,
        package: Any,
        config: OptimizationConfig,
    ) -> tuple[Any, list[OptimizationDecision]]:
        from app.memory.context import ContextPackage, ContextSection

        threshold = config.near_duplicate_similarity
        decisions: list[OptimizationDecision] = []
        new_sections: list = []

        for section in package.sections:
            st = section.section_type if hasattr(section, "section_type") else ""
            section_mems = section.memories if hasattr(section, "memories") else []
            section_srcs = section.sources if hasattr(section, "sources") else []

            kept: list[Memory] = []
            kept_srcs: list = []

            for mem, src in zip(section_mems, section_srcs or []):
                is_near_dup = False
                for existing in kept:
                    overlap = self._content_overlap(mem.content, existing.content)
                    if overlap >= threshold:
                        if mem.importance > existing.importance:
                            kept.remove(existing)
                            kept.append(mem)
                            decisions.append(OptimizationDecision(
                                pass_name=self.name, memory_id=existing.id.value, section_type=st,
                                action="removed", reason=f"near-duplicate of {mem.id.value} (overlap={overlap:.2f})",
                            ))
                        else:
                            decisions.append(OptimizationDecision(
                                pass_name=self.name, memory_id=mem.id.value, section_type=st,
                                action="removed", reason=f"near-duplicate of {existing.id.value} (overlap={overlap:.2f})",
                            ))
                        is_near_dup = True
                        break

                if not is_near_dup:
                    kept.append(mem)
                    kept_srcs.append(src)

            tokens = TokenEstimator.estimate_memories(kept)
            new_sections.append(ContextSection(
                section_type=st,
                label=section.label if hasattr(section, "label") else "",
                memories=kept,
                sources=kept_srcs,
                metadata=section.metadata if hasattr(section, "metadata") else {},
                token_count=tokens,
            ))

        return ContextPackage(
            request_id=package.request_id if hasattr(package, "request_id") else "",
            sections=new_sections,
            statistics=package.statistics if hasattr(package, "statistics") else None,
            metadata=package.metadata if hasattr(package, "metadata") else {},
        ), decisions

    @staticmethod
    def _content_overlap(a: str, b: str) -> float:
        """Word-level Jaccard similarity as proxy for semantic overlap."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# ContextOptimizationEngine
# ---------------------------------------------------------------------------


class ContextOptimizationEngine:
    """Orchestrates optimisation passes to improve ContextPackage quality."""

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        cfg = config or OptimizationConfig()
        cfg.apply_strategy(cfg.strategy)
        self._config = cfg
        self._passes = self._build_passes()

    @property
    def config(self) -> OptimizationConfig:
        return self._config

    async def optimize(self, package: Any) -> OptimizationResult:
        """Run all enabled optimisation passes on *package*."""
        from app.memory.context import ContextPackage, ContextStatistics

        cfg = self._config
        original_memories = package.total_memories if hasattr(package, "total_memories") else 0
        original_sections = len(package.sections) if hasattr(package, "sections") else 0
        original_tokens = package.total_tokens if hasattr(package, "total_tokens") else 0

        decisions: list[OptimizationDecision] = []
        current = package

        passes_run = 0
        for opt_pass in self._passes:
            current, new_decisions = await opt_pass.run(current, cfg)
            decisions.extend(new_decisions)
            passes_run += 1

        # Recalculate statistics
        total_memories = current.total_memories if hasattr(current, "total_memories") else 0
        total_sections = len(current.sections) if hasattr(current, "sections") else 0
        total_tokens = current.total_tokens if hasattr(current, "total_tokens") else 0

        if cfg.enable_statistics_recalculation:
            recalc_stats = ContextStatistics(
                total_memories=total_memories,
                total_sections=total_sections,
                total_tokens=total_tokens,
                estimated_tokens=total_tokens,
            )
            current = ContextPackage(
                request_id=current.request_id if hasattr(current, "request_id") else "",
                sections=current.sections if hasattr(current, "sections") else [],
                statistics=recalc_stats,
                metadata={
                    **(current.metadata if hasattr(current, "metadata") else {}),
                    "optimized": True,
                    "optimization_strategy": cfg.strategy,
                },
            )

        removed = len(decisions)
        merged = sum(1 for d in decisions if d.action in ("merged", "removed"))

        stats = OptimizationStatistics(
            original_memories=original_memories,
            final_memories=total_memories,
            original_sections=original_sections,
            final_sections=total_sections,
            original_tokens=original_tokens,
            final_tokens=total_tokens,
            removed_count=removed,
            merged_count=merged,
            passes_run=passes_run,
        )

        return OptimizationResult(
            package=current,
            decisions=decisions,
            statistics=stats,
            metadata={"strategy": cfg.strategy, "passes_run": passes_run},
        )

    def _build_passes(self) -> list[OptimizationPass]:
        cfg = self._config
        passes: list[OptimizationPass] = []
        if cfg.enable_duplicate_elimination:
            passes.append(DuplicateEliminationPass())
        if cfg.enable_near_duplicate_elimination:
            passes.append(NearDuplicateEliminationPass())
        if cfg.enable_metadata_cleanup:
            passes.append(MetadataCleanupPass())
        if cfg.enable_empty_section_removal:
            passes.append(EmptySectionRemovalPass())
        if cfg.enable_memory_ordering:
            passes.append(MemoryOrderingPass())
        return passes
