"""Plan domain models for the Planning Engine.

A ``Plan`` tracks the execution state of a multi-step reasoning interaction.
The plan is maintained by the ``PlanningEngine`` and owned by the
``AgentRuntime``.

Every model is immutable and provider-independent.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


class PlanStatus(str, enum.Enum):
    """Status of a plan."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, enum.Enum):
    """Status of an individual plan step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PlanStep:
    """A single step in a plan.

    Each step represents one reasoning iteration (provider request + optional
    tool execution).

    Attributes:
        id: Unique identifier for this step.
        title: Human-readable title for the step.
        description: Optional detailed description.
        status: Current status of the step.
        dependencies: List of step IDs that must complete before this one.
        metadata: Optional structured data attached to the step.
    """

    id: str = ""
    title: str = ""
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    """A structured execution plan for the agent.

    Tracks all steps and their statuses throughout a run.

    Attributes:
        id: Unique identifier for this plan.
        steps: Ordered list of plan steps.
        status: Overall plan status.
        metadata: Optional structured data.
    """

    id: str = ""
    steps: tuple[PlanStep, ...] = ()
    status: PlanStatus = PlanStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)

    @property
    def is_finished(self) -> bool:
        return self.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "step_count": self.step_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "status": s.status.value,
                    "dependencies": list(s.dependencies),
                }
                for s in self.steps
            ],
        }


@dataclass(frozen=True)
class PlanResult:
    """Result of a plan execution.

    Attributes:
        plan: The final plan with all step statuses.
        completed: Whether the plan completed successfully.
        total_steps: Total number of steps in the plan.
        completed_steps: Number of steps that completed.
        failed_steps: Number of steps that failed.
    """

    plan: Plan = field(default_factory=Plan)
    completed: bool = False
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
