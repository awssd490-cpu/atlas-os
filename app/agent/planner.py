"""Planning Engine — provider-independent execution state manager.

The planner is responsible for deciding:
- what work remains
- which step is next
- whether execution has completed

It is NOT responsible for:
- executing tools
- calling providers
- memory retrieval
- formatting requests

The planner only manages execution state.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.agent.plan import Plan, PlanResult, PlanStatus, PlanStep, StepStatus


class PlanningEngine:
    """Deterministic execution state manager.

    Creates and tracks a structured plan across the agent's reasoning loop.
    No AI or provider-specific logic lives here.

    Usage::

        engine = PlanningEngine()
        plan = engine.create_plan("Answer: what is 2+2?")
        engine.start_step(plan, step_id=plan.steps[0].id)
        engine.complete_step(plan, step_id=plan.steps[0].id)
        result = engine.finish_plan(plan)
    """

    def __init__(self) -> None:
        self._current_plan: Plan | None = None
        self._current_step_id: str | None = None

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_plan(
        self,
        *,
        title: str = "",
        steps: list[PlanStep] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Plan:
        """Create a new plan.

        If *steps* is provided, those steps become the plan.  Otherwise a
        single default step is created.

        Args:
            title: Title for the default step (ignored if *steps* provided).
            steps: Explicit list of plan steps.
            metadata: Optional plan metadata.

        Returns:
            The newly created ``Plan``.
        """
        plan_id = str(uuid.uuid4())[:8]
        resolved_steps = list(steps) if steps else [
            PlanStep(
                id=str(uuid.uuid4())[:8],
                title=title or "Reasoning iteration",
                description="",
                status=StepStatus.PENDING,
            ),
        ]

        plan = Plan(
            id=plan_id,
            steps=tuple(resolved_steps),
            status=PlanStatus.PENDING,
            metadata=metadata or {},
        )
        self._current_plan = plan
        self._current_step_id = None
        return plan

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def start_step(self, plan: Plan, step_id: str) -> Plan:
        """Mark a step as running.

        Args:
            plan: The plan containing the step.
            step_id: ID of the step to start.

        Returns:
            Updated ``Plan`` with the step marked as RUNNING.

        Raises:
            ValueError: If *step_id* is not found, or if its dependencies
                are not yet completed.
        """
        updated_steps: list[PlanStep] = []
        found = False

        for step in plan.steps:
            if step.id == step_id:
                found = True
                if step.status not in (StepStatus.PENDING,):
                    raise ValueError(
                        f"Step {step_id!r} cannot be started: status is {step.status.value}"
                    )
                # Check dependencies
                for dep_id in step.dependencies:
                    dep_step = next((s for s in plan.steps if s.id == dep_id), None)
                    if dep_step is None:
                        raise ValueError(
                            f"Step {step_id!r} depends on unknown step {dep_id!r}"
                        )
                    if dep_step.status != StepStatus.COMPLETED:
                        raise ValueError(
                            f"Step {step_id!r} depends on {dep_id!r} "
                            f"which is {dep_step.status.value}"
                        )

                updated_steps.append(PlanStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    status=StepStatus.RUNNING,
                    dependencies=step.dependencies,
                    metadata=step.metadata,
                ))
                self._current_step_id = step_id
            else:
                updated_steps.append(step)

        if not found:
            raise ValueError(f"Step {step_id!r} not found in plan")

        new_plan = Plan(
            id=plan.id,
            steps=tuple(updated_steps),
            status=PlanStatus.RUNNING,
            metadata=plan.metadata,
        )
        self._current_plan = new_plan
        return new_plan

    def complete_step(self, plan: Plan, step_id: str) -> Plan:
        """Mark a step as completed.

        Args:
            plan: The plan containing the step.
            step_id: ID of the step to complete.

        Returns:
            Updated plan.
        """
        return self._update_step_status(plan, step_id, StepStatus.COMPLETED)

    def fail_step(
        self,
        plan: Plan,
        step_id: str,
        *,
        error: str = "",
    ) -> Plan:
        """Mark a step as failed.

        Args:
            plan: The plan containing the step.
            step_id: ID of the step to fail.
            error: Error description.

        Returns:
            Updated plan.
        """
        updated = self._update_step_status(plan, step_id, StepStatus.FAILED)
        if updated.steps:
            # Find the failed step and attach error
            final_steps: list[PlanStep] = []
            for step in updated.steps:
                if step.id == step_id:
                    meta = dict(step.metadata)
                    meta["error"] = error
                    final_steps.append(PlanStep(
                        id=step.id,
                        title=step.title,
                        description=step.description,
                        status=step.status,
                        dependencies=step.dependencies,
                        metadata=meta,
                    ))
                else:
                    final_steps.append(step)
            updated = Plan(
                id=updated.id,
                steps=tuple(final_steps),
                status=updated.status,
                metadata=updated.metadata,
            )
        self._current_plan = updated
        return updated

    def skip_step(self, plan: Plan, step_id: str) -> Plan:
        """Mark a step as skipped.

        Args:
            plan: The plan containing the step.
            step_id: ID of the step to skip.

        Returns:
            Updated plan.
        """
        return self._update_step_status(plan, step_id, StepStatus.SKIPPED)

    # ------------------------------------------------------------------
    # Plan completion
    # ------------------------------------------------------------------

    def finish_plan(self, plan: Plan, *, failed: bool = False) -> PlanResult:
        """Mark the plan as completed or failed and return the result.

        Args:
            plan: The plan to finish.
            failed: If ``True``, mark the plan as FAILED.

        Returns:
            A ``PlanResult`` with the final plan state.
        """
        status = PlanStatus.FAILED if failed else PlanStatus.COMPLETED

        # Mark any remaining PENDING steps as SKIPPED
        final_steps: list[PlanStep] = []
        for step in plan.steps:
            if step.status == StepStatus.PENDING:
                final_steps.append(PlanStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    status=StepStatus.SKIPPED,
                    dependencies=step.dependencies,
                    metadata=step.metadata,
                ))
            else:
                final_steps.append(step)

        finished_plan = Plan(
            id=plan.id,
            steps=tuple(final_steps),
            status=status,
            metadata=plan.metadata,
        )
        self._current_plan = finished_plan
        self._current_step_id = None

        return PlanResult(
            plan=finished_plan,
            completed=(status == PlanStatus.COMPLETED),
            total_steps=finished_plan.step_count,
            completed_steps=finished_plan.completed_count,
            failed_steps=finished_plan.failed_count,
        )

    def get_next_step(self, plan: Plan) -> PlanStep | None:
        """Return the next executable step (first PENDING with deps met).

        Args:
            plan: The plan to inspect.

        Returns:
            The next ``PlanStep`` or ``None`` if all steps are done.
        """
        completed_ids = {
            s.id for s in plan.steps if s.status == StepStatus.COMPLETED
        }
        for step in plan.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                return step
        return None

    def is_plan_finished(self, plan: Plan) -> bool:
        """Check if all steps are terminal (completed, failed, or skipped).

        Args:
            plan: The plan to check.

        Returns:
            ``True`` if no steps are PENDING or RUNNING.
        """
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)
            for s in plan.steps
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def current_plan(self) -> Plan | None:
        """The current plan being executed, if any."""
        return self._current_plan

    @property
    def current_step_id(self) -> str | None:
        """The ID of the currently executing step, if any."""
        return self._current_step_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_step_status(
        self,
        plan: Plan,
        step_id: str,
        new_status: StepStatus,
    ) -> Plan:
        """Update a single step's status and return the new plan."""
        updated_steps: list[PlanStep] = []
        found = False

        for step in plan.steps:
            if step.id == step_id:
                found = True
                updated_steps.append(PlanStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    status=new_status,
                    dependencies=step.dependencies,
                    metadata=step.metadata,
                ))
            else:
                updated_steps.append(step)

        if not found:
            raise ValueError(f"Step {step_id!r} not found in plan")

        new_plan = Plan(
            id=plan.id,
            steps=tuple(updated_steps),
            status=plan.status,
            metadata=plan.metadata,
        )
        self._current_plan = new_plan
        return new_plan
