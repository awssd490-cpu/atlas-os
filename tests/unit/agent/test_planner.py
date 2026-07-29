"""Tests for the Planning Engine."""

from __future__ import annotations

import pytest

from app.agent.plan import Plan, PlanResult, PlanStep, PlanStatus, StepStatus
from app.agent.planner import PlanningEngine


class TestPlanningEngine:
    def test_create_plan_default(self) -> None:
        """Create a plan with default single step."""
        engine = PlanningEngine()
        plan = engine.create_plan(title="Test plan")
        assert plan.id != ""
        assert len(plan.steps) == 1
        assert plan.steps[0].title == "Test plan"
        assert plan.steps[0].status == StepStatus.PENDING

    def test_create_plan_with_custom_steps(self) -> None:
        """Create a plan with explicit steps."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.PENDING),
            PlanStep(id="s2", title="Step 2", status=StepStatus.PENDING),
        ]
        plan = engine.create_plan(steps=steps)
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "s1"
        assert plan.steps[1].id == "s2"

    def test_start_step(self) -> None:
        """Start a pending step."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        updated = engine.start_step(plan, step_id)
        assert updated.steps[0].status == StepStatus.RUNNING
        assert engine.current_step_id == step_id

    def test_start_step_not_pending_fails(self) -> None:
        """Starting a non-pending step fails."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        plan = engine.start_step(plan, step_id)
        plan = engine.complete_step(plan, step_id)
        with pytest.raises(ValueError, match="cannot be started"):
            engine.start_step(plan, step_id)

    def test_start_nonexistent_step_fails(self) -> None:
        """Starting unknown step fails."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        with pytest.raises(ValueError, match="not found"):
            engine.start_step(plan, "nonexistent")

    def test_complete_step(self) -> None:
        """Complete a running step."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        plan = engine.start_step(plan, step_id)
        plan = engine.complete_step(plan, step_id)
        assert plan.steps[0].status == StepStatus.COMPLETED

    def test_fail_step(self) -> None:
        """Fail a step with error."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        plan = engine.start_step(plan, step_id)
        plan = engine.fail_step(plan, step_id, error="boom")
        assert plan.steps[0].status == StepStatus.FAILED
        assert "boom" in plan.steps[0].metadata.get("error", "")

    def test_skip_step(self) -> None:
        """Skip a pending step."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        plan = engine.skip_step(plan, step_id)
        assert plan.steps[0].status == StepStatus.SKIPPED

    def test_finish_plan_completed(self) -> None:
        """Finish a plan as completed."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        plan = engine.start_step(plan, plan.steps[0].id)
        plan = engine.complete_step(plan, plan.steps[0].id)
        result = engine.finish_plan(plan)
        assert result.completed is True
        assert result.plan.status == PlanStatus.COMPLETED
        assert result.total_steps == 1
        assert result.completed_steps == 1
        assert result.failed_steps == 0

    def test_finish_plan_failed(self) -> None:
        """Finish a plan as failed."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        plan = engine.start_step(plan, plan.steps[0].id)
        plan = engine.fail_step(plan, plan.steps[0].id, error="boom")
        result = engine.finish_plan(plan, failed=True)
        assert result.completed is False
        assert result.plan.status == PlanStatus.FAILED
        assert result.failed_steps == 1

    def test_finish_plan_skips_pending(self) -> None:
        """Finishing marks pending steps as skipped."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.PENDING),
            PlanStep(id="s2", title="Step 2", status=StepStatus.PENDING),
        ]
        plan = engine.create_plan(steps=steps)
        plan = engine.start_step(plan, "s1")
        plan = engine.complete_step(plan, "s1")
        result = engine.finish_plan(plan)
        assert result.completed is True
        assert result.plan.steps[1].status == StepStatus.SKIPPED

    def test_get_next_step(self) -> None:
        """Get next executable step."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", title="Step 2", status=StepStatus.PENDING, dependencies=("s1",)),
            PlanStep(id="s3", title="Step 3", status=StepStatus.PENDING, dependencies=("s2",)),
        ]
        plan = Plan(id="p1", steps=tuple(steps), status=PlanStatus.RUNNING)
        next_step = engine.get_next_step(plan)
        assert next_step is not None
        assert next_step.id == "s2"

    def test_get_next_step_with_unmet_deps(self) -> None:
        """Next step respects unmet dependencies."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.PENDING),
            PlanStep(id="s2", title="Step 2", status=StepStatus.PENDING, dependencies=("s1",)),
        ]
        plan = Plan(id="p1", steps=tuple(steps), status=PlanStatus.RUNNING)
        next_step = engine.get_next_step(plan)
        assert next_step is not None
        assert next_step.id == "s1"

    def test_get_next_step_none_when_done(self) -> None:
        """No next step when all done."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", title="Step 2", status=StepStatus.COMPLETED),
        ]
        plan = Plan(id="p1", steps=tuple(steps), status=PlanStatus.COMPLETED)
        next_step = engine.get_next_step(plan)
        assert next_step is None

    def test_is_plan_finished(self) -> None:
        """Check plan finished state."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", title="Step 2", status=StepStatus.SKIPPED),
        ]
        plan = Plan(id="p1", steps=tuple(steps), status=PlanStatus.COMPLETED)
        assert engine.is_plan_finished(plan) is True

    def test_start_step_dependency_fails(self) -> None:
        """Starting step with unmet dependency fails."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", title="Step 1", status=StepStatus.PENDING),
            PlanStep(id="s2", title="Step 2", status=StepStatus.PENDING, dependencies=("s1",)),
        ]
        plan = engine.create_plan(steps=steps)
        with pytest.raises(ValueError, match="which is pending"):
            engine.start_step(plan, "s2")

    def test_fail_step_adds_error_metadata(self) -> None:
        """Failed step includes error in metadata."""
        engine = PlanningEngine()
        plan = engine.create_plan()
        step_id = plan.steps[0].id
        plan = engine.start_step(plan, step_id)
        plan = engine.fail_step(plan, step_id, error="custom error")
        assert "custom error" in plan.steps[0].metadata.get("error", "")

    def test_current_plan_state(self) -> None:
        """Engine tracks current plan and step."""
        engine = PlanningEngine()
        assert engine.current_plan is None
        assert engine.current_step_id is None

        plan = engine.create_plan()
        assert engine.current_plan == plan
        assert engine.current_step_id is None

        step_id = plan.steps[0].id
        engine.start_step(plan, step_id)
        assert engine.current_step_id == step_id

    def test_create_plan_with_metadata(self) -> None:
        """Create plan with custom metadata."""
        engine = PlanningEngine()
        plan = engine.create_plan(title="Test", metadata={"key": "value"})
        assert plan.metadata["key"] == "value"

    def test_plan_to_dict(self) -> None:
        """Plan serialization."""
        engine = PlanningEngine()
        plan = engine.create_plan(title="Test plan")
        d = plan.to_dict()
        assert d["id"] == plan.id
        assert d["status"] == PlanStatus.PENDING.value
        assert d["step_count"] == 1
        assert len(d["steps"]) == 1

    def test_plan_properties(self) -> None:
        """Plan computed properties."""
        steps = [
            PlanStep(id="s1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", status=StepStatus.FAILED),
            PlanStep(id="s3", status=StepStatus.PENDING),
        ]
        plan = Plan(id="p1", steps=tuple(steps), status=PlanStatus.RUNNING)
        assert plan.step_count == 3
        assert plan.completed_count == 1
        assert plan.failed_count == 1
        assert plan.is_finished is False

    def test_plan_finished_true(self) -> None:
        """Plan is finished when all steps terminal."""
        steps = [
            PlanStep(id="s1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", status=StepStatus.SKIPPED),
        ]
        plan = Plan(steps=tuple(steps), status=PlanStatus.COMPLETED)
        assert plan.is_finished is True

    def test_engine_is_plan_finished(self) -> None:
        """Engine method to check finished."""
        engine = PlanningEngine()
        steps = [
            PlanStep(id="s1", status=StepStatus.COMPLETED),
            PlanStep(id="s2", status=StepStatus.SKIPPED),
        ]
        plan = Plan(steps=tuple(steps), status=PlanStatus.COMPLETED)
        assert engine.is_plan_finished(plan) is True

        steps = [
            PlanStep(id="s1", status=StepStatus.PENDING),
            PlanStep(id="s2", status=StepStatus.COMPLETED),
        ]
        plan = Plan(steps=tuple(steps), status=PlanStatus.RUNNING)
        assert engine.is_plan_finished(plan) is False