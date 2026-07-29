"""Tests for checkpoint and resume."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.checkpoint import CheckpointManager, RuntimeCheckpoint
from app.agent.config import AgentConfig
from app.agent.plan import Plan, PlanStatus, PlanStep, StepStatus


# ---------------------------------------------------------------------------
# RuntimeCheckpoint tests
# ---------------------------------------------------------------------------


class TestRuntimeCheckpoint:
    def test_create_checkpoint(self) -> None:
        checkpoint = RuntimeCheckpoint(
            checkpoint_id="cp_1",
            iteration=3,
            tool_calls_executed=5,
            provider_requests=4,
            messages=({"role": "user", "content": "hello"},),
            system_prompt="Be helpful.",
        )
        assert checkpoint.checkpoint_id == "cp_1"
        assert checkpoint.iteration == 3
        assert checkpoint.tool_calls_executed == 5
        assert checkpoint.version == "1.0"

    def test_to_dict(self) -> None:
        checkpoint = RuntimeCheckpoint(
            checkpoint_id="cp_1",
            iteration=2,
            messages=({"role": "user", "content": "hi"},),
        )
        d = checkpoint.to_dict()
        assert d["checkpoint_id"] == "cp_1"
        assert d["iteration"] == 2
        assert len(d["messages"]) == 1

    def test_from_dict(self) -> None:
        data = {
            "checkpoint_id": "cp_2",
            "version": "1.0",
            "created_at": 1000.0,
            "iteration": 5,
            "tool_calls_executed": 10,
            "provider_requests": 8,
            "messages": [],
            "system_prompt": "",
            "current_plan": None,
            "memory_enabled": True,
            "metadata": {},
        }
        checkpoint = RuntimeCheckpoint.from_dict(data)
        assert checkpoint.checkpoint_id == "cp_2"
        assert checkpoint.iteration == 5
        assert checkpoint.tool_calls_executed == 10

    def test_from_dict_missing_checkpoint_id_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_id is required"):
            RuntimeCheckpoint.from_dict({"version": "1.0", "checkpoint_id": ""})

    def test_from_dict_wrong_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported checkpoint version"):
            RuntimeCheckpoint.from_dict({"version": "0.5", "checkpoint_id": "x"})

    def test_defaults(self) -> None:
        checkpoint = RuntimeCheckpoint(checkpoint_id="cp_3")
        assert checkpoint.iteration == 0
        assert checkpoint.messages == ()
        assert checkpoint.current_plan is None


class TestCheckpointManager:
    def test_create_checkpoint(self) -> None:
        manager = CheckpointManager()
        cp = manager.create_checkpoint(
            iteration=1,
            tool_calls_executed=2,
        )
        assert cp.iteration == 1
        assert cp.tool_calls_executed == 2
        assert cp.checkpoint_id != ""

    def test_save_and_load(self) -> None:
        manager = CheckpointManager()
        cp = manager.create_checkpoint(iteration=1)
        loaded = manager.load_checkpoint(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.iteration == 1

    def test_load_nonexistent(self) -> None:
        manager = CheckpointManager()
        assert manager.load_checkpoint("nonexistent") is None

    def test_list_checkpoints(self) -> None:
        manager = CheckpointManager()
        cp1 = manager.create_checkpoint(iteration=1)
        cp2 = manager.create_checkpoint(iteration=2)
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2

    def test_delete_checkpoint(self) -> None:
        manager = CheckpointManager()
        cp = manager.create_checkpoint(iteration=1)
        assert manager.delete_checkpoint(cp.checkpoint_id) is True
        assert manager.load_checkpoint(cp.checkpoint_id) is None

    def test_delete_nonexistent(self) -> None:
        manager = CheckpointManager()
        assert manager.delete_checkpoint("ghost") is False

    def test_clear(self) -> None:
        manager = CheckpointManager()
        manager.create_checkpoint(iteration=1)
        manager.create_checkpoint(iteration=2)
        manager.clear()
        assert len(manager.list_checkpoints()) == 0

    def test_restore_checkpoint(self) -> None:
        manager = CheckpointManager()
        cp = manager.create_checkpoint(iteration=3)
        data = cp.to_dict()
        restored = manager.restore_checkpoint(data)
        assert restored.iteration == 3
        assert restored.checkpoint_id == cp.checkpoint_id


class TestCheckpointPlanSerialization:
    def test_plan_from_dict(self) -> None:
        plan_dict = {
            "id": "plan_1",
            "status": "running",
            "steps": [
                {
                    "id": "s1",
                    "title": "Step 1",
                    "status": "completed",
                    "dependencies": [],
                },
                {
                    "id": "s2",
                    "title": "Step 2",
                    "status": "pending",
                    "dependencies": ["s1"],
                },
            ],
        }
        plan = CheckpointManager.plan_from_dict(plan_dict)
        assert plan is not None
        assert plan.id == "plan_1"
        assert plan.status == PlanStatus.RUNNING
        assert len(plan.steps) == 2
        assert plan.steps[0].status == StepStatus.COMPLETED
        assert plan.steps[1].dependencies == ("s1",)

    def test_plan_from_none(self) -> None:
        assert CheckpointManager.plan_from_dict(None) is None

    def test_plan_from_empty_dict(self) -> None:
        plan = CheckpointManager.plan_from_dict({})
        assert plan is not None
        assert plan.id == ""
        assert len(plan.steps) == 0


class TestCheckpointWithPlan:
    def test_create_checkpoint_with_plan(self) -> None:
        plan = Plan(
            id="p1",
            steps=(
                PlanStep(id="s1", title="Step 1", status=StepStatus.COMPLETED),
                PlanStep(id="s2", title="Step 2", status=StepStatus.RUNNING),
            ),
            status=PlanStatus.RUNNING,
        )
        manager = CheckpointManager()
        cp = manager.create_checkpoint(
            iteration=2,
            current_plan=plan,
        )
        assert cp.current_plan is not None
        assert cp.current_plan["id"] == "p1"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestCheckpointConfig:
    def test_default_checkpoint_enabled(self) -> None:
        config = AgentConfig.default()
        assert config.checkpoint_enabled is True

    def test_default_checkpoint_frequency(self) -> None:
        config = AgentConfig.default()
        assert config.checkpoint_frequency == "manual"

    def test_custom_frequency(self) -> None:
        config = AgentConfig(checkpoint_frequency="iteration")
        assert config.checkpoint_frequency == "iteration"

    def test_invalid_frequency_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_frequency"):
            AgentConfig(checkpoint_frequency="invalid")
