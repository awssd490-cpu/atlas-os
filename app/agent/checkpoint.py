"""Checkpoint models for the Agent Runtime.

A ``RuntimeCheckpoint`` captures the agent's execution state at a point
in time so it can be serialized, stored, and resumed later.

Only runtime state is serialized — not provider instances, tool runtime
instances, listeners, or the event dispatcher.  Those are restored through
dependency injection.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agent.plan import Plan, PlanStatus, PlanStep, StepStatus


CHECKPOINT_VERSION = "1.0"


@dataclass(frozen=True)
class RuntimeCheckpoint:
    """A snapshot of agent runtime state for checkpoint/resume.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        version: Schema version for forward/backward compatibility.
        created_at: Unix timestamp when the checkpoint was created.
        iteration: The current iteration count.
        tool_calls_executed: Number of tool calls executed so far.
        provider_requests: Number of provider requests made so far.
        messages: The conversation messages (serialized as dicts).
        system_prompt: The system prompt string.
        current_plan: The current ``Plan``, if planning is enabled.
        memory_enabled: Whether memory was enabled.
        metadata: Optional structured metadata.
    """

    checkpoint_id: str = ""
    version: str = CHECKPOINT_VERSION
    created_at: float = 0.0
    iteration: int = 0
    tool_calls_executed: int = 0
    provider_requests: int = 0
    messages: tuple[dict[str, Any], ...] = ()
    system_prompt: str = ""
    current_plan: dict[str, Any] | None = None
    memory_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the checkpoint to a plain dict.

        Returns:
            A JSON-serializable dict.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "version": self.version,
            "created_at": self.created_at,
            "iteration": self.iteration,
            "tool_calls_executed": self.tool_calls_executed,
            "provider_requests": self.provider_requests,
            "messages": list(self.messages),
            "system_prompt": self.system_prompt,
            "current_plan": self.current_plan,
            "memory_enabled": self.memory_enabled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeCheckpoint":
        """Deserialize a checkpoint from a plain dict.

        Args:
            data: A dict previously produced by ``to_dict()``.

        Returns:
            A ``RuntimeCheckpoint``.

        Raises:
            ValueError: If the version is incompatible or required
                fields are missing.
        """
        version = data.get("version", "")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {version!r} "
                f"(expected {CHECKPOINT_VERSION!r})"
            )

        checkpoint_id = data.get("checkpoint_id", str(uuid.uuid4())[:8])
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required")

        return cls(
            checkpoint_id=checkpoint_id,
            version=version,
            created_at=data.get("created_at", time.time()),
            iteration=data.get("iteration", 0),
            tool_calls_executed=data.get("tool_calls_executed", 0),
            provider_requests=data.get("provider_requests", 0),
            messages=tuple(data.get("messages", [])),
            system_prompt=data.get("system_prompt", ""),
            current_plan=data.get("current_plan"),
            memory_enabled=data.get("memory_enabled", True),
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """Manages checkpoint creation and restoration.

    Storage-agnostic — serializes to/from plain dicts.
    Does not implement any persistence backend.

    Usage::

        manager = CheckpointManager()
        checkpoint = manager.create_checkpoint(runtime_state)
        data = checkpoint.to_dict()
        restored = manager.restore_checkpoint(data)
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, RuntimeCheckpoint] = {}

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        *,
        iteration: int = 0,
        tool_calls_executed: int = 0,
        provider_requests: int = 0,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        current_plan: Plan | None = None,
        memory_enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeCheckpoint:
        """Create a new checkpoint from runtime state.

        Args:
            iteration: Current iteration count.
            tool_calls_executed: Total tool calls executed.
            provider_requests: Total provider requests made.
            messages: Conversation messages as dicts.
            system_prompt: System prompt string.
            current_plan: Current plan (if any).
            memory_enabled: Whether memory was enabled.
            metadata: Optional metadata.

        Returns:
            A new ``RuntimeCheckpoint``.
        """
        plan_dict = current_plan.to_dict() if current_plan is not None else None

        checkpoint = RuntimeCheckpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            created_at=time.time(),
            iteration=iteration,
            tool_calls_executed=tool_calls_executed,
            provider_requests=provider_requests,
            messages=tuple(messages or []),
            system_prompt=system_prompt,
            current_plan=plan_dict,
            memory_enabled=memory_enabled,
            metadata=metadata or {},
        )

        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint: RuntimeCheckpoint) -> None:
        """Store a checkpoint in the in-memory store.

        Args:
            checkpoint: The checkpoint to store.
        """
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> RuntimeCheckpoint | None:
        """Load a checkpoint from the in-memory store.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            The ``RuntimeCheckpoint`` or ``None`` if not found.
        """
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> list[RuntimeCheckpoint]:
        """Return all stored checkpoints."""
        return list(self._checkpoints.values())

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Remove a checkpoint from the in-memory store.

        Args:
            checkpoint_id: The checkpoint identifier.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()

    # ------------------------------------------------------------------
    # Restoration
    # ------------------------------------------------------------------

    def restore_checkpoint(self, data: dict[str, Any]) -> RuntimeCheckpoint:
        """Restore a checkpoint from a serialized dict.

        Args:
            data: The serialized checkpoint data.

        Returns:
            A validated ``RuntimeCheckpoint``.

        Raises:
            ValueError: If validation fails.
        """
        return RuntimeCheckpoint.from_dict(data)

    @staticmethod
    def plan_from_dict(plan_dict: dict[str, Any] | None) -> Plan | None:
        """Convert a serialized plan dict back to a ``Plan`` object.

        Args:
            plan_dict: The plan dict from a checkpoint.

        Returns:
            A ``Plan`` or ``None``.
        """
        if plan_dict is None:
            return None

        steps: list[PlanStep] = []
        for step_data in plan_dict.get("steps", []):
            status_str = step_data.get("status", "pending")
            try:
                status = StepStatus(status_str)
            except ValueError:
                status = StepStatus.PENDING

            steps.append(PlanStep(
                id=step_data.get("id", ""),
                title=step_data.get("title", ""),
                description=step_data.get("description", ""),
                status=status,
                dependencies=tuple(step_data.get("dependencies", [])),
                metadata=step_data.get("metadata", {}),
            ))

        status_str = plan_dict.get("status", "pending")
        try:
            plan_status = PlanStatus(status_str)
        except ValueError:
            plan_status = PlanStatus.PENDING

        return Plan(
            id=plan_dict.get("id", ""),
            steps=tuple(steps),
            status=plan_status,
            metadata=plan_dict.get("metadata", {}),
        )
