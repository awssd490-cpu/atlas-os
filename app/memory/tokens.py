"""Token estimation — rough token counting for memories and text.

Uses a simple 4-char-per-token heuristic with an overhead factor.
This is intentionally approximate — it gives the Token Budget
Manager a baseline to work from.
"""

from __future__ import annotations

import math

from app.memory.memory import Memory


class TokenEstimator:
    """Rough token estimation for memories and text."""

    @staticmethod
    def estimate_text(text: str) -> int:
        """Estimate tokens in plain text."""
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def estimate_memory(memory: Memory) -> int:
        """Estimate tokens consumed by a single memory."""
        total = len(memory.content)
        total += len(memory.memory_type) * 2
        total += len(memory.namespace) * 2
        total += len(memory.source)
        total += len(memory.owner)
        total += sum(len(t) for t in memory.tags)
        total += len(str(memory.metadata))
        return max(1, math.ceil(total / 4) + 40)

    @staticmethod
    def estimate_memories(memories: list[Memory]) -> int:
        """Estimate tokens for a list of memories."""
        return sum(TokenEstimator.estimate_memory(m) for m in memories)
