"""Universal Agent Runtime.

The ``AgentRuntime`` orchestrates the reasoning loop between a provider
and the tool runtime.  It coordinates existing systems — it does not
contain provider or tool logic itself.

Architecture::

    Application
         │
         ▼
    AgentRuntime         ← owns the reasoning loop
         │
         ├─ Provider Runtime    ← generates responses
         ├─ Tool Runtime        ← executes tools
         └─ Integration Layer   ← converts messages ↔ tool calls/results

Usage::

    from app.agent import AgentRuntime, AgentConfig

    runtime = AgentRuntime(
        provider=my_provider,
        tool_runtime=my_tool_runtime,
        config=AgentConfig(max_iterations=5),
    )
    response = await runtime.run(messages)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.runtime import AgentRuntime
    from app.agent.config import AgentConfig
    from app.agent.errors import (
        AgentError,
        IterationLimitExceeded,
        ToolCallLimitExceeded,
        ProviderExecutionError,
    )

__all__ = [
    "AgentRuntime",
    "AgentConfig",
    "AgentError",
    "IterationLimitExceeded",
    "ToolCallLimitExceeded",
    "ProviderExecutionError",
]
