"""Module registry — lifecycle tracking and topological ordering.

**Responsibility:** register modules, validate dependency declarations,
determine boot order (topological sort with cycle detection), and track
state transitions.
"""

from __future__ import annotations

from collections import deque

from app.core.errors import ModuleDependencyError, ModuleNotFoundError_
from app.core.interfaces import Module, ModuleRegistry, ModuleState


class InMemoryModuleRegistry(ModuleRegistry):
    """Module registry backed by in-memory dicts."""

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}
        self._states: dict[str, ModuleState] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, module: Module) -> None:
        """Register *module*.  Raises :class:`ValueError` on duplicate name."""
        if module.name in self._modules:
            raise ValueError(f"Module '{module.name}' is already registered")
        self._modules[module.name] = module
        self._states[module.name] = ModuleState.REGISTERED

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> Module:
        """Get module by name.  Raises :class:`ModuleNotFoundError_`."""
        module = self._modules.get(name)
        if module is None:
            raise ModuleNotFoundError_(
                f"Module '{name}' not found",
                details={"name": name, "registered": list(self._modules.keys())},
            )
        return module

    def all(self) -> list[tuple[str, ModuleState, Module]]:
        """All registered modules with name, state, and instance."""
        return [
            (name, self._states[name], mod)
            for name, mod in self._modules.items()
        ]

    def count(self) -> int:
        return len(self._modules)

    def update_state(self, name: str, state: ModuleState) -> None:
        """Update the runtime state of a registered module."""
        if name not in self._modules:
            raise ModuleNotFoundError_(
                f"Cannot update state for unknown module '{name}'"
            )
        self._states[name] = state

    # ------------------------------------------------------------------
    # Topological sort
    # ------------------------------------------------------------------

    def boot_order(self) -> list[Module]:
        """Topologically sort modules respecting dependency declarations.

        Uses Kahn's algorithm.  Raises :class:`ModuleDependencyError` on:
        - Missing dependencies (declared but not registered)
        - Circular dependencies
        """
        # Build adjacency and in-degree maps
        in_degree: dict[str, int] = {name: 0 for name in self._modules}
        adjacency: dict[str, list[str]] = {name: [] for name in self._modules}

        # Reverse edges: if A depends on B, then B must boot before A
        for name, module in self._modules.items():
            for dep_name in module.manifest.dependencies:
                if dep_name not in self._modules:
                    raise ModuleDependencyError(
                        f"Module '{name}' depends on '{dep_name}' which is not registered",
                        details={
                            "module": name,
                            "missing_dependency": dep_name,
                            "registered": list(self._modules.keys()),
                        },
                    )
                # B -> A means B comes before A
                adjacency.setdefault(dep_name, []).append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

        # Kahn's algorithm
        queue: deque[str] = deque()
        for name, degree in in_degree.items():
            if degree == 0:
                queue.append(name)

        sorted_names: list[str] = []
        while queue:
            current = queue.popleft()
            sorted_names.append(current)
            for neighbor in adjacency.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(self._modules):
            # Some modules are in a cycle
            unresolved = set(self._modules.keys()) - set(sorted_names)
            raise ModuleDependencyError(
                f"Circular dependency detected among modules: {unresolved}",
                details={"unresolved": list(unresolved)},
            )

        return [self._modules[name] for name in sorted_names]
