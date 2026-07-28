"""KernelBuilder — fluent construction for tests and entry points.

Usage::

    kernel = KernelBuilder() \\
        .with_settings(AtlasSettings(app={"environment": "testing"})) \\
        .with_module(MyModule()) \\
        .build()

    await kernel.boot()
"""

from __future__ import annotations

from app.config.settings import AtlasSettings
from app.core.interfaces import Module
from app.kernel.kernel import Kernel


class KernelBuilder:
    """Fluent builder for constructing and populating a Kernel."""

    def __init__(self) -> None:
        self._settings: AtlasSettings | None = None
        self._modules: list[Module] = []

    def with_settings(self, settings: AtlasSettings) -> "KernelBuilder":
        """Use a custom settings object (overrides env defaults)."""
        self._settings = settings
        return self

    def with_module(self, module: Module) -> "KernelBuilder":
        """Add a module to register at build time."""
        self._modules.append(module)
        return self

    def build(self) -> Kernel:
        """Construct and return the Kernel with all registered modules."""
        kernel = Kernel(settings=self._settings)
        for module in self._modules:
            kernel.register(module)
        return kernel
