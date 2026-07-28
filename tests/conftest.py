"""Test configuration and fixtures for ATLAS integration tests.

Uses a test kernel with testing-specific settings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.config.settings import AtlasSettings
from app.core.interfaces import Module
from app.core.manifest import ModuleManifest
from app.kernel.builder import KernelBuilder
from app.kernel.kernel import Kernel


# ---------------------------------------------------------------------------
# Test settings — use testing-specific values
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> AtlasSettings:
    """Settings overridden for the test environment."""
    return AtlasSettings(
        app={"name": "atlas-test", "environment": "testing", "debug": False},
        server={"port": 9099, "reload": False},
        logging={"level": "ERROR", "format": "json"},
    )


# ---------------------------------------------------------------------------
# Minimal no-op module for test
# ---------------------------------------------------------------------------


class _SilentModule(Module):
    """A module that does nothing — used to exercise the boot path."""

    def __init__(self, name: str = "mod_silent") -> None:
        super().__init__()
        self._manifest = ModuleManifest(name=name, version="1.0.0")

    @property
    def manifest(self) -> ModuleManifest:
        return self._manifest


# ---------------------------------------------------------------------------
# Kernel fixture with a single silent module
# ---------------------------------------------------------------------------


@pytest.fixture
def test_kernel(test_settings: AtlasSettings) -> Kernel:
    """Build a bootable test kernel."""
    return (
        KernelBuilder()
        .with_settings(test_settings)
        .with_module(_SilentModule("mod_test"))
        .build()
    )


# ---------------------------------------------------------------------------
# FastAPI app backed by the test kernel
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(test_kernel: Kernel) -> AsyncIterator[AsyncClient]:
    """Create an httpx AsyncClient backed by a booted kernel.

    The kernel is booted *before* creating the app so the kernel is
    running when the test queries it.
    """
    await test_kernel.boot()
    app = create_app(kernel=test_kernel)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
    await test_kernel.shutdown()
