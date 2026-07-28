"""FastAPI application factory.

Wires the kernel into the FastAPI lifespan so that:
- On startup: kernel boots, modules initialize
- On shutdown: kernel shuts down cleanly
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.system import router as system_router
from app.kernel.kernel import Kernel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan that manages the kernel lifecycle."""
    kernel: Kernel = app.state.kernel
    await kernel.boot()
    yield
    await kernel.shutdown()


def create_app(kernel: Kernel | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        kernel: An optional pre-configured kernel.  If not provided, a
            fresh kernel is created.
    """
    if kernel is None:
        kernel = Kernel()

    app = FastAPI(
        title="ATLAS — AI Operating System",
        version=kernel.config.get("app.version"),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store kernel in app state
    app.state.kernel = kernel

    # Register routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")

    return app
