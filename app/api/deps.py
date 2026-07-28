"""FastAPI dependency-injection module.

Provides callable dependencies that FastAPI routes use to access kernel
services.  The kernel instance is stored as application state.
"""

from __future__ import annotations

from fastapi import Request

from app.kernel.kernel import Kernel


def get_kernel(request: Request) -> Kernel:
    """FastAPI dependency that returns the Kernel instance.

    Usage::

        @router.get("/health")
        async def health(kernel: Kernel = Depends(get_kernel)):
            ...
    """
    kernel: Kernel | None = request.app.state.kernel
    if kernel is None:
        raise RuntimeError("Kernel not initialized")
    return kernel
