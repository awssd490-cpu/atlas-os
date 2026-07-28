"""ATLAS entry point.

Boots the kernel, loads the FastAPI app, and starts the HTTP server.

Usage:
    uv run python main.py
    uv run uvicorn main:app --reload
"""

from __future__ import annotations

import uvicorn

from app.api.app import create_app

app = create_app()


def main() -> None:
    """Run the ATLAS HTTP server with kernel-managed lifecycle."""
    config = app.state.kernel.config
    host = config.get("server.host")
    port = config.get("server.port")
    reload_enabled = config.get("server.reload")
    workers = config.get("server.workers")
    timeout_keepalive = config.get("server.timeout_keepalive")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        workers=workers,
        timeout_keepalive=timeout_keepalive,
        log_level="info",
    )


if __name__ == "__main__":
    main()
