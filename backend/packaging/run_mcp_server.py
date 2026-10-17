"""
PyInstaller entrypoint — localist-mcp (port 8003).

Same reasoning as run_backend.py: not mcp_server/main.py's own stale
"if __name__" dev block (reload=True, a pre-restructure import string).

See backend/packaging/README.md for how this gets built.
"""

from __future__ import annotations

import os

import uvicorn

from localist.mcp_server.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host      = "127.0.0.1",
        # Override only for running a frozen build alongside an installed
        # Localist (e.g. smoke-testing a local build) — pair with
        # LOCALIST_MCP_URL on the backend side.
        port      = int(os.environ.get("LOCALIST_MCP_PORT", "8003")),
        reload    = False,
        log_level = "info",
    )
