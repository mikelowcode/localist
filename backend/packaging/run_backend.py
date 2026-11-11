"""
PyInstaller entrypoint — Localist backend (port 8001).

Not `main.py`'s own "if __name__" block: that one is a stale dev-only
convenience (reload=True, a string import path that predates the src/
layout restructure), never called by start_localist.sh. A frozen build
needs genuinely different behavior — no reload (doesn't make sense in a
single frozen process), and the real `app` object passed directly rather
than resolved by import string, so uvicorn never needs its own import
machinery to find `localist.main` inside the frozen bundle.

See backend/packaging/README.md for how this gets built.
"""

from __future__ import annotations

import logging

import uvicorn

from localist.log_file import install_file_logging

# Before importing the app, so an import-time failure in the frozen bundle
# is captured too — stdout/stderr go to /dev/null here (§7.30).
install_file_logging("backend")

try:
    from localist.main import app
except Exception:
    logging.getLogger("localist-backend").exception("Backend failed to import")
    raise

if __name__ == "__main__":
    uvicorn.run(
        app,
        host       = "127.0.0.1",
        port       = 8001,
        reload     = False,
        log_level  = "info",
        # Don't let uvicorn install its own stdout handlers: its startup
        # and error lines propagate to the root logger's file handler.
        # Per-request access lines are off — the UI's /health polling
        # would bury everything else.
        log_config = None,
        access_log = False,
    )
