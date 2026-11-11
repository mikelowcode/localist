"""
Persistent log files for the packaged desktop app (§7.30, 2026-11-11).

The Tauri shell spawns both frozen services with stdout/stderr sent to
/dev/null (deliberately — inheriting LaunchServices' fds crashed the backend
at startup, see localist-ui/src-tauri/src/lib.rs), so before this module
nothing a packaged session logged survived anywhere. Each frozen launcher
(backend/packaging/run_*.py) now calls install_file_logging() first thing,
which adds a size-capped rotating file handler to the root logger:

    ~/Library/Logs/Localist/backend.log   (+ .1 … .3)
    ~/Library/Logs/Localist/mcp.log       (+ .1 … .3)

The Tauri shell's own log (app.log) lands in the same folder. Running from
source (start_localist.sh) never calls this, so dev logging is unchanged —
it still goes to the terminal.

Logs stay on the user's Mac but do contain queries and URLs; API keys are
never logged (they travel in headers / env, never in log lines).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR_ENV: str = "LOCALIST_LOG_DIR"
MAX_BYTES: int = 5 * 1024 * 1024
BACKUP_COUNT: int = 3

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Loggers that emit per-event trace lines at DEBUG — pure noise in a file
# that has to fit in 20 MB. Measured live under LOCALIST_LOG_LEVEL=DEBUG
# (the packaged .env's setting): one news lookup wrote ~35 KB, almost all
# readability's per-node scoring trace and mcp.server.sse's raw JSON-RPC
# dumps (full fetched article text). sse_starlette logs every SSE chunk.
_NOISY_LOGGERS: tuple[str, ...] = (
    "sse_starlette", "httpcore", "mcp.client.sse", "mcp.server.sse", "readability",
)

_active_dir: Path | None = None


def default_log_dir() -> Path:
    override = os.environ.get(LOG_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Logs" / "Localist"


def active_log_dir() -> Path | None:
    """The folder this process is writing its log file to, or None when
    file logging isn't installed (running from source)."""
    return _active_dir


def install_file_logging(name: str, log_dir: Path | None = None) -> Path | None:
    """
    Add a rotating ``<name>.log`` handler to the root logger. Returns the
    log file's path, or None if the folder couldn't be created — logging to
    a file must never stop the service from starting. Idempotent per name.
    """
    global _active_dir
    directory = log_dir or default_log_dir()
    path = directory / f"{name}.log"
    root = logging.getLogger()

    for h in root.handlers:
        if getattr(h, "_localist_log_file", None) == str(path):
            return path

    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8",
        )
    except OSError as exc:
        logging.getLogger(__name__).warning("File logging unavailable (%s): %s", path, exc)
        return None

    handler._localist_log_file = str(path)  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    # The services' own basicConfig() calls become no-ops once a handler
    # exists, so each sets the root level explicitly after loading .env;
    # until then, INFO.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    quiet_noisy_loggers()

    _active_dir = directory
    logging.getLogger(__name__).info("Logging to %s", path)
    return path


def apply_configured_level(level: str | int) -> None:
    """
    Set the root level once a service has loaded its .env. Needed only when
    install_file_logging() ran first — that makes the service's own
    basicConfig() a no-op, level included. Does nothing otherwise, so
    importing a service (e.g. under pytest) never changes the root level.
    """
    if _active_dir is None:
        return
    logging.getLogger().setLevel(level)
    quiet_noisy_loggers()


def quiet_noisy_loggers() -> None:
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.INFO)


def list_log_files(directory: Path) -> list[dict]:
    """Every *.log / *.log.N file in ``directory``, newest first."""
    if not directory.is_dir():
        return []
    files = []
    for p in directory.iterdir():
        if p.is_file() and (p.suffix == ".log" or ".log." in p.name):
            st = p.stat()
            files.append({
                "name":     p.name,
                "size":     st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            })
    files.sort(key=lambda f: f["modified"], reverse=True)
    return files
