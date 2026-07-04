"""
Localist MCP Server — file_op tool implementations
====================================================
Ports the file_op logic from ToolDispatcher._file_read / _file_write /
_file_append (tool_dispatcher.py) verbatim: same sandboxing check (resolved
path must be a descendant of project_root), same _MAX_FILE_READ_CHARS
truncation behaviour, same truncation suffix.

Unlike ToolDispatcher, which swallows errors into an "ERROR: ..." string
inside a always-success-shaped ToolResult, these functions raise on failure
so the MCP protocol layer can surface isError=True to the client — see
mcp/server/lowlevel/server.py's call_tool() decorator, which converts any
raised exception into CallToolResult(isError=True, content=[TextContent(text=str(exc))]).
"""

from __future__ import annotations

import os
from pathlib import Path

# Maximum characters read from a file (mirrors tool_dispatcher._MAX_FILE_READ_CHARS —
# kept duplicated rather than imported so this service has no dependency on the
# backend's agent stack).
_MAX_FILE_READ_CHARS: int = 4000

_project_root: Path | None = None


def get_project_root() -> Path:
    """
    Resolve the sandbox root.

    Configurable via the LOCALIST_MCP_PROJECT_ROOT environment variable.
    Defaults to the backend/ directory (parent of this package), matching
    where main.py and start_localist.sh run the sibling services from.
    """
    global _project_root
    if _project_root is None:
        env_root = os.environ.get("LOCALIST_MCP_PROJECT_ROOT")
        _project_root = (
            Path(env_root).resolve()
            if env_root
            else Path(__file__).resolve().parent.parent
        )
    return _project_root


def set_project_root(path: Path | str) -> None:
    """Override the sandbox root. Used at startup and by tests."""
    global _project_root
    _project_root = Path(path).resolve()


def _sandbox_resolve(rel_path: str) -> Path:
    """
    Resolve rel_path against project_root and enforce sandboxing.

    Raises ValueError if the path is invalid or escapes project_root —
    same checks and same error text as ToolDispatcher._run_file_op.
    """
    project_root = get_project_root()
    try:
        resolved = (project_root / rel_path).resolve()
    except Exception as exc:
        raise ValueError(f"ERROR: invalid path — {exc}") from exc

    if not str(resolved).startswith(str(project_root)):
        raise ValueError(
            "ERROR: path traversal outside project_root is not permitted"
        )
    return resolved


def read_file(path: str) -> str:
    """Read a UTF-8 text file relative to project_root, sandboxed."""
    resolved = _sandbox_resolve(path)

    if not resolved.exists():
        raise ValueError(f"ERROR: file not found — {resolved}")

    text = resolved.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(text) > _MAX_FILE_READ_CHARS:
        text = text[:_MAX_FILE_READ_CHARS]
        truncated = True
    suffix = "\n… [truncated]" if truncated else ""
    return text + suffix


def write_file(path: str, content: str) -> str:
    """Write content to a UTF-8 text file relative to project_root, sandboxed."""
    resolved = _sandbox_resolve(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ERROR: file_op failed — {exc}") from exc
    return f"OK: wrote {len(content)} characters to {resolved.name}"


def append_file(path: str, content: str) -> str:
    """Append content to a UTF-8 text file relative to project_root, sandboxed."""
    resolved = _sandbox_resolve(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as f:
            f.write(content)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"ERROR: file_op failed — {exc}") from exc
    return f"OK: appended {len(content)} characters to {resolved.name}"
