"""
LORA — Tool Dispatcher
=======================
Executes tool calls specified in a RoutingPlan and returns a list of
ToolResult objects for injection into PromptBuilder slot 6.

Tools implemented
-----------------
web_search : Calls runtime.web_search(query) if available; falls back
             to a bounded infer() call that returns structured results.
             Max 3 queries per dispatch call.

file_op    : Read, write, or append to local files. All paths are
             resolved relative to project_root and sandboxed — no
             path traversal outside project_root is permitted.

Reference: §6 of LORA-Architecture.md
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from prompt_builder import ToolResult

logger = logging.getLogger(__name__)

# Maximum characters read from a file (keeps slot 6 within budget).
_MAX_FILE_READ_CHARS: int = 4000

# Maximum number of web_search queries per dispatch call.
_MAX_WEB_QUERIES: int = 3

# System prompt for the web_search fallback inference call.
_WEB_SEARCH_FALLBACK_SYSTEM = (
    "You are a search assistant. Given a search query, return 2–3 concise, "
    "factual bullet points summarising what a web search would likely find. "
    "Format each bullet as: • <fact>. No preamble. No URLs. Plain text only."
)


# ---------------------------------------------------------------------------
# 6.1 — ToolDispatcher interface
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Executes tool calls and returns ToolResult objects for slot 6.

    The dispatcher is stateless between calls. It holds a reference to
    the runtime (for web_search) and a project_root (for file_op sandboxing).

    Parameters
    ----------
    runtime :
        RuntimeClient. Used for web_search (real or fallback inference).
    project_root :
        Absolute path to the project root. All file_op paths are resolved
        relative to this directory. Defaults to the current working directory.
    """

    def __init__(
        self,
        runtime:      Any,
        project_root: Path | str | None = None,
    ) -> None:
        self._runtime      = runtime
        self._project_root = Path(project_root or Path.cwd()).resolve()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def dispatch(
        self,
        tools_to_call: list[str],
        instruction:   str,
        context:       dict[str, Any] | None = None,
    ) -> list[ToolResult]:
        """
        Execute the requested tools and return results.

        Parameters
        ----------
        tools_to_call :
            Ordered list of tool names from RoutingPlan.tools_to_call.
            Supported values: "web_search", "file_op".
            Unknown tool names produce an error ToolResult and are skipped.
        instruction :
            The original user instruction. Used to derive web_search queries
            and file_op parameters when not explicitly provided in context.
        context :
            Optional task context dict. May contain:
              "web_search_queries" : list[str] — explicit queries (max 3)
              "file_op_action"     : "read" | "write" | "append"
              "file_op_path"       : str — relative path from project_root
              "file_op_content"    : str — content for write/append

        Returns
        -------
        list[ToolResult]
            One ToolResult per tool call. Never raises.
        """
        ctx     = context or {}
        results: list[ToolResult] = []

        for tool_name in tools_to_call:
            if tool_name == "web_search":
                results.extend(self._run_web_search(instruction, ctx))
            elif tool_name == "file_op":
                results.append(self._run_file_op(instruction, ctx))
            else:
                logger.warning(
                    "ToolDispatcher: unknown tool %r — skipping.", tool_name
                )
                results.append(ToolResult(
                    tool_name  = tool_name,
                    parameters = "",
                    result     = f"ERROR: unknown tool '{tool_name}'",
                ))

        return results

    # -----------------------------------------------------------------------
    # 6.2 — web_search tool
    # -----------------------------------------------------------------------

    def _run_web_search(
        self,
        instruction: str,
        context:     dict[str, Any],
    ) -> list[ToolResult]:
        """
        Execute web_search for up to _MAX_WEB_QUERIES queries.

        Query resolution order:
          1. context["web_search_queries"] — explicit list (max 3 used)
          2. Derive a single query from the instruction by stripping
             known filler phrases and taking the first 120 characters.

        For each query:
          - If runtime has a web_search(query) method → call it.
          - Otherwise → call runtime.infer() with _WEB_SEARCH_FALLBACK_SYSTEM.

        Returns a list of ToolResult (one per query).
        """
        raw_queries: list[str] = context.get("web_search_queries") or []

        if not raw_queries:
            # Derive a query from the instruction
            derived = instruction.strip()
            # Strip common filler prefixes
            for filler in (
                "what are the ", "what is the ", "what is ", "find the ",
                "search for ", "look up ", "tell me about ",
            ):
                if derived.lower().startswith(filler):
                    derived = derived[len(filler):]
                    break
            raw_queries = [derived[:120]]

        queries = raw_queries[:_MAX_WEB_QUERIES]
        results: list[ToolResult] = []

        for query in queries:
            results.append(self._execute_single_search(query))

        return results

    def _execute_single_search(self, query: str) -> ToolResult:
        """Run one web_search query. Never raises."""
        try:
            if hasattr(self._runtime, "web_search") and callable(
                getattr(self._runtime, "web_search")
            ):
                logger.debug(
                    "ToolDispatcher: web_search (real) query=%r.", query
                )
                raw = self._runtime.web_search(query)
            else:
                logger.debug(
                    "ToolDispatcher: web_search (fallback infer) query=%r.",
                    query,
                )
                raw = self._runtime.infer(
                    system      = _WEB_SEARCH_FALLBACK_SYSTEM,
                    prompt      = f"Search query: {query}",
                    max_tokens  = 120,
                    temperature = 0.2,
                )
            logger.info(
                "ToolDispatcher: web_search complete for query=%r "
                "result_chars=%d.", query, len(str(raw)),
            )
            return ToolResult(
                tool_name  = "web_search",
                parameters = f"query={query!r}",
                result     = str(raw).strip(),
            )
        except Exception as exc:
            logger.warning(
                "ToolDispatcher: web_search failed for query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "web_search",
                parameters = f"query={query!r}",
                result     = f"ERROR: web_search failed — {exc}",
            )

    # -----------------------------------------------------------------------
    # 6.3 — file_op tool
    # -----------------------------------------------------------------------

    def _run_file_op(
        self,
        instruction: str,
        context:     dict[str, Any],
    ) -> ToolResult:
        """
        Execute a file operation: read, write, or append.

        Parameter resolution:
          - action  : context["file_op_action"] or "read"
          - path    : context["file_op_path"] — required; resolved relative
                      to project_root and sandboxed
          - content : context["file_op_content"] — required for write/append

        Sandboxing: the resolved absolute path must be a descendant of
        project_root. Any attempt to escape (e.g. ../../etc/passwd) returns
        an error ToolResult without touching the filesystem.

        Returns a single ToolResult. Never raises.
        """
        action   = context.get("file_op_action", "read")
        rel_path = context.get("file_op_path", "")
        content  = context.get("file_op_content", "")

        if not rel_path:
            return ToolResult(
                tool_name  = "file_op",
                parameters = f"action={action!r} path=<missing>",
                result     = "ERROR: file_op_path not provided in context",
            )

        # Resolve and sandbox
        try:
            resolved = (self._project_root / rel_path).resolve()
        except Exception as exc:
            return ToolResult(
                tool_name  = "file_op",
                parameters = f"action={action!r} path={rel_path!r}",
                result     = f"ERROR: invalid path — {exc}",
            )

        if not str(resolved).startswith(str(self._project_root)):
            logger.warning(
                "ToolDispatcher: file_op path traversal blocked — "
                "resolved=%s project_root=%s", resolved, self._project_root,
            )
            return ToolResult(
                tool_name  = "file_op",
                parameters = f"action={action!r} path={rel_path!r}",
                result     = "ERROR: path traversal outside project_root is not permitted",
            )

        params_str = f"action={action!r} path={str(resolved)!r}"

        try:
            if action == "read":
                return self._file_read(resolved, params_str)
            elif action == "write":
                return self._file_write(resolved, content, params_str)
            elif action == "append":
                return self._file_append(resolved, content, params_str)
            else:
                return ToolResult(
                    tool_name  = "file_op",
                    parameters = params_str,
                    result     = f"ERROR: unknown file_op action '{action}'",
                )
        except Exception as exc:
            logger.warning(
                "ToolDispatcher: file_op failed — action=%r path=%s: %s",
                action, resolved, exc,
            )
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: file_op failed — {exc}",
            )

    def _file_read(self, path: Path, params_str: str) -> ToolResult:
        if not path.exists():
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: file not found — {path}",
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        truncated = False
        if len(text) > _MAX_FILE_READ_CHARS:
            text      = text[:_MAX_FILE_READ_CHARS]
            truncated = True
        suffix = "\n… [truncated]" if truncated else ""
        logger.info(
            "ToolDispatcher: file_op read — path=%s chars=%d truncated=%s",
            path, len(text), truncated,
        )
        return ToolResult(
            tool_name  = "file_op",
            parameters = params_str,
            result     = text + suffix,
        )

    def _file_write(
        self, path: Path, content: str, params_str: str
    ) -> ToolResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("ToolDispatcher: file_op write — path=%s", path)
        return ToolResult(
            tool_name  = "file_op",
            parameters = params_str,
            result     = f"OK: wrote {len(content)} characters to {path.name}",
        )

    def _file_append(
        self, path: Path, content: str, params_str: str
    ) -> ToolResult:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        logger.info("ToolDispatcher: file_op append — path=%s", path)
        return ToolResult(
            tool_name  = "file_op",
            parameters = params_str,
            result     = f"OK: appended {len(content)} characters to {path.name}",
        )
