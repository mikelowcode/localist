"""
LORA — MCP Tool Dispatcher
============================
Was originally a drop-in replacement for the now-deleted ToolDispatcher
(tool_dispatcher.py) at the controller_agent.py dispatch seam; as of Phase
4 (cleanup, 2026-07-03) that legacy class is gone entirely — "file_op",
"url_fetch", and "web_search" are the only tool names Planner ever routes
to tools_to_call (see planner.py's P3/P3b), and all three are served over
the localist-mcp service (mcp_server/, port 8003) via the MCP SSE
transport. Any other tool name is unrecognized and produces an inline
error ToolResult — the one remaining piece of what used to be
ToolDispatcher's "else" branch, ported inline rather than kept as an
excuse to hold onto a whole extra class.

url_fetch (Phase 2): extracts the first http(s):// URL from the
instruction (or context["fetch_url"] if already resolved upstream), calls
the fetch_url MCP tool, and formats the result the same way the legacy
ToolDispatcher._run_url_fetch did. This retired the standalone Fetcher
microservice (port 8002) — fetch_url ports its /extract path in-process on
localist-mcp instead.

web_search (Phase 3): ports ToolDispatcher._run_web_search's query
resolution verbatim (explicit context["web_search_queries"], else derived
from the instruction) and calls the web_search MCP tool once per query, up
to _MAX_WEB_QUERIES. Locked decision: the legacy runtime.infer()
hallucination fallback for a missing LANGSEARCH_API_KEY is gone — that
path now produces a clean success=False ToolResult, same as any other
web_search failure, so controller_agent.py's existing corpus fallback
(Step 3b) is what grounds the answer instead.

Session lifecycle: dispatch() opens one MCP ClientSession (SSE transport)
and reuses it for every tool invocation made during that dispatch() call —
including multiple web_search queries — closing it on the way out
regardless of outcome. Session reuse is scoped to a single dispatch() call
only; it is not persisted across separate HTTP requests/dispatch()
invocations (see MCPToolDispatcher._dispatch_async's docstring).

Reference: §6 of LOCALIST-Architecture.md; Phase 1 MCP migration; Phase 2
url_fetch wiring + Fetcher retirement; Phase 3 web_search migration; Phase
4 cleanup (ToolDispatcher deletion); MCP follow-up (session reuse).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from prompt_builder import ToolResult

logger = logging.getLogger(__name__)

# localist-mcp service endpoint (standalone service on port 8003)
_MCP_SERVER_URL: str = os.environ.get(
    "LOCALIST_MCP_URL", "http://localhost:8003"
) + "/sse"

# file_op_action -> MCP tool name
_FILE_OP_TOOL_MAP: dict[str, str] = {
    "read":   "read_file",
    "write":  "write_file",
    "append": "append_file",
}

# Straightforward http(s):// URL extraction from an instruction — same
# pattern legacy ToolDispatcher._run_url_fetch used.
_URL_RE = re.compile(r"https?://[^\s\"'>]+")

# Maximum number of web_search queries per dispatch call — same cap as
# legacy ToolDispatcher._run_web_search.
_MAX_WEB_QUERIES: int = 3

# Filler prefixes stripped when deriving a single query from the
# instruction — ported verbatim from ToolDispatcher._run_web_search.
_WEB_SEARCH_FILLER_PREFIXES: tuple[str, ...] = (
    "what are the ", "what is the ", "what is ", "find the ",
    "search for ", "look up ", "tell me about ",
)

# file_op action derivation — keyword groups checked in this priority order
# (append > write > read: append/write are less ambiguous signals than a
# bare "read"). Checked against the lowercased instruction; only used when
# context["file_op_action"] is absent.
_FILE_OP_APPEND_KEYWORDS: tuple[str, ...] = ("append", "add to the file", "add this to")
_FILE_OP_WRITE_KEYWORDS:  tuple[str, ...] = ("write", "create", "save", "make a file")
_FILE_OP_READ_KEYWORDS:   tuple[str, ...] = ("read", "open", "show me the file", "what's in the file")

# file_op path derivation — patterns tried in order, first match wins; falls
# back to a bare filename token anywhere in the instruction. Only used when
# context["file_op_path"] is absent.
_FILE_OP_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"name it (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
    re.compile(r"call it (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
    re.compile(r"save (?:it |this )?as (?:.*?\s)?([\w\-]+\.\w+)", re.IGNORECASE),
)
_FILE_OP_PATH_FALLBACK_RE = re.compile(r"\b[\w\-]+\.\w+\b")

# file_op content derivation — patterns tried in order, first match wins.
# Only used when context["file_op_content"] is absent.
_FILE_OP_CONTENT_CODEBLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_FILE_OP_CONTENT_QUOTED_RE    = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_FILE_OP_CONTENT_PHRASE_RE    = re.compile(
    r"(?:with the content|containing|that says)\s+(.*)$", re.IGNORECASE
)


def _derive_file_op_action(instruction: str) -> str:
    lowered = instruction.lower()
    if any(kw in lowered for kw in _FILE_OP_APPEND_KEYWORDS):
        return "append"
    if any(kw in lowered for kw in _FILE_OP_WRITE_KEYWORDS):
        return "write"
    if any(kw in lowered for kw in _FILE_OP_READ_KEYWORDS):
        return "read"
    return "read"


def _derive_file_op_path(instruction: str) -> str:
    for pattern in _FILE_OP_PATH_PATTERNS:
        match = pattern.search(instruction)
        if match:
            return match.group(1).strip()
    match = _FILE_OP_PATH_FALLBACK_RE.search(instruction)
    return match.group(0) if match else ""


def _derive_file_op_content(instruction: str) -> str:
    match = _FILE_OP_CONTENT_CODEBLOCK_RE.search(instruction)
    if match:
        return match.group(1).strip()
    match = _FILE_OP_CONTENT_QUOTED_RE.search(instruction)
    if match:
        return match.group(1) if match.group(1) is not None else match.group(2)
    match = _FILE_OP_CONTENT_PHRASE_RE.search(instruction)
    if match:
        return match.group(1).strip()
    return ""

# FastMCP wraps every raised tool exception as "Error executing tool <name>: <msg>"
# (mcp/server/fastmcp/tools/base.py). Our tool implementations always raise
# ValueError("ERROR: ..."), so stripping this wrapper recovers the exact
# "ERROR: ..." shape ToolDispatcher used to produce — which is what
# controller_agent.py's startswith("ERROR:") slot-6 filter matches on.
_FASTMCP_ERROR_WRAPPER_RE = re.compile(r"^Error executing tool \S+: ")


def _normalize_mcp_error_text(text: str) -> str:
    stripped = _FASTMCP_ERROR_WRAPPER_RE.sub("", text, count=1)
    return stripped if stripped.startswith("ERROR:") else text


class MCPToolDispatcher:
    """
    "file_op", "url_fetch", and "web_search" are served by the localist-mcp
    MCP server; any other tool name is unrecognized (Planner never routes
    tools_to_call to anything else — see planner.py's P3/P3b) and produces
    an inline error ToolResult, same shape the legacy ToolDispatcher's
    "else" branch used to produce.

    Parameters
    ----------
    runtime :
        RuntimeClient. Unused as of Phase 4 — kept as a required parameter
        purely so controller_agent.py's construction call
        (`MCPToolDispatcher(runtime=self._runtime, project_root=...)`)
        doesn't need to change. web_search's runtime.infer() hallucination
        fallback was removed in Phase 3; nothing else here ever used it.
    project_root :
        Accepted for interface stability; not currently used by any tool
        path here. (file_op's actual sandbox root lives on the MCP server
        — see mcp_server/file_ops.py — and is configured independently via
        LOCALIST_MCP_PROJECT_ROOT.)
    mcp_server_url :
        Override the localist-mcp SSE endpoint. Defaults to
        LOCALIST_MCP_URL env var or http://localhost:8003/sse.
    """

    def __init__(
        self,
        runtime:        Any,
        project_root:   Path | str | None = None,
        mcp_server_url: str | None = None,
    ) -> None:
        self._mcp_server_url = mcp_server_url or _MCP_SERVER_URL

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
        Execute the requested tools and return results. Never raises.

        "file_op", "url_fetch", and "web_search" are served over MCP; any
        other tool name is unrecognized and produces an error ToolResult.
        """
        ctx = context or {}
        return asyncio.run(self._dispatch_async(tools_to_call, instruction, ctx))

    async def _dispatch_async(
        self,
        tools_to_call: list[str],
        instruction:   str,
        ctx:           dict[str, Any],
    ) -> list[ToolResult]:
        """
        Open one MCP ClientSession for this dispatch() call, reuse it for
        every tool invocation made during it, and close it cleanly on the
        way out — happy path or not. Scoped to a single dispatch() call;
        not persisted across separate HTTP requests (see module docstring).
        """
        session:       ClientSession | None = None
        connect_error: Exception | None     = None

        async with AsyncExitStack() as stack:
            try:
                session = await self._open_session(stack)
            except Exception as exc:
                logger.warning(
                    "MCPToolDispatcher: localist-mcp unreachable — %s", exc
                )
                session, connect_error = None, exc

            results: list[ToolResult] = []
            for tool_name in tools_to_call:
                if tool_name == "file_op":
                    results.append(
                        await self._run_file_op(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "url_fetch":
                    results.append(
                        await self._run_url_fetch(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "web_search":
                    results.extend(
                        await self._run_web_search(session, connect_error, instruction, ctx)
                    )
                else:
                    logger.warning(
                        "MCPToolDispatcher: unknown tool %r — skipping.", tool_name
                    )
                    results.append(ToolResult(
                        tool_name  = tool_name,
                        parameters = "",
                        result     = f"ERROR: unknown tool '{tool_name}'",
                        success    = False,
                    ))

            _succeeded = sum(1 for r in results if r.success)
            _failed    = len(results) - _succeeded
            logger.info(
                "MCPToolDispatcher: dispatch complete — tools=%s succeeded=%d failed=%d",
                tools_to_call, _succeeded, _failed,
            )
            return results

    async def _open_session(self, stack: AsyncExitStack) -> ClientSession:
        """
        Open the SSE transport + ClientSession for this dispatch() call,
        registering both on `stack` so AsyncExitStack tears them down
        together when the dispatch finishes. Split out from
        _dispatch_async as its own method purely as a test seam — tests
        patch this to hand back a fake session without touching the
        network, while still exercising real connect/teardown behavior in
        live verification.
        """
        read, write = await stack.enter_async_context(sse_client(self._mcp_server_url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # -----------------------------------------------------------------------
    # file_op — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_file_op(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        action = (
            context["file_op_action"] if "file_op_action" in context
            else _derive_file_op_action(instruction)
        )
        rel_path = (
            context["file_op_path"] if "file_op_path" in context
            else _derive_file_op_path(instruction)
        )
        content = (
            context["file_op_content"] if "file_op_content" in context
            else _derive_file_op_content(instruction)
        )
        params_str = f"action={action!r} path={rel_path!r}"

        mcp_tool = _FILE_OP_TOOL_MAP.get(action)
        if mcp_tool is None:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: unknown file_op action '{action}'",
                success    = False,
            )

        if not rel_path:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = "ERROR: file_op_path not provided in context",
                success    = False,
            )

        if session is None:
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        arguments: dict[str, Any] = {"path": rel_path}
        if mcp_tool in ("write_file", "append_file"):
            arguments["content"] = content
        if mcp_tool == "append_file":
            arguments["turn_id"] = context.get("task_id")

        try:
            text, is_error = await self._call_mcp_tool(session, mcp_tool, arguments)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for action=%r path=%r: %s",
                action, rel_path, exc,
            )
            return ToolResult(
                tool_name  = "file_op",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            text = _normalize_mcp_error_text(text)

        return ToolResult(
            tool_name  = "file_op",
            parameters = params_str,
            result     = text,
            success    = not is_error,
        )

    # -----------------------------------------------------------------------
    # url_fetch — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_url_fetch(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult:
        url: str = context.get("fetch_url", "")
        if not url:
            match = _URL_RE.search(instruction)
            url = match.group(0) if match else ""

        if not url:
            logger.warning(
                "MCPToolDispatcher: url_fetch — no URL found in instruction or context."
            )
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = "",
                result     = "ERROR: no URL found in instruction",
                success    = False,
            )

        params_str = f"url={url!r}"

        if session is None:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "fetch_url", {"url": url})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for url_fetch url=%r: %s",
                url, exc,
            )
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "url_fetch",
                parameters = params_str,
                result     = f"ERROR: failed to parse fetch_url response — {exc}",
                success    = False,
            )

        result_text = (
            f"Title: {data.get('title', '')}\n"
            f"Source: {data.get('url', url)}\n"
            f"Words: {data.get('word_count', 0)}\n\n"
            f"{data.get('cleaned_text', '')}"
        )

        logger.info(
            "MCPToolDispatcher: url_fetch complete — url=%r  words=%d  chars=%d",
            url, data.get("word_count", 0), len(result_text),
        )
        return ToolResult(
            tool_name  = "url_fetch",
            parameters = params_str,
            result     = result_text,
            success    = True,
        )

    # -----------------------------------------------------------------------
    # web_search — served by localist-mcp
    # -----------------------------------------------------------------------

    async def _run_web_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Execute web_search for up to _MAX_WEB_QUERIES queries.

        Query resolution order (ported verbatim from
        ToolDispatcher._run_web_search):
          1. context["web_search_queries"] — explicit list (max 3 used)
          2. Derive a single query from the instruction by stripping known
             filler phrases and taking the first 120 characters.
        """
        raw_queries: list[str] = context.get("web_search_queries") or []

        if not raw_queries:
            derived = instruction.strip()
            for filler in _WEB_SEARCH_FILLER_PREFIXES:
                if derived.lower().startswith(filler):
                    derived = derived[len(filler):]
                    break
            raw_queries = [derived[:120]]

        queries = raw_queries[:_MAX_WEB_QUERIES]
        return [
            await self._execute_web_search_query(session, connect_error, query)
            for query in queries
        ]

    async def _execute_web_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
    ) -> ToolResult:
        params_str = f"query={query!r}"

        if session is None:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "web_search", {"query": query})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for web_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "web_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse web_search response — {exc}",
                success    = False,
            )

        return ToolResult(
            tool_name  = "web_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = True,
        )

    async def _call_mcp_tool(
        self, session: ClientSession, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """
        Call an MCP tool on an already-open session. Returns (result_text,
        is_error).

        session.call_tool() internally issues a "tools/list" request the
        first time it validates a successful result's output schema against
        a tool name it hasn't seen yet in this session's cache (see
        mcp.client.session.ClientSession._validate_tool_result) — this is
        the SDK's own bookkeeping, not something we invoke here. With one
        session reused for a whole dispatch() call, that fires at most once
        per dispatch (on the first successful call) instead of once per
        tool call, and it's no longer cancelled mid-flight by an immediate
        session teardown.
        """
        result = await session.call_tool(name, arguments)
        text = "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )
        return text, result.isError
