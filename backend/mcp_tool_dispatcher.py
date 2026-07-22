"""
LORA — MCP Tool Dispatcher
============================
Was originally a drop-in replacement for the now-deleted ToolDispatcher
(tool_dispatcher.py) at the controller_agent.py dispatch seam; as of Phase
4 (cleanup, 2026-07-03) that legacy class is gone entirely — "file_op",
"url_fetch", "web_search", "research", "chart", and "news_search" are the
only tool names Planner ever routes to tools_to_call (see planner.py's
P3/P3-news/P3b), and all are served over the localist-mcp service
(mcp_server/, port 8003) via the MCP SSE transport (research is a
client-side loop over the same web_search/url_fetch MCP tools, not a
distinct MCP tool of its own). Any other tool name is unrecognized and
produces an inline error ToolResult — the one remaining piece of what used
to be ToolDispatcher's "else" branch, ported inline rather than kept as an
excuse to hold onto a whole extra class.

news_search (news-query-routing plan, 2026-07-22): tries the NewsAPI-backed
news_search MCP tool first; on a miss (NewsAPI empty/errored result — see
mcp_server/news_search.py's is_miss) or on localist-mcp being unreachable,
falls through to the same _execute_web_search_query() helper web_search
already uses, reusing the original derived query rather than re-deriving a
generic one. Both the failed tier-1 attempt and the fallback attempt are
returned (not just the winner) so Slot 5 and controller_agent.py's Step 3b
corpus fallback see the full picture — same "return every ToolResult
produced, not just the winning one" convention _run_research_loop already
established. The fallback ToolResult is retagged tool_name="news_search:
brave_fallback" for provenance (so a transcript/log makes it visible when
NewsAPI's key/quota needs attention); Step 3b's web_search-failure check
was extended with an `r.tool_name.startswith("news_search")` clause to
keep matching it after the rename, mirroring the `or r.tool_name ==
"research"` clause already added there for the same reason.

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

research: a bounded search/evaluate/reformulate/fetch loop
(_run_research_loop) that Planner routes to instead of "web_search" when
the instruction's cosine similarity to planner.py's research_intent
template group clears _RESEARCH_INTENT_THRESHOLD (gated behind
LOCALIST_RESEARCH_LOOP_ENABLED, off by default) — for requests that need a
specific, extractable fact (price, spec, plan tier) run down rather than a
single search-and-answer. Up to _MAX_RESEARCH_ITERATIONS rounds of
web_search, each followed by a cheap runtime.infer() yes/no gate check
(and, if inconclusive, a url_fetch of the top candidate result re-checked
against the same gate) and, on failure, a runtime.infer() query
reformulation before retrying. Every ToolResult produced along the way is
returned — not just the winning one — so it drops into the same
dispatched_tool_results handling web_search already uses.

chart (_run_chart): promotes diagnostics/diag_shadow_chart_toolcall_v4_full.
py's measured pipeline to production — a bounded runtime.infer() call with
chart_tool_schema.SYSTEM_PROMPT_FEWSHOT at temperature=0.0 to extract
generate_chart arguments from the instruction, repaired via
json_envelope_repair.repair_envelope() and validated via
chart_tool_schema.validate_chart_arguments(); on a malformed envelope, one
retry at temperature=0.3 (final). On success, dispatches to the
generate_chart MCP tool and returns a ToolResult whose .result is only the
tool's "summary" field (png_path/chart_config ride in .artifact instead —
see prompt_builder.ToolResult.artifact — never in the prompt-facing
.result). On failure (post-retry still malformed, schema-invalid, or the
model legitimately declines), returns None rather than an ERROR-shaped
ToolResult — a deliberate deviation from every other tool here, see
_dispatch_async's chart branch for why.

Session lifecycle: dispatch() opens one MCP ClientSession (SSE transport)
and reuses it for every tool invocation made during that dispatch() call —
including multiple web_search queries and a research loop's internal
search/fetch calls — closing it on the way out regardless of outcome.
Session reuse is scoped to a single dispatch() call only; it is not
persisted across separate HTTP requests/dispatch() invocations (see
MCPToolDispatcher._dispatch_async's docstring).

Reference: §6 of LOCALIST-Architecture.md; Phase 1 MCP migration; Phase 2
url_fetch wiring + Fetcher retirement; Phase 3 web_search migration; Phase
4 cleanup (ToolDispatcher deletion); MCP follow-up (session reuse); research
loop addition (2026-07-16).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from chart_tool_schema import KNOWN_TOOL_NAMES, SYSTEM_PROMPT_FEWSHOT, validate_chart_arguments
from json_envelope_repair import repair_envelope
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
#
# 2026-07-16: ] and ) added to the excluded-character class after a live
# research loop run confirmed mcp_server/web_search.py's result formatting
# (f"• {title}\n  {body}\n  [{url}]") — every URL wrapped in literal
# [...] — caused this regex to capture the trailing "]" as part of the URL
# (e.g. ".../apple]"), which then 404'd when passed to url_fetch. Markdown
# links and parenthetical citations wrap URLs in ()/[] the same way, so
# this is a shared-regex fix, not a research-loop-specific one — this
# pattern also backs _run_url_fetch's instruction-text extraction, where a
# user pasting a bracket- or paren-wrapped URL would hit the identical bug,
# just not yet observed live.
_URL_RE = re.compile(r"https?://[^\s\"'>\]\)]+")

# Maximum number of web_search queries per dispatch call — same cap as
# legacy ToolDispatcher._run_web_search.
_MAX_WEB_QUERIES: int = 3

# Bounded research loop — hard cap on search+evaluate+reformulate cycles.
# Same rationale as _MAX_WEB_QUERIES: an unbounded loop against a live
# search provider is a cost and latency risk, not just a correctness one.
_MAX_RESEARCH_ITERATIONS: int = 3

# Chart argument extraction — matches diag_shadow_chart_toolcall_v4_full.py's
# measured pipeline exactly. _CHART_RETRY_TEMPERATURE is a second,
# independent inference sample on a MALFORMED_ENVELOPE first pass, not a
# repeat of the deterministic temperature=0.0 attempt.
_CHART_MAX_TOKENS:        int   = 400
_CHART_RETRY_TEMPERATURE: float = 0.3

# 2026-07-17: live testing showed a gate-check call inside
# _evaluate_pricing_gate (max_tokens=10) stall for the full 60s
# LOCALIST_STREAM_TIMEOUT before timing out — confirmed via logs that the
# Ollama daemon itself stayed responsive throughout (health-check polling
# to /api/tags kept succeeding every 15s during the stall), so this was a
# cloud-model-side stall, not a local hang. The 60s default is sized for
# the full 1024-token main-dispatch answer; a max_tokens=10/40 classifier
# call sharing that same budget means a stuck one burns a full minute
# before the loop can recover, when it should fail fast and let the loop
# reformulate instead. Applied only to _evaluate_pricing_gate and
# _reformulate_query — every other infer()/infer_stream() call site in the
# codebase keeps the default timeout unchanged.
_RESEARCH_CLASSIFIER_TIMEOUT: float = 15.0

_RESEARCH_GATE_SYSTEM_PROMPT: str = (
    "You are a fact-extraction classifier, not a conversational assistant. "
    "You will be given the ORIGINAL QUESTION the user asked, and a block of "
    "search-result or page TEXT. Decide whether the TEXT contains concrete, "
    "specific information that directly answers the ORIGINAL QUESTION — not "
    "just any pricing-shaped or spec-shaped content in general. A page that "
    "mentions a price or number for a DIFFERENT product, a different "
    "tier/trim than the one asked about, or that only says pricing exists "
    "without stating the number, does NOT count. If the question asks about "
    "a specific tier, trim, plan, or variant, the text must address THAT "
    "specific one, not a different one. Respond with exactly one word: yes "
    "or no."
)

_RESEARCH_REFORMULATE_SYSTEM_PROMPT: str = (
    "You are a search-query rewriter, not a conversational assistant. "
    "The previous web search did not surface concrete pricing information. "
    "Given the original request and the queries already tried, write ONE "
    "new, more specific search query likely to surface a pricing page "
    "with actual numbers (e.g. add \"pricing\", \"plans\", \"per month\", "
    "or the vendor's likely domain). Respond with the query text only, "
    "nothing else."
)

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
    "file_op", "url_fetch", "web_search", and "chart" are served by the
    localist-mcp MCP server; any other tool name is unrecognized (Planner
    never routes tools_to_call to anything else — see planner.py's P3/P3b)
    and produces an inline error ToolResult, same shape the legacy
    ToolDispatcher's "else" branch used to produce.

    Parameters
    ----------
    runtime :
        RuntimeClient. Used by the research loop's pricing-gate evaluation
        and query reformulation (see _run_research_loop) and by chart
        argument extraction (see _run_chart) via a blocking runtime.infer()
        call — the same synchronous-call-from-async-context pattern
        planner.py's _classify_tool_fallback already uses, accepted here for
        the same reason (single-user, non-production app). Prior to the
        research loop's addition, this parameter was accepted but never
        stored — web_search's runtime.infer() hallucination fallback was
        removed in Phase 3 and nothing else here used it.
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
        self._runtime         = runtime
        self._mcp_server_url  = mcp_server_url or _MCP_SERVER_URL

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
                elif tool_name == "news_search":
                    results.extend(
                        await self._run_news_search(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "research":
                    results.extend(
                        await self._run_research_loop(session, connect_error, instruction, ctx)
                    )
                elif tool_name == "chart":
                    chart_result = await self._run_chart(session, connect_error, instruction, ctx)
                    # A failed chart extraction/dispatch degrades to a normal
                    # prose answer rather than surfacing an ERROR: result to
                    # the model — see _run_chart's docstring and
                    # claude/chart-mcp-tool-scoping.md's "Failure handling"
                    # section. Every other tool branch here appends an
                    # ERROR-shaped ToolResult on failure (visible to the
                    # model in Slot 5a); chart deliberately does not.
                    if chart_result is not None:
                        results.append(chart_result)
                    else:
                        logger.warning(
                            "MCPToolDispatcher: chart — extraction/dispatch "
                            "failed; no ToolResult appended (accepted "
                            "residual failure rate — see "
                            "claude/chart-mcp-tool-scoping.md)."
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

    # -----------------------------------------------------------------------
    # news_search — NewsAPI first, falling back to web_search (Brave) on a miss
    # -----------------------------------------------------------------------

    async def _run_news_search(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Execute news_search: NewsAPI (tier 1) first, falling through to the
        existing Brave-backed web_search tool (tier 2) on a miss — see
        news-query-routing plan §4.1/§4.2. Single query only (unlike
        web_search's up-to-_MAX_WEB_QUERIES loop) — reuses
        _derive_initial_query so both tiers see the same derived text; a
        miss on "latest news on X" still asks the fallback tier about "X",
        not a generic re-derivation.

        Always returns the tier-1 ToolResult. Only calls tier 2 (and
        appends its result) when tier 1 didn't succeed — mirrors
        _run_research_loop's "return every ToolResult produced, not just
        the winning one" convention so Slot 5 and controller_agent.py's
        Step 3b corpus fallback see the full picture.
        """
        query = self._derive_initial_query(instruction, context)
        news_result = await self._execute_news_search_query(session, connect_error, query)

        if news_result.success:
            return [news_result]

        logger.info(
            "MCPToolDispatcher: news_search miss/failure for query=%r — "
            "falling back to web_search (Brave).",
            query,
        )
        brave_result = await self._execute_web_search_query(session, connect_error, query)
        # Retag for provenance (§4.3) — Step 3b's corpus-fallback check in
        # controller_agent.py matches on tool_name.startswith("news_search"),
        # so this stays visible to it even after the rename.
        brave_result = replace(brave_result, tool_name="news_search:brave_fallback")

        return [news_result, brave_result]

    async def _execute_news_search_query(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        query:         str,
    ) -> ToolResult:
        params_str = f"query={query!r}"

        if session is None:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {connect_error}",
                success    = False,
            )

        try:
            text, is_error = await self._call_mcp_tool(session, "news_search", {"query": query})
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: localist-mcp unreachable for news_search query=%r: %s",
                query, exc,
            )
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: localist-mcp unreachable — {exc}",
                success    = False,
            )

        if is_error:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = _normalize_mcp_error_text(text),
                success    = False,
            )

        try:
            data = json.loads(text)
        except Exception as exc:
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = f"ERROR: failed to parse news_search response — {exc}",
                success    = False,
            )

        if data.get("is_miss", False):
            return ToolResult(
                tool_name  = "news_search",
                parameters = params_str,
                result     = "",
                success    = False,
            )

        return ToolResult(
            tool_name  = "news_search",
            parameters = params_str,
            result     = data.get("result_text", ""),
            success    = True,
        )

    # -----------------------------------------------------------------------
    # research — bounded search / evaluate / reformulate / fetch loop
    # -----------------------------------------------------------------------

    async def _run_research_loop(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> list[ToolResult]:
        """
        Loop up to _MAX_RESEARCH_ITERATIONS times: web_search -> evaluate
        (cheap yes/no classifier call, same pattern as
        controller_agent._execute_plan's P5 episodic-relevance check) ->
        if the result text already contains concrete pricing, url_fetch the
        top candidate page and re-run the gate on the full text; if not,
        reformulate the query (one more bounded infer call) and retry.

        Returns every ToolResult produced along the way (all search/fetch
        attempts, not just the winning one) so controller_agent's existing
        logging/fallback logic (Step 3b: corpus fallback when every
        web_search result failed) keeps working unmodified.

        Every ToolResult returned carries the same freshly-generated
        `workflow_id` (see ToolResult.workflow_id) — this method is already
        a single bounded call representing one research "workflow", so the
        id is generated once here and stamped on every constituent result
        rather than needing a correlation key threaded in from outside.
        Read by controller_agent.py to build metadata["workflow_steps"] for
        the Episode Browsing UI's step-chain view
        (episode-browsing-ui-plan.md, Phase 2).

        Two distinct "didn't work" outcomes are handled differently:
          - A search/fetch call itself fails (provider/connectivity error):
            the failing ToolResult already has tool_name="web_search"/
            "url_fetch" and success=False, so it's indistinguishable from a
            plain web_search failure — Step 3b's existing
            `r.tool_name == "web_search" and not r.success` check already
            catches it with no changes needed there.
          - Every iteration's search/fetch call *succeeds* but the pricing
            gate never passes (loop exhausts, or reformulation degenerates
            to a repeat): every individual ToolResult in that case has
            success=True (the searches worked; they just didn't find
            pricing), so nothing in the returned list would trip Step 3b's
            web_search check. A synthetic trailing ToolResult
            (tool_name="research", result starting with "ERROR:",
            success=False) is appended in that case only — same
            "ERROR: ..." shape every other failure path in this file uses,
            so it flows into controller_agent's tool_failures prompt slot
            (letting the model honestly say it couldn't find pricing rather
            than guessing) and, via the added `or r.tool_name == "research"`
            in Step 3b, also triggers the corpus fallback.
        """
        results: list[ToolResult] = []
        tried_queries: list[str] = []
        tried_urls:    set[str]  = set()
        connectivity_failed = False
        workflow_id = str(uuid.uuid4())

        query = self._derive_initial_query(instruction, context)

        for iteration in range(_MAX_RESEARCH_ITERATIONS):
            tried_queries.append(query)
            search_result = await self._execute_web_search_query(
                session, connect_error, query
            )
            search_result.workflow_id = workflow_id
            results.append(search_result)

            if not search_result.success:
                # Provider/connectivity failure, not a "no pricing found"
                # outcome — stop the loop, let controller_agent's Step 3b
                # corpus fallback take over exactly as it does for a plain
                # web_search failure today.
                connectivity_failed = True
                break

            gate_pass = await self._evaluate_pricing_gate(instruction, search_result.result)

            candidate_url = self._extract_first_url(search_result.result, tried_urls)

            if not gate_pass and candidate_url:
                # Search snippet alone was inconclusive but pointed at a
                # page — pull the full page before giving up on this query.
                tried_urls.add(candidate_url)
                fetch_result = await self._run_url_fetch(
                    session, connect_error, instruction,
                    {**context, "fetch_url": candidate_url},
                )
                fetch_result.workflow_id = workflow_id
                results.append(fetch_result)
                if fetch_result.success:
                    gate_pass = await self._evaluate_pricing_gate(instruction, fetch_result.result)

            if gate_pass:
                logger.info(
                    "MCPToolDispatcher: research loop — pricing found after "
                    "%d iteration(s), queries=%s.",
                    iteration + 1, tried_queries,
                )
                return results

            if iteration == _MAX_RESEARCH_ITERATIONS - 1:
                break

            query = await self._reformulate_query(instruction, tried_queries)
            if query in tried_queries:
                # Reformulation degenerated to a repeat — stop rather than
                # spend another round-trip on a query we know fails.
                break

        logger.info(
            "MCPToolDispatcher: research loop — exhausted %d iteration(s) "
            "without concrete pricing, queries=%s.",
            len(tried_queries), tried_queries,
        )
        if not connectivity_failed:
            results.append(ToolResult(
                tool_name   = "research",
                parameters  = f"queries={tried_queries!r}",
                result      = (
                    f"ERROR: research loop exhausted {len(tried_queries)} "
                    f"iteration(s) without finding concrete pricing "
                    f"information (queries tried: {tried_queries})."
                ),
                success     = False,
                workflow_id = workflow_id,
            ))
        return results

    def _derive_initial_query(self, instruction: str, context: dict[str, Any]) -> str:
        # Reuse the exact same resolution order _run_web_search already
        # uses (explicit context["web_search_queries"][0], else derived
        # from the instruction) so "research" and "web_search" behave
        # identically on turn one and only diverge once evaluation kicks in.
        raw_queries: list[str] = context.get("web_search_queries") or []
        if raw_queries:
            return raw_queries[0]
        derived = instruction.strip()
        for filler in _WEB_SEARCH_FILLER_PREFIXES:
            if derived.lower().startswith(filler):
                derived = derived[len(filler):]
                break
        return derived[:120]

    async def _evaluate_pricing_gate(self, instruction: str, text: str) -> bool:
        """Single bounded yes/no inference call. Never raises — a failed
        gate check is treated as "no", same fail-open-to-continue posture
        as every other try/except in this file.

        `instruction` is the original question, passed alongside `text` so
        the classifier can judge relevance (does this text specifically
        answer THIS question) rather than merely detecting that some
        pricing/spec-shaped content is present somewhere in `text` — see
        _RESEARCH_GATE_SYSTEM_PROMPT and
        diagnostics/reports/research_loop_qa_assessment_2026-07-20.md for
        the false-positive pattern this fixes (e.g. gate-passing on a
        different product's price, or on content that never actually
        states the requested number)."""
        try:
            raw = self._runtime.infer(
                system      = _RESEARCH_GATE_SYSTEM_PROMPT,
                prompt      = (
                    f"Original question:\n\n{instruction}\n\n"
                    f"Text:\n\n{text[:3000]}\n\n"
                    f"Does the text directly answer the original question with a "
                    f"specific number (yes/no):"
                ),
                max_tokens  = 10,
                temperature = 0.1,
                timeout     = _RESEARCH_CLASSIFIER_TIMEOUT,
            )
            return raw.strip().lower().startswith("yes")
        except Exception as exc:
            logger.debug("MCPToolDispatcher: research gate check failed (%s).", exc)
            return False

    async def _reformulate_query(self, instruction: str, tried: list[str]) -> str:
        try:
            raw = self._runtime.infer(
                system      = _RESEARCH_REFORMULATE_SYSTEM_PROMPT,
                prompt      = (
                    f"Original request: {instruction}\n"
                    f"Queries already tried: {tried}\n\nNew query:"
                ),
                max_tokens  = 40,
                temperature = 0.3,
                timeout     = _RESEARCH_CLASSIFIER_TIMEOUT,
            )
            return raw.strip().strip('"')[:120]
        except Exception as exc:
            logger.debug("MCPToolDispatcher: query reformulation failed (%s).", exc)
            return tried[-1]  # fall through to the repeat-guard, which stops the loop

    @staticmethod
    def _extract_first_url(text: str, exclude: set[str]) -> str | None:
        for match in _URL_RE.finditer(text):
            # _URL_RE already excludes ]/) from the match itself (2026-07-16
            # fix), but a URL pulled out of running text can still end in
            # trailing sentence punctuation a URL is very unlikely to
            # legitimately end with (e.g. "...pricing." at a sentence
            # boundary) — stripped here as a second, cheap layer of defense
            # against a differently-formatted future source hitting the
            # same class of bug the bracket-wrapping case did.
            url = match.group(0).rstrip(".,;:")
            if url not in exclude:
                return url
        return None

    # -----------------------------------------------------------------------
    # chart — served by localist-mcp (generate_chart)
    # -----------------------------------------------------------------------

    async def _run_chart(
        self,
        session:       ClientSession | None,
        connect_error: Exception | None,
        instruction:   str,
        context:       dict[str, Any],
    ) -> ToolResult | None:
        """
        Extract generate_chart arguments from `instruction` via a bounded
        few-shot inference call, then dispatch to the generate_chart MCP
        tool. Promotes diag_shadow_chart_toolcall_v4_full.py's measured
        pipeline to production, unchanged — see the module docstring's
        "chart" paragraph and claude/chart-mcp-tool-scoping.md for the
        reliability numbers this is based on (66.7% MATCH on
        chart-expected instructions, 12.1% residual failure accepted by
        design).

        Returns None — not an ERROR-shaped ToolResult — on any failure
        (post-retry still malformed, schema-invalid, the model legitimately
        declining via {"tool_call": null}, an unreachable localist-mcp
        server, or the generate_chart tool call itself failing). See
        _dispatch_async's chart branch: a failed chart never reaches the
        model as a visible tool error, it just means the turn ends up with
        no chart — the accepted residual-failure behavior.
        """
        arguments = await self._extract_chart_arguments(instruction)
        if arguments is None:
            return None

        params_str = f"chart_type={arguments.get('chart_type')!r}"

        if session is None:
            logger.warning(
                "MCPToolDispatcher: chart — localist-mcp unreachable (%s).",
                connect_error,
            )
            return None

        try:
            text, is_error = await self._call_mcp_tool(session, "generate_chart", arguments)
        except Exception as exc:
            logger.warning("MCPToolDispatcher: chart — localist-mcp call failed: %s", exc)
            return None

        if is_error:
            logger.warning(
                "MCPToolDispatcher: chart — generate_chart tool failed: %s",
                _normalize_mcp_error_text(text),
            )
            return None

        try:
            data = json.loads(text)
        except Exception as exc:
            logger.warning(
                "MCPToolDispatcher: chart — failed to parse generate_chart response: %s", exc
            )
            return None

        logger.info(
            "MCPToolDispatcher: chart complete — %s", params_str,
        )
        return ToolResult(
            tool_name  = "chart",
            parameters = params_str,
            # Only "summary" ever reaches the model (Slot 5's 500-token
            # ceiling) — png_path/chart_config ride in .artifact instead,
            # read directly by controller_agent.py, never rendered into
            # prompt-facing text. See prompt_builder.ToolResult.artifact.
            result     = data.get("summary", ""),
            success    = True,
            artifact   = {
                "png_path":     data.get("png_path"),
                "chart_config": data.get("chart_config"),
            },
        )

    async def _extract_chart_arguments(self, instruction: str) -> dict[str, Any] | None:
        """
        Run the infer -> repair -> validate pipeline at temperature=0.0; on
        a malformed envelope, retry once at _CHART_RETRY_TEMPERATURE. The
        retry's outcome is final — no second retry (matches the "one
        retry" scope diag_shadow_chart_toolcall_v4_full.py measured).

        Returns valid chart arguments, or None on any other outcome
        (schema-invalid, the model declining via null, or still malformed
        after the retry).
        """
        attempt = await self._run_chart_extraction_attempt(instruction, temperature=0.0)
        if attempt["outcome"] == "malformed":
            attempt = await self._run_chart_extraction_attempt(
                instruction, temperature=_CHART_RETRY_TEMPERATURE
            )
        return attempt["arguments"] if attempt["outcome"] == "match" else None

    async def _run_chart_extraction_attempt(
        self, instruction: str, temperature: float
    ) -> dict[str, Any]:
        """
        One infer -> repair -> classify pass. Returns
        {"outcome": ..., "arguments": ...}, where outcome is one of:
          "malformed"      — envelope missing/wrong-shaped/unknown tool
                              name (retry-eligible on the first attempt).
          "no_tool"        — well-formed {"tool_call": null} — the model
                              declined to chart.
          "schema_invalid" — well-formed tool_call, but
                              validate_chart_arguments() found problems.
          "match"          — valid chart arguments ready to dispatch.

        Ported verbatim from diag_shadow_chart_toolcall_v4_full.py's
        _run_one()/_classify_envelope() — same envelope-shape checks,
        same KNOWN_TOOL_NAMES/validate_chart_arguments() calls.
        """
        try:
            raw = self._runtime.infer(
                prompt      = instruction,
                system      = SYSTEM_PROMPT_FEWSHOT,
                max_tokens  = _CHART_MAX_TOKENS,
                temperature = temperature,
            )
        except Exception as exc:
            logger.debug("MCPToolDispatcher: chart — infer() failed (%s).", exc)
            return {"outcome": "malformed", "arguments": None}

        obj, _repair_outcome = repair_envelope(raw)

        if not isinstance(obj, dict) or "tool_call" not in obj:
            return {"outcome": "malformed", "arguments": None}

        call = obj["tool_call"]
        if call is None:
            return {"outcome": "no_tool", "arguments": None}

        if (
            not isinstance(call, dict)
            or "name" not in call
            or "arguments" not in call
            or not isinstance(call["name"], str)
            or not isinstance(call["arguments"], dict)
            or call["name"] not in KNOWN_TOOL_NAMES
        ):
            return {"outcome": "malformed", "arguments": None}

        problems = validate_chart_arguments(call["arguments"])
        if problems:
            return {"outcome": "schema_invalid", "arguments": None}

        return {"outcome": "match", "arguments": call["arguments"]}

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
