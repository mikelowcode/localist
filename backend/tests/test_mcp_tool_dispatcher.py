"""
Phase 1 tests — MCPToolDispatcher (mcp_tool_dispatcher.py).
Phase 2 adds url_fetch coverage. Phase 3 adds web_search coverage. Phase 4
(2026-07-03) drops the legacy ToolDispatcher entirely — an unrecognized
tool name now produces an inline error ToolResult instead of delegating.

Covers:
  - file_op success path: MCP tool call succeeds, ToolResult.success=True
  - file_op error path (tool-level isError): ToolResult.success is explicitly
    set to False — this was the specific gap in the legacy ToolDispatcher's
    dataclass default that Phase 1 fixed.
  - file_op connection failure (MCP server unreachable): graceful
    ToolResult(success=False), never raises.
  - url_fetch: URL found (instruction regex or context override) success
    path, no-URL-found degraded path, tool-level error normalized via the
    same _normalize_mcp_error_text() proven correct for file_op, and
    connection-failure graceful degradation.
  - web_search: query resolution (explicit web_search_queries vs. derived
    from instruction), the 3-query cap, missing-API-key -> success=False
    (the ControllerAgent-level proof that this actually fires Step 3b's
    corpus fallback lives in test_tool_dispatcher_phase6.py), and
    connection-failure graceful degradation.
  - An unrecognized tool name returns an inline error ToolResult
    (success=False) without calling the MCP server at all.
  - research (2026-07-16): the bounded search/evaluate/reformulate/fetch
    loop (_run_research_loop and its helpers) — gate-passes-immediately,
    gate-fails-then-fetch-succeeds, iteration cap with the synthetic
    trailing tool_name="research" failure ToolResult, the reformulation
    repeat-guard, connectivity-failure short-circuit (no synthetic result),
    fail-closed behavior when the gate/reformulate infer() calls raise, the
    dispatch() "research" routing branch, and _derive_initial_query's query
    resolution order. dispatcher._runtime.infer is controlled per-test via
    side_effect, branching on the `system` kwarg
    (_RESEARCH_GATE_SYSTEM_PROMPT vs _RESEARCH_REFORMULATE_SYSTEM_PROMPT)
    since both calls share the same mock.

_call_mcp_tool is monkeypatched per-test rather than opening a real SSE
socket — the real network round-trip is covered by mcp_server's own
in-process tests (test_mcp_server.py) and by live verification.

_open_session (SSE connect + ClientSession.initialize(), the per-dispatch
session-reuse seam added in the MCP follow-up) is likewise mocked at the
`dispatcher` fixture level for every test below except TestSessionReuse,
which asserts directly on its call count/behavior — that class exercises
the real localist-mcp subprocess instead.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_tool_dispatcher import (
    MCPToolDispatcher,
    _CHART_RETRY_TEMPERATURE,
    _MAX_RESEARCH_ITERATIONS,
    _RESEARCH_CLASSIFIER_TIMEOUT,
    _RESEARCH_GATE_SYSTEM_PROMPT,
    _RESEARCH_REFORMULATE_SYSTEM_PROMPT,
)
from prompt_builder import ToolResult

# Sentinel session object handed back by the mocked _open_session below.
# Never actually used to send protocol messages — _call_mcp_tool itself is
# mocked per-test, so this just needs to be a non-None value.
_FAKE_SESSION = object()


@pytest.fixture()
def dispatcher(tmp_path) -> MCPToolDispatcher:
    rt = MagicMock(spec=["infer"])
    rt.infer.return_value = "• Fallback result."
    d = MCPToolDispatcher(runtime=rt, project_root=tmp_path)
    with patch.object(MCPToolDispatcher, "_open_session", return_value=_FAKE_SESSION):
        yield d


class TestFileOpSuccess:
    def test_read_success_maps_to_tool_result(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            assert name == "read_file"
            assert arguments == {"path": "notes.md"}
            return "file content here", False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "read notes",
                context       = {"file_op_action": "read", "file_op_path": "notes.md"},
            )

        assert len(results) == 1
        r = results[0]
        assert r.tool_name == "file_op"
        assert r.result == "file content here"
        assert r.success is True

    def test_write_success_passes_content_argument(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            assert name == "write_file"
            assert arguments == {"path": "out.md", "content": "hello"}
            return "OK: wrote 5 characters to out.md", False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "write file",
                context       = {
                    "file_op_action":  "write",
                    "file_op_path":    "out.md",
                    "file_op_content": "hello",
                },
            )

        assert results[0].success is True
        assert results[0].result.startswith("OK: wrote")


class TestFileOpErrorPaths:
    def test_tool_level_error_sets_success_false(self, dispatcher: MCPToolDispatcher):
        """MCP tool call completes but returns isError=True (e.g. file not found)."""
        async def fake_call(session, name, arguments):
            return "ERROR: file not found — /x/ghost.md", True

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "read ghost",
                context       = {"file_op_action": "read", "file_op_path": "ghost.md"},
            )

        assert results[0].success is False
        assert "ERROR" in results[0].result

    def test_connection_failure_returns_graceful_error(self, dispatcher: MCPToolDispatcher):
        """localist-mcp unreachable — must not raise; success explicitly False."""
        async def fake_call(session, name, arguments):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "read notes",
                context       = {"file_op_action": "read", "file_op_path": "notes.md"},
            )

        assert len(results) == 1
        assert results[0].tool_name == "file_op"
        assert results[0].success is False
        assert "unreachable" in results[0].result

    def test_missing_file_op_path_returns_error_without_calling_mcp(
        self, dispatcher: MCPToolDispatcher
    ):
        with patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "read something",
                context       = {"file_op_action": "read"},  # no file_op_path
            )

        mock_call.assert_not_called()
        assert results[0].success is False

    def test_unknown_action_returns_error_without_calling_mcp(
        self, dispatcher: MCPToolDispatcher
    ):
        with patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["file_op"],
                instruction   = "do something weird",
                context       = {"file_op_action": "delete", "file_op_path": "notes.md"},
            )

        mock_call.assert_not_called()
        assert results[0].success is False


class TestUrlFetch:
    def test_url_extracted_from_instruction_and_success(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            assert name == "fetch_url"
            assert arguments == {"url": "https://example.com/article"}
            return json.dumps({
                "url":               "https://example.com/article",
                "title":             "An Article",
                "author":            "",
                "date_published":    "",
                "cleaned_text":      "Body text here.",
                "word_count":        3,
                "fetch_duration_ms": 5.0,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "summarize this link https://example.com/article please",
                context       = {},
            )

        assert len(results) == 1
        r = results[0]
        assert r.tool_name == "url_fetch"
        assert r.success is True
        assert "Title: An Article" in r.result
        assert "Source: https://example.com/article" in r.result
        assert "Words: 3" in r.result
        assert "Body text here." in r.result

    def test_bracket_wrapped_url_in_instruction_excludes_trailing_bracket(
        self, dispatcher: MCPToolDispatcher
    ):
        """
        2026-07-16 regression: _URL_RE is shared between _run_url_fetch's
        instruction-text extraction (this test) and _extract_first_url
        (the research loop's own regression tests, in TestResearchLoop) —
        a user pasting a bracket- or paren-wrapped URL into their own
        message would hit the same "]"/")" captured as part of the URL bug
        that broke the research loop against web_search's
        "[{url}]"-formatted result text.
        """
        async def fake_call(session, name, arguments):
            assert arguments == {"url": "https://www.t-mobile.com/cell-phones/brand/apple"}
            return json.dumps({
                "url":               "https://www.t-mobile.com/cell-phones/brand/apple",
                "title":             "Apple",
                "author":            "",
                "date_published":    "",
                "cleaned_text":      "Body text here.",
                "word_count":        3,
                "fetch_duration_ms": 5.0,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "check out [https://www.t-mobile.com/cell-phones/brand/apple] for pricing",
                context       = {},
            )

        assert results[0].success is True

    def test_context_fetch_url_overrides_instruction_url(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            assert arguments == {"url": "https://override.example/page"}
            return json.dumps({
                "url": "https://override.example/page", "title": "T", "author": "",
                "date_published": "", "cleaned_text": "x", "word_count": 1,
                "fetch_duration_ms": 1.0,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "fetch this https://instruction.example/ignored",
                context       = {"fetch_url": "https://override.example/page"},
            )

        assert results[0].success is True

    def test_no_url_found_returns_degraded_error_without_calling_mcp(
        self, dispatcher: MCPToolDispatcher
    ):
        with patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "fetch this url please",  # no actual URL
                context       = {},
            )

        mock_call.assert_not_called()
        assert results[0].tool_name == "url_fetch"
        assert results[0].success is False
        assert results[0].result == "ERROR: no URL found in instruction"

    def test_tool_level_error_normalized_via_shared_helper(self, dispatcher: MCPToolDispatcher):
        """
        Proves url_fetch reuses the same _normalize_mcp_error_text() fixed for
        file_op in the Phase 1 follow-up, rather than reintroducing the
        FastMCP-wrapping bug.
        """
        async def fake_call(session, name, arguments):
            return (
                "Error executing tool fetch_url: "
                "ERROR: connection_error — Could not connect to host. (refused)"
            ), True

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "fetch this https://unreachable.example",
                context       = {},
            )

        assert results[0].success is False
        assert results[0].result.startswith("ERROR: connection_error —")
        assert not results[0].result.startswith("Error executing tool")

    def test_connection_failure_returns_graceful_error(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["url_fetch"],
                instruction   = "fetch this https://example.com",
                context       = {},
            )

        assert results[0].tool_name == "url_fetch"
        assert results[0].success is False
        assert "unreachable" in results[0].result


class TestWebSearch:
    def test_explicit_queries_used_over_derived(self, dispatcher: MCPToolDispatcher):
        seen_queries: list[str] = []

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            seen_queries.append(arguments["query"])
            return json.dumps({
                "query": arguments["query"], "result_text": f"result for {arguments['query']}",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "this text should be ignored",
                context       = {"web_search_queries": ["query one", "query two"]},
            )

        assert seen_queries == ["query one", "query two"]
        assert len(results) == 2
        assert all(r.tool_name == "web_search" and r.success is True for r in results)
        assert results[0].result == "result for query one"

    def test_query_derived_from_instruction_strips_filler(self, dispatcher: MCPToolDispatcher):
        seen_queries: list[str] = []

        async def fake_call(session, name, arguments):
            seen_queries.append(arguments["query"])
            return json.dumps({"query": arguments["query"], "result_text": "x", "result_count": 1}), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "what is the latest oMLX release",
                context       = {},
            )

        assert seen_queries == ["latest oMLX release"]

    def test_max_queries_capped_at_three(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            return json.dumps({"query": arguments["query"], "result_text": "x", "result_count": 1}), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "ignored",
                context       = {"web_search_queries": ["q1", "q2", "q3", "q4", "q5"]},
            )

        assert mock_call.call_count == 3
        assert len(results) == 3

    def test_missing_api_key_produces_success_false(self, dispatcher: MCPToolDispatcher):
        """
        The locked Phase 3 decision: no runtime.infer() fallback exists on
        this path anymore. A missing LANGSEARCH_API_KEY is just another
        MCP tool-level error, normalized the same way file_op/url_fetch
        errors are — this is what lets controller_agent.py's Step 3b corpus
        fallback fire. (End-to-end proof that it actually does fire lives
        in test_tool_dispatcher_phase6.py's
        test_web_search_missing_key_triggers_corpus_fallback.)
        """
        async def fake_call(session, name, arguments):
            return (
                "Error executing tool web_search: "
                "ERROR: LANGSEARCH_API_KEY not configured"
            ), True

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "what is the latest oMLX release",
                context       = {},
            )

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].result == "ERROR: LANGSEARCH_API_KEY not configured"

    def test_connection_failure_returns_graceful_error(self, dispatcher: MCPToolDispatcher):
        async def fake_call(session, name, arguments):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "what is the latest oMLX release",
                context       = {},
            )

        assert len(results) == 1
        assert results[0].tool_name == "web_search"
        assert results[0].success is False
        assert "unreachable" in results[0].result


class TestChart:
    """
    Coverage for MCPToolDispatcher._run_chart and its extraction pipeline
    (_extract_chart_arguments / _run_chart_extraction_attempt) — promotes
    diag_shadow_chart_toolcall_v4_full.py's measured infer->repair->
    validate(->retry once)->dispatch pipeline to production.

    dispatcher._runtime.infer is used synchronously (same pattern the
    research loop's gate/reformulate calls already use) to produce the
    few-shot {"tool_call": ...} envelope; _call_mcp_tool is monkeypatched
    for the generate_chart dispatch itself, same as every other tool here.
    """

    _VALID_ENVELOPE = json.dumps({
        "tool_call": {
            "name": "generate_chart",
            "arguments": {
                "chart_type": "bar",
                "title":      "Fruit Inventory",
                "labels":     ["apples", "oranges"],
                "datasets":   [{"label": "Count", "data": [5, 3]}],
            },
        }
    })

    def test_successful_chart_dispatch(self, dispatcher: MCPToolDispatcher):
        dispatcher._runtime.infer.return_value = self._VALID_ENVELOPE

        async def fake_call(session, name, arguments):
            assert name == "generate_chart"
            assert arguments["chart_type"] == "bar"
            return json.dumps({
                "summary":      "Generated bar chart: Fruit Inventory",
                "png_path":     "charts/abc123.png",
                "chart_config": arguments,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["chart"],
                instruction   = "chart this: apples 5, oranges 3",
                context       = {},
            )

        assert dispatcher._runtime.infer.call_count == 1
        assert len(results) == 1
        r = results[0]
        assert r.tool_name == "chart"
        assert r.success is True
        assert r.result == "Generated bar chart: Fruit Inventory"
        assert r.artifact == {
            "png_path":     "charts/abc123.png",
            "chart_config": {
                "chart_type": "bar",
                "title":      "Fruit Inventory",
                "labels":     ["apples", "oranges"],
                "datasets":   [{"label": "Count", "data": [5, 3]}],
            },
        }

    def test_retry_recovers_from_malformed_first_attempt(self, dispatcher: MCPToolDispatcher):
        """First infer() call returns plain prose (no JSON at all — the
        classic "prose instead of null envelope" failure mode); the retry
        at _CHART_RETRY_TEMPERATURE returns a valid envelope, and that
        retry's result is final."""
        responses = iter(["Sure, here's a summary of your data.", self._VALID_ENVELOPE])
        temperatures_seen: list[float] = []

        def infer_side_effect(**kw):
            temperatures_seen.append(kw["temperature"])
            return next(responses)

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            return json.dumps({
                "summary":      "Generated bar chart: Fruit Inventory",
                "png_path":     "charts/abc123.png",
                "chart_config": arguments,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["chart"],
                instruction   = "chart this: apples 5, oranges 3",
                context       = {},
            )

        assert dispatcher._runtime.infer.call_count == 2
        assert temperatures_seen == [0.0, _CHART_RETRY_TEMPERATURE]
        assert len(results) == 1
        assert results[0].success is True

    def test_full_failure_after_retry_appends_no_result(self, dispatcher: MCPToolDispatcher):
        """Both the first attempt and the retry come back malformed — per
        the accepted-failure design (claude/chart-mcp-tool-scoping.md), no
        ToolResult is appended at all, and generate_chart is never called."""
        dispatcher._runtime.infer.return_value = "Sure, here's a summary of your data."

        with patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["chart"],
                instruction   = "chart this: apples 5, oranges 3",
                context       = {},
            )

        assert dispatcher._runtime.infer.call_count == 2
        mock_call.assert_not_called()
        assert results == []


class TestUnknownTool:
    """
    As of Phase 4 (ToolDispatcher deletion, 2026-07-03), an unrecognized
    tool name no longer delegates anywhere — it produces an inline error
    ToolResult, the one surviving piece of the legacy ToolDispatcher's
    "else" branch. Planner never actually routes tools_to_call to anything
    but file_op/url_fetch/web_search (see planner.py's P3/P3b), so this is
    an unreachable-in-practice defensive path.
    """

    def test_unknown_tool_returns_inline_error_result(self, dispatcher: MCPToolDispatcher):
        with patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["totally_unknown_tool"],
                instruction   = "do a thing",
                context       = {},
            )

        mock_call.assert_not_called()
        assert len(results) == 1
        assert results[0].tool_name == "totally_unknown_tool"
        assert results[0].result == "ERROR: unknown tool 'totally_unknown_tool'"
        assert results[0].success is False


class TestSessionReuse:
    """
    MCP follow-up (2026-07-03): a live trace showed _call_mcp_tool opening a
    brand-new SSE session (full connect/handshake/teardown) for every single
    tool invocation, even when one dispatch() call makes several — most
    visibly a multi-query web_search. dispatch() now opens exactly one
    ClientSession per dispatch() call (via _open_session) and reuses it for
    every _call_mcp_tool() invocation made during that call.

    These two tests use their own _open_session patch (overriding the
    `dispatcher` fixture's default mock for the scope of the `with` block)
    so they can assert directly on how many times a session gets opened,
    rather than just on the fixture's fixed sentinel.
    """

    def test_one_session_per_dispatch_for_multi_query_web_search(
        self, dispatcher: MCPToolDispatcher
    ):
        open_call_count = 0

        async def fake_open_session(stack):
            nonlocal open_call_count
            open_call_count += 1
            return _FAKE_SESSION

        async def fake_call(session, name, arguments):
            assert session is _FAKE_SESSION
            return json.dumps({
                "query": arguments["query"], "result_text": f"result for {arguments['query']}",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_open_session", side_effect=fake_open_session), \
             patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "ignored",
                context       = {"web_search_queries": ["q1", "q2", "q3"]},
            )

        assert open_call_count == 1
        assert len(results) == 3
        assert all(r.success is True for r in results)

    def test_one_session_per_dispatch_across_multiple_tools(
        self, dispatcher: MCPToolDispatcher
    ):
        """Same one-session guarantee when a dispatch mixes different tools,
        not just repeated queries for a single tool."""
        open_call_count = 0

        async def fake_open_session(stack):
            nonlocal open_call_count
            open_call_count += 1
            return _FAKE_SESSION

        async def fake_call(session, name, arguments):
            assert session is _FAKE_SESSION
            if name == "read_file":
                return "file content", False
            if name == "fetch_url":
                return json.dumps({
                    "url": arguments["url"], "title": "T", "author": "",
                    "date_published": "", "cleaned_text": "x", "word_count": 1,
                    "fetch_duration_ms": 1.0,
                }), False
            return json.dumps({"query": arguments["query"], "result_text": "x", "result_count": 1}), False

        with patch.object(MCPToolDispatcher, "_open_session", side_effect=fake_open_session), \
             patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["file_op", "url_fetch", "web_search"],
                instruction   = "read notes.md and fetch https://example.com",
                context       = {"file_op_action": "read", "file_op_path": "notes.md"},
            )

        assert open_call_count == 1
        assert len(results) == 3
        assert all(r.success is True for r in results)

    def test_connection_down_degrades_every_tool_call_not_just_first(
        self, dispatcher: MCPToolDispatcher
    ):
        """
        If the session can't be established at all, every tool in the
        dispatch — not just the first — must degrade the same way each tool
        already does individually (success=False, no raise), and
        _call_mcp_tool must never be reached.
        """
        async def fake_open_session_fails(stack):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(MCPToolDispatcher, "_open_session", side_effect=fake_open_session_fails), \
             patch.object(MCPToolDispatcher, "_call_mcp_tool") as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["file_op", "url_fetch", "web_search"],
                instruction   = "read notes.md and fetch https://example.com",
                context       = {"file_op_action": "read", "file_op_path": "notes.md"},
            )

        mock_call.assert_not_called()
        assert len(results) == 3
        assert all(r.success is False for r in results)
        assert all("unreachable" in r.result for r in results)
        assert [r.tool_name for r in results] == ["file_op", "url_fetch", "web_search"]


class TestResearchLoop:
    """
    Coverage for MCPToolDispatcher._run_research_loop and its helpers
    (_derive_initial_query, _evaluate_pricing_gate, _reformulate_query,
    _extract_first_url). This code shipped (2026-07-16) with only the
    763-passed full-suite regression check as evidence — that confirmed
    nothing ELSE broke, not that the loop itself behaves correctly. These
    tests are that missing direct coverage.

    dispatcher._runtime is a MagicMock(spec=["infer"]) (see the `dispatcher`
    fixture) — its .infer.side_effect is set per-test to a function keyed
    off the `system` kwarg, since _evaluate_pricing_gate and
    _reformulate_query share the same mock but use different system
    prompts (_RESEARCH_GATE_SYSTEM_PROMPT vs
    _RESEARCH_REFORMULATE_SYSTEM_PROMPT) to select which one is "talking".

    _evaluate_pricing_gate/_reformulate_query are exercised both indirectly
    (via dispatch(tools_to_call=["research"], ...)) and, for the two
    fail-closed cases, directly via asyncio.run() — no pytest-asyncio
    plugin is installed in this project, so async helpers are awaited the
    same way dispatch() itself does internally (asyncio.run), not via an
    async test function.
    """

    # ------------------------------------------------------------------
    # Gate outcomes
    # ------------------------------------------------------------------

    def test_gate_passes_immediately_stops_after_one_iteration(
        self, dispatcher: MCPToolDispatcher
    ):
        """Gate says "yes" on the very first search result: one ToolResult,
        no url_fetch, no reformulate call."""
        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "The Basic plan costs $10/month.",
                "result_count": 1,
            }), False

        def infer_side_effect(**kw):
            assert kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT, (
                "reformulate must not be called when the gate passes immediately"
            )
            return "yes"

        dispatcher._runtime.infer.side_effect = infer_side_effect

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "what does the Basic plan cost",
                context       = {},
            )

        assert mock_call.call_count == 1
        assert mock_call.call_args.args[1] == "web_search"
        assert len(results) == 1
        assert results[0].tool_name == "web_search"
        assert results[0].success is True
        assert dispatcher._runtime.infer.call_count == 1

    def test_gate_fails_on_snippet_then_passes_after_fetch(
        self, dispatcher: MCPToolDispatcher
    ):
        """Search snippet is inconclusive but names a URL; fetching that URL
        and re-running the gate on the full page text passes. Both
        ToolResults are returned, no synthetic failure result."""
        gate_calls: list[str] = []

        def infer_side_effect(**kw):
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                gate_calls.append(kw["prompt"])
                # First gate call = search snippet (no price stated) -> "no".
                # Second gate call = fetched page text (has a price) -> "yes".
                return "yes" if len(gate_calls) == 2 else "no"
            raise AssertionError("reformulate must not be called once the gate passes")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            if name == "web_search":
                return json.dumps({
                    "query": arguments["query"],
                    "result_text": "See https://vendor.example/pricing for plan details.",
                    "result_count": 1,
                }), False
            if name == "fetch_url":
                assert arguments == {"url": "https://vendor.example/pricing"}
                return json.dumps({
                    "url": "https://vendor.example/pricing", "title": "Pricing", "author": "",
                    "date_published": "", "cleaned_text": "The Basic plan is $10 per month.",
                    "word_count": 6, "fetch_duration_ms": 2.0,
                }), False
            raise AssertionError(f"unexpected tool name {name!r}")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "what does the vendor plan cost",
                context       = {},
            )

        called_tools = [c.args[1] for c in mock_call.call_args_list]
        assert called_tools == ["web_search", "fetch_url"]
        assert len(gate_calls) == 2

        assert len(results) == 2
        assert results[0].tool_name == "web_search" and results[0].success is True
        assert results[1].tool_name == "url_fetch" and results[1].success is True
        assert not any(r.tool_name == "research" for r in results)

    # ------------------------------------------------------------------
    # workflow_id (episode-browsing-ui-plan.md Phase 2) — every ToolResult
    # produced by one _run_research_loop() call shares a single correlation
    # key, read by controller_agent.py to build metadata["workflow_steps"].
    # ------------------------------------------------------------------

    def test_all_results_in_one_loop_share_the_same_workflow_id(
        self, dispatcher: MCPToolDispatcher
    ):
        gate_calls: list[str] = []

        def infer_side_effect(**kw):
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                gate_calls.append(kw["prompt"])
                return "yes" if len(gate_calls) == 2 else "no"
            raise AssertionError("reformulate must not be called once the gate passes")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            if name == "web_search":
                return json.dumps({
                    "query": arguments["query"],
                    "result_text": "See https://vendor.example/pricing for plan details.",
                    "result_count": 1,
                }), False
            if name == "fetch_url":
                return json.dumps({
                    "url": "https://vendor.example/pricing", "title": "Pricing", "author": "",
                    "date_published": "", "cleaned_text": "The Basic plan is $10 per month.",
                    "word_count": 6, "fetch_duration_ms": 2.0,
                }), False
            raise AssertionError(f"unexpected tool name {name!r}")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "what does the vendor plan cost",
                context       = {},
            )

        assert len(results) == 2
        assert results[0].workflow_id is not None
        assert results[0].workflow_id == results[1].workflow_id

    def test_iteration_cap_synthetic_failure_carries_same_workflow_id(
        self, dispatcher: MCPToolDispatcher
    ):
        def infer_side_effect(**kw):
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                return "no"
            return "reformulated query"

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"], "result_text": "no price here", "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "what does it cost",
                context       = {},
            )

        assert results[-1].tool_name == "research" and results[-1].success is False
        workflow_ids = {r.workflow_id for r in results}
        assert len(workflow_ids) == 1
        assert None not in workflow_ids

    def test_two_separate_dispatches_get_different_workflow_ids(
        self, dispatcher: MCPToolDispatcher
    ):
        def infer_side_effect(**kw):
            return "yes"

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            return json.dumps({
                "query": arguments["query"], "result_text": "$10/month.", "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results_a = dispatcher.dispatch(
                tools_to_call=["research"], instruction="cost of plan a", context={},
            )
            results_b = dispatcher.dispatch(
                tools_to_call=["research"], instruction="cost of plan b", context={},
            )

        assert results_a[0].workflow_id != results_b[0].workflow_id

    def test_web_search_tool_call_never_carries_a_workflow_id(
        self, dispatcher: MCPToolDispatcher
    ):
        """workflow_id is a research-loop-only correlation key — a plain
        "web_search" tool call (not routed through the loop) must not have
        one, so controller_agent.py's workflow_id extraction never
        mistakes an ordinary search for a workflow."""
        async def fake_call(session, name, arguments):
            return json.dumps({
                "query": arguments["query"], "result_text": "some text", "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(
                tools_to_call = ["web_search"],
                instruction   = "tell me about zebras",
                context       = {},
            )

        assert results[0].workflow_id is None

    # ------------------------------------------------------------------
    # Loop termination: iteration cap and repeat-guard
    # ------------------------------------------------------------------

    def test_iteration_cap_appends_synthetic_research_failure(
        self, dispatcher: MCPToolDispatcher
    ):
        """Gate always says "no" and reformulate keeps returning a fresh
        query — the loop must stop at _MAX_RESEARCH_ITERATIONS web_search
        calls and append a trailing tool_name="research" success=False
        ToolResult (the piece controller_agent.py's Step 3b depends on)."""
        reformulate_count = 0

        def infer_side_effect(**kw):
            nonlocal reformulate_count
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                return "no"
            if kw["system"] == _RESEARCH_REFORMULATE_SYSTEM_PROMPT:
                reformulate_count += 1
                return f"reformulated query {reformulate_count}"
            raise AssertionError(f"unexpected system prompt {kw['system']!r}")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "No pricing information available here.",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "vendor plan cost",
                context       = {},
            )

        assert mock_call.call_count == _MAX_RESEARCH_ITERATIONS
        assert len(results) == _MAX_RESEARCH_ITERATIONS + 1
        assert all(
            r.tool_name == "web_search" and r.success is True
            for r in results[:_MAX_RESEARCH_ITERATIONS]
        )

        synthetic = results[-1]
        assert synthetic.tool_name == "research"
        assert synthetic.success is False
        assert synthetic.result.startswith("ERROR:")

    def test_repeat_guard_stops_before_iteration_cap(
        self, dispatcher: MCPToolDispatcher
    ):
        """Reformulate degenerates to a query already tried — the loop must
        stop immediately rather than spend another round-trip, and the
        synthetic trailing failure result is still appended."""
        def infer_side_effect(**kw):
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                return "no"
            if kw["system"] == _RESEARCH_REFORMULATE_SYSTEM_PROMPT:
                # Same as the derived initial query for instruction below ->
                # guaranteed repeat after the first iteration.
                return "vendor plan cost"
            raise AssertionError(f"unexpected system prompt {kw['system']!r}")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "No pricing information available here.",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "vendor plan cost",
                context       = {},
            )

        assert mock_call.call_count == 1
        assert mock_call.call_count < _MAX_RESEARCH_ITERATIONS

        assert len(results) == 2
        assert results[0].tool_name == "web_search" and results[0].success is True
        synthetic = results[-1]
        assert synthetic.tool_name == "research"
        assert synthetic.success is False
        assert synthetic.result.startswith("ERROR:")

    # ------------------------------------------------------------------
    # Connectivity failure: no synthetic result appended
    # ------------------------------------------------------------------

    def test_connectivity_failure_stops_immediately_without_synthetic_result(
        self, dispatcher: MCPToolDispatcher
    ):
        """A search/fetch call itself failing (provider/connectivity error)
        already looks like a plain web_search failure to Step 3b's existing
        `r.tool_name == "web_search" and not r.success` clause — the loop
        must stop after this first iteration and must NOT also append the
        synthetic tool_name="research" result (that's reserved for the
        "every call succeeded but no pricing was found" case)."""
        async def fake_call(session, name, arguments):
            raise ConnectionRefusedError("Connection refused")

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "vendor plan cost",
                context       = {},
            )

        assert mock_call.call_count == 1
        assert len(results) == 1
        assert results[0].tool_name == "web_search"
        assert results[0].success is False
        assert "unreachable" in results[0].result
        assert not any(r.tool_name == "research" for r in results)
        dispatcher._runtime.infer.assert_not_called()

    # ------------------------------------------------------------------
    # Fail-closed behavior of the gate/reformulate helpers themselves
    # ------------------------------------------------------------------

    def test_evaluate_pricing_gate_fails_closed_on_infer_error(
        self, dispatcher: MCPToolDispatcher
    ):
        """A raising runtime.infer() must not propagate out of
        _evaluate_pricing_gate — it's treated as "no" (fail-closed), per
        the method's own docstring."""
        dispatcher._runtime.infer.side_effect = RuntimeError("model unavailable")

        result = asyncio.run(
            dispatcher._evaluate_pricing_gate(
                "what does the Basic plan cost", "The Basic plan costs $10/month."
            )
        )
        assert result is False

    # ------------------------------------------------------------------
    # Relevance-aware gate (2026-07-20 fix) — _evaluate_pricing_gate now
    # takes `instruction` alongside `text` so it can judge whether the text
    # answers THIS question, not just whether pricing-shaped content is
    # present anywhere. See diagnostics/reports/
    # research_loop_qa_assessment_2026-07-20.md for the false-positive
    # pattern this closes (e.g. gate-passing on a different product/tier's
    # price, or on content that never actually states the requested fact).
    # ------------------------------------------------------------------

    def test_evaluate_pricing_gate_prompt_includes_original_question(
        self, dispatcher: MCPToolDispatcher
    ):
        """Structural check that the original question is actually threaded
        into the classifier prompt alongside the candidate text — the crux
        of this fix, since the classifier can only be relevance-aware if it
        can see what was asked."""
        captured: dict = {}

        def infer_side_effect(**kw):
            captured.update(kw)
            return "yes"

        dispatcher._runtime.infer.side_effect = infer_side_effect

        asyncio.run(
            dispatcher._evaluate_pricing_gate(
                "What does GitHub Copilot Individual cost per month?",
                "1 AI credit = $0.01 USD. No monthly plan price stated.",
            )
        )

        assert "What does GitHub Copilot Individual cost per month?" in captured["prompt"]
        assert "1 AI credit = $0.01 USD" in captured["prompt"]

    def test_loop_exhausts_on_wrong_tier_content_instead_of_false_positive(
        self, dispatcher: MCPToolDispatcher
    ):
        """The false-positive case from the QA pass: every search result
        found a real price, but for the WRONG tier/product relative to what
        was asked (here: "Enterprise" asked, only "Starter"/"Pro" prices
        found). A relevance-aware gate must say "no" throughout, and the
        loop should exhaust honestly rather than gate-pass on a
        wrong-but-confident-looking answer."""
        call_n = 0

        def infer_side_effect(**kw):
            nonlocal call_n
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                assert "exact enterprise contract price" in kw["prompt"]
                return "no"  # wrong tier every time — never the Enterprise price asked for
            if kw["system"] == _RESEARCH_REFORMULATE_SYSTEM_PROMPT:
                call_n += 1
                return f"reformulated query {call_n}"
            raise AssertionError(f"unexpected system prompt {kw['system']!r}")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "Starter plan: $25/user/month. Pro Suite: $100/user/month.",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "What is the exact enterprise contract price for Salesforce?",
                context       = {},
            )

        assert mock_call.call_count == _MAX_RESEARCH_ITERATIONS
        synthetic = results[-1]
        assert synthetic.tool_name == "research"
        assert synthetic.success is False
        assert "exhausted" in synthetic.result

    def test_loop_gate_passes_when_text_answers_the_specific_tier_asked(
        self, dispatcher: MCPToolDispatcher
    ):
        """Same shape as the wrong-tier case above, but the search result
        this time names the SPECIFIC tier asked about — confirms the
        signature change didn't break the ordinary passing path."""
        def infer_side_effect(**kw):
            assert kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT
            assert "Enterprise" in kw["prompt"]
            return "yes"

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "Enterprise plan: $175/user/month.",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "What does the Enterprise plan cost?",
                context       = {},
            )

        assert mock_call.call_count == 1
        assert len(results) == 1
        assert results[0].success is True

    def test_reformulate_failure_falls_through_to_repeat_guard(
        self, dispatcher: MCPToolDispatcher
    ):
        """_reformulate_query's own except-clause falls back to `tried[-1]`
        on a raising infer() call — that guaranteed repeat should trip the
        repeat-guard and stop the loop cleanly, not propagate the
        exception out of dispatch()."""
        def infer_side_effect(**kw):
            if kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT:
                return "no"
            if kw["system"] == _RESEARCH_REFORMULATE_SYSTEM_PROMPT:
                raise RuntimeError("model unavailable")
            raise AssertionError(f"unexpected system prompt {kw['system']!r}")

        dispatcher._runtime.infer.side_effect = infer_side_effect

        async def fake_call(session, name, arguments):
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"],
                "result_text": "No pricing information available here.",
                "result_count": 1,
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call) as mock_call:
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "vendor plan cost",
                context       = {},
            )

        assert mock_call.call_count == 1
        assert mock_call.call_count < _MAX_RESEARCH_ITERATIONS
        assert len(results) == 2
        synthetic = results[-1]
        assert synthetic.tool_name == "research"
        assert synthetic.success is False

    # ------------------------------------------------------------------
    # Classifier timeout override (2026-07-17)
    # ------------------------------------------------------------------
    #
    # Live testing showed a gate-check call (max_tokens=10) stall for the
    # full 60s default timeout on a cloud-model-side hang — confirmed not a
    # local issue since Ollama daemon health-check polling stayed healthy
    # throughout. _evaluate_pricing_gate/_reformulate_query now pass
    # timeout=_RESEARCH_CLASSIFIER_TIMEOUT (15.0) instead of relying on the
    # runtime's default, so a stuck classifier call fails fast and lets the
    # loop reformulate/exhaust rather than burning a full minute per stall.

    def test_evaluate_pricing_gate_passes_research_classifier_timeout(
        self, dispatcher: MCPToolDispatcher
    ):
        dispatcher._runtime.infer.return_value = "no"

        asyncio.run(
            dispatcher._evaluate_pricing_gate("some question", "some search result text")
        )

        assert dispatcher._runtime.infer.call_args.kwargs["timeout"] == _RESEARCH_CLASSIFIER_TIMEOUT

    def test_reformulate_query_passes_research_classifier_timeout(
        self, dispatcher: MCPToolDispatcher
    ):
        dispatcher._runtime.infer.return_value = "a reformulated query"

        asyncio.run(dispatcher._reformulate_query("original instruction", ["tried one"]))

        assert dispatcher._runtime.infer.call_args.kwargs["timeout"] == _RESEARCH_CLASSIFIER_TIMEOUT

    # ------------------------------------------------------------------
    # dispatch() routing
    # ------------------------------------------------------------------

    def test_dispatch_routes_research_tool_to_the_loop(
        self, dispatcher: MCPToolDispatcher
    ):
        """tools_to_call=["research"] must reach _run_research_loop, not the
        "unknown tool" inline-error branch — mirrors how the file already
        proves the file_op/url_fetch/web_search branches route correctly."""
        captured: dict = {}

        async def fake_loop(session, connect_error, instruction, context):
            captured["instruction"] = instruction
            captured["context"] = context
            return [ToolResult(
                tool_name  = "web_search",
                parameters = "",
                result     = "stubbed research result",
                success    = True,
            )]

        with patch.object(MCPToolDispatcher, "_run_research_loop", side_effect=fake_loop):
            results = dispatcher.dispatch(
                tools_to_call = ["research"],
                instruction   = "what does the vendor plan cost",
                context       = {"web_search_queries": ["vendor plan pricing"]},
            )

        assert captured["instruction"] == "what does the vendor plan cost"
        assert captured["context"] == {"web_search_queries": ["vendor plan pricing"]}
        assert len(results) == 1
        assert results[0].result == "stubbed research result"

    # ------------------------------------------------------------------
    # _derive_initial_query
    # ------------------------------------------------------------------

    def test_derive_initial_query_uses_explicit_queries_first_element(
        self, dispatcher: MCPToolDispatcher
    ):
        result = dispatcher._derive_initial_query(
            "this instruction text is ignored",
            {"web_search_queries": ["explicit query one", "explicit query two"]},
        )
        assert result == "explicit query one"

    def test_derive_initial_query_falls_back_to_filler_stripped_instruction(
        self, dispatcher: MCPToolDispatcher
    ):
        """Same resolution _run_web_search uses (test_query_derived_from_
        instruction_strips_filler in TestWebSearch) — "research" and
        "web_search" must behave identically on turn one."""
        result = dispatcher._derive_initial_query(
            "what is the latest oMLX release", {}
        )
        assert result == "latest oMLX release"

    # ------------------------------------------------------------------
    # _extract_first_url — 2026-07-16 regression
    # ------------------------------------------------------------------
    #
    # Live research loop run confirmed _URL_RE captured the trailing "]"
    # from mcp_server/web_search.py's result formatting
    # (f"• {title}\n  {body}\n  [{url}]") — every URL is wrapped in
    # literal [...] — producing ".../apple]" and a 404 on url_fetch.
    # _URL_RE now excludes ]/) from the match itself, and
    # _extract_first_url additionally rstrips trailing sentence
    # punctuation (".,;:") as a second, cheap layer of defense.

    def test_extract_first_url_strips_trailing_bracket(
        self, dispatcher: MCPToolDispatcher
    ):
        text = "• Apple iPhone pricing\n  Compare plans and pricing.\n  [https://www.t-mobile.com/cell-phones/brand/apple]"
        url = dispatcher._extract_first_url(text, exclude=set())
        assert url == "https://www.t-mobile.com/cell-phones/brand/apple"

    def test_extract_first_url_strips_trailing_paren(
        self, dispatcher: MCPToolDispatcher
    ):
        text = "See the pricing page (https://vendor.example/pricing) for details."
        url = dispatcher._extract_first_url(text, exclude=set())
        assert url == "https://vendor.example/pricing"

    def test_extract_first_url_strips_trailing_sentence_punctuation(
        self, dispatcher: MCPToolDispatcher
    ):
        """Not bracket-wrapped — a URL sitting at the end of a sentence,
        e.g. from a differently-formatted future source. Covered by
        _extract_first_url's rstrip(".,;:"), not by _URL_RE itself."""
        text = "Full pricing details are available at https://vendor.example/pricing."
        url = dispatcher._extract_first_url(text, exclude=set())
        assert url == "https://vendor.example/pricing"

    def test_extract_first_url_unwrapped_url_unaffected(
        self, dispatcher: MCPToolDispatcher
    ):
        """A normal, unwrapped URL with no trailing punctuation must come
        back exactly as it appears — the fix must not over-trim."""
        text = "Direct link: https://vendor.example/pricing/plans/v2"
        url = dispatcher._extract_first_url(text, exclude=set())
        assert url == "https://vendor.example/pricing/plans/v2"
