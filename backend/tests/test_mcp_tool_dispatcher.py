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

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_tool_dispatcher import MCPToolDispatcher

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
