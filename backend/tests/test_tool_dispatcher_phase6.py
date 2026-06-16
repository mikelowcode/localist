"""
Phase 6 integration tests — LORA Tool Dispatcher.

Covers:
  6.1 — ToolDispatcher interface: dispatch(), unknown tool error
  6.2 — web_search: fallback infer path, real web_search method path,
         explicit query list, max-3 enforcement, failure graceful degradation
  6.3 — file_op: read, write, append, truncation, path traversal sandbox,
         missing path error, missing file error, unknown action error,
         parent directory auto-creation
  6.4 — ControllerAgent wiring:
         [TOOL RESULTS] slot in prebuilt prompt when tool fires
         Slot ordering: [TOOL RESULTS] < [INSTRUCTION]
         Token ceiling enforced (slot 6 ≤ 500 tokens)
         Tool dispatch failure → graceful absence of [TOOL RESULTS]
         Quality filter: ERROR/repr/short strings excluded from slot 6
         Ingest path (P1) produces no tool results even when tool keyword present
           unless compound detection fires

All ToolDispatcher tests use tmp_path for file_op. Runtime calls use MagicMock.
ControllerAgent tests use a real SQLite DB via tmp_path.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tool_dispatcher import ToolDispatcher, _MAX_FILE_READ_CHARS, _MAX_WEB_QUERIES
from prompt_builder import ToolResult, PromptBuilder
from controller_agent import ControllerAgent, TaskStatus, AgentResult
from memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def td(tmp_root: Path) -> ToolDispatcher:
    rt = MagicMock(spec=["infer"])
    rt.infer.return_value = "• Fallback result."
    return ToolDispatcher(runtime=rt, project_root=tmp_root)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)
    return path


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


def make_conv_agent(answer="Test answer."):
    received = []
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True
    def run(subtask):
        received.append(subtask)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "conversational_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answer, "sources": [], "grounded": False},
        )
    agent.run.side_effect = run
    agent._received = received
    return agent


# ---------------------------------------------------------------------------
# 6.1 — ToolDispatcher interface
# ---------------------------------------------------------------------------

class TestToolDispatcherInterface:

    def test_dispatch_returns_list(self, td):
        results = td.dispatch(["web_search"], "What is LORA?")
        assert isinstance(results, list)

    def test_dispatch_returns_tool_result_instances(self, td):
        results = td.dispatch(["web_search"], "What is LORA?")
        assert all(isinstance(r, ToolResult) for r in results)

    def test_unknown_tool_returns_error(self, td):
        results = td.dispatch(["unknown_tool"], "test")
        assert len(results) == 1
        assert results[0].tool_name  == "unknown_tool"
        assert results[0].result.startswith("ERROR:")

    def test_multiple_tools_dispatched_in_order(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.return_value = "• Search result."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        # Write a file to read
        (tmp_root / "notes.md").write_text("Note content.", encoding="utf-8")

        results = td.dispatch(
            tools_to_call = ["web_search", "file_op"],
            instruction   = "search and read",
            context       = {
                "file_op_action": "read",
                "file_op_path":   "notes.md",
            },
        )
        assert len(results) == 2
        assert results[0].tool_name == "web_search"
        assert results[1].tool_name == "file_op"

    def test_empty_tools_list_returns_empty(self, td):
        results = td.dispatch([], "test")
        assert results == []


# ---------------------------------------------------------------------------
# 6.2 — web_search tool
# ---------------------------------------------------------------------------

class TestWebSearch:

    def test_fallback_infer_path(self, tmp_root):
        rt = MagicMock(spec=["infer"])   # no web_search method
        rt.infer.return_value = "• oMLX 0.4.2 released."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        results = td.dispatch(["web_search"], "latest oMLX release")
        assert results[0].tool_name == "web_search"
        assert "oMLX" in results[0].result
        assert rt.infer.called

    def test_langsearch_api_called_when_key_set(self, tmp_root):
        rt = MagicMock()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "oMLX Release",
                            "snippet": "Latest oMLX release notes.",
                            "displayUrl": "example.com/omlx",
                            "summary": None,
                        }
                    ]
                }
            }
        }
        fake_response.raise_for_status = MagicMock()
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        with patch.dict("os.environ", {"LANGSEARCH_API_KEY": "test-key"}):
            with patch("requests.post", return_value=fake_response) as mock_post:
                results = td.dispatch(["web_search"], "latest oMLX release")

        assert mock_post.called
        assert "oMLX Release" in results[0].result
        assert not rt.infer.called

    def test_explicit_queries_from_context(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.return_value = "Result."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        results = td.dispatch(
            tools_to_call = ["web_search"],
            instruction   = "search",
            context       = {
                "web_search_queries": ["query A", "query B", "query C"]
            },
        )
        assert len(results) == 3
        assert rt.infer.call_count == 3

    def test_max_queries_enforced(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.return_value = "Result."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        results = td.dispatch(
            tools_to_call = ["web_search"],
            instruction   = "search",
            context       = {
                "web_search_queries": [
                    "q1", "q2", "q3", "q4", "q5"
                ]
            },
        )
        assert len(results) == _MAX_WEB_QUERIES
        assert rt.infer.call_count == _MAX_WEB_QUERIES

    def test_query_derived_from_instruction(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.return_value = "Result."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        td.dispatch(["web_search"], "What are the latest oMLX changes?")
        # One infer call with query derived from instruction
        assert rt.infer.call_count == 1
        call_prompt = rt.infer.call_args.kwargs.get("prompt") or \
                      rt.infer.call_args.args[0]
        assert "oMLX" in call_prompt or "latest" in call_prompt.lower()

    def test_inference_failure_returns_error_result(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.side_effect = Exception("model offline")
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)

        results = td.dispatch(["web_search"], "test query")
        assert len(results) == 1
        assert results[0].result.startswith("ERROR:")

    def test_fallback_uses_correct_max_tokens(self, tmp_root):
        rt = MagicMock(spec=["infer"])
        rt.infer.return_value = "Result."
        td = ToolDispatcher(runtime=rt, project_root=tmp_root)
        td.dispatch(["web_search"], "test")

        call_kwargs = rt.infer.call_args.kwargs
        assert call_kwargs.get("max_tokens", 0) <= 120


# ---------------------------------------------------------------------------
# 6.3 — file_op tool
# ---------------------------------------------------------------------------

class TestFileOp:

    def test_read_existing_file(self, td, tmp_root):
        f = tmp_root / "notes.md"
        f.write_text("Hello, LORA.", encoding="utf-8")

        results = td.dispatch(
            ["file_op"], "read notes",
            context={"file_op_action": "read", "file_op_path": "notes.md"},
        )
        assert "Hello, LORA." in results[0].result

    def test_read_nonexistent_file_returns_error(self, td):
        results = td.dispatch(
            ["file_op"], "read missing",
            context={"file_op_action": "read", "file_op_path": "ghost.md"},
        )
        assert results[0].result.startswith("ERROR:")
        assert "not found" in results[0].result

    def test_write_creates_file(self, td, tmp_root):
        results = td.dispatch(
            ["file_op"], "write output",
            context={
                "file_op_action":  "write",
                "file_op_path":    "out/result.md",
                "file_op_content": "# Result\nContent here.",
            },
        )
        assert "OK:" in results[0].result
        assert (tmp_root / "out" / "result.md").exists()
        assert "Content here." in (tmp_root / "out" / "result.md").read_text()

    def test_write_creates_parent_directories(self, td, tmp_root):
        td.dispatch(
            ["file_op"], "write nested",
            context={
                "file_op_action":  "write",
                "file_op_path":    "a/b/c/deep.md",
                "file_op_content": "Deep content.",
            },
        )
        assert (tmp_root / "a" / "b" / "c" / "deep.md").exists()

    def test_append_adds_to_existing_file(self, td, tmp_root):
        f = tmp_root / "log.txt"
        f.write_text("Line 1.\n", encoding="utf-8")

        td.dispatch(
            ["file_op"], "append",
            context={
                "file_op_action":  "append",
                "file_op_path":    "log.txt",
                "file_op_content": "Line 2.\n",
            },
        )
        assert f.read_text() == "Line 1.\nLine 2.\n"

    def test_append_creates_file_if_absent(self, td, tmp_root):
        td.dispatch(
            ["file_op"], "append to new",
            context={
                "file_op_action":  "append",
                "file_op_path":    "new_log.txt",
                "file_op_content": "First line.\n",
            },
        )
        assert (tmp_root / "new_log.txt").read_text() == "First line.\n"

    def test_read_truncates_large_file(self, td, tmp_root):
        big = tmp_root / "big.txt"
        big.write_text("A" * (_MAX_FILE_READ_CHARS + 1000), encoding="utf-8")

        results = td.dispatch(
            ["file_op"], "read big",
            context={"file_op_action": "read", "file_op_path": "big.txt"},
        )
        assert "[truncated]" in results[0].result
        # Result must not exceed budget (with small suffix tolerance)
        assert len(results[0].result) <= _MAX_FILE_READ_CHARS + 30

    def test_read_does_not_truncate_small_file(self, td, tmp_root):
        small = tmp_root / "small.txt"
        content = "Short content."
        small.write_text(content, encoding="utf-8")

        results = td.dispatch(
            ["file_op"], "read small",
            context={"file_op_action": "read", "file_op_path": "small.txt"},
        )
        assert "[truncated]" not in results[0].result
        assert content in results[0].result

    def test_path_traversal_blocked(self, td):
        results = td.dispatch(
            ["file_op"], "read secret",
            context={
                "file_op_action": "read",
                "file_op_path":   "../../etc/passwd",
            },
        )
        assert results[0].result.startswith("ERROR:")
        assert "traversal" in results[0].result

    def test_missing_file_op_path_returns_error(self, td):
        results = td.dispatch(
            ["file_op"], "read something",
            context={"file_op_action": "read"},   # no file_op_path
        )
        assert results[0].result.startswith("ERROR:")

    def test_unknown_action_returns_error(self, td, tmp_root):
        results = td.dispatch(
            ["file_op"], "do something weird",
            context={
                "file_op_action": "delete",
                "file_op_path":   "notes.md",
            },
        )
        assert results[0].result.startswith("ERROR:")
        assert "unknown" in results[0].result.lower()


# ---------------------------------------------------------------------------
# 6.4 — ControllerAgent integration
# ---------------------------------------------------------------------------

class TestControllerToolIntegration:

    def test_tool_results_slot_present_in_prompt(self, mm):
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",    # P5 episodic relevance
            "• oMLX 0.4.2 released.\n• Supports Gemma 4B quantized.",  # web_search
            "NONE",  # implicit extraction
        ]

        conv = make_conv_agent("Here is what I found.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "What are the latest oMLX release notes?",
            "context":     {"project_context": "LORA"},
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        assert "oMLX" in prompt

    def test_slot_order_user_before_tool_results(self, mm):
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",
            "• Search result with enough content to pass quality filter.",
            "NONE",
        ]

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "What are the latest changes?",
            "context":     {"project_context": "LORA"},
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        if "[TOOL RESULTS]" in prompt:
            assert prompt.index("[TOOL RESULTS]") < prompt.index("[INSTRUCTION]")

    def test_tool_slot_ceiling_enforced(self, mm):
        """Slot 6 must not exceed 500 tokens (2000 chars) in the prompt."""
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        # Return a very long result from web_search
        rt.infer.side_effect = [
            "no",
            "• " + "A" * 3000,  # far exceeds 500-token slot ceiling
            "NONE",
        ]

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "What are the latest changes?",
            "context":     {"project_context": "LORA"},
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        if "[TOOL RESULTS]" in prompt:
            start = prompt.index("[TOOL RESULTS]")
            # Find the next slot label after TOOL RESULTS (or end of string)
            next_slot = len(prompt)
            for label in ["[INSTRUCTION]", "[WORKING MEMORY]",
                          "[EPISODIC MEMORY]", "[CONTEXT]"]:
                pos = prompt.find(label, start + 1)
                if pos != -1:
                    next_slot = min(next_slot, pos)
            tool_section = prompt[start:next_slot]
            assert len(tool_section) // 4 <= 505  # 500 + small label tolerance

    def test_tool_dispatch_failure_graceful(self, mm):
        """If tool dispatch raises, task still completes; [TOOL RESULTS] absent."""
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",
            Exception("tool crashed"),
            "NONE",
        ]

        conv = make_conv_agent("Graceful answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        result = ctrl.handle_task({
            "instruction": "What are the latest changes?",
            "context":     {"project_context": "LORA"},
        })

        assert result["status"] == "complete"
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" not in prompt

    def test_quality_filter_excludes_error_results(self, mm):
        """Tool results beginning with ERROR: must not appear in slot 6."""
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",
            Exception("search failed"),  # causes ERROR: result
            "NONE",
        ]

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "What are the latest changes?",
            "context":     {"project_context": "LORA"},
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "ERROR:" not in prompt

    def test_ingest_path_no_tool_results(self, mm):
        """P1 (ingest) fires alone — no tool results even with 'latest' keyword
        unless compound detection fires (both ingest + web_search keywords)."""
        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.return_value = "NONE"

        wiki = MagicMock()
        wiki.name = "wiki_agent"
        wiki.can_handle.return_value = True
        wiki.run.return_value = AgentResult(
            subtask_id = "w-0",
            agent_name = "wiki_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"new_pages": [], "applied": False},
        )

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "ingest this document",
            "context":     {"raw_path": "/data/notes.md"},
        })

        # wiki_agent was dispatched; routing has no tools
        routing = wiki.run.call_args[0][0].context["_routing"]
        assert routing["tools_to_call"] == []

    def test_file_op_results_appear_in_prompt(self, mm, tmp_path):
        """file_op read result flows through to [TOOL RESULTS] slot."""
        notes = tmp_path / "notes.md"
        notes.write_text(
            "LORA memory system uses SQLite for persistence.", encoding="utf-8"
        )

        rt = MagicMock(spec=["infer", "embed"])
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            "no",   # pre-dispatch episodic check
            "NONE", # implicit extraction
        ]

        conv = make_conv_agent("Here is the file content.")
        ctrl = ControllerAgent(
            runtime=rt, agents=[conv], memory_manager=mm
        )
        ctrl.handle_task({
            "instruction": "read the file notes.md",
            "context": {
                "project_root":   str(tmp_path),
                "file_op_action": "read",
                "file_op_path":   "notes.md",
                "project_context": "LORA",
            },
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" in prompt
        assert "SQLite" in prompt
