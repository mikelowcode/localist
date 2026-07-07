"""
Phase 4 integration tests — LORA ControllerAgent execution pipeline.

Covers:
  - Direct answer path: P6 fallback → PromptBuilder slot 2 only
  - RAG path: P4 corpus hit → fetch_rag=True → rag_sources in prompt
  - Ingest path: P1 → wiki_agent dispatched, Synthesizer called
  - Episodic write path: P2 → EpisodicMemoryWriter called
  - Episodic retrieval path: fetch_episodic=True → bullets in prompt
  - Prebuilt prompt passthrough: ConversationalAgent uses _prebuilt_prompt
  - wiki_doc wiring: _load_persona / _load_user_profile frontmatter handling
  - Working memory: prior turns appear in slot 3
  - Routing metadata: _routing key present in SubTask context
  - Tool stub: tools_to_call logged but not executed
  - Fallback: unregistered agent falls back to conversational_agent

Each test uses a real SQLite DB via tmp_path for paths that exercise
EpisodicMemoryWriter/Reader. Tests that don't need the DB use MagicMock.

Note on RAG test design: MemoryManager.query_corpus() uses Jaccard keyword
overlap scoring. The Planner's Priority 4 threshold is 0.4. Document content
and query strings are chosen to achieve Jaccard >= 0.6 so corpus hits are
reliable regardless of embed availability.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from controller_agent import (
    ControllerAgent,
    Task,
    TaskStatus,
    AgentResult,
    SubTask,
    _memory_key,
    _extract_file_op_content,
    _file_op_confirmation_line,
)
from memory_manager import (
    MemoryManager,
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
    GraphEdgeResult,
)
from planner import RoutingPlan
from prompt_builder import PromptBuilder, WorkingMemoryState, ToolResult as _ToolResult
from wiki_doc import load_wiki_doc, ParsedWikiDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)   # initialises schema v2
    return path


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


def make_runtime(infer_return: str = "Test answer.", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_conv_agent(answer: str = "Test answer."):
    """Conversational agent that captures the SubTask it receives."""
    received: list[SubTask] = []

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


def make_wiki_agent():
    agent = MagicMock()
    agent.name = "wiki_agent"
    agent.can_handle.return_value = True
    agent.run.return_value = AgentResult(
        subtask_id = "wiki-0",
        agent_name = "wiki_agent",
        status     = TaskStatus.COMPLETE,
        output     = {"new_pages": [], "applied": False},
    )
    return agent


# ---------------------------------------------------------------------------
# Path 1 — Direct answer (Priority 6 fallback)
# ---------------------------------------------------------------------------

class TestDirectAnswerPath:

    def test_completes_successfully(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Direct answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "What is 2+2?"})

        assert result["status"] == "complete"
        assert result["answer"] == "Direct answer."

    def test_prebuilt_prompt_passed_to_agent(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        subtask = conv._received[0]
        assert "_prebuilt_prompt" in subtask.context
        assert "[INSTRUCTION]" in subtask.context["_prebuilt_prompt"]

    def test_routing_metadata_in_context(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"]      is False
        assert routing["fetch_episodic"] is False
        assert routing["tools_to_call"]  == []
        assert routing["write_episode"]  is False

    def test_no_rag_sources_in_prompt(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" not in prompt


# ---------------------------------------------------------------------------
# Path 2 — RAG path (Priority 4)
# ---------------------------------------------------------------------------
# P4 now fires on explicit wiki/vault trigger keywords, not corpus scoring.

class TestRAGPath:

    def test_fetch_rag_true_when_wiki_keyword_present(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "check the wiki LORA research assistant agentic",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("RAG answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "check the wiki for LORA research assistant"})

        assert result["status"] == "complete"
        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"] is True

    def test_rag_sources_appear_in_prompt(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "check the wiki LORA SQLite memory storage",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "check the wiki for LORA SQLite memory"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Source:" in prompt

    def test_no_rag_when_corpus_empty(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is the capital of France?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" not in prompt


# ---------------------------------------------------------------------------
# Path 3 — Ingest path (Priority 1)
# ---------------------------------------------------------------------------

class TestIngestPath:

    def test_ingest_routes_to_wiki_agent(self, mm):
        rt   = make_runtime()
        conv = make_conv_agent()
        wiki = make_wiki_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki],
                               memory_manager=mm)

        ctrl.handle_task({
            "instruction": "ingest this document",
            "context":     {"raw_path": "/data/notes.md"},
        })

        assert wiki.run.called
        assert conv.run.called is False

    def test_ingest_does_not_set_fetch_rag(self, mm):
        rt   = make_runtime()
        wiki = make_wiki_agent()

        received: list[SubTask] = []
        def capture(subtask):
            received.append(subtask)
            return wiki.run.return_value
        wiki.run.side_effect = capture

        ctrl = ControllerAgent(runtime=rt, agents=[wiki], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "ingest file",
            "context":     {"raw_path": "/x.md"},
        })

        assert received[0].context["_routing"]["fetch_rag"]      is False
        assert received[0].context["_routing"]["fetch_episodic"] is False

    def test_ingest_fallback_when_wiki_not_registered(self, mm):
        """P1 fires but wiki_agent not registered → fallback to conv agent."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Fallback answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({
            "instruction": "ingest file",
            "context":     {"raw_path": "/x.md"},
        })

        assert result["status"] == "complete"
        assert conv.run.called


# ---------------------------------------------------------------------------
# Episodic write path (Priority 2)
# ---------------------------------------------------------------------------

class TestEpisodicWritePath:

    def test_write_episode_true_on_memory_keyword(self, mm, db_path):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task(
            {"instruction": "remember that I prefer step-by-step instructions"}
        )

        routing = conv._received[0].context["_routing"]
        assert routing["write_episode"] is True

    def test_episode_written_to_db(self, db_path, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task(
            {"instruction": "remember that I prefer step-by-step instructions"}
        )

        reader  = EpisodicMemoryReader(db_path=db_path)
        records = reader.by_recency(project_context="general")
        # At least one episode was written this session
        assert len(records) >= 1


# ---------------------------------------------------------------------------
# Episodic retrieval path (fetch_episodic=True)
# ---------------------------------------------------------------------------

class TestEpisodicRetrievalPath:

    def test_episodic_bullets_appear_in_prompt(self, db_path, mm):
        # Seed an episode directly
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "preference",
            subject         = "output format",
            content         = "User prefers step-by-step instructions.",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        # Instruction contains episodic keyword ("preferences") → P5 keyword match
        rt = make_runtime(infer_return="yes")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What are my formatting preferences?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[EPISODIC MEMORY]" in prompt

    def test_episodic_flag_in_routing(self, db_path, mm):
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "correction",
            subject         = "vault resolver",
            content         = "raw_path passed explicitly.",
            source          = "explicit",
            project_context = "general",
        )

        rt = make_runtime(infer_return="yes")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What is my workflow preference?"})

        routing = conv._received[0].context["_routing"]
        assert routing["fetch_episodic"] is True


# ---------------------------------------------------------------------------
# Working memory path
# ---------------------------------------------------------------------------

class TestWorkingMemoryPath:

    def test_prior_turns_appear_in_slot3(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Answer 2.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        task_id = "wm-test-task"

        # Add a prior turn directly to memory
        mm.add(
            role    = "user",
            content = "What is LORA?",
            task_id = task_id,
        )
        mm.add(
            role    = "agent",
            content = "LORA is a local research assistant.",
            task_id = task_id,
        )

        ctrl.handle_task({
            "task_id":     task_id,
            "instruction": "Tell me more about it.",
        })

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING MEMORY]" in prompt
        assert "LORA is a local research assistant." in prompt


# ---------------------------------------------------------------------------
# Tool stub path (Priority 3)
# ---------------------------------------------------------------------------

class TestToolStubPath:

    def test_tool_signal_sets_tools_to_call(self, mm):
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What are the latest oMLX changes?"})

        routing = conv._received[0].context["_routing"]
        assert "web_search" in routing["tools_to_call"]

    def test_tool_stub_does_not_add_tool_results_slot(self, mm):
        """Tool results slot absent until Phase 6 wires real dispatchers."""
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "What are the latest oMLX changes?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[TOOL RESULTS]" not in prompt


# ---------------------------------------------------------------------------
# _extract_file_op_content / _file_op_confirmation_line — unit tests
# ---------------------------------------------------------------------------

class TestExtractFileOpContent:

    def test_strips_leading_label_and_trailing_parenthetical(self):
        answer = (
            "Haiku about the sea:\n\n"
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky.\n\n"
            "(Attempting to save content to haiku.md)"
        )
        assert _extract_file_op_content(answer) == (
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky."
        )

    def test_no_label_or_parenthetical_passes_through_unchanged(self):
        answer = "Blue expanse so wide,\nWaves that whisper ancient tales,\nHorizon holds sky."
        assert _extract_file_op_content(answer) == answer

    def test_label_only_no_trailing_parenthetical(self):
        answer = "Summary:\n\nThe meeting covered three topics."
        assert _extract_file_op_content(answer) == "The meeting covered three topics."

    def test_trailing_parenthetical_only_no_label(self):
        answer = "The meeting covered three topics.\n\n(Saving this to summary.md)"
        assert _extract_file_op_content(answer) == "The meeting covered three topics."

    def test_content_with_internal_colon_not_mistaken_for_label(self):
        """A single-line answer with a colon must not be treated as a label
        line stripped down to nothing — the guard against zeroing out real
        content should keep it intact."""
        answer = "Remember: buy milk and walk the dog."
        assert _extract_file_op_content(answer) == answer

    def test_multiline_answer_with_colon_in_body_not_first_line(self):
        answer = "Notes:\n\nTODO: buy milk\nTODO: walk the dog"
        assert _extract_file_op_content(answer) == "TODO: buy milk\nTODO: walk the dog"

    def test_markdown_italicized_trailing_aside_is_stripped(self):
        """Observed live: the model wrapped its whole aside in markdown
        italics with a backticked filename, e.g.
        '*(This haiku has been generated and is ready to be saved as
        `haiku.md`.)*' — the plain (...) pattern alone doesn't match this
        because of the leading/trailing '*'."""
        answer = (
            "Blue expanse so wide,\n"
            "Waves crash on the sandy shore,\n"
            "Salt wind fills the air.\n\n"
            "*(This haiku has been generated and is ready to be saved as `haiku.md`.)*"
        )
        assert _extract_file_op_content(answer) == (
            "Blue expanse so wide,\n"
            "Waves crash on the sandy shore,\n"
            "Salt wind fills the air."
        )

    def test_parenthetical_that_is_the_whole_answer_is_not_stripped_to_empty(self):
        """Guard: if stripping the trailing parenthetical would leave nothing,
        keep the original text instead of writing an empty file."""
        answer = "(just kidding, no real content here)"
        assert _extract_file_op_content(answer) == answer


class TestFileOpConfirmationLine:

    def test_success_reports_actual_written_filename(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: wrote 34 characters to haiku.md",
        )
        assert _file_op_confirmation_line(result, "haiku.md") == "\n\n*(Saved to haiku.md)*"

    def test_success_reports_versioned_filename_when_original_existed(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: wrote 34 characters to haiku_2.md",
        )
        # fallback_path is the pre-versioning plan.file_op_path; the actual
        # written name (post version-cap fallback) must win when present.
        assert _file_op_confirmation_line(result, "haiku.md") == "\n\n*(Saved to haiku_2.md)*"

    def test_success_falls_back_to_plan_path_when_message_has_no_filename(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=True,
            result="OK: skipped duplicate append for turn_id=abc (already applied)",
        )
        assert _file_op_confirmation_line(result, "log.md") == "\n\n*(Saved to log.md)*"

    def test_failure_strips_error_prefix(self):
        result = _ToolResult(
            tool_name="file_op", parameters="", success=False,
            result="ERROR: path traversal outside project_root is not permitted",
        )
        assert _file_op_confirmation_line(result, "x.md") == (
            "\n\n*(Could not save — path traversal outside project_root is not permitted)*"
        )


# ---------------------------------------------------------------------------
# Deferred file_op dispatch (_execute_plan Step 7b)
# ---------------------------------------------------------------------------

class TestDeferredFileOpDispatch:
    """
    plan.file_op_deferred means Planner detected a file_op-shaped
    instruction whose content had to be generated by the agent first (see
    planner.py's P3 content-present-vs-deferred split). These tests bypass
    real Planner routing (patch.object on ctrl._planner) and patch
    MCPToolDispatcher wholesale — dispatch()'s own MCP-session behavior is
    already covered by test_mcp_tool_dispatcher.py; these only verify
    _execute_plan's new Step 7b wiring (content extraction, dispatch args,
    and the appended confirmation/failure line).
    """

    def _make_deferred_plan(self, path: str = "haiku.md", action: str = "write") -> RoutingPlan:
        return RoutingPlan(
            agent            = "conversational_agent",
            fetch_episodic   = False,
            fetch_rag        = False,
            priority         = 3,
            compound         = True,
            file_op_deferred = True,
            file_op_path     = path,
            file_op_action   = action,
        )

    def test_success_strips_label_paren_for_content_and_appends_confirmation(self, mm):
        raw_answer = (
            "Haiku about the sea:\n\n"
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky.\n\n"
            "(Attempting to save content to haiku.md)"
        )
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "file_op",
                parameters = "action='write' path='haiku.md'",
                result     = "OK: wrote 74 characters to haiku.md",
                success    = True,
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku about the sea and save it as haiku.md"}
            )

        # The label/parenthetical framing must not leak into the saved content.
        _, kwargs = mock_dispatcher.dispatch.call_args
        assert kwargs["context"]["file_op_content"] == (
            "Blue expanse so wide,\n"
            "Waves that whisper ancient tales,\n"
            "Horizon holds sky."
        )
        assert kwargs["context"]["file_op_path"]   == "haiku.md"
        assert kwargs["context"]["file_op_action"] == "write"
        assert kwargs["tools_to_call"] == ["file_op"]

        # The displayed/persisted answer keeps the model's own text verbatim
        # plus a deterministic (never model-narrated) confirmation line.
        assert result["answer"] == raw_answer + "\n\n*(Saved to haiku.md)*"

    def test_failure_appends_deterministic_failure_line(self, mm):
        raw_answer = "Blue expanse so wide, waves that whisper old secrets, horizon holds the sky."
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan(path="../../etc/passwd", action="write")

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = [
            _ToolResult(
                tool_name  = "file_op",
                parameters = "action='write' path='../../etc/passwd'",
                result     = "ERROR: path traversal outside project_root is not permitted",
                success    = False,
            )
        ]

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku and save it as ../../etc/passwd"}
            )

        assert result["answer"] == raw_answer + (
            "\n\n*(Could not save — path traversal outside project_root is not permitted)*"
        )

    def test_dispatch_exception_appends_failure_line_without_raising(self, mm):
        raw_answer = "Blue expanse so wide, waves that whisper old secrets, horizon holds the sky."
        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent(raw_answer)
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        plan = self._make_deferred_plan()

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.side_effect = RuntimeError("localist-mcp unreachable")

        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch("controller_agent.MCPToolDispatcher", return_value=mock_dispatcher):
            result = ctrl.handle_task(
                {"instruction": "write a haiku about the sea and save it as haiku.md"}
            )

        assert result["status"] == "complete"
        assert result["answer"] == raw_answer + "\n\n*(Could not save — localist-mcp unreachable)*"


# ---------------------------------------------------------------------------
# Prebuilt prompt passthrough in ConversationalAgent
# ---------------------------------------------------------------------------

class TestPrebuiltPromptPassthrough:

    def test_prebuilt_path_skips_internal_rag(self):
        """When _prebuilt_prompt is present, ConversationalAgent skips
        its own corpus query and uses the prebuilt prompt verbatim."""
        from conversational_agent import ConversationalAgent

        mm = MagicMock()   # mock MM — must NOT be called for query_corpus
        rt = MagicMock()
        rt.infer.return_value = "Prebuilt answer."

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)

        subtask = MagicMock()
        subtask.subtask_id = "pb-test"
        subtask.instruction = "What is LORA?"
        subtask.context = {
            "_prebuilt_prompt": "[USER]\nWhat is LORA?",
            "_prebuilt_system": "You are LORA.",
            "_routing":         {"fetch_rag": True},
        }

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == "Prebuilt answer."
        mm.query_corpus.assert_not_called()

    def test_without_prebuilt_normal_path_runs(self):
        from conversational_agent import ConversationalAgent

        rt = MagicMock()
        rt.infer.return_value = "Normal answer."

        agent = ConversationalAgent(runtime=rt, memory_manager=None)

        subtask = MagicMock()
        subtask.subtask_id  = "normal-test"
        subtask.instruction = "What is 2+2?"
        subtask.context     = {}

        result = agent.run(subtask)
        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == "Normal answer."


# ---------------------------------------------------------------------------
# wiki_doc wiring — _load_persona() frontmatter handling
# ---------------------------------------------------------------------------

_LORA_PERSONA_CONTENT = (
    "You are LORA, a local‑first thinking partner.\n"
    "You speak clearly, directly, and in a natural conversational tone.\n"
    "You use tools when they are needed and follow tool instructions precisely.\n"
    "When you state facts, you cite where they came from."
)


def _mock_doc(path_str: str, content: str):
    doc = MagicMock()
    doc.path = Path(path_str)
    doc.content = content
    return doc


class TestLoadPersonaWikiDoc:

    def test_no_frontmatter_byte_identical(self):
        """Zero-behavior-change: plain content → cache equals content[:2000] exactly."""
        mm = MagicMock()
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/lora-persona.md", _LORA_PERSONA_CONTENT)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert ctrl._persona_cache == _LORA_PERSONA_CONTENT[:2000]

    def test_frontmatter_stripped_body_only(self):
        """Forward-looking: frontmatter lines are excluded; body text is present."""
        content = (
            "---\n"
            "title: LORA Persona\n"
            "type: system\n"
            "created: 2026-06-01\n"
            "---\n"
            "\n"
            "You are LORA, a local-first thinking partner.\n"
            "You are helpful, concise, and precise.\n"
        )
        mm = MagicMock()
        mm.query_corpus.return_value = [
            _mock_doc("/wiki/lora-persona.md", content)
        ]

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)
        ctrl._load_persona()

        assert "title: LORA Persona" not in ctrl._persona_cache
        assert "type: system" not in ctrl._persona_cache
        assert "---" not in ctrl._persona_cache
        assert "You are LORA" in ctrl._persona_cache
        assert "You are helpful" in ctrl._persona_cache


# ---------------------------------------------------------------------------
# wiki_doc wiring — _load_user_profile() frontmatter handling
# ---------------------------------------------------------------------------

_PROFILE_CONTENT_NO_FM = (
    "## About Michael\n"
    "\n"
    "- Name: Michael\n"
    "- Role: Solo developer\n"
    "\n"
    "## Preferences\n"
    "\n"
    "- I prefer concise answers.\n"
)


class TestLoadUserProfileWikiDoc:

    def _make_ctrl_with_embed(self):
        mm = MagicMock()
        mm._embed_fn = lambda _: [0.0] * 768
        return ControllerAgent(runtime=make_runtime(), agents=[], memory_manager=mm)

    def test_no_frontmatter_byte_identical(self, tmp_path: Path):
        """Zero-behavior-change: no frontmatter → profile_lines identical to old logic."""
        profile_file = tmp_path / "michael.md"
        profile_file.write_text(_PROFILE_CONTENT_NO_FM, encoding="utf-8")

        # Reproduce what the old logic (raw.splitlines()) would have produced
        expected = [
            line.lstrip("- ").strip()
            for line in _PROFILE_CONTENT_NO_FM.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        ctrl = self._make_ctrl_with_embed()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("controller_agent.load_wiki_doc",
                   side_effect=lambda _: load_wiki_doc(profile_file)):
            ctrl._load_user_profile()

        assert ctrl._profile_lines == expected

    def test_frontmatter_excluded_from_profile_lines(self, tmp_path: Path):
        """Forward-looking: frontmatter YAML lines are not ingested as fact lines."""
        content = (
            "---\n"
            "title: Michael Profile\n"
            "type: user-profile\n"
            "created: 2026-06-01\n"
            "---\n"
            "\n"
            "## About Michael\n"
            "\n"
            "- Name: Michael\n"
            "- Role: Solo developer\n"
        )
        profile_file = tmp_path / "michael.md"
        profile_file.write_text(content, encoding="utf-8")

        ctrl = self._make_ctrl_with_embed()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("controller_agent.load_wiki_doc",
                   side_effect=lambda _: load_wiki_doc(profile_file)):
            ctrl._load_user_profile()

        lines = ctrl._profile_lines
        assert not any("title:" in l for l in lines)
        assert not any("type:" in l for l in lines)
        assert not any("---" in l for l in lines)
        assert "Name: Michael" in lines
        assert "Role: Solo developer" in lines


# ---------------------------------------------------------------------------
# _memory_key() and session_id cross-turn working memory
# ---------------------------------------------------------------------------

class TestMemoryKey:

    def test_prefers_session_id_when_present(self):
        """session_id in context takes precedence over task_id."""
        task = Task(task_id="xyz", instruction="hi", context={"session_id": "abc"})
        assert _memory_key(task) == "abc"

    def test_falls_back_to_task_id_when_absent(self):
        """Callers without session_id (e.g. ingest path) keep today's behavior."""
        task = Task(task_id="xyz", instruction="hi", context={})
        assert _memory_key(task) == "xyz"

    def test_same_session_id_shares_working_memory(self, db_path: Path):
        """Two handle_task() calls with the same session_id accumulate in one log."""
        mm   = MemoryManager(db_path=db_path)
        rt   = make_runtime()
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({
            "task_id":     "task-1",
            "instruction": "First question",
            "context":     {"session_id": "test-session"},
        })
        ctrl.handle_task({
            "task_id":     "task-2",
            "instruction": "Second question",
            "context":     {"session_id": "test-session"},
        })

        entries = mm.get_context_window(task_id="test-session", limit=20)
        user_instructions = [e["content"] for e in entries if e["role"] == "user"]
        assert "First question" in user_instructions
        assert "Second question" in user_instructions

    def test_different_task_ids_without_session_are_isolated(self, db_path: Path):
        """Callers without session_id keep isolated per-request memory."""
        mm   = MemoryManager(db_path=db_path)
        rt   = make_runtime()
        conv = make_conv_agent("Answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({
            "task_id":     "task-a",
            "instruction": "Question A",
        })
        ctrl.handle_task({
            "task_id":     "task-b",
            "instruction": "Question B",
        })

        entries_a = mm.get_context_window(task_id="task-a", limit=20)
        entries_b = mm.get_context_window(task_id="task-b", limit=20)

        contents_a = [e["content"] for e in entries_a]
        contents_b = [e["content"] for e in entries_b]

        assert "Question A" in contents_a
        assert "Question B" not in contents_a
        assert "Question B" in contents_b
        assert "Question A" not in contents_b


# ---------------------------------------------------------------------------
# wiki_doc wiring — RAG source frontmatter stripping
# ---------------------------------------------------------------------------

# Shape of how-localist-works.md frontmatter — confirmed from real corpus.
_RAG_WITH_FRONTMATTER = (
    "---\n"
    "title: Localist Agent Framework Manifest and Schema\n"
    "type: research-note\n"
    "query: Analyze how-localist-works.md\n"
    "created: 2026-06-18\n"
    "updated: 2026-06-18\n"
    "---\n"
    "\n"
    "## Summary\n"
    "\n"
    "Localist is a local-first AI agent framework.\n"
    "It runs entirely on-device with no cloud dependencies.\n"
)

_RAG_WITHOUT_FRONTMATTER = (
    "## Overview\n"
    "\n"
    "This document has no frontmatter block at all.\n"
    "All content is body text.\n"
)


def _make_rag_ctrl(docs: list) -> tuple:
    """Return (ctrl, conv_agent) with query_corpus() returning *docs*."""
    mm = MagicMock()
    mm.query_corpus.return_value = docs
    conv = make_conv_agent()
    ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
    return ctrl, conv


class TestRagSourceFrontmatterStripping:

    def test_frontmatter_stripped_from_rag_source(self):
        """Frontmatter lines must not appear in the RagSource content passed to PromptBuilder."""
        doc = _mock_doc("/wiki/how-localist-works.md", _RAG_WITH_FRONTMATTER)
        doc.relevance_score = 0.9  # above threshold

        ctrl, conv = _make_rag_ctrl([doc])
        ctrl.handle_task({"instruction": "check the wiki for Localist framework"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "title: Localist Agent Framework" not in prompt
        assert "type: research-note"              not in prompt
        assert "query: Analyze how-localist-works" not in prompt
        assert "created: 2026-06-18"              not in prompt
        assert "---"                              not in prompt
        # Body text must still be present
        assert "Localist is a local-first AI agent framework" in prompt

    def test_no_frontmatter_rag_source_unchanged(self):
        """Zero-behavior-change: content without frontmatter is passed through unmodified."""
        doc = _mock_doc("/wiki/plain-doc.md", _RAG_WITHOUT_FRONTMATTER)
        doc.relevance_score = 0.9

        ctrl, conv = _make_rag_ctrl([doc])
        ctrl.handle_task({"instruction": "check the wiki for plain doc overview"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # The full body text must appear in the prompt, character-for-character
        assert "This document has no frontmatter block at all." in prompt
        assert "All content is body text." in prompt

    def test_rag_filter_and_exclusion_unaffected(self):
        """Score filter and lora-persona.md exclusion must be unchanged after the fix."""
        low_score_doc   = _mock_doc("/wiki/low-score.md", "low relevance content")
        low_score_doc.relevance_score = 0.3   # below 0.55 threshold — must be excluded

        persona_doc     = _mock_doc("/wiki/lora-persona.md", "persona content")
        persona_doc.relevance_score = 0.9     # above threshold but excluded by path rule

        good_doc        = _mock_doc("/wiki/good.md", "relevant body text about Localist")
        good_doc.relevance_score = 0.9

        ctrl, conv = _make_rag_ctrl([low_score_doc, persona_doc, good_doc])
        ctrl.handle_task({"instruction": "check the wiki for Localist"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        # Only the good_doc should appear
        assert "relevant body text about Localist" in prompt
        assert "low relevance content"             not in prompt
        assert "persona content"                   not in prompt


# ---------------------------------------------------------------------------
# Graph query fetch — Step 5c wiring
# ---------------------------------------------------------------------------

class TestGraphQueryFetch:
    """
    Tests for _execute_plan() Step 5c: graph query fetch and Slot 5b rendering.

    Tests 1-5 inject a specific RoutingPlan directly (bypassing real Planner
    routing) via patch.object so the graph_query field can be set precisely.
    Test 6 routes through the real Planner with a real MemoryManager to verify
    the "pure/minimal" guarantee end-to-end.
    """

    def _make_graph_plan(self, direction: str, node_id: int, stem: str) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            compound       = False,
            priority       = 3,
            graph_query    = (direction, node_id, stem),
        )

    # 1. Incoming, 2 edges — [GRAPH RESULT] appears with both source page stems
    def test_incoming_populated(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.return_value = [
            GraphEdgeResult(
                link_text       = "[[lora-persona]]",
                target_path     = "lora-persona",
                target_resolved = True,
                node_title      = "How Localist Works",
                node_doc_path   = "/wiki/how-localist-works.md",
            ),
            GraphEdgeResult(
                link_text       = "[[lora-persona]]",
                target_path     = "lora-persona",
                target_resolved = True,
                node_title      = "Localist Build Order",
                node_doc_path   = "/wiki/localist-build-order.md",
            ),
        ]

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Incoming answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 7, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "Pages linking to lora-persona:" in prompt
        assert "how-localist-works" in prompt
        assert "localist-build-order" in prompt

    # 2. Outgoing, mixed resolved+unresolved — confirms link_text used for
    #    unresolved display (not target_path), catching the link_text-vs-
    #    target_path bug described in the prompt spec.
    def test_outgoing_mixed_link_text_vs_target_path(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_outgoing_links.return_value = [
            GraphEdgeResult(
                link_text       = "localist-master-project-outline",
                target_path     = "localist-master-project-outline",
                target_resolved = True,
                node_title      = "Localist Master Project Outline",
                node_doc_path   = "/wiki/localist-master-project-outline.md",
            ),
            GraphEdgeResult(
                link_text       = "Localist Software Stack Overview",  # original casing
                target_path     = "localist-software-stack-overview",  # normalized — must NOT appear
                target_resolved = False,
                node_title      = None,
                node_doc_path   = None,
            ),
        ]

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Outgoing answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("outgoing", 3, "localist-build-order")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what does localist-build-order link to"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "localist-master-project-outline" in prompt
        # Unresolved entry must show original link_text, not the normalized target_path
        assert '"Localist Software Stack Overview"' in prompt
        assert "localist-software-stack-overview" not in prompt

    # 3. Zero edges — [GRAPH RESULT] still present (clean-omission exception)
    def test_zero_edges_slot_still_emitted(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.return_value = []

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("No backlinks answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 5, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" in prompt
        assert "No pages link to lora-persona." in prompt

    # 4. Fetch failure — _execute_plan does not raise; [GRAPH RESULT] absent
    def test_fetch_failure_degrades_gracefully(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []
        mm_mock.get_backlinks.side_effect = RuntimeError("SQLite locked")

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Degraded answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)
        plan = self._make_graph_plan("incoming", 5, "lora-persona")

        with patch.object(ctrl._planner, "route", return_value=plan):
            # Must not raise
            result = ctrl.handle_task({"instruction": "what links to lora-persona"})

        assert result["status"] == "complete"
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" not in prompt

    # 5. Non-graph-query plan — get_backlinks/get_outgoing_links never called
    def test_no_graph_query_no_edge_fetch(self):
        mm_mock = MagicMock()
        mm_mock.query_corpus.return_value = []

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("Direct answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm_mock)

        # "What is 2+2?" triggers no graph pattern → plan.graph_query is None
        ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[GRAPH RESULT]" not in prompt
        mm_mock.get_backlinks.assert_not_called()
        mm_mock.get_outgoing_links.assert_not_called()

    # 6. Purity end-to-end: real Planner + real MemoryManager.
    #    Episodic records and RAG docs exist in the DB and WOULD appear if
    #    their fetch conditions fired. Verify they do not leak into the prompt.
    def test_p3c_purity_no_rag_or_episodic_slots(self, tmp_path):
        db_path = tmp_path / "purity.db"
        mm = MemoryManager(db_path=db_path)

        # Graph: "lora-persona" node + "how-localist-works" backlink source
        lp_id  = mm.upsert_graph_node(
            doc_path  = str(tmp_path / "lora-persona.md"),
            node_type = "wiki",
            title     = "LORA Persona",
        )
        src_id = mm.upsert_graph_node(
            doc_path  = str(tmp_path / "how-localist-works.md"),
            node_type = "wiki",
            title     = "How Localist Works",
        )
        mm.upsert_graph_edge(
            source_node_id  = src_id,
            source_doc_path = str(tmp_path / "how-localist-works.md"),
            target_path     = "lora-persona",
            target_node_id  = lp_id,
            target_resolved = True,
            link_text       = "lora-persona",
        )

        # Episodic record (would appear in [EPISODIC MEMORY] if fetch_episodic fired)
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "preference",
            subject         = "output format",
            content         = "PURITY_LEAK_EPISODIC: should not appear in prompt",
            source          = "explicit",
            confidence      = 1.0,
            project_context = "general",
        )

        # RAG document (would appear in [CONTEXT] if fetch_rag fired)
        mm.index_document(
            path     = tmp_path / "background-doc.md",
            doc_type = "wiki",
            content  = "PURITY_LEAK_RAG localist persona links wiki architecture",
        )

        # Real Planner routes "what links to lora-persona" → P3c
        rt   = make_runtime(infer_return="yes")  # "yes" would fire episodic if reached
        conv = make_conv_agent("Graph answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]

        # Graph result must be present
        assert "[GRAPH RESULT]" in prompt
        assert "how-localist-works" in prompt

        # No other context slots must leak — purity guarantee
        assert "[CONTEXT]"       not in prompt
        assert "[EPISODIC MEMORY]" not in prompt
        assert "[USER PROFILE]"  not in prompt
        assert "PURITY_LEAK_EPISODIC" not in prompt
        assert "PURITY_LEAK_RAG"      not in prompt



# ---------------------------------------------------------------------------
# Step 5d — WorkingMemoryState (Slot 6A) wiring
# ---------------------------------------------------------------------------

class TestWorkingStateSlot6A:
    """
    Verifies Step 5d: WorkingMemoryState assembly and the P3c exclusivity guard.

    Tests inject RoutingPlans directly via patch.object so the graph_query
    field can be set precisely — matching the pattern used in TestGraphQueryFetch.
    """

    def _make_rag_plan(self, *, fetch_rag: bool = True, graph_query=None) -> RoutingPlan:
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = fetch_rag,
            priority       = 4,
            graph_query    = graph_query,
        )

    # 1. Non-P3c route with RAG sources present → working_state constructed,
    #    active_artifacts matches the RAG source paths exactly.
    def test_non_p3c_with_rag_sources_builds_working_state(self):
        doc = _mock_doc("/wiki/localist-arch.md", "Localist architecture content here.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=True)

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for Localist architecture"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" in prompt
        assert "active_artifacts:" in prompt
        # Path must be the exact path from the RAG source
        assert "/wiki/localist-arch.md" in prompt

    # 2. Non-P3c route with no RAG sources and no usable current_project →
    #    working_state stays None; no [WORKING STATE] block in prompt.
    def test_non_p3c_no_rag_no_project_working_state_absent(self):
        mm = MagicMock()
        mm.query_corpus.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=False)

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "What is 2+2?"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" not in prompt

    # 3. P3c exclusivity guard: graph_query is not None with fetch_rag=True and
    #    docs present in scope → working_state is NOT constructed regardless.
    #    This is the regression test for the Phase C purity guarantee.
    def test_p3c_graph_query_excludes_working_state(self):
        doc = _mock_doc("/wiki/lora-persona.md", "Some persona content.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        mm.get_backlinks.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        # Plan with both graph_query set AND fetch_rag=True — hypothetical scenario
        # that exercises the guard directly, regardless of what the Planner produces.
        plan = self._make_rag_plan(
            fetch_rag    = True,
            graph_query  = ("incoming", 5, "lora-persona"),
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "what links to lora-persona"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[WORKING STATE]" not in prompt, (
            "Slot 6A must not be constructed on P3c routes (graph_query is not None)"
        )

    # 4. Regression guard: existing RAG path behaviour is unchanged — answer,
    #    sources, and [CONTEXT] slot are unaffected by the Step 5d addition.
    def test_regression_rag_answer_and_sources_unchanged(self):
        doc = _mock_doc("/wiki/localist-arch.md", "check the wiki Localist architecture content.")
        doc.relevance_score = 0.9

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]
        conv = make_conv_agent("Architecture answer.")
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)
        plan = self._make_rag_plan(fetch_rag=True)

        with patch.object(ctrl._planner, "route", return_value=plan):
            result = ctrl.handle_task(
                {"instruction": "check the wiki for Localist architecture"}
            )

        # Core result unchanged
        assert result["status"] == "complete"
        assert result["answer"] == "Architecture answer."
        # [CONTEXT] slot still present — RAG wiring unaffected
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "[CONTEXT]" in prompt
        assert "Localist architecture content" in prompt


# ---------------------------------------------------------------------------
# Post-P4a removal: query_corpus always called without doc_type (Part 4B)
# ---------------------------------------------------------------------------

class TestQueryCorpusNeverReceivesDocType:
    """
    After removing force_rag, Step 4's query_corpus() call must never pass
    doc_type — the kwarg was dropped entirely (not set to None explicitly),
    so it should be absent from call_kwargs.

    Any plan with fetch_rag=True exercises this code path.
    """

    def test_step4_query_corpus_has_no_doc_type_kwarg(self):
        """Step 4 query_corpus() must not pass doc_type under any plan."""
        mm = MagicMock()
        mm.query_corpus.return_value = []
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = True,
            priority       = 4,
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for Localist"})

        assert mm.query_corpus.call_count >= 1
        _, step4_kwargs = mm.query_corpus.call_args_list[0]
        assert "doc_type" not in step4_kwargs, (
            f"doc_type must not be passed to query_corpus after force_rag removal; "
            f"got doc_type={step4_kwargs.get('doc_type')!r}"
        )


# ---------------------------------------------------------------------------
# Post-P4a removal: relevance threshold is now unconditional (Part 4C)
# ---------------------------------------------------------------------------

class TestRelevanceThresholdUnconditional:
    """
    Under the old code, force_rag=True bypassed the 0.55 relevance_score
    threshold so documents below it were included in rag_sources.  After
    removing force_rag, the threshold is unconditional: a low-scoring document
    must always be excluded regardless of plan contents.
    """

    def test_low_score_doc_excluded_no_bypass(self):
        """A doc with relevance_score < 0.55 must not appear in the prompt."""
        low_doc = _mock_doc("/wiki/low-relevance.md", "Some low-relevance content.")
        low_doc.relevance_score = 0.40   # below 0.55 — would have been included by force_rag

        mm = MagicMock()
        mm.query_corpus.return_value = [low_doc]
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=make_runtime(), agents=[conv], memory_manager=mm)

        plan = RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = True,
            priority       = 4,
        )

        with patch.object(ctrl._planner, "route", return_value=plan):
            ctrl.handle_task({"instruction": "check the wiki for LORA"})

        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "Some low-relevance content." not in prompt, (
            "Low-score document must be excluded: threshold is now unconditional"
        )
        assert "[CONTEXT]" not in prompt


# ---------------------------------------------------------------------------
# RoutingPlan no longer accepts force_rag keyword argument (Part 4D)
# ---------------------------------------------------------------------------

class TestRoutingPlanNoForceRagField:
    """
    Confirm force_rag was genuinely removed from the RoutingPlan dataclass,
    not merely left unused.  Passing force_rag=True must raise TypeError.
    """

    def test_force_rag_kwarg_raises_type_error(self):
        import pytest
        with pytest.raises(TypeError):
            RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                force_rag      = True,   # no longer a field — must raise
            )
