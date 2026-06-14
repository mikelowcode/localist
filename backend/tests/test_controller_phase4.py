"""
Phase 4 integration tests — LORA ControllerAgent execution pipeline.

Covers:
  - Direct answer path: P6 fallback → PromptBuilder slot 2 only
  - RAG path: P4 corpus hit → fetch_rag=True → rag_sources in prompt
  - Ingest path: P1 → wiki_agent dispatched, Synthesizer called
  - Episodic write path: P2 → EpisodicMemoryWriter called
  - Episodic retrieval path: fetch_episodic=True → bullets in prompt
  - Prebuilt prompt passthrough: ConversationalAgent uses _prebuilt_prompt
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
    TaskStatus,
    AgentResult,
    SubTask,
)
from memory_manager import (
    MemoryManager,
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
)
from prompt_builder import PromptBuilder


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
#
# Document content and instruction are chosen so Jaccard keyword overlap
# scores 0.6 (above the 0.4 Planner threshold).
# "LORA research assistant" vs "LORA research assistant agentic local":
#   intersection={lora,research,assistant}=3  union=5  → 0.60
# "LORA SQLite memory" vs "LORA SQLite memory episodic storage":
#   intersection={lora,sqlite,memory}=3  union=5  → 0.60

class TestRAGPath:

    def test_fetch_rag_true_when_corpus_hits(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "LORA research assistant agentic local",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent("RAG answer.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        result = ctrl.handle_task({"instruction": "LORA research assistant"})

        assert result["status"] == "complete"
        routing = conv._received[0].context["_routing"]
        assert routing["fetch_rag"] is True

    def test_rag_sources_appear_in_prompt(self, mm, db_path):
        mm.index_document(
            path     = db_path.parent / "fake_wiki.md",
            doc_type = "wiki",
            content  = "LORA SQLite memory episodic storage",
        )

        rt   = make_runtime(infer_return="no")
        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)

        ctrl.handle_task({"instruction": "LORA SQLite memory"})

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
