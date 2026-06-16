"""
Phase 3 integration tests — LORA rule-based Planner.

Covers:
  - Each priority level fires correctly (P1–P6)
  - Compound detection fires before P1
  - ControllerAgent._execute() uses RoutingPlan for agent selection
  - Fallback when requested agent is not registered
  - RoutingPlan metadata is passed into SubTask context

All tests use mocks — no SQLite, no runtime calls for P1–P4 and P6.
"""

from unittest.mock import MagicMock, patch
from planner import Planner, RoutingPlan
from controller_agent import (
    ControllerAgent, Task, TaskStatus, SubTask, AgentResult
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_runtime(infer_return="no", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_agent(name, answer="Mock answer."):
    agent = MagicMock()
    agent.name = name
    agent.can_handle.return_value = True
    agent.run.return_value = AgentResult(
        subtask_id = "test-0",
        agent_name = name,
        status     = TaskStatus.COMPLETE,
        output     = {"answer": answer, "sources": [], "grounded": False},
    )
    return agent


# ---------------------------------------------------------------------------
# Planner unit tests — priority firing
# ---------------------------------------------------------------------------

class TestPlannerPriorities:

    def test_p1_raw_path_routes_to_wiki(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("do something", context={"raw_path": "/f.md"})
        assert plan.agent == "wiki_agent"
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False

    def test_p1_keyword_routes_to_wiki(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this document", context={})
        assert plan.agent == "wiki_agent"

    def test_p1_beats_p2(self):
        """raw_path in context + memory keyword → Priority 1 wins."""
        p = Planner(runtime=make_runtime())
        plan = p.route("remember that you should ingest this",
                       context={"raw_path": "/x.md"})
        assert plan.agent         == "wiki_agent"
        assert plan.write_episode is False

    def test_p2_memory_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("remember that I prefer dark mode", context={})
        assert plan.write_episode is True
        assert plan.compound      is True
        assert plan.agent         == "conversational_agent"

    def test_p2_forget_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("forget that preference", context={})
        assert plan.write_episode is True

    def test_p3_web_search_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("What are the latest oMLX changes?", context={})
        assert "web_search" in plan.tools_to_call
        assert plan.compound is True

    def test_p3_file_op_keyword(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("read the file notes.md", context={})
        assert "file_op" in plan.tools_to_call

    def test_p4_explicit_wiki_keyword_fires(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("check the wiki for LORA memory system", context={})
        assert plan.fetch_rag      is True
        assert plan.fetch_episodic is True

    def test_p4_no_wiki_keyword_falls_through(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("Tell me about the LORA memory system", context={})
        assert plan.fetch_rag is False

    def test_p5_yes_returns_episodic(self):
        p = Planner(runtime=make_runtime(infer_return="yes"))
        plan = p.route("What are my formatting preferences?", context={})
        assert plan.fetch_episodic is True

    def test_p5_no_falls_to_p6(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("What is 2+2?", context={})
        assert plan.fetch_episodic is False
        assert plan.fetch_rag      is False
        assert plan.agent          == "conversational_agent"

    def test_p6_direct_all_false(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("What is 2+2?", context={})
        assert plan.tools_to_call == []
        assert plan.write_episode  is False
        assert plan.compound       is False


# ---------------------------------------------------------------------------
# Compound detection
# ---------------------------------------------------------------------------

class TestCompoundDetection:

    def test_tool_ingest_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Search for the latest oMLX release notes and add to wiki",
            context={},
        )
        assert plan.agent         == "wiki_agent"
        assert "web_search"       in plan.tools_to_call
        assert plan.compound      is True

    def test_compound_fires_before_p1(self):
        """Without compound detection, P1 would win and drop the tool signal."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "Get the latest release notes and ingest them",
            context={},
        )
        assert "web_search" in plan.tools_to_call

    def test_ingest_only_not_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("ingest this document", context={})
        assert plan.tools_to_call == []
        assert plan.compound      is False

    def test_tool_only_not_wiki_compound(self):
        p = Planner(runtime=make_runtime())
        plan = p.route("What are the latest changes?", context={})
        assert plan.agent    == "conversational_agent"
        assert plan.compound is True   # P3 sets compound=True for tool+response


# ---------------------------------------------------------------------------
# ControllerAgent integration
# ---------------------------------------------------------------------------

class TestControllerAgentRouting:

    def test_conversational_query_short_circuits_synthesizer(self):
        """ConversationalAgent result bypasses Synthesizer."""
        rt = make_runtime(infer_return="no")
        conv = make_agent("conversational_agent", answer="42 is the answer.")
        wiki = make_agent("wiki_agent")

        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki])
        result = ctrl.handle_task({"instruction": "What is 42?"})

        assert result["status"]  == "complete"
        assert result["answer"]  == "42 is the answer."
        assert conv.run.called
        assert wiki.run.called   is False

    def test_ingest_routes_to_wiki_agent(self):
        """raw_path in context → wiki_agent is dispatched."""
        rt = make_runtime()
        conv = make_agent("conversational_agent")
        wiki = make_agent("wiki_agent")
        # WikiAgent returns ingest-style output (no "answer" key → Synthesizer)
        wiki.run.return_value = AgentResult(
            subtask_id = "test-0",
            agent_name = "wiki_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"new_pages": [], "applied": False},
        )

        ctrl = ControllerAgent(runtime=rt, agents=[conv, wiki])
        result = ctrl.handle_task({
            "instruction": "ingest this file",
            "context":     {"raw_path": "/data/notes.md"},
        })

        assert wiki.run.called
        assert conv.run.called is False

    def test_routing_metadata_in_subtask_context(self):
        """RoutingPlan fields are forwarded into SubTask.context._routing."""
        rt = make_runtime(infer_return="no")
        captured_subtasks = []

        conv = MagicMock()
        conv.name = "conversational_agent"
        conv.can_handle.return_value = True

        def capture_run(subtask):
            captured_subtasks.append(subtask)
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = "conversational_agent",
                status     = TaskStatus.COMPLETE,
                output     = {"answer": "ok", "sources": [], "grounded": False},
            )
        conv.run.side_effect = capture_run

        ctrl = ControllerAgent(runtime=rt, agents=[conv])
        ctrl.handle_task({"instruction": "What is LORA?"})

        assert len(captured_subtasks) == 1
        routing = captured_subtasks[0].context.get("_routing")
        assert routing is not None, "_routing key must be present in SubTask.context"
        assert "fetch_rag"      in routing
        assert "fetch_episodic" in routing
        assert "tools_to_call"  in routing
        assert "write_episode"  in routing

    def test_fallback_when_requested_agent_not_registered(self):
        """If wiki_agent not registered but instruction triggers P1,
        fall back to conversational_agent."""
        rt = make_runtime()
        conv = make_agent("conversational_agent")

        ctrl = ControllerAgent(runtime=rt, agents=[conv])
        # P1 will fire (raw_path present) → requests wiki_agent → not found
        # → fallback to conversational_agent
        result = ctrl.handle_task({
            "instruction": "process this",
            "context":     {"raw_path": "/x.md"},
        })

        assert result["status"] == "complete"
        assert conv.run.called

    def test_no_agents_returns_failed(self):
        """No agents registered at all → failed result."""
        rt = make_runtime()
        ctrl = ControllerAgent(runtime=rt, agents=[])
        result = ctrl.handle_task({"instruction": "What is LORA?"})
        assert result["status"] == "failed"
