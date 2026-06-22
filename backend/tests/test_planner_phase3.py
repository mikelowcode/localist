"""
Phase 3 integration tests — LORA rule-based Planner.

Covers:
  - Each priority level fires correctly (P1–P6)
  - Compound detection fires before P1
  - ControllerAgent._execute() uses RoutingPlan for agent selection
  - Fallback when requested agent is not registered
  - RoutingPlan metadata is passed into SubTask context
  - Priority 3c graph-query wiring (P3c)

All tests use mocks — no SQLite, no runtime calls for P1–P4 and P6.
P3c tests use a real MemoryManager with a temporary SQLite database.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memory_manager import MemoryManager
from planner import Planner, RoutingPlan, extract_graph_query, resolve_graph_target
from controller_agent import (
    ControllerAgent, Task, TaskStatus, SubTask, AgentResult
)


def make_mm_with_nodes(tmp_path: Path) -> tuple[MemoryManager, dict[str, int]]:
    """Create a MemoryManager with five known graph_nodes for P3c tests."""
    db = tmp_path / "planner_p3c_test.db"
    mm = MemoryManager(db_path=db)
    node_ids: dict[str, int] = {}
    for stem in [
        "how-localist-works",
        "localist-build-order",
        "localist-master-project-outline",
        "localist-software-stack",
        "lora-persona",
    ]:
        nid = mm.upsert_graph_node(
            doc_path  = str(tmp_path / f"{stem}.md"),
            node_type = "wiki",
            title     = stem.replace("-", " ").title(),
        )
        node_ids[stem] = nid
    return mm, node_ids


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

    def test_p4a_identity_returns_priority_4(self):
        """P4a must set priority=4 — not inherit the default 6 used by P6."""
        p = Planner(runtime=make_runtime(infer_return="no"))
        for instruction in ("Who are you?", "What is Localist?", "What can you do?"):
            plan = p.route(instruction, context={})
            assert plan.force_rag is True, f"P4a did not fire for {instruction!r}"
            assert plan.priority == 4, (
                f"Expected priority=4 for {instruction!r}; got priority={plan.priority}"
            )


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


# ---------------------------------------------------------------------------
# Graph-query extraction and name resolution
# ---------------------------------------------------------------------------

# 5-stem test set per LOCALIST-Architecture.md §8.7.
# Note: the real wiki/ directory also contains a "michael" page not in this
# set. Tests use this fixed list so they are isolated from filesystem state.
_TEST_STEMS = [
    "how-localist-works",
    "localist-build-order",
    "localist-master-project-outline",
    "localist-software-stack",
    "lora-persona",
]


class TestGraphQueryExtraction:

    # 1. Pattern A matches (outgoing, anchored regex)
    def test_pattern_a_outgoing(self):
        result = extract_graph_query("What does localist-build-order link to?")
        assert result == ("outgoing", "localist-build-order")

    # 2. Pattern A does NOT match trailing content after "link to"
    def test_pattern_a_no_trailing_content(self):
        result = extract_graph_query("what does X link to and why")
        assert result is None

    # 3. Pattern B longest-phrase-first: "show me backlinks for" wins.
    #    If phrases were checked shortest-first, a hypothetical shorter prefix
    #    could steal the match; this test confirms the longest phrase fires
    #    and the remainder is correctly extracted.
    def test_pattern_b_longest_phrase_first(self):
        result = extract_graph_query("show me backlinks for lora-persona")
        assert result == ("incoming", "lora-persona")

    # 4. Pattern B basic case (case + trailing ?)
    def test_pattern_b_what_links_to(self):
        result = extract_graph_query("What links to lora-persona?")
        assert result == ("incoming", "lora-persona")

    # 5. Pattern C matches (outgoing lead-phrase)
    def test_pattern_c_outgoing(self):
        result = extract_graph_query("links from localist-build-order")
        assert result == ("outgoing", "localist-build-order")

    # 6. Degenerate empty remainder — extraction still returns a match;
    #    resolution handles the failure, not extraction.
    def test_pattern_b_empty_remainder(self):
        result = extract_graph_query("what links to")
        assert result == ("incoming", "")

    # 7. No pattern matches at all
    def test_no_match(self):
        result = extract_graph_query("what is the weather today")
        assert result is None


class TestGraphNameResolution:

    # 8. Tier 1: remainder is a substring of exactly one stem
    def test_tier1_remainder_in_stem(self):
        result = resolve_graph_target("software stack", _TEST_STEMS)
        assert result == "localist-software-stack"

    # 9. Tier 1: stem is a substring of remainder (other direction)
    def test_tier1_stem_in_remainder(self):
        # "lora-persona" is a substring of the normalized remainder
        result = resolve_graph_target("info about lora-persona page", _TEST_STEMS)
        assert result == "lora-persona"

    # 10. Tier 1: ambiguous — "localist" substring-matches multiple stems
    def test_tier1_ambiguous(self):
        # "localist" appears in 4 of the 5 test stems → multi-match → None
        result = resolve_graph_target("localist", _TEST_STEMS)
        assert result is None

    # 11. Tier 2 fallback: Tier 1 finds zero, Tier 2 succeeds with ratio ≥ 0.5
    def test_tier2_fallback_single_match(self):
        # Normalized: "build-order-for-localist"
        # Tier 1: no stem is a substring of this and it is not in any stem.
        # Tier 2: query_tokens={"build","order","localist"} (3, after removing "for")
        #   localist-build-order: intersection={"build","order","localist"} → ratio 3/3 = 1.0 ✓
        #   all others: only "localist" overlaps → ratio 1/3 < 0.5 ✗
        result = resolve_graph_target("build order for localist", _TEST_STEMS)
        assert result == "localist-build-order"

    # 12. Tier 2 skipped: <2 meaningful tokens after stopword removal.
    #     "the overview" → normalized "the-overview"; Tier 1 finds no matches;
    #     Tier 2 tokens = {"the","overview"} → after stopwords: {"overview"} → 1 < 2.
    #     This is the "skipped, not scored" path (not a ratio failure).
    def test_tier2_skipped_too_few_tokens(self):
        result = resolve_graph_target("the overview", _TEST_STEMS)
        # Confirm the token count before the skip:
        from planner import _normalize_graph_text, _GRAPH_STOPWORDS
        normalized = _normalize_graph_text("the overview")
        meaningful = set(normalized.split("-")) - _GRAPH_STOPWORDS
        assert len(meaningful) < 2, (
            f"Expected <2 meaningful tokens, got {meaningful} — "
            "this would exercise Tier 2 scoring, not the skip path"
        )
        assert result is None

    # 13. Tier 2 ambiguous: two stems both clear the 0.5 ratio threshold
    def test_tier2_ambiguous(self):
        # Normalized: "localist-project-build"
        # Tier 1: no substring match either direction.
        # Tier 2: query_tokens={"localist","project","build"} (3 tokens)
        #   localist-build-order: intersection={"localist","build"} → 2/3 ≈ 0.67 ✓
        #   localist-master-project-outline: intersection={"localist","project"} → 2/3 ≈ 0.67 ✓
        #   two matches → ambiguous → None
        result = resolve_graph_target("localist project build", _TEST_STEMS)
        assert result is None

    # 14. Empty remainder: Tier 1 treats "" as substring of every stem
    #     (all 5 match) → ambiguous → None. No Tier 2 is reached.
    def test_empty_remainder(self):
        result = resolve_graph_target("", _TEST_STEMS)
        assert result is None

    # 15. Completely unrelated remainder: no substring matches, no token overlap
    def test_unrelated_remainder(self):
        result = resolve_graph_target("the weather forecast", _TEST_STEMS)
        assert result is None


# ---------------------------------------------------------------------------
# Priority 3c — graph-query wiring (uses real SQLite via make_mm_with_nodes)
# ---------------------------------------------------------------------------

class TestPlannerP3c:

    # 1. Basic incoming graph-query matches and returns graph_query tuple
    def test_incoming_graph_query_resolves(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        assert plan.tools_to_call == []
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False
        assert plan.compound       is False

    # 2. Outgoing graph-query ("what does X link to") resolves correctly
    def test_outgoing_graph_query_resolves(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("What does lora-persona link to?", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "outgoing"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]

    # 3. file_op guard: "save" triggers inline guard → P3c returns None,
    #    P3 fires on "save" and returns file_op plan
    def test_file_op_guard_defers_to_p3(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, save the results", context={})
        # P3c defers; P3 fires
        assert "file_op" in plan.tools_to_call
        assert plan.graph_query is None

    # 4. Ordering regression: graph-query wins over a web_search-only P3 match.
    #    "today" is a web_search keyword; under the old (wrong) ordering where
    #    P3c ran AFTER P3, P3 would win. P3c's inline guard does NOT block
    #    web_search, so with correct ordering P3c resolves the graph query first.
    def test_p3c_beats_web_search_p3(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona today", context={})
        assert plan.graph_query is not None
        direction, node_id, resolved_stem = plan.graph_query
        assert direction     == "incoming"
        assert resolved_stem == "lora-persona"
        assert node_id       == node_ids["lora-persona"]
        # Must NOT have triggered a web_search
        assert "web_search" not in plan.tools_to_call

    # 5. Unresolvable name → P3c returns None and falls through to P6
    def test_unresolvable_name_falls_through(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(infer_return="no"), memory_manager=mm)
        plan = p.route("what links to the localist pages", context={})
        # Name resolution fails for ambiguous "localist" → P3c returns None.
        # No other priority fires → P6 direct answer.
        assert plan.graph_query is None
        assert plan.agent == "conversational_agent"
        assert plan.tools_to_call == []

    # 6. P1 beats P3c: ingest keyword ensures wiki_agent routing before P3c runs
    def test_p1_beats_p3c(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("ingest this file, what links to lora-persona", context={})
        assert plan.agent == "wiki_agent"
        assert plan.graph_query is None

    # 7. No MemoryManager: P3c skips and route() falls through to P6
    def test_no_memory_manager_skips_p3c(self):
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route("what links to lora-persona", context={})
        assert plan.graph_query is None
        assert plan.agent == "conversational_agent"
        assert plan.tools_to_call == []
