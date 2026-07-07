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

import math
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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

    def test_p3_file_op_content_present_quoted_dispatches_immediately(self):
        """Destination phrase + quoted literal content → unchanged old
        behavior: tools_to_call=["file_op"], not deferred."""
        p = Planner(runtime=make_runtime())
        plan = p.route('save it as notes.md: "buy milk"', context={})
        assert "file_op" in plan.tools_to_call
        assert plan.file_op_deferred is False
        assert plan.compound is True

    def test_p3_file_op_content_present_fenced_dispatches_immediately(self):
        """Destination phrase + fenced literal content → unchanged old
        behavior: tools_to_call=["file_op"], not deferred."""
        p = Planner(runtime=make_runtime())
        plan = p.route("save it as notes.md: ```buy milk```", context={})
        assert "file_op" in plan.tools_to_call
        assert plan.file_op_deferred is False

    def test_p3_file_op_generation_required_is_deferred(self):
        """No literal content in the instruction — content must be composed
        by the agent first — so file_op is NOT dispatched yet."""
        p = Planner(runtime=make_runtime())
        plan = p.route(
            "write a haiku about the sea and save it as haiku.md", context={}
        )
        assert "file_op" not in plan.tools_to_call
        assert plan.file_op_deferred is True
        assert plan.file_op_path   == "haiku.md"
        assert plan.file_op_action == "write"
        assert plan.compound is True

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
# Post-P4a-removal: former identity phrasings now route to P6
# ---------------------------------------------------------------------------
#
# Discovery run (2026-06-26): all 13 former _IDENTITY_KEYWORDS phrasings
# resolve to priority=6 (P6 direct-answer fallback) with no MemoryManager.
# P4 Path B is skipped because there is no corpus to score against; if a
# corpus were present with a sufficiently high-scoring document, some of
# these might reach P4. The assertions below lock in the P6 outcome for
# the no-corpus case, which is the clean-room unit-test baseline.

class TestFormerP4aIdentityPhrasingsRouteToPSix:

    def _check(self, phrase: str) -> None:
        p = Planner(runtime=make_runtime(infer_return="no"))
        plan = p.route(phrase, context={})
        assert plan.priority == 6, (
            f"Expected priority=6 for {phrase!r}; got priority={plan.priority}"
        )
        assert plan.fetch_rag is False, (
            f"Expected fetch_rag=False for {phrase!r}; got {plan.fetch_rag}"
        )
        assert plan.fetch_episodic is False, (
            f"Expected fetch_episodic=False for {phrase!r}; got {plan.fetch_episodic}"
        )
        assert plan.agent == "conversational_agent"

    def test_who_are_you(self):
        self._check("who are you")

    def test_what_are_you(self):
        self._check("what are you")

    def test_tell_me_about_yourself(self):
        self._check("tell me about yourself")

    def test_what_can_you_do(self):
        self._check("what can you do")

    def test_are_you_an_ai(self):
        self._check("are you an ai")

    def test_are_you_a_bot(self):
        self._check("are you a bot")

    def test_what_is_lora(self):
        self._check("what is lora")

    def test_who_is_lora(self):
        self._check("who is lora")

    def test_what_is_localist(self):
        self._check("what is localist")

    def test_are_you_made_by_google(self):
        self._check("are you made by google")

    def test_are_you_chatgpt(self):
        self._check("are you chatgpt")

    def test_are_you_gemma(self):
        self._check("are you gemma")

    def test_introduce_yourself(self):
        self._check("introduce yourself")


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

    # 3. file_op guard: "create a file" triggers inline guard → P3c returns None,
    #    P3 fires on "create a file" and returns file_op plan
    def test_file_op_guard_defers_to_p3(self, tmp_path):
        mm, _ = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, create a file with the results", context={})
        # P3c defers; P3 fires. "the results" isn't literal content present in
        # the instruction (it's the not-yet-computed graph-query output), so
        # this is a generation-required file_op — deferred, not dispatched.
        assert plan.priority == 3
        assert plan.file_op_deferred is True
        assert "file_op" not in plan.tools_to_call
        assert plan.graph_query is None

    # 4. Ordering regression: graph-query wins over a web_search-only P3 match.
    #    "recent" is a web_search keyword; under the old (wrong) ordering where
    #    P3c ran AFTER P3, P3 would win. P3c's inline guard does NOT block
    #    web_search, so with correct ordering P3c resolves the graph query first.
    def test_p3c_beats_web_search_p3(self, tmp_path):
        mm, node_ids = make_mm_with_nodes(tmp_path)
        p = Planner(runtime=make_runtime(), memory_manager=mm)
        plan = p.route("what links to lora-persona, recent activity", context={})
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


# ---------------------------------------------------------------------------
# Diagnostic Slot 1 — _semantic_search_intent tests
# ---------------------------------------------------------------------------

def _unit_vector(dim: int = 4) -> list[float]:
    """Return a simple normalised vector [1/sqrt(dim), ...] of length `dim`."""
    v = 1.0 / math.sqrt(dim)
    return [v] * dim


class TestSemanticSearchIntent:
    """Unit tests for Planner._semantic_search_intent (Diagnostic Slot 1)."""

    def test_returns_none_when_embed_fn_is_none(self):
        """No embed_fn → always None, no templates cached."""
        p = Planner(runtime=make_runtime())
        assert p._embed_fn is None
        assert p._template_embeddings == []
        result = p._semantic_search_intent("why don't you search for something")
        assert result is None

    def test_returns_none_when_negative_filter_matches_and_embed_fn_not_called(self):
        """A phrase from _SEARCH_NEGATIVE_FILTER short-circuits before calling embed_fn."""
        spy = MagicMock(return_value=_unit_vector())
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        # Reset call count after __init__ (which calls embed_fn for templates)
        spy.reset_mock()

        result = p._semantic_search_intent("did you search for that already?")
        assert result is None
        # embed_fn must NOT have been called for the query
        spy.assert_not_called()

    def test_returns_group_and_score_for_matching_vector(self):
        """
        When embed_fn always returns the same unit vector, every template gets
        that vector at __init__ time and the query also gets it — cosine
        similarity of 1.0 with every template, so the returned score ≈ 1.0.
        The returned group must be one of the known template groups.
        """
        from planner import _SEARCH_INTENT_TEMPLATES
        fixed_vec = _unit_vector(8)
        embed_fn = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)

        embed_fn.reset_mock()
        result = p._semantic_search_intent("why don't you do a web search for APC")
        assert result is not None
        best_group, best_score, all_scores = result
        assert best_group in _SEARCH_INTENT_TEMPLATES
        assert abs(best_score - 1.0) < 1e-6  # cosine(v, v) == 1.0

    def test_returns_none_gracefully_when_embed_fn_raises(self):
        """embed_fn raising RuntimeError must not propagate — returns None instead."""
        embed_fn = MagicMock(side_effect=RuntimeError("model unavailable"))
        p = Planner(runtime=make_runtime(), embed_fn=embed_fn)
        # __init__ will fail to embed templates (error swallowed), so
        # _template_embeddings will be empty → returns None immediately.
        # OR embed_fn raises during the query call if templates somehow cached.
        # Either way, the method must return None, never raise.
        result = p._semantic_search_intent("look it up online")
        assert result is None

    def test_returns_none_gracefully_when_embed_fn_raises_only_on_query(self):
        """
        embed_fn succeeds for template embedding at __init__ but raises when
        called for the query string — _semantic_search_intent must return None.
        """
        call_count = {"n": 0}
        template_total = sum(
            len(v) for v in __import__("planner")._SEARCH_INTENT_TEMPLATES.values()
        )

        def embed_fn_that_fails_later(text: str) -> list[float]:
            call_count["n"] += 1
            if call_count["n"] > template_total:
                raise RuntimeError("model unavailable on query call")
            return _unit_vector(8)

        p = Planner(runtime=make_runtime(), embed_fn=embed_fn_that_fails_later)
        assert len(p._template_embeddings) == template_total  # templates loaded OK

        result = p._semantic_search_intent("look it up")
        assert result is None


class TestSemanticSearchIntentDiag2:
    """Diagnostic Slot 2 additions — per-group score dict shape and correctness."""

    def test_all_group_scores_contains_exactly_four_expected_keys(self):
        """all_group_scores must contain exactly the four template-group keys."""
        from planner import _SEARCH_INTENT_TEMPLATES
        expected_keys = set(_SEARCH_INTENT_TEMPLATES.keys())
        assert expected_keys == {
            "explicit_search_action",
            "lookup_request",
            "knowledge_request_open",
            "freshness_request",
        }

        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))
        result = p._semantic_search_intent("find out about this topic online")
        assert result is not None
        _, _, all_scores = result
        assert set(all_scores.keys()) == expected_keys

    def test_per_group_max_scores_are_independent(self):
        """
        When embed_fn returns different vectors for different template strings,
        the per-group score in all_group_scores reflects each group's own best
        cosine similarity, not the global best.

        Strategy: engineer two orthogonal basis vectors v1 and v2.
        At __init__ time, embed_fn always returns v1, so all template
        embeddings are v1. Then for the query, we switch the spy to return v2.
        cosine(v2, v1) ≈ 0 (orthogonal) for all templates.

        Then we patch _template_embeddings directly to give two groups
        different representative vectors and confirm per-group reporting.
        """
        from planner import _SEARCH_INTENT_TEMPLATES

        # Build a planner with any embed_fn to get past __init__ validation.
        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))

        # Manually install two distinct group vectors:
        # group A ("explicit_search_action") gets [1, 0, 0, 0]
        # group B ("lookup_request")         gets [0, 1, 0, 0]
        # all others get [0, 0, 0, 1] (low similarity to both query vectors)
        def normalise(v: list[float]) -> list[float]:
            import math
            n = math.sqrt(sum(x * x for x in v))
            return [x / n for x in v]

        vec_a = normalise([1.0, 0.0, 0.0, 0.0])
        vec_b = normalise([0.0, 1.0, 0.0, 0.0])
        vec_other = normalise([0.0, 0.0, 0.0, 1.0])

        group_to_vec = {
            "explicit_search_action": vec_a,
            "lookup_request":         vec_b,
            "knowledge_request_open": vec_other,
            "freshness_request":      vec_other,
        }
        p._template_embeddings = [
            (g, group_to_vec[g]) for g in _SEARCH_INTENT_TEMPLATES
        ]

        # Query vector halfway between A and B: [1, 1, 0, 0] normalised.
        query_vec = normalise([1.0, 1.0, 0.0, 0.0])
        p._embed_fn = MagicMock(return_value=query_vec)

        result = p._semantic_search_intent("some query")
        assert result is not None
        best_group, best_score, all_scores = result

        # cos([1,1,0,0]/√2, [1,0,0,0]) = 1/√2 ≈ 0.707 for both A and B
        # cos([1,1,0,0]/√2, [0,0,0,1]) = 0 for others
        import math
        expected_ab = 1.0 / math.sqrt(2)
        assert abs(all_scores["explicit_search_action"] - expected_ab) < 1e-5
        assert abs(all_scores["lookup_request"] - expected_ab) < 1e-5
        assert abs(all_scores["knowledge_request_open"]) < 1e-5
        assert abs(all_scores["freshness_request"]) < 1e-5

        # best_group is either A or B (tied — max() picks the first alphabetically)
        assert best_group in ("explicit_search_action", "lookup_request")
        assert abs(best_score - expected_ab) < 1e-5


def _planner_with_mocked_semantic(all_scores: dict[str, float]) -> "Planner":
    """
    Return a Planner whose _semantic_search_intent is patched to return
    all_scores directly, bypassing embed_fn entirely.  Used to test the
    gate logic in _priority3_tool with precise, normalization-free control.
    """
    p = Planner(runtime=make_runtime())
    best_group = max(all_scores, key=lambda g: all_scores[g])
    best_score = all_scores[best_group]
    p._semantic_search_intent = MagicMock(
        return_value=(best_group, best_score, all_scores)
    )
    return p


class TestPriority3SemanticGating:
    """
    Slot [Fix 1]: verifies the semantic gate correctly controls tools_to_call.

    Replaces TestPriority3ToolUnaffectedBySemantic (from Diagnostics 1/2).
    That class contained three methods:
      - test_no_literal_keyword_returns_none_despite_semantic_match
        REMOVED: asserted semantic never adds web_search — false after this slot.
      - test_route_unchanged_by_semantic_for_p6_instruction
        REMOVED: asserted that a fixed-vector instruction falls to P6 — false
        after this slot because a fixed unit vector scores 1.0 on all groups
        including explicit_search_action ≥ 0.68, so it now routes via P3.
      - test_existing_web_search_keyword_still_fires_with_embed_fn
        PRESERVED below as test_literal_keyword_still_fires_with_embed_fn:
        the invariant (literal keyword → web_search) is unchanged by this slot.

    Seven new tests cover the gate boundary, the protected negative group, and
    the deduplication guard.
    """

    def test_literal_keyword_still_fires_with_embed_fn(self):
        """Literal _WEB_SEARCH_KEYWORDS match still produces web_search."""
        fixed_vec = _unit_vector(8)
        p = Planner(runtime=make_runtime(), embed_fn=MagicMock(return_value=fixed_vec))
        plan = p.route("what is the latest news on APC?", context={})
        assert "web_search" in plan.tools_to_call
        assert plan.compound is True

    def test_explicit_search_action_at_threshold_fires_web_search(self):
        """explicit_search_action ≥ 0.68 alone → gate fires → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.90,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you check the internet for that")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_lookup_request_at_threshold_fires_web_search(self):
        """lookup_request ≥ 0.65 alone → gate fires → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.90,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("go ahead and look it up")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_knowledge_group_alone_does_not_fire_gate(self):
        """
        knowledge_request_open at 0.95 with both gating groups below threshold
        must NOT add web_search — this is the exact failure mode the
        group-specific gate design prevents.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.95,
            "freshness_request":      0.20,
        })
        result = p._priority3_tool("explain this code to me")
        assert result is None

    def test_score_just_below_lookup_threshold_does_not_fire(self):
        """
        lookup_request at 0.59 (below the 0.60 threshold; analogous to
        NEG-03 from Diagnostic 2 which scored 0.601) must not trigger the gate.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.59,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("find me a good name for this variable")
        assert result is None

    def test_score_just_below_explicit_threshold_does_not_fire(self):
        """explicit_search_action at 0.67 (below the current 0.72 threshold) must not trigger the gate."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.67,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you just check something for me")
        assert result is None

    def test_no_duplicate_web_search_when_literal_and_semantic_both_match(self):
        """
        An instruction with a literal _WEB_SEARCH_KEYWORDS match AND a high
        semantic score must produce exactly one 'web_search' in tools_to_call.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.90,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        # "latest" is a literal keyword; semantic gate also fires; no duplicate.
        result = p._priority3_tool("what's the latest on this topic, look it up")
        assert result is not None
        assert result.tools_to_call.count("web_search") == 1

    def test_route_falls_to_p6_when_all_semantic_scores_below_threshold(self):
        """
        An instruction with all semantic scores below both gating thresholds
        and no literal P3 keywords must reach P6 (no tools, no retrieval).
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.30,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.40,
            "freshness_request":      0.20,
        })
        plan = p.route("what is 2 + 2?", context={})
        assert plan.tools_to_call == []
        assert plan.fetch_rag      is False
        assert plan.fetch_episodic is False
        assert plan.agent          == "conversational_agent"

    # ------------------------------------------------------------------
    # 2026-06-25: template-coverage fix for question-form lookup_request
    # (§8.8 Open Item 11)
    # ------------------------------------------------------------------

    def test_set1_lookup_request_templates_present(self):
        """
        All four Candidate Set 1 (object-specificity fix) templates are in lookup_request.
        The four 2026-06-25 question-form templates were removed 2026-06-28 — this test
        was renamed and updated from test_new_lookup_request_templates_present (pass→fail
        change: old test asserted the removed templates; updated to assert the new Set 1
        templates that replaced them).
        """
        from planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for expected in (
            "can you look up the release date for this",
            "could you look up what year this happened",
            "can you look up information about the latest Apple products",
            "could you find out the current stock price for me",
        ):
            assert expected in templates, f"Missing Set 1 template: {expected!r}"

    def test_old_2026_06_25_lookup_request_templates_removed(self):
        """The 4 collision-prone templates added 2026-06-25 were removed 2026-06-28."""
        from planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for removed in (
            "can you look up",
            "can you look that up for me",
            "could you look up",
            "can you look into this for me",
        ):
            assert removed not in templates, f"Removed template still present: {removed!r}"

    def test_original_lookup_request_templates_unchanged(self):
        """Regression guard: original five lookup_request templates are present and unmodified."""
        from planner import _SEARCH_INTENT_TEMPLATES
        templates = _SEARCH_INTENT_TEMPLATES["lookup_request"]
        for expected in (
            "look up this",
            "look that up",
            "go ahead and look it up",
            "find information on this",
            "find out about this",
        ):
            assert expected in templates, f"Original template missing or edited: {expected!r}"

    def test_semantic_gate_thresholds_current_values(self):
        """Regression lock: _SEMANTIC_GATE_THRESHOLDS must match current calibrated values.
        lookup_request lowered 0.65 → 0.60 on 2026-06-25 (§10.4 Open Item 3 revisit).
        explicit_search_action raised 0.68 → 0.72 on 2026-06-28 per
        explicit_search_action_margin_assessment_2026-06-28.md (pass→fail change:
        old assertion was 0.68; updated to 0.72 after the threshold raise)."""
        from planner import _SEMANTIC_GATE_THRESHOLDS
        assert _SEMANTIC_GATE_THRESHOLDS == {
            "explicit_search_action": 0.72,
            "lookup_request": 0.60,
        }

    def test_lookup_request_score_at_new_threshold_fires_gate(self):
        """lookup_request at 0.605 (≥ 0.60) must fire the gate → web_search added."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.605,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up something for me")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_lookup_request_score_just_below_new_threshold_does_not_fire(self):
        """lookup_request at 0.595 (< 0.60) must NOT fire the gate."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.595,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up something for me")
        assert result is None

    def test_other_template_groups_unmodified(self):
        """Regression guard: explicit_search_action, knowledge_request_open, freshness_request are unchanged."""
        from planner import _SEARCH_INTENT_TEMPLATES
        assert _SEARCH_INTENT_TEMPLATES["explicit_search_action"] == (
            "search the web for this",
            "do a web search for this",
            "search online for this",
            "google this",
            "go look it up",
        )
        assert _SEARCH_INTENT_TEMPLATES["knowledge_request_open"] == (
            "what is this",
            "what do you know about this",
            "tell me about this",
            "explain this to me",
        )
        assert _SEARCH_INTENT_TEMPLATES["freshness_request"] == (
            "what's the latest on this",
            "what's the current status of this",
            "is there anything new about this",
        )


class TestWebSearchKeywordLiteralFix2:
    """
    Slot Fix 2: 'web search' and 'do a search' added to _WEB_SEARCH_KEYWORDS.

    All tests use embed_fn=None so only the literal keyword path is reachable —
    proving the fix does not depend on the embedding layer.
    """

    def test_web_search_phrase_triggers_literal_path(self):
        """'web search' in instruction fires web_search via literal keyword match."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route(
            "Why don't you do a web search for APC and then tell me if you "
            "still stand by your previous answer.",
            context={},
        )
        assert "web_search" in plan.tools_to_call

    def test_do_a_search_phrase_triggers_literal_path(self):
        """'do a search' in instruction fires web_search via literal keyword match."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route("Can you do a search for recent AI papers?", context={})
        assert "web_search" in plan.tools_to_call

    def test_unrelated_sentence_does_not_trigger(self):
        """Sentence containing neither new phrase does not pick up a false positive."""
        p = Planner(runtime=make_runtime())  # embed_fn=None — literal path only
        plan = p.route("I searched online for new shoes yesterday.", context={})
        assert "web_search" not in plan.tools_to_call


class TestEmbedFnWiringSmoke:
    """Smoke-level test: Planner receives a non-None embed_fn when supplied."""

    def test_planner_stores_embed_fn_from_constructor(self):
        """Constructor argument embed_fn is stored as _embed_fn."""
        fn = MagicMock(return_value=_unit_vector())
        p = Planner(runtime=make_runtime(), embed_fn=fn)
        assert p._embed_fn is fn

    def test_controller_agent_threads_embed_fn_to_planner(self):
        """
        ControllerAgent accepts embed_fn and passes it through to the Planner.
        Verified by inspecting _planner._embed_fn on the constructed controller.
        """
        fn = MagicMock(return_value=_unit_vector())
        rt = make_runtime()
        agent = make_agent("conversational_agent")
        ctrl = ControllerAgent(runtime=rt, agents=[agent], embed_fn=fn)
        assert ctrl._planner._embed_fn is fn

    def test_controller_agent_embed_fn_defaults_to_none(self):
        """ControllerAgent with no embed_fn leaves Planner._embed_fn as None."""
        rt = make_runtime()
        agent = make_agent("conversational_agent")
        ctrl = ControllerAgent(runtime=rt, agents=[agent])
        assert ctrl._planner._embed_fn is None


# ---------------------------------------------------------------------------
# 2026-06-26: identity/capability negative-filter entries
# (confirmed false positives via diagnostics/score_lookup_request_templates.py)
# ---------------------------------------------------------------------------

class TestIdentityCapabilityNegativeFilter:
    """
    Five identity/capability phrases added to _SEARCH_NEGATIVE_FILTER on
    2026-06-26 after live diagnostic confirmed they cross the lookup_request
    0.60 gate via syntactic similarity with the four 2026-06-25 question-form
    templates ("can you look up", etc.).

    Tests 1–5: _semantic_search_intent returns None for each exact phrase.
    Test 6:    _priority3_tool returns None for "Who are you?" end-to-end.
    Test 7:    Non-regression — original 2026-06-25 incident utterances still
               fire the gate (negative filter does not intercept them).
    """

    def _make_planner_with_embed(self) -> "Planner":
        """Return a Planner with a stub embed_fn so _semantic_search_intent is reachable."""
        fixed_vec = _unit_vector(8)
        spy = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        spy.reset_mock()
        return p

    def test_who_are_you_filtered(self):
        """'who are you' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("who are you") is None

    def test_what_are_you_filtered(self):
        """'what are you' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what are you") is None

    def test_what_can_you_do_filtered(self):
        """'what can you do' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what can you do") is None

    def test_what_can_you_help_with_filtered(self):
        """'what can you help with' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what can you help with") is None

    def test_what_do_you_do_filtered(self):
        """'what do you do' is caught by negative filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what do you do") is None

    def test_priority3_tool_returns_none_for_who_are_you(self):
        """
        End-to-end: _priority3_tool("who are you?") returns None (no tools scheduled).
        Verifies the observed false-positive behavior is now blocked at the
        _priority3_tool level, not just the helper.
        """
        p = self._make_planner_with_embed()
        result = p._priority3_tool("who are you?")
        assert result is None

    def test_original_lookup_incident_utterances_still_fire_gate(self):
        """
        Non-regression: the 2026-06-25 incident's utterance family
        ("Can you look up ...") is NOT intercepted by the new negative-filter
        entries and still fires the semantic gate when scores are above threshold.

        Uses _planner_with_mocked_semantic to inject a score of 0.62 on
        lookup_request (above the 0.60 gate), exactly as measured for those
        live utterances post-update-B. Confirms gate_fired=True is preserved.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.62,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up apple's price hike for the macbook neo and ipad?"
        )
        assert result is not None, "Gate should fire for a real lookup-request utterance"
        assert "web_search" in result.tools_to_call


# ---------------------------------------------------------------------------
# 2026-06-27: greeting false-positive filter — §10.4
# ---------------------------------------------------------------------------

class TestGreetingFalsePositiveFilter:
    """
    Tests for the 4-phrase greeting block added to _SEARCH_NEGATIVE_FILTER
    on 2026-06-27 after the live diagnostic in
    diagnostics/score_greeting_collisions.py confirmed these utterances cross
    the lookup_request 0.60 gate on the real embedding model.

    What these tests DO verify:
        Substring-match behavior: the filter's `phrase in lowered` check fires
        correctly for each new entry and the surrounding confirmed-live
        utterance forms (trailing punctuation, mixed case, etc.). This is
        independent of any embedding model — the filter fires before embed_fn
        is ever called.

    What these tests CANNOT verify with a stub embed_fn:
        Whether these utterances would actually clear the 0.60 lookup_request
        gate on the REAL mlx-community/embeddinggemma-300m-4bit model. That
        was confirmed by the live diagnostic (run separately from this suite),
        not here. A stub embed_fn returning a synthetic vector has no
        relationship to the real model's embedding geometry.
    """

    def _make_planner_with_embed(self) -> "Planner":
        """Planner with a stub embed_fn; spy reset after __init__ template pre-embedding."""
        fixed_vec = _unit_vector(8)
        spy = MagicMock(return_value=fixed_vec)
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        spy.reset_mock()
        return p

    # ------------------------------------------------------------------
    # Group 1 — filter membership
    # ------------------------------------------------------------------

    def test_hey_lora_in_negative_filter(self):
        """'hey lora' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "hey lora" in _SEARCH_NEGATIVE_FILTER

    def test_hi_there_in_negative_filter(self):
        """'hi there' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "hi there" in _SEARCH_NEGATIVE_FILTER

    def test_hey_there_in_negative_filter(self):
        """'hey there' is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "hey there" in _SEARCH_NEGATIVE_FILTER

    def test_whats_up_in_negative_filter(self):
        """"what's up" is a member of _SEARCH_NEGATIVE_FILTER (2026-06-27 addition)."""
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "what's up" in _SEARCH_NEGATIVE_FILTER

    # ------------------------------------------------------------------
    # Group 2 — behavioral: filter fires → _semantic_search_intent returns None
    # ------------------------------------------------------------------

    def test_hey_lora_exclamation_filtered(self):
        """
        'Hey LORA!' — the confirmed live false positive (2026-06-27) — is
        intercepted by the 'hey lora' substring match before embedding runs.

        Inlines the spy (rather than using _make_planner_with_embed) so that
        spy.assert_not_called() can verify the filter truly short-circuits
        before the embed call, not just that the method returns None.

        Note: this stub cannot verify the real model would have scored this
        ≥ 0.60 — that was confirmed by the live diagnostic only.
        """
        spy = MagicMock(return_value=_unit_vector(8))
        p = Planner(runtime=make_runtime(), embed_fn=spy)
        spy.reset_mock()

        result = p._semantic_search_intent("hey lora!")
        assert result is None
        spy.assert_not_called()

    def test_hi_there_filtered(self):
        """'hi there' is caught by the filter → _semantic_search_intent returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("hi there") is None

    def test_whats_up_with_question_mark_filtered(self):
        """"what's up?" (trailing ?) matches 'what's up' substring → returns None."""
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("what's up?") is None

    def test_hey_lora_trailing_question_mark_filtered(self):
        """
        'hey lora?' — trailing punctuation — is caught by 'hey lora' substring.
        Confirms the substring check is not accidentally anchored or
        punctuation-sensitive at the tail of the string.
        """
        p = self._make_planner_with_embed()
        assert p._semantic_search_intent("hey lora?") is None

    # ------------------------------------------------------------------
    # Group 3 — non-regression: genuine lookup utterance not swallowed
    # ------------------------------------------------------------------

    def test_genuine_lookup_not_caught_by_greeting_filter(self):
        """
        'can you look up this' — a genuine lookup-request utterance — is NOT
        caught by any of the 4 new greeting-filter entries and still fires
        the semantic gate → web_search is in tools_to_call.

        Uses _planner_with_mocked_semantic (0.62 on lookup_request, above the
        0.60 threshold) matching the existing pattern in
        TestIdentityCapabilityNegativeFilter.test_original_lookup_incident_utterances_still_fire_gate.

        Guards against a future edit accidentally widening one of the 4 new
        phrases so it swallows real search intent.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.10,
            "lookup_request":         0.62,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you look up this")
        assert result is not None, (
            "Genuine lookup utterance must not be swallowed by the greeting filter"
        )
        assert "web_search" in result.tools_to_call

    # ------------------------------------------------------------------
    # Group 4 — documented gap: bare "hi" / "hey" deliberately NOT filtered
    # ------------------------------------------------------------------

    def test_bare_hi_not_filtered_documented_gap(self):
        """
        'hi' is deliberately NOT in _SEARCH_NEGATIVE_FILTER and is not
        intercepted by the filter.

        This is a DOCUMENTED GAP, not an omission. Bare 'hi' collides with
        common substrings in legitimate queries under this filter's
        substring-match mechanism ('history', 'this', 'high', 'vehicle',
        etc.) — adding it would silently suppress the semantic gate on
        unrelated queries.

        See the 2026-06-27 comment block in planner.py (_SEARCH_NEGATIVE_FILTER)
        and LOCALIST-Architecture.md §10.4 open items for the pending fix path
        (word-boundary-matched filter or a different mechanism).

        If you are reading this because you want to add 'hi' to the filter:
        resolve the collision risk in the §10.4 open item first.
        """
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "hi" not in _SEARCH_NEGATIVE_FILTER

        # Not intercepted → reaches the embedding path.
        # With a stub embed_fn the result is non-None (synthetic scores returned).
        p = self._make_planner_with_embed()
        result = p._semantic_search_intent("hi")
        assert result is not None, (
            "'hi' must reach the embedding path; the filter must not intercept it"
        )

    def test_bare_hey_not_filtered_documented_gap(self):
        """
        'hey' is deliberately NOT in _SEARCH_NEGATIVE_FILTER and is not
        intercepted by the filter.

        This is a DOCUMENTED GAP, not an omission. Bare 'hey' collides with
        'they' (and potentially other substrings) under this filter's
        substring-match mechanism — adding it would suppress the gate for
        utterances like 'what did they say?', 'can they look it up?', etc.

        See the 2026-06-27 comment block in planner.py (_SEARCH_NEGATIVE_FILTER)
        and LOCALIST-Architecture.md §10.4 open items for the pending fix path.

        If you are reading this because you want to add 'hey' to the filter:
        resolve the collision risk in the §10.4 open item first.
        """
        from planner import _SEARCH_NEGATIVE_FILTER
        assert "hey" not in _SEARCH_NEGATIVE_FILTER

        p = self._make_planner_with_embed()
        result = p._semantic_search_intent("hey")
        assert result is not None, (
            "'hey' must reach the embedding path; the filter must not intercept it"
        )


# ---------------------------------------------------------------------------
# 2026-06-28: Candidate Set 1 template fix + explicit_search_action 0.68 → 0.72
# ---------------------------------------------------------------------------

class TestSet1TemplateFix20260628:
    """
    Validates the two 2026-06-28 planner.py changes:
      1. lookup_request: 4 collision-prone templates replaced with Candidate Set 1
         (object-specificity fix). Source: lookup_request_template_rework_2026-06-28.md,
         full_pertable_lr_set1_esa_2026-06-28.md.
      2. explicit_search_action threshold: 0.68 → 0.72. Source:
         explicit_search_action_margin_assessment_2026-06-28.md.

    Negative-filter protection for the 5 identity and 4 greeting phrases is
    unchanged — the filter fires pre-gate independent of template content.
    Verified by existing TestIdentityCapabilityNegativeFilter and
    TestGreetingFalsePositiveFilter; no new filter tests added here.

    All LR and ESA scores injected via _planner_with_mocked_semantic are
    the actual measured values from full_pertable_lr_set1_esa_2026-06-28.md
    and explicit_search_action_margin_assessment_2026-06-28.md.
    """

    # ------------------------------------------------------------------
    # (a) Cat C — 2026-06-25 incident utterances still fire via LR at Set 1
    # ------------------------------------------------------------------

    def test_cat_c1_still_fires_via_lr_set1(self):
        """
        "Can you look up Apple's price hike for the MacBook Neo and iPad?"
        LR(Set1)=0.7653, ESA=0.5424. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5424,
            "lookup_request":         0.7653,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up apple's price hike for the macbook neo and ipad?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_cat_c2_still_fires_via_lr_set1(self):
        """
        "Can you look up their next-generation in-house Microsoft AI models?"
        LR(Set1)=0.6522, ESA=0.5785. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5785,
            "lookup_request":         0.6522,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up their next-generation in-house microsoft ai models?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_cat_c3_still_fires_via_lr_set1(self):
        """
        "Can you look up Microsoft's next-generation in-house AI models?"
        LR(Set1)=0.6409, ESA=0.5735. LR ≥ 0.60 → gate fires.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5735,
            "lookup_request":         0.6409,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool(
            "can you look up microsoft's next-generation in-house ai models?"
        )
        assert result is not None
        assert "web_search" in result.tools_to_call

    # ------------------------------------------------------------------
    # (b) Cat D — 2 utterances Set 1 fixes (LR now < 0.60 AND ESA < 0.72)
    # ------------------------------------------------------------------

    def test_can_you_help_does_not_fire_with_set1(self):
        """
        "Can you help?" — LR(Set1)=0.5901, ESA=0.5810. Both below thresholds.
        Nearest miss in the fixed-8 group (0.0099 below LR threshold).
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.5810,
            "lookup_request":         0.5901,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you help?")
        assert result is None

    def test_trip_to_japan_does_not_fire_with_set1(self):
        """
        "Would you help me plan a trip to Japan?" — LR(Set1)=0.4869, ESA=0.4735.
        Both well below thresholds — generic-domain domain item.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.4735,
            "lookup_request":         0.4869,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("would you help me plan a trip to japan?")
        assert result is None

    # ------------------------------------------------------------------
    # (c) ESA positive path still fires at the new 0.72 threshold
    # ------------------------------------------------------------------

    def test_esa_positive_still_fires_at_new_threshold(self):
        """
        A genuine explicit_search_action utterance with ESA score well above 0.72
        still triggers the gate at the raised threshold.
        LR injected below 0.60 to isolate the ESA path.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.85,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("can you search the web for the latest apple news")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_esa_score_just_above_new_threshold_fires(self):
        """explicit_search_action at 0.73 (just above the new 0.72 threshold) must fire."""
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.73,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("go search for that")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_esa_score_at_old_threshold_no_longer_fires(self):
        """
        explicit_search_action at 0.69 (above the old 0.68 but below the new 0.72)
        must NOT fire the gate after the threshold raise.
        The two ESA-floor Cat D items ("Would you look at this?" ESA=0.6990,
        "Will you look into this?" ESA=0.6874) fall in this range and would
        have fired before the 2026-06-28 change.
        """
        p = _planner_with_mocked_semantic({
            "explicit_search_action": 0.69,
            "lookup_request":         0.10,
            "knowledge_request_open": 0.10,
            "freshness_request":      0.10,
        })
        result = p._priority3_tool("would you look at this")
        assert result is None
