"""
PromptBuilder unit tests — persona injection and slot ordering.

Covers:
  PB-A — No persona: system message is identity constant only
  PB-B — Persona present: appended to system message; absent from user message
  PB-C — Persona truncation at 500-token ceiling
  PB-D — Slot ordering with all slots populated
"""

import pytest

from prompt_builder import (
    PromptBuilder,
    Turn,
    EpisodeBullet,
    UserProfileFact,
    RagSource,
    ToolResult,
    GraphLinkEntry,
    GraphQueryResult,
    WorkingMemoryState,
)


def test_pb_a_no_persona_system_is_identity_only():
    """build() with no persona returns bare _SYSTEM as system_prompt."""
    pb = PromptBuilder()
    system_prompt, user_prompt = pb.build(instruction="hello")

    assert system_prompt == PromptBuilder._SYSTEM
    assert "[INSTRUCTION]" in user_prompt
    assert "hello" in user_prompt


def test_pb_b_persona_appended_to_system_not_in_user():
    """Persona is appended to system_prompt and does not appear in user_prompt."""
    pb = PromptBuilder()
    persona = "I am LORA, your assistant."
    system_prompt, user_prompt = pb.build(instruction="hello", persona=persona)

    assert system_prompt == PromptBuilder._SYSTEM + "\n\n" + persona
    assert "[INSTRUCTION]" in user_prompt
    assert "I am LORA" not in user_prompt


def test_pb_c_persona_truncated_at_ceiling():
    """Persona longer than 500 tokens (2000 chars) is hard-truncated."""
    pb = PromptBuilder()
    long_persona = "word " * 600   # 3000 chars >> 2000-char (500-token) ceiling
    system_prompt, _ = pb.build(instruction="x", persona=long_persona)

    assert len(system_prompt) < len(PromptBuilder._SYSTEM) + len(long_persona)
    assert "… [truncated]" in system_prompt


def test_pb_d_slot_ordering_all_slots():
    """User message slots appear in strict static-first order."""
    pb = PromptBuilder()
    _, user_prompt = pb.build(
        instruction      = "final question",
        episodic_summary = [EpisodeBullet("pref x", "preference", 0.9)],
        rag_snippets     = [RagSource("doc.md", "some context content here")],
        tool_results     = [ToolResult("search", "q=test", "search result text")],
        working_memory   = [Turn("user", "prior turn content")],
    )

    ep_pos    = user_prompt.index("[EPISODIC MEMORY]")
    ctx_pos   = user_prompt.index("[CONTEXT]")
    tools_pos = user_prompt.index("[TOOL RESULTS]")
    wm_pos    = user_prompt.index("[WORKING MEMORY]")
    inst_pos  = user_prompt.index("[INSTRUCTION]")

    assert ep_pos < ctx_pos < tools_pos < wm_pos < inst_pos, (
        f"Slot order wrong: ep={ep_pos} ctx={ctx_pos} "
        f"tools={tools_pos} wm={wm_pos} inst={inst_pos}"
    )


def test_pb_e_build_enforces_dynamic_suffix_slot_order():
    """
    Locks the dynamic-suffix slot order per LOCALIST-Architecture.md §3.7a.

    All optional parameters populated simultaneously. Asserts:
    - [EPISODIC MEMORY] < [USER PROFILE] < [CONTEXT] < [TOOL RESULTS]
      < [WORKING MEMORY] < [INSTRUCTION] in user_prompt
    - stable-prefix boundary: persona in system_prompt, absent from user_prompt
    - identity constant present in system_prompt

    This test must fail if PromptBuilder.build()'s slots list is reordered.
    Any failure here is a deliberate contract break requiring doc + review.
    """
    pb = PromptBuilder()
    persona = "Test persona content — stable prefix fixture."
    system_prompt, user_prompt = pb.build(
        instruction      = "test instruction",
        persona          = persona,
        episodic_summary = [EpisodeBullet("episodic fact A", "preference", 0.9)],
        profile_facts    = [UserProfileFact("profile fact B")],
        rag_snippets     = [RagSource("test-doc.md", "rag context content C")],
        tool_results     = [ToolResult("web_search", "q=test", "tool result D")],
        working_memory   = [Turn("user", "prior turn E")],
    )

    ep_pos      = user_prompt.index("[EPISODIC MEMORY]")
    profile_pos = user_prompt.index("[USER PROFILE]")
    ctx_pos     = user_prompt.index("[CONTEXT]")
    tools_pos   = user_prompt.index("[TOOL RESULTS]")
    wm_pos      = user_prompt.index("[WORKING MEMORY]")
    inst_pos    = user_prompt.index("[INSTRUCTION]")

    assert ep_pos < profile_pos < ctx_pos < tools_pos < wm_pos < inst_pos, (
        f"Dynamic-suffix slot order violated (§3.7a): "
        f"ep={ep_pos} profile={profile_pos} ctx={ctx_pos} "
        f"tools={tools_pos} wm={wm_pos} inst={inst_pos}"
    )

    # Stable-prefix / dynamic-suffix boundary: persona goes into system, not user.
    assert PromptBuilder._SYSTEM in system_prompt
    assert persona in system_prompt
    assert persona not in user_prompt


def test_pb_f_slot3_profile_only_precedes_context():
    """
    Locks Slot 3 profile-only sub-ordering per LOCALIST-Architecture.md §3.7a.

    When episodic_summary is absent but profile_facts are present, [USER PROFILE]
    must appear before [CONTEXT] and [EPISODIC MEMORY] must be absent entirely.
    Confirms the dynamic-suffix contract holds for the profile-only routing path
    (P4/P4a turns that fire profile injection but not episodic retrieval).
    """
    pb = PromptBuilder()
    _, user_prompt = pb.build(
        instruction   = "test instruction",
        profile_facts = [UserProfileFact("profile fact only")],
        rag_snippets  = [RagSource("test-doc.md", "rag context content")],
    )

    assert "[EPISODIC MEMORY]" not in user_prompt, (
        "Slot 3a must be cleanly absent when episodic_summary is empty (§3.7a)"
    )

    profile_pos = user_prompt.index("[USER PROFILE]")
    ctx_pos     = user_prompt.index("[CONTEXT]")

    assert profile_pos < ctx_pos, (
        f"[USER PROFILE] must precede [CONTEXT] in dynamic suffix (§3.7a): "
        f"profile={profile_pos} ctx={ctx_pos}"
    )


# ---------------------------------------------------------------------------
# TestSlotGraph — Slot 5b [GRAPH RESULT]
# ---------------------------------------------------------------------------

class TestSlotGraph:

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    # 1. graph_result=None → slot returns ""
    def test_none_returns_empty(self):
        pb = self._pb()
        assert pb._slot_graph(None) == ""

    # 2. Incoming, populated — exact format match
    def test_incoming_populated(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="localist-software-stack",
            links=[
                GraphLinkEntry("how-localist-works", True),
                GraphLinkEntry("localist-master-project-outline", True),
            ],
        )
        expected = (
            "[GRAPH RESULT]\n"
            "Pages linking to localist-software-stack:\n"
            "- how-localist-works\n"
            "- localist-master-project-outline"
        )
        assert pb._slot_graph(gr) == expected

    # 3. Incoming, zero results — exact format match
    def test_incoming_empty(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="lora-persona",
            links=[],
        )
        expected = "[GRAPH RESULT]\nNo pages link to lora-persona."
        assert pb._slot_graph(gr) == expected

    # 4. Outgoing, all resolved — no unresolved / "also references" section
    def test_outgoing_all_resolved(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[GraphLinkEntry("localist-master-project-outline", True)],
        )
        result = pb._slot_graph(gr)
        assert result == (
            "[GRAPH RESULT]\n"
            "localist-build-order links to:\n"
            "- localist-master-project-outline"
        )
        assert "also references" not in result
        assert "does not exist" not in result

    # 5. Outgoing, all unresolved — "references" WITHOUT "also"
    def test_outgoing_all_unresolved(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[GraphLinkEntry("Localist Software Stack Overview", False)],
        )
        result = pb._slot_graph(gr)
        assert result == (
            "[GRAPH RESULT]\n"
            "localist-build-order references a page that does not exist:\n"
            '- "Localist Software Stack Overview" (no matching page found)'
        )
        assert "also references" not in result

    # 6. Outgoing, mixed — exact format match including "also" and blank line
    def test_outgoing_mixed(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="localist-build-order",
            links=[
                GraphLinkEntry("localist-master-project-outline", True),
                GraphLinkEntry("Localist Software Stack Overview", False),
            ],
        )
        expected = (
            "[GRAPH RESULT]\n"
            "localist-build-order links to:\n"
            "- localist-master-project-outline\n"
            "\n"
            "localist-build-order also references a page that does not exist:\n"
            '- "Localist Software Stack Overview" (no matching page found)'
        )
        assert pb._slot_graph(gr) == expected

    # 7. Outgoing, zero results — exact format match
    def test_outgoing_empty(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="outgoing",
            page_name="lora-persona",
            links=[],
        )
        expected = "[GRAPH RESULT]\nlora-persona does not link to any other pages."
        assert pb._slot_graph(gr) == expected

    # 8. build() end-to-end: [GRAPH RESULT] positioned after [TOOL RESULTS]
    #    and before [WORKING MEMORY]
    def test_build_slot_order_with_graph(self):
        pb = self._pb()
        gr = GraphQueryResult(
            direction="incoming",
            page_name="localist-software-stack",
            links=[GraphLinkEntry("how-localist-works", True)],
        )
        _, user_prompt = pb.build(
            instruction    = "test",
            tool_results   = [ToolResult("search", "q=wiki", "some result")],
            graph_result   = gr,
            working_memory = [Turn("user", "prior message")],
        )
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        graph_pos = user_prompt.index("[GRAPH RESULT]")
        wm_pos    = user_prompt.index("[WORKING MEMORY]")
        assert tools_pos < graph_pos < wm_pos, (
            f"Slot 5b order wrong: tools={tools_pos} graph={graph_pos} wm={wm_pos}"
        )

    # 9. build() with graph_result=None and all other optionals absent →
    #    no [GRAPH RESULT] in output, no stray separators
    def test_build_none_graph_no_stray_output(self):
        pb = self._pb()
        _, user_prompt = pb.build(instruction="hello")
        assert "[GRAPH RESULT]" not in user_prompt
        assert user_prompt == "[INSTRUCTION]\nhello"


# ---------------------------------------------------------------------------
# TestSlot6AWorkingState — Slot 6A [WORKING STATE]
# ---------------------------------------------------------------------------

class TestSlot6AWorkingState:

    def _pb(self) -> PromptBuilder:
        return PromptBuilder()

    # 1. None state → clean omission; build() output byte-identical to no-arg baseline
    def test_none_state_clean_omission_regression_guard(self):
        pb = self._pb()
        baseline_sys, baseline_user = pb.build(instruction="test")
        sys_with_none, user_with_none = pb.build(instruction="test", working_state=None)
        assert sys_with_none == baseline_sys
        assert user_with_none == baseline_user
        assert "[WORKING STATE]" not in user_with_none

    # 2. Empty WorkingMemoryState → clean omission (both fields falsy)
    def test_empty_state_clean_omission(self):
        pb = self._pb()
        _, user_prompt = pb.build(
            instruction   = "test",
            working_state = WorkingMemoryState(),
        )
        assert "[WORKING STATE]" not in user_prompt

    # 3. current_project set, active_artifacts empty → only current_project line
    def test_current_project_only(self):
        pb = self._pb()
        state = WorkingMemoryState(current_project="localist-v2")
        result = pb._slot6a_working_state(state)
        assert result == "[WORKING STATE]\ncurrent_project: localist-v2"
        assert "active_artifacts" not in result

    # 4. active_artifacts set, current_project None → only active_artifacts line
    def test_active_artifacts_only(self):
        pb = self._pb()
        state = WorkingMemoryState(active_artifacts=["wiki/lora.md", "wiki/planner.md"])
        result = pb._slot6a_working_state(state)
        assert result == (
            "[WORKING STATE]\n"
            "active_artifacts: wiki/lora.md, wiki/planner.md"
        )
        assert "current_project" not in result

    # 5. Both fields set → both lines render in documented order
    def test_both_fields_render_in_order(self):
        pb = self._pb()
        state = WorkingMemoryState(
            current_project  = "localist-v2",
            active_artifacts = ["wiki/lora.md", "wiki/planner.md"],
        )
        result = pb._slot6a_working_state(state)
        assert result == (
            "[WORKING STATE]\n"
            "current_project: localist-v2\n"
            "active_artifacts: wiki/lora.md, wiki/planner.md"
        )
        assert result.index("current_project") < result.index("active_artifacts")

    # 6. Ceiling enforcement: active_artifacts truncated from the end
    def test_ceiling_truncates_artifacts_from_end(self):
        pb = self._pb()
        # Each artifact is ~20 chars; 30 artifacts × ~20 chars >> 400 chars (100-token ceiling)
        artifacts = [f"wiki/article-{i:04d}.md" for i in range(30)]
        state = WorkingMemoryState(
            current_project  = "localist-v2",
            active_artifacts = artifacts,
        )
        result = pb._slot6a_working_state(state)
        assert "[WORKING STATE]" in result
        assert "current_project: localist-v2" in result
        # Must be within ceiling
        assert pb._estimate_tokens(result) <= pb._CEIL_WORKING_STATE
        # Last artifact should be absent (truncated from end)
        assert "wiki/article-0029.md" not in result
        # First artifact should still be present (kept from front)
        assert "wiki/article-0000.md" in result

    # 7. Ordering: [WORKING STATE] after [TOOL RESULTS] and before [WORKING MEMORY]
    def test_build_slot_order_with_working_state(self):
        pb = self._pb()
        _, user_prompt = pb.build(
            instruction   = "test",
            tool_results  = [ToolResult("search", "q=test", "some result")],
            working_state = WorkingMemoryState(
                current_project  = "localist-v2",
                active_artifacts = ["wiki/lora.md"],
            ),
            working_memory = [Turn("user", "prior turn")],
        )
        tools_pos = user_prompt.index("[TOOL RESULTS]")
        ws_pos    = user_prompt.index("[WORKING STATE]")
        wm_pos    = user_prompt.index("[WORKING MEMORY]")
        assert tools_pos < ws_pos < wm_pos, (
            f"Slot 6A order wrong: tools={tools_pos} ws={ws_pos} wm={wm_pos}"
        )
