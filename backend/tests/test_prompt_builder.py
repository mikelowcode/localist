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
    RagSource,
    ToolResult,
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
