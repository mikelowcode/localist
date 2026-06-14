"""
LORA — PromptBuilder
====================
Single point of prompt assembly for all LORA agents.
Implements the 7-slot prompt contract defined in §3 of
LORA-Architecture.md. Every agent calls PromptBuilder.build();
no agent assembles its own prompt string.

This module has no dependencies on FastAPI, SQLite, or any runtime
client. It is pure Python and safe to import anywhere in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role:    str         # "user" | "assistant" | "tool"
    content: str
    label:   str | None = None   # tool name, if role == "tool"


@dataclass
class EpisodeBullet:
    content:      str
    episode_type: str
    confidence:   float


@dataclass
class RagSource:
    path:    str
    content: str


@dataclass
class ToolResult:
    tool_name:  str
    parameters: str
    result:     str


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the canonical 7-slot prompt defined in §3 of
    LORA-Architecture.md.

    Slot layout
    -----------
    SYSTEM MESSAGE
      Slot 1a [identity]        — _SYSTEM constant; always present; ~50 tokens
      Slot 1b [persona]         — wiki persona doc; injected when provided;
                                  500-token ceiling; appended to system msg

    USER MESSAGE (static-first ordering for KV-cache prefix reuse)
      Slot 3  [EPISODIC MEMORY] — durable facts; conditional; 150-token ceiling
      Slot 4  [CONTEXT]         — RAG snippets; conditional; 450-token ceiling
      Slot 5  [TOOL RESULTS]    — tool output; conditional; 500-token ceiling
      Slot 6  [WORKING MEMORY]  — recent turns; conditional; 300-token ceiling
      Slot 7  [INSTRUCTION]     — raw user instruction; always present; uncapped

    Slots are ordered most-stable-first so that the KV-cache prefix
    is maximally reused across consecutive turns. [INSTRUCTION] is always
    last because it changes on every turn.

    Token estimation: 1 token ≈ 4 characters (len(text) // 4).
    This is consistent with the convention used elsewhere in memory_manager.py.

    Returns
    -------
    build() → (system_prompt: str, user_prompt: str)
        system_prompt : Slot 1a + optional 1b. Pass as the `system=` argument
                        to runtime.
        user_prompt   : Slots 3–7 assembled in order, empty slots cleanly
                        omitted (no label, no whitespace placeholder).

    Invariants
    ----------
    - Stateless. Safe to call from multiple threads concurrently.
    - Callers pass full content; PromptBuilder enforces all token ceilings.
    - Empty optional slots produce no output whatsoever — not even a newline.
    - Slot 1a is a constant. It is never overridden by callers.
    """

    # -----------------------------------------------------------------------
    # Slot 1a — canonical system prompt (§3.2, Slot 1)
    # Ceiling: 50 tokens = 200 chars. This value is 174 chars / ~43 tokens.
    # -----------------------------------------------------------------------
    _SYSTEM: str = (
        "You are LORA, a local research assistant. "
        "You reason carefully, cite your sources, and acknowledge when you "
        "don't know something. You do not simulate certainty."
    )

    # -----------------------------------------------------------------------
    # Token ceilings (in tokens; multiply by 4 for char equivalent)
    # -----------------------------------------------------------------------
    _CEIL_SYSTEM:   int = 50    # hard; slot 1a is a constant so this is advisory
    _CEIL_PERSONA:  int = 500   # slot 1b; persona injected into system msg
    _CEIL_EPISODIC: int = 150   # slot 3
    _CEIL_RAG:      int = 450   # slot 4
    _CEIL_TOOL:     int = 500   # slot 5
    _CEIL_WORKING:  int = 300   # slot 6

    # -----------------------------------------------------------------------
    # Internal token helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count as len(text) // 4."""
        return len(text) // 4

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Hard-truncate text to max_tokens (estimated).
        Truncation appends '… [truncated]' and never cuts mid-word.
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0]
        return cut + "… [truncated]"

    # -----------------------------------------------------------------------
    # Slot builders (private)
    # -----------------------------------------------------------------------

    def _slot1_system(self, persona: str | None = None) -> str:
        """
        Return Slot 1: system prompt, optionally with persona appended.

        When persona is None or empty, returns the identity constant only.
        Otherwise appends the truncated persona (500-token ceiling) raw,
        separated by a double newline — no label is added.
        """
        if not persona:
            return self._SYSTEM
        truncated = self._truncate_to_tokens(persona, self._CEIL_PERSONA)
        return self._SYSTEM + "\n\n" + truncated

    def _slot3_episodic(
        self,
        bullets: list[EpisodeBullet] | None,
    ) -> str:
        """
        Return Slot 3: episodic memory block, or "" if None/empty.

        Format:
            [EPISODIC MEMORY]
            - {content} ({episode_type}, {confidence:.1f})

        The 150-token ceiling is enforced by dropping lowest-priority bullets
        from the tail (callers should pre-sort by priority; this method
        enforces the budget by truncating from the end).
        """
        if not bullets:
            return ""

        lines = ["[EPISODIC MEMORY]"]
        max_chars = self._CEIL_EPISODIC * 4

        for bullet in bullets:
            line = f"- {bullet.content} ({bullet.episode_type}, {bullet.confidence:.1f})"
            candidate = "\n".join(lines + [line])
            if self._estimate_tokens(candidate) > self._CEIL_EPISODIC:
                break    # budget exhausted; remaining bullets are dropped
            lines.append(line)

        if len(lines) == 1:
            # Only the label survived — no bullets fit; emit nothing
            return ""

        return "\n".join(lines)

    def _slot4_rag(
        self,
        sources: list[RagSource] | None,
    ) -> str:
        """
        Return Slot 4: RAG context block, or "" if None/empty.

        Format:
            [CONTEXT]
            Source: {path}
            {content snippet}

            Source: {path}
            {content snippet}

        The 450-token ceiling is enforced across all sources combined.
        Each source's content is truncated at a sentence boundary when
        possible. Maximum 3 sources (callers should pre-rank; this method
        takes the first 3).
        """
        if not sources:
            return ""

        MAX_SOURCES = 3
        max_chars   = self._CEIL_RAG * 4

        lines   = ["[CONTEXT]"]
        budget  = max_chars - len("[CONTEXT]\n")

        for source in sources[:MAX_SOURCES]:
            header  = f"Source: {source.path}"
            content = source.content.strip()

            # Truncate content at sentence boundary
            entry_budget = budget - len(header) - 2  # 2 for newlines
            if entry_budget <= 0:
                break
            if len(content) > entry_budget:
                # Try to cut at last sentence boundary within budget
                truncated = content[:entry_budget]
                last_period = max(
                    truncated.rfind("."),
                    truncated.rfind("!"),
                    truncated.rfind("?"),
                )
                if last_period > entry_budget // 2:
                    content = truncated[: last_period + 1]
                else:
                    content = truncated + "…"

            block  = f"{header}\n{content}"
            lines.append(block)
            budget -= len(block) + 1   # +1 for separator newline

            if budget <= 0:
                break

        if len(lines) == 1:
            return ""

        return "\n\n".join(lines[:1] + lines[1:])

    def _slot5_tools(
        self,
        tool_results: list[ToolResult] | None,
    ) -> str:
        """
        Return Slot 5: tool results block, or "" if None/empty.

        Format:
            [TOOL RESULTS]
            {tool_name}({parameters}):
              {result}

        The 500-token ceiling is enforced. Each result is truncated to fit
        within the remaining budget before being appended.
        """
        if not tool_results:
            return ""

        max_chars = self._CEIL_TOOL * 4
        lines     = ["[TOOL RESULTS]"]
        budget    = max_chars - len("[TOOL RESULTS]\n")

        for tr in tool_results:
            header = f"{tr.tool_name}({tr.parameters}):"
            result = tr.result.strip()

            entry_budget = budget - len(header) - 4  # 4 for "\n  " + "\n"
            if entry_budget <= 0:
                break
            if len(result) > entry_budget:
                result = result[:entry_budget] + "… [truncated]"

            block  = f"{header}\n  {result}"
            lines.append(block)
            budget -= len(block) + 1

            if budget <= 0:
                break

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _slot6_working_memory(
        self,
        turns: list[Turn] | None,
    ) -> str:
        """
        Return Slot 6: working memory block, or empty string if no turns.

        Format:
            [WORKING MEMORY]
            Turn -N [role]: content
            ...
            Turn -1 [role]: content

        Turns are listed chronologically (oldest surviving first, newest last).
        The 300-token ceiling is enforced by dropping oldest turns first.
        Each turn is formatted as a single line before ceiling enforcement.

        Returns "" if turns is None or empty.
        """
        if not turns:
            return ""

        # Format all turns first
        formatted: list[str] = []
        for i, turn in enumerate(turns):
            offset = -(len(turns) - i)   # -N … -1
            if turn.role == "tool" and turn.label:
                role_str = f"tool:{turn.label}"
            else:
                role_str = turn.role
            formatted.append(f"Turn {offset} [{role_str}]: {turn.content}")

        # Enforce 300-token ceiling: drop oldest until budget is met
        max_chars = self._CEIL_WORKING * 4
        while formatted:
            total = sum(len(f) for f in formatted)
            if total <= max_chars:
                break
            formatted.pop(0)

        if not formatted:
            return ""

        body = "\n".join(formatted)
        return f"[WORKING MEMORY]\n{body}"

    def _slot7_instruction(self, instruction: str) -> str:
        """
        Return Slot 7: the raw user instruction. Always present; uncapped.

        Format:
            [INSTRUCTION]
            {instruction}
        """
        return f"[INSTRUCTION]\n{instruction}"

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        instruction:      str,
        persona:          str | None            = None,
        episodic_summary: list[EpisodeBullet]   | None = None,
        rag_snippets:     list[RagSource]        | None = None,
        tool_results:     list[ToolResult]       | None = None,
        working_memory:   list[Turn]             | None = None,
    ) -> tuple[str, str]:
        """
        Assemble the canonical 7-slot prompt.

        Parameters
        ----------
        instruction :
            The raw user instruction (Slot 7). Never transformed.
        persona :
            Optional persona string (Slot 1b). Injected into the system
            message when provided; 500-token ceiling enforced.
        episodic_summary :
            Pre-sorted EpisodeBullet list (Slot 3). Caller is responsible
            for priority ordering. Pass None or [] to omit.
        rag_snippets :
            Ranked RagSource list (Slot 4). At most 3 used. Pass None or
            [] to omit.
        tool_results :
            ToolResult list in dispatch order (Slot 5). Pass None or []
            to omit.
        working_memory :
            Recent conversation turns (Slot 6). Oldest dropped first when
            the 300-token ceiling is exceeded. Pass None or [] to omit.

        Returns
        -------
        (system_prompt, user_prompt) : tuple[str, str]
            system_prompt : Slot 1a + optional 1b. Pass as `system=` to
                            the runtime.
            user_prompt   : Slots 3–7 joined with double newlines.
                            Empty slots are cleanly absent.
        """
        system_prompt = self._slot1_system(persona)

        slots = [
            self._slot3_episodic(episodic_summary),
            self._slot4_rag(rag_snippets),
            self._slot5_tools(tool_results),
            self._slot6_working_memory(working_memory),
            self._slot7_instruction(instruction),
        ]

        user_prompt = "\n\n".join(s for s in slots if s)
        return system_prompt, user_prompt
