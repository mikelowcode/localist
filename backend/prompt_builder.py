"""
LORA — PromptBuilder
====================
Single point of prompt assembly for all LORA agents.
Implements the 7-slot prompt contract defined in §3 of
LOCALIST-Architecture.md. Every agent calls PromptBuilder.build();
no agent assembles its own prompt string.

This module has no dependencies on FastAPI, SQLite, or any runtime
client. It is pure Python and safe to import anywhere in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
class UserProfileFact:
    content: str   # single fact line, already scored and selected by caller


@dataclass
class RagSource:
    path:    str
    content: str


@dataclass
class SessionFile:
    filename: str   # original filename, used as the label in the slot
    content:  str   # full extracted text; truncation enforced at render-time in slot builder


@dataclass
class ToolResult:
    tool_name:  str
    parameters: str
    result:     str
    success:    bool = True


@dataclass
class GraphLinkEntry:
    name:     str    # resolved page's stem, or raw link_text if unresolved
    resolved: bool   # True if this entry points to a real page


@dataclass
class GraphQueryResult:
    direction: str                  # "incoming" | "outgoing"
    page_name: str                  # display name of the resolved page
    links:     list[GraphLinkEntry] # may be empty — zero results is valid


@dataclass
class WorkingMemoryState:
    current_project:  str | None = None
    active_artifacts: list[str]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the canonical 7-slot prompt defined in §3 of
    LOCALIST-Architecture.md.

    Slot layout
    -----------
    SYSTEM MESSAGE
      Slot 1a [identity]        — _SYSTEM constant; always present; ~50 tokens
      Slot 1b [persona]         — wiki persona doc; injected when provided;
                                  500-token ceiling; appended to system msg

    USER MESSAGE (static-first ordering for KV-cache prefix reuse)
      Slot 3  [EPISODIC MEMORY] — durable facts; conditional; 150-token ceiling
      Slot 4  [CONTEXT]         — RAG snippets; conditional; 800-token ceiling
      Slot 5  [TOOL RESULTS]    — tool output; conditional; 500-token ceiling
      Slot 5b [GRAPH RESULT]    — graph query result; emitted whenever a graph
                                  query resolved, even with zero edges;
                                  300-token ceiling
      Slot 6A [WORKING STATE]   — structured working state; conditional;
                                  100-token ceiling
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
        user_prompt   : Slots 3–7 assembled in order. Most empty slots are
                        cleanly omitted (no label, no whitespace placeholder).
                        Exception: Slot 5b ([GRAPH RESULT]) is always rendered
                        when graph_result is not None, even with zero links —
                        see _slot_graph() for the full contract.

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
    _CEIL_EPISODIC: int = 150   # slot 3a; episodic bullets
    _CEIL_PROFILE:  int = 100   # slot 3b; user profile facts
    _CEIL_RAG:      int = 800   # slot 4
    _CEIL_TOOL:     int = 500   # slot 5
    _CEIL_GRAPH:    int = 300   # slot 5b
    _CEIL_WORKING_STATE: int = 100   # slot 6a
    _CEIL_WORKING:       int = 300   # slot 6
    _CEIL_SESSION_FILES_EACH:  int = 4000   # per-file ceiling, tokens
    _CEIL_SESSION_FILES_TOTAL: int = 20000  # total slot ceiling, tokens

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

    def _slot_session_files(
        self,
        files: list[SessionFile] | None,
    ) -> str:
        """
        Return the [SESSION FILES] slot, or "" if files is None/empty.

        This slot is unnumbered and positioned before all other user-message
        slots (before Slot 3 / [EPISODIC MEMORY]) so that uploaded file
        content sits as early as possible in the KV-cache prefix. It is
        populated directly from the backend ephemeral session file cache and
        bypasses the Planner routing ladder entirely.

        Per-file ceiling: 4,000 tokens (16,000 chars).
        Total slot ceiling: 20,000 tokens.

        Each file is rendered as:
            [SESSION FILES]
            --- filename.ext ---
            {content}
            --- end filename.ext ---

        Multiple files are separated by a single blank line.
        Truncation appends the standard '… [truncated]' marker via
        _truncate_to_tokens(), consistent with all other slot builders.
        """
        if not files:
            return ""

        rendered_files: list[str] = []
        total_tokens = 0

        for f in files:
            truncated = self._truncate_to_tokens(f.content, self._CEIL_SESSION_FILES_EACH)
            block = f"--- {f.filename} ---\n{truncated}\n--- end {f.filename} ---"
            block_tokens = self._estimate_tokens(block)
            if total_tokens + block_tokens > self._CEIL_SESSION_FILES_TOTAL:
                break
            rendered_files.append(block)
            total_tokens += block_tokens

        if not rendered_files:
            return ""

        body = "\n\n".join(rendered_files)
        return f"[SESSION FILES]\n{body}"

    def _slot3_combined(
        self,
        bullets:       list[EpisodeBullet] | None,
        profile_facts: list[UserProfileFact] | None,
    ) -> str:
        """
        Return Slot 3: episodic memory block and/or user profile facts,
        or "" if both are None/empty.

        Two independent sub-budgets:
          Episodic bullets : 150-token ceiling (unchanged)
          User profile     : 100-token ceiling (new)

        Each sub-block is omitted cleanly when empty.
        Combined output is returned as a single string.

        Format:
            [EPISODIC MEMORY]
            - {content} ({episode_type}, {confidence:.1f})

            [USER PROFILE]
            - {fact line}
        """
        parts: list[str] = []

        # -- Episodic sub-block (150-token ceiling) ---------------------------
        if bullets:
            lines = ["[EPISODIC MEMORY]"]
            for bullet in bullets:
                line = (
                    f"- {bullet.content} "
                    f"({bullet.episode_type}, {bullet.confidence:.1f})"
                )
                candidate = "\n".join(lines + [line])
                if self._estimate_tokens(candidate) > self._CEIL_EPISODIC:
                    break
                lines.append(line)
            if len(lines) > 1:
                parts.append("\n".join(lines))

        # -- User profile sub-block (100-token ceiling) -----------------------
        if profile_facts:
            lines = ["[USER PROFILE]"]
            max_chars = self._CEIL_PROFILE * 4
            used_chars = len("[USER PROFILE]\n")
            for fact in profile_facts:
                line = f"- {fact.content}"
                if used_chars + len(line) + 1 > max_chars:
                    break
                lines.append(line)
                used_chars += len(line) + 1
            if len(lines) > 1:
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

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

        The 800-token ceiling is enforced across all sources combined.
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
                    content = truncated[: last_period + 1] + " … [truncated]"
                else:
                    content = truncated + "… [truncated]"

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

    def _slot_graph(self, graph_result: GraphQueryResult | None) -> str:
        """
        Return Slot 5b: graph query result, or "" if graph_result is None.

        This slot does NOT follow the clean-omission rule of other slots.
        When graph_result is not None the slot is always rendered — even when
        graph_result.links is an empty list. Zero edges is a real, correct
        answer that must be visible to the model. The only omission case is
        graph_result=None, meaning no graph query was resolved this turn.

        300-token ceiling enforced as a single post-render truncation.
        """
        if graph_result is None:
            return ""

        page  = graph_result.page_name
        links = graph_result.links

        if graph_result.direction == "incoming":
            if not links:
                content = f"No pages link to {page}."
            else:
                body    = "\n".join(f"- {e.name}" for e in links)
                content = f"Pages linking to {page}:\n{body}"
        else:  # outgoing
            if not links:
                content = f"{page} does not link to any other pages."
            else:
                resolved   = [e for e in links if e.resolved]
                unresolved = [e for e in links if not e.resolved]
                sections: list[str] = []
                if resolved:
                    res_body = "\n".join(f"- {e.name}" for e in resolved)
                    sections.append(f"{page} links to:\n{res_body}")
                if unresolved:
                    header = (
                        f"{page} also references a page that does not exist:"
                        if resolved
                        else f"{page} references a page that does not exist:"
                    )
                    unres_body = "\n".join(
                        f'- "{e.name}" (no matching page found)'
                        for e in unresolved
                    )
                    sections.append(f"{header}\n{unres_body}")
                content = "\n\n".join(sections)

        rendered = f"[GRAPH RESULT]\n{content}"
        return self._truncate_to_tokens(rendered, self._CEIL_GRAPH)

    def _slot6a_working_state(
        self,
        state: WorkingMemoryState | None,
    ) -> str:
        """
        Return Slot 6A: structured working state, or "" if state is None/empty.

        Clean-omission: returns "" when state is None or both current_project
        is falsy and active_artifacts is empty — no label, no placeholder.

        active_artifacts is truncated by dropping entries from the end until
        the block fits within the 100-token ceiling. current_project is emitted
        as-is (ceiling is soft for this single line).

        Format:
            [WORKING STATE]
            current_project: {current_project}
            active_artifacts: {artifact1}, {artifact2}, ...

        Each line is only emitted when its field is non-empty/non-None.
        """
        if state is None:
            return ""
        if not state.current_project and not state.active_artifacts:
            return ""

        lines = ["[WORKING STATE]"]

        if state.current_project:
            lines.append(f"current_project: {state.current_project}")

        if state.active_artifacts:
            artifacts = list(state.active_artifacts)
            while artifacts:
                line = f"active_artifacts: {', '.join(artifacts)}"
                candidate = "\n".join(lines + [line])
                if self._estimate_tokens(candidate) <= self._CEIL_WORKING_STATE:
                    lines.append(line)
                    break
                artifacts.pop()

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
        session_files:    list[SessionFile]        | None = None,
        persona:          str | None              = None,
        episodic_summary: list[EpisodeBullet]     | None = None,
        profile_facts:    list[UserProfileFact]   | None = None,
        rag_snippets:     list[RagSource]          | None = None,
        tool_results:     list[ToolResult]         | None = None,
        graph_result:     GraphQueryResult         | None = None,
        working_state:    WorkingMemoryState       | None = None,
        working_memory:   list[Turn]               | None = None,
    ) -> tuple[str, str]:
        """
        Assemble the canonical 7-slot prompt.

        Parameters
        ----------
        instruction :
            The raw user instruction (Slot 7). Never transformed.
        session_files :
            Uploaded session files ([SESSION FILES], unnumbered slot before
            Slot 3). Per-file ceiling 4,000 tokens; total slot ceiling
            20,000 tokens. Pass None or [] to omit.
        persona :
            Optional persona string (Slot 1b). Injected into the system
            message when provided; 500-token ceiling enforced.
        episodic_summary :
            Pre-sorted EpisodeBullet list (Slot 3a). Caller is responsible
            for priority ordering. Pass None or [] to omit.
        profile_facts :
            Pre-scored UserProfileFact list (Slot 3b). Caller is responsible
            for relevance scoring and line selection. Pass None or [] to omit.
            100-token sub-budget within Slot 3.
        rag_snippets :
            Ranked RagSource list (Slot 4). At most 3 used. Pass None or
            [] to omit.
        tool_results :
            ToolResult list in dispatch order (Slot 5). Pass None or []
            to omit.
        graph_result :
            GraphQueryResult for the current turn (Slot 5b). When not None,
            always rendered — even with zero links. Pass None to omit.
        working_state :
            Structured working state (Slot 6A). Deterministic; not inferred.
            Rendered between tool-results and working-memory slots.
            100-token ceiling; active_artifacts truncated from the end.
            Pass None to omit (zero-impact on output when absent).
        working_memory :
            Recent conversation turns (Slot 6). Oldest dropped first when
            the 300-token ceiling is exceeded. Pass None or [] to omit.

        Returns
        -------
        (system_prompt, user_prompt) : tuple[str, str]
            system_prompt : Slot 1a + optional 1b. Pass as `system=` to
                            the runtime.
            user_prompt   : [SESSION FILES] (when present) followed by
                            Slots 3–7 joined with double newlines.
                            Empty slots are cleanly absent. Slot 3 includes
                            an optional [USER PROFILE] sub-block (Slot 3b)
                            when profile_facts are provided. Slot 5b
                            ([GRAPH RESULT]) is always rendered when
                            graph_result is not None; see _slot_graph().
        """
        system_prompt = self._slot1_system(persona)

        slots = [
            self._slot_session_files(session_files),   # unnumbered — before Slot 3
            self._slot3_combined(episodic_summary, profile_facts),
            self._slot4_rag(rag_snippets),
            self._slot5_tools(tool_results),
            self._slot_graph(graph_result),
            self._slot6a_working_state(working_state),
            self._slot6_working_memory(working_memory),
            self._slot7_instruction(instruction),
        ]

        user_prompt = "\n\n".join(s for s in slots if s)
        return system_prompt, user_prompt
