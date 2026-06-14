"""
LORA — Planner (Rule Engine)
=============================
Deterministic routing engine. Evaluates a priority-ordered set of
conditions against the instruction and context, and produces a RoutingPlan.

The Planner never answers. It produces a RoutingPlan. The ControllerAgent
executes the plan.

Inference is invoked in exactly one place: Priority 5 (episodic relevance).
Priorities 1–4 and 6 are pure rule evaluations — no model calls.

Reference: §4 of LORA-Architecture.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    import math
    if len(a) != len(b) or not a:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# RoutingPlan  (§4.4)
# ---------------------------------------------------------------------------

@dataclass
class RoutingPlan:
    """
    The output of the Planner. Consumed by ControllerAgent.handle_task().

    Fields
    ------
    agent :
        "wiki_agent" | "conversational_agent"
    fetch_episodic :
        True → retrieve from episodes table and inject into slot 4.
    fetch_rag :
        True → query_corpus() before responding; results populate slot 5.
    tools_to_call :
        Tool names in dispatch order. Empty list if no tools needed.
    write_episode :
        True → EpisodicMemoryWriter runs before the agent call.
    episode_type :
        Type hint for episodic extraction. None when write_episode is False.
    compound :
        True → multiple signal types detected; ControllerAgent sequences
        execution in priority order.
    """
    agent:          str
    fetch_episodic: bool
    fetch_rag:      bool
    tools_to_call:  list[str]  = field(default_factory=list)
    write_episode:  bool       = False
    episode_type:   str | None = None
    compound:       bool       = False


# ---------------------------------------------------------------------------
# Keyword sets  (§4.2)
# ---------------------------------------------------------------------------

# Priority 1 — ingest keywords (checked against lowercased instruction)
_INGEST_KEYWORDS: frozenset[str] = frozenset({
    "ingest",
    "process this file",
    "add to wiki",
    "index this",
})

# Priority 2 — explicit memory command keywords
_MEMORY_KEYWORDS: frozenset[str] = frozenset({
    "remember that",
    "my preference is",
    "that's wrong",
    "the correct value is",
    "forget that",
    "mark complete",
    "that's no longer true",
})

# Priority 3 — web search trigger keywords
# "current" was removed (too broad — matches "currently", valid non-news uses);
# replaced with "current price" as a compound phrase to reduce false positives.
_WEB_SEARCH_KEYWORDS: frozenset[str] = frozenset({
    "latest",
    "current price",
    "today",
    "news",
    "recent",
})

# Priority 3 — file operation trigger keywords
_FILE_OP_KEYWORDS: frozenset[str] = frozenset({
    "read",
    "write",
    "open the file",
    "save",
    "create a file",
})

# Relevance threshold for Priority 4 corpus scoring
_CORPUS_SCORE_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Rule engine that produces a RoutingPlan from an instruction and context.

    Priority order (first match wins):
      1. Ingest signal       — deterministic keyword/context check
      2. Memory command      — deterministic keyword check
      3. Tool signal         — deterministic keyword check
      4. Corpus signal       — deterministic score threshold check
      5. Episodic relevance  — single bounded inference call (added in 3.3)
      6. Direct answer       — fallback (added in 3.4)

    The Planner never calls agents, never writes to the database,
    and never assembles a prompt for the final answer.

    Parameters
    ----------
    runtime :
        RuntimeClient. Used only for the Priority 5 inference call.
        Not used in Priorities 1–4 or 6.
    memory_manager :
        Optional MemoryManager. Required for Priority 4 corpus scoring.
        When absent, Priority 4 is skipped (no corpus to query).
    """

    # Class-level aliases so callers can access via instance (e.g. p._WEB_SEARCH_KEYWORDS)
    _WEB_SEARCH_KEYWORDS: frozenset[str] = _WEB_SEARCH_KEYWORDS
    _FILE_OP_KEYWORDS:    frozenset[str] = _FILE_OP_KEYWORDS

    def __init__(
        self,
        runtime:        Any,
        memory_manager: "MemoryManager | None" = None,
    ) -> None:
        self._runtime        = runtime
        self._memory_manager = memory_manager
        # Session state for Priority 5 caching (§4.3)
        # _episodic_injected: True once episodic bullets have been injected
        #   this session; causes all further Priority 5 checks to return True
        #   without an inference call (relevance assumed to persist).
        # _episodic_cache_pairs: parallel list of (embedding, result)
        #   pairs used for cosine similarity lookup, since dict keys cannot
        #   be float lists.
        self._episodic_injected: bool = False
        self._episodic_cache_pairs: list[tuple[list[float], bool]] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def mark_episodic_injected(self) -> None:
        """
        Signal that episodic bullets were injected into the prompt this session.

        After this is called, Priority 5 will return fetch_episodic=True for
        all subsequent instructions in the same session without making another
        inference call. Relevance is assumed to persist within a session.

        Called by ControllerAgent after a successful episodic retrieval and
        prompt injection.
        """
        self._episodic_injected = True
        logger.debug("Planner: episodic_injected flag set for this session.")

    def _detect_compound(
        self,
        lowered: str,
        context: dict[str, Any],
    ) -> RoutingPlan | None:
        """
        Detect compound instructions that trigger multiple priority conditions
        simultaneously and require special resolution (§4.5).

        Currently handles one compound case:

        Tool + Ingest
        -------------
        Condition: instruction matches both Priority 1 (ingest) AND Priority 3
        (web search tool).
        Resolution: wiki_agent with tools_to_call=["web_search"], compound=True.
        The ControllerAgent will execute the web_search tool first and pass
        the result as context to WikiAgent.

        Returns a RoutingPlan if a compound pattern is detected, or None if
        the instruction should proceed through normal priority evaluation.
        """
        has_ingest = (
            "raw_path" in context
            or any(kw in lowered for kw in _INGEST_KEYWORDS)
        )
        has_web_search = bool(self._any_whole_word(_WEB_SEARCH_KEYWORDS, lowered))

        if has_ingest and has_web_search:
            logger.debug(
                "Planner: compound detected — Tool + Ingest "
                "(ingest=%s, web_search=%s).",
                has_ingest, has_web_search,
            )
            return RoutingPlan(
                agent          = "wiki_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                tools_to_call  = ["web_search"],
                compound       = True,
            )

        return None

    def route(
        self,
        instruction: str,
        context:     dict[str, Any],
    ) -> RoutingPlan:
        """
        Evaluate priorities 1–6 in order and return the first matching plan.

        Parameters
        ----------
        instruction :
            The raw user instruction string.
        context :
            The task context dict (may contain raw_path, wiki_dir, etc.).

        Returns
        -------
        RoutingPlan
            Never raises. Guaranteed to return a valid plan.
        """
        lowered = instruction.lower()

        # Compound detection — must run before priority evaluation
        plan = self._detect_compound(lowered, context)
        if plan is not None:
            return plan

        # Priority 1 — Ingest signal
        plan = self._priority1_ingest(lowered, context)
        if plan is not None:
            return plan

        # Priority 2 — Explicit memory command
        plan = self._priority2_memory(lowered)
        if plan is not None:
            return plan

        # Priority 3 — Tool signal
        plan = self._priority3_tool(lowered)
        if plan is not None:
            return plan

        # Priority 4 — Corpus signal
        plan = self._priority4_corpus(instruction)
        if plan is not None:
            # P4 matched — also run P5 to check episodic relevance.
            # If both match, merge into a compound plan so the controller
            # fetches both RAG context and episodic memory for this turn.
            p5 = self._priority5_episodic(instruction)
            if p5 is not None:
                logger.debug(
                    "Planner: P4+P5 compound — fetch_rag=True fetch_episodic=True."
                )
                plan.fetch_episodic = True
            return plan

        # Priority 5 — Episodic relevance (single bounded inference call)
        plan = self._priority5_episodic(instruction)
        if plan is not None:
            return plan

        # Priority 6 — Direct answer fallback
        return self._priority6_direct()

    # -----------------------------------------------------------------------
    # Priority 1 — Ingest signal  (§4.2, Priority 1)
    # -----------------------------------------------------------------------

    def _priority1_ingest(
        self,
        lowered: str,
        context: dict[str, Any],
    ) -> RoutingPlan | None:
        """
        Match condition: raw_path key present in context OR ingest keyword
        in lowercased instruction.

        Returns a RoutingPlan routed to wiki_agent, or None if no match.
        fetch_rag and fetch_episodic are explicitly False — ingest is never
        augmented with retrieval.
        """
        has_raw_path = "raw_path" in context
        has_keyword  = any(kw in lowered for kw in _INGEST_KEYWORDS)

        if has_raw_path or has_keyword:
            logger.debug(
                "Planner: Priority 1 matched (raw_path=%s, keyword=%s).",
                has_raw_path, has_keyword,
            )
            return RoutingPlan(
                agent          = "wiki_agent",
                fetch_episodic = False,
                fetch_rag      = False,
            )
        return None

    # -----------------------------------------------------------------------
    # Priority 2 — Explicit memory command  (§4.2, Priority 2)
    # -----------------------------------------------------------------------

    def _priority2_memory(self, lowered: str) -> RoutingPlan | None:
        """
        Match condition: any memory command keyword present in lowercased
        instruction.

        Sets write_episode=True. The episode_type hint is left as None here
        — the EpisodicMemoryWriter will infer type from content. After
        writing, routing continues to Priority 4 or 6 for the response; the
        compound flag is set to True to signal that sequencing is needed.

        Returns a RoutingPlan or None if no match.
        """
        matched_kw = next(
            (kw for kw in _MEMORY_KEYWORDS if kw in lowered), None
        )
        if matched_kw is not None:
            logger.debug(
                "Planner: Priority 2 matched (keyword=%r).", matched_kw
            )
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                write_episode  = True,
                episode_type   = None,   # extracted by EpisodicMemoryWriter
                compound       = True,   # write first, then respond
            )
        return None

    # -----------------------------------------------------------------------
    # Priority 3 — Tool signal  (§4.2, Priority 3)
    # -----------------------------------------------------------------------

    @staticmethod
    def _any_whole_word(keywords: frozenset[str], text: str) -> str | None:
        """
        Return the first keyword from `keywords` that appears as a whole
        word (or whole phrase) in `text`, or None if no match.

        Multi-word keywords (e.g. "create a file") are matched as a literal
        phrase with word boundaries on each end. Single-word keywords are
        matched with \\b anchors. Matching is case-insensitive; callers
        should pass already-lowercased text.
        """
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                return kw
        return None

    def _priority3_tool(self, lowered: str) -> RoutingPlan | None:
        """
        Match condition: web search keyword OR file operation keyword present
        in lowercased instruction.

        Populates tools_to_call with "web_search" or "file_op" as
        appropriate. Sets compound=True when a tool is scheduled alongside
        a response agent, since the tool must run before the agent call.

        Returns a RoutingPlan or None if no match.
        """
        tools: list[str] = []

        ws_kw = self._any_whole_word(_WEB_SEARCH_KEYWORDS, lowered)
        if ws_kw:
            tools.append("web_search")
            logger.debug(
                "Planner: Priority 3 — web_search signal detected (%r).", ws_kw
            )

        fo_kw = self._any_whole_word(_FILE_OP_KEYWORDS, lowered)
        if fo_kw:
            tools.append("file_op")
            logger.debug(
                "Planner: Priority 3 — file_op signal detected (%r).", fo_kw
            )

        if tools:
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                tools_to_call  = tools,
                compound       = True,
            )
        return None

    # -----------------------------------------------------------------------
    # Priority 4 — Corpus signal  (§4.2, Priority 4)
    # -----------------------------------------------------------------------

    def _priority4_corpus(self, instruction: str) -> RoutingPlan | None:
        """
        Match condition: MemoryManager.query_corpus() returns at least one
        result with relevance_score >= _CORPUS_SCORE_THRESHOLD.

        When no MemoryManager is available, this priority is skipped entirely
        (returns None) — no corpus to query.

        Returns a RoutingPlan with fetch_rag=True, or None if no match.
        """
        if self._memory_manager is None:
            logger.debug(
                "Planner: Priority 4 skipped (no MemoryManager available)."
            )
            return None

        try:
            results = self._memory_manager.query_corpus(
                instruction,
                max_results    = 1,      # only need to know if any result clears the bar
                use_embeddings = True,
            )
            above_threshold = [
                r for r in results
                if r.relevance_score >= _CORPUS_SCORE_THRESHOLD
            ]
            if above_threshold:
                logger.debug(
                    "Planner: Priority 4 matched "
                    "(top score=%.3f >= threshold=%.1f).",
                    above_threshold[0].relevance_score,
                    _CORPUS_SCORE_THRESHOLD,
                )
                return RoutingPlan(
                    agent          = "conversational_agent",
                    fetch_episodic = False,
                    fetch_rag      = True,
                )
        except Exception as exc:
            logger.warning(
                "Planner: Priority 4 corpus query failed (%s) — skipping.", exc
            )

        return None

    # -----------------------------------------------------------------------
    # Priority 5 — Episodic relevance  (§4.2, §4.3)
    # -----------------------------------------------------------------------

    def _priority5_episodic(self, instruction: str) -> RoutingPlan | None:
        """
        Priority 5 — Episodic relevance (§4.2, §4.3).

        Replaced inference call with deterministic keyword check for
        Gemma 4B compatibility. The model requires max_tokens=300 to
        produce binary classifier output, making inference-based routing
        too expensive for a per-turn call.

        Returns a RoutingPlan with fetch_episodic=True if the instruction
        contains episodic relevance signals, or None if not.

        Session flag caching is preserved: once episodic bullets have been
        injected this session, all subsequent turns return fetch_episodic=True
        without keyword evaluation.
        """
        # Cache check: session-level flag (preserved from original)
        if self._episodic_injected:
            logger.debug(
                "Planner: Priority 5 — episodic already injected this "
                "session; returning fetch_episodic=True."
            )
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = True,
                fetch_rag      = False,
            )

        # Deterministic keyword check
        _EPISODIC_KEYWORDS: frozenset[str] = frozenset({
            "preference", "preferences", "remember", "remembered",
            "you know about me", "what do you know",
            "decision", "decisions", "decided",
            "correction", "corrections", "wrong",
            "workflow", "workflows",
            "last time", "previously", "before",
            "my project", "my setup", "my environment",
        })

        lowered = instruction.lower()
        matched = next(
            (kw for kw in _EPISODIC_KEYWORDS if kw in lowered), None
        )

        if matched:
            logger.debug(
                "Planner: Priority 5 — episodic keyword matched %r → "
                "fetch_episodic=True.", matched
            )
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = True,
                fetch_rag      = False,
            )

        logger.debug("Planner: Priority 5 — no episodic keyword matched.")
        return None

    # -----------------------------------------------------------------------
    # Priority 6 — Direct answer fallback  (§4.2, Priority 6)
    # -----------------------------------------------------------------------

    def _priority6_direct(self) -> RoutingPlan:
        """
        Priority 6 — Direct answer fallback (§4.2, Priority 6).

        Reached only when no prior priority matched. Routes to
        ConversationalAgent with no retrieval — the model answers from
        its own weights plus working memory (slots 1–3 only).
        """
        logger.debug("Planner: Priority 6 — direct answer fallback.")
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
        )
