"""
LORA — Planner (Rule Engine)
=============================
Deterministic routing engine. Evaluates a priority-ordered set of
conditions against the instruction and context, and produces a RoutingPlan.

The Planner never answers. It produces a RoutingPlan. The ControllerAgent
executes the plan.

Inference is invoked in exactly one place: Priority 5 (episodic relevance).
Priorities 1–4 and 6 are pure rule evaluations — no model calls.

Reference: §4 of LOCALIST-Architecture.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

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
    graph_query :
        (direction, node_id, resolved_stem) when Priority 3c matched a
        structural graph lookup; None otherwise. direction is "incoming"
        or "outgoing". resolved_stem is carried alongside node_id so
        downstream consumers skip a second MemoryManager round-trip just
        to recover the display name.
    """
    agent:          str
    fetch_episodic: bool
    fetch_rag:      bool
    tools_to_call:  list[str]  = field(default_factory=list)
    write_episode:  bool       = False
    episode_type:   str | None = None
    compound:       bool       = False
    priority:       int        = 6
    graph_query:    tuple[str, int, str] | None = None


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
# Multi-word phrases carry no false-positive risk with _any_whole_word().
# Single words ("today", "recent", "news") are protected by \b anchors.
_WEB_SEARCH_KEYWORDS: frozenset[str] = frozenset({
    "latest",
    "current price",
    "current version",
    "current ceo",
    "current status",
    "current rate",
    "today",
    "news",
    "recent",
    "web search",
    "do a search",
})

# Priority 3b — factual query keywords (trigger web search when corpus misses)
_FACTUAL_QUERY_KEYWORDS: frozenset[str] = frozenset({
    "when did",
    "what year",
    "who founded",
    "who invented",
    "who created",
    "where was",
    "how many",
    "what is the",
    "which company",
    "who was the first",
    "what was the first",
})

# Priority 4 — explicit wiki/vault query triggers
_WIKI_QUERY_KEYWORDS: frozenset[str] = frozenset({
    "check the wiki",
    "search the wiki",
    "what's in my vault",
    "what is in my vault",
    "look in the wiki",
    "from the wiki",
    "in the wiki",
    "vault",
})

# Priority 3 — URL fetch trigger keywords (explicit only)
_FETCH_KEYWORDS: frozenset[str] = frozenset({
    "fetch this",
    "fetch the url",
    "fetch this url",
    "read this link",
    "read this url",
    "open this link",
    "summarize this url",
    "summarize this link",
    "extract this",
})

# Priority 3 — file operation trigger keywords
_FILE_OP_KEYWORDS: frozenset[str] = frozenset({
    "read the file",
    "read file",
    "write",
    "open the file",
    "save",
    "create a file",
})

# Diagnostic Slot 1 — canonical template groups for semantic search-intent scoring.
# These strings are embedded at startup and used only for cosine-similarity logging;
# they do not gate any routing decision until a threshold prompt replaces this comment.
_SEARCH_INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "explicit_search_action": (
        "search the web for this",
        "do a web search for this",
        "search online for this",
        "google this",
        "go look it up",
    ),
    "lookup_request": (
        "look up this",
        "look that up",
        "go ahead and look it up",
        "find information on this",
        "find out about this",
        "can you look up",
        "can you look that up for me",
        "could you look up",
        "can you look into this for me",
    ),
    "knowledge_request_open": (
        "what is this",
        "what do you know about this",
        "tell me about this",
        "explain this to me",
    ),
    "freshness_request": (
        "what's the latest on this",
        "what's the current status of this",
        "is there anything new about this",
    ),
}

_SEARCH_NEGATIVE_FILTER: frozenset[str] = frozenset({
    "search your memory",
    "search my previous messages",
    "search this conversation",
    "what did i just say",
    "what did you just say",
    "why didn't you search",
    "why did you search",
    "did you search",
    "what tool did you use",
    # 2026-06-26: identity/capability questions confirmed as false positives via
    # diagnostics/score_lookup_request_templates.py — these utterances cross the
    # lookup_request 0.60 gate due to syntactic (not semantic) similarity with the
    # four question-form templates added 2026-06-25 ("can you look up", etc.).
    "who are you",
    "what are you",
    "what can you do",
    "what can you help with",
    "what do you do",
})

# Relevance threshold for Priority 4 corpus scoring
_CORPUS_SCORE_THRESHOLD: float = 0.55

# Slot [Fix 1] — semantic search-intent gating thresholds.
# Derived from Diagnostic 2's live-backend score table (18 real
# utterances against mlx-community/embeddinggemma-300m-4bit).
# knowledge_request_open and freshness_request are deliberately
# excluded from gating — see Diagnostic 2 findings: a non-search
# utterance ("Explain this code to me.") scored 0.795 on
# knowledge_request_open, higher than 5 of 10 real positive
# paraphrases, because that group's templates are generically
# conversational. freshness_request showed one ambiguous negative
# (0.642, inside the positive range) and has not been independently
# stress-tested. Both groups remain computed and logged for future
# tuning but must never gate tools_to_call until separately re-evaluated.
#
# 2026-06-25 update A (§8.8 Open Item 11): lookup_request templates expanded
# from 5 to 9. The original 5 were bare imperatives with vague pronoun
# objects ("look up this", "look that up", etc.); they did not represent
# the "Can/Could you + look up/look into + [specific object]" question-form
# frame. Three live utterances using that frame scored 0.593, 0.598, and
# 0.598 on lookup_request — consistently below the 0.65 gate — causing
# gate_fired=False and no web_search dispatch. Four new templates appended:
# "can you look up", "can you look that up for me", "could you look up",
# "can you look into this for me". This was a template-coverage fix, not a
# threshold adjustment; the 0.65 threshold was deliberately left unchanged.
#
# 2026-06-25 update B (§10.4 Open Item 3 revisit, §8.8 Open Item 11):
# lookup_request threshold lowered from 0.65 to 0.60. Live re-verification
# after update A showed the same three utterances scored 0.608, 0.621, and
# 0.617 — a real, consistent improvement (+0.015 to +0.023) but still below
# the 0.65 gate. Template coverage alone cannot close this remaining gap for
# this phrasing family at this scale of addition. The 0.03–0.04 shortfall
# across all three utterances matches Open Item 3's stated revisit criterion
# ("if live false negatives are observed"). explicit_search_action (0.68) is
# NOT changed. Known risk: the original 18-utterance diagnostic pass did not
# retain per-utterance scores for lookup_request's adversarial negatives, so
# the margin to the new 0.60 line is unknown. Accepted and named risk — any
# live false positive on lookup_request is the trigger to revisit this value.
_SEMANTIC_GATE_THRESHOLDS: dict[str, float] = {
    "explicit_search_action": 0.68,
    "lookup_request": 0.60,
}


# ---------------------------------------------------------------------------
# Graph-query extraction and name resolution  (P3c, Phase C Step 3a)
#
# extract_graph_query() and resolve_graph_target() are pure, stateless
# functions with zero MemoryManager dependency. The caller (ControllerAgent /
# route(), wired in Step 3b) injects the real stem list at call time.
# ---------------------------------------------------------------------------

# Pattern B — incoming backlink lead-phrases, sorted longest-first.
# Longest-first ordering is load-bearing: a shorter phrase must not
# shadow a longer, more specific one that is also a prefix of the
# instruction (even if no collision exists in the current set, any
# future addition could introduce one).
_BACKLINK_LEAD_PHRASES: tuple[str, ...] = (
    "show me backlinks for",
    "what pages link to",
    "backlinks for",
    "what links to",
    "what references",
    "what mentions",
)

# Pattern C — outgoing lead-phrases, same longest-first principle.
_OUTGOING_LEAD_PHRASES: tuple[str, ...] = (
    "show outgoing links for",
    "outgoing links for",
    "what links from",
    "links from",
)

# Token-overlap stopwords for Tier 2 name resolution.
_GRAPH_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "is", "are",
    "and", "or", "page", "pages", "about", "that",
})

# Pattern A compiled regex (module-level, evaluated once at import time).
_PATTERN_A_OUTGOING: re.Pattern[str] = re.compile(
    r"what does (.+?) link to\??\.?$", re.IGNORECASE
)


def _normalize_graph_text(text: str) -> str:
    # Identical to wiki_agent._validate_links() / build_graph._normalize():
    # link_text.lower().replace(" ", "-").
    # .strip() is added here because remainder may carry incidental leading/
    # trailing whitespace that the write-time normalizer never sees (link_text
    # there always comes pre-trimmed from XML parsing).
    return text.lower().strip().replace(" ", "-")


def extract_graph_query(instruction: str) -> tuple[str, str] | None:
    """
    Attempt to extract a (direction, remainder) pair from `instruction`
    using three deterministic patterns. Returns None if no pattern matches.

    direction : "incoming" | "outgoing"
    remainder : whatever text follows/is captured by the matched
                pattern — passed directly to resolve_graph_target()
                with no further cleanup, even if empty.
    """
    instruction = instruction.strip()

    # Pattern A is checked first because it is a regex anchored at both
    # ends (re.match + $ anchor) and syntactically more specific than B/C's
    # startswith checks. Checking it first avoids any ambiguity on inputs
    # that match "what does X link to" — no B/C phrase can compete with that
    # anchored form, but testing A before B/C makes the priority explicit.
    m = _PATTERN_A_OUTGOING.match(instruction)
    if m:
        return ("outgoing", m.group(1).strip())

    # Normalize for B and C: lowercase, strip surrounding whitespace, then
    # strip trailing punctuation so "What links to X?" and "what links to X"
    # reach the same startswith comparison.
    normalized = instruction.strip().lower().rstrip("?.")

    # Pattern B — incoming: longest-first ordering prevents a shorter phrase
    # from matching before a longer, more specific one.
    for phrase in _BACKLINK_LEAD_PHRASES:
        if normalized.startswith(phrase):
            return ("incoming", normalized[len(phrase):].strip())

    # Pattern C — outgoing: same longest-first principle as Pattern B.
    for phrase in _OUTGOING_LEAD_PHRASES:
        if normalized.startswith(phrase):
            return ("outgoing", normalized[len(phrase):].strip())

    return None


def resolve_graph_target(
    remainder:       str,
    candidate_stems: list[str],
) -> str | None:
    """
    Resolve `remainder` (raw extracted text) against `candidate_stems`
    (a list of existing page stems, e.g. ["how-localist-works",
    "localist-build-order", ...]) using a three-tier deterministic
    pipeline. Returns the single matching stem, or None if resolution
    is ambiguous or finds no match (both cases are identical from the
    caller's perspective — never distinguish "zero" from "multiple").
    """
    normalized = _normalize_graph_text(remainder)

    # Tier 1 — symmetric substring match.
    # Match if: normalized remainder is a substring of the stem, OR the
    # stem is a substring of the normalized remainder (either direction).
    tier1 = [
        stem for stem in candidate_stems
        if normalized in stem or stem in normalized
    ]
    if len(tier1) == 1:
        return tier1[0]
    if len(tier1) > 1:
        # Multi-match: ambiguous. Never tiebreak — return None immediately.
        return None
    # Zero Tier 1 matches → fall through to Tier 2.

    # Tier 2 — token-overlap fallback (only reached when Tier 1 found zero).
    query_tokens = set(normalized.split("-")) - _GRAPH_STOPWORDS
    if len(query_tokens) < 2:
        # Too few meaningful tokens to score reliably; skip Tier 2 entirely
        # and fall through to Tier 3 (return None). This is the "skipped,
        # not scored" path — a low ratio is not the reason for the None here.
        return None

    # Asymmetric ratio: intersection / query-token count (NOT Jaccard).
    # Rewards stems where most of the query's meaningful words are present;
    # does not penalise stems for having more tokens than the query.
    tier2 = [
        stem for stem in candidate_stems
        if len(query_tokens & set(stem.split("-"))) / len(query_tokens) >= 0.5
    ]
    if len(tier2) == 1:
        return tier2[0]
    return None  # zero or ambiguous — Tier 3 fallthrough (return None)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Rule engine that produces a RoutingPlan from an instruction and context.

    Priority order (first match wins):
      1.  Ingest signal      — deterministic keyword/context check
      2.  Memory command     — deterministic keyword check
      3c. Graph-query        — structural link lookup; wins over web_search
                               but defers to file_op/url_fetch (inline guard)
      3.  Tool signal        — deterministic keyword check
      3b. Factual query      — keyword + corpus miss → web_search
      4.  Corpus signal      — deterministic score threshold check
      5.  Episodic relevance — single bounded inference call (added in 3.3)
      6.  Direct answer      — fallback (added in 3.4)

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
        embed_fn:       Callable[[str], list[float]] | None = None,
    ) -> None:
        self._runtime        = runtime
        self._memory_manager = memory_manager
        self._embed_fn       = embed_fn
        # Session state for Priority 5 caching (§4.3)
        # _episodic_injected: True once episodic bullets have been injected
        #   this session; causes all further Priority 5 checks to return True
        #   without an inference call (relevance assumed to persist).
        # _episodic_cache_pairs: parallel list of (embedding, result)
        #   pairs used for cosine similarity lookup, since dict keys cannot
        #   be float lists.
        self._episodic_injected: bool = False
        self._episodic_cache_pairs: list[tuple[list[float], bool]] = []

        # Diagnostic Slot 1 — flat list of (group_name, vector) pairs, one per
        # template string across all groups in _SEARCH_INTENT_TEMPLATES.
        # Only populated when embed_fn is available; left empty on failure so
        # _semantic_search_intent() safely returns None without logging noise.
        self._template_embeddings: list[tuple[str, list[float]]] = []
        if embed_fn is not None:
            try:
                for group_name, phrases in _SEARCH_INTENT_TEMPLATES.items():
                    for phrase in phrases:
                        vec = embed_fn(phrase)
                        self._template_embeddings.append((group_name, vec))
                logger.debug(
                    "Planner: pre-embedded %d search-intent templates across %d groups.",
                    len(self._template_embeddings), len(_SEARCH_INTENT_TEMPLATES),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Planner: failed to embed search-intent templates — "
                    "semantic search-intent check will be skipped. Error: %s", exc,
                )
                self._template_embeddings = []

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

        # Priority 3c — Graph-query (runs before P3 so a graph-query wins over a
        # web_search-only P3 match; P3c's inline guard defers to P3 when
        # file_op/url_fetch signals are present)
        plan = self._priority3c_graph_query(instruction, lowered)
        if plan is not None:
            return plan

        # Priority 3 — Tool signal
        plan = self._priority3_tool(lowered)
        if plan is not None:
            return plan

        # Priority 3b — Factual query + corpus miss
        plan = self._priority3b_factual(instruction, lowered)
        if plan is not None:
            return plan

        # Priority 4 — Corpus signal
        plan = self._priority4_corpus(lowered, instruction)
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
                priority       = 1,
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
                priority       = 2,
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

    def _semantic_search_intent(
        self, lowered: str
    ) -> tuple[str, float, dict[str, float]] | None:
        """
        Diagnostic-only (Slot [Diagnostic 2]): returns the best-matching
        group, its score, AND the max score per group across all groups —
        so collisions between groups can be inspected directly, not just
        inferred from the argmax.

        Returns (best_group, best_score, all_group_scores) or None under
        the same conditions as Diagnostic 1 (no embed_fn, negative filter
        match, embedding failure).
        """
        if self._embed_fn is None or not self._template_embeddings:
            return None

        if any(phrase in lowered for phrase in _SEARCH_NEGATIVE_FILTER):
            logger.debug(
                "Planner: Priority 3 semantic check skipped — negative filter matched (%r).",
                next(p for p in _SEARCH_NEGATIVE_FILTER if p in lowered),
            )
            return None

        try:
            query_vec = self._embed_fn(lowered)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Planner: _semantic_search_intent — embed_fn raised: %s", exc,
            )
            return None

        # Per-group max scores: accumulate highest similarity for each group.
        group_scores: dict[str, float] = {g: -1.0 for g in _SEARCH_INTENT_TEMPLATES}
        for group_name, tmpl_vec in self._template_embeddings:
            score = _cosine_similarity(query_vec, tmpl_vec)
            if score > group_scores[group_name]:
                group_scores[group_name] = score

        best_group = max(group_scores, key=lambda g: group_scores[g])
        best_score = group_scores[best_group]

        if best_score < 0:
            return None

        return (best_group, best_score, group_scores)

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

        semantic_result = self._semantic_search_intent(lowered)
        semantic_triggered = False
        if semantic_result is not None:
            best_group, best_score, all_scores = semantic_result
            semantic_triggered = any(
                all_scores.get(group, 0.0) >= threshold
                for group, threshold in _SEMANTIC_GATE_THRESHOLDS.items()
            )
            logger.debug(
                "Planner: Priority 3 — semantic signal best=%r(%.3f) all=%s "
                "gate_fired=%s.",
                best_group, best_score, all_scores, semantic_triggered,
            )

        if semantic_triggered and "web_search" not in tools:
            tools.append("web_search")
            logger.debug(
                "Planner: Priority 3 — web_search signal detected via semantic "
                "gate (best_group=%r, best_score=%.3f).", best_group, best_score,
            )

        fo_kw = self._any_whole_word(_FILE_OP_KEYWORDS, lowered)
        if fo_kw:
            tools.append("file_op")
            logger.debug(
                "Planner: Priority 3 — file_op signal detected (%r).", fo_kw
            )

        if self._any_whole_word(_FETCH_KEYWORDS, lowered) or re.search(
            r"https?://", lowered
        ):
            tools.append("url_fetch")
            logger.debug("Planner: Priority 3 — url_fetch signal detected.")

        if tools:
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = False,
                tools_to_call  = tools,
                compound       = True,
                priority       = 3,
            )
        return None

    # -----------------------------------------------------------------------
    # Priority 3c — Graph-query  (structural link lookup)
    # -----------------------------------------------------------------------

    def _priority3c_graph_query(
        self,
        instruction: str,
        lowered:     str,
    ) -> RoutingPlan | None:
        """
        Priority 3c — Graph-query (structural "what links to X" /
        "what does X link to" lookup).

        Wins over web_search/P3b/P4a/P4 but loses to P1/P2 (handled by
        route()'s call order, not here) and to P3's file_op/url_fetch
        signals specifically (handled by the inline guard below — there is
        no standalone pre-check method for those two signals; the module-
        level note above _priority3_tool() explains the design rationale —
        do NOT remove that comment when editing nearby code).

        Returns None (deferring to normal priority evaluation) if:
          - file_op or url_fetch keywords are present (inline guard), OR
          - no MemoryManager is available, OR
          - no extraction pattern matches the instruction, OR
          - name resolution fails at all tiers (zero or ambiguous matches).

        On success, returns a RoutingPlan with graph_query set and
        fetch_rag/fetch_episodic all False — graph-query turns are deliberately
        pure/minimal, never combined with RAG or episodic context (Phase C design).
        """
        # 1. Inline file_op/url_fetch guard.
        #    Intentionally duplicates three lines from _priority3_tool() —
        #    this duplication is deliberate (no standalone pre-check exists
        #    for those two signals without modifying _priority3_tool()).
        #    _WEB_SEARCH_KEYWORDS is NOT guarded here: P3c must win over a
        #    web_search-only P3 match (hence its placement before P3 in route()).
        if (
            self._any_whole_word(_FILE_OP_KEYWORDS, lowered)
            or self._any_whole_word(_FETCH_KEYWORDS, lowered)
            or re.search(r"https?://", lowered)
        ):
            return None

        # 2. MemoryManager availability check (same pattern as P3b/P4).
        if self._memory_manager is None:
            logger.debug("Planner: Priority 3c skipped — no MemoryManager.")
            return None

        # 3. Extraction.
        extracted = extract_graph_query(instruction)
        if extracted is None:
            return None
        direction, remainder = extracted

        # 4. Fetch candidate stems from the live graph_nodes table.
        try:
            candidate_stems = self._memory_manager.list_graph_node_stems()
        except Exception as exc:
            logger.warning(
                "Planner: Priority 3c — list_graph_node_stems failed: %s", exc
            )
            return None

        # 5. Name resolution (three-tier deterministic pipeline).
        resolved_stem = resolve_graph_target(remainder, candidate_stems)
        if resolved_stem is None:
            logger.debug(
                "Planner: Priority 3c — name resolution failed for %r.", remainder
            )
            return None

        # 6. Node id lookup.  Should always succeed since resolved_stem came
        #    from the same graph_nodes table, but guard defensively in case of
        #    a race between step 4 and step 6 in a concurrent-write scenario.
        try:
            node = self._memory_manager.resolve_node_by_stem(resolved_stem)
        except Exception as exc:
            logger.warning(
                "Planner: Priority 3c — resolve_node_by_stem failed: %s", exc
            )
            return None

        if node is None:
            logger.warning(
                "Planner: Priority 3c — resolve_node_by_stem returned None for "
                "%r (race condition or DB inconsistency).", resolved_stem
            )
            return None

        node_id = node["id"]
        logger.debug(
            "Planner: Priority 3c matched — direction=%r stem=%r node_id=%d.",
            direction, resolved_stem, node_id,
        )

        # 7. Build the plan.
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            compound       = False,
            priority       = 3,
            graph_query    = (direction, node_id, resolved_stem),
        )

    # -----------------------------------------------------------------------
    # Priority 3b — Factual query + corpus miss
    # -----------------------------------------------------------------------

    def _priority3b_factual(self, instruction: str, lowered: str) -> RoutingPlan | None:
        """
        Match condition: instruction contains a factual query keyword AND
        corpus query returns no result above _CORPUS_SCORE_THRESHOLD.

        When no MemoryManager is available, this priority is skipped entirely
        (returns None) — corpus cannot be checked.

        Returns a RoutingPlan with tools_to_call=["web_search"], or None.
        """
        if not any(kw in lowered for kw in _FACTUAL_QUERY_KEYWORDS):
            return None

        if self._memory_manager is None:
            logger.debug("Planner: Priority 3b skipped — no MemoryManager.")
            return None

        try:
            results = self._memory_manager.query_corpus(
                instruction, max_results=1
            )
        except Exception as exc:
            logger.warning("Planner: Priority 3b corpus check failed: %s", exc)
            results = []

        top_score = results[0].relevance_score if results else 0.0

        if top_score >= _CORPUS_SCORE_THRESHOLD:
            logger.debug(
                "Planner: Priority 3b — corpus hit (score=%.3f), "
                "deferring to Priority 4.", top_score,
            )
            return None

        logger.debug(
            "Planner: Priority 3b matched — factual keyword, "
            "corpus miss (score=%.3f), scheduling web_search.", top_score,
        )
        return RoutingPlan(
            agent          = "conversational_agent",
            fetch_episodic = False,
            fetch_rag      = False,
            tools_to_call  = ["web_search"],
            compound       = True,
            priority       = 3,
        )

    # -----------------------------------------------------------------------
    # Priority 4 — Corpus signal  (§4.2, Priority 4)
    # -----------------------------------------------------------------------

    def _priority4_corpus(self, lowered: str, instruction: str) -> RoutingPlan | None:
        """
        Match condition (either sufficient):
          A) instruction contains an explicit wiki/vault trigger keyword, OR
          B) MemoryManager is available AND query_corpus() returns a top result
             with relevance_score >= _CORPUS_SCORE_THRESHOLD.

        Path A keeps routing deterministic for explicit wiki requests.
        Path B restores score-based RAG injection for natural-language corpus
        queries that lack a trigger keyword (e.g. "summarize the LORA Master
        Project Outline").

        Returns a RoutingPlan with fetch_rag=True, or None if no match.
        """
        # Path A — explicit keyword trigger
        matched_kw = next(
            (kw for kw in _WIKI_QUERY_KEYWORDS if kw in lowered), None
        )
        if matched_kw is not None:
            logger.debug(
                "Planner: Priority 4 matched via keyword (%r).", matched_kw
            )
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = True,
                fetch_rag      = True,
                compound       = False,
                priority       = 4,
            )

        # Path B — corpus score threshold
        if self._memory_manager is None:
            logger.debug("Planner: Priority 4 Path B skipped — no MemoryManager.")
            return None

        try:
            results = self._memory_manager.query_corpus(
                instruction, max_results=1
            )
        except Exception as exc:
            logger.warning("Planner: Priority 4 corpus check failed: %s", exc)
            return None

        top_score = results[0].relevance_score if results else 0.0

        if top_score >= _CORPUS_SCORE_THRESHOLD:
            logger.debug(
                "Planner: Priority 4 matched via corpus score (%.3f >= %.3f).",
                top_score, _CORPUS_SCORE_THRESHOLD,
            )
            return RoutingPlan(
                agent          = "conversational_agent",
                fetch_episodic = False,
                fetch_rag      = True,
                compound       = False,
                priority       = 4,
            )

        logger.debug(
            "Planner: Priority 4 — corpus miss (top_score=%.3f).", top_score
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
        # Cache check: session-level flag.
        # When set, we skip the (previously: inference) call and go straight
        # to keyword evaluation — but we do NOT return True unconditionally.
        # A turn with no episodic keyword still returns None so P6 or P4 can
        # handle it correctly.
        _skip_inference = self._episodic_injected
        if _skip_inference:
            logger.debug(
                "Planner: Priority 5 — episodic_injected=True; "
                "skipping inference, proceeding to keyword check."
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
            # Personal reference signals (unambiguous — always fetch episodic)
            "my name", "do you remember", "who am i",
            "what do you know about me", "my preference",
            "what did i tell you", "what have i told you",
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
                priority       = 5,
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
            priority       = 6,
        )
