"""
LORA — Episodic Extraction Pipeline
=====================================
Converts raw user instructions and agent responses into typed, scored
EpisodeRecord candidates for storage in the episodes table.

Pipeline
--------
1. Deterministic signal detection (5.1)
   Inspects the instruction for explicit memory command keywords.
   Maps trigger phrases to episode_type + a retraction flag.
   Returns ExtractionSignal or None.

2. Model-based content extraction (5.2)
   When a deterministic signal is detected but the durable content
   cannot be inferred from surface keywords alone, a single bounded
   inference call extracts a clean one-sentence content string from
   the full instruction.
   Also used for implicit extraction from the post-response turn pair.

3. Confidence scoring (5.3)
   Explicit signals → confidence = 1.0
   Model-extracted  → confidence scored 0.6–0.9 by response heuristic.

Reference: §2 and Phase 5 of LOCALIST-Architecture.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from memory_manager import EpisodicMemoryWriter, WorkingStateStore, WorkingStateRecord, VALID_EPISODE_TYPES
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_PROMPT_BUILDER = PromptBuilder()

# ---------------------------------------------------------------------------
# Deterministic signal tables  (§4.2 Priority 2 + §2.5 retraction rule)
# ---------------------------------------------------------------------------

# Maps a trigger phrase (lowercased) to the episode_type it implies.
# Retraction phrases are handled separately in _RETRACTION_SIGNALS.
_EXPLICIT_SIGNALS: dict[str, str] = {
    "remember that":         "preference",
    "my preference is":      "preference",
    "i prefer":              "preference",
    "that's wrong":          "correction",
    "that is wrong":         "correction",
    "the correct value is":  "correction",
    "actually,":             "correction",
    "mark complete":         "task_completion",
    "mark as complete":      "task_completion",
    "that's done":           "task_completion",
    "we decided":            "decision",
    "we've decided":         "decision",
    "the decision is":       "decision",
    "always":                "workflow",
    "every time":            "workflow",
    "my workflow is":        "workflow",
    "note that":             "project_fact",
    "fyi":                   "project_fact",
    "for the record":        "project_fact",
    "should be called":      "naming_convention",
    "not called":            "naming_convention",
    "the correct name is":   "naming_convention",
}

# Retraction phrases — matched before _EXPLICIT_SIGNALS.
# When matched: call retract() instead of insert().
_RETRACTION_SIGNALS: frozenset[str] = frozenset({
    "forget that",
    "that's no longer true",
    "that is no longer true",
    "ignore that",
    "disregard that",
    "scratch that",
})

# System prompt for the model-based extraction call.
_EXTRACTION_SYSTEM = (
    "You are a memory assistant. When given a user instruction, "
    "respond with exactly one sentence that captures the key fact "
    "as a third-person statement starting with \"The user\". "
    "Include specific details like names, versions, and platforms. "
    "Do not explain. Do not add preamble. "
    "If there is no durable fact to record, respond with: NONE\n\n"
    "Example:\n"
    "User says: Remember that I prefer dark mode.\n"
    "You respond: The user prefers dark mode.\n\n"
    "User says: Remember that I'm building on an M1 MacBook Air.\n"
    "You respond: The user is building the LORA project on an M1 "
    "MacBook Air running macOS."
)

# System prompt for the implicit extraction call (post-response hook).
_IMPLICIT_EXTRACTION_SYSTEM = (
    "You are an episodic memory extractor for a local AI research assistant. "
    "Given a conversation turn (user instruction + assistant response), "
    "determine if a durable personal fact, preference, decision, or workflow "
    "pattern was revealed. If so, output it as one self-contained sentence. "
    "If nothing durable was revealed, output the single word: NONE. "
    "No preamble. No explanation. One sentence or NONE."
)

# System prompt for the working-state update call (post-response hook, Slot 6A Tier 2).
_WORKING_STATE_UPDATE_SYSTEM = (
    "You are a session-state tracker for a local AI research assistant. "
    "After each turn, you update three working-state fields based on the previous "
    "state and the new turn (user instruction + assistant response).\n\n"
    "Output exactly three lines in this exact order, with no preamble, no explanation:\n"
    "FOCUS: <one short phrase summarising what the user is currently working on, "
    "or NONE to keep the previous focus unchanged>\n"
    "OPEN_LOOPS: <comma-separated short phrases for unresolved threads, "
    "or NONE if nothing is currently open>\n"
    "DECISIONS: <comma-separated short phrases for decisions made this turn, "
    "or NONE if no new decisions were made>\n\n"
    "Rules:\n"
    "- FOCUS: NONE means 'carry the previous focus forward unchanged, do not clear it'. "
    "Only emit a new focus phrase if this turn clearly shifts what the user is working on.\n"
    "- OPEN_LOOPS: NONE means the list is empty. Reconcile against the previous open "
    "loops: if a loop was closed or answered by this turn's response, remove it. "
    "Do not blindly append; edit the list to reflect what is still unresolved.\n"
    "- DECISIONS: NONE means no new decisions were made this turn. "
    "Only capture clear explicit choices or commitments, not tentative suggestions.\n"
    "- Never add preamble, explanation, or extra lines beyond the three."
)

# Per-bullet truncation ceiling for open_loops and recent_decisions entries.
# Mirrors memory_manager._MAX_BULLET_CHARS (80 chars) — same 20-token convention.
_MAX_BULLET_CHARS = 80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionSignal:
    """
    The output of deterministic signal detection.

    Fields
    ------
    episode_type :
        The inferred type from the trigger phrase.
    is_retraction :
        True → call retract(), not insert(). content will be empty.
    trigger_phrase :
        The phrase that matched, for logging.
    source :
        Always "explicit" for deterministic signals.
    confidence :
        Always 1.0 for deterministic signals.
    """
    episode_type:   str
    is_retraction:  bool
    trigger_phrase: str
    source:         str   = "explicit"
    confidence:     float = 1.0


@dataclass
class ExtractionResult:
    """
    The fully resolved episode candidate, ready for EpisodicMemoryWriter.

    None is returned (not an ExtractionResult) when:
      - No signal detected and implicit extraction returns NONE
      - Retraction was performed (the write is a side effect, not a result)
    """
    episode_type:    str
    subject:         str
    content:         str
    source:          str    # "explicit" | "model_extracted"
    confidence:      float  # 1.0 for explicit; 0.6–0.9 for model_extracted
    task_id:         str | None = None
    conversation_id: str | None = None
    project_context: str        = "general"


# ---------------------------------------------------------------------------
# 5.1 — Deterministic signal detection
# ---------------------------------------------------------------------------

def detect_explicit_signal(instruction: str) -> ExtractionSignal | None:
    """
    Scan the instruction for explicit memory command phrases.

    Retraction phrases are checked first. If a retraction phrase is found,
    an ExtractionSignal with is_retraction=True is returned — the caller
    must call EpisodicMemoryWriter.retract() rather than insert().

    Then explicit insert phrases are checked. The first match wins.

    Returns None if no signal is detected.

    Parameters
    ----------
    instruction :
        The raw user instruction string (not lowercased — this function
        lowercases internally).
    """
    lowered = instruction.lower()

    # Retraction check first (§2.5 retraction rule)
    for phrase in _RETRACTION_SIGNALS:
        if phrase in lowered:
            logger.debug(
                "detect_explicit_signal: retraction matched %r.", phrase
            )
            return ExtractionSignal(
                episode_type   = "preference",   # placeholder; retract() ignores type
                is_retraction  = True,
                trigger_phrase = phrase,
            )

    # Explicit insert signals
    for phrase, episode_type in _EXPLICIT_SIGNALS.items():
        if phrase in lowered:
            logger.debug(
                "detect_explicit_signal: explicit signal matched %r → %s.",
                phrase, episode_type,
            )
            return ExtractionSignal(
                episode_type   = episode_type,
                is_retraction  = False,
                trigger_phrase = phrase,
            )

    return None


# ---------------------------------------------------------------------------
# 5.3 — Confidence scoring
# ---------------------------------------------------------------------------

def score_model_extraction(raw_response: str) -> float:
    """
    Score a model-extracted episode content string.

    Heuristic rules (applied in order, first match wins):
      - Response is empty or "NONE"          → 0.0  (caller discards)
      - Contains strong hedging language      → 0.6
      - Response is very short (< 7 words)   → 0.7  (may lack specificity)
      - Response contains specific nouns/     → 0.9
        numbers/proper names
      - Default                               → 0.8

    Returns a float in [0.0, 0.9]. The caller is responsible for
    discarding results that score 0.0.

    Parameters
    ----------
    raw_response :
        The raw text returned by the model extraction call.
    """
    text = raw_response.strip()

    if not text or text.upper() == "NONE":
        return 0.0

    lowered = text.lower()

    # Strong hedging → low confidence
    hedging_phrases = (
        "might", "may", "could", "perhaps", "possibly",
        "i think", "i believe", "it seems", "unclear",
    )
    if any(h in lowered for h in hedging_phrases):
        return 0.6

    words = text.split()

    # Very short response → possibly too vague
    if len(words) < 7:
        return 0.7

    # Specificity signals → high confidence
    # Proper nouns (capitalised mid-sentence), numbers, version strings
    has_proper_noun = any(
        w[0].isupper() for w in words[1:]   # skip first word
    )
    has_number = bool(re.search(r"\d", text))
    if has_proper_noun or has_number:
        return 0.9

    return 0.8


# ---------------------------------------------------------------------------
# 5.2 — Model-based content extraction
# ---------------------------------------------------------------------------

def extract_content_from_instruction(
    instruction: str,
    episode_type: str,
    runtime: Any,
) -> tuple[str, float]:
    """
    Use a single bounded inference call to extract a clean one-sentence
    content string from the instruction.

    Used when a deterministic signal identifies the episode_type but the
    durable content string cannot be reliably inferred from surface keywords.

    Parameters
    ----------
    instruction :
        The raw user instruction.
    episode_type :
        The type already identified by deterministic detection.
    runtime :
        RuntimeClient. Used for a single infer() call.

    Returns
    -------
    (content, confidence) : tuple[str, float]
        content    : Extracted one-sentence string, or "" if NONE returned.
        confidence : Scored by score_model_extraction(). 0.0 means discard.
    """
    user_prompt = (
        f"A user said: {instruction!r}\n"
        f"Write one sentence about them starting with 'The user'. "
        f"If no durable fact is present, write: NONE"
    )

    try:
        raw = runtime.infer(
            system      = _EXTRACTION_SYSTEM,
            prompt      = user_prompt,
            max_tokens  = 200,
            temperature = 0.1,
        )
    except Exception as exc:
        logger.warning(
            "extract_content_from_instruction: inference failed (%s).", exc
        )
        return "", 0.0

    confidence = score_model_extraction(raw)
    if confidence == 0.0:
        logger.debug(
            "extract_content_from_instruction: model returned NONE/empty."
        )
        return "", 0.0

    content = raw.strip()
    logger.debug(
        "extract_content_from_instruction: extracted %r (confidence=%.2f).",
        content[:60], confidence,
    )
    return content, confidence


def _has_implicit_signal(instruction: str, response: str) -> bool:
    """
    Deterministic gate for implicit episodic extraction.

    Returns True if the turn pair contains signals that suggest a durable
    personal fact may be present. Only when True is the model extraction
    call made.

    Checks (any match → True):
    - Instruction contains first-person factual statements about environment,
      tools, hardware, or preferences
    - Instruction contains project-specific proper nouns alongside context words
    - Response contains explicit acknowledgement phrases suggesting a fact
      was registered
    """
    lowered_inst = instruction.lower()
    lowered_resp = response.lower()

    # First-person factual signals in the instruction
    _IMPLICIT_INST_SIGNALS: frozenset[str] = frozenset({
        "i'm using", "i am using", "i use",
        "i'm running", "i am running",
        "i'm building", "i am building",
        "i'm working", "i am working",
        "i prefer", "i like", "i always",
        "my setup", "my environment", "my project",
        "my workflow", "my preference",
        "i have a", "i've been", "i have been",
        "we use", "we're using", "we decided",
        "the project is", "the stack is",
    })

    # Acknowledgement signals in the response — suggest the agent
    # recognised a durable fact worth storing
    _IMPLICIT_RESP_SIGNALS: frozenset[str] = frozenset({
        "noted", "i'll remember", "i will remember",
        "recorded", "stored", "got it",
        "understood", "i've noted", "i have noted",
        "keep that in mind", "i'll keep",
    })

    if any(sig in lowered_inst for sig in _IMPLICIT_INST_SIGNALS):
        return True
    if any(sig in lowered_resp for sig in _IMPLICIT_RESP_SIGNALS):
        return True
    return False


def extract_implicit_episode(
    instruction: str,
    response:    str,
    runtime:     Any,
) -> tuple[str, str, float] | None:
    """
    Attempt to extract an implicit episode from a full conversation turn.

    Used in the post-response hook to catch durable facts the user revealed
    without an explicit memory command.

    Parameters
    ----------
    instruction :
        The user's instruction for this turn.
    response :
        The agent's response for this turn.
    runtime :
        RuntimeClient. Used for a single infer() call.

    Returns
    -------
    (episode_type, content, confidence) or None
        None when the model returns NONE or inference fails.
        episode_type is inferred from content keywords; defaults to
        "project_fact" when no stronger signal is present.
    """
    # Deterministic gate — skip model call if no implicit signal detected
    if not _has_implicit_signal(instruction, response):
        logger.debug(
            "extract_implicit_episode: no implicit signal detected — skipping."
        )
        return None

    # Gate passed — run targeted extraction on the instruction only.
    # We already know a fact is present; the model only needs to extract it.
    user_prompt = (
        f"A user said: {instruction!r}\n"
        f"Write one sentence about them starting with 'The user'. "
        f"If no durable fact is present, write: NONE"
    )

    try:
        raw = runtime.infer(
            system      = _EXTRACTION_SYSTEM,
            prompt      = user_prompt,
            max_tokens  = 200,
            temperature = 0.1,
        )
    except Exception as exc:
        logger.warning(
            "extract_implicit_episode: inference failed (%s).", exc
        )
        return None

    confidence = score_model_extraction(raw)
    if confidence == 0.0:
        logger.debug(
            "extract_implicit_episode: model returned NONE/empty."
        )
        return None

    content = raw.strip()
    episode_type = _infer_type_from_content(content)

    logger.debug(
        "extract_implicit_episode: extracted %r "
        "(type=%s, confidence=%.2f).",
        content[:60], episode_type, confidence,
    )
    return episode_type, content, confidence


def _infer_type_from_content(content: str) -> str:
    """
    Heuristically infer episode_type from extracted content string.

    Checked in priority order. First match wins. Defaults to
    "project_fact" when no pattern matches.
    """
    lowered = content.lower()
    if any(w in lowered for w in ("prefer", "like", "want", "always use")):
        return "preference"
    if any(w in lowered for w in ("wrong", "incorrect", "should be", "actually")):
        return "correction"
    if any(w in lowered for w in ("decided", "decision", "committed", "chosen")):
        return "decision"
    if any(w in lowered for w in ("every time", "always", "workflow", "process")):
        return "workflow"
    if any(w in lowered for w in ("called", "named", "refers to", "known as")):
        return "naming_convention"
    if any(w in lowered for w in ("completed", "done", "finished", "milestone")):
        return "task_completion"
    return "project_fact"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def process_explicit_signal(
    instruction:     str,
    runtime:         Any,
    db_path:         Any,   # Path
    task_id:         str | None = None,
    project_context: str        = "general",
) -> ExtractionResult | None:
    """
    Full explicit extraction pipeline: detect → extract → write → return.

    1. Run detect_explicit_signal(). If None → return None.
    2. If retraction: call EpisodicMemoryWriter.retract() and return None.
    3. Call extract_content_from_instruction() for the content string.
    4. If confidence == 0.0: return None (model said NONE).
    5. Derive subject from instruction (first 80 chars, stripped).
    6. Write to DB via EpisodicMemoryWriter.insert().
    7. Return ExtractionResult.

    Parameters
    ----------
    instruction :
        Raw user instruction.
    runtime :
        RuntimeClient for the extraction inference call.
    db_path :
        Path to lora_memory.db. Must match the MemoryManager db_path.
    task_id :
        Optional task_id for provenance.
    project_context :
        Project scope for retrieval. Defaults to "general".

    Returns
    -------
    ExtractionResult or None
        None when: no signal, retraction performed, or model returned NONE.
    """
    signal = detect_explicit_signal(instruction)
    if signal is None:
        return None

    writer = EpisodicMemoryWriter(db_path=db_path)

    # Retraction path
    # Use the model to extract the subject being retracted so that
    # writer.retract() (which does exact subject matching) can find the
    # previously stored record. Fall back to the instruction text if the
    # model returns NONE or inference fails.
    if signal.is_retraction:
        extracted_subject, _ = extract_content_from_instruction(
            instruction  = instruction,
            episode_type = "retraction",
            runtime      = runtime,
        )
        subject = (extracted_subject or instruction.strip())[:80]
        count = writer.retract(subject=subject, episode_type="preference")
        logger.info(
            "process_explicit_signal: retraction — %d record(s) retracted "
            "for subject=%r.", count, subject,
        )
        return None

    # Content extraction
    content, confidence = extract_content_from_instruction(
        instruction  = instruction,
        episode_type = signal.episode_type,
        runtime      = runtime,
    )
    if not content:
        return None

    # Derive subject from the already-normalized content string (same approach
    # as process_implicit_extraction). content is already in "The user..."
    # form from extract_content_from_instruction — no second model call needed.
    # Falls back to the raw instruction if content is unexpectedly empty.
    subject = (content[:80] if content else instruction.strip()[:80])

    writer.insert(
        episode_type    = signal.episode_type,
        subject         = subject,
        content         = content,
        source          = "explicit",
        confidence      = 1.0,   # explicit signal always 1.0 (§2.3)
        task_id         = task_id,
        project_context = project_context,
    )

    logger.info(
        "process_explicit_signal: wrote %s episode (confidence=1.0) "
        "subject=%r.", signal.episode_type, subject,
    )

    return ExtractionResult(
        episode_type    = signal.episode_type,
        subject         = subject,
        content         = content,
        source          = "explicit",
        confidence      = 1.0,
        task_id         = task_id,
        project_context = project_context,
    )


def process_implicit_extraction(
    instruction:     str,
    response:        str,
    runtime:         Any,
    db_path:         Any,   # Path
    task_id:         str | None = None,
    project_context: str        = "general",
) -> ExtractionResult | None:
    """
    Full implicit extraction pipeline: extract from turn pair → write → return.

    1. Call extract_implicit_episode(). If None → return None.
    2. If confidence < 0.6 → discard (below minimum threshold).
    3. Derive subject from content (first 80 chars).
    4. Write to DB via EpisodicMemoryWriter.insert() with source="model_extracted".
    5. Return ExtractionResult.

    Parameters
    ----------
    instruction :
        The user's instruction for this turn.
    response :
        The agent's response for this turn.
    runtime :
        RuntimeClient for the extraction inference call.
    db_path :
        Path to lora_memory.db.
    task_id :
        Optional task_id for provenance.
    project_context :
        Project scope. Defaults to "general".

    Returns
    -------
    ExtractionResult or None
    """
    result = extract_implicit_episode(instruction, response, runtime)
    if result is None:
        return None

    episode_type, content, confidence = result

    if confidence < 0.6:
        logger.debug(
            "process_implicit_extraction: discarding low-confidence "
            "extraction (confidence=%.2f).", confidence,
        )
        return None

    subject = content[:80]

    writer = EpisodicMemoryWriter(db_path=db_path)
    writer.insert(
        episode_type    = episode_type,
        subject         = subject,
        content         = content,
        source          = "model_extracted",
        confidence      = confidence,
        task_id         = task_id,
        project_context = project_context,
    )

    logger.info(
        "process_implicit_extraction: wrote %s episode "
        "(confidence=%.2f) subject=%r.",
        episode_type, confidence, subject,
    )

    return ExtractionResult(
        episode_type    = episode_type,
        subject         = subject,
        content         = content,
        source          = "model_extracted",
        confidence      = confidence,
        task_id         = task_id,
        project_context = project_context,
    )


def extract_working_state_update(
    instruction:    str,
    response:       str,
    previous_state: "WorkingStateRecord | None",
    runtime:        Any,
) -> "tuple[str | None, list[str], list[str]] | None":
    """
    Extract updated Slot 6A Tier 2 fields from a conversation turn.

    Calls the model with the previous working state and the new turn, then
    parses the mandatory three-line structured response.

    Parameters
    ----------
    instruction :
        The user's instruction for this turn.
    response :
        The agent's response for this turn.
    previous_state :
        The stored WorkingStateRecord for this session, or None on the
        first turn. Used to provide reconciliation context to the model.
    runtime :
        RuntimeClient. Used for a single infer() call.

    Returns
    -------
    (current_focus, open_loops, recent_decisions) or None
        current_focus    : New focus phrase, or None meaning 'keep previous'.
        open_loops       : List of open-loop strings (empty list = none open).
        recent_decisions : List of decision strings (empty list = none).
        Returns None when the model response fails to parse — caller must
        keep the previous state unchanged (fail-closed contract).
    """
    # Serialize previous state for the model's reconciliation context.
    if previous_state is not None:
        prev_focus = previous_state.current_focus or "NONE"
        prev_loops = ", ".join(previous_state.open_loops) or "NONE"
        prev_decisions = ", ".join(previous_state.recent_decisions) or "NONE"
    else:
        prev_focus = "NONE"
        prev_loops = "NONE"
        prev_decisions = "NONE"

    user_prompt = (
        f"Previous working state:\n"
        f"current_focus: {prev_focus}\n"
        f"open_loops: {prev_loops}\n"
        f"recent_decisions: {prev_decisions}\n\n"
        f"New turn:\n"
        f"User: {instruction}\n"
        f"Assistant: {response}"
    )

    try:
        raw = runtime.infer(
            system      = _WORKING_STATE_UPDATE_SYSTEM,
            prompt      = user_prompt,
            max_tokens  = 200,
            temperature = 0.0,
        )
    except Exception as exc:
        logger.warning(
            "extract_working_state_update: inference failed (%s).", exc
        )
        return None

    # Parse the three-line structured response. All three labels must be present.
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]

    focus_val      = None
    open_loops_val = None
    decisions_val  = None

    for line in lines:
        upper = line.upper()
        if upper.startswith("FOCUS:"):
            focus_val = line[len("FOCUS:"):].strip()
        elif upper.startswith("OPEN_LOOPS:"):
            open_loops_val = line[len("OPEN_LOOPS:"):].strip()
        elif upper.startswith("DECISIONS:"):
            decisions_val = line[len("DECISIONS:"):].strip()

    if any(v is None for v in (focus_val, open_loops_val, decisions_val)):
        logger.warning(
            "extract_working_state_update: parse failed — "
            "missing label(s) in response %r.", raw[:120],
        )
        return None

    # Resolve FOCUS — NONE means "keep previous value"
    current_focus: str | None = (
        None if focus_val.upper() == "NONE" else focus_val
    )

    # Parse OPEN_LOOPS — NONE means empty list
    if open_loops_val.upper() == "NONE":
        open_loops: list[str] = []
    else:
        open_loops = [
            s[:_MAX_BULLET_CHARS] for s in
            [item.strip() for item in open_loops_val.split(",")]
            if s.strip()
        ]

    # Parse DECISIONS — NONE means empty list
    if decisions_val.upper() == "NONE":
        recent_decisions: list[str] = []
    else:
        recent_decisions = [
            s[:_MAX_BULLET_CHARS] for s in
            [item.strip() for item in decisions_val.split(",")]
            if s.strip()
        ]

    logger.debug(
        "extract_working_state_update: parsed — "
        "focus=%r loops=%d decisions=%d.",
        current_focus, len(open_loops), len(recent_decisions),
    )
    return current_focus, open_loops, recent_decisions


def process_working_state_update(
    instruction: str,
    response:    str,
    mem_key:     str,
    runtime:     Any,
    db_path:     Any,   # Path
) -> "WorkingStateRecord | None":
    """
    Full working-state update pipeline: read → extract → resolve → write → return.

    1. Read previous state via WorkingStateStore.get(mem_key).
    2. Call extract_working_state_update(). If None → log and return
       previous_state unchanged without calling upsert() (fail-closed).
    3. Resolve FOCUS: None from extraction means carry forward
       previous_state.current_focus (or None when no previous state).
    4. Call WorkingStateStore.upsert() with resolved fields.
    5. Return the resulting WorkingStateRecord.

    Never raises — wraps all logic in try/except; returns previous_state
    (or None if no previous state) on any exception.

    Parameters
    ----------
    instruction :
        The user's instruction for this turn.
    response :
        The agent's response for this turn.
    mem_key :
        Session key (same key used by conversation_log / get_context_window).
    runtime :
        RuntimeClient for the extraction inference call.
    db_path :
        Path to lora_memory.db.

    Returns
    -------
    WorkingStateRecord | None
        The newly stored record, or the previous record unchanged on failure.
    """
    store = WorkingStateStore(db_path=db_path)

    try:
        previous_state = store.get(mem_key)

        result = extract_working_state_update(
            instruction    = instruction,
            response       = response,
            previous_state = previous_state,
            runtime        = runtime,
        )

        if result is None:
            logger.warning(
                "process_working_state_update: extraction returned None "
                "for mem_key=%r — keeping previous state.", mem_key,
            )
            return previous_state

        extracted_focus, open_loops, recent_decisions = result

        # Resolve FOCUS: None means "keep previous value"
        if extracted_focus is None:
            current_focus = (
                previous_state.current_focus if previous_state is not None else None
            )
        else:
            current_focus = extracted_focus

        store.upsert(
            mem_key          = mem_key,
            current_focus    = current_focus,
            open_loops       = open_loops,
            recent_decisions = recent_decisions,
        )

        logger.info(
            "process_working_state_update: working state written "
            "for mem_key=%r focus=%r loops=%d.",
            mem_key, current_focus, len(open_loops),
        )

        return store.get(mem_key)

    except Exception as exc:
        logger.warning(
            "process_working_state_update: failed for mem_key=%r (%s) — "
            "continuing.", mem_key, exc,
        )
        try:
            return store.get(mem_key)
        except Exception:
            return None
