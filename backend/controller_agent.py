"""
LORA — Controller Agent
=======================
The central coordinator of the multi-agent reasoning system.

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no UI imports.
- FastAPI calls into this module; this module never calls FastAPI.
- Agents are called directly as Python objects via their AgentInterface.
- All model inference is requested through the RuntimeClient abstraction.
- All public methods are synchronous; async variants can be added later
  by wrapping the runtime calls in asyncio.to_thread if needed.

Memory
------
ControllerAgent accepts an optional ``memory_manager`` argument.  When
supplied (the normal production case), it is the SQLite-backed MemoryManager
from memory_manager.py and all conversation-log writes are persisted across
requests.  When absent (tests, standalone runs), a lightweight in-process
_EphemeralMemory shim is used instead so nothing else in the codebase changes.

Layer placement
---------------
  Svelte UI  →  FastAPI  →  ControllerAgent  →  Sub-agents / Runtime
                                             →  MemoryManager (SQLite)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Avoid a hard circular import at runtime; memory_manager imports nothing
    # from controller_agent, but being explicit about the direction keeps
    # the dependency graph clean.
    from memory_manager import MemoryManager

from planner import Planner as _RulePlanner, RoutingPlan
from pathlib import Path
from prompt_builder import PromptBuilder, Turn, EpisodeBullet, RagSource, UserProfileFact, GraphQueryResult, GraphLinkEntry, WorkingMemoryState
from memory_manager import (
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
    format_episodic_summary,
    EpisodeRecord,
    _cosine_similarity,
)
from episodic_extractor import (
    process_explicit_signal,
    process_implicit_extraction,
    process_working_state_update,
)
from mcp_tool_dispatcher import MCPToolDispatcher
from prompt_builder import ToolResult as _ToolResult
from wiki_doc import load_wiki_doc, parse_wiki_doc
import session_files as _session_files

logger = logging.getLogger(__name__)

_PROMPT_BUILDER = PromptBuilder()


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"


@dataclass
class Task:
    """The canonical task object passed from FastAPI into the Controller."""
    task_id:     str
    instruction: str
    context:     dict[str, Any] = field(default_factory=dict)
    metadata:    dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Validate and construct a Task from a raw FastAPI payload."""
        required = {"instruction"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Task payload missing required fields: {missing}")
        return cls(
            task_id     = data.get("task_id", str(uuid.uuid4())),
            instruction = data["instruction"],
            context     = data.get("context", {}),
            metadata    = data.get("metadata", {}),
        )


@dataclass
class SubTask:
    """A unit of work delegated to a single sub-agent."""
    subtask_id:  str
    agent_name:  str
    instruction: str
    context:     dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Structured output returned by a sub-agent."""
    subtask_id: str
    agent_name: str
    status:     TaskStatus
    output:     dict[str, Any]
    error:      str | None = None


@dataclass
class ControllerResult:
    """The final structured payload the Controller returns to FastAPI."""
    task_id:    str
    status:     TaskStatus
    answer:     str
    sources:    list[dict[str, Any]] = field(default_factory=list)
    metadata:   dict[str, Any]       = field(default_factory=dict)
    error:      str | None           = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id":  self.task_id,
            "status":   self.status.value,
            "answer":   self.answer,
            "sources":  self.sources,
            "metadata": self.metadata,
            "error":    self.error,
        }


# ---------------------------------------------------------------------------
# Agent interface protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentInterface(Protocol):
    """
    Every sub-agent must satisfy this protocol.
    Agents are pure Python — they receive a SubTask and return an AgentResult.
    They must not know about HTTP, FastAPI, or the UI.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this agent, e.g. 'wiki_agent'."""
        ...

    def run(self, subtask: SubTask) -> AgentResult:
        """Execute the subtask and return a structured result."""
        ...

    def can_handle(self, instruction: str) -> bool:
        """
        Return True if this agent is capable of handling the given instruction.
        Used by the planner to route tasks without hard-coded conditionals.
        """
        ...


# ---------------------------------------------------------------------------
# Runtime client interface
# ---------------------------------------------------------------------------

@runtime_checkable
class RuntimeClient(Protocol):
    """
    Abstraction over the Local Runtime Layer (Azure AI Foundry / oMLX).
    The Controller and sub-agents call this — never a model API directly.
    """

    def infer(
        self,
        prompt:      str,
        system:      str  = "",
        max_tokens:  int  = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Return the model's text completion for the given prompt."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for the given text."""
        ...


# ---------------------------------------------------------------------------
# Memory key helper
# ---------------------------------------------------------------------------

def _memory_key(task: Task) -> str:
    """
    The key used to group conversation_log entries for working-memory
    continuity. Prefers a client-supplied session_id (set once per page
    load by the frontend) so multiple turns in one conversation share
    working memory. Falls back to task.task_id for any caller that
    doesn't supply session_id (e.g. the one-shot ingest path), matching
    today's behavior for that path exactly.
    """
    return task.context.get("session_id") or task.task_id


# ---------------------------------------------------------------------------
# Ephemeral memory shim
# ---------------------------------------------------------------------------
# Used only when no MemoryManager is injected (tests, standalone scripts).
# The public API intentionally mirrors MemoryManager so call sites are
# identical regardless of which object is in use.

class _EphemeralMemory:
    """
    In-process memory store — zero persistence, zero dependencies.

    Accepts the same keyword arguments as MemoryManager.add() / .add_agent_result()
    so ControllerAgent._execute() and Synthesizer.synthesize() call both
    objects identically.  task_id is accepted but ignored — this shim holds
    only one flat list of entries.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def add(
        self,
        role:     str,
        content:  str,
        metadata: dict[str, Any] | None = None,
        task_id:  str = "global",        # accepted, ignored
    ) -> None:
        self._entries.append({
            "role":     role,
            "content":  content,
            "metadata": metadata or {},
        })
        # Naive eviction
        if len(self._entries) > 200:
            self._entries = self._entries[-200:]

    def add_agent_result(
        self,
        result:  AgentResult,
        task_id: str = "global",         # accepted, ignored
    ) -> None:
        self.add(
            role     = "agent",
            content  = str(result.output),
            metadata = {"agent": result.agent_name, "subtask_id": result.subtask_id},
            task_id  = task_id,
        )

    def get_context_window(
        self,
        task_id: str = "global",
        limit:   int = 50,
    ) -> list[dict[str, Any]]:
        return list(self._entries[-limit:])

    def format_for_prompt(
        self,
        task_id: str = "global",
        limit:   int = 50,
    ) -> str:
        entries = self.get_context_window(task_id=task_id, limit=limit)
        return "\n".join(f"[{e['role'].upper()}] {e['content']}" for e in entries)

    def clear(self, task_id: str | None = None) -> None:
        self._entries.clear()


# Union type used for internal type hints — both objects satisfy this duck type.
_AnyMemory = Any   # MemoryManager | _EphemeralMemory


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Decomposes an instruction into an ordered list of SubTasks.

    Strategy (initial):  ask the runtime to produce a JSON plan, then
    validate each step against the registered agent roster.  Falls back
    to a single-agent plan if decomposition fails.
    """

    def __init__(self, runtime: RuntimeClient) -> None:
        self._runtime = runtime

    def plan(
        self,
        task:   Task,
        agents: list[AgentInterface],
    ) -> list[SubTask]:
        """Return an ordered list of SubTasks for the given task."""
        agent_names = [a.name for a in agents]

        raw_plan = self._runtime.infer(
            system  = self._system_prompt(agent_names),
            prompt  = self._user_prompt(task),
        )

        subtasks = self._parse_plan(task.task_id, raw_plan, agents, task.context)
        if not subtasks:
            logger.warning("Planner produced no subtasks; falling back to single-agent plan.")
            subtasks = self._fallback_plan(task, agents)

        logger.info("Planner produced %d subtask(s) for task %s.", len(subtasks), task.task_id)
        return subtasks

    # -- internal helpers --

    @staticmethod
    def _system_prompt(agent_names: list[str]) -> str:
        return (
            "You are a task planner inside a local multi-agent research system.\n"
            "Given a user instruction, output a JSON array of subtasks.\n"
            "Each subtask must have keys: \"agent\" and \"instruction\".\n\n"
            "Available agents and when to use them:\n"
            "- conversational_agent : use for ALL questions, research, analysis, chitchat,\n"
            "                         and anything the user wants to know or discuss.\n"
            "                         This is the default agent for every non-ingest task.\n"
            "- wiki_agent           : use ONLY when the instruction explicitly mentions\n"
            "                         ingesting, uploading, or adding a specific file to\n"
            "                         the wiki (must include a file path or file name).\n\n"
            "Rules:\n"
            "1. For all questions and general instructions -> use conversational_agent ONLY.\n"
            "2. For wiki ingestion tasks that include a file -> use wiki_agent ONLY.\n"
            "3. Never schedule both agents for the same task. Each task uses exactly one agent.\n"
            "4. Output ONLY the JSON array. No prose, no markdown fences, no commentary.\n\n"
            "Example outputs:\n"
            "[{\"agent\": \"conversational_agent\", \"instruction\": \"What do we know about X?\"}]\n"
            "[{\"agent\": \"wiki_agent\", \"instruction\": \"Ingest /path/to/file.md\"}]\n"
        )

    @staticmethod
    def _user_prompt(task: Task) -> str:
        return f"User instruction: {task.instruction}\nContext: {task.context}"

    @staticmethod
    def _parse_plan(
        task_id:      str,
        raw:          str,
        agents:       list[AgentInterface],
        task_context: dict[str, Any] = {},
    ) -> list[SubTask]:
        import json, re
        agent_map = {a.name: a for a in agents}
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            steps = json.loads(match.group())
            subtasks = []
            for i, step in enumerate(steps):
                agent_name = step.get("agent", "")
                if agent_name not in agent_map:
                    logger.warning("Planner referenced unknown agent '%s'; skipping.", agent_name)
                    continue
                # Merge task_context (raw_path, wiki_dir, schema_path, etc.) with
                # any context the model chose to emit.  Model-supplied keys win so
                # the Planner can in principle override defaults, but in practice
                # the model never emits a context key — task_context carries everything.
                merged_context = {**task_context, **step.get("context", {})}
                subtasks.append(SubTask(
                    subtask_id  = f"{task_id}-{i}",
                    agent_name  = agent_name,
                    instruction = step.get("instruction", ""),
                    context     = merged_context,
                ))
            return subtasks
        except Exception as exc:
            logger.error("Planner failed to parse plan: %s", exc)
            return []

    @staticmethod
    def _fallback_plan(task: Task, agents: list[AgentInterface]) -> list[SubTask]:
        """Route to conversational_agent by preference, then first can_handle() match."""
        for agent in agents:
            if agent.name == "conversational_agent":
                return [SubTask(
                    subtask_id  = f"{task.task_id}-0",
                    agent_name  = agent.name,
                    instruction = task.instruction,
                    context     = task.context,
                )]
        for agent in agents:
            if agent.can_handle(task.instruction):
                return [SubTask(
                    subtask_id  = f"{task.task_id}-0",
                    agent_name  = agent.name,
                    instruction = task.instruction,
                    context     = task.context,
                )]
        if agents:
            return [SubTask(
                subtask_id  = f"{task.task_id}-0",
                agent_name  = agents[0].name,
                instruction = task.instruction,
                context     = task.context,
            )]
        return []


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """
    Combines the outputs of one or more AgentResults into a single coherent
    answer, then validates schema compliance before returning.
    """

    def __init__(self, runtime: RuntimeClient) -> None:
        self._runtime = runtime

    def synthesize(
        self,
        task:    Task,
        results: list[AgentResult],
        memory:  _AnyMemory,
    ) -> ControllerResult:
        """Produce the final ControllerResult from all collected AgentResults."""

        if not results:
            return ControllerResult(
                task_id = task.task_id,
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = "No agent results to synthesize.",
            )

        failed = [r for r in results if r.status == TaskStatus.FAILED]
        if len(failed) == len(results):
            errors = "; ".join(r.error or "unknown" for r in failed)
            return ControllerResult(
                task_id = task.task_id,
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = f"All sub-agents failed: {errors}",
            )

        # Both MemoryManager and _EphemeralMemory accept task_id as a kwarg.
        context_str = memory.format_for_prompt(task_id=_memory_key(task))
        results_str = self._format_results(results)

        answer = self._runtime.infer(
            system = (
                "You are the synthesis step of a local multi-agent research system. "
                "Given sub-agent outputs, produce a clear, accurate, well-structured answer "
                "to the original user instruction. Do not hallucinate. Cite sources where present."
            ),
            prompt = (
                f"Original instruction: {task.instruction}\n\n"
                f"Context window:\n{context_str}\n\n"
                f"Sub-agent outputs:\n{results_str}\n\n"
                "Produce the final answer."
            ),
        )

        sources = self._collect_sources(results)

        return ControllerResult(
            task_id  = task.task_id,
            status   = TaskStatus.COMPLETE,
            answer   = answer,
            sources  = sources,
            metadata = {"subtask_count": len(results)},
        )

    # -- helpers --

    @staticmethod
    def _format_results(results: list[AgentResult]) -> str:
        lines = []
        for r in results:
            lines.append(f"[{r.agent_name}] {r.output}")
        return "\n".join(lines)

    @staticmethod
    def _collect_sources(results: list[AgentResult]) -> list[dict[str, Any]]:
        sources = []
        for r in results:
            if isinstance(r.output, dict) and "sources" in r.output:
                sources.extend(r.output["sources"])
        return sources


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Classifies a user instruction into one of two buckets before routing.

    Buckets
    -------
    query   — anything the user wants to know, understand, or discuss.
              Routes to ConversationalAgent (corpus RAG + single inference).
    ingest  — explicit requests to add/upload a specific file to the wiki.
              Routes to WikiAgent via the Planner.

    One fast model call with a constrained output format.  Falls back to
    "query" on any parse failure so the pipeline always continues.

    Note: the former "conversational" and "research" buckets have been merged
    into "query".  ConversationalAgent now handles both chitchat and deep
    factual questions via corpus-grounded inference.
    """

    _SYSTEM = (
        "You are an intent classifier for a local research agent system. "
        "Classify the user instruction into exactly one of these categories:\n"
        "  query  — questions, research, analysis, chitchat, "
        "anything the user wants to know or discuss\n"
        "  ingest — explicit requests to add, upload, or ingest "
        "a specific file into the wiki\n\n"
        "Output ONLY one word: query or ingest. Nothing else."
    )

    def __init__(self, runtime: RuntimeClient) -> None:
        self._runtime = runtime

    def classify(self, instruction: str) -> str:
        """Return 'query' or 'ingest'.  Never raises."""
        try:
            raw = self._runtime.infer(
                system      = self._SYSTEM,
                prompt      = f"Instruction: {instruction}",
                max_tokens  = 8,
                temperature = 0.0,
            ).strip().lower()

            first_word = raw.split()[0].rstrip(".,!") if raw else ""
            if first_word in {"query", "ingest"}:
                logger.info(
                    "IntentClassifier: '%s' → %s", instruction[:60], first_word
                )
                return first_word
        except Exception as exc:
            logger.warning(
                "IntentClassifier failed (%s); defaulting to query.", exc
            )

        logger.info("IntentClassifier: unrecognised output, defaulting to query.")
        return "query"


# ---------------------------------------------------------------------------
# ControllerAgent
# ---------------------------------------------------------------------------

class ControllerAgent:
    """
    The executive function of the Localist Framework.

    FastAPI instantiates one ControllerAgent at startup and calls
    ``handle_task(task_dict)`` for every incoming request.

    Parameters
    ----------
    runtime :
        RuntimeClient used for planning, synthesis, classification, and
        conversational turns.
    agents :
        Sub-agents to register.  Each must satisfy AgentInterface.
    memory_manager :
        Optional SQLite-backed MemoryManager from memory_manager.py.
        When supplied, all conversation-log writes persist across requests
        and process restarts.  When absent, an _EphemeralMemory shim is
        used — identical API, zero persistence.

    This class does not know about HTTP, request/response objects,
    streaming, or the Svelte UI.  It is a pure Python reasoning coordinator.
    """

    def __init__(
        self,
        runtime:        RuntimeClient,
        agents:         list[AgentInterface],
        memory_manager: "MemoryManager | None" = None,
        embed_fn:       Callable[[str], list[float]] | None = None,
    ) -> None:
        self._runtime        = runtime
        self._agents         = {a.name: a for a in agents}
        self._planner        = _RulePlanner(runtime=runtime, memory_manager=memory_manager, embed_fn=embed_fn)
        self._synthesizer    = Synthesizer(runtime)
        # IntentClassifier retired — routing is now handled by _RulePlanner
        self._memory_manager = memory_manager   # None → use ephemeral shim per request
        self._persona_cache: str | None = None

        # User profile cache — loaded once per process from
        # wiki/users/michael.md, split into (line, embedding) pairs.
        # Empty list until _load_user_profile() succeeds.
        self._profile_lines:      list[str]        = []
        self._profile_embeddings: list[list[float]] = []

        if not agents:
            logger.warning("ControllerAgent initialized with no sub-agents.")

        if memory_manager is not None:
            logger.info(
                "ControllerAgent: using persistent MemoryManager (%r).", memory_manager
            )
        else:
            logger.info(
                "ControllerAgent: no MemoryManager supplied — using ephemeral memory per request."
            )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_persona(self) -> str | None:
        """
        Load the LORA persona document from the wiki corpus.
        Result is cached in self._persona_cache after first successful load.
        Returns None when MemoryManager is absent or corpus has no persona doc.
        """
        if self._persona_cache is not None:
            return self._persona_cache
        if self._memory_manager is None:
            return None
        try:
            docs = self._memory_manager.query_corpus(
                "LORA persona identity research assistant",
                max_results    = 3,
                use_embeddings = True,
            )
            # Filter to the persona document by filename — never load a
            # different wiki doc into the persona slot.
            persona_doc = next(
                (d for d in docs if "lora-persona" in str(d.path)), None
            )
            if persona_doc is not None:
                parsed = parse_wiki_doc(persona_doc.content)
                self._persona_cache = parsed.body[:2000]
                logger.debug(
                    "_load_persona: persona loaded and cached — "
                    "path=%s  chars=%d",
                    persona_doc.path,
                    len(self._persona_cache),
                )
            else:
                logger.warning(
                    "_load_persona: lora-persona not found in top-3 "
                    "corpus results — proceeding without persona."
                )
        except Exception as exc:
            logger.warning(
                "_load_persona: persona fetch failed (%s) — "
                "proceeding without persona.", exc,
            )
        return self._persona_cache

    def _load_user_profile(self) -> None:
        """
        Load wiki/users/michael.md and embed each fact line.

        Splits the profile into non-empty, non-header lines (strips lines
        starting with '#' and blank lines). Each line is embedded using
        the runtime's embed() method and stored in parallel lists:
          self._profile_lines      — raw fact strings
          self._profile_embeddings — corresponding embedding vectors

        Called once during the first request after startup (lazy init).
        Results are cached for the process lifetime — the profile is a
        session constant. Re-load requires a process restart.

        Silently no-ops when:
          - Profile file does not exist
          - Runtime has no embed() method
          - Any individual line fails to embed (that line is skipped)
        """
        if self._profile_lines:
            return   # already loaded

        profile_path = Path(__file__).parent / "wiki" / "users" / "michael.md"
        if not profile_path.exists():
            logger.warning("_load_user_profile: %s not found — skipping.", profile_path)
            return

        try:
            parsed = load_wiki_doc(profile_path)
        except Exception as exc:
            logger.warning("_load_user_profile: read failed (%s).", exc)
            return

        # Extract fact lines — skip headers (##) and blank lines
        fact_lines = [
            line.lstrip("- ").strip()
            for line in parsed.body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not fact_lines:
            logger.warning("_load_user_profile: no fact lines found in profile.")
            return

        # Embed each line; skip lines that fail
        loaded = 0
        for line in fact_lines:
            try:
                vec = self._embed(line)
                self._profile_lines.append(line)
                self._profile_embeddings.append(vec)
                loaded += 1
            except Exception as exc:
                logger.debug(
                    "_load_user_profile: embed failed for line %r (%s) — skipping.",
                    line[:40], exc,
                )

        logger.info(
            "_load_user_profile: loaded %d/%d fact lines from %s.",
            loaded, len(fact_lines), profile_path,
        )

    def _embed(self, text: str) -> list[float]:
        """
        Embed text using the MemoryManager's embed function.
        Raises RuntimeError if no embed function is available.
        """
        fn = getattr(self._memory_manager, "_embed_fn", None)
        if fn is None:
            raise RuntimeError("No embed function available on MemoryManager.")
        return fn(text)

    def _score_profile_facts(
        self,
        instruction_embedding: list[float],
        top_n:     int   = 5,
        threshold: float = 0.45,
    ) -> list[UserProfileFact]:
        """
        Score all cached profile fact lines against the instruction embedding.

        Returns the top_n lines whose cosine similarity exceeds threshold,
        ordered by score descending. Returns [] when:
          - Profile is not loaded (_profile_lines is empty)
          - instruction_embedding is empty
          - No lines exceed threshold

        Parameters
        ----------
        instruction_embedding :
            Dense embedding of the current instruction. Caller is
            responsible for generating this.
        top_n :
            Maximum facts to return. Default 5.
        threshold :
            Minimum cosine similarity. Default 0.45 — lower than the
            0.55 RAG threshold because profile facts are short lines
            whose embeddings are naturally less similar to full questions.
        """
        if not self._profile_lines or not instruction_embedding:
            return []

        scored: list[tuple[float, str]] = []
        for line, vec in zip(self._profile_lines, self._profile_embeddings):
            try:
                sim = _cosine_similarity(instruction_embedding, vec)
                if sim >= threshold:
                    scored.append((sim, line))
            except Exception:
                continue

        scored.sort(reverse=True)
        return [UserProfileFact(content=line) for _, line in scored[:top_n]]

    # -----------------------------------------------------------------------
    # Public API (called by FastAPI)
    # -----------------------------------------------------------------------

    def handle_task(self, task_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Entry point called by FastAPI.

        Parameters
        ----------
        task_dict:
            Raw payload from FastAPI, containing at minimum ``instruction``.

        Returns
        -------
        dict
            A serialized ControllerResult suitable for JSON response.
        """
        try:
            task = Task.from_dict(task_dict)
        except ValueError as exc:
            return ControllerResult(
                task_id = task_dict.get("task_id", "unknown"),
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = f"Invalid task payload: {exc}",
            ).to_dict()

        logger.info("Controller received task %s: '%s'", task.task_id, task.instruction[:80])

        # Use the persistent manager if available; otherwise a fresh ephemeral shim.
        memory: _AnyMemory = self._memory_manager or _EphemeralMemory()
        mem_key = _memory_key(task)
        memory.add(
            "user",
            task.instruction,
            metadata={"task_id": task.task_id},
            task_id=mem_key,
        )

        try:
            result = self._execute(task, memory)
        except Exception as exc:
            logger.exception("Unhandled exception in controller for task %s.", task.task_id)
            result = ControllerResult(
                task_id = task.task_id,
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = str(exc),
            )

        logger.info("Controller completed task %s with status '%s'.", task.task_id, result.status)
        return result.to_dict()

    def route_task(self, instruction: str, context: dict[str, Any]) -> RoutingPlan:
        """
        Run the routing step standalone, without executing the rest of the
        pipeline. Used by the streaming endpoint to surface the routing
        decision as an SSE event before the heavy execution begins.

        Must be called inside asyncio.to_thread — some priority branches
        invoke embed_fn and/or runtime.infer() which are blocking calls.
        """
        return self._planner.route(instruction=instruction, context=context)

    def handle_task_with_plan(
        self,
        task_dict:      dict[str, Any],
        plan:           RoutingPlan,
        on_token:       Callable[[str], None]            | None = None,
        on_status:      Callable[[str], None]            | None = None,
        on_answer_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Entry point for callers that have already run routing separately.

        Identical to handle_task() except it skips _execute() (which calls
        _planner.route()) and calls _execute_plan() directly with the
        precomputed plan. Routing runs exactly once per request on this path.

        handle_task() is left unchanged — POST /task and any other caller
        that doesn't need the early routing event still uses it unmodified.
        """
        try:
            task = Task.from_dict(task_dict)
        except ValueError as exc:
            return ControllerResult(
                task_id = task_dict.get("task_id", "unknown"),
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = f"Invalid task payload: {exc}",
            ).to_dict()

        logger.info("Controller received task %s: '%s'", task.task_id, task.instruction[:80])
        logger.info(
            "Planner plan (precomputed): agent=%s  fetch_rag=%s  fetch_episodic=%s  "
            "tools=%s  write_episode=%s  compound=%s",
            plan.agent, plan.fetch_rag, plan.fetch_episodic,
            plan.tools_to_call, plan.write_episode, plan.compound,
        )

        memory: _AnyMemory = self._memory_manager or _EphemeralMemory()
        mem_key = _memory_key(task)
        memory.add(
            "user",
            task.instruction,
            metadata={"task_id": task.task_id},
            task_id=mem_key,
        )

        try:
            result = self._execute_plan(
                task, plan, memory,
                on_token=on_token,
                on_status=on_status,
                on_answer_ready=on_answer_ready,
            )
        except Exception as exc:
            logger.exception("Unhandled exception in controller for task %s.", task.task_id)
            result = ControllerResult(
                task_id = task.task_id,
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = str(exc),
            )

        logger.info("Controller completed task %s with status '%s'.", task.task_id, result.status)
        return result.to_dict()

    def register_agent(self, agent: AgentInterface) -> None:
        """Dynamically register a new sub-agent at runtime."""
        self._agents[agent.name] = agent
        logger.info("Registered agent '%s'.", agent.name)

    # -----------------------------------------------------------------------
    # Internal execution pipeline
    # -----------------------------------------------------------------------

    def _execute_plan(
        self,
        task:            Task,
        plan:            "RoutingPlan",
        memory:          _AnyMemory,
        on_token:        Callable[[str], None]            | None = None,
        on_status:       Callable[[str], None]            | None = None,
        on_answer_ready: Callable[[dict[str, Any]], None] | None = None,
    ) -> ControllerResult:
        """
        Execute the 7-step contract from §4.4 of LOCALIST-Architecture.md.

        Steps
        -----
        1.  RoutingPlan already received from Planner (done in _execute).
        2.  If write_episode: run EpisodicMemoryWriter.
        3.  If tools_to_call: stub — logs tool names, does not execute.
            Full tool dispatch arrives in Phase 6.
        4.  If fetch_rag: run MemoryManager.query_corpus().
        5.  If fetch_episodic: run EpisodicMemoryReader.by_recency().
        5c. If graph_query: fetch links via get_backlinks() / get_outgoing_links(),
            convert to GraphQueryResult for PromptBuilder Slot 5b.
        6.  Call PromptBuilder.build() with all collected content.
        7.  Pass prebuilt prompt to agent via SubTask.context["_prebuilt_prompt"].
            Agent short-circuits its own prompt assembly when this key is present.
        """
        db_path = getattr(self._memory_manager, "_db_path", None)
        mem_key = _memory_key(task)

        # -- Step 2: Episodic write ------------------------------------------
        if plan.write_episode and db_path is not None:
            try:
                extraction = process_explicit_signal(
                    instruction     = task.instruction,
                    runtime         = self._runtime,
                    db_path         = db_path,
                    task_id         = task.task_id,
                    project_context = task.context.get("project_context", "general"),
                )
                if extraction is not None:
                    logger.info(
                        "_execute_plan: explicit episode written — "
                        "type=%s confidence=%.2f subject=%r.",
                        extraction.episode_type,
                        extraction.confidence,
                        extraction.subject,
                    )
                else:
                    logger.debug(
                        "_execute_plan: process_explicit_signal returned None "
                        "(retraction, no signal, or model said NONE)."
                    )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: episodic write failed (%s) — continuing.", exc
                )

        # -- Pre-step 3: Episodic relevance for compound P3 queries ------------
        # The Planner's Priority 3 path short-circuits before Priority 5
        # (episodic relevance). For queries that schedule tool calls we run the
        # P5 check here so episodic context is fetched alongside tool results.
        if plan.tools_to_call and not plan.fetch_episodic:
            try:
                _ep_raw = self._runtime.infer(
                    system      = (
                        "You are a routing classifier. "
                        "Reply with a single word: yes or no."
                    ),
                    prompt      = (
                        "Does this instruction relate to personal preferences, "
                        "past corrections, project decisions, or workflow patterns?\n\n"
                        f"Instruction: {task.instruction}\n\n"
                        "Answer (yes or no):"
                    ),
                    max_tokens  = 10,
                    temperature = 0.1,
                )
                if _ep_raw.strip().lower().startswith("yes"):
                    plan.fetch_episodic = True
                    logger.debug(
                        "_execute_plan: pre-dispatch episodic check → yes; "
                        "fetch_episodic updated."
                    )
            except Exception as _ep_exc:
                logger.debug(
                    "_execute_plan: pre-dispatch episodic check failed (%s).",
                    _ep_exc,
                )

        # -- Step 3: Tool dispatch ------------------------------------------
        dispatched_tool_results: list[_ToolResult] = []
        if plan.tools_to_call:
            try:
                dispatcher = MCPToolDispatcher(
                    runtime      = self._runtime,
                    project_root = task.context.get("project_root"),
                )
                dispatched_tool_results = dispatcher.dispatch(
                    tools_to_call = plan.tools_to_call,
                    instruction   = task.instruction,
                    context       = task.context,
                )
                logger.info(
                    "_execute_plan: tool dispatch complete — "
                    "tools=%s results=%d",
                    plan.tools_to_call,
                    len(dispatched_tool_results),
                )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: tool dispatch failed (%s) — "
                    "continuing without tool results.", exc,
                )

        # -- Step 3b: Corpus fallback when web_search fails -----------------
        # Runs only when the P3 tool path was taken and every web_search
        # result came back failed. Queries the corpus with the original
        # instruction using the same threshold (_priority4_corpus reuses 0.55)
        # so the answer can be grounded in vault content even when live
        # search is unavailable.
        rag_sources: list[RagSource] = []
        _web_search_failed = any(
            r.tool_name == "web_search" and not r.success
            for r in dispatched_tool_results
        )
        if _web_search_failed and self._memory_manager is not None:
            try:
                _fallback_docs = self._memory_manager.query_corpus(
                    task.instruction,
                    max_results    = 3,
                    use_embeddings = True,
                )
                rag_sources = [
                    RagSource(
                        path    = str(doc.path),
                        content = parse_wiki_doc(doc.content).body[:2000],
                    )
                    for doc in _fallback_docs
                    if doc.relevance_score >= 0.55
                    and not str(doc.path).endswith("lora-persona.md")
                ]
                if rag_sources:
                    logger.info(
                        "_execute_plan: web_search failed — corpus fallback "
                        "found %d relevant source(s).",
                        len(rag_sources),
                    )
                else:
                    logger.info(
                        "_execute_plan: web_search failed — corpus fallback "
                        "found no sources clearing threshold; proceeding "
                        "without grounding.",
                    )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: corpus fallback fetch failed (%s) — "
                    "continuing without context.", exc,
                )

        # -- Step 4: RAG fetch -----------------------------------------------
        # rag_sources is initialized above (Step 3b).  Step 4 populates it
        # on fetch_rag (P4) routes; on P3 routes fetch_rag is False so the
        # fallback-populated list flows through untouched.
        if plan.fetch_rag and self._memory_manager is not None:
            try:
                docs = self._memory_manager.query_corpus(
                    task.instruction,
                    max_results    = 3,
                    use_embeddings = True,
                )
                rag_sources = [
                    RagSource(
                        path    = str(doc.path),
                        content = parse_wiki_doc(doc.content).body[:2000],
                    )
                    for doc in docs
                    if doc.relevance_score >= 0.55
                    and not str(doc.path).endswith("lora-persona.md")
                ]
                logger.info(
                    "_execute_plan: RAG fetch complete — %d source(s).",
                    len(rag_sources),
                )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: RAG fetch failed (%s) — continuing without context.", exc
                )

        # -- Step 5: Episodic retrieval --------------------------------------
        episodic_bullets: list[EpisodeBullet] = []
        if plan.fetch_episodic and db_path is not None:
            try:
                reader  = EpisodicMemoryReader(db_path=db_path)
                records = reader.by_recency(
                    project_context=task.context.get("project_context", "general")
                )
                summary = format_episodic_summary(records)
                if summary:
                    for record in records:
                        if record.confidence >= 0.7 and record.status == "active":
                            episodic_bullets.append(EpisodeBullet(
                                content      = record.content,
                                episode_type = record.episode_type,
                                confidence   = record.confidence,
                            ))
                    self._planner.mark_episodic_injected()
                    logger.info(
                        "_execute_plan: episodic retrieval complete — %d bullet(s).",
                        len(episodic_bullets),
                    )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: episodic retrieval failed (%s) — continuing.", exc
                )

        # -- Step 5b: User profile injection ---------------------------------
        profile_facts: list[UserProfileFact] = []
        _should_inject_profile = (
            plan.fetch_episodic          # P5 route
            or plan.fetch_rag            # P4 corpus route
            or bool(episodic_bullets)    # episodic bullets fired
            or bool(rag_sources)         # corpus fallback fired (web_search failed)
        )
        if _should_inject_profile:
            try:
                self._load_user_profile()
                if self._profile_lines:
                    instr_vec = self._embed(task.instruction)
                    profile_facts = self._score_profile_facts(instr_vec)
                    if profile_facts:
                        logger.info(
                            "_execute_plan: user profile injection — "
                            "%d fact(s) selected.", len(profile_facts),
                        )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: user profile scoring failed (%s) — "
                    "continuing without profile facts.", exc,
                )

        # -- Step 5c: Graph query fetch ---------------------------------------
        graph_result: GraphQueryResult | None = None
        if plan.graph_query is not None and self._memory_manager is not None:
            direction, node_id, resolved_stem = plan.graph_query
            try:
                if direction == "incoming":
                    edges = self._memory_manager.get_backlinks(node_id)
                else:  # "outgoing"
                    edges = self._memory_manager.get_outgoing_links(node_id)

                links = [
                    GraphLinkEntry(
                        name     = (
                            # incoming: interesting side is the source page.
                            # outgoing+resolved: interesting side is the target page.
                            # outgoing+unresolved: no real target page; fall back to
                            # link_text (the original [[...]] text) — NOT target_path,
                            # which is the normalized stem-shaped string and loses
                            # the original author casing.
                            Path(edge.node_doc_path).stem
                            if edge.node_doc_path is not None
                            else edge.link_text
                        ),
                        resolved = edge.target_resolved,
                    )
                    for edge in edges
                ]

                graph_result = GraphQueryResult(
                    direction = direction,
                    page_name = resolved_stem,
                    links     = links,
                )
                logger.info(
                    "_execute_plan: graph query fetch complete — "
                    "direction=%s page=%s edges=%d",
                    direction, resolved_stem, len(links),
                )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: graph query fetch failed (%s) — "
                    "continuing without graph result.", exc,
                )

        # -- Step 5d: Working state assembly (deterministic) ----------------
        working_state: WorkingMemoryState | None = None
        if plan.graph_query is None:
            try:
                active_artifacts = [s.path for s in rag_sources]
                current_project = (
                    task.context.get("project_context")
                    if task.context.get("project_context") not in (None, "general")
                    else None
                )
                if current_project or active_artifacts:
                    working_state = WorkingMemoryState(
                        current_project  = current_project,
                        active_artifacts = active_artifacts,
                    )
                    logger.info(
                        "_execute_plan: working state assembled — "
                        "current_project=%r artifacts=%d",
                        current_project, len(active_artifacts),
                    )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: working state assembly failed (%s) — "
                    "continuing without working state.", exc,
                )

        # -- Step 6: PromptBuilder assembly ----------------------------------
        working_turns: list[Turn] = []
        try:
            entries = memory.get_context_window(
                task_id    = mem_key,
                limit      = 5,
                max_tokens = 300,
            )
            working_turns = [
                Turn(role=e["role"], content=e["content"])
                for e in entries
            ]
        except Exception as exc:
            logger.debug("_execute_plan: working memory fetch failed (%s).", exc)

        system_prompt, user_prompt = _PROMPT_BUILDER.build(
            instruction      = task.instruction,
            persona          = self._load_persona(),
            episodic_summary = episodic_bullets         or None,
            profile_facts    = profile_facts            or None,
            rag_snippets     = rag_sources              or None,
            tool_results     = [
                r for r in dispatched_tool_results
                if not r.result.startswith("ERROR:")
                and not r.result.startswith("<")
                and len(r.result.strip()) >= 5
            ] or None,
            graph_result     = graph_result             or None,
            working_state    = working_state,
            working_memory   = working_turns            or None,
            session_files    = _session_files.get_files() or None,
        )

        logger.debug(
            "_execute_plan: prompt assembled — "
            "system_chars=%d  user_chars=%d  "
            "working_turns=%d  episodic_bullets=%d  rag_sources=%d  "
            "working_state_artifacts=%d  session_files=%d",
            len(system_prompt), len(user_prompt),
            len(working_turns), len(episodic_bullets), len(rag_sources),
            len(working_state.active_artifacts) if working_state is not None else 0,
            len(_session_files.get_files()),
        )
        logger.debug(
            "_execute_plan: assembled system_prompt:\n%s",
            system_prompt,
        )
        logger.debug(
            "_execute_plan: assembled user_prompt:\n%s",
            user_prompt,
        )

        # -- Step 7: Build SubTask with prebuilt prompt ----------------------
        effective_agent_name = plan.agent
        agent = self._agents.get(effective_agent_name)
        if agent is None:
            agent = self._agents.get("conversational_agent")
            if agent is None:
                return ControllerResult(
                    task_id = task.task_id,
                    status  = TaskStatus.FAILED,
                    answer  = "",
                    error   = (
                        f"Planner requested '{plan.agent}' which is not registered "
                        "and no conversational_agent fallback exists."
                    ),
                )
            effective_agent_name = "conversational_agent"
            logger.warning(
                "_execute_plan: '%s' not registered; falling back to "
                "conversational_agent.", plan.agent,
            )

        prebuilt_sources = [s.path for s in rag_sources] + [
            f"session://{sf.filename}" for sf in _session_files.get_files()
        ]

        subtask_context = {
            **task.context,
            "_prebuilt_prompt":  user_prompt,
            "_prebuilt_system":  system_prompt,
            "_prebuilt_sources": prebuilt_sources,
            "_routing": {
                "fetch_rag":      plan.fetch_rag,
                "fetch_episodic": plan.fetch_episodic,
                "tools_to_call":  plan.tools_to_call,
                "write_episode":  plan.write_episode,
                "episode_type":   plan.episode_type,
                "compound":       plan.compound,
                "graph_query":    plan.graph_query,
            },
        }
        if on_token is not None:
            subtask_context["_on_token"] = on_token

        subtask = SubTask(
            subtask_id  = f"{task.task_id}-0",
            agent_name  = effective_agent_name,
            instruction = task.instruction,
            context     = subtask_context,
        )

        logger.info("TIMING dispatch_start t=%.4f", time.monotonic())
        results = self._dispatch([subtask], memory, mem_key)
        logger.info("TIMING dispatch_end t=%.4f", time.monotonic())

        # Early-completion signal — fire before memory hooks so the streaming
        # endpoint can emit 'sources'/'done' and unblock the client immediately
        # after the answer is ready, rather than waiting 18-23 s for hooks.
        if on_answer_ready is not None:
            _early = self._build_conversational_result(task, plan, effective_agent_name, results)
            if _early is not None:
                on_answer_ready(_early.to_dict())

        # -- Post-response hook: implicit episodic extraction ----------------
        # Run after every dispatch. Catches durable facts the user revealed
        # without an explicit memory command. Skipped when:
        #   - No MemoryManager (no db_path to write to)
        #   - Agent failed (no useful response to extract from)
        #   - write_episode was already True (explicit extraction already ran;
        #     avoid double-writing the same turn)
        if (
            db_path is not None
            and not plan.write_episode
            and results
            and results[0].status == TaskStatus.COMPLETE
        ):
            logger.info("TIMING implicit_extraction_start t=%.4f", time.monotonic())
            try:
                agent_response = results[0].output.get("answer", "")
                if agent_response:
                    implicit = process_implicit_extraction(
                        instruction     = task.instruction,
                        response        = agent_response,
                        runtime         = self._runtime,
                        db_path         = db_path,
                        task_id         = task.task_id,
                        project_context = task.context.get(
                            "project_context", "general"
                        ),
                    )
                    if implicit is not None:
                        logger.info(
                            "_execute_plan: implicit episode written — "
                            "type=%s confidence=%.2f subject=%r.",
                            implicit.episode_type,
                            implicit.confidence,
                            implicit.subject,
                        )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: implicit extraction failed (%s) — "
                    "continuing.", exc,
                )
            logger.info("TIMING implicit_extraction_end t=%.4f", time.monotonic())
            if on_status is not None:
                on_status("Updating working memory…")

            # Working state update — separate try for independent error isolation.
            # Runs regardless of plan.graph_query: P3c exclusivity guards Slot 6A
            # *rendering* on the next turn, not whether working state is captured
            # after a graph-query turn. The instruction/response pair from any
            # route represents real conversational state worth persisting.
            logger.info("TIMING working_state_start t=%.4f", time.monotonic())
            try:
                ws_response = results[0].output.get("answer", "")
                if ws_response:
                    process_working_state_update(
                        instruction = task.instruction,
                        response    = ws_response,
                        mem_key     = mem_key,
                        runtime     = self._runtime,
                        db_path     = db_path,
                        persona     = self._load_persona(),
                    )
            except Exception as exc:
                logger.warning(
                    "_execute_plan: working state update failed (%s) — "
                    "continuing.", exc,
                )
            logger.info("TIMING working_state_end t=%.4f", time.monotonic())

        logger.info("TIMING execute_plan_end t=%.4f", time.monotonic())
        final_result = self._build_conversational_result(task, plan, effective_agent_name, results)
        if final_result is not None:
            return final_result
        return self._synthesizer.synthesize(task, results, memory)

    def _execute(self, task: Task, memory: _AnyMemory) -> ControllerResult:
        """
        Route via rule-based Planner → execute plan → return.
        """
        plan = self._planner.route(
            instruction = task.instruction,
            context     = task.context,
        )

        logger.info(
            "Planner plan: agent=%s  fetch_rag=%s  fetch_episodic=%s  "
            "tools=%s  write_episode=%s  compound=%s",
            plan.agent, plan.fetch_rag, plan.fetch_episodic,
            plan.tools_to_call, plan.write_episode, plan.compound,
        )

        return self._execute_plan(task, plan, memory)

    def _build_conversational_result(
        self,
        task:                 Task,
        plan:                 "RoutingPlan",
        effective_agent_name: str,
        results:              list[AgentResult],
    ) -> "ControllerResult | None":
        """
        Build a ControllerResult for a successful single-agent conversational dispatch.
        Returns None when the synthesizer path should be used instead (non-conversational
        agent, multi-result, or failed dispatch).

        Used by both the early on_answer_ready callback and the final return path so
        the answer-extraction logic is never duplicated.
        """
        if not (effective_agent_name == "conversational_agent" and len(results) == 1):
            return None
        r = results[0]
        if r.status != TaskStatus.COMPLETE:
            return None
        return ControllerResult(
            task_id  = task.task_id,
            status   = TaskStatus.COMPLETE,
            answer   = r.output.get("answer", ""),
            sources  = [
                (
                    {
                        "path": s[len("session://"):],
                        "type": "session",
                        "name": s[len("session://"):],
                    }
                    if s.startswith("session://")
                    else {
                        "path": s,
                        "type": "wiki" if "/wiki/" in s else "raw",
                        "name": s.split("/")[-1].replace(".md", "").replace("-", " ").title(),
                    }
                )
                for s in r.output.get("sources", [])
            ],
            metadata = {
                "agent":              effective_agent_name,
                "priority":           plan.priority,
                "fetch_rag":          plan.fetch_rag,
                "fetch_episodic":     plan.fetch_episodic,
                "tools_fired":        plan.tools_to_call,
                "tool_signal_source": plan.tool_signal_source,
                "grounded":           r.output.get("grounded", False),
            },
        )

    def _dispatch(
        self,
        subtasks: list[SubTask],
        memory:   _AnyMemory,
        task_id:  str,
    ) -> list[AgentResult]:
        """
        Execute subtasks in order, feeding each result into memory so that
        downstream agents have access to prior outputs.
        """
        results: list[AgentResult] = []

        for subtask in subtasks:
            agent = self._agents.get(subtask.agent_name)
            if agent is None:
                logger.error("No agent registered for name '%s'.", subtask.agent_name)
                results.append(AgentResult(
                    subtask_id = subtask.subtask_id,
                    agent_name = subtask.agent_name,
                    status     = TaskStatus.FAILED,
                    output     = {},
                    error      = f"Unknown agent: {subtask.agent_name}",
                ))
                continue

            logger.info("Dispatching subtask %s to '%s'.", subtask.subtask_id, agent.name)
            try:
                result = agent.run(subtask)
            except Exception as exc:
                logger.exception("Agent '%s' raised an exception.", agent.name)
                result = AgentResult(
                    subtask_id = subtask.subtask_id,
                    agent_name = subtask.agent_name,
                    status     = TaskStatus.FAILED,
                    output     = {},
                    error      = str(exc),
                )

            memory.add_agent_result(result, task_id=task_id)
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Usage example (not executed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class _MockRuntime:
        def infer(self, prompt: str, system: str = "", **_) -> str:
            return '[{"agent": "wiki_agent", "instruction": "Look up the topic."}]'
        def embed(self, text: str) -> list[float]:
            return [0.0] * 768

    class _MockWikiAgent:
        @property
        def name(self) -> str:
            return "wiki_agent"
        def can_handle(self, instruction: str) -> bool:
            return True
        def run(self, subtask: SubTask) -> AgentResult:
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.COMPLETE,
                output     = {"text": "Mock wiki result.", "sources": []},
            )

    runtime    = _MockRuntime()
    controller = ControllerAgent(runtime=runtime, agents=[_MockWikiAgent()])
    output     = controller.handle_task({"instruction": "What is the capital of France?"})
    print(output)