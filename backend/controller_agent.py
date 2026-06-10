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

Layer placement
---------------
  Svelte UI  →  FastAPI  →  ControllerAgent  →  Sub-agents / Runtime
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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
    Abstraction over the Local Runtime Layer (Azure AI Foundry).
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
# Memory manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Manages the context window across a multi-step workflow.

    In the initial skeleton this is an in-process list.  Replace the
    storage backend (e.g. with a vector store or SQLite) without changing
    the interface.
    """

    def __init__(self, max_tokens: int = 8192) -> None:
        self._entries:    list[dict[str, Any]] = []
        self._max_tokens: int                  = max_tokens

    # -- write --

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._entries.append({
            "role":     role,
            "content":  content,
            "metadata": metadata or {},
        })
        self._evict_if_needed()

    def add_agent_result(self, result: AgentResult) -> None:
        self.add(
            role    = "agent",
            content = str(result.output),
            metadata = {"agent": result.agent_name, "subtask_id": result.subtask_id},
        )

    # -- read --

    def get_context_window(self) -> list[dict[str, Any]]:
        """Return entries suitable for passing to the runtime as conversation history."""
        return list(self._entries)

    def format_for_prompt(self) -> str:
        """Flatten memory into a single string for inclusion in a prompt."""
        lines = []
        for e in self._entries:
            lines.append(f"[{e['role'].upper()}] {e['content']}")
        return "\n".join(lines)

    # -- maintenance --

    def clear(self) -> None:
        self._entries.clear()

    def _evict_if_needed(self) -> None:
        """
        Naive eviction: drop oldest entries when the list exceeds a soft cap.
        Replace with a token-counting strategy once a tokenizer is available.
        """
        soft_cap = 200  # entries
        if len(self._entries) > soft_cap:
            dropped = len(self._entries) - soft_cap
            self._entries = self._entries[dropped:]
            logger.debug("MemoryManager evicted %d old entries.", dropped)


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

        # Ask the model to propose a plan.
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
            "- research_agent : use for questions, research requests, analysis, synthesis,\n"
            "                   cross-document reasoning, and corpus-wide queries.\n"
            "- wiki_agent     : use ONLY when the instruction explicitly mentions ingesting,\n"
            "                   uploading, or adding a specific file to the wiki\n"
            "                   (must include a file path).\n\n"
            "Rules:\n"
            "1. For questions and all general research instructions -> use research_agent ONLY.\n"
            "2. For wiki ingestion tasks that include a file path -> use wiki_agent ONLY.\n"
            "   DO NOT add research_agent as a follow-on step after wiki_agent.\n"
            "   The wiki_agent creates the research note itself. Adding research_agent\n"
            "   is redundant and wrong.\n"
            "3. Never schedule both agents for the same task. Each task uses exactly one agent.\n"
            "4. Output ONLY the JSON array. No prose, no markdown fences, no commentary.\n\n"
            "Example outputs:\n"
            "[{\"agent\": \"research_agent\", \"instruction\": \"What do we know about X?\"}]\n"
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
                # Merge task-level context (raw_path, wiki_dir, etc.) with any
                # per-step context the model supplies.  Model values win on conflict.
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
        """Route to research_agent by preference, then first can_handle() match."""
        # Prefer research_agent as the safe fallback — it handles open-ended
        # instructions without requiring context keys that wiki_agent needs.
        for agent in agents:
            if agent.name == "research_agent":
                return [SubTask(
                    subtask_id  = f"{task.task_id}-0",
                    agent_name  = agent.name,
                    instruction = task.instruction,
                    context     = task.context,
                )]
        # Then try can_handle() routing
        for agent in agents:
            if agent.can_handle(task.instruction):
                return [SubTask(
                    subtask_id  = f"{task.task_id}-0",
                    agent_name  = agent.name,
                    instruction = task.instruction,
                    context     = task.context,
                )]
        # Last resort: first registered agent
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
        memory:  MemoryManager,
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

        context_str  = memory.format_for_prompt()
        results_str  = self._format_results(results)

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
    Classifies a user instruction into one of three buckets before planning.

    Buckets
    -------
    conversational  — greetings, chitchat, meta questions about the system
    research        — questions, analysis, synthesis, anything needing lookup
    ingest          — explicit requests to add/upload a specific file to the wiki

    One fast model call with a constrained output format.  Falls back to
    "research" on any parse failure so the pipeline always continues.
    """

    _SYSTEM = (
        "You are an intent classifier for a local research agent system. "
        "Classify the user instruction into exactly one of these categories:\n"
        "  conversational — greetings, small talk, meta questions "
        "(e.g. 'hello', 'how are you', 'what can you do')\n"
        "  research       — questions, analysis, research requests, "
        "anything that needs looking up\n"
        "  ingest         — explicit requests to add, upload, or ingest "
        "a specific file into the wiki\n\n"
        "Output ONLY one word: conversational, research, or ingest. Nothing else."
    )

    def __init__(self, runtime: RuntimeClient) -> None:
        self._runtime = runtime

    def classify(self, instruction: str) -> str:
        """Return 'conversational', 'research', or 'ingest'.  Never raises."""
        try:
            raw = self._runtime.infer(
                system      = self._SYSTEM,
                prompt      = f"Instruction: {instruction}",
                max_tokens  = 8,
                temperature = 0.0,
            ).strip().lower()

            # Accept the first word in case the model adds punctuation
            first_word = raw.split()[0].rstrip(".,!") if raw else ""
            if first_word in {"conversational", "research", "ingest"}:
                logger.info(
                    "IntentClassifier: '%s' → %s", instruction[:60], first_word
                )
                return first_word
        except Exception as exc:
            logger.warning(
                "IntentClassifier failed (%s); defaulting to research.", exc
            )

        logger.info("IntentClassifier: unrecognised output, defaulting to research.")
        return "research"


# ---------------------------------------------------------------------------
# Conversational agent
# ---------------------------------------------------------------------------

class ConversationalAgent:
    """
    Fast-path agent for natural-language chat.

    Bypasses the Planner, corpus, embeddings, and Synthesizer entirely.
    Satisfies AgentInterface so it can be called uniformly by _execute(),
    but can_handle() always returns False — routing is exclusively by
    IntentClassifier, never by the Planner's fallback logic.

    Output contract
    ---------------
    AgentResult.output["answer"] is a single plain-text paragraph.
    Never JSON, never a list, never a code block.
    """

    _SYSTEM = (
        "You are LORA, a helpful local research assistant. "
        "Respond naturally and concisely to the user's message in plain text only. "
        "Write a single short paragraph — two to four sentences at most. "
        "Be warm, direct, and helpful. "
        "Do not output JSON, XML, bullet lists, numbered lists, or code blocks. "
        "Do not reference tools, agents, internal system details, or file paths. "
        "Do not start your reply with a label, heading, or role description."
    )

    def __init__(self, runtime: RuntimeClient) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "conversational_agent"

    def can_handle(self, instruction: str) -> bool:
        # Always False — the Planner must never route here.
        # Routing is done exclusively via IntentClassifier in _execute().
        return False

    def run(self, subtask: SubTask) -> AgentResult:
        """
        Handle a conversational turn with a single low-token model call.

        No corpus access.  No embeddings.  No file I/O.
        max_tokens and temperature are intentionally small — fast path.
        """
        ctx         = subtask.context
        max_tokens  = int(ctx.get("max_tokens",   256))
        temperature = float(ctx.get("temperature", 0.7))

        try:
            answer = self._runtime.infer(
                system      = self._SYSTEM,
                prompt      = subtask.instruction,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
        except Exception as exc:
            logger.warning(
                "[conversational_agent] infer() failed (%s); using fallback.", exc
            )
            answer = "Hello! I'm LORA, your local research assistant. How can I help?"

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answer, "sources": []},
        )

class ControllerAgent:
    """
    The executive function of the LORA system.

    FastAPI instantiates one ControllerAgent at startup and calls
    ``handle_task(task_dict)`` for every incoming request.

    This class does not know about HTTP, request/response objects,
    streaming, or the Svelte UI.  It is a pure Python reasoning coordinator.
    """

    def __init__(
        self,
        runtime: RuntimeClient,
        agents:  list[AgentInterface],
    ) -> None:
        self._runtime              = runtime
        self._agents               = {a.name: a for a in agents}
        self._planner              = Planner(runtime)
        self._synthesizer          = Synthesizer(runtime)
        self._classifier           = IntentClassifier(runtime)
        self._conversational_agent = ConversationalAgent(runtime)

        if not agents:
            logger.warning("ControllerAgent initialized with no sub-agents.")

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

        memory = MemoryManager()
        memory.add("user", task.instruction, metadata={"task_id": task.task_id})

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

    def register_agent(self, agent: AgentInterface) -> None:
        """Dynamically register a new sub-agent at runtime."""
        self._agents[agent.name] = agent
        logger.info("Registered agent '%s'.", agent.name)

    # -----------------------------------------------------------------------
    # Internal execution pipeline
    # -----------------------------------------------------------------------

    def _execute(self, task: Task, memory: MemoryManager) -> ControllerResult:
        """Classify intent → route → [Dispatch →] Synthesize."""

        # 0. Classify intent — fast single-token call, always returns a valid bucket
        intent = self._classifier.classify(task.instruction)

        # Conversational shortcut — skip Planner, corpus, and Synthesizer entirely
        if intent == "conversational":
            logger.info("Intent=conversational — routing direct to ConversationalAgent.")
            subtask = SubTask(
                subtask_id  = f"{task.task_id}-0",
                agent_name  = self._conversational_agent.name,
                instruction = task.instruction,
                context     = task.context,
            )
            result = self._conversational_agent.run(subtask)
            memory.add_agent_result(result)
            return ControllerResult(
                task_id  = task.task_id,
                status   = TaskStatus.COMPLETE,
                answer   = result.output.get("answer", ""),
                sources  = [],
                metadata = {"intent": intent},
            )

        # Research / ingest — run the full Planner → Dispatch → Synthesize pipeline
        logger.info("Intent=%s — routing through Planner pipeline.", intent)
        agent_list = list(self._agents.values())
        subtasks   = self._planner.plan(task, agent_list)

        if not subtasks:
            return ControllerResult(
                task_id = task.task_id,
                status  = TaskStatus.FAILED,
                answer  = "",
                error   = "Planner could not construct a valid execution plan.",
            )

        # 2. Dispatch
        results = self._dispatch(subtasks, memory)

        # 3. Synthesize
        return self._synthesizer.synthesize(task, results, memory)

    def _dispatch(
        self,
        subtasks: list[SubTask],
        memory:   MemoryManager,
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

            memory.add_agent_result(result)
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Usage example (not executed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This block exists only to illustrate how FastAPI wires things together.
    # It is NOT FastAPI code — it shows the pure-Python call pattern.

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