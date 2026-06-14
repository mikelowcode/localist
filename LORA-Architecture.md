# LORA — Canonical Architecture Specification

> **Status: Authoritative**
> This document is the canonical reference for LORA's substrate architecture.
> No implementation begins until it is reflected here. No deviation from this
> specification is made without updating this document first.

---

## Table of Contents

1. [System Identity](#1-system-identity)
2. [Episodic Memory Schema](#2-episodic-memory-schema)
3. [Unified Prompt Contract](#3-unified-prompt-contract)
4. [Planner Routing Model](#4-planner-routing-model)
5. [Build-Order Checklist](#5-build-order-checklist)

---

## 1. System Identity

LORA is a **local-first, agentic research assistant**. Every architectural
decision is evaluated against five constraints:

| Constraint | Meaning |
|---|---|
| **Local** | All inference, embeddings, memory, and tools run on-device. No cloud calls except explicit user-initiated web search. |
| **Sparse** | Memory is high-value semantic events, not transcripts. Prompts carry only what is needed. |
| **Predictable** | The same input produces the same routing decision. Inference is used for reasoning, not for control flow, except where explicitly specified. |
| **Minimal** | System prompts are small. Persona lives in the wiki. Agents are single-purpose. |
| **Auditable** | Every prompt can be logged and read. Every memory write has provenance. Every routing decision has a named rule. |

These constraints are not preferences. They are the identity of the system.

---

## 2. Episodic Memory Schema

### 2.1 Design Principles

An **episode** is a meaningful, durable semantic event extracted from a
conversation. It is not a session log, a turn record, or a summary of what
was said. It is a specific, typed fact that is worth remembering across
sessions.

Episodes are **sparse by design**. Most turns produce no episode. A turn
that contains a user preference, a correction, a decision, or a workflow
pattern produces one or more episodes. The goal is a store of high-value
records, not a compressed transcript.

### 2.2 Table Definition

```sql
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_type    TEXT    NOT NULL,
    subject         TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    source          TEXT    NOT NULL,
    task_id         TEXT,
    conversation_id TEXT,
    project_context TEXT,
    status          TEXT    NOT NULL DEFAULT 'active',
    created_at      REAL    NOT NULL,
    last_accessed   REAL,
    embedding       BLOB
);

CREATE INDEX IF NOT EXISTS idx_episodes_type_status
    ON episodes (episode_type, status);

CREATE INDEX IF NOT EXISTS idx_episodes_subject
    ON episodes (subject, status);

CREATE INDEX IF NOT EXISTS idx_episodes_project
    ON episodes (project_context, status);
```

### 2.3 Field Reference

| Field | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key. |
| `episode_type` | TEXT | One of the seven canonical types. See §2.4. |
| `subject` | TEXT | What the episode is about. Used for exact-match retrieval and deduplication. |
| `content` | TEXT | The durable fact or event, in plain language. One sentence preferred. |
| `confidence` | REAL | 0.0–1.0. Code-extracted events = 1.0. Model-extracted events = 0.6–0.9. |
| `source` | TEXT | `"explicit"` for code-detected signals. `"model_extracted"` for inference-detected signals. |
| `task_id` | TEXT | The `task_id` of the originating request. Nullable. |
| `conversation_id` | TEXT | The originating conversation identifier. Nullable. |
| `project_context` | TEXT | Scopes retrieval. e.g. `"LORA"`, `"general"`. Nullable defaults to `"general"`. |
| `status` | TEXT | `"active"` \| `"superseded"` \| `"retracted"`. See §2.5. |
| `created_at` | REAL | Unix timestamp (from `time.time()`). |
| `last_accessed` | REAL | Updated on every retrieval. Enables LRU decay. Nullable until first access. |
| `embedding` | BLOB | Optional 768-dim float vector. Same encoding as `document_index`. Nullable. |

### 2.4 Type Taxonomy

The `episode_type` field is a **closed set**. Adding a new type is a
deliberate architectural decision, not an ad-hoc extraction choice.

| Type | Meaning | Example content |
|---|---|---|
| `preference` | A stated or inferred user preference about style, process, or output. | `"Prefers step-by-step swap instructions over inline diffs"` |
| `correction` | A factual correction the user made to a prior assistant output or assumption. | `"raw_path is passed explicitly from the UI, not resolved by fuzzy match"` |
| `decision` | An architectural or design decision that has been committed to. | `"Committed to SQLite-backed MemoryManager over in-process shim"` |
| `workflow` | A repeating pattern or process the user follows. | `"Always uploads source files before accepting generated code"` |
| `project_fact` | A durable fact about an ongoing project or its components. | `"oMLX 0.4.2 is the current inference runtime running Gemma 4B quantized"` |
| `task_completion` | A task or milestone that has been reached. | `"File ingestion pipeline is functional end-to-end"` |
| `naming_convention` | A naming or terminology rule that must be respected. | `"The local inference server is called oMLX, not OMLX or omlx"` |

### 2.5 Lifecycle Rules

Episodes are **never deleted**. The `status` field manages their lifecycle.

| Status | Meaning | Transition |
|---|---|---|
| `active` | Trusted. Eligible for injection into context. | Default on creation. |
| `superseded` | Replaced by a newer episode with the same `subject` and `episode_type`. | Set on the old record when a conflicting new record is inserted as `active`. |
| `retracted` | Explicitly invalidated by user command or model detection of contradiction. | Set directly; no new record required. |

**Supersession rule:** When a new episode is inserted and an `active` record
with the same `subject` and `episode_type` already exists, the existing
record is updated to `status = 'superseded'` before the new record is
inserted. Both records are retained for audit.

**Retraction rule:** Explicit user commands (`"forget that"`,
`"that's no longer true"`) trigger a retraction write. The record is marked
`status = 'retracted'`; no replacement is inserted unless the user provides
a corrected value.

### 2.6 Retrieval Modes

Three retrieval modes cover all Planner use cases.

**Mode 1 — Exact subject match**
Used when the Planner knows the specific subject to retrieve.

```sql
SELECT * FROM episodes
WHERE subject = :subject
  AND status = 'active'
ORDER BY confidence DESC, created_at DESC
LIMIT 5;
```

**Mode 2 — Type-filtered recency**
Used for session priming. Loads high-priority durable context.

```sql
SELECT * FROM episodes
WHERE episode_type IN ('preference', 'correction', 'decision', 'workflow')
  AND status = 'active'
  AND project_context = :project_context
ORDER BY last_accessed DESC, confidence DESC
LIMIT 5;
```

**Mode 3 — Semantic similarity**
Used for open-ended queries. Cosine ranking over the `embedding` column,
using the same infrastructure as `document_index` in `MemoryManager`.
Falls back to keyword overlap scoring when embeddings are absent.

### 2.7 Summarization Contract

When episodic memory is injected into the prompt, it must conform to this
contract exactly. The contract is a token budget constraint, not a display
preference.

| Rule | Value |
|---|---|
| Maximum bullets | 5 |
| Maximum tokens per bullet | 20 |
| Minimum confidence threshold | 0.7 |
| Priority order | `correction` > `decision` > `preference` > `workflow` > `project_fact` > `naming_convention` > `task_completion` |
| Eligible statuses | `active` only |

**Output format:**

```
[EPISODIC MEMORY]
- Prefers explicit raw_path passing over fuzzy vault resolution (preference, 1.0)
- XML pre-validation must shield Markdown content blocks before parsing (correction, 1.0)
- oMLX 0.4.2 is the current inference runtime, Gemma 4B quantized (project_fact, 1.0)
```

The `[EPISODIC MEMORY]` label is mandatory. It tells the model the
provenance of these bullets. Without it, the model may weight them as
user-provided facts rather than retrieved memory.

The type annotation and confidence score in parentheses are mandatory. They
tell the model how to weight each bullet. A `correction` should override the
model's prior. A `preference` should shape style. A `project_fact` is
background context.

---

## 3. Unified Prompt Contract

### 3.1 Design Principles

The prompt contract is a **cognitive architecture**, not a formatting
convention. The order of slots, the presence of labels, and the token
ceilings all affect how Gemma 4B weights information under attention
constraints. A poorly ordered prompt does not just look wrong — it produces
worse answers.

The ordering principle is: **static before dynamic, stable before volatile.**

All content that is invariant across turns is placed first, forming a stable
prefix that inference backends can cache and reuse. All content that changes
per-turn is placed last. This is a KV-cache architectural constraint, not a
style preference: every backend that supports prefix caching (oMLX, MLX-LM,
Foundry Local, vLLM, llama.cpp, TGI, TensorRT-LLM, ONNX Runtime) requires
exact byte-identity from the start of the token sequence. A single character
change anywhere in the prefix causes a complete cache miss for everything
after it.

Content stability ranking, most stable to least:

| Rank | Content | Changes when |
|---|---|---|
| 1 | Identity constant | Never |
| 2 | Persona | Wiki page updated |
| 3 | Episodic memory | New episode written |
| 4 | RAG snippets | Query topic changes |
| 5 | Tool results | New tool call issued |
| 6 | Working memory | Every turn |
| 7 | Current instruction | Every turn (always last) |

### 3.2 Slot Definitions

The prompt is assembled as two runtime arguments: a **system message**
(passed as `system=` to the runtime client) and a **user message** (passed
as the user turn). Slots 1a and 1b form the system message. Slots 3–7 form
the user message in strict stability order.

---

#### Slot 1a — Identity

**Purpose:** Establishes who LORA is and how it reasons. The invariant
anchor of every prompt. Never changes.

**Token ceiling:** ~50 tokens (the canonical value is 43 tokens)

**Content:** Identity name, core behavioral constraint, epistemic stance.

**Canonical value:**
```
You are LORA, a local research assistant. You reason carefully, cite your
sources, and acknowledge when you don't know something. You do not simulate
certainty.
```

**Rules:**
- This is a constant defined in `PromptBuilder._SYSTEM`. It is never
  modified at runtime.
- Keep it minimal. Every token here is cached unconditionally by all
  backends — there is no cost to including it, but expanding it narrows
  the headroom for dynamic slots.

---

#### Slot 1b — Persona

**Purpose:** LORA's voice, style, and project-specific behavioral context.
Loaded once per session from `wiki/lora-persona.md` and appended to the
system message after the identity constant.

**Token ceiling:** 500 tokens (hard limit). Truncated by `PromptBuilder`
when the wiki page exceeds this budget.

**Format:** Appended to Slot 1a with a double newline separator. No label
is added; the persona content is inserted raw:

```
You are LORA, a local research assistant. You reason carefully, cite your
sources, and acknowledge when you don't know something. You do not simulate
certainty.

{persona content from wiki/lora-persona.md}
```

**Rules:**
- Loaded by `ControllerAgent._load_persona()`, which caches the result in
  `self._persona_cache` after the first successful corpus query.
- Passed to `PromptBuilder.build()` as the `persona=` keyword argument.
- When persona is `None` or empty, the system message is Slot 1a only —
  no separator, no placeholder.
- WikiAgent's XML-only system prompt is a protected contract. WikiAgent
  does not pass `persona=` and never receives Slot 1b. See §3.5.
- The persona must remain byte-stable within a session. Re-querying the
  corpus on every turn would break prefix caching. The cache is
  invalidated only when WikiAgent writes a new persona page.

---

#### Slot 3 — Episodic Memory

**Purpose:** Durable facts about the user, project, and preferences that
are relevant to this specific request.

**Token ceiling:** 150 tokens (hard limit)

**Format:** See §2.7 Summarization Contract.

**Rules:**
- This slot is **conditional**. It is omitted entirely when not relevant.
  No empty label, no placeholder.
- Relevance is determined by the Planner (Priority 5). See §4.
- When injected: 3–5 bullets, confidence ≥ 0.7, type-ordered per §2.7.
- The `[EPISODIC MEMORY]` label and inline type annotations are mandatory.
- Placed first in the user message because episodic content changes rarely
  (only when a new episode is written), maximising the stable prefix shared
  across consecutive turns.

---

#### Slot 4 — RAG Snippets

**Purpose:** Relevant content from the wiki corpus and document index.

**Token ceiling:** 450 tokens (hard limit)

**Format:**
```
[CONTEXT]
Source: wiki/XML Parsing.md
{2–3 sentences of relevant content, not truncated mid-sentence}

Source: wiki/WikiAgent Architecture.md
{2–3 sentences of relevant content, not truncated mid-sentence}
```

**Rules:**
- This slot is **conditional**. Omitted entirely when corpus yields no
  results above threshold.
- Maximum 3 sources. Wiki sources are preferred over raw doc sources.
- Content is never truncated mid-sentence. If a passage exceeds the per-
  source budget, it is cut at the nearest sentence boundary.
- The `[CONTEXT]` label is mandatory. Source paths are mandatory.
- Placed after episodic memory because RAG results change per query topic
  but are deterministic for the same query, giving partial prefix reuse on
  follow-up questions about the same topic.

---

#### Slot 5 — Tool Results

**Purpose:** Fresh, request-specific evidence from tool calls made during
this request's routing phase.

**Token ceiling:** 500 tokens (hard limit)

**Format:**
```
[TOOL RESULTS]
{tool_name}({call_parameters}):
  {truncated result content}
```

**Rules:**
- This slot is **conditional**. Omitted entirely when no tools were called.
- Tool results are injected in the order tools were called.
- Results are truncated to essential content. Long results (file reads, web
  search) are summarized to 3–5 lines unless the full content is the
  explicit purpose of the request.
- The `[TOOL RESULTS]` label is mandatory. Tool name and call parameters
  are mandatory for auditability.

---

#### Slot 6 — Working Memory

**Purpose:** The immediate conversational context. What just happened.

**Token ceiling:** 300 tokens (hard limit). Oldest turns are dropped first
when the ceiling is exceeded.

**Format:**
```
[WORKING MEMORY]
Turn -2 [user]: {prior user message}
Turn -2 [assistant]: {prior assistant response}
Turn -1 [user]: {most recent prior user message}
Turn -1 [assistant]: {most recent prior assistant response}
```

**Rules:**
- Default window: last 3 turns. Maximum window: 5 turns.
- Turns are listed in chronological order (oldest first, newest last).
- Tool results from prior turns are included as
  `[tool: {tool_name}] {truncated result}` entries, capped at 2–3 lines.
- The 300-token ceiling is enforced by `MemoryManager.get_context_window()`
  via a `max_tokens` parameter. Truncation drops oldest turns first, never
  mid-turn.
- This slot is conditional. It is omitted when no prior turns exist.
- Placed second-to-last because working memory changes every turn. Placing
  it here rather than first means only the instruction (Slot 7) consistently
  invalidates the cache prefix on every turn.

---

#### Slot 7 — Instruction

**Purpose:** The raw instruction from the current turn.

**Token ceiling:** Uncapped.

**Format:**
```
[INSTRUCTION]
{instruction}
```

**Rules:**
- No transformation of the instruction.
- Always the last slot in the user message. This is a KV-cache invariant:
  the most volatile content must be at the end so that all stable content
  above it can be prefix-cached.
- If the instruction references a file path or prior result, those are
  resolved before this slot is populated. The instruction text itself is
  never modified.

---

### 3.3 Aggregate Token Budget

| Slot | Label | Ceiling | Presence | Message |
|---|---|---|---|---|
| 1a — Identity | *(none)* | ~50 | Always | System |
| 1b — Persona | *(none)* | 500 | Conditional | System |
| 3 — Episodic memory | `[EPISODIC MEMORY]` | 150 | Conditional | User |
| 4 — RAG snippets | `[CONTEXT]` | 450 | Conditional | User |
| 5 — Tool results | `[TOOL RESULTS]` | 500 | Conditional | User |
| 6 — Working memory | `[WORKING MEMORY]` | 300 | Conditional | User |
| 7 — Instruction | `[INSTRUCTION]` | Uncapped | Always | User |
| **Worst-case total** | | **~1,950** | | |

Slot numbers 2 and the old Slot 2 label `[USER]` are retired. The gap
between 1b and 3 is intentional: slot numbering reflects cognitive role
and stability rank, not sequential position in the output string.

Gemma 4B quantized has an effective context window of approximately 8,000
tokens. The prompt contract consumes under 2,000 tokens in the worst case
(persona + all dynamic slots fully populated), leaving substantial headroom
for model output. A prompt contract that routinely approaches the context
ceiling will silently degrade on long sessions. This budget prevents that.

### 3.4 PromptBuilder Interface

`PromptBuilder` is the single point of prompt assembly. Every agent calls
it. No agent assembles its own prompt string.

```python
@dataclass
class Turn:
    role:    str   # "user" | "assistant" | "tool"
    content: str
    label:   str | None = None  # tool name, if role == "tool"

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


class PromptBuilder:
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
        Assembles the canonical 7-slot prompt (static-first ordering).

        Returns
        -------
        (system_prompt, user_prompt)
            system_prompt : Slots 1a + 1b. Passed as the system= argument
                            to the runtime client. Byte-stable when persona
                            is unchanged — maximises KV-cache prefix reuse
                            across the system message.
            user_prompt   : Slots 3–7, assembled in stability order (most
                            stable first, instruction always last). Empty
                            slots are omitted cleanly — no label, no
                            whitespace. Passed as the user= argument to
                            the runtime client.
        """
        ...
```

**Enforcement rules:**
- `PromptBuilder` enforces all token ceilings internally. Callers do not
  truncate; they pass full content and let the builder enforce budgets.
- Empty optional slots produce no output — not an empty label, not
  whitespace, nothing.
- The builder is stateless. It is safe to call concurrently.

### 3.5 WikiAgent Prompt Exception

WikiAgent's system prompt is a protected contract: a compact XML-only
instruction block that must not be replaced by `PromptBuilder._SYSTEM` or
extended with a persona. WikiAgent calls `PromptBuilder.build()` with
`instruction=` only. The returned `system_prompt` is discarded; WikiAgent
passes its own `SYSTEM_PROMPT` constant to `runtime.infer()` directly.

This exception is intentional and permanent. The XML-only prompt is tuned
to the ingest task. General identity or persona context would pollute the
structured-output contract WikiAgent depends on.

---

## 4. Planner Routing Model

### 4.1 Design Principles

The Planner is a **rule engine**, not a classifier and not a free-form
inference call. It evaluates a priority-ordered set of conditions against
the instruction and current context. The first matching condition wins.

Inference is invoked sparingly. Priority 5 uses a deterministic keyword check rather than a model call — Gemma 4B requires `max_tokens ≥ 300` to produce reliable output on binary classification tasks, making inference-based routing too expensive for a per-turn call. See §4.3 for the updated Priority 5 implementation.

The Planner **never answers**. It produces a `RoutingPlan`. The
`ControllerAgent` executes the plan.

### 4.2 Priority-Ordered Decision Tree

Conditions are evaluated in strict priority order. The first match wins.
All lower priorities are skipped.

---

**PRIORITY 1 — INGEST SIGNAL**

| | |
|---|---|
| **Condition** | Explicit file path present in context (`raw_path` key) OR ingest keyword detected in instruction (`"ingest"`, `"process this file"`, `"add to wiki"`, `"index this"`) |
| **Action** | Route to `WikiAgent`. Set `fetch_rag = False`, `fetch_episodic = False`. |
| **Rationale** | Ingest is never ambiguous. Fast-pathing prevents any possibility of ResearchAgent or ConversationalAgent being scheduled as a follow-on, which was the source of a known routing bug. |

---

**PRIORITY 2 — EXPLICIT MEMORY COMMAND**

| | |
|---|---|
| **Condition** | Explicit memory signal detected in instruction: `"remember that"`, `"my preference is"`, `"that's wrong"`, `"the correct value is"`, `"forget that"`, `"mark complete"`, `"that's no longer true"` |
| **Action** | Route to `EpisodicMemoryWriter` first (extract and store). Then proceed to Priority 4 or 6 for the response. Set `write_episode = True`. |
| **Rationale** | These signals are deterministic and safe. No model judgment is needed for extraction. The memory write always precedes the response. |

---

**PRIORITY 3 — TOOL SIGNAL**

| | |
|---|---|
| **Condition** | Instruction requires information that cannot come from the corpus or episodic store. Web search keywords: `"latest"`, `"current"`, `"today"`, `"news"`, `"recent"`. File operation keywords: `"read"`, `"write"`, `"open"`, `"save"`, `"create a file"`. |
| **Action** | Dispatch the appropriate tool. Populate `RoutingPlan.tools_to_call`. Tool results will populate slot 6 before ConversationalAgent runs. |
| **Rationale** | Tool need is usually detectable from surface signals. Tool results are the freshest possible evidence and must be gathered before RAG to avoid stale corpus content taking precedence. |

---

**PRIORITY 4 — CORPUS SIGNAL**

| | |
|---|---|
| **Condition** | Instruction references a known project entity OR `MemoryManager.query_corpus()` returns results above the relevance threshold (default: `score >= 0.4`). |
| **Action** | Run RAG retrieval. Set `fetch_rag = True`. Snippets will populate slot 5. |
| **Rationale** | Default path for knowledge questions within LORA's domain. Cheaper and more reliable than episodic retrieval for factual questions. Always attempted before the episodic inference call. |

---

**PRIORITY 5 — EPISODIC RELEVANCE**

| | |
|---|---|
| **Condition** | None of the above triggered episodic retrieval, AND the instruction contains an episodic relevance keyword: `"preference"`, `"preferences"`, `"remember"`, `"remembered"`, `"you know about me"`, `"what do you know"`, `"decision"`, `"decisions"`, `"decided"`, `"correction"`, `"corrections"`, `"wrong"`, `"workflow"`, `"workflows"`, `"last time"`, `"previously"`, `"before"`, `"my project"`, `"my setup"`, `"my environment"`. |
| **Action** | Run episodic retrieval. Set `fetch_episodic = True`. Bullets will populate slot 4. |
| **Rationale** | Gemma 4B requires `max_tokens ≥ 300` to produce reliable output on binary yes/no classification tasks, making an inference-based routing call too expensive per turn. Deterministic keyword matching is faster, cheaper, and sufficiently accurate for the episodic relevance signal. The session flag (once episodic bullets have been injected, all subsequent turns return `fetch_episodic=True`) is preserved. When P4 fires, P5 is also evaluated and merged if it matches, producing `fetch_rag=True, fetch_episodic=True` as a compound plan. |

---

**PRIORITY 6 — DIRECT ANSWER**

| | |
|---|---|
| **Condition** | None of the above triggered. |
| **Action** | Route to `ConversationalAgent` with slots 1–3 only. |
| **Rationale** | General knowledge questions need no retrieval. The model answers from its own weights plus working memory. |

---

### 4.3 Priority 5 — Deterministic Episodic Relevance Check

Priority 5 uses a deterministic keyword check. No inference call is made.

**Implementation:** Scan the lowercased instruction for membership in `_EPISODIC_KEYWORDS` (defined in `planner.py`). First match wins. If any keyword is present, return `fetch_episodic=True`. If none match, return `None`.

**Session flag caching (preserved):** Once episodic bullets have been injected in a turn this session, `mark_episodic_injected()` is called and all subsequent Priority 5 checks return `fetch_episodic=True` without keyword evaluation. Relevance is assumed to persist within a session.

**P4+P5 compound merge:** When Priority 4 (corpus signal) fires, Priority 5 is also evaluated immediately after. If both match, the P4 `RoutingPlan` is updated: `fetch_episodic=True` is set on the existing plan rather than constructing a new one. This ensures queries that are both corpus-relevant and episodically relevant receive both slot 4 and slot 5 content.

**Why inference was removed:** Gemma 4B (`gemma-4-e4b-it-4bit`) requires `max_tokens ≥ 300` to produce reliable output on binary classification prompts. Below this threshold the model consistently returns a bare newline regardless of system prompt content or temperature. A 300-token budget for a yes/no routing decision is incompatible with the **Sparse** and **Predictable** constraints in §1.

### 4.4 RoutingPlan Structure

```python
@dataclass
class RoutingPlan:
    agent:             str            # "wiki_agent" | "conversational_agent"
    fetch_episodic:    bool           # True → retrieve from episodes table
    fetch_rag:         bool           # True → query_corpus() before responding
    tools_to_call:     list[str]      # tool names in dispatch order; [] if none
    write_episode:     bool           # True → EpisodicMemoryWriter runs first
    episode_type:      str | None     # type hint for extraction; None if not write
    compound:          bool           # True → multiple signal types detected
```

**Execution contract for `ControllerAgent.handle_task()`:**

1. Receive `RoutingPlan` from Planner.
2. If `write_episode`: run `EpisodicMemoryWriter`, wait for completion.
3. If `tools_to_call`: dispatch tools in listed order, collect results.
4. If `fetch_rag`: run `MemoryManager.query_corpus()`, collect snippets for Slot 4.
5. If `fetch_episodic`: run episodic retrieval, collect bullets for Slot 3.
6. Call `PromptBuilder.build()` with all collected content; persona is loaded
   from `_load_persona()` (cached) and passed as `persona=` for Slot 1b.
7. Call `RoutingPlan.agent` with the assembled prompt.

The persona fetch (formerly Step 4a) has been removed from the per-request
execution path. Persona is now a session-level cached value loaded on first
request by `_load_persona()` and injected into the system message via
`PromptBuilder.build(persona=...)`. This eliminates one corpus query per
turn and moves the persona from the volatile user message (old Slot 5) into
the stable system message prefix (Slot 1b).

The Planner never calls agents, never calls tools, and never touches the
database. It is pure decision logic.

### 4.5 Compound Instruction Handling

A compound instruction triggers two or more priority conditions simultaneously.

**Tool + Ingest compound**
Example: *"Search for the latest oMLX release notes and update the wiki."*

Triggers: Priority 1 (ingest) and Priority 3 (tool).

Resolution: Tool call executes first. Result is passed as `raw_path`
context to `WikiAgent`. The `RoutingPlan` sets
`tools_to_call = ["web_search"]`, `agent = "wiki_agent"`,
`compound = True`.

**Episodic + RAG compound**
Example: *"What did we decide about the vault resolver?"*

Triggers: Potentially Priority 4 (RAG, if documented) and Priority 5
(episodic, if stored as a decision).

Resolution: RAG runs first (Priority 4). If RAG returns results above
threshold, Priority 5 is skipped. If RAG returns nothing useful, Priority 5
runs. Double-fetching is never performed on well-documented topics.

**General compound rule:** When `compound = True`, the `ControllerAgent`
sequences execution in priority order. Higher-priority results populate
their slots first. Lower-priority fetches are skipped if the higher-priority
result fully resolves the information need.

---

### 4.6 Gemma 4B Behavioral Constraints

Live testing during Phase 7 revealed several behavioral constraints of the `gemma-4-e4b-it-4bit` model that affect prompt and inference call design. These are documented here as architectural constraints, not implementation details.

**Binary classification floor (`max_tokens`)**
Gemma 4B returns a bare newline (`'\n'`) on binary yes/no classification tasks when `max_tokens < 300`. This affects any inference call that expects a short, structured response. Confirmed thresholds:

| `max_tokens` | Result |
|---|---|
| 5–200 | `'\n'` only |
| 300 | Correct response, often with leading newline and markdown formatting (e.g. `'\n**Yes.**\n\n...'`) |

**Consequence:** All bounded inference calls in LORA that expect short output (routing classifiers, extractors) must use `max_tokens ≥ 200` or be replaced with deterministic Python logic. The preference is always deterministic Python over a model call for binary decisions.

**Extraction call minimum (`max_tokens`)**
The episodic extraction call (`extract_content_from_instruction`) requires `max_tokens = 200` to reliably produce a one-sentence output. The extraction system prompt is ~155 tokens, leaving ~45 tokens for the completion — sufficient for a single sentence.

**PromptBuilder `[USER]\n` wrapper incompatibility**
The `[USER]\n` slot label produced by `PromptBuilder.build()` combined with imperative instructions (e.g. `"Remember that..."`) causes Gemma 4B to return bare newlines on short-budget inference calls. Extraction calls that use bounded `max_tokens` must construct their user prompt directly rather than passing through `PromptBuilder.build()`. This is a documented architectural exception, analogous to WikiAgent's bypass of PromptBuilder for ingest prompts.

**Temperature**
`temperature = 0.0` produces degenerate output on extraction tasks. All bounded extraction calls use `temperature = 0.1` as the minimum viable value.

**Markdown output on constrained calls**
Even when Gemma 4B produces output on binary classification tasks, it may use markdown formatting (`**Yes.**`) rather than plain text. Parse logic must use `strip().lower()` and check `startswith()` rather than exact equality.

---

## 5. Build-Order Checklist

The dependency chain is strict. Each item depends on all items above it.
No item is begun until all items above it are complete and tested.

> **Session progress** — Phases 1–7 complete, plus KV-Cache Prompt Refactor.
> Test suite: **184 tests, 0 failures** across 7 test files.
> Files added/modified in Phases 1–7: `memory_manager.py` (extended), `prompt_builder.py` (extended),
> `planner.py` (extended), `episodic_extractor.py` (extended), `tool_dispatcher.py`,
> `controller_agent.py` (extended), `conversational_agent.py` (extended),
> `wiki_agent.py` (extended), `wiki/lora-persona.md` (new),
> `tests/test_memory_phase1.py`, `tests/test_planner_phase3.py`,
> `tests/test_controller_phase4.py`, `tests/test_episodic_phase5.py`,
> `tests/test_tool_dispatcher_phase6.py`, `tests/test_integration_phase7.py`.
>
> **Phase 7 live testing discoveries** — the following architectural corrections were made
> during live validation with Gemma 4B and are now reflected in §4.3 and §4.6:
> - `_CEIL_RAG` raised 400 → 450 tokens to accommodate `[CONTEXT]` header and `Source:` path overhead
> - Persona document (`wiki/lora-persona.md`) deduplication added to `_execute_plan` Step 4a
> - Episodic extraction bypasses `PromptBuilder` wrapper; uses direct prompt construction
> - `max_tokens` raised to 200 on all extraction inference calls
> - Priority 5 replaced with deterministic keyword check (Gemma 4B binary classification incompatibility)
> - P4+P5 compound merge added to `route()` so corpus-relevant + episodically-relevant queries receive both Slot 3 (episodic) and Slot 4 (RAG)
> - Implicit extraction gated by `_has_implicit_signal()` deterministic check before any inference call
>
> **KV-Cache Prompt Refactor** — the following changes were made after Phase 7 to maximise
> prefix-cache reuse across all inference backends. Reflected in §3:
> - Slot ordering redesigned: static-first, volatile-last (identity → persona → episodic → RAG → tools → working memory → instruction)
> - Persona moved from volatile user message (old Slot 5 RAG prepend) to stable system message (Slot 1b)
> - `PromptBuilder.build()` gained `persona: str | None` parameter; `working_memory` moved to last
> - `[USER]` label renamed `[INSTRUCTION]`; old Slot 2 retired; slots renumbered 3–7
> - `ControllerAgent._load_persona()` added: loads and caches persona once per session
> - Step 4a (per-request persona fetch + dedup) removed from `_execute_plan`
> - `tests/test_prompt_builder.py` (new file): 4 persona/ordering tests
> - 4 tests updated across `test_controller_phase4.py`, `test_tool_dispatcher_phase6.py`, `test_integration_phase7.py`

---

### Phase 1 — Memory Substrate

- [x] **1.1** Add `episodes` table to `MemoryManager` SQLite schema
- [x] **1.2** Write and run migration script against existing `lora_memory.db`
- [x] **1.3** Implement `EpisodicMemoryWriter`: insert, supersede, retract
- [x] **1.4** Implement `EpisodicMemoryReader`: all three retrieval modes (§2.6)
- [x] **1.5** Implement summarization contract (§2.7) as `format_episodic_summary()`
- [x] **1.6** Add `max_tokens` parameter to `get_context_window()` with 300-token ceiling
- [x] **1.7** Unit tests: lifecycle transitions, retrieval modes, summarization output

---

### Phase 2 — Prompt Contract

- [x] **2.1** Implement `PromptBuilder` class with all six slot methods
- [x] **2.2** Implement token ceiling enforcement for slots 3, 4, 5, 6
- [x] **2.3** Implement clean omission of empty optional slots (no empty labels)
- [x] **2.4** Replace prompt assembly in `ConversationalAgent` with `PromptBuilder.build()`
- [x] **2.5** Replace prompt assembly in `WikiAgent` with `PromptBuilder.build()`
- [x] **2.6** Unit tests: slot ordering, ceiling enforcement, empty slot omission, round-trip output

---

### Phase 3 — Planner Rewrite

- [x] **3.1** Implement `RoutingPlan` dataclass
- [x] **3.2** Implement Priority 1–4 as deterministic rule evaluations (no inference)
- [x] **3.3** Implement Priority 5 episodic relevance — deterministic keyword check (inference call removed; see §4.3 and §4.6)
- [x] **3.4** Implement Priority 6 direct answer fallback
- [x] **3.5** Implement compound instruction detection and sequencing
- [x] **3.6** Replace existing `Planner` inference-based routing with new rule engine
- [x] **3.7** Integration tests: each priority level fires correctly, compound cases sequence correctly

---

### Phase 4 — Controller Integration

- [x] **4.1** Update `ControllerAgent.handle_task()` to consume `RoutingPlan`
- [x] **4.2** Implement the 7-step execution contract (§4.4)
- [x] **4.3** Wire `PromptBuilder.build()` as the single prompt assembly point
- [x] **4.4** End-to-end integration test: ingest path, RAG path, direct answer path

---

### Phase 5 — Episodic Extraction Pipeline

- [x] **5.1** Implement deterministic signal detection (explicit memory commands)
- [x] **5.2** Implement model-based extraction call with `PromptBuilder`-conformant prompt
- [x] **5.3** Implement confidence scoring for model-extracted episodes (0.6–0.9 range)
- [x] **5.4** Wire extraction pipeline into `ControllerAgent` post-response hook
- [x] **5.5** Integration tests: explicit signals produce confidence=1.0 records, model
             extraction produces correctly typed and scored records

---

### Phase 6 — Tool Dispatcher

- [x] **6.1** Define `ToolResult` dataclass and tool dispatcher interface
- [x] **6.2** Implement `web_search` sub-agent (1–3 searches, structured results)
- [x] **6.3** Implement local file tools (read, write, append)
- [x] **6.4** Wire tool results into Slot 5 via `PromptBuilder`
- [x] **6.5** Integration tests: tool results appear in correct slot, token ceiling enforced

---

### Phase 7 — Final Integration

- [x] **7.1** Full pipeline test: instruction → Planner → fetches → PromptBuilder → agent → response
- [x] **7.2** Episodic extraction fires correctly on real conversations
- [x] **7.3** Working memory window enforces 300-token ceiling across session
- [x] **7.4** Persona loaded from wiki via RAG and injected into system message as Slot 1b (moved from Slot 5 RAG during KV-cache refactor; see session progress note above)
- [x] **7.5** All agents use `PromptBuilder.build()`. No agent assembles its own prompt string.
- [x] **7.6** Prompt logging enabled: every inference call writes its assembled prompt to debug log

---

*End of LORA Canonical Architecture Specification*
