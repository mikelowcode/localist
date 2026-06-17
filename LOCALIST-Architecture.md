# Localist Framework — Canonical Architecture Specification

> **Status: Authoritative**
> This document is the canonical reference for Localist Framework's substrate architecture.
> No implementation begins until it is reflected here. No deviation from this
> specification is made without updating this document first.

---

## Table of Contents

1. [System Identity](#1-system-identity)
2. [Episodic Memory Schema](#2-episodic-memory-schema)
3. [Unified Prompt Contract](#3-unified-prompt-contract)
4. [Planner Routing Model](#4-planner-routing-model)
5. [Fetcher Service](#5-fetcher-service)
6. [Build-Order Checklist](#6-build-order-checklist)
7. [Localist UI](#7-localist-ui)

---

## 1. System Identity

Localist Framework is a **local-first, agentic general assistant**. Every architectural
decision is evaluated against five constraints:

| Constraint | Meaning |
|---|---|
| **Local** | All inference, embeddings, memory, and tools run on-device. No cloud calls except explicit user-initiated web search or page fetch. |
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
| `subject` | TEXT | What the episode is about. Normalized to a clean third-person fact by the extraction pipeline. Used for exact-match retrieval and deduplication. |
| `content` | TEXT | The durable fact or event, in plain language. One sentence preferred. |
| `confidence` | REAL | 0.0–1.0. Code-extracted events = 1.0. Model-extracted events = 0.6–0.9. |
| `source` | TEXT | `"explicit"` for code-detected signals. `"model_extracted"` for inference-detected signals. |
| `task_id` | TEXT | The `task_id` of the originating request. Nullable. |
| `conversation_id` | TEXT | The originating conversation identifier. Nullable. |
| `project_context` | TEXT | Scopes retrieval. e.g. `"localist"`, `"general"`. Nullable defaults to `"general"`. |
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

### 2.8 Explicit Extraction Subject Normalization

When a Priority 2 explicit memory command fires (`"remember that"`,
`"my preference is"`, etc.), the raw instruction must not be stored as the
episode `subject`. The extraction pipeline normalizes it through the same
model-based extraction used by the implicit path:

- The already-normalized `content` string (output of `extract_content_from_instruction`)
  is used as the `subject` value (truncated to 80 chars).
- This ensures `subject` is always a clean third-person fact
  (e.g. `"The user's name is Michael."`) rather than the raw command
  (e.g. `"My name is Michael. Please remember that."`).
- Confidence for explicit episodes remains `1.0` regardless of normalization.
- If the model call fails, the pipeline falls back to the raw instruction — the
  write is never blocked.

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

**Purpose:** LORA's voice, style, tool awareness, and honor code.
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

**Persona structure (current `wiki/lora-persona.md`):**
- **Who You Are** — second-person voice, direct register, "thinking partner" framing
- **How You Work** — four-pillar trust hierarchy (tools → vault → episodic → priors)
- **Your Tools** — web search (LangSearch), page fetch (Fetcher service), file ops, wiki ingestion
- **Your Honor Code** — citation obligations, epistemic honesty, no hallucinated sources,
  one question at a time, memory consistency
- **What You Are Not** — internet access disclaimer, training cutoff boundary

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
- `lora-persona.md` is filtered from RAG results — it is already in the
  system message and must not appear twice in Slot 4.

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

**Purpose:** Relevant content from the wiki corpus and document index,
retrieved only when the user explicitly requests it.

**Token ceiling:** 800 tokens (800-token hard limit)

**Format:**
```
[CONTEXT]
Source: wiki/XML Parsing.md
{2–3 sentences of relevant content, not truncated mid-sentence}

Source: wiki/WikiAgent Architecture.md
{2–3 sentences of relevant content, not truncated mid-sentence}
```

**Rules:**
- This slot is **conditional**. Omitted entirely when Priority 4 does not fire.
- Priority 4 fires **only on explicit wiki/vault trigger keywords** — never
  on corpus scoring alone. See §4.2.
- Maximum 3 sources. Wiki sources are preferred over raw doc sources.
- When a source is truncated to fit the budget, the truncated content is suffixed
  with `… [truncated]` at the cut point (sentence boundary if possible, otherwise
  mid-content). This signals to the model that the source was incomplete so it can
  reflect that in its response rather than presenting a partial summary as complete.
- Content is truncated at a sentence boundary when possible.
- The `[CONTEXT]` label is mandatory. Source paths are mandatory.
- `lora-persona.md` is excluded from RAG results (already in system message).

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
- The `[TOOL RESULTS]` label is mandatory. Tool name and call parameters
  are mandatory for auditability.
- `url_fetch` results include title, source URL, word count, and full
  extracted text. PromptBuilder enforces the 500-token ceiling.

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

---

### 3.3 Aggregate Token Budget

| Slot | Label | Ceiling | Presence | Message |
|---|---|---|---|---|
| 1a — Identity | *(none)* | ~50 | Always | System |
| 1b — Persona | *(none)* | 500 | Conditional | System |
| 3 — Episodic memory | `[EPISODIC MEMORY]` | 150 | Conditional | User |
| 4 — RAG snippets | `[CONTEXT]` | 800 | Conditional | User |
| 5 — Tool results | `[TOOL RESULTS]` | 500 | Conditional | User |
| 6 — Working memory | `[WORKING MEMORY]` | 300 | Conditional | User |
| 7 — Instruction | `[INSTRUCTION]` | Uncapped | Always | User |
| **Worst-case total** | | **~2,300** | | |

Slot numbers 2 and the old Slot 2 label `[USER]` are retired. The gap
between 1b and 3 is intentional: slot numbering reflects cognitive role
and stability rank, not sequential position in the output string.

Gemma 4B quantized has an effective context window of approximately 8,000
tokens. The prompt contract consumes under 2,000 tokens in the worst case,
leaving substantial headroom for model output.

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
            system_prompt : Slots 1a + 1b. Byte-stable when persona is
                            unchanged — maximises KV-cache prefix reuse.
            user_prompt   : Slots 3–7, in stability order. Empty slots
                            are omitted cleanly — no label, no whitespace.
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

This exception is intentional and permanent.

---

## 4. Planner Routing Model

### 4.1 Design Principles

The Planner is a **rule engine**, not a classifier and not a free-form
inference call. It evaluates a priority-ordered set of conditions against
the instruction and current context. The first matching condition wins.

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
| **Rationale** | Ingest is never ambiguous. Fast-pathing prevents any possibility of incorrect agent scheduling. |

---

**PRIORITY 2 — EXPLICIT MEMORY COMMAND**

| | |
|---|---|
| **Condition** | Explicit memory signal detected: `"remember that"`, `"my preference is"`, `"that's wrong"`, `"the correct value is"`, `"forget that"`, `"mark complete"`, `"that's no longer true"` |
| **Action** | Route to `EpisodicMemoryWriter` first (extract and store, with subject normalization per §2.8). Then proceed to Priority 4 or 6 for the response. Set `write_episode = True`. |
| **Rationale** | These signals are deterministic and safe. The memory write always precedes the response. |

---

**PRIORITY 3 — TOOL SIGNAL**

| | |
|---|---|
| **Condition** | Web search keywords (`"latest"`, `"current price"`, `"current version"`, `"current ceo"`, `"current status"`, `"current rate"`, `"today"`, `"news"`, `"recent"`); OR file operation keywords (`"read the file"`, `"read file"`, `"write"`, `"open the file"`, `"save"`, `"create a file"`); OR URL fetch keywords (`"fetch this"`, `"fetch the url"`, `"read this link"`, `"read this url"`, `"open this link"`, `"summarize this url"`, `"summarize this link"`, `"extract this"`); OR any `http://` or `https://` URL present in the instruction. |
| **Action** | Dispatch appropriate tool(s). Populate `RoutingPlan.tools_to_call`. Tool results populate Slot 5 before ConversationalAgent runs. |
| **Rationale** | Tool results are the freshest possible evidence and must be gathered before any RAG retrieval. |
| **Notes** | All single-word keywords use `_any_whole_word()` with `\b` regex anchors to prevent substring false positives. Multi-word phrases (`"current version"`, `"read the file"`) carry no false-positive risk. The URL regex (`https?://`) automatically triggers `url_fetch` when any link is dropped into the instruction. |

---

**PRIORITY 3b — FACTUAL QUERY + CORPUS MISS**

| | |
|---|---|
| **Condition** | Instruction contains a factual query keyword (`"when did"`, `"what year"`, `"who founded"`, `"who invented"`, `"who created"`, `"where was"`, `"how many"`, `"what is the"`, `"which company"`, `"who was the first"`, `"what was the first"`) AND `MemoryManager.query_corpus()` returns no result with `relevance_score >= 0.55`. |
| **Action** | Schedule `web_search` via `tools_to_call`. Route to `ConversationalAgent`. |
| **Rationale** | Factual queries about the external world should go to web search when the corpus has no strong hit. Corpus is checked first to avoid unnecessary API calls when the answer is already in the vault. |
| **Notes** | Requires `MemoryManager` to be available. Skipped entirely when no MemoryManager is present. When corpus returns a hit (score ≥ 0.55), Priority 3b returns `None` and evaluation falls through to Priority 4. |

---

**PRIORITY 4 — CORPUS SIGNAL**

| | |
|---|---|
| **Condition** | **Path A:** Instruction contains an explicit wiki/vault trigger keyword (`"check the wiki"`, `"search the wiki"`, `"from the wiki"`, `"in the wiki"`, `"vault"`, etc.). **Path B:** `MemoryManager.query_corpus()` returns a top result with `relevance_score >= 0.55`. Either path is sufficient to match. |
| **Action** | Run RAG retrieval. Set `fetch_rag = True`. Snippets populate slot 4. Path A also sets `fetch_episodic = True`. Path B sets `fetch_episodic = False` (episodic is evaluated independently at P5). |
| **Rationale** | Path A keeps routing deterministic for explicit wiki requests. Path B restores score-based RAG injection for natural-language corpus queries that carry no trigger keyword (e.g. "summarize the Localist Master Project Outline"). Without Path B, ingested documents are unreachable unless the user knows to say "check the wiki". |

---

**PRIORITY 5 — EPISODIC RELEVANCE**

| | |
|---|---|
| **Condition** | Instruction contains a personal reference or episodic relevance keyword. Personal reference keywords (always return `fetch_episodic=True` immediately): `"my name"`, `"do you remember"`, `"who am i"`, `"what do you know about me"`, `"my preference"`, `"my setup"`, `"what did i tell you"`, `"what have i told you"`. General episodic keywords: `"preference"`, `"preferences"`, `"remember"`, `"remembered"`, `"you know about me"`, `"what do you know"`, `"decision"`, `"decisions"`, `"decided"`, `"correction"`, `"corrections"`, `"wrong"`, `"workflow"`, `"workflows"`, `"last time"`, `"previously"`, `"before"`, `"my project"`, `"my environment"`. |
| **Action** | Run episodic retrieval. Set `fetch_episodic = True`. Bullets populate Slot 3. |
| **Rationale** | Deterministic keyword matching is faster and cheaper than a model-based relevance call. Personal reference phrases are unambiguous and bypass keyword evaluation — they always fetch episodic memory. |
| **Session flag:** | Once episodic bullets have been injected this session, `mark_episodic_injected()` is called. **Session flag caching:** Once episodic bullets have been injected this session, the relevance inference call is skipped on subsequent turns — but keyword evaluation still runs. A turn with no episodic keyword returns `None` and falls through to P6. The flag suppresses the inference cost only, not the routing decision. |

---

**PRIORITY 6 — DIRECT ANSWER**

| | |
|---|---|
| **Condition** | None of the above triggered. |
| **Action** | Route to `ConversationalAgent` with Slots 1–3 only (system + working memory + instruction). |
| **Rationale** | General knowledge questions need no retrieval. The model answers from its own weights plus working memory. |

---

### 4.3 Priority 5 — Deterministic Episodic Relevance Check

Priority 5 uses a deterministic keyword check. No inference call is made.

**Implementation:** Scan the lowercased instruction for membership in
`_EPISODIC_KEYWORDS` and `_PERSONAL_REF_KEYWORDS` (defined in `planner.py`).
Personal reference keywords return `fetch_episodic=True` immediately.
General episodic keywords also return `fetch_episodic=True` on first match.

**Caching rule:**
- Once episodic bullets have been injected this session (`_episodic_injected = True`),
  the inference call is skipped on subsequent turns.
- Keyword evaluation still runs regardless of the flag. A turn with no matching
  episodic keyword returns `fetch_episodic = False` and falls through to P6.
- The flag suppresses inference cost only — it does not force `fetch_episodic = True`
  unconditionally.

**Why inference was removed:** Gemma 4B (`gemma-4-e4b-it-4bit`) requires
`max_tokens ≥ 300` to produce reliable output on binary classification
prompts. Below this threshold the model consistently returns a bare newline.
A 300-token budget for a yes/no routing decision is incompatible with the
**Sparse** and **Predictable** constraints in §1.

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
    priority:          int            # 1–6; which priority rule matched (default 6)
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

The Planner never calls agents, never calls tools, and never touches the
database. It is pure decision logic.

### 4.4a ControllerResult — API Response Schema

`ControllerAgent.handle_task()` returns a `ControllerResult` dict that is
serialised directly to the HTTP response by `main.py`.

```python
{
    "task_id":  str,
    "status":   "complete" | "failed",
    "answer":   str,
    "sources":  list[SourceItem],   # see below
    "metadata": ResponseMetadata,   # see below
    "error":    str | None,
}
```

**`SourceItem`** — typed source reference:
```python
{
    "path": str,              # absolute path on disk
    "type": "wiki" | "raw",  # classified by path prefix
    "name": str,              # human-readable title derived from filename
}
```

**`ResponseMetadata`** — routing provenance:
```python
{
    "agent":          str,         # agent that produced the answer
    "priority":       int,         # 1–6; which Planner rule matched
    "fetch_rag":      bool,        # True if RAG retrieval ran
    "fetch_episodic": bool,        # True if episodic memory was injected
    "tools_fired":    list[str],   # tool names that executed this turn
    "grounded":       bool,        # True if any corpus context was injected
}
```

This metadata is emitted in the SSE stream as the `"done"` event payload
and consumed by Localist UI's provenance bar (see §7).

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

Triggers: Potentially Priority 4 (explicit wiki query) and Priority 5
(episodic, if stored as a decision).

Resolution: Both `fetch_rag=True` and `fetch_episodic=True` are set on
the same `RoutingPlan`. Both retrievals run before the agent call.

### 4.6 Tool Dispatcher

The `ToolDispatcher` executes tool calls specified in a `RoutingPlan` and
returns `ToolResult` objects for injection into Slot 5.

**Registered tools:**

| Tool name | Trigger | Implementation |
|---|---|---|
| `web_search` | P3 web keywords or P3b factual + corpus miss | LangSearch API (`https://api.langsearch.com/v1/web-search`). Returns top 3 results as formatted bullets. Falls back to inference stub when `LANGSEARCH_API_KEY` is absent. Max 3 queries per dispatch call. |
| `file_op` | P3 file keywords (`"read the file"`, `"write"`, `"open the file"`, `"save"`, `"create a file"`) | Read, write, or append local files. All paths resolved relative to `project_root` and sandboxed — no path traversal outside `project_root` permitted. Max 4000 chars on read. |
| `url_fetch` | P3 URL fetch keywords or any `https?://` URL in instruction | HTTP POST to Fetcher service `/extract` endpoint (`http://localhost:8002/extract`). Returns title, source URL, word count, and full extracted text. PromptBuilder enforces Slot 5 ceiling. |

**LangSearch integration:**
- Endpoint: `POST https://api.langsearch.com/v1/web-search`
- Auth: `Authorization: Bearer {LANGSEARCH_API_KEY}` (from `backend/.env`)
- Request: `{"query": q, "summary": true, "count": 3, "freshness": "noLimit"}`
- Result format: `• {name}\n  {body[:300]}\n  [{displayUrl}]` per result
- Key loaded via `load_dotenv()` at server startup in `main.py`

### 4.7 Gemma 4B Behavioral Constraints

Live testing revealed several behavioral constraints of `gemma-4-e4b-it-4bit`
that affect prompt and inference call design. These are architectural
constraints, not implementation details.

**Binary classification floor (`max_tokens`)**
Gemma 4B returns a bare newline (`'\n'`) on binary yes/no classification
tasks when `max_tokens < 300`. All bounded inference calls that expect short
output must use `max_tokens ≥ 200` or be replaced with deterministic Python
logic. The preference is always deterministic Python over a model call for
binary decisions.

**Extraction call minimum (`max_tokens`)**
The episodic extraction call requires `max_tokens = 200` to reliably produce
a one-sentence output.

**PromptBuilder `[USER]\n` wrapper incompatibility**
The `[USER]\n` slot label combined with imperative instructions causes Gemma 4B
to return bare newlines on short-budget inference calls. Extraction calls
construct their user prompt directly rather than passing through
`PromptBuilder.build()`. This is a documented architectural exception.

**Temperature**
`temperature = 0.0` produces degenerate output on extraction tasks. All
bounded extraction calls use `temperature = 0.1` as the minimum viable value.

**Separate normalization prompt incompatibility**
A standalone normalization prompt (`max_tokens=60`, `temperature=0.1`)
reliably returns `'\n'` from Gemma 4B 4-bit — insufficient output budget
for the model to produce a complete sentence. Subject normalization must
derive from the already-normalized `content` string produced by the main
extraction call, not from a separate model call. See §2.8.

---

## 5. Fetcher Service

### 5.1 Overview

The Fetcher is a **standalone FastAPI microservice** running on port 8002.
It is separate from the main LORA backend (port 8001) and has no shared
code with it. The main backend's `ToolDispatcher` calls it over HTTP as
part of `url_fetch` tool execution.

**Start command (from `backend/` with venv activated):**
```bash
python -m uvicorn fetcher.main:app --host 127.0.0.1 --port 8002 --reload
```

**Directory layout:**
```
backend/fetcher/
├── __init__.py      (empty)
├── main.py          FastAPI app, lifespan, three endpoints
├── models.py        Pydantic request/response models
├── extractor.py     readability-lxml + lxml.html extraction logic
└── client.py        httpx async fetch logic
```

**Dependencies:** `httpx`, `readability-lxml`, `lxml` (all in venv).

### 5.2 Endpoints

**`POST /fetch`** — Raw HTTP fetch.
Returns status code, content-type, raw HTML, and response headers.
Used for debugging and inspection.

```
Request:  FetchRequest  { url, timeout=10.0, headers={} }
Response: FetchResponse { url, status_code, content_type, html,
                          headers, fetch_duration_ms }
```

**`POST /extract`** — Fetch + readability extraction.
Primary endpoint called by `ToolDispatcher`. Returns cleaned article text.
Full content returned — PromptBuilder enforces Slot 5 truncation.

```
Request:  ExtractRequest  { url, timeout=10.0 }
Response: ExtractResponse { url, title, author, date_published,
                            cleaned_text, word_count,
                            fetch_duration_ms, extractor_used }
```

**`POST /api`** — JSON REST endpoint fetch.
Strictly for `application/json` responses. Returns parsed JSON data.
Returns HTTP 422 if content-type is not `application/json`.

```
Request:  ApiRequest  { url, timeout=10.0, headers={} }
Response: ApiResponse { url, status_code, content_type, data,
                        fetch_duration_ms }
```

**`GET /health`** — Service health check.
Returns `{"healthy": true, "service": "localist-fetcher", "port": 8002}`.

### 5.3 Error Handling

All endpoints return structured `ErrorResponse` on failure. Never raises
through the endpoint boundary.

| Condition | `error_code` | HTTP status |
|---|---|---|
| DNS / connection failure | `connection_error` | 502 |
| Timeout | `timeout` | 504 |
| HTTP 4xx from target | `http_client_error` | 502 |
| HTTP 5xx from target | `http_server_error` | 502 |
| Readability extraction failed | `extraction_failed` | 422 |
| Non-JSON response on `/api` | `not_json` | 422 |

### 5.4 Implementation Notes

- **`readability-lxml` 0.8.4.1** expects a decoded string, not bytes.
  `html.decode("utf-8", errors="replace")` is applied before passing to
  `Document()`.
- **Browser-like User-Agent** is set in `client.py` to reduce bot-blocking
  on common sites.
- **Async model:** `httpx.AsyncClient` with `follow_redirects=True`. The
  service runs its own uvicorn event loop entirely independently of the
  main backend.
- **URL fetch UX note:** The URL regex in `_priority3_tool()` fires on any
  `http://` or `https://` in the instruction, enabling drop-a-link UX.
  Edge case: "what's the difference between http and https?" will also
  trigger `url_fetch`. Monitor in practice; fix if it becomes noisy by
  requiring explicit keyword + URL rather than either/or.

### 5.5 Environment Variable

```
LOCALIST_FETCHER_URL=http://localhost:8002   # in backend/.env
```

---

## 6. Build-Order Checklist

The dependency chain is strict. Each item depends on all items above it.
No item is begun until all items above it are complete and tested.

> **Session progress** — Phases 1–7 complete, plus KV-Cache Prompt Refactor,
> LangSearch integration, HTTP Fetcher service, Priority 4 rewrite,
> Priority 3b, persona rewrite, episodic memory bug fixes, Localist rebrand,
> and Localist UI overhaul (provenance bar, episodic memory panel, full rebrand),
> and Fetcher service restored (lxml, readability-lxml pinned in requirements.txt).
> Test suite: **184 tests, 0 failures** across 7 test files.
>
> **Files added/modified (all phases):**
> `memory_manager.py`, `prompt_builder.py`, `planner.py`,
> `episodic_extractor.py`, `tool_dispatcher.py`, `controller_agent.py`,
> `conversational_agent.py`, `wiki_agent.py`, `main.py`,
> `wiki/lora-persona.md`, `backfill_embeddings.py`, `embedding_engine.py`,
> `fetcher/__init__.py`, `fetcher/main.py`, `fetcher/models.py`,
> `fetcher/client.py`, `fetcher/extractor.py`,
> `tests/test_memory_phase1.py`, `tests/test_prompt_builder.py`,
> `tests/test_planner_phase3.py`, `tests/test_controller_phase4.py`,
> `tests/test_episodic_phase5.py`, `tests/test_tool_dispatcher_phase6.py`,
> `tests/test_integration_phase7.py`.
>
> **Post-Phase-7 architectural changes (all reflected above):**
>
> *KV-Cache Prompt Refactor:*
> - Slot ordering redesigned: static-first, volatile-last
> - Persona moved to stable system message (Slot 1b)
> - `PromptBuilder.build()` gained `persona=` parameter
> - `[USER]` label renamed `[INSTRUCTION]`; slots renumbered 3–7
> - `ControllerAgent._load_persona()` added: loads and caches persona once per session
> - KV-cache efficiency: 79.7% confirmed in live session (oMLX dashboard)
>
> *LangSearch integration:*
> - `ToolDispatcher._execute_single_search()` replaced with real LangSearch HTTP call
> - `load_dotenv()` added to `main.py` so `LANGSEARCH_API_KEY` loads at server startup
> - `LANGSEARCH_API_KEY` added to `backend/.env`
>
> *Priority 3b:*
> - New priority between P3 and P4: factual keyword + corpus miss → web search
> - `_FACTUAL_QUERY_KEYWORDS` frozenset added to `planner.py`
> - `_priority3b_factual()` method added to `Planner`
>
> *Priority 4 rewrite:*
> - Scoring-based RAG injection eliminated entirely
> - Priority 4 now fires only on explicit wiki/vault trigger keywords
> - `_WIKI_QUERY_KEYWORDS` frozenset added; `_priority4_corpus()` rewritten
> - `_CORPUS_SCORE_THRESHOLD` no longer used for routing (retained for P3b)
>
> *`_WEB_SEARCH_KEYWORDS` expansion:*
> - `"current"` replaced with multi-word phrases: `"current price"`, `"current version"`,
>   `"current ceo"`, `"current status"`, `"current rate"`
> - `_any_whole_word()` helper added with `\b` regex anchors for single-word keywords
>
> *`_FILE_OP_KEYWORDS` fix:*
> - `"read"` replaced with `"read the file"`, `"read file"` to prevent false
>   positive on `"read this link"` / `"read this URL"`
> - `"open"` replaced with `"open the file"` (same pattern)
>
> *HTTP Fetcher service:*
> - Standalone FastAPI microservice on port 8002
> - Three endpoints: `/fetch`, `/extract`, `/api`; plus `/health`
> - `url_fetch` tool added to `ToolDispatcher`
> - `_FETCH_KEYWORDS` frozenset + URL regex added to `_priority3_tool()`
> - `LOCALIST_FETCHER_URL` added to `backend/.env`
>
> *Persona rewrite:*
> - `wiki/lora-persona.md` rewritten: second-person voice, trust hierarchy,
>   tool awareness (LangSearch + Fetcher), honor code
> - Removed third-person documentation register; added direct behavioral instructions
>
> *Episodic memory bug fixes:*
> - Priority 5 personal reference keywords added: `"my name"`, `"do you remember"`,
>   `"who am i"`, `"what do you know about me"`, etc.
> - Explicit extraction subject normalization: `subject` now derived from normalized
>   `content` string, not raw instruction (see §2.8)

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

- [x] **2.1** Implement `PromptBuilder` class with all seven slot methods
- [x] **2.2** Implement token ceiling enforcement for slots 3, 4, 5, 6
- [x] **2.3** Implement clean omission of empty optional slots (no empty labels)
- [x] **2.4** Replace prompt assembly in `ConversationalAgent` with `PromptBuilder.build()`
- [x] **2.5** Replace prompt assembly in `WikiAgent` with `PromptBuilder.build()`
- [x] **2.6** Unit tests: slot ordering, ceiling enforcement, empty slot omission, round-trip output

---

### Phase 3 — Planner Rewrite

- [x] **3.1** Implement `RoutingPlan` dataclass
- [x] **3.2** Implement Priority 1–4 as deterministic rule evaluations (no inference)
- [x] **3.3** Implement Priority 5 episodic relevance — deterministic keyword check
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
- [x] **5.2** Implement model-based extraction call with direct prompt construction
- [x] **5.3** Implement confidence scoring for model-extracted episodes (0.6–0.9 range)
- [x] **5.4** Wire extraction pipeline into `ControllerAgent` post-response hook
- [x] **5.5** Integration tests: explicit signals produce confidence=1.0 records, model
             extraction produces correctly typed and scored records
- [x] **5.6** Subject normalization: explicit extraction derives subject from normalized
             content string, not raw instruction (§2.8)

---

### Phase 6 — Tool Dispatcher

- [x] **6.1** Define `ToolResult` dataclass and tool dispatcher interface
- [x] **6.2** Implement `web_search` tool — LangSearch API integration
- [x] **6.3** Implement `file_op` tool (read, write, append — sandboxed)
- [x] **6.4** Implement `url_fetch` tool — calls Fetcher service `/extract`
- [x] **6.5** Wire tool results into Slot 5 via `PromptBuilder`
- [x] **6.6** Integration tests: tool results appear in correct slot, token ceiling enforced

---

### Phase 7 — Final Integration

- [x] **7.1** Full pipeline test: instruction → Planner → fetches → PromptBuilder → agent → response
- [x] **7.2** Episodic extraction fires correctly on real conversations
- [x] **7.3** Working memory window enforces 300-token ceiling across session
- [x] **7.4** Persona loaded from wiki and injected into system message as Slot 1b
- [x] **7.5** All agents use `PromptBuilder.build()`. No agent assembles its own prompt string.
- [x] **7.6** Prompt logging enabled: every inference call writes its assembled prompt to debug log

---

### Fetcher Service

- [x] **F.1** Standalone FastAPI service on port 8002
- [x] **F.2** `POST /fetch` endpoint — raw HTTP fetch
- [x] **F.3** `POST /extract` endpoint — readability extraction
- [x] **F.4** `POST /api` endpoint — JSON REST fetch
- [x] **F.5** `url_fetch` tool wired into `ToolDispatcher`
- [x] **F.6** URL regex + explicit keyword triggers in `_priority3_tool()`
- [x] **F.7** End-to-end verified: GitHub release URL fetched and summarized correctly

---

---

## 7. Localist UI

### 7.1 Overview

Localist UI is the SvelteKit frontend sub-product. It communicates with
the Localist backend exclusively via the REST/SSE API on port 8001. All
rendering is client-side; the backend has no knowledge of the UI.

**Tech stack:** SvelteKit, TypeScript, IBM Plex Sans / IBM Plex Mono,
CSS custom properties (no Tailwind, no component library).

**Directory:** `localist-ui/` at the project root.

**Dev server:** `npm run dev` from `localist-ui/` (port 5173).

### 7.2 Routes

| Route | Component | Purpose |
|---|---|---|
| `/conversation` | `ChatPanel.svelte` | Primary chat interface. Streams SSE responses. |
| `/memory` | `EpisodesPanel.svelte` | Episodic memory browser. |
| `/files` | FileBrowser | Raw and wiki file listing; file upload; wiki ingest trigger. |
| `/settings` | Settings | Runtime status, version info. |

### 7.3 Provenance Bar

Every completed assistant turn renders a **provenance bar** between the
response body and the source chips. It is driven by the `metadata` field
in the SSE `"done"` event.

**Chips rendered:**

| Chip | Condition | Colour |
|---|---|---|
| `P1 · Direct` | `priority === 1` | Muted |
| `P2 · Memory write` | `priority === 2` | Green |
| `P3 · Web search` | `priority === 3` | Blue |
| `P4 · Vault` | `priority === 4` | Purple |
| `P5 · Episodic` | `priority === 5` | Amber |
| `P6 · Inference` | `priority === 6` | Muted |
| `⚙ {tool_name}` | each entry in `tools_fired` | Orange |
| `◎ episodic` | `fetch_episodic === true` | Amber |
| `◈ grounded` | `grounded === true` | Green |

Source chips (wiki/raw type + human-readable name) are rendered below the
provenance bar from the `sources` array.

### 7.4 Episodic Memory Panel

The `/memory` route renders `EpisodesPanel.svelte`, which calls
`GET /api/memory/episodes` on mount and on manual refresh.

**State management:** `localist-ui/src/lib/stores/episodes.ts`
- `episodesStore` — writable store with episodes list, loading/error state,
  pagination, and active type filter
- `loadEpisodes(opts)` — fetches from `/api/memory/episodes` with optional
  `episode_type`, `offset`, `limit` query params
- `EPISODE_TYPES`, `TYPE_LABELS`, `TYPE_COLORS` — constants for the 7
  canonical episode types

**Episode card fields displayed:** type chip (colour-coded), subject,
date, content, confidence percentage, project context, source.

**Type filter bar:** All | Preference | Correction | Decision | Workflow |
Fact | Relationship | Context

### 7.5 API Proxy

Localist UI proxies all `/api/*` requests to `http://localhost:8001` via
the Vite dev server config. Production deployments should configure an
equivalent reverse proxy. The `/api` prefix is stripped before forwarding.

---

*End of Localist Framework Canonical Architecture Specification*
