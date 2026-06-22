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
8. [Graph Retrieval Layer](#8-graph-retrieval-layer)
9. [Slot 6A — Structured Working State](#9-slot-6a--structured-working-state)

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

#### Cache eligibility under the current runtime contract

The table above describes conceptual content stability, not cache eligibility under the current runtime contract.

Per the oMLX single-turn finding (detailed fully in §3.7, now a resolved finding), only Slot 1a + Slot 1b (the system message) are byte-identical across separate HTTP requests today. Slots 3a–7 (the user message) are structurally single-shot per `OMLXRuntimeClient.infer()` and cannot participate in cross-request prefix matching regardless of internal ordering or byte-stability.

The canonical vocabulary for this distinction: **stable prefix** (system message: identity + persona) and **dynamic suffix** (user message: all of Slots 3a–7). These terms are used consistently in §3.7a and forward.

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
As of the 2026-06-20 rewrite, `wiki/lora-persona.md` is five plain prose sentences with no internal section headers (~476 chars / ~119 tokens, roughly 24% of the 500-token hard ceiling). Persona content is intentionally undifferentiated prose rather than a fixed section template — this description should not be treated as a contract that future personas must follow.

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
- `_load_persona()` fetches top-3 corpus results and filters by
  `"lora-persona" in str(d.path)` before accepting any document into
  Slot 1b. If `lora-persona.md` is not in the top-3 results, persona
  is absent for that session and a warning is logged.
- Persona content is no longer required to stay minimal. It may grow, as plain undifferentiated prose with no internal section headers, to absorb durable, non-instruction-dependent behavioral content (the "static rules" referenced in §3.7a), up to the existing 500-token hard ceiling. No soft checkpoint or review threshold applies below that ceiling.

---

#### Slot 3 — Episodic Memory + User Profile

**Purpose:** Durable facts about the user, project, and preferences (Slot 3a —
episodic bullets) and relevant lines from the user profile document (Slot 3b —
user profile facts).

**Token ceiling:** 250 tokens total. Two independent sub-budgets enforced by
`PromptBuilder._slot3_combined()`: 150 tokens for episodic bullets (Slot 3a),
100 tokens for user profile facts (Slot 3b).

**Format:**
```
[EPISODIC MEMORY]
- {content} ({episode_type}, {confidence:.1f})

[USER PROFILE]
- {fact line}
```

**Rules:**
- This slot is **conditional**. It is omitted entirely when both sub-blocks
  are empty. No empty label, no placeholder.
- Slot 3a (episodic): relevance determined by Priority 5. See §4.
- Slot 3b (profile): injected on P4, P4a, P5 routes and any turn where
  episodic bullets fire. See §3.6.
- When episodic bullets injected: 3–5 bullets, confidence ≥ 0.7, type-ordered per §2.7.
- Both `[EPISODIC MEMORY]` and `[USER PROFILE]` labels are mandatory when
  their respective sub-block is present.
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

#### Slot 5b — Graph Result

**Purpose:** Structured answer to a P3c graph-query turn — which pages link
to a target page (incoming), or which pages a target page links to (outgoing).
Carries the complete graph-query answer to the model so the response can be
grounded in real graph state rather than generated from weights alone.

**Token ceiling:** 300 tokens

**Format (incoming direction):**
```
[GRAPH RESULT]
Pages linking to {page}:
- {page_name}
```

**Format (outgoing direction):**
```
[GRAPH RESULT]
{page} links to:
- {page_name}
{page} also references a page that does not exist:
- "{link_text}" (no matching page found)
```

When zero edges exist, the content is a single declarative sentence:
`No pages link to {page}.` or `{page} does not link to any other pages.`

**Rules:**
- **Exception to the clean-omission contract.** This slot renders whenever a
  graph query resolved a target page this turn — even when the edge list is
  empty. Zero edges is a real, correct answer that must be visible to the model.
  The only omission case is `graph_result is None`, meaning no graph query
  resolved this turn. Implemented in `_slot_graph()` in `prompt_builder.py`.
- **Mutual-exclusivity with RAG, episodic, profile, and tools.** P3c
  graph-query turns never co-render with Slot 3 (episodic/profile), Slot 4
  (RAG), or Slot 5 (tool results). This guarantee is not enforced by
  `PromptBuilder` — it falls out of the `RoutingPlan` produced by
  `planner._priority3c_graph_query()`, which sets `fetch_rag=False`,
  `fetch_episodic=False`, `force_rag=False`, and `tools_to_call=[]`.
  `PromptBuilder` itself has no exclusivity guard.
- Ceiling enforced as a single post-render truncation via `_truncate_to_tokens()`.
- Conditional: omitted entirely (no label, no whitespace) when `graph_result is None`.

---

#### Slot 6A — Structured Working State

**Purpose:** Deterministic, per-turn working context — what RAG sources were
active this turn and (when Tier 2 rendering is wired in) what the model-assisted
session state is. Positioned after Slot 5b and before Slot 6 to place
state-carrying content after retrieval evidence and before conversational
scaffolding.

**Token ceiling:** 100 tokens

**Format:**
```
[WORKING STATE]
current_project: {current_project}
active_artifacts: {artifact1}, {artifact2}, ...
```

Each line is emitted only when its field is non-empty/non-None.
`active_artifacts` is truncated by dropping entries from the end until the
100-token ceiling is met.

**Tier 1 fields (deterministic — implemented and rendering):**
- `current_project` — derived from `task.context["project_context"]` in
  `controller_agent.py` Step 5d. Currently always `None` at all real call
  sites: `project_context` is a DB-scoping key (e.g., `"general"` or a project
  slug), not a human-readable project name. The Step 5d code explicitly sets
  `current_project = None` when the value is `None` or `"general"`. The field is
  wired and ready; it will activate without a code change once a real
  project-name source is available at this context key.
- `active_artifacts` — `[s.path for s in rag_sources]`: the filesystem paths of
  RAG documents retrieved during the same turn. Non-empty only on turns where
  RAG retrieval ran (P4, P4a, P5 routes).

**Tier 2 fields (model-assisted — updating, not yet rendering):**
Three fields — `current_focus`, `open_loops`, `recent_decisions` — are updated
post-response via `process_working_state_update()` in `episodic_extractor.py`
and persisted in the `working_state` SQLite table via `WorkingStateStore`. These
fields do **not** yet appear in the rendered `[WORKING STATE]` output. Wiring
them into this slot is a separate, future step.

The Tier 2 update runs after every completed turn, **regardless of route** —
including P3c (graph-query) turns that are excluded from rendering. A graph-query
turn still produces real conversational state worth persisting for the next
turn's render. The update-vs-render distinction is deliberate: suppressing the
update on P3c turns would discard valid session state to maintain a boundary that
only applies to rendering.

**Rules:**
- **Clean omission.** Returns `""` when `state` is `None` or when both
  `current_project` is falsy and `active_artifacts` is empty.
- **P3c render-gating.** This slot does **not** render on P3c (graph-query)
  routes. The gate is in `controller_agent.py` Step 5d:
  `if plan.graph_query is None:`. Rationale: P3c is a deliberate
  no-interpretation zone — structural, non-personalized, deterministic relative
  to graph state. Slot 6A is interpretive and state-carrying by definition.
  Co-rendering on the same P3c turn would blur that boundary by accident rather
  than by deliberate future design.
- Implemented in `_slot6a_working_state()` in `prompt_builder.py`.
  Input dataclass: `WorkingMemoryState(current_project: str | None,
  active_artifacts: list[str])`.

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
| 3a — Episodic memory | `[EPISODIC MEMORY]` | 150 | Conditional | User |
| 3b — User profile | `[USER PROFILE]` | 100 | Conditional | User |
| 4 — RAG snippets | `[CONTEXT]` | 800 | Conditional | User |
| 5 — Tool results | `[TOOL RESULTS]` | 500 | Conditional | User |
| 5b — Graph result | `[GRAPH RESULT]` | 300 | Conditional | User |
| 6A — Working state | `[WORKING STATE]` | 100 | Conditional | User |
| 6 — Working memory | `[WORKING MEMORY]` | 300 | Conditional | User |
| 7 — Instruction | `[INSTRUCTION]` | Uncapped | Always | User |
| **Worst-case total** | | **~2,850** | | |

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

### 3.6 User Profile

`ControllerAgent` maintains a per-session user profile sourced from
`backend/wiki/users/michael.md`. This file is hand-authored (not produced
by WikiAgent) and contains structured fact lines in five sections: Identity,
Active Projects, Preferences, Working Patterns, and Decisions.

Each section has a maximum of 5 lines to enforce minimalism and prevent
prompt bloat. The ideal injection returns only the top relevant facts
for the current instruction, not the entire profile.

#### Embedding and scoring

On first request after startup, `ControllerAgent._load_user_profile()`
reads the file, strips section headers and blank lines, and embeds each
remaining fact line using `ControllerAgent._embed()` — which delegates
to `MemoryManager._embed_fn` (the EmbeddingEngine callable). Embeddings
are stored in parallel lists (`_profile_lines`, `_profile_embeddings`)
and cached for the session.

Per-turn scoring: `_score_profile_facts(instruction_embedding, top_n=5,
threshold=0.45)` computes cosine similarity between the instruction
embedding and each cached fact embedding. Lines scoring ≥ 0.45 are
returned as `UserProfileFact` objects, sorted by score descending,
capped at 5.

The 0.45 threshold is lower than the 0.55 RAG corpus threshold because
profile fact lines are short and produce lower raw cosine similarity
against full instruction embeddings even when semantically relevant.

#### Injection trigger

Profile facts are injected into Slot 3b on any turn where:
- `plan.fetch_rag` is True (P4 and P4a routes)
- `plan.fetch_episodic` is True (P5 routes)
- Episodic bullets fired in the same turn

P6 direct-answer turns do not inject profile facts.

#### Update path

Manual for now: edit `wiki/users/michael.md` directly and restart the
backend to reload the cache. Automatic promotion from episodic memory
is planned as part of the graph retrieval layer (future session).

---

### 3.7 Resolved: Prefix Stability Is a Structural Property of the Stable Prefix, Not a User-Message Defect

The §3.1 design principle ("static before dynamic, stable before volatile") describes the intended ordering. The investigation in the 2026-06-20 session established that the zero-cache-hit result on the user turn (see Observed evidence below) is an **expected, structural property** of the current `infer()` contract — not a defect in `PromptBuilder`'s slot ordering.

#### Mechanism — routing-dependent slot presence

Slots 3–6 are all conditional. The first slot that actually appears in the user message
varies by routing path:

| Routing path | First user slot |
|---|---|
| P6 (direct answer) | `[WORKING MEMORY]` (if non-empty), else `[INSTRUCTION]` |
| P4 / P4a (RAG) | `[USER PROFILE]` or `[EPISODIC MEMORY]` (if either fires), else `[CONTEXT]` |
| P5 (episodic) | `[EPISODIC MEMORY]` |

A P6 turn followed by a P4 turn produces a different first byte in the user message.
Because KV-cache prefix matching must be exact from byte 0, this is a complete cache
miss for the user turn — not a partial miss from the point of divergence. This variability is expected and accepted under the dynamic-suffix model — it is no longer being treated as a defect to fix via reordering.

#### Compounding factor — similarity-scored profile selection

Even when Slot 3 is present on two consecutive turns (both are P4 routes, both fire
`[USER PROFILE]`), the Slot 3 content is not stable. `_score_profile_facts()` scores
each fact line by cosine similarity against the **current instruction embedding**.
Different instructions → different top-5 facts → different Slot 3 bytes → cache miss
at byte 1 of the user message, even within the same routing path. This per-turn variation is expected and accepted under the dynamic-suffix model — profile facts are deliberately instruction-dependent by design, and freezing them into a stable prefix would defeat their purpose.

#### Observed evidence (2026-06-19 live session)

Three-turn live session. Stable system message: `system_chars = 403` (≈ 101 tokens). *(Historical — this figure predates the persona rewrite completed later in the 2026-06-20 session. At current persona size, the system message would be approximately ~40 tokens identity + ~119 tokens persona ≈ ~159 tokens / ~636 chars under the same `len // 4` estimation convention used elsewhere in the codebase.)*
User turn longest common prefix (LCP):

| Turn pair | User LCP | Cause |
|---|---|---|
| T1 → T2 | 1 char | Slot 3/4 absent T1, present T2 — first byte differs |
| T2 → T3 | 1 char | Slot 3 content changed (different profile facts scored) |

oMLX dashboard: **zero KV-cache hits** across all three turns. The 403-char system
message may cache at the system-turn level (oMLX caching behavior not confirmed from
the codebase), but the user turn achieves no prefix reuse. This zero-cache-hit result on the user turn is now understood to be **expected**, not anomalous — see Root cause below.

#### What this is not

This is not a regression introduced by any change in the 2026-06-19 session. The
session_id working-memory fix and the RAG frontmatter-stripping fix are independently
verified and correct. The prefix-stability gap pre-dates this session and results from
the conditional-slot architecture as originally designed.

#### Root cause — oMLX single-turn request shape

`OMLXRuntimeClient.infer_stream()` sends exactly one system message + one user message per HTTP call, with no message-history accumulation and no session identifier in the payload. This was verified by reading `omlx_runtime_client.py` directly — there is no cache-control parameter, session field, or accumulated `messages` array anywhere in the request construction. The backend therefore has no mechanism to recognize a request as a continuation of a prior call. The system message is the only content that can ever be compared byte-for-byte against a previous request, and it is already stable (cached per-session via `ControllerAgent._load_persona()`). The user message can never achieve cross-request prefix reuse under this contract, independent of slot ordering.

#### Status

Resolved as a documentation/framing correction. No code change was required — `PromptBuilder.build()`'s existing slot order already matches the dynamic-suffix ordering this finding converged on. See §3.7a for the stable-prefix / dynamic-suffix contract and §3.7b for the future APC direction.

#### Design direction (historical record — both options superseded)

Two approaches were identified before the root-cause finding; tradeoffs were not evaluated:

1. **Emit all conditional slots with empty-state markers.** When Slot 3 is absent,
   emit `[EPISODIC MEMORY]\n(none)` rather than omitting the slot entirely. This
   guarantees the user message starts with the same bytes regardless of routing path —
   at the cost of a small fixed token overhead on every P6 turn.

2. **Reorder volatile content after a longer stable prefix.** Place a
   routing-invariant block first in the user message so that conditional slots always
   follow a guaranteed-stable prefix. Preserves clean omission but requires defining
   and maintaining a new static anchor that is byte-identical across all routing paths.

Both options are **superseded by the single-turn-request finding**, not rejected on their own merits. Both were designed to fix user-message prefix instability for a backend capable of comparing the user message across requests; since `OMLXRuntimeClient.infer()` structurally cannot make such a comparison today, neither option would produce any measurable effect. Retained here as a historical record of the options considered.

---

### 3.7a Stable Prefix / Dynamic Suffix Contract

This contract is the canonical boundary between cached and per-turn content. It is enforced by an automated test (`tests/test_prompt_builder.py`, `test_pb_e_build_enforces_dynamic_suffix_slot_order`) so that any future change to slot order is a deliberate, reviewed decision rather than a silent regression.

**Stable prefix** (system message, passed as `system=`): Slot 1a (identity) + Slot 1b (persona). Byte-identical for the lifetime of a session once persona is cached by `ControllerAgent._load_persona()`. No new slot is introduced. "Static rules" is not a separate artifact — it denotes invariant scaffolding that may be written directly into `lora-persona.md` as plain prose, blended with voice and style content, with no internal section headers and no structural separation from the rest of the persona text. Persona may grow to absorb this kind of durable, non-instruction-dependent content, with no soft checkpoint or review threshold below the cap. The only constraint is the existing hard ceiling, `_CEIL_PERSONA = 500` tokens / 2000 chars in `prompt_builder.py`, which is unchanged and still governs KV-cache prefix stability. Current actual persona size: ~476 chars / ~119 tokens (roughly 24% of the cap) — substantial headroom (~381 tokens) exists.

**Dynamic suffix** (user message): Slot 3a (episodic) → Slot 3b (profile) → Slot 4 (RAG) → Slot 5 (tool results) → Slot 6 (working memory) → Slot 7 (instruction). This order is unchanged from the existing implementation and is preserved deliberately — it reflects conceptual layering (contextualizers → evidence providers → conversation scaffolding → instruction), not cache eligibility. Episodic and profile facts are *not* eligible to move into the stable prefix: profile is re-scored per turn via live cosine similarity; episodic presence is gated by routing path and session state. Freezing either into the prefix would defeat their purpose.

---

### 3.7b Future Direction: APC Layer for MLX-Based Engines

*(Future / unscheduled — direction statement only, no build sequence, no priority.)*

Localist is architected toward a future automatic/algorithmic prefix caching (APC) layer intended to work across MLX-based inference engines (oMLX, MLX-LM, and potentially other MLX-backed runtimes), either as a capability those engines add natively or as a plugin Localist builds itself.

The stable-prefix/dynamic-suffix contract locked in §3.7a is the groundwork for that future layer: if a future engine or plugin sends a growing multi-turn `messages` array (rather than today's single system+user shot) or otherwise gains the ability to compare prefixes across requests, the dynamic-suffix ordering already locked here becomes the correct stable-history/dynamic-tail shape for it to exploit, with no further prompt-assembly redesign needed.

This is explicitly a forward-looking architectural bet, not a fix for a currently-observed metric. No current backend in this codebase can produce a user-message cache hit; this is expected and accepted (§3.7) and should not be treated as a regression in future live-testing sessions.

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

**PRIORITY 4a — IDENTITY TRIGGER**

| | |
|---|---|
| **Condition** | Instruction contains any keyword from `_IDENTITY_KEYWORDS` (whole-phrase match via `_any_whole_word()`): `"who are you"`, `"what are you"`, `"tell me about yourself"`, `"what can you do"`, `"are you an ai"`, `"are you a bot"`, `"what is lora"`, `"who is lora"`, `"what is localist"`, `"are you made by google"`, `"are you chatgpt"`, `"are you gemma"`, `"introduce yourself"`. |
| **Action** | Route to `conversational_agent`. Set `fetch_rag = True`, `force_rag = True`. `how-localist-works.md` is injected into Slot 4 regardless of embedding score. |
| **Rationale** | Without this priority, identity questions fall to P6 (direct answer). Gemma 4B's RLHF fine-tuning then overrides the system prompt and produces "I'm a large language model made by Google." P4a forces RAG retrieval of `how-localist-works.md`, giving the model explicit first-person identity context. |
| **Notes** | Fires before Priority 4 (explicit wiki/vault) so identity questions are never absorbed by general corpus routing. Does not set `fetch_episodic=True`. A trailing-content guard prevents false positives: if the matched keyword is followed by further meaningful words (e.g. `"what can you do with this file"`), the match is discarded. |

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

#### Priority 4a — Identity trigger

**Match condition:** Instruction contains any keyword from
`_IDENTITY_KEYWORDS` (whole-word match via `_any_whole_word()`).

**Effect:** Routes to `conversational_agent` with `fetch_rag=True` and
`force_rag=True`. The `force_rag` flag bypasses the `relevance_score >=
0.55` threshold in the RAG filter, guaranteeing that `how-localist-works.md`
is injected into Slot 4 regardless of embedding similarity score.

**Purpose:** Prevents identity questions from falling to P6 (direct answer),
where Gemma 4B's RLHF fine-tuning overrides the system prompt and produces
"I'm a large language model made by Google." With `how-localist-works.md`
in Slot 4, the model has explicit first-person identity context to draw from.

**Keywords (`_IDENTITY_KEYWORDS`):**
`"who are you"`, `"what are you"`, `"tell me about yourself"`,
`"what can you do"`, `"are you an ai"`, `"are you a bot"`,
`"what is lora"`, `"who is lora"`, `"what is localist"`,
`"are you made by google"`, `"are you chatgpt"`, `"are you gemma"`,
`"introduce yourself"`

**Implementation notes:**
- Uses `_any_whole_word()` for whole-phrase matching — prevents false
  positives (e.g. `"what can you do with this file"` does not trigger).
- Fires before Priority 4 (explicit wiki/vault) so identity questions
  are never absorbed by general corpus routing.
- Does not set `fetch_episodic=True` — identity questions do not require
  episodic context.

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
    force_rag:         bool           # True → bypass relevance_score >= 0.55 RAG threshold (set by P4a)
    priority:          int            # 1–6; which priority rule matched (default 6)
```

**Execution contract for `ControllerAgent.handle_task()`:**

1. Receive `RoutingPlan` from Planner.
2. If `write_episode`: run `EpisodicMemoryWriter`, wait for completion.
3. If `tools_to_call`: dispatch tools in listed order, collect results.
4. If `fetch_rag`: run `MemoryManager.query_corpus()`, collect snippets for Slot 4.
   RAG results are filtered by `relevance_score >= 0.55` unless `plan.force_rag is True`,
   in which case the threshold is bypassed and all returned documents are included (still
   filtered for `lora-persona.md` exclusion). Maximum 3 sources regardless of `force_rag`.
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

**Structured-output field label "SUMMARY" triggers EOS at position 1 (word-level sensitivity)**
Controlled A/B testing at `temperature=0.0` on the Slot 6A working-state
extraction prompt revealed that adding a `SUMMARY:` field as a fourth
structured-output label causes Gemma 4B to emit near-100%-probability EOS
at the first output token — producing zero content — on every sample (0/3
success rate). The same prompt with only three labels (`FOCUS:`,
`OPEN_LOOPS:`, `DECISIONS:`) succeeded on every sample (3/3). All other
variables were held constant: system prompt structure, user prompt format,
`max_tokens`, and `temperature=0.0`.

This is evidence for word-level prompt sensitivity, not a token-budget issue.
The `SUMMARY` label itself — not the added length — appears to trigger the
failure. `max_tokens` was unchanged between the A and B conditions.

**Working theory (unverified hypothesis):** "SUMMARY" carries
document-closing semantics from pretraining — summary sections
characteristically appear near the end of documents, making EOS a
high-probability continuation after that token. This hypothesis is consistent
with all observed evidence but the underlying mechanism has not been confirmed.
Do not treat it as established fact.

**Distinction from the nearby temperature finding above:** The entry
"Temperature — `temperature = 0.0` produces degenerate output on extraction
tasks" refers to episodic extraction tasks and concerns output quality across
a general extraction contract. This finding concerns a specific structured-output
prompt where a single field label name drives near-certain EOS independently of
temperature or token budget. These are not the same root cause and should not
be conflated. See §9.2 for the full diagnostic arc and the decision to remove
the SUMMARY field entirely from Slot 6A Tier 2.

**Implication for the rest of the codebase:** Any structured-output prompt that
includes a `SUMMARY:` or similar document-closing field label should be treated
as a risk for this failure mode, particularly at `temperature=0.0`.

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
> Localist UI overhaul (provenance bar, episodic memory panel, full rebrand),
> Fetcher service restored (lxml, readability-lxml pinned in requirements.txt),
> and Graph Retrieval Layer Phase A/B (wiki_doc.py shared parsing helper,
> graph schema v3 migration, offline link-graph builder, WikiAgent link validation).
> Test suite: **224 tests, 0 failures** across 9 test files.
>
> **Files added/modified (all phases):**
> `memory_manager.py`, `prompt_builder.py`, `planner.py`,
> `episodic_extractor.py`, `tool_dispatcher.py`, `controller_agent.py`,
> `conversational_agent.py`, `wiki_agent.py`, `main.py`,
> `wiki/lora-persona.md`, `backfill_embeddings.py`, `embedding_engine.py`,
> `fetcher/__init__.py`, `fetcher/main.py`, `fetcher/models.py`,
> `fetcher/client.py`, `fetcher/extractor.py`,
> `wiki_doc.py` (new), `build_graph.py` (new), `requirements.txt`,
> `tests/test_memory_phase1.py`, `tests/test_prompt_builder.py`,
> `tests/test_planner_phase3.py`, `tests/test_controller_phase4.py`,
> `tests/test_episodic_phase5.py`, `tests/test_tool_dispatcher_phase6.py`,
> `tests/test_integration_phase7.py`,
> `tests/test_wiki_doc.py` (new), `tests/test_wiki_agent.py` (new),
> `tests/test_build_graph.py` (new).
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
>
> *Graph Retrieval Layer Phase A/B:*
> - `wiki_doc.py` added: `parse_wiki_doc()` / `load_wiki_doc()` returns `ParsedWikiDoc(frontmatter, body, links)`; PyYAML parses ISO dates as `datetime.date`; 12 tests in `tests/test_wiki_doc.py`
> - `controller_agent.py`: `_load_persona()` and `_load_user_profile()` now strip frontmatter via `parse_wiki_doc()` / `load_wiki_doc()` before operating on body; verified zero-behavior-change for current `lora-persona.md` and `wiki/users/michael.md`; `PyYAML>=6.0` added to `requirements.txt`
> - `wiki_agent.py`: `_validate_links()` added — scans Mapped Pages (H3) and Related Pages (H2) only; normalization `link_text.lower().replace(" ", "-")`; wired between `parse_model_xml()` and journaling; flagged links appear in `AgentResult.output["unresolved_links"]` and are logged as warnings; page content is never modified; 8 tests in `tests/test_wiki_agent.py` (new); `_FakeRuntime` established as convention for `run()` tests
> - `memory_manager.py`: `graph_nodes` and `graph_edges` tables added as v2→v3 migration (`_SCHEMA_VERSION = 3`); four new public methods: `upsert_graph_node()`, `upsert_graph_edge()`, `clear_graph_for_doc()`, `clear_graph_edges()`; 6 new tests in `TestGraphSchema` class in `tests/test_memory_phase1.py`
> - `build_graph.py` added: offline two-pass link-graph builder; same normalization rule as `_validate_links()`; same-page-same-target duplicate links collapse to one edge row per `(source_doc_path, target_path)` pair; whole-corpus `clear_graph_edges()` between passes; `doc_path` uses absolute resolved paths matching `document_index.path` convention; 10 tests in `tests/test_build_graph.py` (new)
> - Validation run against real 5-document corpus: 5 nodes, 11 edges, 8 resolved, 3 unresolved — see §8.7

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

---

### Session — 2026-06-18

*Identity RAG fix (Item 1):*
- `backend/raw/how-localist-works.md` authored (1,009 chars, under 1,600)
  and ingested into wiki corpus as a research note.
- `planner.py`: `_IDENTITY_KEYWORDS` frozenset (13 keywords) added;
  `_priority4a_identity()` method added with `_any_whole_word()` matching
  and trailing-content false-positive guard; wired into `route()` between
  Priority 3b and Priority 4; `RoutingPlan.force_rag` field added.
- `controller_agent.py`: `_load_persona()` path filter added — only
  `lora-persona` path accepted into Slot 1b (top-3 fetch, path filter);
  RAG filter updated to bypass score threshold when `plan.force_rag=True`.
- Live verified: "Who are you?" → P4a matched → `lora-persona.md` in
  Slot 1b (system_chars=403) → `how-localist-works.md` in Slot 4
  (rag_sources=2) → LORA identifies correctly.
- Test suite: 184 tests, 0 failures.

*Known issue resolved:*
- SQLite-persisted retrieval cache (`retrieval_cache` table, `valid` flag)
  does not invalidate on direct `MemoryManager.index_document()` calls that
  bypass the write path. Workaround: manual `UPDATE retrieval_cache SET
  valid = 0`. Long-term fix: ensure all ingest paths go through the
  canonical `/ingest` HTTP endpoint.

*User profile continuity (Item 2 — partial):*
- `backend/wiki/users/michael.md` authored: 5 sections (Identity,
  Active Projects, Preferences, Working Patterns, Decisions), 20 fact
  lines, no prose.
- `prompt_builder.py`: `UserProfileFact` dataclass added; `_CEIL_PROFILE
  = 100` added; `_slot3_episodic()` replaced by `_slot3_combined(bullets,
  profile_facts)` with independent 150/100-token sub-budgets; `build()`
  gains `profile_facts=` parameter (backwards-compatible).
- `controller_agent.py`: `_embed()` helper delegates to
  `MemoryManager._embed_fn`; `_load_user_profile()` lazy-loads and
  embeds 20 fact lines at first request; `_score_profile_facts()` scores
  via cosine similarity (threshold 0.45, top 5); Step 5b wired into
  `_execute_plan()` firing on P4, P4a, P5, and episodic-bullet turns;
  `profile_facts=` passed to `PromptBuilder.build()`.
- Live verified: "What are my working patterns?" → P4 corpus route →
  20/20 fact lines loaded → 5 working-pattern facts injected →
  `[USER PROFILE]` block in assembled prompt → LORA answered correctly.
- Graph retrieval layer (concept relationship reasoning) deferred to
  next session as planned.

---

## 8. Graph Retrieval Layer

### 8.1 Scope

**Implemented (Phases A, B, and C):**

- `wiki_doc.py` — shared frontmatter/body/link parsing helper consumed by
  `controller_agent.py` and `build_graph.py`.
- `memory_manager.py` v2→v3 schema migration — `graph_nodes` and `graph_edges`
  tables; `_SCHEMA_VERSION = 3`.
- `build_graph.py` — offline two-pass link-graph builder.
- `_validate_links()` in `wiki_agent.py` — write-time link validation wired
  between XML parsing and journaling.
- `memory_manager.py` — three new graph read methods: `resolve_node_by_stem()`,
  `get_backlinks()`, `get_outgoing_links()`, plus `list_graph_node_stems()` (added
  during Planner wiring once a gap was found — no existing method listed all stems).
  New result type `GraphEdgeResult`.
- `prompt_builder.py` — new `[GRAPH RESULT]` slot (`_slot_graph()`), positioned
  after Tool Results, before Working Memory. New input dataclasses
  `GraphQueryResult`/`GraphLinkEntry` (deliberately separate from
  `memory_manager.GraphEdgeResult` — `prompt_builder.py` remains free of any
  `memory_manager` import, preserving its pure-Python constraint). New
  `_CEIL_GRAPH = 300` ceiling. This slot is the one documented exception to the
  module's clean-omission contract: it is emitted whenever a graph query resolves
  a target page, even with zero edges, and is omitted only when resolution itself
  fails.
- `planner.py` — new standalone functions `extract_graph_query()` and
  `resolve_graph_target()` (three deterministic extraction patterns; three-tier
  stem-based name resolution: exact/substring, then token-overlap with a 2-token
  minimum and 0.5 ratio threshold, then ambiguous/no-match fallthrough — never a
  tiebreak). New `RoutingPlan` field `graph_query: tuple[str, int, str] | None`.
  New method `_priority3c_graph_query()`, checked in `route()` **before**
  `_priority3_tool()` — see ordering-correction note below. P3c's own inline guard
  checks `_FILE_OP_KEYWORDS`/`_FETCH_KEYWORDS`/the URL regex directly; when either
  fires, P3c defers and normal `_priority3_tool()` evaluation proceeds.
- `controller_agent.py` — new Step 5c in `_execute_plan()`: fetches
  `get_backlinks()`/`get_outgoing_links()` when `plan.graph_query` is set, translates
  `GraphEdgeResult` → `GraphLinkEntry`/`GraphQueryResult` (using `link_text`, not
  `target_path`, as the display name for unresolved targets, to preserve original
  casing per the locked output format), and passes the result into
  `PromptBuilder.build()`'s new `graph_result` parameter. The "pure/minimal"
  guarantee (graph-query turns never combine with RAG/episodic/profile context)
  requires no extra guard code — it falls out for free because P3c's `RoutingPlan`
  already sets `fetch_rag`/`fetch_episodic`/`force_rag` to `False` and
  `tools_to_call` to `[]`; confirmed end-to-end with a dedicated leak-marker test.
- `build_graph.py` — fixed: the `__main__` block previously called `MemoryManager()`
  with no path argument, which resolved to `MemoryManager`'s bare default
  (`lora_memory.db`) rather than the live backend's actual database
  (`localist_memory.db`, per `main.py:254`). Found via live manual testing, not by
  any automated test. Fixed by hardcoding `_BACKEND_DIR / "localist_memory.db"`.

**Locked-design ordering correction (found during implementation):** The design's
requirement that graph-query win over a web_search-only match is only satisfiable
if P3c is checked **before** `_priority3_tool()` runs — not after.
`_priority3_tool()` returns a plan whenever *any* of its three signals match,
including web_search alone; if P3c ran after it, a web_search-only match would
cause `route()` to return before P3c ever ran. The implemented ordering checks P3c
first, with P3c's inline guard (checking only `_FILE_OP_KEYWORDS`/`_FETCH_KEYWORDS`/
URL-regex — deliberately not `_WEB_SEARCH_KEYWORDS`) deferring to P3 only when
file_op or url_fetch signals are present. Locked in by `test_p3c_beats_web_search_p3`
in `tests/test_planner_phase3.py`, which would fail under the naive "after
`_priority3_tool()`" ordering.

**Explicitly deferred:**

- Phase D — automatic promotion from episodic memory to graph: not started.

### 8.2 Design Decisions

**Link graph, not LLM extraction.** Phase B parses existing `[[wiki-link]]`
references deterministically — no inference call, no entity/relationship
extraction. Rationale: matches the **Predictable** constraint (§1); edges only
exist where a human or WikiAgent explicitly linked two pages; avoids the
validation burden of LLM-extracted entities before the graph schema has any
real production usage. Richer NER/relationship extraction is a possible future
Phase C, not cancelled.

**Offline script, not WikiAgent post-ingest hook.** `build_graph.py` runs
manually and touches only `graph_nodes`/`graph_edges`. Rationale: WikiAgent's
system prompt is a protected XML-only contract (§3.5); embedding a hook would
carry the `/ingest` → `retrieval_cache` invalidation blast radius for an
unrelated concern; an offline script keeps both responsibilities isolated.
Migration path: the `build_graph()` function's signature is caller-agnostic —
a future WikiAgent post-ingest hook could call it without changing the function
itself, only the call site.

**Whole-corpus clear between passes, not per-document.** `build_graph.py`
calls `clear_graph_edges()` once between the node-upsert pass and the
edge-upsert pass. Ensures stale edges from since-removed `[[...]]` links never
survive a rebuild. `clear_graph_for_doc()` is implemented for future
per-document incremental updates but not called by the offline script.

**`doc_path` uses absolute resolved paths.** `str(Path(p).resolve())`,
consistent with the existing `document_index.path` convention in
`MemoryManager`. Future Phase C retrieval code can look up graph nodes the same
way it already looks up indexed documents.

**`raw/` documents are not graph nodes.** Only curated wiki pages are nodes.
A `[[...]]` link whose normalized target coincidentally matches a filename in
`raw/` still counts as unresolved.

### 8.3 Schema

`_SCHEMA_VERSION` is now **3** (v2→v3 migration added to `memory_manager.py`).

```sql
CREATE TABLE IF NOT EXISTS graph_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_path        TEXT    NOT NULL UNIQUE,
    node_type       TEXT,
    title           TEXT,
    source_doc_path TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_path
    ON graph_nodes(doc_path);

CREATE TABLE IF NOT EXISTS graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id  INTEGER NOT NULL REFERENCES graph_nodes(id),
    target_path     TEXT    NOT NULL,
    target_node_id  INTEGER REFERENCES graph_nodes(id),
    target_resolved INTEGER NOT NULL DEFAULT 0,
    link_text       TEXT    NOT NULL,
    source_doc_path TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source
    ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target_path
    ON graph_edges(target_path);
CREATE INDEX IF NOT EXISTS idx_graph_edges_resolved
    ON graph_edges(target_resolved);
```

`target_node_id` is nullable. Unresolved links are written with
`target_node_id = NULL` and `target_resolved = 0`. When the target page is
later created and the script rerun, `upsert_graph_edge()` updates the existing
row in-place — enabling automatic resolution on the next rebuild without a
separate cleanup pass.

### 8.4 Resolution Rule

```
link_text.lower().replace(" ", "-")
```

Applied identically by:

| Layer | Location |
|---|---|
| Write-time (WikiAgent) | `wiki_agent._validate_links()`, lines 663/665 |
| Read-time (builder) | `build_graph._normalize()` |

A link resolves if the normalized form matches the **stem** of an existing
`graph_nodes.doc_path` — matching how `wiki_pages` keys are built in
`wiki_agent.py` (`p.stem`), so resolution is apples-to-apples across both
layers.

Unresolved links are never dropped. A `target_resolved=False` row is always
written, enabling corpus gap analysis from the graph tables directly.

### 8.5 wiki_doc.py — Shared Parsing Helper

New module `backend/wiki_doc.py`:

```python
@dataclass(frozen=True)
class WikiLink:
    link_text: str
    target_path: str   # same as link_text; Phase B normalizes independently

@dataclass(frozen=True)
class ParsedWikiDoc:
    frontmatter: dict[str, Any]   # PyYAML 6.0 — ISO dates parse as datetime.date
    body: str
    links: list[WikiLink]         # all [[...]] in body; not section-scoped

def parse_wiki_doc(content: str) -> ParsedWikiDoc: ...
def load_wiki_doc(path: Path) -> ParsedWikiDoc: ...
```

`links` contains every `[[...]]` reference in the body. Section scoping is
`_validate_links()`'s concern at write time; the helper is scope-agnostic by
design so a future caller can impose any scoping it needs.

**Regression closed by this module:**
- `_load_persona()` previously truncated raw file content at 2,000 characters
  with no frontmatter awareness. Now operates on `body` only.
- `_load_user_profile()` had no frontmatter-skip logic. Now calls
  `load_wiki_doc()` and parses `body` lines.
- Both fixes verified zero-behavior-change for `lora-persona.md` and
  `wiki/users/michael.md`, neither of which has frontmatter today.

### 8.6 WikiAgent Link Validation

`_validate_links(actions, wiki_pages) -> dict[page_name, list[target]]` added
to `wiki_agent.py` and wired into `run()` between XML parsing and journaling.

**Scope:** `### Mapped Pages` (H3) and `## Related Pages` (H2) sections only.

**Behavior:** For each `[[link]]` in the scanned sections, if the normalized
form does not match an existing page stem or a self-proposed page name, the
link is flagged. Page content reaching disk is **never modified**. Flagged
links are logged at WARNING level and returned as
`AgentResult.output["unresolved_links"]`. This is intentional layered defense:
the read-time graph builder independently detects any unresolved link regardless
of what the write-time check catches.

A complementary write-time rule lives in the WikiAgent prompt templates themselves. Rule 7, added to both `build_user_prompt()` and `build_slim_prompt()` in `wiki_agent.py`, instructs the model to use the verbatim `page_name` as the `[[...]]` link target rather than a paraphrased title or longer description, reducing how often `_validate_links()` has anything to flag. This is a model-prompting measure only — `_validate_links()`'s normalization rule and section scope are unchanged, and it continues to flag every link that does not resolve exactly as before. Rule 7 reduces false positives at the source; it does not change what counts as resolved.

### 8.7 Validation-Run Results

`python build_graph.py` run against the real 5-document `wiki/` corpus
(2026-06-19 session).

| Metric | Count |
|---|---|
| Nodes | 5 |
| Edges | 11 |
| Resolved | 8 |
| Unresolved | 3 |

**Per-page breakdown:**

| Source page | Resolved edges | Unresolved edges |
|---|---|---|
| `how-localist-works` | 4 (→ `localist-build-order`, `localist-master-project-outline`, `localist-software-stack`, `lora-persona`) | 0 |
| `localist-build-order` | 1 (→ `localist-master-project-outline`) | 1 |
| `localist-master-project-outline` | 2 (→ `localist-build-order`, `localist-software-stack`) | 2 |
| `localist-software-stack` | 1 (→ `localist-master-project-outline`) | 0 |
| `lora-persona` | 0 | 0 |

**The three unresolved cases (precisely characterized):**

1. `localist-software-stack-overview` — from `localist-build-order.md`'s
   `[[Localist Software Stack Overview]]`. **Word-count mismatch**, not a casing
   issue. The actual page stem is `localist-software-stack`; the link text has
   an extra word ("Overview"). Will not resolve via the narrow normalization
   rule. This is the expected, correct behavior — not a defect in the
   normalization logic.

2. `localist-design-philosophy` — from `localist-master-project-outline.md`.
   Genuinely nonexistent page, proposed in that file's "Proposed New Pages"
   section but never created.

3. `localist-wiki-evolution-ideas` — from `localist-master-project-outline.md`.
   Same: genuinely nonexistent page.

**Incidental finding (recorded, not acted on):** `how-localist-works.md` is
the only page in the corpus whose `[[...]]` link targets are already correctly
kebab-cased, matching their target filenames exactly. Every other page exhibits
the title-case defect described in §8.8 Open Item 1. This suggests the model
can produce correct kebab-case link generation under at least some conditions —
relevant evidence for the prompt-tightening follow-up but not acted on here.

### 8.8 Open Items (Explicitly Deferred)

*Cross-reference (2026-06-21): Slot 5b (`[GRAPH RESULT]`) is now documented canonically in §3.2 and §3.3, not only in §8.1 Scope. The documentation gap from Phase C is closed.*

**Open Item 1 — WikiAgent prompt wording (highest-priority follow-up).**
WikiAgent's prompt does not state that `[[...]]` link targets must equal an
existing or self-proposed `page_name` verbatim. The real corpus confirms this
is a live defect (title-case vs. kebab-case throughout; word-count mismatch in
`localist-build-order.md`). A prompt-tightening change to Rule 5 and/or the
`_EXAMPLE` block was **agreed in principle (2026-06-19 session) but not
scheduled or implemented.** Recommended as a small standalone follow-up kept
separate from this build so it can be tested in isolation.

**Open Item 2 — `wiki/users/michael.md` frontmatter.** No decision has been
made about whether this file will ever receive OKF-style frontmatter.
`_load_user_profile()`'s frontmatter-skip logic handles it correctly if added,
per test coverage — but the decision to add frontmatter to that file has not
been made.

**Open Item 3 — Phase C retrieval path.** Implemented and live-verified
(2026-06-20 session); see session-log entry for detail.

**Open Item 4 — Phase D automatic promotion.** Not started, unchanged.

**Open Item 5 — Future LLM-based entity/relationship extraction.** Whether
this lives inside WikiAgent or remains a separate offline process is the same
structural question already resolved for link-parsing (offline), but has not
been decided for the richer extraction case.

**Open Item 6 — RAG frontmatter regression. CLOSED 2026-06-21.**

*Root cause (identified via read-only diagnostic, 2026-06-21):* `parse_model_xml()` in
`wiki_agent.py` extracted `content` from `create_page` actions via
`action.findtext("content")` (and the `__CONTENT_N__` placeholder path from
`_shield_content_blocks()`) without `.strip()`. Unlike `page_name`/`page_type` — both
stripped two lines above in the same function — `content` was assigned raw. Gemma's
generated XML consistently places a newline immediately after the opening `<content>` tag
(the few-shot `_EXAMPLE` template does not show one); that leading `\n` was written
verbatim to disk, becoming line 0 and pushing the real `---` frontmatter fence to line 1.
`parse_wiki_doc()` checks only `lines[0].rstrip("\r\n") == "---"` for fence detection;
when line 0 is a stray blank, the frontmatter branch is never entered and the entire raw
content — YAML block included — passes through as `body`, reaching `[CONTEXT]` via the
already-correct Step 4 call site (`parse_wiki_doc(doc.content).body[:2000]`). That
2026-06-19 fix was correctly placed; it was defeated by malformed input it had no way to
detect.

*Confirmed affected (both on-disk and in `document_index`):* four model-generated
`research-note` docs — `how-localist-works.md`, `localist-build-order.md`,
`localist-master-project-outline.md`, `localist-software-stack.md` — all written with a
leading blank line by Gemma, all returning `parse_wiki_doc().body == content` (full raw
file with YAML block intact).

*Confirmed unaffected:* `lora-persona.md` and `wiki/users/michael.md` (human-authored,
never pass through `parse_model_xml()`). Both verified byte-identical (`body == content`,
`frontmatter == {}`, `fence_idx = None`) via direct fresh disk-read in a follow-up
confirmation pass. The persona-cache call site in `_load_persona()` also verified
unaffected — `parse_wiki_doc()` takes the `fence_idx = None` path for that file.

*Fix, two layers (locked together — symptom-only fix would leave the malformed files
silently producing stale output on the next ingest cycle):*
1. **Write-time** (`wiki_agent.py`, `parse_model_xml()`): `.strip()` added to
   `raw_content` before assignment into `entry["content"]`, covering both the
   `__CONTENT_N__` placeholder path and the direct-`findtext` path identically.
   Prevents future model-generated pages from carrying a stray leading/trailing blank
   line to disk.
2. **Read-time** (`wiki_doc.py`, `parse_wiki_doc()`): `fence_idx` detection hardened
   to tolerate exactly one leading blank line before the `---` opening fence (bounded,
   not unbounded, to avoid masking unrelated malformed-doc cases). Existing
   no-closing-fence fallback (`frontmatter = {}`, `body = content`) preserved exactly
   unchanged. Fixes the four already-affected files immediately on next RAG fetch —
   no re-indexing required (`document_index.content` stores raw file text; `parse_wiki_doc()`
   runs at read time in Step 4, not at index time; confirmed by re-reading `index_document()`
   and the Step 4 call site fresh).

*Live verification:* query `"localist build order phases development roadmap"` against
live `localist_memory.db` returned all previously-affected docs with clean `[CONTEXT]`
bodies — each starts at `## Summary` with no `---`, `title:`, `type:`, or `query:` YAML
lines. Actual excerpt captured as evidence.

*Test suite:* 279 → 286 (+7: 4 in `test_wiki_doc.py` — leading-blank-parses-frontmatter,
body-clean, no-close-fence-fallback-unchanged, standard-fence-at-line-zero-unaffected;
3 in `test_wiki_agent.py` — strips-leading-newline, strips-trailing-whitespace,
strips-trailing-only), 0 failures.

**Open Item 7 — `build_graph.py` manual-trigger gap.** No automated trigger (no
hook, no CI step, no runbook) runs `build_graph.py` after wiki content changes.
This is what allowed the live P3c resolution failure to go undetected until manual
testing — the graph was simply never built against the production database. Not
urgent, but warrants a deliberate decision: wire into the WikiAgent post-ingest
path (the migration path noted in §8.2), add a CI/startup check, or leave manual
with an explicit runbook note. No decision made; flagged for a future session.

**Open Item 8 — `raw/`-in-RAG via `force_rag` bypass. CLOSED 2026-06-21.**

*Originally:* an unscoped inline observation from the 2026-06-19 evening live-testing session
(not a numbered Open Item at the time — logged as "flagged for evaluation in a future session").
Promoted to a tracked item and closed in the same 2026-06-21 session.

*Root cause:* `controller_agent.py` Step 4's `query_corpus()` call passed no `doc_type` filter,
so `wiki` and `raw` documents were ranked together in a single pool. The Step 4 filter condition
`if (plan.force_rag or doc.relevance_score >= 0.55)` meant that when `plan.force_rag=True` (set
by Priority 4a in `planner.py`, the identity-question route triggered by keywords such as "who
are you", "what can you do", "what is localist"), every top-3 `query_corpus()` result was
included in `[CONTEXT]` with no quality floor. `raw/` source files — structurally different from
curated wiki pages and not intended as direct grounding material for identity questions — could
backfill `[CONTEXT]` slots at scores as low as 0.0070.

*Live reproduction (three real P4a-triggering queries against the running backend —
`"What is Localist?"`, `"Who are you?"`, `"What can you do?"`):* `raw/` files reached `[CONTEXT]`
on every test, always via the `force_rag` bypass (no `raw/` result in any test would have cleared
the 0.55 threshold on its own merit — scores 0.0070–0.4206). Worst case: on `"Who are you?"`,
`lora-persona.md` scored highest (0.5023) but was excluded by the existing persona-exclusion guard,
leaving both remaining `[CONTEXT]` slots backfilled entirely by `raw/` files
(`raw/how-localist-works.md` at 0.4206, `raw/Localist Master Project Outline.md` at 0.4166).

*Design decision:* `raw/` files remain fully eligible for RAG in the normal (non-identity) routing
path, unchanged. For the `force_rag=True` identity-route path specifically, `[CONTEXT]` must never
be backfilled with `raw/` material; the persona doc or other curated wiki content should fill those
slots instead.

*Fix:* `controller_agent.py` Step 4's `query_corpus()` call now passes
`doc_type="wiki" if plan.force_rag else None` — restricting the candidate pool at the source for
the identity-route path rather than adding a second filter pass after the fact. No changes to
`memory_manager.py` — `query_corpus()`'s existing `doc_type` parameter already supported this.

*Live-verified post-fix:* Same three reproduction queries re-run — no `doc_type='raw'` document
appeared in any of the three. Normal (non-identity) RAG path confirmed unaffected:
`query_corpus(doc_type=None)` returns `raw/` documents as before; the `doc_type="wiki"` filter is
applied only when `force_rag=True`.

*Test suite:* 286 → 288 (+2 tests in `test_controller_phase4.py`, class
`TestForceRagDocTypeFilter`: `test_force_rag_true_calls_query_corpus_with_wiki_doc_type` and
`test_force_rag_false_calls_query_corpus_with_no_doc_type_filter`), 0 failures.

**Open Item 9 — Empty `[CONTEXT]` on identity-route queries. OPEN, deferred as of 2026-06-21.**

*Observed in the same live-verification pass as Open Item 8's fix:* for two of the three
identity-route reproduction queries (`"Who are you?"` and `"What can you do?"`),
`query_corpus(doc_type="wiki")` returned only `lora-persona.md` as a relevant wiki candidate —
which is then excluded by the existing persona-exclusion guard
(`not str(doc.path).endswith("lora-persona.md")`). The net result: `[CONTEXT]` is empty for these
identity questions post-fix, rather than backfilled with irrelevant `raw/` material as before.

Strictly better than the prior bug — no incorrect material is supplied. But it means some identity
questions currently get no RAG grounding at all. This gap was pre-existing but obscured by the
`raw/` backfill. Not a regression introduced by the Open Item 8 fix.

*No decision made on a fix direction.* Not scoped further in this session; flagged for a future
session consistent with this project's standard discipline for under-specified findings — gather
more occurrences and evaluate scope before treating as a fix-ready item.

**Open Item 10 — `_priority4a_identity()` missing `priority` field. CLOSED 2026-06-21.**

*Originally:* an unanalyzed observation noticed during the same live-reproduction pass used for
Open Item 8 — `_priority4a_identity()` was described informally as "returning `priority=4` in
its RoutingPlan but live runs showed `priority=6`." On fresh investigation this description was
inaccurate in its premise: the function does not set `priority` at all.

*Root cause:* `_priority4a_identity()` in `planner.py` constructs its `RoutingPlan` return value
without passing a `priority=` argument. `RoutingPlan.priority` defaults to `6` — the same default
used by `_priority6_direct()`, an unrelated fallback at the opposite end of the routing chain. Every
other `_priorityN_*` method in `planner.py` sets this field explicitly (priorities 1, 2, 3, 3, 4);
`_priority4a_identity()` was the sole outlier, so every identity-route plan silently inherited the
P6 default.

*Impact:* purely metadata/observability. `plan.priority` is consumed in exactly one place in the
entire codebase — `controller_agent.py`'s `ControllerResult.metadata` dict — with no influence on
actual routing control flow. `route()`'s evaluation order, and the returned `agent`, `fetch_rag`,
and `force_rag` were all already correct. Only the reported `priority` value in response metadata
was wrong: every identity question's response metadata reported `"priority": 6` when it should
have reported `"priority": 4`.

*Fix:* `priority = 4` added to the `RoutingPlan(...)` construction in `_priority4a_identity()`,
matching the function's name and its documented position in the evaluation order. The
`RoutingPlan.priority` default (`6`) is unchanged — it is correct and intentional for
`_priority6_direct()`'s use; the bug was the missing explicit override in P4a.

*Live-verified:* re-ran the three reproduction queries (`"What is Localist?"`, `"Who are you?"`,
`"What can you do?"`) — all three now report `priority=4` (previously `6`); `force_rag=True` and
`agent='conversational_agent'` unchanged, confirming no behavioral change, only the metadata
correction.

*Test suite:* 288 → 289 (+1: `test_p4a_identity_returns_priority_4` in
`tests/test_planner_phase3.py`, class `TestPlannerPriorities`), 0 failures.

---

### Session — 2026-06-19

*Graph Retrieval Layer Phase A/B:*
- `wiki_doc.py` created: `parse_wiki_doc()` / `load_wiki_doc()` returns `ParsedWikiDoc(frontmatter, body, links)` as frozen dataclasses; PyYAML parses ISO dates as `datetime.date` objects (PyYAML 6.0 behavior, empirically confirmed); 12 tests in `tests/test_wiki_doc.py` using verbatim real corpus fixtures. `PyYAML>=6.0` added to `requirements.txt`.
- `controller_agent.py` updated: `_load_persona()` strips frontmatter before truncating persona body; `_load_user_profile()` calls `load_wiki_doc()` and operates on body lines only. Both verified zero-behavior-change for current files. 4 tests added to `tests/test_controller_phase4.py`; profile test isolation fix applied (patch `pathlib.Path.exists` + `controller_agent.load_wiki_doc`).
- `wiki_agent.py` updated: `_validate_links()` added — section-scoped (`### Mapped Pages` H3, `## Related Pages` H2); normalization `link_text.lower().replace(" ", "-")`; wired between `parse_model_xml()` and journaling; unresolved links in `AgentResult.output["unresolved_links"]`; content never modified. 8 tests in `tests/test_wiki_agent.py` (new file); `_FakeRuntime` protocol-shaped fake established as convention for `WikiAgent.run()` tests to prevent `hasattr(rt, "infer_with_file")` false positives from `MagicMock`.
- `memory_manager.py` updated: `graph_nodes` and `graph_edges` tables added as v2→v3 migration (`_SCHEMA_VERSION = 3`); four new public methods: `upsert_graph_node()` (INSERT ... ON CONFLICT DO UPDATE, returns id), `upsert_graph_edge()` (SELECT + UPDATE-or-INSERT by natural key), `clear_graph_for_doc()` (per-document edge clear), `clear_graph_edges()` (whole-corpus edge clear). 6 new tests in `TestGraphSchema` in `tests/test_memory_phase1.py`.
- `build_graph.py` created: offline two-pass builder; normalization rule byte-for-byte identical to `_validate_links()`; same-page-same-target duplicate `[[...]]` links collapse to one edge row per unique `(source_doc_path, target_path)` pair (graph counts relationships, not link-mention occurrences); `doc_path` uses absolute resolved paths matching `document_index.path` convention; 10 tests in `tests/test_build_graph.py` (new file).
- Validation run (real corpus, 2026-06-19): 5 nodes, 11 edges, 8 resolved, 3 unresolved. See §8.7.
- Test suite: **224 tests, 0 failures** across 9 test files.

---

### Session — 2026-06-19 (Live Testing, Evening)

*Live verification of fixes from earlier session:*
- **Persona/profile frontmatter stripping** (`_load_persona()`, `_load_user_profile()`): confirmed correct — system_chars stable at 403; no stray YAML visible in assembled prompt.
- **RAG frontmatter stripping** (`parse_wiki_doc(doc.content).body[:2000]` in Step 4 of `_execute_plan()`): confirmed correct — RAG source content begins at body text, not YAML block.
- **Cross-turn working memory (`session_id`)**: confirmed correct — second turn's `[WORKING MEMORY]` slot contains first turn's instruction and assistant response; memory persists across turns within the same page load.

*New finding (recorded, not evaluated):*
- `query_corpus()` is surfacing content from `raw/` as a RAG source alongside wiki pages. Observed in live session but not investigated or acted on. Flagged for evaluation in a future session — may indicate the document index includes `raw/` files without the distinction that `build_graph.py` enforces (wiki-only nodes).

*Confirmed not a bug:*
- LLM response appearing to cut off mid-sentence in one live turn was model/sampling behavior, not a streaming or PromptBuilder truncation defect. Response resumed correctly on the next token sample. No code change warranted.

*Prefix stability (§3.7):*
- Zero KV-cache hits confirmed across two independent live data points (T1→T2, T2→T3). Root cause documented in §3.7. Named as highest-priority item for the next session.

*Test suite:* **231 tests, 0 failures** across 9 test files (7 new tests added this session: 3 RAG frontmatter + 4 session_id/working-memory).

---

### Session — 2026-06-20

*Prefix-stability investigation closed as a documentation/framing correction:*
- Root cause identified: `OMLXRuntimeClient.infer_stream()` sends one system message + one user message per HTTP call with no message-history accumulation and no session identifier — verified by reading `omlx_runtime_client.py` directly. Zero user-turn cache hits is an expected structural property of this contract, not a slot-ordering defect.
- No code changes required. `PromptBuilder.build()`'s existing slot order already matches the dynamic-suffix ordering this finding converged on.
- Stable-prefix / dynamic-suffix contract formalized in §3.7a: system message (Slot 1a + 1b) = stable prefix; user message (Slots 3a–7) = dynamic suffix; terms added to §3.1 as canonical vocabulary.
- §3.7 retitled and reframed as a resolved finding. "Design direction" options preserved as historical record, marked superseded.
- APC future direction noted in §3.7b (unscheduled, direction statement only).
- Slot-order regression test added: `test_pb_e_build_enforces_dynamic_suffix_slot_order` and `test_pb_f_slot3_profile_only_precedes_context` in `tests/test_prompt_builder.py`; both reference §3.7a in their docstrings.
- Persona-minimalism constraint superseded: `lora-persona.md` may now grow as plain undifferentiated prose (no internal section headers) to absorb durable static-rules content, up to the existing 500-token hard ceiling. No ceiling change made. Current actual size ~241 chars / ~60 tokens (~12% of cap).
- §3.2 Slot 1b updated: stale five-section structure description replaced with accurate plain-prose description; new growth-allowance rule added to Rules list.
- Test suite: **233 tests, 0 failures** across 9 test files (2 new tests added this session).

*WikiAgent prompt tightening — Rule 7 (same session, later prompt):*
- Rule 7 added to both `build_user_prompt()` and `build_slim_prompt()` in `wiki_agent.py`: `[[...]]` link targets in `### Mapped Pages` and `## Related Pages` must match an existing or self-proposed `page_name` verbatim — not a paraphrase, not a title, not a longer or shorter description. `_EXAMPLE` placeholder updated from `existing-page` to `localist-software-stack` with reason text illustrating the verbatim constraint. No change to `_validate_links()` normalization or scope (locked per §8.4).
- 3 new tests in `tests/test_wiki_agent.py`: Rule 7 present in both prompt functions with identical wording; Rules 1–6 text unchanged. Test suite: **236 tests, 0 failures**.
- §8.6 cross-reference note added (this session) orienting future readers to Rule 7 as the write-time companion to `_validate_links()`.
- §8.8 Open Item 1 (WikiAgent prompt wording) is now implemented and can be considered closed.

*Documentation refresh — persona figures (same session, latest prompt):*
- Persona token-count figures updated throughout the doc to reflect the post-rewrite size: **~476 chars / ~119 tokens (~24% of the 500-token cap)**. The earlier session-log reference "~241 chars / ~60 tokens (~12% of cap)" remains in this entry as a historical record of the size at that point in the session.
- §3.7 `system_chars = 403` (2026-06-19 live-session record) annotated inline to clarify it predates the persona rewrite; current estimate (~159 tokens / ~636 chars) noted alongside without altering the recorded value.
- §3.2 Slot 1b persona-structure description updated: sentence count corrected from four to five; size figures updated to match actual file.
- No code files modified.

---

### Session — 2026-06-20 (Phase C: Graph Retrieval Layer)

*Five-prompt Phase C build sequence:*
- **Prompt 1 (MemoryManager graph read methods):** `resolve_node_by_stem()`,
  `get_backlinks()`, `get_outgoing_links()`, `GraphEdgeResult`. `list_graph_node_stems()`
  was not in the original scope — the gap was found during Prompt 4 (Planner P3c wiring);
  the implementation lives here because it is a `MemoryManager` method. Test suite:
  236 → 242 (+6 tests).
- **Prompt 2 (PromptBuilder graph slot):** `GraphQueryResult`/`GraphLinkEntry` input
  dataclasses (deliberately decoupled from `memory_manager.GraphEdgeResult` to preserve
  `prompt_builder.py`'s pure-Python constraint); `_slot_graph()` method; `_CEIL_GRAPH = 300`
  ceiling; `graph_result=` parameter wired into `build()`. Test suite: 242 → 251 (+9 tests).
- **Prompt 3 (Planner extraction + name resolution):** `extract_graph_query()` and
  `resolve_graph_target()` standalone functions (`TestGraphQueryExtraction` and
  `TestGraphNameResolution` classes in `tests/test_planner_phase3.py`). Scope strictly limited
  to the two parsing/resolution helpers — no `route()` changes, no `RoutingPlan` changes, no
  `MemoryManager` calls. Test suite: 251 → 266 (+15 tests).
- **Prompt 4 (Planner P3c wiring):** `RoutingPlan.graph_query` field;
  `_priority3c_graph_query()` method; P3c inserted in `route()` before P3; and
  `list_graph_node_stems()` added to `MemoryManager` (gap found in this prompt — no existing
  method listed all node stems). Test suite: 266 → 273 (+7 tests).
- **Prompt 5 (ControllerAgent wiring):** Step 5c in `_execute_plan()` — fetches edges when
  `plan.graph_query` is set, converts `GraphEdgeResult` → `GraphLinkEntry`/`GraphQueryResult`,
  passes to `PromptBuilder.build()`. `link_text` (not `target_path`) used as display name for
  unresolved targets to preserve original author casing. Pure/minimal guarantee confirmed by
  a dedicated leak-marker test. Test suite: 273 → 279 (+6 tests). Zero failures throughout.

*Two discrepancies found and resolved against the locked design during implementation:*
- **No `page_name` field in `graph_nodes`.** The design assumed a `page_name` column for name
  resolution. Actual schema (§8.3) has no such field. Resolution is stem-based via
  `Path(doc_path).stem`. Prompted the addition of `list_graph_node_stems()` — no pre-existing
  method listed all stems for the candidate list passed to `resolve_graph_target()`.
- **P3c-before-P3 ordering.** The design required graph-query to win over a web_search-only
  P3 match. This is only satisfiable if P3c runs before `_priority3_tool()` — if P3c ran
  after, a web_search-only match would already have caused `route()` to return. See §8.1
  ordering-correction note. Locked in by `test_p3c_beats_web_search_p3` in
  `tests/test_planner_phase3.py`.

*Live-testing arc:*
- `"What links to localist-build-order?"` failed on first live test. Root cause:
  `graph_nodes`/`graph_edges` were empty — `build_graph.py` had never been run against the
  live backend database. Not a code bug; the Planner correctly logged "name resolution failed"
  because the candidate stem list was empty.
- After running `build_graph.py`, P3c still failed. The actual bug: the `__main__` block called
  `MemoryManager()` with no path, defaulting to `lora_memory.db`, while the live backend uses
  `localist_memory.db` (per `main.py:254`). The script reported success but silently populated
  the wrong file.
- Fixed by hardcoding `_BACKEND_DIR / "localist_memory.db"` in the `__main__` block. After
  re-running: 5 nodes, 11 edges, 8 resolved, 3 unresolved — matching §8.7 exactly.
  `Planner.route("What links to localist-build-order?")` returned
  `graph_query=('incoming', 2, 'localist-build-order')` against the correct database,
  confirming end-to-end P3c resolution.

*`wiki/users/michael.md` exclusion confirmed:* `build_graph.py`'s non-recursive
`wiki_dir.iterdir()` walk correctly excludes subdirectory entries, including
`wiki/users/michael.md`. This is correct behavior by design. Resolves a question raised
but left open during the Planner prompt's report.

*Stray `lora_memory.db`:* Created by the first wrongly-targeted `build_graph.py` run.
Contains graph data only (no `document_index` content); not referenced by any code path.
Left in place; safe to delete in a future cleanup session.

*Test suite:* **279 tests, 0 failures** across 10 test files. No new test files; all
Phase C tests added to existing files.

*One open item carried forward (see §8.8, Item 7):* the `build_graph.py` manual-trigger
gap that allowed the production database to remain empty until manual testing caught it.
Item 6 (RAG frontmatter regression) was closed in the 2026-06-21 session — see that
session-log entry below.

---

### Session — 2026-06-21

*Open Item 6 closed — RAG frontmatter regression root-caused and fixed:*

**Diagnostic phase (read-only):** A fresh diagnostic pass confirmed the root cause was
not in `controller_agent.py` Step 4 (already correct), not in `memory_manager.py`'s
retrieval cache (stores scores/paths only; content re-fetched fresh from `document_index`
on every cache hit), and not in `prompt_builder.py`'s `_slot4_rag()` (renders
`RagSource.content` as-is, no re-fetch). Static scan of all six `wiki/` files on disk
plus the six `document_index` rows in the live `localist_memory.db` revealed the actual
defect: four model-generated docs (`how-localist-works.md`, `localist-build-order.md`,
`localist-master-project-outline.md`, `localist-software-stack.md`) each have a leading
blank line (`'\n'`) as line 0, with the `---` frontmatter fence on line 1. This was
confirmed by `repr()` inspection of the raw bytes — no BOM, no `\r\n` mismatch, no
missing closing fence; simply a stray blank line prepended by Gemma before writing.

**Root-cause trace:** `parse_model_xml()` in `wiki_agent.py` assigns
`entry["content"] = raw_content` without `.strip()`. `page_name` and `page_type` are
stripped two lines above; `content` is not. Gemma's XML output places a newline
immediately after the opening `<content>` tag in practice (the few-shot `_EXAMPLE`
template does not demonstrate one). That `\n` is never stripped anywhere downstream —
not in `_shield_content_blocks()`, not in `parse_model_xml()`, not in
`write_text_file()`. It becomes line 0 on disk. `parse_wiki_doc()` evaluates only
`lines[0].rstrip("\r\n") == "---"`; when line 0 is blank, the frontmatter branch is
never entered, `frontmatter = {}` and `body = content` (full file including the YAML
block) are returned silently. The Step 4 call site `parse_wiki_doc(doc.content).body[:2000]`
executes correctly; the body it receives is already the full file, so the frontmatter
passes through to `RagSource.content` and into `[CONTEXT]`. The 2026-06-19 fix was real
and correctly placed — it was defeated by write-time malformation it could not detect.

**Fix:**
- Write-time: `.strip()` added to `raw_content` in `parse_model_xml()` before
  `entry["content"]` assignment, covering both the `__CONTENT_N__` placeholder path (the
  normal model-output path) and the direct-`findtext` path identically.
- Read-time: `parse_wiki_doc()` in `wiki_doc.py` hardened with `fence_idx` detection
  tolerating exactly one leading blank line before the opening `---` fence. The condition
  is bounded (`len(lines) >= 2 and lines[0].rstrip("\r\n") == "" and lines[1].rstrip("\r\n") == "---"`)
  — not unbounded stripping, to avoid masking deeper structural problems. The existing
  no-closing-fence fallback (`frontmatter = {}`, `body = content`) is preserved exactly.
  No re-indexing required: `parse_wiki_doc()` runs at read time in Step 4; `index_document()`
  stores raw file content and calls no parser. Confirmed by re-reading both `index_document()`
  and the Step 4 call site fresh.

**Follow-up confirmation pass (same session, after fix landed):** Verified that the two
new tests covering the no-fence-found fallback and the standard-fence-at-line-0 case use
generic inline fixture strings rather than the real `lora-persona.md` or
`wiki/users/michael.md` content. Closed the gap with a direct disk-read check: both files
confirmed `body == content` exactly (`frontmatter == {}`, no truncation, no mutation,
`fence_idx = None` for both — `lora-persona.md` starts with `"You are LORA…"`,
`users/michael.md` starts with `"## Identity"`). The persona-cache call site in
`_load_persona()` confirmed unaffected — `parse_wiki_doc()` returns full file body
unchanged for a no-fence document, identical pre/post-fix.

**Live verification:** `MemoryManager.query_corpus("localist build order phases development
roadmap")` against live `localist_memory.db` (keyword-only, `use_embeddings=False`)
returned three previously-affected docs; all produced clean `[CONTEXT]` bodies starting
at `## Summary` with no YAML block. Actual excerpt for `localist-build-order`:

```
## Summary

This document outlines the nine-phase development roadmap for the Localist project...
```

**Test suite:** 279 → 286 (+7 tests, 0 failures). New tests: 4 in `test_wiki_doc.py`
(leading-blank-parses-frontmatter, leading-blank-body-clean,
leading-blank-no-close-fence-fallback-unchanged, standard-fence-at-line-zero-unaffected);
3 in `test_wiki_agent.py` (strips-leading-newline, strips-trailing-whitespace,
strips-trailing-only). All 286 tests pass.

*Open Item 7 remains open.* No code change was made to `build_graph.py` or its trigger
path in this session.

---

### Session — 2026-06-21 (`raw/`-in-RAG fix, Open Items 8–9)

*`raw/`-in-RAG finding reproduced, root-caused, and fixed (see §8.8, Open Item 8):*

**Reproduction:** The inline observation from the 2026-06-19 evening session was reproduced
live. Three P4a identity-route queries (`"What is Localist?"`, `"Who are you?"`, `"What can you
do?"`) each confirmed `raw/` files reaching `[CONTEXT]`, always via the `force_rag=True` bypass
set by Priority 4a. No `raw/` result in any test cleared the 0.55 threshold on its own merit
(scores 0.0070–0.4206). Worst case — `"Who are you?"`: `lora-persona.md` scored highest (0.5023)
but was excluded by the persona-exclusion guard, leaving both remaining `[CONTEXT]` slots filled
by `raw/how-localist-works.md` (0.4206) and `raw/Localist Master Project Outline.md` (0.4166).

**Root cause:** `controller_agent.py` Step 4's `query_corpus()` call passed no `doc_type` filter,
so `wiki` and `raw` documents competed in a single ranked pool. Because `plan.force_rag=True`
bypasses the 0.55 threshold entirely, any `raw/` file landing in the top 3 was automatically
included with no quality floor.

**Fix:** `doc_type="wiki" if plan.force_rag else None` added to the `query_corpus()` call in Step
4. Restricts the candidate pool at the source for the identity-route path; no second filter pass
after the fact. No changes to `memory_manager.py` — `query_corpus()`'s existing `doc_type`
parameter already supported this. `raw/` files remain fully eligible for the normal non-identity
routing path, unchanged.

**Live verification post-fix:** Same three reproduction queries re-run — no `doc_type='raw'`
document appeared in any. Normal RAG path confirmed unaffected: a sample non-identity query
(`"local first agent framework oMLX Python SQLite multi-agent reasoning pipeline"`) returned three
`raw/` docs in its top-5 `query_corpus(doc_type=None)` results, confirming the filter is additive
and does not disturb the base ranking.

**Test suite:** 286 → 288 (+2 tests in `test_controller_phase4.py`,
`TestForceRagDocTypeFilter`), 0 failures.

*New deferred finding opened (see §8.8, Open Item 9):* With `doc_type="wiki"` applied on the
identity route, two of the three reproduction queries (`"Who are you?"`, `"What can you do?"`)
returned only `lora-persona.md` from `query_corpus()`, which the persona-exclusion guard then
removes — leaving `[CONTEXT]` empty. Strictly better than the prior bug; no incorrect material
supplied. The gap was pre-existing but obscured by the `raw/` backfill. No fix direction decided;
logged as Open Item 9.

*`_priority4a_identity()` priority-field bug identified and fixed (see §8.8, Open Item 10):*
The same live-reproduction run that produced Open Items 8 and 9 also revealed that the three
reproduction queries all returned `priority=6` in the `RoutingPlan`, despite P4a correctly firing.
Root cause: `_priority4a_identity()` never sets `priority=` in its `RoutingPlan(...)` construction,
so the field silently inherits `RoutingPlan.priority`'s default of `6` — the same value used by
`_priority6_direct()`. All other `_priorityN_*` methods set this field explicitly; P4a was the
sole outlier. Impact: metadata-only — `plan.priority` feeds only `ControllerResult.metadata`, with
no bearing on routing control flow or behavior. Fix: `priority = 4` added to the P4a
`RoutingPlan(...)` construction. All three reproduction queries now report `priority=4`; `force_rag`
and `agent` unchanged. Test suite: 288 → 289 (+1 test), 0 failures.

*Open Items 7 remains open.* No further code changes this session.

---

## 9. Slot 6A — Structured Working State

### 9.1 Scope

**Implemented:**

- `WorkingMemoryState` dataclass in `prompt_builder.py` — two Tier 1 fields:
  `current_project: str | None` and `active_artifacts: list[str]`.
- `_slot6a_working_state()` and `_CEIL_WORKING_STATE = 100` in `prompt_builder.py` —
  renders `[WORKING STATE]` block with clean omission when both fields are empty/None.
- Step 5d in `controller_agent._execute_plan()` — assembles `WorkingMemoryState`
  deterministically from current-turn RAG sources and `task.context["project_context"]`;
  gated on `plan.graph_query is None` to suppress rendering on P3c turns.
- `working_state` SQLite table (`memory_manager.py`, v3→v4 migration).
- `WorkingStateRecord` dataclass and `WorkingStateStore` class in `memory_manager.py` —
  `get()`, `upsert()`, `clear()` methods.
- Tier 2 extraction in `episodic_extractor.py` — `_WORKING_STATE_UPDATE_SYSTEM` (3-field
  structured prompt), `extract_working_state_update()` (parse + return tuple), and
  `process_working_state_update()` (orchestrate + upsert). Wired into
  `controller_agent.py` post-response hook; runs regardless of `plan.graph_query`.
- v4→v5 migration (`memory_manager.py`) — removes `turn_summaries_json` column dropped
  after live diagnostic testing confirmed 4-field format failure (see §9.2).
- `tests/test_episodic_phase5.py` — covers full Tier 2 extraction arc including the
  new `test_exactly_three_labels_required` test added in this session.
- `tests/test_memory_phase1.py` — updated for v5 schema (version assertion, column
  checks, `upsert()` call sites).

**Explicitly deferred or not yet wired:**

- Tier 2 fields (`current_focus`, `open_loops`, `recent_decisions`) are stored and
  updated but **do not yet appear in the rendered `[WORKING STATE]` output**. The
  update pipeline is complete end-to-end; the render side is a separate future step.
- A deterministic pre-gate on Tier 2 updates (analogous to `_has_implicit_signal()`
  for episodic extraction) has not been decided. See §9.5 Open Item 1.
- Automatic promotion of working state to episodic memory — not started.

### 9.2 Design Decisions

**Literal/interpretive split with Slot 6 (Working Memory).** Slot 6 carries
raw conversational turns: what was said, by whom, in what order. Slot 6A carries
structured state about what is happening and what has been decided — a layer of
interpretation above the raw transcript. The split respects the Predictable
constraint: Slot 6 is deterministic; Slot 6A's Tier 1 fields are deterministic;
only Slot 6A's Tier 2 fields involve inference, and that inference runs in a
post-response hook, not in the routing or prompt-assembly path.

**Deterministic Tier 1 first, model-assisted Tier 2 second.** The system
establishes a working rendering slot (`[WORKING STATE]`) before Tier 2 is
wired in. This allows the slot to carry real value in early deployment (active
RAG sources are immediately visible in the assembled prompt) while Tier 2 is
validated in parallel. It also avoids the risk of a Tier 2 extraction failure
silently leaving the rendered slot empty — Tier 1 renders unconditionally when
either of its fields is non-empty.

**P3c render-gating (no `[WORKING STATE]` on graph-query turns).** Graph-query
turns (P3c route) are a deliberate no-interpretation zone: structural,
non-personalized, deterministic relative to graph state. Injecting `[WORKING
STATE]` on the same turn would introduce interpretive, session-specific content
into what should be a clean, repeatable graph answer — blurring the boundary
by accident rather than by deliberate future design. The gate is a single
`if plan.graph_query is None:` check in `controller_agent.py` Step 5d.

**Update-vs-render distinction.** Tier 2 update (`process_working_state_update()`)
runs in the post-response hook regardless of `plan.graph_query`. A graph-query
turn produces real conversational state — the user issued an instruction and the
assistant responded — and that state is worth persisting so the next non-P3c
turn can render it. Suppressing the update on P3c turns would discard valid
session state to maintain a render boundary that applies only to a single slot's
output.

**SUMMARY field removed, not deferred.** The Slot 6A Tier 2 prompt was
originally designed with four output fields: `FOCUS`, `OPEN_LOOPS`, `DECISIONS`,
`SUMMARY`. Live diagnostic testing at `temperature=0.0` showed a categorical
failure with the 4-field format: on every sample (0/3), the model emitted bare
EOS with zero output content. The 3-field format (removing only `SUMMARY`) succeeded
on every sample (3/3), all else held constant. The SUMMARY field was therefore
permanently removed from the system prompt, the extraction function, the
`WorkingStateRecord` dataclass, the `upsert()` signature, the database schema
(v4→v5 migration), and all test fixtures. This was a removal decision, not a
deferral — the field will not be re-added unless the underlying failure mode is
understood and a workaround exists. The working theory for the failure (see §4.7)
is that "SUMMARY" carries document-closing pretraining semantics that dominate EOS
probability at position 1; this is an unverified hypothesis consistent with the
evidence, not a confirmed mechanism.

**SQLite DROP COLUMN safety check.** The v4→v5 migration drops
`turn_summaries_json` using `ALTER TABLE working_state DROP COLUMN`. This
requires SQLite ≥ 3.35.0; the project uses 3.50.4, so the operation is safe.
A PRAGMA guard (`PRAGMA table_info(working_state)` before executing the DROP)
makes the migration idempotent: fresh databases opened via the v2→v3 or v3→v4
path have `working_state` pre-created by `_init_db()` without `turn_summaries_json`
(since `_init_db()` already reflects the v5 schema), so the column is absent
when the v4→v5 block runs. The conditional prevents a spurious
`sqlite3.OperationalError: no such column` on those paths.

### 9.3 Schema

`_SCHEMA_VERSION` is now **5** (up from 3 at the time §8 was written; v3→v4
created `working_state`, v4→v5 dropped `turn_summaries_json`).

**`working_state` table (current, v5):**
```sql
CREATE TABLE IF NOT EXISTS working_state (
    mem_key                TEXT    PRIMARY KEY,
    current_focus          TEXT,
    open_loops_json        TEXT    NOT NULL DEFAULT '[]',
    recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
    updated_at             REAL    NOT NULL
);
```

**`WorkingStateRecord` (Python dataclass, `memory_manager.py`):**
```python
@dataclass
class WorkingStateRecord:
    mem_key:          str
    current_focus:    str | None
    open_loops:       list[str]
    recent_decisions: list[str]
    updated_at:       float
```

**v3→v4 migration (added `working_state` with `turn_summaries_json`):**
```python
if from_version < 4:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_state (
            mem_key                TEXT    PRIMARY KEY,
            current_focus          TEXT,
            open_loops_json        TEXT    NOT NULL DEFAULT '[]',
            recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
            turn_summaries_json    TEXT    NOT NULL DEFAULT '[]',
            updated_at             REAL    NOT NULL
        );
    """)
```

**v4→v5 migration (dropped `turn_summaries_json`, conditional via PRAGMA):**
```python
if from_version < 5:
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(working_state)"
    ).fetchall()}
    if "turn_summaries_json" in cols:
        conn.executescript(
            "ALTER TABLE working_state DROP COLUMN turn_summaries_json;"
        )
```

`_init_db()` pre-creates `working_state` in the v5 form (without
`turn_summaries_json`) for fresh databases, so the PRAGMA guard above handles
both upgrade paths (v4→v5: column exists, DROP executes) and fresh/v2/v3 paths
(column never existed, DROP skipped).

### 9.4 Test Suite

Final state as of 2026-06-21: **318 tests, 0 failures** across all test files.

Per-file delta confirmed for the SUMMARY-removal session (the last session in
the Slot 6A build arc):

| File | Before | After | Delta |
|---|---|---|---|
| `test_episodic_phase5.py` | 61 | 62 | +1 (`test_exactly_three_labels_required`) |
| `test_memory_phase1.py` | unchanged count | unchanged count | tests updated for v5 (version assertion, column checks, `upsert()` signatures) |
| `test_tool_dispatcher_phase6.py` | pre-existing test broken | fixed | +0 net (fix, not new test) |

The total rose from approximately 314 (pre-SUMMARY-removal, with the tool ceiling
test already broken) to 318 after the session. Per-file breakdowns across the
full Slot 6A build arc (prior to the SUMMARY-removal session) are not
reconstructed here — the total of 318 passing, 0 failures is the authoritative
figure.

### 9.5 Open Items

**Open Item 1 — Tier 2 pre-gate (unresolved, flagged for live observation).**
The episodic extraction pipeline has a deterministic pre-gate
(`_has_implicit_signal()`) that avoids an inference call on turns with no
plausible implicit signal. Tier 2's `process_working_state_update()` has no
equivalent gate — it calls the model on every completed non-episodic turn,
including P6 (direct answer) turns. Whether a similar pre-gate is warranted
depends on how often Tier 2 updates produce useful state changes vs. `NONE`
returns on short, low-context turns. No decision has been made; flagged for
live observation rather than speculative pre-optimization.

**Open Item 2 — Tier 2 render wiring not yet done.** `current_focus`,
`open_loops`, and `recent_decisions` are stored and updated per turn but
never appear in the rendered `[WORKING STATE]` block. Wiring them in requires
reading the stored `WorkingStateRecord` in `controller_agent.py` Step 5d and
merging those fields into the `WorkingMemoryState` passed to
`PromptBuilder.build()` — or a separate dataclass path for Tier 2 fields.
Not started; the schema and extraction pipeline are ready when this step is
tackled.

**Open Item 3 — SUMMARY/EOS mechanism hypothesis unverified.** The working
theory that "SUMMARY" triggers early EOS via document-closing pretraining
semantics is consistent with the diagnostic evidence but the mechanism is not
confirmed. Anyone considering a structured-output prompt with similar field
labels elsewhere in this codebase should treat §4.7's documented finding as
a risk, not a certain prediction, and run their own diagnostic at the target
temperature and prompt shape before committing.

---

### Session — 2026-06-21 (Slot 6A — Structured Working State)

*Two-part build arc: Tier 1 + Tier 2 implementation, then SUMMARY-field removal.*

- **Tier 1** (`WorkingMemoryState`, `_slot6a_working_state()`, Step 5d gate,
  P3c render-gating) shipped deterministically: `active_artifacts` carries
  current-turn RAG source paths into `[WORKING STATE]`; `current_project` is
  wired but inert at all current call sites (DB-scoping key, not a project name).
- **Tier 2** (`WorkingStateStore`, `WorkingStateRecord`, `working_state` table
  via v3→v4 migration, `episodic_extractor.py` Tier 2 extraction hook) built and
  wired. Post-response hook runs on every completed turn regardless of routing
  path, including P3c; update-vs-render distinction deliberate.
- **SUMMARY field diagnostic and removal:** live testing confirmed that a 4-field
  extraction format (adding `SUMMARY:`) fails at `temperature=0.0` with 0/3 success
  rate (model emits bare EOS); 3-field format succeeds 3/3 at the same temperature,
  all else constant. SUMMARY removed permanently from system prompt, extraction
  function, `WorkingStateRecord`, `upsert()`, DB schema (v4→v5 migration), and all
  tests. See §4.7 and §9.2 for the full finding and decision record.
- **Test suite:** 318 tests, 0 failures (final state for this session). 1 new test
  added (`test_exactly_three_labels_required` in `test_episodic_phase5.py`); 1
  pre-existing broken test in `test_tool_dispatcher_phase6.py` fixed (missing
  `[WORKING STATE]` label in slot-ceiling measurement).

*End of Localist Framework Canonical Architecture Specification*
