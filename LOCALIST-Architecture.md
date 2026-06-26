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
10. [Semantic Search-Intent Classifier](#10-semantic-search-intent-classifier)

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

*Flagged 2026-06-23: live `/admin/api/cache/probe` evidence in §3.7c below directly contradicts the "no current backend... can produce a user-message cache hit" claim above. §3.7 and this paragraph are NOT yet edited to reflect this — that edit is deliberately deferred to a future session per standing discipline (lock the finding, decide on action items, then update the doc). Treat the claim above as superseded-but-not-yet-corrected in the prose until that edit happens.*

---

### 3.7c Open Item — Live Cache-Mechanism Findings and Candidate Follow-Up Actions (2026-06-23)

**Status: forward-looking open item. Findings below are confirmed via live source-reading and a live probe call. The four candidate actions are NOT yet decided, scoped, or scheduled — this section exists to preserve the option space before the next session picks a subset to act on.**

#### Confirmed mechanism (supersedes §3.7's session-continuity framing)

Source-read directly from the installed oMLX package
(`/opt/homebrew/Cellar/omlx/0.4.2/libexec/lib/python3.11/site-packages/omlx/`)
and confirmed live via `POST /admin/api/cache/probe`:

- oMLX's prefix cache is **role-blind and session-blind**. It hashes fixed-size
  token-ID blocks (`compute_block_hash`, `paged_cache.py:78–119`) chained from a
  fixed root seed (`b"omlx-root"`) — it has no concept of "session," "history,"
  or "system vs. user message." §3.7's premise that caching requires the
  *client* to assert continuity (a session ID, a growing `messages` array) is
  incorrect. A client sending one isolated system+user pair per HTTP call —
  exactly what `OMLXRuntimeClient.infer()` does — can and does produce real
  cache hits on identical leading token blocks across separate, unrelated
  calls, with no client-side change required.
- Block size for the currently-loaded model is **512 tokens**, not the
  256-token default documented in `scheduler.py:904` —
  `_align_block_size_with_rotating_window()` overrides this at load time for
  Gemma 4's rotating-window attention. Confirmed live via probe response
  (`block_size: 512`), not assumed from source alone.
- Live probe data (two real Localist-shaped prompts, ~780–800 tokens each,
  system prompt + `[TOOL RESULTS]` + `[WORKING MEMORY]` held byte-identical,
  only `[INSTRUCTION]` varying): both prompts produced exactly 2 blocks. Block
  0 (tokens 0–511) was cached (`ssd_disk`) for both, confirmed already-warmed
  from a prior 20-call diagnostic run. Block 1 (the remaining ~270–280 tokens)
  was cold for both. Partial trailing blocks are never stored by
  `store_cache` — block 1 is **unconditionally** cold regardless of content,
  not cold because the instruction text differed. At current prompt lengths,
  instruction-content divergence has **zero effect** on cache behavior, since
  the only token range it could affect never gets cached in the first place.
- `blocks_ssd_hot: 0` on both probes — the cached block was sitting in the SSD
  cold tier, not the in-memory hot tier, at probe time (server/model had been
  reloaded since the diagnostic run). A real inference call right now would
  pay one SSD read for block 0 rather than a RAM hit.
- The endpoint is `POST /admin/api/cache/probe`, not `/admin/probe_cache` as
  an earlier investigation guessed from release notes alone — corrected here
  for any future session that wants to re-run this check. Response is
  aggregate counts only (`total_tokens`, `block_size`, `total_blocks`,
  `blocks_ssd_hot`, `blocks_ssd_disk`, `blocks_cold`, `ssd_hit_tokens`,
  `cold_tokens`) — no per-block detail.
- **Three structurally distinct system messages compete for block 0, with
  zero cross-sharing, confirmed by direct source read (2026-06-23).** The
  warm-up fixture's system message (Lever 3, below), the main
  conversational call's per-turn system message (persona + `PromptBuilder`
  slots), and `_WORKING_STATE_UPDATE_SYSTEM` (the fixed constant used by
  `extract_working_state_update()`'s post-response Tier 2 call) are three
  different byte sequences from token 0. Since the cache is content-hashed
  with no session or role awareness, none of these three call types can
  ever produce a block-0 cache hit against either of the other two — they
  are three separate, mutually non-overlapping cache lineages, not one
  cache being destabilized by another. This was identified live: a
  production session showed `blocks_cold` accumulating tokens turn over
  turn with `blocks_ssd_hot` frozen at its single post-warm-up value (512
  tokens cached out of 6,019 total prefill tokens after several turns —
  8.5% efficiency, down from 25% after the first conversational call). An
  embedding-call interference theory was proposed and **disproven by direct
  source read**: `EmbeddingEngine` (in `embedding_engine.py`) runs entirely
  in-process via `mlx_embeddings`, makes no HTTP call to oMLX at all, and
  the module's own docstring states "OMLXRuntimeClient.embed() is NOT
  called anywhere for corpus embeddings" — there is no code path by which
  an embedding call touches oMLX's cache. The three-system-message finding
  is the better-supported explanation, confirmed by reading
  `_WORKING_STATE_UPDATE_SYSTEM`'s literal content against the other two
  system messages, not yet confirmed by a live diagnostic isolating each
  lineage's contribution independently. **Not yet actioned** — see Lever 3
  update below and the live working-state-update outcome diagnostic
  currently in progress (§10.4-adjacent; not yet a numbered open item as
  of this update).

#### Candidate follow-up actions (Lever 3 now implemented and confirmed; 1, 2, 4 remain option space)

**Lever 1 — Grow the persona to widen the structurally-cacheable portion of
block 0.** The system message currently occupies only ~159 of block 0's 512
tokens; the remainder is dynamic-suffix content that happens to fit in block
0 only by coincidence of current prompt length. Growing persona content
toward the existing 500-token/2000-char ceiling (already sanctioned by
§3.7a, ~381 tokens of headroom) would increase the *guaranteed*-stable
portion of block 0, making cache reliability less sensitive to small changes
in dynamic-suffix length. Cheap; no routing or slot-order change required.

*Added context (2026-06-23, post-Lever-3 verification): the oMLX dashboard's
own Serving Stats reports 58.4% cache efficiency (512 of 877 total prefill
tokens cached) for the warm-up fixture's prompt shape in isolation — this is
the real per-call baseline Lever 1 would be improving on. The ceiling on
efficiency for any single prompt of this approximate length is structural:
block 1 (the trailing partial block) is unconditionally uncacheable
regardless of what Lever 1 does to block 0's composition. Separately, in a
real multi-turn session, overall efficiency falls well below even this
58.4% figure (observed: 8.5% after several turns) — see the new
three-system-message finding above. Lever 1 addresses block 0's *internal*
reliability for a single call shape; it does not address the
multi-lineage problem.*

**Lever 2 — Reconsider whether dynamic-suffix ordering has real cache payoff
at longer prompt lengths.** §3.7a currently states the dynamic-suffix slot
order (episodic → profile → RAG → tool results → working memory →
instruction) "reflects conceptual layering... not cache eligibility." That
statement was correct under the old (incorrect) session-continuity model. If
real multi-turn sessions push prompts past ~1024 tokens (a third block),
placing the most-stable dynamic slots (episodic, profile) immediately after
the system message could make a second block cacheable on turns where that
content doesn't change — but this is conditional on real session data
showing episodic/profile content is actually stable enough turn-to-turn, not
just stable within one frozen diagnostic fixture. Needs a live diagnostic
before any slot-order change; not assumed to pay off.

**Lever 3 — Pre-warm block 0 at startup. IMPLEMENTED AND CONFIRMED
(2026-06-23).** Implemented as `run_cache_warmup()` in `backend/warmup.py`,
hooked into `main.py`'s `lifespan()` immediately after
`_state.controller = controller` and before the "ControllerAgent ready" log
line. A single best-effort `runtime.infer()` call, built via the real
`PromptBuilder.build()` against a dedicated fixture
(`templates/warmup_fixture.md`, parsed by `parse_warmup_fixture()`), runs
once per backend process boot. Fails open: any failure (oMLX unreachable,
fixture missing/malformed, prompt assembly error) is logged as a warning
and startup proceeds normally with no warm cache, never raising and never
delaying server readiness beyond the warm-up call's own duration.

Design decision, locked during implementation: success is defined as block 0
reaching *any* cache-resident tier (disk or hot), not specifically the hot
tier. Disk-resident-but-not-yet-hot is an acceptable end state, since
disk→hot promotion was separately confirmed (see below) to be cheap and
synchronous whenever it happens. The hook issues exactly one call — no
polling, no retry, no second call to force hot-tier promotion.

Verification chain (all live, against the real running system, in order):
1. Initial probe-tool diagnostic (this section's "Confirmed mechanism"
   findings above) established the baseline mechanism and predicted Lever
   3's effect.
2. A zero-delay follow-up diagnostic confirmed disk→hot promotion is
   **synchronous, complete before `infer()` returns** (visible at a
   measured 2.3 µs post-call probe latency) — resolving an open question
   from the first Lever-3-shaped diagnostic about whether promotion was
   async. (The reverse direction — cold→disk write timing — was not
   resolved and remains genuinely untested; see "Still open" below.)
3. Implementation (`warmup.py` + `warmup_fixture.md` + `main.py` hook),
   352/352 tests passing, 0 regressions from the pre-implementation
   342-test baseline.
4. Fixture content reviewed directly for realism (concrete wiki-search and
   note-fetch tool outputs, a substantive six-turn research conversation) —
   judged qualitatively realistic, not a synthetic placeholder.
5. Fixture's real token/block shape confirmed against the actual tokenizer
   and probe endpoint: 877 tokens, 2 blocks, block 0 = tokens 0–511 —
   structurally identical in composition to the original diagnostic
   prompts despite being meaningfully longer in content. No 3-block
   overflow; no shape mismatch with production traffic's expected block 0.
6. Live verification against a real `start_localist.sh` boot (oMLX
   pre-confirmed reachable via health check; model not yet loaded —
   `loaded_count: 0` — so this run also exercised cold model load, not
   cache promotion alone): startup logs showed
   `Cache warm-up complete — block 0 promoted to cache-resident tier
   (8874 ms)`, followed immediately by `ControllerAgent ready` and
   `Application startup complete`, confirming hook ordering is correct —
   it runs and completes before the server accepts any request. A
   post-boot probe (preceded only by passive health-check polls, no real
   chat requests) showed: `total_tokens: 877`, `total_blocks: 2`,
   `blocks_ssd_hot: 1`, `blocks_cold: 1` (the unconditionally-cold tail,
   as expected), `ssd_hit_tokens: 512`. Cross-checked independently via
   the oMLX dashboard's Serving Stats panel (a separate UI/instrumentation
   path from the `/admin/api/cache/probe` endpoint): Total Prefill Tokens
   877, Cached Tokens 512 — exact agreement with the probe figures via a
   wholly independent measurement surface.

**Verdict: confirmed, with a real limitation surfaced after live
deployment.** A single warm-up call at backend boot reliably promotes
block 0 to a cache-resident tier before any real user request, with
negligible added risk (fail-open, single call, no production dependency on
the diagnostic-only probe endpoint) — this part of the original prediction
holds exactly as tested. **What was not anticipated:** in real multi-turn
production use, the warm-up's cache-resident block 0 is never hit again,
because neither the main conversational call nor the Tier 2 working-state
call shares a byte-identical system message with the warm-up fixture (see
the three-system-message finding above). The hook does exactly what it was
built and tested to do; it does not, by itself, improve steady-state
multi-turn cache efficiency, which depends on a separate question (whether
any of the three call types can be made to share a leading block) not
addressed by this implementation.

**One nuance for future reference:** the 8874 ms warm-up duration observed
in live verification includes cold *model load* time (the model was not
resident in oMLX at all before this test — `loaded_count: 0`), not warm-up
prefill time alone. On a boot where oMLX already has the model loaded (e.g.
if the backend restarts more often than oMLX does), the warm-up call's
duration would be substantially lower — this number should not be read as
the steady-state cost of the hook.

**Still open, not resolved by this work:**
- Cold→disk write timing (the original ambiguity from the first
  Lever-3-shaped diagnostic) was never directly tested — every subsequent
  run either found disk already populated or jumped straight from cold to
  hot in a way that didn't isolate the write step. Low practical urgency
  now, since disk→hot promotion is confirmed cheap regardless of how long
  the write itself takes, but it remains a genuine gap in the mechanism
  picture, not a closed question.
- The oMLX dashboard displays an idle-unload countdown for the loaded model
  (observed: "idle 4m 54s / ~10m 6s left" before eviction in one session,
  and separately "idle 13m 59s / ~1m 1s left" in another). If this is a
  real eviction policy and not just a display artifact, it has direct
  bearing on Lever 3: a long-idle Localist session could see oMLX unload
  the model entirely, after which the *next* real request — not the
  warm-up hook, which only runs once at backend boot — would pay the full
  cold-load cost again. Unscoped and unconfirmed; flagging only so it
  isn't lost.
- The dashboard's separate "Runtime Cache Observability" panel reports a
  distinct "Memory: _ MB / 2.0 GB · N entries" figure (observed values
  varying across sessions: 28 MB/1 entry shortly after one boot, 392
  MB/14 entries after a multi-turn session reached 8.5% efficiency). This
  appears to be a different cache-accounting surface from the
  `blocks_ssd_hot`/`blocks_ssd_disk` block-tier model documented above —
  its relationship to block-level tier state, and whether "entries" tracks
  per-cache-lineage state (which would make 14 entries plausibly
  correspond to the three-system-message finding's competing lineages,
  accumulating over turns), is not understood and was not investigated.
  Noted as an unexplained observation, not folded into the confirmed
  mechanism above.
- **New, highest-priority follow-up:** whether any of Levers 1/2/4, or a
  new fifth option (making the Tier 2 working-state call and/or the main
  conversational call share a byte-identical leading system message with
  each other or with the warm-up fixture), would address the real
  multi-turn efficiency problem the three-system-message finding
  describes. Not scoped. A live diagnostic logging pass on
  `extract_working_state_update()`'s outcomes (separate from cache
  mechanics — see working-state-update Tier 2 pre-gate work) is in
  progress as of this update and may independently reduce one of the
  three competing lineages if its outcome leads to gating that call on
  some turns.

**Highest-priority follow-up — IMPLEMENTED 2026-06-23 (same day as
diagnosis): all three calls now share a byte-identical leading prefix.**

Implemented via a new `_build_wsu_system(persona)` in `episodic_extractor.py`,
which constructs the Tier 2 system message as
`PromptBuilder._slot1_system(persona) + "\n\n" + _WSU_TASK_INSTRUCTIONS`
(the renamed, content-unchanged former `_WORKING_STATE_UPDATE_SYSTEM`).
`extract_working_state_update()` and `process_working_state_update()` each
gained a `persona: str | None = None` parameter, threaded from
`controller_agent.py`'s existing `self._load_persona()` call at the Tier 2
call site (~line 1249) — reusing the already-cached persona load from the
same turn's main-call construction, not a second corpus query. Lever 1
(persona growth) was folded into this same fix, since a shared prefix only
has practical cache value if it's long enough to cover a meaningful portion
of block 0: `lora-persona.md` was grown from ~476 chars (~119 tokens) to
1,951 chars (~487 tokens) — real, previously-authored content (an earlier
draft of the persona, trimmed back down to fit the existing 500-token
`_CEIL_PERSONA` ceiling) rather than invented padding. `_SYSTEM` (160 chars)
+ this persona lands at ~528 tokens combined — 16 tokens past the 512-token
block-0 boundary, meaning block 0 is now covered entirely by content
genuinely shared across all three call sites, with a small uncontested
margin into block 1.

*What's confirmed, and at what strength of evidence:*
- **Confirmed by test, with the real on-disk persona file (not a
  placeholder string):** `_build_wsu_system(actual_persona)` produces output
  whose leading bytes are identical, string-for-string, to
  `PromptBuilder()._slot1_system(actual_persona)` — proven via a dedicated
  test that reads `wiki/lora-persona.md` from disk, parses it through the
  same `parse_wiki_doc().body[:2000]` path `_load_persona()` itself uses,
  and asserts both the byte-identical prefix and the exact suffix shape
  (`"\n\n" + _WSU_TASK_INSTRUCTIONS`, nothing duplicated or mangled at the
  seam). This is real, code-level proof that the *construction* is correct.
- **Not yet confirmed live:** whether this construction-level fix actually
  changes `blocks_ssd_hot` / cache-efficiency behavior in a real multi-turn
  session — i.e., whether the original 8.5%-efficiency finding improves.
  The byte-identical-prefix test proves the prompts *would* hash to the
  same lineage; it does not, by itself, re-run the live diagnostic that
  measured the original problem. That re-verification (re-running the same
  kind of multi-turn session and re-probing `/admin/api/cache/probe` or the
  oMLX dashboard's Serving Stats) has not yet been done as of this writing
  and is the natural next step before this item can be marked fully closed
  rather than "implemented and unit-verified."
- **A separate, narrower gap worth naming:** `lora-persona.md`'s on-disk
  edit required re-indexing into `document_index` for `_load_persona()`
  (which reads via `query_corpus()` from the DB, never from disk directly)
  to actually serve the new content — this was done via direct SQL update
  against `localist_memory.db` (content, token_set, and content_hash
  refreshed; existing embedding preserved rather than recomputed, since the
  identity/persona document is not typically retrieved via semantic
  similarity). The stray, unreferenced `lora_memory.db` was also updated in
  the same operation — outside the stated scope of the request that
  prompted it. Harmless given that file's confirmed unreferenced status,
  but logged explicitly per the project's standing discipline of not
  letting out-of-scope side-effects pass without comment, however benign.

**Lever 4 — Treat the trailing partial block's unconditional coldness as a
budget signal, not a defect to fix.** Since the trailing partial block is
*always* cold regardless of content or ordering, there may be limited ROI in
optimizing slot order within it. The more relevant move may be ensuring
expensive-to-recompute content (long RAG snippets, long tool results) lands
in the cacheable leading block(s) wherever possible, while accepting that
cheap, naturally-volatile content (the instruction itself) is what gets
recomputed every time regardless of ordering.

**Explicitly NOT recommended, regardless of which levers are pursued:**
artificially padding prompts to force a block boundary purely to improve a
cache-efficiency number — this trades a real, certain prefill cost for an
uncertain caching benefit with no evidence of net positive ROI.

**Current state (updated 2026-06-23):** Lever 3 is implemented, tested, and
live-verified as doing exactly what it was built to do — but live
multi-turn deployment surfaced that the original four-lever framing did not
anticipate the real driver of steady-state cache efficiency: three
structurally distinct system messages in rotation, none sharing a leading
block with any other. This is now the most important open question in this
section, ahead of Levers 1, 2, and 4 as originally scoped. Levers 1, 2, and
4 remain undecided option space, unchanged from the original framing, and
none has been scoped into a Claude Code prompt.

---

### 3.7c Update — Live Re-Verification of the Shared-Prefix Fix: Negative (2026-06-24)

The "not yet confirmed live" item flagged in the implementation writeup above has now
been tested directly, in a real multi-turn session, with a real `/admin/api/cache/probe`
cross-check. The result is negative — documented here in full so it is not mistaken for
"still pending" in any future session.

1. **Persona content and combined system-message length independently re-confirmed,
   byte-exact, from three separate sources this session:** `cat`'d directly from
   `wiki/lora-persona.md` (1,951 chars, matching the implementation record above exactly);
   reconstructing `_SYSTEM + "\n\n" + persona` from this disk content produces exactly
   **2,113 characters**, matching both a real live backend log's own `system_chars=2113`
   debug field (from a `13:19:04` conversational turn this session) and an independently
   built `/admin/api/cache/probe` payload. All three agree exactly — there is no remaining
   doubt about what the real per-turn system message currently contains, nor any doubt that
   the construction described above is what's actually running in production right now.

2. **Two real conversational turns ran with this exact 2,113-character / 525-token system
   message** (confirmed via live backend log `_execute_plan: assembled system_prompt:`
   dumps, identical text on both turns), the model already warm for the second of the two
   (no cold-load confound).

3. **`POST /admin/api/cache/probe`, run immediately after, against this exact payload,
   twice in direct succession:** identical result both times —

   ```
   total_tokens: 525, block_size: 512, total_blocks: 2,
   blocks_ssd_hot: 0, blocks_ssd_disk: 0, blocks_cold: 2,
   ssd_hit_tokens: 0, cold_tokens: 525
   ```

   Fully cold, both blocks, both calls. Not the "block 0 hit, block 1 cold" pattern the fix
   was designed to produce for this lineage.

4. **Probe self-write ruled out as a confound:** sending identical content to the probe
   twice in a row with no real inference call in between also showed fully cold both times
   — consistent with the endpoint's own documented behavior (a hash-and-check walk against
   existing cache state, not a prefill) and confirming the probe itself never writes to
   cache. The two real conversational turns were the only events in this test capable of
   writing cache state for this content; neither produced a hit on the immediately-following
   probe.

5. **Aggregate dashboard efficiency climbed from 23.8% → 30.3%** over the same live session
   window (session-scoped, cleared beforehand; cached tokens 1,536 → 3,072, both exact
   multiples of the 512-token block size) — a real, positive trend, but uninformative about
   *which* of the three competing system-message lineages produced it, since the figure
   aggregates all prefill traffic without attribution. Fully consistent with this
   persona-bearing lineage getting zero hits while some other lineage (most plausibly the
   warm-up fixture re-hitting itself) accounts for the entire observed gain.

**Verdict: the fix remains correctly implemented at the construction level (re-confirmed
independently this session) but is NOT producing the cache benefit it was designed to
produce, based on direct live evidence rather than absence of evidence.** This should be
read as a genuine negative result going forward, not as "still awaiting verification."

**Root cause is not yet diagnosed; per standing project discipline, no fix should be
proposed before it is.** Candidate explanations, none yet investigated:
- The two turns observed may not have been the first time this content was ever sent —
  if the most recent model reload (confirmed to have happened earlier in this session)
  invalidated prior cache state, these two turns would both be "after invalidation, before
  re-caching," not proof the content has never cached. Not ruled out.
- Whether `OMLXRuntimeClient.infer_stream()`'s real streaming call path triggers the same
  `store_cache` behavior as the warm-up hook's non-streaming `infer()` call has not been
  directly checked — Lever 3's cache-writing behavior is confirmed only for the non-streaming
  path. If streaming and non-streaming calls are handled differently by oMLX's own caching
  logic, this would be a previously-unconsidered mechanism gap.
- Whether oMLX's block-hash chaining has any dependency beyond the system message's own
  token range (e.g. generation parameters such as `temperature`/`max_tokens`) is speculative
  and not source-confirmed, but not yet ruled out either.

**Next step (not yet started):** a direct source read of oMLX's `store_cache` call path
(same installed-package directory used for the original 2026-06-23 mechanism read), followed
by a targeted live test isolating streaming vs. non-streaming calls against identical content.

---

### 3.7c Update — Dashboard Observation, Sustained High Efficiency Across Restarts (2026-06-25)

A new aggregate data point, recorded as supporting evidence for this open investigation — **not**
a probe-confirmed finding, and explicitly not given the same evidentiary weight as the byte-level
verification above. Per this project's "verify the mechanism, not just the correlation" discipline,
this is logged as a dashboard reading pending live-probe confirmation, not as a resolved result.

**Correction, added immediately after this entry was first written:** this session was run on
**oMLX v0.4.4**, confirmed completed before this session began and consistent as a single binary for
the session's entire duration — not a mid-session change, and not a variable that differs between
the dashboard observation below and the rest of this session's work. Every prior entry in this §3.7c
thread — the original 2026-06-23 mechanism finding, the 2026-06-24 negative-result probe, and the
source-code read citing `/opt/homebrew/Cellar/omlx/0.4.2/...` — was conducted against v0.4.2, on
separate earlier sessions. The confound is at the boundary between this session and that prior work,
not within this session. The upgrade was not staged as a planned, controlled variable for this
investigation; it is logged here as a confound identified after this entry was first drafted, not as
a deliberate variable the analysis below was designed to isolate. **The "why is this notable"
analysis below was originally written assuming same-binary continuity with the 2026-06-24 probe —
that assumption is false.** The analysis is left in place rather than deleted, because the question
it raises (does sustained high efficiency survive restart/standby) is still a real question — but
every conclusion below must now be read as "possibly a v0.4.2 mechanism finding, possibly simply how
v0.4.4 behaves," with no way to distinguish those from a dashboard reading alone.

**Observation:** during this session's live testing (the Open Item 11 reproduction and gate-
calibration/backstop verification work, recorded above), the oMLX Serving Stats dashboard reported,
for `gemma-4-e4b-it-4bit`, all-time/session figures of: 41,236 total prefill tokens, 26,112 cached
tokens, **63.3% cache efficiency**. Reported as sustained — i.e., not reset to near-zero — despite
the Localist Runtime being restarted and the model going idle/into standby multiple times over the
course of the session.

**Why this is notable relative to the existing negative result — now qualified by the version
confound above:** the 2026-06-24 update found fully cold probe results (`blocks_ssd_hot: 0,
cold_tokens: 525`) for two real, identically-worded conversational turns on **v0.4.2**, against a
dashboard efficiency climbing only modestly (23.8% → 30.3%) over that session — and explicitly
concluded that modest aggregate gain was uninformative and likely attributable to a different
lineage entirely (the warm-up fixture self-hitting). 63.3%, persisting across multiple
restart/standby cycles, is a substantially larger and more durable figure than anything previously
recorded for this investigation — but it was recorded on **v0.4.4**, a different binary, so it
cannot be directly compared to the 23.8%/30.3%/cold-probe figures as if they were the same system
at two points in time. If this dashboard number reflects real, mechanism-confirmed cache reuse,
restart/standby persistence would be new information not covered by either candidate explanation
already on file (model reload invalidating cache state; streaming vs. non-streaming `store_cache`
differences) — but it would be new information about **v0.4.4 specifically**, not necessarily a
correction to what was found on v0.4.2.

**What this does NOT establish, stated plainly — now a longer list given the version confound:**
this is one dashboard reading, not a live `/admin/api/cache/probe` call against known, byte-confirmed
content, and not attributed to any specific lineage (system message vs. user message vs. some other
prefill source). It carries the same aggregation problem the 2026-06-24 entry already named — the
figure sums all prefill traffic without distinguishing which calls, paths, or content produced the
hits. It does not, on its own, overturn the 2026-06-24 negative result for the specific
persona-bearing system-message lineage that result was about — both because of the aggregation
problem and now because of the version difference, either of which independently breaks any direct
comparison. It also does not establish that v0.4.2's documented cold-probe behavior would reproduce
or fail to reproduce on v0.4.4; that is now an open, distinct question this entry cannot answer.

**Status: logged as supporting evidence, investigation remains open — now with an added
prerequisite.** This raises the priority of the next step already named above (direct source read
of `store_cache`, followed by a targeted live test isolating streaming vs. non-streaming calls) —
the restart/standby persistence detail is a new and specific enough observation that it may be worth
probing directly rather than waiting for it to recur. **However, given the v0.4.2 → v0.4.4 version
confound identified above, that source read must now target the v0.4.4 installed package path, not
the v0.4.2 path cited in the original 2026-06-23 finding (`/opt/homebrew/Cellar/omlx/0.4.2/...`) —
the two versions' `store_cache` implementations cannot be assumed identical without checking.** Not
yet investigated as of this entry; no mechanism claim is made for either version.

---

*(Reminder, unchanged from before: §3.7 and §3.7b's "no current backend...
can produce a user-message cache hit" line is still flagged inline as
superseded but not yet corrected — see line 824 of the current doc. That
edit remains a deliberately separate, unbundled task and was not addressed
in this update.)*

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

**Open Item 9 — Empty `[CONTEXT]` on identity-route queries. CLOSED 2026-06-22.**

*Originally:* observed in the same live-verification pass as Open Item 8's fix (2026-06-21) — for
two of the three identity-route reproduction queries (`"Who are you?"` and `"What can you do?"`),
`query_corpus(doc_type="wiki")` returned only `lora-persona.md` as a relevant wiki candidate, which
the existing persona-exclusion guard then removed, leaving `[CONTEXT]` empty. Logged as open, no
fix direction decided, pending a live-tested diagnostic pass across more identity-phrasing variants.

*Diagnostic pass (2026-06-22):* all 13 `_IDENTITY_KEYWORDS` phrasings from `planner.py` were run
through a read-only probe against the live backend, capturing each query's full top-3
`query_corpus(doc_type="wiki")` result set plus a direct cosine-similarity score against
`lora-persona.md` specifically (independent of whether persona made the top-3). Result: 11 of 13
phrasings returned populated `[CONTEXT]` (1–2 survivors after persona exclusion); the same two
phrasings from the original observation (`"Who are you?"`, `"What can you do?"`) remained empty.
Persona similarity for the two empty cases (0.490, 0.484) was solidly mid-range, ruling out
"persona's score is unusually dominant" as the mechanism — both cases returned only one document
in their top-3 entirely, with that document being `lora-persona.md`.

*Two candidate mechanisms were proposed and disproven before the actual root cause was found —
preserved here deliberately, not smoothed over, per this project's standing discipline of stating
plainly when an informal description turns out wrong on fresh investigation:*

1. *Keyword-Jaccard bottleneck (disproven).* Hypothesis: `query_corpus()`'s two-stage pipeline
   (rank all docs by keyword Jaccard overlap, re-rank the top `2×max_results` by cosine) was
   producing a shrunken candidate pool for these two low-keyword-overlap phrasings. Direct
   inspection of `query_corpus()` disproved this: `pool = scored[:max_results*2]` and
   `top = scored[:max_results]` are unconditional slices with no internal threshold, dedupe, or
   early-exit — the function's own logic guarantees exactly `max_results` results whenever at
   least that many documents of the requested `doc_type` exist, regardless of score values. A
   live corpus-size check (`document_count(doc_type="wiki")` = 6) confirmed the corpus itself
   was never the constraint either.
2. *Relative-path cache drift (disproven).* A first live trace of the two failing queries showed
   `_check_cache()` returning a hit, with a cached payload whose paths appeared to be short
   filenames (`lora-persona.md`) rather than the absolute paths `document_index` currently stores
   — suggesting a path-format migration had silently broken cache hydration. A second, deeper
   trace disproved this directly: the short filenames were a display artifact of the trace
   script itself (printing `Path(e["path"]).name` instead of the full stored path); the underlying
   cache payload always contained correct, matching absolute paths. `git log` confirmed
   `index_document()` has used `Path(path).resolve()` since the very first commit that introduced
   `MemoryManager` — there was never a relative-path era for this table.

*Actual root cause:* `_query_hash(query, top_n)` in `memory_manager.py` hashed only the query
string and `max_results` — `doc_type` was never part of the cache key. `query_corpus()` calls this
hash with the same `query`/`max_results` regardless of `doc_type`, so a `retrieval_cache` entry
written for one `doc_type` (e.g. `None`, wiki+raw combined) could be served as a hit for a later
call with a different `doc_type` (e.g. `"wiki"`). `_hydrate_cache_result()` then filters the
already-hydrated cached docs down to the requested `doc_type` *after* retrieval, silently dropping
any cached docs of the wrong type. Both originally-failing queries had real, valid (`valid=1`)
cache entries written for `doc_type=None` at an earlier point — `"Who are you?"` on 2026-06-18,
`"What can you do?"` on 2026-06-21 — each containing 3 absolute paths (a mix of `wiki/` and `raw/`
docs). On a `doc_type="wiki"` call, only the single `wiki/` doc in each cached payload survived
the post-hoc filter, and that doc was `lora-persona.md` in both cases — which the persona-exclusion
guard then removed, yielding empty `[CONTEXT]`. This is not specific to P4a or to identity
questions: any caller of `query_corpus()` that varies `doc_type` across calls sharing the same
query text and `max_results` is exposed to the same collision. It happened to surface through the
P4a route because P4a is the only caller that forces `doc_type="wiki"` on text that other routes
or earlier sessions may have queried with `doc_type=None`.

*Fix:* `_query_hash()`'s signature extended to `_query_hash(query: str, top_n: int, doc_type: str
| None)`, with `doc_type` folded into the hashed string. Its one call site, inside `query_corpus()`,
updated to pass `doc_type` through. No other method (`_write_cache`, `_check_cache`,
`_hydrate_cache_result`) required modification — `_write_cache` already accepted a pre-computed
hash string and `_check_cache`/`_hydrate_cache_result` are agnostic to how the hash was derived.
No schema change — `doc_type` enters the hash input only, not a stored column. Existing cache rows
computed under the old 2-field hash become unreachable under the new 3-field key and are left in
place rather than purged; this is harmless and intentional — a fresh 3-field-keyed cache miss now
falls through correctly to a real keyword+embedding re-rank for any query previously polluted by a
cross-`doc_type` collision.

*Separately found, separately fixed (not folded into this root cause, by deliberate choice — see
§10's precedent for treating co-occurring failure shapes independently):* `backfill_embeddings.py`
writes directly to `document_index.embedding` via its own raw `sqlite3.Connection`, bypassing
`MemoryManager` and never calling `_invalidate_cache()`. A single `UPDATE retrieval_cache SET
valid = 0` was added once after the script's embedding-update loop completes (not per-row),
matching the script's existing raw-SQL pattern rather than refactoring it to construct a
`MemoryManager`.

*Live-verified, in stages, against three different conditions before the real one was confirmed —
preserved here as a worked example of the project's "verify the mechanism, not just a
symptom-correlation" discipline, the second such pattern this arc surfaced after mount-staleness:*

1. A first re-run returned 3 docs for both queries — but against a freshly-reindexed, *empty*
   database using the keyword-only fallback path (no embed model loaded), which is a different
   code branch than the one that produced the original bug. Confirmed the fix's mechanism in
   isolation; did not confirm it against the original failure's actual conditions.
2. A second re-run, intended to use the real database, was discovered to have connected to
   `lora_memory.db` — a known stray, empty, unreferenced database left over from an earlier
   wrong-target `build_graph.py` run (see §8, Validation-Run Results) — rather than the real
   production database. This was caught before being accepted as evidence, the same discipline
   applied to source-file mount staleness now applied to database-file ambiguity.
3. A corrected final run confirmed, from source (`main.py` → `backend/.env`'s
   `LOCALIST_MEMORY_DB` setting → resolved working-directory path), the real database path
   (`backend/localist_memory.db`); confirmed the original two stale `retrieval_cache` rows
   (same `query_hash`, same `created_at` timestamps as originally traced) were still present and
   still `valid=1` in that real database; computed both the old 2-field hash and the new 3-field
   hash for both queries side by side, showing them to be different values (non-collision
   demonstrated directly, not inferred); and re-ran both queries with the real `EmbeddingEngine`
   against the real database, returning 3 documents each at cosine-similarity-range scores
   (0.39–0.49, as opposed to the 0.0–0.05 range a keyword-only fallback would produce — confirmed
   explicitly to rule out a repeat of stage 1's branch ambiguity).

*Known, accepted gap:* verification in stage 3 constructed a standalone `MemoryManager` pointed at
the confirmed real database and real `embed_fn`, rather than exercising the actual running FastAPI
backend end-to-end through its HTTP endpoint — the backend was not running at verification time.
`controller_agent.py`'s P4a branch is a thin wrapper around the identical `query_corpus()` call
shape that was tested, so divergence risk is low, but this was not a full HTTP-level confirmation
and is recorded as such rather than overstated.

*Test suite:* 339 → 342 (+3 tests in `tests/test_memory_phase1.py`, class `TestQueryHash`:
hash differs for `doc_type=None` vs `"wiki"`, differs for `"wiki"` vs `"raw"`, and is stable for
identical inputs), 0 failures.

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

**Open Item 11 — Fabricated tool-call syntax in generation output. OPEN, mechanism unknown,
2026-06-22.**

*Originally:* a single live turn produced a fabricated tool-call string as the model's entire
visible output, in place of a synthesized answer. Instruction: `"Do a web search then tell when
Microsoft's first formal investment in OpenAI was?"`. Backend logs confirmed routing, LangSearch
dispatch, and prompt assembly all completed correctly — `[TOOL RESULTS]` in the assembled user
prompt contained three real search results before generation. The model's raw completion was:

```
<|toolcall>call:websearch{query: "when was microsoft's first formal investment in openai"}<tool_call|>
```

This tag matches no real format used anywhere in this codebase. `OMLXRuntimeClient.infer()`'s
chat-completions payload contains no `tools` or `tool_choice` field at all — confirmed by direct
inspection of `omlx_runtime_client.py` — so there is no real tool-calling contract for the model
to be honoring or malforming. The string was invented by the model, most likely reflecting
tool-call-shaped patterns present in its training data despite this harness never exposing that
capability.

*Diagnostic (read-only, same day):* a standalone script (`diagnostics/diag_toolcall_fabrication.py`)
reconstructed the exact system prompt, `[TOOL RESULTS]` block, and `[WORKING MEMORY]` block from
the incident's backend log as a fixed fixture, varying only the final `[INSTRUCTION]` line across
4 phrasing variants — including the original instruction verbatim (Variant A) — at 5 repeat runs
each (20 total live `OMLXRuntimeClient.infer()` calls, `temperature=0.30`, `max_tokens=1024`,
matching the incident's real call parameters). Variants tested: (A) original exact phrasing, (B)
search reframed as already-done ("Based on the search results..."), (C) no mention of search at
all, (D) explicit statement that search already happened.

*Result:* 0/20 fabrications. Every run across all four variants correctly treated `[TOOL RESULTS]`
as already-resolved search content and produced a grounded (if sometimes hedged/inconclusive)
answer rather than fabricating a tool-call string. This closes the original phrasing hypothesis —
the literal instruction "do a web search" is not, on its own, a reliable trigger — but does not
explain the original incident, which did occur once, live, under what appears to be the same
prompt shape.

*Mechanism: unknown.* The diagnostic fixture is a faithful reconstruction of what the backend log
*displayed*, but is not a guaranteed faithful reconstruction of full live session state at the
exact moment of the incident — e.g. the true stored `[WORKING MEMORY]` turn content (persisted via
`MemoryManager.get_context_window()`) could in principle diverge from what a finite log excerpt
showed, and that possibility has not been ruled out. No diagnostic has yet tested temperatures
other than 0.30, run counts beyond 20 per variant, or working-memory content other than the one
fixture pulled from the original log excerpt.

*Status:* not reproduced; not root-caused; no fix direction proposed or evaluated. Logged as a
single confirmed live occurrence with unknown recurrence rate. Per this project's standard
discipline for under-specified findings, this should not be treated as fix-ready until either (a)
it recurs and a fuller live state capture is available, or (b) a wider diagnostic sweep (higher
run count, varied temperature, varied working-memory content) establishes a non-zero reproduction
rate. A passive detection guard in `conversational_agent.py` (flagging this output pattern at
generation time and logging the full real prompt that produced it) has been suggested as a future
non-fix instrumentation step, not yet scheduled or implemented.

*Cross-reference (2026-06-23):* §9.5 Open Item 4 confirms, via live diagnostic, a structurally
different but topically related issue on a different call (`extract_working_state_update()`,
`max_tokens=200`) — the model emits a `reasoning_content` delta stream that consumes the full
token budget before any parseable output reaches `content`. This is **not** offered as an
explanation for this item's fabricated tool-call string, which occurred on the main conversational
call (`max_tokens=1024`) under different parameters and remains independently unreproduced and
unexplained. Noted only because both findings involve this model/serving setup producing unexpected
output shaped around its own internal process, on calls this codebase's parsers were not written
expecting. Do not treat Open Item 4 as having root-caused this item.

*Second live occurrence, 2026-06-24, 12:34 — different trigger shape, real backend log captured
directly (not reconstructed from a screenshot/chat excerpt).* Instruction:
`"What do you know about LangSmith Engine?"`. Unlike the original incident, **no tool fired**:
Priority 3's semantic gate scored `knowledge_request_open` highest (0.643) with `gate_fired=False`,
so the plan carried `tools=[]`. The conversational call (`temp=0.30, max_tokens=1024,
prompt_chars=610`, full `[TOOL RESULTS]` block absent from the prompt — there was none to include)
returned, as the model's entire visible answer:

```
<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|>
```

This is the **inverse trigger condition** from the original incident, not a repeat of it. The
2026-06-22 case fabricated a tool-call string *after* a real `web_search` had already run and
real results were sitting in `[TOOL RESULTS]` — fabrication there meant ignoring grounded content
already provided. This 2026-06-24 case fabricated the *same shaped* string when **no tool was ever
offered or dispatched for that turn** — `tools=[]` — on a topic outside the model's training
knowledge. Read naturally, this looks less like a malformed reaction to tool output already present
and more like the model attempting to request a tool call that this harness simply does not expose
(`OMLXRuntimeClient.infer()`'s payload has no `tools`/`tool_choice` field, confirmed previously and
still true). Both incidents share the same malformed delimiter pattern (`<|tool_call...` /
`...<tool_call|>`, never a real matched tag pair in any format this codebase uses), which is itself
notable — two independent live incidents, twelve days apart, different trigger shapes, producing
near-identical syntactically-broken tool-call tokens suggests the *string itself* is something
the base model reaches for, rather than something assembled fresh from prompt content each time.
This is offered as an observation, not a confirmed mechanism.

*New finding not present in the original incident: propagation into a second, independent call.*
The fabricated string was stored verbatim as that turn's answer in `[WORKING MEMORY]`
(`Turn -2 [agent]: {'answer': '\n<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|>', ...}`).
The Tier 2 working-state-update call for that same turn — a separate `infer_stream()` call,
`temp=0.00`, prompt built from this same contaminated working-memory content — returned a near-
identical string (`'\n<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|><eos>'`),
and `extract_working_state_update()` correctly logged this as `PARSE_FAILURE` (`missing label(s)`)
rather than silently accepting it — the existing parse-failure guard from Open Item 4's diagnostic
work did its job here. This establishes that a fabrication in the main conversational answer can
**propagate into a second, structurally unrelated call** simply by virtue of being stored as normal
turn history and later re-read as context — a blast-radius fact, not a root-cause fact. It does not
mean Open Item 11 and Open Item 4 share a mechanism (they remain logged separately, per the
cross-reference above); it means Open Item 11's failure mode, once it occurs, is not necessarily
contained to the single turn it occurs on.

*Adjacent, unverified observation — not part of this finding, logged separately so it isn't lost:*
the same live chat session reportedly included a model-generated remark about oMLX cache state
("cache is building with each turn"). No `/admin/api/cache/probe` call or dashboard read appears
anywhere in the captured backend log for this session, so this claim cannot be checked against
real cache telemetry from the evidence in hand. Flagged because, if accurate as a description of
what the model said, it would be a third instance of the same class of behavior as this item and
Open Item 4 — the model narrating something about its own serving/runtime internals that it has no
actual introspection path to — but on a different surface (plain conversational prose instead of
malformed tool-call tokens) and with no raw evidence yet captured. Not logged as its own Open Item
pending an actual occurrence with backend log coverage.

*Status (updated 2026-06-24):* now two confirmed live occurrences, not one — still not root-caused,
still no fix direction proposed or evaluated, recurrence rate still unknown (n=2 live, against
indeterminate live turn volume). The original diagnostic's 0/20 isolation result is **not**
contradicted by this new incident, since the new incident's prompt shape (`tools=[]`, no
`[TOOL RESULTS]` block) was never one of the four variants tested — the diagnostic sweep covered
only the "tool already ran" trigger shape. A natural next diagnostic step (not yet scheduled) would
extend `diag_toolcall_fabrication.py` with a fifth variant matching this incident's actual shape:
no tool dispatched, `[TOOL RESULTS]` absent, topic outside training knowledge, `temp=0.30`. Two data
points with different trigger shapes still does not license a unified mechanism claim — it licenses
widening the diagnostic, which remains the next concrete step if this is picked up before it
recurs again.

*Third, fourth, and fifth live occurrences, 2026-06-25 — deliberate live reproduction attempt,
three turns, real backend logs, isolating priming vs. instruction phrasing as candidate variables.*
Following the 2026-06-24 update's open question (what trigger shape actually produces this), three
live turns were run specifically to test whether fabrication requires priming from an immediately
preceding real tool-dispatch turn, or is driven by something else.

**Turn 1 ("Test A"), 10:49 — priming present, turn lands on Priority 6, `[TOOL RESULTS]` absent.**
Instruction: `"Can you look up Apples price hike for the MacBook Neo and iPad?"`. Preceded in the
same session by a real `tools=['web_search']` turn (`"What's the latest Apple News?"`) with results
delivered normally. On the test turn: `lookup_request` scored 0.593 (`gate_fired=False`); Priority 4
missed (`top_score=0.424`); Priority 5 no match; **Priority 6 — direct answer fallback**, `tools=[]`.
Assembled user prompt contained only `[WORKING MEMORY]` + `[INSTRUCTION]` — no `[TOOL RESULTS]`, no
`[CONTEXT]`. Model's entire output:

```
<|toolcall>call:web search{queries:[<|"|>Apple price hike MacBook Neo iPad<|"|>,<|"|>MacBook Neo price change<|"|>,<|"|>iPad price increase<|"|>]}<toolcall|>
```

**Turn 2 ("Test B"), 11:11 — priming present, turn lands on Priority 4 (corpus), `[TOOL RESULTS]`
absent but `[CONTEXT]`/`[USER PROFILE]`/`[WORKING STATE]` all present and populated, topic-mismatched.**
Instruction: `"Can you look up their next-generation in-house Microsoft AI models?"`, following a real
`tools=['web_search']` turn (`"What's the latest Microsoft news?"`) in the same session. `lookup_request`
scored 0.598 (`gate_fired=False`); **Priority 4 matched via corpus score (0.582 ≥ 0.550)** — `tools=[]`,
`fetch_rag=True`. The RAG hit pulled two Localist-architecture wiki docs (`localist-master-project-
outline.md`, `localist-software-stack.md`) that have no topical relevance to Microsoft's AI models —
matched on shared technical vocabulary ("AI models," embeddings) rather than subject. `prompt_chars=4874`,
including real prior-turn search results in `[WORKING MEMORY]`. Chat-pane tag: `P4 · Vault ◈ grounded`.
Model's entire output:

```
<|toolcall>call:websearch{query: "next-generation in-house Microsoft AI models Build 2026"}<tool_call|>
```

**Turn 3 ("B1"), 11:17 — no priming (fresh task, no preceding turn in working memory at all), turn
lands on Priority 4 (corpus), same topic-mismatch shape as Test B.** Instruction: `"Can you look up
Microsoft's next-generation in-house AI models?"` — first and only turn in this task; `Turn -1` is the
sole `[WORKING MEMORY]` entry, no prior agent response, fresh `mem_key`. `lookup_request` scored 0.598
(`gate_fired=False`); Priority 4 matched via corpus score (0.584 ≥ 0.550) — `tools=[]`, `fetch_rag=True`,
pulling the same two irrelevant Localist-architecture docs. `prompt_chars=3883`. Chat-pane tag: `P4 ·
Vault ◈ grounded`. Model's entire output:

```
<|toolcall>call:websearch{query:<|"|>Microsoft next-generation in-house AI models<|"|>}<tool_call|>
```

*Interpretation.* Turn 3 (B1) is the decisive result: it reproduces fabrication with **no priming
turn at all**, ruling out "immediately preceded by a real tool-dispatch turn" as a necessary
condition — Test A had priming with an empty downstream prompt, Test B had priming with a populated
(but topically irrelevant) downstream prompt, and B1 had neither priming nor relevant context, yet
produced the same failure. The one factor constant across all three of today's reproductions, the
2026-06-22 original incident, and the 2026-06-24 second incident is **`tools=[]` on the turn that
produced the fabrication** — no exception across five live occurrences to date. The three 2026-06-25
turns additionally share an instruction phrased with an explicit "look up" verb, and a `lookup_request`
semantic score consistently in a narrow 0.593–0.598 band — below the 0.65 gate threshold but well
above a clean miss — across all three, despite three different downstream routing outcomes (Priority
6 empty fallback; Priority 4 RAG hit with irrelevant content; Priority 4 RAG hit with irrelevant
content and no priming). This is read as suggestive that "look up"-phrased instructions landing on a
`tools=[]` turn are a stronger candidate trigger than priming, tool-result-emptiness, or RAG-content
relevance individually — each of which varied across the three turns while the outcome did not.

*This remains a hypothesis, not a confirmed mechanism.* Promoted here from "candidate" to "leading
hypothesis" on the strength of three converging live data points plus one clean disconfirmation
(B1 against the priming theory), per this project's standard for distinguishing hypothesis-consistent-
evidence from confirmed mechanism. Not yet tested: (a) whether the "look up" phrasing is doing real
work versus any instruction landing on `tools=[]`-with-lookup-shaped-semantic-score regardless of
literal verb choice — the originally-proposed B2 variant (priming held constant, non-"look up"
phrasing) was not run this session and remains a natural next check if this is revisited; (b) whether
the 0.59–0.60 score band itself is load-bearing (a near-miss specifically) versus any `lookup_request`
score below 0.65; (c) whether the system prompt's "Your Tools" section framing — "Web search fires
automatically on factual queries" — is contributing by setting an expectation the model then
"completes" via fabricated syntax when that automatic firing doesn't happen on a given turn; this is
plausible given the consistent malformed-but-tool-call-shaped string across all five occurrences, but
untested.

*Status (updated 2026-06-25, superseded later same day — see the generation-time backstop and gate
threshold entries cross-referenced below):* five confirmed live occurrences total (2026-06-22 ×1,
2026-06-24 ×1, 2026-06-25 ×3). Reproduction rate within today's deliberate three-turn attempt: 3/3.
Leading hypothesis: instructions using explicit lookup/search phrasing, landing on a turn where
`tools=[]` regardless of cause (Priority 6 fallback or a Priority 4 RAG hit that doesn't satisfy the
lookup intent), reliably produce fabricated tool-call syntax as the entire model output. Still not
root-caused at the mechanism level (why the model reaches for this specific malformed string remains
unexplained — see the cross-session observation above that the same broken delimiter pattern recurs
across unrelated trigger shapes). A two-part fix was implemented and live-verified later the same
day: see "Gate-Calibration Fix" and "Generation-Time Backstop" entries immediately below.

**Gate-Calibration Fix (Prompt 1), 2026-06-25.** `_SEARCH_INTENT_TEMPLATES["lookup_request"]` in
`planner.py` was missing coverage for the "Can/Could you + look up/look into + [specific object]"
question-form frame that all three of today's reproductions used — the existing five templates were
all bare imperatives with a vague pronoun object. Four templates were added (`"can you look up"`,
`"can you look that up for me"`, `"could you look up"`, `"can you look into this for me"`), with
`_SEMANTIC_GATE_THRESHOLDS` deliberately left unchanged at first, on the reasoning that this looked
like a paraphrase-coverage gap rather than a miscalibration. Live re-verification of the three
original utterances showed real but insufficient movement: 0.593→0.608, 0.598→0.617, 0.598→0.621 —
all three remained below the 0.65 threshold, and two of three still fabricated on re-test (the third
hit a stale query-cache from an earlier same-day run, not a new confound).

Given this evidence — three consistent live measurements, each landing 0.029–0.042 short of
threshold — and per §10.4 Open Item 3's own stated revisit criterion ("revisit if live false
negatives are observed," now satisfied), `_SEMANTIC_GATE_THRESHOLDS["lookup_request"]` was lowered
from 0.65 to 0.60 (`explicit_search_action` at 0.68 left untouched). **Known, named risk:** the
original 18-utterance diagnostic's per-utterance scores for `lookup_request`'s 7 adversarial
negatives are not available in this document or in any retained diagnostic output, so the new
threshold's negative-side margin is unverified. This is an accepted risk consistent with this
project's existing "shippable-but-not-fully-validated" posture for these thresholds; the named
mitigation is that any live false positive on `lookup_request` (gate fires when no search was
intended) is the signal to revisit this value.

**Full live re-verification, all three utterances, post-threshold-fix:**

| Utterance | Score | gate_fired | Result |
|---|---|---|---|
| "Can you look up Apple's price hike for the MacBook Neo and iPad?" | 0.608 | True | Real `web_search` dispatch, 3 real results, grounded answer, no fabrication |
| "Can you look up Microsoft's next-generation in-house AI models?" | 0.617 | True | Real `web_search` dispatch, 3 real results (Microsoft MAI/Build 2026 announcements), grounded answer, no fabrication |
| "Can you look up their next-generation in-house Microsoft AI models?" | 0.621 | True | Real `web_search` dispatch, same real results, grounded answer, no fabrication |

All three routed via `_priority3_tool()`'s semantic-gate path, confirming Priority 3 evaluates and
short-circuits `route()` before Priority 4 is ever reached on these turns. Test suite (file-scoped,
`tests/test_planner_phase3.py`): 65 → 69 (template addition) → 71 (threshold adjustment + two new
boundary tests), 0 failures throughout. Note: these are file-scoped counts, not full-suite figures —
the last confirmed full-suite total remains 339 (2026-06-22); a full-suite re-run to establish the
current project-wide total has not yet been done.

**Generation-Time Backstop (Prompt 2), 2026-06-25 — closes Open Item 11's user-facing impact, not
mechanism.** The gate-calibration fix reduces exposure for one phrasing family but does not address
generation-time behavior on any `tools=[]` turn regardless of cause. A detection-and-substitution
guard was added directly to `conversational_agent.py`, the call site all five live incidents shared.

Placement was confirmed by tracing real code, not assumed: `controller_agent._dispatch()` writes
each agent's `AgentResult` to memory via `memory.add_agent_result()` immediately, before
`_execute_plan()`'s implicit-extraction and working-state-update post-hooks read the same
`results[0].output["answer"]` value — confirming the only point early enough to prevent propagation
into working memory and Tier 2 extraction is inside `ConversationalAgent.run()` itself, before it
returns.

Detection: `_is_fabricated_toolcall()`, a module-level regex
(`<\|?tool_?call.*?call:web.*?tool_?call\|>`, case-insensitive, dotall), matched against all seven
real fabricated strings observed across the five live incidents to date — covering delimiter
variants (`toolcall`/`tool_call`) and call-target variants (`websearch`/`web_search`/`web search`).
Verified against five negative-control strings, including an adversarial near-miss ("You can call
the web_search tool if needed.") that contains both "call" and "web_search" as separate words
without the contiguous `call:web` substring or the `<|tool...tool_call|>` bracketing — correctly not
matched. No real tool-calling contract exists in any runtime client in this codebase, so any match
is unambiguously fabrication.

On detection, at both the prebuilt-prompt call site (all five live incidents) and the legacy RAG
call site (no live incidents, but identical structural exposure — included for consistency): `answer`
is replaced with a fixed fallback message ("I don't have live search results for that — here's what
I know from training, which may be stale or incomplete."), and `output["grounded"]`/`output["sources"]`
are forced to `False`/`[]` regardless of what they would otherwise have been — confirmed by dedicated
tests that the guard overrides a real `plan.fetch_rag=True` on the prebuilt path and a real
corpus-hit-derived `grounded=True` on the legacy path. No retry is attempted. New test file
`tests/test_conversational_agent_toolcall_guard.py`: 0 → 36, 0 failures.

**What this closes and what it does not.** This closes Open Item 11's user-facing impact: a turn that
fabricates this pattern can no longer surface the malformed string to the user, store it in working
memory, or have it re-read as context by a later turn — the propagation behavior documented above and
re-confirmed live during this same fix's verification pass (the Apple-utterance fabrication appearing
as `Turn -2 [agent]` context on the following turn, before the threshold fix was applied) is now
structurally prevented at the source. This does **not** close Open Item 11's "mechanism unknown"
status — why the model reaches for this specific malformed string when it does remains unexplained.
The model may still attempt to emit the pattern internally; this guard ensures it never reaches the
user or persists anywhere.

**Live verification of the backstop is explicitly limited, not papered over.** Fabrication is
non-deterministic and cannot be reliably forced on demand, unlike the gate-calibration fix's
live-verifiable score. The 36 mocked-runtime tests are the primary confirmation that the guard works
mechanically. If a live recurrence is observed in normal use going forward, the check is: confirm the
returned answer is the fallback message and `grounded=False`/`sources=[]` for that turn.

**Status: Open Item 11's user-facing impact closed (2026-06-25); mechanism remains open and
unexplained.** Both halves of the two-prompt plan (gate calibration; generation-time backstop) are
implemented and verified to the extent each could be.

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

**Open Item 4 — `extract_working_state_update()` PARSE_FAILURE root-caused:
reasoning-token exhaustion against `max_tokens=200`. CONFIRMED by live
diagnostic, FIXED 2026-06-23 (same day, bundled with §3.7c shared-prefix
fix below). See full status note at end of this entry.**

*Originally:* logged via newly-added `WSU_DIAG` read-only diagnostic
instrumentation (no gating logic, no behavior change) added to
`process_working_state_update()` in a prior session, in response to a single
live incident where a Tier 2 call returned an unparseable bare `'\n'`.

*Live session evidence (2026-06-23):* a real session of 4 conversational
turns — a simple knowledge question, a no-tool lookup, and a tool-call turn —
produced **4/4 `PARSE_FAILURE` outcomes**, zero `CHANGED`, zero
`UNCHANGED_NONE`, zero `INFER_FAILURE`. Every failure returned the identical
bare `'\n'` raw response. Every call's `infer_elapsed_s` fell in the same
12–14s band (11.958–13.972s) regardless of turn content or prompt length
(`prompt_chars` ranged 334–1905 across the four calls). This moved the
finding from "single historical incident, mechanism unknown" (§8.8 Open Item
11 — see cross-reference below) to deterministically reproducible under this
specific call's exact conditions.

*Isolation diagnostic (read-only, same day, direct-to-oMLX, bypassing the
FastAPI app entirely):* a standalone script sent the exact
`_WORKING_STATE_UPDATE_SYSTEM` system message and a reconstructed user prompt
matching the real call shape directly to `http://localhost:8000/v1/chat/completions`,
mirroring `OMLXRuntimeClient.infer_stream()`'s real request/response contract
exactly (`stream=True`, OpenAI-compatible SSE envelope, `choices[0].delta.content`)
after an initial mis-shaped non-streaming test attempt failed with `KeyError`
and was corrected by reading `omlx_runtime_client.py`'s real `infer_stream()`
and `foundry_runtime_client.py`'s real `_iter_sse_chunks()` source directly
rather than assuming the response shape.

Three cases run, `temperature=0.0` throughout:
1. **All-NONE previous state, `max_tokens=200`** (reproduction attempt): bare
   `'\n'` returned, `finish_reason: "length"`. Reproduced the live failure
   exactly.
2. **Realistic non-empty previous state** (`current_focus`, `open_loops`,
   `recent_decisions` all populated with real-looking values), `max_tokens=200`:
   bare `'\n'` returned, `finish_reason: "length"`. Identical failure despite
   non-empty state — **this rules out the all-NONE-previous-state-confuses-the-model
   hypothesis directly**, rather than merely deprioritizing it.
3. **Same all-NONE prompt as Case 1, `max_tokens=2000`**: succeeded.
   `finish_reason: "stop"`. Returned a clean, fully parseable three-line
   response: `FOCUS: Explaining the Localist Framework / OPEN_LOOPS: NONE /
   DECISIONS: NONE`.

*Mechanism, visible directly in the raw SSE stream (not inferred):* the model
is emitting a distinct `reasoning_content` delta stream — a multi-step
internal "Thinking Process" narrating its own analysis of the previous state,
the new turn, and each of FOCUS/OPEN_LOOPS/DECISIONS in turn — entirely
separate from the `content` delta stream the parser reads. In both 200-token
failures, the reasoning trace was still mid-step (step 3–5 of its own
6-step internal outline) when the token ceiling was hit; the only token(s)
that reached `content` before the forced cutoff was the stray `'\n'` the
parser then correctly rejected. In the 2000-token success case, the
*identical* reasoning trace ran to its own natural conclusion (its own final
step: *"Format Output: Apply the strict three-line format"*) and only then
did the delta stream switch from `reasoning_content` to `content`, emitting
the three labeled lines cleanly. The reasoning trace alone consumed roughly
500–600 tokens before any `content` token appeared in the successful case —
this is the real floor this call needs for reasoning, before the three-line
answer's own token cost is added on top.

*This explicitly rules out, by direct evidence rather than deprioritization:*
- The SUMMARY/EOS mechanism from Open Item 3 — no "SUMMARY" token appears
  anywhere in `_WORKING_STATE_UPDATE_SYSTEM`, and the failure is a hard
  `finish_reason: "length"` truncation, not an EOS emission.
- Previous-state emptiness as a causal factor (Case 2, above).
- Turn-content dependence — the live session's 4/4 spanned three
  meaningfully different turn shapes with identical failure behavior.

*Not yet established:*
- The true minimum viable `max_tokens` for this call. 2000 was chosen as
  generous headroom for the isolation test, not as a tuned value — it is
  very likely far larger than necessary. No bisection or systematic search
  has been run.
- Whether reasoning-trace length is roughly constant across turn content, or
  varies meaningfully with longer/more complex turns (e.g. a turn involving
  a tool-call result, which the live session's 4th turn was, but which has
  not yet been isolation-tested directly — only reproduced live, not
  isolation-tested with non-trivial response content).
- Whether this same reasoning-token-exhaustion mechanism is present (and
  currently silently absorbed by larger headroom) in any of this codebase's
  other bounded extraction calls — `extract_content_from_instruction()`
  (`max_tokens=60`) and `extract_implicit_episode()` have not been checked
  against this same diagnostic approach. Given `extract_content_from_instruction()`'s
  budget is smaller than the WSU call's, it is at least as exposed in
  principle; not yet verified empirically either way.
- Whether this is a property of the `gemma-4-e4b-it-4bit` model file itself,
  an oMLX 0.4.2 serving configuration/default, or something that changed
  between sessions — no comparison against a non-reasoning call configuration
  has been attempted, and no version/config history has been checked.

*Cross-reference:* §8.8 Open Item 11 (fabricated tool-call syntax, mechanism
unknown) is a structurally different failure on the *main conversational*
call (`max_tokens=1024`, not this call's `max_tokens=200`) and should **not**
be conflated with this finding — they were produced by different calls with
different parameters, and Open Item 11 remains unreproduced and
unexplained on its own terms. The two are noted together only because both
involve this model/serving setup producing output around its own internal
process (an invented tool-call string; an exposed reasoning trace) in a way
this codebase's call sites were not written expecting. Whether they share a
deeper common cause (e.g. reasoning-capable generation behavior this harness
doesn't yet account for anywhere) is an open question, not a finding.

*Status:* root cause confirmed by direct live diagnostic, **fix implemented and
verified, 2026-06-23 (same day).** `max_tokens` raised from 200 to 1024 on this
call — chosen to match the main conversational call's existing budget, not a
newly invented value, and confirmed comfortably above the ~570–600-token
reasoning floor measured above. Implemented as part of the same Claude Code
prompt as the system-message-sharing fix below (Open Item 4 and the §3.7c
shared-prefix fix were deliberately bundled — both touch
`extract_working_state_update()`'s call site). 73/73 tests passing
post-implementation (66 pre-fix → 72 from the sharing/budget change → 73
after one additional realistic-length verification test was requested and
added). No live re-verification of the *fix itself* (i.e. running real turns
and confirming `WSU_DIAG` no longer logs `PARSE_FAILURE` at the new budget)
has been done yet — the isolation diagnostic proved the mechanism and that
2000 tokens resolves it; 1024 tokens has not been independently
isolation-tested, only reasoned to be sufficient by analogy to the
measured ~570–600 token floor plus the main call's own budget convention.
This is a real, if probably minor, gap — flagged rather than assumed away.

The items below, all logged at diagnosis time, remain genuinely open even
after this fix and were not addressed by it:
- The true minimum viable `max_tokens` — 1024 is a reused convention, not a
  tuned or bisected value, and remains unverified against the 1024 ceiling
  specifically (only 200 and 2000 were ever tested).
- Whether reasoning-trace length varies meaningfully across turn content
  (e.g. tool-result turns) — still untested.
- Whether `extract_content_from_instruction()` (`max_tokens=60`) or
  `extract_implicit_episode()` share this exposure — still unchecked.
- Whether this is a model-file property, an oMLX serving default, or
  something that changed recently — still unestablished.

---

## 10. Semantic Search-Intent Classifier

### 10.1 Scope

Before this change, `_priority3_tool()` in `planner.py` triggered `web_search` only when the
instruction contained a literal keyword from `_WEB_SEARCH_KEYWORDS` — a small frozenset of
recency and freshness terms ("latest", "today", "news", "current price", and similar). This
produced a confirmed false negative in a real session.

**The incident:** the user asked "What do you know about APC (Auto Prefix Cache)?" The corpus
and episodic stores lacked APC-specific content; the model correctly stated it lacked specific
information. The user then issued the follow-up: "Go ahead and look it up." No keyword from
`_WEB_SEARCH_KEYWORDS` appeared in that instruction. `_priority3_tool()` returned `None`;
routing fell to Priority 6 (direct answer); and the model responded with fabricated claims about
APC — falsely presented as coming from "Web Search" despite no search tool ever being invoked.

A second, distinct gap was found in the same incident: the full original APC instruction that
preceded the follow-up — "Why don't you do a web search for APC and then tell me if you still
stand by your previous answer?" — contains the literal phrase "web search". That instruction
failed because of a keyword-coverage miss, not for any semantic reason.

This section documents the resulting two-part fix: an embedding-based semantic classifier
layered onto Priority 3 to catch natural-language search-action phrasings, and a literal-keyword
addition to `_WEB_SEARCH_KEYWORDS` to close the separate coverage gap. Each fix targets a
distinct failure mode; they were applied at different layers for that reason.

### 10.2 Design Decisions

**Two distinct failure shapes; two distinct fixes.** "Go ahead and look it up." is a
bare-affirmative follow-up with no topic keyword; no literal phrase reliably covers the space of
such instructions, which require semantic generalization. "Why don't you do a web search for
APC..." contains the exact literal phrase "web search" and failed only because that phrase was
absent from `_WEB_SEARCH_KEYWORDS`. Fixing the first gap by extending the keyword list would
have required open-ended enumeration with no principled stopping point. Fixing the second gap by
tuning the semantic threshold down to capture that one sentence (actual score: 0.638 vs. the 0.68
threshold; margin of 0.023 above the closest false positive) would have changed gate behavior for
all future utterances without any broader justification. Conflating these two failure shapes would
have produced a worse fix in either direction.

**Per-group cosine similarity against four canonical template groups, using the EmbeddingGemma
model already resident in the process.** `_SEARCH_INTENT_TEMPLATES` defines four named groups —
`explicit_search_action`, `lookup_request`, `knowledge_request_open`, `freshness_request` — with
17 template strings total (5+5+4+3). At startup, `Planner.__init__()` embeds all 17 using the
same EmbeddingGemma model (`mlx-community/embeddinggemma-300m-4bit`, 768-dimensional) that
`EmbeddingEngine` already uses for corpus retrieval. The `embed_fn` callable is threaded into
`Planner` as a new optional constructor parameter (`embed_fn: Callable[[str], list[float]] | None
= None`), passed from the `main.py` lifespan function through `ControllerAgent`. `MemoryManager`
already holds this callable for corpus scoring; `Planner` receives its own copy of the same
already-initialized function rather than reaching into `MemoryManager._embed_fn` as a shortcut.
A reach-through pattern exists elsewhere in the codebase (`controller_agent.py`'s `/embed`
endpoint helper); it was noted and deliberately not replicated here.

**Gating uses per-group scores from `all_scores`, not the global argmax.** `_semantic_search_intent()`
returns a 3-tuple `(best_group, best_score, all_scores)` where `all_scores` is a dict mapping
each group name to its own maximum cosine similarity across that group's templates. The gate in
`_priority3_tool()` evaluates each gating group against its own score independently:

```python
semantic_triggered = any(
    all_scores.get(group, 0.0) >= threshold
    for group, threshold in _SEMANTIC_GATE_THRESHOLDS.items()
)
```

This is load-bearing. If gating were evaluated on `best_group` only, a non-gating group winning
the argmax would suppress the gate even when a gating group independently cleared its own
threshold. In the live diagnostic evaluation, `knowledge_request_open` won the argmax in 3 of 7
adversarial negative test cases — demonstrating how frequently this scenario arises in practice.

**Only two of the four groups gate routing; the other two are informational only.** `_SEMANTIC_GATE_THRESHOLDS`
contains exactly two entries: `explicit_search_action` (≥ 0.68) and `lookup_request` (≥ 0.65).
`knowledge_request_open` and `freshness_request` are computed and logged on every turn but are
excluded from gating. The evidence for `knowledge_request_open`: a live diagnostic pass found
that "Explain this code to me." scored 0.795 on that group — higher than 5 of the 10 real
positive search-intent paraphrases tested in the same pass — because that group's canonical
templates ("tell me about this", "what do you know about this", "what is this", "explain this to
me") are generically conversational phrasings that collide with ordinary non-search chat. For
`freshness_request`: one adversarial negative scored inside the positive range during evaluation,
and the group has not been independently stress-tested at a larger sample size. Both groups remain
in the computation pipeline and are emitted to the debug log; neither may gate `tools_to_call`
without a separate evaluation pass. See §10.4, Open Item 1.

**Thresholds were derived from live-backend diagnostics, not tuned to fit any incident utterance.**
The values 0.68 (`explicit_search_action`) and 0.65 (`lookup_request`) were determined from a
structured evaluation pass run before the gating logic was written: 10 positive search-intent
paraphrases, 7 adversarial negatives, and 1 negative-filter case (18 utterances total) submitted
against the live EmbeddingGemma model. The second incident instruction ("Why don't you do a web
search for APC...") was not used to tune these numbers — it scored 0.638, fell below the 0.68
threshold, and was fixed at the literal-keyword layer specifically to avoid post-hoc threshold
adjustment for one known utterance.

**A negative filter short-circuits before the embedding call.** `_SEARCH_NEGATIVE_FILTER` is a
frozenset of 9 phrases identifying meta-instructions that reference the conversation itself or
the search tool — "did you search", "what tool did you use", "search my previous messages", and
similar — rather than requesting a world-facing search. When any of these phrases appears in the
lowered instruction, `_semantic_search_intent()` returns `None` immediately without invoking
`embed_fn`. Verified live: "Did you search for that already?" triggered the filter and produced
no embedding call.

**"web search" and "do a search" added to `_WEB_SEARCH_KEYWORDS`.** The second incident gap was
closed by adding these two phrases to the existing `_WEB_SEARCH_KEYWORDS` frozenset, matched via
the existing `_any_whole_word()` boundary function with no new matching logic. "search for" was
considered and deliberately excluded: it matches sentences like "search for a workaround in my
own code" with no search-tool intent, and adding it would reintroduce the over-broad literal-match
problem this arc was correcting.

**Priority 3's semantic embedding call and Priority 4's corpus-retrieval embedding call are not
shared.** Each computes a separate `embed_fn` invocation on the same query string. Sharing the
computed query vector across both call sites was considered — on any turn where P3 semantic
evaluation and P4 corpus retrieval both run, the vector is identical — and was explicitly
deferred, not rejected, pending latency profiling on the 16 GB development machine. The
EmbeddingGemma model is resident in memory; per-call overhead has not been measured at
production-style turn rates. Revisit only if profiling shows the duplicate call contributes
meaningfully to observed latency. This is the same accept-now/optimize-later posture applied to
other unmeasured-cost decisions in this codebase.

### 10.3 Live Verification — Original Incident, Recreated

The following is a live, two-turn recreation of the original incident, run against the deployed
backend after all four slots of the fix arc were applied.

**Turn 1 — "What do you know about APC (Auto Prefix Cache)?"** The semantic gate did not fire:
`knowledge_request_open` scored 0.624; neither `explicit_search_action` nor `lookup_request`
cleared its threshold. Corpus and episodic retrieval both missed on APC specifically. The model
correctly stated it lacked specific information rather than fabricating claims. **This turn's
correct, non-hallucinating response is not attributable to this fix.** The gate did not fire;
credit for the model's behavior here lies with prompt wording (e.g. the behavioral constraint
"you do not simulate certainty"), not with the classifier.

**Turn 2 — "Go ahead and look up APC (Auto Prefix Cache)."** This is a live, unscripted
paraphrase — not identical to any string in the diagnostic dataset and not identical to the
original incident's exact wording. `lookup_request` scored 0.740, clearing the 0.65 threshold.
`tools_to_call = ['web_search']`. LangSearch returned three real, correctly-disambiguated results
identifying APC as automatic prefix caching in LLM inference serving — including an arXiv paper
and a Chinese-language vLLM technical article independently confirming the same expansion. These
results directly contradicted the original incident's fabricated claims (networking/routing,
database indexing, compression/streaming), confirming those claims were not merely unsupported
but factually wrong.

This is the first live, non-scripted confirmation that the semantic fix generalizes beyond the
diagnostic dataset's curated test strings.

### 10.4 Open Items

**Open Item 1 — `freshness_request` gating status unresolved.** One adversarial negative
example scored inside the positive range during the Diagnostic 2 evaluation pass; the group has
not been stress-tested at a larger sample size. Remains informational-only — not a confirmed-safe
candidate for routing gating — until a separate evaluation pass establishes a defensible
threshold and negative margin.

**Open Item 2 — Embedding call sharing deferred.** Priority 3's semantic classification and
Priority 4's corpus-retrieval check each invoke `embed_fn` independently on the same query
string. Sharing a single computed embedding across both call sites was scoped and explicitly
deferred pending real latency data on the 16 GB development machine. Revisit only if profiling
shows the cost matters in practice; do not optimize without measurement.

**Open Item 3 — Threshold sample size.** 0.68 and 0.65 are derived from a single diagnostic
pass: 10 positive paraphrases and 7 adversarial negatives. Treat as shippable-but-not-fully-validated.
Revisit if live false positives (gate fires when no search was intended) or false negatives (gate
misses a clear search instruction) are observed.

**Open Item 4 — Live near-miss on `explicit_search_action` threshold, compound
instruction (2026-06-23).** A live, unscripted turn — "Look up karpathy llm
wiki then propose ways it implement it into Localist design." — scored
`explicit_search_action: 0.618`, the highest of all four groups
(`lookup_request: 0.597`, `knowledge_request_open: 0.462`,
`freshness_request: 0.404`), but fell short of the 0.68 gating threshold.
`tools_to_call` was not populated; no LangSearch call occurred. Priority 4
matched instead via corpus score (0.638 ≥ 0.550), routing to
`conversational_agent` with wiki-only RAG context. The model correctly
stated it did not have the Karpathy material and asked the user to supply
it, rather than fabricating a claim about Karpathy's content — the
fail-safe (prompt-level "you do not simulate certainty" framing, not the
classifier) held, consistent with Turn 1 of the §10.3 live recreation.

This is structurally similar to §10.3 Turn 1 (a real informational-intent
turn scoring sub-threshold, correctly falling back to an honest non-answer)
but is a distinct data point: a different group won the argmax
(`explicit_search_action` here vs. `knowledge_request_open` in the §10.3
recreation), and the instruction was compound — a lookup clause ("Look up
karpathy llm wiki") joined to a proposal clause ("propose ways... into
Localist design") in a single 81-character instruction. Whether the
proposal clause's embedding signal diluted the lookup clause's score below
what it would have scored alone is a plausible mechanism, not a confirmed
one — no isolated test of the lookup clause alone has been run.

**Not actioned.** Per Open Item 3's standing posture and the project's
single-occurrence discipline (a single proposed mechanism is a hypothesis
to verify, not a finding to act on), no threshold change, no compound-
instruction-splitting logic, and no new gating behavior follows from this
one turn. Logged so it counts toward Open Item 3's "revisit if live false
negatives are observed" criterion — this is one such occurrence, not yet a
pattern. Revisit if additional compound or near-threshold instructions are
observed scoring in the 0.60–0.68 band for `explicit_search_action`.

### 10.5 Test Suite

Final state: **339 tests, 0 failures** across all test files (verified against the live test
suite as of 2026-06-22).

The classifier was built across four sequential slots, all in `backend/tests/test_planner_phase3.py`:

| Slot | Purpose | Before | After | Net |
|---|---|---|---|---|
| Diagnostic 1 | `_semantic_search_intent()` scaffold; `embed_fn` wiring; logging only, no routing change | 318 | 329 | +11 |
| Diagnostic 2 | Expand return type to `(best_group, best_score, all_scores)`; per-group score logging | 329 | 331 | +2 |
| Fix 1 | Live gating via `_SEMANTIC_GATE_THRESHOLDS`; first routing change in `_priority3_tool()` | 331 | 336 | +5 |
| Fix 2 | "web search" and "do a search" added to `_WEB_SEARCH_KEYWORDS` | 336 | 339 | +3 |
| **Total** | | **318** | **339** | **+21** |

Fix 1's net of +5 reflects 7 new tests in `TestPriority3SemanticGating` minus 2 tests removed
from its predecessor class `TestPriority3ToolUnaffectedBySemantic`, whose premise — "semantic
signal never affects routing" — became false after that slot.

*End of Localist Framework Canonical Architecture Specification*
