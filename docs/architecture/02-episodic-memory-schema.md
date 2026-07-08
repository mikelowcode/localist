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

