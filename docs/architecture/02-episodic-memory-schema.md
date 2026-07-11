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
| `embedding` | BLOB | Optional 768-dim float vector, same encoding as `document_index`. Populated by `EpisodicMemoryWriter.insert()` whenever it holds an `embed_fn` (embeds `"{subject}. {content}"`). Nullable — rows written before embedding support existed, or written with no `embed_fn` available, fall back to keyword scoring in Mode 3 (§2.6) on a per-row basis. |

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
| `active` | Trusted. Eligible for injection into context, retrieval, and `MEMORY.md` (§2.9). | Default on creation, or set by `EpisodicMemoryWriter.approve()` from `pending`. |
| `pending` | Staged, unreviewed. **Not** eligible for injection, retrieval, or `MEMORY.md` — invisible to every retrieval mode in §2.6 until resolved. | Set on creation only when the write-approval gate is active for that write (§2.11); never a transition target. |
| `superseded` | Replaced by a newer episode with the same `subject` and `episode_type`. | Set on the old record when a conflicting new **`active`** record is inserted — a `pending` write does not supersede or get superseded by anything until it is approved (§2.11). |
| `retracted` | Explicitly invalidated by user command, or a `pending` write that was rejected. | Set directly; no new record required. |

**Supersession rule:** When a new episode is inserted with `initial_status =
'active'` and an `active` record with the same `subject` and `episode_type`
already exists, the existing record is updated to `status = 'superseded'`
before the new record is inserted. Both records are retained for audit.
Writes with `initial_status = 'pending'` skip this step entirely — an
unreviewed guess must never retire a confirmed fact.

**Retraction rule:** Explicit user commands (`"forget that"`,
`"that's no longer true"`) trigger a retraction write. The record is marked
`status = 'retracted'`; no replacement is inserted unless the user provides
a corrected value.

Retraction resolution went through two implementations. The original one
matched `(subject, episode_type)` exactly, with `episode_type` hardcoded to
`"preference"` — meaning retraction silently no-opped for every other type.
The current implementation (`process_explicit_signal()`'s retraction branch
in `episodic_extractor.py`) instead calls
`EpisodicMemoryReader.best_match(subject, min_score=0.55)` — a semantic
lookup across every active episode regardless of type, using the same
cosine-with-keyword-fallback scoring as Mode 3 (§2.6) — and retracts the
single best-scoring record by primary key via
`EpisodicMemoryWriter.retract_by_id()`, which is strictly precise (no
string-matching ambiguity). This also fixes a second, independent problem:
the retraction path's model-extracted "what to retract" phrasing rarely
matched the stored `subject` string character-for-character, so exact-match
retraction was fragile even for the one type it covered. When no `embed_fn`
is available, or nothing clears the 0.55 threshold, retraction falls back to
looping `retract(subject, episode_type)` over every `VALID_EPISODE_TYPES`
value (cheap and safe — each call is a no-op `UPDATE ... WHERE` if it
doesn't match).

### 2.6 Retrieval Modes

Three retrieval modes cover all Planner use cases. All three implicitly
exclude `pending` rows — none of them filter on `status = 'active'` loosely;
`'active'` is the only status any of them accept.

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
Used for session priming. Loads high-priority durable context. Deliberately
scoped to the four "durable stance" types — `preference`, `correction`,
`decision`, `workflow` — and **not** `project_fact`, `task_completion`, or
`naming_convention`. This scoping is intentional (session priming should
front-load how the user wants to be worked with, not every fact about the
project), but it has a sharp edge covered in Mode 3 below: for a long time,
Mode 2 was the *only* retrieval mode wired into the live pipeline
(`controller_agent.py`'s `_execute_plan`), which meant those three excluded
types were retrievable in principle but unreachable in practice — a
`project_fact` episode could be written, shown correctly in the Memory UI
tab, and still be reported as "no memory of that" the moment the user asked
about it. Fixed by wiring Mode 3 in alongside Mode 2 — see below.

```sql
SELECT * FROM episodes
WHERE episode_type IN ('preference', 'correction', 'decision', 'workflow')
  AND status = 'active'
  AND project_context = :project_context
ORDER BY last_accessed DESC, confidence DESC
LIMIT 5;
```

Since Mode 2's result set depends only on `project_context` — never on the
current instruction — `ControllerAgent` caches it in-memory per
`project_context`, invalidated (cleared entirely, not per-key) on any
episodic write. This is a backend-efficiency measure only (fewer redundant
SQLite reads/`last_accessed` touch-writes on consecutive turns with no
write in between) — it is explicitly **not** a KV-cache prefix-stability
win for the LLM runtime, because the assembled `[EPISODIC MEMORY]` block
combines Mode 2's (now-cacheable) results with Mode 3's
(instruction-dependent, still varies every turn) results before injection,
so the prompt slot downstream of Slot 2 still breaks the runtime's prefix
cache every turn regardless. See §3's stable-prefix/dynamic-suffix
investigation for the actual KV-cache contract; a genuine prefix-cache win
here would require restructuring slot order, which is out of scope for this
section.

**Mode 3 — Semantic similarity**
Used for open-ended queries, and — as of this fix — merged into every
`fetch_episodic` turn in `controller_agent.py` alongside Mode 2, specifically
so `project_fact`/`task_completion`/`naming_convention` episodes are
reachable at all. `EpisodicMemoryReader._score_all_active(query)` is the
shared scoring core: it embeds `query` once (via the reader's `embed_fn`, if
supplied) and scores every active episode — real cosine similarity
(`_cosine_similarity`) against any row with a stored `embedding`, falling
back to keyword (Jaccard) overlap per-row for un-embedded rows or when no
`embed_fn` is available at all. `by_similarity(query, top_n, min_score)`
slices the top N above threshold; `best_match(query, min_score)` (used by
retraction, §2.5) returns just the single top scorer.

`controller_agent.py`'s Step 5 calls both Mode 2 and Mode 3 per turn
(`by_similarity(task.instruction, top_n=5, min_score=0.45)`), merges the
two lists by `id` (deduplicating), and builds `[EPISODIC MEMORY]` from the
union. The 0.45 threshold mirrors the profile-fact scoring threshold used
elsewhere in `controller_agent.py` (`_score_profile_facts`); retraction's
`best_match()` uses a stricter 0.55, since a false-positive retraction
silently destroys the wrong memory where a false-negative in recall is
merely a miss.

Existing rows written before embedding support existed have `embedding =
NULL` and score via keyword fallback until backfilled —
`backend/backfill_episode_embeddings.py` is a one-off script that embeds
every active row missing a vector and regenerates `MEMORY.md` (§2.9) from
current state; safe to re-run.

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

### 2.9 MEMORY.md — Human-Readable Snapshot

`EpisodicMemoryWriter` regenerates a single Markdown file
(`backend/wiki/MEMORY.md` by default, path supplied at construction as
`memory_md_path`) after every successful `insert()`, `retract()`,
`retract_by_id()`, `approve()`, and `reject()` call. It is a **generated
view, not a source of truth** — SQLite remains canonical, including the
full superseded/retracted audit trail; `MEMORY.md` deliberately only shows
`active` episodes (matching what the Memory UI tab shows), grouped by
calendar date (newest first), one line per episode:
`- **{episode_type}** ({confidence}%, {project_context}, {source}) — {content}`.
A `regenerate_memory_md()` public method exists for on-demand rebuilds
outside the write path (used by the backfill script, §2.6, and available
for any future one-off resync). The file header states plainly that it is
auto-generated and will be overwritten — it is not meant to be hand-edited.

This exists because episodic memory before this point was only inspectable
through the Memory UI tab's paginated card view or direct SQL — there was
no single place to read the full current state of "what does LORA
currently believe" at a glance.

### 2.10 Content Safety Scanning on Write

`EpisodicMemoryWriter.insert()` scans both `subject` and `content` (via
`backend/content_safety.py`'s `scan_content()`) before writing, rejecting
the write (`insert()` returns `None`, no row written) if either matches a
known threat pattern: prompt-injection phrasing (e.g. "ignore previous
instructions", forged role-marker strings), credential/key-material
patterns (API key prefixes, PEM/SSH key headers, high-entropy token runs),
or invisible/control Unicode (category `Cf` characters). This is a bounded,
regex/pattern-based check — not an inference call, and not a full
prompt-injection classifier — deliberately kept synchronous and fast since
it runs on every write.

The motivating concern: episodic memory content is replayed forward into
every future system prompt (`[EPISODIC MEMORY]`, `MEMORY.md`, and now
`[USER PROFILE]`-adjacent context). A single unreviewed `model_extracted`
episode containing injected or malicious content, once written, would
re-inject itself into every subsequent session indefinitely. Blocked writes
degrade to "nothing was remembered" — `process_explicit_signal()` and
`process_implicit_extraction()` both treat a `None` return the same as any
other no-op extraction outcome (no durable fact, model said NONE), never
raising.

Out of scope, deliberately: retroactively scanning rows written before this
check existed. If wanted, that should be a separate, explicit one-off
script in the shape of `backfill_episode_embeddings.py`, not a silent side
effect of a future write.

### 2.11 Write-Approval Gate (Pending Review)

Configurable via `LOCALIST_EPISODIC_WRITE_APPROVAL`
(`Settings.episodic_write_approval`, default `false`). When enabled,
`process_implicit_extraction()` (the `model_extracted` path only —
`process_explicit_signal()` is never gated, since an explicit "remember
that X" is direct user consent already) writes with `initial_status =
"pending"` instead of `"active"`, via a `require_approval` parameter
threaded from `Settings` through `ControllerAgent.__init__` →
`self._episodic_write_approval` → the implicit-write call site in
`_execute_plan`.

Two endpoints resolve a pending episode:

```
POST /memory/episodes/{id}/approve   → pending → active  (EpisodicMemoryWriter.approve())
POST /memory/episodes/{id}/reject    → pending → retracted (EpisodicMemoryWriter.reject())
```

Both are idempotent — a repeat call on an already-resolved or nonexistent
id returns `updated: false` rather than a 404/409, consistent with this
being a single-user local app where race conditions are not a primary
concern. `GET /memory/episodes` already accepted an arbitrary `status`
filter and needed no changes to serve `status=pending`, except one fix
along the way: its `total` field was `len(rows)` (silently capped by
`limit`), which meant a `?status=pending&limit=1` query — the exact shape
the UI badge/count needs — could only ever report `0` or `1`, never the
real count. `MemoryManager.count_episodes()` (a true `SELECT COUNT(*)` with
the same filters as `list_episodes()`) fixed this.

The Memory UI tab (`localist-ui/src/lib/components/EpisodesPanel.svelte`)
surfaces this with a distinct "Pending (N)" status filter (separate
dimension from the existing type filters), Approve/Reject buttons on
pending cards, and a live pending count synced via a shared store
(`episodes.ts`'s `pendingCount`) — read by both the Memory tab itself and a
badge next to the "Memory" item in the left sidebar nav
(`Sidebar.svelte`), so a newly staged episode is noticeable without having
to navigate into the tab.

This is the mechanism, not a claim that model-extracted writes are
currently gated by default — they are not (`false` by default, matching
Hermes Agent's own default of "write freely"). Turning it on trades
immediacy for review: a `model_extracted` fact sits invisible to every
retrieval mode in §2.6 until a human resolves it.

