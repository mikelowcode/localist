## 12. Chat History Tab

*Shipped 2026-07-01 across six sequential Claude Code prompts; see
session-log entry for the step-by-step account (`sessions-log.md`,
"2026-07-01 — Chat History Tab").*

### 12.1 Scope

Durable, searchable, user-manageable persistence of chat turns — distinct from
both:
- `conversation_log` — task-scoped working memory used to build prompts
  (§ Schema — three tables in `memory_manager.py`'s module docstring), and
- the existing `chatHistoryStore` (§7.8) — a session-only, in-memory Svelte
  store that resets on every full page reload.

Shipped end-to-end across six sequential steps: schema migration, backend
write path, settings CRUD endpoints, a read/search endpoint, and a new
SvelteKit route (`/history`) with a retention-preset dropdown and a
searchable, paginated turn list.

**Implemented:**
- `memory_manager.py` — `chat_turns`, `chat_turns_fts`, three sync triggers,
  and `chat_history_settings` (v5→v6 schema migration). New public methods:
  `add_chat_turn()`, `get_chat_turns()`, `get_chat_history_eviction_preset()`,
  `set_chat_history_eviction_preset()`.
- `main.py` — `_persist_chat_turn()` write helper wired into `post_task()` and
  `_stream_task()`; `_require_memory_manager()` dependency helper; three new
  endpoints (`GET`/`PUT /chat/history/settings`, `GET /chat/history`).
- `localist-ui` — new route `src/routes/history/+page.svelte`; new stores
  `chatHistorySettings.ts` and `chatHistoryList.ts`; new `Sidebar.svelte` nav
  entry.

**Explicitly not implemented (see §12.7):** eviction sweep execution.

### 12.2 Design Decisions

**`conversation_log` rejected as a reuse target.** Two mismatches, both
structural, not stylistic:
- *Scoping key.* `conversation_log` rows are grouped by `task_id`, and the
  frontend mints a new `task_id` per turn via `crypto.randomUUID()` in
  `ChatPanel.svelte` (`const task_id = crypto.randomUUID();`), not once per
  conversation. A durable, cross-session, searchable transcript needs a
  table that is not keyed to a value that changes every message.
- *Eviction semantics.* `conversation_log` evicts FIFO-by-count via
  `_evict_conversation_log()` (`_CONV_LOG_CAP_PER_TASK = 200` rows per
  `task_id`). Chat History's eviction requirement is age-based (7d/30d/90d/
  forever), which has no natural expression in a count-capped-per-task_id
  table.

**Eviction preset is user-set only — no default.** `chat_history_settings`
starts with **zero rows**; `get_chat_history_eviction_preset()` returns
`None` until the user explicitly picks a preset via the UI dropdown or
`PUT /chat/history/settings`. This is a deliberate absence, not an
oversight — the dropdown renders an explicit disabled placeholder option
("Choose a retention policy…") rather than silently defaulting to any of the
four real presets. **Eviction sweep execution is not yet implemented** —
this arc shipped the settings storage and its read/write API only. Any
future sweep implementation must treat "no row in `chat_history_settings`"
as "do nothing," not as an error or an implicit default preset.

**Write path lives in `main.py`, not `controller_agent.py`.**
`_persist_chat_turn()` sits at the request/response boundary in `main.py` —
called from `post_task()` and both of `_stream_task()`'s two
mutually-exclusive completion paths (the `answer_ready` event and the
fallback path) — deliberately decoupled from `controller_agent.py`'s
agent-internal `conversation_log` writes. `_persist_chat_turn()` no-ops
silently when `_state.memory_manager` is `None` and otherwise wraps the call
in `try/except`, logging a warning on failure without raising — a
chat_turns write failure must never break the actual task response, since
the source of truth for an in-flight answer is the SSE stream / `TaskResponse`,
not this table.

**FTS5 full-text search, not substring matching.** `get_chat_turns()` queries
`chat_turns_fts` via `MATCH` and orders by `bm25(chat_turns_fts) ASC`.
Confirmed empirically (not assumed) that ascending is the correct direction:
SQLite FTS5's `bm25()` returns more-negative values for better matches, so a
row that repeats the search term multiple times outranks a row mentioning it
once under `ASC` ordering — verified both in `test_fts_search_ranks_best_match_first`
and live against real data (§12.6). The raw query string is wrapped as a
single quoted FTS5 phrase (embedded `"` doubled) before being passed to
`MATCH`, so punctuation-heavy input (`"what's"`, hyphens, FTS5 operator
keywords) is treated as literal text instead of `MATCH` syntax; a residual
`sqlite3.OperationalError` from the FTS5 query is caught and degrades to an
empty result set rather than raising.

### 12.3 Schema

Current `_SCHEMA_VERSION` is **7** (v6→v7 migration added to
`memory_manager.py`; v5→v6 was the original Chat History Tab migration
above; v4→v5 was the working-state `turn_summaries_json`-removal migration
from §9).

```sql
CREATE TABLE IF NOT EXISTS chat_turns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT    NOT NULL,
    role                TEXT    NOT NULL,
    content             TEXT    NOT NULL,
    sources_json        TEXT    NOT NULL DEFAULT '[]',
    status_message      TEXT,
    metadata_json       TEXT    NOT NULL DEFAULT '{}',
    conversation_id     TEXT    NOT NULL DEFAULT 'legacy',
    conversation_title  TEXT,
    created_at          REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_turns_created
    ON chat_turns(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_turns_task
    ON chat_turns(task_id);
CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation
    ON chat_turns(conversation_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS chat_turns_fts USING fts5(
    content,
    content='chat_turns',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chat_turns_ai AFTER INSERT ON chat_turns BEGIN
    INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chat_turns_ad AFTER DELETE ON chat_turns BEGIN
    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chat_turns_au AFTER UPDATE ON chat_turns BEGIN
    INSERT INTO chat_turns_fts(chat_turns_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chat_turns_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS chat_history_settings (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    eviction_preset TEXT
);
```

`chat_turns_fts` is an external-content FTS5 table (`content='chat_turns'`,
`content_rowid='id'`) — it stores no independent copy of `content`, only the
inverted index, and is kept in sync purely by the three triggers above:
`chat_turns_ai` (`AFTER INSERT`) indexes the new row; `chat_turns_ad`
(`AFTER DELETE`) removes the old row's index entries via FTS5's
`('delete', rowid, content)` special-insert form; `chat_turns_au`
(`AFTER UPDATE`) does both — deletes the old index entry, then indexes the
new content — since FTS5 external-content tables have no native `UPDATE`
support.

`chat_history_settings` is a single-row table (`id INTEGER PRIMARY KEY CHECK
(id = 1)`) with **no seed row inserted anywhere** — neither in the fresh-DB
`_init_db()` script nor in the v5→v6 `_migrate()` block. Absence of a row
means "no policy set," by design (§12.2).

**v6→v7 migration — conversation grouping.** Adds two columns to
`chat_turns` — `conversation_id TEXT NOT NULL DEFAULT 'legacy'` and
`conversation_title TEXT` — plus `idx_chat_turns_conversation`
(`ON chat_turns(conversation_id, created_at)`), the composite index that
`get_chat_turns()`'s `conversation_id`-filtered queries and
`get_conversations()`'s `GROUP BY conversation_id` rely on. Both
`ALTER TABLE ... ADD COLUMN` statements are gated on a
`PRAGMA table_info(chat_turns)` column-existence check (idempotent —
skipped if the column is already present), matching the v4→v5 migration's
existing pattern. **No separate backfill `UPDATE` is needed:** every
pre-v7 row lacks a stored `conversation_id`, and SQLite applies an
`ADD COLUMN ... DEFAULT 'legacy'` column default to all existing rows as
part of the same `ALTER TABLE` statement (this only works because the
default is a constant, not an expression) — so every row written before
this migration reads back with `conversation_id = 'legacy'` with no extra
migration step.

**`_init_db()` control-flow fix, found during live verification of the
v6→v7 migration.** Previously, `_init_db()` ran the full fresh-install
`executescript()` unconditionally on every startup and only afterward
checked `schema_version` to decide whether to also call `_migrate()`. This
was harmless as long as every statement in the fresh-install script was
independently idempotent against an existing database — but it broke the
first time a schema change added an index on a column that only
`_migrate()`, not the fresh-install script's prior revision, would have
added on an existing (non-fresh) database. Concretely: on a database still
at `schema_version = 6`, the fresh-install script's
`CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation ON
chat_turns(conversation_id, created_at)` ran before `_migrate()` ever got a
chance to `ALTER TABLE chat_turns ADD COLUMN conversation_id`, so the index
referenced a column that did not yet exist and startup failed. The fix
restructures `_init_db()` so `schema_version` is created first (its own
idempotent `CREATE TABLE IF NOT EXISTS`) and read *before* either the
fresh-install script or `_migrate()` runs — the two are now mutually
exclusive branches: `row is None` → genuinely fresh database, run the full
DDL once; `row["version"] < _SCHEMA_VERSION` → existing database, run only
`_migrate()`'s incremental blocks for the versions it's behind. This is a
structural fix to `_init_db()`'s control flow, not specific to
`chat_turns` — it affects every future migration that adds an index (or
any DDL) referencing a column introduced by that same migration, not just
this one.

**New/changed `MemoryManager` methods.** `add_chat_turn()`'s signature now
requires `conversation_id: str` (previously absent) and accepts an
optional `conversation_title: str | None = None`; both are written
straight through to the new columns on `INSERT`. `get_chat_turns()` gained
an optional `conversation_id: str | None = None` filter — when set, both
the unfiltered and FTS-filtered query paths add a
`WHERE conversation_id = ?` / `AND c.conversation_id = ?` clause, and the
accompanying `COUNT(*)` total is filtered the same way, preserving the
§12.4 true-total-count guarantee on a per-conversation basis; when omitted,
behavior is unchanged from v6 (searches/lists across all conversations).
New `get_conversations()` returns one summary row per distinct
`conversation_id` — `conversation_id`, `conversation_title`,
`last_created_at` (`MAX(created_at)`), `first_created_at`
(`MIN(created_at)`) — from a single `GROUP BY conversation_id` query,
ordered by `last_created_at DESC`. Since `conversation_title` is written on
only the first turn of a conversation (every later turn passes
`conversation_title=None`), the query relies on SQLite's bare-column
aggregate behavior to surface that one title-bearing row's value per
group rather than an arbitrary row's — empirically confirmed (not just
assumed) that with two `MIN`/`MAX` aggregates in the same `SELECT`, the
bare `conversation_title` column tracks whichever of the two is listed
*last* (`MIN(created_at) AS first_created_at`, i.e. the earliest row —
exactly the row the title is written on). This is a real SQLite quirk, not
a literal correlated subquery; it is also order-fragile — swapping the
`MAX`/`MIN` lines in the `SELECT` would silently make `conversation_title`
track the *latest* row instead, which is always `NULL`, breaking every
conversation's displayed title without a query error.

### 12.4 Backend API

Four endpoints in `main.py` (the original three from the v6 arc plus
`GET /chat/history/conversations`, added with the v6→v7 conversation-
grouping migration), all guarded by a new `_require_memory_manager()`
dependency helper (503 if `_state.memory_manager is None`, matching
`_require_controller()`/`_require_runtime()`'s exact style):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/chat/history/settings` | Returns `ChatHistorySettingsResponse{eviction_preset: str \| None}`. `None` when the user has never set one. |
| `PUT` | `/chat/history/settings` | Body `ChatHistorySettingsRequest{eviction_preset: Literal["7d","30d","90d","forever"]}` — the `Literal` type rejects invalid values at the Pydantic/FastAPI validation layer (422) before the handler runs. Calls `set_chat_history_eviction_preset()`, then re-reads via `get_chat_history_eviction_preset()` and returns the re-read value (confirms the write landed rather than echoing the request back). |
| `GET` | `/chat/history` | Query params `q: str \| None = None`, `limit: int = 50` (clamped to 200 at the endpoint, matching `list_episodes()`'s ceiling, and clamped again inside `get_chat_turns()`), `offset: int = 0`, and a new optional `conversation_id: str \| None = None` — when provided, restricts results to one conversation; when omitted, searches/lists across all conversations (unchanged v6 behavior). Returns `ChatHistoryResponse{turns: ChatTurnItem[], total: int, offset: int, limit: int}`. |
| `GET` | `/chat/history/conversations` | No query params. Returns `ConversationListResponse{conversations: ConversationSummary[]}`, one entry per distinct `conversation_id` ordered by `last_created_at` descending — powers the sidebar's conversation sub-list (§12.5). Read-only; guarded by the same `_require_memory_manager()` convention as the other two endpoints. |

`ChatTurnItem` fields: `id: int`, `task_id: str`, `role: str`, `content: str`,
`sources: list[dict] = []`, `status_message: str | None = None`,
`metadata: dict = {}`, `conversation_id: str`,
`conversation_title: str | None = None`, `created_at: float`.

`ConversationSummary` fields (new): `conversation_id: str`,
`conversation_title: str | None = None`, `last_created_at: float`,
`first_created_at: float` — a direct mirror of `get_conversations()`'s
return dict (§12.3).

**`total` is a true count, deliberately not `GET /memory/episodes`'s
`total=len(rows)` quirk.** `get_chat_turns()` returns `(rows, total_count)`
where `total_count` comes from a separate `SELECT COUNT(*)` (unfiltered) or
`SELECT COUNT(*) FROM chat_turns_fts WHERE chat_turns_fts MATCH ?` (filtered)
— it reflects the full matching set, not just the current page. This was a
deliberate choice, not an oversight: `GET /memory/episodes`'s existing
`total = len(episodes)` (the page size, not the true total) was noted during
implementation and not copied.

### 12.5 Frontend

*Merged with the Chat tab 2026-07-02 — this subsection previously described
a standalone `History` nav entry (per-turn card view, search box, retention
dropdown). That implementation still exists on disk (see the last bullet
below) but is no longer reachable from navigation; what follows describes
the current merged architecture.*

**Sidebar (`Sidebar.svelte`).** The previously separate `Conversation` and
`History` nav entries are now one `Chat` entry (`{ href: '/conversation',
label: 'Chat', icon: 'chat' }`). Whenever the active route starts with
`/conversation`, the sidebar additionally renders a sub-list of
conversations beneath the main nav, sourced from
`GET /chat/history/conversations` and refetched on every navigation to a
`/conversation*` route (a reactive `$:` block keyed off the current path,
so it also re-fires on `/conversation/[id]` → `/conversation/[id]`
transitions). A **"+ New chat"** button is pinned above the conversation
sub-list; it calls `startNewConversation()` (below) and navigates to the
freshly minted conversation's route. Each sub-list entry links to
`/conversation/<conversation_id>` and is labelled with
`conversation_title` when set, or a formatted "New conversation — <date>"
fallback (derived from `last_created_at`) when not.

**New dynamic route `src/routes/conversation/[id]/+page.svelte`.** Renders
`ChatPanel` (unchanged) and owns loading the selected conversation's
history. A reactive block syncs `currentConversationId` (new store, below)
to `$page.params.id`, clears `chatHistoryStore` immediately (so switching
conversations doesn't flash the previous conversation's turns while the
new ones load), and re-fetches from `GET /chat/history?conversation_id=...
&limit=200` on every `id` change. The response is reversed (backend orders
`created_at DESC`; the feed renders oldest-first) and each row's
`created_at` (seconds) is multiplied by 1000 to match `Turn.timestamp`'s
millisecond convention (§7.8's `chatHistory.ts`). A failed fetch degrades
to an empty store rather than throwing (logged via `console.warn`), matching
the fail-soft convention established elsewhere in this arc — e.g. §12.2's
`_persist_chat_turn()`.

**Bare `/conversation` route.** `src/routes/conversation/+page.svelte`
redirects client-side to `/conversation/<currentConversationId>` via
`onMount()` + `goto(..., { replaceState: true })` — explicitly **not** a
`+page.ts` `load()`/`redirect()`, because `currentConversationId` reads
`localStorage`, which is unavailable during SSR; a universal `load()`
would see a fresh, non-persisted id on the server and redirect to the
wrong conversation.

**New store `src/lib/stores/conversation.ts`.** Owns three exports,
distinct from `tasks.ts`'s `SESSION_ID` (see below):
- `currentConversationId: Writable<string>` — backed by `localStorage`
  under key `localist:conversationId`; on first-ever load it mints a
  `crypto.randomUUID()` and persists it, otherwise loads the stored value.
  A `subscribe()` write-through keeps `localStorage` in sync with every
  later update (including from `startNewConversation()`).
- `startNewConversation(): string` — mints and persists a fresh id, sets
  `currentConversationId`, and resets `isFirstTurnOfConversation` to
  `true`. Called by the sidebar's "+ New chat" button.
- `isFirstTurnOfConversation: Writable<boolean>` — starts `true`, flipped
  to `false` by `ChatPanel.svelte` the instant a turn is submitted (before
  the request goes out, guarding against a rapid double-submit sending a
  title twice), and reset to `true` by `startNewConversation()`. Controls
  whether the next submitted turn sends a `conversation_title` — the
  backend contract is that `conversation_title` is sent on exactly the
  first turn of a `conversation_id` and never after. `ChatPanel.svelte`
  derives the title itself: the submitted message text, truncated to 60
  characters with a trailing `…` if longer.

**`tasks.ts`'s `submitTask()`**, extended with two new trailing optional
params, `conversation_id?: string` and `conversation_title?: string`,
both passed straight through in the `POST /task/stream` request body.
This is separate from, and does not touch, the pre-existing
`SESSION_ID` constant (`tasks.ts:53`) — `SESSION_ID` is a page-load-scoped
id used for backend `conversation_log` working-memory grouping (§12.2);
`conversation_id` is the durable, user-facing chat-thread identifier
persisted across reloads. The two must not be conflated: one resets on
every full page reload by design, the other explicitly does not.

**`src/routes/history/+page.svelte` — retired 2026-07-21, see §20.** Was
unlinked from nav since the 2026-07-02 merge above; deleted outright once
the Episode Browsing UI's `/episodes` route (§20) made it a strict subset
(same FTS search + list, superseded by that route's semantic search,
filters, detail pane, and tool-result rendering). `chatHistorySettings.ts`
is unaffected — its retention-preset control already lived on `/settings`
independently and still does; only the duplicate copy on `/history` and
the dedicated `chatHistoryList.ts` list store were removed.

### 12.6 Live Verification

With a live backend (`main:app`, port 8001) and a live LLM runtime (oMLX,
port 8000) already running, three real requests were sent through the
actual production write path — two via `POST /task` ("what is a zebra?",
"fun fact about pangolins") and one via `POST /task/stream` ("what color is
the sky?") — writing 6 real rows into `chat_turns`.

- **Direct SQLite inspection** of all 6 rows: `role` values exactly `"user"`/
  `"assistant"` (no stray values); `content` non-empty and matching the real
  conversation; `sources_json`/`metadata_json` valid JSON round-tripping
  correctly (`[]`/`{}` for user turns, the expected
  `{agent, priority, fetch_rag, fetch_episodic, tools_fired, grounded}` dict
  for assistant turns); `status_message` consistently `NULL` (expected — no
  call site populates it yet); `created_at` timestamps strictly increasing.
- **`GET /chat/history`** (direct and through the Vite dev proxy at
  `/api/chat/history`) returned all 6 rows, newest first, `total: 6`.
- **FTS search correctness in both directions:** `?q=zebra` returned exactly
  the 2 rows containing "zebra" (the user question *and* the assistant
  answer); `?q=sky` returned exactly the 2 "sky" rows; `?q=nonexistentxyz`
  returned `{turns: [], total: 0}`.
- **Pagination correctness:** `?limit=2&offset=2` returned rows 3–4 of the
  unfiltered set while `total` remained `6` (the full count, not the page
  size) — confirming the §12.4 total-count design decision holds under real
  data, not just unit tests.
- The rendered `/history` page (fetched through the Vite dev proxy) showed
  the expected markup (`turn-card` elements, search input, retention
  dropdown) against this real data.

Test rows were deleted after verification, restoring `chat_turns` to empty.

This live-fire pass also closed an outstanding gap from step 2: the
`_stream_task()` write path had shipped on test-suite-pass alone, with no
direct live verification at the time (no SSE test harness exists in this
repo) — the `POST /task/stream` request in this pass was the first live
confirmation that its `answer_ready` persistence branch actually fires
correctly end-to-end.

### 12.7 Open Items

**Open Item 1 — Eviction sweep execution not yet implemented.**
`chat_history_settings` can be read and written via the two settings
endpoints, but nothing currently acts on the stored preset. No scheduled
job, no request-time sweep, no manual trigger exists. A future
implementation must treat an absent row as "no policy set — do nothing,"
per §12.2.

**Open Item 2 — Leading-`\n` in assistant answers (cosmetic, out-of-scope).**
Assistant `content` values sometimes begin with a literal `\n` (e.g.
`"\nA zebra is a genus of equid..."`, observed live in §12.6). This is
inherited verbatim from `result.get("answer")` — a synthesizer/prompt-output
quirk upstream of persistence, not a `chat_turns` or `add_chat_turn()` bug.
`chat_turns` is correctly persisting exactly what the pipeline produced.

**Open Item 3 — No automated frontend test coverage.** No test framework
(`vitest`, `jest`, or otherwise) exists anywhere in this repo — confirmed
twice independently during this arc (once during the settings-store step,
once during the searchable-list step) via `package.json` and a repo-wide
search for `*.test.ts`. Not introduced as part of this feature; frontend
verification for this arc relied on `svelte-check`, `vite build`, and the
live-fire pass in §12.6.

**Open Item 4 — CLOSED 2026-07-21, see §20.** `src/routes/history/
+page.svelte` and its dedicated `chatHistoryList.ts` store were deleted
outright rather than folded in — the new Episode Browsing UI's
`/episodes` route (§20) is a strict superset of `/history`'s turn-list +
FTS-search functionality (plus semantic search, filters, a detail pane,
and tool-result rendering), and `/history`'s retention-preset dropdown
duplicated a control that already lives on `/settings`
(`chatHistorySettings.ts`, unaffected, still used there). No functionality
was lost.

**Open Item 5 — Mid-stream navigate-away-and-back does not reconstruct
in-progress streaming state.** If the user leaves a `/conversation/[id]`
route while an answer is still streaming and returns before it completes,
`loadConversationHistory()` (§12.5) only has `GET /chat/history` to work
from, which reflects committed rows — the in-flight assistant turn's
partial tokens, `status`, and `status_message` are not reconstructed. The
completed turn simply appears, in full, the next time the fetch runs (on
return to that route, or on a subsequent navigation), rather than the feed
picking the stream back up mid-flight.

Startup/lifecycle tooling for running the full stack (backend, fetcher, and
this UI) together is documented separately in §13 — Localist CLI.

### 12.8 Test Suite

Backend test suite: **447 → 489 (+42)** across the full six-step arc, 0 real
failures. The full-suite run at the end of this arc reports 9 failures
(`test_tool_dispatcher_phase6.py::TestWebSearch` and one
`test_controller_phase4.py` test), all attributable to a pre-existing,
unrelated network flake — live HTTP calls to `api.langsearch.com`
intermittently returning 500s when run alongside the rest of the suite.
Both files pass 73/73 when run in isolation; this flake predates this
feature and recurred identically (with a varying failure count, 7–9) across
every step of this arc.

No frontend test coverage was added (§12.7, Open Item 3).

