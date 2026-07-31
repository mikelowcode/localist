## 20. Episode Browsing UI

### 20.1 Overview

A three-pane (Filters / Episode list / Episode detail) view at `/episodes` for browsing
`chat_turns` as a semantic event stream rather than a linear chat log — the pitch, scoped
2026-07-21, described `chat_turns` as the browsing spine (Path B) with `episodes` as a read-only
annotation overlay, rather than extending `VALID_EPISODE_TYPES` (Path A) to cover tool-result and
multi-turn-workflow event kinds. Path B was chosen because it reuses `chat_turns`' existing FTS5
infrastructure and `metadata_json` persistence path unmodified, and leaves `episodes`' existing
contract (implicit extraction, `format_episodic_summary()`'s 5-bullet prompt-injection cap) untouched.

Built in seven phases, three backend + four frontend:

1. `chat_turns` semantic search (embedding column, provenance tracking, `mode="semantic"`)
2. Research-loop `workflow_id` correlation key
3. Multi-diff turns — verified already correct end to end, not a code change
4. Frontend route, list, and filter pane
5. Detail pane with type-specific renderers (chart, diff, workflow step-chain)
6. Episodes overlay (read-only "related memory" per turn)
7. `/history` route retirement

### 20.2 Backend — `chat_turns` Semantic Search (Phase 1)

Schema v9 adds an `embedding BLOB` column to `chat_turns` (`memory_manager.py`). `add_chat_turn()`
embeds `content` (truncated to 500 chars — same convention as `index_document()`/`reembed_corpus()`)
whenever `embed_fn` is configured; embed failures degrade to a `NULL` embedding for that row rather
than blocking the write, so the row stays findable via keyword/FTS.

`embedding_provenance` (§16.4) gains a third store, `'chat_turns'`, alongside `'corpus'` and
`'episodes'`. It follows `'corpus'`'s never-automatic-reembed path (own `self._chat_turns_stale`
flag, cleared by the new `MemoryManager.reembed_chat_turns()` / `POST /memory/reembed-chat-turns`)
rather than `'episodes'`'s auto-reembed-in-place path — `chat_turns` can grow arbitrarily large
under the `"forever"` eviction preset, the same reasoning that keeps `'corpus'` manual.

`MemoryManager.get_chat_turns()` gains `mode: "keyword" | "semantic"` and `min_score: float = 0.3`.
`mode="semantic"` does a full-table cosine scan via `_get_chat_turns_semantic()` — `chat_turns` has
no `token_set`-style column to cheaply pre-filter with the way `document_index` does, so every row
with a stored embedding is scored directly (`_CHAT_TURNS_SEMANTIC_SCAN_WARN_ROW_COUNT = 2000` logs a
warning past that row count, does not bound anything). Silently falls back to keyword/FTS when
`query` is empty, `embed_fn` is unavailable, or `self._chat_turns_stale` is `True` — same fail-safe
posture as `query_corpus()`/`EpisodicMemoryReader.by_similarity()`.

Also gains `date_from`/`date_to` (inclusive `created_at` bounds) and `has_tool_result` (a LIKE-based
substring check on `metadata_json` for `"chart"`/`"pending_diffs"`/`"workflow_id"` — the only three
keys `controller_agent.py` ever writes there) — the Episode Browsing UI's filter-pane dimensions.
Both compose with keyword and semantic mode. `GET /chat/history` exposes all of this
(`mode`, `min_score`, `date_from`, `date_to`, `has_tool_result`); `ChatTurnItem` gains an optional
`score` field, populated only in semantic-mode results.

**Correction to the original scoping session's read of the codebase**, worth recording since it
changed what Phase 1 actually needed to build: `EpisodicMemoryReader.by_similarity()`/`best_match()`
already did real cosine-similarity search when `embed_fn` was present (confirmed live at
`controller_agent.py`'s episodic-retrieval step and `episodic_extractor.py`'s retraction
best-match) — the scoping session's claim that this needed wiring was stale/wrong. The actual gap
was `chat_turns` having zero embedding infrastructure at all, not `episodes`.

Test coverage: `test_chat_turns_semantic_search.py` (migration, embed-on-write, provenance
seed/mismatch/reembed, semantic scoring/ranking/pagination/fallback, date-range and
has-tool-result filters in both modes), `test_main_chat_turns_semantic_endpoint.py` (the two new/
extended endpoints), extensions to `test_chat_turns_schema.py`'s `TestGetChatTurns`.

### 20.3 Backend — Research-Loop `workflow_id` (Phase 2)

`ToolResult` (`prompt_builder.py`) gains `workflow_id: str | None = None`.
`MCPToolDispatcher._run_research_loop()` (§18.4) generates one `uuid.uuid4()` per call and stamps
it on every `ToolResult` it produces or appends, including the synthetic `tool_name="research"`
exhaustion result (§18.4/§18.9) — no other tool ever sets this field. `controller_agent.py`'s
`_execute_plan` pulls it back out right after Step 3's `chart_artifact` extraction (identical
pattern): the first non-`None` `workflow_id` found becomes `metadata["workflow_id"]`, and every
`ToolResult` sharing it becomes an ordered `metadata["workflow_steps"]` list (`tool_name`,
`parameters`, `success`, `result` truncated to 500 chars). Both keys are omitted from `metadata`
entirely on non-research turns. Full detail: §18.10.

### 20.4 Backend — Multi-Diff Turns Verified, Not Fixed (Phase 3)

Investigated as a scoping question — "does a turn proposing 2+ wiki diffs actually work end to
end?" — and found every layer (`WikiAgent` → `_build_wiki_diff_result()` → `mark_diff_applied()` →
`POST /wiki/apply-diff` → `ChatPanel.svelte`'s per-`page_name`-keyed rendering) already operated on
the full `pending_diffs` list independently, with no single-diff assumption anywhere. Closed
docs/architecture/17-wiki-agent-diff-target.md §17.8's long-open "Multi-diff turns" item as
verified rather than fixed — 3 new end-to-end tests, no production code changed. Full detail: §17.11.

### 20.5 Backend — Episodes Overlay Support (Phase 6)

`MemoryManager.list_episodes()`/`count_episodes()` gain a `task_id` filter (same pattern as the
existing `project_context`/`episode_type` filters), exposed via `GET /memory/episodes?task_id=...`.
Originally backed the detail pane's "related memory" section via an exact `task_id` match — episodes
implicit extraction stamped with the same `task_id` as the selected `chat_turns` row
(`process_implicit_extraction(..., task_id=task.task_id, ...)` in `controller_agent.py` already did
this stamping — no change needed there, only a way to query by it). **Superseded 2026-07-24
(§2.13 of the Episodic Memory Schema doc):** the exact-match query almost always returned empty
(episodes are sparse by design, §2.1 there), so "related memory" is now a real Mode 3 semantic-
similarity lookup — `GET /memory/episodes/related?content=...&task_id=...` — scored against the
selected turn's own content, with `task_id` now used only to *exclude* that turn's own episode(s)
from its own results rather than to select them. `list_episodes()`/`count_episodes()`'s `task_id`
filter itself is unchanged and still backs the plain `GET /memory/episodes?task_id=...` filter used
elsewhere (e.g. any future exact-task_id lookup) — only the overlay's own query changed.

### 20.6 Frontend

**Route** — `src/routes/episodes/+page.svelte`, a CSS-grid three-pane layout (`220px 340px 1fr`).
New Sidebar nav entry (`Episodes`, between Chat and Files) and `StatusBar.svelte` title-map entry.

**Store** — `src/lib/stores/episodeBrowser.ts`. Wraps the extended `GET /chat/history` (filters:
query/mode/conversationId/dateFrom/dateTo/hasToolResult) and `GET /chat/history/conversations`.
Kept separate from the (now-deleted) `chatHistoryList.ts` — different shape (carries `score`, the
extra filter dimensions), different route.

**Components:**
- `EpisodeFilterPane.svelte` — search box with a keyword/semantic mode toggle (debounced 300ms,
  same convention as the old `/history` search), conversation dropdown, date-range inputs,
  has-tool-result checkbox. Every filter change resets pagination to page 1 and reloads.
- `EpisodeList.svelte` — paginated turn list, role badge, truncated content, tool-result badges
  (Chart/Diff/Workflow, derived from `metadata`), semantic match-score badge when present.
- `EpisodeDetailPane.svelte` — full turn content plus type-specific renderers (below), sources,
  metadata, and the episodes overlay.
- `DiffBlock.svelte` — **extracted from `ChatPanel.svelte`** (which previously inlined diff
  rendering/apply/discard directly). Both `ChatPanel.svelte` and `EpisodeDetailPane.svelte` now
  render diffs through this one component; each caller owns its own post-apply store sync via an
  `onApplied(pageName)` callback (`ChatPanel` syncs `chatHistoryStore` + `tasksStore`,
  `EpisodeDetailPane` syncs `episodeTurns`). This extraction is what made Phase 5's multi-diff
  detail-pane rendering free — §20.4 already confirmed the underlying logic handles 2+ diffs
  correctly, so reusing it (rather than writing a second diff renderer) carries that correctness
  over with no new risk.
- `WorkflowSteps.svelte` — read-only step-chain view for `metadata.workflow_steps` (§20.3); a
  connector-dot-and-line layout, one entry per tool call, failed steps flagged.
- `EpisodeAnnotations.svelte` — the episodes overlay (§20.5): fetches
  `GET /memory/episodes/related?content=...&task_id=...` on mount for the selected turn (`content`
  is the turn's own text, `task_id` excludes that turn's own episode(s) from its own results —
  §2.13 of the Episodic Memory Schema doc), wrapped in a `{#key selected.task_id}` block in the
  caller so a turn-selection change remounts it, renders each episode as a small type-chip +
  content card. Read-only by design — approve/reject stays `EpisodesPanel.svelte`'s job on the
  existing `/memory` route; every episode surfaced here is already `status=active`.
- `ChartRenderer.svelte` — reused as-is from `ChatPanel.svelte`, no changes.

### 20.7 `/history` Retirement (Phase 7)

`src/routes/history/+page.svelte` and `src/lib/stores/chatHistoryList.ts` deleted outright.
Rationale: the route had zero nav-link reachability already (unlinked since the 2026-07-02 Chat +
History merge, §12.5/§12.7 Open Item 4), its retention-preset section duplicated a control that
already lived independently on `/settings` (`chatHistorySettings.ts` at the time, later renamed —
see §20.10), and its turn-list + FTS-search section is a strict subset of `/episodes`. No
functionality was lost; see §12.7's closed Open Item 4 for the full before/after.

### 20.8 Test Coverage

Backend: 987 → 1042 passed (+55), 0 failed, across all three backend phases — see §20.2's
`test_chat_turns_semantic_search.py`/`test_main_chat_turns_semantic_endpoint.py`, §20.3's
`workflow_id` coverage in `test_mcp_tool_dispatcher.py::TestResearchLoop` and
`test_controller_phase4.py::TestWorkflowStepsMetadata`, §20.4's multi-diff verification tests in
`test_controller_phase4.py`/`test_chat_turns_schema.py`/`test_wiki_apply_diff_endpoint.py`, and
§20.5's `task_id`-filter tests in `test_memory_phase1.py`/`test_main_memory_episodes.py`.

Frontend: no automated test coverage — no test framework exists in this repo (§12.7 Open Item 3,
still open, not introduced by this feature). Verified via `svelte-check` (0 errors) and
`vite build` (succeeds, `/episodes` route present in the build output) after every phase.

### 20.9 Open Items

- No live-browser verification was performed as part of this build — `svelte-check`/`vite build`
  confirm the code compiles and type-checks, not that the three-pane layout renders correctly or
  that live filter/search/apply interactions behave as intended against a running backend.
- `EpisodeAnnotations.svelte`'s per-turn fetch is not cached — reselecting a previously-viewed turn
  re-fetches `GET /memory/episodes` rather than reusing the prior result. Not expected to matter at
  this app's scale (single user, local backend) but worth revisiting if it does.
- No pinning, export, replay, or episode-based RAG — explicitly out of scope for this build, per
  the original scoping session, as downstream features that depend on the spine decision (Path B)
  landing first.
- `EpisodeList.svelte`'s tool-result badges (Chart/Diff/Workflow) are derived client-side from
  `metadata` shape on every render; `has_tool_result`'s backend filter uses the same three keys via
  a LIKE substring check rather than a real JSON predicate (sqlite's json1 extension is not assumed
  to be compiled in) — both are correct today because exactly three keys exist, but either would
  need revisiting if a fourth tool-result metadata key is ever added under a name that could
  collide with the substring check.

### 20.10 Global Retention TTL (2026-07-31)

The user had Chat History's 7-day retention preset set in Settings and asked for the same TTL
control over episodes (808 saved at the time). Investigation found the existing preset was stored
but **never enforced** — no sweep existed anywhere in the codebase (§12.7's now-closed Open Item 1)
— so this arc both extended the setting to episodes and built real enforcement for the first time.

**One shared global preset, not two.** Per explicit user choice, a single retention preset governs
both `chat_turns` and episodes rather than two independent settings — "Data Retention" is one
segmented control on `/settings` (7d/30d/90d/Forever), same as before.

**Renamed, not duplicated.** `chat_history_settings` was renamed to `retention_settings` outright
(`memory_manager.py` migration v12→v13 — `ALTER TABLE ... RENAME TO`, with a self-heal fallback
mirroring the existing drift-recovery pattern for `chat_turns.embedding`/`news_preferences`/etc.),
rather than adding a second table — this repo's convention is no backwards-compat shims. Renamed
throughout: `get_chat_history_eviction_preset()`/`set_chat_history_eviction_preset()` →
`get_retention_preset()`/`set_retention_preset()`; `GET`/`PUT /chat/history/settings` →
`GET`/`PUT /settings/retention`; `ChatHistorySettingsResponse`/`Request` →
`RetentionSettingsResponse`/`Request`; `chatHistorySettings.ts` → `retentionSettings.ts`. Deliberately
**not** named "memory retention" — the app already has a separate `/memory` nav tab (episode
browsing), and that name would have read as if it configured that tab instead.

**Episodes soft-retract, chat_turns hard-delete.** `episodes` has no lifecycle column comparable to
`chat_turns` — it already has one (`status`, values `active`/`pending`/`superseded`/`retracted`),
used by the existing approve/reject/retract flow (§20's `EpisodeAnnotations.svelte` /
`EpisodesPanel.svelte`). The sweep reuses that convention (`UPDATE episodes SET status='retracted'
WHERE status='active' AND created_at < cutoff`) rather than hard-deleting — reversible, and
automatically excluded from every existing `status='active'` retrieval path (`by_subject`,
`by_recency`, `_score_all_active`, `get_by_ids`, `list_episodes()`/`count_episodes()`'s default
filter — no query-site changes needed anywhere). `chat_turns` rows are hard-deleted
(`DELETE FROM chat_turns WHERE created_at < cutoff`), matching the "evicted" language the original
§12 UI copy already used. A new `idx_episodes_created` index (`episodes(created_at)`) was added in
the same migration for the sweep's range scan — cheap at 808 rows today, insurance at scale.

**`MemoryManager.sweep_expired_memory()`** is the new enforcement method: no-ops
(`{"chat_turns_deleted": 0, "episodes_retracted": 0}`) when no preset is set or the preset is
`"forever"` — absence means "keep everything," not "sweep with an infinite TTL," preserving the
original §12.2 semantics. It does **not** regenerate `MEMORY.md` itself — that snapshot is owned by
the separate `EpisodicMemoryWriter` class (`_write_memory_md()`/`regenerate_memory_md()`), which
`MemoryManager` has no reference to; the caller is responsible for triggering it when
`episodes_retracted > 0`, exactly like `POST /memory/episodes/{id}/approve`/`reject` already do in
`main.py`.

**Startup-only sweep, no on-write pruning, no manual trigger.** Wired into `lifespan()`
(`main.py`) immediately after the existing wiki-snapshot TTL sweep (§17) — same non-fatal
`try/except`, same logging convention. `_state.memory_manager` is already non-`None` by that point
in startup. If `episodes_retracted > 0`, `lifespan()` also constructs an `EpisodicMemoryWriter` and
calls `regenerate_memory_md()`, per the note above.

**Test coverage:** new `backend/tests/test_retention_sweep.py` (no-preset/`"forever"` no-op,
mixed-age chat_turns+episodes sweep, already-retracted episodes not double-counted, idempotency).
`test_chat_turns_schema.py` and `test_main_task_chat_turns.py` updated in place for the rename
(table/method/endpoint names); `test_memory_manager_github_watch.py`/`test_memory_manager_hacker_news.py`'s
minimal synthetic pre-v13 DB fixtures (no `episodes` table) surfaced a real bug in the first version
of the v12→v13 migration/self-heal blocks — the new `CREATE INDEX ... ON episodes(created_at)` was
unconditional and failed with `no such table: episodes` against those fixtures; fixed by gating the
index creation on `episodes` actually being present, same defensive style already used for the
table-rename check. Full suite: 1376 passed, 0 failed.
