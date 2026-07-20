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
| `/files` | `FileBrowser.svelte` | Full-width file preview pane. Wiki/Raw/Generated listing, upload, and ingest moved into the sidebar's Files sub-nav (§7.11, 2026-07-13) — no longer part of this route's own component. |
| `/settings` | Settings | Runtime backend status/health, live runtime-backend switch + per-backend chat-model dropdown (§7.10, §16.6), chat-history eviction preset, theme. |

### 7.3 Provenance Bar

Every completed assistant turn renders a **provenance bar** between the
response body and the source chips. It is driven by the `metadata` field
in the SSE `"done"` event.

**Chips rendered:**

| Chip | Condition | Colour |
|---|---|---|
| `P1 · Direct` | `priority === 1` | Muted |
| `P2 · Memory write` | `priority === 2` | Green |
| `P3 · Web search` / `P3 · File operation` / `P3 · Page fetch` / `P3 · Tool` | `priority === 3`, labeled by whichever tool fired (see below) | Blue |
| `P4 · Vault` | `priority === 4` | Purple |
| `P5 · Episodic` | `priority === 5` | Amber |
| `P6 · Inference` | `priority === 6` | Muted |
| `⚙ {tool_name}` | each entry in `tools_fired`; also rendered for a deferred `file_op` (see below) | Orange |
| `◎ episodic` | `fetch_episodic === true` | Amber |
| `◈ grounded` | `grounded === true` | Green |

Source chips (wiki/raw type + human-readable name) are rendered below the
provenance bar from the `sources` array.

**Correction 2026-07-07 — P3 chip was hardcoded to "Web search" regardless
of which tool actually fired.** Priority 3 (§4.2) is a generic tool-signal
priority — `web_search`, `file_op`, or `url_fetch` can each independently
match it — but `ChatPanel.svelte`'s provenance bar rendered a literal
`"P3 · Web search"` string for every Priority-3 turn, including ones where
only `file_op` or `url_fetch` had fired. Fix: a new `p3Provenance()`
helper reads `tools_fired` and picks the label from whichever tool actually
matched (`"P3 · Web search"` / `"P3 · File operation"` / `"P3 · Page
fetch"`), falling back to a generic `"P3 · Tool"` when none or more than
one match (compound turns). A deferred `file_op` (§4.4b) counts as a
`file_op` match for this purpose even though `tools_fired` stays empty for
it — the write happens after generation completes, so `file_op` never
enters `tools_to_call` — by checking `metadata.file_op_deferred` alongside
`tools_fired`. The same deferred case previously showed *no* tool chip at
all (the `⚙ {tool_name}` loop iterates `tools_fired`, which is empty here);
a matching `⚙ file_op` chip is now rendered explicitly whenever
`file_op_deferred` is true and `file_op` isn't already in `tools_fired`.
Verified by code-trace of `ResponseMetadata` (§4.4a) plus inspecting the
live task metadata for the `moon.md`/`ocean.md` deferred-dispatch repros
from §4.4b (`file_op_deferred: true`, `tools_fired: []`, chip renders `⚙
file_op` with no duplicate). A headless-browser screenshot check
(Playwright) was attempted for full pixel-level confirmation but the
install stalled in this environment; this is noted as a limitation on the
verification depth, not skipped as a shortcut — the API-level metadata
trace plus source-read of `p3Provenance()`'s branch logic against every
`tools_fired`/`file_op_deferred` combination is the verification actually
performed.

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

### 7.6 Status Bar & Live Turn Rendering

#### Header status chips

**Superseded 2026-07-13 — see §7.10.** The three-chip layout described in this subsection
(agents/model/connectivity) was replaced by a single consolidated chip (green dot + active
inference-engine name, Chat screen only) plus a separate pending-count chip (Memory screen only)
as part of the 1a desktop-UI port. The agents chip/popover described below no longer exists in the
UI at all — `agents.ts`'s store and polling are still running but unconsumed by any component (see
§7.10's open items). The rest of this subsection is retained as the historical record of the
pre-2026-07-13 SSE/status-event mechanics, which are unchanged by the redesign.

`StatusBar.svelte` (`localist-ui/src/lib/components/StatusBar.svelte`) renders three chip types in the right section of the application header: agents, model, and connectivity. A fourth chip — a "streaming" indicator driven by `$tasksStore.streaming` — existed in earlier versions and was removed 2026-06-28. It duplicated the in-bubble status line already present in `ChatPanel.svelte`; the in-bubble status line is now the canonical live-status indicator for in-flight tasks.

**Agents chip.** Reads `$agents.agents` (a `string[]` of agent names) and `$agents.loaded` (boolean) from `$lib/stores/server`. Hidden until `$agents.loaded === true && $agents.agents.length > 0`. The store exposes agent names only — not per-agent activity state, task assignment, or health. The chip label is the agent count (e.g. `2 agents`). Clicking it toggles a popover anchored below the chip (`position: absolute; top: calc(100% + 6px); right: 0`) listing each name as a `role="menuitem"` row. The button carries `aria-expanded` and `aria-haspopup="true"`.

**Popover close behavior.** Three paths close the popover: re-clicking the chip (`on:click={() => (agentsOpen = !agentsOpen)}`), clicking outside the `.agents-wrap` container, and pressing Escape. Click-away and Escape are handled via `window.addEventListener` calls registered in `onMount`. Cleanup runs in `onDestroy`, guarded by `if (browser)` — where `browser` is imported from `$app/environment`:

```svelte
onDestroy(() => {
  if (browser) {
    window.removeEventListener('click', handleWindowClick);
    window.removeEventListener('keydown', handleWindowKeydown);
  }
});
```

The guard is necessary because `onDestroy` runs during SSR teardown (SvelteKit pre-renders routes on the server), not only during client-side unmount. `window` is undefined in the server environment. An earlier unguarded version of this code caused a live `ReferenceError: window is not defined` crash on `/conversation` page load (2026-06-28); the `browser` guard was the direct fix.

#### SSE status event sequence

The streaming endpoint (`POST /api/task/stream`, `_stream_task()` in `backend/main.py`) yields the following sequence for a normal chat request:

| Order | Event type | `message` / payload | Emitted after |
|---|---|---|---|
| 1 | `status` | `"Planning task…"` | Immediately, before any blocking work |
| 2 | `status` | `"Routed to {agent}"` | `controller.route_task()` returns |
| 3–N | `token` | one chunk per event | Real-time from `on_token` drain loop; ConversationalAgent routes only (see §7.7) |
| N+1 | `status` | `"Updating working memory…"` | Before `process_working_state_update()`; conditional on post-dispatch gate (see §7.7) |
| N+2 | `sources` | sources array | After `handle_task_with_plan()` returns |
| N+3 | `done` | task_id, status, metadata, answer | — |
| N+4 | `task_complete` | task_id | Only after the full task — including `process_implicit_extraction()` and `process_working_state_update()` — has resolved. Emitted on every terminal path, success or error alike (see §7.7 Update 2026-07-05). |
| N+5 | `[DONE]` | (raw sentinel, not JSON) | — |

Event 1 is emitted unconditionally at the top of `_stream_task()` (`main.py:885`). Events 2 onward follow only after the corresponding blocking work completes. `task_complete` always immediately precedes `[DONE]` and is distinct from `done`: `done` can fire as soon as the visible answer is ready (see `on_answer_ready`, below), while `task_complete` fires only once the entire backend pipeline — memory hooks included — has actually finished. Clients must not treat `done` as the signal that it is safe to submit another task; see §7.7 Update 2026-07-05.

**Routing split.** The "Routed to {agent}" event is emitted after `controller.route_task()` (`controller_agent.py:836`) returns. `route_task()` is a thin wrapper around `self._planner.route()`, dispatched via `asyncio.to_thread` because some priority branches in `Planner.route()` call `embed_fn` or `runtime.infer()`, both synchronous. Once `route_task()` returns a `RoutingPlan`, `_stream_task()` yields event 2, then dispatches `controller.handle_task_with_plan()` (`controller_agent.py:847`) in a second `asyncio.to_thread`. `handle_task_with_plan()` calls `_execute_plan()` directly with the precomputed `RoutingPlan`, bypassing `_execute()` and therefore not calling `_planner.route()` again. Routing runs exactly once per streaming request.

**Unchanged surface.** `controller.handle_task()` (`controller_agent.py:786`) retains its original signature. `POST /task` (non-streaming) still calls `handle_task()` via `asyncio.to_thread`, unchanged.

#### Known limitation — word-replay resolved for ConversationalAgent; WikiAgent buffer path retained by design

**RESOLVED for ConversationalAgent (2026-06-28).** `_stream_task()` now uses a real producer/consumer bridge: an `asyncio.Queue[dict[str, str]]` populated from the worker thread via `loop.call_soon_threadsafe`, drained by `await asyncio.wait_for(queue.get(), timeout=0.05)` while the producer task runs. `ConversationalAgent.run()` calls `on_token(chunk)` for each chunk emitted by `infer_stream()`, and those chunks reach the SSE layer in real time — not as a post-hoc buffer replay. The word-split loop and its `asyncio.sleep(0)` separator are removed entirely. Full architecture in §7.7.

**WikiAgent — buffer path retained permanently.** WikiAgent's output contract is structured XML consumed by `parse_model_xml()`; streaming partial XML before the full response is available would surface malformed output to the user. For WikiAgent-routed plans, `on_token` is never called and the drain loop terminates promptly with an empty queue once `producer_task.done()`.

*Historical record of the pre-2026-06-28 word-replay approach (superseded):* `_stream_task()`'s former token loop split the completed answer on whitespace and yielded one `"token"` event per word, separated only by `await asyncio.sleep(0)`. Because both agents called `self._runtime.infer()` synchronously, the full answer was already in memory when the loop began; it completed faster than a single browser paint cycle, making the "Streaming answer…" status frame and the full answer content effectively simultaneous in the UI. The routing-status frames were the only phases where visible progressive rendering occurred. Documented as an accepted cosmetic gap in the original §7.6 entry.

**Correction to in-session pacing/streaming diagnosis (2026-06-28).** During this session — after the `infer_stream()` wiring described in the RESOLVED block above was already applied — specific test cases appeared to show "no visible word-by-word streaming" or "all arrived at once." An earlier draft of this subsection tentatively attributed this to residual blocking-`infer()` usage (the same causal mechanism as the pre-fix era, based on a stale in-session read of `conversational_agent.py`). **This diagnosis was wrong.** Confirmed against current on-disk `conversational_agent.py`: both the prebuilt-prompt branch (lines 219–229) and the main RAG branch (lines 364–375) call `self._runtime.infer_stream()` when `on_token` is not `None`. The wiring was complete at the time the incorrect diagnosis was made — the apparent "missing streaming" was not caused by `infer()` usage or missing pacing/`sleep(0)`. The pacing-and-sleep explanation was removed rather than softened; it was simply not the actual mechanism. The actual cause was a separate bug: the fabrication-correction propagation gap, documented in the next subsection. Garbled fabricated-tool-call text was being streamed live to the client with no correction ever arriving, which made genuine streaming look like "nothing is happening" or "all arrived at once" in the specific test cases triggered during this session.

#### Fabrication-correction propagation gap (fixed 2026-06-28)

A companion bug to the open-item fabrication detection (§8.8 Open Item 11): even when `_is_fabricated_toolcall()` correctly detected fabricated tool-call syntax and substituted `_SEARCH_UNAVAILABLE_FALLBACK`, the corrected answer never reached the client's chat bubble.

**Detection sequence (unchanged, pre-existing).** In `ConversationalAgent.run()`, both the prebuilt-prompt branch and the main RAG branch run the `on_token` streaming loop first, yielding each chunk to the SSE queue as inference progresses. `_is_fabricated_toolcall()` is called *after* that loop completes, on the fully-assembled `answer` string. On a positive match, the method returns an `AgentResult` with `output["answer"] = _SEARCH_UNAVAILABLE_FALLBACK`, `output["sources"] = []`, `output["grounded"] = False`. This correctly threads through `controller_agent.py`'s `_execute_plan()` fast-path into `ControllerResult.answer` → `result["answer"]` in the dict returned by `handle_task_with_plan()`.

**The bug.** `_stream_task()` in `main.py` emitted the `"done"` SSE event with only `task_id`, `status`, and `metadata`. The corrected `result["answer"]` existed in scope at that point but was never included in the event payload. `tasks.ts`'s `case 'done'` handler received no `answer` field, so it never overwrote the task's accumulated streamed text. The chat bubble permanently displayed whatever garbled fabricated-tool-call chunks had already been streamed, with no correction arriving — ever.

**Fix.**
- `main.py` (`_stream_task()`): The `"done"` SSE event now includes `"answer": result.get("answer", "")`. (See `main.py` lines 974–982, which also updated the SSE event table above from `task_id, status, metadata` to `task_id, status, metadata, answer`.)
- `tasks.ts` (`case 'done'`): Conditionally overwrites `task.answer` and clears `task.tokens` only when `event.answer` is present and differs from the already-accumulated streamed answer. For the normal (no fabrication) case, the condition `correctedAnswer !== t.answer` evaluates false and the patch is a no-op — no disruption to correctly streamed turns.
- `ChatPanel.svelte`: No changes required. Its existing reactive block already syncs bubble content from `tasksStore` on any store change.

**Live verification (same day).** The same incident shape was re-triggered after the fix: LangSearch returned a 500, the model fabricated tool-call syntax as its entire output, `_is_fabricated_toolcall()` detected and substituted the fallback. The chat bubble now shows `_SEARCH_UNAVAILABLE_FALLBACK` ("I don't have live search results for that — here's what I know from training, which may be stale or incomplete.") instead of the permanently garbled text seen before the fix.

#### Turn/task_id reconciliation — historical fix (2026-06-28)

**Prior bug.** `ChatPanel.svelte`'s `handleSubmit()` previously created the assistant turn with a temporary placeholder id (`const tempId = \`pending-${Date.now()}\``). The real `task_id` was only available once `submitTask()` resolved, which on the SSE path does not happen until the `[DONE]` sentinel arrives. `ChatPanel.svelte`'s reactive block (lines 30–46) matches live store updates to turns by `t.task_id === activeTask.task_id`; because the turn held `tempId` and `tasksStore` was keyed by the real UUID, no match ever occurred during an in-flight request. Every SSE status event updated the store correctly but nothing in `turns` reflected any of it until the stream ended. Only the final committed state was ever rendered; live status transitions and token-by-token content were never visible.

**Fix.** `submitTask()` in `tasks.ts` now accepts an optional third parameter (`task_id?: string`, line 88). Internally it uses `const id = task_id ?? crypto.randomUUID()` (line 90) for all store operations and the request body. `handleSubmit()` in `ChatPanel.svelte` generates `const task_id = crypto.randomUUID()` before creating either turn and passes it as the third argument to `submitTask(text, {}, task_id)`. Both the user turn and assistant turn receive the real `task_id` at creation time, so the reactive block finds a match from the first SSE event onward. The `tempId` variable and the post-`await` `turns.map(...)` reconciliation patch were removed entirely — no remaining references to `tempId` exist in `ChatPanel.svelte`.

This bug predated 2026-06-28's other changes. It was only surfaced when the addition of the "Routed to {agent}" status event created a visible gap: for the first time there was a status transition (routing → execution) that should have been visible mid-stream but was not, revealing that no live update ever reached the turn.

**Update 2026-06-29 — submitTask() resolves on 'done', not [DONE].** After the on_answer_ready fix caused `tasksStore.streaming` to flip false at the `'done'` SSE event, a secondary gap emerged: `submitTask()` in `tasks.ts` still resolved its `Promise<string>` only on the `[DONE]` sentinel, which `main.py` emits only after `producer_task` fully resolves (hooks included). This meant `submitting` in `ChatPanel.svelte`'s `handleSubmit()` stayed `true` for the full hooks duration — the textarea re-enabled visually but a fast follow-up Send was silently swallowed by the guard clause (`if (!text || submitting || $tasksStore.streaming) return`).

**Fix:** `submitTask()` restructured from `async function` to `function returning new Promise<string>((resolve) => { (async () => { ... })(); })`. `resolve(id)` is called at the `'done'` event (after `handleSSEEvent`'s store patch completes; the SSE read loop continues running un-awaited by the caller) and at `[DONE]` as a no-op safety-net (Promise spec: subsequent settle calls are silently ignored). Three additional safety-net resolves cover abrupt stream close, fetch error, and non-200 response. External signature (`(instruction, context?, task_id?) => Promise<string>`) is unchanged; the single call site in `ChatPanel.svelte` (`const task_id = await submitTask(text, {}, task_id)`) required no change.

*Minor open item:* a network drop after `'done'` resolves but before `[DONE]` causes `catch` to run post-resolve; `patchTask({status: 'failed'})` still executes, meaning a task the user experienced as complete could transiently show `status: 'failed'` in the store. Pre-existing race shape — the window is slightly wider now. Revisit on live evidence only.

---

### 7.7 Real-Time Token Streaming and In-Flight Status Visibility (2026-06-28)

#### Real-time token streaming — ConversationalAgent only

**Callback threading.** `handle_task_with_plan()` in `controller_agent.py` gained a fourth optional parameter `on_token: Callable[[str], None] | None = None`, and `_execute_plan()` gained the same parameter. When `on_token` is not None, `_execute_plan()` injects it into `subtask_context` under the key `"_on_token"`, alongside the existing `"_prebuilt_prompt"`, `"_prebuilt_system"`, and `"_routing"` keys. The `AgentInterface.run()` Protocol signature and `_dispatch()` are **unchanged** — the callback travels via `SubTask.context`, not the dispatch layer.

**`ConversationalAgent.run()`.** `on_token = context.get("_on_token")` is read once at the top of `run()`. At both existing `infer()` call sites — the prebuilt-prompt branch and the main RAG branch:
- If `on_token` is None: `self._runtime.infer(...)` is called exactly as before; behavior is unchanged.
- If `on_token` is not None: `self._runtime.infer_stream(...)` is called instead. Each yielded chunk is passed to `on_token(chunk)` and appended to a local list; the list is joined into the same `answer` variable that all downstream lines — `AgentResult` construction, `output["answer"]`, sources, grounded — already read. Both branches are wrapped in the same `try/except Exception` as the original blocking path.

**WikiAgent exclusion — permanent.** WikiAgent is not touched. Its output is raw XML consumed by `parse_model_xml()`; streaming partial XML before the full response is parseable would surface malformed output. This exclusion is structural: it falls out of WikiAgent never receiving `"_on_token"` in its `SubTask.context`.

**Queue-based SSE bridge in `main.py`.** `_stream_task()` uses an `asyncio.Queue[dict[str, str]]` (named `event_queue`) populated from the worker thread via `loop.call_soon_threadsafe(event_queue.put_nowait, item)`. Items are tagged dicts:
- `{"_kind": "token", "chunk": chunk}` — pushed by `on_token`
- `{"_kind": "status", "message": message}` — pushed by `on_status` (see next section)

`call_soon_threadsafe` was chosen over per-get `asyncio.to_thread` to avoid thread-crossing overhead for every token. The drain loop uses `await asyncio.wait_for(event_queue.get(), timeout=0.05)` while `producer_task.done()` is False, then a final synchronous `get_nowait()` drain after the task completes. A `_drain_item(item)` helper dispatches on `_kind`: `"status"` items yield `_sse({"type": "status", "message": ..., "task_id": ...})`; `"token"` items yield `_sse({"type": "token", "token": item["chunk"]})`. Both the live loop and the post-completion drain call `_drain_item()`.

For WikiAgent-routed plans, `on_token` is never called, the queue stays empty, the drain loop exhausts its 50 ms timeout on each iteration until `producer_task.done()`, and terminates immediately — no stall.

`handle_task_with_plan` is called with keyword arguments for both optional callbacks — `on_token=on_token, on_status=on_status` — making the binding explicit and position-safe against any future signature change.

#### on_status visibility event

A fifth optional parameter `on_status: Callable[[str], None] | None = None` follows `on_token` in both `handle_task_with_plan()` and `_execute_plan()`, threaded identically to `on_token`. Unlike `on_token`, it is **not** injected into `SubTask.context` — its sole call site is inside `_execute_plan()` itself, after the implicit extraction phase completes.

`on_status` is called exactly once per request at most: immediately after the `"TIMING implicit_extraction_end"` log line and before the `"TIMING working_state_start"` log line — i.e., right before `process_working_state_update()` runs. The call:
```python
if on_status is not None:
    on_status("Updating working memory…")
```
fires only inside the existing post-dispatch gate:
```python
if (db_path is not None and not plan.write_episode
        and results and results[0].status == TaskStatus.COMPLETE):
```
Turns where `plan.write_episode` is True, WikiAgent turns, failed results, or missing MemoryManager never enter this block — `on_status` is simply never called and no SSE event is emitted. No "done" counterpart is emitted for this status; the existing `"done"` SSE event already covers completion.

**Update 2026-06-29 — silently dropped after on_answer_ready.** Once `answer_ready_emitted` is set to `True` in `main.py`'s drain loop, subsequent queue events — including this `on_status("Updating working memory…")` — are dropped before reaching the SSE layer. The `'done'` event has already been sent by the time `on_status` fires (hooks run after `on_answer_ready` returns), so this status event no longer reaches the frontend on any qualifying conversational turn. The call site in `_execute_plan()` is unchanged; the suppression is entirely in the drain loop.

**Frontend compatibility — zero changes required.** `tasks.ts`'s `handleSSEEvent()` has a `case 'status'` handler at line 164 that patches `status_message` for any message string. `"Updating working memory…"` is rendered by the same in-bubble status line as `"Planning task…"` and `"Routed to {agent}"` — no frontend change was needed. This was confirmed by reading `tasks.ts` directly before writing code.

#### TIMING instrumentation

Seven `logger.info("TIMING %s t=%.4f", label, time.monotonic())` lines were added to `_execute_plan()` in `controller_agent.py` as permanent diagnostic instrumentation (not stripped). `import time` was added at module level. Labels and positions:

| Label | Position in `_execute_plan()` |
|---|---|
| `dispatch_start` | Immediately before `results = self._dispatch(...)` |
| `dispatch_end` | Immediately after `_dispatch()` returns |
| `implicit_extraction_start` | Before the `process_implicit_extraction` try block (inside the post-dispatch gate) |
| `implicit_extraction_end` | After that try/except closes |
| `working_state_start` | Before the `process_working_state_update` try block |
| `working_state_end` | After that try/except closes |
| `execute_plan_end` | Before the `if effective_agent_name == "conversational_agent"` final branching block |

`dispatch_end` marks the moment the full answer is already known — `ConversationalAgent.run()` has returned its complete `AgentResult` and `_dispatch()` has unblocked. For ConversationalAgent routes with streaming enabled, the last token was sent to the SSE queue before `_dispatch()` returned. The wall-clock gap `dispatch_end → execute_plan_end` is the total post-dispatch hook cost visible to the user as tail latency.

Grep pattern to isolate: `grep "TIMING" <server-log>`.

#### Tail-latency finding — process_working_state_update() dominates post-dispatch cost

Live timing (2026-06-28) confirmed `process_working_state_update()` accounts for the observed pause between the last streamed token and the "done" SSE event. Two live data points (normal conversational turns, `plan.write_episode=False`, `results[0].status=COMPLETE`):

| Turn | `working_state_start → working_state_end` | Outcome |
|---|---|---|
| 1 | 23.134 s | CHANGED |
| 2 | 18.835 s | CHANGED |

Both produced real state changes. Cross-reference: §9.5 Open Item 1 (pre-gate decision) and §9.5 Open Item 4 (reasoning-token exhaustion mechanism, previously confirmed at the inference layer). The `on_status("Updating working memory…")` event above was added specifically because of this finding — a 20+ second silent pause was the user-visible symptom.

*Update 2026-06-29 — user-visible impact closed.* The `on_answer_ready` early-completion callback (see next section) causes the `'done'` SSE event to fire immediately after dispatch, before either memory hook runs. The 18–23s pause still occurs server-side but `tasksStore.streaming` flips to `false` and the input re-enables before the hooks begin. Cross-reference: §9.5 Open Item 1 (pre-gate decision) remains open — this fix removes the user-visible consequence of the latency, not the latency itself.

#### Early-completion callback — on_answer_ready (2026-06-29)

The tail-latency finding above (18–23s post-dispatch pause) was confirmed to also cause full input lockout — the chat textarea and send button remained disabled for the entire duration of both memory hooks on every turn, not just during active streaming. Root cause: `on_status("Updating working memory…")` fired before `process_working_state_update()` ran, but the `'done'` SSE event (which flips `tasksStore.streaming` to `false` in `tasks.ts`, which drives `ChatPanel.svelte`'s `disabled` bindings) was only emitted after `_execute_plan()` returned — which required both hooks to complete first.

**Fix:** new `on_answer_ready: Callable[[dict[str, Any]], None] | None = None` parameter added to both `handle_task_with_plan()` and `_execute_plan()`, threaded identically to `on_token`/`on_status`. A new `_build_conversational_result()` helper (factored from the conversational_agent fast-path synthesizer block) constructs the `answer`/`sources`/`status`/`metadata` payload used by both the early callback and the final return path. `on_answer_ready` is called immediately after `results = self._dispatch(...)` returns, before either memory hook runs. Only fires on complete single-agent conversational dispatch — WikiAgent, failed, and synthesizer paths are unchanged.

In `main.py`, `on_answer_ready` bridges to `event_queue` via `call_soon_threadsafe` (same pattern as `on_token`/`on_status`). The drain loop handles `_kind == "answer_ready"` by immediately yielding `sources` + `done` SSE events and setting `answer_ready_emitted = True`. Subsequent queue events after this point (including the `on_status("Updating working memory…")` from the hooks) are silently dropped to avoid flickering the task status back to `'planning'` after `'done'`. After `await producer_task` completes (hooks finished), only the `[DONE]` sentinel is emitted. If `producer_task` raises after `'done'` was already sent, the error is logged but no error SSE event is emitted over the already-completed stream. All failure paths (routing exception, producer exception before `'done'`, `result.status == 'failed'`) are unaffected — they apply only when `on_answer_ready` was never called.

#### Update 2026-07-05 — runtime-level serialization + `finalizing` gate close the concurrency gap `on_answer_ready` reopened

`on_answer_ready` (above) removed the user-visible *wait*, but unlocking `tasksStore.streaming` at the same moment as `'done'` reopened a concurrency hazard: a user could submit turn N+1 while turn N's `process_implicit_extraction()` / `process_working_state_update()` were still running. Both calls invoke `runtime.infer()` against `omlx_runtime_client.py`, which talks to a single oMLX instance (port 8000, one Gemma 4B model) — so turn N+1's `main_dispatch` call and turn N's still-running background hook call ended up contending for the same instance. Confirmed by direct timestamp cross-reference (turn N+1's submission falling inside turn N's still-open `working_state_start`→`working_state_end` window) across two independent live sessions; full investigation trail, including the ruled-out call-frequency and thermal-throttling hypotheses that preceded this finding, is in `sessions-log.md` §16.

**Fix, defense in depth — either layer alone is sufficient:**
- **Backend — `omlx_runtime_client.py`.** A module-level `threading.Lock` (not `asyncio.Lock`: the full call chain from `conversational_agent.run()` down to the runtime client is synchronous, dispatched via `asyncio.to_thread()` from `main.py`, so serialization has to hold across worker threads, not coroutines) brackets the HTTP call + SSE consumption inside `infer_stream()`. `infer()` delegates to `infer_stream()`, so it is covered without separate locking. A `label` parameter (`"main_dispatch"`, `"implicit_extraction"`, `"working_state"`) threads through from each call site for diagnostic correlation with the existing `_log_infer_throughput()` lines. A `RUNTIME_OVERLAP` warning logs if any call ever finds the lock already held — structurally, this should never fire once the lock is in place (a contending call blocks at `.acquire()` rather than reaching the check); it ships as a live tripwire, not an expected event.
- **Backend — `main.py`.** New `task_complete` SSE event (see updated table, §7.6), emitted on every terminal path — success and error alike — only after `await producer_task` resolves, i.e. only after both memory hooks have actually finished.
- **Frontend — `tasks.ts` / `ChatPanel.svelte` / `ResearchView.svelte`.** New `finalizing` store field, separate from `streaming` (`streaming` is unchanged and still drives token-visible UI state). `finalizing` starts `true` at submission and clears only on `task_complete`, with fail-safes on `error`, `[DONE]`, and dropped-connection paths so a missed event cannot permanently disable submission. **`finalizing` gates only the submit action** — `handleSubmit()`'s guard clause and the send/query button's `disabled` binding in both components. The compose textarea (`ChatPanel.svelte`) and query input (`ResearchView.svelte`) are never disabled by it and remain freely editable throughout, including during the post-`'done'` finalizing window; the attach button in `ChatPanel.svelte` is likewise ungated by `finalizing` (only by the pre-existing local `submitting` flag), since attaching a file is a compose-time action, not a submission. A first pass disabled the whole textarea for the finalizing window and swapped its placeholder to a "saving" message; this was corrected same-day after review — for a sub-30s window, a disabled send button (with a native `title` tooltip explaining why) was judged sufficient, and a disabled compose box was not. `streaming` retains its prior meaning and call sites unchanged.

**Net effect:** `process_implicit_extraction()` and `process_working_state_update()` are now guaranteed to run without contending against the next turn's `main_dispatch` call, enforced at two independent layers (backend lock, frontend gate) so that either one holding is sufficient defense in depth. Live-verified via a deliberately fast-paced session (turn N+1 submitted during turn N's `working_state` window): zero `RUNTIME_OVERLAP` warnings, and a visible serialization gap in the timestamps (turn N+1's `main_dispatch` call blocked ~27s at the lock before its own HTTP POST completed) — the expected shape for a working blocking mutex.

*Scope note.* This closes the confirmed concurrency bug and its UX side effect (premature input unlock) only. The investigation that led here started from a laptop-heat observation; thermal throttling was directly tested and explicitly *not* confirmed as the mechanism (throughput varied non-monotonically across live samples, falsifying simple throttling before any fix was built around it). No heat/thermal claim is made here beyond what was verified: no concurrent runtime calls, no premature UI unlock. See `sessions-log.md` §16 for the full ruled-out-hypothesis trail.

#### Process note — mount staleness (recurring pattern, 2026-06-28)

Context staleness from mount-time reads recurred across multiple files in this session: `main.py`, `ChatPanel.svelte`, `controller_agent.py`, and `episodic_extractor.py` each exhibited stale-context issues traceable to reading file or variable state at initialization rather than at use time. Each instance was a new occurrence of the pattern documented in §3.7 (persona-cache staleness), §8.8 Open Item 6 (database-path disambiguation), and §8.8 Open Item 9 (cache-read disambiguation) — not a new principle. The existing discipline — verify the mechanism from current on-disk source, not from earlier in-context descriptions — applied uniformly across all four files. No new architectural rule is warranted.

### 7.8 Chat History Persistence (2026-06-29)

Conversation history (`turns: Turn[]`) was previously local component state in `ChatPanel.svelte`. Because Conversations and Files are separate SvelteKit routes (`+page.svelte` files rendered into `+layout.svelte`'s `<slot />`), navigating between tabs unmounted and remounted `ChatPanel`, resetting `turns` to `[]` on every navigation.

**Fix:** new `$lib/stores/chatHistory.ts` exports `chatHistoryStore: writable<Turn[]>([])` and the `Turn` interface (moved from `ChatPanel.svelte`). `ChatPanel.svelte` reads and writes through `$chatHistoryStore` / `chatHistoryStore.update()` exclusively — no local `turns` variable remains. The store lives at module level and survives any number of route navigations, resetting only on full page reload (by design — `SESSION_ID` in `tasks.ts` has the same lifecycle).

**Open item:** `chatHistoryStore` has no programmatic clear/reset path. Only a full page reload empties it. Not yet addressed; flagged for live observation.

*Forward reference: durable, cross-session, searchable persistence of chat turns — a separate concern from this session-only store — shipped later as the Chat History Tab; see §12.*

### 7.9 File Browser — `type` Discriminator Fix & Generated Files Listing (2026-07-07)

*Numbering note:* the File Browser (`/files`, §7.2) previously had no
dedicated subsection in this document — only a one-line row in the §7.2
routes table. Given this, the `type`-field fix below is filed as its own
new numbered entry rather than shoehorned into an unrelated existing
section; the provenance-badge fix above (§7.3) had a natural existing home
and was handled as a sub-entry there instead — different treatment for two
genuinely different situations, not an inconsistency.

**Bug: undefined preview badge and dead ingest-footer gating.**
`FileBrowser.svelte`'s content pane has always keyed off `selectedFile.type`
— a badge (`{selectedFile.type === 'wiki' ? 'badge-success' :
'badge-warning'}`) and a footer that only renders raw-file-only ingest
controls when `selectedFile.type === 'raw'`. But `FileEntry` (both the
Pydantic model in `main.py` and the TypeScript interface in
`stores/files.ts`) never actually had a `type` field — `/files/raw` and
`/files/wiki` returned metadata with no type discriminator at all, so
`selectedFile.type` was always `undefined`: the badge rendered the literal
text "undefined" with the `badge-warning` fallback style, and the
raw-only ingest footer never rendered for any file, `.type === 'raw'`
being false for every entry. Discovered live this session by opening
`generated_files/water.md` (created the prior session, 2026-07-06 18:39,
still 0 bytes — see `sessions-log.md`) in the browser and observing the
broken badge directly.

**Fix.** `FileEntry.type: Literal["raw", "wiki", "generated"]` added to
both the backend model (`main.py`) and the frontend interface
(`stores/files.ts`). `_file_entry()` (`main.py`) now takes an explicit
`type` parameter, threaded through from each call site (`get_files_raw`
→ `"raw"`, `get_files_wiki` → `"wiki"`, `post_file_upload` → `"raw"`, and
the new `get_files_generated` below → `"generated"`) rather than being
inferred — no ambiguity about which directory a listing came from.

**Shipped alongside: Generated Files listing.** Files written by `file_op`
(§4.6, §4.4b) land in `mcp_server/file_ops.py`'s sandboxed
`generated_files/` directory (§14.7) but had no UI surface at all before
this session — `water.md`'s 0-byte state was only discoverable by direct
filesystem inspection. New `GET /files/generated` endpoint (`main.py`,
mirrors `/files/raw`/`/files/wiki`'s shape exactly) backed by a new
`_state.generated_dir` (defaults to `project_root/generated_files`,
overridable via `LOCALIST_GENERATED_DIR`), and a matching entry added to
`/files/content`'s `allowed_roots` sandbox check. Frontend: new
`generatedFiles`/`generatedLoading`/`generatedError` stores and
`loadGeneratedFiles()` in `stores/files.ts`, and a new "Generated Files"
section in `FileBrowser.svelte` (loaded on mount alongside raw/wiki),
giving the file browser three panes total.

**Live-verified:** `GET /files/generated` confirmed returning `water.md`
(0 bytes, `type: "generated"`) both before and after the backend restart
that shipped this fix (`logs/backend.log`, `GET /files/generated` calls
either side of the `10:20:37` restart); `GET /files/content` for
`water.md` confirmed serving correctly through the sandbox check.

**Test suite:** no dedicated backend test added for the `type` field or
the new endpoint in this pass — covered only by the existing
`FilesResponse`/`FileEntry` Pydantic validation (a missing `type` on
construction is a hard `ValidationError`, not a silent gap) and the live
`GET /files/generated` round trip above. Flagged here as a real gap, not
elided silently: add an explicit `test_get_files_generated` case
alongside the existing `/files/raw`/`/files/wiki` tests if this endpoint
grows more logic than a directory listing.


### 7.10 Desktop UI Direction "1a — Inline Provenance" Ported to Web (2026-07-13)

A desktop-app UI direction, designed and approved separately as an HTML/React click-through
reference (`design_handoff_desktop_ui/reference.dc.html` + accompanying `README.md`), was ported
into this SvelteKit app as a visual/structural pass — existing stores' data-fetching logic, backend
contracts, and SSE handling were explicitly out of scope. Full session narrative and rationale for
individual decisions: `sessions-log.md` §30. This subsection is the current-state reference.

**Design tokens (`app.css`).** New dark palette (`--bg: #121214`, `--bg-panel: #1a1a1d`, etc.), a
new `--accent-2` (logo gradient only), `color-mix()`-based `--accent-dim/-mid/-glow`, and a
`[data-theme="light"]` override block (the `theme` store/`data-theme` attribute mechanism already
existed; only the light-theme values were added). `--topbar-h` 44px, `--radius`/`--radius-lg`
8px/12px, `--sidebar-w` default 236px.

**Sidebar (`Sidebar.svelte`).** Two-tone CSS gradient logo mark (`.brand-mark`, no image asset);
20×20 mono-letter nav icons (C/M/F/S). Chat and Files nav rows expand their sub-lists in place on
click (local `chatHistoryExpanded`/`filesNavExpanded` state) rather than always rendering them.
Files' sub-list contains the Wiki/Raw/Generated listing, upload, and per-file ingest that used to
live in `FileBrowser.svelte` — see §7.11. New `$lib/stores/sidebar.ts` (`sidebarWidth`,
`sidebarCollapsed`, both `localStorage`-persisted) backs a drag-to-resize divider (180–320px,
collapses fully below a 120px threshold) and the sidebar footer's theme-toggle switch.
`+layout.svelte` applies `sidebarWidth`/`sidebarCollapsed` to `#app-shell`'s `grid-template-columns`
directly via `document.getElementById('app-shell')` (necessary because `#app-shell` is defined in
`app.html`, outside the component tree `+layout.svelte` renders into) — animating between two fixed-
length grid tracks this way needs no `@property` registration, unlike animating the `--sidebar-w`
custom property value itself would have.

**Appbar (`StatusBar.svelte`).** Single consolidated chip (green dot + active inference-engine
name — `Ollama`/`oMLX`/`Foundry`, not the model id), shown only on the Chat screen; a separate
"N pending" chip shown only on Memory. New sidebar show/hide toggle button at the start of the bar.
Screen title now derives from `$page.url.pathname` rather than the component's previously-unused
`<slot />`. See §7.6 for what this replaced.

**Chat (`ChatPanel.svelte`).** The always-visible provenance bar (§7.3) is now collapsed by default:
a single `prov-toggle` pill (priority label + chevron) per completed assistant turn, expanding on
click to reveal the same tool/episodic/grounded/source chips §7.3 already documents — same
`metadata`/`sources` data and `p3Provenance()` labeling, purely a disclosure restructuring. Per-turn
expand state is a local `expandedProv` map keyed by `task_id` (falling back to `timestamp`).

**Memory (`EpisodesPanel.svelte`, `episodes.ts`).** `TYPE_COLORS` for `preference`/`decision`/
`workflow`/`correction` now reference `var(--accent-dim/mid)` / success / warning / error tokens
instead of bespoke hex triples. `fact`/`relationship`/`context` keep their pre-existing bespoke hex
colors — not covered by the design handoff, a known gap. Active filter-pill state is now solid
`--accent` background + white text; the pre-existing amber-tinted `Pending` chip's distinct active
state was intentionally preserved rather than unified into the same treatment.

**Settings.** Restyled as stacked cards. Runtime Backend segmented control
(`RUNTIME_BACKENDS`/`RUNTIME_BACKEND_LABELS`/`runtimeBackendLabel` in `model.ts`) drives the appbar
chip's label. **Live-wired as of §16.6 (2026-07-15)** — clicking a backend other than the active one
now calls `switchRuntimeBackend()` (a new `runtimeBackendSwitch.ts` store) against the real
`POST /settings/runtime-backend`, gated by a `confirm()` dialog; a no-op guard skips the confirm/
switch when the clicked backend is already active. The real active backend is synced from
`GET /health`'s new `backend` field (`model.ts`'s `health.subscribe(...)`), not just read from
`localStorage` — the browser's cached value is now only a paint-before-first-poll placeholder. The
former read-only "Chat Model" card is now a `<select>` populated from `fetchBackendModels()` for
whichever backend is currently selected in the segmented control (independent of which one is
actually active, so a different backend's models can be previewed before switching to it);
`onchange` calls `pinChatModel()` against `POST /settings/runtime-backend/{backend}/chat-model`. The
free-text "Embedding model ID" field (and the rest of the by-then-empty "Model Configuration" card)
was deleted — confirmed inert for every backend per §33/§34. New Streaming-responses /
Episodic-write-approval toggles are UI-only placeholders with no backing endpoint — flagged in the
code as such.

**Corpus Embeddings (2026-07-17).** New card directly below Runtime Backend, wired to the real
`POST /memory/reembed` (§16.4) via a new `reembedCorpus.ts` store — not a placeholder. A
"Re-embed Corpus Now" button, gated by the same `confirm()`-dialog pattern as the Runtime Backend
switch, calls `reembed_corpus()` and blocks (single `asyncio.to_thread` call, no progress
callback) until every wiki/raw document has been re-embedded; an indeterminate spinner covers the
wait, matching the Runtime Backend card's loading treatment. A `corpus_stale` badge — sourced from
the new `corpus_stale` field on `GET /memory/stats`, fetched once on mount and refreshed after a
re-embed completes — reads "Corpus embeddings out of date — re-ranking is running keyword-only"
when `MemoryManager._corpus_stale` is set. On success the card shows "Re-embedded X of Y
documents."; on failure it shows `reembedError` in `var(--error)`, same as the Runtime Backend
card's error paragraph.

**Files.** See §7.11.

**Verification posture.** `npm run check` and a production `npm run build` both clean throughout.
No browser-automation tool is available in this environment; verification was structural (SSR HTML
diffed for expected markup against a live backend) rather than a rendered visual/pixel check —
explicitly flagged as a limitation, not asserted as a full visual pass.

**Open items:**
- `fact`/`relationship`/`context` episode-type colors don't participate in the token system.
- `agents.ts`'s `loadAgents()` polling is now unconsumed by any component (the agents chip/popover
  it fed no longer exists in the UI) — not removed this session.
- No live human browser/visual QA pass has been performed as of this writing.

### 7.11 File Browser Restructure: Sidebar-Driven Listing, Download, and Two-Step Delete (2026-07-13)

Continuation of §7.10's Files change, with two new real capabilities added in the same session.
Full narrative: `sessions-log.md` §31.

**Structure.** `FileBrowser.svelte` is now a full-width preview-only pane. Wiki/Raw/Generated
listing, upload, and per-file ingest live in `Sidebar.svelte`'s expandable Files sub-nav instead —
each group independently collapsible, all expanded by default. Selection state moved to a new
`$lib/stores/fileSelection.ts` (`selectedFile`, `fileContent`, `fileContentLoading`,
`fileContentError`, `selectFile()`, `closeFile()`), shared between the sidebar (selection UI) and
`FileBrowser.svelte` (preview rendering) — previously this was local component state inside
`FileBrowser.svelte` alone.

**New endpoint — `GET /files/download`.** Returns a `FileResponse` with an explicit
`Content-Disposition: attachment` header and a `mimetypes.guess_type()`-derived media type, so the
browser performs a real download (Safari's Downloads queue, specifically) rather than navigating to
raw content the way `GET /files/content`'s JSON response would. Gated by the same raw/wiki/generated
allowed-roots check as `/files/content`, but using `Path.is_relative_to()` rather than those
endpoints' string-prefix check (`str(target).startswith(str(root))`) — the prefix check is
vulnerable to a sibling-directory bypass (an allowed root of `/data/wiki` also matches
`/data/wiki_evil`); `/files/content` and `/files/upload` still use the older, narrower check
(flagged as an open item below, not fixed in this pass). Frontend: a plain `<a
href=".../files/download?path=..." download="filename">` in `FileBrowser.svelte`'s footer for
`type === 'generated'` files — no blob/JS handling; the anchor's `download` attribute plus the
response header alone trigger the native download.

**New endpoint — `DELETE /files`.** Same `is_relative_to()`-gated path check. Unlinks the file and
calls `MemoryManager.remove_document()` to purge any `document_index` row for that path — a no-op
for generated files (never indexed), necessary for raw/wiki so a deleted file doesn't linger in RAG
retrieval. Frontend: every sidebar file row gets a trash-icon button. First click swaps that row in
place for an inline `Delete "<filename>"?` / Confirm / Cancel prompt (`confirmDeletePath` local
state in `Sidebar.svelte`) — a two-step confirmation by design requirement, using the app's existing
inline-review pattern (cf. the wiki diff apply/discard flow, §17) rather than a browser-native
`confirm()` dialog. Confirming refreshes the affected list and closes the preview pane if the
deleted file was open.

**Verification.** Both endpoints exercised directly against the running backend: correct headers
and byte-identical content for download; real throwaway files created and deleted for the delete
path, confirmed removed from disk; a path-traversal attempt (`/etc/passwd` / `/etc/hosts`) 403s on
both. Each check repeated through the Vite dev server's `/api` proxy — the actual path the browser
UI uses — not just direct-to-backend, to rule out a proxy-layer discrepancy.

**Test suite.** No backend unit tests added for either endpoint — covered only by the live-request
verification above, following the same precedent set by §7.9's `GET /files/generated` (which also
shipped without dedicated tests). Flagged as a real gap.

**Open items:**
- No backend test coverage for `/files/download` or `DELETE /files`.
- `/files/content` and `/files/upload`'s path-containment checks remain on the older, narrower
  string-prefix pattern — not fixed in this pass, worth a consistency sweep later.

### 7.12 Math Rendering (KaTeX) in `MarkdownRenderer.svelte` (2026-07-18)

Live use surfaced literal LaTeX source (e.g. `$\rightarrow$`, quote-escape commands) appearing
verbatim in assistant replies instead of the symbols they encode. Investigation (full trail:
`sessions-log.md` §41) traced the entire backend prompt-assembly path — user-profile injection,
episodic memory, `PromptBuilder`'s slot rendering — and found no formatting bug there; the model
itself emits real LaTeX/MathJax source (`\rightarrow` is the standard command for →, wrapped in
`$...$` inline-math delimiters) because its training data biases arrow/symbol notation toward math
mode even in plain prose. This only surfaces as broken text because `MarkdownRenderer.svelte`
renders CommonMark, not KaTeX/MathJax — the delimiters previously passed through uninterpreted.
Since model output can't be controlled from this side, the fix is scoped entirely to the renderer.

**Fix.** `katex` (0.18.0) added as `localist-ui`'s first runtime dependency — a deliberate,
documented exception to this file's previous "no third-party deps" design comment, not a silent
violation of it. `katex/dist/katex.min.css` is imported once at the top of the component; Vite
bundles KaTeX's font files as hashed static assets, so there is no CDN dependency (consistent with
the project's local-first constraint).

`inlineFormat()` extracts math spans to placeholder tokens *before* the existing bold/italic/
code/link regexes run, then swaps in the real KaTeX HTML at the end — otherwise KaTeX's own markup
(full of literal `< > " '`) would get mangled by the later substitutions. Tokens use a
Private-Use-Area sentinel (`U+E000`) that survives `escape()` untouched and can't collide with
anything the other regexes match. A new `renderMath()` helper wraps `katex.renderToString()` with
`throwOnError: false, trust: false`; on any unexpected exception it falls back to the literal
source, so malformed LaTeX degrades to an inline KaTeX error span rather than breaking the render.

**Currency disambiguation.** `$$…$$` is always treated as display math — plain prose never doubles
dollar signs like that. Single `$…$` is only treated as inline math when its content contains a
backslash command, which covers every LaTeX symbol a model emits this way (`\rightarrow`,
quote/accent commands, `\alpha`, …) while leaving plain currency mentions (`$5`, `$10 total`)
completely untouched.

**Known gap.** No change to the pre-existing per-line paragraph architecture — a `$$...$$` display
block spanning multiple lines within one paragraph won't be recognized, since paragraph lines are
still formatted independently (§ design predates this change). Accepted rather than fixed: the
reported symptom is always single-line inline math embedded in prose, not multi-line display blocks.

**Verification.** `npm run check` (0 errors) and `npm run build` (succeeds, KaTeX assets bundle
correctly) both clean. A standalone Node smoke test against the real `katex` package confirmed
`$\rightarrow$` renders to a proper `→` glyph, `$5 and $10 total` passes through unchanged, and a
malformed math span degrades to an inline error span without crashing. No browser-automation tool
was available to screenshot the live UI directly (same limitation noted at §7.3, §7.10) — but the
user live-confirmed it afterward with a real chat transcript showing LORA's own
`Request → Tool/Vault Search → Grounding against Local Truth → Cited Response → Memory Update`
summary rendering with actual `→` glyphs instead of literal `$\rightarrow$` text.
