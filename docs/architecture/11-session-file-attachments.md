## 11. Session File Attachments

### 11.1 Overview

Session file attachments allow the user to upload text and code files directly
into the chat interface. Uploaded files are injected into every subsequent prompt
for the duration of the session via the `[SESSION FILES]` slot (Slot SF — see §3.2).
They are not wiki documents: no ingestion, no embedding, no graph indexing, no
persistence across backend restarts.

This feature bypasses the Planner routing ladder entirely. File content reaches
the model as literal prompt text, not as retrieved corpus context.

**Design constraints confirmed before implementation:**
- oMLX context window confirmed live at 131,072 tokens (`GET /v1/models` →
  `max_model_len: 131072`). All existing slot ceilings sum to ~3,924 tokens
  worst-case, leaving ~127K tokens of headroom. Per-file ceiling (4,000 tokens)
  and total slot ceiling (20,000 tokens) were chosen conservatively relative to
  this headroom; the binding constraint is Gemma 4B attention fidelity at long
  contexts, not budget exhaustion.
- Render-time truncation only: full content stored in cache; ceiling applied at
  `PromptBuilder.build()` call, not at upload time. Consistent with "never
  silently drop" principle.
- Reject-with-error on budget exceeded: no silent LRU eviction.

### 11.2 Backend Cache (`session_files.py`)

New module `backend/session_files.py`. Public API:

| Function | Signature | Behaviour |
|---|---|---|
| `add_file` | `(filename, content) -> str \| None` | Returns `None` on success; user-readable error string on rejection. Three rejection conditions checked in order: extension not allowlisted, content exceeds `MAX_FILE_TOKENS` (4,000), adding file would exceed `MAX_TOTAL_TOKENS` (20,000). Same-filename re-upload replaces in-place with budget re-check. |
| `remove_file` | `(filename) -> bool` | Returns `True` if removed, `False` if not found. |
| `get_files` | `() -> list[SessionFile]` | Returns all cached files in insertion order. Returns `[]` when cache is empty. |
| `clear` | `() -> None` | Removes all cached files. Intended for future "clear chat" affordance. |

**Internal structure:** `OrderedDict[str, str]` (filename → content). Insertion
order preserved for deterministic prompt assembly. Module-level (process-lifetime).
No threading locks required — FastAPI's async model serialises access in practice
for this single-user local deployment.

**Token estimation:** `len(content) // 4` — identical to `PromptBuilder._estimate_tokens()`.

### 11.3 API Endpoints

Two new endpoints in `main.py`. Route prefix follows the existing convention
(no `/api` prefix in the route string — Vite dev proxy strips `/api` before
forwarding):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/files` | Upload a file. Reads bytes, decodes UTF-8 (422 on failure), calls `session_files.add_file()`. Returns `{filename, token_estimate}` on success; 400 + readable `detail` on rejection. |
| `DELETE` | `/chat/files/{filename}` | Remove a file by name. Returns `{removed: true}` on success; 404 if not found. |

**Proxy convention note (2026-06-30):** The initial implementation registered
these routes as `/api/chat/files` — breaking the existing convention and causing
404s through the Vite proxy in browser sessions (masked by direct `curl`
verification against port 8001). Fixed before shipping. **Standing rule:** any
new endpoint verified by direct `curl` must also be spot-checked against the
existing proxy-convention route prefix before the prompt is closed.

### 11.4 UI (`ChatPanel.svelte`)

**Paperclip button:** Rendered inside `.input-wrap` immediately left of the
textarea. Triggers a hidden `<input type="file">` on click. Disabled during
active streaming (`$tasksStore.streaming`). Client-side extension allowlist
applied in `handleFileSelect()` as defense-in-depth before the network call.

**Attached-files pill strip:** Rendered below `.input-wrap` when
`attachedFiles.length > 0 || attachError`. Each pill shows:
- Paperclip SVG icon
- Filename (truncated with `text-overflow: ellipsis` at 180px)
- Token estimate (`~{N}t`)
- `×` remove button (calls `DELETE /api/chat/files/{filename}` via Vite proxy,
  then filters local `attachedFiles` array)

**Error display:** `attachError` rendered inline in the pill strip area as
`.attach-error` span. No `alert()` or modal.

**State:** `attachedFiles: AttachedFile[]` is local component state (not a Svelte
store). Source of truth for the attachment list is the backend cache; the frontend
array is a display mirror updated optimistically on upload success and remove
confirmation. A backend restart clears the cache without clearing the frontend
display — acceptable for a local development tool; surfaced as a known limitation.

### 11.5 Prompt Assembly Integration

Session files are injected in `ControllerAgent._execute_plan()` at Step 6:

```python
system_prompt, user_prompt = _PROMPT_BUILDER.build(
    ...
    session_files = _session_files.get_files() or None,
)
```

The `or None` is consistent with the call site's convention for all other optional
args. `get_files()` returning `[]` and `get_files()` returning `None` both produce
a cleanly absent `[SESSION FILES]` slot — no label, no whitespace.

`ConversationalAgent.run()` also passes `session_files=_session_files.get_files()`
to its own `_PROMPT_BUILDER.build()` call as defense-in-depth for the rare
non-prebuilt path. In production all turns take the prebuilt path (controller
assembles the prompt at Step 6 and passes it via
`subtask.context["_prebuilt_prompt"]`); `conversational_agent`'s own `build()`
call is only reached when `_prebuilt_prompt` is absent from the subtask context.

### 11.6 Open Items

**Open Item 1 — `[SESSION FILES]` not surfaced in UI badge strip.**
`AgentResult.sources` currently carries corpus document paths only. When LORA
grounds an answer in an attached session file, no `📎 filename` badge appears in
`ChatPanel.svelte`'s sources row. Revisit when sources rendering is next in scope.

**Open Item 2 — Frontend display state not cleared on backend restart.**
`attachedFiles` in `ChatPanel.svelte` is local component state. A backend restart
clears `session_files.py`'s in-memory cache, but the frontend pill strip still
shows the previously attached files. The model will behave as if no files are
attached (cache is empty), while the UI shows them as present. Acceptable for a
local dev tool; log for future "clear chat" affordance.

**Open Item 3 — `chatHistoryStore` persistence and 30-day eviction.**
Eviction scheduling is meaningless until chat history is durable. `chatHistoryStore`
is currently an in-memory Svelte `writable` — nothing persists it to SQLite or any
store. A future design pass is required: new `chat_turns` SQLite table, schema
migration (v5 → v6), and Memory tab UI changes to surface chat history detail
alongside a clear affordance. Not scoped.

**Open Item 4 — PDF and image support.**
oMLX and Gemma 4B natively support OCR, image, and PDF. The extension allowlist
in `session_files.py` and the client-side set in `ChatPanel.svelte` are the only
gate. Removing `.pdf` and image extensions from both and adding multipart/form-data
content handling to the `POST /chat/files` endpoint is the full scope of the
future work. Not scheduled.

**Open Item 5 — UI freeze on tab navigation during in-flight streaming task; root
cause unconfirmed, browser profiling required (2026-07-06).** A live incident: the
UI became fully unresponsive — including the in-app refresh button and a full
browser F5 reload — after navigating from `/conversation` to the Files tab while a
`file_op`-triggering chat turn was in-flight (~49s end-to-end). The backend/Vite
dev-server layer was live-refuted as the cause: a real ~31s-open SSE connection
plus a concurrent request burst through a connection-capped agent showed no
queuing or stalling there. Two safe mitigations were landed regardless — a
throttled Markdown re-parse in `MarkdownRenderer.svelte` (max one `parse()` call
per 75ms while streaming, with a guaranteed final parse on stream-end) and a
microtask yield between coalesced SSE lines in `tasks.ts` — but neither is
confirmed to address the actual freeze, and the microtask yield specifically does
not let the browser paint or handle input mid-burst (a macrotask yield would be
needed for that, not yet implemented). Blocked on the lack of any browser
automation/profiling tool (no `chromium-cli`, no Playwright) in this environment.
Full diagnostic detail, ranked candidate causes with confidence levels, and the
two mitigations: see `sessions-log.md` §20 (2026-07-06).

### 11.7 Test Suite

Test suite at feature completion: **445 tests, 0 failures.**
No new test files were added for this feature. The `SessionFile` dataclass and
`_slot_session_files()` builder are covered by the smoke test run during Prompt 1
implementation (throwaway script, not committed). A permanent test class for
`session_files.py` and the new `PromptBuilder` slot is a future addition.

