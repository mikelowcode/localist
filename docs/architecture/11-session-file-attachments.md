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
| `add_file` | `(filename, content, source="upload") -> str \| None` | Returns `None` on success; user-readable error string on rejection. Three rejection conditions checked in order: extension not allowlisted, content exceeds `MAX_FILE_TOKENS` (4,000), adding file would exceed `MAX_TOTAL_TOKENS` (20,000). Same-filename re-upload replaces in-place with budget re-check. `source="wiki_pin"` (added 2026-07-21, see §11.8) marks a pinned wiki page rather than an uploaded file — same budget rules, but the per-file-too-large message says "too large to pin" instead of "is too large", since the user didn't choose the page's size. |
| `remove_file` | `(filename) -> bool` | Returns `True` if removed, `False` if not found. Used for both uploads and wiki pins — a pin is cached under `filename = f"{stem}.md"` like any other entry, so no separate unpin path exists. |
| `get_files` | `() -> list[SessionFile]` | Returns all cached files in insertion order, each carrying its `source`. Returns `[]` when cache is empty. |
| `clear` | `() -> None` | Removes all cached files. Intended for future "clear chat" affordance. |

**Internal structure:** `OrderedDict[str, tuple[str, str]]` (filename →
`(content, source)`). Insertion order preserved for deterministic prompt
assembly. Module-level (process-lifetime). No threading locks required —
FastAPI's async model serialises access in practice for this single-user
local deployment.

**Token estimation:** `len(content) // 4` — identical to `PromptBuilder._estimate_tokens()`.

### 11.3 API Endpoints

Two new endpoints in `main.py`. Route prefix follows the existing convention
(no `/api` prefix in the route string — Vite dev proxy strips `/api` before
forwarding):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/files` | Upload a file. Reads bytes, decodes UTF-8 (422 on failure), calls `session_files.add_file()`. Returns `{filename, token_estimate}` on success; 400 + readable `detail` on rejection. |
| `DELETE` | `/chat/files/{filename}` | Remove a file by name. Returns `{removed: true}` on success; 404 if not found. Also used to unpin a wiki page — see §11.8. |
| `POST` | `/chat/pin-wiki-page` | Added 2026-07-21 (§11.8). Pins an existing wiki page by `{stem}` into the same cache. Returns `{filename, token_estimate, source: "wiki_pin"}` on success; 404 if no `{stem}.md` exists on disk; 400 + readable `detail` on budget rejection; 503 if `wiki_dir` isn't configured. |

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

### 11.8 Wiki Page Pinning (2026-07-21)

Extends the attachment picker so a user can pin an existing wiki page — not
just an uploaded local file — into the session file cache. The motivating
case: when asking LORA to propose a diff against a specific wiki page, the
user can pin that exact page first so the model gets the real current file
content instead of guessing or being asked to paste it in. (This also
supplies the `source` tag a planned Planner P1b enhancement depends on, to
short-circuit diff-target resolution when a page is already pinned.)

**No new listing endpoint.** `GET /files/wiki` (already existed for the
Sidebar's file browser) returns `FileEntry{name, filename, path, size,
modified, type}` for every `.md` file in `wiki_dir` — sufficient for the
picker (`name` is the stem). The frontend reuses the existing
`loadWikiFiles()` / `wikiFiles` store (`localist-ui/src/lib/stores/files.ts`)
directly rather than adding a parallel fetch path.

**Pin validates against disk, not the graph index.** `POST
/chat/pin-wiki-page` checks `wiki_dir / f"{stem}.md"` exists directly,
rather than resolving against `MemoryManager.list_graph_node_stems()` (used
by Planner P1b/P3c for free-text stem resolution). The graph index is only
rebuilt on an explicit trigger and can lag behind real files on disk —
pinning must not fail just because the graph hasn't caught up.

**`SessionFile.source` field** (`prompt_builder.py`): `Literal["upload",
"wiki_pin"]`, defaulting to `"upload"` so all existing callers are
unaffected. `_slot_session_files()` renders a `wiki_pin` entry's opening
label as `--- {filename} (from the vault) ---` instead of the plain
`--- {filename} ---` used for uploads, so the model can cite a pinned page
distinctly per LORA's honor-code citation style.

**UI (`ChatPanel.svelte`):** A second small icon button sits beside the
paper-clip (unchanged) — a bookmark icon that opens a lightweight popover
listing `$wikiFiles`. Selecting a page calls `POST /chat/pin-wiki-page`
and pushes the result into the same `attachedFiles` array uploads use; the
resulting pill swaps its icon (bookmark vs. paperclip) based on `source` but
is otherwise identical, including reusing `removeAttachedFile()` unchanged
for unpinning. No tab strip or unified two-source modal — a second trigger
button was chosen over restructuring the paper-clip's existing
one-click-to-Finder flow.

### 11.9 Test Suite

Test suite at feature completion: **958 tests, 0 failures**, up from 445 at
§11.7's writing (unrelated feature work landed in between) plus new coverage
added for wiki-page pinning: `tests/test_session_files.py` (new — direct
`add_file()`/`get_files()` coverage for the `source` parameter, both
defaulted and `wiki_pin`, including the pin-specific oversized-page message),
`tests/test_chat_pin_wiki_page.py` (new — `POST /chat/pin-wiki-page` success,
404, 400, 503, and validation cases), and a new `TestSlotSessionFilesSource`
class in `tests/test_prompt_builder.py` covering the "(from the vault)"
labeling.

