## 24. WikiAgent Ingest Alignment with the Local OCR Service

### 24.1 Overview

Closes the gap §22.8 explicitly flagged and deferred: `WikiAgent`'s raw-document ingest path
(`wiki_agent.py`) accepted a PDF/image `raw_path` only via oMLX's `infer_with_file()` — not a real
OCR step, but MarkItDown-based server-side document conversion, oMLX-only. On any other backend
(Ollama, Foundry) the same file hit `read_text_file()`, a plain UTF-8 decode that raises
`UnicodeDecodeError` on binary content, failing the whole ingest with a generic "File load error."
Neither path used Apple Vision/PyMuPDF (§22). This section makes wiki ingestion of images/PDFs use
the same `ocr_extract` MCP tool chat uploads already use (§22.2), and widens the product surface
(`POST /files/upload`, the wiki upload UI) so a PDF/image can actually reach that path, mirroring
how §22 did the same for chat attachments.

### 24.2 Shared Extension→Mime Map (`mcp_server/ocr.py`)

`_OCR_MIME_BY_EXTENSION`, previously private to `main.py`, is now the public
`OCR_MIME_BY_EXTENSION` constant in `mcp_server/ocr.py` (`.png`/`.jpg`/`.jpeg`/`.webp`/`.heic` →
`image/*`, `.pdf` → `application/pdf`). `main.py` imports it (aliased back to
`_OCR_MIME_BY_EXTENSION` so its call sites were untouched) rather than keeping its own copy —
needed so `wiki_agent.py` (which cannot import `main.py` without a circular dependency) has a
single source of truth to check `raw_path`'s extension against, shared with `POST /chat/files`'s
existing routing.

### 24.3 `POST /files/upload` / `GET /files/raw` Widened (`main.py`)

`POST /files/upload` (`main.py:2117`) accepted only `.md`/`.txt`; now also accepts
`OCR_MIME_BY_EXTENSION`'s extensions, saving raw bytes to `raw/` unchanged — deliberately **no**
OCR at upload time. Unlike chat uploads (where OCR'd text *is* the thing that needs to reach the
prompt), the wiki's `raw/` file is the canonical source document, ingested (and potentially
re-ingested) later by `WikiAgent`; OCR happens lazily at that point (§24.4), same as `.md`/`.txt`
raw files were never touched at upload time either.
`MemoryManager.index_document(dest, "raw", None, False)` — called right after upload to make the
raw file searchable immediately — already catches a UTF-8-decode failure on binary content and
logs+skips (`memory_manager.py:2467-2472`), so an OCR-eligible raw file is expected to sit
un-indexed (only the resulting *wiki page*, not the raw source, ever gets indexed by
`WikiAgent` — see its class docstring) until it's actually ingested. Self-resolving, not a bug.

Widening the upload endpoint alone would have been a dead end: `GET /files/raw` (`main.py:1924`)
filtered its directory listing to `.md`/`.txt` only, so an uploaded PDF/image would have been
invisible in the sidebar the moment after upload. Widened the same way.

`GET /files/content` (the plain-text preview endpoint) is deliberately **not** widened — it still
reads as UTF-8 and 500s on binary content by design; OCR-eligible raw files route to
`/files/download` instead on the frontend (§24.5), never to `/files/content`.

### 24.4 `WikiAgent.run()` Ingest Routing (`wiki_agent.py`) — the core fix

Two changes, both gated on the raw file's extension:

1. **`_resolve_raw_path()`** (`wiki_agent.py:1788`) previously hard-rejected anything but `.md`/
   `.txt` and required the file to decode as UTF-8 (`is_text_file()`) — a validation gate that ran
   *before* any inference-path logic. Caught live (not by inspection, see §24.6): an OCR-eligible
   file never even reached the new routing below without first widening this gate. Now: an
   extension present in `OCR_MIME_BY_EXTENSION` is accepted without the UTF-8 check (it's binary by
   design); `.md`/`.txt` keep the existing check unchanged; anything else is still rejected.

2. **Load-inputs step** (`wiki_agent.py:1273` area): previously always called
   `read_text_file(raw_path)` to build `raw_content`. Now checks
   `OCR_MIME_BY_EXTENSION.get(raw_path.suffix.lower())` first — if the extension matches, a new
   module-level `extract_raw_content_via_ocr(runtime, raw_path, mime_type)` (`wiki_agent.py:249`)
   supplies `raw_content` instead, with its own "OCR extraction error" message on failure (kept
   outside the pre-existing generic "File load error" try/except so OCR failures are legible on
   their own terms). `extract_raw_content_via_ocr()` mirrors `main.py`'s
   `_extract_text_via_ocr()` exactly: `ocr_extract`'s sandboxing only resolves paths under its own
   `mcp_server.ocr.get_upload_root()` (a `chat_uploads/` subdirectory), which is a *different*
   sandbox than the wiki's `raw/` directory, so `raw_path`'s bytes are copied to a temp file there
   first, `ocr_extract` is dispatched via a new `MCPToolDispatcher(runtime=...)` instance
   (`WikiAgent` did not previously hold one), and the temp file is removed in a `finally` block
   regardless of outcome.

3. **Inference-path selection** (`wiki_agent.py:1327`): `use_file_upload` (which picks
   `infer_with_file()` vs. `infer()`) is now `ocr_mime_type is None and hasattr(self._runtime,
   "infer_with_file")` — i.e. an OCR-extracted raw file **always** takes the `infer()`
   string-prompt path, even on oMLX. Once OCR output is plain text, this is exactly the design
   principle §22.1 already established for chat uploads: nothing downstream needs to know OCR
   happened. This is the actual fix — ingestion of a PDF/image raw file is now identical across
   oMLX/Ollama/Foundry, where before it silently depended on which backend was active.

`.md`/`.txt` raw files are completely unaffected by any of this — `ocr_mime_type` is `None` for
them, so every branch above falls through to the pre-existing behavior byte-for-byte.

### 24.5 Frontend (`Sidebar.svelte`, `FileBrowser.svelte`, `fileSelection.ts`)

New shared `localist-ui/src/lib/utils/ocr.ts` (`OCR_EXTENSIONS`/`extOf`/`isOcrExtension`) — the
frontend mirror of `OCR_MIME_BY_EXTENSION` (§24.2). `ChatPanel.svelte` now imports from it instead
of keeping its own local copy (previously the only place this list existed on the frontend).

`Sidebar.svelte`'s upload control (extension gate + `accept` attribute) widened to `.md`/`.txt` +
`OCR_EXTENSIONS`; label changed from "Upload .md / .txt" to "Upload file" since the specific list
no longer fits cleanly in a button label.

`fileSelection.ts`'s `selectFile()` previously always fetched `/api/files/content` — which 500s on
binary content (§24.3). Now checks `isOcrExtension(file.filename)` first and, if true, skips the
fetch entirely and sets a new `filePreviewUnavailable` store instead of `fileContent`.
`FileBrowser.svelte` renders a "No text preview for this file type" message for that state, plus a
Download link (reusing the same `/api/files/download` pattern already used for `generated`-type
files) alongside the existing "Ingest to wiki" button in the raw-file footer — OCR-eligible raw
files can be downloaded (to inspect the original image/PDF) or ingested (which triggers the real
`ocr_extract` pass, §24.4), but never text-previewed.

### 24.6 Bug Caught During Build

`_resolve_raw_path()`'s `.md`/`.txt`-only gate (§24.4, point 1) was not visible from reading
`WikiAgent.run()`'s inference-path branching alone — an initial live test (§24.7) against the
real, already-running backend failed immediately with `"raw_path must be a .md or .txt file, got:
.png"`, before the new OCR-routing code was ever reached. Caught by testing, not inspection; fixed
by widening the gate as described above.

### 24.7 Test Coverage

`tests/test_wiki_agent.py`: `TestOcrRawPathIngest` (OCR routing taken with and without
`infer_with_file` present on the fake runtime — using genuinely non-UTF-8 raw bytes so a regression
back to `read_text_file()` would fail loudly rather than silently; OCR failure surfaces via
`_fail()`), `TestExtractRawContentViaOcr` (temp-file created and cleaned up on both success and
failure, correct `MCPToolDispatcher` call shape), `TestResolveRawPathOcrExtensions` (binary
`.png`/`.pdf` now accepted, unsupported extensions and non-UTF-8 `.txt` still rejected exactly as
before). New `tests/test_files_upload_endpoint.py` backfills coverage that never existed for
`POST /files/upload`/`GET /files/raw` at all (`.md` upload, OCR-eligible `.png`/`.pdf` uploads
saved byte-for-byte unchanged, unsupported extension still 422s, overwrite-on-same-name, widened
listing). `npm run check` (frontend): 0 errors, 0 warnings. Full backend suite: 1443 → 1460
passed, 0 failed.

Live-verified against the real running stack (real Apple Vision OCR, real Ollama inference — the
actually-active backend at verification time, chosen specifically because it has no
`infer_with_file` and would previously have hit the UTF-8-decode failure — real MCP round trip, no
mocks): a real PNG containing the text "Localist OCR wiki ingest test", submitted via `POST /task`
with `context.raw_path` pointing at it and `auto_apply: false`, correctly proposed a wiki page
reflecting the OCR'd content; disk was left untouched (as `auto_apply: false` requires) and the
`chat_uploads/` temp file was confirmed removed afterward. Also live-verified: a real PNG uploaded
through `POST /files/upload` correctly appears in `GET /files/raw` and was cleaned up after.

No live-browser verification of the frontend changes (Sidebar upload control, preview-unavailable
state) was performed — no browser-automation tool was available in the build environment; only
`svelte-check`/backend round-trips were used, matching the same constraint noted in §22.9 for the
original OCR service build.
