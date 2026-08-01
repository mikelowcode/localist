# 21 — Multimodal Image Upload: oMLX Gemma 4 OCR Routing (Scope)

> **Status: Rejected (2026-08-01) — superseded by §22.** Never built. After this
> outline was completed, a second alternative (a dedicated, inference-engine-agnostic
> local OCR service — Apple Vision + PyMuPDF, no dependency on which chat backend is
> active) was scoped and chosen instead; see §22.8 for the full reasoning. Kept in
> place, not moved to `archive/`, as a documented record of the alternative
> considered. §11.6's Open Item 4 and §3's Slot SF note are now resolved by §22, not
> by this document.

## 21.1 Goal

Let users attach image files in the chat panel's upload UI and have Gemma 4
(oMLX) read them directly via its built-in OCR/vision capability — no separate
OCR service, no new model, no image preprocessing pipeline beyond what's needed
to get bytes to oMLX in its existing multimodal wire format.

## 21.2 Non-goals

- No dedicated OCR microservice or third-party OCR library.
- No PDF support in this pass (mentioned alongside images in the original §11.6
  note, but out of scope here — different content-type handling, revisit
  separately).
- No attempt to give Ollama or Foundry backends image support in this pass (see
  §21.3, Decision 1).

## 21.3 Decisions already made

1. **oMLX-only.** Matches the docs' existing framing ("oMLX and Gemma 4B
   natively support OCR, image, and PDF") and reuses the one piece of prior art
   that already exists, `OMLXRuntimeClient.infer_with_file()`. When Ollama or
   Foundry is the active runtime (resolved fresh from `_state.runtime` per the
   live-swap rule in §16), image attachment is disabled/rejected rather than
   silently dropped or attempted. Ollama and Foundry are not extended to carry
   images in this scope.
2. **Images flow through `PromptBuilder`, not around it.** The existing
   `infer_with_file()` bypasses `PromptBuilder` entirely (used only by
   `WikiAgent`'s ingest path) — full session/episodic/RAG/persona context never
   gets built for that call. For chat, images ride *alongside* the normal
   prompt-assembly pipeline so the model has full context in the same turn,
   not just the image and a bare instruction.
3. **Separate image cache, not a widened `session_files.py`.** Text session
   files (`session_files.py`) are an `OrderedDict[str, tuple[content: str,
   source: str]]` with a token-based budget (4k/file, 20k total). Images need
   byte-based sizing and binary storage — a parallel cache (`session_images.py`
   or equivalent) keeps that existing, well-tested module's contract untouched.

## 21.4 Current-state findings this scope is grounded in

- `POST /chat/files` (`backend/main.py:2209-2235`) hard-decodes uploads as
  UTF-8 (`main.py:2221`) — any binary file 422s today. This, not the extension
  allowlist, is the real blocker.
- `ChatPanel.svelte`'s client-side `ALLOWED_EXTENSIONS` (lines 58-63) has no
  image types; server-side enforcement lives in `session_files.py`'s own
  allowlist (lines 45-50) — the actual gate, the frontend list is decorative.
- `session_files.py` cache is text-only, in-memory, ephemeral (no disk write,
  no RAG indexing, no MCP `file_op` involvement) — the model for the new image
  cache to mirror structurally, not share.
- `backend/mcp_server/file_ops.py`'s `read_file`/`write_file`/`append_file` are
  unrelated — text-only (`read_text`/`write_text`), sandboxed to
  `generated_files/` under `LOCALIST_MCP_PROJECT_ROOT`, used for agent-initiated
  writes, not user uploads. Not part of this feature.
- `PromptBuilder.build()` (`backend/prompt_builder.py:808+`) returns a flat
  `(system_prompt: str, user_prompt: str)` pair; `BaseRuntimeClient.infer()`/
  `infer_stream()` (`backend/base_runtime_client.py:89-208`) are string-only —
  no content-block concept anywhere in the Protocol.
- `OMLXRuntimeClient.infer_with_file()` (`backend/omlx_runtime_client.py:
  551-696`) is the only existing multimodal call: base64-encodes a file,
  guesses MIME type, sends an oMLX content-block array
  (`{"type": "file", "file": {...}}` + `{"type": "text", "text": prompt}`).
  Currently called only from `WikiAgent._run_ingest()`
  (`backend/wiki_agent.py:1272-1291`). No streaming variant exists.
- Gemma 4 naming: oMLX's default chat model is `gemma-4-e4b-it-4bit`
  (`backend/runtime_factory.py:90`). The "natively supports OCR" claim is
  already asserted (not yet load-bearing) in §11.6 and §3 (Slot SF note).

## 21.5 Phased outline

**Phase 1 — Storage: `session_images.py`**
New module parallel to `session_files.py`. `OrderedDict[filename, (bytes,
mime_type)]`, byte-size ceiling (per-image + total), independent of the
existing 20k-token text budget. No disk write, no RAG indexing.

**Phase 2 — Backend upload routing**
Extend `POST /chat/files` (or add a sibling endpoint) to branch on
MIME/extension before the UTF-8 decode: image extensions route to
`session_images.add_image()`; everything else keeps today's text path
unchanged.

**Phase 3 — `PromptBuilder` integration**
Add an image-aware marker to the text prompt (mirroring how text session files
render today, e.g. `--- attached image: filename.png ---`) so the model's
textual context reflects what's attached. Widen `PromptBuilder.build()`'s
return shape to also carry the raw image bytes/mime list out-of-band (e.g. an
`images: list[ImageAttachment]` field) for `ConversationalAgent` to pass to the
runtime. All other slot assembly (memory, RAG, persona, working state)
proceeds unchanged.

**Phase 4 — `ConversationalAgent` + `OMLXRuntimeClient` routing**
When `PromptBuilder` returns non-empty `images` and the active runtime
(resolved fresh, not cached) is `OMLXRuntimeClient`: call a new
`infer_with_images(prompt, system, images, ...)` that generalizes
`infer_with_file()`'s content-block format to the fully-assembled prompt plus
1+ images. Needs a streaming variant if chat responses must stream while an
image is attached (open item, §21.6). When images are attached but a
non-oMLX runtime is active, reject clearly rather than silently dropping the
image.

**Phase 5 — Frontend**
`ChatPanel.svelte`: add image extensions to `ALLOWED_EXTENSIONS` and the
`accept` attribute; attached-file chips get an image/thumbnail variant. Gate
the attach affordance on the currently active runtime backend (read from the
existing runtime-backend settings store) — disable when Ollama/Foundry is
active, with an explanatory tooltip, so the Phase 4 rejection is a backstop,
not the primary UX.

**Phase 6 — Tests**
Mock-based, following the existing suite's pattern (no live oMLX required):
`session_images.py` eviction/ceiling logic; upload-routing branch (image vs.
text vs. rejected-binary); `PromptBuilder` image-slot + `images` passthrough;
`ConversationalAgent`'s runtime-gating decision (oMLX-with-images vs.
non-oMLX-with-images-rejected).

**Phase 7 — Docs**
Update §11.6 (remove "Not scheduled," describe the shipped shape) and the Slot
SF note in §3; add any new env vars (max image bytes, allowed image MIME
types) to `backend/.env.example`; retire this file's Draft status to
Authoritative (or fold its content into §11) once shipped.

## 21.6 Open items to resolve before Phase 1 starts

1. **Per-turn resend cost.** Text session files are re-injected into every
   subsequent turn's prompt for the session's duration. Doing the same for a
   base64-encoded image on every turn is a real payload/latency cost and works
   against `PromptBuilder`'s KV-cache-reuse design principle (§3). Undecided:
   resend every turn (consistent with text files, but costly) vs. one-shot
   (image only enters the prompt on the turn it's uploaded/referenced, then
   evicted from future turns). Leaning one-shot, not decided.
2. **Streaming.** `infer_with_file()` has no streaming variant today. Decide
   whether image-attached turns must stream (needs new work in
   `OMLXRuntimeClient`) or can fall back to non-streaming for that turn only.

## 21.7 Prior art / cross-references

- §11 (`docs/architecture/11-session-file-attachments.md`) §11.6 Open Item 4 —
  the original (undersized) framing of this gap.
- §3 (`docs/architecture/03-unified-prompt-contract.md`) — Slot SF note,
  same "deferred" framing.
- §16 (`docs/architecture/16-runtime-backend-layer.md`) — live runtime-backend
  switching and the "resolve fresh, never cache" rule this scope's Phase 4/5
  gating depends on.
- `WikiAgent._run_ingest()` (`backend/wiki_agent.py:1272-1291`) — the only
  existing caller of `infer_with_file()`, the closest prior art for the wire
  format.
