## 16. Runtime Backend Layer (BaseRuntimeClient)

### 16.1 Available Backends

`base_runtime_client.py` defines the `BaseRuntimeClient` Protocol (`infer`, `embed`,
`infer_stream`) that every concrete runtime client satisfies structurally, without inheritance.
`runtime_factory.py`'s `create_runtime()` is the single entry point that constructs the active
backend at process startup, selected via `LOCALIST_RUNTIME_BACKEND`. Concurrency posture is
swap-only: exactly one backend is active per process; dual-runtime/concurrent-backend operation is
explicitly out of scope for all three backends below.

Three backends are registered in `_REGISTRY` today:

- **`FoundryRuntimeClient`** (`foundry_runtime_client.py`) — Azure AI Foundry, local execution,
  ephemeral-port resolution via `foundry service status`.
- **`OMLXRuntimeClient`** (`omlx_runtime_client.py`) — oMLX local inference, fixed port 8000,
  OpenAI-compatible SSE streaming, native MarkItDown file ingestion (`infer_with_file()`, an
  oMLX-only capability not part of the Protocol).
- **`OllamaRuntimeClient`** (`ollama_runtime_client.py`, added 2026-07-08) — Ollama local
  inference, fixed port 11434, native NDJSON streaming (not OpenAI-compatible SSE). See §16.2.

### 16.2 OllamaRuntimeClient — Added 2026-07-08

> **Superseded in part (2026-07-14) — see §16.4.** `embed()` is no longer a permanent stub (it now
> implements `POST /api/embed`, with a real "not yet configured" `embedding_model` parameter
> mirroring oMLX's convention) and the "chat model left as an unfilled constructor default" framing
> below was superseded first by a hardcoded `gemma4:e4b-mlx` default (2026-07-08, later in this
> section) and then by a fail-fast `ValueError` (2026-07-14). The scoping narrative below is
> retained as a historical record of what was decided and why at the time.

**Scoping decisions (made before any code was written).** Full interchangeable-backend tier, same
as Foundry/oMLX — not an offload-style secondary backend as sketched in earlier dual-runtime
planning (the Apfel spec). Concurrency: swap-only, no dual-runtime logic added. `embed()`: stubbed
to always raise `NotImplementedError`, not left as a "not configured yet" placeholder the way
oMLX's empty-`embedding_model` case is — this backend has no path to an embedding model at all
under this scope, so its constructor doesn't even accept an `embedding_model` parameter. Chat
model left as an unfilled constructor default pending live confirmation rather than guessed up
front.

**Live verification before writing the Claude Code prompt.** Base URL confirmed as Ollama's actual
default (`http://localhost:11434` — no port conflict with 8000/8001/8003/5173). `GET /api/tags`
response shape confirmed via live `curl`: `{"models": [{"model": "...", "name": "...", ...}]}` — a
`models`/`model` shape, not oMLX's `{"data": [{"id": "..."}]}`. `POST /api/chat` streaming
confirmed via live `curl` to be NDJSON (one raw JSON object per line, terminated on `"done":
true`), not an SSE `data: {...}` envelope — meaning `foundry_runtime_client.py`'s
`_iter_sse_chunks` could not be reused; `ollama_runtime_client.py` implements its own
`_iter_ndjson_chunks`.

**Model selection — a live correction, not an assumption.** Initial framing considered Gemma 3
variants (the assistant's most recent reliable training knowledge). The user corrected this:
Ollama's model library now lists native Gemma 4 MLX-format models, and `gemma4:e4b-mlx` was chosen
specifically to match the oMLX baseline's `gemma-4-e4b-it-4bit` architecture class (E4B
effective-parameter size, 4-bit-class quantization) — not picked arbitrarily. This surfaced that
both Gemma 4 and Ollama's native MLX backend support post-date the assistant's reliable training
data; the live-verification step above, rather than the assistant's own knowledge of what Ollama
supports, is why the final choice held up.

**Implementation.** New file `backend/ollama_runtime_client.py`, structurally mirroring
`omlx_runtime_client.py` (docstring conventions, method ordering, `__repr__`,
`_assert_protocol_conformance()`). One flagged deviation from the `_make_omlx` registration
pattern: `_make_ollama` (`runtime_factory.py`) does not pass an `embedding_model` kwarg at all,
since `OllamaRuntimeClient.__init__` has no such parameter — `embed()` here is a permanent stub,
not an unconfigured state, so there is nothing to configure.

**Two mount-staleness false alarms surfaced and resolved during this work, not implementation
defects:**

- The assistant initially flagged the `label: str = ""` parameter (present in
  `BaseRuntimeClient`, `FoundryRuntimeClient`, and `OMLXRuntimeClient`, and carried into
  `OllamaRuntimeClient`) as a possibly-hallucinated, unverified claim — based on a stale
  project-knowledge mount that didn't contain it. A live `grep` against the actual repo confirmed
  `label` is genuinely present in all three; the original implementation report was correct
  throughout, and the mount, not the code, was stale.
- Separately, the assistant's mount showed `env_prefix="LORA_"` in `main.py`; a live `grep`
  confirmed the actual prefix is `env_prefix="LOCALIST_"`. Caught and corrected before any `.env`
  edit was made.

**Full live-verification chain, in order:** `GET /api/tags` reachable → backend startup with
`LOCALIST_RUNTIME_BACKEND=ollama` (clean init log chain, no errors) → health-check equivalent
confirmed via startup log (`chat_found=True embed_found=None`) → `POST /task` non-streaming, real
content, correct answer ("7 times 6 is 42") → `POST /task/stream` SSE relay, correct
token-by-token streaming, clean `[DONE]` termination. One call returning empty content
(`warmup.py`'s cache-warm call, `max_tokens=16`) was investigated and confirmed expected —
`run_cache_warmup()` deliberately discards completion content by design (KV-prefill exercise only),
not a bug in the new client.

**Test suite:** 595 tests collected/passed before this change, 595 passed after (file-scoped count
via a direct `.venv` pytest run, not carried forward from an earlier baseline). No new dedicated
test file was in scope for this addition.

**Open items — both closed 2026-07-08. OllamaRuntimeClient has no known open items as of this
update.**

**1. `LOCALIST_CHAT_MODEL` coupling. CLOSED 2026-07-08.**

*Originally:* `main.py`'s `Settings.chat_model` defaulted to the Foundry-specific string
`"Phi-4-mini-instruct-generic-gpu:5"` and was always passed explicitly to `create_runtime()`, so
`runtime_factory.py`'s per-backend `kwargs.get("chat_model", default)` fallback never triggered —
the key was never absent.

*Fix:* `Settings.chat_model` changed to `str | None = None`; `runtime_factory.py`'s
`_make_foundry`/`_make_omlx`/`_make_ollama` changed from `kwargs.get("chat_model", default)` to
`kwargs.get("chat_model") or default`, so `None`/empty string now correctly falls through to each
backend's own default.

*Live-verified:* with `LOCALIST_CHAT_MODEL` unset and `LOCALIST_RUNTIME_BACKEND=ollama`, the
constructed `OllamaRuntimeClient` reports `chat_model="gemma4:e4b-mlx"` (not the Foundry string,
not `None`). Foundry/oMLX fallback and explicit-override passthrough also spot-checked with no
regression.

*Test suite:* 595 passed (before and after).

**2. Missing `LOCALIST_OLLAMA_URL`. CLOSED 2026-07-08.**

*Originally:* `Settings` had `foundry_url` and `omlx_url` fields but no `ollama_url` field;
`create_runtime()` had no `ollama_url` kwarg wired in for the `ollama` backend.

*Fix:* added `ollama_url: str = "http://localhost:11434"` to `Settings`, documented
`LOCALIST_OLLAMA_URL` in the module docstring's environment variable list, and passed
`ollama_url=settings.ollama_url` into the `create_runtime()` call.

*Live-verified:* with `runtime_backend="ollama"` and `ollama_url` overridden to
`http://localhost:19999`, `health_check()` returns `base_url: 'http://localhost:19999'` — the
override, not the hardcoded `11434` default — confirmed even with nothing listening on that port,
since `health_check()` never raises.

*Test suite:* 595 passed (before and after).

### 16.3 Proposed — Live-Switchable Runtime Backend from Settings (scoped, not built, 2026-07-13)

> **Built and live-verified 2026-07-15 — see §16.5.** The proposal below was implemented
> essentially as scoped, with both open decisions resolved (persist to `.env`, not session-scoped;
> independent switch/pin endpoints, not folded together) and point 6's concurrency claim now
> live-verified rather than assumed. The narrative below is retained as the historical record of
> what was scoped and why; §16.5 is the current-state record of what shipped, and §16.6 closes the
> two follow-on items code review found there (the frontend now wired, the lock now `asyncio.Lock`).

**Status: proposal only. No code has been changed for this item.** The Settings page's Runtime
Backend segmented control (§7.10) is a display preference — selecting oMLX/Ollama/Foundry there
updates only the appbar's status-chip label, not the actual active runtime. This subsection records
a scoping pass done at the user's explicit request ("don't perform any edits, just scope out how
many steps it would take"). Full narrative: `sessions-log.md` §34.

**Why this is a real rebuild-and-replace, not a one-line state mutation.** At startup,
`main.py`'s `lifespan()` constructs one `runtime` object via `create_runtime()` and passes it **by
value** into the constructors of `WikiAgent`, `ConversationalAgent`, and `ControllerAgent`
(`self._runtime = runtime`). `ControllerAgent.__init__` further constructs its own
`Synthesizer(runtime)` and `_RulePlanner(runtime=runtime, ...)` from that same captured reference.
None of these look the runtime back up dynamically. The sole exception is `GET /health`, which reads
`_state.runtime` fresh via `_require_runtime()` on every call. So mutating `_state.runtime` alone
post-startup would make `/health` report a new backend while every actual chat turn kept running
inference through the *original* client.

**Proposed implementation shape:**
1. Extract the agent-construction block currently inline in `lifespan()` into a reusable function
   (e.g. `_build_controller(settings, runtime, memory_manager, embed_fn)`).
2. New `POST /settings/runtime-backend`: validate the target backend name; call `create_runtime()`
   for it; call `.health_check()` on the *new* client before committing anything, so an unreachable
   target fails cleanly and leaves the current backend running.
3. Rebuild `WikiAgent`/`ConversationalAgent`/`ControllerAgent` via the function from step 1 (which
   transparently rebuilds `Synthesizer`/`_RulePlanner` too, since they're constructed inside
   `ControllerAgent.__init__`).
4. Re-run `_run_cache_warmup` against the new controller/runtime, since persona caching
   (`ControllerAgent._load_persona()`) is per-instance and would otherwise start cold.
5. Atomically swap `_state.runtime`/`_state.wiki_agent`/`_state.controller` only after steps 3–4
   succeed, under a lock (so two rapid switch requests can't interleave).
6. In-flight requests are expected to be safe by construction — each request resolves
   `_state.controller` once at request time, so an active stream should finish on the pre-swap
   controller rather than crashing. **Live-verified 2026-07-15 (§16.5):** a switch fired mid-stream
   let the in-flight request complete its entire lifecycle on the pre-switch client, unaffected by
   the concurrent rebuild — no hang, no cross-contamination.

**The `chat_model`-per-backend interaction with §16.2's existing fix.** §16.2 already fixed the
*unset* case (`Settings.chat_model = None` correctly falls through to each backend's own hardcoded
default). This proposal is about the *explicit-override* case: a `LOCALIST_CHAT_MODEL` value set
for one backend would still get handed unchanged to whichever backend is live-switched to next
(e.g. a Foundry model id passed into `OllamaRuntimeClient`). Proposed fix: per-backend chat-model
storage (three settings fields or a dict) plus a new small `GET
/settings/runtime-backend/{backend}/models` endpoint — builds a throwaway client for that backend,
calls `.health_check()`, returns its `models` list, discards the client without touching `_state` —
letting the Settings UI's "Chat model ID" free-text field become a dropdown of models actually
available on the target backend instead of a field a user can mistype.

**A separate finding from the same scoping conversation, already acted on.** While tracing whether
Ollama could serve as an alternative embedding backend (unrelated to the switch proposal itself, but
raised in the same conversation), it became clear the Settings page's "Embedding model ID" field is
fully inert today — see `sessions-log.md` §33 for the finding and the fix (a `requirements.txt`
dependency gap, unrelated to runtime-backend switching). Recorded here only because it materially
affects what "fold Model Configuration into backend-switch config" should mean: that field should
likely be retired rather than made per-backend.

**Open decisions blocking a build session:**
- Does a live switch persist across a process restart (would need writing back to `.env` or a small
  runtime-config store), or is it session-scoped only?
- Accept the per-backend hardcoded chat-model fallback always, or let users pin a model per backend
  via the proposed dropdown?

**If built, would also require:** dedicated unit tests for the new endpoint(s) (unknown backend
rejected; unreachable backend rejected without side effects; successful swap updates `_state`);
updating `CLAUDE.md`'s current "Swapping backends is a config change only" line; updating this
section; updating the README's Configuration section, which currently and correctly describes the
control as display-only.

### 16.4 Embedding Support Across Backends, Chat-Model Fail-Fast, and Platform-Gated MLX
Fallback — Added 2026-07-14

`OllamaRuntimeClient.embed()` is no longer a permanent stub. It POSTs to Ollama's native
`/api/embed` endpoint (distinct from the OpenAI-compatible `/v1/embeddings` used internally by
`FoundryRuntimeClient` and `OMLXRuntimeClient` — Ollama's own API returns a plural `embeddings`
list rather than the singular `data[0].embedding` shape). The client accepts an `embedding_model`
constructor argument, defaulting to `""` ("not yet configured" — same convention as
`OMLXRuntimeClient`, not "this backend can never do this").

**Chat model configuration is now mandatory, not defaulted.** `OllamaRuntimeClient.DEFAULT_CHAT_MODEL`
was previously `"gemma4:e4b-mlx"` — a silent 8.8GB local-model fallback whenever `LOCALIST_CHAT_MODEL`
was unset. This has been removed. The constructor now raises `ValueError` immediately if
`chat_model` is falsy, failing at construction time (and therefore at FastAPI startup, surfacing as
`Application startup failed. Exiting.` with a clear traceback) rather than producing a confusing
failure on the first chat request. `runtime_factory._make_ollama()` was updated to match. This is
deliberate: a chat application should never start without a chat model configured, and Ollama in
particular spans model sizes from tiny local quantizations to 700B-parameter cloud models — there
is no safe default to assume. Live-verified 2026-07-14: unsetting `LOCALIST_CHAT_MODEL` produces a
clean startup crash with the expected `ValueError` message; restoring it produces normal startup.

**Embedding source precedence (all backends):**

At startup, `lifespan()` in `main.py` selects an embedding source via
`_configure_embedding_source(settings, runtime, health)` (extracted 2026-07-14 into a standalone,
unit-testable function — see Testing note below) in this order:

1. **Runtime-backend embed** — if `Settings.embedding_model` (`LOCALIST_EMBEDDING_MODEL`) is set
   and the active runtime's `health_check()` reports the model present, `MemoryManager.embed_fn`
   is bound to `runtime.embed`. This is wired through for the Foundry and Ollama backends and is
   platform-agnostic for both. **Not yet wired for the oMLX backend** —
   `runtime_factory._make_omlx()` hardcodes `embedding_model=""` regardless of
   `Settings.embedding_model`, so this tier can never actually engage while `runtime_backend="omlx"`
   is active, even though `OMLXRuntimeClient.embed()` itself supports a configurable
   `embedding_model` when constructed directly. Closing this gap is a follow-up, not part of this
   change.
2. **`EmbeddingEngine` (MLX-LM, standalone)** — used only when tier 1 isn't active, AND only on
   Apple Silicon. Gated behind
   `is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")`,
   added 2026-07-14, since `mlx_lm` cannot run on Intel Mac, Windows, or Linux. On non-Apple-Silicon
   hosts with this tier otherwise eligible, the load attempt is skipped entirely (no
   `EmbeddingEngine()` construction, no import attempt) and an INFO-level log line fires — distinct
   from the WARNING-level "failed to load" message used when an actual load attempt fails, so the
   two situations ("this platform can't run this by design" vs. "this platform can but something
   went wrong") are distinguishable in logs.
3. **Keyword-only** — universal fallback when neither of the above is available.

`GET /health`'s `embed_model_found` field mirrors this same precedence (fixed 2026-07-14 —
previously hardcoded to check `EmbeddingEngine` only, which misreported `false` whenever tier 1 was
actually active).

**Testing note:** the project's test suite convention (established in
`test_main_memory_episodes.py`) is to never trigger the real FastAPI `lifespan()` in tests — tests
swap `AppState` fields directly instead. The four-branch embedding-source selection was therefore
extracted out of `lifespan()`'s inline body into `_configure_embedding_source()` specifically to
make it unit-testable without violating that convention. `tests/test_main_embedding_source_selection.py`
covers all four branches, including the Apple-Silicon skip branch under mocked
`platform.system()`/`platform.machine()` calls (Linux/x86_64, Windows/AMD64) — this is
mocked/forced-condition testing, not real cross-platform execution; no non-Darwin hardware was
available to verify the skip branch by actually running on it.

**Verified configuration (2026-07-14, fully live end-to-end):** cloud chat model
(`gemma4:31b-cloud`, proxied through ollama.com — never resident locally) + local embedding model
(`nomic-embed-text:latest`, served on-device via `/api/embed`), both routed through the same local
Ollama daemon at `localhost:11434`. A real turn produced ~30 real embed calls (768-dim vectors)
plus a completed chat response and working-state extraction, all logged and confirmed via
`GET /health`. This is the reference configuration for Localist PC (non-Apple-Silicon) users, since
MLX/`EmbeddingEngine` is unavailable on that hardware entirely — Ollama's `/api/embed` is the only
local embedding path on Windows/Linux, and the platform gate above ensures those hosts never
attempt (and fail) an MLX load. One tag-mismatch false start along the way (`.env` needed the
`:latest` suffix Ollama's `/api/tags` actually reports) — corrected, not a code defect. Full
narrative: `sessions-log.md` §24 (2026-07-14 entry).

**Test suite:** 728 passed / 0 failed after `embed()` + wiring + `/health` fix; 732 passed / 0
failed after the chat-model fail-fast change (+4 new tests); 739 passed / 0 failed after the
platform-gating change (+7 new tests).

**Open items:**
- ~~Corpus-retrieval similarity thresholds may need to become embedding-model-aware~~ — **confirmed,
  2026-07-16.** The low top-score corpus miss (`0.028` under `nomic-embed-text`) noted below was the
  first hint; live testing the same day made it unambiguous: the identical `lookup_request`
  utterance scored `0.7119` under `mlx-community/embeddinggemma-300m-4bit` (the model
  `planner.py`'s semantic-gating thresholds were tuned against) vs `0.578` under `nomic-embed-text`
  — a swing large enough to flip `gate_fired` from `True` to `False` for the same input. Two more
  diagnostics required assuming a fixed embedding model to be meaningful at all:
  `diagnostics/reports/research_intent_threshold_assessment_2026-07-16*.md` and
  `diagnostics/reports/negative_filter_tiebreak_assessment_2026-07-16.md`. Cosine similarity is not
  portable across embedding models' geometries; a threshold tuned on one model's score distribution
  has no guaranteed meaning on another's.

  **Fixed on the Planner side (2026-07-16).** `planner.py` now declares `_TUNED_EMBEDDING_MODEL =
  "mlx-community/embeddinggemma-300m-4bit"`. `Planner.__init__` takes an `embedding_model_name`
  parameter; when it's set and doesn't match `_TUNED_EMBEDDING_MODEL`, semantic search-intent gating
  (`_semantic_search_intent()`, backing `explicit_search_action` / `lookup_request` /
  `research_intent`) is disabled for that Planner instance — a startup-time guard rather than a
  silent runtime degradation, with a `logger.warning` naming both models. `main.py` derives this name
  via the new `_derive_active_embedding_model_name()` (mirroring `_configure_embedding_source()`'s
  own three-tier precedence), stores it on `_state.active_embedding_model_name`, and threads it
  through `_build_controller()` into `ControllerAgent` at every construction site — startup
  (`lifespan()`) and both live-switch endpoints (`/settings/runtime-backend`,
  `/settings/runtime-backend/{backend}/chat-model`), read fresh from `_state` at request time rather
  than captured once. `embedding_model_name=None` (keyword-only mode, no embedding source
  configured) is not treated as a mismatch — `embed_fn` is already `None` in that case, which already
  short-circuits semantic scoring.

  **Fixed on the MemoryManager side too (2026-07-16).** A new `embedding_provenance` table
  (`store TEXT PRIMARY KEY` — `'corpus'` | `'episodes'` — `model TEXT NOT NULL`, schema v8) records
  which embedding model actually produced the vectors currently stored in `document_index` and
  `episodes`. `MemoryManager.__init__` gains `embedding_model_name`, threaded from `main.py`'s
  `_state.active_embedding_model_name` (the same value `ControllerAgent`/`Planner` now receive), and
  runs `_check_embedding_provenance()` once at construction time. Per store, per the recorded row:

  - No row, no embedded data yet → nothing to compare against — deferred; `'corpus'` seeds its own
    row the first time `index_document()` actually writes an embedding
    (`_maybe_seed_corpus_provenance()`).
  - No row, but embedded data already present → the pre-existing-database migration case (provenance
    tracking shipped after data was already embedded under whatever model was active at the time).
    Seeded silently, no warning, no re-embed — treating this as a mismatch would have triggered a
    surprise re-embed for every existing user on their very next boot.
  - Row present and it matches → no-op.
  - Row present and it disagrees → a genuine mismatch, handled per the decided split:
    - **`'corpus'`** (wiki/raw documents): potentially large/expensive to re-embed, so never
      automatic. `logger.warning`, `self._corpus_stale = True`, and an immediate retrieval-cache
      flush (stale rankings from the old model are wrong the instant the mismatch is detected, even
      before any re-embed happens). `query_corpus()` checks `not self._corpus_stale` alongside its
      existing `self._embed_fn is not None` check before taking the embedding re-rank path — same
      fail-safe-to-keyword-only posture as no `embed_fn` at all. Cleared only by the new
      `MemoryManager.reembed_corpus()` (exposed as `POST /memory/reembed`), which re-embeds every
      `document_index` row, flushes the cache, updates the provenance row, and clears
      `_corpus_stale` — idempotent, callable regardless of whether the corpus is currently flagged
      stale.
    - **`'episodes'`**: small and bounded, so auto-corrected in place before `__init__` returns —
      every `episodes` row is re-embedded via the active `embed_fn` (`"{subject}. {content}"`, the
      same convention `EpisodicMemoryWriter.insert()` uses) and the provenance row is advanced to the
      new model. Provenance only advances if every row succeeded; a partial failure leaves the old
      provenance value in place so the mismatch is detected and retried on the next boot rather than
      claiming a clean re-embed that didn't fully happen. A row-count tripwire
      (`_EPISODES_REEMBED_WARN_ROW_COUNT`) logs a warning if this ever needs revisiting at scale, but
      re-embedding runs regardless.

  This is the same detect-and-fail-safe pattern as the Planner-side fix above — a tuned-model
  comparison at construction time, disabling (not silently degrading) on a mismatch — applied to
  stored vectors instead of threshold constants, split into an automatic path where the fix is cheap
  and a manual path where it isn't.
- Apple-Silicon skip branch remains verified only via mocked `platform` calls; real non-Darwin
  hardware verification is still outstanding, pending access to such a machine.
- oMLX's `embedding_model` wiring gap noted above (tier 1 never engages for that backend) — a
  follow-up, not yet scheduled.

### 16.5 Live-Switchable Runtime Backend — Built and Live-Verified, 2026-07-15

§16.3's proposal shipped as scoped. Implemented entirely in `backend/main.py`;
`runtime_factory.py`'s existing `create_runtime()`/`available_backends()` required no changes.
Full narrative, including the step-by-step live-verification checklist: `sessions-log.md` §36.

**What was built.** `_build_controller(settings, runtime, memory_manager, embed_fn, project_root,
templates_dir)` — the WikiAgent/ConversationalAgent/ControllerAgent construction + cache-warmup
block, extracted out of `lifespan()` so both startup and a live switch share one code path.
`_resolve_chat_model(settings, backend)` — single precedence rule: `Settings.chat_model` (global
override) beats a new per-backend pin (`chat_model_omlx`/`chat_model_ollama`/`chat_model_foundry`,
`LOCALIST_CHAT_MODEL_OMLX`/`_OLLAMA`/`_FOUNDRY`) beats `None` (falls through to
`runtime_factory.py`'s existing hardcoded per-backend default). `_write_env_var(project_root, key,
value)` — atomic (`tempfile` + `os.replace()`) read-modify-write against `.env`, preserving
comments/blank lines/unrelated keys, skipping commented-out lines when matching the target key.
`_runtime_switch_lock` (`threading.Lock`) serializes any sequence that reads-then-swaps
`_state.runtime`/`.wiki_agent`/`.controller`.

Three new endpoints:
- `POST /settings/runtime-backend` `{backend, chat_model?}` — validates the backend name;
  health-checks the candidate client before mutating anything; on success, rebuilds via
  `_build_controller()`, atomically swaps `_state`, then persists `LOCALIST_RUNTIME_BACKEND` to
  `.env`. An optional `chat_model` on the request also pins that backend (not a one-shot override —
  see the open call noted in the original Claude Code prompt, resolved this way to keep one source
  of truth). If the in-memory swap succeeds but the `.env` write itself fails, the swap is **not**
  rolled back — the response returns `persisted: false` plus a warning instead.
- `GET /settings/runtime-backend/{backend}/models` — builds a throwaway client for `backend`,
  health-checks it, returns its reported models, discards the client. Never touches `_state`.
- `POST /settings/runtime-backend/{backend}/chat-model` `{chat_model}` — always persists the pin
  (Settings field + `.env`) regardless of which backend is active; additionally triggers a live
  rebuild via the same lock/health-check/`_build_controller()` sequence if — and only if —
  `backend` is the currently active one.

**Test suite:** 757 passed / 0 failed (739 baseline + 18 new,
`backend/tests/test_main_runtime_backend_switch.py`), covering unknown-backend rejection,
unreachable-target rejection without side effects, successful-switch state/`.env` assertions, both
chat-model-pin paths (active vs. inactive backend), and `_write_env_var()`/`_resolve_chat_model()`
directly.

**Live verification (2026-07-15, full 9-step checklist against real oMLX/Ollama/Foundry
services):** every mechanic in the proposal confirmed working end to end — health-check-before-swap,
zero-side-effect read-only model listing, `.env` persistence surviving an actual process restart,
clean rejection of an unreachable target with the prior backend provably untouched, and (closing
§16.3 point 6's open item) an in-flight streaming request completing unaffected by a concurrent
switch. Full step-by-step results, including three findings that turned out to be pre-existing
design decisions or environmental constraints rather than defects (the Ollama fail-fast/pin
interaction, oMLX's memory ceiling, and `chmod`'s inability to trigger the `persisted: false` path
via POSIX rename semantics) are in `sessions-log.md` §36.

**Open items:**
- **`embed_fn` staleness across a switch — confirmed already-correct behavior, now hardened.**
  A follow-up code review found that neither `switch_runtime_backend()` nor
  `set_runtime_backend_chat_model()` ever re-derives or reassigns the embedding function from the
  candidate/new runtime: both read it once from `MemoryManager` and thread it through unchanged
  into `_build_controller()`. So a chat-backend switch already never touched the embedding source —
  this was not the bug the original finding above described, just an implicit consequence of how
  the code happened to be written. Hardened rather than fixed: `MemoryManager` gained a read-only
  public `embed_fn` property (replacing a private-attribute reach-through via
  `getattr(memory_manager, "_embed_fn", None)` at both call sites), both call sites gained an inline
  comment stating the decoupling is deliberate, and
  `backend/tests/test_main_runtime_backend_switch.py` gained two identity-check tests
  (`test_successful_switch_leaves_embed_fn_untouched`,
  `test_pin_for_active_backend_leaves_embed_fn_untouched`) asserting `embed_fn` is the exact same
  object before and after a live switch/pin. This guarantee is now explicit and test-covered instead
  of an accidental side effect a future edit could casually "fix" into a coupling bug. See
  `sessions-log.md` §37.
- ~~**Lock held across `await`.**~~ **Closed in §16.6** — `_runtime_switch_lock` is now an
  `asyncio.Lock`, acquired via `async with`.
- The `persisted: false` `.env`-write-failure path needs a different provocation technique for any
  future repeatable verification — a read-only *directory* (not a read-only file) would trigger it
  per POSIX rename semantics; not yet tried.
- ~~The Settings UI (`localist-ui`, §7.10) is not wired to any of these three endpoints.~~
  **Closed in §16.6** — the Runtime Backend segmented control, Chat Model dropdown, and per-backend
  model preview are now wired to all three endpoints.

### 16.6 Settings UI Wired to the Live Switch, and the Lock-Across-`await` Item Closed (2026-07-15)

Two remaining §16.5 open items closed in one pass — full narrative in `sessions-log.md` §38.

**Lock fix.** `_runtime_switch_lock` (`backend/main.py`) changed from `threading.Lock()` to
`asyncio.Lock()`, and both call sites from `with _runtime_switch_lock:` to
`async with _runtime_switch_lock:`. The prior synchronous lock, held across
`await asyncio.to_thread(...)`, didn't just serialize two overlapping switch/pin requests — it
busy-blocked the *entire event loop* while the second waited to acquire, since a plain
`threading.Lock.acquire()` has no cooperative-yield path back to the loop. Confirmed by hand: with
the lock reverted to `threading.Lock`, a new regression test
(`TestConcurrentSwitchRequestsDoNotBlockEventLoop` in
`backend/tests/test_main_runtime_backend_switch.py`) deadlocks outright rather than merely running
slow — the loop can never process the `asyncio.to_thread` completion callback that would let the
lock-holding coroutine resume and release, so nothing ever unblocks. That test now runs the
switch/heartbeat coroutines inside a daemon thread with an external `join(timeout=...)`, since an
in-process `asyncio` timeout can't fire either once the loop itself is frozen. With the
`asyncio.Lock` fix, the same test passes: two concurrent `switch_runtime_backend()` calls both
complete, and a concurrent heartbeat coroutine keeps ticking throughout (proving the loop stayed
responsive) rather than stalling for the ~150ms one request holds the lock.

**Backend prerequisite for the frontend.** `HealthResponse` (`backend/main.py`) gained a `backend:
str` field, sourced from `settings.runtime_backend` in `get_health()` — the one authoritative
signal the frontend needs to know which of `omlx`/`ollama`/`foundry` is actually live; `base_url`
alone can't disambiguate if two backends happen to share a host/port shape.

**Frontend — `localist-ui`.**
- `server.ts`'s `HealthState` gained the matching `backend: string` field (default `''`); no other
  change needed since `checkHealth()` already spreads the full response into the store.
- `model.ts`: `readStoredBackend()`'s `localStorage` value is now only a paint-before-first-health-
  check cache, not the record of truth. A new `health.subscribe(...)` sets `modelConfig.backend =
  $health.backend` whenever `$health.reachable` is true and `$health.backend` is non-empty — so a
  process restart (which reverts to `.env`) or an out-of-band switch (e.g. via `curl`) is reflected
  once the next health poll lands, not just whatever the browser last remembered.
- New `runtimeBackendSwitch.ts` store (follows `chatHistorySettings.ts`'s pattern — loading/error
  writables, no optimistic update on failure): `switchRuntimeBackend(backend, chatModel?)` →
  `POST /api/settings/runtime-backend`; `fetchBackendModels(backend)` →
  `GET /api/settings/runtime-backend/{backend}/models` (returns `[]` on failure rather than
  throwing); `pinChatModel(backend, chatModel)` → `POST /api/settings/runtime-backend/{backend}/chat-model`.
- `routes/settings/+page.svelte`: the Runtime Backend segmented control now calls
  `switchRuntimeBackend()` for real, gated by a plain `confirm()` dialog naming the target backend
  and noting in-flight requests are unaffected but the switch itself takes a few seconds; a no-op
  guard skips the confirm/switch entirely when the clicked backend is already active. The control
  disables and shows a spinner while `runtimeBackendSwitchLoading` is true. A new `selectedUiBackend`
  local variable — deliberately independent of the real active backend — lets a segmented-control
  click preview a *different* backend's models (via `fetchBackendModels()`, re-run whenever
  `selectedUiBackend` changes) before or even without confirming an actual switch to it; a
  `previewing` CSS modifier (`app.css`) distinguishes "previewing" from "active" on the segmented
  buttons. The former read-only "Chat Model" card is now a `<select>` sourced from that backend's
  real model list, `onchange` calling `pinChatModel()`; an inline note states whether the pin applies
  immediately (selected backend is the active one) or only takes effect on a future switch. The
  free-text "Embedding model ID" field (and the rest of the now-empty "Model Configuration" card) was
  deleted — confirmed inert for every backend per §33/§34, and there is still no independent
  embedding-source switch endpoint (out of scope, tracked separately).

**Test suite:** 761 passed / 0 failed (759 baseline + 2 new — the asyncio.Lock type assertion and
the concurrent-requests-don't-block-the-loop regression test). `npm run check` and `npm run build`
both clean.

**Live verification (2026-07-15).** No browser-automation tool is available in this environment
(same limitation §7.10 already flags), so verification was: (1) structural — SSR HTML for
`/settings` diffed for the expected segmented-control/dropdown markup; (2) direct, against the real
running backend (Ollama active) and through the Vite dev server's `/api` proxy — `GET /health`
confirmed the new `backend` field; two genuinely concurrent `POST
.../ollama/chat-model` requests (idempotent re-pin of the already-active model, so no functional
change) both returned `200`/`applied_live: true` in ~3s total with no hang, exercising the fixed
lock on the live server rather than only the mocked test; a `POST` through the `:5173/api` proxy
confirmed the frontend's exact fetch path reaches the backend correctly. `.env` was restored
byte-for-byte after each test. No live click-through/visual QA pass was performed — flagged as a
limitation, not asserted as done, consistent with §7.10's own posture.

**Open items:**
- No live human browser/visual QA pass of the wired Settings UI has been performed as of this
  writing — structural/API-level verification only (see above).
- The `persisted: false` `.env`-write-failure path still needs a different provocation technique
  (carried forward from §16.5 — a read-only *directory*, not a read-only file, would trigger it).
