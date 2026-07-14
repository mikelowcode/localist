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
   controller rather than crashing — but this has **not** been verified live and should be before
   being trusted.

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
- Corpus-retrieval similarity thresholds may need to become embedding-model-aware — a low top-score
  corpus miss (`0.028`) was observed under `nomic-embed-text` on a query that plausibly should have
  matched; not yet enough samples to confirm this is a real threshold-transfer issue versus normal
  variance across embedding models' similarity distributions.
- Apple-Silicon skip branch remains verified only via mocked `platform` calls; real non-Darwin
  hardware verification is still outstanding, pending access to such a machine.
- oMLX's `embedding_model` wiring gap noted above (tier 1 never engages for that backend) — a
  follow-up, not yet scheduled.
