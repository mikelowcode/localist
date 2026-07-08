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
