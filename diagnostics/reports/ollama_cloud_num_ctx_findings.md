# Is `num_ctx` actually honored server-side by Ollama Cloud?

**Date:** 2026-07-18
**Type:** Investigation only — no application code changed. Probe script:
`diagnostics/diag_ollama_num_ctx_probe.py`.

## Verdict: **Not honored — the real enforced ceiling is each model's native context length, completely independent of the requested `num_ctx` value**

This holds for **both** Ollama Cloud and local Ollama (v0.31.1) — it is a general
Ollama runtime behavior, not something specific to the Cloud product. Requesting a
*smaller* `num_ctx` than what you actually send has **zero observed truncating
effect** at any value tested (8,000–200,000 for cloud; 2,000–20,000 for local). The
only hard limit that ever appeared was each model's own architectural maximum
context length (262,144 for `gemma4:31b-cloud`; 131,072 for `gemma4:e4b-mlx`),
enforced as a hard `400` rejection that explicitly names the model's real number —
never the app's configured `num_ctx`.

**Practical consequence for this app:** `CLOUD_PROFILE.total_context_tokens=100_000`
(what `OllamaRuntimeClient` sends as `options.num_ctx`) currently has **no functional
effect whatsoever** on Ollama Cloud. It is not a safety ceiling, not a cost control,
not a truncation boundary — the server will process however much is sent, up to
262,144 tokens, regardless of this value. The only thing actually protecting the app
from an oversized request is its own client-side budget
(`working_memory_tokens=60_000`, enforced by `MemoryManager.get_context_window()` and
`PromptBuilder`), which happens to sit comfortably under the model's real ceiling —
so nothing is currently broken in practice, but that safety is accidental, not
provided by `num_ctx`.

---

## 1. Response-metadata check

Ollama Cloud's `/api/chat` response reports `prompt_eval_count` (tokens actually
evaluated). Held `num_ctx=100,000` (the app's configured `CLOUD_PROFILE` value) fixed
and varied prompt length:

| target tokens | prompt_eval_count | ratio | TTFT (s) |
|---|---|---|---|
| 10,000 | 10,010 | 1.001 | 1.36 |
| 50,000 | 50,114 | 1.002 | 8.36 |
| 90,000 | 90,319 | 1.004 | 18.74 |
| 100,000 | 100,168 | 1.002 | 22.99 |
| 110,000 | 110,004 | 1.000 | 25.69 |
| 120,000 | 120,243 | 1.002 | 29.34 |

`prompt_eval_count` scales **perfectly linearly** with input, including 20,000 tokens
*past* the configured `num_ctx=100,000` — no plateau, no cliff, no sign of the server
respecting our ceiling at all. On its own, this only proves the server *can* process
more than we configured — it doesn't yet prove `num_ctx` is inert, since `num_ctx`
was never varied in this pass. That's step 1b below.

No field describing an "effective context length actually used" (as opposed to what
was requested) exists anywhere in the response — only `prompt_eval_count`,
`eval_count`, `total_duration`, `done_reason`. Nothing settles the question directly
from response metadata alone; the causal test below does.

---

## 1b. `num_ctx` sweep at a *fixed* prompt length — the actual causal test

Held a single, fixed 50,000-token haystack constant (needle planted at the very
start — the position most exposed by front-truncation, the most common truncation
strategy) and varied only the requested `num_ctx`:

| num_ctx | prompt_eval_count | needle found | TTFT (s) |
|---|---|---|---|
| 8,000 | 50,235 | **True** | 8.09 |
| 30,000 | 50,235 | **True** | 0.77 |
| 50,000 | 50,235 | **True** | 0.73 |
| 100,000 | 50,235 | **True** | 1.01 |
| 200,000 | 50,235 | **True** | 0.83 |

**Identical `prompt_eval_count` across a 25× range of requested `num_ctx` values,
including `num_ctx=8,000` against a 50,000-token prompt (6.25× larger than
requested) — and the needle planted at the very start still came back correctly
every time.** If `num_ctx=8,000` caused any real front-truncation, the start-planted
needle should have been the first thing dropped. It wasn't, at any tested value.

---

## 2. Local-backend cross-check (methodology sanity check)

The plan was to confirm this methodology detects a *known* truncation boundary
before trusting a "cloud passes cleanly" result — using local Ollama's well-documented
strict context enforcement as ground truth. **This did not go as expected, which
turned out to be the more important finding.**

First attempt — sweep `num_ctx` at a fixed 10,000-token prompt against
`gemma4:e4b-mlx` (native max 131,072):

| num_ctx | prompt_eval_count | needle found | TTFT (s) |
|---|---|---|---|
| 2,000 | 10,056 | **True** | 55.81 |
| 8,000 | 10,056 | **True** | 0.39 |
| 20,000 | 10,056 | **True** | 0.26 |

No truncation locally either — surprising, since local llama.cpp-based runners are
widely documented to enforce `n_ctx` strictly. Hypothesis: the model was already
resident from an earlier, unrelated call in this session with a larger effective
context, and a smaller subsequent request doesn't force a downward resize. **Ruled
out directly:** ran `ollama stop gemma4:e4b-mlx`, confirmed via `ollama ps` that
nothing was resident, then sent a **fresh-load** request with `num_ctx=2,000`
against the same 10,000-token prompt:

```
prompt_eval_count: 10049   needle_found: True   response: 'zebra-4471-quartz'
```

Even a genuinely fresh model load with an explicit, small `num_ctx=2,000` did not
truncate a 10,000-token prompt. **This confirms the behavior is not cloud-specific —
it is how this Ollama version (0.31.1) handles a `num_ctx` request smaller than the
actual input, for both backends.** The local cross-check didn't validate the
methodology by reproducing an *expected* truncation — it validated it by reproducing
the *same anomalous* non-truncation, which independent evidence (§3 below) shows is
itself a documented, known Ollama behavior rather than a flaw in this probe.

---

## 3. Independent corroboration — Ollama's own documentation and GitHub

- **[docs.ollama.com/context-length](https://docs.ollama.com/context-length):**
  describes context length as auto-scaling by available VRAM tier (4K / 32K / 256K)
  since Ollama v0.17.0, configurable via `OLLAMA_CONTEXT_LENGTH`. Does not document a
  precedence order between this, the Modelfile `PARAMETER num_ctx`, and a per-request
  `num_ctx` — nor whether a smaller per-request value is guaranteed to shrink the
  actual runtime allocation.
- **[GitHub issue #11964 — "context size larger than set"](https://github.com/ollama/ollama/issues/11964)**
  (opened 2025-08-19, **closed**). Independently reports the *exact* phenomenon
  found here: a user set `'num_ctx': 3182` via the API expecting a smaller context to
  fit their GPUs, but the actual runner process launched with `--ctx-size 40960` —
  more than 12× larger than requested, causing an unwanted CPU fallback. This is
  dated nearly a year before this investigation and confirms the behavior is
  reproducible, known, and not unique to this project's setup or this specific model.

Combined with §2's fresh-load control, this is strong enough that the "not honored"
verdict rests on more than this project's own probe.

---

## 4. Capstone test — does *anything* ever cap it?

Pushed prompts past each model's own stated native maximum, still requesting the
app's actual configured `num_ctx`:

**Cloud** (`gemma4:31b-cloud`, native max 262,144, requested `num_ctx=100,000`,
280,282-token prompt):
```
HTTP 400: {"error":"The prompt is too long: 280282, model maximum context length: 262144 (ref: ...)"}
```

**Local** (`gemma4:e4b-mlx`, native max 131,072, requested `num_ctx=8,000`,
140,274-token prompt):
```
HTTP 400: {"error":"input length (140274 tokens) exceeds the model's maximum context length (131072 tokens)"}
```

Both errors explicitly cite the **model's own native maximum** — 262,144 and 131,072
respectively — never the requested `num_ctx` (100,000 and 8,000). This is the
cleanest possible confirmation: the real gate is the model's built-in ceiling, and
`num_ctx` plays no visible role in either direction (neither shrinking the effective
window below the model's max, nor being reflected in the rejection message).

---

## 5. Lost-in-the-middle check (a separate question from truncation)

Since truncation was already ruled out as the mechanism, ran a smaller confirmatory
grid to check whether *recall quality* degrades with length independent of hard
truncation — this is a different phenomenon and shouldn't be conflated with a cap:

| tokens | position | trial 1 | trial 2 |
|---|---|---|---|
| 10,000 | start | found | found |
| 10,000 | late | found | found |
| 120,000 | start | found | found |
| 120,000 | late | found | found |

**8/8 recalled**, at both a short and a very long length (120,000 tokens — 20,000
past the app's own `num_ctx=100,000` setting), at both an early and a late needle
position. No lost-in-the-middle degradation observed in this range. (Not exhaustive —
see Scope note below.)

---

## Scope note — grid size reduced from the original request, and why

The original investigation prompt asked for a fuller grid (5 lengths × 2 positions ×
3 trials for cloud, plus a matching local grid). After the §1b `num_ctx` sweep gave
an unambiguous, clean answer to the core causal question (identical
`prompt_eval_count` across a 25× range of `num_ctx` values, at multiple independent
prompt lengths, confirmed with a fresh-load control and corroborated by a year-old
independent GitHub report), continuing to spend cloud tokens on a larger grid aimed
at the same already-answered question wasn't a good tradeoff. The remaining budget
went to the two things that pass wasn't answering: whether an ultimate ceiling exists
at all (§4, capstone) and whether a separate quality-degradation effect exists
independent of truncation (§5). Trials were reduced from 3 to 2 for the confirmatory
grid; a genuinely open question (e.g. an inconsistent/partial signal, the way the
prior prefix-cache investigation found) would have warranted the full 3-trial rigor,
but this question's signal was unusually unambiguous at every value tested.

**Approximate total token spend, this investigation:** ~1.53M tokens sent to Ollama
Cloud (metadata check ~480K, `num_ctx` sweep ~250K, capstone ~280K, final needle grid
~520K), plus ~180K tokens against the local backend (free/on-device, no cloud cost).
Flagging per the original prompt's request that this be visible, not just the
conclusions — this is a materially larger spend than the casual single-turn testing
recorded in `sessions-log.md`'s 2026-07-08 entry ("0.4% of session limit... Free
tier").

---

## 6. Recommendation

- **`CLOUD_PROFILE.total_context_tokens=100_000` should not be trusted as a real
  ceiling or cost control.** It is currently inert on Ollama Cloud. Whether to revise
  it down, remove it as a meaningless config value, or keep it purely as
  documentation of intent is a design decision for a follow-up implementation
  session — not changed here per the investigation-only scope.
- **No functional bug exists today.** The app's own client-side budget
  (`working_memory_tokens=60_000`) is well under both models' real native ceilings
  (262,144 cloud / 131,072 local), so nothing is silently getting cut off in
  practice. The risk is latent, not active: if `working_memory_tokens` (or the sum
  of all prompt slots) were ever raised close to or past a model's real native
  maximum, the failure mode would be a hard `400` rejection of the entire request
  (not a graceful degrade), and `num_ctx` would do nothing to prevent it.
  `MemoryManager`/`PromptBuilder`'s own ceilings are the only real backstop.
- **Two independent, uncoordinated truncation mechanisms do *not* both apply here**
  — worth stating explicitly since the original prompt asked this to be checked.
  There aren't two competing truncation layers each silently fighting over the same
  budget; there is exactly one that actually does anything (the app's own
  client-side ceiling), and one that's currently a no-op (`num_ctx`). That's a
  simpler, if less obviously intentional, situation than "two mechanisms
  double-truncating."
- If a follow-up implementation task is warranted, worth deciding then whether
  `num_ctx` should be dropped entirely (since it does nothing), kept as
  forward-looking documentation of intent (in case a future Ollama version honors
  it), or replaced with a request that queries each model's real native max (e.g.
  via `ollama show`/`/api/show`) and clamps the app's own client-side budget against
  *that*, rather than assuming a number.
