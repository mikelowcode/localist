# Does Ollama Cloud do cross-request prefix-cache reuse?

**Date:** 2026-07-18
**Type:** Investigation only — no application code changed. Probe script:
`diagnostics/diag_ollama_cloud_prefix_cache_probe.py`.

## Verdict: **Plausible, not confirmed** — and the strongest available evidence leans toward "not a dependable feature today"

There is a real, token-count-controlled TTFT gap in the live test below, but it is
inconsistent within its own condition, does not survive a 180-second gap in the one
data point tested, and is contradicted by the more rigorous piece of third-party
evidence found (a 7-day, 1,129-session token-accounting measurement showing 0.64%
cache utilization on Ollama Cloud). Treat prefix caching as **not something to plan
around** for the `CLOUD_PROFILE` budget decision — see Implications below.

---

## 0. What prompted this — and why it shouldn't be trusted as-is

A live in-app chat turn (`chat_turns.id=580`, 2026-07-18 18:31:02, found by querying
`backend/localist_memory.db` directly) reads:

> "the performance you're seeing—specifically the millisecond Time to First Token
> (TTFT)—strongly suggests that server-side prefix caching is indeed active, even if
> it isn't explicitly detailed in the public-facing documentation I found during my
> search."

This is a single anecdotal TTFT observation being read as confirmation. A fast TTFT
is equally consistent with "Ollama Cloud just runs on faster data-center GPUs than an
M1" — no caching required. **This should not be treated as a settled finding.**
Checked whether it was ever written down as one elsewhere: it wasn't. The episodic
fact actually extracted from that same conversation (`episodes` table, 2026-07-18
18:29:33) is correctly hedged — *"The user is evaluating whether Ollama Cloud
supports cross-request prefix-cache reuse... "* — and none of the durable
architecture docs (`docs/architecture/*.md`) or `wiki/MEMORY.md` assert caching as
confirmed. So there is nothing durable to correct — only the raw chat transcript
overclaims, and chat history is an append-only log, not something edited after the
fact. **Recommend posting a follow-up correction turn in the app itself** referencing
this report, rather than treating turn 580 as established.

---

## 1. Documentation / API-surface sweep

**Ollama's own docs/blog:** describe automatic KV-cache reuse for the **local**
daemon only — prefix must be byte-identical, and the cache is dropped when the model
is unloaded from VRAM (5-minute default idle timeout). Nothing found describing this
as a feature of the Cloud product specifically.

**GitHub — three directly relevant, currently-open-or-recent issues on
`ollama/ollama`:**

- **[#15600 — "Ollama Cloud Prompt Caching"](https://github.com/ollama/ollama/issues/15600)**
  (opened 2026-04-15, closed as duplicate of #15758). Body, verbatim: *"Does Ollama
  Cloud currently support Prompt Caching (or KV caching)? If it does, how long is the
  cache typically persisted?"* No maintainer answer is visible before it was closed
  as a duplicate.
- **[#15758 — "Ollama's Cloud doesn't report number of cached tokens"](https://github.com/ollama/ollama/issues/15758)**
  (opened 2026-04-23, **still open**). Body, verbatim: *"Behind the scenes requests
  are sped up with caches, but we currently always report 0 cached tokens."* This is
  a user's *claim*, not a maintainer confirmation — and it is directly contradicted
  by the stronger measurement below.
- **[#16714 — "Ollama Cloud - Prompt Cache Support"](https://github.com/ollama/ollama/issues/16714)**
  (opened 2026-06-14, **still open**, labeled feature request, assigned to a maintainer).
  Body, verbatim: *"I request ollama-cloud to support the provider cache... we are
  either losing out completely on savings and performance - or the savings are not
  passed on to end users."* An open feature *request* for something to be added is
  itself evidence the requester does not currently observe it working.

**Strongest single piece of evidence found — [`NousResearch/hermes-agent` issue
#55422](https://github.com/NousResearch/hermes-agent/issues/55422)** (opened
2026-06-30, still open — closer to today than the issues above). An independent
developer instrumented their own agent's per-profile SQLite state over a **7-day
rolling window, 15 profiles, 1,129 Ollama Cloud sessions**, using **OpenAI Codex on
the identical system/workload as a control**:

| | Ollama Cloud | OpenAI Codex (control) |
|---|---|---|
| Sessions measured | 1,129 | 1,541 |
| Cache utilization | **0.64%** | **82.73%** |
| Sessions with zero cache reads | 1,123 / 1,129 (99.5%) | — |

Their conclusion: the gap is specific to Ollama Cloud's provider handling (not a
generic accounting bug in their own tool, since the identical instrumentation shows
expected behavior for OpenAI Codex on the same machine), and users may "burn Ollama
Cloud quota much faster because repeated large prompt prefixes are not being cached
or are not being measured." This is exactly the billing/usage signal step 3 of the
original investigation prompt asked for — it's a third party's data, not this
project's own dashboard, since no Ollama Cloud account/billing UI is accessible from
this environment (no browser session), but it is a real, cited, dated measurement
rather than reasoning from documentation silence.

A `x.com/ollama` post surfaced in search ("Improved caching for more responsiveness...
Ollama will now reuse its cache across conversations... more cache hits when
branching") could not be fetched directly (HTTP 402 from this environment) to confirm
whether it refers to the local Ollama app/daemon cache or the Cloud service — the
surrounding language ("memory utilization") reads as describing the local, on-device
cache manager, not a cloud API concern, but this is unconfirmed and not relied upon
either way.

---

## 2. Controlled empirical test — `diag_ollama_cloud_prefix_cache_probe.py`

Ran live against the real Ollama Cloud endpoint (`gemma4:31b-cloud`, proxied through
the local daemon at `localhost:11434` — same one the app uses), bypassing
`OllamaRuntimeClient`/`ControllerAgent`/`PromptBuilder` entirely: raw `requests.post`
calls to `/api/chat`, streaming, measuring wall-clock time-to-first-chunk.

**Methodology note, itself a finding:** Ollama Cloud's response omits
`prompt_eval_duration`, `eval_duration`, and `load_duration` entirely — confirmed by
a side-by-side calibration call against the **local** `gemma4:e4b-mlx` model, which
*does* populate all of them (`prompt_eval_duration=1358664041`,
`eval_duration=340800542`, `load_duration=8935636250` ns). Only `total_duration`
(a single combined figure) and `prompt_eval_count`/`eval_count` (token counts, not
durations) come back from the Cloud endpoint. So TTFT (wall-clock, measured locally)
is the only precise per-phase signal available — the original investigation prompt's
hope of comparing `prompt_eval_duration` directly isn't possible against this API.

**A methodological correction made mid-investigation, worth flagging on its own:**
The first run's Condition B (no-shared-prefix control) used random lowercase letter
salad to match Condition A's *character* length. That silently produced **~14,900
prompt tokens vs. Condition A's 7,030** — gibberish tokenizes far less efficiently
under BPE than real text, so B was processing roughly 2.1× the actual token count
despite matching char-for-char. That run's ~1.07s TTFT gap was mostly or entirely
explained by that confound, not caching, and is **not used** as evidence below. Fixed
by rebuilding Condition B from shuffled real English words
(`/usr/share/dict/words`), calibrated to within ~1% of Condition A's actual token
count, then rerun. All numbers below are from the corrected run.

**Condition A — repeated identical ~7,875-char / 7,030-token prefix, 5 calls, varying
short suffix:**

| Call | TTFT (s) | total_duration (s) |
|---|---|---|
| 1 | 0.934 | 0.921 |
| 2 | 0.204 | 0.190 |
| 3 | 0.385 | 0.418 |
| 4 | 0.374 | 0.384 |
| 5 | 0.975 | 0.936 |
| **mean / stdev** | **0.574 / 0.354** | |

**Condition B — no shared prefix, freshly randomized real-word content each call,
token-matched (7,029–7,114 tokens), 5 calls:**

| Call | TTFT (s) | total_duration (s) |
|---|---|---|
| 1 | 0.924 | 0.914 |
| 2 | 1.145 | 1.148 |
| 3 | 1.074 | 0.976 |
| 4 | 1.226 | 1.482 |
| 5 | 0.947 | 0.986 |
| **mean / stdev** | **1.063 / 0.129** | |

**Condition C — same prefix, immediate repeat vs. after a 180s gap:**

| | TTFT (s) |
|---|---|
| Immediate repeat | 0.623 |
| After 180s gap | **9.100** |

### Reading the data honestly

- Condition A's mean TTFT (0.574s) is meaningfully lower than Condition B's (1.063s)
  at matched token counts — a real, not-confounded gap.
- But Condition A's own spread is the more interesting signal: 3 of 5 calls landed
  fast (0.204–0.385s) while 2 landed slow (0.934–0.975s) — squarely inside Condition
  B's range. **A repeated, byte-identical prefix did not reliably produce a fast
  response.** This is the opposite of what a dependable, always-on cache would look
  like (which should make *every* repeat fast) — it's more consistent with either
  (a) an inconsistent/partial cache that only sometimes hits, e.g. a multi-replica
  backend where a repeated request doesn't always land on the instance that has the
  KV state warm, or (b) generic multi-tenant load variance on a shared cloud service
  that has nothing to do with caching at all, coincidentally producing a similar
  spread. This data cannot distinguish between those two explanations.
- Condition C's single after-gap data point (9.1s) is **far slower than anything
  else observed** in the entire probe (everything else was 0.2–2.2s) — the opposite
  of "cache persisted across the gap." This is one sample and could be a cold-replica
  routing artifact, a rate-limit backoff, or genuine cache eviction — not enough
  data to tell which, but it directly weighs against "caching reliably speeds up a
  resumed conversation after any idle time," which is the actual behavior the
  `CLOUD_PROFILE` design would need to lean on for a latency/cost win.

Net: real signal, but **not a clean, reproducible "cached is always fast" pattern** —
exactly the kind of ambiguous result the original investigation prompt warned against
over-reading from a single anecdote, now reproduced with more data points and still
ambiguous.

---

## 3. Cost/billing signal

No Ollama Cloud account/usage dashboard is accessible from this environment (no
browser session available to Claude Code). The Hermes-agent issue in §1 substitutes
for this — it's exactly the token-accounting metric (`cache_read_tokens`) a billing
surface would expose, measured independently by a third party over a real 7-day
production workload, showing 0.64% utilization. That is the strongest single data
point in this whole investigation, and it points toward "caching is not reliably
happening or not being counted" — either way, not something to bank on for cost
planning today.

---

## 4. Implications for `CLOUD_PROFILE` (context_profile.py)

The 2026-07-18 session that added `CLOUD_PROFILE` (60,000-token working memory /
100,000-token `num_ctx`) already flagged this exact uncertainty as an open item
rather than assuming a caching win. This investigation doesn't overturn that
caution — it reinforces it:

- **Do not budget on caching reducing the cost/latency of a growing working-memory
  prompt.** The evidence here is at best "sometimes, unreliably, for some requests"
  — not a dependable per-turn discount. Every new turn should be planned as if it
  pays full prompt-eval cost for the entire accumulated prefix, because on the
  current evidence it often will.
- The reasoning-quality half of the original goal (the model can see full history)
  stands regardless of caching and is unaffected by this finding.
- If a cost/latency guardrail for the cloud tier is ever built (explicitly out of
  scope in the prior session), this finding is a reason to prioritize it sooner
  rather than later — a 60K-token prompt on every turn of a long conversation is a
  real, recurring, likely-uncached cost, not a one-time one.
- Given this, whether `CLOUD_PROFILE.working_memory_tokens` should stay at 60,000
  is worth revisiting with cost/latency (not just context-budget) framing in mind —
  flagged here as a recommendation, not implemented, since this was explicitly an
  investigation-only pass.

---

## 5. Correcting the record

`chat_turns.id=580`'s "strongly suggests... server-side prefix caching is indeed
active" should not be treated as established anywhere in the project. It was never
written into a durable doc or episodic-memory fact (checked directly against
`backend/wiki/MEMORY.md`, `episodes` table, and all `docs/architecture/*.md` files —
no match), so there's nothing to edit there. The chat transcript itself is
append-only by design (Chat History Tab, §12) — recommend a follow-up chat turn in
the live app acknowledging the correction with a pointer to this report, rather than
attempting to alter history.
