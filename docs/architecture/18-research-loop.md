## 18. Research Loop

### 18.1 Overview

A plain `web_search` dispatch (§4.6, §14.2) fires exactly once with a
single query and returns whatever the provider gives back — for an
open-ended lookup ("what's the latest on X") that's sufficient, but for a
request that needs a *specific, extractable fact* (a price, a spec, a
plan tier) a single search snippet frequently doesn't contain the number
at all, only a link to a page that does.

The research loop (`MCPToolDispatcher._run_research_loop`,
`backend/mcp_tool_dispatcher.py`) replaces that single fire-and-return
with a bounded search → evaluate → (fetch) → reformulate cycle: search,
cheaply classify whether the result actually contains concrete pricing,
fetch the top candidate page if the snippet alone was inconclusive,
re-classify, and — only if still inconclusive — ask the model to rewrite
the query once before retrying. It is not a new MCP tool on
`localist-mcp`; it is a client-side loop inside `MCPToolDispatcher` over
the existing `web_search`/`fetch_url` MCP tools (§14.2).

**Not yet enabled by default.** The whole feature is gated behind
`LOCALIST_RESEARCH_LOOP_ENABLED`, which ships `false` in `.env.example`
(§18.7). Everything described below exists in the codebase and is
covered by tests, but a fresh checkout with default settings never routes
a turn to it.

### 18.2 Routing — `research_intent` and the `web_search` → `research` Upgrade

`research_intent` is one of the semantic template groups in
`planner.py`'s `_SEARCH_INTENT_TEMPLATES` (alongside
`explicit_search_action`, `lookup_request`, `knowledge_request_open`, and
`freshness_request` — §10). Its 8 templates are verb-anchored lookup
phrasings naming a concrete cost/spec object (e.g. `"look up the pricing
for this product"`, `"find the cost of this plan per month"`) — see
§18.5 for why that specific phrasing shape was chosen over a v1 set that
named cost/price vocabulary without a lookup verb.

Unlike `explicit_search_action`/`lookup_request`, `research_intent` is
**not** a member of `_SEMANTIC_GATE_THRESHOLDS` and never independently
decides whether a tool fires at all. It only ever *upgrades* an
already-firing `web_search` to `"research"`, inside `_priority3_tool()`,
after every other P3 signal (literal keyword, `explicit_search_action`,
`lookup_request`) has already put `"web_search"` into `tools_to_call` for
this turn:

```python
if (
    "web_search" in tools
    and semantic_result is not None
    and self._research_loop_enabled()
):
    research_score = all_scores.get("research_intent", -1.0)
    if research_score >= _RESEARCH_INTENT_THRESHOLD:
        tools[tools.index("web_search")] = "research"
```

`_RESEARCH_INTENT_THRESHOLD = 0.65` (module constant, `planner.py`).
`_research_loop_enabled()` reads `LOCALIST_RESEARCH_LOOP_ENABLED` from
the environment **at call time**, not cached at Planner construction —
the same pattern every other feature flag in this file uses, so flipping
`.env` doesn't require reconstructing the Planner. Default (unset, or any
value other than `"true"` case-insensitively) is disabled — the same
fail-safe-to-existing-behavior direction every gate in this file already
defaults to.

The `semantic_result` computation itself (the embedding call and
per-group cosine scores) is shared with the existing
`explicit_search_action`/`lookup_request` gating — no second embed call
is made just to score `research_intent`.

**Embedding-model dependency, by reference.** `research_intent`'s
threshold has the identical embedding-model-portability problem every
other semantic gate in this file has (§10, §16.4): it was tuned against
`mlx-community/embeddinggemma-300m-4bit`, and cosine similarity does not
transfer across embedding models. `Planner.__init__`'s
`_TUNED_EMBEDDING_MODEL` guard (§16.4) disables `_semantic_search_intent()`
entirely — `research_intent` included — the moment the active embedding
model doesn't match, rather than letting `research_intent` silently
compute meaningless scores under an unvalidated model. See §16.4 for the
guard itself; not re-explained here.

### 18.3 Negative-Filter Redesign and the Tie-Break Exception

`_SEARCH_NEGATIVE_FILTER` and `_RESEARCH_NEGATIVE_FILTER` (`planner.py`)
are substring-matched phrase sets that catch utterances whose embedding
score collides with real search intent for syntactic, not semantic,
reasons — identity/capability questions and greetings for the former
(§10.4), subjective price-opinion phrasing (`"worth the price"`, `"too
expensive"`) for the latter, added specifically for `research_intent`'s
own collision (§18.5).

**Before this work**, both filters were checked *before* scoring: a
matched phrase short-circuited `_semantic_search_intent()` straight to
`None`, and — because the check ran ahead of the embedding call — no
score was ever computed or logged for a filtered turn, and a match on
one group's known collision suppressed *every* group's gating for that
turn, `research_intent` included, not just the group the filter actually
targets.

**As of this work**, `_semantic_search_intent()` always scores first.
The negative filters are checked *after* scoring, and only escalate to a
model call when there's a genuine conflict — a filter phrase matched
*and* some gated group's score cleared its own threshold anyway. If
nothing cleared threshold, the filter and the embedding already agree
(nothing was going to fire) and no call is made. Empirically this is the
common case: only 4–8 of 11 filter-matched utterances in the validating
diagnostic (§18.5) actually reached a conflict, depending on
`LOCALIST_RESEARCH_LOOP_ENABLED`.

When a genuine conflict occurs, `_resolve_negative_filter_conflict()`
makes a single bounded `runtime.infer()` call
(`_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT`, one-word `lookup`/`other`
response) to decide whether the filter or the embedding score is right
for this specific utterance. It fails closed: any runtime error, timeout,
or unparseable response confirms the filter's suppression (returns
`False`) rather than risking a known collision phrase through gating.

**This is a documented exception, not an oversight.** `planner.py`'s
module docstring states the general rule — "Inference is invoked in
exactly one place: Priority 5 (episodic relevance). Priorities 1–4 and 6
are pure rule evaluations — no model calls." `Planner.__init__`'s
`runtime` parameter docstring names the carve-out explicitly: `runtime`
is used in Priority 3 *only* for this tie-break, "never for the base P3
keyword/semantic routing decision itself," and only reached when a
negative-filter phrase matched **and** a gated group's score cleared its
own threshold anyway — not on every P3 turn.

### 18.4 The Loop Itself — `MCPToolDispatcher._run_research_loop`

`_run_research_loop(session, connect_error, instruction, context)` runs
when `"research"` appears in `tools_to_call` (`_dispatch_async`'s
per-tool dispatch, alongside the existing `file_op`/`url_fetch`/
`web_search` branches).

**Bounded iteration.** `_MAX_RESEARCH_ITERATIONS = 3` — a hard cap on
search+evaluate+reformulate cycles, same cost/latency rationale as
`_MAX_WEB_QUERIES` capping plain `web_search` at 3 queries per dispatch.

**Per iteration:**
1. `web_search` the current query (first iteration: `_derive_initial_query()`
   — identical resolution order to plain `web_search`'s own query
   derivation, so turn one behaves identically to a plain `web_search`
   dispatch and only diverges once evaluation kicks in).
2. If the search call itself fails (provider/connectivity error, not "no
   pricing found"), stop immediately — no reformulation, no synthetic
   result (see below).
3. `_evaluate_pricing_gate()` — a single bounded `runtime.infer()` call
   (`max_tokens=10`, `_RESEARCH_GATE_SYSTEM_PROMPT`) asking whether the
   search snippet already contains concrete pricing. Fails closed to
   `False` on any exception.
4. If the gate is inconclusive but the snippet names a URL
   (`_extract_first_url()`), `url_fetch` that page and re-run the gate on
   the full page text.
5. If the gate passes (snippet or fetched page), return every
   `ToolResult` produced so far — not just the winning one.
6. Otherwise, if this wasn't the last allowed iteration,
   `_reformulate_query()` — another single bounded `runtime.infer()` call
   (`max_tokens=40`, `_RESEARCH_REFORMULATE_SYSTEM_PROMPT`) rewriting the
   query to be more likely to surface a pricing page. **Repeat-guard:**
   if the reformulated query is identical to one already tried (including
   the fallback-on-failure behavior of `_reformulate_query()` itself,
   which returns the last query unchanged on an `infer()` error), the
   loop stops immediately rather than spend another round-trip on a query
   already known not to work.

**Exhaustion — the synthetic trailing `ToolResult`.** If the loop runs
out of iterations (or the repeat-guard fires) without ever finding
pricing, and the reason wasn't a connectivity failure, a synthetic
`ToolResult(tool_name="research", result="ERROR: research loop exhausted
N iteration(s)...", success=False)` is appended to the returned list.
This exists because every individual search/fetch call along the way
*succeeded* (the searches worked; they just never found pricing), so
none of them would trip `controller_agent.py`'s Step 3b corpus-fallback
check on its own (`r.tool_name == "web_search" and not r.success`).
`controller_agent.py`'s Step 3b was extended with `or r.tool_name ==
"research"` specifically to catch this synthetic entry, so a "loop
exhausted, no pricing found" outcome gets the same corpus-grounding
attempt as a "search API down" outcome, rather than leaving the final
answer ungrounded. A genuine connectivity failure inside the loop, by
contrast, already produces a normal `tool_name="web_search"`,
`success=False` entry indistinguishable from a plain `web_search`
failure — Step 3b's original clause catches that case unmodified, so no
synthetic entry is appended for it.

**`_RESEARCH_CLASSIFIER_TIMEOUT = 15.0`**, applied only to
`_evaluate_pricing_gate()` and `_reformulate_query()`'s `infer()` calls —
every other `infer()`/`infer_stream()` call site in the codebase keeps
the default `LOCALIST_STREAM_TIMEOUT` (60s) unchanged. The default is
sized for a full-length (up to 1024-token) main-dispatch answer; a
`max_tokens=10`/`40` classifier call sharing that same budget means a
single stuck cloud-side call burns most of a minute before the loop can
even attempt to recover, when it should fail fast and let the
repeat-guard/reformulation machinery move on instead. This required a new
`timeout: float | None = None` parameter on `BaseRuntimeClient.infer()` /
`infer_stream()` (and every concrete implementation —
`OllamaRuntimeClient`, `OMLXRuntimeClient`, `FoundryRuntimeClient`) —
`None` (the default for every pre-existing call site) means "use the
client's configured default timeout," so this is additive, not a
behavior change for anything except the two research-loop call sites
that pass it explicitly.

### 18.5 Diagnostic Provenance

Three read-only diagnostics (`diagnostics/reports/`, per this repo's
diagnostics discipline — see `CLAUDE.md`) established the numbers above
before they were wired into `planner.py`:

- **`research_intent_threshold_assessment_2026-07-16.md` (v1).** Scored
  an initial 8-template `research_intent` candidate set (bare cost/price
  phrasing, e.g. `"what does this cost"`) against 30 utterances across 7
  categories. Found a **threshold-unfixable collision**: Category E
  (subjective price opinion — `"Do you think this is worth the price?"`)
  scored *above* every true positive (max FP-pool score 0.8451 vs.
  minimum true-positive score 0.6908), the same failure shape as
  `lookup_request`'s 2026-06-25 incident (§10) — no threshold can
  separate a negative that outscores the positives, only different
  template wording can.
- **`research_intent_threshold_assessment_2026-07-16-v2.md`.** Re-tested
  a verb-anchored v2 template set (every template pairs a lookup verb
  with a concrete cost object) plus the new `_RESEARCH_NEGATIVE_FILTER`
  pre-filter. Category E's collision is fully resolved (all 4 Category E
  utterances now match the negative filter and never reach the
  `research_intent` FP-pool analysis at all), but a smaller residual
  collision against `lookup_request`-shaped utterances remains
  (`"Can you look up information about the latest Apple products?"` →
  0.6748, above the true-positive minimum of 0.6342). At the chosen
  threshold of 0.65: 9/10 true positives survive (only a marginal,
  non-brand-specific plumber-pricing phrasing drops out) and 2/16 of the
  remaining FP-pool items leak — both low-cost leaks, since they're
  Category L utterances that already trigger `web_search` via
  `lookup_request` regardless, so a false positive here just burns extra
  bounded loop iterations rather than producing a wrong answer.
- **`negative_filter_tiebreak_assessment_2026-07-16.md`.** Validated
  `_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT` against 37 utterances (LIVE,
  `gemma4:31b-cloud` via Ollama, real `EmbeddingEngine`) before it was
  wired into `_resolve_negative_filter_conflict`. Full-set accuracy
  32/37 (86.5%), but every error fell on utterances (Category K/F) that
  never match either negative filter and so never actually reach the
  tie-break call in production. Restricted to the operationally-reachable
  subset — utterances that actually matched a filter — accuracy was
  **11/11 (100%)**, 0 false positives and 0 false negatives. Also
  measured the tie-break's real fire rate against the "fires noticeably
  more often" claim in the original design sketch: 4/11 filter-matched
  utterances reach it with the research loop flag off, 8/11 with it on —
  real but more of a minority than the sketch's framing implied.

### 18.6 Test Coverage

- **`TestResearchLoop`** (`backend/tests/test_mcp_tool_dispatcher.py`,
  16 tests) — gate-passes-immediately, gate-fails-then-passes-after-fetch,
  iteration-cap exhaustion appending the synthetic failure result, the
  repeat-guard stopping before the iteration cap, a connectivity failure
  stopping without a synthetic result, `_evaluate_pricing_gate`/
  `_reformulate_query` failing closed on an `infer()` exception,
  `_RESEARCH_CLASSIFIER_TIMEOUT` actually reaching the underlying
  `infer()` calls, `dispatch()` routing `"research"` to the loop,
  `_derive_initial_query`'s two resolution paths, and four
  `_extract_first_url` variants (bracket-wrapped, paren-wrapped, trailing
  sentence punctuation, and an unwrapped URL left unaffected).
- **`test_ollama_runtime_client.py`** (14 tests, new file) — the NDJSON
  mid-stream-error / done-less-stream fix (§18.7) and the `timeout`
  parameter override, both described below.
- **`planner.py` semantic-gating tests** (`test_planner_phase3.py`) —
  the negative-filter conflict-resolution redesign has dedicated coverage
  (no-conflict-no-model-call, conflict-confirmed-by-tiebreak,
  conflict-overridden-by-tiebreak), and the `_TUNED_EMBEDDING_MODEL` guard
  that also disables `research_intent` scoring has its own suite
  (`TestTunedEmbeddingModelGuard`, §16.4).

**Known gap.** There is currently no test that directly exercises
`_priority3_tool()`'s `web_search` → `"research"` upgrade branch itself
(the `tools[tools.index("web_search")] = "research"` line in §18.2) —
neither the threshold check nor the `LOCALIST_RESEARCH_LOOP_ENABLED` gate
around it has a dedicated assertion in `test_planner_phase3.py`. The
surrounding machinery (`_semantic_search_intent()`'s scoring, the
negative-filter tie-break, the embedding-model guard) is well covered;
the upgrade decision that consumes `research_intent`'s score is not.
Carried forward as an open item (§18.8).

### 18.7 Live-Verification Status

Extensively live-tested this session against real search-provider and
cloud-model traffic (Brave Search, Ollama Cloud) — not just the mocked
`TestResearchLoop` suite. Three real bugs were found and fixed along the
way, none of them hypothetical:

1. **Bracket-wrapped URL parsing.** `mcp_server/web_search.py`'s result
   formatting wraps every URL in literal `[...]` (`f"• {title}\n
   {body}\n  [{url}]"`). `_URL_RE`'s original character class didn't
   exclude `]`/`)`, so a URL extracted from a search snippet captured the
   trailing bracket as part of the URL (e.g. `.../pricing]`), which then
   404'd when passed to `fetch_url`. Fixed by adding `]`/`)` to `_URL_RE`'s
   excluded-character class (`mcp_tool_dispatcher.py`) — a shared-regex
   fix, since the same pattern also backs `_run_url_fetch`'s
   instruction-text URL extraction, not a research-loop-only bug.
   `_extract_first_url()` additionally strips trailing sentence
   punctuation (`.,;:`) as a second, cheap layer of defense against a
   differently-formatted future source hitting the same class of bug.
2. **Silent-failure NDJSON stream bug** (`ollama_runtime_client.py`,
   unrelated to the research loop's own logic but discovered via its
   repeated tool-heavy turns). `_iter_ndjson_chunks()` previously treated
   two real failure modes as an unremarkable empty completion: a
   mid-stream `{"error": "..."}` line (rate limit, context-length
   overflow, moderation block, mid-generation crash) has no `"message"`
   key, so its content resolved to `""` and was skipped the same as any
   other empty delta; and a connection that closed without ever sending
   `"done": true` just ended the generator with zero chunks yielded and
   no exception. Both are now raised as `RuntimeError` instead of
   resolving silently. Found via repeated `output_chars=0` completions on
   tool-heavy turns during this session's testing, root-caused by reading
   the actual streaming code path rather than assuming a transient cloud
   hiccup.
3. **Classifier-timeout tuning** — the `_RESEARCH_CLASSIFIER_TIMEOUT`
   mechanism described in §18.4, added after a live `max_tokens=10` gate
   check stalled for the full default 60s stream timeout on a cloud-side
   hang (confirmed not a local issue — health-check polling to
   `/api/tags` stayed healthy throughout the stall).

**Not yet enabled by default.** `LOCALIST_RESEARCH_LOOP_ENABLED=false` in
`.env.example` — a fresh checkout never routes to the research loop
without an explicit opt-in. This is a deliberate, not accidental, default
pending more live use (§18.8), consistent with how this codebase treats
every other feature-flagged mechanism (§15).

### 18.8 Open Items

- `LOCALIST_RESEARCH_LOOP_ENABLED` remains off by default pending more
  live use — no shadow-mode rollout was used for this feature (unlike
  §15's classifier); see the design decision recorded in
  `sessions-log.md` for why a plain on/off switch was judged acceptable
  for this single-user, non-production app.
- No dedicated test exercises the `web_search` → `"research"` upgrade
  branch in `_priority3_tool()` itself (§18.6).
- No live human QA of the research loop's actual *answer quality* beyond
  the specific queries exercised during this session's live testing —
  the bugs found and fixed (§18.7) were mechanical (URL parsing, stream
  handling, timeouts), not an evaluation of whether the loop reliably
  converges on correct pricing/spec answers across a broad query set.
- `MemoryManager`'s corpus/episodic embedding-provenance gap is now
  closed (§16.4); the oMLX `embedding_model` wiring gap noted in §16.4
  is unrelated to this feature and remains separately open.
