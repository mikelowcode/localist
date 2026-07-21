# Research Loop Answer-Quality QA Pass — Ollama Cloud, Gemma 4-31B

**Date:** 2026-07-20
**Script:** `diagnostics/research_loop_qa_pass.py`
**Corpus:** `diagnostics/research_loop_qa_corpus.md` (18 queries, 7 categories A–G)
**Raw data:** `diagnostics/research_loop_qa_2026-07-20.csv`
**Runtime:** backend=`ollama` chat_model=`gemma4:31b-cloud` (LIVE, no mocks) — confirmed
against `backend/.env` (`LOCALIST_RUNTIME_BACKEND=ollama`,
`LOCALIST_CHAT_MODEL_OLLAMA=gemma4:31b-cloud`) before running, matching the corpus
author's stated intent.
**Search provider:** live Brave Search via localist-mcp (port 8003) — real network
calls throughout, not mocked.
**Status:** READ-ONLY. No source file, database, or episodic memory touched.
Closes the open item in `docs/architecture/18-research-loop.md` §18.8.

## Methodological caveat — read before interpreting §2/§3

This script calls `MCPToolDispatcher.dispatch(["research"], instruction, ...)`
**directly**, bypassing `Planner` and `ControllerAgent`/`ConversationalAgent`
entirely — a deliberate scoping choice (routing is already covered by the prior
session's slash-command tests; this pass measures the loop's own mechanics). One
consequence that matters for reading the results below: **there is no
answer-synthesis step in this pass.** "Final answer" in the CSV and below is the
raw winning `ToolResult.result` text (a search-snippet block, a fetched page's
extracted text, or the synthetic exhaustion message) — never a model-composed
natural-language reply. Where the corpus's category descriptions ask "did the
final answer acknowledge ambiguity" or "did it fabricate a number," those
questions live at the synthesis layer this pass does not exercise. What this pass
*can* and does answer: what raw material the loop's gate accepts as sufficient
and hands upstream, and whether the gate's pass/fail decision is actually
responsive to the question asked — which turned out to be the more useful thing
to measure (§3).

## §1. Summary Table

| # | Cat | Query (truncated) | Iterations | Fetch | Reformulated | Outcome | Elapsed |
|---|:---:|---|:---:|:---:|:---:|---|---:|
| 1 | A | Netflix Standard plan price | 1 | N | N | gate_passed_on_snippet | 0.71s |
| 2 | A | GitHub Copilot Individual cost | 1 | N | N | gate_passed_on_snippet | 1.03s |
| 3 | A | 2024 Honda Civic LX base price | 1 | N | N | gate_passed_on_snippet | 1.07s |
| 4 | B | Backblaze B2 storage price/GB | 1 | N | N | gate_passed_on_snippet | 1.02s |
| 5 | B | Tesla Model 3 entry price | 1 | N | N | gate_passed_on_snippet | 1.14s |
| 6 | B | Notion Business plan/user/mo | 1 | N | N | gate_passed_on_snippet | 1.35s |
| 7 | C | t3.medium EC2 us-east-1 cost | 1 | N | N | gate_passed_on_snippet | 2.05s |
| 8 | C | Figma Enterprise per-seat price | 1 | N | N | gate_passed_on_snippet | 1.00s |
| 9 | C | Ford F-150 XLT crew cab start | 1 | N | N | gate_passed_on_snippet | 1.07s |
| 10 | D | Adobe Creative Cloud cost | 1 | N | N | gate_passed_on_snippet | 1.15s |
| 11 | D | iPhone 16 price | 1 | N | N | gate_passed_on_snippet | 1.13s |
| 12 | E | Salesforce exact enterprise contract price | 1 | N | N | gate_passed_on_snippet | 5.69s |
| 13 | E | Discontinued Stadia controller cost | 1 | N | N | gate_passed_on_snippet | 1.00s |
| 14 | E | Obscure NFT collection floor price | 1 | N | N | gate_passed_on_snippet | 1.11s |
| 15 | F | Ford F-150 Lightning max payload | 2 | Y | Y | gate_passed_after_fetch | 15.80s |
| 16 | F | Tesla Model Y LR battery kWh | 3 | Y | Y | exhausted_honestly | 9.26s |
| 17 | G | "worth the price" (iPhone, opinion) | 2 | Y | Y | gate_passed_on_snippet | 3.79s |
| 18 | G | "overpriced vs competitors" (Tesla, opinion) | 1 | Y | N* | exhausted_honestly | 16.22s |

`*` Query 18's `reformulated=N` reflects that only one `web_search` call is present
in the returned results — the repeat-guard (§18.4) appears to have fired inside
the same iteration after a fetch, stopping the loop before a second `web_search`
call ever happened. This script cannot distinguish "reformulation was never
attempted" from "reformulation was attempted and returned an unusably-similar
query, triggering the repeat-guard" purely from the `ToolResult` list — that
requires the dispatcher's own debug logs, which this script did not capture (no
logging handler was configured). Noted as a limitation, not resolved here.

## §2. Per-Category Findings

### Category A — single-iteration easy (queries 1–3)

**Probing for:** the easy case firing cleanly on iteration 1, no fetch, no
reformulation.

**Confirmed as designed** — all three gate-passed on the first snippet. But
worth flagging inside the "easy" label: query 2 (GitHub Copilot Individual)
gate-passed on a snippet that never actually states Copilot Individual's
monthly price (it surfaces a per-AI-credit rate, "$0.01 USD," and general
licensing text — no "$10/month" figure appears anywhere in the returned text).
The gate said yes anyway. This is the first instance of a pattern that recurs
throughout §2/§3: the gate accepting content that contains *some* dollar figure
as sufficient, without verifying that figure actually answers the specific
question asked.

### Category B — needs a page fetch (queries 4–6)

**Probing for:** the fetch-and-reevaluate step (§18.4 step 4) actually engaging
on pages that bury the number below the fold.

**Did not fire as designed** — all three gate-passed on the search snippet
alone, no fetch. Not a defect: Brave's snippets for all three (Backblaze B2,
Tesla Model 3, Notion Business) came back with the actual number already
bolded in the result text (`<strong>` tags around `$0.006/GB/month`, `$39,990`,
`$15`/`$18`/`$20` per user/month respectively). This looks like a corpus
assumption that didn't hold up against current live search-snippet quality
rather than a loop defect — worth noting for future corpus revisions, since
this category currently provides zero coverage of the fetch path.

### Category C — needs reformulation (queries 7–9)

**Probing for:** `_reformulate_query()` actually improving a first-miss query.

**Did not fire, same story as Category B** — all three gate-passed on
iteration 1. Same read: today's Brave snippets for AWS/Figma/Ford pricing
already contain a bolded number on the first query. This category currently
provides zero live coverage of the reformulation path either — the only two
reformulation events observed anywhere in this run came from Category F/G
(queries 15–18), not C.

### Category D — ambiguous / multiple valid tiers (queries 10–11) — most important qualitative finding

**Probing for:** whether the loop silently narrows a multi-tier product down
to one number and presents it as *the* answer, vs. surfacing the ambiguity.

Per the methodological caveat above, this pass cannot judge what a synthesized
answer would say — but the raw material both queries hand upstream is
genuinely and clearly multi-valued, which is the precondition for the risk the
corpus describes:

- **Query 10 (Creative Cloud):** the accepted snippet contains **three
  different numbers** with no single one flagged as canonical — "$60/month"
  (bundle, one source), "$54.99/month" (Standard) and "$69.99/month" (Pro,
  same source). A downstream synthesis step reading only this snippet has no
  signal for which of three real, valid numbers to lead with, or that all
  three are legitimate tier prices rather than conflicting reports of one price.
- **Query 11 (iPhone 16):** the accepted snippet is arguably worse — none of
  the three results states a clean base retail price at all. It surfaces a
  trade-in discount range ("$35–$695 off"), a relative delta ("$100 more than
  previous model"), and a *refurbished-market* price range ("$564–$750, Grade
  A condition"). The actual current base retail price ($799, confirmed
  incidentally in query 17's results for the same product) never surfaces in
  query 11's own accepted snippet at all.

**Read:** the loop's gate is binary — "does this text contain concrete
pricing" — with no notion of "is this a single unambiguous answer to the
question asked" or "is this even the right kind of price (retail vs.
refurb vs. discount-off)." Ambiguity resolution is entirely unhandled at the
loop layer and deferred wholesale to synthesis, which this pass didn't
exercise. This is the clearest evidence in the whole run that Category D's
concern is real, even though it manifests one layer below where the corpus
expected to observe it.

### Category E — should fail honestly (queries 12–14) — second most important finding

**Probing for:** the exhaustion path firing on genuinely unanswerable
questions rather than converging on a fabricated-sounding number.

**None of the three exhausted. All three gate-passed on the first snippet** —
the opposite of the designed expectation, and the single biggest surprise of
the whole run:

- **Query 12 (Salesforce "exact enterprise contract price"):** the question
  specifically asks about unlisted, negotiated enterprise pricing. The
  accepted snippet contains only published list-tier prices ("Enterprise
  $175/user/mo," "$50/user/month" for an unrelated add-on product) — none of
  which is the negotiated contract price the question asked about. The gate
  passed because *a* price was present, not because *the requested* price was
  found.
- **Query 13 (discontinued Stadia controller):** more defensible — the
  accepted snippet surfaces real historical Stadia controller pricing ($59.99,
  $99.99 bundle) from a 2021 article. Since the product is genuinely
  discontinued, a historical price is arguably the most honest answer
  available; this is the one Category E query where the gate's "yes" holds up
  reasonably well on inspection, even though "new from Google" (the question's
  framing) is now moot.
- **Query 14 (obscure NFT collection floor price):** the clearest false
  positive in the entire run. The question names no specific collection (by
  design, to test genuine unfindability), and the accepted snippet contains
  **no floor price for any specific collection at all** — only a generic
  definition of what "NFT floor price" means and an aggregate cross-collection
  trading-volume statistic ($3.1M/day across 1,798 collections). The gate said
  "yes, concrete pricing" on content that does not address the question in any
  way.

**Read:** across all three, the gate never once triggered the honest-failure
path it was specifically designed to exercise. Query 14 in particular shows
the gate can pass on content with zero relevant information, purely because
some dollar-figure-shaped text is present nearby. This is a real, structural
weak point: the gate appears to check "does this text contain pricing-shaped
content" rather than "does this text answer the specific thing asked" — the
distinction Category E was built to probe, and the one place it clearly fails.

### Category F — spec lookups, not pricing (queries 15–16)

**Probing for:** whether `_evaluate_pricing_gate()` — a prompt literally
designed around dollar amounts — recognizes a non-price spec number as
"concrete enough," or fails/exhausts on non-price lookups as an
out-of-designed-scope case.

**Not a simple scope-boundary finding either way — a more specific and more
concerning result than the corpus anticipated:**

- **Query 15 (F-150 Lightning payload capacity):** engaged the *full*
  machinery as designed — 2 iterations, a reformulation, a page fetch — so the
  gate is clearly not pricing-*keyword*-only; it doesn't just check for a `$`
  sign. But the fetched page it ultimately gate-passed on (180 words, a
  dealership site) is pure MSRP/fees legal boilerplate — it contains **no
  payload-capacity number whatsoever.** The gate said "yes" on a page that,
  read in full, does not contain the requested fact at all.
- **Query 16 (Model Y battery kWh):** correctly exhausted after all 3
  iterations, including a fetch and two reformulations. Worth noting on the
  reformulation quality itself: the second reformulated query appended
  `site:tesla.com` as literal query text (`"...official | ...specifications
  site:tesla.com"`) — a search-operator syntax that a full-text `web_search`
  MCP call almost certainly does not honor as an actual domain filter (it's
  passed through as plain query text, not a structured search parameter). The
  reformulation model appears to be reaching for a strategy (domain-restrict to
  the manufacturer's own site) that this integration has no mechanism to
  actually execute.

**Read:** this reinforces §2 Category E's finding rather than opening a new,
separate scope-boundary question — the gate is not narrowly pricing-only (it
happily engaged for a non-price spec query), but it shares the same underlying
weakness: passing on content that doesn't actually contain the specific fact
requested. Query 16's honest exhaustion is the better outcome of the two, but
is arguably closer to lucky (three failed attempts) than a demonstration the
gate reliably distinguishes "found it" from "found something dollar/number-
shaped nearby."

### Category G — negative-filter-adjacent / subjective (queries 17–18)

**Probing for:** loop behavior on a question that is unanswerable by search
because it's an opinion, not a fact — a different failure mode than Category
E's objectively-unfindable framing.

Two genuinely different outcomes:

- **Query 17 ("is the new iPhone worth the price"):** the reformulation step
  silently substituted a factual query ("iPhone 16 price plans and models cost
  official Apple store") for the subjective one, then gate-passed on real,
  legitimate iPhone 16 pricing content. This isn't fabrication — the numbers
  surfaced are real — but the loop never engaged with the "worth it"
  (value-judgment) framing at all; reformulation quietly reframed an
  unanswerable subjective question into an answerable factual one and reported
  success. Downstream, this risks presenting "found pricing" as if it settles
  a question it never actually addressed.
- **Query 18 ("is the Model 3 overpriced vs. competitors"):** exhausted
  honestly — a fetch was attempted and still didn't yield gate-passing
  content, and reformulation apparently repeated (or was skipped per the
  repeat-guard — see the §1 footnote), correctly ending in the same
  "ERROR: ... without finding concrete pricing" honest-failure message
  Category E was designed to produce. This is the cleanest, most correct
  outcome in the entire Category E/F/G group.

**Read:** two structurally different behaviors for the same category — one
quietly reframes an unanswerable question into an answerable one (arguably a
subtler problem than a bare failure, since it looks like success), the other
fails honestly. Sample size (n=2) is too small to say which is more typical.

## §3. Cross-Cutting Findings

1. **The pricing/fact gate's dominant failure mode is a false positive, not a
   false negative.** Across Categories A, D, E, and F, the gate repeatedly
   accepted content that contains *some* numeric/dollar-shaped figure as
   sufficient, without verifying that figure is actually responsive to the
   specific question asked (Query 2's unrelated credit-rate; Query 12's
   published tier price standing in for negotiated contract pricing; Query
   14's complete non-answer; Query 15's fetched page with zero payload
   figures). Category E — designed to be the one place false positives would
   be most visible and most costly — showed a 0/3 honest-exhaustion rate in
   this run.
2. **Categories B and C, as currently written, provide no live coverage of the
   fetch and reformulation paths** — current Brave snippet quality for these
   six queries already surfaces a bolded number on the first try. The only
   fetch/reformulation activity observed anywhere came from Categories F and
   G. If continued QA coverage of the fetch/reformulate mechanics specifically
   is wanted, this corpus's B/C queries would need harder targets (or the
   corpus would need queries deliberately chosen against sparser search
   providers/pages).
3. **Category D's ambiguity risk is real and is confirmed to live in the raw
   material the loop hands upstream**, not something this pass can rule in or
   out at the synthesis layer.
4. **Reformulation, when it did engage (queries 15–17), showed mixed quality**
   — one clearly effective narrowing (query 15's trim-level-specific rewrite
   that led to a successful fetch), one strategy the integration can't
   actually execute (query 16's inert `site:` operator), and one silent
   reframing of a subjective question into a factual one (query 17).
5. No go/no-go recommendation is made here, per this repo's diagnostic
   discipline — the data above states what was measured. Whether the gate's
   false-positive tendency is acceptable given the loop's "run down a specific
   fact" framing, or needs a stricter prompt/design change, is a product
   decision.

## §4. Known Limitations of This Pass

- No answer-synthesis step was exercised (see the methodological caveat above)
  — Category D/E/G's ambiguity-acknowledgment and honesty questions are
  answered here only at the raw-material level, not the final-user-facing
  level.
- Query 18's `reformulated=False` classification has a known ambiguity (§1
  footnote) — this script did not configure a logging handler, so the
  dispatcher's own `logger.debug`/`logger.info` calls (which would show the
  exact reformulated query text and repeat-guard firing) were not captured.
  A re-run with `logging.basicConfig(level=logging.DEBUG)` would resolve this
  if the exact reformulation behavior on repeat-guard cases becomes
  important.
- n=2–3 per category is enough to surface behavior patterns, not to establish
  rates with statistical confidence — several of the findings above (Category
  E's 0/3, Category G's 1/2 reframing) are read as qualitative signals, not
  measured proportions.
- Per the requesting instructions, no independent fact-checking of the
  numbers/specs themselves was performed in this pass — that is a separate,
  planned follow-up.

---

*Generated by `diagnostics/research_loop_qa_pass.py`. Raw per-query data:
`diagnostics/research_loop_qa_2026-07-20.csv`.*
