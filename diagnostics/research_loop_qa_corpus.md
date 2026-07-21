# Research loop answer-quality QA corpus (Ollama Cloud, Gemma 4-31B)

Companion to `docs/architecture/18-research-loop.md` §18.8's open item:
"No live human QA of the research loop's actual answer quality... beyond
the specific queries exercised during this session's live testing." This
corpus is designed to close that gap with a deliberately wider,
categorized set, run via `/research` so every query actually invokes the
loop regardless of the ambient semantic gate.

Each query below is chosen for a specific reason — noted so the person
running/reading results knows what failure mode each one is probing for,
not just "did it find a number."

## Category A — single-iteration easy (snippet should contain the answer directly)

1. `/research What is the monthly price of Netflix's Standard plan?`
2. `/research What does GitHub Copilot Individual cost per month?`
3. `/research What is the base price of a 2024 Honda Civic LX?`

*Probing for:* the easy case — gate should pass on iteration 1, no fetch
needed. If any of these need reformulation or fail, that's a signal
something's off with the basic gate/query-derivation logic, not just hard
queries.

## Category B — needs a page fetch (snippet alone won't contain it)

4. `/research What is the storage price per GB for Backblaze B2?`
5. `/research What's the price of the entry-level Tesla Model 3 in the US?`
6. `/research What does the Notion Business plan cost per user per month?`

*Probing for:* the fetch-and-reevaluate step (§18.4 step 4) actually
working — these are chosen because pricing pages for these products
often bury the exact number below a marketing snippet's fold.

## Category C — needs reformulation (first query likely to miss)

7. `/research How much does it cost to run a t3.medium EC2 instance in us-east-1?`
8. `/research What's the per-seat price for Figma's Enterprise tier?`
9. `/research What does a Ford F-150 XLT crew cab start at?`

*Probing for:* `_reformulate_query()` actually improving the query rather
than just burning an iteration — worth comparing iteration 1's query vs.
the reformulated one in the transcript.

## Category D — ambiguous / multiple valid tiers (no single correct number)

10. `/research What does Adobe Creative Cloud cost?` (there are several
    plans — All Apps, single-app, student — no single right answer)
11. `/research What's the price of an iPhone 16?` (multiple storage
    tiers/colors, same base question ambiguity as the chart tool's
    5-department pie-chart boundary case)

*Probing for:* whether the loop's gate accepts the FIRST plausible number
it finds and reports it as *the* answer (a real correctness risk — a
"Creative Cloud costs $X" answer that silently picked one tier out of
several is subtly wrong even though a number was found), or whether the
model's final answer acknowledges the ambiguity. This is the most
important category for judging *answer quality* rather than just
*loop mechanics* — a technically-successful gate-pass can still produce a
misleading answer here.

## Category E — should fail honestly (no findable/stable answer)

12. `/research What is the exact enterprise contract price for Salesforce?`
    (deliberately unlisted/negotiated pricing — should NOT converge on a
    fabricated number)
13. `/research What does the discontinued Google Stadia controller cost
    new from Google?` (discontinued product, no longer sold)
14. `/research What is the current spot price of a specific obscure
    NFT collection's floor price?` (rapidly-changing, likely genuinely
    unfindable via a clean search snippet)

*Probing for:* the exhaustion path (§18.4's synthetic trailing
`ToolResult`) firing correctly and the final answer honestly reporting
"couldn't find this" rather than confidently fabricating a plausible-
sounding number — this is the correctness failure mode that matters most
(a fabricated enterprise price is worse than an admitted failure).

## Category F — spec lookups, not pricing (tests the loop isn't pricing-only in practice even though it's named/designed around pricing)

15. `/research What is the max payload capacity of a Ford F-150 Lightning?`
16. `/research What is the battery capacity in kWh of a Tesla Model Y
    Long Range?`

*Probing for:* `_evaluate_pricing_gate()`'s prompt (per §18.4, literally
named/designed around pricing) — does it correctly recognize a spec
number as "concrete enough" to pass the gate, or does it only recognize
dollar amounts specifically and fail/exhaust on non-price factual
lookups that would benefit from the same search→evaluate→fetch loop?
This is worth flagging clearly either way: if it turns out the gate is
pricing-specific and these fail/exhaust, that's not necessarily a bug —
it may just mean the feature is narrower than its "answer quality" framing
suggests, worth documenting as a scope boundary rather than a defect.

## Category G — negative-filter-adjacent phrasing (subjective, not a real lookup)

17. `/research Do you think the new iPhone is worth the price?`
18. `/research Is the Tesla Model 3 overpriced compared to competitors?`

*Probing for:* since `/research` bypasses the Planner-level negative
filter entirely (it forces `tools_to_call=["research"]` directly, no
semantic gating involved), these will always invoke the loop even though
in normal (non-slash-command) usage they're designed to be filtered out
as subjective-opinion phrasing, not real price lookups. Worth seeing what
the loop actually does when asked a genuinely unanswerable-by-search
question — does it exhaust honestly, or try to force a numeric-sounding
answer out of opinion content it fetches? Different failure mode than
Category E (which is objectively unfindable) — this is subjectively
unanswerable.

---

## What to capture per query

- Iteration count actually used (1–3, or exhausted)
- Whether a fetch occurred (step 4) or the gate passed on the snippet alone
- Whether the query was reformulated, and if so, old vs. new query text
- The final answer text verbatim
- Elapsed time
- Your own quick read: did the final answer's claimed number/fact look
  plausible on its face (not yet independently verified — that's a
  separate pass)

Once you have raw results back, I'll independently verify a sample of
the "found pricing/spec" claims (especially Categories A–C, and both
Category D queries) against real current sources using my own web search
— an actual ground-truth check, not just "the loop said it found
something." Category E/F/G are more about behavior-under-honest-failure
than verifiable correctness, so I'll read those qualitatively rather than
fact-check a number that shouldn't exist.
