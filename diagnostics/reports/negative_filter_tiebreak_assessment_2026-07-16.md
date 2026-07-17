# Negative-Filter Tie-Break — Accuracy & Reachability Assessment

**Date:** 2026-07-16
**Script:** `diagnostics/score_negative_filter_tiebreak.py`
**Runtime:** backend='ollama' chat_model='gemma4:31b-cloud' (LIVE, no mocks)
**Embedding model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine
**Status:** READ-ONLY. `planner.py` unmodified — the tie-break prompt is local to this script.

**Purpose:** Validate `_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT` from the "Negative-filter
redesign" addendum before wiring `_resolve_negative_filter_conflict` into planner.py — same
"don't hand-pick, measure" discipline as the research_intent template/threshold diagnostics.

**Tie-break system prompt under test:**

> You are a routing classifier, not a conversational assistant. The instruction below matched a known phrase pattern that is usually NOT a request to search for or look up information (e.g. a subjective opinion, a greeting, an identity question) but also scored high on semantic similarity to real search-intent phrasing — an ambiguous case. Decide which one it actually is. Respond with exactly one word: lookup or other.

## §1. Accuracy — Full Test Set

32/37 correct (86.5%).

Categories T/L/L-price-adj (ground truth: lookup) and K/F/E/G (ground truth: other) are
reused verbatim from `research_intent_threshold_assessment_2026-07-16-v2.md`. Category A is
new: utterances that combine a `_RESEARCH_NEGATIVE_FILTER` substring with an explicit lookup
verb — the actual case the tie-break exists to resolve correctly.

| Cat | Utterance | Truth | Predicted | Raw response | Correct |
|-----|-----------|:-----:|:---------:|---------------|:-------:|
| T | How much does the Tesla Model 3 cost? | lookup | lookup | 'lookup' | Y |
| T | What's the price of the new iPhone? | lookup | lookup | 'lookup' | Y |
| T | Can you find out how much AWS charges for S3 storage? | lookup | lookup | 'lookup' | Y |
| T | What are the pricing tiers for Notion? | lookup | lookup | 'lookup' | Y |
| T | Look up the specs on the RTX 4090 | lookup | lookup | 'lookup' | Y |
| T | Find me the cost of a one-bedroom apartment in Austin | lookup | lookup | 'lookup' | Y |
| T | What does ChatGPT Plus cost per month? | lookup | lookup | 'lookup' | Y |
| T | Track down pricing for Salesforce Enterprise | lookup | lookup | 'lookup' | Y |
| T | Can you check what a Peloton subscription costs? | lookup | lookup | 'lookup' | Y |
| T | What's the going rate for a plumber in this area? | lookup | lookup | 'lookup' | Y |
| L | Can you look up the release date for this? | lookup | lookup | 'lookup' | Y |
| L | Could you look up what year this happened? | lookup | lookup | 'lookup' | Y |
| L | Can you look up information about the latest Apple products? | lookup | lookup | 'lookup' | Y |
| L | Can you look up Apple's price hike for the MacBook Neo and … | lookup | lookup | 'lookup' | Y |
| L | Can you look up their next-generation in-house Microsoft AI… | lookup | lookup | 'lookup' | Y |
| L-price-adj | Could you find out the current stock price for me? | lookup | lookup | 'lookup' | Y |
| K | What is this? | other | other | 'other' | Y |
| K | Tell me about this company | other | lookup | 'lookup' | **N** |
| K | Explain how blockchain works | other | lookup | 'lookup' | **N** |
| K | What do you know about this? | other | other | 'other' | Y |
| F | What's the latest on this? | other | lookup | 'lookup' | **N** |
| F | Is there anything new about this? | other | lookup | 'lookup' | **N** |
| F | What's the current status of this project? | other | lookup | 'lookup' | **N** |
| E | Is this too expensive for me? | other | other | 'other' | Y |
| E | Do you think this is worth the price? | other | other | 'other' | Y |
| E | I can't believe how expensive rent is these days | other | other | 'other' | Y |
| E | That seems like a lot of money for what you get | other | other | 'other' | Y |
| G | Can you help me with this? | other | other | 'other' | Y |
| G | What's up? | other | other | 'other' | Y |
| G | How are you doing today? | other | other | 'other' | Y |
| G | Thanks, that's helpful | other | other | 'other' | Y |
| A | Can you look up if this laptop is worth the price? | lookup | lookup | 'lookup' | Y |
| A | Could you search for reviews on whether this is worth the p… | lookup | lookup | 'lookup' | Y |
| A | Find out if people think this hotel is too expensive for wh… | lookup | lookup | 'lookup' | Y |
| A | Look up whether this subscription is worth the price compar… | lookup | lookup | 'lookup' | Y |
| A | Can you check if this is too expensive compared to other pl… | lookup | lookup | 'lookup' | Y |
| A | Search online to see if this is worth the price | lookup | lookup | 'lookup' | Y |

## §2. Confusion Matrix

|  | Predicted: lookup | Predicted: other |
|---|:---:|:---:|
| **Truth: lookup** | 22 (TP) | 0 (FN) |
| **Truth: other** | 5 (FP) | 10 (TN) |

FN = a real lookup request wrongly classified as "other" — the tie-break would let the
negative filter incorrectly suppress a genuine search request that happened to use
filter-listed phrasing.

FP = a genuine non-lookup utterance wrongly classified as "lookup" — if this utterance also
matched a negative filter, the tie-break would override the filter and reintroduce the exact
collision the filter was built to prevent. IMPORTANT CAVEAT, expanded in §3: most FP/error
categories below (K, F) never match `_SEARCH_NEGATIVE_FILTER`/`_RESEARCH_NEGATIVE_FILTER` in
the first place, so `_resolve_negative_filter_conflict` is never invoked on them in
production regardless of what this classifier would say about them — their errors here are
an out-of-distribution robustness probe, not evidence of a live bug. §3 cross-references which
errors land on filter-matched (operationally reachable) utterances.

| Category | N | Correct | Accuracy |
|----------|---|---------|----------|
| T | 10 | 10 | 100.0% |
| L | 5 | 5 | 100.0% |
| L-price-adj | 1 | 1 | 100.0% |
| K | 4 | 2 | 50.0% |
| F | 3 | 0 | 0.0% |
| E | 4 | 4 | 100.0% |
| G | 4 | 4 | 100.0% |
| A | 6 | 6 | 100.0% |

## §3. Reachability — Would the Tie-Break Call Actually Fire?

The redesign only invokes `_resolve_negative_filter_conflict` when a negative-filter
substring matched AND at least one gated group's score clears its threshold (explicit_
search_action ≥0.72, lookup_request ≥0.60 always; research_intent ≥0.65 only when
`LOCALIST_RESEARCH_LOOP_ENABLED=true`). This section checks the sketch's own claim that the
call "is expected to fire noticeably more often [than the P5/P6 classifiers], not
negligibly" against live embedding scores, rather than accepting it as asserted.

**11/37** test utterances matched `_RESEARCH_NEGATIVE_FILTER` or
`_SEARCH_NEGATIVE_FILTER` at all — categories A, E, G. Category E and A
were deliberately constructed to match; the rest weren't, so an unplanned match (e.g. an
existing `_SEARCH_NEGATIVE_FILTER` greeting/identity phrase colliding with one of these
utterances by coincidence) is itself worth surfacing, not filtering out of this report.

**Accuracy restricted to filter-matched (operationally reachable) utterances: 11/11 (100.0%).** Zero errors landed on an utterance the tie-break would actually be invoked on — every classification error in §2 is confined to Category K/F, which never reach `_resolve_negative_filter_conflict` because they never match either negative filter.

- **`LOCALIST_RESEARCH_LOOP_ENABLED=false`** (current default): **4/11** filter-matched utterances would reach the tie-break call (i.e. also clear explicit_search_action ≥0.72 or lookup_request ≥0.60).
- **`LOCALIST_RESEARCH_LOOP_ENABLED=true`**: **8/11** filter-matched utterances would reach the tie-break call (adds research_intent ≥0.65 to the gated set).

| Cat | Utterance | Filter matched on | ESA | LR | RI | Reachable (flag OFF) | Reachable (flag ON) |
|-----|-----------|--------------------|----:|----:|----:|:---:|:---:|
| E | Is this too expensive for me? | `too expensive` | 0.4897 | 0.5701 | 0.7079 | n | Y |
| E | Do you think this is worth the price? | `worth the price` | 0.5181 | 0.5846 | 0.7427 | n | Y |
| E | I can't believe how expensive rent is these … | `can't believe how expensive` | 0.3722 | 0.4611 | 0.5144 | n | n |
| E | That seems like a lot of money for what you … | `lot of money for` | 0.4838 | 0.5251 | 0.6215 | n | n |
| G | What's up? | `what's up` | 0.6436 | 0.6931 | 0.5532 | Y | Y |
| A | Can you look up if this laptop is worth the … | `worth the price` | 0.6147 | 0.6442 | 0.7444 | Y | Y |
| A | Could you search for reviews on whether this… | `worth the price` | 0.6036 | 0.6008 | 0.7504 | Y | Y |
| A | Find out if people think this hotel is too e… | `too expensive` | 0.5109 | 0.5139 | 0.6077 | n | n |
| A | Look up whether this subscription is worth t… | `worth the price` | 0.5824 | 0.5987 | 0.8329 | n | Y |
| A | Can you check if this is too expensive compa… | `too expensive` | 0.5097 | 0.5311 | 0.7534 | n | Y |
| A | Search online to see if this is worth the pr… | `worth the price` | 0.7384 | 0.6750 | 0.8181 | Y | Y |

**Partial support for the sketch's reachability claim.** 4/11 (flag off) and 8/11 (flag on) filter-matched utterances would reach the tie-break call — a real minority-not-negligible rate (36% / 73% of filter-matched turns), not the "fires noticeably more often" framing the sketch asserted, but not zero either. Notably one unplanned match — a Category G item (a pre-existing `_SEARCH_NEGATIVE_FILTER` greeting/identity phrase, not one of the new research-specific ones) — is independently reachable under both configs because it clears `lookup_request` on its own; see the table above.

## §4. Summary

**Accuracy (full test set):** 32/37 (86.5%). 5 false positive(s) and 0 false negative(s) overall — see §2.

**Accuracy (operationally reachable subset — filter-matched utterances only):** 11/11 (100.0%). This is the number that actually matters for whether wiring `_resolve_negative_filter_conflict` into planner.py is safe: every error in the full-set number above falls on Category K/F utterances that never reach the tie-break call in production (see §3), so they don't represent live risk.

**Partial support for the sketch's reachability claim.** 4/11 (flag off) and 8/11 (flag on) filter-matched utterances would reach the tie-break call — a real minority-not-negligible rate (36% / 73% of filter-matched turns), not the "fires noticeably more often" framing the sketch asserted, but not zero either. Notably one unplanned match — a Category G item (a pre-existing `_SEARCH_NEGATIVE_FILTER` greeting/identity phrase, not one of the new research-specific ones) — is independently reachable under both configs because it clears `lookup_request` on its own; see the table above.

No go/no-go recommendation is made here — per this repo's established diagnostic discipline,
the data above states what was measured; whether this accuracy/reachability profile is good
enough to wire `_resolve_negative_filter_conflict` into planner.py is a product decision.

---

*Generated by `diagnostics/score_negative_filter_tiebreak.py`.*
