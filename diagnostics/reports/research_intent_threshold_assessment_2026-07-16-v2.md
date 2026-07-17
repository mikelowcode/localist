# `research_intent` Threshold Assessment

**Date:** 2026-07-16
**Script:** `diagnostics/score_research_intent_templates.py`
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Status:** READ-ONLY. `planner.py` unmodified (candidate templates are local to this script).

**Purpose:** Re-test `research_intent` with the v2 template set from
`research_loop_design.md` plus a `_RESEARCH_NEGATIVE_FILTER` pre-filter, after the v1 pass
(`research_intent_threshold_assessment_2026-07-16.md`) found Category E (subjective price
opinion) scored higher than every true positive — a threshold-unfixable collision.

**Candidate templates (8, v2 — verb-anchored, from research_loop_design.md):**

- `look up the pricing for this product`
- `find out how much this subscription costs`
- `search for the price of this item`
- `check the current price of this plan`
- `find the pricing tiers for this service`
- `track down how much this product costs`
- `look up the specs and price for this product`
- `find the cost of this plan per month`

**Negative filter (mirrors `_SEARCH_NEGATIVE_FILTER`):**

- `can't believe how expensive`
- `lot of money for`
- `too expensive`
- `worth the price`

## Category Definitions

| Cat | Description | N | Expected research_intent behavior |
|-----|-------------|---|-----------------------------------|
| T | True positives — real price/spec-lookup requests, paraphrased (not verbatim template overlap) | 10 | should fire |
| L | lookup_request positives, non-pricing — already trigger web_search via LR; must NOT be needlessly upgraded to the research loop | 5 | should NOT fire |
| L-price-adj | "current stock price" — a lookup_request template that IS itself a price fact | 1 | ambiguous by design, reported separately, not counted as FP/TP |
| K | knowledge_request_open positives, non-pricing | 4 | should NOT fire |
| F | freshness_request positives, non-pricing | 3 | should NOT fire |
| E | pricing-adjacent but subjective/non-lookup (opinion, no concrete fact to extract) | 4 | should NOT fire |
| G | generic conversational, no search semantics | 4 | should NOT fire |

## Full Score Table (all groups)

`filtered` = matched `_RESEARCH_NEGATIVE_FILTER`; RI score shown for visibility but excluded
from the FP-pool analysis below (see Negative Filter section).

| Cat | Utterance | ESA | LR | KRO | FR | RI | filtered |
|-----|-----------|----:|----:|----:|----:|----:|:--------:|
| T | How much does the Tesla Model 3 cost? | 0.4379 | 0.5709 | 0.4700 | 0.4654 | **0.6724** |  |
| T | What's the price of the new iPhone? | 0.4423 | 0.7119 | 0.4692 | 0.4892 | **0.7023** |  |
| T | Can you find out how much AWS charges for S3 storage? | 0.4948 | 0.5636 | 0.4993 | 0.4374 | **0.6983** |  |
| T | What are the pricing tiers for Notion? | 0.4728 | 0.5479 | 0.4885 | 0.4751 | **0.7900** |  |
| T | Look up the specs on the RTX 4090 | 0.6348 | 0.6423 | 0.5610 | 0.4732 | **0.7736** |  |
| T | Find me the cost of a one-bedroom apartment in Austin | 0.5687 | 0.5839 | 0.5162 | 0.4373 | **0.6790** |  |
| T | What does ChatGPT Plus cost per month? | 0.4693 | 0.5337 | 0.4731 | 0.4318 | **0.7376** |  |
| T | Track down pricing for Salesforce Enterprise | 0.5939 | 0.5964 | 0.5099 | 0.4636 | **0.7283** |  |
| T | Can you check what a Peloton subscription costs? | 0.5168 | 0.6027 | 0.5089 | 0.4495 | **0.8414** |  |
| T | What's the going rate for a plumber in this area? | 0.4667 | 0.5172 | 0.4735 | 0.4560 | **0.6342** |  |
| L | Can you look up the release date for this? | 0.6580 | 0.9940 | 0.6464 | 0.6435 | **0.6532** |  |
| L | Could you look up what year this happened? | 0.6675 | 0.9938 | 0.6081 | 0.5862 | **0.5587** |  |
| L | Can you look up information about the latest Apple pro… | 0.6130 | 0.9955 | 0.5535 | 0.6149 | **0.6748** |  |
| L | Can you look up Apple's price hike for the MacBook Neo… | 0.5424 | 0.7653 | 0.4684 | 0.4938 | **0.6483** |  |
| L | Can you look up their next-generation in-house Microso… | 0.5785 | 0.6522 | 0.4935 | 0.5053 | **0.5761** |  |
| L-price-adj | Could you find out the current stock price for me? | 0.5767 | 0.9940 | 0.5706 | 0.5834 | **0.6876** |  |
| K | What is this? | 0.6818 | 0.7606 | 0.9893 | 0.7269 | **0.6456** |  |
| K | Tell me about this company | 0.6091 | 0.6883 | 0.7590 | 0.5369 | **0.6143** |  |
| K | Explain how blockchain works | 0.5625 | 0.6032 | 0.6974 | 0.4904 | **0.5223** |  |
| K | What do you know about this? | 0.6625 | 0.8677 | 0.9929 | 0.7387 | **0.6105** |  |
| F | What's the latest on this? | 0.6232 | 0.7157 | 0.7483 | 0.9900 | **0.5688** |  |
| F | Is there anything new about this? | 0.6205 | 0.7117 | 0.7427 | 0.9924 | **0.5321** |  |
| F | What's the current status of this project? | 0.5030 | 0.5459 | 0.5959 | 0.8776 | **0.6185** |  |
| E | Is this too expensive for me? | 0.4897 | 0.5701 | 0.5842 | 0.5239 | **0.7079** | Y |
| E | Do you think this is worth the price? | 0.5181 | 0.5846 | 0.6251 | 0.5353 | **0.7427** | Y |
| E | I can't believe how expensive rent is these days | 0.3722 | 0.4611 | 0.4398 | 0.3698 | **0.5144** | Y |
| E | That seems like a lot of money for what you get | 0.4838 | 0.5251 | 0.5167 | 0.4550 | **0.6215** | Y |
| G | Can you help me with this? | 0.6187 | 0.6146 | 0.7517 | 0.6426 | **0.5355** |  |
| G | What's up? | 0.6436 | 0.6931 | 0.7493 | 0.7086 | **0.5532** |  |
| G | How are you doing today? | 0.4723 | 0.5713 | 0.5014 | 0.6243 | **0.5158** |  |
| G | Thanks, that's helpful | 0.5494 | 0.5432 | 0.5497 | 0.4656 | **0.4427** |  |

## Threshold Trade-off

FP pool = categories L + K + F + E + G, MINUS any utterance intercepted by
`_RESEARCH_NEGATIVE_FILTER` (reported separately below). L-price-adj is excluded from both
pools and reported separately below.

| Threshold | T survivors (of 10) | FP pool fires (of 16) |
|:---------:|:----------------------:|:--------------------------:|
| 0.45 | 10/10 | 15/16 |
| 0.50 | 10/10 | 15/16 |
| 0.55 | 10/10 | 11/16 |
| 0.60 | 10/10 | 7/16 |
| 0.65 | 9/10 | 2/16 |
| 0.70 | 6/10 | 0/16 |

**Full separation** (fine 0.005 grid, 0.300–0.950): a threshold where all T survive AND
zero FP-pool items fire.

No threshold in the scanned range achieves full separation.

## Category T — Per-Utterance research_intent Scores

Lowest T score is the load-bearing constraint: any threshold above it starts losing
true positives.

| Utterance | RI score |
|-----------|---------:|
| What's the going rate for a plumber in this area? | 0.6342 |
| How much does the Tesla Model 3 cost? | 0.6724 |
| Find me the cost of a one-bedroom apartment in Austin | 0.6790 |
| Can you find out how much AWS charges for S3 storage? | 0.6983 |
| What's the price of the new iPhone? | 0.7023 |
| Track down pricing for Salesforce Enterprise | 0.7283 |
| What does ChatGPT Plus cost per month? | 0.7376 |
| Look up the specs on the RTX 4090 | 0.7736 |
| What are the pricing tiers for Notion? | 0.7900 |
| Can you check what a Peloton subscription costs? | 0.8414 |

**Minimum T score:** 0.6342

## FP Pool — Per-Utterance research_intent Scores (highest first)

| Cat | Utterance | RI score |
|-----|-----------|---------:|
| L | Can you look up information about the latest Apple pro… | 0.6748 |
| L | Can you look up the release date for this? | 0.6532 |
| L | Can you look up Apple's price hike for the MacBook Neo… | 0.6483 |
| K | What is this? | 0.6456 |
| F | What's the current status of this project? | 0.6185 |
| K | Tell me about this company | 0.6143 |
| K | What do you know about this? | 0.6105 |
| L | Can you look up their next-generation in-house Microso… | 0.5761 |
| F | What's the latest on this? | 0.5688 |
| L | Could you look up what year this happened? | 0.5587 |
| G | What's up? | 0.5532 |
| G | Can you help me with this? | 0.5355 |
| F | Is there anything new about this? | 0.5321 |
| K | Explain how blockchain works | 0.5223 |
| G | How are you doing today? | 0.5158 |
| G | Thanks, that's helpful | 0.4427 |

**Maximum FP-pool score:** 0.6748

## L-price-adjacent (reported separately, not scored as FP or TP)

`Could you find out the current stock price for me?` → RI = 0.6876

This utterance is a verbatim lookup_request template ("could you find out the current
stock price for me") that also happens to name a concrete price fact. Whether it *should*
fire research_intent is a product judgment call (does a stock-price lookup benefit from the
search→evaluate→fetch loop, or does the answer already appear in a plain search snippet?),
not something this diagnostic resolves. Reported for visibility only.

## Negative Filter — Regression Reference

Utterances intercepted by `_RESEARCH_NEGATIVE_FILTER` before they could contribute a
research_intent score to the FP-pool analysis. Scored here for visibility only, same as
Category B in `score_lookup_request_templates.py` — no re-decision is made; this confirms
what score each would have carried had the filter not caught it.

| Cat | Utterance | Raw RI score (pre-filter) |
|-----|-----------|---------------------------:|
| E | Do you think this is worth the price? | 0.7427 |
| E | Is this too expensive for me? | 0.7079 |
| E | That seems like a lot of money for what you get | 0.6215 |
| E | I can't believe how expensive rent is these days | 0.5144 |

4/4 Category E utterances matched the filter — the entire category is intercepted pre-gate in production, so none of it reaches the FP-pool analysis below.

## Post-Filter Separation — Remaining Categories

With the negative filter applied, the FP pool consists of: F, G, K, L.

Top remaining offender: Category L, "Can you look up information about the latest Apple products?" → RI = 0.6748, which exceeds the T-minimum (0.6342) and is higher than 2/10 Category T scores.

**The remaining categories are still not threshold-separable from T** on their own — the negative filter resolved Category E, but at least one L/K/F/G utterance still collides with the true-positive range at the score level, not just near a boundary.

## Summary

**No clean separation**: minimum T score (0.6342) is below the maximum FP-pool score (0.6748). No single threshold achieves both 10/10 T survival and 0/16 FP-pool false positives — see the trade-off table above for the actual cost at each candidate value.

No threshold recommendation is made here — per this repo's established diagnostic
discipline (see `lookup_request_margin_assessment_2026-06-28.md`), the data above states
the cost at each candidate threshold; the choice of `_RESEARCH_INTENT_THRESHOLD` is a
product decision made from this table, not something this script decides.

Per the implementation sketch, `research_intent` should ship shadow-only
(scored and logged, excluded from `_SEMANTIC_GATE_THRESHOLDS`, threshold read from
`LOCALIST_RESEARCH_INTENT_THRESHOLD` defaulting to `float("inf")`) regardless of which
value is picked from this table, until it has been observed against live traffic — the
same rollout discipline already applied to `LOCALIST_TOOL_FALLBACK_CLASSIFIER`.

---

*Generated by `diagnostics/score_research_intent_templates.py` (v2 template set + negative
filter). Compare against `research_intent_threshold_assessment_2026-07-16.md` (v1).*
