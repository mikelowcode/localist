# `research_intent` Threshold Assessment

**Date:** 2026-07-16
**Script:** `diagnostics/score_research_intent_templates.py`
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Status:** READ-ONLY. `planner.py` unmodified (candidate templates are local to this script).

**Purpose:** Establish a defensible `_RESEARCH_INTENT_THRESHOLD` for the `research_intent`
semantic group proposed in the research-loop implementation sketch, before it is added to
`_SEARCH_INTENT_TEMPLATES` / wired into `_priority3_tool`'s web_search→research upgrade.

**Candidate templates (8, verbatim from the sketch):**

- `find the pricing for this`
- `what does this cost`
- `how much does this cost`
- `look up the pricing plans for this`
- `find out how much this costs`
- `what are the pricing tiers for this`
- `track down the price of this`
- `find the specs for this product`

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

| Cat | Utterance | ESA | LR | KRO | FR | RI |
|-----|-----------|----:|----:|----:|----:|----:|
| T | How much does the Tesla Model 3 cost? | 0.4379 | 0.5709 | 0.4700 | 0.4654 | **0.7321** |
| T | What's the price of the new iPhone? | 0.4423 | 0.7119 | 0.4692 | 0.4892 | **0.7104** |
| T | Can you find out how much AWS charges for S3 storage? | 0.4948 | 0.5636 | 0.4993 | 0.4374 | **0.7194** |
| T | What are the pricing tiers for Notion? | 0.4728 | 0.5479 | 0.4885 | 0.4751 | **0.8429** |
| T | Look up the specs on the RTX 4090 | 0.6348 | 0.6423 | 0.5610 | 0.4732 | **0.7648** |
| T | Find me the cost of a one-bedroom apartment in Austin | 0.5687 | 0.5839 | 0.5162 | 0.4373 | **0.6999** |
| T | What does ChatGPT Plus cost per month? | 0.4693 | 0.5337 | 0.4731 | 0.4318 | **0.6908** |
| T | Track down pricing for Salesforce Enterprise | 0.5939 | 0.5964 | 0.5099 | 0.4636 | **0.7467** |
| T | Can you check what a Peloton subscription costs? | 0.5168 | 0.6027 | 0.5089 | 0.4495 | **0.7289** |
| T | What's the going rate for a plumber in this area? | 0.4667 | 0.5172 | 0.4735 | 0.4560 | **0.7238** |
| L | Can you look up the release date for this? | 0.6580 | 0.9940 | 0.6464 | 0.6435 | **0.6755** |
| L | Could you look up what year this happened? | 0.6675 | 0.9938 | 0.6081 | 0.5862 | **0.5893** |
| L | Can you look up information about the latest Apple pro… | 0.6130 | 0.9955 | 0.5535 | 0.6149 | **0.6335** |
| L | Can you look up Apple's price hike for the MacBook Neo… | 0.5424 | 0.7653 | 0.4684 | 0.4938 | **0.6281** |
| L | Can you look up their next-generation in-house Microso… | 0.5785 | 0.6522 | 0.4935 | 0.5053 | **0.5587** |
| L-price-adj | Could you find out the current stock price for me? | 0.5767 | 0.9940 | 0.5706 | 0.5834 | **0.6780** |
| K | What is this? | 0.6818 | 0.7606 | 0.9893 | 0.7269 | **0.7230** |
| K | Tell me about this company | 0.6091 | 0.6883 | 0.7590 | 0.5369 | **0.6109** |
| K | Explain how blockchain works | 0.5625 | 0.6032 | 0.6974 | 0.4904 | **0.5438** |
| K | What do you know about this? | 0.6625 | 0.8677 | 0.9929 | 0.7387 | **0.6893** |
| F | What's the latest on this? | 0.6232 | 0.7157 | 0.7483 | 0.9900 | **0.5920** |
| F | Is there anything new about this? | 0.6205 | 0.7117 | 0.7427 | 0.9924 | **0.5746** |
| F | What's the current status of this project? | 0.5030 | 0.5459 | 0.5959 | 0.8776 | **0.5628** |
| E | Is this too expensive for me? | 0.4897 | 0.5701 | 0.5842 | 0.5239 | **0.8266** |
| E | Do you think this is worth the price? | 0.5181 | 0.5846 | 0.6251 | 0.5353 | **0.8451** |
| E | I can't believe how expensive rent is these days | 0.3722 | 0.4611 | 0.4398 | 0.3698 | **0.5452** |
| E | That seems like a lot of money for what you get | 0.4838 | 0.5251 | 0.5167 | 0.4550 | **0.6828** |
| G | Can you help me with this? | 0.6187 | 0.6146 | 0.7517 | 0.6426 | **0.6348** |
| G | What's up? | 0.6436 | 0.6931 | 0.7493 | 0.7086 | **0.6191** |
| G | How are you doing today? | 0.4723 | 0.5713 | 0.5014 | 0.6243 | **0.4962** |
| G | Thanks, that's helpful | 0.5494 | 0.5432 | 0.5497 | 0.4656 | **0.4837** |

## Threshold Trade-off

FP pool = categories L + K + F + E + G (all utterances research_intent should NOT fire on).
L-price-adj is excluded from both pools and reported separately below.

| Threshold | T survivors (of 10) | FP pool fires (of 20) |
|:---------:|:----------------------:|:--------------------------:|
| 0.45 | 10/10 | 20/20 |
| 0.50 | 10/10 | 18/20 |
| 0.55 | 10/10 | 16/20 |
| 0.60 | 10/10 | 11/20 |
| 0.65 | 10/10 | 6/20 |
| 0.70 | 8/10 | 3/20 |

**Full separation** (fine 0.005 grid, 0.300–0.950): a threshold where all T survive AND
zero FP-pool items fire.

No threshold in the scanned range achieves full separation.

## Category T — Per-Utterance research_intent Scores

Lowest T score is the load-bearing constraint: any threshold above it starts losing
true positives.

| Utterance | RI score |
|-----------|---------:|
| What does ChatGPT Plus cost per month? | 0.6908 |
| Find me the cost of a one-bedroom apartment in Austin | 0.6999 |
| What's the price of the new iPhone? | 0.7104 |
| Can you find out how much AWS charges for S3 storage? | 0.7194 |
| What's the going rate for a plumber in this area? | 0.7238 |
| Can you check what a Peloton subscription costs? | 0.7289 |
| How much does the Tesla Model 3 cost? | 0.7321 |
| Track down pricing for Salesforce Enterprise | 0.7467 |
| Look up the specs on the RTX 4090 | 0.7648 |
| What are the pricing tiers for Notion? | 0.8429 |

**Minimum T score:** 0.6908

## FP Pool — Per-Utterance research_intent Scores (highest first)

| Cat | Utterance | RI score |
|-----|-----------|---------:|
| E | Do you think this is worth the price? | 0.8451 |
| E | Is this too expensive for me? | 0.8266 |
| K | What is this? | 0.7230 |
| K | What do you know about this? | 0.6893 |
| E | That seems like a lot of money for what you get | 0.6828 |
| L | Can you look up the release date for this? | 0.6755 |
| G | Can you help me with this? | 0.6348 |
| L | Can you look up information about the latest Apple pro… | 0.6335 |
| L | Can you look up Apple's price hike for the MacBook Neo… | 0.6281 |
| G | What's up? | 0.6191 |
| K | Tell me about this company | 0.6109 |
| F | What's the latest on this? | 0.5920 |
| L | Could you look up what year this happened? | 0.5893 |
| F | Is there anything new about this? | 0.5746 |
| F | What's the current status of this project? | 0.5628 |
| L | Can you look up their next-generation in-house Microso… | 0.5587 |
| E | I can't believe how expensive rent is these days | 0.5452 |
| K | Explain how blockchain works | 0.5438 |
| G | How are you doing today? | 0.4962 |
| G | Thanks, that's helpful | 0.4837 |

**Maximum FP-pool score:** 0.8451

## L-price-adjacent (reported separately, not scored as FP or TP)

`Could you find out the current stock price for me?` → RI = 0.6780

This utterance is a verbatim lookup_request template ("could you find out the current
stock price for me") that also happens to name a concrete price fact. Whether it *should*
fire research_intent is a product judgment call (does a stock-price lookup benefit from the
search→evaluate→fetch loop, or does the answer already appear in a plain search snippet?),
not something this diagnostic resolves. Reported for visibility only.

## Collision Analysis — Category E (subjective price opinion)

The maximum FP-pool score belongs to Category E, not L/K/F/G:

- `Do you think this is worth the price?` → RI = 0.8451
- `Is this too expensive for me?` → RI = 0.8266
- `That seems like a lot of money for what you get` → RI = 0.6828
- `I can't believe how expensive rent is these days` → RI = 0.5452

The top Category E score (0.8451, "Do you think this is worth the price?") exceeds 10/10 of the Category T true-positive scores — higher than most genuine price-lookup requests. This is the same failure shape documented for `lookup_request` in the 2026-06-25 incident (planner.py's own comment on that group): a threshold cannot separate these two categories because the higher-scoring negative sits *above* most positives, not just close to the boundary. Raising the threshold to exclude Category E only sheds more true positives faster than it sheds this one.

Excluding Category E, the next-highest FP-pool score drops to 0.7230 (Category K, "What is this?") — still above the T-minimum of 0.6908, meaning the L/K/F/G collisions alone would not be threshold-fixable on their own.

**Implication for template design, not just threshold value**: the current 8 candidate templates all name cost/pricing vocabulary directly ("cost", "price", "pricing") without any framing that distinguishes a *lookup request* (find a number) from a *value judgment* (is the number acceptable). Embedding models trained on general text correlate "cost"/"price" tokens with both framings equally. A template rework aimed at the lookup-vs-judgment distinction — e.g. adding an explicit lookup/search verb ("find", "look up", "search for") to every template rather than mixing pure cost phrasing ("what does this cost") with lookup phrasing ("find the pricing for this") — is a more promising fix than threshold tuning alone, mirroring how the 2026-06-28 `lookup_request` candidate-rework diagnostic resolved an analogous unfixable collision by changing templates rather than the threshold. Not attempted here — this diagnostic's scope is the threshold question for the templates as given in the sketch; template rework is a follow-up if Category E's collision is judged unacceptable.

## Summary

**No clean separation**: minimum T score (0.6908) is below the maximum FP-pool score (0.8451). No single threshold achieves both 10/10 T survival and 0/20 FP-pool false positives — see the trade-off table above for the actual cost at each candidate value.

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

*Generated by `diagnostics/score_research_intent_templates.py`.*
