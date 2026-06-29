# Full Per-Utterance Table — LR(Set1) + ESA(original) — Cat D and Cat A

**Date:** 2026-06-28
**Script:** `diagnostics/score_lookup_request_templates.py` (full per-utterance section)
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Status:** READ-ONLY. `planner.py` unmodified. `_SEARCH_INTENT_TEMPLATES` unmodified.

Closes the data gap flagged in `combined_lr_esa_crosscheck_2026-06-28.md` §4:
the cross-check report inferred LR(Set1) < 0.60 for 8 Cat D items from the
aggregate 6/14 trade-off count. This section prints every score explicitly.

## §1. LR Templates Used (Candidate Set 1)

These are the 4 Candidate Set 1 templates scored as LR. The original 5 pre-2026-06-25
templates are also included in this config (9 total → Set 1 replaces the 4 suspect ones).
For this diagnostic, LR score = max cosine similarity against these 4 templates only
(the original 5 contribute separately to the production 9-template pool; see cross-check
report for how they interact — this section shows Set 1's contribution in isolation).

- `can you look up the release date for this`
- `could you look up what year this happened`
- `can you look up information about the latest Apple products`
- `could you find out the current stock price for me`

**ESA templates:** original 5 (`explicit_search_action`), unchanged.

## §2. Gate Definitions

Two gate variants evaluated per utterance:

- **Current + Set1:** `LR(Set1) >= 0.60 OR ESA(orig) >= 0.68`  [Set1 LR templates, ESA threshold unchanged]
- **Combined fix:** `LR(Set1) >= 0.60 OR ESA(orig) >= 0.72`  [proposed joint configuration]

## §3. Full Per-Utterance Score Table

All 17 utterances. `fires_curr` = current-ESA gate with Set1 LR; `fires_comb` = joint fix gate.
Both columns differ only when `0.68 <= ESA < 0.72` — those are the ESA-floor items.

| Cat | Domain | Utterance | LR(Set1) | ESA(orig) | Best LR template | fires_curr | fires_comb |
|-----|--------|-----------|:--------:|:---------:|------------------|:----------:|:----------:|
| A | — | Tell me how Localist works? | 0.5478 | 0.5570 | could you find out the current stoc… | n | n |
| A | — | Can you read my wiki files? | 0.5361 | 0.5948 | can you look up the release date fo… | n | n |
| A | — | List the files in my vault? | 0.5452 | 0.5832 | can you look up the release date fo… | n | n |
| D-verb-swap | short/generic | Can you help me with this? | 0.6051 | 0.6187 | can you look up the release date fo… | **Y** | **Y** |
| D-verb-swap | short/generic | Could you check this for me? | 0.6461 | 0.6497 | could you look up what year this ha… | **Y** | **Y** |
| D-verb-swap | short/generic | Would you look at this? | 0.6962 | 0.6990 | could you look up what year this ha… | **Y** | **Y** |
| D-verb-swap | short/generic | Can you tell me about this? | 0.6671 | 0.6781 | can you look up the release date fo… | **Y** | **Y** |
| D-modal-swap | short/generic | Will you look into this? | 0.7011 | 0.6874 | could you look up what year this ha… | **Y** | **Y** |
| D-modal-swap | short/generic | Do you mind looking at this? | 0.6562 | 0.6764 | could you look up what year this ha… | **Y** | **Y** |
| D-length | bare-minimum | Can you help? | 0.5901 | 0.5810 | could you find out the current stoc… | n | n |
| D-length | fuller-sentence | Can you help me understand this particular … | 0.5609 | 0.5553 | could you find out the current stoc… | n | n |
| D-domain | project-referential | Can you help me understand how Localist wor… | 0.5649 | 0.5226 | could you find out the current stoc… | n | n |
| D-domain | project-referential | Could you explain what this system does? | 0.5349 | 0.5567 | could you find out the current stoc… | n | n |
| D-domain | file-referencing | Can you look at my notes and help me organi… | 0.5140 | 0.5173 | could you find out the current stoc… | n | n |
| D-domain | file-referencing | Could you read through this document for me? | 0.5776 | 0.5867 | can you look up the release date fo… | n | n |
| D-domain | generic/unrelated | Would you help me plan a trip to Japan? | 0.4869 | 0.4735 | could you find out the current stoc… | n | n |
| D-domain | generic/unrelated | Can you tell me a joke? | 0.5303 | 0.4823 | could you find out the current stoc… | n | n |

## §4. Totals

| Category | N | fires_curr (LR≥0.60 OR ESA≥0.68) | fires_comb (LR≥0.60 OR ESA≥0.72) |
|----------|:-:|:---------------------------------:|:---------------------------------:|
| Cat A false positives | 3 | 0/3 | 0/3 |
| Cat D false positives | 14 | 6/14 | 6/14 |

## §5. Items Where fires_curr ≠ fires_comb

Items where the two gate variants disagree = items with `0.68 <= ESA < 0.72`
and `LR < 0.60`. These are the utterances where raising ESA from 0.68 to 0.72
would actually change the gate outcome (all others are identical under both gates).

_No items differ between the two gate variants — raising ESA from 0.68 to 0.72__has zero effect on the gate outcome for all 17 utterances under LR-Set1 at 0.60._

---

*No recommendation is made. All scores are explicit — no aggregation or inference.*

*Generated by `diagnostics/score_lookup_request_templates.py` — full per-utterance section — 2026-06-28.*
