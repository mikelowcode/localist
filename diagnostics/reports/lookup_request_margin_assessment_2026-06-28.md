# Negative-Side Margin Assessment — `lookup_request` @ 0.60

**Date:** 2026-06-28
**Script:** `diagnostics/score_lookup_request_templates.py` (extended section)
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Purpose:** Establish the negative-side margin of `_SEMANTIC_GATE_THRESHOLDS["lookup_request"] = 0.60`
**Status:** READ-ONLY diagnostic. `planner.py` unmodified (verified via git diff).

## 1. Category Definitions

| Cat | Description | N | Expected behavior |
|-----|-------------|---|-------------------|
| A | Live false positives, 2026-06-28 — misrouted to web_search | 3 | gate=False |
| B-identity | `_SEARCH_NEGATIVE_FILTER` identity phrases (added 2026-06-26) | 5 | filtered before gate |
| B-greeting | `_SEARCH_NEGATIVE_FILTER` greeting phrases (added 2026-06-27) | 4 | filtered before gate |
| C | 2026-06-25 confirmed true positives — MUST fire gate | 3 | gate=True |
| D-\* | Fresh adversarial negatives, systematic (14 utterances) | 14 | gate=False |

## 2. Category D Generation Axes

Collision shape confirmed 2026-06-26: **"can/could/would you + [verb]"** modal-question
scaffolding with no actual search/lookup semantics. Four axes used (each utterance is tagged
with its primary axis and domain in `_CAT_D`; the table in §5 repeats those tags):

1. **Verb swap** (modal=can/could/would): replace "look up" with non-lookup verbs
   — help, check, "look at" [≠ "look up"], tell me about
2. **Modal swap** (verb=look into): replace can/could/would with "will" / "do you mind"
3. **Length/specificity**: short bare form ("Can you help?") vs. fuller sentence
4. **Subject domain**: project/self-referential | file/document-referencing | generic/unrelated

Axes 1–2 directly probe the confirmed collision vector.
Axes 3–4 probe whether length or subject domain shifts the collision rate.
All 14 utterances share the property that a reasonable user would NOT expect web search.

## 3. Full Score Table (all categories A–D, all 4 groups)

- ESA = `explicit_search_action` (gate threshold fixed at 0.68, not under evaluation)
- LR = `lookup_request` (the threshold under evaluation)
- KRO = `knowledge_request_open` (no gate threshold — diagnostic logging only)
- FR = `freshness_request` (no gate threshold — diagnostic logging only)
- @0.60 / @0.65 / @0.68: gate fires? = LR ≥ candidate threshold **OR** ESA ≥ 0.68

| Cat | Utterance | ESA | LR | KRO | FR | Best | @0.60 | @0.65 | @0.68 |
|-----|-----------|-----|----|-----|----|------|-------|-------|-------|
| A | Tell me how Localist works? | 0.5570 | 0.5770 | 0.6366 | 0.5305 | KRO:0.6366 | n | n | n |
| A | Can you read my wiki files? | 0.5948 | 0.5761 | 0.5554 | 0.4971 | ESA:0.5948 | n | n | n |
| A | List the files in my vault? | 0.5832 | 0.5923 | 0.5720 | 0.5549 | LR:0.5923 | n | n | n |
| B-identity | who are you | 0.5558 | 0.5656 | 0.6476 | 0.5307 | KRO:0.6476 | n | n | n |
| B-identity | what are you | 0.5713 | 0.5835 | 0.6921 | 0.5714 | KRO:0.6921 | n | n | n |
| B-identity | what can you do | 0.6029 | 0.6076 | 0.6585 | 0.5796 | KRO:0.6585 | **Y** | n | n |
| B-identity | what can you help with | 0.5744 | 0.5864 | 0.6789 | 0.5691 | KRO:0.6789 | n | n | n |
| B-identity | what do you do | 0.6006 | 0.5780 | 0.6632 | 0.5432 | KRO:0.6632 | n | n | n |
| B-greeting | hey lora | 0.5481 | 0.5674 | 0.5425 | 0.4685 | LR:0.5674 | n | n | n |
| B-greeting | hi there | 0.5642 | 0.5879 | 0.5836 | 0.5141 | LR:0.5879 | n | n | n |
| B-greeting | hey there | 0.5834 | 0.5999 | 0.6012 | 0.5353 | KRO:0.6012 | n | n | n |
| B-greeting | what's up | 0.6478 | 0.7009 | 0.7383 | 0.7153 | KRO:0.7383 | **Y** | **Y** | **Y** |
| C | Can you look up Apple's price hike for the MacB… | 0.5424 | 0.7653 | 0.4684 | 0.4938 | LR:0.7653 | **Y** | **Y** | **Y** |
| C | Can you look up their next-generation in-house … | 0.5785 | 0.6522 | 0.4935 | 0.5053 | LR:0.6522 | **Y** | **Y** | n |
| C | Can you look up Microsoft's next-generation in-… | 0.5735 | 0.6409 | 0.4863 | 0.4934 | LR:0.6409 | **Y** | n | n |
| D-verb-swap | Can you help me with this? | 0.6187 | 0.6146 | 0.7517 | 0.6426 | KRO:0.7517 | **Y** | n | n |
| D-verb-swap | Could you check this for me? | 0.6497 | 0.6833 | 0.7160 | 0.6479 | KRO:0.7160 | **Y** | **Y** | **Y** |
| D-verb-swap | Would you look at this? | 0.6990 | 0.7818 | 0.7321 | 0.6706 | LR:0.7818 | **Y** | **Y** | **Y** |
| D-verb-swap | Can you tell me about this? | 0.6781 | 0.8619 | 0.9622 | 0.7137 | KRO:0.9622 | **Y** | **Y** | **Y** |
| D-modal-swap | Will you look into this? | 0.6874 | 0.7062 | 0.7446 | 0.6836 | KRO:0.7446 | **Y** | **Y** | **Y** |
| D-modal-swap | Do you mind looking at this? | 0.6764 | 0.7503 | 0.7169 | 0.6512 | LR:0.7503 | **Y** | **Y** | **Y** |
| D-length | Can you help? | 0.5810 | 0.5901 | 0.6218 | 0.5614 | KRO:0.6218 | n | n | n |
| D-length | Can you help me understand this particular conc… | 0.5553 | 0.5874 | 0.7147 | 0.5365 | KRO:0.7147 | n | n | n |
| D-domain | Can you help me understand how Localist works? | 0.5226 | 0.5649 | 0.6675 | 0.5138 | KRO:0.6675 | n | n | n |
| D-domain | Could you explain what this system does? | 0.5567 | 0.6198 | 0.7359 | 0.5883 | KRO:0.7359 | **Y** | n | n |
| D-domain | Can you look at my notes and help me organize t… | 0.5173 | 0.5456 | 0.5545 | 0.4866 | KRO:0.5545 | n | n | n |
| D-domain | Could you read through this document for me? | 0.5867 | 0.6097 | 0.6437 | 0.5510 | KRO:0.6437 | **Y** | n | n |
| D-domain | Would you help me plan a trip to Japan? | 0.4735 | 0.4869 | 0.4910 | 0.3901 | KRO:0.4910 | n | n | n |
| D-domain | Can you tell me a joke? | 0.4823 | 0.5303 | 0.5642 | 0.4538 | KRO:0.5642 | n | n | n |

## 4. Category C — Confirmed Positives: Threshold Survival

Survival = gate still fires at the candidate LR threshold (LR ≥ T or ESA ≥ 0.68).
Last live-verified LR scores (2026-06-26): 0.6077 / 0.6172 / 0.6208.
A **Y** in every column means that threshold would not un-gate any confirmed real positive.
A **n** means raising to that threshold kills a confirmed positive — this is the
load-bearing constraint on how far the threshold can be raised.

| Utterance | LR score | Survives @0.60 | Survives @0.65 | Survives @0.68 |
|-----------|----------|:--------------:|:--------------:|:--------------:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.7653 | **Y** | **Y** | **Y** |
| Can you look up their next-generation in-house Microso… | 0.6522 | **Y** | **Y** | n |
| Can you look up Microsoft's next-generation in-house A… | 0.6409 | **Y** | n | n |

- **@0.60:** all 3 survive — raising to this threshold does not kill any confirmed positive
- **@0.65:** ⚠ only 2/3 survive — raising to 0.65 KILLS 1 confirmed positive(s)
- **@0.68:** ⚠ only 1/3 survive — raising to 0.68 KILLS 2 confirmed positive(s)

## 5. Category D — Fresh Adversarial Negatives: False Positive Counts

N = 14 utterances (see §2 for generation axes).
A **false positive** = gate fires for an utterance that should NOT trigger web_search.
Gate fires = LR ≥ threshold OR ESA ≥ 0.68.

### LR threshold @ 0.60 — False positives: 8/14

| Utterance | LR | ESA | Trigger |
|-----------|----|-----|---------|
| Can you help me with this? | 0.6146 | 0.6187 | LR=0.6146≥0.60 |
| Could you check this for me? | 0.6833 | 0.6497 | LR=0.6833≥0.60 |
| Would you look at this? | 0.7818 | 0.6990 | LR=0.7818≥0.60; ESA=0.6990≥0.68 |
| Can you tell me about this? | 0.8619 | 0.6781 | LR=0.8619≥0.60 |
| Will you look into this? | 0.7062 | 0.6874 | LR=0.7062≥0.60; ESA=0.6874≥0.68 |
| Do you mind looking at this? | 0.7503 | 0.6764 | LR=0.7503≥0.60 |
| Could you explain what this system does? | 0.6198 | 0.5567 | LR=0.6198≥0.60 |
| Could you read through this document for me? | 0.6097 | 0.5867 | LR=0.6097≥0.60 |

### LR threshold @ 0.65 — False positives: 5/14

| Utterance | LR | ESA | Trigger |
|-----------|----|-----|---------|
| Could you check this for me? | 0.6833 | 0.6497 | LR=0.6833≥0.65 |
| Would you look at this? | 0.7818 | 0.6990 | LR=0.7818≥0.65; ESA=0.6990≥0.68 |
| Can you tell me about this? | 0.8619 | 0.6781 | LR=0.8619≥0.65 |
| Will you look into this? | 0.7062 | 0.6874 | LR=0.7062≥0.65; ESA=0.6874≥0.68 |
| Do you mind looking at this? | 0.7503 | 0.6764 | LR=0.7503≥0.65 |

### LR threshold @ 0.68 — False positives: 5/14

| Utterance | LR | ESA | Trigger |
|-----------|----|-----|---------|
| Could you check this for me? | 0.6833 | 0.6497 | LR=0.6833≥0.68 |
| Would you look at this? | 0.7818 | 0.6990 | LR=0.7818≥0.68; ESA=0.6990≥0.68 |
| Can you tell me about this? | 0.8619 | 0.6781 | LR=0.8619≥0.68 |
| Will you look into this? | 0.7062 | 0.6874 | LR=0.7062≥0.68; ESA=0.6874≥0.68 |
| Do you mind looking at this? | 0.7503 | 0.6764 | LR=0.7503≥0.68 |

### Full Category D Per-Utterance Detail

| Axis | Domain | Utterance | LR | @0.60 | @0.65 | @0.68 |
|------|--------|-----------|----|:-----:|:-----:|:-----:|
| D-verb-swap | short/generic | Can you help me with this? | 0.6146 | **Y** | n | n |
| D-verb-swap | short/generic | Could you check this for me? | 0.6833 | **Y** | **Y** | **Y** |
| D-verb-swap | short/generic | Would you look at this? | 0.7818 | **Y** | **Y** | **Y** |
| D-verb-swap | short/generic | Can you tell me about this? | 0.8619 | **Y** | **Y** | **Y** |
| D-modal-swap | short/generic | Will you look into this? | 0.7062 | **Y** | **Y** | **Y** |
| D-modal-swap | short/generic | Do you mind looking at this? | 0.7503 | **Y** | **Y** | **Y** |
| D-length | bare-minimum | Can you help? | 0.5901 | n | n | n |
| D-length | fuller-sentence | Can you help me understand this particular conc… | 0.5874 | n | n | n |
| D-domain | project-referential | Can you help me understand how Localist works? | 0.5649 | n | n | n |
| D-domain | project-referential | Could you explain what this system does? | 0.6198 | **Y** | n | n |
| D-domain | file-referencing | Can you look at my notes and help me organize t… | 0.5456 | n | n | n |
| D-domain | file-referencing | Could you read through this document for me? | 0.6097 | **Y** | n | n |
| D-domain | generic/unrelated | Would you help me plan a trip to Japan? | 0.4869 | n | n | n |
| D-domain | generic/unrelated | Can you tell me a joke? | 0.5303 | n | n | n |

## 6. Trade-off Statement (numerical only — no recommendation)

| LR threshold | Cat C survivors (must-fire) | Cat A false positives | Cat D false positives |
|:------------:|:---------------------------:|:--------------------:|:--------------------:|
| 0.60 | 3/3 | 0/3 | 8/14 |
| 0.65 | 2/3 | 0/3 | 5/14 |
| 0.68 | 1/3 | 0/3 | 5/14 |

**Cat C minimum LR scores** (load-bearing — any threshold above the lowest kills a confirmed positive):
- `Can you look up Apple's price hike for the MacBook Neo and iPad?` → LR = 0.7653
- `Can you look up their next-generation in-house Microsoft AI models?` → LR = 0.6522
- `Can you look up Microsoft's next-generation in-house AI models?` → LR = 0.6409

**Cat A (live false positives, 2026-06-28) LR scores** (evidence that triggered this diagnostic):
- `Tell me how Localist works?` → LR = 0.5770
- `Can you read my wiki files?` → LR = 0.5761
- `List the files in my vault?` → LR = 0.5923

---

*No threshold recommendation is made. The data above states the cost at each candidate*
*threshold in terms of confirmed-positive survival (Cat C) and false positive rate (Cat A, Cat D).*

*Generated by `diagnostics/score_lookup_request_templates.py` — extended section — 2026-06-28.*
