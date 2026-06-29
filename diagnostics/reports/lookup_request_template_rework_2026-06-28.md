# `lookup_request` Template Rework — Candidate Scoring Findings

**Date:** 2026-06-28
**Script:** `diagnostics/score_lookup_request_templates.py` (candidate rework section)
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Status:** READ-ONLY. `planner.py` unmodified. `_SEARCH_INTENT_TEMPLATES` unmodified.

Candidate scoring for the 4 suspect templates added 2026-06-25 (`can you look up`,
`can you look that up for me`, `could you look up`, `can you look into this for me`).
The 2026-06-28 margin assessment identified these as the source of a severe,
threshold-unfixable false-positive surface (6/14 Cat D negatives score 0.81–0.90 —
higher than every Cat C true positive).

## §1. Template Sets Under Evaluation

**Current templates (9 total — original 5 + 4 added 2026-06-25):**

| # | Template | Epoch |
|---|----------|-------|
| 1 | `look up this` | pre-2026-06-25 |
| 2 | `look that up` | pre-2026-06-25 |
| 3 | `go ahead and look it up` | pre-2026-06-25 |
| 4 | `find information on this` | pre-2026-06-25 |
| 5 | `find out about this` | pre-2026-06-25 |
| 6 | `can you look up` | 2026-06-25 |
| 7 | `can you look that up for me` | 2026-06-25 |
| 8 | `could you look up` | 2026-06-25 |
| 9 | `can you look into this for me` | 2026-06-25 |

**Candidate Set 1 — object-specificity fix** *(replaces the 4 suspect templates)*:
Modal+verb frame kept; vague pronoun object replaced with a concrete queryable object.

- `can you look up the release date for this`
- `could you look up what year this happened`
- `can you look up information about the latest Apple products`
- `could you find out the current stock price for me`

**Candidate Set 2 — verb-anchored, modal-light** *(replaces the 4 suspect templates)*:
Modal-question scaffolding dropped; anchored on search-specific verb phrases.

- `look up information about`
- `search for details on`
- `find out the facts about`
- `go find out about`

**Candidate Set 3 — removal** *(reverts to original 5, no replacements)*:
Tests whether removing the 4 suspect templates reopens the 2026-06-25 incident.

- `look up this`
- `look that up`
- `go ahead and look it up`
- `find information on this`
- `find out about this`

## §2. Trade-off Tables (same format as 2026-06-28 report §6)

Gate fires = LR ≥ threshold **OR** ESA ≥ 0.68 (ESA templates and threshold unchanged).

- **Cat C survivors**: gate fires for a confirmed true positive. Target: 3/3.
- **Cat A false positives**: gate fires for a 2026-06-28 confirmed misroute. Target: 0/3.
- **Cat D false positives**: gate fires for a confirmed non-search utterance. Target: 0/14.

> **ESA floor note:** Two Cat D utterances fire via ESA ≥ 0.68 independently of LR —
> `Would you look at this?` (ESA=0.6990) and `Will you look into this?` (ESA=0.6874).
> These will appear as false positives under any LR template change at any threshold
> unless the ESA templates or threshold are also changed (not under evaluation here).
> The achievable Cat D floor across all candidates is therefore at minimum 2/14.

### Reference: Current 9 Templates (from 2026-06-28 margin assessment, not recomputed)

| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |
|:------------:|:---------------:|:--------------------:|:--------------------:|
| 0.60 | 2/3 | 3/3 | 13/14 |
| 0.65 | 0/3 | 1/3 | 10/14 |
| 0.68 | 0/3 | 0/3 | 10/14 |

### Set 1 — object-specificity fix

| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |
|:------------:|:---------------:|:--------------------:|:--------------------:|
| 0.60 | 3/3 | 0/3 | 6/14 |
| 0.65 | 2/3 | 0/3 | 4/14 |
| 0.68 | 1/3 | 0/3 | 2/14 |

**Cat C per-utterance LR scores:**

| Utterance | LR | @0.60 | @0.65 | @0.68 |
|-----------|---:|:-----:|:-----:|:-----:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.7653 | **Y** | **Y** | **Y** |
| Can you look up their next-generation in-house Microso… | 0.6522 | **Y** | **Y** | n |
| Can you look up Microsoft's next-generation in-house A… | 0.6409 | **Y** | n | n |

### Set 2 — verb-anchored, modal-light

| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |
|:------------:|:---------------:|:--------------------:|:--------------------:|
| 0.60 | 1/3 | 1/3 | 5/14 |
| 0.65 | 0/3 | 0/3 | 4/14 |
| 0.68 | 0/3 | 0/3 | 3/14 |

**Cat C per-utterance LR scores:**

| Utterance | LR | @0.60 | @0.65 | @0.68 |
|-----------|---:|:-----:|:-----:|:-----:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.5605 | n | n | n |
| Can you look up their next-generation in-house Microso… | 0.6092 | **Y** | n | n |
| Can you look up Microsoft's next-generation in-house A… | 0.5987 | n | n | n |

### Set 3 — original 5 templates only (pre-2026-06-25)

| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |
|:------------:|:---------------:|:--------------------:|:--------------------:|
| 0.60 | 0/3 | 0/3 | 8/14 |
| 0.65 | 0/3 | 0/3 | 5/14 |
| 0.68 | 0/3 | 0/3 | 5/14 |

**Cat C per-utterance LR scores:**

| Utterance | LR | @0.60 | @0.65 | @0.68 |
|-----------|---:|:-----:|:-----:|:-----:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.5712 | n | n | n |
| Can you look up their next-generation in-house Microso… | 0.5829 | n | n | n |
| Can you look up Microsoft's next-generation in-house A… | 0.5791 | n | n | n |

## §3. Six Hardest Cat D Collisions — Per-Utterance Scores (Sets 1 and 2)

The 6 Cat D utterances that scored 0.81–0.90 on LR against the current 9 templates.
These are the hardest test: the modal+verb collision frame at its most extreme.
A score above 0.70 on any candidate means that candidate has not resolved the
core collision for that utterance — flagged explicitly per utterance.

### Set 1 — object-specificity fix

| Rank | Utterance | Prior LR | New LR | ESA | Still >0.70? |
|:----:|-----------|--------:|-------:|----:|:------------:|
| 1 | Could you check this for me? | 0.8980 | 0.6461 | 0.6497 | No |
| 2 | Will you look into this? | 0.8864 | 0.7011 | 0.6874 | **Yes** (ESA=0.6874≥0.68 — ESA-driven, not LR) |
| 3 | Can you tell me about this? | 0.8619 | 0.6671 | 0.6781 | No |
| 4 | Would you look at this? | 0.8483 | 0.6962 | 0.6990 | **Yes** (ESA=0.6990≥0.68 — ESA-driven, not LR) |
| 5 | Do you mind looking at this? | 0.8158 | 0.6562 | 0.6764 | No |
| 6 | Can you help me with this? | 0.8134 | 0.6051 | 0.6187 | No |

### Set 2 — verb-anchored, modal-light

| Rank | Utterance | Prior LR | New LR | ESA | Still >0.70? |
|:----:|-----------|--------:|-------:|----:|:------------:|
| 1 | Could you check this for me? | 0.8980 | 0.6183 | 0.6497 | No |
| 2 | Will you look into this? | 0.8864 | 0.6728 | 0.6874 | **Yes** (ESA=0.6874≥0.68 — ESA-driven, not LR) |
| 3 | Can you tell me about this? | 0.8619 | 0.7580 | 0.6781 | **Yes** (LR=0.7580) |
| 4 | Would you look at this? | 0.8483 | 0.6667 | 0.6990 | **Yes** (ESA=0.6990≥0.68 — ESA-driven, not LR) |
| 5 | Do you mind looking at this? | 0.8158 | 0.6514 | 0.6764 | No |
| 6 | Can you help me with this? | 0.8134 | 0.5431 | 0.6187 | No |

## §4. Candidate Set 3 — 2026-06-25 Incident Regression Check

**Question:** Does removing the 4 suspect templates reopen the 2026-06-25
false-negative incident? Score the 3 incident utterances against the 5-template-only
config at thresholds 0.55, 0.60, 0.65.

| Utterance | LR (5-tmpl only) | @0.55 | @0.60 | @0.65 |
|-----------|----------------:|:-----:|:-----:|:-----:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.5712 | **Y** | n | n |
| Can you look up their next-generation in-house Microso… | 0.5829 | **Y** | n | n |
| Can you look up Microsoft's next-generation in-house A… | 0.5791 | **Y** | n | n |

- At 0.55: 3/3 incident utterances caught by original 5 templates
- At 0.60: 0/3 incident utterances caught by original 5 templates
- At 0.65: 0/3 incident utterances caught by original 5 templates

**Answer:** The original 5 templates catch all 3 incident utterances only at ≤ 0.55. At 0.60: 0/3 caught. Removing the suspect templates reopens the 2026-06-25 incident at threshold 0.60 unless the threshold is also lowered to 0.55.

## §5. Separation Summary (numerical only — no recommendation)

Full separation = there exists a threshold T where Cat C = 3/3 AND Cat D = 0/14.
Partial improvement = Cat D false positives reduced vs. current (≥1 improvement)
while Cat C ≥ 2/3 at the same threshold.

| Candidate Set | Cat C @0.60 | Cat D @0.60 | Cat C @0.65 | Cat D @0.65 | Full separation? |
|:--------------|:-----------:|:-----------:|:-----------:|:-----------:|:---------------:|
| Set 1 — object-specificity fix | 3/3 | 6/14 | 2/3 | 4/14 | No |
| Set 2 — verb-anchored, modal-light | 1/3 | 5/14 | 0/3 | 4/14 | No |
| Set 3 — original 5 templates only (pre-2026-06-25) | 0/3 | 8/14 | 0/3 | 5/14 | No |

---

*No recommendation is made. The data above states per-candidate separation quality numerically.*

*Generated by `diagnostics/score_lookup_request_templates.py` — candidate rework section — 2026-06-28.*
