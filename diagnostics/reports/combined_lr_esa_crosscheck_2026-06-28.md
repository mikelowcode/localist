# Combined LR+ESA Cross-Tabulation — Joint Fix Assessment

**Date:** 2026-06-28
**Method:** Arithmetic cross-tabulation over already-computed scores — no new embeddings run.
**Status:** READ-ONLY. `planner.py` unmodified. Both source report files unmodified.

**Proposed joint configuration:**
- LR templates: Candidate Set 1 (object-specificity fix — 4 modal+concrete-object templates)
- LR threshold: 0.60 (current production value, unchanged)
- ESA templates: original 5 (unchanged)
- ESA threshold: 0.72 (raised from 0.68)

**Gate:** `gate_fires = (LR_set1 >= 0.60) OR (ESA_orig >= 0.72)`

---

## §1. Source Data and Key Assumption

**Score sources:**
- `LR_set1` values: extracted from `lookup_request_template_rework_2026-06-28.md` §3
  (Set 1 hardest-6 table) and §2 Set 1 Cat C per-utterance table.
- `ESA_orig` values: extracted from `explicit_search_action_margin_assessment_2026-06-28.md`
  §2 full per-utterance table and §4 full Cat D detail table.

**Critical assumption (state for verification):**
Candidate Set 1 modifies only the LR (`lookup_request`) templates. It does not touch the
`explicit_search_action` template group. Therefore, the ESA scores computed in the ESA
margin report — which used the original, unmodified ESA templates — are valid inputs for
this cross-tabulation. The ESA columns below are identical to what the proposed joint
configuration would produce. **If this assumption is wrong (e.g. the EmbeddingEngine
caches across groups), the ESA column must be re-verified by re-running the script under
the joint config — but given the implementation uses per-query embedding with no shared
cache, the assumption holds.**

---

## §2. Full Cross-Tabulation Table

`fires_combined` = `(LR_set1 >= 0.60) OR (ESA_orig >= 0.72)`

Cells showing `—` indicate LR(Set1) score not printed in source report (data gap — flagged
in §4). Gate outcome is still derivable from the aggregate trade-off table.

### Cat A (3 utterances)

| Cat | Utterance | LR(Set1) | ESA(orig) | fires_combined |
|-----|-----------|:--------:|:---------:|:--------------:|
| A | Tell me how Localist works? | < 0.60 (no score printed) | 0.5570 | **n** |
| A | Can you read my wiki files? | < 0.60 (no score printed) | 0.5948 | **n** |
| A | List the files in my vault? | < 0.60 (no score printed) | 0.5832 | **n** |

> Cat A LR(Set1) scores not printed in source report; inferred < 0.60 from Set 1 @0.60
> Cat A false positive count = 0/3 (same as Report 1 Set 1 @0.60).

### Cat C (3 utterances)

| Cat | Utterance | LR(Set1) | ESA(orig) | fires_combined |
|-----|-----------|:--------:|:---------:|:--------------:|
| C | Can you look up Apple's price hike for the MacBook Neo… | 0.7653 | 0.5424 | **Y** (via LR) |
| C | Can you look up their next-generation in-house Microso… | 0.6522 | 0.5785 | **Y** (via LR) |
| C | Can you look up Microsoft's next-generation in-house A… | 0.6409 | 0.5735 | **Y** (via LR) |

### Cat D — 6 hardest items (scores explicit in both source reports)

| Cat | Utterance | LR(Set1) | ESA(orig) | fires_combined | Driver |
|-----|-----------|:--------:|:---------:|:--------------:|--------|
| D-verb-swap | Could you check this for me? | 0.6461 | 0.6497 | **Y** | LR |
| D-modal-swap | Will you look into this? | 0.7011 | 0.6874 | **Y** | LR |
| D-verb-swap | Can you tell me about this? | 0.6671 | 0.6781 | **Y** | LR |
| D-verb-swap | Would you look at this? | 0.6962 | 0.6990 | **Y** | LR |
| D-modal-swap | Do you mind looking at this? | 0.6562 | 0.6764 | **Y** | LR |
| D-verb-swap | Can you help me with this? | 0.6051 | 0.6187 | **Y** | LR |

> None of the 6 fire via ESA at 0.72 (max ESA in this group is 0.6990 < 0.72).
> All 6 fire via LR at 0.60.

### Cat D — 8 remaining items (LR scores not printed in source report — §4 gap flag applies)

| Cat | Utterance | LR(Set1) | ESA(orig) | fires_combined |
|-----|-----------|:--------:|:---------:|:--------------:|
| D-length | Can you help? | < 0.60 (inferred) | 0.5810 | **n** |
| D-length | Can you help me understand this particular concept… | < 0.60 (inferred) | 0.5553 | **n** |
| D-domain | Can you help me understand how Localist works? | < 0.60 (inferred) | 0.5226 | **n** |
| D-domain | Could you explain what this system does? | < 0.60 (inferred) | 0.5567 | **n** |
| D-domain | Can you look at my notes and help me organize them? | < 0.60 (inferred) | 0.5173 | **n** |
| D-domain | Could you read through this document for me? | < 0.60 (inferred) | 0.5867 | **n** |
| D-domain | Would you help me plan a trip to Japan? | < 0.60 (inferred) | 0.4735 | **n** |
| D-domain | Can you tell me a joke? | < 0.60 (inferred) | 0.4823 | **n** |

> Basis for inference: Report 1 §2 Set 1 @0.60 trade-off table shows 6/14 Cat D fires.
> All 6 fires are fully accounted for by the hardest-6 items above (all 6 have LR ≥ 0.60).
> Therefore these 8 items have LR(Set1) < 0.60. ESA is also < 0.72 for all 8 (max 0.5867).
> This inference is **not** verified by individual LR scores — flagged as a data gap in §4.

---

## §3. Combined Totals

| Category | Combined count | Notes |
|----------|:-------------:|-------|
| **Cat C survivors** | **3/3** ✓ | All 3 fire exclusively via LR (0.6409–0.7653). ESA max = 0.5785 < 0.72. |
| **Cat A false positives** | **0/3** ✓ | No Cat A item fires via LR (Set 1 @0.60) or ESA (max 0.5948 < 0.72). |
| **Cat D false positives** | **6/14** | All 6 via LR. ESA ≥ 0.72: 0/14 Cat D items — ESA contributes zero at 0.72. |

### Marginal value of combining both fixes (vs. each fix alone)

| Configuration | Gate | Cat D FP | Cat C |
|:--------------|:-----|:--------:|:-----:|
| No fix (current production) | LR-9-tmpl@0.60 OR ESA-orig@0.68 | 13/14 | 2/3 |
| LR-Set1 alone (ESA unchanged @0.68) | LR-Set1@0.60 OR ESA-orig@0.68 | 6/14 | 3/3 |
| ESA-0.72 alone (LR original templates @0.60) | LR-9-tmpl@0.60 OR ESA-orig@0.72 | 13/14 | 2/3 |
| **Combined: LR-Set1@0.60 + ESA@0.72** | LR-Set1@0.60 OR ESA-orig@0.72 | **6/14** | **3/3** |

**Key finding: Marginal value of the ESA@0.72 raise, given LR-Set1 at 0.60 = zero.**

At LR threshold 0.60, both previously ESA-driven false positives fire independently via
LR-Set1 (LR scores 0.6962 and 0.7011, both ≥ 0.60). Removing their ESA path by raising
the threshold to 0.72 does not remove them from the gate. The combined Cat D count is
unchanged at 6/14.

The ESA fix provides marginal value only if the LR threshold is also raised above 0.7011
(the higher of the two items' LR scores under Set 1). At that LR threshold, neither item
fires via LR, and ESA@0.72 would keep them suppressed. The combined Cat D count in that
scenario (LR-Set1@0.72 OR ESA@0.72) is 2/14 from Report 1 §2 (at LR 0.68: 2/14; both
are ESA-driven at that row, and at 0.72 ESA drops those too → 0/14). Neither report
computes LR-Set1@0.72 explicitly, so that scenario is not evaluated here.

> **Note on the "2/14 (ESA-0.72 fix alone)" comparison point in the task brief:**
> The "2/14" figure refers to the ESA-isolated count at 0.68 from the ESA margin report
> (not ESA@0.72, which is 0/14 isolated). In the production OR-gate, raising ESA from 0.68
> to 0.72 while leaving LR original at 0.60 gives 13/14 (not 2/14), because the 13 items
> that fire via original LR@0.60 continue to fire regardless. The "2/14" is the ESA-only
> marginal contribution at 0.68, not the gate outcome.

---

## §4. Data Gaps and Annotation Issues

### Gap 1 — LR(Set1) scores for 8 non-hardest-6 Cat D items (inference only)

Report 1 prints LR(Set1) scores only for the 6 hardest Cat D items in §3. The other 8
Cat D items' exact LR(Set1) scores are not published. Their gate outcome at 0.60 is
inferred from the aggregate 6/14 trade-off table (all 6 fires are accounted for by the
hardest-6). The "< 0.60 (inferred)" entries in the table above are not verified per
utterance. To close this gap, re-run `score_lookup_request_templates.py` with full
per-utterance logging for Set 1 against all 14 Cat D items.

### Gap 2 — Cat A LR(Set1) scores (inference only)

Report 1 prints no per-utterance LR(Set1) scores for Cat A — only the aggregate 0/3 at
@0.60. Gate outcome for Cat A is inferred from that aggregate. Not a concern for correctness
of the combined totals, but individual scores are not available.

### Gap 3 — "ESA-driven, not LR" annotation for "Will you look into this?"

Report 1 §3 annotates "Will you look into this?" as **"ESA=0.6874≥0.68 — ESA-driven, not
LR"**. This annotation is ambiguous: `LR(Set1) = 0.7011 ≥ 0.68`, meaning this item also
fires via LR at the 0.68 threshold. It is not exclusively ESA-driven even at 0.68. At the
production LR threshold of 0.60, it fires via LR unambiguously (LR=0.7011). The annotation
was accurate in the context of Report 1's ESA-floor note (explaining why items persist
regardless of LR template changes), but reading it as "LR alone would not fire this item"
is incorrect.

Practical consequence: the two "ESA-floor" items contribute exactly 0 marginal Cat D false
positives when combined with LR-Set1@0.60. Both are LR-positive at 0.60, not ESA-exclusive.

---

*No recommendation is made. The combined numbers are reported above.*

*Arithmetic only — no new embeddings run. Source data from two previously generated reports.*

*Generated 2026-06-28. `planner.py` unmodified. Both source reports unmodified. Test count: 436 → 436.*
