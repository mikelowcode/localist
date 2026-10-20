# Search-Intent v2 Assessment — Phase B of the search-intent plan

**Date:** 2026-10-20
**Script:** `diagnostics/diag_search_intent_v2.py`
**Raw data:** `diagnostics/search_intent_v2_results.csv` (held-out battery rows)
**New fixtures:** `threshold_calibration_fixtures.*_V2` and `heldout_split()`. Data only; not wired
into calibration.
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest`, `embeddinggemma-2:latest`
(real Ollama), plus the model-independent lexical fallback.
**Status:** READ-ONLY. Nothing that runs was changed.

**Background:** `search_intent_baseline_2026-10-20.md` (Phase A) found four problems:

- calibration scores raw text while production scores lowercased text;
- the battery can't measure the search gates;
- the search templates don't match how searches are actually phrased;
- the BM25 fallback is biased by message length.

## Method

- **Battery v2.** It adds own positives for the explicit-search gate (16) and the lookup gate (22,
  including short commands and "research <topic>"), plus 36 shared negatives. The negatives cover:
  attached files and images, the assistant's own knowledge, opinions, statements, follow-ups,
  local lookups (notes, files, memory), greetings, verb-swap near misses, and four long pasted texts.
  - research_intent gets 6 positives and 6 negatives; episodic_relevance gets 8 and 8.
  - Each gate's pool also keeps the current battery's phrases.
  - It was written before any scoring and never copied from real requests.
- **Split.** Every third item is held out (`heldout_split()`). Thresholds are chosen with the same
  `_youden_calibrate()` used by calibration, on the calibration part only, and scored on the
  held-out part.
- **Search decision.** "Explicit search OR lookup fires", since both trigger `web_search`. Held-out:
  13 positives, 17 negatives.
- **Configurations per model:**

  | Config | Templates | Calibrated on |
  |---|---|---|
  | A | current | raw text (today) |
  | B | current | lowercased text |
  | C | candidate search templates | lowercased text |

  - The candidate templates were written before scoring: lowercase, no specific subjects, no "look
    at" wording, plus "research this topic" forms.
  - No revision round was needed or done.
- **Lexical variants:**

  | Variant | Scoring |
  |---|---|
  | L0 | today's BM25 |
  | L1 | BM25 per request token |
  | L2 | template content-word coverage, 0–1, stopwords removed |
  | L3 | L2 × min(1, 12 / request tokens) |

- **Real requests.** The user's 32 typed requests labelled in Phase A (6 should search), scored once
  at the end. Personal data: kept in the session scratchpad; only aggregates and generic
  descriptions are recorded here.

## Results — search decision

Held-out and real requests, as deployed: embedding where a gate calibrates cleanly, otherwise the
lexical fallback.

| Setup | Held-out caught / false | J | Real caught / false |
|---|---|---|---|
| nomic, hand-validated values (production today) | 15% / 18% | −0.03 | 1 / 6, 1 false (Phase A) |
| Any model, A or B (search gates inconclusive → L0) | 77% / 29% | +0.48 | 5 / 6, **12 false** |
| nomic, C (explicit-search embedding 0.52 + lookup L3) | 77% / 41% | +0.36 | 4 / 6, 6 false |
| embeddinggemma, C (both gates → L3) | **100% / 24%** | **+0.76** | **5 / 6, 1 false** |
| embeddinggemma-2, C (both gates → L3) | **100% / 24%** | **+0.76** | **5 / 6, 1 false** |

**Embedding search gates on battery v2.** Every model's explicit-search and lookup gates were
inconclusive (J below 0.5) in every configuration. The one exception is nomic's explicit-search gate
with the candidate templates: threshold 0.52, held-out 60% / 12%. Lowercased calibration (B) didn't
change this.

**Lexical variants** (search decision, model-independent):

| Templates | Variant | Held-out caught / false | J | Real caught / false |
|---|---|---|---|---|
| current | L0 (today) | 77% / 29% | +0.48 | 5 / 6, 12 false |
| current | L1 | 85% / 65% | +0.20 | 4 / 6, 3 false |
| current | L2 | 85% / 35% | +0.49 | 4 / 6, 5 false |
| current | L3 | 85% / 29% | +0.55 | 3 / 6, 2 false |
| candidate | L0 | 77% / 47% | +0.30 | 4 / 6, 16 false |
| candidate | L1 | 92% / 59% | +0.33 | 5 / 6, 6 false |
| candidate | L2 | 100% / 35% | +0.65 | 5 / 6, 4 false |
| candidate | **L3** | **100% / 24%** | **+0.76** | **5 / 6, 1 false** |

- **L3 thresholds:** explicit-search 0.50; lookup 0.33. The lookup threshold means one content word
  of a two- or three-word template is enough on a short message.
- **L3's real errors:**
  - The one false alarm is a "look at this image" request: "look" matches a lookup template.
  - The one miss is a "what can you tell me about <specific model>" question, whose content words
    don't overlap the templates.

## Results — research_intent and episodic_relevance

Current templates; battery v2 adds to the current pools.

| Model | Gate | A threshold → held-out | B (lowercased) | Real |
|---|---|---|---|---|
| nomic | research_intent | 0.52 → 100% / 43% | same | — |
| nomic | episodic_relevance | 0.48 → 86% / 22% | same | fired on 10 of 32 |
| embeddinggemma | research_intent | 0.54 → 80% / 14% | same | — |
| embeddinggemma | episodic_relevance | 0.40 → 100% / 44% | same | fired on 7 of 32 |
| embeddinggemma-2 | research_intent | 0.76 → 100% / 14% | same | — |
| embeddinggemma-2 | episodic_relevance | 0.70 → 86% / 56% | 0.68 → 100% / 56% | fired on 15 → **26** of 32 |

None of the 32 real requests needed memory, so every episodic firing is a false alarm.

- On the current battery (Phase A), the episodic gate fired on 0 (embeddinggemma) and 15
  (embeddinggemma-2) real requests.
- The v2 battery's broader personal-reference positives pull the episodic thresholds down, and real
  false alarms go up.
- Lexical variants don't rescue it: L0 is 71% / 33% held-out and fires on 8 real requests; L3 falls
  below the J floor.

## Findings

1. **Short-template cosine similarity can't separate search intent from realistic non-search
   requests, for any of the three models.**
   - This held on a battery with real hard negatives, even with revised templates and lowercased
     calibration.
   - nomic's hand-validated search values do no better than chance on held-out data: 15% caught,
     18% false.
2. **A length-aware coverage score with the revised templates (L3) is the best search detector
   measured.**
   - Held-out: 100% caught, 24% false. Real: 5 of 6 caught, 1 false.
   - Every model would get the same result, since it doesn't use embeddings.
   - It is crude: a single search verb in a short message is enough for the lookup gate. Its
     remaining false alarms are "look at …" requests.
3. **Lowercased calibration made no measurable difference** here, except one gate
   (embeddinggemma-2 episodic), where it lowered the threshold and real false alarms rose. It is
   still the consistent choice, since production scores lowercased text.
4. **The episodic gate is loose on every model and v2 makes it looser.** It needs its own study
   (templates, battery, or a different mechanism). Shipping v2's episodic additions as they stand
   would raise real false alarms.
5. **research_intent is reasonable on the gemma models; on nomic its calibrated threshold is loose**
   (43% held-out false). nomic's hand-validated 0.58 was tighter in Phase A (80% / 18%).

## Options for Phase C

- **Search gates, recommended.**
  - Adopt the candidate templates and L3 coverage scoring.
  - Use L3 as the search detector for every model, rather than only as a fallback: where an
    embedding gate did calibrate (nomic explicit-search), combining it with L3 was worse (4 / 6, 6
    false) than L3 alone.
  - Retire nomic's hand-validated explicit-search and lookup values.
  - Ship battery v2's search pools, so the L3 thresholds can be re-derived by the existing offline
    lexical calibration.
- **research_intent:** no change beyond lowercased calibration. Optionally keep nomic's
  hand-validated 0.58.
- **episodic_relevance:** keep the current battery and templates for now. Study it separately.
- **Calibration:** lowercase phrases before embedding, and bump `CALIBRATION_VERSION`.
