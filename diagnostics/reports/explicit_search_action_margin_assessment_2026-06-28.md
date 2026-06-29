# Negative-Side Margin Assessment — `explicit_search_action` @ 0.68

**Date:** 2026-06-28
**Script:** `diagnostics/score_lookup_request_templates.py` (ESA margin section)
**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs
**Purpose:** Establish the negative-side margin of `_SEMANTIC_GATE_THRESHOLDS["explicit_search_action"] = 0.68`.
**Gate metric:** ESA in isolation (`esa >= threshold`) — **not** OR'd with LR,
so ESA's own margin is visible independent of the lookup_request gate.
**Status:** READ-ONLY. `planner.py` unmodified. All templates and thresholds unmodified.

**Trigger:** The 2026-06-28 LR candidate rework diagnostic found that 2 Cat D
utterances fire at the current 0.68 threshold *via ESA alone*, not LR:
`Would you look at this?` (ESA ≈ 0.699) and `Will you look into this?` (ESA ≈ 0.687).
ESA has never been independently margin-tested. This assessment is the first.

## §1. ESA Templates and Test Categories

**`_SEARCH_INTENT_TEMPLATES["explicit_search_action"]` (5 templates, unchanged):**

- `search the web for this`
- `do a web search for this`
- `search online for this`
- `google this`
- `go look it up`

**Threshold candidates:** 0.60 / 0.65 / 0.68 (current production) / 0.72

**Test set:** all 29 utterances from the 2026-06-28 LR margin assessment, unchanged.

| Cat | Description | N | Expected ESA behavior |
|-----|-------------|---|-----------------------|
| A | Live false positives (2026-06-28) | 3 | ESA should **not** fire |
| B-identity | `_SEARCH_NEGATIVE_FILTER` identity phrases | 5 | Filtered pre-gate in prod; ESA should **not** fire |
| B-greeting | `_SEARCH_NEGATIVE_FILTER` greeting phrases | 4 | Filtered pre-gate in prod; ESA should **not** fire |
| C | 2026-06-25 confirmed true positives | 3 | ESA **should** fire if ESA is load-bearing for any |
| D-\* | Fresh adversarial negatives (14) | 14 | ESA should **not** fire |

## §2. Full Per-Utterance ESA Score Table (all 29 utterances)

ESA-isolated gate: **Y** = ESA ≥ threshold. LR not included in this check.

| Cat | Utterance | ESA | Best ESA Template | @0.60 | @0.65 | @0.68 | @0.72 |
|-----|-----------|-----|-------------------|:-----:|:-----:|:-----:|:-----:|
| A | Tell me how Localist works? | 0.5570 | google this | n | n | n | n |
| A | Can you read my wiki files? | 0.5948 | search the web for this | n | n | n | n |
| A | List the files in my vault? | 0.5832 | go look it up | n | n | n | n |
| B-identity | who are you | 0.5558 | google this | n | n | n | n |
| B-identity | what are you | 0.5713 | google this | n | n | n | n |
| B-identity | what can you do | 0.6029 | google this | **Y** | n | n | n |
| B-identity | what can you help with | 0.5744 | google this | n | n | n | n |
| B-identity | what do you do | 0.6006 | google this | **Y** | n | n | n |
| B-greeting | hey lora | 0.5481 | go look it up | n | n | n | n |
| B-greeting | hi there | 0.5642 | google this | n | n | n | n |
| B-greeting | hey there | 0.5834 | search online for this | n | n | n | n |
| B-greeting | what's up | 0.6478 | go look it up | **Y** | n | n | n |
| C | Can you look up Apple's price hike for the … | 0.5424 | go look it up | n | n | n | n |
| C | Can you look up their next-generation in-ho… | 0.5785 | go look it up | n | n | n | n |
| C | Can you look up Microsoft's next-generation… | 0.5735 | go look it up | n | n | n | n |
| D-verb-swap | Can you help me with this? | 0.6187 | google this | **Y** | n | n | n |
| D-verb-swap | Could you check this for me? | 0.6497 | go look it up | **Y** | n | n | n |
| D-verb-swap | Would you look at this? | 0.6990 | go look it up | **Y** | **Y** | **Y** | n |
| D-verb-swap | Can you tell me about this? | 0.6781 | go look it up | **Y** | **Y** | n | n |
| D-modal-swap | Will you look into this? | 0.6874 | go look it up | **Y** | **Y** | **Y** | n |
| D-modal-swap | Do you mind looking at this? | 0.6764 | go look it up | **Y** | **Y** | n | n |
| D-length | Can you help? | 0.5810 | search online for this | n | n | n | n |
| D-length | Can you help me understand this particular … | 0.5553 | go look it up | n | n | n | n |
| D-domain | Can you help me understand how Localist wor… | 0.5226 | go look it up | n | n | n | n |
| D-domain | Could you explain what this system does? | 0.5567 | go look it up | n | n | n | n |
| D-domain | Can you look at my notes and help me organi… | 0.5173 | search online for this | n | n | n | n |
| D-domain | Could you read through this document for me? | 0.5867 | go look it up | n | n | n | n |
| D-domain | Would you help me plan a trip to Japan? | 0.4735 | go look it up | n | n | n | n |
| D-domain | Can you tell me a joke? | 0.4823 | go look it up | n | n | n | n |

## §3. Category C — Confirmed Positives: ESA Threshold Survival

**Key question:** Is any Cat C utterance dependent on ESA to fire the gate?
If a Cat C item fires via ESA independently of LR, raising ESA could kill
a confirmed positive — the same load-bearing constraint Cat C posed for LR.

| Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |
|-----------|-----|:-----:|:-----:|:-----:|:-----:|
| Can you look up Apple's price hike for the MacBook Neo… | 0.5424 | n | n | n | n |
| Can you look up their next-generation in-house Microso… | 0.5785 | n | n | n | n |
| Can you look up Microsoft's next-generation in-house A… | 0.5735 | n | n | n | n |

- **@0.60:** 0/3 — **ESA is not load-bearing for Cat C at 0.60**
- **@0.65:** 0/3 — **ESA is not load-bearing for Cat C at 0.65**
- **@0.68:** 0/3 — **ESA is not load-bearing for Cat C at 0.68**
- **@0.72:** 0/3 — **ESA is not load-bearing for Cat C at 0.72**

**Conclusion:** No Cat C utterance clears the ESA gate at 0.68 or 0.72.
Cat C fires **exclusively via LR** in production.
Raising the ESA threshold carries **no Cat C coverage cost**.

## §4. Category D — False Positive Counts (ESA-isolated)

N = 14 utterances (unchanged from LR diagnostic).
ESA-isolated false positive = ESA ≥ threshold.
Items marked `†` are the 2 utterances previously flagged as ESA-driven in the
LR candidate rework diagnostic. All 14 are checked for ESA score regardless
of whether ESA was their argmax in the prior report.

### ESA @ 0.60 — Cat D false positives: 6/14
- Previously flagged as ESA-driven: 2 (`Would you look at this?`, `Will you look into this?`)
- Novel (not previously flagged as ESA-driven): 4 (`Can you help me with this?`, `Could you check this for me?`, `Can you tell me about this?`, `Do you mind looking at this?`)

| Utterance | ESA | Best Template | Status |
|-----------|----|---------------|--------|
| Can you help me with this? | 0.6187 | google this | **novel** |
| Could you check this for me? | 0.6497 | go look it up | **novel** |
| Would you look at this? | 0.6990 | go look it up | known † |
| Can you tell me about this? | 0.6781 | go look it up | **novel** |
| Will you look into this? | 0.6874 | go look it up | known † |
| Do you mind looking at this? | 0.6764 | go look it up | **novel** |

### ESA @ 0.65 — Cat D false positives: 4/14
- Previously flagged as ESA-driven: 2 (`Would you look at this?`, `Will you look into this?`)
- Novel (not previously flagged as ESA-driven): 2 (`Can you tell me about this?`, `Do you mind looking at this?`)

| Utterance | ESA | Best Template | Status |
|-----------|----|---------------|--------|
| Would you look at this? | 0.6990 | go look it up | known † |
| Can you tell me about this? | 0.6781 | go look it up | **novel** |
| Will you look into this? | 0.6874 | go look it up | known † |
| Do you mind looking at this? | 0.6764 | go look it up | **novel** |

### ESA @ 0.68 (current production threshold) — Cat D false positives: 2/14
- Previously flagged as ESA-driven: 2 (`Would you look at this?`, `Will you look into this?`)
- Novel (not previously flagged as ESA-driven): 0

| Utterance | ESA | Best Template | Status |
|-----------|----|---------------|--------|
| Would you look at this? | 0.6990 | go look it up | known † |
| Will you look into this? | 0.6874 | go look it up | known † |

### ESA @ 0.72 — Cat D false positives: 0/14
- Previously flagged as ESA-driven: 0
- Novel (not previously flagged as ESA-driven): 0
*(no Cat D false positives at this threshold)*

### Full Category D Per-Utterance ESA Detail

Items marked `†` are the 2 known ESA-driven false positives.

| Axis | Domain | Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |
|------|--------|-----------|----|:-----:|:-----:|:-----:|:-----:|
| D-verb-swap | short/generic | Can you help me with this? | 0.6187 | **Y** | n | n | n |
| D-verb-swap | short/generic | Could you check this for me? | 0.6497 | **Y** | n | n | n |
| D-verb-swap | short/generic | Would you look at this? † | 0.6990 | **Y** | **Y** | **Y** | n |
| D-verb-swap | short/generic | Can you tell me about this? | 0.6781 | **Y** | **Y** | n | n |
| D-modal-swap | short/generic | Will you look into this? † | 0.6874 | **Y** | **Y** | **Y** | n |
| D-modal-swap | short/generic | Do you mind looking at this? | 0.6764 | **Y** | **Y** | n | n |
| D-length | bare-minimum | Can you help? | 0.5810 | n | n | n | n |
| D-length | fuller-sentence | Can you help me understand this particula… | 0.5553 | n | n | n | n |
| D-domain | project-referential | Can you help me understand how Localist w… | 0.5226 | n | n | n | n |
| D-domain | project-referential | Could you explain what this system does? | 0.5567 | n | n | n | n |
| D-domain | file-referencing | Can you look at my notes and help me orga… | 0.5173 | n | n | n | n |
| D-domain | file-referencing | Could you read through this document for … | 0.5867 | n | n | n | n |
| D-domain | generic/unrelated | Would you help me plan a trip to Japan? | 0.4735 | n | n | n | n |
| D-domain | generic/unrelated | Can you tell me a joke? | 0.4823 | n | n | n | n |

## §5. Cat A and Cat B — ESA Scores

These items are either live false positives (Cat A) or pre-filtered phrases (Cat B).
**Key question:** Even if LR were corrected, would any of these independently clear ESA?
A Cat A item clearing ESA at 0.68 means the misroute survives even a corrected LR gate.
A Cat B item clearing ESA means the negative filter's protection would be bypassed
at the ESA layer (the filter runs pre-embedding, so ESA represents an independent risk).

| Cat | Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |
|-----|-----------|-----|:-----:|:-----:|:-----:|:-----:|
| A | Tell me how Localist works? | 0.5570 | n | n | n | n |
| A | Can you read my wiki files? | 0.5948 | n | n | n | n |
| A | List the files in my vault? | 0.5832 | n | n | n | n |
| B-identity | who are you | 0.5558 | n | n | n | n |
| B-identity | what are you | 0.5713 | n | n | n | n |
| B-identity | what can you do | 0.6029 | **Y** | n | n | n |
| B-identity | what can you help with | 0.5744 | n | n | n | n |
| B-identity | what do you do | 0.6006 | **Y** | n | n | n |
| B-greeting | hey lora | 0.5481 | n | n | n | n |
| B-greeting | hi there | 0.5642 | n | n | n | n |
| B-greeting | hey there | 0.5834 | n | n | n | n |
| B-greeting | what's up | 0.6478 | **Y** | n | n | n |

- Cat A items clearing ESA @ 0.68: 0/3 — corrected LR gate would be sufficient
- Cat B items clearing ESA @ 0.68: 0/9 — negative filter covers all B items at ESA level

## §6. Trade-off Table (numerical only — no recommendation)

ESA-isolated gate = ESA ≥ threshold (LR not included).

| ESA threshold | Cat C survivors | Cat A false positives | Cat D false positives |
|:-------------:|:---------------:|:--------------------:|:--------------------:|
| 0.60 | 0/3 | 0/3 | 6/14 |
| 0.65 | 0/3 | 0/3 | 4/14 |
| 0.68 ← current | 0/3 | 0/3 | 2/14 |
| 0.72 | 0/3 | 0/3 | 0/14 |

**Cat C ESA scores** (shown to confirm whether ESA is load-bearing):
- `Can you look up Apple's price hike for the MacBook Neo and iPad?` → ESA = 0.5424
- `Can you look up their next-generation in-house Microsoft AI models?` → ESA = 0.5785
- `Can you look up Microsoft's next-generation in-house AI models?` → ESA = 0.5735

## §7. Failure Mode Classification

**LR failure mode (2026-06-25 suspect templates, from prior report):**
Frame-genericity — the modal-question scaffold (`can/could you + verb + vague-object`)
causes any polite request using that frame to score 0.81–0.90, which is **above**
every Cat C true positive (max 0.61). The negatives invert with the positives.

**ESA failure mode observed in this diagnostic:**

At the production threshold of 0.68, 2/14 Cat D utterances clear ESA:

- `Would you look at this?` → ESA=0.6990, driven by template: `go look it up`
- `Will you look into this?` → ESA=0.6874, driven by template: `go look it up`

**Pattern classification: verb-overlap (structurally different from LR's frame-genericity)**

The `go look it up` template appears to drive 2/all false positive(s).
This template contains the bare verb `look`. Utterances using `look into` or `look at`
— structurally distinct from `look up` — share the root verb and embed close to it.
The other 4 ESA templates (`search the web`, `do a web search`, `search online`,
`google this`) anchor on the word `search`/`google` and do not exhibit this problem.

**Comparison with LR:**

| Dimension | LR failure (2026-06-25 templates) | ESA failure (current) |
|-----------|----------------------------------|----------------------|
| Root cause | Frame-genericity: modal-question scaffold matches all polite requests | Verb-overlap: `go look it up` attracts `look into`/`look at` |
| Severity at production threshold | 6/14 Cat D score 0.81–0.90, **above** Cat C max (0.61) — inversion | 2/14 Cat D score 0.69–0.70; Cat C max ≈ 0.58 — no inversion |
| Cat C visibility via this gate | Cat C fires on LR (0.60–0.61) | Cat C does **not** fire on ESA at any evaluated threshold |
| Threshold-fixable? | No — negatives score above all positives regardless of threshold | Yes — raising ESA to 0.72 drops both false positives below threshold with zero Cat C cost |

---

*No threshold or template change is recommended.*
*The data above classifies the ESA failure pattern numerically and structurally.*

*Generated by `diagnostics/score_lookup_request_templates.py` — ESA margin section — 2026-06-28.*
