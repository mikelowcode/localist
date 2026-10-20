# Profile-Fact Threshold Assessment — the single 0.45 across three embedding models

**Date:** 2026-10-20
**Script:** `diagnostics/diag_profile_fact_threshold.py`
**Raw data:** `diagnostics/profile_fact_threshold_results.csv`
**Fixture:** `threshold_calibration_fixtures.PROFILE_*` (new; shared with any future live calibration)
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest` (EmbeddingGemma 300M,
standing in for the oMLX `embeddinggemma-300m-4bit`, which wasn't running), `embeddinggemma-2:latest`
— real Ollama `/api/embed`, no stubs.
**Status:** READ-ONLY. Nothing in the codebase or any database was changed (the fixture file gained
data only).

**Trigger:** §16.28 open item. `ControllerAgent._score_profile_facts(instruction_embedding, top_n=5,
threshold=0.45)` picks Slot 3b profile lines (03-unified-prompt-contract §3.6). 0.45 was set by hand,
never measured, and is the same for every model. Phase 1 measured episodic recall floors from 0.36 to
0.71 across these models.

## Spec/code drift found on the way

§3.6 says profile facts are injected on P4/P5 turns and turns where episodic bullets fire, and that
"P6 direct-answer turns do not inject profile facts". `_execute_plan` Step 5b has no route check: it
scores and injects on every turn. So the threshold is the only gate on profile injection. Seen live in
the Phase 1 sandbox: under `embeddinggemma-2`, "How does photosynthesis convert light into chemical
energy?" got 5 profile facts.

## Method

- **Synthetic battery:** a made-up person's profile, 30 bare fact lines in six sections, embedded raw
  one line at a time exactly as `_load_user_profile()` does. 30 positive questions, each labelled with
  1–4 relevant lines (several have more than one). 16 general-knowledge negatives, including
  explanation requests where style lines could leak. Following §3.6 ("only the top relevant facts for
  the current instruction"), any line injected on a negative counts as a false positive, style lines
  included.
- **Metrics at a threshold:** recall (a relevant line in the top 5 and ≥ threshold), all-relevant
  recall, off-topic FP, irrelevant lines per positive, lines per off-topic question.
- **Selection:** `threshold_calibration._recall_floor_from_scores()`, the episodic recall floor's
  objective (max recall − FP, ties to the highest threshold). So a calibrated profile threshold built on
  it would select exactly these values. Leave-one-out cross-validation (LOOCV) reported alongside.
- **Real profile (out-of-sample):** the sandbox copy of the real profile (88 lines), 16 hand-labelled
  personal questions (11 about facts or interest domains, 5 about response style) and 10 off-topic
  questions. Thresholds from the synthetic battery, **not retuned**. The script and questions are
  personal data and stayed in the session scratchpad; only aggregates are recorded here.

## Results — synthetic battery

Recall / all-relevant / off-topic FP / irrelevant lines per positive / lines per off-topic.

| Model | Best relevant line | Off-topic best | At 0.45 (today) | Calibrated | LOOCV recall / FP |
|---|---|---|---|---|---|
| nomic | 0.429–0.767 | 0.408–0.544 | 93% / 90% / **69%** / 2.9 / 1.4 | **0.52**: 67% / 63% / 12% / 0.7 / 0.1 | 53% / 31% (unstable) |
| embeddinggemma | 0.363–0.706 | 0.264–0.480 | 73% / 70% / 6% / 0.5 / 0.1 | **0.40**: 90% / 87% / 12% / 1.5 / 0.2 | 83% / 12% |
| embeddinggemma-2 | 0.692–0.846 | 0.638–0.737 | 100% / 87% / **100%** / 3.9 / **5.0** | **0.72**: 97% / 83% / 6% / 0.9 / 0.1 | 97% / 6% |

Off-topic leaks at 0.45, by section: `embeddinggemma-2` leaked lines from every section (Communication
Style 30, Cognitive Preferences 15, Identity 13, Interests 12, Values 6, Active Projects 4). nomic
mostly leaked Active Projects. EmbeddingGemma 300M leaked one Communication Style line.

## Results — real profile (out-of-sample)

| Model | Threshold | Fact-question recall | Style-question recall | Irrelevant lines / question | Off-topic FP | Lines / off-topic (style + fact) |
|---|---|---|---|---|---|---|
| nomic | 0.45 | 100% | 100% | 3.2 | 70% | 2.4 (19 + 5) |
| nomic | **0.52** | 100% | 100% | 1.8 | **20%** | 0.3 (3 + 0) |
| embeddinggemma | **0.45** | 100% | 100% | 1.6 | **10%** | 0.4 (4 + 0) |
| embeddinggemma | 0.40 | 100% | 100% | 2.7 | 50% | 1.4 (12 + 2) |
| embeddinggemma-2 | 0.45 | 91% | 100% | 3.6 | 100% | 5.0 (37 + 13) |
| embeddinggemma-2 | **0.72** | 91% | 100% | 2.2 | **20%** | 0.4 (4 + 0) |

(`embeddinggemma-2`'s one fact miss ranks below the top 5 at any threshold.)

## Findings

- **0.45 is badly wrong for `embeddinggemma-2`:** every off-topic question gets the maximum 5 profile
  lines, on both the synthetic and the real profile. Its calibrated 0.72 cuts that to 0.1–0.4 lines with
  no recall loss on the real profile, and is stable under LOOCV.
- **0.45 is too loose for nomic:** 70% of real off-topic questions get profile lines. 0.52 cuts that to
  20% with no recall loss on the real profile. On the synthetic battery 0.52 is a knife edge (LOOCV
  53% / 31%), the same overlap seen in nomic's episodic scores, but real data supports it.
- **0.45 is already right for EmbeddingGemma 300M,** and its calibrated 0.40 is worse on the real profile
  (off-topic FP 10% → 50%, no recall gain). The synthetic battery's paraphrased questions reward a lower
  threshold that real questions don't need. So live calibration alone would make this model worse.
- **The real profile is about 80% style and meta lines** (70 of 88: guidance on answer structure,
  tone and what to avoid). Those are what leak into off-topic questions on every model, and similarity
  gating will only ever surface a few of them on questions that happen to be about style. If they're
  meant to shape every answer, a threshold is the wrong gate for them. That's a design question beyond
  this threshold.

## Options for Phase 5

1. **Hand-validated table plus calibration (the recall-floor pattern).** Hand-validated nomic 0.52
   and EmbeddingGemma 300M 0.45 (both oMLX `embeddinggemma-300m-4bit` and Ollama `embeddinggemma:latest`,
   since live calibration picks a worse value for it). Unlisted models use their calibrated threshold
   (`embeddinggemma-2` → 0.72), else 0.45.
2. **Calibration only.** Simpler, but it would move EmbeddingGemma 300M to 0.40, which is worse on real
   data.
3. Either of the above, plus restoring §3.6's route gate (no profile facts on P6), or else updating §3.6
   to match the code. Either way the spec and code should agree.
