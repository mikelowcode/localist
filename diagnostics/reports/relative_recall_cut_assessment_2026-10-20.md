# Relative Recall Cut Assessment — absolute floor vs. margin / gap / rank / prominence gate

**Date:** 2026-10-20
**Script:** `diagnostics/diag_relative_recall_cut.py`
**Raw data:** `diagnostics/relative_recall_cut_results.csv`
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest` (EmbeddingGemma 300M,
standing in for the oMLX `embeddinggemma-300m-4bit`, which wasn't running), `embeddinggemma-2:latest`
— real Ollama `/api/embed`, no stubs.
**Status:** READ-ONLY. Nothing in the codebase or any database was changed.

**Trigger:** docs/architecture/16-runtime-backend-layer.md §16.28 open item: nomic's right-episode and
off-topic scores overlap (0.50–0.57), so its calibrated 0.55 floor gives 68% recall / 7% FP on the
synthetic battery. Is a relative cut the next lever? Phase 1
(`unlisted_model_calibration_assessment_2026-10-20.md`) added that both EmbeddingGemma floors sit
≤0.005 above their worst off-topic score.

## Method

Production keeps the top 5 episodes scoring ≥ floor. Each variant is applied to that same top 5, on
the shared `RECALL_*` battery (30 episodes, 31 positive / 15 negative queries, no prefixes):

| Variant | Rule |
|---|---|
| A floor | keep sᵢ ≥ f (today) |
| B floor + margin | … and sᵢ ≥ s₁ − d |
| C floor + gap | … truncated at the first drop sᵢ − sᵢ₊₁ > g |
| D floor + rank | … and rank ≤ k |
| E gated | inject the floor-f set (optionally with margin d) only if s₁ ≥ f_high or prominence ≥ m, where prominence = s₁ − mean(s₂..s₁₀) |

B–D always keep the top episode when it passes f, so they can only cut wrong extras, never off-topic
FP. E is the only variant that can lower FP without raising the floor.

Selection: maximise J = recall − FP, ties to fewer wrong episodes, the objective
`calibrate_recall_floor()` already uses; A reproduces the calibrated floors (0.55 / 0.36 / 0.71).
B–E tune 2–4 parameters on 46 queries, so leave-one-out cross-validation (LOOCV) is reported:
parameters re-chosen on 45 queries, scored on the held-out one.

## Results — synthetic battery

Recall / off-topic FP / wrong episodes per question, in-sample and LOOCV.

| Model | Variant | Params | In-sample | J | LOOCV | J |
|---|---|---|---|---|---|---|
| nomic | A floor | f=0.55 | 68% / 7% / 0.4 | 0.61 | 58% / 33% / 0.4 | 0.25 |
| nomic | B margin | f=0.55, d=0.03 | 68% / 7% / 0.1 | 0.61 | 65% / 33% / 0.1 | 0.31 |
| nomic | C gap | f=0.55, g=0.03 | 68% / 7% / 0.1 | 0.61 | 65% / 33% / 0.1 | 0.31 |
| nomic | D rank | f=0.55, k=2 | 68% / 7% / 0.3 | 0.61 | 55% / 33% / 0.3 | 0.22 |
| nomic | E gated (full) | f=0.50, f_high=0.58, m=0.07, d=0.12 | 90% / 13% / 0.6 | 0.77 | 87% / 27% / 0.6 | 0.60 |
| nomic | **E floor + prominence** | f=0.50, m=0.07 | 90% / 13% / 1.5 | 0.77 | 90% / 20% / 1.5 | **0.70** |
| embeddinggemma | A floor | f=0.36 | 90% / 0% / 0.7 | 0.90 | 90% / 7% / 0.7 | **0.84** |
| embeddinggemma | B margin | f=0.35, d=0.01 | 90% / 0% / 0.1 | 0.90 | 90% / 7% / 0.1 | 0.84 |
| embeddinggemma | E floor + prominence | f=0.33, m=0.08 | 94% / 0% / 1.1 | 0.94 | 90% / 7% / 1.1 | 0.84 |
| embeddinggemma-2 | A floor | f=0.71 | 84% / 0% / 0.5 | 0.84 | 84% / 13% / 0.5 | **0.71** |
| embeddinggemma-2 | B margin | f=0.71, d=0.02 | 84% / 0% / 0.1 | 0.84 | 81% / 13% / 0.1 | 0.67 |
| embeddinggemma-2 | E floor + prominence | f=0.71, m=0.00 (= floor) | 84% / 0% / 0.5 | 0.84 | 77% / 13% / 0.3 | 0.64 |

(Full table for every variant and model in the CSV.)

- **On the battery, the prominence gate is a large win for nomic and only for nomic.** Prominence
  separates nomic's positives (median 0.143) from off-topic (median 0.039) better than raw score does,
  so the floor can drop to 0.50 while the gate holds FP down. LOOCV J goes from 0.25 to 0.70, and it's
  stable for m 0.06–0.09. On the EmbeddingGemma models the calibrated m falls to 0 or ties the floor:
  they don't need it.
- **The absolute floor is unstable for nomic under LOOCV** (FP 7% → 33%): drop one off-topic query and
  the chosen floor moves below another. The knife edge is real on this battery.
- **Margin and gap (B, C) cut wrong extras to ~0.1 per question at no recall cost** on every model.
  But every battery query has exactly one right episode, so this battery rewards a tight margin by
  construction. A real question with several relevant episodes would lose all but one. The battery can't
  measure that cost, so it can't justify a margin.

## Results — real episodes (out-of-sample)

The sandbox snapshot of the live DB (21 active episodes, clustered in gaming / Localist / Ollama), 16
hand-written personal questions labelled with their relevant episodes (5 have 3 relevant episodes
each), 10 off-topic questions. Parameters from the battery, **not retuned**. The script and questions
are personal data and stayed in the session scratchpad; only aggregates are recorded here.

| Model | Variant | Any relevant kept | All relevant kept | Off-topic FP | Wrong / question |
|---|---|---|---|---|---|
| nomic | A floor 0.55 | 94% | 75% | 0% | 1.2 |
| nomic | floor 0.50 + prominence 0.07 | 94% | 75% | 0% | 2.4 |
| embeddinggemma | A floor 0.36 | 94% | 75% | 0% | 1.8 |
| embeddinggemma | floor 0.33 + prominence 0.08 | 94% | 75% | 0% | 2.2 |
| embeddinggemma-2 | A floor 0.71 (gate off, m=0) | 100% | 81% | 0% | 1.4 |

Prominence medians: single-relevant questions 0.149–0.312, multi-relevant 0.082–0.160, off-topic
0.029–0.049. Multi-relevant questions do have lower prominence, as feared, but they still cleared the
gate here.

- **nomic's battery recall problem doesn't show up on real episodes.** At 0.55 it already keeps a
  relevant episode for 94% of real questions with 0% off-topic FP. The gate's battery win comes from
  the battery's deliberately indirect paraphrases. On real questions, its only effect is to lower the
  floor, which doubles wrong episodes (1.2 → 2.4).
- "Wrong" here is loose. The real corpus is clustered, so a "wrong" gaming episode on a gaming question
  is often arguably related. Still, the gate adds prompt noise without measured benefit.

## Conclusion

A relative cut is **not supported by the evidence for shipping**:

- The prominence gate is the only variant that addresses nomic's overlap, and it does so on the
  synthetic battery (LOOCV J 0.25 → 0.70). But on real episodes it gave no recall gain and twice the
  wrong episodes.
- Margin, gap and rank cuts look good only because the battery has one right episode per question.
- The absolute floors held 0% off-topic FP on real episodes for all three models, including both
  knife-edge EmbeddingGemma floors.

Small N on both sides: 46 synthetic and 26 real queries over 21 real episodes. What would change this:
a real corpus where nomic visibly misses relevant episodes. The cheapest way to see that is to log the
top-5 scores and prominence on live turns, which costs nothing at request time.
