# Unlisted-Model Calibration Assessment — `embeddinggemma-2:latest` (+ `embeddinggemma:latest` control)

**Date:** 2026-10-20
**Script:** `diagnostics/diag_unlisted_model_calibration.py`
**Raw data:** `diagnostics/unlisted_model_calibration_results.csv`
**Models under test:** `embeddinggemma-2:latest`, `embeddinggemma:latest`, `nomic-embed-text:latest`
(real Ollama `/api/embed`). `embeddinggemma-300m-4bit` (oMLX) was skipped: oMLX wasn't running.
**Status:** READ-ONLY diagnostic, plus a live end-to-end check in a sandbox (scratch copy of the DB,
`.env`, wiki/ and raw/ via `LOCALIST_BACKEND_ROOT`). The real DB and repo `.env` were not touched.

**Trigger:** docs/architecture/16-runtime-backend-layer.md §16.28. Option B's per-model calibrated
recall floor had never run live on the case it exists for: a model with no entry in
`_EPISODIC_RECALL_FLOORS` / `_VALIDATED_MODEL_THRESHOLDS`.

## Models

| Model | Size | Dim | Norm | In hand-validated tables? |
|---|---|---|---|---|
| `embeddinggemma-2:latest` | 1.3 GB, 744M params, nvfp4 | 768 | 1.0 | No |
| `embeddinggemma:latest` | 621 MB (EmbeddingGemma 300M) | 768 | 1.0 | No (same model family as `embeddinggemma-300m-4bit`, which is) |
| `nomic-embed-text:latest` | 274 MB | 768 | 1.0 | Yes |

`ollama show embeddinggemma-2` reports `embedding length 512`, but `/api/embed` returns 768-dim
vectors. Presumably a projection after the base model; harmless, since vectors are stored per model.

## Method

`threshold_calibration.calibrate_thresholds()` — the exact function the app runs on an
embedding-model switch or Re-embed Memory — per model, then a re-score of the shared
`RECALL_*` battery (30 episodes, 31 positive / 15 negative queries, no prefixes, top_n=5) at the
0.45 default, at the calibrated floor, and at what `resolve_episodic_recall_floor(model,
calibrated)` returns for that model name.

## Results

### Recall floor (recall / off-topic FP / wrong episodes per question)

| Model | Median right | Median off-topic max | Top-1 | Top-5 | @ 0.45 | Calibrated | Production resolves |
|---|---|---|---|---|---|---|---|
| `embeddinggemma-2:latest` | 0.758 | 0.658 | 87% | 97% | 97% / **100%** / 4.0 | **0.71**: 84% / 0% / 0.5 | 0.71 calibrated |
| `embeddinggemma:latest` | 0.463 | 0.297 | 90% | 100% | 55% / 0% / 0.1 | **0.36**: 90% / 0% / 0.7 | 0.36 calibrated |
| `nomic-embed-text:latest` | 0.609 | 0.461 | 81% | 94% | 94% / 60% / 3.0 | 0.55: 68% / 7% / 0.4 | 0.55 hand-validated |

- **`embeddinggemma-2` scores high, the opposite of its predecessor.** Right episodes 0.611–0.814,
  but off-topic questions also score up to 0.705. At the 0.45 default it would inject memory into
  every off-topic question and ~4 wrong episodes into every real one — without calibration this model
  is unusable for episodic memory. Calibrated 0.71 is clean (0% FP), but sits 0.005 above the worst
  off-topic score: the same knife edge as the 300m model's 0.37 vs 0.367.
- **Control reproduces the hand-validated value.** `embeddinggemma:latest` calibrates to 0.36 vs the
  hand-validated 0.37 for `embeddinggemma-300m-4bit` (different quantisation and server, one grid step
  apart). Live calibration also reproduces nomic's 0.55 exactly.

### Gates (threshold, TPR / FPR; — = degenerate, J < 0.5)

| Model | explicit_search_action | lookup_request | research_intent | episodic_relevance |
|---|---|---|---|---|
| `embeddinggemma-2:latest` | — (J 0.06) | — (J 0.35) | 0.74 (90% / 6%) | 0.70 (85% / 30%) |
| `embeddinggemma:latest` | — | — | 0.52 (100% / 12%) | 0.48 (85% / 15%) |
| `nomic-embed-text:latest` (calibrated; hand-validated values win in production) | — | 0.62 (100% / 24%) | 0.52 (100% / 35%) | 0.58 (77% / 5%) |

For `embeddinggemma-2`, the two degenerate gates fall to the lexical/BM25 tier, as designed.
`episodic_relevance` clears J but at 30% FPR — loose; worth watching.

## Live end-to-end (sandbox, dev backend on :8011)

1. `POST /settings/embedding-model {"model":"embeddinggemma-2:latest"}` → calibration identical to the
   diagnostic; `recall_floor` 0.71, `recall_floor_in_use` 0.71; `.env` (sandbox) persisted
   `LOCALIST_EMBEDDING_MODEL_OLLAMA=embeddinggemma-2:latest`.
2. `embedding_model_thresholds` row: `embeddinggemma-2:latest | NULL | NULL | 0.74 | 0.70 | 0.71`.
3. `POST /memory/reembed` → 57 corpus + 114 chat turns re-embedded, 71 episodes already current;
   recalibrated to the same 0.71.
4. Live turns (`gemma4:31b-cloud`):
   - A personal question about a fact stored in one episode → log `1 new-from-semantic (floor
     0.71)`; answered correctly from episodic memory.
   - "How does photosynthesis convert light into chemical energy?" → `0 new-from-semantic (floor 0.71)`.

The unlisted-model path (calibrate → persist → resolve → controller uses it) works live.

## Implications for later phases

- Score scales now span 0.36 (EmbeddingGemma 300M) to 0.71 (EmbeddingGemma 2) for the recall floor
  alone, which strengthens the case that the profile-facts 0.45 (Phase 4) needs to be per model:
  under `embeddinggemma-2` it is very likely far too permissive.
- Both EmbeddingGemma floors are knife-edge (≤0.005 margin over the worst off-topic). Phase 2's
  relative-cut measurement should include both, not just nomic.
- Not addressed: `embeddinggemma-2` is not added to the hand-validated table, so it stays the live
  test case for unlisted models.
