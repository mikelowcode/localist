# Document RAG Cutoff Assessment — Part 2, Phase 2A

**Date:** 2026-10-20
**Script:** `diagnostics/diag_rag_cutoff.py`
**New fixture:** `threshold_calibration_fixtures.RAG_DOCS` / `RAG_POSITIVES` / `RAG_NEGATIVES` (data only)
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest`, `embeddinggemma-2:latest`
(real Ollama).
**Status:** READ-ONLY (temporary databases only). Nothing that runs was changed.

## What the cutoff controls

One fixed cosine cutoff of 0.55 decides two things for every embedding model:

- **Routing.** `Planner._priority4_corpus` Path B sends a turn to RAG when `query_corpus(q,
  max_results=1)`'s top document scores ≥ 0.55. Only the top 2 BM25 candidates are cosine re-ranked
  here.
- **Sources.** `ControllerAgent` Step 4 (RAG fetch), Step 3b (fallback when web search fails) and
  Step 4b (graph assist) keep documents of `query_corpus(q, max_results=3)` scoring ≥ 0.55. The top 6
  BM25 candidates are re-ranked.

## Method

- **Battery.** `RAG_DOCS` is a synthetic knowledge base shaped like a real corpus:
  - 12 wiki pages with OKF front matter — `index_document()` embeds only the first 500 characters, so
    front matter is most of the embedded text — plus plain raw versions of 6 of them;
  - 24 paraphrased positives, each listing every document that answers it;
  - 20 negatives: 8 general off-topic and 12 near-topic (same domain as a document, different
    subject).
  
  It was written before any scoring.
- **Real code path.** Documents are indexed into a temporary `MemoryManager` with each model's live
  embeddings and queried through the real `query_corpus()`. Both decisions are measured.
- **Selection.** The cutoff is chosen with the episodic recall floor's objective: maximise sources
  recall − FP over 0.10–0.90, ties to the highest. It is chosen on the calibration part and scored on
  the held-out part (`heldout_split()`: 8 positives / 6 negatives).
- **Real check.** Copies of the user's corpus: 7 wiki pages and 7 raw files, with persona, index,
  logs, MEMORY and profile files skipped as production does.
  - Questions: 12, labelled with the wiki and raw copies that answer them.
  - Negatives, run separately: 10 general off-topic questions, and the user's 23 Live Feed
    news-button requests (AI and tech news, near these documents).
  - Personal data, kept in the session scratchpad; only aggregates are recorded.

## Results — synthetic battery (held-out)

| Model | Right-document scores | Off-topic best | At 0.55 | Calibrated |
|---|---|---|---|---|
| nomic | 0.46–0.77 | 0.38–0.59 | 100% recall / 17% FP, 0.0 wrong | **0.56**: same |
| embeddinggemma | 0.36–0.73 | 0.13–0.46 | **50% recall** / 0% FP | **0.42**: 75% recall / 0% FP |
| embeddinggemma-2 | 0.67–0.84 | 0.58–0.78 | 100% recall / **100% FP**, 1.5 wrong | **0.76**: 62% recall / 0% FP |

Routing moved with sources for every model.

## Results — the user's real documents

Sources recall / FP and wrong per question; routing TP / FP.

| Model | Cutoff | 12 questions | General off-topic | News buttons |
|---|---|---|---|---|
| nomic | 0.55 | 100%, 0.8 wrong; routing 100% | 0% | 52% FP; routing 35% |
| nomic | 0.56 | 100%, 0.8 wrong; routing 92% | 0% | 39% FP; routing 26% |
| embeddinggemma | 0.55 | **50%**, 0.2 wrong; routing 42% | 0% | 4% FP |
| embeddinggemma | **0.42** | **100%**, 0.8 wrong; routing 75% | 0% | 17% FP |
| embeddinggemma-2 | 0.55 | 100%, 1.1 wrong | **100%** FP, routing 100% | **100%** FP |
| embeddinggemma-2 | **0.76** | 100%, 0.5 wrong; routing 83% | **0%** | **4%** FP |

## Findings

1. **0.55 is right for nomic and wrong for both EmbeddingGemma models**, in opposite directions:
   - EmbeddingGemma 300M scores low: at 0.55 it drops half the right documents on real questions.
     Calibrated 0.42 keeps all of them, with 0% general off-topic.
   - embeddinggemma-2 scores high: at 0.55 every question, including general off-topic, gets
     documents and is routed to RAG. Calibrated 0.76 keeps every right document on real questions
     with 0–4% off-topic.
2. **Calibration on the synthetic battery transfers to real documents** for all three models.
3. **nomic's calibrated value (0.56) differs from 0.55 by one grid step.** It trades a little routing
   recall (100% → 92%) for fewer news-button sources (52% → 39%).
4. **Near-topic news still draws documents under nomic** (39–52%). This is the same near-topic
   overlap the recall floor showed. It is bounded at 3 sources × 2,000 characters.

## Proposal for Phase 2B (same pattern as the recall floor and the profile threshold)

- **Thresholds.** `_RAG_SOURCE_FLOORS` (hand-validated): `nomic-embed-text:latest` 0.55, which keeps
  nomic's behaviour exactly. `resolve_rag_floor()` resolves hand-validated, then calibrated, then
  0.55. EmbeddingGemma 300M and embeddinggemma-2 calibrate, to about 0.42 and 0.76.
- **Calibration.** `calibrate_rag_floor()` runs inside `calibrate_thresholds()` on `RAG_*` through a
  temporary `MemoryManager`, exactly as measured here. It is stored in a new
  `embedding_model_thresholds` column (schema v21), and `CALIBRATION_VERSION` goes to 3.
- **Wiring.** The Planner's Priority 4 Path B and the controller's three source filters use the
  resolved floor.
- **API and UI.** `CalibrationResponse.rag_floor` / `rag_floor_in_use`; Settings adds "Document
  cutoff 0.xx".

## Phase 2B — as built (2026-10-20)

- **Decision (user):** nomic stays hand-validated at 0.55; every other model calibrates.
- **Calibrated values.** Production calibration (`calibrate_rag_floor()`) runs on the whole `RAG_*`
  battery, not just the calibration part, and lands slightly lower than the Phase 2A
  calibration-part values:

  | Model | Phase 2B (whole battery) | Phase 2A (calibration part) |
  |---|---|---|
  | embeddinggemma | 0.40 | 0.42 |
  | embeddinggemma-2 | 0.72 | 0.76 |
  | nomic | 0.57 (overridden by the hand-validated 0.55) | 0.56 |

- **Re-checked on the user's real documents at the shipped values.** Both EmbeddingGemma models
  still keep every right document with 0% general off-topic. Near-topic news-button turns that draw
  documents rise to 22% for both, against 17% at 0.42 and 4% at 0.76: the cost of a single
  calibration procedure for every model instead of per-model tuning.
- **Live (sandbox):** a question about the user's Localist docs routed to RAG under
  embeddinggemma-2 (0.869 ≥ 0.72); "How does photosynthesis work?" got no documents.
- **Related issue, not in scope.** The live check returned three copies of one page. Each backend root
  sharing the database (source checkout, packaged app) indexes its own copy of the same file, so
  duplicates can take RAG slots.
