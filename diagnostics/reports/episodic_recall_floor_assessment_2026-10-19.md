# Episodic Recall Floor Assessment — `embeddinggemma-300m-4bit` vs `nomic-embed-text:latest`

**Date:** 2026-10-19
**Script:** `diagnostics/diag_embeddinggemma_episodic_recall_floor.py`
**Raw data:** `diagnostics/embeddinggemma_episodic_recall_floor_results.csv`
**Models under test:** `embeddinggemma-300m-4bit` (real oMLX `/v1/embeddings`) and
`nomic-embed-text:latest` (real Ollama `/api/embed`). No stubs.
**Status:** READ-ONLY. Nothing in the codebase or any database was changed.

**Trigger:** docs/architecture/16-runtime-backend-layer.md §16.23 / §16.27 open item. Episodic
recall (`ControllerAgent` → `EpisodicMemoryReader.by_similarity(task.instruction, top_n=5,
min_score=0.45)`) uses one floor, set while nomic was the embedding model. On a live turn with
EmbeddingGemma the right episode scored 0.387 and was dropped. EmbeddingGemma also never gets the
task prompts it was trained with.

## Method

- **Fixture:** synthetic, committed in the script. Real episodes are personal data.
  - 30 episodes shaped like real ones: one sentence starting "The user …", with subject = content
    cut to 80 characters, embedded as `"{subject}. {content}"` exactly as `EpisodicMemoryWriter`
    does. They include near-miss distractors in the same domains (gaming, hardware, APIs, food,
    location).
  - 31 positive queries, each with one right episode. They are paraphrased and indirect, phrased as
    full user instructions (production embeds `task.instruction`), and never copy the episode's
    wording.
  - 15 negative queries (general-knowledge questions) that no episode is relevant to.
- **Variants:**
  - `gemma-raw`: production as it is today.
  - `gemma-prefixed`: query `task: search result | query: …`, document `title: none | text: …`.
  - `nomic-raw`: the baseline the 0.45 floor was tuned on.
  - `nomic-prefixed`: `search_query: ` / `search_document: `.
- **Metrics, per floor:**
  - **Recall:** the right episode is in the top 5 and scores at or above the floor.
  - **Off-topic false positives:** a negative query has any episode at or above the floor, so
    irrelevant memory gets injected.
  - **Wrong episodes injected per real question:** top-5 episodes at or above the floor that aren't
    the right one.

## Results

| Variant | Median right-episode score | Median best off-topic score | Right episode ranked 1st | In top 5 | Recall @0.45 | Off-topic FP @0.45 |
|---|---:|---:|---:|---:|---:|---:|
| gemma-raw      | 0.453 | 0.292 | 84% | 100% | **52%** | 0% |
| gemma-prefixed | 0.424 | 0.178 | 87% |  97% | 42% | 0% |
| nomic-raw      | 0.609 | 0.461 | 81% |  94% | 94% | **60%** |
| nomic-prefixed | 0.644 | 0.559 | 74% |  94% | 94% | 100% |

Each cell below is recall / off-topic false positives / wrong episodes injected per real question:

| Floor | gemma-raw | gemma-prefixed | nomic-raw |
|---:|---|---|---|
| 0.30 | 97% / 27% / 2.4 | **90% / 0% / 0.9** | 94% / 100% / 4.1 |
| 0.37 | **87% / 0% / 0.7** | 71% / 0% / 0.3 | 94% / 100% / 4.1 |
| 0.45 | 52% / 0% / 0.1 | 42% / 0% / 0.0 | 94% / 60% / 3.0 |
| 0.50 | 45% / 0% / – | 23% / 0% / – | 94% / 33% / – |
| 0.55 | 19% / 0% / 0.0 | 10% / 0% / 0.0 | **68% / 7% / 0.4** |

Worst off-topic score, the highest floor that still leaks: gemma-raw 0.367, gemma-prefixed 0.294,
nomic-raw 0.570.

## Findings

1. **The 0.45 floor is wrong for EmbeddingGemma, and systematically so.** It drops about half the
   right episodes (52% recall), even though the right episode was in the top 5 for every query.
   The live 0.387 miss was typical, not a fluke. EmbeddingGemma ranks well but scores low overall.
2. **The same floor is too loose for nomic, the daily-driver model.** At 0.45, nomic injects some
   memory into 60% of off-topic questions, and about 3 wrong episodes into every real question
   alongside the right one. nomic's scores are compressed into a narrow band: right episodes
   0.50–0.65, off-topic up to 0.57. That's a pre-existing prompt-noise cost on today's default
   setup, separate from EmbeddingGemma.
3. **Without prefixes, EmbeddingGemma has a usable but knife-edge floor of about 0.37:** 87% recall,
   0% off-topic false positives, 0.7 wrong episodes per question. The worst off-topic score is
   0.367, only 0.003 below it.
4. **Task prefixes help EmbeddingGemma's separation, not its absolute scores.** Prefixed off-topic
   scores fall much further than right-episode scores (median gap 0.25 vs 0.16 raw). At a floor of
   0.30 the result is 90% recall, 0% off-topic false positives and 0.9 wrong episodes per question,
   the best balance measured, with a little more headroom over the worst off-topic score (0.294).
   It also makes three different misses than raw, all low-information queries such as "Find me a
   dinner recipe" against a vegetarian episode.
5. **Prefixes don't help nomic.** Off-topic scores rise more than right-episode scores.
6. **No single floor serves both models.** The best floors measured are about 0.37 (gemma raw), 0.30
   (gemma prefixed) and about 0.55 (nomic). Like the Planner's gate thresholds, the recall floor has
   to be per model.

## Caveats

- Synthetic fixture with small N (31 positive, 15 negative). Treat the floors as estimates to
  within a few hundredths, not exact values. Embeddings are deterministic, so a rerun reproduces
  these numbers exactly, but a different fixture would move them.
- Off-topic negatives here are general-knowledge questions. Real off-topic turns that share
  vocabulary with stored episodes would score higher.
- Only the reader's floor was measured. The Planner's `episodic_relevance` gate, which decides
  whether episodic fetch runs at all, is separate and already calibrated per model.

## Options (not built)

- **A. Per-model recall floor, hand-validated.** A small table next to
  `_VALIDATED_MODEL_THRESHOLDS`, e.g. nomic ≈ 0.55 and embeddinggemma ≈ 0.37, with 0.45 as the
  fallback for unknown models. Smallest change. It would also cut nomic's wrong-memory injection.
- **B. Calibrate the floor per model**, alongside the existing gate calibration
  (`threshold_calibration.py`), from a committed episode/query fixture like this one. It covers any
  model without hand tuning, but it's more machinery.
- **C. EmbeddingGemma task prefixes.** A modest gain in separation. It needs every episode
  re-embedded with the document prefix. Because §16.26's archive is keyed by model name, prefixed
  vectors must be stored under a distinct key (e.g. `embeddinggemma-300m-4bit#prompted`) so they
  never mix with unprefixed ones. Worth doing only together with A or B.

## Retest after option A (2026-10-19)

Option A is built: `planner._EPISODIC_RECALL_FLOORS` (nomic 0.55, EmbeddingGemma 0.37, 0.45 default)
via `resolve_episodic_recall_floor()`, resolved once per `ControllerAgent`.

**Diagnostic rerun** ("model floor" column = the floor production now uses):

| Variant | Before (0.45): recall / off-topic FP / wrong per question | Now: recall / off-topic FP / wrong per question |
|---|---|---|
| gemma-raw | 52% / 0% / 0.1 | **87% / 0% / 0.7** (floor 0.37) |
| nomic-raw | 94% / 60% / 3.0 | **68% / 7% / 0.4** (floor 0.55) |

nomic floor sweep (recall / off-topic FP / wrong per question): 0.50 94%/33%/1.5 · 0.52 74%/20%/1.1 ·
0.53 71%/20%/0.7 · 0.55 68%/7%/0.4 · 0.56 65%/7%/0.3. nomic's right-episode scores (0.50–0.54 for the
10 lost queries) overlap its off-topic scores (up to 0.57), so no fixed floor separates them
cleanly. Of the 10 queries lost at 0.55, 7 still had the right episode ranked first or second. One is
the almond/tree-nut-allergy query, the kind of fact that shouldn't be dropped. That points at a
relative cut (score margin over the rest, or rank) rather than an absolute floor for nomic, which is
worth folding into option B's design.

**Real episodes** (sandbox copy of the packaged database). The same `by_similarity()` path, with five
questions about real episodes and two off-topic. The queries were written after seeing the
episodes, so their scores run high.
- nomic: the right episode is still ranked first for all five; extra injected episodes fell (5→3,
  3→1); the off-topic "TCP vs UDP" question went from 5 irrelevant episodes to 0.
- EmbeddingGemma: the right episode is ranked first for all five (0.55–0.57); 1–2 more neighbours
  admitted; still 0 for off-topic.

**Live turns through the controller** (sandbox, `POST /task`):
- The log shows `floor 0.55` on Ollama/nomic and `floor 0.37` on oMLX/EmbeddingGemma, so the
  wiring is confirmed. "What console do I usually play on?" was answered correctly on both (that
  episode is a `preference`, so recency priming supplied it).
- **New finding, pre-existing and not caused by option A.** "What machine did I build and test my
  orchestration layer on?" (a `project_fact`): on both backends semantic recall added 3 episodes at
  the new floor, but the prompt still got exactly 5 bullets and both models answered "I don't have
  that information". `select_top_episodes()` sorts by episode type first (correction → decision →
  preference → workflow → project_fact → …) and caps at 5, ignoring relevance. Recency priming
  returns up to 5 preference/workflow episodes, so any `project_fact` from semantic recall, which is
  what Mode 3 exists to surface, is cut whenever the user has 5 or more such episodes (the sandbox
  has 8). In that situation floor tuning can't help facts reach the prompt. This dates from the
  2026-09-06 5-bullet cap fix.
