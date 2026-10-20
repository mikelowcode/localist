# Memory-Check Assessment — Phase 1A of the memory-check plan

**Date:** 2026-10-20
**Script:** `diagnostics/diag_memory_check.py`
**New fixtures:** `threshold_calibration_fixtures.EP_HARD_NEGATIVES_V3` (data only)
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest`, `embeddinggemma-2:latest`
(real Ollama), plus the model-independent keyword candidates.
**Status:** READ-ONLY. Nothing that runs was changed.

## What the check does today

`Planner._priority5_episodic` fires on a deterministic keyword list (`_EPISODIC_KEYWORDS`, substring
match), or on the embedding gate against `_EPISODIC_RELEVANCE_TEMPLATES` (or its BM25 fallback).

It **does not decide whether memory is recalled**: Step 5 recall runs on every conversational turn
(04-planner-routing-model.md §4.3a), and the per-model recall floor decides what is injected. A
firing check:

- sets `fetch_episodic`, which drives the chat's "◎ episodic" provenance chip;
- makes the turn priority 5 instead of 6, which pre-empts the optional P6 tool-fallback classifier
  (off by default).

So a false positive gives a misleading chip, not extra memory in the prompt.

## Method

- **Candidates.** Each candidate's decision is the keyword part OR the gate part, as in
  production. Thresholds are chosen by Youden's J on the calibration part only (`heldout_split()`).

  | Candidate | Rule |
  |---|---|
  | K | current keyword list |
  | K+E | production today: current templates at the model's current threshold (hand-validated, else calibrated v2) |
  | K′ | a tightened keyword list: first-person and recall phrases matched on word boundaries; no generic single words such as "before", "wrong", "decided", "previously", "what do you know" |
  | K′+T | K′ plus revised templates, cosine, calibrated per model |
  | K′+C | K′ plus model-independent keyword coverage on the revised templates |

  Candidates were written before scoring.
- **Battery.**
  - Positives: the existing episodic positives plus v2's, 21 in all.
  - Negatives: 44 — the existing ones, v2's, and 16 new hard negatives. The hard negatives contain
    the keyword list's generic words ("what's wrong with this regex", "what happened before the
    french revolution", "what do you know about black holes") or are first-person requests that need
    no memory.
  - A few hard negatives are borderline ("write a birthday message for my sister" could use a stored
    name). They are labelled as written, not relabelled after scoring.
- **Real data** (personal; kept in the session scratchpad, only aggregates recorded):
  - Negatives: the user's 32 typed requests, none of which needed memory.
  - Positives: 16 questions about the user's own stored facts, from the 2026-10-20 relative-recall
    real-data check.
- **Chip replay.** All 55 stored chat turns were replayed through `EpisodicMemoryReader.by_similarity()`
  (nomic, floor 0.55) against a sandbox copy of the user's episodes, and compared with whether the chip
  was shown (stored `fetch_episodic`).
  - The replay uses today's episode bank, which includes episodes created from those same chats.

## Results — the check

Held-out: 7 positives / 14 negatives. Real: 16 memory questions / 32 typed requests.

| Candidate | Held-out caught / false | Real caught / false |
|---|---|---|
| K (today's keywords) | 3 / 4 | 0 / **4** |
| **K′ (tightened keywords)** | 3 / **0** | 0 / **0** |
| K′+C (coverage ≥ 0.33) | 6 / 1 | 3 / 5 |
| nomic: K+E (production) | 6 / 4 | 0 / 5 |
| nomic: **K′+E** (today's gate, 0.62) | 6 / 0 | 0 / **1** |
| nomic: K′+T (revised, 0.58) | 6 / 1 | 2 / 2 |
| embeddinggemma: K+E (production) | 6 / 5 | 0 / 4 |
| embeddinggemma: **K′+E** (today's gate, 0.48) | 6 / 1 | 0 / **0** |
| embeddinggemma: K′+T (revised, 0.48) | 6 / 3 | 4 / 4 |
| embeddinggemma-2: K+E (production) | 6 / 5 | 1 / 11 |
| embeddinggemma-2: **K′+E** (today's gate, 0.74) | 6 / 2 | 1 / **7** |
| embeddinggemma-2: K′+T (revised, 0.72) | 7 / 5 | 5 / 11 |

- **The tightened keyword list is a strict improvement.** It removes every keyword false alarm
  (4 real, 4 held-out) and catches the same positives.
- **Revised templates and keyword coverage made real false alarms worse**, so both are rejected.
- **Recalibrating today's gate with the hard negatives added didn't move any threshold.**
  `embeddinggemma-2` stays at 0.74 with 7 real false alarms. Those come from phrasings the battery
  doesn't cover.
- **No candidate recognises plain factual memory questions** ("where do I live?", "what console do I
  play on?"), since they contain no recall wording. This costs nothing in answers, because recall
  runs anyway. But it means `fetch_episodic` can't serve as a "memory was used" signal.

## Results — the chip (55 replayed turns)

| | Relevant memory recalled | No relevant memory recalled |
|---|---|---|
| Chip shown | 2 | 1 |
| Chip not shown | **34** | 18 |

- The chip misled once, but stayed silent on 34 turns where at least one episode cleared the recall
  floor and went into the prompt.
- A spot check of 10 of those 34 found about half genuinely related (a gaming news story recalling
  the user's console; Localist release notes recalling the Localist repository) and about half loosely
  related (an AI-model news story recalling an unrelated open-source project).
- On real tech-news traffic, nomic's 0.55 recall floor admits loosely related memories more often
  than the 2026-10-19 synthetic battery suggested. That battery's off-topic questions were general
  knowledge, not topics adjacent to the user's own episodes.

## Recommendations for Phase 1B

1. **Replace `_EPISODIC_KEYWORDS` with the tightened, word-boundary list (K′),** for every model.
2. **Keep the current embedding gate and templates.** The revised-template and coverage candidates
   are rejected.
3. **Drive the chip from what recall actually did.** Show it only when at least one
   relevance-selected episode (semantic hit or graph neighbour, not recency priming) reached the
   prompt, reported as a new metadata field.
   - `fetch_episodic` keeps its routing role.
   - On this user's traffic the honest chip would show on most turns, because memories really are
     injected that often.
4. **Follow-up, not in this plan:** re-measure the recall floor against off-topic questions adjacent
   to the user's own topics (tech news, products they own), since real traffic shows loose matches
   the synthetic battery missed.
