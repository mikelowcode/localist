# Recall Floor vs. Off-Topic Questions Near the User's Own Topics

**Date:** 2026-10-20
**Script:** `diagnostics/diag_recall_floor_adjacent.py`
**New fixture:** `threshold_calibration_fixtures.RECALL_ADJACENT_NEGATIVES` (data only)
**Models under test:** `nomic-embed-text:latest`, `embeddinggemma:latest`, `embeddinggemma-2:latest`
(real Ollama).
**Status:** READ-ONLY. Nothing that runs was changed.

**Trigger:** `memory_check_assessment_2026-10-20.md`. On real tech-news traffic, about half the
episodes clearing nomic's 0.55 recall floor were only loosely related. The recall battery's off-topic
pool (`RECALL_NEGATIVES`) is general knowledge, far from every episode, so it can't show this.

## Method

- **New fixture.** `RECALL_ADJACENT_NEGATIVES` holds 20 off-topic questions in the same domain as a
  `RECALL_EPISODES` entry but about a different thing, so the user's fact is not relevant. For
  example, the PlayStation owner is asked about Xbox Game Pass titles, and the Pixel 8 owner about the
  iPhone 17 camera. The set is mostly news-button style ("Tell me more about this news story: …"), and
  was written before scoring.
- **Battery comparison.** Per model, compare the floor in use today with the floor re-selected by the
  same objective (`_recall_floor_from_scores`) after adding the calibration part of the adjacent pool.
  The re-selected floor is scored on the held-out part.
- **Real check.** A sandbox copy of the user's 21 active episodes:
  - 16 questions labelled with the episodes they need;
  - 32 typed requests needing no memory;
  - 23 Live Feed news-button requests ("Tell me more about this news story: …").
  
  Personal data, kept in the session scratchpad; only aggregates are recorded.
- **Prominence gate.** The 2026-10-20 relative-recall variant "floor + prominence" (top score minus
  the mean of ranks 2–10) was re-run on the real data with its battery parameters.

## Results — battery

| Model | Floor today | Right episode | General off-topic max | Adjacent off-topic | Adjacent FP (all) | General FP |
|---|---|---|---|---|---|---|
| nomic | 0.55 (hand) | 0.38–0.75 | 0.57 | 0.40–0.60 | **45%** | 7% |
| embeddinggemma | 0.36 (calibrated) | 0.27–0.65 | 0.35 | 0.29–0.45 | **40%** | 0% |
| embeddinggemma-2 | 0.71 (calibrated) | 0.61–0.81 | 0.71 | 0.65–0.75 | **20%** | 0% |

Re-selecting with the adjacent pool: embeddinggemma and embeddinggemma-2 keep the same floor; nomic is
inconclusive (best J 0.48). **Adjacent off-topic questions score inside the range of right episodes on
every model, so no absolute floor separates them.**

## Results — real episodes

| Model | Setup | Right memory recalled | Injected: typed (none needed) | Injected: news buttons |
|---|---|---|---|---|
| nomic | floor 0.55 | 15 / 16 | 18 / 32 | 17 / 23 |
| nomic | floor 0.50 + prominence 0.07 | 15 / 16 | 16 / 32 | **11 / 23** |
| embeddinggemma | floor 0.36 | 15 / 16 | 13 / 32 | 9 / 23 |
| embeddinggemma | floor 0.33 + prominence 0.08 | 15 / 16 | 11 / 32 | 9 / 23 |
| embeddinggemma-2 | floor 0.71 | 16 / 16 | 18 / 32 | 6 / 23 |
| embeddinggemma-2 | floor 0.71 (prominence 0) | 16 / 16 | 18 / 32 | 6 / 23 |

(Counted with top-5 retrieval as production does. The real typed numbers differ slightly from the
`diag_recall_floor_adjacent.py` printout, which reports the share of requests whose single best
episode clears the floor.)

## Findings

1. **Near-topic memory injection is built into similarity recall when a user's episodes and questions
   share a domain.**
   - The user's episodes are mostly about their tech projects and hardware, and much of their traffic
     is tech news and project talk.
   - On every model, an off-topic question about a neighbouring product scores like a real recall.
2. **Tuning the floor can't fix it.** Floors re-selected with near-topic negatives didn't move, or
   became inconclusive.
3. **The prominence gate helps a little, on nomic only.**
   - nomic news-button injections fall from 17 to 11 of 23, with no recall loss on the 16 real
     questions.
   - It barely changes embeddinggemma and does nothing for embeddinggemma-2.
   - On 2026-10-20 the same gate showed no recall gain on real questions and more wrong episodes per
     question, because it lowers the floor to 0.50.
4. **Cost per injection is small.** Memories are capped at 5 bullets and 150 tokens (Slot 3a), and
   relevant ones rank first. With the 2026-10-20 chip change, every injection is now visible to the
   user.

## Options (not built)

- **A. Accept it.** The prompt cost is bounded, and the chip now shows when memory is used.
- **B. Skip semantic recall on Live Feed button turns.** These are "Tell me more about this news
  story" and "Summarize the release notes" requests, which arrive with the article or release already
  pinned in the request context. It is a deterministic and cheap change. It would remove most news
  injections, but also the occasional useful one (the user's own console, recalled on a story about it).
- **C. Adopt the prominence gate for nomic only,** or per model where it measures as a gain. It is a
  modest gain, and adds a second recall parameter to calibrate per model.

**Decision (user, 2026-10-20): A — accept.** Near-topic injection stays. It is bounded, it ranks
relevance first, and it is now visible through the episodic chip. No recall change was built.
