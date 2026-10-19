"""
Diagnostic: diag_embeddinggemma_episodic_recall_floor.py
=========================================================
Does EmbeddingGemma need a different episodic recall floor, or task
prefixes, before it can replace nomic for episodic memory?

Background: docs/architecture/16-runtime-backend-layer.md §16.23 / §16.27.
ControllerAgent's semantic episodic recall calls
EpisodicMemoryReader.by_similarity(task.instruction, top_n=5,
min_score=0.45). That floor was set while nomic-embed-text:latest was the
embedding model. On a live turn with embeddinggemma-300m-4bit (oMLX) the
right episode scored 0.387 and was dropped. EmbeddingGemma was also trained
with task prompts ("task: search result | query: …" for queries,
"title: none | text: …" for documents) that production never sends.

What this measures, per variant:
  - production (no prefixes) vs. the model's trained prefixes, for both
    embeddinggemma-300m-4bit (oMLX) and nomic-embed-text:latest (Ollama;
    its trained prefixes are "search_query: " / "search_document: ") —
    nomic-raw is the baseline the 0.45 floor was tuned on;
  - for each positive query: the right episode's cosine score and rank;
  - for each negative query (no relevant episode): the highest score of
    any episode, i.e. what would be injected as irrelevant memory;
  - recall at a floor (right episode in the top 5 AND >= floor), the
    false-positive rate at a floor (negative query with any episode >=
    floor), wrong episodes injected per positive query (top-5 episodes >=
    floor that aren't the right one), and the floor that maximises
    recall − false-positive rate;
  - the same at the floor production now uses for that model
    (planner.resolve_episodic_recall_floor(), §16.28 option A). Production
    sends no prefixes, so that column is the deployed behaviour only for
    the -raw variants.

The fixture (localist.threshold_calibration_fixtures.RECALL_*, shared with
live calibration) is SYNTHETIC — made-up facts shaped like real episodes (one
sentence starting "The user …", subject = content cut to 80 chars, embedded
as "{subject}. {content}" exactly as EpisodicMemoryWriter does), with
near-miss distractors in the same domains. Real episodes are personal data
and never belong in the repo. Queries are phrased as full user
instructions, since production embeds task.instruction.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
oMLX (:8000) and Ollama (:11434) embedding endpoints through the real
runtime clients — no stubs. Not part of the pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_embeddinggemma_episodic_recall_floor.py
Writes diagnostics/embeddinggemma_episodic_recall_floor_results.csv and
prints a summary.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.planner import resolve_episodic_recall_floor
from localist.threshold_calibration_fixtures import (
    RECALL_EPISODES,
    RECALL_NEGATIVES,
    RECALL_POSITIVES,
)

PRODUCTION_FLOOR = 0.45
TOP_N = 5
GEMMA = "embeddinggemma-300m-4bit"
NOMIC = "nomic-embed-text:latest"
OUT_CSV = pathlib.Path(__file__).with_name("embeddinggemma_episodic_recall_floor_results.csv")

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

EPISODES  = RECALL_EPISODES
POSITIVES = RECALL_POSITIVES
NEGATIVES = RECALL_NEGATIVES

VARIANTS: list[tuple[str, str, str, str]] = [
    # (name, model, query template, document template)
    ("gemma-raw",      GEMMA, "{}",                             "{}"),
    ("gemma-prefixed", GEMMA, "task: search result | query: {}", "title: none | text: {}"),
    ("nomic-raw",      NOMIC, "{}",                             "{}"),
    ("nomic-prefixed", NOMIC, "search_query: {}",               "search_document: {}"),
]


def _doc_text(content: str) -> str:
    """Exactly what EpisodicMemoryWriter embeds: f"{subject}. {content}"[:500]."""
    return f"{content[:80]}. {content}"[:500]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _embedders() -> dict[str, object]:
    return {
        GEMMA: OMLXRuntimeClient(chat_model="unused", embedding_model=GEMMA, base_url="http://127.0.0.1:8000"),
        NOMIC: OllamaRuntimeClient(chat_model="unused", embedding_model=NOMIC, base_url="http://127.0.0.1:11434"),
    }


def _metrics(pos_rows: list[dict], neg_max: list[float], floor: float) -> tuple[float, float]:
    recall = sum(1 for r in pos_rows if r["rank"] <= TOP_N and r["right_score"] >= floor) / len(pos_rows)
    fp = sum(1 for s in neg_max if s >= floor) / len(neg_max)
    return recall, fp


def main() -> int:
    embedders = _embedders()
    keys = list(EPISODES)
    rows: list[dict] = []
    summary: list[tuple] = []

    for name, model, qtpl, dtpl in VARIANTS:
        embed = embedders[model].embed
        try:
            doc_vecs = {k: embed(dtpl.format(_doc_text(EPISODES[k][1]))) for k in keys}
        except Exception as exc:
            print(f"{name}: embedding failed ({exc}) — is {model}'s backend running?", file=sys.stderr)
            return 1

        pos_rows = []
        pos_scores: list[tuple[str, list[tuple[float, str]]]] = []
        for query, right in POSITIVES:
            qv = embed(qtpl.format(query))
            scores = sorted(((_cos(qv, doc_vecs[k]), k) for k in keys), reverse=True)
            pos_scores.append((right, scores))
            rank = next(i for i, (_, k) in enumerate(scores, 1) if k == right)
            right_score = next(s for s, k in scores if k == right)
            row = {
                "variant": name, "kind": "positive", "query": query, "right": right,
                "right_score": round(right_score, 4), "rank": rank,
                "top1": scores[0][1], "top1_score": round(scores[0][0], 4),
                "best_wrong_score": round(max(s for s, k in scores if k != right), 4),
            }
            pos_rows.append(row)
            rows.append(row)

        neg_max = []
        for query in NEGATIVES:
            qv = embed(qtpl.format(query))
            best, best_k = max((_cos(qv, doc_vecs[k]), k) for k in keys)
            neg_max.append(best)
            rows.append({
                "variant": name, "kind": "negative", "query": query, "right": "",
                "right_score": "", "rank": "", "top1": best_k, "top1_score": round(best, 4),
                "best_wrong_score": round(best, 4),
            })

        def wrong_per_question(floor: float) -> float:
            return statistics.mean(
                sum(1 for s, k in scores[:TOP_N] if s >= floor and k != right)
                for right, scores in pos_scores
            )

        rec_prod, fp_prod = _metrics(pos_rows, neg_max, PRODUCTION_FLOOR)
        model_floor = resolve_episodic_recall_floor(model)
        rec_model, fp_model = _metrics(pos_rows, neg_max, model_floor)
        sweep = [(f / 100, *_metrics(pos_rows, neg_max, f / 100)) for f in range(10, 91)]
        best_floor, best_rec, best_fp = max(sweep, key=lambda t: (t[1] - t[2], -abs(t[0] - PRODUCTION_FLOOR)))
        summary.append((
            name,
            statistics.median(r["right_score"] for r in pos_rows),
            statistics.median(neg_max),
            sum(1 for r in pos_rows if r["rank"] == 1) / len(pos_rows),
            sum(1 for r in pos_rows if r["rank"] <= TOP_N) / len(pos_rows),
            rec_prod, fp_prod, wrong_per_question(PRODUCTION_FLOOR),
            best_floor, best_rec, best_fp,
            model_floor, rec_model, fp_model, wrong_per_question(model_floor),
        ))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(EPISODES)} episodes, {len(POSITIVES)} positive / {len(NEGATIVES)} negative queries, top_n={TOP_N}\n")
    print(f"{'variant':16} {'med right':>9} {'med neg max':>11} {'top1':>5} {'top5':>5} "
          f"{'recall@.45':>10} {'FP@.45':>7} {'wrong@.45':>9}   {'best floor':>10} {'recall':>6} {'FP':>5}"
          f"   {'model floor':>11} {'recall':>6} {'FP':>5} {'wrong':>5}")
    for (name, med_r, med_n, top1, top5, rec, fp, wrong, bf, brec, bfp, mf, mrec, mfp, mwrong) in summary:
        print(f"{name:16} {med_r:9.3f} {med_n:11.3f} {top1:5.0%} {top5:5.0%} "
              f"{rec:10.0%} {fp:7.0%} {wrong:9.1f}   {bf:10.2f} {brec:6.0%} {bfp:5.0%}"
              f"   {mf:11.2f} {mrec:6.0%} {mfp:5.0%} {mwrong:5.1f}")
    print("\n'model floor' = planner.resolve_episodic_recall_floor(model), what production uses now "
          "(production sends no prefixes, so it applies as deployed to the -raw variants only).")
    print(f"\nPer-query rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
