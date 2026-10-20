"""
Diagnostic: diag_recall_floor_adjacent.py
==========================================
Does the per-model episodic recall floor (planner.resolve_episodic_recall_floor)
hold up against off-topic questions that sit close to the user's own topics —
tech news and products in the same domain as a stored episode — rather than
the far-off general-knowledge questions it was calibrated on?

Background: diagnostics/reports/memory_check_assessment_2026-10-20.md found
that on real tech-news traffic about half the episodes clearing nomic's 0.55
floor were only loosely related. The calibration battery's off-topic pool
(RECALL_NEGATIVES) is general knowledge, far from every episode.

Per model, on the shared battery (RECALL_EPISODES, embedded as
EpisodicMemoryWriter does; no prefixes):
  - the floor in use today (hand-validated, else what calibrate_recall_floor()
    picks), with recall, general off-topic FP and adjacent off-topic FP
    (RECALL_ADJACENT_NEGATIVES: same domain as an episode, about a different
    thing);
  - the floor re-selected by the same objective (_recall_floor_from_scores)
    with the calibration part of the adjacent pool added to the off-topic
    pool, scored on the held-out part (heldout_split()).

Optional real check (--real-db, --real-positives, --real-negatives): the
user's real episodes (a COPY of the DB, read-only), questions labelled with
the episode ids they need, and requests needing no memory. Reports, at each
floor, the share of positives whose right episode is recalled (top 5, at or
above the floor) and the share of negatives that get any episode injected.
Personal data: keep those files outside the repo; only aggregates print.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama embedding endpoint through the real runtime client. Not part of the
pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_recall_floor_adjacent.py \
        [--real-db COPY.db --real-positives P.json --real-negatives N.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import _EPISODIC_RECALL_FLOORS
from localist.threshold_calibration import (
    _cosine_similarity,
    _recall_floor_from_scores,
    recall_document_text,
)
from localist.threshold_calibration_fixtures import (
    RECALL_ADJACENT_NEGATIVES,
    RECALL_EPISODES,
    RECALL_NEGATIVES,
    RECALL_POSITIVES,
    heldout_split,
)

MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest", "embeddinggemma-2:latest"]
TOP_N = 5


def _ranked(embed, docs: dict, query: str) -> list[tuple[float, object]]:
    qv = embed(query)
    return sorted(((_cosine_similarity(qv, v), k) for k, v in docs.items()), reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-db", type=pathlib.Path)
    ap.add_argument("--real-positives", type=pathlib.Path, help='[{"text", "ids": [episode ids]}]')
    ap.add_argument("--real-negatives", type=pathlib.Path, help='[{"text"}]')
    args = ap.parse_args()
    real = None
    if args.real_db and args.real_positives and args.real_negatives:
        conn = sqlite3.connect(f"file:{args.real_db}?mode=ro", uri=True)
        real_eps = dict(conn.execute("SELECT id, content FROM episodes WHERE status = 'active'").fetchall())
        conn.close()
        real = (real_eps, json.loads(args.real_positives.read_text()), json.loads(args.real_negatives.read_text()))

    adj_cal, adj_ho = heldout_split(RECALL_ADJACENT_NEGATIVES)
    print(f"battery: {len(RECALL_EPISODES)} episodes, {len(RECALL_POSITIVES)} positives, "
          f"{len(RECALL_NEGATIVES)} general off-topic, {len(RECALL_ADJACENT_NEGATIVES)} adjacent off-topic "
          f"({len(adj_cal)} calibration / {len(adj_ho)} held-out)")
    if real:
        print(f"real: {len(real[0])} episodes, {len(real[1])} questions needing memory, "
              f"{len(real[2])} requests needing none")

    for model in MODELS:
        embed = OllamaRuntimeClient(chat_model="unused", embedding_model=model).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"\n{model}: skipped ({type(exc).__name__})", file=sys.stderr)
            continue
        docs = {k: embed(recall_document_text(c)) for k, (_t, c) in RECALL_EPISODES.items()}
        positives = []
        for q, right in RECALL_POSITIVES:
            r = _ranked(embed, docs, q)
            rank, score = next((i, s) for i, (s, k) in enumerate(r, 1) if k == right)
            positives.append((score, rank))
        general = [_ranked(embed, docs, q)[0][0] for q in RECALL_NEGATIVES]
        adj = {q: _ranked(embed, docs, q)[0][0] for q in RECALL_ADJACENT_NEGATIVES}

        today_cal = _recall_floor_from_scores(positives, general).threshold
        today = _EPISODIC_RECALL_FLOORS.get(model, today_cal)
        new = _recall_floor_from_scores(positives, general + [adj[q] for q in adj_cal])

        def at(floor):
            rec = sum(s >= floor and r <= TOP_N for s, r in positives) / len(positives)
            fp_g = sum(s >= floor for s in general) / len(general)
            fp_a = sum(adj[q] >= floor for q in adj_ho) / len(adj_ho)
            fp_all = sum(s >= floor for s in adj.values()) / len(adj)
            return f"recall {rec:4.0%}  general FP {fp_g:4.0%}  adjacent FP held-out {fp_a:4.0%} (all {fp_all:4.0%})"

        print(f"\n{model}")
        print(f"  adjacent off-topic best score {min(adj.values()):.3f}–{max(adj.values()):.3f}; "
              f"general off-topic max {max(general):.3f}; right episode {min(s for s, _ in positives):.3f}–"
              f"{max(s for s, _ in positives):.3f}")
        print(f"  today {today:.2f}{' (hand-validated)' if model in _EPISODIC_RECALL_FLOORS else ' (calibrated)'}: {at(today)}")
        if new.threshold is None:
            print(f"  re-selected with adjacent negatives: inconclusive ({new.reason})")
        else:
            print(f"  re-selected with adjacent negatives {new.threshold:.2f}: {at(new.threshold)}")

        if real:
            eps, rpos, rneg = real
            rdocs = {i: embed(recall_document_text(c)) for i, c in eps.items()}
            for label, floor in (("today", today), ("re-selected", new.threshold)):
                if floor is None:
                    continue
                hit = sum(any(k in set(p["ids"]) and s >= floor for s, k in _ranked(embed, rdocs, p["text"])[:TOP_N])
                          for p in rpos) / len(rpos)
                inj = sum(_ranked(embed, rdocs, n["text"])[0][0] >= floor for n in rneg) / len(rneg)
                print(f"  real at {label} {floor:.2f}: right memory recalled {hit:4.0%}; "
                      f"memory injected into {inj:4.0%} of requests needing none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
