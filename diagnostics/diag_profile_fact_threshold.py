"""
Diagnostic: diag_profile_fact_threshold.py
===========================================
Is the single 0.45 profile-fact threshold right for each embedding model?

Background: ControllerAgent._score_profile_facts(instruction_embedding,
top_n=5, threshold=0.45) picks which lines of the user profile
(wiki/users/<name>.md, embedded raw, one line at a time) go into Slot 3b
(docs/architecture/03-unified-prompt-contract.md §3.6). 0.45 was set by
hand, never measured, and is the same for every model — while Phase 1/2 of
the 2026-10-20 plan measured episodic recall floors from 0.36 to 0.71
across models. Step 5b also runs on every turn regardless of route (§3.6
says P6 turns don't inject profile facts; the code has no such check), so
this threshold is the only gate on profile injection.

What this measures, per model, on the shared PROFILE_* battery (synthetic
profile, 30 lines / 30 positive / 16 negative queries):
  - per positive query: the best relevant line's score and rank;
  - per negative query: the best score of any line (what would be injected);
  - at a threshold: recall (a relevant line in the top 5 and >= threshold),
    all-relevant recall (every relevant line kept), off-topic FP (negative
    query with any line >= threshold), irrelevant lines kept per positive
    query, lines kept per negative query;
  - the threshold maximising recall − FP — threshold_calibration.
    _recall_floor_from_scores(), the episodic recall floor's objective, so a
    calibrated profile threshold would select exactly this value — plus its
    leave-one-out cross-validated recall/FP;
  - which profile sections leak into off-topic queries at 0.45.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama (and oMLX, if reachable) embedding endpoints through the real runtime
clients. Not part of the pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_profile_fact_threshold.py
Writes diagnostics/profile_fact_threshold_results.csv and prints a summary.
"""

from __future__ import annotations

import collections
import csv
import pathlib
import statistics
import sys

import numpy as np

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.threshold_calibration import _recall_floor_from_scores
from localist.threshold_calibration_fixtures import (
    PROFILE_LINES,
    PROFILE_NEGATIVES,
    PROFILE_POSITIVES,
)

PRODUCTION_THRESHOLD = 0.45
TOP_N = 5
OUT_CSV = pathlib.Path(__file__).with_name("profile_fact_threshold_results.csv")

MODELS: list[tuple[str, str]] = [
    ("nomic-embed-text:latest",  "ollama"),
    ("embeddinggemma:latest",    "ollama"),
    ("embeddinggemma-2:latest",  "ollama"),
    ("embeddinggemma-300m-4bit", "omlx"),
]


def _client(model: str, backend: str):
    if backend == "omlx":
        return OMLXRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:8000")
    return OllamaRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:11434")


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def _metrics(pos: list[tuple[list[tuple[float, str]], tuple[str, ...]]],
             neg: list[list[tuple[float, str]]], t: float) -> dict:
    def kept(ranked):
        return {k for s, k in ranked[:TOP_N] if s >= t}
    hits = [bool(kept(r) & set(rel)) for r, rel in pos]
    all_hits = [set(rel) <= kept(r) or (len(rel) > TOP_N) for r, rel in pos]
    return {
        "recall": statistics.mean(hits),
        "all_recall": statistics.mean(all_hits),
        "fp": statistics.mean(bool(kept(r)) for r in neg),
        "wrong": statistics.mean(len(kept(r) - set(rel)) for r, rel in pos),
        "neg_lines": statistics.mean(len(kept(r)) for r in neg),
    }


def _calibration_inputs(pos, neg):
    """(best relevant line's score, its rank) per positive; best score per negative."""
    positives = []
    for ranked, rel in pos:
        rank, score = next((i, s) for i, (s, k) in enumerate(ranked, 1) if k in rel)
        positives.append((score, rank))
    return positives, [r[0][0] for r in neg]


def _loocv(pos, neg) -> dict:
    positives, negatives = _calibration_inputs(pos, neg)
    hits, fps = [], []
    for i in range(len(positives)):
        t = _recall_floor_from_scores(positives[:i] + positives[i + 1:], negatives).threshold
        s, rank = positives[i]
        hits.append(t is not None and rank <= TOP_N and s >= t)
    for i in range(len(negatives)):
        t = _recall_floor_from_scores(positives, negatives[:i] + negatives[i + 1:]).threshold
        fps.append(t is not None and negatives[i] >= t)
    return {"recall": statistics.mean(hits), "fp": statistics.mean(fps)}


def main() -> int:
    keys = list(PROFILE_LINES)
    csv_rows: list[dict] = []
    summary: list[tuple] = []

    for model, backend in MODELS:
        embed = _client(model, backend).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"{model} ({backend}): skipped — embed failed ({type(exc).__name__})", file=sys.stderr)
            continue
        print(f"{model} ({backend}): embedding battery…", file=sys.stderr)
        L = np.stack([_unit(embed(PROFILE_LINES[k][1])) for k in keys])

        def ranked(q: str) -> list[tuple[float, str]]:
            s = L @ _unit(embed(q))
            return sorted(((float(s[i]), keys[i]) for i in range(len(keys))), reverse=True)

        pos = [(ranked(q), rel) for q, rel in PROFILE_POSITIVES]
        neg = [ranked(q) for q in PROFILE_NEGATIVES]
        for (r, rel), (q, _) in zip(pos, PROFILE_POSITIVES):
            rank, score = next((i, s) for i, (s, k) in enumerate(r, 1) if k in rel)
            csv_rows.append({"model": model, "kind": "positive", "query": q, "relevant": " ".join(rel),
                             "best_relevant_score": round(score, 4), "best_relevant_rank": rank,
                             "top1": r[0][1], "top1_score": round(r[0][0], 4)})
        for r, q in zip(neg, PROFILE_NEGATIVES):
            csv_rows.append({"model": model, "kind": "negative", "query": q, "relevant": "",
                             "best_relevant_score": "", "best_relevant_rank": "",
                             "top1": r[0][1], "top1_score": round(r[0][0], 4)})

        calib = _recall_floor_from_scores(*_calibration_inputs(pos, neg))
        leaks = collections.Counter(
            PROFILE_LINES[k][0] for r in neg for s, k in r[:TOP_N] if s >= PRODUCTION_THRESHOLD
        )
        summary.append((model, pos, neg, calib, _loocv(pos, neg), leaks))

    if not summary:
        print("No model could be embedded — is Ollama running?", file=sys.stderr)
        return 1

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n{len(PROFILE_LINES)} profile lines, {len(PROFILE_POSITIVES)} positive / "
          f"{len(PROFILE_NEGATIVES)} negative queries, top_n={TOP_N}")
    print("Cells: recall / all-relevant recall / off-topic FP / irrelevant lines per positive / lines per off-topic")
    for model, pos, neg, calib, cv, leaks in summary:
        positives, negatives = _calibration_inputs(pos, neg)
        rel_scores = [s for s, _ in positives]

        def cell(t):
            m = _metrics(pos, neg, t)
            return (f"{m['recall']:4.0%} / {m['all_recall']:4.0%} / {m['fp']:4.0%} / "
                    f"{m['wrong']:.1f} / {m['neg_lines']:.1f}")

        print(f"\n{model}")
        print(f"  best relevant line: {min(rel_scores):.3f}–{max(rel_scores):.3f} (median {statistics.median(rel_scores):.3f}); "
              f"off-topic best: {min(negatives):.3f}–{max(negatives):.3f} (median {statistics.median(negatives):.3f}); "
              f"relevant line ranked first {sum(r == 1 for _, r in positives) / len(positives):.0%}")
        print(f"  at 0.45 (today):     {cell(PRODUCTION_THRESHOLD)}")
        if calib.degenerate:
            print(f"  calibrated:          — ({calib.reason})")
        else:
            print(f"  calibrated {calib.threshold:.2f}:     {cell(calib.threshold)}"
                  f"   LOOCV recall {cv['recall']:.0%}, FP {cv['fp']:.0%}")
        print(f"  off-topic leaks at 0.45 by section: "
              + (", ".join(f"{s} {n}" for s, n in leaks.most_common()) or "none"))

    print(f"\nPer-query rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
