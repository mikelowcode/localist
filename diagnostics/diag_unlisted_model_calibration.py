"""
Diagnostic: diag_unlisted_model_calibration.py
===============================================
Does live calibration produce usable thresholds for an embedding model that
has no hand-validated entry, and does it reproduce the hand-validated values
when given the same weights under a name the tables don't list?

Background: docs/architecture/16-runtime-backend-layer.md §16.28. Option B
made the episodic recall floor calibrate per model, but the case it exists
for — a model with no _EPISODIC_RECALL_FLOORS / _VALIDATED_MODEL_THRESHOLDS
entry — was never run live: only nomic-embed-text:latest and
embeddinggemma-300m-4bit were installed, and both are in the tables.

Models (all through the real runtime clients, no stubs):
  - embeddinggemma-2:latest (Ollama) — the unlisted test model.
  - embeddinggemma:latest (Ollama) — the control: EmbeddingGemma 300M, the
    model behind the hand-validated embeddinggemma-300m-4bit (oMLX) floor
    of 0.37, under a name no table lists. Calibration should land near 0.37.
  - nomic-embed-text:latest (Ollama) — hand-validated reference; live
    calibration should reproduce its 0.55 floor.
  - embeddinggemma-300m-4bit (oMLX) — hand-validated reference, only if
    oMLX is reachable on :8000; skipped otherwise.

For each model this runs threshold_calibration.calibrate_thresholds() — the
exact function the app runs on an embedding-model switch or Re-embed Memory —
and reports the four gate thresholds (with Youden's J) and the recall floor.
It then re-scores the shared RECALL_* battery (no prefixes, as production
sends) to report recall / off-topic FP / wrong episodes per question at the
0.45 default, at the calibrated floor, and at the floor production would
resolve via planner.resolve_episodic_recall_floor(model, calibrated) — i.e.
what the controller would actually use for that model name.

READ-ONLY. Changes nothing in the codebase or any database. Not part of the
pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_unlisted_model_calibration.py
Writes diagnostics/unlisted_model_calibration_results.csv and prints a
summary.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.planner import (
    _EPISODIC_RECALL_FLOORS,
    _VALIDATED_MODEL_THRESHOLDS,
    resolve_episodic_recall_floor,
)
from localist.threshold_calibration import (
    calibrate_thresholds,
    recall_document_text,
)
from localist.threshold_calibration_fixtures import (
    RECALL_EPISODES,
    RECALL_NEGATIVES,
    RECALL_POSITIVES,
)

DEFAULT_FLOOR = 0.45
TOP_N = 5
OUT_CSV = pathlib.Path(__file__).with_name("unlisted_model_calibration_results.csv")

MODELS: list[tuple[str, str]] = [
    # (model, backend)
    ("embeddinggemma-2:latest",  "ollama"),
    ("embeddinggemma:latest",    "ollama"),
    ("nomic-embed-text:latest",  "ollama"),
    ("embeddinggemma-300m-4bit", "omlx"),
]


def _client(model: str, backend: str):
    if backend == "omlx":
        return OMLXRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:8000")
    return OllamaRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:11434")


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _score_battery(embed) -> tuple[list[tuple[str, list[tuple[float, str]]]], list[float], list[dict]]:
    keys = list(RECALL_EPISODES)
    doc_vecs = {k: embed(recall_document_text(RECALL_EPISODES[k][1])) for k in keys}
    pos_scores, rows = [], []
    for query, right in RECALL_POSITIVES:
        qv = embed(query)
        scores = sorted(((_cos(qv, doc_vecs[k]), k) for k in keys), reverse=True)
        pos_scores.append((right, scores))
        rank = next(i for i, (_, k) in enumerate(scores, 1) if k == right)
        rows.append({
            "kind": "positive", "query": query, "right": right,
            "right_score": round(next(s for s, k in scores if k == right), 4), "rank": rank,
            "top1": scores[0][1], "top1_score": round(scores[0][0], 4),
        })
    neg_max = []
    for query in RECALL_NEGATIVES:
        qv = embed(query)
        best, best_k = max((_cos(qv, doc_vecs[k]), k) for k in keys)
        neg_max.append(best)
        rows.append({
            "kind": "negative", "query": query, "right": "", "right_score": "", "rank": "",
            "top1": best_k, "top1_score": round(best, 4),
        })
    return pos_scores, neg_max, rows


def _at_floor(pos_scores, neg_max, floor: float) -> tuple[float, float, float]:
    recall = sum(
        1 for right, scores in pos_scores
        if any(k == right and s >= floor for s, k in scores[:TOP_N])
    ) / len(pos_scores)
    fp = sum(1 for s in neg_max if s >= floor) / len(neg_max)
    wrong = statistics.mean(
        sum(1 for s, k in scores[:TOP_N] if s >= floor and k != right)
        for right, scores in pos_scores
    )
    return recall, fp, wrong


def main() -> int:
    all_rows: list[dict] = []
    results: list[tuple[str, object, tuple | None]] = []

    for model, backend in MODELS:
        embed = _client(model, backend).embed
        try:
            dim = len(embed("probe"))
        except Exception as exc:
            print(f"{model} ({backend}): skipped — embed failed ({exc})", file=sys.stderr)
            continue
        print(f"{model} ({backend}): {dim}-dim, calibrating…", file=sys.stderr)

        calib = calibrate_thresholds(embed)
        pos_scores, neg_max, rows = _score_battery(embed)
        for r in rows:
            r["model"] = model
        all_rows.extend(rows)
        results.append((model, calib, (pos_scores, neg_max, dim)))

    if not results:
        print("No model could be embedded — is Ollama running?", file=sys.stderr)
        return 1

    with OUT_CSV.open("w", newline="") as f:
        fields = ["model", "kind", "query", "right", "right_score", "rank", "top1", "top1_score"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(RECALL_EPISODES)} episodes, {len(RECALL_POSITIVES)} positive / "
          f"{len(RECALL_NEGATIVES)} negative recall queries, top_n={TOP_N}\n")

    print("Gate calibration (threshold, TPR/FPR; '—' = degenerate, J below floor)")
    gate_names = list(results[0][1].gates)
    print(f"{'model':26} {'listed?':8} " + " ".join(f"{g:>24}" for g in gate_names))
    for model, calib, _ in results:
        listed = "yes" if model in _VALIDATED_MODEL_THRESHOLDS else "no"
        cells = []
        for g in gate_names:
            gc = calib.gates[g]
            cells.append(f"{'—':>24}" if gc.degenerate else
                         f"{gc.threshold:.2f} ({gc.tpr_at_threshold:.0%}/{gc.fpr_at_threshold:.0%})".rjust(24))
        print(f"{model:26} {listed:8} " + " ".join(cells))

    print("\nEpisodic recall floor (recall / off-topic FP / wrong episodes per question)")
    print(f"{'model':26} {'dim':>4} {'med right':>9} {'med neg':>7} {'top1':>5} {'top5':>5}"
          f"   {'@0.45':>17}   {'calibrated':>24}   {'production resolves':>24}")
    for model, calib, (pos_scores, neg_max, dim) in results:
        rights = [next(s for s, k in sc if k == r) for r, sc in pos_scores]
        ranks = [next(i for i, (_, k) in enumerate(sc, 1) if k == r) for r, sc in pos_scores]
        rf = calib.recall_floor
        cal_floor = None if rf is None or rf.degenerate else rf.threshold
        prod_floor = resolve_episodic_recall_floor(model, calib.thresholds)
        source = ("hand-validated" if model in _EPISODIC_RECALL_FLOORS
                  else "calibrated" if cal_floor is not None else "default")

        def fmt(floor):
            rec, fp, wrong = _at_floor(pos_scores, neg_max, floor)
            return f"{rec:4.0%} /{fp:4.0%} /{wrong:4.1f}"

        cal_cell = f"{cal_floor:.2f}: {fmt(cal_floor)}" if cal_floor is not None else f"— ({rf.reason})"
        print(f"{model:26} {dim:4d} {statistics.median(rights):9.3f} {statistics.median(neg_max):7.3f}"
              f" {sum(r == 1 for r in ranks) / len(ranks):5.0%} {sum(r <= TOP_N for r in ranks) / len(ranks):5.0%}"
              f"   {fmt(DEFAULT_FLOOR):>17}   {cal_cell:>24}   {f'{prod_floor:.2f} {source}: {fmt(prod_floor)}':>24}")
        print(f"{'':26} right-episode score range {min(rights):.3f}–{max(rights):.3f}; "
              f"worst off-topic {max(neg_max):.3f}")

    print(f"\nPer-query rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
