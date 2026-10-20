"""
Diagnostic: diag_relative_recall_cut.py
========================================
Can a relative cut on episodic recall beat the per-model absolute floor —
above all for nomic-embed-text:latest, whose right-episode and off-topic
scores overlap (0.50–0.57), capping it at 68% recall / 7% FP at 0.55?

Background: docs/architecture/16-runtime-backend-layer.md §16.28 open item;
diagnostics/reports/unlisted_model_calibration_assessment_2026-10-20.md
(both EmbeddingGemma floors sit ≤0.005 above their worst off-topic score).

Production today: by_similarity(query, top_n=5, min_score=floor) keeps the
top-5 episodes scoring >= floor. Variants, each applied to the same top 5:

  A  floor           keep s_i >= f                              (today)
  B  floor + margin  ... and s_i >= s_1 - d                     (wrong extras)
  C  floor + gap     ... truncated at the first drop s_i - s_i+1 > g
  D  floor + rank    ... and rank <= k
  E  gated           inject the floor-f set (optionally margin d) only if
                     s_1 >= f_high OR prominence >= m, where prominence =
                     s_1 - mean(s_2..s_10): how far the top episode stands
                     out from the background. The only variant that can
                     lower off-topic FP without raising the floor, since
                     B–D always keep the top episode when it passes f.

Metrics per variant (shared RECALL_* battery, no prefixes, as production
sends): recall (right episode kept), off-topic FP (negative query with
anything kept), wrong episodes kept per positive query. Parameters chosen
to maximise J = recall − FP, ties to fewer wrong episodes — the same
objective as threshold_calibration.calibrate_recall_floor(), so variant A
reproduces the calibrated floors.

Small battery (31 positive / 15 negative), and B–E tune 2–4 parameters on
it, so in-sample numbers flatter the richer variants. Leave-one-out
cross-validation is reported alongside: for each query, parameters are
re-chosen on the other 45 and scored on the held-out one. Judge variants on
the LOOCV columns.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama (and oMLX, if reachable) embedding endpoints through the real runtime
clients. Not part of the pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_relative_recall_cut.py
Writes diagnostics/relative_recall_cut_results.csv (per-model, per-variant
best parameters and in-sample / LOOCV metrics) and prints a summary.
"""

from __future__ import annotations

import csv
import itertools
import pathlib
import sys

import numpy as np

from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.threshold_calibration import recall_document_text
from localist.threshold_calibration_fixtures import (
    RECALL_EPISODES,
    RECALL_NEGATIVES,
    RECALL_POSITIVES,
)

TOP_N = 5
PROMINENCE_BACKGROUND = 10      # s_1 vs mean(s_2..s_10)
OUT_CSV = pathlib.Path(__file__).with_name("relative_recall_cut_results.csv")

MODELS: list[tuple[str, str]] = [
    ("nomic-embed-text:latest",  "ollama"),
    ("embeddinggemma:latest",    "ollama"),
    ("embeddinggemma-2:latest",  "ollama"),
    ("embeddinggemma-300m-4bit", "omlx"),
]

FLOORS   = [round(x / 100, 2) for x in range(10, 91)]
FLOORS_E = [round(x / 100, 2) for x in range(10, 91, 2)]
MARGINS  = [round(x / 100, 2) for x in range(1, 31)]
GAPS     = [round(x / 100, 2) for x in range(1, 21)]
RANKS    = [1, 2, 3, 4]
PROMS    = [round(x / 100, 2) for x in range(0, 21)]
MARGINS_E = [None] + [round(x / 100, 2) for x in range(2, 21, 2)]


def _client(model: str, backend: str):
    if backend == "omlx":
        return OMLXRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:8000")
    return OllamaRuntimeClient(chat_model="unused", embedding_model=model, base_url="http://127.0.0.1:11434")


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def _ranked_battery(embed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted scores R (Q x episodes, descending), right-episode mask
    (Q x episodes; all False for negatives), is_positive (Q,)."""
    keys = list(RECALL_EPISODES)
    docs = np.stack([_unit(embed(recall_document_text(RECALL_EPISODES[k][1]))) for k in keys])
    queries = [(q, keys.index(r)) for q, r in RECALL_POSITIVES] + [(q, -1) for q in RECALL_NEGATIVES]
    R, right = [], []
    for q, ri in queries:
        s = docs @ _unit(embed(q))
        order = np.argsort(-s)
        R.append(s[order])
        right.append(order == ri)
    return np.array(R), np.array(right), np.array([ri >= 0 for _, ri in queries])


class Battery:
    def __init__(self, R: np.ndarray, right: np.ndarray, pos: np.ndarray):
        self.R5 = R[:, :TOP_N]
        self.right5 = right[:, :TOP_N]
        self.pos = pos
        self.top1 = R[:, 0]
        self.prom = R[:, 0] - R[:, 1:PROMINENCE_BACKGROUND].mean(axis=1)
        gaps = self.R5[:, :-1] - self.R5[:, 1:]
        self.gap_sorted = gaps
        self.P, self.N = int(pos.sum()), int((~pos).sum())

    def outcome(self, kept: np.ndarray) -> np.ndarray:
        """Per query: (hit-or-fp, wrong). Positives: hit = right episode kept;
        negatives: fp = anything kept. wrong = non-right episodes kept (positives)."""
        hit_or_fp = np.where(self.pos, (kept & self.right5).any(axis=1), kept.any(axis=1))
        wrong = np.where(self.pos, (kept & ~self.right5).sum(axis=1), 0)
        return np.stack([hit_or_fp.astype(float), wrong.astype(float)])


def _variant_masks(b: Battery):
    """Yield (variant, params, kept mask Q x TOP_N)."""
    for f in FLOORS:
        yield "A floor", {"f": f}, b.R5 >= f
    for f, d in itertools.product(FLOORS, MARGINS):
        yield "B floor+margin", {"f": f, "d": d}, (b.R5 >= f) & (b.R5 >= b.top1[:, None] - d)
    for f, g in itertools.product(FLOORS, GAPS):
        ok = np.concatenate([np.ones((len(b.pos), 1), bool),
                             np.cumprod(b.gap_sorted <= g, axis=1).astype(bool)], axis=1)
        yield "C floor+gap", {"f": f, "g": g}, (b.R5 >= f) & ok
    rank_idx = np.arange(TOP_N)[None, :]
    for f, k in itertools.product(FLOORS, RANKS):
        yield "D floor+rank", {"f": f, "k": k}, (b.R5 >= f) & (rank_idx < k)
    for f, fh, m, d in itertools.product(FLOORS_E, FLOORS_E, PROMS, MARGINS_E):
        if fh <= f:
            continue
        gate = (b.top1 >= fh) | (b.prom >= m)
        kept = (b.R5 >= f) & gate[:, None]
        if d is not None:
            kept &= b.R5 >= b.top1[:, None] - d
        yield "E gated", {"f": f, "f_high": fh, "m": m, "d": d}, kept


def _evaluate(b: Battery) -> list[dict]:
    by_variant: dict[str, tuple[list[dict], list[np.ndarray]]] = {}
    for name, params, kept in _variant_masks(b):
        params_list, outcomes = by_variant.setdefault(name, ([], []))
        params_list.append(params)
        outcomes.append(b.outcome(kept))

    rows = []
    for name, (params_list, outcomes) in by_variant.items():
        O = np.stack(outcomes)                      # C x 2 x Q
        hit_fp, wrong = O[:, 0, :], O[:, 1, :]
        pos, neg = b.pos, ~b.pos

        def score(mask_q: np.ndarray) -> np.ndarray:
            """Selection score per combo over the queries in mask_q."""
            p, n = (mask_q & pos), (mask_q & neg)
            recall = hit_fp[:, p].sum(axis=1) / p.sum()
            fp = hit_fp[:, n].sum(axis=1) / n.sum()
            w = wrong[:, p].sum(axis=1) / p.sum()
            return (recall - fp) - 1e-3 * w

        all_q = np.ones(len(pos), bool)
        best = int(np.argmax(score(all_q)))
        recall = hit_fp[best, pos].mean()
        fp = hit_fp[best, neg].mean()
        w = wrong[best, pos].mean()

        # LOOCV: re-choose on the other queries, score the held-out one.
        cv = np.zeros((len(pos), 2))
        for q in range(len(pos)):
            mask = all_q.copy()
            mask[q] = False
            c = int(np.argmax(score(mask)))
            cv[q] = hit_fp[c, q], wrong[c, q]
        rows.append({
            "variant": name,
            "params": ", ".join(f"{k}={'—' if v is None else v}" for k, v in params_list[best].items()),
            "recall": recall, "fp": fp, "wrong": w, "j": recall - fp,
            "cv_recall": cv[pos, 0].mean(), "cv_fp": cv[neg, 0].mean(), "cv_wrong": cv[pos, 1].mean(),
            "cv_j": cv[pos, 0].mean() - cv[neg, 0].mean(),
            "combos": len(params_list),
        })
    return rows


def main() -> int:
    results: list[tuple[str, Battery, list[dict]]] = []
    for model, backend in MODELS:
        embed = _client(model, backend).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"{model} ({backend}): skipped — embed failed ({type(exc).__name__})", file=sys.stderr)
            continue
        print(f"{model} ({backend}): embedding battery, sweeping…", file=sys.stderr)
        b = Battery(*_ranked_battery(embed))
        results.append((model, b, _evaluate(b)))

    if not results:
        print("No model could be embedded — is Ollama running?", file=sys.stderr)
        return 1

    with OUT_CSV.open("w", newline="") as f:
        fields = ["model", "variant", "params", "recall", "fp", "wrong", "j",
                  "cv_recall", "cv_fp", "cv_wrong", "cv_j", "combos"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, _, rows in results:
            for r in rows:
                w.writerow({"model": model, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}})

    print(f"\n{len(RECALL_EPISODES)} episodes, {len(RECALL_POSITIVES)} positive / "
          f"{len(RECALL_NEGATIVES)} negative queries, top_n={TOP_N}; "
          f"prominence = s1 - mean(s2..s{PROMINENCE_BACKGROUND})")
    for model, b, rows in results:
        pos_prom, neg_prom = b.prom[b.pos], b.prom[~b.pos]
        print(f"\n{model}")
        print(f"  top-1 score: positives {b.top1[b.pos].min():.3f}–{b.top1[b.pos].max():.3f}, "
              f"off-topic {b.top1[~b.pos].min():.3f}–{b.top1[~b.pos].max():.3f}")
        print(f"  prominence:  positives {pos_prom.min():.3f}–{pos_prom.max():.3f} (median {np.median(pos_prom):.3f}), "
              f"off-topic {neg_prom.min():.3f}–{neg_prom.max():.3f} (median {np.median(neg_prom):.3f})")
        print(f"  {'variant':16} {'params':38} {'recall':>6} {'FP':>5} {'wrong':>5} {'J':>5}"
              f"   {'LOOCV recall':>12} {'FP':>5} {'wrong':>5} {'J':>5}")
        for r in rows:
            print(f"  {r['variant']:16} {r['params']:38} {r['recall']:6.0%} {r['fp']:5.0%} {r['wrong']:5.1f} {r['j']:5.2f}"
                  f"   {r['cv_recall']:12.0%} {r['cv_fp']:5.0%} {r['cv_wrong']:5.1f} {r['cv_j']:5.2f}")

    print(f"\nPer-variant rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
