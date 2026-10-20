"""
Diagnostic: diag_search_intent_baseline.py
===========================================
Baseline for the four semantic gates (explicit_search_action, lookup_request,
research_intent, episodic_relevance) before the search-intent test set is
rebuilt — Phase A of the 2026-10-20 search-intent plan.

Per embedding model, and per threshold source a gate can resolve to
(planner.resolve_gate_tiers()):
  - hand-validated  _VALIDATED_MODEL_THRESHOLDS (nomic only);
  - calibrated      what threshold_calibration.calibrate_thresholds() picks
                    today, re-derived here with the same _youden_calibrate();
  - lexical         BM25 against the same templates, at
                    _LEXICAL_FALLBACK_THRESHOLDS (model-independent);
  - deployed        whichever of those production actually resolves to.

Two scorings of the same battery:
  - "raw"     each phrase embedded as written — what calibration does;
  - "lowered" each phrase lowercased first — what production does
              (Planner scores `lowered`; templates are embedded as written in
              both). The gap between the two is a calibration/production
              mismatch for case-sensitive tokenizers.

Optional real-request check (--real PATH): a JSON list of labelled requests
({"text", "search", "research", "memory", "ambiguous"}). It reports, for each
model's deployed tiers, whether a web search would fire — from the semantic
gates alone, and together with the deterministic keyword / explicit-date
rules — against the "search" label, plus research and memory. Real requests
are personal data: keep that file outside the repo; only aggregates are
printed. The negative-filter tie-break (a model call) isn't simulated; the
number of requests it could affect is reported instead.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama embedding endpoint through the real runtime client. Not part of the
pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_search_intent_baseline.py [--real /path/to/labels.json]
Writes diagnostics/search_intent_baseline_results.csv (battery only).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

from localist import bm25
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import (
    _ALL_SEARCH_NEGATIVE_FILTERS,
    _EPISODIC_RELEVANCE_TEMPLATES,
    _LEXICAL_FALLBACK_THRESHOLDS,
    _SEARCH_INTENT_TEMPLATES,
    _VALIDATED_MODEL_THRESHOLDS,
    _WEB_SEARCH_KEYWORDS,
    _has_explicit_date,
)
from localist.threshold_calibration import (
    _LR_ESA_GRID,
    _RI_EP_GRID,
    _build_gate_phrase_pools,
    _cosine_similarity,
    _youden_calibrate,
)

MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest", "embeddinggemma-2:latest"]
GATES = ["explicit_search_action", "lookup_request", "research_intent", "episodic_relevance"]
GRIDS = {"explicit_search_action": _LR_ESA_GRID, "lookup_request": _LR_ESA_GRID,
         "research_intent": _RI_EP_GRID, "episodic_relevance": _RI_EP_GRID}
OUT_CSV = pathlib.Path(__file__).with_name("search_intent_baseline_results.csv")


def _templates(gate: str) -> tuple[str, ...]:
    return _EPISODIC_RELEVANCE_TEMPLATES if gate == "episodic_relevance" else _SEARCH_INTENT_TEMPLATES[gate]


def _bm25_score(text: str, gate: str) -> float:
    phrases = _templates(gate)
    scores = bm25.score_documents(text.lower(), [(i, p) for i, p in enumerate(phrases)])
    return max(scores.values()) if scores else 0.0


def _rates(pos: list[float], neg: list[float], t: float | None) -> tuple[float, float] | None:
    if t is None or not pos or not neg:
        return None
    return (sum(s >= t for s in pos) / len(pos), sum(s >= t for s in neg) / len(neg))


def _fmt(r: tuple[float, float] | None) -> str:
    return "—" if r is None else f"{r[0]:4.0%}/{r[1]:4.0%} J{r[0] - r[1]:+.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=pathlib.Path, help="labelled real requests (JSON); kept outside the repo")
    args = ap.parse_args()

    pools = _build_gate_phrase_pools()
    real = json.loads(args.real.read_text()) if args.real else None
    rows: list[dict] = []

    # Lexical tier: model-independent, BM25 lowercases internally.
    lexical = {g: _rates([_bm25_score(p, g) for p in pools[g][0]],
                         [_bm25_score(p, g) for p in pools[g][1]],
                         _LEXICAL_FALLBACK_THRESHOLDS[g]) for g in GATES}

    print("Battery (current fixtures): pool sizes " + ", ".join(
        f"{g} {len(pools[g][0])}+/{len(pools[g][1])}-" for g in GATES))
    print("Cells: TPR/FPR and Youden's J at that tier's threshold. '—' = no threshold at that tier.\n")
    print(f"lexical (all models): " + "  ".join(f"{g} {_fmt(lexical[g])}" for g in GATES) + "\n")

    for model in MODELS:
        embed = OllamaRuntimeClient(chat_model="unused", embedding_model=model).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"{model}: skipped ({type(exc).__name__})", file=sys.stderr)
            continue
        tvecs = {g: [embed(p) for p in _templates(g)] for g in GATES}

        def score(text: str, g: str) -> float:
            v = embed(text)
            return max(_cosine_similarity(v, t) for t in tvecs[g])

        print(model)
        hand = _VALIDATED_MODEL_THRESHOLDS.get(model, {})
        deployed: dict[str, tuple[str, float | None]] = {}
        for g in GATES:
            pos, neg = pools[g]
            raw_p, raw_n = [score(p, g) for p in pos], [score(p, g) for p in neg]
            low_p, low_n = [score(p.lower(), g) for p in pos], [score(p.lower(), g) for p in neg]
            cal = _youden_calibrate(raw_p, raw_n, GRIDS[g])      # what calibration picks (raw text)
            cal_t = None if cal.degenerate else cal.threshold
            best_j = max((sum(s >= t for s in raw_p) / len(raw_p) - sum(s >= t for s in raw_n) / len(raw_n))
                         for t in GRIDS[g])
            if g in hand:
                deployed[g] = ("hand", hand[g])
            elif cal_t is not None:
                deployed[g] = ("calibrated", cal_t)
            else:
                deployed[g] = ("lexical", None)
            dep_rates = (lexical[g] if deployed[g][0] == "lexical"
                         else _rates(low_p, low_n, deployed[g][1]))
            print(f"  {g:23} pos {min(raw_p):.2f}–{max(raw_p):.2f}  neg max {max(raw_n):.2f}  "
                  f"best J {best_j:+.2f}")
            print(f"  {'':23} hand {_fmt(_rates(low_p, low_n, hand.get(g)))}   "
                  f"calibrated {('—' if cal_t is None else f'{cal_t:.2f} ')}"
                  f"raw {_fmt(_rates(raw_p, raw_n, cal_t))} / lowered {_fmt(_rates(low_p, low_n, cal_t))}   "
                  f"deployed: {deployed[g][0]} {_fmt(dep_rates)}")
            print(f"  {'':23} lowercasing shifts scores by {statistics.mean(abs(a - b) for a, b in zip(raw_p + raw_n, low_p + low_n)):.3f} on average")
            for kind, phrases, sc in (("positive", pos, raw_p), ("negative", neg, raw_n)):
                for p, s in zip(phrases, sc):
                    rows.append({"model": model, "gate": g, "kind": kind, "phrase": p, "raw_score": round(s, 4),
                                 "lowered_score": round(score(p.lower(), g), 4),
                                 "bm25": round(_bm25_score(p, g), 4)})

        if real is not None:
            typed = [r for r in real if not r.get("ambiguous")]

            def fires(text: str, g: str) -> bool:
                tier, t = deployed[g]
                lowered = text.lower()
                if tier == "lexical":
                    return _bm25_score(lowered, g) >= _LEXICAL_FALLBACK_THRESHOLDS[g]
                return score(lowered, g) >= t

            def keyword_search(text: str) -> bool:
                lowered = text.lower()
                return any(k in lowered for k in _WEB_SEARCH_KEYWORDS) or _has_explicit_date(lowered)

            gate_search = [fires(r["text"], "explicit_search_action") or fires(r["text"], "lookup_request") for r in typed]
            any_search = [g or keyword_search(r["text"]) for g, r in zip(gate_search, typed)]
            want = [r["search"] for r in typed]

            def cm(pred, truth):
                tp = sum(p and t for p, t in zip(pred, truth)); fn = sum(t and not p for p, t in zip(pred, truth))
                fp = sum(p and not t for p, t in zip(pred, truth))
                return f"{tp}/{tp + fn} caught, {fp} false"

            research = [r for r, s in zip(typed, any_search) if s]
            ri = [fires(r["text"], "research_intent") for r in research]
            # Embedding check only; _priority5_episodic's keyword list is local to that method.
            mem_pred = [fires(r["text"], "episodic_relevance") for r in typed]
            filt = sum(any(f in r["text"].lower() for f in _ALL_SEARCH_NEGATIVE_FILTERS) for r in typed)
            print(f"  real requests ({len(typed)} typed, {sum(want)} should search):")
            print(f"    search, embedding checks alone: {cm(gate_search, want)};  with keyword/date rules: {cm(any_search, want)}")
            print(f"    research upgrade (of searches): {cm(ri, [r['research'] for r in research])};  "
                  f"memory check fired on {sum(mem_pred)} ({sum(r['memory'] for r in typed)} labelled as needing memory);  "
                  f"negative-filter phrases matched: {filt}")
        print()

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-phrase rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
