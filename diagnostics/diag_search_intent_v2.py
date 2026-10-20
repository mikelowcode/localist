"""
Diagnostic: diag_search_intent_v2.py
=====================================
Phase B of the 2026-10-20 search-intent plan: does a rebuilt test set,
lowercased calibration, revised search templates and a length-robust lexical
fallback improve the semantic gates for every embedding model?

Background: diagnostics/reports/search_intent_baseline_2026-10-20.md (Phase A).

Configurations, per model:
  A  today       current templates, calibrated on phrases as written (raw)
  B  lowercase   current templates, calibrated on lowercased phrases, which is
                 how production scores the request
  C  candidate   CANDIDATE_TEMPLATES below for the two search gates, plus B
research_intent and episodic_relevance keep their templates (A vs B only).

Battery: threshold_calibration_fixtures *_V2 lists plus the current battery's
pools, split by heldout_split(): thresholds are chosen (Youden's J, the same
_youden_calibrate() calibration uses) on the calibration part only and scored
on the held-out part. A gate whose J falls below _MIN_ACCEPTABLE_J would fall
to the lexical tier in production; the "deployed" search decision models
that.

Lexical fallback variants (model-independent), each calibrated the same way:
  L0 bm25        today: BM25 max over templates, the request as the query
  L1 bm25/token  L0 divided by the request's token count
  L2 coverage    max over templates of the share of the template's content
                 words present in the request (stopwords removed), 0–1
  L3 coverage×len  L2 × min(1, 12 / request tokens): long messages are
                 rarely search commands

The search decision is "explicit_search_action OR lookup_request fires"
(both trigger web_search).

Optional real-request check (--real PATH): labelled requests, as in Phase A.
Personal data: keep the file outside the repo; only aggregates are printed.
research_intent's real-request row is not reported: it only upgrades a search
to the price/spec research loop, which the labels don't capture.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama embedding endpoint through the real runtime client. Not part of the
pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_search_intent_v2.py [--real /path/to/labels.json]
Writes diagnostics/search_intent_v2_results.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

from localist import bm25
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import _EPISODIC_RELEVANCE_TEMPLATES, _SEARCH_INTENT_TEMPLATES
from localist.threshold_calibration import (
    _LR_ESA_GRID,
    _MIN_ACCEPTABLE_J,
    _RI_EP_GRID,
    _build_gate_phrase_pools,
    _cosine_similarity,
    _youden_calibrate,
)
from localist.threshold_calibration_fixtures import (
    EP_NEGATIVES_V2,
    EP_POSITIVES_V2,
    ESA_POSITIVES_V2,
    LR_POSITIVES_V2,
    RI_NEGATIVES_V2,
    RI_POSITIVES_V2,
    SEARCH_NEGATIVES_V2,
    heldout_split,
)

MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest", "embeddinggemma-2:latest"]
SEARCH = ["explicit_search_action", "lookup_request"]
OTHER = ["research_intent", "episodic_relevance"]
GRIDS = {"explicit_search_action": _LR_ESA_GRID, "lookup_request": _LR_ESA_GRID,
         "research_intent": _RI_EP_GRID, "episodic_relevance": _RI_EP_GRID}
OUT_CSV = pathlib.Path(__file__).with_name("search_intent_v2_results.csv")

# Written before any v2 scoring. Explicit-search: commands to search the web
# or a named site. Lookup: requests to fetch facts from outside, in command and
# question form, including "research <topic>". All lowercase, no specific
# subjects, no "look at" wording (it collides with "look at this image").
CANDIDATE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "explicit_search_action": (
        "search the web for this",
        "do a web search for this",
        "search online for this",
        "google this",
        "search the internet for this",
        "run a web search on this",
        "search github for this",
        "find web results about this",
    ),
    "lookup_request": (
        "look this up",
        "look up this topic",
        "look up information about this",
        "find out about this",
        "find information on this",
        "can you look this up for me",
        "what can you find out about this",
        "check online what the latest is on this",
        "research this topic",
        "do some research on this",
    ),
}

_STOPWORDS = frozenset(
    "a an the this that these those it is are was be to of for on in at about and or "
    "me my you your i we can could would will please do does what how".split()
)


def _pools() -> dict[str, tuple[list[str], list[str]]]:
    old = _build_gate_phrase_pools()
    lr_old_pos, search_old_neg = old["lookup_request"]
    return {
        "explicit_search_action": (ESA_POSITIVES_V2, SEARCH_NEGATIVES_V2 + search_old_neg),
        "lookup_request":         (LR_POSITIVES_V2 + lr_old_pos, SEARCH_NEGATIVES_V2 + search_old_neg),
        "research_intent":        (old["research_intent"][0] + RI_POSITIVES_V2,
                                   old["research_intent"][1] + RI_NEGATIVES_V2),
        "episodic_relevance":     (old["episodic_relevance"][0] + EP_POSITIVES_V2,
                                   old["episodic_relevance"][1] + EP_NEGATIVES_V2),
    }


def _templates(gate: str, candidate: bool) -> tuple[str, ...]:
    if gate == "episodic_relevance":
        return _EPISODIC_RELEVANCE_TEMPLATES
    if candidate and gate in CANDIDATE_TEMPLATES:
        return CANDIDATE_TEMPLATES[gate]
    return _SEARCH_INTENT_TEMPLATES[gate]


# -- lexical variants -----------------------------------------------------------

def _lexical(variant: str, text: str, templates: tuple[str, ...]) -> float:
    tokens = bm25.tokenize(text)
    if variant in ("L0", "L1"):
        scores = bm25.score_documents(text.lower(), [(i, t) for i, t in enumerate(templates)])
        s = max(scores.values()) if scores else 0.0
        return s if variant == "L0" else s / max(1, len(tokens))
    present = set(tokens)
    cov = max((len(c & present) / len(c)) for t in templates
              if (c := {w for w in bm25.tokenize(t) if w not in _STOPWORDS}))
    return cov if variant == "L2" else cov * min(1.0, 12 / max(1, len(tokens)))


def _youden_any(pos: list[float], neg: list[float]) -> tuple[float | None, float]:
    """Best threshold over the observed scores (lexical scales differ per variant)."""
    best = (None, -1.0)
    for t in sorted(set(pos + neg)):
        j = sum(s >= t for s in pos) / len(pos) - sum(s >= t for s in neg) / len(neg)
        if j > best[1]:
            best = (t, j)
    return best


def _rate(scores: list[float], t: float | None) -> float:
    return 0.0 if t is None or not scores else sum(s >= t for s in scores) / len(scores)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=pathlib.Path)
    args = ap.parse_args()
    real = [r for r in json.loads(args.real.read_text()) if not r.get("ambiguous")] if args.real else None

    pools = _pools()
    split = {g: (heldout_split(p), heldout_split(n)) for g, (p, n) in pools.items()}
    rows: list[dict] = []

    # Held-out search decision: a held-out positive of either search gate should
    # fire; a held-out shared negative shouldn't.
    ho_search_pos = split["explicit_search_action"][0][1] + split["lookup_request"][0][1]
    ho_search_neg = split["lookup_request"][1][1]

    print("Pools (calibration + held-out): " + ", ".join(
        f"{g} {len(p)}+/{len(n)}-" for g, (p, n) in pools.items()))

    # -- Lexical variants, on current and candidate templates ------------------
    print("\nLexical fallback (model-independent): search decision on held-out "
          f"({len(ho_search_pos)} positives / {len(ho_search_neg)} negatives)"
          + (f" and on real requests ({sum(r['search'] for r in real)} / "
             f"{sum(not r['search'] for r in real)})" if real else ""))
    lexical_choice: dict[tuple[str, bool], dict] = {}
    for candidate in (False, True):
        for v in ("L0", "L1", "L2", "L3"):
            thr, ok = {}, True
            for g in SEARCH:
                (cp, _hp), (cn, _hn) = split[g]
                t, j = _youden_any([_lexical(v, x, _templates(g, candidate)) for x in cp],
                                   [_lexical(v, x, _templates(g, candidate)) for x in cn])
                thr[g] = t
                ok &= j >= _MIN_ACCEPTABLE_J

            def fires(text):
                return any(_lexical(v, text, _templates(g, candidate)) >= thr[g] for g in SEARCH)

            tpr = sum(fires(x) for x in ho_search_pos) / len(ho_search_pos)
            fpr = sum(fires(x) for x in ho_search_neg) / len(ho_search_neg)
            line = (f"  {'candidate' if candidate else 'current  '} templates {v}: held-out "
                    f"{tpr:4.0%} caught / {fpr:4.0%} false (J {tpr - fpr:+.2f}){'' if ok else ' [a gate below J floor]'}")
            if real:
                tp = sum(fires(r["text"]) and r["search"] for r in real)
                fp = sum(fires(r["text"]) and not r["search"] for r in real)
                line += f";  real {tp}/{sum(r['search'] for r in real)} caught, {fp} false"
            print(line)
            lexical_choice[(v, candidate)] = {"thr": thr, "fires": fires}

    # -- Embedding configurations ----------------------------------------------
    for model in MODELS:
        embed = OllamaRuntimeClient(chat_model="unused", embedding_model=model).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"\n{model}: skipped ({type(exc).__name__})", file=sys.stderr)
            continue
        cache: dict[str, list[float]] = {}

        def vec(text: str) -> list[float]:
            if text not in cache:
                cache[text] = embed(text)
            return cache[text]

        print(f"\n{model}")
        for name, candidate, lowercase in (("A today", False, False), ("B lowercase", False, True),
                                           ("C candidate", True, True)):
            gates = SEARCH + (OTHER if not candidate else [])
            thr: dict[str, float | None] = {}
            for g in gates:
                tv = [vec(t) for t in _templates(g, candidate)]

                def score(text: str, lower: bool) -> float:
                    return max(_cosine_similarity(vec(text.lower() if lower else text), t) for t in tv)

                (cp, hp), (cn, hn) = split[g]
                cal = _youden_calibrate([score(x, lowercase) for x in cp], [score(x, lowercase) for x in cn], GRIDS[g])
                thr[g] = None if cal.degenerate else cal.threshold
                # production always scores lowercased text
                hp_s, hn_s = [score(x, True) for x in hp], [score(x, True) for x in hn]
                print(f"  {name:12} {g:23} threshold {('—' if thr[g] is None else f'{thr[g]:.2f}'):>4}  "
                      f"held-out {_rate(hp_s, thr[g]):4.0%} / {_rate(hn_s, thr[g]):4.0%}"
                      + ("" if thr[g] is not None else "  (falls to lexical)"))
                for kind, xs, ss in (("positive", hp, hp_s), ("negative", hn, hn_s)):
                    for x, s in zip(xs, ss):
                        rows.append({"model": model, "config": name, "gate": g, "kind": kind,
                                     "phrase": x[:120], "score": round(s, 4), "threshold": thr[g]})
                gate_score = score
                thr[g + ":score"] = gate_score  # type: ignore[assignment]

            # Deployed search decision: embedding where calibrated, else the
            # lexical fallback (L0 today, L3 with the candidate setup).
            lex = lexical_choice[("L0", False) if not candidate else ("L3", True)]

            def search_fires(text: str) -> bool:
                for g in SEARCH:
                    if thr[g] is not None:
                        if thr[g + ":score"](text, True) >= thr[g]:  # type: ignore[operator]
                            return True
                    elif _lexical("L0" if not candidate else "L3", text, _templates(g, candidate)) >= lex["thr"][g]:
                        return True
                return False

            tpr = sum(search_fires(x) for x in ho_search_pos) / len(ho_search_pos)
            fpr = sum(search_fires(x) for x in ho_search_neg) / len(ho_search_neg)
            line = f"  {name:12} search decision (deployed): held-out {tpr:4.0%} caught / {fpr:4.0%} false (J {tpr - fpr:+.2f})"
            if real:
                tp = sum(search_fires(r["text"]) and r["search"] for r in real)
                fp = sum(search_fires(r["text"]) and not r["search"] for r in real)
                line += f";  real {tp}/{sum(r['search'] for r in real)} caught, {fp} false"
                if not candidate and "episodic_relevance:score" in thr and thr["episodic_relevance"] is not None:
                    ep = sum(thr["episodic_relevance:score"](r["text"], True) >= thr["episodic_relevance"]  # type: ignore[operator]
                             for r in real)
                    line += f";  memory check fired on {ep}"
            print(line)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-phrase held-out rows: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
