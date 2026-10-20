"""
Diagnostic: diag_memory_check.py
=================================
Phase 1A of the 2026-10-20 memory-check plan: can the episodic-relevance
check (Planner._priority5_episodic) stop firing on requests that need no
stored memory, without missing the ones that do?

The check is "deterministic keyword list OR embedding gate (or BM25 lexical
fallback)". It no longer decides whether memory is recalled — recall runs on
every turn (04-planner-routing-model.md §4.3a) — but it sets fetch_episodic,
which drives the chat's "episodic" provenance chip and makes the turn P5
instead of P6.

Candidates (decision = keyword part OR gate part, as production):
  K       current _EPISODIC_KEYWORDS (substring match), no gate
  K+E     production today: K + current templates at the model's current
          threshold (hand-validated, else calibrated v2: lowercased battery)
  K'      tightened keyword list (K_TIGHT below), word-boundary match
  K'+T    K' + revised templates (T_REVISED), cosine, calibrated per model
  K'+C    K' + keyword coverage on T_REVISED (model-independent): at least
          two content words matched, first-person words kept, length scaled

Battery: EP_POSITIVES + EP_POSITIVES_LIVE + EP_POSITIVES_V2 against
EP_NEGATIVES + EP_NEGATIVES_V2 + EP_HARD_NEGATIVES_V3 (all
threshold_calibration_fixtures), split by heldout_split(); thresholds are
chosen (Youden's J) on the calibration part only.

Real requests (--real-negatives, --real-positives): JSON lists of {"text"}.
The negatives are the user's typed requests (none needed memory); the
positives are questions about the user's own stored facts. Personal data:
keep the files outside the repo; only aggregates are printed.

READ-ONLY. Changes nothing in the codebase or any database. Calls the live
Ollama embedding endpoint through the real runtime client. Not part of the
pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_memory_check.py \
        [--real-negatives /path/a.json] [--real-positives /path/b.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from localist import bm25
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import _EPISODIC_RELEVANCE_TEMPLATES, _VALIDATED_MODEL_THRESHOLDS
from localist.threshold_calibration import (
    _RI_EP_GRID,
    _build_gate_phrase_pools,
    _cosine_similarity,
    _youden_calibrate,
)
from localist.threshold_calibration_fixtures import (
    EP_HARD_NEGATIVES_V3,
    EP_NEGATIVES,
    EP_NEGATIVES_V2,
    EP_POSITIVES,
    EP_POSITIVES_LIVE,
    EP_POSITIVES_V2,
    heldout_split,
)

MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest", "embeddinggemma-2:latest"]

# Copy of Planner._priority5_episodic's local _EPISODIC_KEYWORDS (2026-10-20).
K_CURRENT = frozenset({
    "preference", "preferences", "remember", "remembered",
    "you know about me", "what do you know",
    "decision", "decisions", "decided",
    "correction", "corrections", "wrong",
    "workflow", "workflows",
    "last time", "previously", "before",
    "my project", "my setup", "my environment",
    "my name", "do you remember", "who am i",
    "what do you know about me", "my preference",
    "what did i tell you", "what have i told you",
})

# Written before scoring: phrases that refer to what the user told the
# assistant or to the user's own stored context; no generic single words.
K_TIGHT = (
    "do you remember", "you remember", "what do you know about me", "you know about me",
    "what did i tell you", "what have i told you", "did i tell you", "did i mention",
    "i told you", "as i mentioned", "remind me what", "who am i", "my name",
    "my project", "my setup", "my environment", "my preference", "my preferences",
    "my usual", "last time we", "we discussed", "we decided", "we talked about",
    "we planned", "what we decided",
)
_K_TIGHT_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in K_TIGHT) + r")\b")

T_REVISED = (
    "what did i tell you about this",
    "do you remember what i said about this",
    "remind me what we decided about this",
    "what do you know about me",
    "based on what you know about me, help with this",
    "catch me up on what we discussed before",
    "help me prepare for my upcoming event",
    "what is on my schedule this week",
    "what are my preferences for this",
    "what was my plan for this",
)

# Coverage stopwords: as planner._KEYWORD_STOPWORDS, but first-person and
# second-person words are kept — they carry the memory signal here.
_COVER_STOP = frozenset(
    "a an the this that these those it is are was be to of for on in at about and or "
    "can could would will please do does what how with".split()
)


def _k_current(text: str) -> bool:
    return any(k in text for k in K_CURRENT)


def _k_tight(text: str) -> bool:
    return _K_TIGHT_RE.search(text) is not None


def _coverage(text: str) -> float:
    tokens = bm25.tokenize(text)
    present = set(tokens)
    best = 0.0
    for t in T_REVISED:
        words = {w for w in bm25.tokenize(t) if w not in _COVER_STOP}
        matched = len(words & present)
        if words and matched >= min(2, len(words)):
            best = max(best, matched / len(words))
    return best * min(1.0, 12 / max(1, len(tokens)))


def _youden_any(pos: list[float], neg: list[float]) -> float | None:
    g = _youden_calibrate(pos, neg, grid=None)
    return g.threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-negatives", type=pathlib.Path)
    ap.add_argument("--real-positives", type=pathlib.Path)
    args = ap.parse_args()
    real_neg = [r["text"].lower() for r in json.loads(args.real_negatives.read_text())] if args.real_negatives else []
    real_pos = [r["text"].lower() for r in json.loads(args.real_positives.read_text())] if args.real_positives else []

    pos = [p.lower() for p in EP_POSITIVES + EP_POSITIVES_LIVE + EP_POSITIVES_V2]
    neg = [p.lower() for p in EP_NEGATIVES + EP_NEGATIVES_V2 + EP_HARD_NEGATIVES_V3]
    (cal_p, ho_p), (cal_n, ho_n) = heldout_split(pos), heldout_split(neg)
    old_pos, old_neg = (_build_gate_phrase_pools()["episodic_relevance"])

    print(f"battery: {len(pos)} positives / {len(neg)} negatives "
          f"(held-out {len(ho_p)} / {len(ho_n)}); real: {len(real_pos)} memory questions / "
          f"{len(real_neg)} typed requests (none needing memory)\n")

    def report(name: str, decide) -> None:
        tp = sum(map(decide, ho_p)); fp = sum(map(decide, ho_n))
        rtp = sum(map(decide, real_pos)); rfp = sum(map(decide, real_neg))
        print(f"  {name:28} held-out {tp:2}/{len(ho_p)} caught, {fp:2}/{len(ho_n)} false   "
              f"real {rtp:2}/{len(real_pos)} caught, {rfp:2}/{len(real_neg)} false")

    print("model-independent")
    report("K  (current keywords)", _k_current)
    report("K' (tightened keywords)", _k_tight)
    c_thr = _youden_any([_coverage(x) for x in cal_p], [_coverage(x) for x in cal_n])
    report(f"K'+C (coverage >= {c_thr:.2f})" if c_thr is not None else "K'+C (degenerate)",
           lambda x: _k_tight(x) or (c_thr is not None and _coverage(x) >= c_thr))

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

        def gate(templates: tuple[str, ...]):
            tv = [vec(t) for t in templates]
            return lambda text: max(_cosine_similarity(vec(text), t) for t in tv)

        cur, rev = gate(_EPISODIC_RELEVANCE_TEMPLATES), gate(T_REVISED)
        # Today's threshold: hand-validated, else what calibration v2 picks
        # (current battery, lowercased).
        hand = _VALIDATED_MODEL_THRESHOLDS.get(model, {}).get("episodic_relevance")
        today = hand if hand is not None else _youden_calibrate(
            [cur(p.lower()) for p in old_pos], [cur(p.lower()) for p in old_neg], _RI_EP_GRID).threshold
        t_rev = _youden_calibrate([rev(x) for x in cal_p], [rev(x) for x in cal_n], _RI_EP_GRID)
        print(f"\n{model}  (today's threshold {today}{' hand-validated' if hand is not None else ''}; "
              f"revised-template threshold {t_rev.threshold}{' DEGENERATE' if t_rev.degenerate else ''})")
        report("K+E (production today)", lambda x: _k_current(x) or (today is not None and cur(x) >= today))
        report("E alone (today's gate)", lambda x: today is not None and cur(x) >= today)
        report("K'+T (revised, calibrated)",
               lambda x: _k_tight(x) or (t_rev.threshold is not None and rev(x) >= t_rev.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
