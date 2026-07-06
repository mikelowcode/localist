"""
Diagnostic: score_referential_followup_lookup_request.py
==========================================================
Read-only live diagnostic. Makes NO changes to planner.py, the negative
filter, the template groups, or the gate thresholds. Same method as
score_lookup_request_templates.py: constructs a REAL EmbeddingEngine
(mlx-community/embeddinggemma-300m-4bit) and a REAL Planner, then calls
Planner._semantic_search_intent() directly for every test utterance — no
mocking, no synthetic vectors.

Purpose
-------
"What do you make of all that?" was live-confirmed to score 0.6158 on
lookup_request — above the current 0.60 gate. That utterance is a
referential/follow-up ("that" = something already in context), not a
lookup request. This is a new adversarial category, distinct from the
Cat A/B/C/D sets already explored in score_lookup_request_templates.py:
those probed modal-question scaffolding ("can/could/would you + verb"),
identity questions, and greetings. Referential follow-ups are a different
collision shape — vague-pronoun-object questions about prior turns — and
have not been scored before.

This script scores 10 referential/follow-up phrases against all four
_SEARCH_INTENT_TEMPLATES groups, alongside the existing Cat C true-positive
set (the 2026-06-25 confirmed live incident, from
score_lookup_request_templates.py._CAT_C) so the referential category's
scores can be read relative to real lookup requests, not in isolation.
Cat C is reused directly (imported, not re-typed) because it is the only
true-positive set from the original diagnostic passes whose per-utterance
scores are still retained in this repo — the original 18-utterance
Diagnostic 2 baseline's raw scores were not kept (see
LOCALIST-Architecture.md §8.8 OI 3).

This script does NOT propose a fix. It reports scores against the two
gating groups and their current production thresholds
(explicit_search_action 0.72, lookup_request 0.60) and states how many
referential phrases cross either gate. Whether that calls for a fix
(negative-filter addition, template change, threshold change) is left to
Michael, consistent with the posture of every prior diagnostic in this
family.

Usage
-----
    cd <localist repo root>
    python diagnostics/score_referential_followup_lookup_request.py

Exit code is always 0 (observation only, no pass/fail assertion).
"""

from __future__ import annotations

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

# New adversarial category — NOT part of the original 18-utterance baseline.
# "What do you make of all that?" is the live-confirmed trigger (0.6158 on
# lookup_request, above the 0.60 gate). The rest are variations on the same
# collision shape: a short question referring back to unspecified prior
# context ("that", "there") with no actual search/lookup semantics.
_REFERENTIAL_FOLLOWUP: list[str] = [
    "What do you make of all that?",
    "What do you think about that?",
    "Any thoughts on that?",
    "Is that concerning?",
    "What does that mean?",
    "So what's the deal there?",
    "Anything else notable in that?",
    "Can you unpack that a bit?",
    "Does that seem right to you?",
    "What's your take?",
]

_GROUP_ABBR: dict[str, str] = {
    "explicit_search_action": "ESA",
    "lookup_request": "LR",
    "knowledge_request_open": "KRO",
    "freshness_request": "FR",
}


def _trunc(s: str, n: int = 42) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    try:
        from embedding_engine import EmbeddingEngine
    except ImportError as exc:
        print(f"FATAL: could not import embedding_engine — run this from the "
              f"repo root with the project venv active. {exc}", file=sys.stderr)
        return 1

    try:
        from planner import Planner, _SEMANTIC_GATE_THRESHOLDS, _SEARCH_NEGATIVE_FILTER
    except ImportError as exc:
        print(f"FATAL: could not import planner — run this from the repo "
              f"root with the project venv active. {exc}", file=sys.stderr)
        return 1

    try:
        from score_lookup_request_templates import _CAT_C
    except ImportError as exc:
        print(f"FATAL: could not import _CAT_C from "
              f"score_lookup_request_templates.py — {exc}", file=sys.stderr)
        return 1

    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit) …")
    engine = EmbeddingEngine()
    if not engine.available:
        print("FATAL: EmbeddingEngine.available is False — model failed to "
              "load. This script requires the real model; it will not fall "
              "back to a stub.", file=sys.stderr)
        return 1

    print("Constructing Planner with live embed_fn (no MemoryManager — "
          "Priority 4 corpus paths are not exercised by this diagnostic) …")
    planner = Planner(runtime=None, memory_manager=None, embed_fn=engine.embed)

    if not planner._template_embeddings:
        print("FATAL: planner._template_embeddings is empty — template "
              "pre-embedding failed at construction.", file=sys.stderr)
        return 1

    esa_t = _SEMANTIC_GATE_THRESHOLDS["explicit_search_action"]
    lr_t = _SEMANTIC_GATE_THRESHOLDS["lookup_request"]
    print(f"\nGate thresholds in effect: explicit_search_action={esa_t}, "
          f"lookup_request={lr_t}")
    print(f"Negative filter has {len(_SEARCH_NEGATIVE_FILTER)} entries.\n")
    print("=" * 100)

    # (category, utterance) — true_positive rows reuse Cat C verbatim from
    # score_lookup_request_templates.py for direct comparison.
    items: list[tuple[str, str]] = (
        [("referential", u) for u in _REFERENTIAL_FOLLOWUP]
        + [("true_positive", u) for _, u in _CAT_C]
    )

    rows: list[dict] = []

    for category, utterance in items:
        lowered = utterance.lower()

        filter_hit = next(
            (p for p in _SEARCH_NEGATIVE_FILTER if p in lowered), None
        )

        print(f"[{category:>13}] {utterance!r}")
        if filter_hit is not None:
            print(f"              → negative filter intercepted "
                  f"(matched: {filter_hit!r}). No embedding run.")
            rows.append({
                "category": category, "utterance": utterance,
                "filter_hit": filter_hit, "all_scores": None,
                "esa_fires": False, "lr_fires": False,
            })
            print("-" * 100)
            continue

        result = planner._semantic_search_intent(lowered)
        if result is None:
            print("              → _semantic_search_intent returned None "
                  "unexpectedly.", file=sys.stderr)
            rows.append({
                "category": category, "utterance": utterance,
                "filter_hit": None, "all_scores": None,
                "esa_fires": False, "lr_fires": False,
            })
            print("-" * 100)
            continue

        _best_group, _best_score, all_scores = result
        esa = all_scores["explicit_search_action"]
        lr = all_scores["lookup_request"]
        esa_fires = esa >= esa_t
        lr_fires = lr >= lr_t

        gates_fired = []
        if esa_fires:
            gates_fired.append("ESA")
        if lr_fires:
            gates_fired.append("LR")
        gates_str = "+".join(gates_fired) if gates_fired else "none"

        print(f"              → ESA={esa:.4f}  LR={lr:.4f}  "
              f"KRO={all_scores['knowledge_request_open']:.4f}  "
              f"FR={all_scores['freshness_request']:.4f}  gates_fired={gates_str}")

        rows.append({
            "category": category, "utterance": utterance,
            "filter_hit": None, "all_scores": all_scores,
            "esa_fires": esa_fires, "lr_fires": lr_fires,
        })
        print("-" * 100)

    # ── Full score table ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"FULL SCORE TABLE  (gates: ESA >= {esa_t}, LR >= {lr_t})")
    print("=" * 100)
    header = (
        f"{'Category':<14} {'Phrase':<42} {'ESA':>7} {'LR':>7} "
        f"{'KRO':>7} {'FR':>7}  Gate(s)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["all_scores"] is None:
            print(f"{r['category']:<14} {_trunc(r['utterance']):<42} "
                  f"{'--':>7} {'--':>7} {'--':>7} {'--':>7}  "
                  f"filtered ({r['filter_hit']!r})")
            continue
        gs = r["all_scores"]
        gates_fired = []
        if r["esa_fires"]:
            gates_fired.append("ESA")
        if r["lr_fires"]:
            gates_fired.append("LR")
        gates_str = "+".join(gates_fired) if gates_fired else "none"
        print(
            f"{r['category']:<14} {_trunc(r['utterance']):<42} "
            f"{gs['explicit_search_action']:>7.4f} {gs['lookup_request']:>7.4f} "
            f"{gs['knowledge_request_open']:>7.4f} {gs['freshness_request']:>7.4f}  "
            f"{gates_str}"
        )

    # ── Markdown table (same data, for copy/paste into a report) ──────────
    print("\n" + "=" * 100)
    print("Markdown table")
    print("=" * 100)
    md_lines = [
        "| Category | Phrase | ESA | LR | KRO | FR | Gate(s) fired |",
        "|----------|--------|----:|---:|----:|---:|---------------|",
    ]
    for r in rows:
        if r["all_scores"] is None:
            md_lines.append(
                f"| {r['category']} | {_trunc(r['utterance'], 50)} "
                f"| -- | -- | -- | -- | filtered (`{r['filter_hit']}`) |"
            )
            continue
        gs = r["all_scores"]
        gates_fired = []
        if r["esa_fires"]:
            gates_fired.append("ESA")
        if r["lr_fires"]:
            gates_fired.append("LR")
        gates_str = "+".join(gates_fired) if gates_fired else "none"
        md_lines.append(
            f"| {r['category']} | {_trunc(r['utterance'], 50)} "
            f"| {gs['explicit_search_action']:.4f} | {gs['lookup_request']:.4f} "
            f"| {gs['knowledge_request_open']:.4f} | {gs['freshness_request']:.4f} "
            f"| {gates_str} |"
        )
    for line in md_lines:
        print(line)

    # ── Summary ─────────────────────────────────────────────────────────────
    referential_rows = [r for r in rows if r["category"] == "referential"]
    true_positive_rows = [r for r in rows if r["category"] == "true_positive"]

    referential_crossing = [
        r for r in referential_rows
        if r["all_scores"] is not None and (r["esa_fires"] or r["lr_fires"])
    ]
    tp_crossing = [
        r for r in true_positive_rows
        if r["all_scores"] is not None and (r["esa_fires"] or r["lr_fires"])
    ]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"Referential/follow-up phrases crossing either gate "
        f"(ESA>={esa_t} or LR>={lr_t}): "
        f"{len(referential_crossing)}/{len(referential_rows)}"
    )
    if referential_crossing:
        for r in referential_crossing:
            gs = r["all_scores"]
            gates_fired = []
            if r["esa_fires"]:
                gates_fired.append(f"ESA={gs['explicit_search_action']:.4f}")
            if r["lr_fires"]:
                gates_fired.append(f"LR={gs['lookup_request']:.4f}")
            print(f"  - {r['utterance']!r}  ({', '.join(gates_fired)})")
    print(
        f"\nTrue-positive (Cat C) phrases still crossing either gate "
        f"(sanity check — should be {len(true_positive_rows)}/{len(true_positive_rows)}): "
        f"{len(tp_crossing)}/{len(true_positive_rows)}"
    )

    print("\n" + "=" * 100)
    print("Diagnostic complete. No files modified; no routing/threshold changes made.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
