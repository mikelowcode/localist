"""
Diagnostic: score_research_intent_templates.py
================================================
Establishes a real threshold for the proposed `research_intent` semantic
template group (see "Research loop — implementation sketch") before it is
added to planner.py. Mirrors score_lookup_request_templates.py's method:
embed a candidate template set and a curated set of test utterances using
the live EmbeddingEngine, score via max-cosine-per-group (the same
mechanism as Planner._semantic_search_intent), and report true-positive
survival vs. false-positive rate across threshold candidates.

v2 (this revision): swaps in the v2 template set from research_loop_design.md
(every template anchored on an explicit lookup verb — "look up"/"find"/
"search for"/"check"/"track down" — rather than mixing pure cost phrasing
with lookup phrasing, per the v1 diagnostic's template-design finding) and
adds `_RESEARCH_NEGATIVE_FILTER`, a substring pre-filter mirroring
planner.py's `_SEARCH_NEGATIVE_FILTER`, to intercept subjective
price-opinion phrasing (Category E in the test set) before it can
contribute a research_intent score at all. The v1 report
(`research_intent_threshold_assessment_2026-07-16.md`) found no threshold
separates Category E from true positives; this revision tests whether the
combination of new templates + pre-filter resolves it instead.

READ-ONLY: does not modify planner.py or _SEARCH_INTENT_TEMPLATES. The
research_intent templates and negative filter below are local to this
script for evaluation only.

Scope note: research_intent is only meaningful as an *upgrade* of an
already-gated web_search dispatch (see the sketch's _priority3_tool splice
— it only fires when "web_search" is already in tools). So the questions
that matter are:
  1. Do real price/spec-lookup requests score high on research_intent?
  2. Do OTHER web_search-triggering utterances (lookup_request,
     knowledge_request_open, freshness_request, explicit_search_action
     positives that are NOT about pricing/specs) stay low on
     research_intent, so a plain "look up the release date for this"
     doesn't get needlessly upgraded into a 3-iteration search loop?
  3. Do non-search utterances (generic conversation, subjective
     price opinions) stay low, as an extra sanity check?

No files are modified. Run from the repo root or the backend/ directory.

Usage:
    cd backend
    python ../diagnostics/score_research_intent_templates.py
"""

from __future__ import annotations

import sys
import os
import pathlib
from datetime import date

# Ensure backend/ is on the path so planner and embedding_engine can be imported.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from embedding_engine import EmbeddingEngine
from planner import _SEARCH_INTENT_TEMPLATES, _cosine_similarity  # type: ignore[attr-defined]

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Candidate research_intent templates — v2, from research_loop_design.md.
# Evaluated as a local, additional group; _SEARCH_INTENT_TEMPLATES in
# planner.py is read but never written to.
#
# v1 (superseded — see research_intent_threshold_assessment_2026-07-16.md):
#   "find the pricing for this", "what does this cost", "how much does this
#   cost", "look up the pricing plans for this", "find out how much this
#   costs", "what are the pricing tiers for this", "track down the price of
#   this", "find the specs for this product"
# v1 mixed pure cost phrasing ("what does this cost") with lookup phrasing
# ("find the pricing for this"), which collided badly with subjective price
# opinions (Category E). v2 anchors every template on an explicit lookup
# verb.
# ---------------------------------------------------------------------------
_RESEARCH_INTENT_TEMPLATES: tuple[str, ...] = (
    "look up the pricing for this product",
    "find out how much this subscription costs",
    "search for the price of this item",
    "check the current price of this plan",
    "find the pricing tiers for this service",
    "track down how much this product costs",
    "look up the specs and price for this product",
    "find the cost of this plan per month",
)

# Mirrors planner.py's _SEARCH_NEGATIVE_FILTER pattern: substrings that,
# when present, intercept the utterance before it can contribute a
# research_intent score — same "filtered before the gate, not decided by
# the gate" mechanism used there for identity/greeting collisions.
_RESEARCH_NEGATIVE_FILTER: frozenset[str] = frozenset({
    "worth the price",
    "too expensive",
    "lot of money for",
    "can't believe how expensive",
})

_THRESHOLD_CANDIDATES: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

_GROUP_ABBR: dict[str, str] = {
    "explicit_search_action": "ESA",
    "lookup_request": "LR",
    "knowledge_request_open": "KRO",
    "freshness_request": "FR",
    "research_intent": "RI",
}

# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

# Category T — true positives. Real price/spec-lookup requests, phrased
# differently from the templates themselves (paraphrase, not verbatim
# overlap) so the score reflects semantic generalization, not template
# memorization.
_CAT_T: list[str] = [
    "How much does the Tesla Model 3 cost?",
    "What's the price of the new iPhone?",
    "Can you find out how much AWS charges for S3 storage?",
    "What are the pricing tiers for Notion?",
    "Look up the specs on the RTX 4090",
    "Find me the cost of a one-bedroom apartment in Austin",
    "What does ChatGPT Plus cost per month?",
    "Track down pricing for Salesforce Enterprise",
    "Can you check what a Peloton subscription costs?",
    "What's the going rate for a plumber in this area?",
]

# Category L — lookup_request positives (existing templates/incident
# utterances) that are NOT about pricing/specs. These already trigger
# web_search via lookup_request; research_intent must stay low on them or
# every generic lookup gets needlessly upgraded into a multi-iteration loop.
# One deliberate exception is included and flagged: "current stock price"
# is itself a price fact, so a research_intent hit there is plausible, not
# a bug — noted separately in the report rather than folded into the FP count.
_CAT_L: list[str] = [
    "Can you look up the release date for this?",
    "Could you look up what year this happened?",
    "Can you look up information about the latest Apple products?",
    "Can you look up Apple's price hike for the MacBook Neo and iPad?",
    "Can you look up their next-generation in-house Microsoft AI models?",
]
_CAT_L_PRICE_ADJACENT: str = "Could you find out the current stock price for me?"

# Category K — knowledge_request_open positives, non-pricing. Ungated in
# production (never trigger tools_to_call on their own) but worth checking
# research_intent doesn't spuriously fire on them.
_CAT_K: list[str] = [
    "What is this?",
    "Tell me about this company",
    "Explain how blockchain works",
    "What do you know about this?",
]

# Category F — freshness_request positives, non-pricing.
_CAT_F: list[str] = [
    "What's the latest on this?",
    "Is there anything new about this?",
    "What's the current status of this project?",
]

# Category E — pricing-adjacent but subjective/non-lookup. Mentions cost
# vocabulary without asking for a concrete number to be found — should NOT
# fire research_intent (there is nothing to search for and extract).
_CAT_E: list[str] = [
    "Is this too expensive for me?",
    "Do you think this is worth the price?",
    "I can't believe how expensive rent is these days",
    "That seems like a lot of money for what you get",
]

# Category G — generic conversational negatives, no search semantics at all.
_CAT_G: list[str] = [
    "Can you help me with this?",
    "What's up?",
    "How are you doing today?",
    "Thanks, that's helpful",
]


def _trunc(s: str, n: int = 55) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _yn(b: bool) -> str:
    return "**Y**" if b else "n"


def main() -> None:
    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit)…")
    engine = EmbeddingEngine()
    if not engine.available:
        print("ERROR: EmbeddingEngine failed to load. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print("EmbeddingEngine ready.\n")

    # Pre-embed the 4 production groups plus the local research_intent candidate.
    all_groups: dict[str, tuple[str, ...]] = dict(_SEARCH_INTENT_TEMPLATES)
    all_groups["research_intent"] = _RESEARCH_INTENT_TEMPLATES

    print("Pre-embedding templates (4 production groups + research_intent candidate)…")
    template_vecs: dict[str, dict[str, list[float]]] = {}
    for group, phrases in all_groups.items():
        template_vecs[group] = {phrase: engine.embed(phrase) for phrase in phrases}
    print("Done.\n")

    all_items: list[tuple[str, str]] = (
        [("T", u) for u in _CAT_T]
        + [("L", u) for u in _CAT_L]
        + [("L-price-adj", _CAT_L_PRICE_ADJACENT)]
        + [("K", u) for u in _CAT_K]
        + [("F", u) for u in _CAT_F]
        + [("E", u) for u in _CAT_E]
        + [("G", u) for u in _CAT_G]
    )

    print(f"Scoring {len(all_items)} utterances against {len(all_groups)} groups…")
    rows: list[dict] = []
    for cat, utt in all_items:
        print(f"  [{cat}] {utt!r}")
        qv = engine.embed(utt)
        gs = {
            group: max(_cosine_similarity(qv, tv) for tv in tvecs.values())
            for group, tvecs in template_vecs.items()
        }
        filtered = any(neg in utt.lower() for neg in _RESEARCH_NEGATIVE_FILTER)
        rows.append({
            "category": cat, "utterance": utt, "scores": gs,
            "ri": gs["research_intent"], "filtered": filtered,
        })
    print("Done.\n")

    md: list[str] = []
    md += [
        "# `research_intent` Threshold Assessment",
        "",
        f"**Date:** {TODAY}",
        "**Script:** `diagnostics/score_research_intent_templates.py`",
        "**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs",
        "**Status:** READ-ONLY. `planner.py` unmodified (candidate templates are local to this script).",
        "",
        "**Purpose:** Re-test `research_intent` with the v2 template set from",
        "`research_loop_design.md` plus a `_RESEARCH_NEGATIVE_FILTER` pre-filter, after the v1 pass",
        "(`research_intent_threshold_assessment_2026-07-16.md`) found Category E (subjective price",
        "opinion) scored higher than every true positive — a threshold-unfixable collision.",
        "",
        "**Candidate templates (8, v2 — verb-anchored, from research_loop_design.md):**",
        "",
    ]
    for t in _RESEARCH_INTENT_TEMPLATES:
        md.append(f"- `{t}`")
    md += [
        "",
        "**Negative filter (mirrors `_SEARCH_NEGATIVE_FILTER`):**",
        "",
    ]
    for f in sorted(_RESEARCH_NEGATIVE_FILTER):
        md.append(f"- `{f}`")
    md += [
        "",
        "## Category Definitions",
        "",
        "| Cat | Description | N | Expected research_intent behavior |",
        "|-----|-------------|---|-----------------------------------|",
        f"| T | True positives — real price/spec-lookup requests, paraphrased (not verbatim template overlap) | {len(_CAT_T)} | should fire |",
        f"| L | lookup_request positives, non-pricing — already trigger web_search via LR; must NOT be needlessly upgraded to the research loop | {len(_CAT_L)} | should NOT fire |",
        "| L-price-adj | \"current stock price\" — a lookup_request template that IS itself a price fact | 1 | ambiguous by design, reported separately, not counted as FP/TP |",
        f"| K | knowledge_request_open positives, non-pricing | {len(_CAT_K)} | should NOT fire |",
        f"| F | freshness_request positives, non-pricing | {len(_CAT_F)} | should NOT fire |",
        f"| E | pricing-adjacent but subjective/non-lookup (opinion, no concrete fact to extract) | {len(_CAT_E)} | should NOT fire |",
        f"| G | generic conversational, no search semantics | {len(_CAT_G)} | should NOT fire |",
        "",
        "## Full Score Table (all groups)",
        "",
        "`filtered` = matched `_RESEARCH_NEGATIVE_FILTER`; RI score shown for visibility but excluded",
        "from the FP-pool analysis below (see Negative Filter section).",
        "",
        "| Cat | Utterance | ESA | LR | KRO | FR | RI | filtered |",
        "|-----|-----------|----:|----:|----:|----:|----:|:--------:|",
    ]
    for r in rows:
        gs = r["scores"]
        md.append(
            f"| {r['category']} | {_trunc(r['utterance'])} "
            f"| {gs['explicit_search_action']:.4f} "
            f"| {gs['lookup_request']:.4f} "
            f"| {gs['knowledge_request_open']:.4f} "
            f"| {gs['freshness_request']:.4f} "
            f"| **{gs['research_intent']:.4f}** "
            f"| {'Y' if r['filtered'] else ''} |"
        )
    md.append("")

    # ── Threshold trade-off: T survival vs. FP rate across L/K/F/E/G ─────────
    # Filtered utterances (matched _RESEARCH_NEGATIVE_FILTER) are excluded from
    # both pools here, mirroring production: a filter match means
    # _semantic_search_intent never computes a score, so the utterance can
    # never contribute a false positive (or, hypothetically, a true positive)
    # to the gate in the first place.
    fp_pool = [r for r in rows if r["category"] in ("L", "K", "F", "E", "G") and not r["filtered"]]
    t_pool = [r for r in rows if r["category"] == "T" and not r["filtered"]]
    filtered_pool = [r for r in rows if r["filtered"]]
    n_fp_pool = len(fp_pool)

    md += [
        "## Threshold Trade-off",
        "",
        "FP pool = categories L + K + F + E + G, MINUS any utterance intercepted by",
        "`_RESEARCH_NEGATIVE_FILTER` (reported separately below). L-price-adj is excluded from both",
        "pools and reported separately below.",
        "",
        "| Threshold | T survivors (of " + str(len(t_pool)) + ") | FP pool fires (of " + str(n_fp_pool) + ") |",
        "|:---------:|:----------------------:|:--------------------------:|",
    ]
    for th in _THRESHOLD_CANDIDATES:
        t_ok = sum(1 for r in t_pool if r["ri"] >= th)
        fp = sum(1 for r in fp_pool if r["ri"] >= th)
        md.append(f"| {th:.2f} | {t_ok}/{len(t_pool)} | {fp}/{n_fp_pool} |")
    md.append("")

    # Full separation search (fine grid) — does any threshold cleanly split T from FP pool?
    fine_grid = [x / 1000 for x in range(300, 951, 5)]
    full_sep_thresholds = [
        t for t in fine_grid
        if sum(1 for r in t_pool if r["ri"] >= t) == len(t_pool)
        and sum(1 for r in fp_pool if r["ri"] >= t) == 0
    ]
    md += [
        "**Full separation** (fine 0.005 grid, 0.300–0.950): a threshold where all T survive AND",
        "zero FP-pool items fire.",
        "",
    ]
    if full_sep_thresholds:
        md.append(
            f"Full separation achieved for threshold ∈ [{min(full_sep_thresholds):.3f}, "
            f"{max(full_sep_thresholds):.3f}] ({len(full_sep_thresholds)} grid points)."
        )
    else:
        md.append("No threshold in the scanned range achieves full separation.")
    md.append("")

    # ── Category T per-utterance detail (load-bearing minimum) ──────────────
    md += [
        "## Category T — Per-Utterance research_intent Scores",
        "",
        "Lowest T score is the load-bearing constraint: any threshold above it starts losing",
        "true positives.",
        "",
        "| Utterance | RI score |",
        "|-----------|---------:|",
    ]
    for r in sorted(t_pool, key=lambda r: r["ri"]):
        md.append(f"| {_trunc(r['utterance'])} | {r['ri']:.4f} |")
    min_t = min(r["ri"] for r in t_pool) if t_pool else float("nan")
    md += ["", f"**Minimum T score:** {min_t:.4f}", ""]

    # ── FP pool per-utterance detail, highest-scoring first ─────────────────
    md += [
        "## FP Pool — Per-Utterance research_intent Scores (highest first)",
        "",
        "| Cat | Utterance | RI score |",
        "|-----|-----------|---------:|",
    ]
    for r in sorted(fp_pool, key=lambda r: -r["ri"]):
        md.append(f"| {r['category']} | {_trunc(r['utterance'])} | {r['ri']:.4f} |")
    max_fp = max(r["ri"] for r in fp_pool) if fp_pool else float("nan")
    md += ["", f"**Maximum FP-pool score:** {max_fp:.4f}", ""]

    # ── L-price-adjacent, reported separately ────────────────────────────────
    price_adj_row = next(r for r in rows if r["category"] == "L-price-adj")
    md += [
        "## L-price-adjacent (reported separately, not scored as FP or TP)",
        "",
        f"`{_CAT_L_PRICE_ADJACENT}` → RI = {price_adj_row['ri']:.4f}",
        "",
        "This utterance is a verbatim lookup_request template (\"could you find out the current",
        "stock price for me\") that also happens to name a concrete price fact. Whether it *should*",
        "fire research_intent is a product judgment call (does a stock-price lookup benefit from the",
        "search→evaluate→fetch loop, or does the answer already appear in a plain search snippet?),",
        "not something this diagnostic resolves. Reported for visibility only.",
        "",
    ]

    # ── Negative filter — regression reference (mirrors Cat B in the LR script) ──
    md += [
        "## Negative Filter — Regression Reference",
        "",
        "Utterances intercepted by `_RESEARCH_NEGATIVE_FILTER` before they could contribute a",
        "research_intent score to the FP-pool analysis. Scored here for visibility only, same as",
        "Category B in `score_lookup_request_templates.py` — no re-decision is made; this confirms",
        "what score each would have carried had the filter not caught it.",
        "",
    ]
    if filtered_pool:
        md += [
            "| Cat | Utterance | Raw RI score (pre-filter) |",
            "|-----|-----------|---------------------------:|",
        ]
        for r in sorted(filtered_pool, key=lambda r: -r["ri"]):
            md.append(f"| {r['category']} | {_trunc(r['utterance'])} | {r['ri']:.4f} |")
        md.append("")
        all_cat_e_filtered = all(
            r["filtered"] for r in rows if r["category"] == "E"
        )
        md.append(
            f"{len(filtered_pool)}/{len(_CAT_E)} Category E utterances matched the filter"
            + (
                " — the entire category is intercepted pre-gate in production, so none of it "
                "reaches the FP-pool analysis below."
                if all_cat_e_filtered and len(filtered_pool) == len(_CAT_E)
                else ". Some Category E utterances were NOT caught by the filter and remain in the "
                "FP-pool analysis below — the filter alone does not fully resolve the v1 collision."
            )
        )
    else:
        md.append("*(no utterances matched the filter)*")
    md.append("")

    if min_t > max_fp:
        margin_note = (
            f"**Clean separation exists**: minimum T score ({min_t:.4f}) exceeds maximum FP-pool "
            f"score ({max_fp:.4f}). Any threshold in ({max_fp:.4f}, {min_t:.4f}] achieves "
            f"{len(t_pool)}/{len(t_pool)} T survival and 0/{n_fp_pool} FP-pool false positives."
        )
    else:
        margin_note = (
            f"**No clean separation**: minimum T score ({min_t:.4f}) is below the maximum FP-pool "
            f"score ({max_fp:.4f}). No single threshold achieves both {len(t_pool)}/{len(t_pool)} T "
            f"survival and 0/{n_fp_pool} FP-pool false positives — see the trade-off table above for "
            f"the actual cost at each candidate value."
        )

    # ── Post-filter separation: is the remaining FP pool (L/K/F/G, since E is
    # intercepted by the filter) separable from T on its own? Generalized —
    # does not assume Category E remains in fp_pool, and identifies whichever
    # category is the top remaining offender rather than hardcoding one.
    remaining_cats = sorted({r["category"] for r in fp_pool})
    top_offender = max(fp_pool, key=lambda r: r["ri"]) if fp_pool else None

    md += [
        "## Post-Filter Separation — Remaining Categories",
        "",
        f"With the negative filter applied, the FP pool consists of: {', '.join(remaining_cats) or '(none)'}.",
        "",
    ]
    if top_offender is not None:
        n_t_below_top = sum(1 for r in t_pool if r["ri"] < top_offender["ri"])
        md += [
            (
                f"Top remaining offender: Category {top_offender['category']}, "
                f"\"{top_offender['utterance']}\" → RI = {top_offender['ri']:.4f}, which "
                f"{'exceeds' if top_offender['ri'] >= min_t else 'is below'} the T-minimum "
                f"({min_t:.4f}) and is higher than {n_t_below_top}/{len(t_pool)} Category T scores."
            ),
            "",
        ]
        if min_t > max_fp:
            md.append(
                "**The remaining categories are threshold-separable from T** — see the clean-"
                "separation range in the Summary below."
            )
        else:
            md.append(
                "**The remaining categories are still not threshold-separable from T** on their "
                "own — the negative filter resolved Category E, but at least one L/K/F/G utterance "
                "still collides with the true-positive range at the score level, not just near a "
                "boundary."
            )
    else:
        md.append("FP pool is empty after filtering — nothing left to separate from T.")
    md.append("")

    md += [
        "## Summary",
        "",
        margin_note,
        "",
        "No threshold recommendation is made here — per this repo's established diagnostic",
        "discipline (see `lookup_request_margin_assessment_2026-06-28.md`), the data above states",
        "the cost at each candidate threshold; the choice of `_RESEARCH_INTENT_THRESHOLD` is a",
        "product decision made from this table, not something this script decides.",
        "",
        "Per the implementation sketch, `research_intent` should ship shadow-only",
        "(scored and logged, excluded from `_SEMANTIC_GATE_THRESHOLDS`, threshold read from",
        "`LOCALIST_RESEARCH_INTENT_THRESHOLD` defaulting to `float(\"inf\")`) regardless of which",
        "value is picked from this table, until it has been observed against live traffic — the",
        "same rollout discipline already applied to `LOCALIST_TOOL_FALLBACK_CLASSIFIER`.",
        "",
        "---",
        "",
        "*Generated by `diagnostics/score_research_intent_templates.py` (v2 template set + negative",
        "filter). Compare against `research_intent_threshold_assessment_2026-07-16.md` (v1).*",
    ]

    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"research_intent_threshold_assessment_{TODAY}-v2.md"
    report_path.write_text("\n".join(md) + "\n")

    print("=" * 72)
    print(f"Report written to:\n  {report_path}")
    print("=" * 72)
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
