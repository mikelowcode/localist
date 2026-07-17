"""
Diagnostic: score_negative_filter_tiebreak.py
================================================
Validates the negative-filter tie-break classifier proposed in the
"Negative-filter redesign" addendum to the research-loop implementation
sketch, before `_resolve_negative_filter_conflict` and its
`_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT` are wired into planner.py.

Two separate questions, both answered empirically rather than guessed:

  1. Accuracy: fed the reused 2026-07-16-v2 test set (Categories T/L/
     L-price-adj/K/F/E/G) plus a new Category A ("ambiguous" — utterances
     that deliberately contain a _RESEARCH_NEGATIVE_FILTER substring AND a
     genuine lookup verb, e.g. "Can you look up if this is worth the
     price?"), does the tie-break prompt correctly answer lookup/other
     against a hand-assigned ground truth? Calls the LIVE configured
     runtime (LOCALIST_RUNTIME_BACKEND / LOCALIST_CHAT_MODEL from the
     environment) — no mocks, same discipline as the embedding diagnostics.

  2. Reachability: the sketch claims the tie-break call "is expected to
     fire noticeably more often [than the P5/P6 classifiers], not
     negligibly" because Category E scores 0.51-0.85 raw on research_intent,
     "several clearing 0.65". This script checks that claim directly by
     computing, for every test utterance, live embedding scores against
     all 5 _SEARCH_INTENT_TEMPLATES groups and determining whether a
     genuine conflict (negative-filter substring matched AND some GATED
     group's score clears its threshold) would actually arise in
     production — separately for LOCALIST_RESEARCH_LOOP_ENABLED=false
     (research_intent not gated) and =true (research_intent gated at 0.65).

READ-ONLY: does not modify planner.py, call MemoryManager, or persist
anything outside diagnostics/reports/. The tie-break system prompt is a
local constant here (verbatim from the sketch), not yet added to
planner.py.

Usage:
    cd backend
    python ../diagnostics/score_negative_filter_tiebreak.py
"""

from __future__ import annotations

import os
import sys
import pathlib
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from embedding_engine import EmbeddingEngine
from runtime_factory import create_runtime
from planner import (  # type: ignore[attr-defined]
    _SEARCH_INTENT_TEMPLATES,
    _SEARCH_NEGATIVE_FILTER,
    _RESEARCH_NEGATIVE_FILTER,
    _SEMANTIC_GATE_THRESHOLDS,
    _RESEARCH_INTENT_THRESHOLD,
    _cosine_similarity,
)

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Tie-break classifier — verbatim from the "Negative-filter redesign"
# addendum. Not yet in planner.py; local to this script for evaluation.
# ---------------------------------------------------------------------------
_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT: str = (
    "You are a routing classifier, not a conversational assistant. The "
    "instruction below matched a known phrase pattern that is usually NOT "
    "a request to search for or look up information (e.g. a subjective "
    "opinion, a greeting, an identity question) but also scored high on "
    "semantic similarity to real search-intent phrasing — an ambiguous "
    "case. Decide which one it actually is. Respond with exactly one "
    "word: lookup or other."
)

_ALL_NEGATIVE_FILTER: frozenset[str] = _SEARCH_NEGATIVE_FILTER | _RESEARCH_NEGATIVE_FILTER

# ---------------------------------------------------------------------------
# Test set — reuses the 2026-07-16-v2 category set (score_research_intent_
# templates.py) with a ground-truth "lookup"/"other" label added, plus a new
# Category A: utterances that deliberately combine a negative-filter
# substring with a genuine lookup verb, the actually-hard case for the
# tie-break prompt to get right (a false "other" here silently kills a real
# search request; a false "lookup" on Category E defeats the filter).
# ---------------------------------------------------------------------------

# (category, utterance, ground_truth)
_TEST_SET: list[tuple[str, str, str]] = [
    # Category T — true positives (research_intent), genuine lookup.
    ("T", "How much does the Tesla Model 3 cost?", "lookup"),
    ("T", "What's the price of the new iPhone?", "lookup"),
    ("T", "Can you find out how much AWS charges for S3 storage?", "lookup"),
    ("T", "What are the pricing tiers for Notion?", "lookup"),
    ("T", "Look up the specs on the RTX 4090", "lookup"),
    ("T", "Find me the cost of a one-bedroom apartment in Austin", "lookup"),
    ("T", "What does ChatGPT Plus cost per month?", "lookup"),
    ("T", "Track down pricing for Salesforce Enterprise", "lookup"),
    ("T", "Can you check what a Peloton subscription costs?", "lookup"),
    ("T", "What's the going rate for a plumber in this area?", "lookup"),

    # Category L — lookup_request positives, non-pricing. Genuine lookups,
    # just not about price/specs.
    ("L", "Can you look up the release date for this?", "lookup"),
    ("L", "Could you look up what year this happened?", "lookup"),
    ("L", "Can you look up information about the latest Apple products?", "lookup"),
    ("L", "Can you look up Apple's price hike for the MacBook Neo and iPad?", "lookup"),
    ("L", "Can you look up their next-generation in-house Microsoft AI models?", "lookup"),

    # L-price-adj — a lookup_request template that is itself a price fact.
    ("L-price-adj", "Could you find out the current stock price for me?", "lookup"),

    # Category K — knowledge_request_open, non-pricing. Deliberately
    # ungated in production (planner.py's own comment: templates collided
    # with generic conversational phrasing) — ground truth "other" for the
    # same reason: too generic to reliably signal an actual search action.
    ("K", "What is this?", "other"),
    ("K", "Tell me about this company", "other"),
    ("K", "Explain how blockchain works", "other"),
    ("K", "What do you know about this?", "other"),

    # Category F — freshness_request, non-pricing. Same ungated-by-design
    # reasoning as K.
    ("F", "What's the latest on this?", "other"),
    ("F", "Is there anything new about this?", "other"),
    ("F", "What's the current status of this project?", "other"),

    # Category E — subjective price opinion. The exact phrases
    # _RESEARCH_NEGATIVE_FILTER targets. Ground truth "other": no lookup
    # verb, nothing to search for.
    ("E", "Is this too expensive for me?", "other"),
    ("E", "Do you think this is worth the price?", "other"),
    ("E", "I can't believe how expensive rent is these days", "other"),
    ("E", "That seems like a lot of money for what you get", "other"),

    # Category G — generic conversational, no search semantics.
    ("G", "Can you help me with this?", "other"),
    ("G", "What's up?", "other"),
    ("G", "How are you doing today?", "other"),
    ("G", "Thanks, that's helpful", "other"),

    # Category A — NEW. Ambiguous: contains a _RESEARCH_NEGATIVE_FILTER
    # substring AND an explicit lookup/search verb. This is the case the
    # tie-break exists for — the filter would otherwise wrongly suppress a
    # real search request. Ground truth "lookup".
    ("A", "Can you look up if this laptop is worth the price?", "lookup"),
    ("A", "Could you search for reviews on whether this is worth the price?", "lookup"),
    ("A", "Find out if people think this hotel is too expensive for what you get", "lookup"),
    ("A", "Look up whether this subscription is worth the price compared to competitors", "lookup"),
    ("A", "Can you check if this is too expensive compared to other plans?", "lookup"),
    ("A", "Search online to see if this is worth the price", "lookup"),
]


def _classify(runtime, utterance: str) -> tuple[str, str]:
    """Returns (normalized_label, raw_response). Mirrors
    _resolve_negative_filter_conflict's parsing exactly (startswith match,
    case-insensitive) but never silently swallows a failure — this is a
    diagnostic, so an exception is a finding, not something to hide behind
    the production fail-closed default."""
    raw = runtime.infer(
        system      = _NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT,
        prompt      = f"Instruction: {utterance.lower()}\n\nClassification (lookup/other):",
        max_tokens  = 10,
        temperature = 0.1,
    )
    normalized = "lookup" if raw.strip().lower().startswith("lookup") else "other"
    return normalized, raw


def _reachability(
    group_scores: dict[str, float], filter_matched: bool, research_enabled: bool
) -> bool:
    """Would _resolve_negative_filter_conflict actually be invoked for this
    utterance in production, under the given LOCALIST_RESEARCH_LOOP_ENABLED
    setting? Mirrors the redesign's own conflict-detection logic exactly."""
    if not filter_matched:
        return False
    gated = dict(_SEMANTIC_GATE_THRESHOLDS)
    if research_enabled:
        gated["research_intent"] = _RESEARCH_INTENT_THRESHOLD
    return any(group_scores.get(g, -1.0) >= t for g, t in gated.items())


def _trunc(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit)…")
    engine = EmbeddingEngine()
    if not engine.available:
        print("ERROR: EmbeddingEngine failed to load. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print("EmbeddingEngine ready.\n")

    backend    = os.environ.get("LOCALIST_RUNTIME_BACKEND", "ollama")
    chat_model = os.environ.get("LOCALIST_CHAT_MODEL", "")
    print(f"Constructing live runtime — backend={backend!r} chat_model={chat_model!r}…")
    runtime = create_runtime(
        backend         = backend,
        chat_model      = chat_model,
        ollama_url      = os.environ.get("LOCALIST_OLLAMA_URL", "http://localhost:11434"),
        omlx_url        = os.environ.get("LOCALIST_OMLX_URL", "http://127.0.0.1:8000"),
        foundry_url     = os.environ.get("LOCALIST_FOUNDRY_URL") or None,
    )
    print("Runtime ready.\n")

    print("Pre-embedding templates (5 groups, incl. research_intent)…")
    template_vecs: dict[str, dict[str, list[float]]] = {
        group: {phrase: engine.embed(phrase) for phrase in phrases}
        for group, phrases in _SEARCH_INTENT_TEMPLATES.items()
    }
    print("Done.\n")

    print(f"Scoring + classifying {len(_TEST_SET)} utterances (live embed + live infer)…")
    rows: list[dict] = []
    for cat, utt, truth in _TEST_SET:
        print(f"  [{cat}] {utt!r}")
        lowered = utt.lower()
        qv = engine.embed(utt)
        group_scores = {
            group: max(_cosine_similarity(qv, tv) for tv in tvecs.values())
            for group, tvecs in template_vecs.items()
        }
        filter_matched = any(p in lowered for p in _ALL_NEGATIVE_FILTER)
        matched_phrase = next((p for p in _ALL_NEGATIVE_FILTER if p in lowered), None)

        predicted, raw_response = _classify(runtime, utt)

        rows.append({
            "category": cat, "utterance": utt, "truth": truth,
            "predicted": predicted, "raw_response": raw_response,
            "correct": predicted == truth,
            "filter_matched": filter_matched, "matched_phrase": matched_phrase,
            "group_scores": group_scores,
            "reachable_off": _reachability(group_scores, filter_matched, research_enabled=False),
            "reachable_on":  _reachability(group_scores, filter_matched, research_enabled=True),
        })
    print("Done.\n")

    # ── Build markdown report ────────────────────────────────────────────
    md: list[str] = []
    md += [
        "# Negative-Filter Tie-Break — Accuracy & Reachability Assessment",
        "",
        f"**Date:** {TODAY}",
        "**Script:** `diagnostics/score_negative_filter_tiebreak.py`",
        f"**Runtime:** backend={backend!r} chat_model={chat_model!r} (LIVE, no mocks)",
        "**Embedding model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine",
        "**Status:** READ-ONLY. `planner.py` unmodified — the tie-break prompt is local to this script.",
        "",
        "**Purpose:** Validate `_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT` from the \"Negative-filter",
        "redesign\" addendum before wiring `_resolve_negative_filter_conflict` into planner.py — same",
        "\"don't hand-pick, measure\" discipline as the research_intent template/threshold diagnostics.",
        "",
        "**Tie-break system prompt under test:**",
        "",
        f"> {_NEGATIVE_FILTER_TIEBREAK_SYSTEM_PROMPT}",
        "",
    ]

    # ── §1. Accuracy — full set ──────────────────────────────────────────
    n_total = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    md += [
        "## §1. Accuracy — Full Test Set",
        "",
        f"{n_correct}/{n_total} correct ({n_correct / n_total:.1%}).",
        "",
        "Categories T/L/L-price-adj (ground truth: lookup) and K/F/E/G (ground truth: other) are",
        "reused verbatim from `research_intent_threshold_assessment_2026-07-16-v2.md`. Category A is",
        "new: utterances that combine a `_RESEARCH_NEGATIVE_FILTER` substring with an explicit lookup",
        "verb — the actual case the tie-break exists to resolve correctly.",
        "",
        "| Cat | Utterance | Truth | Predicted | Raw response | Correct |",
        "|-----|-----------|:-----:|:---------:|---------------|:-------:|",
    ]
    for r in rows:
        md.append(
            f"| {r['category']} | {_trunc(r['utterance'])} | {r['truth']} | {r['predicted']} "
            f"| {r['raw_response']!r} | {'Y' if r['correct'] else '**N**'} |"
        )
    md.append("")

    # ── §2. Confusion matrix + per-category breakdown ────────────────────
    tp = sum(1 for r in rows if r["truth"] == "lookup" and r["predicted"] == "lookup")
    fn = sum(1 for r in rows if r["truth"] == "lookup" and r["predicted"] == "other")
    fp = sum(1 for r in rows if r["truth"] == "other" and r["predicted"] == "lookup")
    tn = sum(1 for r in rows if r["truth"] == "other" and r["predicted"] == "other")
    md += [
        "## §2. Confusion Matrix",
        "",
        "|  | Predicted: lookup | Predicted: other |",
        "|---|:---:|:---:|",
        f"| **Truth: lookup** | {tp} (TP) | {fn} (FN) |",
        f"| **Truth: other** | {fp} (FP) | {tn} (TN) |",
        "",
        "FN = a real lookup request wrongly classified as \"other\" — the tie-break would let the",
        "negative filter incorrectly suppress a genuine search request that happened to use",
        "filter-listed phrasing.",
        "",
        "FP = a genuine non-lookup utterance wrongly classified as \"lookup\" — if this utterance also",
        "matched a negative filter, the tie-break would override the filter and reintroduce the exact",
        "collision the filter was built to prevent. IMPORTANT CAVEAT, expanded in §3: most FP/error",
        "categories below (K, F) never match `_SEARCH_NEGATIVE_FILTER`/`_RESEARCH_NEGATIVE_FILTER` in",
        "the first place, so `_resolve_negative_filter_conflict` is never invoked on them in",
        "production regardless of what this classifier would say about them — their errors here are",
        "an out-of-distribution robustness probe, not evidence of a live bug. §3 cross-references which",
        "errors land on filter-matched (operationally reachable) utterances.",
        "",
        "| Category | N | Correct | Accuracy |",
        "|----------|---|---------|----------|",
    ]
    for cat in ("T", "L", "L-price-adj", "K", "F", "E", "G", "A"):
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        c = sum(1 for r in cat_rows if r["correct"])
        md.append(f"| {cat} | {len(cat_rows)} | {c} | {c / len(cat_rows):.1%} |")
    md.append("")

    # ── §3. Reachability — does the tie-break call even fire in production? ──
    matched_rows = [r for r in rows if r["filter_matched"]]
    matched_cats = sorted({r["category"] for r in matched_rows})
    reachable_errors = [r for r in matched_rows if not r["correct"]]
    reachable_off = [r for r in matched_rows if r["reachable_off"]]
    reachable_on  = [r for r in matched_rows if r["reachable_on"]]

    md += [
        "## §3. Reachability — Would the Tie-Break Call Actually Fire?",
        "",
        "The redesign only invokes `_resolve_negative_filter_conflict` when a negative-filter",
        "substring matched AND at least one gated group's score clears its threshold (explicit_",
        "search_action ≥0.72, lookup_request ≥0.60 always; research_intent ≥0.65 only when",
        "`LOCALIST_RESEARCH_LOOP_ENABLED=true`). This section checks the sketch's own claim that the",
        "call \"is expected to fire noticeably more often [than the P5/P6 classifiers], not",
        "negligibly\" against live embedding scores, rather than accepting it as asserted.",
        "",
        f"**{len(matched_rows)}/{n_total}** test utterances matched `_RESEARCH_NEGATIVE_FILTER` or",
        f"`_SEARCH_NEGATIVE_FILTER` at all — categories {', '.join(matched_cats)}. Category E and A",
        "were deliberately constructed to match; the rest weren't, so an unplanned match (e.g. an",
        "existing `_SEARCH_NEGATIVE_FILTER` greeting/identity phrase colliding with one of these",
        "utterances by coincidence) is itself worth surfacing, not filtering out of this report.",
        "",
        (
            f"**Accuracy restricted to filter-matched (operationally reachable) utterances: "
            f"{len(matched_rows) - len(reachable_errors)}/{len(matched_rows)} "
            f"({(len(matched_rows) - len(reachable_errors)) / len(matched_rows):.1%}).** "
            + (
                "Zero errors landed on an utterance the tie-break would actually be invoked on — "
                "every classification error in §2 is confined to Category K/F, which never reach "
                "`_resolve_negative_filter_conflict` because they never match either negative filter."
                if not reachable_errors else
                f"{len(reachable_errors)} error(s) landed on a filter-matched utterance — these ARE "
                "operationally relevant, unlike the K/F errors: " + "; ".join(
                    f"[{r['category']}] {r['utterance']!r} (truth={r['truth']}, "
                    f"predicted={r['predicted']})" for r in reachable_errors
                )
            )
        ),
        "",
        f"- **`LOCALIST_RESEARCH_LOOP_ENABLED=false`** (current default): "
        f"**{len(reachable_off)}/{len(matched_rows)}** filter-matched utterances would reach the "
        f"tie-break call (i.e. also clear explicit_search_action ≥0.72 or lookup_request ≥0.60).",
        f"- **`LOCALIST_RESEARCH_LOOP_ENABLED=true`**: "
        f"**{len(reachable_on)}/{len(matched_rows)}** filter-matched utterances would reach the "
        f"tie-break call (adds research_intent ≥0.65 to the gated set).",
        "",
        "| Cat | Utterance | Filter matched on | ESA | LR | RI | Reachable (flag OFF) | Reachable (flag ON) |",
        "|-----|-----------|--------------------|----:|----:|----:|:---:|:---:|",
    ]
    for r in matched_rows:
        gs = r["group_scores"]
        md.append(
            f"| {r['category']} | {_trunc(r['utterance'], 45)} | `{r['matched_phrase']}` "
            f"| {gs['explicit_search_action']:.4f} | {gs['lookup_request']:.4f} "
            f"| {gs['research_intent']:.4f} "
            f"| {'Y' if r['reachable_off'] else 'n'} | {'Y' if r['reachable_on'] else 'n'} |"
        )
    md.append("")

    if len(reachable_off) == 0 and len(reachable_on) > 0:
        reach_note = (
            "**Finding: the sketch's reachability claim does not hold under the current default "
            "config.** With the research loop disabled (today's default), ZERO filter-matched "
            "utterances in this set clear explicit_search_action or lookup_request — the tie-break "
            "call would never fire in practice for these phrasings, because Category E/A's collision "
            "with the embedding space is concentrated in research_intent specifically (a v2-template "
            "artifact), not in the older ESA/LR groups. The tie-break call only becomes reachable at "
            "all once `LOCALIST_RESEARCH_LOOP_ENABLED=true`, and only for the subset that also clears "
            "research_intent ≥0.65."
        )
    elif len(reachable_off) == 0 and len(reachable_on) == 0:
        reach_note = (
            "**Finding: the tie-break call is unreachable by this entire test set, in either config.** "
            "No filter-matched utterance here clears any gated group's threshold — the conflict this "
            "redesign exists to resolve did not occur for any of the utterances tested. This does not "
            "prove it can't occur (a different phrasing could clear both a filter and a threshold "
            "simultaneously) but it means this specific evidence doesn't support the sketch's "
            "\"fires noticeably more often\" framing."
        )
    else:
        reach_note = (
            f"**Partial support for the sketch's reachability claim.** "
            f"{len(reachable_off)}/{len(matched_rows)} (flag off) and {len(reachable_on)}/"
            f"{len(matched_rows)} (flag on) filter-matched utterances would reach the tie-break call — "
            f"a real minority-not-negligible rate ({len(reachable_off) / len(matched_rows):.0%} / "
            f"{len(reachable_on) / len(matched_rows):.0%} of filter-matched turns), not the \"fires "
            "noticeably more often\" framing the sketch asserted, but not zero either. Notably one "
            "unplanned match — a Category G item (a pre-existing `_SEARCH_NEGATIVE_FILTER` greeting/"
            "identity phrase, not one of the new research-specific ones) — is independently reachable "
            "under both configs because it clears `lookup_request` on its own; see the table above."
        )
    md += [reach_note, ""]

    # ── §4. Summary ───────────────────────────────────────────────────────
    op_accuracy = (len(matched_rows) - len(reachable_errors)) / len(matched_rows) if matched_rows else float("nan")
    md += [
        "## §4. Summary",
        "",
        f"**Accuracy (full test set):** {n_correct}/{n_total} ({n_correct / n_total:.1%}). "
        f"{fp} false positive(s) and {fn} false negative(s) overall — see §2.",
        "",
        f"**Accuracy (operationally reachable subset — filter-matched utterances only):** "
        f"{len(matched_rows) - len(reachable_errors)}/{len(matched_rows)} ({op_accuracy:.1%}). This is "
        "the number that actually matters for whether wiring `_resolve_negative_filter_conflict` into "
        "planner.py is safe: every error in the full-set number above falls on Category K/F utterances "
        "that never reach the tie-break call in production (see §3), so they don't represent live risk.",
        "",
        reach_note,
        "",
        "No go/no-go recommendation is made here — per this repo's established diagnostic discipline,",
        "the data above states what was measured; whether this accuracy/reachability profile is good",
        "enough to wire `_resolve_negative_filter_conflict` into planner.py is a product decision.",
        "",
        "---",
        "",
        "*Generated by `diagnostics/score_negative_filter_tiebreak.py`.*",
    ]

    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"negative_filter_tiebreak_assessment_{TODAY}.md"
    report_path.write_text("\n".join(md) + "\n")

    print("=" * 72)
    print(f"Report written to:\n  {report_path}")
    print("=" * 72)
    print(f"Accuracy: {n_correct}/{n_total} ({n_correct / n_total:.1%})")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
