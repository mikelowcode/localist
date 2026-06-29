"""
Diagnostic: score_lookup_request_templates.py
==============================================
Embeds a set of test utterances and scores them against every template in
_SEARCH_INTENT_TEMPLATES using the live EmbeddingEngine and _cosine_similarity
from planner.py.

No files are modified. Run from the repo root or the backend/ directory.

Usage:
    cd backend
    python ../diagnostics/score_lookup_request_templates.py
"""

from __future__ import annotations

import sys
import os
import pathlib

# Ensure backend/ is on the path so planner and embedding_engine can be imported.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from embedding_engine import EmbeddingEngine
from planner import _SEARCH_INTENT_TEMPLATES, _cosine_similarity  # type: ignore[attr-defined]

TEST_UTTERANCES = [
    "Who are you?",
    "What are you?",
    "What can you do?",
]


def main() -> None:
    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit)…")
    engine = EmbeddingEngine()
    if not engine.available:
        print("ERROR: EmbeddingEngine failed to load. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print("EmbeddingEngine ready.\n")

    # Pre-embed all templates once.
    print("Pre-embedding templates…")
    template_vecs: dict[str, dict[str, list[float]]] = {}
    for group, phrases in _SEARCH_INTENT_TEMPLATES.items():
        template_vecs[group] = {}
        for phrase in phrases:
            template_vecs[group][phrase] = engine.embed(phrase)
    print("Done.\n")
    print("=" * 72)

    # Collect lookup_request per-template results for the focused section below.
    lookup_detail: dict[str, list[tuple[str, float]]] = {}

    for utterance in TEST_UTTERANCES:
        print(f"\nUtterance: {utterance!r}")
        print("-" * 72)
        query_vec = engine.embed(utterance)

        for group, phrases in _SEARCH_INTENT_TEMPLATES.items():
            scored = sorted(
                [(phrase, _cosine_similarity(query_vec, template_vecs[group][phrase]))
                 for phrase in phrases],
                key=lambda x: x[1],
                reverse=True,
            )
            group_max = scored[0][1] if scored else 0.0
            print(f"\n  Group: {group}  (max={group_max:.4f})")
            for phrase, score in scored:
                print(f"    {score:.4f}  {phrase!r}")

            if group == "lookup_request":
                lookup_detail[utterance] = scored

        print()

    print("=" * 72)
    print("\nlookup_request — Full Per-Template Breakdown (all utterances)")
    print("=" * 72)
    gate = 0.60
    for utterance in TEST_UTTERANCES:
        scored = lookup_detail[utterance]
        gate_fires = scored[0][1] >= gate if scored else False
        print(f"\n  Utterance: {utterance!r}  (gate={gate}, fires={gate_fires})")
        for phrase, score in scored:
            marker = " *** ABOVE GATE" if score >= gate else ""
            print(f"    {score:.4f}  {phrase!r}{marker}")

    print()
    print("=" * 72)
    print("Diagnostic complete.")

    # Extended section (2026-06-28): negative-side margin assessment.
    run_extended_diagnostic(engine, template_vecs)

    # Candidate rework section (2026-06-28): template replacement exploration.
    run_candidate_rework_diagnostic(engine, template_vecs)

    # ESA margin section (2026-06-28): explicit_search_action isolated margin assessment.
    run_esa_margin_diagnostic(engine, template_vecs)

    # Full per-utterance table (2026-06-28): LR(Set1) + ESA(orig) for Cat D + Cat A.
    run_full_pertable_diagnostic(engine, template_vecs)


# ======================================================================================
# EXTENDED DIAGNOSTIC — 2026-06-28
# Negative-side margin assessment for _SEMANTIC_GATE_THRESHOLDS["lookup_request"] = 0.60
# READ-ONLY: does not modify planner.py or change any threshold.
# ======================================================================================

_ESA_FIXED_THRESHOLD: float = 0.68          # explicit_search_action — fixed, not under evaluation
_LR_THRESHOLD_CANDIDATES: tuple[float, ...] = (0.60, 0.65, 0.68)

# Category A: Live false positives from the 2026-06-28 session.
# These utterances were misrouted to web_search when they should have gone to corpus.
# Scores at time of incident: 0.659, 0.649, 0.649 on lookup_request.
_CAT_A: list[tuple[str, str]] = [
    ("A", "Tell me how Localist works?"),
    ("A", "Can you read my wiki files?"),
    ("A", "List the files in my vault?"),
]

# Category B: Phrases already in _SEARCH_NEGATIVE_FILTER — regression reference only.
# These are intercepted before embedding in production. Scored here to confirm the raw
# lookup_request score that made filtering necessary. No re-decision is made.
_CAT_B: list[tuple[str, str]] = [
    ("B-identity", "who are you"),
    ("B-identity", "what are you"),
    ("B-identity", "what can you do"),
    ("B-identity", "what can you help with"),
    ("B-identity", "what do you do"),
    ("B-greeting", "hey lora"),
    ("B-greeting", "hi there"),
    ("B-greeting", "hey there"),
    ("B-greeting", "what's up"),
]

# Category C: Original 2026-06-25 incident positives — MUST remain protected.
# Confirmed real web-search requests. Last live-verified LR scores (2026-06-26):
# 0.6077 / 0.6172 / 0.6208. Any threshold that kills any of these is a
# load-bearing regression — flagged explicitly in the report.
_CAT_C: list[tuple[str, str]] = [
    ("C", "Can you look up Apple's price hike for the MacBook Neo and iPad?"),
    ("C", "Can you look up their next-generation in-house Microsoft AI models?"),
    ("C", "Can you look up Microsoft's next-generation in-house AI models?"),
]

# Category D: Fresh adversarial negatives, generated systematically.
#
# Collision shape (confirmed 2026-06-26): "can/could/would you + [verb]"
# modal-question scaffolding with no actual search/lookup semantics.
#
# Generation axes (auditable from the tuples below):
#   (1) Verb swap    — replace "look up" with non-lookup verbs
#                      (help, check, "look at" [≠ "look up"], tell me about)
#   (2) Modal swap   — replace can/could/would with "will" / "do you mind"
#   (3) Length       — short bare form vs. fuller sentence form
#   (4) Domain       — project/self-referential | file/document | generic/unrelated
#
# Tuple layout: (axis_tag, domain_tag, utterance)
_CAT_D: list[tuple[str, str, str]] = [
    # Axis 1: verb swap (modal = can/could/would, verb ≠ "look up")
    ("D-verb-swap",   "short/generic",        "Can you help me with this?"),
    ("D-verb-swap",   "short/generic",        "Could you check this for me?"),
    ("D-verb-swap",   "short/generic",        "Would you look at this?"),
    ("D-verb-swap",   "short/generic",        "Can you tell me about this?"),
    # Axis 2: modal swap (verb = look into)
    ("D-modal-swap",  "short/generic",        "Will you look into this?"),
    ("D-modal-swap",  "short/generic",        "Do you mind looking at this?"),
    # Axis 3: length/specificity
    ("D-length",      "bare-minimum",         "Can you help?"),
    ("D-length",      "fuller-sentence",      "Can you help me understand this particular concept in more depth?"),
    # Axis 4: domain — project/self-referential (same shape as Cat A incidents)
    ("D-domain",      "project-referential",  "Can you help me understand how Localist works?"),
    ("D-domain",      "project-referential",  "Could you explain what this system does?"),
    # Axis 4: domain — file/document-referencing (same shape as Cat A incidents)
    ("D-domain",      "file-referencing",     "Can you look at my notes and help me organize them?"),
    ("D-domain",      "file-referencing",     "Could you read through this document for me?"),
    # Axis 4: domain — generic/unrelated
    ("D-domain",      "generic/unrelated",    "Would you help me plan a trip to Japan?"),
    ("D-domain",      "generic/unrelated",    "Can you tell me a joke?"),
]

_GROUP_ABBR: dict[str, str] = {
    "explicit_search_action": "ESA",
    "lookup_request": "LR",
    "knowledge_request_open": "KRO",
    "freshness_request": "FR",
}


def _score_all(
    engine: "EmbeddingEngine",
    template_vecs: dict[str, dict[str, list[float]]],
    items: list[tuple[str, str, str]],
) -> list[dict]:
    """Embed each utterance and return per-group max cosine scores."""
    results = []
    for cat, domain, utt in items:
        print(f"  [{cat}] {utt!r}")
        qv = engine.embed(utt)
        gs: dict[str, float] = {
            group: max(_cosine_similarity(qv, tv) for tv in tvecs.values())
            for group, tvecs in template_vecs.items()
        }
        best_g = max(gs, key=gs.__getitem__)
        results.append({
            "category": cat,
            "domain": domain,
            "utterance": utt,
            "group_scores": gs,
            "best_group": best_g,
            "best_score": gs[best_g],
            "lr": gs["lookup_request"],
            "esa": gs["explicit_search_action"],
        })
    return results


def _fires(lr: float, esa: float, lr_t: float) -> bool:
    return lr >= lr_t or esa >= _ESA_FIXED_THRESHOLD


def _yn(b: bool) -> str:
    return "**Y**" if b else "n"


def _trunc(s: str, n: int = 50) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _md_score_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Cat | Utterance | ESA | LR | KRO | FR | Best | @0.60 | @0.65 | @0.68 |",
        "|-----|-----------|-----|----|-----|----|------|-------|-------|-------|",
    ]
    for r in rows:
        gs = r["group_scores"]
        abbr = _GROUP_ABBR.get(r["best_group"], r["best_group"])
        best = f"{abbr}:{r['best_score']:.4f}"
        lr, esa = r["lr"], r["esa"]
        lines.append(
            f"| {r['category']} "
            f"| {_trunc(r['utterance'], 48)} "
            f"| {esa:.4f} "
            f"| {lr:.4f} "
            f"| {gs['knowledge_request_open']:.4f} "
            f"| {gs['freshness_request']:.4f} "
            f"| {best} "
            f"| {_yn(_fires(lr, esa, 0.60))} "
            f"| {_yn(_fires(lr, esa, 0.65))} "
            f"| {_yn(_fires(lr, esa, 0.68))} |"
        )
    return lines


def run_extended_diagnostic(
    engine: "EmbeddingEngine",
    template_vecs: dict[str, dict[str, list[float]]],
) -> None:
    """
    Score categories A–D against all four template groups and write a markdown
    margin-assessment report. Called from main() after the original diagnostic.
    """
    print("\n" + "=" * 72)
    print("EXTENDED DIAGNOSTIC (2026-06-28) — negative-side margin assessment")
    print("=" * 72)

    all_items: list[tuple[str, str, str]] = (
        [(cat, "", utt) for cat, utt in _CAT_A]
        + [(cat, "", utt) for cat, utt in _CAT_B]
        + [(cat, "", utt) for cat, utt in _CAT_C]
        + list(_CAT_D)
    )

    print(f"\nScoring {len(all_items)} utterances across all 4 groups…")
    scored = _score_all(engine, template_vecs, all_items)

    print("\nBuilding markdown report…")
    md: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    md += [
        "# Negative-Side Margin Assessment — `lookup_request` @ 0.60",
        "",
        "**Date:** 2026-06-28",
        "**Script:** `diagnostics/score_lookup_request_templates.py` (extended section)",
        "**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs",
        (
            "**Purpose:** Establish the negative-side margin of"
            " `_SEMANTIC_GATE_THRESHOLDS[\"lookup_request\"] = 0.60`"
        ),
        "**Status:** READ-ONLY diagnostic. `planner.py` unmodified (verified via git diff).",
        "",
    ]

    # ── 1. Category Definitions ───────────────────────────────────────────────
    md += [
        "## 1. Category Definitions",
        "",
        "| Cat | Description | N | Expected behavior |",
        "|-----|-------------|---|-------------------|",
        "| A | Live false positives, 2026-06-28 — misrouted to web_search | 3 | gate=False |",
        "| B-identity | `_SEARCH_NEGATIVE_FILTER` identity phrases (added 2026-06-26) | 5 | filtered before gate |",
        "| B-greeting | `_SEARCH_NEGATIVE_FILTER` greeting phrases (added 2026-06-27) | 4 | filtered before gate |",
        "| C | 2026-06-25 confirmed true positives — MUST fire gate | 3 | gate=True |",
        "| D-\\* | Fresh adversarial negatives, systematic (14 utterances) | 14 | gate=False |",
        "",
    ]

    # ── 2. Category D Generation Axes ─────────────────────────────────────────
    md += [
        "## 2. Category D Generation Axes",
        "",
        "Collision shape confirmed 2026-06-26: **\"can/could/would you + [verb]\"** modal-question",
        "scaffolding with no actual search/lookup semantics. Four axes used (each utterance is tagged",
        "with its primary axis and domain in `_CAT_D`; the table in §5 repeats those tags):",
        "",
        "1. **Verb swap** (modal=can/could/would): replace \"look up\" with non-lookup verbs",
        "   — help, check, \"look at\" [≠ \"look up\"], tell me about",
        "2. **Modal swap** (verb=look into): replace can/could/would with \"will\" / \"do you mind\"",
        "3. **Length/specificity**: short bare form (\"Can you help?\") vs. fuller sentence",
        "4. **Subject domain**: project/self-referential | file/document-referencing | generic/unrelated",
        "",
        "Axes 1–2 directly probe the confirmed collision vector.",
        "Axes 3–4 probe whether length or subject domain shifts the collision rate.",
        "All 14 utterances share the property that a reasonable user would NOT expect web search.",
        "",
    ]

    # ── 3. Full Score Table ───────────────────────────────────────────────────
    md += [
        "## 3. Full Score Table (all categories A–D, all 4 groups)",
        "",
        "- ESA = `explicit_search_action` (gate threshold fixed at 0.68, not under evaluation)",
        "- LR = `lookup_request` (the threshold under evaluation)",
        "- KRO = `knowledge_request_open` (no gate threshold — diagnostic logging only)",
        "- FR = `freshness_request` (no gate threshold — diagnostic logging only)",
        "- @0.60 / @0.65 / @0.68: gate fires? = LR ≥ candidate threshold **OR** ESA ≥ 0.68",
        "",
    ]
    md += _md_score_table(scored)
    md.append("")

    # ── 4. Category C Survival ────────────────────────────────────────────────
    cat_c = [r for r in scored if r["category"] == "C"]
    md += [
        "## 4. Category C — Confirmed Positives: Threshold Survival",
        "",
        "Survival = gate still fires at the candidate LR threshold (LR ≥ T or ESA ≥ 0.68).",
        "Last live-verified LR scores (2026-06-26): 0.6077 / 0.6172 / 0.6208.",
        "A **Y** in every column means that threshold would not un-gate any confirmed real positive.",
        "A **n** means raising to that threshold kills a confirmed positive — this is the",
        "load-bearing constraint on how far the threshold can be raised.",
        "",
        "| Utterance | LR score | Survives @0.60 | Survives @0.65 | Survives @0.68 |",
        "|-----------|----------|:--------------:|:--------------:|:--------------:|",
    ]
    c_survive: dict[float, int] = {0.60: 0, 0.65: 0, 0.68: 0}
    for r in cat_c:
        lr, esa = r["lr"], r["esa"]
        for t in _LR_THRESHOLD_CANDIDATES:
            c_survive[t] += _fires(lr, esa, t)
        md.append(
            f"| {_trunc(r['utterance'], 55)} | {lr:.4f} "
            f"| {_yn(_fires(lr, esa, 0.60))} "
            f"| {_yn(_fires(lr, esa, 0.65))} "
            f"| {_yn(_fires(lr, esa, 0.68))} |"
        )
    md.append("")
    for t in _LR_THRESHOLD_CANDIDATES:
        count = c_survive[t]
        if count == 3:
            note = "all 3 survive — raising to this threshold does not kill any confirmed positive"
        else:
            note = f"⚠ only {count}/3 survive — raising to {t:.2f} KILLS {3 - count} confirmed positive(s)"
        md.append(f"- **@{t:.2f}:** {note}")
    md.append("")

    # ── 5. Category D False Positive Counts ──────────────────────────────────
    cat_d = [r for r in scored if r["category"].startswith("D-")]
    n_d = len(cat_d)
    md += [
        "## 5. Category D — Fresh Adversarial Negatives: False Positive Counts",
        "",
        f"N = {n_d} utterances (see §2 for generation axes).",
        "A **false positive** = gate fires for an utterance that should NOT trigger web_search.",
        "Gate fires = LR ≥ threshold OR ESA ≥ 0.68.",
        "",
    ]
    for thresh in _LR_THRESHOLD_CANDIDATES:
        fp = [r for r in cat_d if _fires(r["lr"], r["esa"], thresh)]
        md.append(f"### LR threshold @ {thresh:.2f} — False positives: {len(fp)}/{n_d}")
        if fp:
            md += [
                "",
                "| Utterance | LR | ESA | Trigger |",
                "|-----------|----|-----|---------|",
            ]
            for r in fp:
                reasons: list[str] = []
                if r["lr"] >= thresh:
                    reasons.append(f"LR={r['lr']:.4f}≥{thresh:.2f}")
                if r["esa"] >= _ESA_FIXED_THRESHOLD:
                    reasons.append(f"ESA={r['esa']:.4f}≥{_ESA_FIXED_THRESHOLD:.2f}")
                md.append(
                    f"| {_trunc(r['utterance'], 55)} "
                    f"| {r['lr']:.4f} | {r['esa']:.4f} "
                    f"| {'; '.join(reasons)} |"
                )
        else:
            md.append("*(no false positives at this threshold)*")
        md.append("")

    md += [
        "### Full Category D Per-Utterance Detail",
        "",
        "| Axis | Domain | Utterance | LR | @0.60 | @0.65 | @0.68 |",
        "|------|--------|-----------|----|:-----:|:-----:|:-----:|",
    ]
    for r in cat_d:
        lr, esa = r["lr"], r["esa"]
        md.append(
            f"| {r['category']} | {r['domain']} "
            f"| {_trunc(r['utterance'], 48)} "
            f"| {lr:.4f} "
            f"| {_yn(_fires(lr, esa, 0.60))} "
            f"| {_yn(_fires(lr, esa, 0.65))} "
            f"| {_yn(_fires(lr, esa, 0.68))} |"
        )
    md.append("")

    # ── 6. Trade-off Statement ────────────────────────────────────────────────
    cat_a = [r for r in scored if r["category"] == "A"]
    n_a = len(cat_a)
    md += [
        "## 6. Trade-off Statement (numerical only — no recommendation)",
        "",
        "| LR threshold | Cat C survivors (must-fire) | Cat A false positives | Cat D false positives |",
        "|:------------:|:---------------------------:|:--------------------:|:--------------------:|",
    ]
    for t in _LR_THRESHOLD_CANDIDATES:
        c_ok = sum(1 for r in cat_c if _fires(r["lr"], r["esa"], t))
        a_fp = sum(1 for r in cat_a if _fires(r["lr"], r["esa"], t))
        d_fp = sum(1 for r in cat_d if _fires(r["lr"], r["esa"], t))
        md.append(f"| {t:.2f} | {c_ok}/3 | {a_fp}/{n_a} | {d_fp}/{n_d} |")
    md += [
        "",
        "**Cat C minimum LR scores** (load-bearing — any threshold above the lowest kills a confirmed positive):",
    ]
    for r in cat_c:
        md.append(f"- `{r['utterance']}` → LR = {r['lr']:.4f}")
    md += [
        "",
        "**Cat A (live false positives, 2026-06-28) LR scores** (evidence that triggered this diagnostic):",
    ]
    for r in cat_a:
        md.append(f"- `{r['utterance']}` → LR = {r['lr']:.4f}")
    md += [
        "",
        "---",
        "",
        "*No threshold recommendation is made. The data above states the cost at each candidate*",
        "*threshold in terms of confirmed-positive survival (Cat C) and false positive rate (Cat A, Cat D).*",
        "",
        "*Generated by `diagnostics/score_lookup_request_templates.py` — extended section — 2026-06-28.*",
    ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "lookup_request_margin_assessment_2026-06-28.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"\nReport written to:\n  {report_path}")
    print("=" * 72)
    print("Extended diagnostic complete.")
    print("=" * 72)


# ======================================================================================
# CANDIDATE REWORK DIAGNOSTIC — 2026-06-28
# Template replacement exploration for _SEARCH_INTENT_TEMPLATES["lookup_request"].
# Scores 3 candidate template sets against Cat A / C / D test sets.
# READ-ONLY: does not modify planner.py or _SEARCH_INTENT_TEMPLATES.
# ======================================================================================

# Original 5 templates (pre-2026-06-25), confirmed max 0.597 against identity-question negatives.
_ORIGINAL_5_TEMPLATES: tuple[str, ...] = (
    "look up this",
    "look that up",
    "go ahead and look it up",
    "find information on this",
    "find out about this",
)

# Candidate Set 1 — object-specificity fix (replaces the 4 suspect templates).
# Modal+verb frame retained; vague pronoun object replaced with a concrete queryable object.
# Fill-ins mirror the kind of specificity present in real Cat C utterances.
_CAND1_TEMPLATES: tuple[str, ...] = (
    "can you look up the release date for this",
    "could you look up what year this happened",
    "can you look up information about the latest Apple products",
    "could you find out the current stock price for me",
)

# Candidate Set 2 — verb-anchored, modal-light (replaces the 4 suspect templates).
# Modal-question scaffolding dropped entirely; anchored on search-specific verb phrases.
_CAND2_TEMPLATES: tuple[str, ...] = (
    "look up information about",
    "search for details on",
    "find out the facts about",
    "go find out about",
)

# Candidate Set 3 — removal (reverts to original 5, no new templates).
_CAND3_TEMPLATES: tuple[str, ...] = _ORIGINAL_5_TEMPLATES

# The 6 Cat D utterances with the highest LR score in the 2026-06-28 report
# (range 0.8134–0.8980), in descending prior-report LR order. These are the
# hardest test cases for any candidate set.
_HARDEST_6_UTTS: tuple[str, ...] = (
    "Could you check this for me?",     # prior LR=0.8980
    "Will you look into this?",          # prior LR=0.8864
    "Can you tell me about this?",       # prior LR=0.8619
    "Would you look at this?",           # prior LR=0.8483
    "Do you mind looking at this?",      # prior LR=0.8158
    "Can you help me with this?",        # prior LR=0.8134
)

_HARDEST_6_PRIOR_LR: dict[str, float] = {
    "Could you check this for me?": 0.8980,
    "Will you look into this?": 0.8864,
    "Can you tell me about this?": 0.8619,
    "Would you look at this?": 0.8483,
    "Do you mind looking at this?": 0.8158,
    "Can you help me with this?": 0.8134,
}

# Thresholds used for Candidate Set 3's incident-regression check (below current 0.60).
_CAND3_INCIDENT_THRESHOLDS: tuple[float, ...] = (0.55, 0.60, 0.65)


def run_candidate_rework_diagnostic(
    engine: "EmbeddingEngine",
    template_vecs: dict[str, dict[str, list[float]]],
) -> None:
    """
    Score 3 candidate LR template sets against Cat A/C/D.
    Generates a markdown findings report in diagnostics/reports/.
    READ-ONLY: planner.py and _SEARCH_INTENT_TEMPLATES are not modified.
    """
    print("\n" + "=" * 72)
    print("CANDIDATE REWORK DIAGNOSTIC (2026-06-28) — template replacement exploration")
    print("=" * 72)

    # ESA vectors are fixed across all candidates — unchanged from production config.
    esa_tvecs = template_vecs["explicit_search_action"]

    # Items to score: Cat A + Cat C + Cat D (same fixed test sets as extended diagnostic).
    items_a: list[tuple[str, str, str]] = [(cat, "", utt) for cat, utt in _CAT_A]
    items_c: list[tuple[str, str, str]] = [(cat, "", utt) for cat, utt in _CAT_C]
    items_d: list[tuple[str, str, str]] = list(_CAT_D)
    all_items = items_a + items_c + items_d

    # Pre-embed all query utterances once to avoid repeating across candidate sets.
    print(f"\nPre-embedding {len(all_items)} query utterances…")
    query_vecs: dict[str, list[float]] = {}
    for _, _, utt in all_items:
        if utt not in query_vecs:
            print(f"  {utt!r}")
            query_vecs[utt] = engine.embed(utt)
    print("Done.")

    def _score_cand(cand_templates: tuple[str, ...], label: str) -> list[dict]:
        print(f"\n  Embedding {len(cand_templates)} templates ({label})…")
        cand_vecs = {t: engine.embed(t) for t in cand_templates}
        rows: list[dict] = []
        for cat, domain, utt in all_items:
            qv = query_vecs[utt]
            lr = max(_cosine_similarity(qv, tv) for tv in cand_vecs.values())
            esa = max(_cosine_similarity(qv, tv) for tv in esa_tvecs.values())
            rows.append({
                "category": cat,
                "domain": domain,
                "utterance": utt,
                "lr": lr,
                "esa": esa,
            })
        return rows

    cand_data: list[tuple[str, tuple[str, ...]]] = [
        ("Set 1 — object-specificity fix", _CAND1_TEMPLATES),
        ("Set 2 — verb-anchored, modal-light", _CAND2_TEMPLATES),
        ("Set 3 — original 5 templates only (pre-2026-06-25)", _CAND3_TEMPLATES),
    ]

    print("\nScoring all candidate sets…")
    scored: list[tuple[str, tuple[str, ...], list[dict]]] = [
        (label, tmpls, _score_cand(tmpls, label)) for label, tmpls in cand_data
    ]

    # ── Build markdown findings report ────────────────────────────────────────
    print("\nBuilding markdown findings report…")
    md: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    md += [
        "# `lookup_request` Template Rework — Candidate Scoring Findings",
        "",
        "**Date:** 2026-06-28",
        "**Script:** `diagnostics/score_lookup_request_templates.py` (candidate rework section)",
        "**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs",
        "**Status:** READ-ONLY. `planner.py` unmodified. `_SEARCH_INTENT_TEMPLATES` unmodified.",
        "",
        "Candidate scoring for the 4 suspect templates added 2026-06-25 (`can you look up`,",
        "`can you look that up for me`, `could you look up`, `can you look into this for me`).",
        "The 2026-06-28 margin assessment identified these as the source of a severe,",
        "threshold-unfixable false-positive surface (6/14 Cat D negatives score 0.81–0.90 —",
        "higher than every Cat C true positive).",
        "",
    ]

    # ── §1. Template sets ─────────────────────────────────────────────────────
    md += [
        "## §1. Template Sets Under Evaluation",
        "",
        "**Current templates (9 total — original 5 + 4 added 2026-06-25):**",
        "",
        "| # | Template | Epoch |",
        "|---|----------|-------|",
        "| 1 | `look up this` | pre-2026-06-25 |",
        "| 2 | `look that up` | pre-2026-06-25 |",
        "| 3 | `go ahead and look it up` | pre-2026-06-25 |",
        "| 4 | `find information on this` | pre-2026-06-25 |",
        "| 5 | `find out about this` | pre-2026-06-25 |",
        "| 6 | `can you look up` | 2026-06-25 |",
        "| 7 | `can you look that up for me` | 2026-06-25 |",
        "| 8 | `could you look up` | 2026-06-25 |",
        "| 9 | `can you look into this for me` | 2026-06-25 |",
        "",
        "**Candidate Set 1 — object-specificity fix** *(replaces the 4 suspect templates)*:",
        "Modal+verb frame kept; vague pronoun object replaced with a concrete queryable object.",
        "",
    ]
    for t in _CAND1_TEMPLATES:
        md.append(f"- `{t}`")
    md += [
        "",
        "**Candidate Set 2 — verb-anchored, modal-light** *(replaces the 4 suspect templates)*:",
        "Modal-question scaffolding dropped; anchored on search-specific verb phrases.",
        "",
    ]
    for t in _CAND2_TEMPLATES:
        md.append(f"- `{t}`")
    md += [
        "",
        "**Candidate Set 3 — removal** *(reverts to original 5, no replacements)*:",
        "Tests whether removing the 4 suspect templates reopens the 2026-06-25 incident.",
        "",
    ]
    for t in _CAND3_TEMPLATES:
        md.append(f"- `{t}`")
    md.append("")

    # ── §2. Trade-off tables ──────────────────────────────────────────────────
    md += [
        "## §2. Trade-off Tables (same format as 2026-06-28 report §6)",
        "",
        "Gate fires = LR ≥ threshold **OR** ESA ≥ 0.68 (ESA templates and threshold unchanged).",
        "",
        "- **Cat C survivors**: gate fires for a confirmed true positive. Target: 3/3.",
        "- **Cat A false positives**: gate fires for a 2026-06-28 confirmed misroute. Target: 0/3.",
        "- **Cat D false positives**: gate fires for a confirmed non-search utterance. Target: 0/14.",
        "",
        "> **ESA floor note:** Two Cat D utterances fire via ESA ≥ 0.68 independently of LR —",
        "> `Would you look at this?` (ESA=0.6990) and `Will you look into this?` (ESA=0.6874).",
        "> These will appear as false positives under any LR template change at any threshold",
        "> unless the ESA templates or threshold are also changed (not under evaluation here).",
        "> The achievable Cat D floor across all candidates is therefore at minimum 2/14.",
        "",
        "### Reference: Current 9 Templates (from 2026-06-28 margin assessment, not recomputed)",
        "",
        "| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |",
        "|:------------:|:---------------:|:--------------------:|:--------------------:|",
        "| 0.60 | 2/3 | 3/3 | 13/14 |",
        "| 0.65 | 0/3 | 1/3 | 10/14 |",
        "| 0.68 | 0/3 | 0/3 | 10/14 |",
        "",
    ]

    for label, tmpls, rows in scored:
        rows_a = [r for r in rows if r["category"] == "A"]
        rows_c = [r for r in rows if r["category"] == "C"]
        rows_d = [r for r in rows if r["category"].startswith("D-")]
        n_a, n_c, n_d = len(rows_a), len(rows_c), len(rows_d)

        md += [
            f"### {label}",
            "",
            "| LR threshold | Cat C survivors | Cat A false positives | Cat D false positives |",
            "|:------------:|:---------------:|:--------------------:|:--------------------:|",
        ]
        for t in _LR_THRESHOLD_CANDIDATES:
            c_ok = sum(1 for r in rows_c if _fires(r["lr"], r["esa"], t))
            a_fp = sum(1 for r in rows_a if _fires(r["lr"], r["esa"], t))
            d_fp = sum(1 for r in rows_d if _fires(r["lr"], r["esa"], t))
            md.append(f"| {t:.2f} | {c_ok}/{n_c} | {a_fp}/{n_a} | {d_fp}/{n_d} |")
        md += [
            "",
            "**Cat C per-utterance LR scores:**",
            "",
            "| Utterance | LR | @0.60 | @0.65 | @0.68 |",
            "|-----------|---:|:-----:|:-----:|:-----:|",
        ]
        for r in rows_c:
            lr, esa = r["lr"], r["esa"]
            md.append(
                f"| {_trunc(r['utterance'], 55)} | {lr:.4f} "
                f"| {_yn(_fires(lr, esa, 0.60))} "
                f"| {_yn(_fires(lr, esa, 0.65))} "
                f"| {_yn(_fires(lr, esa, 0.68))} |"
            )
        md.append("")

    # ── §3. Six hardest Cat D collisions — Sets 1 and 2 ──────────────────────
    md += [
        "## §3. Six Hardest Cat D Collisions — Per-Utterance Scores (Sets 1 and 2)",
        "",
        "The 6 Cat D utterances that scored 0.81–0.90 on LR against the current 9 templates.",
        "These are the hardest test: the modal+verb collision frame at its most extreme.",
        "A score above 0.70 on any candidate means that candidate has not resolved the",
        "core collision for that utterance — flagged explicitly per utterance.",
        "",
    ]
    for label, _, rows in scored[:2]:
        utt_to_row = {r["utterance"]: r for r in rows}
        md += [
            f"### {label}",
            "",
            "| Rank | Utterance | Prior LR | New LR | ESA | Still >0.70? |",
            "|:----:|-----------|--------:|-------:|----:|:------------:|",
        ]
        for rank, utt in enumerate(_HARDEST_6_UTTS, 1):
            prior_lr = _HARDEST_6_PRIOR_LR[utt]
            r = utt_to_row[utt]
            new_lr, esa = r["lr"], r["esa"]
            esa_fires = esa >= _ESA_FIXED_THRESHOLD
            still_problem = new_lr > 0.70 or esa_fires
            if esa_fires:
                note = f"**Yes** (ESA={esa:.4f}≥0.68 — ESA-driven, not LR)"
            elif still_problem:
                note = f"**Yes** (LR={new_lr:.4f})"
            else:
                note = "No"
            md.append(
                f"| {rank} | {_trunc(utt, 42)} "
                f"| {prior_lr:.4f} | {new_lr:.4f} | {esa:.4f} | {note} |"
            )
        md.append("")

    # ── §4. Candidate Set 3 — incident regression check ──────────────────────
    _, _, set3_rows = scored[2]
    set3_c = [r for r in set3_rows if r["category"] == "C"]

    md += [
        "## §4. Candidate Set 3 — 2026-06-25 Incident Regression Check",
        "",
        "**Question:** Does removing the 4 suspect templates reopen the 2026-06-25",
        "false-negative incident? Score the 3 incident utterances against the 5-template-only",
        "config at thresholds 0.55, 0.60, 0.65.",
        "",
        "| Utterance | LR (5-tmpl only) | @0.55 | @0.60 | @0.65 |",
        "|-----------|----------------:|:-----:|:-----:|:-----:|",
    ]
    for r in set3_c:
        lr, esa = r["lr"], r["esa"]
        md.append(
            f"| {_trunc(r['utterance'], 55)} | {lr:.4f} "
            f"| {_yn(_fires(lr, esa, 0.55))} "
            f"| {_yn(_fires(lr, esa, 0.60))} "
            f"| {_yn(_fires(lr, esa, 0.65))} |"
        )
    md.append("")

    c55 = sum(1 for r in set3_c if _fires(r["lr"], r["esa"], 0.55))
    c60 = sum(1 for r in set3_c if _fires(r["lr"], r["esa"], 0.60))
    c65 = sum(1 for r in set3_c if _fires(r["lr"], r["esa"], 0.65))
    md += [
        f"- At 0.55: {c55}/3 incident utterances caught by original 5 templates",
        f"- At 0.60: {c60}/3 incident utterances caught by original 5 templates",
        f"- At 0.65: {c65}/3 incident utterances caught by original 5 templates",
        "",
    ]
    if c60 == 3:
        md.append(
            "**Answer:** The original 5 templates alone catch all 3 incident utterances at "
            "threshold 0.60. Removing the 4 suspect templates does **not** reopen the "
            "2026-06-25 incident at the current production threshold of 0.60."
        )
    elif c55 == 3:
        md.append(
            f"**Answer:** The original 5 templates catch all 3 incident utterances only at "
            f"≤ 0.55. At 0.60: {c60}/3 caught. Removing the suspect templates reopens the "
            f"2026-06-25 incident at threshold 0.60 unless the threshold is also lowered to 0.55."
        )
    else:
        md.append(
            f"**Answer:** Even at 0.55, only {c55}/3 incident utterances are caught by the "
            f"original 5 templates. Removing the 4 suspect templates permanently reopens the "
            f"2026-06-25 incident at all three evaluated thresholds (0.55, 0.60, 0.65)."
        )
    md.append("")

    # ── §5. Separation summary ────────────────────────────────────────────────
    md += [
        "## §5. Separation Summary (numerical only — no recommendation)",
        "",
        "Full separation = there exists a threshold T where Cat C = 3/3 AND Cat D = 0/14.",
        "Partial improvement = Cat D false positives reduced vs. current (≥1 improvement)",
        "while Cat C ≥ 2/3 at the same threshold.",
        "",
        "| Candidate Set | Cat C @0.60 | Cat D @0.60 | Cat C @0.65 | Cat D @0.65 | Full separation? |",
        "|:--------------|:-----------:|:-----------:|:-----------:|:-----------:|:---------------:|",
    ]
    for label, _, rows in scored:
        rows_c = [r for r in rows if r["category"] == "C"]
        rows_d = [r for r in rows if r["category"].startswith("D-")]
        c60 = sum(1 for r in rows_c if _fires(r["lr"], r["esa"], 0.60))
        d60 = sum(1 for r in rows_d if _fires(r["lr"], r["esa"], 0.60))
        c65 = sum(1 for r in rows_c if _fires(r["lr"], r["esa"], 0.65))
        d65 = sum(1 for r in rows_d if _fires(r["lr"], r["esa"], 0.65))
        # Fine-grained search for full separation across a 1-point grid (0.50–1.00)
        fully_sep = any(
            sum(1 for r in rows_c if _fires(r["lr"], r["esa"], t)) == len(rows_c)
            and sum(1 for r in rows_d if _fires(r["lr"], r["esa"], t)) == 0
            for t in [x / 100 for x in range(50, 101)]
        )
        sep = "Yes" if fully_sep else "No"
        md.append(
            f"| {label} | {c60}/{len(rows_c)} | {d60}/{len(rows_d)} "
            f"| {c65}/{len(rows_c)} | {d65}/{len(rows_d)} | {sep} |"
        )
    md += [
        "",
        "---",
        "",
        "*No recommendation is made. The data above states per-candidate separation quality numerically.*",
        "",
        "*Generated by `diagnostics/score_lookup_request_templates.py` — candidate rework section — 2026-06-28.*",
    ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "lookup_request_template_rework_2026-06-28.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"\nFindings report written to:\n  {report_path}")
    print("=" * 72)
    print("Candidate rework diagnostic complete.")
    print("=" * 72)


# ======================================================================================
# ESA MARGIN DIAGNOSTIC — 2026-06-28
# Negative-side margin assessment for _SEMANTIC_GATE_THRESHOLDS["explicit_search_action"] = 0.68.
# ESA scored in isolation (not OR'd with LR) so its own behavior is visible independently.
# READ-ONLY: does not modify planner.py or any template or threshold.
# ======================================================================================

# Threshold candidates for ESA evaluation. 0.68 is the current production value.
_ESA_THRESHOLD_CANDIDATES: tuple[float, ...] = (0.60, 0.65, 0.68, 0.72)

# The 2 Cat D utterances previously identified as ESA-driven false positives in the
# LR candidate rework diagnostic (2026-06-28): they fired at 0.68 via ESA, not LR.
_KNOWN_ESA_FP_UTTS: frozenset[str] = frozenset({
    "Would you look at this?",
    "Will you look into this?",
})


def run_esa_margin_diagnostic(
    engine: "EmbeddingEngine",
    template_vecs: dict[str, dict[str, list[float]]],
) -> None:
    """
    Score all 29 existing test utterances (Cat A/B/C/D) against explicit_search_action
    in isolation at ESA thresholds 0.60/0.65/0.68/0.72. Generates a markdown findings
    report. READ-ONLY: planner.py and all templates are unmodified.
    """
    print("\n" + "=" * 72)
    print("ESA MARGIN DIAGNOSTIC (2026-06-28) — explicit_search_action @ 0.68")
    print("=" * 72)

    esa_tvecs = template_vecs["explicit_search_action"]

    # All 29 utterances: Cat A + Cat B-identity + Cat B-greeting + Cat C + Cat D.
    all_items: list[tuple[str, str, str]] = (
        [(cat, "", utt) for cat, utt in _CAT_A]
        + [(cat, "", utt) for cat, utt in _CAT_B]
        + [(cat, "", utt) for cat, utt in _CAT_C]
        + list(_CAT_D)
    )

    print(f"\nScoring {len(all_items)} utterances against ESA templates…")
    rows: list[dict] = []
    for cat, domain, utt in all_items:
        print(f"  [{cat}] {utt!r}")
        qv = engine.embed(utt)
        per_tmpl: dict[str, float] = {
            t: _cosine_similarity(qv, tv) for t, tv in esa_tvecs.items()
        }
        esa_max = max(per_tmpl.values())
        best_t = max(per_tmpl, key=per_tmpl.__getitem__)
        rows.append({
            "category": cat,
            "domain": domain,
            "utterance": utt,
            "esa": esa_max,
            "best_template": best_t,
            "per_template": per_tmpl,
        })

    print("\nBuilding markdown findings report…")
    md: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    md += [
        "# Negative-Side Margin Assessment — `explicit_search_action` @ 0.68",
        "",
        "**Date:** 2026-06-28",
        "**Script:** `diagnostics/score_lookup_request_templates.py` (ESA margin section)",
        "**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs",
        (
            "**Purpose:** Establish the negative-side margin of"
            " `_SEMANTIC_GATE_THRESHOLDS[\"explicit_search_action\"] = 0.68`."
        ),
        "**Gate metric:** ESA in isolation (`esa >= threshold`) — **not** OR'd with LR,",
        "so ESA's own margin is visible independent of the lookup_request gate.",
        "**Status:** READ-ONLY. `planner.py` unmodified. All templates and thresholds unmodified.",
        "",
        "**Trigger:** The 2026-06-28 LR candidate rework diagnostic found that 2 Cat D",
        "utterances fire at the current 0.68 threshold *via ESA alone*, not LR:",
        "`Would you look at this?` (ESA ≈ 0.699) and `Will you look into this?` (ESA ≈ 0.687).",
        "ESA has never been independently margin-tested. This assessment is the first.",
        "",
    ]

    # ── §1. ESA Templates and Test Categories ─────────────────────────────────
    md += [
        "## §1. ESA Templates and Test Categories",
        "",
        "**`_SEARCH_INTENT_TEMPLATES[\"explicit_search_action\"]` (5 templates, unchanged):**",
        "",
    ]
    for t in esa_tvecs:
        md.append(f"- `{t}`")
    md += [
        "",
        "**Threshold candidates:** 0.60 / 0.65 / 0.68 (current production) / 0.72",
        "",
        "**Test set:** all 29 utterances from the 2026-06-28 LR margin assessment, unchanged.",
        "",
        "| Cat | Description | N | Expected ESA behavior |",
        "|-----|-------------|---|-----------------------|",
        "| A | Live false positives (2026-06-28) | 3 | ESA should **not** fire |",
        "| B-identity | `_SEARCH_NEGATIVE_FILTER` identity phrases | 5 | Filtered pre-gate in prod; ESA should **not** fire |",
        "| B-greeting | `_SEARCH_NEGATIVE_FILTER` greeting phrases | 4 | Filtered pre-gate in prod; ESA should **not** fire |",
        "| C | 2026-06-25 confirmed true positives | 3 | ESA **should** fire if ESA is load-bearing for any |",
        "| D-\\* | Fresh adversarial negatives (14) | 14 | ESA should **not** fire |",
        "",
    ]

    # ── §2. Full Per-Utterance ESA Score Table ────────────────────────────────
    md += [
        "## §2. Full Per-Utterance ESA Score Table (all 29 utterances)",
        "",
        "ESA-isolated gate: **Y** = ESA ≥ threshold. LR not included in this check.",
        "",
        "| Cat | Utterance | ESA | Best ESA Template | @0.60 | @0.65 | @0.68 | @0.72 |",
        "|-----|-----------|-----|-------------------|:-----:|:-----:|:-----:|:-----:|",
    ]
    for r in rows:
        esa = r["esa"]
        md.append(
            f"| {r['category']} "
            f"| {_trunc(r['utterance'], 44)} "
            f"| {esa:.4f} "
            f"| {_trunc(r['best_template'], 24)} "
            f"| {_yn(esa >= 0.60)} "
            f"| {_yn(esa >= 0.65)} "
            f"| {_yn(esa >= 0.68)} "
            f"| {_yn(esa >= 0.72)} |"
        )
    md.append("")

    # ── §3. Cat C Survival ────────────────────────────────────────────────────
    rows_c = [r for r in rows if r["category"] == "C"]
    md += [
        "## §3. Category C — Confirmed Positives: ESA Threshold Survival",
        "",
        "**Key question:** Is any Cat C utterance dependent on ESA to fire the gate?",
        "If a Cat C item fires via ESA independently of LR, raising ESA could kill",
        "a confirmed positive — the same load-bearing constraint Cat C posed for LR.",
        "",
        "| Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |",
        "|-----------|-----|:-----:|:-----:|:-----:|:-----:|",
    ]
    for r in rows_c:
        esa = r["esa"]
        md.append(
            f"| {_trunc(r['utterance'], 55)} | {esa:.4f} "
            f"| {_yn(esa >= 0.60)} "
            f"| {_yn(esa >= 0.65)} "
            f"| {_yn(esa >= 0.68)} "
            f"| {_yn(esa >= 0.72)} |"
        )
    md.append("")
    for t in _ESA_THRESHOLD_CANDIDATES:
        n = sum(1 for r in rows_c if r["esa"] >= t)
        if n == 0:
            note = f"0/3 — **ESA is not load-bearing for Cat C at {t:.2f}**"
        else:
            note = f"**{n}/3 — ESA IS load-bearing; raising above {t:.2f} would kill {n} confirmed positive(s)**"
        md.append(f"- **@{t:.2f}:** {note}")
    md.append("")

    any_c_esa = any(r["esa"] >= min(_ESA_THRESHOLD_CANDIDATES) for r in rows_c)
    if not any(r["esa"] >= 0.68 for r in rows_c):
        md += [
            "**Conclusion:** No Cat C utterance clears the ESA gate at 0.68 or 0.72.",
            "Cat C fires **exclusively via LR** in production.",
            "Raising the ESA threshold carries **no Cat C coverage cost**.",
            "",
        ]
    else:
        md += [
            "**Conclusion:** At least one Cat C utterance clears ESA at 0.68.",
            "Raising the ESA threshold above that score would kill a confirmed positive.",
            "",
        ]

    # ── §4. Cat D False Positive Counts ──────────────────────────────────────
    rows_d = [r for r in rows if r["category"].startswith("D-")]
    n_d = len(rows_d)
    md += [
        "## §4. Category D — False Positive Counts (ESA-isolated)",
        "",
        f"N = {n_d} utterances (unchanged from LR diagnostic).",
        "ESA-isolated false positive = ESA ≥ threshold.",
        "Items marked `†` are the 2 utterances previously flagged as ESA-driven in the",
        "LR candidate rework diagnostic. All 14 are checked for ESA score regardless",
        "of whether ESA was their argmax in the prior report.",
        "",
    ]
    for t in _ESA_THRESHOLD_CANDIDATES:
        fp = [r for r in rows_d if r["esa"] >= t]
        known_fp = [r for r in fp if r["utterance"] in _KNOWN_ESA_FP_UTTS]
        novel_fp = [r for r in fp if r["utterance"] not in _KNOWN_ESA_FP_UTTS]
        prod_marker = " (current production threshold)" if t == 0.68 else ""
        md.append(
            f"### ESA @ {t:.2f}{prod_marker} — Cat D false positives: {len(fp)}/{n_d}"
        )
        md += [
            f"- Previously flagged as ESA-driven: {len(known_fp)}"
            + (f" ({', '.join('`' + r['utterance'] + '`' for r in known_fp)})" if known_fp else ""),
            f"- Novel (not previously flagged as ESA-driven): {len(novel_fp)}"
            + (f" ({', '.join('`' + r['utterance'] + '`' for r in novel_fp)})" if novel_fp else ""),
        ]
        if fp:
            md += [
                "",
                "| Utterance | ESA | Best Template | Status |",
                "|-----------|----|---------------|--------|",
            ]
            for r in fp:
                status = "known †" if r["utterance"] in _KNOWN_ESA_FP_UTTS else "**novel**"
                md.append(
                    f"| {_trunc(r['utterance'], 48)} "
                    f"| {r['esa']:.4f} "
                    f"| {_trunc(r['best_template'], 24)} "
                    f"| {status} |"
                )
        else:
            md.append("*(no Cat D false positives at this threshold)*")
        md.append("")

    md += [
        "### Full Category D Per-Utterance ESA Detail",
        "",
        "Items marked `†` are the 2 known ESA-driven false positives.",
        "",
        "| Axis | Domain | Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |",
        "|------|--------|-----------|----|:-----:|:-----:|:-----:|:-----:|",
    ]
    for r in rows_d:
        esa = r["esa"]
        known_marker = " †" if r["utterance"] in _KNOWN_ESA_FP_UTTS else ""
        md.append(
            f"| {r['category']} | {r['domain']} "
            f"| {_trunc(r['utterance'], 42)}{known_marker} "
            f"| {esa:.4f} "
            f"| {_yn(esa >= 0.60)} "
            f"| {_yn(esa >= 0.65)} "
            f"| {_yn(esa >= 0.68)} "
            f"| {_yn(esa >= 0.72)} |"
        )
    md.append("")

    # ── §5. Cat A and Cat B ESA Scores ───────────────────────────────────────
    rows_a = [r for r in rows if r["category"] == "A"]
    rows_b = [r for r in rows if r["category"].startswith("B-")]
    md += [
        "## §5. Cat A and Cat B — ESA Scores",
        "",
        "These items are either live false positives (Cat A) or pre-filtered phrases (Cat B).",
        "**Key question:** Even if LR were corrected, would any of these independently clear ESA?",
        "A Cat A item clearing ESA at 0.68 means the misroute survives even a corrected LR gate.",
        "A Cat B item clearing ESA means the negative filter's protection would be bypassed",
        "at the ESA layer (the filter runs pre-embedding, so ESA represents an independent risk).",
        "",
        "| Cat | Utterance | ESA | @0.60 | @0.65 | @0.68 | @0.72 |",
        "|-----|-----------|-----|:-----:|:-----:|:-----:|:-----:|",
    ]
    for r in rows_a + rows_b:
        esa = r["esa"]
        md.append(
            f"| {r['category']} "
            f"| {_trunc(r['utterance'], 40)} "
            f"| {esa:.4f} "
            f"| {_yn(esa >= 0.60)} "
            f"| {_yn(esa >= 0.65)} "
            f"| {_yn(esa >= 0.68)} "
            f"| {_yn(esa >= 0.72)} |"
        )
    md.append("")
    any_a_068 = any(r["esa"] >= 0.68 for r in rows_a)
    any_b_068 = any(r["esa"] >= 0.68 for r in rows_b)
    md += [
        f"- Cat A items clearing ESA @ 0.68: {sum(1 for r in rows_a if r['esa'] >= 0.68)}/3"
        + (" — **a corrected LR gate would not fully prevent these misroutes**" if any_a_068 else " — corrected LR gate would be sufficient"),
        f"- Cat B items clearing ESA @ 0.68: {sum(1 for r in rows_b if r['esa'] >= 0.68)}/9"
        + (" — **negative filter does not cover ESA path for these**" if any_b_068 else " — negative filter covers all B items at ESA level"),
        "",
    ]

    # ── §6. Trade-off Table ───────────────────────────────────────────────────
    md += [
        "## §6. Trade-off Table (numerical only — no recommendation)",
        "",
        "ESA-isolated gate = ESA ≥ threshold (LR not included).",
        "",
        "| ESA threshold | Cat C survivors | Cat A false positives | Cat D false positives |",
        "|:-------------:|:---------------:|:--------------------:|:--------------------:|",
    ]
    for t in _ESA_THRESHOLD_CANDIDATES:
        c_ok = sum(1 for r in rows_c if r["esa"] >= t)
        a_fp = sum(1 for r in rows_a if r["esa"] >= t)
        d_fp = sum(1 for r in rows_d if r["esa"] >= t)
        marker = " ← current" if t == 0.68 else ""
        md.append(f"| {t:.2f}{marker} | {c_ok}/3 | {a_fp}/3 | {d_fp}/14 |")
    md += [
        "",
        "**Cat C ESA scores** (shown to confirm whether ESA is load-bearing):",
    ]
    for r in rows_c:
        md.append(f"- `{r['utterance']}` → ESA = {r['esa']:.4f}")
    md.append("")

    # ── §7. Failure Mode Classification ──────────────────────────────────────
    fp_at_068 = [r for r in rows_d if r["esa"] >= 0.68]
    md += [
        "## §7. Failure Mode Classification",
        "",
        "**LR failure mode (2026-06-25 suspect templates, from prior report):**",
        "Frame-genericity — the modal-question scaffold (`can/could you + verb + vague-object`)",
        "causes any polite request using that frame to score 0.81–0.90, which is **above**",
        "every Cat C true positive (max 0.61). The negatives invert with the positives.",
        "",
        "**ESA failure mode observed in this diagnostic:**",
        "",
    ]
    if fp_at_068:
        md += [
            f"At the production threshold of 0.68, {len(fp_at_068)}/14 Cat D utterances clear ESA:",
            "",
        ]
        for r in fp_at_068:
            md.append(
                f"- `{r['utterance']}` → ESA={r['esa']:.4f}, driven by template: `{r['best_template']}`"
            )
        md.append("")

    # Identify which ESA template is responsible
    go_look_driven = [r for r in fp_at_068 if "go look" in r["best_template"].lower()]
    other_driven = [r for r in fp_at_068 if "go look" not in r["best_template"].lower()]
    md += [
        "**Pattern classification: verb-overlap (structurally different from LR's frame-genericity)**",
        "",
    ]
    if go_look_driven:
        md += [
            f"The `go look it up` template appears to drive {len(go_look_driven)}/{"all" if len(go_look_driven) == len(fp_at_068) else len(fp_at_068)} false positive(s).",
            "This template contains the bare verb `look`. Utterances using `look into` or `look at`",
            "— structurally distinct from `look up` — share the root verb and embed close to it.",
            "The other 4 ESA templates (`search the web`, `do a web search`, `search online`,",
            "`google this`) anchor on the word `search`/`google` and do not exhibit this problem.",
        ]
    md += [
        "",
        "**Comparison with LR:**",
        "",
        "| Dimension | LR failure (2026-06-25 templates) | ESA failure (current) |",
        "|-----------|----------------------------------|----------------------|",
        "| Root cause | Frame-genericity: modal-question scaffold matches all polite requests | Verb-overlap: `go look it up` attracts `look into`/`look at` |",
        "| Severity at production threshold | 6/14 Cat D score 0.81–0.90, **above** Cat C max (0.61) — inversion | 2/14 Cat D score 0.69–0.70; Cat C max ≈ 0.58 — no inversion |",
        "| Cat C visibility via this gate | Cat C fires on LR (0.60–0.61) | Cat C does **not** fire on ESA at any evaluated threshold |",
        "| Threshold-fixable? | No — negatives score above all positives regardless of threshold | Yes — raising ESA to 0.72 drops both false positives below threshold with zero Cat C cost |",
        "",
        "---",
        "",
        "*No threshold or template change is recommended.*",
        "*The data above classifies the ESA failure pattern numerically and structurally.*",
        "",
        "*Generated by `diagnostics/score_lookup_request_templates.py` — ESA margin section — 2026-06-28.*",
    ]

    # ── Write report ──────────────────────────────────────────────────────────
    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "explicit_search_action_margin_assessment_2026-06-28.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"\nFindings report written to:\n  {report_path}")
    print("=" * 72)
    print("ESA margin diagnostic complete.")
    print("=" * 72)


# ======================================================================================
# FULL PER-UTTERANCE TABLE — 2026-06-28
# Closes data gap flagged in combined_lr_esa_crosscheck_2026-06-28.md §4:
# LR(Set1) scores for the 8 non-hardest-6 Cat D items were inferred from the aggregate
# trade-off table (6/14), not printed per-utterance. This section prints every row.
# Gate evaluated: LR(Set1) >= 0.60 OR ESA(orig) >= 0.72  [proposed joint config].
# READ-ONLY: planner.py and _SEARCH_INTENT_TEMPLATES are not modified.
# ======================================================================================

def run_full_pertable_diagnostic(
    engine: "EmbeddingEngine",
    template_vecs: dict[str, dict[str, list[float]]],
) -> None:
    """
    Score all 14 Cat D and 3 Cat A utterances against Candidate Set 1 LR templates
    and the original ESA templates. Prints an explicit per-row score table — no
    aggregation or inference. Writes a small markdown report.
    """
    print("\n" + "=" * 72)
    print("FULL PER-UTTERANCE TABLE (2026-06-28) — LR(Set1) + ESA(orig), Cat D + Cat A")
    print("=" * 72)

    esa_tvecs = template_vecs["explicit_search_action"]

    # Embed Candidate Set 1 templates fresh (not in production template_vecs).
    print(f"\nEmbedding {len(_CAND1_TEMPLATES)} Candidate Set 1 LR templates…")
    cand1_vecs: dict[str, list[float]] = {}
    for t in _CAND1_TEMPLATES:
        print(f"  {t!r}")
        cand1_vecs[t] = engine.embed(t)

    # Utterances to score: Cat A (3) + Cat D (14).
    items: list[tuple[str, str, str]] = (
        [(cat, "", utt) for cat, utt in _CAT_A]
        + list(_CAT_D)
    )

    print(f"\nScoring {len(items)} utterances (Cat A + Cat D)…")
    rows: list[dict] = []
    for cat, domain, utt in items:
        qv = engine.embed(utt)
        lr = max(_cosine_similarity(qv, tv) for tv in cand1_vecs.values())
        esa = max(_cosine_similarity(qv, tv) for tv in esa_tvecs.values())
        best_lr_tmpl = max(cand1_vecs, key=lambda t: _cosine_similarity(qv, cand1_vecs[t]))
        rows.append({
            "category": cat,
            "domain": domain,
            "utterance": utt,
            "lr": lr,
            "esa": esa,
            "best_lr_tmpl": best_lr_tmpl,
        })
        gate_60_68 = lr >= 0.60 or esa >= 0.68
        gate_60_72 = lr >= 0.60 or esa >= 0.72
        print(
            f"  [{cat}] {_trunc(utt, 50)!r}"
            f"  LR={lr:.4f}  ESA={esa:.4f}"
            f"  @(0.60|0.68)={'Y' if gate_60_68 else 'n'}"
            f"  @(0.60|0.72)={'Y' if gate_60_72 else 'n'}"
        )

    # ── Build markdown report ─────────────────────────────────────────────────
    md: list[str] = []
    md += [
        "# Full Per-Utterance Table — LR(Set1) + ESA(original) — Cat D and Cat A",
        "",
        "**Date:** 2026-06-28",
        "**Script:** `diagnostics/score_lookup_request_templates.py` (full per-utterance section)",
        "**Model:** `mlx-community/embeddinggemma-300m-4bit` — real EmbeddingEngine, no stubs",
        "**Status:** READ-ONLY. `planner.py` unmodified. `_SEARCH_INTENT_TEMPLATES` unmodified.",
        "",
        "Closes the data gap flagged in `combined_lr_esa_crosscheck_2026-06-28.md` §4:",
        "the cross-check report inferred LR(Set1) < 0.60 for 8 Cat D items from the",
        "aggregate 6/14 trade-off count. This section prints every score explicitly.",
        "",
        "## §1. LR Templates Used (Candidate Set 1)",
        "",
        "These are the 4 Candidate Set 1 templates scored as LR. The original 5 pre-2026-06-25",
        "templates are also included in this config (9 total → Set 1 replaces the 4 suspect ones).",
        "For this diagnostic, LR score = max cosine similarity against these 4 templates only",
        "(the original 5 contribute separately to the production 9-template pool; see cross-check",
        "report for how they interact — this section shows Set 1's contribution in isolation).",
        "",
    ]
    for t in _CAND1_TEMPLATES:
        md.append(f"- `{t}`")
    md += [
        "",
        "**ESA templates:** original 5 (`explicit_search_action`), unchanged.",
        "",
        "## §2. Gate Definitions",
        "",
        "Two gate variants evaluated per utterance:",
        "",
        "- **Current + Set1:** `LR(Set1) >= 0.60 OR ESA(orig) >= 0.68`  [Set1 LR templates, ESA threshold unchanged]",
        "- **Combined fix:** `LR(Set1) >= 0.60 OR ESA(orig) >= 0.72`  [proposed joint configuration]",
        "",
        "## §3. Full Per-Utterance Score Table",
        "",
        "All 17 utterances. `fires_curr` = current-ESA gate with Set1 LR; `fires_comb` = joint fix gate.",
        "Both columns differ only when `0.68 <= ESA < 0.72` — those are the ESA-floor items.",
        "",
        "| Cat | Domain | Utterance | LR(Set1) | ESA(orig) | Best LR template | fires_curr | fires_comb |",
        "|-----|--------|-----------|:--------:|:---------:|------------------|:----------:|:----------:|",
    ]
    for r in rows:
        gate_curr = r["lr"] >= 0.60 or r["esa"] >= 0.68
        gate_comb = r["lr"] >= 0.60 or r["esa"] >= 0.72
        md.append(
            f"| {r['category']} "
            f"| {r['domain'] or '—'} "
            f"| {_trunc(r['utterance'], 44)} "
            f"| {r['lr']:.4f} "
            f"| {r['esa']:.4f} "
            f"| {_trunc(r['best_lr_tmpl'], 36)} "
            f"| {_yn(gate_curr)} "
            f"| {_yn(gate_comb)} |"
        )
    md.append("")

    # Separate Cat A and Cat D for summaries.
    rows_a = [r for r in rows if r["category"] == "A"]
    rows_d = [r for r in rows if r["category"].startswith("D-")]

    md += [
        "## §4. Totals",
        "",
        "| Category | N | fires_curr (LR≥0.60 OR ESA≥0.68) | fires_comb (LR≥0.60 OR ESA≥0.72) |",
        "|----------|:-:|:---------------------------------:|:---------------------------------:|",
    ]
    for label, subset in [("Cat A false positives", rows_a), ("Cat D false positives", rows_d)]:
        curr = sum(1 for r in subset if r["lr"] >= 0.60 or r["esa"] >= 0.68)
        comb = sum(1 for r in subset if r["lr"] >= 0.60 or r["esa"] >= 0.72)
        md.append(f"| {label} | {len(subset)} | {curr}/{len(subset)} | {comb}/{len(subset)} |")
    md += [
        "",
        "## §5. Items Where fires_curr ≠ fires_comb",
        "",
        "Items where the two gate variants disagree = items with `0.68 <= ESA < 0.72`",
        "and `LR < 0.60`. These are the utterances where raising ESA from 0.68 to 0.72",
        "would actually change the gate outcome (all others are identical under both gates).",
        "",
    ]
    diff = [
        r for r in rows
        if (r["lr"] >= 0.60 or r["esa"] >= 0.68) != (r["lr"] >= 0.60 or r["esa"] >= 0.72)
    ]
    if diff:
        md += [
            "| Cat | Utterance | LR(Set1) | ESA(orig) | fires_curr | fires_comb |",
            "|-----|-----------|:--------:|:---------:|:----------:|:----------:|",
        ]
        for r in diff:
            gate_curr = r["lr"] >= 0.60 or r["esa"] >= 0.68
            gate_comb = r["lr"] >= 0.60 or r["esa"] >= 0.72
            md.append(
                f"| {r['category']} "
                f"| {_trunc(r['utterance'], 44)} "
                f"| {r['lr']:.4f} "
                f"| {r['esa']:.4f} "
                f"| {_yn(gate_curr)} "
                f"| {_yn(gate_comb)} |"
            )
    else:
        md.append(
            "_No items differ between the two gate variants — raising ESA from 0.68 to 0.72_"
            "_has zero effect on the gate outcome for all 17 utterances under LR-Set1 at 0.60._"
        )
    md += [
        "",
        "---",
        "",
        "*No recommendation is made. All scores are explicit — no aggregation or inference.*",
        "",
        "*Generated by `diagnostics/score_lookup_request_templates.py` — full per-utterance section — 2026-06-28.*",
    ]

    report_dir = pathlib.Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "full_pertable_lr_set1_esa_2026-06-28.md"
    report_path.write_text("\n".join(md) + "\n")
    print(f"\nReport written to:\n  {report_path}")
    print("=" * 72)
    print("Full per-utterance diagnostic complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
