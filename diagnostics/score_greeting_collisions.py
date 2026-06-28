"""
Diagnostic: score_greeting_collisions.py
=========================================
Read-only live diagnostic. Makes NO changes to planner.py, the negative
filter, the template groups, or the gate thresholds.

Purpose
-------
2026-06-27 live finding: "Hey LORA!" scored 0.612 against the
lookup_request group, clearing the 0.60 gate set in the 2026-06-25
update B threshold-lowering commit — the same change whose comment
explicitly named "any live false positive on lookup_request" as the
trigger to revisit that value. This is that trigger.

This script does NOT propose a fix. It exists to answer the open
question from the 2026-06-27 diagnosis: is "Hey LORA!" a single
artifact (e.g. of the exclamation mark or the "LORA" token specifically),
or does it represent a class of bare greetings sitting near the 0.60
line generally? That distinction determines whether the eventual fix
is a negative-filter addition, a threshold revisit, or a template-set
change — and that decision is explicitly deferred to Michael, not made
here.

2026-06-27 follow-up — length-controlled extension
---------------------------------------------------
The first run (exact_repeat / isolation / common_greeting / known_anchor
probe groups) showed every short greeting clustering in a 0.59-0.65 band
on lookup_request regardless of content — "Hey" (1 word) at 0.648,
"Hey LORA!" (2 words) at 0.612, suggesting the score moves with length,
not with greeting-ness specifically. But that run never held length
constant while varying content (greeting vs. non-greeting small talk),
so "short strings collide" and "greetings collide" were both still
consistent with the data.

This extension adds three length-matched probe groups — greetings,
non-greeting small talk, and the lookup_request templates themselves —
at 1, 2, 3, and 4-word lengths, so length can be read off as a
controlled variable instead of an incidental one. If non-greeting
small talk at the same word count scores comparably to greetings, that
points at "low-information short string" as the mechanism rather than
"greeting" as a semantic category. If greetings score consistently
higher than length-matched small talk, that would instead point at
something greeting-specific (e.g. proximity to "look" / address-form
syntax). Either way, this remains an observation script — it reports
the numbers and does not pick a side.

Method
------
Constructs a REAL EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit)
and a REAL Planner, then calls Planner._semantic_search_intent() directly
— the exact method Priority 3 calls in production — for each probe
utterance. No mocking, no synthetic vectors, no monkeypatching.

This mirrors the method of the original 18-utterance Diagnostic 2 pass
and the 2026-06-26 four-template A/B test: live model, live gate logic,
scores reported raw with no interpretation baked in.

Usage
-----
    cd <localist repo root>
    python diagnostics/score_greeting_collisions.py

Output
------
For each probe utterance: best_group, best_score, full per-group score
dict, gate_fired (per the real _SEMANTIC_GATE_THRESHOLDS), and whether
the negative filter intercepted it before embedding ever ran.

The length-controlled extension additionally prints a track × word-count
comparison table of mean lookup_request scores, to make the length
question readable at a glance rather than buried in per-utterance rows.

Exit code is always 0. This script only observes; it never asserts
pass/fail, since "pass/fail" here is a judgment call for Michael once
the numbers are in front of him.
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Probe set
# ---------------------------------------------------------------------------
# Group A — the confirmed live false positive, exact string, run twice
# in the actual log (same string both times per Michael's confirmation).
# Included multiple times here only to confirm determinism of the live
# model at temperature-irrelevant embedding (embedding has no temperature
# knob, but re-running confirms no nondeterminism in the mlx-embeddings
# path itself).
_PROBE_EXACT_REPEAT = [
    "Hey LORA!",
    "Hey LORA!",
    "Hey LORA!",
]

# Group B — minimal variations isolating the punctuation and the "LORA"
# token specifically. If only "Hey LORA!" (with exclamation, with name)
# scores high and the bare "hey" variants do not, that points at a
# narrow artifact rather than a general greeting collision.
_PROBE_ISOLATION = [
    "Hey LORA",       # no exclamation mark
    "Hey LORA.",      # period instead of exclamation
    "hey lora!",      # already-lowercased — should be identical post lower()
    "Hey",            # bare greeting, no name, no punctuation
    "Hey!",           # bare greeting with exclamation
]

# Group C — common bare greetings with no name reference, to test
# whether this is a "greeting" class problem or specific to "Hey LORA!".
_PROBE_COMMON_GREETINGS = [
    "Hi",
    "Hi!",
    "Hello",
    "Hello!",
    "Hey there",
    "Good morning",
    "Good morning!",
    "What's up",
    "Yo",
]

# Group D — known negatives from the original diagnostic passes, included
# as a sanity check that this script reproduces prior confirmed scores
# (does NOT re-verify the full 18-utterance set — just enough anchor
# points to catch a setup error in this script, e.g. wrong model path,
# wrong prefix, stale template list).
_PROBE_KNOWN_ANCHORS = [
    "who are you",                 # confirmed negative-filter catch (2026-06-26)
    "can you look up",             # confirmed lookup_request positive template itself
    "what is this",                # knowledge_request_open template itself (ungated)
]

ALL_PROBES: list[tuple[str, str]] = (
    [("exact_repeat", p) for p in _PROBE_EXACT_REPEAT]
    + [("isolation", p) for p in _PROBE_ISOLATION]
    + [("common_greeting", p) for p in _PROBE_COMMON_GREETINGS]
    + [("known_anchor", p) for p in _PROBE_KNOWN_ANCHORS]
)

# ---------------------------------------------------------------------------
# Length-controlled extension (2026-06-27 follow-up)
# ---------------------------------------------------------------------------
# Three parallel tracks, matched word-for-word at lengths 1-4, so length
# can be read as a controlled variable rather than an incidental one in
# the results. Word count is by whitespace split, counting contractions
# ("what's") as one word, consistent with how a person would count them.
#
# Track 1 — greetings. Same intent as the original common_greeting probe
# but explicitly bucketed by length here instead of mixed together.
_PROBE_LENGTH_GREETING: dict[int, list[str]] = {
    1: ["Hi", "Hey", "Yo"],
    2: ["Hi there", "Hey LORA", "Good day"],
    3: ["Hey there friend", "Good morning LORA", "What's up LORA"],
    4: ["Hey there, how's it", "Good morning to you", "Hello there my friend"],
}

# Track 2 — non-greeting small talk. Deliberately NOT greetings and NOT
# search-intent — ordinary conversational filler a person might say to a
# present, already-engaged assistant. This is the comparison condition:
# if these score as high as Track 1 at matched length, the mechanism is
# "short string" in general, not "greeting" specifically.
_PROBE_LENGTH_SMALLTALK: dict[int, list[str]] = {
    1: ["Sure", "Okay", "Nice"],
    2: ["Sounds good", "That's great", "Fair enough"],
    3: ["That makes sense", "I appreciate that", "Sounds good actually"],
    4: ["That makes a lot", "I think that's right", "Yeah that works fine"],
}

# Track 3 — lookup_request's own templates, length-bucketed, as the
# positive-control reference line. These are NOT length-matched 1:1 with
# tracks 1/2 (the group has no genuine 1-word search-intent phrase — that
# itself may be informative) but bucketing the existing template set by
# length lets us see whether the group's own positives cluster at a
# particular length, which would explain why short strings drift toward it.
# Word counts below are verified by whitespace split (matching the same
# convention used in tracks 1/2), not eyeballed — a mis-bucketed template
# would quietly corrupt the comparison table this script exists to produce.
_PROBE_LENGTH_LOOKUP_REFERENCE: dict[int, list[str]] = {
    2: ["google this"],
    3: ["look that up", "look up this", "find information on"],
    4: ["go ahead and look", "can you look up",
        "go look it up", "find information on this",
        "could you look up", "find out about this"],
}


def _length_probes() -> list[tuple[str, int, str]]:
    """
    Flatten the three length-matched tracks into (track, word_count,
    utterance) triples for iteration and later tabulation.
    """
    out: list[tuple[str, int, str]] = []
    for track_name, buckets in (
        ("greeting", _PROBE_LENGTH_GREETING),
        ("smalltalk", _PROBE_LENGTH_SMALLTALK),
        ("lookup_reference", _PROBE_LENGTH_LOOKUP_REFERENCE),
    ):
        for word_count, utterances in buckets.items():
            for u in utterances:
                out.append((track_name, word_count, u))
    return out


LENGTH_PROBES: list[tuple[str, int, str]] = _length_probes()


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

    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit) …")
    engine = EmbeddingEngine()
    if not engine.available:
        print("FATAL: EmbeddingEngine.available is False — model failed to "
              "load. Check the warning logged during _load(). This script "
              "requires the real model; it will not fall back to a stub.",
              file=sys.stderr)
        return 1

    print("Constructing Planner with live embed_fn (no MemoryManager — "
          "Priority 4 corpus paths are not exercised by this diagnostic) …")
    planner = Planner(runtime=None, memory_manager=None, embed_fn=engine.embed)

    if not planner._template_embeddings:
        print("FATAL: planner._template_embeddings is empty — template "
              "pre-embedding failed at construction. See warning above.",
              file=sys.stderr)
        return 1

    print(f"\nGate thresholds in effect: {_SEMANTIC_GATE_THRESHOLDS}")
    print(f"Negative filter has {len(_SEARCH_NEGATIVE_FILTER)} entries.\n")
    print("=" * 100)

    rows: list[dict] = []

    for probe_group, utterance in ALL_PROBES:
        lowered = utterance.lower()

        # Replicate the exact pre-check Planner._semantic_search_intent()
        # performs, so we can report whether the negative filter would
        # have intercepted this utterance BEFORE we call the real method
        # (the real method also does this internally and returns None
        # silently on a filter hit — we want that fact surfaced, not hidden).
        filter_hit = next(
            (p for p in _SEARCH_NEGATIVE_FILTER if p in lowered), None
        )

        result = planner._semantic_search_intent(lowered)

        print(f"[{probe_group:>15}] {utterance!r}")
        if filter_hit is not None:
            print(f"                  → negative filter intercepted "
                  f"(matched phrase: {filter_hit!r}). No embedding run.")
            rows.append({
                "group": probe_group, "utterance": utterance,
                "filter_hit": filter_hit, "best_group": None,
                "best_score": None, "gate_fired": False,
                "all_scores": None,
            })
            print("-" * 100)
            continue

        if result is None:
            print("                  → _semantic_search_intent returned None "
                  "(embed_fn unavailable or embedding call failed — should "
                  "not happen given the checks above).")
            rows.append({
                "group": probe_group, "utterance": utterance,
                "filter_hit": None, "best_group": None,
                "best_score": None, "gate_fired": False,
                "all_scores": None,
            })
            print("-" * 100)
            continue

        best_group, best_score, all_scores = result
        gate_fired = any(
            all_scores.get(g, 0.0) >= thresh
            for g, thresh in _SEMANTIC_GATE_THRESHOLDS.items()
        )

        scores_str = ", ".join(
            f"{g}={s:.3f}" for g, s in sorted(all_scores.items())
        )
        print(f"                  → best={best_group}({best_score:.3f})  "
              f"gate_fired={gate_fired}")
        print(f"                  → all: {scores_str}")

        rows.append({
            "group": probe_group, "utterance": utterance,
            "filter_hit": None, "best_group": best_group,
            "best_score": best_score, "gate_fired": gate_fired,
            "all_scores": all_scores,
        })
        print("-" * 100)

    # ---------------------------------------------------------------------
    # Length-controlled extension — same method, no shortcuts. Each
    # utterance goes through the identical negative-filter check +
    # _semantic_search_intent() call as the first pass.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("LENGTH-CONTROLLED EXTENSION — greeting vs. small talk vs. "
          "lookup_request templates, matched by word count\n")
    print("=" * 100)

    length_rows: list[dict] = []

    for track_name, word_count, utterance in LENGTH_PROBES:
        lowered = utterance.lower()

        filter_hit = next(
            (p for p in _SEARCH_NEGATIVE_FILTER if p in lowered), None
        )
        result = None if filter_hit is not None else planner._semantic_search_intent(lowered)

        print(f"[{track_name:>17} | {word_count}w] {utterance!r}")
        if filter_hit is not None:
            print(f"                       → negative filter intercepted "
                  f"(matched: {filter_hit!r}).")
            length_rows.append({
                "track": track_name, "words": word_count, "utterance": utterance,
                "filter_hit": filter_hit, "best_group": None,
                "best_score": None, "lookup_score": None, "gate_fired": False,
            })
            print("-" * 100)
            continue

        if result is None:
            print("                       → returned None (unexpected — "
                  "embed_fn should be available here).")
            length_rows.append({
                "track": track_name, "words": word_count, "utterance": utterance,
                "filter_hit": None, "best_group": None,
                "best_score": None, "lookup_score": None, "gate_fired": False,
            })
            print("-" * 100)
            continue

        best_group, best_score, all_scores = result
        lookup_score = all_scores.get("lookup_request", 0.0)
        gate_fired = any(
            all_scores.get(g, 0.0) >= thresh
            for g, thresh in _SEMANTIC_GATE_THRESHOLDS.items()
        )
        print(f"                       → best={best_group}({best_score:.3f})  "
              f"lookup_request={lookup_score:.3f}  gate_fired={gate_fired}")

        length_rows.append({
            "track": track_name, "words": word_count, "utterance": utterance,
            "filter_hit": None, "best_group": best_group,
            "best_score": best_score, "lookup_score": lookup_score,
            "gate_fired": gate_fired,
        })
        print("-" * 100)

    # -----------------------------------------------------------------
    # Comparison table — mean lookup_request score per (track, word_count)
    # cell. This is the table that actually answers the question: read
    # down a column (fixed length, varying track) to see whether
    # greeting and small-talk track together or diverge at that length.
    # Filter-intercepted / failed rows are excluded from the mean and
    # their exclusion is reported, not silently dropped.
    # -----------------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPARISON TABLE — mean lookup_request score by track × word count\n")

    all_word_counts = sorted({r["words"] for r in length_rows})
    track_order = ["greeting", "smalltalk", "lookup_reference"]

    header = "track".ljust(18) + "".join(f"{wc}w".rjust(10) for wc in all_word_counts)
    print(header)
    print("-" * len(header))

    excluded_cells: list[str] = []

    for track_name in track_order:
        row_cells = []
        for wc in all_word_counts:
            cell_rows = [
                r for r in length_rows
                if r["track"] == track_name and r["words"] == wc
            ]
            scored = [r["lookup_score"] for r in cell_rows if r["lookup_score"] is not None]
            n_excluded = len(cell_rows) - len(scored)
            if n_excluded:
                excluded_cells.append(
                    f"{track_name}/{wc}w: {n_excluded} of {len(cell_rows)} "
                    f"excluded (filtered or failed)"
                )
            if not cell_rows:
                row_cells.append("—".rjust(10))
            elif not scored:
                row_cells.append("n/a".rjust(10))
            else:
                mean_score = sum(scored) / len(scored)
                row_cells.append(f"{mean_score:.3f}".rjust(10))
        print(track_name.ljust(18) + "".join(row_cells))

    if excluded_cells:
        print("\nCells with excluded rows (not averaged in):")
        for line in excluded_cells:
            print(f"  - {line}")

    print(
        "\nReading guide: compare the 'greeting' row to the 'smalltalk' row "
        "at each word count. Close values at a given column = length is "
        "doing the work, not greeting-ness. A consistent gap = something "
        "about greetings specifically. The 'lookup_reference' row is the "
        "template group's own positive scores for context — not a strict "
        "control, since lookup_request has no natural 1-word phrasing."
    )


    print("\n" + "=" * 100)
    print("SUMMARY — utterances that would fire the semantic gate "
          "(tools_to_call would include web_search):\n")
    fired = [r for r in rows if r["gate_fired"]]
    fired_length = [r for r in length_rows if r["gate_fired"]]
    if not fired and not fired_length:
        print("  (none)")
    else:
        print("  -- from first pass --")
        for r in fired:
            print(f"  {r['utterance']!r}  →  {r['best_group']}"
                  f"({r['best_score']:.3f})")
        print("  -- from length-controlled pass --")
        for r in fired_length:
            print(f"  [{r['track']}/{r['words']}w] {r['utterance']!r}  →  "
                  f"{r['best_group']}({r['best_score']:.3f})")

    print("\nSUMMARY — utterances intercepted by the negative filter "
          "(no embedding run, no gate check possible):\n")
    filtered = [r for r in rows if r["filter_hit"] is not None]
    filtered_length = [r for r in length_rows if r["filter_hit"] is not None]
    if not filtered and not filtered_length:
        print("  (none)")
    else:
        for r in filtered:
            print(f"  {r['utterance']!r}  →  matched {r['filter_hit']!r}")
        for r in filtered_length:
            print(f"  [{r['track']}/{r['words']}w] {r['utterance']!r}  →  "
                  f"matched {r['filter_hit']!r}")

    print("\nRaw rows are available in-process as `rows` if you want to "
          "pipe this through further analysis; this script intentionally "
          "does not write a file or touch the database.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
