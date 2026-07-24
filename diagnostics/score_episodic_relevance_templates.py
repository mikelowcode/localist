"""
Diagnostic: score_episodic_relevance_templates.py
==================================================
Embeds a battery of true-positive (should trigger episodic fetch) and
true-negative (should not) utterances and scores each against the candidate
_EPISODIC_RELEVANCE_TEMPLATES set using the live EmbeddingEngine and
_cosine_similarity from planner.py.

Background: Planner Priority 5 (episodic-fetch gate) was keyword-only
("remember", "preference", "decision", ...) and missed a real instruction —
"Help me prepare for the upcoming Claude Impact Lab on August 6th." — that
should have triggered episodic retrieval. This script was used to pick both
the template phrases and the threshold for a new semantic gate added
alongside the keyword check, mirroring the pattern _semantic_search_intent()
already uses for Priority 3 (own template set, own threshold, own instance
state — kept structurally separate from P3's negative-filter/tie-break
machinery, which is search-intent-specific and irrelevant here).

No files are modified. Run from the repo root or the backend/ directory.

Usage:
    cd backend
    python ../diagnostics/score_episodic_relevance_templates.py
"""

from __future__ import annotations

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from embedding_engine import EmbeddingEngine
from planner import _cosine_similarity  # type: ignore[attr-defined]

TEMPLATES = [
    "help me prepare for my upcoming event",
    "what do I have coming up on my calendar",
    "remind me what we planned for this",
    "catch me up on what we discussed before",
    "what is on my schedule this week",
    "help me get ready for my appointment",
    "what did we decide about this before",
]

# (utterance, expected_positive)
POSITIVES = [
    "Help me prepare for the upcoming Claude Impact Lab on August 6th.",
    "What do I need to bring to my dentist appointment next week?",
    "Can you help me get ready for my presentation on Friday?",
    "What did we decide about the migration plan?",
    "What do I have going on this week?",
    "Catch me up on my project status.",
    "What was the plan we settled on last time?",
    "Help me get ready for my trip to Japan.",
    "Prep me for my meeting with the investors tomorrow.",
    "What did I say I would do about the server migration?",
]

NEGATIVES = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Search the web for the latest news on SpaceX.",
    "What is 2+2?",
    "Summarize this document for me.",
    "Fetch this URL and tell me what it says.",
    "Remind me how photosynthesis works.",
    "Help me write a cover letter for a job application.",
    "Prepare a report on quarterly earnings.",
    "What is on the front page of the New York Times today?",
    "Explain how neural networks work.",
    "Get ready to receive a large file upload.",
    "I have a headache, what should I do?",
    "My favorite color is blue, what goes well with it?",
    "Can you help me plan a birthday party for my friend?",
    "What is my IP address?",
    "Prepare a Python script that reverses a string.",
    "I need help understanding this error message.",
    "What time is it in Tokyo right now?",
    "Draft an email to my landlord about the leak.",
]

THRESHOLD = 0.70


def best_score(engine: EmbeddingEngine, template_vecs: list[tuple[str, list[float]]], text: str) -> tuple[float, str]:
    qv = engine.embed(text)
    scored = [(_cosine_similarity(qv, tv), t) for t, tv in template_vecs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0]


def main() -> None:
    print("Loading EmbeddingEngine (mlx-community/embeddinggemma-300m-4bit)…")
    engine = EmbeddingEngine()
    if not engine.available:
        print("ERROR: EmbeddingEngine failed to load. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print("EmbeddingEngine ready.\n")

    template_vecs = [(t, engine.embed(t)) for t in TEMPLATES]

    print(f"Threshold under test: {THRESHOLD}\n")
    print("=" * 72)
    print("POSITIVES (want score >= threshold)")
    print("=" * 72)
    fn = 0
    for p in POSITIVES:
        score, template = best_score(engine, template_vecs, p)
        ok = score >= THRESHOLD
        if not ok:
            fn += 1
        print(f"{'OK  ' if ok else 'MISS'}  {score:.4f}  [{template[:30]}]  {p}")

    print()
    print("=" * 72)
    print("NEGATIVES (want score < threshold)")
    print("=" * 72)
    fp = 0
    for n in NEGATIVES:
        score, template = best_score(engine, template_vecs, n)
        ok = score < THRESHOLD
        if not ok:
            fp += 1
        print(f"{'OK       ' if ok else 'FALSE-POS'}  {score:.4f}  [{template[:30]}]  {n}")

    print()
    print("=" * 72)
    print(
        f"Summary: {len(POSITIVES) - fn}/{len(POSITIVES)} true positives caught, "
        f"{len(NEGATIVES) - fp}/{len(NEGATIVES)} true negatives correctly excluded."
    )


if __name__ == "__main__":
    main()
