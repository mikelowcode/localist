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


if __name__ == "__main__":
    main()
