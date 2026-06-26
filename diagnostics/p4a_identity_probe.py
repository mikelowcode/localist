"""
p4a_identity_probe.py — Open Item 9 Diagnostic (read-only)

Probes the live MemoryManager / embedding backend for the 13 _IDENTITY_KEYWORDS
instructions to diagnose why P4a identity-route queries intermittently return
an empty [CONTEXT] block.

Run from the project root:
    python3 diagnostics/p4a_identity_probe.py

No existing file is modified. No new dependency is added.
"""

from __future__ import annotations

import sqlite3
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — backend modules live one directory up from this script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR  = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from embedding_engine import EmbeddingEngine  # noqa: E402
from memory_manager import (  # noqa: E402
    MemoryManager,
    _cosine_similarity,
    _unpack_embedding,
)

# ---------------------------------------------------------------------------
# Config — must match main.py lifespan exactly
# ---------------------------------------------------------------------------
DB_PATH       = PROJECT_ROOT / "backend" / "localist_memory.db"
PERSONA_PATH  = str(BACKEND_DIR / "wiki" / "lora-persona.md")

# The 13 _IDENTITY_KEYWORDS instructions — verbatim, exact case + punctuation
INSTRUCTIONS: list[str] = [
    "Who are you?",
    "What are you?",
    "Tell me about yourself.",
    "What can you do?",
    "Are you an AI?",
    "Are you a bot?",
    "What is LORA?",
    "Who is LORA?",
    "What is Localist?",
    "Are you made by Google?",
    "Are you ChatGPT?",
    "Are you Gemma?",
    "Introduce yourself.",
]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _load_persona_embedding(db_path: Path) -> list[float] | None:
    """Fetch lora-persona.md's stored embedding blob directly from document_index."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        row = conn.execute(
            "SELECT embedding FROM document_index WHERE path = ?",
            (PERSONA_PATH,),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row["embedding"] is None:
        return None
    return _unpack_embedding(row["embedding"])


def _simulate_p4a_filter(docs: list) -> list:
    """Reproduce the existing force_rag=True filter from controller_agent.py Step 4."""
    return [
        doc for doc in docs
        if not str(doc.path).endswith("lora-persona.md")
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("P4a Identity Probe — Open Item 9 Diagnostic")
    print(f"DB:     {DB_PATH}")
    print(f"Persona path: {PERSONA_PATH}")
    print("=" * 72)

    # Load the embedding engine (same path as main.py)
    print("\nLoading EmbeddingEngine …")
    engine = EmbeddingEngine()
    if not engine.available:
        print("ERROR: EmbeddingEngine failed to load — cannot run embedding probe.")
        sys.exit(1)
    print(f"EmbeddingEngine ready — model={engine.model_path!r}\n")

    # Construct MemoryManager against the live DB with the live embed_fn
    mm = MemoryManager(db_path=DB_PATH, embed_fn=engine.embed)

    # Pre-load persona embedding once (independent of any query_corpus call)
    persona_vec = _load_persona_embedding(DB_PATH)
    if persona_vec is None:
        print(f"WARNING: lora-persona.md embedding not found in DB at path:\n  {PERSONA_PATH}")
        print("Persona-similarity column will show N/A.\n")

    # Collect per-instruction results for the summary table
    summary_rows: list[dict] = []

    for instruction in INSTRUCTIONS:
        print(f'\n=== "{instruction}" ===')

        # Step 1 — query_corpus (identical call shape to P4a route)
        docs = mm.query_corpus(
            instruction,
            max_results    = 3,
            use_embeddings = True,
            doc_type       = "wiki",
        )

        print("Top-3 (query_corpus, doc_type=wiki):")
        for i, doc in enumerate(docs, 1):
            stem = Path(doc.path).name
            print(f"  {i}. {stem:<55} score={doc.relevance_score:.3f}")
        if not docs:
            print("  (no results)")

        # Step 2 — direct persona similarity (regardless of top-3 membership)
        if persona_vec is not None and mm._embed_fn is not None:
            instruction_vec   = mm._embed_fn(instruction)
            persona_sim       = _cosine_similarity(instruction_vec, persona_vec)
            persona_sim_str   = f"{persona_sim:.3f}"
        else:
            persona_sim       = None
            persona_sim_str   = "N/A"
        print(f"Persona similarity (direct, regardless of top-3 membership): {persona_sim_str}")

        # Step 3 — simulate P4a filter (force_rag=True path, persona exclusion only)
        survivors   = _simulate_p4a_filter(docs)
        surv_names  = [Path(d.path).name for d in survivors]
        print(f"Simulated rag_sources after P4a filter: {surv_names}")

        if survivors:
            print(f"Result: POPULATED ({len(survivors)} source(s))")
            result_label = "POPULATED"
        else:
            print("Result: EMPTY")
            result_label = "EMPTY"

        # Record for summary table
        top1_name  = Path(docs[0].path).name if docs else "(none)"
        top1_score = docs[0].relevance_score if docs else 0.0
        summary_rows.append({
            "instruction":   instruction,
            "top1":          top1_name,
            "top1_score":    top1_score,
            "persona_sim":   persona_sim if persona_sim is not None else -1.0,
            "result":        result_label,
            "surv_count":    len(survivors),
        })

    # ---------------------------------------------------------------------------
    # Summary table — sorted by persona_sim descending
    # ---------------------------------------------------------------------------
    summary_rows.sort(key=lambda r: r["persona_sim"], reverse=True)

    print("\n" + "=" * 72)
    print("SUMMARY TABLE (sorted by persona similarity, descending)")
    print("=" * 72)

    col_instr   = 34
    col_top1    = 40
    col_score   = 8
    col_psim    = 8
    col_result  = 11
    col_surv    = 7

    header = (
        f"{'Instruction':<{col_instr}} "
        f"{'Top-1 Doc':<{col_top1}} "
        f"{'T1Score':>{col_score}} "
        f"{'PsnSim':>{col_psim}} "
        f"{'Result':<{col_result}} "
        f"{'Survivors':>{col_surv}}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        psim_str = f"{row['persona_sim']:.3f}" if row["persona_sim"] >= 0 else "N/A"
        print(
            f"{row['instruction']:<{col_instr}} "
            f"{row['top1']:<{col_top1}} "
            f"{row['top1_score']:>{col_score}.3f} "
            f"{psim_str:>{col_psim}} "
            f"{row['result']:<{col_result}} "
            f"{row['surv_count']:>{col_surv}}"
        )

    print("=" * 72)
    n_empty     = sum(1 for r in summary_rows if r["result"] == "EMPTY")
    n_populated = sum(1 for r in summary_rows if r["result"] == "POPULATED")
    print(f"POPULATED: {n_populated}   EMPTY: {n_empty}   Total: {len(summary_rows)}")


if __name__ == "__main__":
    main()
