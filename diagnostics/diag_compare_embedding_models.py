"""
Embedding Model Comparison Diagnostic — EmbeddingEngine (MLX) vs. Ollama (nomic-embed-text)
============================================================================================

Live-verification script: independently embeds the real indexed corpus
(wiki + raw, from the production `localist_memory.db` via MemoryManager's
existing read-only accessors) and a fixed set of test queries under two
embedding sources, then reports per-query, per-document top-5 rankings and
raw cosine scores side by side.

Motivation
----------
Corpus retrieval under `nomic-embed-text:latest` produced one observed
low-score miss (top_score=0.028) on "Tell me how Localist works?" — a query
that plausibly should have matched wiki/how-localist-works.md. Per this
project's "verify the mechanism, not just the correlation" convention, this
script exists to gather per-utterance comparative data before concluding
anything about the cause (model behavior difference vs. threshold-tuning
issue vs. normal variance) or proposing any threshold change.

This is observational tooling only. It draws NO conclusions about whether
any similarity threshold should change — that is a separate, follow-up
decision once a human has reviewed this output.

Read-only guarantees
---------------------
- Uses MemoryManager.get_all_documents() (a SELECT-only accessor) against
  the real localist_memory.db to load the actual indexed corpus.
- MemoryManager's own __init__ / _init_db() only issues idempotent
  `CREATE TABLE IF NOT EXISTS` DDL and a schema-version check; it performs
  no migration when the on-disk schema is already current, which it is on
  this production database.
- All embedding vectors computed by this script exist only in this
  process's memory. Nothing is written back to any `embedding` column,
  any cache table, or any file. No production module (memory_manager.py,
  main.py, runtime_factory.py, any runtime client) is modified.
- Does not touch the similarity threshold used by planner.py or anywhere
  else.

Requirements to run
--------------------
- Apple Silicon machine, mlx-embeddings installed (EmbeddingEngine loads
  mlx-community/embeddinggemma-300m-4bit).
- Ollama daemon running locally with `nomic-embed-text:latest` pulled
  (`ollama list` should show it).
- Run from the project root:
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_compare_embedding_models.py

Not a pytest test — lives under diagnostics/, filename does not match
test_*.py, and is never collected by `pytest tests/`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from backend/
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from embedding_engine import EmbeddingEngine  # noqa: E402
from ollama_runtime_client import OllamaRuntimeClient  # noqa: E402
from memory_manager import MemoryManager, DocumentResult, _cosine_similarity  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = BACKEND_DIR / "localist_memory.db"

OLLAMA_BASE_URL     = "http://localhost:11434"
OLLAMA_EMBED_MODEL  = "nomic-embed-text:latest"
# embed() is the only method this script calls on the Ollama client; chat_model
# is required by the constructor but never exercised here.
OLLAMA_CHAT_MODEL   = "gemma4:31b-cloud"

TOP_K = 5

# The exact query that produced the observed miss, plus a spread of
# representative queries: near-exact title match, paraphrase, broad/vague,
# and a true-negative control (topic absent from the corpus entirely).
TEST_QUERIES: list[str] = [
    "Tell me how Localist works?",                          # observed miss
    "How does Localist work?",                               # paraphrase of the miss
    "localist design philosophy",                            # near-exact title match
    "What inference engines does Localist support?",         # moderately specific
    "What is this project about?",                            # vague/broad
    "What's a good recipe for chocolate chip cookies?",       # true-negative control
]


@dataclass
class ScoredDoc:
    name: str
    doc_type: str
    score: float


def load_corpus() -> list[DocumentResult]:
    """Read-only load of the real indexed corpus via MemoryManager."""
    mm = MemoryManager(db_path=DB_PATH)  # _init_db() is idempotent CREATE-IF-NOT-EXISTS only
    docs = mm.get_all_documents()  # SELECT-only accessor, no writes
    return docs


def embed_corpus(
    docs: list[DocumentResult],
    embed_fn,
    label: str,
) -> dict[str, list[float]]:
    """
    Embed every document's content under one model.

    Keyed by f"{doc_type}:{name}" to disambiguate wiki/raw docs that share a
    name (e.g. "how-localist-works" exists as both). Returns an in-memory
    dict only — nothing is persisted.
    """
    vectors: dict[str, list[float]] = {}
    failures = 0
    for doc in docs:
        key = f"{doc.doc_type}:{doc.name}"
        try:
            vectors[key] = embed_fn(doc.content)
        except Exception as exc:
            failures += 1
            print(f"  [{label}] WARNING: failed to embed doc {key!r}: {exc}")
    print(f"  [{label}] embedded {len(vectors)}/{len(docs)} documents "
          f"({failures} failure(s)).")
    return vectors


def rank_top_k(
    query_vec: list[float],
    doc_vectors: dict[str, list[float]],
    docs_by_key: dict[str, DocumentResult],
    k: int = TOP_K,
) -> list[ScoredDoc]:
    scored = [
        ScoredDoc(name=docs_by_key[key].name, doc_type=docs_by_key[key].doc_type,
                  score=_cosine_similarity(query_vec, vec))
        for key, vec in doc_vectors.items()
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]


def format_side_by_side(query: str, mlx_top: list[ScoredDoc], ollama_top: list[ScoredDoc]) -> str:
    lines = [f"### Query: {query!r}", ""]
    lines.append(f"{'Rank':<5}{'MLX (embeddinggemma)':<45}{'Ollama (nomic-embed-text)':<45}")
    lines.append("-" * 95)
    for i in range(max(len(mlx_top), len(ollama_top))):
        m = mlx_top[i] if i < len(mlx_top) else None
        o = ollama_top[i] if i < len(ollama_top) else None
        m_str = f"{m.doc_type}/{m.name}  ({m.score:.4f})" if m else "-"
        o_str = f"{o.doc_type}/{o.name}  ({o.score:.4f})" if o else "-"
        lines.append(f"{i + 1:<5}{m_str:<45}{o_str:<45}")
    agree = bool(mlx_top and ollama_top and
                 mlx_top[0].name == ollama_top[0].name and
                 mlx_top[0].doc_type == ollama_top[0].doc_type)
    lines.append("")
    lines.append(f"Top-1 agreement: {'YES' if agree else 'NO'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not DB_PATH.exists():
        print(f"FAIL: production database not found: {DB_PATH}")
        sys.exit(1)

    print(f"=== Loading real corpus from {DB_PATH} (read-only) ===")
    docs = load_corpus()
    docs_by_key = {f"{d.doc_type}:{d.name}": d for d in docs}
    print(f"Loaded {len(docs)} documents "
          f"({sum(1 for d in docs if d.doc_type == 'wiki')} wiki, "
          f"{sum(1 for d in docs if d.doc_type == 'raw')} raw).\n")

    print("=== Loading EmbeddingEngine (MLX, embeddinggemma-300m-4bit) ===")
    mlx_engine = EmbeddingEngine()
    if not mlx_engine.available:
        print("FAIL: EmbeddingEngine did not load — mlx-embeddings unavailable "
              "or model load failed. Cannot run this comparison. Aborting.")
        sys.exit(1)
    print("EmbeddingEngine loaded OK.\n")

    print("=== Connecting to Ollama (nomic-embed-text:latest) ===")
    ollama_client = OllamaRuntimeClient(
        chat_model=OLLAMA_CHAT_MODEL,
        embedding_model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    health = ollama_client.health_check()
    if not health.get("reachable"):
        print(f"FAIL: Ollama daemon not reachable at {OLLAMA_BASE_URL}. Aborting.")
        sys.exit(1)
    try:
        # Smoke-test the embed model with a short probe before committing to
        # embedding the whole corpus with it.
        ollama_client.embed("connectivity probe")
    except Exception as exc:
        print(f"FAIL: Ollama embed() smoke test failed for model "
              f"{OLLAMA_EMBED_MODEL!r}: {exc}. Aborting.")
        sys.exit(1)
    print("Ollama embed model reachable OK.\n")

    print("=== Embedding corpus under both models (in-memory only, no writes) ===")
    mlx_doc_vectors = embed_corpus(docs, mlx_engine.embed, "MLX")
    ollama_doc_vectors = embed_corpus(docs, ollama_client.embed, "Ollama")
    print()

    report_lines: list[str] = []
    report_lines.append("# Embedding Model Comparison — MLX (embeddinggemma) vs. Ollama (nomic-embed-text)")
    report_lines.append("")
    report_lines.append(f"Corpus: {len(docs)} documents from {DB_PATH}")
    report_lines.append(f"MLX vectors: {len(mlx_doc_vectors)}  |  Ollama vectors: {len(ollama_doc_vectors)}")
    report_lines.append("")
    report_lines.append(
        "**Observational only — no threshold conclusions drawn here.** "
        "This report exists to compare score distributions and rankings; any "
        "similarity-threshold change is a separate follow-up decision."
    )
    report_lines.append("")

    mlx_top1_scores: list[float] = []
    ollama_top1_scores: list[float] = []
    agreements = 0

    for query in TEST_QUERIES:
        print(f"Embedding query: {query!r}")
        mlx_q_vec = mlx_engine.embed(query)
        ollama_q_vec = ollama_client.embed(query)

        mlx_top = rank_top_k(mlx_q_vec, mlx_doc_vectors, docs_by_key)
        ollama_top = rank_top_k(ollama_q_vec, ollama_doc_vectors, docs_by_key)

        if mlx_top:
            mlx_top1_scores.append(mlx_top[0].score)
        if ollama_top:
            ollama_top1_scores.append(ollama_top[0].score)
        if (mlx_top and ollama_top and
                mlx_top[0].name == ollama_top[0].name and
                mlx_top[0].doc_type == ollama_top[0].doc_type):
            agreements += 1

        report_lines.append(format_side_by_side(query, mlx_top, ollama_top))

    def _stats(scores: list[float]) -> str:
        if not scores:
            return "n/a"
        return f"mean={sum(scores) / len(scores):.4f}  min={min(scores):.4f}  max={max(scores):.4f}"

    report_lines.append("## Aggregate stats across all test queries")
    report_lines.append("")
    report_lines.append(f"- MLX top-1 score:    {_stats(mlx_top1_scores)}")
    report_lines.append(f"- Ollama top-1 score: {_stats(ollama_top1_scores)}")
    report_lines.append(
        f"- Top-1 doc agreement between models: {agreements}/{len(TEST_QUERIES)} queries"
    )
    report_lines.append("")

    report = "\n".join(report_lines)
    print("\n" + "=" * 95)
    print(report)

    out_path = Path(__file__).parent / "reports" / "embedding_model_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport also written to {out_path}")


if __name__ == "__main__":
    main()
