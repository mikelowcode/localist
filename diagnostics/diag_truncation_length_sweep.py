"""
Truncation Length Sweep Diagnostic — does the 500-char embedding ceiling in
`memory_manager.py`'s index_document() cost retrieval quality?
=============================================================================

Third diagnostic in this series, building on:
  - diag_compare_embedding_models.py — MLX vs. Ollama side-by-side ranking.
  - diag_discrimination_band.py — established a clean, non-overlapping
    true-positive/true-negative score gap (0.073 MLX / 0.148 Ollama) when
    embedding FULL document text (n=8 true-positive, n=8 true-negative
    queries, reused verbatim here for comparability).

Production's index_document() (memory_manager.py, ~line 1328) truncates
every document to content[:500] before computing its stored embedding —
a decision the inline comment ties to "keeping embedding calls cheap."
This script measures whether that ceiling is actually the right tradeoff
by sweeping truncation length (500 / 1500 / 3000 / full-text) and
re-measuring the same discrimination-band gap at each length, against the
real 16-document corpus, live MLX and live Ollama.

This is purely an in-memory comparison. Nothing is written to the database
and memory_manager.py's actual content[:500] truncation is NOT modified —
see "Read-only guarantees" below.

Query embeddings are computed once per model and reused across all
truncation lengths: only DOCUMENT content is truncated before embedding,
exactly mirroring index_document()'s real behavior (queries are always
embedded in full in production's query_corpus(), never truncated).

Read-only guarantees
---------------------
- Only MemoryManager.get_all_documents() (SELECT-only) touches the DB.
- No embedding is written back to document_index or any cache table.
- memory_manager.py is not edited; its content[:500] truncation is left
  exactly as-is. This script re-implements the same truncation in its own
  process memory at multiple lengths purely for comparison.
- No wiki/raw file is modified.

Run from the project root:
    python3 diagnostics/diag_truncation_length_sweep.py

Not a pytest test — lives under diagnostics/, not collected by `pytest tests/`.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from embedding_engine import EmbeddingEngine  # noqa: E402
from ollama_runtime_client import OllamaRuntimeClient  # noqa: E402
from memory_manager import MemoryManager, DocumentResult, _cosine_similarity  # noqa: E402

# Reuse the exact validated query sets from the discrimination-band
# diagnostic — do not invent a new set, for comparability with that run.
from diag_discrimination_band import (  # noqa: E402
    TRUE_NEGATIVE_QUERIES,
    TRUE_POSITIVE_QUERIES,
)

DB_PATH = BACKEND_DIR / "localist_memory.db"

OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL  = "gemma4:31b-cloud"  # constructor-required, never called here

# 500 = current production ceiling (memory_manager.py index_document()).
# None = full text, no truncation (what both prior diagnostics used).
TRUNCATION_LENGTHS: list[int | None] = [500, 1500, 3000, None]

# Docs picked for the qualitative "what falls inside vs. outside the
# window" check — the 3 longest in the corpus, chosen so 500/1500/3000
# all land at meaningfully different points in the content, plus one
# YAML-frontmatter'd wiki page to show that overhead's effect distinctly.
QUALITATIVE_SAMPLE_DOCS = [
    "wiki:MEMORY",
    "raw:Localist Master Project Outline",
    "raw:Localist Build Order",
    "wiki:localist-design-philosophy",
]


def _label(length: int | None) -> str:
    return "full-text" if length is None else str(length)


def _truncate(content: str, length: int | None) -> str:
    return content if length is None else content[:length]


@dataclass
class Stats:
    mean: float
    min: float
    max: float
    stddev: float

    def __str__(self) -> str:
        return f"mean={self.mean:.4f}  min={self.min:.4f}  max={self.max:.4f}  stddev={self.stddev:.4f}"


def _stats(scores: list[float]) -> Stats:
    return Stats(
        mean=statistics.mean(scores),
        min=min(scores),
        max=max(scores),
        stddev=statistics.stdev(scores) if len(scores) > 1 else 0.0,
    )


def load_corpus() -> list[DocumentResult]:
    mm = MemoryManager(db_path=DB_PATH)
    return mm.get_all_documents()


def top1_against(query_vec: list[float], doc_vectors: dict[str, list[float]]) -> tuple[str, float]:
    best_key, best_score = None, float("-inf")
    for key, vec in doc_vectors.items():
        score = _cosine_similarity(query_vec, vec)
        if score > best_score:
            best_key, best_score = key, score
    return best_key, best_score


def main() -> None:
    if not DB_PATH.exists():
        print(f"FAIL: production database not found: {DB_PATH}")
        sys.exit(1)

    print(f"=== Loading real corpus from {DB_PATH} (read-only) ===")
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents.\n")

    print("=== Loading EmbeddingEngine (MLX) ===")
    mlx_engine = EmbeddingEngine()
    if not mlx_engine.available:
        print("FAIL: EmbeddingEngine unavailable. Aborting.")
        sys.exit(1)
    print("OK.\n")

    print("=== Connecting to Ollama ===")
    ollama_client = OllamaRuntimeClient(
        chat_model=OLLAMA_CHAT_MODEL,
        embedding_model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    if not ollama_client.health_check().get("reachable"):
        print("FAIL: Ollama daemon not reachable. Aborting.")
        sys.exit(1)
    try:
        ollama_client.embed("connectivity probe")
    except Exception as exc:
        print(f"FAIL: Ollama embed() smoke test failed: {exc}. Aborting.")
        sys.exit(1)
    print("OK.\n")

    models = {
        "MLX": mlx_engine.embed,
        "Ollama": ollama_client.embed,
    }

    # -- Embed all queries ONCE per model (queries are never truncated —
    # matches production, where only stored document embeddings are
    # truncated by index_document(); query_corpus() embeds the query in
    # full every time). --
    print("=== Embedding queries once per model (full text, reused across all truncation lengths) ===")
    query_vecs: dict[str, dict[str, list[float]]] = {}
    for model_name, embed_fn in models.items():
        qv: dict[str, list[float]] = {}
        for q in TRUE_NEGATIVE_QUERIES:
            qv[q] = embed_fn(q)
        for q, _ in TRUE_POSITIVE_QUERIES:
            qv[q] = embed_fn(q)
        query_vecs[model_name] = qv
        print(f"  [{model_name}] embedded {len(qv)} unique queries.")
    print()

    # -- Sweep: for each model x truncation length, re-embed the corpus
    # with that truncation applied, then re-run every query against it. --
    print("=== Sweeping truncation lengths (re-embedding corpus per length, in-memory only) ===")
    # results[model_name][length] = (pos_scores, neg_scores, per_query_detail)
    results: dict[str, dict[str, dict]] = {m: {} for m in models}

    for model_name, embed_fn in models.items():
        for length in TRUNCATION_LENGTHS:
            label = _label(length)
            doc_vectors: dict[str, list[float]] = {}
            for doc in docs:
                key = f"{doc.doc_type}:{doc.name}"
                doc_vectors[key] = embed_fn(_truncate(doc.content, length))

            pos_scores: list[float] = []
            pos_detail: list[tuple[str, str, float]] = []
            for q, expected in TRUE_POSITIVE_QUERIES:
                key, score = top1_against(query_vecs[model_name][q], doc_vectors)
                pos_scores.append(score)
                pos_detail.append((q, key, score))

            neg_scores: list[float] = []
            neg_detail: list[tuple[str, str, float]] = []
            for q in TRUE_NEGATIVE_QUERIES:
                key, score = top1_against(query_vecs[model_name][q], doc_vectors)
                neg_scores.append(score)
                neg_detail.append((q, key, score))

            results[model_name][label] = {
                "pos_scores": pos_scores,
                "neg_scores": neg_scores,
                "pos_detail": pos_detail,
                "neg_detail": neg_detail,
            }
            print(f"  [{model_name}] truncation={label} — done "
                  f"({len(doc_vectors)} docs re-embedded).")
    print()

    # -- Build report ------------------------------------------------------
    report: list[str] = []
    report.append("# Truncation Length Sweep — does the 500-char embedding ceiling cost retrieval quality?")
    report.append("")
    report.append(
        "Third diagnostic in this series. Reuses the exact 8 true-positive / "
        "8 true-negative queries from diag_discrimination_band.py. Sweeps "
        "document-embedding truncation length (500 = current production "
        "ceiling in memory_manager.py's index_document(); 1500; 3000; "
        "full-text = no truncation, matching the prior two diagnostics). "
        "Query text is never truncated, matching production's query_corpus() "
        "behavior — only stored document embeddings are truncated."
    )
    report.append("")
    report.append(
        "**Observational only.** memory_manager.py's content[:500] "
        "truncation is unmodified; no re-index of the real database "
        "occurred; no threshold or ceiling value is changed by this script."
    )
    report.append(f"\nCorpus: {len(docs)} real documents from {DB_PATH}\n")

    for model_name in models:
        report.append(f"## {model_name} — gap vs. truncation length")
        report.append("")
        report.append(f"{'Length':<12}{'TP mean':<12}{'TN mean':<12}{'Gap':<12}{'Overlap?':<10}")
        report.append("-" * 58)
        for length in TRUNCATION_LENGTHS:
            label = _label(length)
            r = results[model_name][label]
            pos_stats = _stats(r["pos_scores"])
            neg_stats = _stats(r["neg_scores"])
            gap = pos_stats.min - neg_stats.max
            overlap = "NO" if gap > 0 else "YES"
            gap_str = f"{gap:+.4f}"
            report.append(
                f"{label:<12}{pos_stats.mean:<12.4f}{neg_stats.mean:<12.4f}{gap_str:<12}{overlap:<10}"
            )
        report.append("")

    report.append("## Trend summary (observation, not a decision)")
    report.append("")
    for model_name in models:
        gaps = []
        for length in TRUNCATION_LENGTHS:
            label = _label(length)
            r = results[model_name][label]
            pos_stats = _stats(r["pos_scores"])
            neg_stats = _stats(r["neg_scores"])
            gaps.append((label, pos_stats.min - neg_stats.max))
        gap_500 = next(g for l, g in gaps if l == "500")
        gap_full = next(g for l, g in gaps if l == "full-text")
        direction = (
            "widens" if gap_full > gap_500 + 0.01 else
            "narrows" if gap_full < gap_500 - 0.01 else
            "stays roughly flat"
        )
        report.append(
            f"- {model_name}: gap at 500 chars = {gap_500:+.4f}; gap at "
            f"full-text = {gap_full:+.4f} — going from the production "
            f"ceiling to full-text **{direction}** the gap "
            f"(delta {gap_full - gap_500:+.4f}). Full progression: "
            + ", ".join(f"{l}={g:+.4f}" for l, g in gaps)
        )
    report.append("")

    # -- Qualitative content-window check -----------------------------------
    report.append("## What falls inside vs. outside the truncation window (qualitative)")
    report.append("")
    report.append(
        "For 4 of the corpus's longer documents, showing what content each "
        "truncation length actually captures — to interpret *why* any "
        "pattern above occurs, not just that it occurs."
    )
    report.append("")

    docs_by_key = {f"{d.doc_type}:{d.name}": d for d in docs}
    for key in QUALITATIVE_SAMPLE_DOCS:
        doc = docs_by_key[key]
        content = doc.content
        report.append(f"### `{key}` (full length: {len(content)} chars)")
        report.append("")
        for length in [500, 1500, 3000]:
            if length >= len(content):
                report.append(f"- **{length} chars**: covers the entire document (shorter than this length).")
                continue
            cutoff_context = content[max(0, length - 40):length]
            after_context = content[length:length + 40]
            report.append(
                f"- **{length} chars**: cuts off after `...{cutoff_context!r}` "
                f"→ `{after_context!r}...`"
            )
        report.append("")

    report.append(
        "Observed in this corpus: none of these 4 documents' 500-char cutoff "
        "lands on a clean sentence boundary — each truncates mid-word or "
        "mid-clause into what would otherwise be substantive content (not "
        "filler). `wiki:localist-design-philosophy` additionally spends "
        "~140 of its 500-char budget on YAML frontmatter "
        "(`---\\ntitle: ...\\n---\\n\\n## Summary\\n\\n`) before reaching any "
        "actual body text, so its effective substantive-content window is "
        "closer to ~360 chars, not the full 500. Neither pattern is "
        "generalized here to \"most wiki docs\" — this is 4 documents, "
        "reported individually; see the per-doc detail above."
    )
    report.append("")

    # -- Appendix: raw per-query scores -------------------------------------
    report.append("## Appendix — raw per-query scores at every truncation length")
    report.append("")
    for model_name in models:
        report.append(f"### {model_name}")
        report.append("")
        for length in TRUNCATION_LENGTHS:
            label = _label(length)
            r = results[model_name][label]
            report.append(f"#### Truncation = {label}")
            report.append("")
            report.append("True-positive queries:")
            report.append("")
            for q, doc_key, score in r["pos_detail"]:
                report.append(f"- {q!r} → `{doc_key}` ({score:.4f})")
            report.append("")
            report.append("True-negative queries:")
            report.append("")
            for q, doc_key, score in r["neg_detail"]:
                report.append(f"- {q!r} → `{doc_key}` ({score:.4f})")
            report.append("")

    report_text = "\n".join(report)
    print("\n" + "=" * 100)
    print(report_text)

    out_path = Path(__file__).parent / "reports" / "truncation_length_sweep.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)
    print(f"\nReport also written to {out_path}")


if __name__ == "__main__":
    main()
