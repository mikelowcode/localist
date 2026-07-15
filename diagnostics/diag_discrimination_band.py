"""
Discrimination Band Diagnostic — true-negative vs. true-positive top-1 score
spread, MLX (embeddinggemma) vs. Ollama (nomic-embed-text)
=============================================================================

Companion to `diag_compare_embedding_models.py`. That script's first run
surfaced two findings; this script investigates ONLY the first of them:

  Finding 1 — a single true-negative query ("chocolate chip cookies") scored
  0.39-0.41 top-1 against the real 16-doc corpus, while a genuinely relevant
  but broad query ("What is this project about?") scored only 0.55. A ~0.15
  gap between "irrelevant" and "relevant" is narrow, and it rested on one
  true-negative sample.

(Finding 2 — wiki/raw corpus duplication — is investigated separately, by
reading memory_manager.py / wiki_agent.py and diffing on-disk file pairs.
See diagnostics/reports/discrimination_band_and_duplication.md for both
write-ups. This script produces only the Finding-1 numbers.)

This script does NOT re-litigate the original 0.028 "miss" — that is
considered closed/noise per the prior session and is out of scope here.

What this script does
----------------------
1. Loads the real indexed corpus (read-only, via MemoryManager.get_all_documents()).
2. Embeds a widened true-negative query set (8 queries spanning cooking,
   sports, history, small talk, an unrelated well-known software product,
   math, travel, biology) and a widened true-positive query set (8 queries,
   each written to unambiguously target one specific known corpus document).
3. Computes top-1-score mean/min/max/stddev for each set, per model, and
   reports the gap (or overlap) between the two distributions.
4. Adds 3 topically unrelated "control" documents (Lorem Ipsum, a generic
   cooking-recipe paragraph, a generic astronomy paragraph) purely in-memory
   and re-scores the true-negative queries against them, to distinguish
   "corpus is thematically narrow" from "model score calibration is coarse."

Read-only guarantees — identical to diag_compare_embedding_models.py:
- Only MemoryManager.get_all_documents() (SELECT-only) touches the DB.
- No embedding is written back anywhere; no wiki/raw file is modified.
- No production module is edited; no similarity threshold is touched.

Run from the project root:
    python3 diagnostics/diag_discrimination_band.py

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

DB_PATH = BACKEND_DIR / "localist_memory.db"

OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL  = "gemma4:31b-cloud"  # constructor-required, never called here

# ---------------------------------------------------------------------------
# Widened query sets
# ---------------------------------------------------------------------------

# Each targets a different unrelated-to-Localist domain.
TRUE_NEGATIVE_QUERIES: list[str] = [
    "What's a good recipe for chocolate chip cookies?",          # cooking (original)
    "Who won the most recent Super Bowl?",                        # sports
    "What caused the fall of the Roman Empire?",                  # history
    "How's the weather looking for the weekend?",                 # small talk
    "How do I create a pivot table in Microsoft Excel?",          # unrelated well-known software
    "What is the derivative of x squared?",                       # math
    "What are the best beaches to visit in Thailand?",            # travel
    "How does photosynthesis work in plants?",                    # biology
]

# Each written to unambiguously target ONE specific known corpus document
# (not broad/vague — contrast with "What is this project about?" from the
# original run).
TRUE_POSITIVE_QUERIES: list[tuple[str, str]] = [
    ("What are the sequential build phases for developing Localist?",
     "localist-build-order"),
    ("What are the five core design pillars of the Localist design philosophy?",
     "localist-design-philosophy"),
    ("What tools does LORA have access to, like web search and file operations?",
     "how-localist-works"),  # / lora-persona also plausible target
    ("Where does Michael live?",
     "michael"),
    ("What inference engines does the Localist Runtime Backend Layer support?",
     "localist-runtime-tooling-update"),
    ("What is the high-level vision and roadmap for the Localist project?",
     "localist-master-project-outline"),
    ("What hardware and models make up the Localist software stack?",
     "localist-software-stack"),
    ("What is the MEMORY.md human-readable snapshot?",
     "MEMORY"),
]

# Topically unrelated control documents — no relation to Localist/LORA/AI-
# assistant development whatsoever.
CONTROL_DOCS: dict[str, str] = {
    "control/lorem-ipsum": (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim "
        "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut "
        "aliquip ex ea commodo consequat. Duis aute irure dolor in "
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat "
        "nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
        "sunt in culpa qui officia deserunt mollit anim id est laborum."
    ),
    "control/cookie-recipe": (
        "Preheat the oven to 375 degrees Fahrenheit. In a large bowl, cream "
        "together softened butter, white sugar, and brown sugar until light "
        "and fluffy. Beat in eggs one at a time, then stir in vanilla "
        "extract. Combine flour, baking soda, and salt in a separate bowl; "
        "gradually blend into the butter mixture. Fold in chocolate chips. "
        "Drop rounded spoonfuls onto ungreased baking sheets and bake for "
        "nine to eleven minutes until golden brown around the edges."
    ),
    "control/astronomy": (
        "Jupiter is the largest planet in the solar system, a gas giant "
        "composed primarily of hydrogen and helium. Its Great Red Spot is a "
        "giant storm that has raged for at least 350 years and is large "
        "enough to swallow the Earth several times over. Jupiter has dozens "
        "of moons, the largest of which, Ganymede, is bigger than the "
        "planet Mercury. Its powerful magnetic field traps charged "
        "particles, producing intense radiation belts and vivid auroras."
    ),
}


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


def _overlap_report(neg: Stats, pos: Stats) -> str:
    gap = pos.min - neg.max
    if gap > 0:
        return f"NO OVERLAP — clean gap of {gap:.4f} between neg.max and pos.min."
    return (
        f"OVERLAP of {-gap:.4f} — neg.max ({neg.max:.4f}) exceeds pos.min "
        f"({pos.min:.4f})."
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
    docs_by_key = {f"{d.doc_type}:{d.name}": d for d in docs}
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

    print("=== Embedding real corpus (both models, in-memory only) ===")
    mlx_doc_vectors = {f"{d.doc_type}:{d.name}": mlx_engine.embed(d.content) for d in docs}
    ollama_doc_vectors = {f"{d.doc_type}:{d.name}": ollama_client.embed(d.content) for d in docs}
    print(f"MLX: {len(mlx_doc_vectors)} vectors  |  Ollama: {len(ollama_doc_vectors)} vectors\n")

    print("=== Embedding control documents (both models, in-memory only) ===")
    mlx_control_vectors = {k: mlx_engine.embed(v) for k, v in CONTROL_DOCS.items()}
    ollama_control_vectors = {k: ollama_client.embed(v) for k, v in CONTROL_DOCS.items()}
    print(f"{len(CONTROL_DOCS)} control docs embedded under both models.\n")

    report: list[str] = []
    report.append("# Discrimination Band Diagnostic — Finding 1 only")
    report.append("")
    report.append(
        "Companion to diag_compare_embedding_models.py. Investigates ONLY the "
        "narrow true-negative/true-positive score band; the 0.028 corpus-miss "
        "is out of scope and considered closed. Corpus duplication (Finding 2) "
        "is written up separately — see the accompanying prose report, not "
        "this script."
    )
    report.append("")
    report.append(f"Corpus: {len(docs)} real documents from {DB_PATH}")
    report.append("")

    # -- True-negative pass ---------------------------------------------
    report.append("## True-negative queries (8) — top-1 score against real corpus")
    report.append("")
    report.append(f"{'Query':<55}{'MLX doc / score':<40}{'Ollama doc / score':<40}")
    report.append("-" * 135)

    mlx_neg_scores: list[float] = []
    ollama_neg_scores: list[float] = []
    for q in TRUE_NEGATIVE_QUERIES:
        mlx_vec = mlx_engine.embed(q)
        ollama_vec = ollama_client.embed(q)
        mlx_key, mlx_score = top1_against(mlx_vec, mlx_doc_vectors)
        ollama_key, ollama_score = top1_against(ollama_vec, ollama_doc_vectors)
        mlx_neg_scores.append(mlx_score)
        ollama_neg_scores.append(ollama_score)
        report.append(
            f"{q[:53]:<55}{mlx_key + ' (' + f'{mlx_score:.4f}' + ')':<40}"
            f"{ollama_key + ' (' + f'{ollama_score:.4f}' + ')':<40}"
        )
    report.append("")

    # -- True-positive pass -----------------------------------------------
    report.append("## True-positive queries (8) — top-1 score against real corpus")
    report.append("")
    report.append(f"{'Query':<55}{'MLX doc / score':<40}{'Ollama doc / score':<40}")
    report.append("-" * 135)

    mlx_pos_scores: list[float] = []
    ollama_pos_scores: list[float] = []
    mlx_pos_hits = 0
    ollama_pos_hits = 0
    for q, expected_name in TRUE_POSITIVE_QUERIES:
        mlx_vec = mlx_engine.embed(q)
        ollama_vec = ollama_client.embed(q)
        mlx_key, mlx_score = top1_against(mlx_vec, mlx_doc_vectors)
        ollama_key, ollama_score = top1_against(ollama_vec, ollama_doc_vectors)
        mlx_pos_scores.append(mlx_score)
        ollama_pos_scores.append(ollama_score)
        if expected_name in mlx_key:
            mlx_pos_hits += 1
        if expected_name in ollama_key:
            ollama_pos_hits += 1
        report.append(
            f"{q[:53]:<55}{mlx_key + ' (' + f'{mlx_score:.4f}' + ')':<40}"
            f"{ollama_key + ' (' + f'{ollama_score:.4f}' + ')':<40}"
        )
    report.append("")
    report.append(
        f"Expected-doc top-1 hit rate — MLX: {mlx_pos_hits}/{len(TRUE_POSITIVE_QUERIES)}  "
        f"Ollama: {ollama_pos_hits}/{len(TRUE_POSITIVE_QUERIES)} "
        "(informal check that these queries do target an unambiguous doc; "
        "not the object of this diagnostic)."
    )
    report.append("")

    # -- Distribution stats ------------------------------------------------
    mlx_neg_stats = _stats(mlx_neg_scores)
    mlx_pos_stats = _stats(mlx_pos_scores)
    ollama_neg_stats = _stats(ollama_neg_scores)
    ollama_pos_stats = _stats(ollama_pos_scores)

    report.append("## Distribution stats — real corpus")
    report.append("")
    report.append(f"- MLX true-negative:    {mlx_neg_stats}")
    report.append(f"- MLX true-positive:    {mlx_pos_stats}")
    report.append(f"- MLX gap: {_overlap_report(mlx_neg_stats, mlx_pos_stats)}")
    report.append("")
    report.append(f"- Ollama true-negative: {ollama_neg_stats}")
    report.append(f"- Ollama true-positive: {ollama_pos_stats}")
    report.append(f"- Ollama gap: {_overlap_report(ollama_neg_stats, ollama_pos_stats)}")
    report.append("")

    # -- Control-document pass ---------------------------------------------
    report.append("## True-negative queries vs. control documents (corpus-homogeneity check)")
    report.append("")
    report.append(
        "If scores against these topically unrelated control docs are "
        "similarly 'moderate' to scores against the real corpus, that points "
        "to model score-calibration (the model just doesn't produce very low "
        "cosine scores for short natural-language queries against any "
        "prose). If scores drop noticeably lower here, that points to "
        "corpus homogeneity instead (the real corpus's moderate negative "
        "scores are because all 16 docs share Localist/AI-assistant "
        "vocabulary, not because the model can't discriminate)."
    )
    report.append("")
    report.append(f"{'Query':<55}{'MLX control top-1':<40}{'Ollama control top-1':<40}")
    report.append("-" * 135)

    mlx_control_scores: list[float] = []
    ollama_control_scores: list[float] = []
    for q in TRUE_NEGATIVE_QUERIES:
        mlx_vec = mlx_engine.embed(q)
        ollama_vec = ollama_client.embed(q)
        mlx_key, mlx_score = top1_against(mlx_vec, mlx_control_vectors)
        ollama_key, ollama_score = top1_against(ollama_vec, ollama_control_vectors)
        mlx_control_scores.append(mlx_score)
        ollama_control_scores.append(ollama_score)
        report.append(
            f"{q[:53]:<55}{mlx_key + ' (' + f'{mlx_score:.4f}' + ')':<40}"
            f"{ollama_key + ' (' + f'{ollama_score:.4f}' + ')':<40}"
        )
    report.append("")

    mlx_control_stats = _stats(mlx_control_scores)
    ollama_control_stats = _stats(ollama_control_scores)
    report.append("## Control-document stats vs. real-corpus true-negative stats")
    report.append("")
    report.append(f"- MLX    true-neg vs. real corpus:   {mlx_neg_stats}")
    report.append(f"- MLX    true-neg vs. control docs:  {mlx_control_stats}")
    mlx_drop = mlx_neg_stats.mean - mlx_control_stats.mean
    report.append(f"- MLX    mean drop (real - control): {mlx_drop:+.4f}")
    report.append("")
    report.append(f"- Ollama true-neg vs. real corpus:   {ollama_neg_stats}")
    report.append(f"- Ollama true-neg vs. control docs:  {ollama_control_stats}")
    ollama_drop = ollama_neg_stats.mean - ollama_control_stats.mean
    report.append(f"- Ollama mean drop (real - control): {ollama_drop:+.4f}")
    report.append("")
    report.append(
        "**Read this section, not a hardcoded conclusion below** — a large "
        "positive drop (scores meaningfully lower against control docs than "
        "against the real corpus) supports the corpus-homogeneity hypothesis; "
        "a near-zero or negative drop supports the model-calibration "
        "hypothesis. This script does not decide between them."
    )
    report.append("")
    report.append(
        "**Observational only.** No threshold change, no corpus change, no "
        "code change is proposed by this script."
    )

    report_text = "\n".join(report)
    print("\n" + "=" * 135)
    print(report_text)

    out_path = Path(__file__).parent / "reports" / "discrimination_band.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text)
    print(f"\nReport also written to {out_path}")


if __name__ == "__main__":
    main()
