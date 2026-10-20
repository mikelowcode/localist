"""
Diagnostic: diag_rag_cutoff.py
===============================
Part 2, Phase 2A of the 2026-10-20 memory-check / RAG plan: what document
cutoff does each embedding model need, in place of the fixed 0.55?

The cutoff controls two decisions (both cosine-scored when the documents have
embeddings for the active model):
  routing  Planner._priority4_corpus Path B: query_corpus(q, max_results=1)
           top score >= cutoff -> fetch_rag
  sources  ControllerAgent Step 4 / Step 3b: of query_corpus(q, max_results=3),
           keep documents scoring >= cutoff as Slot 4 sources
This measures both through the real code: documents are indexed into a
temporary MemoryManager with the model's live embeddings
(index_document(): first 500 characters, front matter included) and queried
with the real query_corpus() (BM25 prefilter, then cosine re-rank of the top
2 x max_results).

Battery: threshold_calibration_fixtures.RAG_* (synthetic). The cutoff is
chosen like the episodic recall floor (_recall_floor_from_scores' objective:
maximise sources recall - FP over 0.10-0.90, ties to the highest) on the
calibration part (heldout_split()) and scored on the held-out part.
  sources recall  a right document is among the top 3 and >= cutoff
  sources FP      a negative gets any document >= cutoff
  wrong           non-answering documents kept per positive
  routing         positive routed to RAG / negative routed to RAG

Optional real check (--real-wiki, --real-raw, --real-questions,
--real-negatives): COPIES of the user's wiki/ and raw/ (persona, index, logs,
MEMORY and user-profile files are skipped, as production excludes them),
questions labelled with the file names that answer them, and requests that
need none. Personal data: keep those outside the repo; only aggregates print.

READ-ONLY. Writes only to a temporary directory. Calls the live Ollama
embedding endpoint through the real runtime client. Not part of the pytest
suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_rag_cutoff.py [--real-wiki DIR --real-raw DIR \
        --real-questions Q.json --real-negatives N.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import tempfile

from localist.memory_manager import MemoryManager
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.threshold_calibration_fixtures import RAG_DOCS, RAG_NEGATIVES, RAG_POSITIVES, heldout_split

MODELS = ["nomic-embed-text:latest", "embeddinggemma:latest", "embeddinggemma-2:latest"]
TODAY = 0.55
GRID = [x / 100 for x in range(10, 91)]
SKIP = {"persona.md", "index.md", "logs.md", "MEMORY.md", "michael.md"}


def _index(mm: MemoryManager, docs: dict[str, tuple[str, str]], root: pathlib.Path) -> dict[str, str]:
    """Index docs; return {resolved path: key}. A key may carry a "wiki/" or
    "raw/" prefix (real files, where one name can exist in both folders)."""
    paths = {}
    for key, (doc_type, content) in docs.items():
        p = (root / doc_type / f"{key.split('/')[-1]}.md").resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        mm.index_document(path=p, doc_type=doc_type, content=content)
        paths[str(p)] = key
    return paths


def _results(mm: MemoryManager, paths: dict[str, str], query: str):
    """(routing top score, [(score, key)] top 3) through the real query_corpus()."""
    top1 = mm.query_corpus(query, max_results=1, use_embeddings=True)
    top3 = mm.query_corpus(query, max_results=3, use_embeddings=True)
    route = top1[0].relevance_score if top1 and top1[0].scored_by_embedding else 0.0
    return route, [(d.relevance_score, paths.get(str(d.path), str(d.path))) for d in top3 if d.scored_by_embedding]


def _metrics(pos, neg, t):
    """pos: [(route, top3, answer_keys)], neg: [(route, top3)]."""
    rec = sum(any(k in ans and s >= t for s, k in top3) for _, top3, ans in pos) / len(pos)
    fp = sum(any(s >= t for s, _ in top3) for _, top3 in neg) / len(neg)
    wrong = sum(sum(1 for s, k in top3 if s >= t and k not in ans) for _, top3, ans in pos) / len(pos)
    r_tp = sum(r >= t for r, _, _ in pos) / len(pos)
    r_fp = sum(r >= t for r, _ in neg) / len(neg)
    return rec, fp, wrong, r_tp, r_fp


def _choose(pos, neg) -> tuple[float | None, float]:
    best = (None, -1.0)
    for t in GRID:
        rec, fp, *_ = _metrics(pos, neg, t)
        if rec - fp >= best[1]:
            best = (t, rec - fp)
    return best if best[1] >= 0.5 else (None, best[1])


def _fmt(m) -> str:
    rec, fp, wrong, r_tp, r_fp = m
    return (f"sources {rec:4.0%} recall / {fp:4.0%} FP / {wrong:.1f} wrong;  "
            f"routing {r_tp:4.0%} / {r_fp:4.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-wiki", type=pathlib.Path)
    ap.add_argument("--real-raw", type=pathlib.Path)
    ap.add_argument("--real-questions", type=pathlib.Path, help='[{"text", "docs": [file names]}]')
    ap.add_argument("--real-negatives", type=pathlib.Path, help='[{"text"}]')
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    (pos_cal, pos_ho), (neg_cal, neg_ho) = heldout_split(RAG_POSITIVES), heldout_split(RAG_NEGATIVES)
    real_docs = None
    if args.real_wiki and args.real_raw and args.real_questions and args.real_negatives:
        real_docs = {}
        for d, doc_type in ((args.real_wiki, "wiki"), (args.real_raw, "raw")):
            for f in sorted(d.glob("*.md")):
                if f.name not in SKIP:
                    real_docs[f"{doc_type}:{f.name}"] = (doc_type, f.read_text())
        real_q = json.loads(args.real_questions.read_text())
        real_n = json.loads(args.real_negatives.read_text())

    print(f"battery: {len(RAG_DOCS)} documents, {len(RAG_POSITIVES)} positives / {len(RAG_NEGATIVES)} "
          f"negatives (held-out {len(pos_ho)} / {len(neg_ho)})"
          + (f"; real: {len(real_docs)} documents, {len(real_q)} questions / {len(real_n)} requests needing none"
             if real_docs else ""))

    for model in MODELS:
        embed = OllamaRuntimeClient(chat_model="unused", embedding_model=model).embed
        try:
            embed("probe")
        except Exception as exc:
            print(f"\n{model}: skipped ({type(exc).__name__})", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            mm = MemoryManager(db_path=root / "rag.db", embed_fn=embed, embedding_model_name=model)
            paths = _index(mm, RAG_DOCS, root)
            run_p = lambda items: [(*_results(mm, paths, q), set(ans)) for q, ans in items]
            run_n = lambda items: [_results(mm, paths, q) for q in items]
            cal_p, ho_p, cal_n, ho_n = run_p(pos_cal), run_p(pos_ho), run_n(neg_cal), run_n(neg_ho)
            t, j = _choose(cal_p, cal_n)
            right = [s for _, top3, ans in cal_p + ho_p for s, k in top3 if k in ans]
            off = [top3[0][0] for _, top3 in cal_n + ho_n if top3]
            print(f"\n{model}")
            print(f"  right-document scores {min(right):.3f}–{max(right):.3f}; "
                  f"off-topic best {min(off):.3f}–{max(off):.3f}")
            print(f"  today 0.55        held-out: {_fmt(_metrics(ho_p, ho_n, TODAY))}")
            if t is None:
                print(f"  calibrated        inconclusive (best J {j:.2f})")
            else:
                print(f"  calibrated {t:.2f}   held-out: {_fmt(_metrics(ho_p, ho_n, t))}")

        if real_docs:
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                mm = MemoryManager(db_path=root / "real.db", embed_fn=embed, embedding_model_name=model)
                # keys "wiki/<stem>" / "raw/<stem>", as the labels use
                keyed = {k.split(":", 1)[0] + "/" + k.split(":", 1)[1].removesuffix(".md"): v
                         for k, v in real_docs.items()}
                paths = _index(mm, keyed, root)
                rp = [(*_results(mm, paths, q["text"]), set(q["docs"])) for q in real_q]
                rn = [_results(mm, paths, n["text"]) for n in real_n]
                for label, cut in (("today 0.55", TODAY), (f"calibrated {t:.2f}" if t else None, t)):
                    if cut is not None:
                        print(f"  real at {label:16} {_fmt(_metrics(rp, rn, cut))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
