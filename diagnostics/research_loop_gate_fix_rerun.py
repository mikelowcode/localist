"""
diagnostics/research_loop_gate_fix_rerun.py

READ-ONLY DIAGNOSTIC — does not modify any source file, database, or
episodic memory. Before/after companion to research_loop_qa_pass.py: reuses
its exact direct-dispatcher-call pattern and helper functions
(_run_query, _classify_outcome, _extract_query_text) — no logic duplicated
— to re-run the 6 queries diagnostics/reports/
research_loop_qa_assessment_2026-07-20.md flagged as the clearest false
positives, against the relevance-aware _evaluate_pricing_gate() fix
(mcp_tool_dispatcher.py, 2026-07-20).

"Before" data for each query is read directly from
diagnostics/research_loop_qa_2026-07-20.csv (the original pass's raw
output), not re-derived from the report's prose, so the before/after delta
table is accurate per query.

Usage
-----
    cd diagnostics/
    python research_loop_gate_fix_rerun.py

Output
------
  diagnostics/research_loop_gate_fix_2026-07-20.csv  (raw re-run data)
  diagnostics/reports/research_loop_gate_fix_delta_2026-07-20.md
  stdout — per-query progress
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_loop_qa_pass import (  # noqa: E402
    _CORPUS,
    _FIELDNAMES,
    _run_query,
)
from ollama_runtime_client import OllamaRuntimeClient  # noqa: E402
from mcp_tool_dispatcher import MCPToolDispatcher  # noqa: E402

_BEFORE_CSV = Path(__file__).resolve().parent / "research_loop_qa_2026-07-20.csv"
_AFTER_CSV = Path(__file__).resolve().parent / "research_loop_gate_fix_2026-07-20.csv"
_REPORT_PATH = (
    Path(__file__).resolve().parent
    / "reports"
    / "research_loop_gate_fix_delta_2026-07-20.md"
)

_FLAGGED_NUMS = {2, 11, 12, 14, 15, 17}


def _load_before_rows() -> dict[int, dict]:
    with _BEFORE_CSV.open(encoding="utf-8") as f:
        rows = {int(r["num"]): r for r in csv.DictReader(f)}
    missing = _FLAGGED_NUMS - rows.keys()
    if missing:
        raise RuntimeError(f"Original CSV missing expected query numbers: {missing}")
    return {n: rows[n] for n in _FLAGGED_NUMS}


def main() -> None:
    backend = os.environ.get("LOCALIST_RUNTIME_BACKEND", "")
    chat_model = (
        os.environ.get("LOCALIST_CHAT_MODEL_OLLAMA")
        or os.environ.get("LOCALIST_CHAT_MODEL", "")
    )
    print(f"Configured runtime backend: {backend!r}  chat_model: {chat_model!r}")
    if backend.strip().lower() != "ollama" or "31b" not in chat_model:
        raise RuntimeError(
            f"Expected ollama/gemma4:31b-cloud, got backend={backend!r} "
            f"chat_model={chat_model!r}. Confirm backend/.env before running."
        )

    runtime = OllamaRuntimeClient(chat_model=chat_model)
    health = runtime.health_check()
    if not health.get("reachable") or not health.get("chat_model_found"):
        raise RuntimeError(f"Ollama not ready: {health}")
    print(f"Ollama reachable — chat_model_found={health['chat_model_found']}\n")

    before_rows = _load_before_rows()
    dispatcher = MCPToolDispatcher(runtime=runtime)

    corpus_by_num = {num: (num, cat, q) for num, cat, q in _CORPUS}

    after_rows: list[dict] = []
    for num in sorted(_FLAGGED_NUMS):
        _, category, query = corpus_by_num[num]
        print(f"[{num:2d}] ({category}) {query}")
        row = _run_query(dispatcher, num, category, query)
        after_rows.append(row)
        print(
            f"     outcome={row['outcome']:24s} "
            f"iterations={row['iteration_count']} "
            f"fetch={row['fetch_occurred']} "
            f"reformulated={row['reformulated']} "
            f"elapsed={row['elapsed_seconds']}s"
        )

    with _AFTER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(after_rows)
    print(f"\nRaw re-run data written to: {_AFTER_CSV}")

    _write_delta_report(before_rows, {r["num"]: r for r in after_rows})
    print(f"Delta report written to: {_REPORT_PATH}")


def _write_delta_report(before: dict[int, dict], after: dict[int, dict]) -> None:
    lines: list[str] = []
    lines.append("# Research Loop Gate Fix — Before/After Delta, 2026-07-20\n")
    lines.append("**Fix:** `_evaluate_pricing_gate()` (`backend/mcp_tool_dispatcher.py`) "
                  "now takes `instruction` alongside `text` and judges whether the text "
                  "specifically answers the question asked, not just whether "
                  "pricing/spec-shaped content is present anywhere. See "
                  "`diagnostics/reports/research_loop_qa_assessment_2026-07-20.md` for "
                  "the false-positive pattern this targets.\n")
    lines.append("**Runtime:** backend=`ollama` chat_model=`gemma4:31b-cloud` (LIVE, "
                  "no mocks) — confirmed before running, same check as the original pass.\n")
    lines.append("**Before data source:** `diagnostics/research_loop_qa_2026-07-20.csv` "
                  "(original pass, read directly, not re-derived from the report's prose).\n")
    lines.append("**After data source:** `diagnostics/research_loop_gate_fix_2026-07-20.csv` "
                  "(this re-run).\n")
    lines.append("## Per-Query Delta\n")

    for num in sorted(_FLAGGED_NUMS):
        b = before[num]
        a = after[num]
        lines.append(f"### Query {num} — {b['query']}\n")
        lines.append(f"| | Before | After |")
        lines.append(f"|---|---|---|")
        lines.append(f"| Outcome | `{b['outcome']}` | `{a['outcome']}` |")
        lines.append(f"| Iterations | {b['iteration_count']} | {a['iteration_count']} |")
        lines.append(f"| Fetch occurred | {b['fetch_occurred']} | {a['fetch_occurred']} |")
        lines.append(f"| Reformulated | {b['reformulated']} | {a['reformulated']} |")
        lines.append(f"| Elapsed | {b['elapsed_seconds']}s | {a['elapsed_seconds']}s |")
        lines.append("")
        lines.append("**Before answer:**")
        lines.append("```")
        lines.append(b["final_answer"][:1000])
        lines.append("```")
        lines.append("**After answer:**")
        lines.append("```")
        lines.append(a["final_answer"][:1000])
        lines.append("```")
        lines.append("")
        lines.append("**Read:** _TODO — filled in by hand after reviewing the raw text above._\n")

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
