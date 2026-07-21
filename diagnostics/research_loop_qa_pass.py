"""
diagnostics/research_loop_qa_pass.py

READ-ONLY DIAGNOSTIC — does not modify any source file, database, or
episodic memory. Closes the open item in
docs/architecture/18-research-loop.md §18.8: "No live human QA of the
research loop's actual answer quality... beyond the specific queries
exercised during this session's live testing."

Corpus: diagnostics/research_loop_qa_corpus.md (18 queries across 7
categories A-G, each probing a specific failure mode — read that file
for the full rationale per category before reading results here).

Isolation constraint — calls MCPToolDispatcher.dispatch() directly
------------------------------------------------------------------
Each corpus query is dispatched as tools_to_call=["research"] straight to
MCPToolDispatcher — Planner and the HTTP layer are never involved. This is
a deliberate scoping choice, not an oversight: routing (does "/research"
actually reach the loop, does it bypass LOCALIST_RESEARCH_LOOP_ENABLED) is
already covered by the prior session's slash-command Planner tests. What
this script measures is the loop's own mechanical behavior — iteration
count, fetch-on-inconclusive-snippet, reformulation, and the gate's
pass/exhaust decision.

METHODOLOGICAL CAVEAT (read before interpreting results, especially
Category D/E/G): because ConversationalAgent's answer-synthesis step is
never invoked here, "the final answer" recorded per query is the raw
winning ToolResult's .result text (a search snippet, a fetched page's
extracted text, or the synthetic exhaustion message) — NOT a model's
synthesized natural-language reply. A synthesized answer's tone (does it
hedge on ambiguity, does it invent a number) is a downstream concern this
script cannot observe. What this script CAN observe, and what the
per-category judgments below are actually grounded in: whether the raw
material the loop hands upstream already contains multiple tiers/prices
(Category D), whether the gate correctly refuses to pass on content that
doesn't actually contain the requested fact rather than pass on a
document snippet that just happens to be dollar-amount-shaped (Category
E/F), and whether the loop exhausts honestly or gate-passes on marginal
content when asked something search cannot answer (Category G).

Corpus queries are stored below with their leading "/research " token
already stripped — that token is a Planner-level slash-command concern
(§ Planner._priority0_slash_command), not part of the query text itself,
and this script bypasses Planner entirely.

Usage
-----
    cd diagnostics/
    python research_loop_qa_pass.py

Requires (checked at startup, not assumed):
  - LOCALIST_RUNTIME_BACKEND=ollama, LOCALIST_CHAT_MODEL_OLLAMA (or
    LOCALIST_CHAT_MODEL) resolving to a cloud Gemma 4-31B model, reachable
    at http://localhost:11434.
  - localist-mcp reachable at http://localhost:8003 (real Brave/LangSearch
    web_search + fetch_url — no mocks; this is a live network diagnostic).

Output
------
  diagnostics/research_loop_qa_2026-07-20.csv
  diagnostics/reports/research_loop_qa_assessment_2026-07-20.md
  stdout — per-query progress and a final per-category summary table
"""

from __future__ import annotations

import ast
import csv
import os
import re
import sys
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_BACKEND_DIR / ".env")

from ollama_runtime_client import OllamaRuntimeClient  # noqa: E402
from mcp_tool_dispatcher import MCPToolDispatcher  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "research_loop_qa_2026-07-20.csv"

# ---------------------------------------------------------------------------
# Corpus — diagnostics/research_loop_qa_corpus.md, "/research " prefix
# stripped (see module docstring for why).
# ---------------------------------------------------------------------------

_CORPUS: list[tuple[int, str, str]] = [
    (1,  "A", "What is the monthly price of Netflix's Standard plan?"),
    (2,  "A", "What does GitHub Copilot Individual cost per month?"),
    (3,  "A", "What is the base price of a 2024 Honda Civic LX?"),
    (4,  "B", "What is the storage price per GB for Backblaze B2?"),
    (5,  "B", "What's the price of the entry-level Tesla Model 3 in the US?"),
    (6,  "B", "What does the Notion Business plan cost per user per month?"),
    (7,  "C", "How much does it cost to run a t3.medium EC2 instance in us-east-1?"),
    (8,  "C", "What's the per-seat price for Figma's Enterprise tier?"),
    (9,  "C", "What does a Ford F-150 XLT crew cab start at?"),
    (10, "D", "What does Adobe Creative Cloud cost?"),
    (11, "D", "What's the price of an iPhone 16?"),
    (12, "E", "What is the exact enterprise contract price for Salesforce?"),
    (13, "E", "What does the discontinued Google Stadia controller cost new from Google?"),
    (14, "E", "What is the current spot price of a specific obscure NFT collection's floor price?"),
    (15, "F", "What is the max payload capacity of a Ford F-150 Lightning?"),
    (16, "F", "What is the battery capacity in kWh of a Tesla Model Y Long Range?"),
    (17, "G", "Do you think the new iPhone is worth the price?"),
    (18, "G", "Is the Tesla Model 3 overpriced compared to competitors?"),
]

_CATEGORY_LABELS = {
    "A": "single-iteration easy",
    "B": "needs a page fetch",
    "C": "needs reformulation",
    "D": "ambiguous / multiple tiers",
    "E": "should fail honestly",
    "F": "spec lookups, not pricing",
    "G": "negative-filter-adjacent (subjective)",
}

_QUERY_PARAM_RE = re.compile(r"^query=(.*)$")


def _extract_query_text(parameters: str) -> str | None:
    """Recover the literal query string from a web_search ToolResult's
    parameters field (f"query={query!r}" — see mcp_tool_dispatcher.py's
    _execute_web_search_query)."""
    m = _QUERY_PARAM_RE.match(parameters)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


def _classify_outcome(results: list) -> str:
    """Classify the loop's outcome purely from the returned ToolResult
    list's shape — see _run_research_loop's docstring (mcp_tool_dispatcher.py)
    for the three distinguishing shapes this mirrors."""
    if not results:
        return "no_results"
    last = results[-1]
    if last.tool_name == "research" and not last.success:
        return "exhausted_honestly"
    if last.tool_name == "url_fetch" and last.success:
        return "gate_passed_after_fetch"
    if last.tool_name == "web_search" and last.success:
        return "gate_passed_on_snippet"
    if not last.success:
        return "connectivity_failure"
    return "unclassified"


def _run_query(dispatcher: MCPToolDispatcher, num: int, category: str, query: str) -> dict:
    start = time.perf_counter()
    results = dispatcher.dispatch(["research"], query, context={})
    elapsed = time.perf_counter() - start

    web_search_results = [r for r in results if r.tool_name == "web_search"]
    fetch_results = [r for r in results if r.tool_name == "url_fetch"]

    queries_tried: list[str] = []
    for r in web_search_results:
        q = _extract_query_text(r.parameters)
        if q is not None:
            queries_tried.append(q)

    outcome = _classify_outcome(results)
    winning = results[-1] if results else None

    row = {
        "num":              num,
        "category":         category,
        "query":            query,
        "iteration_count":  len(web_search_results),
        "fetch_occurred":   bool(fetch_results),
        "reformulated":     len(web_search_results) > 1,
        "queries_tried":    " | ".join(queries_tried),
        "outcome":          outcome,
        "elapsed_seconds":  round(elapsed, 2),
        "final_answer":     winning.result if winning else "",
        "final_success":    winning.success if winning else False,
    }
    return row


_FIELDNAMES = [
    "num", "category", "query", "iteration_count", "fetch_occurred",
    "reformulated", "queries_tried", "outcome", "elapsed_seconds",
    "final_success", "final_answer",
]


def main() -> None:
    backend = os.environ.get("LOCALIST_RUNTIME_BACKEND", "")
    chat_model = os.environ.get("LOCALIST_CHAT_MODEL_OLLAMA") or os.environ.get("LOCALIST_CHAT_MODEL", "")
    print(f"Configured runtime backend: {backend!r}  chat_model: {chat_model!r}")
    if backend.strip().lower() != "ollama":
        raise RuntimeError(
            f"LOCALIST_RUNTIME_BACKEND={backend!r}, expected 'ollama' for this "
            f"QA pass (corpus author's intent: Ollama Cloud, Gemma 4-31B). "
            f"Switch backend/.env before running."
        )
    if "31b" not in chat_model:
        raise RuntimeError(
            f"LOCALIST_CHAT_MODEL_OLLAMA/LOCALIST_CHAT_MODEL={chat_model!r} does "
            f"not look like the intended gemma4:31b-cloud model. Confirm "
            f"backend/.env before running."
        )

    runtime = OllamaRuntimeClient(chat_model=chat_model)
    health = runtime.health_check()
    if not health.get("reachable") or not health.get("chat_model_found"):
        raise RuntimeError(
            f"Ollama not ready for this QA pass: {health}. "
            f"Confirm the Ollama daemon is running and {chat_model!r} is pulled/available."
        )
    print(f"Ollama reachable — chat_model_found={health['chat_model_found']}\n")

    dispatcher = MCPToolDispatcher(runtime=runtime)

    rows: list[dict] = []
    for num, category, query in _CORPUS:
        print(f"[{num:2d}/{len(_CORPUS)}] ({category}) {query}")
        row = _run_query(dispatcher, num, category, query)
        rows.append(row)
        print(
            f"         outcome={row['outcome']:24s} "
            f"iterations={row['iteration_count']} "
            f"fetch={row['fetch_occurred']} "
            f"reformulated={row['reformulated']} "
            f"elapsed={row['elapsed_seconds']}s"
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRaw data written to: {CSV_PATH}")

    print("\n" + "=" * 78)
    print("SUMMARY BY CATEGORY")
    print("=" * 78)
    for cat in sorted(_CATEGORY_LABELS):
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        print(f"\n{cat} — {_CATEGORY_LABELS[cat]} (n={len(cat_rows)})")
        for r in cat_rows:
            print(
                f"  [{r['num']:2d}] {r['outcome']:24s} "
                f"it={r['iteration_count']} fetch={r['fetch_occurred']} "
                f"reform={r['reformulated']} t={r['elapsed_seconds']}s"
            )


if __name__ == "__main__":
    main()
