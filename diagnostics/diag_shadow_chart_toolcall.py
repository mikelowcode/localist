"""
diagnostics/diag_shadow_chart_toolcall.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Adapted from diagnostics/diag_shadow_toolcall.py for the proposed
`generate_chart` tool. Measures how reliably gemma-4-e4b-it-4bit (via
OMLXRuntimeClient, same as production) produces a well-formed tool_call
envelope AND schema-valid `generate_chart` arguments when given the
chart_tool_schema.py system prompt, across a fixed corpus of chart and
non-chart instructions.

This exists to answer one question before any production code is written:
is the chart argument-extraction step (Planner gate → model emits JSON
chart config → parse/validate) reliable enough on this quantized local
model to ship, and how narrow does the schema need to stay? See
chart_tool_schema.py's module docstring for why the schema is scoped to
3 chart types / flat datasets rather than the full Chart.js surface.

Isolation constraint (same as diag_shadow_toolcall.py)
-------------------------------------------------------
This script NEVER calls MemoryManager, NEVER writes to working memory,
NEVER calls MCPToolDispatcher.dispatch(), and NEVER executes any tool
(no chart is actually rendered). It only calls:
  - Planner.route()             — for the real (keyword-gate) routing decision
  - OMLXRuntimeClient.infer()   — for Gemma's shadow chart-tool-call proposal
and logs the outputs. Planner is constructed with memory_manager=None and
embed_fn=None (same as the original diagnostic), which makes every
MemoryManager-dependent priority a documented no-op — and, since every
self._runtime.infer() call inside planner.py is itself gated behind
embed_fn/memory_manager checks that fire first, runtime=None is safe here
too: no code path in Planner.route() can reach the runtime when
memory_manager and embed_fn are both None.

Usage
-----
    cd diagnostics/
    python diag_shadow_chart_toolcall.py

Requires the same running oMLX backend the original diagnostic needs
(LOCALIST's inference engine on its configured port — see
omlx_runtime_client.py's _DEFAULT_BASE_URL / LORA_OMLX_URL). Connectivity
is checked up front (same as diag_shadow_toolcall.py) so an unreachable
backend fails fast with a clear error instead of surfacing as a wall of
MALFORMED_ENVELOPE trials that look like a model-reliability finding but
are actually a connectivity problem.

Output
------
  diagnostics/shadow_chart_toolcall_results.csv  — one row per trial
  stdout                                          — summary table
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — same pattern as diag_shadow_toolcall.py. Adjust if this
# file is not placed at diagnostics/ alongside the original.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))
# chart_tool_schema.py is expected to sit next to this script in diagnostics/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import Planner  # noqa: E402
from omlx_runtime_client import OMLXRuntimeClient  # noqa: E402

from chart_tool_schema import (  # noqa: E402
    KNOWN_TOOL_NAMES,
    SYSTEM_PROMPT,
    validate_chart_arguments,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "shadow_chart_toolcall_results.csv"

# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------
# Categories mirror diag_shadow_toolcall.py's structure:
#   chart_keyword_clear   — obvious chart-trigger keywords + inline data
#                            (what a P3-style Planner keyword gate should catch)
#   chart_semantic_implicit — chart intent without an obvious trigger keyword
#                            (tests whether the *gate*, not just the model,
#                            would miss these — separate concern from
#                            argument-extraction reliability, but worth
#                            recording since it affects real coverage)
#   negative_control       — ordinary non-chart instructions; measures
#                            false-positive rate (chart proposed when it
#                            shouldn't be)
#   ambiguous              — data-adjacent but no explicit visualize/plot
#                            ask; genuinely arguable either way
#
# expects_tool encodes the ground-truth judgment call used for
# classification. Reasonable people can disagree on a couple of the
# "semantic_implicit"/"ambiguous" rows — that's expected and is exactly
# what separates gate design from argument-extraction reliability.

_CORPUS: list[tuple[str, str, bool]] = [
    # (instruction, category, expects_tool)
    ("Chart this data for me: apples 5, oranges 3, bananas 7",
     "chart_keyword_clear", True),
    ("Make a bar chart of Q3 revenue by region: North 120, South 90, East 150, West 80",
     "chart_keyword_clear", True),
    ("Plot the last 5 months of signups: Jan 10, Feb 15, Mar 22, Apr 18, May 30",
     "chart_keyword_clear", True),
    ("Can you graph these numbers? 12, 19, 8, 24, 16",
     "chart_keyword_clear", True),
    ("Visualize this: rent 40, food 25, transport 15, other 20",
     "chart_keyword_clear", True),
    ("Turn this into a pie chart: Chrome 65, Safari 19, Firefox 8, Edge 6, Other 2",
     "chart_keyword_clear", True),
    ("Graph the temperature readings over the week: Mon 68, Tue 71, Wed 69, Thu 74, Fri 77",
     "chart_keyword_clear", True),

    ("Show me a trend line of these values over time: 10, 20, 15, 25, 30",
     "chart_semantic_implicit", True),
    ("I want to see this as a breakdown: rent 40%, food 25%, transport 15%, other 20%",
     "chart_semantic_implicit", True),
    ("Turn this into something visual: Q1 100, Q2 150, Q3 130, Q4 180",
     "chart_semantic_implicit", True),
    ("Compare these three products' sales side by side: A 45, B 60, C 38",
     "chart_semantic_implicit", True),

    ("What's the capital of France?",
     "negative_control", False),
    ("Summarize this article for me.",
     "negative_control", False),
    ("Remember that I prefer dark mode.",
     "negative_control", False),
    ("What did I tell you about my project setup?",
     "negative_control", False),
    ("Search the web for the latest AI news.",
     "negative_control", False),
    ("Read the file at notes/todo.md",
     "negative_control", False),
    ("What is the current CEO of OpenAI?",
     "negative_control", False),
    ("Explain what a bar chart is.",
     "negative_control", False),

    ("Can you show me this data?",
     "ambiguous", False),
    ("I have some numbers, what should I do with them?",
     "ambiguous", False),
    ("Here's my budget: rent 1200, food 400, transport 200. Thoughts?",
     "ambiguous", False),
]

_EXPECTS_TOOL: dict[str, bool] = {
    "chart_keyword_clear":     True,
    "chart_semantic_implicit": True,
    "negative_control":        False,
    "ambiguous":               False,
}

_N_RUNS = 3  # repeat each instruction this many times — small local models
             # are not perfectly deterministic even at low temperature.

# ---------------------------------------------------------------------------
# Parsing — envelope check (same contract/shape as diag_shadow_toolcall.py's
# _parse_gemma_output), then schema-level validation via
# chart_tool_schema.validate_chart_arguments.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _parse_envelope(raw: str) -> tuple[str | None, dict | None, bool]:
    """
    Returns (tool_name_or_None, arguments_dict_or_None, envelope_malformed).

    envelope_malformed=True whenever the text does not parse as JSON, or
    does not conform to the {"tool_call": null | {"name":..., "arguments":...}}
    contract, or names a tool outside KNOWN_TOOL_NAMES. This mirrors
    diag_shadow_toolcall.py's _parse_gemma_output exactly, so envelope
    malformed-rates are comparable across the two diagnostics.
    """
    cleaned = _strip_code_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, None, True

    if not isinstance(obj, dict) or "tool_call" not in obj:
        return None, None, True

    call = obj["tool_call"]
    if call is None:
        return None, None, False

    if (
        not isinstance(call, dict)
        or "name" not in call
        or "arguments" not in call
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
        or call["name"] not in KNOWN_TOOL_NAMES
    ):
        return None, None, True

    return call["name"], call["arguments"], False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
# Buckets (wider than the original diagnostic's five, to separate envelope
# failure from argument-quality failure — the distinction that matters for
# deciding whether to narrow the schema further or widen it):
#
#   MALFORMED_ENVELOPE — output isn't valid {"tool_call": ...} JSON at all
#   SCHEMA_INVALID     — envelope parsed, tool proposed, but arguments fail
#                        validate_chart_arguments() (wrong types, mismatched
#                        lengths, missing fields, etc.)
#   MATCH              — envelope valid, tool expected, arguments valid
#   FALSE_POSITIVE     — tool proposed (validly or not) when none expected
#   MISS               — tool expected, model proposed tool_call: null
#   NULL_CORRECT       — tool not expected, model correctly proposed null
#
# This classification is about argument-extraction reliability only. The
# Planner's own routing decision (planner_tools_to_call / planner_priority)
# is still recorded per row in the CSV for manual cross-reference against
# category (e.g. does the keyword gate already cover chart_semantic_implicit?)
# but deliberately does not feed into the classification below — that would
# be measuring gate design, a separate concern from whether the model itself
# produces usable chart arguments.

def _classify(
    category: str,
    tool_name: str | None,
    arguments: dict | None,
    envelope_malformed: bool,
) -> str:
    expects_tool = _EXPECTS_TOOL[category]

    if envelope_malformed:
        return "MALFORMED_ENVELOPE"

    proposed = tool_name is not None

    if not expects_tool:
        return "FALSE_POSITIVE" if proposed else "NULL_CORRECT"

    # expects_tool == True from here down
    if not proposed:
        return "MISS"

    problems = validate_chart_arguments(arguments or {})
    if problems:
        return "SCHEMA_INVALID"
    return "MATCH"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_FIELDNAMES = [
    "instruction", "category", "run",
    "planner_tools_to_call", "planner_priority",
    "gemma_raw_output", "gemma_parsed_tool", "gemma_parsed_args",
    "schema_problems", "classification",
]


def main() -> None:
    print("Checking oMLX connectivity at http://localhost:8000 …")
    runtime = OMLXRuntimeClient()
    health = runtime.health_check()
    if not health.get("reachable"):
        raise RuntimeError(
            "Cannot reach oMLX at http://localhost:8000. "
            f"Health response: {health}. "
            "Start the oMLX inference server before running this diagnostic."
        )
    print(f"  oMLX reachable — model: {health.get('chat_model_found')}\n")

    planner = Planner(runtime=None, memory_manager=None, embed_fn=None)

    rows: list[dict] = []
    counts: dict[str, int] = {}

    for instruction, category, _ in _CORPUS:
        plan = planner.route(instruction, context={})
        planner_tools = plan.tools_to_call

        for run in range(1, _N_RUNS + 1):
            try:
                raw = runtime.infer(
                    prompt=instruction,
                    system=SYSTEM_PROMPT,
                    max_tokens=400,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                raw = f"__INFER_ERROR__: {exc}"

            tool_name, arguments, envelope_malformed = _parse_envelope(raw)
            problems = (
                []
                if envelope_malformed or tool_name is None
                else validate_chart_arguments(arguments or {})
            )
            classification = _classify(
                category, tool_name, arguments, envelope_malformed
            )
            counts[classification] = counts.get(classification, 0) + 1

            rows.append({
                "instruction":           instruction,
                "category":              category,
                "run":                   run,
                "planner_tools_to_call": ";".join(planner_tools),
                "planner_priority":      plan.priority,
                "gemma_raw_output":      raw,
                "gemma_parsed_tool":     tool_name or "",
                "gemma_parsed_args":     json.dumps(arguments) if arguments else "",
                "schema_problems":       " | ".join(problems),
                "classification":        classification,
            })

            print(f"[{category:24s}] run {run} — {classification:20s} — {instruction[:60]!r}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    total = sum(counts.values())
    for label in [
        "MATCH", "SCHEMA_INVALID", "MALFORMED_ENVELOPE",
        "MISS", "FALSE_POSITIVE", "NULL_CORRECT",
    ]:
        n = counts.get(label, 0)
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {label:20s} {n:4d}  ({pct:5.1f}%)")
    print(f"\n  Total trials: {total}")
    print(f"  Results written to: {OUTPUT_PATH}")

    match_n = counts.get("MATCH", 0)
    expected_tool_n = sum(
        1 for _, cat, _ in _CORPUS for _ in range(_N_RUNS) if _EXPECTS_TOOL[cat]
    )
    if expected_tool_n:
        print(
            f"\n  Argument-extraction success rate on chart-expected trials: "
            f"{match_n}/{expected_tool_n} ({100.0 * match_n / expected_tool_n:.1f}%)"
        )
    print(
        "\n  Read SCHEMA_INVALID rows' schema_problems column to see which "
        "specific schema fields the model gets wrong most often — that's "
        "the concrete signal for whether to simplify further (e.g. drop "
        "multi-dataset support) or whether the schema is already fine."
    )


if __name__ == "__main__":
    main()
