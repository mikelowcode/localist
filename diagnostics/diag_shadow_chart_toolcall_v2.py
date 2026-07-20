"""
diagnostics/diag_shadow_chart_toolcall_v2.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Follow-up to diag_shadow_chart_toolcall.py's first run (66 trials:
40.9% MALFORMED_ENVELOPE, 0% SCHEMA_INVALID, 36.4% MATCH). That run
showed the schema itself isn't the bottleneck — envelope well-formedness
is. This script re-runs the SAME corpus with two independent changes and
reports both pre- and post-repair classification per trial, so you can
see how much each change buys separately:

  1. Few-shot system prompt (chart_tool_schema_fewshot.SYSTEM_PROMPT_FEWSHOT)
     — targets the "prose instead of null envelope" failures.
  2. Bracket-balanced repair pass (json_envelope_repair.repair_envelope)
     — targets the "stray token after valid JSON" failures. Applied
     AFTER inference, so it's visible as its own lever independent of
     the prompt change.

Genuine mid-array truncation is NOT repaired (see json_envelope_repair.py's
docstring) — those trials will still show up as a distinct outcome
("truncated") so the truncation rate is measured, not hidden by the
repair layer. Nor is arbitrary bracket corruption where no candidate
excision produces a fully valid parse — that shows up as "unrepairable"
rather than being guessed at (see json_envelope_repair.py's module
docstring for why guessing is unsafe: a decoy delimiter inside the stray
tokens themselves can make a wrong excision look plausible while
producing a structure the model never actually said).

Isolation constraint: identical to diag_shadow_chart_toolcall.py — no
MemoryManager writes, no MCPToolDispatcher, no chart actually rendered.

Usage
-----
    cd diagnostics/
    python diag_shadow_chart_toolcall_v2.py

Output
------
  diagnostics/shadow_chart_toolcall_v2_results.csv
  stdout — summary table, including pre-repair vs post-repair MATCH counts
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import Planner  # noqa: E402
from omlx_runtime_client import OMLXRuntimeClient  # noqa: E402

from chart_tool_schema import KNOWN_TOOL_NAMES, validate_chart_arguments  # noqa: E402
from chart_tool_schema_fewshot import SYSTEM_PROMPT_FEWSHOT  # noqa: E402
from json_envelope_repair import repair_envelope  # noqa: E402

# Reuse the exact same corpus AND the same fence-stripping helper as v1 so
# results are a like-for-like delta. _strip_code_fences is imported (not
# reimplemented) specifically so the pre-repair parse below has the exact
# same strictness as v1's _parse_envelope — without this, a fenced
# response (plausible given the few-shot examples below) would be
# miscounted as MALFORMED_ENVELOPE pre-repair when v1 would have parsed
# it fine, corrupting the pre-repair-vs-v1 comparison this script exists
# to make.
from diag_shadow_chart_toolcall import (  # noqa: E402
    _CORPUS, _EXPECTS_TOOL, _N_RUNS, _strip_code_fences,
)

OUTPUT_PATH = Path(__file__).resolve().parent / "shadow_chart_toolcall_v2_results.csv"


def _classify_envelope(obj) -> tuple[str | None, dict | None, bool]:
    """Same envelope-shape check as v1's _parse_envelope, operating on an
    already-parsed value rather than raw text (repair_envelope does its
    own parsing)."""
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


def _classify(category: str, tool_name: str | None, arguments: dict | None, envelope_bad: bool) -> str:
    expects_tool = _EXPECTS_TOOL[category]
    if envelope_bad:
        return "MALFORMED_ENVELOPE"
    proposed = tool_name is not None
    if not expects_tool:
        return "FALSE_POSITIVE" if proposed else "NULL_CORRECT"
    if not proposed:
        return "MISS"
    problems = validate_chart_arguments(arguments or {})
    return "SCHEMA_INVALID" if problems else "MATCH"


_FIELDNAMES = [
    "instruction", "category", "run",
    "gemma_raw_output",
    "pre_repair_classification",
    "repair_outcome",
    "post_repair_classification",
    "improved_by_repair",
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
    pre_counts: dict[str, int] = {}
    post_counts: dict[str, int] = {}
    repair_outcome_counts: dict[str, int] = {}

    for instruction, category, _ in _CORPUS:
        planner.route(instruction, context={})  # kept for parity with v1; result unused here

        for run in range(1, _N_RUNS + 1):
            try:
                raw = runtime.infer(
                    prompt=instruction,
                    system=SYSTEM_PROMPT_FEWSHOT,
                    max_tokens=400,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                raw = f"__INFER_ERROR__: {exc}"

            # Pre-repair: direct parse only (same strictness as v1,
            # including fence-stripping — see the import comment above).
            try:
                direct_obj = json.loads(_strip_code_fences(raw))
                pre_name, pre_args, pre_bad = _classify_envelope(direct_obj)
            except json.JSONDecodeError:
                pre_name, pre_args, pre_bad = None, None, True
            pre_class = _classify(category, pre_name, pre_args, pre_bad)

            # Post-repair: bracket-balanced repair pass, then re-classify.
            repaired_obj, repair_outcome = repair_envelope(raw)
            repair_outcome_counts[repair_outcome] = repair_outcome_counts.get(repair_outcome, 0) + 1
            post_name, post_args, post_bad = _classify_envelope(repaired_obj)
            post_class = _classify(category, post_name, post_args, post_bad)

            pre_counts[pre_class] = pre_counts.get(pre_class, 0) + 1
            post_counts[post_class] = post_counts.get(post_class, 0) + 1

            rows.append({
                "instruction":                 instruction,
                "category":                    category,
                "run":                         run,
                "gemma_raw_output":            raw,
                "pre_repair_classification":   pre_class,
                "repair_outcome":              repair_outcome,
                "post_repair_classification":  post_class,
                "improved_by_repair":          pre_class != post_class,
            })

            marker = "  ← repaired" if pre_class != post_class else ""
            print(f"[{category:24s}] run {run} — pre={pre_class:20s} post={post_class:20s}{marker}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(pre_counts.values())
    labels = ["MATCH", "SCHEMA_INVALID", "MALFORMED_ENVELOPE", "MISS", "FALSE_POSITIVE", "NULL_CORRECT"]

    print("\n" + "=" * 72)
    print("PRE-REPAIR (few-shot prompt only, strict direct-parse)")
    print("=" * 72)
    for label in labels:
        n = pre_counts.get(label, 0)
        print(f"  {label:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    print("\n" + "=" * 72)
    print("POST-REPAIR (few-shot prompt + bracket-balanced repair)")
    print("=" * 72)
    for label in labels:
        n = post_counts.get(label, 0)
        print(f"  {label:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    print("\n" + "=" * 72)
    print("REPAIR OUTCOME BREAKDOWN")
    print("=" * 72)
    for outcome, n in sorted(repair_outcome_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    expected_tool_n = sum(1 for _, cat, _ in _CORPUS for _ in range(_N_RUNS) if _EXPECTS_TOOL[cat])
    pre_match = pre_counts.get("MATCH", 0)
    post_match = post_counts.get("MATCH", 0)
    print(f"\n  MATCH rate on chart-expected trials:")
    print(f"    pre-repair:  {pre_match}/{expected_tool_n} ({100.0 * pre_match / expected_tool_n:.1f}%)")
    print(f"    post-repair: {post_match}/{expected_tool_n} ({100.0 * post_match / expected_tool_n:.1f}%)")
    print(f"\n  Compare pre-repair MATCH% here against v1's 36.4% to isolate the")
    print(f"  few-shot prompt's effect; compare post-repair MATCH% against")
    print(f"  pre-repair here to isolate the repair layer's effect.")
    print(f"\n  Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
