"""
diagnostics/diag_shadow_chart_toolcall_v4_full.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

The "trustworthy number" run. v1-v3 established the pieces separately
(few-shot prompt, bracket repair, retry-on-malformed) but v3's combined
estimate was extrapolated from a 4-sample retry set applied back across
the v2 corpus — not a real measurement. This script wires all three
pieces into ONE per-trial pipeline and runs the full v1-style N=3 corpus
end to end, so the final MATCH rate is measured directly rather than
projected.

Per-trial pipeline (matches what a production dispatch path would do):
  1. infer() at temperature=0.0 with the few-shot system prompt.
  2. repair_envelope() — bracket-balanced recovery of trailing garbage.
  3. classify. If MALFORMED_ENVELOPE, retry ONCE at temperature=0.3
     (an independent sample, not a repeat of the deterministic failure),
     repair, and reclassify. The retry's outcome is final — no second
     retry, matching the "one retry" scope this was scoped to measure.

Both the first-pass and final classification are recorded per trial, so
you can see the retry's marginal contribution directly (same idea as
v2's pre-repair/post-repair columns, one level further).

Isolation constraint: identical to v1/v2/v3 — no MemoryManager writes,
no MCPToolDispatcher, no chart actually rendered.

Usage
-----
    cd diagnostics/
    python diag_shadow_chart_toolcall_v4_full.py

Output
------
  diagnostics/shadow_chart_toolcall_v4_results.csv
  stdout — first-pass vs final classification breakdown, retry effectiveness,
           final MATCH rate on chart-expected trials (the number that decides
           ship/no-ship)
"""

from __future__ import annotations

import csv
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
from diag_shadow_chart_toolcall import _CORPUS, _EXPECTS_TOOL, _N_RUNS  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "shadow_chart_toolcall_v4_results.csv"

_RETRY_TEMPERATURE = 0.3


def _classify_envelope(obj: dict | None) -> tuple[str | None, dict | None, bool]:
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


def _run_one(runtime: OMLXRuntimeClient, instruction: str, category: str, temperature: float) -> dict:
    """One infer→repair→classify pass. Returns a dict with raw output,
    repair outcome, and classification — the unit both the first attempt
    and the retry attempt share."""
    try:
        raw = runtime.infer(
            prompt=instruction,
            system=SYSTEM_PROMPT_FEWSHOT,
            max_tokens=400,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        raw = f"__INFER_ERROR__: {exc}"

    obj, repair_outcome = repair_envelope(raw)
    tool_name, arguments, envelope_bad = _classify_envelope(obj)
    classification = _classify(category, tool_name, arguments, envelope_bad)

    return {
        "raw": raw,
        "repair_outcome": repair_outcome,
        "classification": classification,
    }


_FIELDNAMES = [
    "instruction", "category", "run",
    "first_pass_classification", "first_pass_repair_outcome",
    "retry_attempted", "retry_classification", "retry_repair_outcome",
    "final_classification",
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
    first_pass_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    retry_attempted_n = 0
    retry_recovered_n = 0

    for instruction, category, _ in _CORPUS:
        planner.route(instruction, context={})  # parity with v1/v2; result unused

        for run in range(1, _N_RUNS + 1):
            first = _run_one(runtime, instruction, category, temperature=0.0)
            first_pass_counts[first["classification"]] = (
                first_pass_counts.get(first["classification"], 0) + 1
            )

            retry_attempted = first["classification"] == "MALFORMED_ENVELOPE"
            retry_result = None

            if retry_attempted:
                retry_attempted_n += 1
                retry_result = _run_one(runtime, instruction, category, temperature=_RETRY_TEMPERATURE)
                final_classification = retry_result["classification"]
                if final_classification != "MALFORMED_ENVELOPE":
                    retry_recovered_n += 1
            else:
                final_classification = first["classification"]

            final_counts[final_classification] = final_counts.get(final_classification, 0) + 1

            rows.append({
                "instruction":                instruction,
                "category":                   category,
                "run":                        run,
                "first_pass_classification":  first["classification"],
                "first_pass_repair_outcome":  first["repair_outcome"],
                "retry_attempted":            retry_attempted,
                "retry_classification":       retry_result["classification"] if retry_result else "",
                "retry_repair_outcome":       retry_result["repair_outcome"] if retry_result else "",
                "final_classification":       final_classification,
            })

            marker = ""
            if retry_attempted:
                marker = "  ← retried, recovered" if final_classification != "MALFORMED_ENVELOPE" else "  ← retried, still failed"
            print(f"[{category:24s}] run {run} — final={final_classification:20s}{marker}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(first_pass_counts.values())
    labels = ["MATCH", "SCHEMA_INVALID", "MALFORMED_ENVELOPE", "MISS", "FALSE_POSITIVE", "NULL_CORRECT"]

    print("\n" + "=" * 72)
    print(f"FIRST PASS (temperature=0.0, few-shot + repair, no retry) — n={total}")
    print("=" * 72)
    for label in labels:
        n = first_pass_counts.get(label, 0)
        print(f"  {label:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    print("\n" + "=" * 72)
    print(f"FINAL (first pass, + one temperature={_RETRY_TEMPERATURE} retry on MALFORMED_ENVELOPE) — n={total}")
    print("=" * 72)
    for label in labels:
        n = final_counts.get(label, 0)
        print(f"  {label:20s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    print("\n" + "=" * 72)
    print("RETRY EFFECTIVENESS")
    print("=" * 72)
    print(f"  Trials that needed a retry:  {retry_attempted_n}/{total}")
    if retry_attempted_n:
        print(f"  Retry recovered (any non-MALFORMED outcome): "
              f"{retry_recovered_n}/{retry_attempted_n} "
              f"({100.0 * retry_recovered_n / retry_attempted_n:.1f}%)")

    expected_tool_n = sum(1 for _, cat, _ in _CORPUS for _ in range(_N_RUNS) if _EXPECTS_TOOL[cat])
    first_match = first_pass_counts.get("MATCH", 0)
    final_match = final_counts.get("MATCH", 0)
    print(f"\n  MATCH rate on chart-expected trials (n={expected_tool_n}):")
    print(f"    first pass (no retry): {first_match}/{expected_tool_n} "
          f"({100.0 * first_match / expected_tool_n:.1f}%)")
    print(f"    final (with retry):    {final_match}/{expected_tool_n} "
          f"({100.0 * final_match / expected_tool_n:.1f}%)  ← this is the real, measured number")
    print(f"\n  Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
