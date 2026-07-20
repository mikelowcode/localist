"""
diagnostics/diag_shadow_chart_toolcall_v3_retry.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Follow-up to v2 (54.5% MATCH on chart-expected trials, 12/66 still
MALFORMED_ENVELOPE after few-shot + bracket repair). Tests whether a
single temperature-bumped retry on those specific failures recovers
enough of them to clear a production ship bar, without touching the
schema, prompt, or parser any further.

This does NOT re-run the whole corpus. It reads
diagnostics/shadow_chart_toolcall_v2_results.csv, pulls out exactly the
rows that were still MALFORMED_ENVELOPE after repair, and re-queries
only those (instruction, category) pairs — deduplicated, since the same
instruction can appear malformed on more than one of its 3 original runs
— at a bumped temperature to get an independent sample rather than the
same deterministic failure repeated.

Isolation constraint: identical to v1/v2 — no MemoryManager writes, no
MCPToolDispatcher, no chart actually rendered.

Usage
-----
    cd diagnostics/
    python diag_shadow_chart_toolcall_v3_retry.py

Requires shadow_chart_toolcall_v2_results.csv to already exist (i.e. v2
must have been run first) and the oMLX backend to be up.

Output
------
  diagnostics/shadow_chart_toolcall_v3_retry_results.csv
  stdout — recovery rate + combined (v2 post-repair ∪ v3 retry) MATCH rate
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omlx_runtime_client import OMLXRuntimeClient  # noqa: E402

from chart_tool_schema import KNOWN_TOOL_NAMES, validate_chart_arguments  # noqa: E402
from chart_tool_schema_fewshot import SYSTEM_PROMPT_FEWSHOT  # noqa: E402
from json_envelope_repair import repair_envelope  # noqa: E402
from diag_shadow_chart_toolcall import _EXPECTS_TOOL, _CORPUS, _N_RUNS  # noqa: E402

V2_RESULTS_PATH = Path(__file__).resolve().parent / "shadow_chart_toolcall_v2_results.csv"
OUTPUT_PATH     = Path(__file__).resolve().parent / "shadow_chart_toolcall_v3_retry_results.csv"

# Independent sample, not a re-ask of the identical deterministic query —
# temperature 0 on the same input+model gives the same failure every time.
# 0.3 is a modest bump: enough to perturb token choice at the failure
# point (stray-token insertion, mid-generation drift) without abandoning
# the low-temperature regime the rest of the system relies on for
# reproducibility.
_RETRY_TEMPERATURE = 0.3
_RETRY_ATTEMPTS_PER_TRIAL = 1  # exactly the "one retry" being measured


def _load_v2_malformed_pairs() -> list[tuple[str, str]]:
    """
    Return deduplicated (instruction, category) pairs that were
    MALFORMED_ENVELOPE in v2's post-repair classification.

    Deduplicated because the same instruction can fail on more than one
    of its 3 original runs — retrying it 3 more times would inflate the
    apparent sample size without adding information; one retry per
    distinct failing instruction is what "give it one more shot" means
    in production (a single dispatch, not 3 parallel dispatches).
    """
    if not V2_RESULTS_PATH.exists():
        raise SystemExit(
            f"{V2_RESULTS_PATH} not found — run diag_shadow_chart_toolcall_v2.py first."
        )

    seen: set[tuple[str, str]] = set()
    with V2_RESULTS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["post_repair_classification"] == "MALFORMED_ENVELOPE":
                seen.add((row["instruction"], row["category"]))
    return sorted(seen)


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


_FIELDNAMES = [
    "instruction", "category",
    "retry_raw_output", "repair_outcome", "retry_classification",
]


def main() -> None:
    pairs = _load_v2_malformed_pairs()
    if not pairs:
        print("No MALFORMED_ENVELOPE rows found in v2 results — nothing to retry.")
        return

    runtime = OMLXRuntimeClient()
    rows: list[dict] = []
    retry_counts: dict[str, int] = {}

    print(f"Retrying {len(pairs)} distinct instructions that were still "
          f"MALFORMED_ENVELOPE after v2's repair pass (temperature={_RETRY_TEMPERATURE}).\n")

    for instruction, category in pairs:
        try:
            raw = runtime.infer(
                prompt=instruction,
                system=SYSTEM_PROMPT_FEWSHOT,
                max_tokens=400,
                temperature=_RETRY_TEMPERATURE,
            )
        except Exception as exc:  # noqa: BLE001
            raw = f"__INFER_ERROR__: {exc}"

        obj, repair_outcome = repair_envelope(raw)
        tool_name, arguments, envelope_bad = _classify_envelope(obj)
        classification = _classify(category, tool_name, arguments, envelope_bad)
        retry_counts[classification] = retry_counts.get(classification, 0) + 1

        rows.append({
            "instruction":          instruction,
            "category":             category,
            "retry_raw_output":     raw,
            "repair_outcome":       repair_outcome,
            "retry_classification": classification,
        })

        print(f"[{category:24s}] retry → {classification:20s} — {instruction[:60]!r}")

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print("RETRY RESULTS (temperature-bumped, one attempt per prior failure)")
    print("=" * 72)
    total_retried = len(pairs)
    for label in ["MATCH", "SCHEMA_INVALID", "MALFORMED_ENVELOPE", "MISS", "FALSE_POSITIVE", "NULL_CORRECT"]:
        n = retry_counts.get(label, 0)
        pct = (100.0 * n / total_retried) if total_retried else 0.0
        print(f"  {label:20s} {n:4d}  ({pct:5.1f}%)")

    recovered = retry_counts.get("MATCH", 0) + retry_counts.get("NULL_CORRECT", 0)
    print(f"\n  Recovered by retry: {recovered}/{total_retried} "
          f"({100.0 * recovered / total_retried:.1f}% of previously-failing instructions)")

    # ------------------------------------------------------------------
    # Combined effective rate: v2's post-repair MATCH count, plus every
    # distinct instruction that flipped from MALFORMED_ENVELOPE to MATCH
    # on retry — counted once per distinct instruction recovered, applied
    # across its original _N_RUNS occurrences, since in production a
    # retry-on-failure policy would apply per-dispatch, not per-run-in-
    # this-diagnostic. This is an estimate, not a live re-run of the full
    # corpus at N=3 with retry wired in — flagged explicitly rather than
    # presented as equivalent to v1/v2's numbers.
    # ------------------------------------------------------------------
    with V2_RESULTS_PATH.open(newline="", encoding="utf-8") as f:
        v2_rows = list(csv.DictReader(f))

    v2_match_count = sum(1 for r in v2_rows if r["post_repair_classification"] == "MATCH")
    v2_expected_total = sum(1 for r in v2_rows if _EXPECTS_TOOL[r["category"]])

    recovered_instructions = {
        (r["instruction"], r["category"]) for r in rows if r["retry_classification"] == "MATCH"
    }
    # Count how many original v2 trial-rows those recovered instructions covered.
    newly_covered_trials = sum(
        1 for r in v2_rows
        if r["post_repair_classification"] == "MALFORMED_ENVELOPE"
        and (r["instruction"], r["category"]) in recovered_instructions
        and _EXPECTS_TOOL[r["category"]]
    )

    combined_match = v2_match_count + newly_covered_trials
    print(f"\n  ESTIMATED combined MATCH rate (v2 post-repair + retry-recovered, "
          f"per-trial extrapolation):")
    print(f"    {combined_match}/{v2_expected_total} "
          f"({100.0 * combined_match / v2_expected_total:.1f}%)")
    print(f"\n  This is an estimate extrapolated from a single retry sample per")
    print(f"  distinct failing instruction, not a full re-run at N={_N_RUNS} with")
    print(f"  retry wired into the main loop. If this estimate looks promising,")
    print(f"  the honest next step is wiring retry into a full v1-style N={_N_RUNS}")
    print(f"  run (retry-on-MALFORMED_ENVELOPE inline, not post-hoc) to confirm")
    print(f"  the rate holds up rather than being an artifact of which specific")
    print(f"  trials happened to fail the first time.")


if __name__ == "__main__":
    main()
