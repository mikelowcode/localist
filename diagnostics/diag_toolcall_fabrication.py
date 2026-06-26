"""
diagnostics/diag_toolcall_fabrication.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Reproduces the toolcall-fabrication incident observed 2026-06-22, where the
model returned a raw `<|tool_call>...` string instead of a synthesized answer
after web_search results had already been injected into the prompt.

Tests 4 instruction-phrasing variants × 5 runs = 20 inference calls.
Isolates phrasing as the only variable; system prompt, tool results, and
working memory are held fixed at the exact values logged during the incident.

Output:
  diagnostics/toolcall_fabrication_results.jsonl   — one JSON object per call
  stdout                                             — summary table
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — import OMLXRuntimeClient without touching ControllerAgent,
# Planner, ConversationalAgent, or MemoryManager.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from omlx_runtime_client import OMLXRuntimeClient  # noqa: E402

# ---------------------------------------------------------------------------
# Incident fixture — verbatim from backend.log, task d4641550-…, 17:25:22
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are LORA, a local research assistant. You reason carefully, cite your \
sources, and acknowledge when you don't know something. You do not simulate \
certainty.

You are LORA, a local‑first thinking partner.
You speak clearly, directly, and in a natural conversational tone.
You use tools when they are needed and follow tool instructions precisely.
You never claim to have called a tool or received a result that didn't actually happen \
— if a tool fails or returns nothing useful, you say so plainly.
When you state facts, you cite where they came from, distinguishing what you \
found in the wiki or vault from what you found on the web."""

# Exact [TOOL RESULTS] block from the log (lines 559-590).
# The [TOOL RESULTS] label itself is included here so the assembled user
# prompt is identical to what PromptBuilder produced on the failing turn.
TOOL_RESULTS_BLOCK = """\
[TOOL RESULTS]
web_search(query="Do a web search then tell when Microsoft's first formal investment in OpenAI was?"):
  • Microsoft Corporation's partnership with OpenAI, Inc. Decision on relevant merger situation
  1
 microsoft corporation 's partnership with openai , inc.
 decision on relevant merger situation
 me / 7084 / 23
 the competition and markets authority (cma) 's decision on relevant merger situation under section 22 ( 1 ) of the enterprise act 2002 ( the act ) given on 5 march 2025 . full text
  [https://assets.publishing.service.gov.uk/media/67fe26ef712bf73dea135449/Full_text_decision__.pdf]

• OpenAI - Wikipedia
  contents
 founding
 corporate structure
 2.1 transition from nonprofit
 2.1.1 creation of for - profit subsidiaries ( 2019 )
 2.1.2 conversion to public benefit corporation ( 2025 )
 2.2 partnership with microsoft
 2.3 finances
 3
 business model
 3.1 acquisitions
 3.2 corporate
  [https://en.wikipedia.org/wiki/OpenAI?country=48ps0w]

• Are Microsoft and OpenAI Breaking Up? What To Know | Built In
  are microsoft and openai breaking up ? it 's complicated .
 after billions invested and nearly a decade together , the tech giants are testing the boundaries of their ai partnership — and eyeing what life apart might look like .
 written by brooke becher
 image : built in / shutterstock

  [https://builtin.com/artificial-intelligence/microsoft-openai-breakup]"""

# Exact [WORKING MEMORY] block from the log (lines 592-597).
# Kept fixed across all variants — exactly as the model saw it on the failing turn.
WORKING_MEMORY_BLOCK = """\
[WORKING MEMORY]
Turn -5 [user]: Can you tell me when was Microsoft's first formal investment in OpenAI?
Turn -4 [agent]: {'answer': "\\nI found information regarding Microsoft's involvement with OpenAI.\\n\\nAccording to my search, Microsoft's initial investment and partnership with OpenAI began in **2013**.\\n\\nIf you'd like more detail about the timeline or the specifics of that first investment, just let me know!", 'sources': [], 'grounded': False}
Turn -3 [user]: 2013 that doesn't sound correct. Why don't you look it up first before you answer next time.
Turn -2 [agent]: {'answer': '\\nI apologize for the error. You are correct that the information I provided regarding the timeline was inaccurate. I will ensure I verify my sources more thoroughly before presenting facts in the future.', 'sources': [], 'grounded': False}
Turn -1 [user]: Do a web search then tell when Microsoft's first formal investment in OpenAI was?"""

# ---------------------------------------------------------------------------
# Variants — only the final [INSTRUCTION] line differs.
# ---------------------------------------------------------------------------

VARIANTS: list[tuple[str, str]] = [
    (
        "A",
        "Do a web search then tell when Microsoft's first formal investment in OpenAI was?",
    ),
    (
        "B",
        "Based on the search results, when was Microsoft's first formal investment in OpenAI?",
    ),
    (
        "C",
        "When was Microsoft's first formal investment in OpenAI?",
    ),
    (
        "D",
        "You've already searched the web for this — now tell me when Microsoft's first formal investment in OpenAI was.",
    ),
]

RUNS_PER_VARIANT = 5

# Confirmed from conversational_agent.py (lines 165-166): defaults 1024 / 0.3
MAX_TOKENS  = 1024
TEMPERATURE = 0.30

# ---------------------------------------------------------------------------
# Fabrication detection patterns (case-insensitive substring search)
# ---------------------------------------------------------------------------

_FABRICATION_PATTERNS = [
    "<|toolcall",
    "<tool_call",
    "tool_call>",
    "call:websearch",
    "call:web_search",
]

_YEAR_PATTERN = re.compile(r"\b(199[0-9]|20[0-2][0-9]|2030)\b")

OUTPUT_PATH = Path(__file__).resolve().parent / "toolcall_fabrication_results.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(instruction: str) -> str:
    return (
        TOOL_RESULTS_BLOCK
        + "\n\n"
        + WORKING_MEMORY_BLOCK
        + "\n\n[INSTRUCTION]\n"
        + instruction
    )


def _has_fabricated_tag(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in _FABRICATION_PATTERNS)


def _has_year(text: str) -> bool:
    return bool(_YEAR_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- Connectivity check: fail loudly rather than hanging ------------------
    print("Checking oMLX connectivity at http://localhost:8000 …")
    client = OMLXRuntimeClient()
    health = client.health_check()
    if not health.get("reachable"):
        raise RuntimeError(
            "Cannot reach oMLX at http://localhost:8000. "
            f"Health response: {health}. "
            "Start the oMLX inference server before running this diagnostic."
        )
    print(f"  oMLX reachable — model: {health.get('chat_model_found')}\n")

    results: list[dict] = []

    total_calls = len(VARIANTS) * RUNS_PER_VARIANT
    call_n = 0

    for variant_label, instruction in VARIANTS:
        user_prompt = _build_user_prompt(instruction)
        print(f"=== Variant {variant_label}: {instruction[:70]!r} ===")

        for run in range(1, RUNS_PER_VARIANT + 1):
            call_n += 1
            print(f"  run {run}/{RUNS_PER_VARIANT}  (call {call_n}/{total_calls}) … ", end="", flush=True)

            ts = datetime.now(timezone.utc).isoformat()
            try:
                response_text = client.infer(
                    prompt      = user_prompt,
                    system      = SYSTEM_PROMPT,
                    max_tokens  = MAX_TOKENS,
                    temperature = TEMPERATURE,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                raise

            fabricated = _has_fabricated_tag(response_text)
            has_year   = _has_year(response_text)
            char_len   = len(response_text)

            flag = " *** FABRICATION ***" if fabricated else ""
            print(f"{char_len} chars  fabricated={fabricated}  year={has_year}{flag}")

            results.append({
                "variant":           variant_label,
                "run":               run,
                "instruction":       instruction,
                "response_text":     response_text,
                "char_len":          char_len,
                "has_fabricated_tag": fabricated,
                "has_year_pattern":  has_year,
                "timestamp":         ts,
            })

        print()

    # -- Write JSONL ----------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Results written to: {OUTPUT_PATH}  ({len(results)} lines)\n")

    # -- Summary table --------------------------------------------------------
    print("=" * 80)
    print(f"{'VARIANT':<8} {'RUNS':>5} {'FABRICATIONS':>13} {'RATE':>8}  SAMPLE (first fabricated, or 'none observed')")
    print("-" * 80)

    for variant_label, instruction in VARIANTS:
        variant_results = [r for r in results if r["variant"] == variant_label]
        fab_results = [r for r in variant_results if r["has_fabricated_tag"]]
        fab_count = len(fab_results)
        rate = fab_count / len(variant_results) if variant_results else 0.0
        sample = fab_results[0]["response_text"][:80].replace("\n", "\\n") if fab_results else "none observed"
        print(f"{variant_label:<8} {len(variant_results):>5} {fab_count:>13} {rate:>7.0%}  {sample!r}")

    print("=" * 80)
    print()

    # -- Full text of every fabricated response --------------------------------
    fab_all = [r for r in results if r["has_fabricated_tag"]]
    if fab_all:
        print(f"\n{'=' * 80}")
        print(f"FABRICATED RESPONSES — {len(fab_all)} of {total_calls} calls")
        print("=" * 80)
        for r in fab_all:
            print(f"\n  Variant {r['variant']}  run {r['run']}  ({r['char_len']} chars)")
            print(f"  Instruction: {r['instruction']!r}")
            print(f"  Raw response text:")
            print(f"  {r['response_text']!r}")
    else:
        print("No fabricated responses observed across all 20 calls.")


if __name__ == "__main__":
    main()
