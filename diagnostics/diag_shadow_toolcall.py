"""
diagnostics/diag_shadow_toolcall.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Shadow tool-call diagnostic: for a fixed corpus of instructions, compares the
Planner's real (rule-engine) routing decision against a "shadow" proposal
obtained by asking Gemma directly whether a tool call is warranted, given an
explicit tool-schema system prompt. The goal is to measure how often the
deterministic keyword-based Planner misses a tool-worthy instruction that a
schema-aware Gemma call would have caught (semantic-miss recovery), and how
often such a shadow call would introduce false positives instead.

Isolation constraint (hard requirement)
----------------------------------------
This script NEVER calls MemoryManager, NEVER writes to working memory, NEVER
calls MCPToolDispatcher.dispatch(), and NEVER executes any tool. It only
calls:
  - Planner.route()            — for the real routing decision
  - OMLXRuntimeClient.infer()   — for Gemma's shadow tool-call proposal
and logs the outputs. Planner is constructed with memory_manager=None and
embed_fn=None, which makes every MemoryManager-dependent priority (3b, 3c, 4)
a documented no-op inside planner.py itself (they check for None and return
None) and makes the semantic-intent gate in Priority 3 skip immediately
(no embedding calls). Nothing this script produces is persisted anywhere in
the running system — output is confined to diagnostics/shadow_toolcall_results.csv.

Output
------
  diagnostics/shadow_toolcall_results.csv   — one row per trial
  stdout                                     — summary table (classification counts by category)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — import Planner + OMLXRuntimeClient without touching
# ControllerAgent, ConversationalAgent, MemoryManager, or MCPToolDispatcher.
# ---------------------------------------------------------------------------

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from planner import Planner  # noqa: E402
from omlx_runtime_client import OMLXRuntimeClient  # noqa: E402
from prompt_builder import PromptBuilder, ToolResult, Turn  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "shadow_toolcall_results.csv"

# ---------------------------------------------------------------------------
# Tool schema — built directly from the real MCP signatures/docstrings in
# backend/mcp_server/main.py (read_file, write_file, append_file, fetch_url,
# web_search). Not approximated.
# ---------------------------------------------------------------------------

TOOL_SCHEMA: list[dict] = [
    {
        "name": "web_search",
        "description": "Run one web search query via LangSearch. Requires LANGSEARCH_API_KEY.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file. path is resolved relative to project_root and sandboxed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a UTF-8 text file. path is resolved relative to project_root and sandboxed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_file",
        "description": "Append content to a UTF-8 text file. path is resolved relative to project_root and sandboxed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and extract clean article text (title, author, date, cleaned_text, word_count).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout": {"type": "number", "default": 10.0},
            },
            "required": ["url"],
        },
    },
]

KNOWN_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOL_SCHEMA)

SYSTEM_PROMPT = f"""You have access to the following tools:

{json.dumps(TOOL_SCHEMA, indent=2)}

When the user's instruction requires calling one of these tools, respond with
ONLY a single JSON object of the form:
  {{"tool_call": {{"name": "<tool_name>", "arguments": {{...}}}}}}

When no tool is required, respond with ONLY:
  {{"tool_call": null}}

Rules:
- Output exactly one JSON object and nothing else.
- No prose, no explanation, no markdown code fences.
- "name" must be one of: web_search, read_file, write_file, append_file, fetch_url.
- "arguments" must contain exactly the parameters listed in that tool's schema."""

MAX_TOKENS  = 300
TEMPERATURE = 0.2
RUNS_PER_INSTRUCTION = 5

# ---------------------------------------------------------------------------
# Instruction corpus — (instruction, category)
# ---------------------------------------------------------------------------

CAT_SEMANTIC_MISS                  = "semantic_miss"
CAT_KEYWORD_CLEAR                  = "keyword_clear"
CAT_TRUE_NEGATIVE                  = "true_negative"
CAT_TOOL_FLAVORED_RESOLVED         = "tool_flavored_resolved"
CAT_TOOL_FLAVORED_RESOLVED_GROUNDED = "tool_flavored_resolved_grounded"

CORPUS: list[tuple[str, str]] = [
    # Semantic-miss — genuinely needs web_search, but avoids every literal
    # _WEB_SEARCH_KEYWORDS / _FACTUAL_QUERY_KEYWORDS match in planner.py.
    ("Go head and look up the weather in Oceanside CA.", CAT_SEMANTIC_MISS),  # verbatim, live testing 2026-xx-xx
    ("Could you look into what the weather's like in Oceanside CA right now?", CAT_SEMANTIC_MISS),
    ("Can you check what the weather is doing in Oceanside CA?", CAT_SEMANTIC_MISS),
    ("Find out what's going on with the weather in Oceanside CA.", CAT_SEMANTIC_MISS),
    ("See what the weather looks like over in Oceanside CA.", CAT_SEMANTIC_MISS),

    # Keyword-clear — literally contains a _WEB_SEARCH_KEYWORDS entry
    # ("latest" / "today" / "news"), so Planner P3 should fire.
    ("Look up the latest Oceanside CA weather report.", CAT_KEYWORD_CLEAR),  # verbatim
    ("What's the weather in Oceanside CA today?", CAT_KEYWORD_CLEAR),
    ("Get me the news on the weather in Oceanside CA.", CAT_KEYWORD_CLEAR),

    # True-negative — no tool is ever warranted.
    ("Can you explain what causes a low-pressure weather system?", CAT_TRUE_NEGATIVE),
    ("What's a good synonym for 'rainy'?", CAT_TRUE_NEGATIVE),
    ("Summarize the plot of Moby Dick for me.", CAT_TRUE_NEGATIVE),

    # Tool-flavored-but-resolved — references an already-completed search,
    # same trigger shape as the Open Item 11 fabrication incident (see
    # diag_toolcall_fabrication.py variant D). Correct behavior is tool_call=null.
    ("You already looked up the Oceanside CA weather — just tell me what you found.", CAT_TOOL_FLAVORED_RESOLVED),
    ("Since you've already checked the forecast, what's the weather like in Oceanside CA?", CAT_TOOL_FLAVORED_RESOLVED),

    # Tool-flavored-but-resolved, GROUNDED — the actual Open Item 11 condition:
    # a real [TOOL RESULTS] slot (web_search results already present) plus
    # working memory showing the search was requested and run, assembled via
    # the real PromptBuilder.build() slot ordering (see _build_grounded_prompt
    # below). Correct behavior is still tool_call=null — the model should
    # synthesize from the results already in front of it, not re-call the tool.
    ("Since you've already checked the forecast, what's the outlook for tomorrow?", CAT_TOOL_FLAVORED_RESOLVED_GROUNDED),
    ("Based on the search results above, summarize the Oceanside forecast.", CAT_TOOL_FLAVORED_RESOLVED_GROUNDED),
]

_EXPECTS_TOOL: dict[str, bool] = {
    CAT_SEMANTIC_MISS:                   True,
    CAT_KEYWORD_CLEAR:                   True,
    CAT_TRUE_NEGATIVE:                   False,
    CAT_TOOL_FLAVORED_RESOLVED:          False,
    CAT_TOOL_FLAVORED_RESOLVED_GROUNDED: False,
}

# ---------------------------------------------------------------------------
# Grounded fixture — real [TOOL RESULTS] slot for CAT_TOOL_FLAVORED_RESOLVED_GROUNDED.
# Assembled via the real PromptBuilder.build() (backend/prompt_builder.py),
# same slot ordering PromptBuilder uses in production (system → … → tool
# results → working memory → instruction). Bullet format mirrors the real
# LangSearch formatting in backend/mcp_server/web_search.py:
#   "• {name}\n  {body}\n  [{url}]" joined with "\n\n".
# This is a diagnostic-only fixture — no live LangSearch call is made, no
# MCPToolDispatcher.dispatch() is invoked, nothing is executed.
# ---------------------------------------------------------------------------

_GROUNDED_QUERY = "Oceanside CA weather forecast"

_GROUNDED_BULLETS = "\n\n".join([
    "• National Weather Service — Oceanside, CA 7-Day Forecast\n"
    "  Partly cloudy through Thursday with a persistent morning marine layer "
    "clearing by early afternoon. Highs in the mid-70s, lows near 60. Winds "
    "light and variable, becoming west 10-15 mph in the afternoon. No rain "
    "expected through the forecast period.\n"
    "  [https://forecast.weather.gov/MapClick.php?lat=33.1959&lon=-117.3795]",

    "• Oceanside, CA Weather Forecast | Weather Underground\n"
    "  Tomorrow: sunny skies with a high near 74°F and a low around 61°F. "
    "Humidity elevated in the morning due to coastal fog, dropping through "
    "the day. UV index moderate.\n"
    "  [https://www.wunderground.com/weather/us/ca/oceanside]",

    "• AccuWeather — Oceanside CA Daily Forecast\n"
    "  RealFeel high of 76°F tomorrow, mostly sunny after early clouds burn "
    "off. Chance of rain 0%. Winds from the WSW at 8 mph.\n"
    "  [https://www.accuweather.com/en/us/oceanside-ca/92054/daily-weather-forecast/331979]",
])

_GROUNDED_TOOL_RESULTS = [
    ToolResult(
        tool_name  = "web_search",
        parameters = f"query={_GROUNDED_QUERY!r}",
        result     = _GROUNDED_BULLETS,
    ),
]

_GROUNDED_WORKING_MEMORY = [
    Turn(role="user", content="Can you check the weather forecast for Oceanside CA this week?"),
    Turn(role="assistant", content="One moment — let me look that up for you."),
]

_prompt_builder = PromptBuilder()


def _build_grounded_prompt(instruction: str) -> str:
    """
    Assemble the Slot 3-7 user prompt via the real PromptBuilder.build(),
    with a populated [TOOL RESULTS] slot (web_search already run) and a
    [WORKING MEMORY] slot showing the search was requested and executed.
    Returns the user_prompt half of build()'s (system_prompt, user_prompt)
    tuple — the system half is not used here since every category in this
    script shares the same schema-augmented SYSTEM_PROMPT (see below) so the
    comparison across categories stays apples-to-apples.
    """
    _, user_prompt = _prompt_builder.build(
        instruction    = instruction,
        tool_results   = _GROUNDED_TOOL_RESULTS,
        working_memory = _GROUNDED_WORKING_MEMORY,
    )
    return user_prompt


def _gemma_prompt_for(instruction: str, category: str) -> str:
    if category == CAT_TOOL_FLAVORED_RESOLVED_GROUNDED:
        return _build_grounded_prompt(instruction)
    return instruction

# ---------------------------------------------------------------------------
# Control arm — local-only augmented keyword set. Never written to planner.py.
# Planner._any_whole_word is a @staticmethod: calling it with a locally-built
# frozenset does not touch (or even read) the module-level _WEB_SEARCH_KEYWORDS.
# ---------------------------------------------------------------------------

_CONTROL_ARM_ADDITIONS: frozenset[str] = frozenset({"look up", "look into", "find out"})


def _control_arm_would_fire(lowered_instruction: str) -> bool:
    augmented = Planner._WEB_SEARCH_KEYWORDS | _CONTROL_ARM_ADDITIONS
    return bool(Planner._any_whole_word(augmented, lowered_instruction))


# ---------------------------------------------------------------------------
# Gemma output parsing — strict contract: {"tool_call": {...}} or {"tool_call": null}
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _parse_gemma_output(raw: str) -> tuple[str | None, dict | None, bool]:
    """
    Returns (tool_name_or_None, arguments_dict_or_None, malformed).

    malformed=True whenever the text does not parse as JSON, or does not
    conform to the {"tool_call": null | {"name": ..., "arguments": {...}}}
    contract, or names a tool outside KNOWN_TOOL_NAMES.
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
# The task names exactly five buckets: MATCH, CATCH, FALSE_POSITIVE,
# MALFORMED, NULL_CORRECT. Those five do not cover every combination of
# (Planner flagged?, Gemma flagged?, category ground truth) — specifically,
# a category that genuinely needs a tool (semantic_miss/keyword_clear) where
# Gemma *also* fails to propose one has no natural home in that list. Rather
# than force that case into a misleading bucket, it is reported as MISS. This
# is the only deviation from the requested taxonomy and is called out here
# and in the printed summary.

def _classify(category: str, planner_tools: list[str], gemma_tool: str | None, malformed: bool) -> str:
    if malformed:
        return "MALFORMED"

    expects_tool    = _EXPECTS_TOOL[category]
    planner_flagged = bool(planner_tools)
    gemma_flagged   = gemma_tool is not None

    if expects_tool:
        if gemma_flagged:
            return "MATCH" if planner_flagged else "CATCH"
        return "MISS"  # tool genuinely needed; Gemma also failed to propose one
    else:
        return "FALSE_POSITIVE" if gemma_flagged else "NULL_CORRECT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_FIELDNAMES = [
    "instruction", "category", "run",
    "planner_tools_to_call", "planner_priority",
    "gemma_raw_output", "gemma_parsed_tool", "gemma_parsed_args",
    "classification", "control_arm_would_fire",
]


def _existing_categories(path: Path) -> set[str]:
    """Categories already present in the CSV, so re-runs only add new ones."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["category"] for row in csv.DictReader(fh)}


def main() -> None:
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

    # memory_manager=None and embed_fn=None: every MemoryManager-dependent
    # priority (3b/3c/4) short-circuits to None inside planner.py itself, and
    # the Priority 3 semantic-intent gate skips (no embed_fn to call). No
    # MemoryManager instance is ever constructed or referenced by this script.
    planner = Planner(runtime=client, memory_manager=None, embed_fn=None)

    # Idempotent append: only run trials for categories not already in the
    # CSV, so a prior run's rows are never overwritten or duplicated.
    already_done = _existing_categories(OUTPUT_PATH)
    corpus_to_run = [(instr, cat) for instr, cat in CORPUS if cat not in already_done]

    if not corpus_to_run:
        print("Nothing new to run — every category in CORPUS is already present in the CSV.")
        return

    skipped = sorted(already_done)
    if skipped:
        print(f"Skipping categories already in {OUTPUT_PATH.name}: {skipped}\n")

    results: list[dict] = []
    total_trials = len(corpus_to_run) * RUNS_PER_INSTRUCTION
    trial_n = 0

    for instruction, category in corpus_to_run:
        lowered = instruction.lower()
        control_arm = _control_arm_would_fire(lowered)
        gemma_prompt = _gemma_prompt_for(instruction, category)

        plan = planner.route(instruction, {})
        planner_tools    = list(plan.tools_to_call)
        planner_priority = plan.priority

        print(f"=== [{category}] {instruction[:70]!r} ===")
        print(f"    Planner: tools_to_call={planner_tools} priority={planner_priority}  control_arm_would_fire={control_arm}")

        for run in range(1, RUNS_PER_INSTRUCTION + 1):
            trial_n += 1
            print(f"    run {run}/{RUNS_PER_INSTRUCTION}  (trial {trial_n}/{total_trials}) … ", end="", flush=True)

            raw = client.infer(
                prompt      = gemma_prompt,
                system      = SYSTEM_PROMPT,
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )

            gemma_tool, gemma_args, malformed = _parse_gemma_output(raw)
            classification = _classify(category, planner_tools, gemma_tool, malformed)

            print(f"gemma_tool={gemma_tool!r} malformed={malformed} → {classification}")

            results.append({
                "instruction":              instruction,
                "category":                 category,
                "run":                      run,
                "planner_tools_to_call":    ";".join(planner_tools),
                "planner_priority":         planner_priority,
                "gemma_raw_output":         raw,
                "gemma_parsed_tool":        gemma_tool or "",
                "gemma_parsed_args":        json.dumps(gemma_args) if gemma_args is not None else "",
                "classification":           classification,
                "control_arm_would_fire":   control_arm,
            })

        print()

    # -- Append to CSV (write header only if the file doesn't exist yet) --
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_PATH.exists()
    with OUTPUT_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for rec in results:
            writer.writerow(rec)
    print(f"Appended to: {OUTPUT_PATH}  (+{len(results)} rows)\n")

    # -- Summary table: classification counts by category, over the WHOLE
    #    CSV (prior rows + rows just appended), so previously-run categories
    #    still show up alongside the new one. ---------------------------
    with OUTPUT_PATH.open(encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    categories = list(dict.fromkeys(r["category"] for r in all_rows))
    classifications = ["MATCH", "CATCH", "FALSE_POSITIVE", "MALFORMED", "NULL_CORRECT", "MISS"]

    print("=" * 100)
    header = f"{'CATEGORY':<28}" + "".join(f"{c:>15}" for c in classifications) + f"{'TOTAL':>10}"
    print(header)
    print("-" * 100)
    for category in categories:
        cat_rows = [r for r in all_rows if r["category"] == category]
        counts = {c: sum(1 for r in cat_rows if r["classification"] == c) for c in classifications}
        row = f"{category:<28}" + "".join(f"{counts[c]:>15}" for c in classifications) + f"{len(cat_rows):>10}"
        print(row)
    print("-" * 100)
    overall = {c: sum(1 for r in all_rows if r["classification"] == c) for c in classifications}
    print(f"{'TOTAL':<28}" + "".join(f"{overall[c]:>15}" for c in classifications) + f"{len(all_rows):>10}")
    print("=" * 100)

    print(
        "\nNote: MISS is not one of the five requested buckets (MATCH/CATCH/"
        "FALSE_POSITIVE/MALFORMED/NULL_CORRECT). It covers the residual case "
        "of a semantic_miss/keyword_clear instruction where Gemma's shadow "
        "proposal ALSO failed to call a tool — a real outcome the requested "
        "taxonomy has no bucket for. See module docstring for rationale."
    )

    # -- Full (untruncated) raw outputs for any newly-run category --------
    new_categories = sorted({cat for _, cat in corpus_to_run})
    for category in new_categories:
        cat_results = [r for r in results if r["category"] == category]
        print(f"\n{'=' * 100}\nFULL RAW OUTPUTS — category={category!r} ({len(cat_results)} trials)\n{'=' * 100}")
        for r in cat_results:
            print(f"\n--- {r['instruction']!r}  run {r['run']}  classification={r['classification']} ---")
            print(r["gemma_raw_output"])


if __name__ == "__main__":
    main()
