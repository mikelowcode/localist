"""
diagnostics/chart_tool_schema_fewshot.py

Few-shot variant of chart_tool_schema.SYSTEM_PROMPT, built for
diag_shadow_chart_toolcall_v2.py's prompt-side lever. Same tool schema,
same envelope contract — the only change is worked examples added after
the schema, specifically targeting the "prose instead of null envelope"
failure mode observed in the first diagnostic run: on negative-control
instructions with nothing concrete to chart ("Summarize this article for
me.", "Explain what a bar chart is."), gemma-4-e4b-it-4bit answered in
plain prose instead of emitting {"tool_call": null} — content that was
arguably reasonable, but broke the strict envelope contract the
downstream parser requires.

The third example below is deliberately built in that same shape (an
instruction that invites an explanation) to show the model directly:
even here, the required output is the null envelope, never prose.

Does not change CHART_TOOL_SCHEMA or validate_chart_arguments — this
file only adds a prompt variant. See chart_tool_schema.py for the schema
itself and diag_shadow_chart_toolcall.py's first-run results for why this
variant exists.
"""

from __future__ import annotations

import json

from chart_tool_schema import CHART_TOOL_SCHEMA

_EXAMPLES = [
    (
        "Chart this: cats 12, dogs 20, birds 5",
        {
            "tool_call": {
                "name": "generate_chart",
                "arguments": {
                    "chart_type": "bar",
                    "title": "Pet Counts",
                    "labels": ["cats", "dogs", "birds"],
                    "datasets": [{"label": "Count", "data": [12, 20, 5]}],
                },
            }
        },
    ),
    (
        "Break this down as a pie chart: rent 50%, food 30%, other 20%",
        {
            "tool_call": {
                "name": "generate_chart",
                "arguments": {
                    "chart_type": "pie",
                    "title": "Budget Breakdown",
                    "labels": ["rent", "food", "other"],
                    "datasets": [{"label": "Share", "data": [50, 30, 20]}],
                },
            }
        },
    ),
    (
        "Explain what a pie chart is.",
        {"tool_call": None},
    ),
]

_EXAMPLES_TEXT = "\n\n".join(
    f"Instruction: {instr}\nResponse: {json.dumps(resp)}"
    for instr, resp in _EXAMPLES
)

SYSTEM_PROMPT_FEWSHOT = f"""You have access to the following tool:

{json.dumps(CHART_TOOL_SCHEMA, indent=2)}

When the user's instruction asks you to chart, plot, graph, or visualize
data, respond with ONLY a single JSON object of the form:
  {{"tool_call": {{"name": "generate_chart", "arguments": {{...}}}}}}

The "arguments" object must exactly match the parameters schema above:
- "chart_type" must be one of "bar", "line", "pie".
- "labels" must be an array of strings.
- "datasets" must be an array of objects, each with "label" (string) and
  "data" (array of numbers) whose length equals the length of "labels".

When no chart is requested — including when you would normally explain
something, ask a clarifying question, or say you need more information —
respond with ONLY:
  {{"tool_call": null}}

Never answer in prose under any circumstances, even when no chart is
warranted. The null envelope IS the correct response in that case, not a
fallback for when you're unsure what else to say.

Examples:

{_EXAMPLES_TEXT}

Rules:
- Output exactly one JSON object and nothing else.
- No prose, no explanation, no markdown code fences.
- Do not invent data the user did not provide or clearly reference."""
