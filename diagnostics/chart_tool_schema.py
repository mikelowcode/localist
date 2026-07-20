"""
chart_tool_schema.py

Candidate tool schema + system prompt contract for a `generate_chart` tool,
scoped deliberately narrow for a first reliability measurement against
gemma-4-e4b-it-4bit (4-bit quantized, oMLX). Three chart types only
(bar/line/pie), flat dataset shape, no colors/options/nested style objects.

Rationale for the narrow scope (see chat discussion): the existing
diagnostics/diag_shadow_toolcall.py measured JSON-tool-call reliability
against single-string-argument tools (web_search's `query`, fetch_url's
`url`). This schema is deliberately the smallest useful nested schema that
still resembles a real Chart.js config, so the MALFORMED/SCHEMA_INVALID
rate measured against it tells you where a 4B quantized model's ceiling is
for structured chart-argument generation — before deciding whether to
widen the schema (colors, multi-axis, stacked, options.*) or keep it this
narrow in production.

This module defines the schema + prompt contract only. It does not call
any runtime, dispatch any tool, or touch production code paths — same
read-only-diagnostic posture as diag_shadow_toolcall.py.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
# Deliberately flat: chart_type (enum of 3), title, labels (string array),
# datasets (array of {label, data}). No colors, no options, no stacking,
# no multi-axis. This is the minimum viable nested schema — one level of
# array-of-objects nesting, one level of array-of-numbers nesting within
# that. If this schema shows a high MALFORMED/SCHEMA_INVALID rate, a wider
# schema (real Chart.js config surface) will only be worse; if it shows a
# low rate, that's the signal to consider widening incrementally and
# re-measuring rather than jumping straight to the full Chart.js surface.

CHART_TOOL_SCHEMA: dict = {
    "name": "generate_chart",
    "description": (
        "Render a chart from structured data. Use this when the user asks "
        "to chart, plot, graph, or visualize data they have provided or "
        "referenced in the conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie"],
            },
            "title": {
                "type": "string",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Category labels — x-axis ticks for bar/line, slice names for pie.",
            },
            "datasets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "data": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["label", "data"],
                },
                "description": (
                    "One or more series. Each dataset's `data` array must be "
                    "the same length as `labels`. Pie charts should use "
                    "exactly one dataset."
                ),
            },
        },
        "required": ["chart_type", "labels", "datasets"],
    },
}

KNOWN_TOOL_NAMES: frozenset[str] = frozenset({CHART_TOOL_SCHEMA["name"]})

# ---------------------------------------------------------------------------
# System prompt — same envelope contract as diag_shadow_toolcall.py
# ({"tool_call": {...}} | {"tool_call": null}), so results are comparable
# and the parser can be reused with minimal changes.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You have access to the following tool:

{json.dumps(CHART_TOOL_SCHEMA, indent=2)}

When the user's instruction asks you to chart, plot, graph, or visualize
data, respond with ONLY a single JSON object of the form:
  {{"tool_call": {{"name": "generate_chart", "arguments": {{...}}}}}}

The "arguments" object must exactly match the parameters schema above:
- "chart_type" must be one of "bar", "line", "pie".
- "labels" must be an array of strings.
- "datasets" must be an array of objects, each with "label" (string) and
  "data" (array of numbers) whose length equals the length of "labels".

When no chart is requested, respond with ONLY:
  {{"tool_call": null}}

Rules:
- Output exactly one JSON object and nothing else.
- No prose, no explanation, no markdown code fences.
- Do not invent data the user did not provide or clearly reference."""


# ---------------------------------------------------------------------------
# Argument-level validation — deeper than the envelope check in
# diag_shadow_toolcall.py's _parse_gemma_output. That function only checks
# "does this look like a tool_call envelope"; this checks "are the
# arguments actually usable to render a chart" (right types, non-empty,
# labels/data length match per dataset). Both checks are needed: an
# envelope can parse fine as JSON while still being useless to a renderer
# (e.g. mismatched array lengths, non-numeric data).
# ---------------------------------------------------------------------------

def validate_chart_arguments(arguments: dict) -> list[str]:
    """
    Validate `arguments` against CHART_TOOL_SCHEMA's shape and internal
    consistency rules that a JSON-Schema validator alone won't catch
    (labels/data length agreement per dataset).

    Returns a list of human-readable problem strings. Empty list = valid.
    Never raises — a malformed/wrong-typed `arguments` value (e.g. a
    string instead of a dict) is reported as a problem, not an exception.
    """
    problems: list[str] = []

    if not isinstance(arguments, dict):
        return [f"arguments is not an object (got {type(arguments).__name__})"]

    chart_type = arguments.get("chart_type")
    if chart_type not in ("bar", "line", "pie"):
        problems.append(f"chart_type invalid or missing: {chart_type!r}")

    labels = arguments.get("labels")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        problems.append("labels missing, not an array, or contains non-strings")
        labels = None
    elif len(labels) == 0:
        problems.append("labels is an empty array")

    datasets = arguments.get("datasets")
    if not isinstance(datasets, list) or len(datasets) == 0:
        problems.append("datasets missing, not an array, or empty")
        datasets = []

    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            problems.append(f"datasets[{i}] is not an object")
            continue
        if not isinstance(ds.get("label"), str):
            problems.append(f"datasets[{i}].label missing or not a string")
        data = ds.get("data")
        if not isinstance(data, list) or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in data
        ):
            problems.append(f"datasets[{i}].data missing, not an array, or contains non-numbers")
        elif labels is not None and len(data) != len(labels):
            problems.append(
                f"datasets[{i}].data length ({len(data)}) != labels length ({len(labels)})"
            )

    if chart_type == "pie" and len(datasets) > 1:
        problems.append("pie chart_type should have exactly one dataset, got more than one")

    return problems
