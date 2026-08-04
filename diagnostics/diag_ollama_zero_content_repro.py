"""
diagnostics/diag_ollama_zero_content_repro.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Phase 2 of the still-open §4.6.2 zero-content-Ollama-completion
investigation (docs/architecture/04-planner-routing-model.md,
LOCALIST-Architecture.md index row 4). Phase 1 (2026-08-04) added passive
instrumentation to OllamaRuntimeClient (captures done_reason/eval_count/
prompt_eval_count/total_duration on the terminal NDJSON line, previously
discarded, and warns on zero-content). This script exists to actually try
to trigger a reproduction and correlate it against candidate factors, per
that section's "suggested future scope" item (2) — no diagnostic run had
been done before this one; both confirmed occurrences to date are single
anecdotal data points.

Two confirmed live occurrences to date:
  - 2026-07-17: bare, ungrounded query naming a specific date the Planner's
    P3 gate didn't recognize as needing a tool call — the documented
    trigger condition ("no tool grounding for a query it structurally
    can't answer from training alone").
  - 2026-07-30: recurrence with REAL, populated hacker_news_search tool
    grounding already present in the prompt (title/points/comment count/
    real top-comment text) — contradicting that trigger condition, since
    the model only needed to summarize, not answer from bare training
    knowledge.
Both were temperature=0.30, both OllamaRuntimeClient / gemma4:31b-cloud,
both a validly-terminated "done": true stream with zero content, ~1s round
trip, no timeout, no exception.

Methodology
-----------
Builds real (system_prompt, user_prompt) pairs via the actual
PromptBuilder.build() — not hand-rolled strings — so prompt shape matches
production exactly, then sends them directly to Ollama's /api/chat via
`requests` (same transport OllamaRuntimeClient uses internally). Every
trial's terminal NDJSON line is captured (done_reason, eval_count,
prompt_eval_count, total_duration) — not just zero-content ones — because
establishing a baseline distribution needs the full population, not only
the anecdotes.

Four prompt SHAPEs:
  A  ungrounded_short      — bare instruction naming a specific date, no
                              tool grounding. Mirrors the 2026-07-17 trigger.
  B  grounded_hacker_news  — tool_results=[hacker_news_search] with
                              realistic title/points/comment-count/
                              top-comment text. Mirrors the 2026-07-30
                              recurrence exactly (same tool identity).
  C  grounded_web_search   — tool_results=[web_search], different tool
                              identity, to test whether B's result (if any)
                              is HN-specific or general to any populated
                              grounding.
  D  ungrounded_long       — no tool grounding, but a long pasted-document
                              instruction, to isolate raw prompt length
                              from grounding presence.

Crossed with MODEL (gemma4:31b-cloud, gemma4:e4b-mlx — the only two chat
models present in this machine's `GET /api/tags`, giving a cloud vs. local
comparison the doc flagged as unconfirmed) at the baseline temperature
(0.30, matching both real incidents exactly), plus a secondary
temperature sweep (0.0, 0.70) on shapes A and B specifically, cloud-only
(that's the backend actually in production use per LOCALIST_CHAT_MODEL).

Isolation constraint: only PromptBuilder.build() and raw `requests` calls
to the live Ollama endpoint are used. No ControllerAgent, no
MCPToolDispatcher, no MemoryManager, nothing written to SQLite or working
memory, no OllamaRuntimeClient construction — a pure prompt-shape ->
completion probe, same isolation discipline as diag_shadow_toolcall.py.

Output
------
  diagnostics/ollama_zero_content_repro_results.csv  — one row per trial
  stdout                                              — running progress +
                                                         summary + repro callouts

Run from the project root (Ollama must already be reachable at
localhost:11434 with both models present):
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_ollama_zero_content_repro.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from prompt_builder import PromptBuilder, ToolResult  # noqa: E402

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

MODELS           = ["gemma4:31b-cloud", "gemma4:e4b-mlx"]
BASELINE_TEMP    = 0.30   # matches both real incidents exactly
SECONDARY_TEMPS  = [0.0, 0.70]
MAX_TOKENS       = 200
N_PRIMARY        = 6      # per (shape, model) cell at baseline temp
N_SECONDARY      = 5      # per (shape, temp) cell, cloud-only temp sweep
CALL_DELAY_S     = 0.5
REQUEST_TIMEOUT  = 120

_BUILDER = PromptBuilder()

_UNGROUNDED_SHORT_QUESTIONS = [
    "What happened in the news on August 3rd, 2026?",
    "What were the top Hacker News stories yesterday?",
    "What's the weather forecast for tomorrow in San Francisco?",
    "Who won the game last night?",
    "What's the latest version of the Claude API as of this week?",
    "What did the Fed announce at its meeting this morning?",
]

_HN_ARTICLES = [
    dict(title="Show HN: A local-first note-taking app in Rust", points=342,
         comments=128, top_comment="This is impressively fast — I switched "
         "from Obsidian and haven't looked back. The sync story is the "
         "only thing I'm still nervous about long-term."),
    dict(title="The end of the free tier: why cloud costs are eating startups alive",
         points=567, comments=310, top_comment="We hit this exact wall last "
         "year. Moved half our workload back to bare metal and cut spend by 60%."),
    dict(title="Ask HN: What's your favorite underrated programming language feature?",
         points=210, comments=440, top_comment="Erlang's let-it-crash "
         "philosophy changed how I think about error handling entirely."),
    dict(title="A new proof of the Collatz conjecture for a restricted case",
         points=890, comments=95, top_comment="The restriction to a bounded "
         "starting range does a lot of the heavy lifting here, but still a "
         "nice technique."),
    dict(title="Why we rewrote our build system in Zig", points=433,
         comments=176, top_comment="Curious what your CI cold-cache build "
         "times look like now versus before — that's usually where these "
         "rewrites pay off most."),
    dict(title="Show HN: I built a CLI that turns any repo into a podcast summary",
         points=198, comments=61, top_comment="Tried this on our monorepo "
         "and it correctly identified the three most-changed modules "
         "without being told anything about the codebase."),
]

_WEB_RESULTS = [
    dict(title="Apple Silicon M5 benchmarks leak ahead of announcement",
         source="theverge.com", summary="Early benchmark results suggest a "
         "roughly 20% single-core improvement over the M4 generation, with "
         "GPU gains concentrated in ray-tracing workloads."),
    dict(title="Local LLM inference frameworks compared: MLX vs llama.cpp vs Ollama",
         source="arstechnica.com", summary="A hands-on comparison across "
         "throughput, memory footprint, and setup complexity on Apple "
         "Silicon hardware, with MLX pulling ahead on M-series unified memory."),
    dict(title="New study finds remote work productivity gains plateau after year two",
         source="hbr.org", summary="Researchers tracked distributed teams "
         "across 40 companies and found initial productivity gains largely "
         "leveled off, with communication overhead the most cited factor."),
]

_LONG_FILLER = (
    "In the following report, our team reviewed quarterly performance "
    "across four regional offices, focusing on customer retention, "
    "support ticket resolution time, and net revenue retention. "
) * 40   # ~2,800 chars of filler, no tool grounding


@dataclass
class Trial:
    shape:             str
    model:             str
    temperature:       float
    prompt_chars:      int
    content_chars:     int
    done_reason:       str | None
    eval_count:        int | None
    prompt_eval_count: int | None
    total_duration_ns: int | None
    wall_s:            float
    zero_content:      bool
    error:             str = ""


def _build_prompt(shape: str) -> tuple[str, str]:
    now = datetime.now().astimezone()

    if shape == "A_ungrounded_short":
        instruction = random.choice(_UNGROUNDED_SHORT_QUESTIONS)
        return _BUILDER.build(instruction=instruction, current_datetime=now)

    if shape == "B_grounded_hacker_news":
        a = random.choice(_HN_ARTICLES)
        result_text = (
            f"Title: {a['title']}\n"
            f"Points: {a['points']}  Comments: {a['comments']}\n"
            f"Top comment: {a['top_comment']}"
        )
        tool_results = [ToolResult(
            tool_name="hacker_news_search",
            parameters=f"query='{a['title'][:40]}'",
            result=result_text,
        )]
        instruction = ("What's the top story on Hacker News right now, and "
                        "what are people saying about it?")
        return _BUILDER.build(instruction=instruction, current_datetime=now,
                               tool_results=tool_results)

    if shape == "C_grounded_web_search":
        w = random.choice(_WEB_RESULTS)
        result_text = f"{w['title']} — {w['source']} — {w['summary']}"
        tool_results = [ToolResult(
            tool_name="web_search",
            parameters=f"query='{w['title'][:40]}'",
            result=result_text,
        )]
        instruction = "Search the web for the latest on this and summarize it for me."
        return _BUILDER.build(instruction=instruction, current_datetime=now,
                               tool_results=tool_results)

    if shape == "D_ungrounded_long":
        instruction = _LONG_FILLER + "\n\nSummarize the above report in three sentences."
        return _BUILDER.build(instruction=instruction, current_datetime=now)

    raise ValueError(f"unknown shape: {shape}")


def _call(system: str, prompt: str, model: str, temperature: float) -> tuple[str, dict, float]:
    """Same transport OllamaRuntimeClient.infer_stream() uses internally.
    Returns (content, terminal_done_line, wall_seconds)."""
    t0 = time.time()
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"num_predict": MAX_TOKENS, "temperature": temperature},
        },
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()

    content_parts: list[str] = []
    last_line: dict = {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        data = json.loads(raw)
        if data.get("error"):
            raise RuntimeError(f"Ollama mid-stream error: {data['error']}")
        c = data.get("message", {}).get("content", "")
        if c:
            content_parts.append(c)
        if data.get("done"):
            last_line = data
            break
    wall_s = time.time() - t0
    return "".join(content_parts), last_line, wall_s


def _run_trial(shape: str, model: str, temperature: float) -> Trial:
    system, prompt = _build_prompt(shape)
    try:
        content, meta, wall_s = _call(system, prompt, model, temperature)
        return Trial(
            shape=shape, model=model, temperature=temperature,
            prompt_chars=len(system) + len(prompt),
            content_chars=len(content),
            done_reason=meta.get("done_reason"),
            eval_count=meta.get("eval_count"),
            prompt_eval_count=meta.get("prompt_eval_count"),
            total_duration_ns=meta.get("total_duration"),
            wall_s=wall_s,
            zero_content=(len(content) == 0),
        )
    except Exception as exc:
        return Trial(
            shape=shape, model=model, temperature=temperature,
            prompt_chars=len(system) + len(prompt), content_chars=-1,
            done_reason=None, eval_count=None, prompt_eval_count=None,
            total_duration_ns=None, wall_s=time.time(), zero_content=False,
            error=str(exc),
        )


def _print_trial(i: int, total: int, t: Trial) -> None:
    flag = "  <<< ZERO-CONTENT REPRO" if t.zero_content else ""
    if t.error:
        print(f"  [{i}/{total}] {t.shape} model={t.model} temp={t.temperature} "
              f"ERROR: {t.error}")
    else:
        print(f"  [{i}/{total}] {t.shape} model={t.model} temp={t.temperature} "
              f"prompt_chars={t.prompt_chars} content_chars={t.content_chars} "
              f"eval_count={t.eval_count} done_reason={t.done_reason} "
              f"wall={t.wall_s:.2f}s{flag}")
    sys.stdout.flush()


def main() -> None:
    tags = requests.get(OLLAMA_TAGS_URL, timeout=10).json()
    available = [m["model"] for m in tags.get("models", [])]
    for m in MODELS:
        if m not in available:
            print(f"FAIL: {m!r} not found in GET /api/tags — {available}")
            return
    print(f"OK: {MODELS} reachable via local Ollama daemon.\n")

    shapes = ["A_ungrounded_short", "B_grounded_hacker_news",
              "C_grounded_web_search", "D_ungrounded_long"]

    trials: list[Trial] = []

    print(f"=== Primary matrix: {shapes} x {MODELS} @ temp={BASELINE_TEMP}, "
          f"N={N_PRIMARY} each ===")
    primary_cells = [(s, m) for s in shapes for m in MODELS]
    total_primary = len(primary_cells) * N_PRIMARY
    i = 0
    for shape, model in primary_cells:
        for _ in range(N_PRIMARY):
            i += 1
            t = _run_trial(shape, model, BASELINE_TEMP)
            trials.append(t)
            _print_trial(i, total_primary, t)
            time.sleep(CALL_DELAY_S)

    print(f"\n=== Secondary temperature sweep: ['A_ungrounded_short', "
          f"'B_grounded_hacker_news'] x temps={SECONDARY_TEMPS}, "
          f"cloud-only, N={N_SECONDARY} each ===")
    secondary_shapes = ["A_ungrounded_short", "B_grounded_hacker_news"]
    secondary_cells = [(s, t) for s in secondary_shapes for t in SECONDARY_TEMPS]
    total_secondary = len(secondary_cells) * N_SECONDARY
    j = 0
    for shape, temp in secondary_cells:
        for _ in range(N_SECONDARY):
            j += 1
            t = _run_trial(shape, "gemma4:31b-cloud", temp)
            trials.append(t)
            _print_trial(j, total_secondary, t)
            time.sleep(CALL_DELAY_S)

    # -- Write CSV ------------------------------------------------------
    out_path = Path(__file__).resolve().parent / "ollama_zero_content_repro_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shape", "model", "temperature", "prompt_chars", "content_chars",
            "done_reason", "eval_count", "prompt_eval_count",
            "total_duration_ns", "wall_s", "zero_content", "error",
        ])
        for t in trials:
            writer.writerow([
                t.shape, t.model, t.temperature, t.prompt_chars, t.content_chars,
                t.done_reason, t.eval_count, t.prompt_eval_count,
                t.total_duration_ns, f"{t.wall_s:.3f}", t.zero_content, t.error,
            ])
    print(f"\nWrote {len(trials)} trial rows to {out_path}")

    # -- Summary ----------------------------------------------------------
    print("\n=== Summary ===")
    valid = [t for t in trials if not t.error]
    errors = [t for t in trials if t.error]
    repros = [t for t in valid if t.zero_content]
    print(f"Total trials: {len(trials)}  (errors: {len(errors)}, valid: {len(valid)})")
    print(f"Zero-content repros: {len(repros)} / {len(valid)}")

    if repros:
        print("\n--- REPRO DETAIL ---")
        for t in repros:
            print(f"  shape={t.shape} model={t.model} temp={t.temperature} "
                  f"eval_count={t.eval_count} done_reason={t.done_reason} "
                  f"prompt_eval_count={t.prompt_eval_count} "
                  f"prompt_chars={t.prompt_chars} wall_s={t.wall_s:.2f}")
        eval_counts = {t.eval_count for t in repros}
        print(f"\neval_count values across repros: {eval_counts}")
        if eval_counts == {0}:
            print("All repros have eval_count=0 — Ollama never generated a "
                  "token at all (upstream of generation, not a parsing/"
                  "transport drop).")
        elif 0 not in eval_counts:
            print("No repro has eval_count=0 — tokens WERE generated but "
                  "never surfaced as content. Different failure class than "
                  "the eval_count=0 case; parsing/transport, not generation.")
        else:
            print("Repros split between eval_count=0 and eval_count>0 — "
                  "likely two distinct causes producing the same symptom.")
    else:
        print("\nNo reproduction across this run. Negative result — does not "
              "rule out the bug, but this sweep's specific shapes/temps/"
              "models did not trigger it. See §4.6.2 for what to try next.")

    if errors:
        print("\n--- ERRORS (excluded from repro-rate denominator) ---")
        for t in errors:
            print(f"  shape={t.shape} model={t.model} temp={t.temperature}: {t.error}")

    print("\n--- Per-cell zero-content rate ---")
    cells: dict[tuple[str, str, float], list[Trial]] = {}
    for t in valid:
        cells.setdefault((t.shape, t.model, t.temperature), []).append(t)
    for key in sorted(cells):
        cell_trials = cells[key]
        n_zero = sum(1 for t in cell_trials if t.zero_content)
        print(f"  {key}: {n_zero}/{len(cell_trials)} zero-content")


if __name__ == "__main__":
    main()
