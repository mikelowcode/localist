"""
diagnostics/diag_ollama_thinking_fabrication_repro.py

READ-ONLY DIAGNOSTIC — does not modify any source file or database.

Pilot for the "Option B" scratchpad question: does re-enabling Ollama's
native `thinking` field (currently hardcoded off in ollama_runtime_client.py
— see the "think": False comment there, and §4.6.2 of
docs/architecture/04-planner-routing-model.md for why it was disabled)
change the rate of fabricated tool-call syntax the model emits on tools=[]
turns (conversational_agent.py's _is_fabricated_toolcall() guard, §8.8 Open
Item 11)?

Scope note: tool *selection* in this codebase is fully deterministic
(planner.py's P1-P6 ladder resolves tools_to_call before any inference
runs) — thinking cannot influence which tools get called. This script does
NOT test that. It tests the one inference-side, orchestration-adjacent,
already-instrumented failure mode that exists: the model inventing a fake
tool-call token on a turn where it was given no tools at all. This is a
narrower, honest substitute for "does thinking improve tool orchestration,"
not a claim that the two are the same thing.

Two conditions, same model, same prompts (sampled), same max_tokens:
  - think=False — current production behavior (baseline)
  - think=True  — the treatment being piloted

For each trial, three outcomes are tracked *separately* so they don't get
conflated:
  1. fabricated   — _is_fabricated_toolcall(content) is True
  2. empty        — content is empty (the known §4.6.2 budget-exhaustion
                     failure mode; expected to rise under think=True since
                     reasoning tokens draw from the same num_predict budget)
  3. neither       — an ordinary completion

Methodology mirrors diag_ollama_zero_content_repro.py: real
(system_prompt, user_prompt) via the actual PromptBuilder.build(), sent
directly to Ollama's /api/chat with `requests` (same transport
OllamaRuntimeClient uses internally) — no OllamaRuntimeClient construction,
no ControllerAgent, no MemoryManager, nothing written to SQLite.

Output
------
  diagnostics/ollama_thinking_fabrication_repro_results.csv — one row per trial
  stdout                                                     — progress + summary

Run from the project root (Ollama must already be reachable at
localhost:11434 with gemma4:e4b-mlx present):
    python3 diagnostics/diag_ollama_thinking_fabrication_repro.py
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

from prompt_builder import PromptBuilder          # noqa: E402
from conversational_agent import _is_fabricated_toolcall  # noqa: E402

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

MODEL            = "gemma4:e4b-mlx"   # the only model confirmed to use "thinking"
TEMPERATURE      = 0.30               # matches conversational_agent.py's default
MAX_TOKENS       = 1024               # matches conversational_agent.py's default max_tokens
N_PER_CONDITION  = 20
CALL_DELAY_S     = 0.5
REQUEST_TIMEOUT  = 180                # generous — think=True calls may run long

_BUILDER = PromptBuilder()

# Ungrounded, search-shaped instructions — no tool_results, no RAG — modeled
# on the topical shape of the 7 live-observed fabrication strings in
# conversational_agent.py's _REFERENCE_FABRICATIONS (tech/product news
# lookups the model has no training-knowledge answer for and no tool to
# reach for, which is the documented trigger condition).
_UNGROUNDED_SEARCH_QUESTIONS = [
    "Did Apple announce a price hike on the MacBook Neo or iPad this week?",
    "What are Microsoft's next-generation in-house AI models called?",
    "When was Microsoft's first formal investment in OpenAI?",
    "What's the latest news on LangSmith Engine?",
    "What did Apple announce at its most recent hardware event?",
    "Is there a new version of the Claude API released this week?",
    "What's the current status of the Fed's interest rate decision?",
    "What were the top Hacker News stories in the last day?",
    "Did any major AI lab announce a new model release recently?",
    "What's the latest on Microsoft's AI model roadmap?",
]


@dataclass
class Trial:
    think:          bool
    instruction:    str
    prompt_chars:   int
    content_chars:  int
    thinking_chars: int
    fabricated:     bool
    empty:          bool
    done_reason:    str | None
    eval_count:     int | None
    wall_s:         float
    error:          str = ""


def _build_prompt() -> tuple[str, str, str]:
    """Returns (system, prompt, instruction) for an ungrounded P6-shaped turn."""
    instruction = random.choice(_UNGROUNDED_SEARCH_QUESTIONS)
    now = datetime.now().astimezone()
    system, prompt = _BUILDER.build(instruction=instruction, current_datetime=now)
    return system, prompt, instruction


def _call(system: str, prompt: str, think: bool) -> tuple[str, str, dict, float]:
    """Same transport OllamaRuntimeClient.infer_stream() uses internally.
    Returns (content, thinking, terminal_done_line, wall_seconds)."""
    t0 = time.time()
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "think": think,
            "options": {"num_predict": MAX_TOKENS, "temperature": TEMPERATURE},
        },
        timeout=REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()

    content_parts:  list[str] = []
    thinking_parts: list[str] = []
    last_line: dict = {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        data = json.loads(raw)
        if data.get("error"):
            raise RuntimeError(f"Ollama mid-stream error: {data['error']}")
        msg = data.get("message", {})
        if msg.get("content"):
            content_parts.append(msg["content"])
        if msg.get("thinking"):
            thinking_parts.append(msg["thinking"])
        if data.get("done"):
            last_line = data
            break
    wall_s = time.time() - t0
    return "".join(content_parts), "".join(thinking_parts), last_line, wall_s


def _run_trial(think: bool) -> Trial:
    system, prompt, instruction = _build_prompt()
    try:
        content, thinking, meta, wall_s = _call(system, prompt, think)
        return Trial(
            think=think, instruction=instruction,
            prompt_chars=len(system) + len(prompt),
            content_chars=len(content), thinking_chars=len(thinking),
            fabricated=_is_fabricated_toolcall(content),
            empty=(len(content.strip()) == 0),
            done_reason=meta.get("done_reason"), eval_count=meta.get("eval_count"),
            wall_s=wall_s,
        )
    except Exception as exc:
        return Trial(
            think=think, instruction=instruction, prompt_chars=0,
            content_chars=-1, thinking_chars=-1, fabricated=False, empty=False,
            done_reason=None, eval_count=None, wall_s=0.0, error=str(exc),
        )


def _print_trial(i: int, total: int, t: Trial) -> None:
    flags = ""
    if t.fabricated:
        flags += "  <<< FABRICATED TOOLCALL"
    if t.empty:
        flags += "  <<< EMPTY CONTENT"
    if t.error:
        print(f"  [{i}/{total}] think={t.think} ERROR: {t.error}")
    else:
        print(f"  [{i}/{total}] think={t.think} content_chars={t.content_chars} "
              f"thinking_chars={t.thinking_chars} eval_count={t.eval_count} "
              f"done_reason={t.done_reason} wall={t.wall_s:.2f}s{flags}")
    sys.stdout.flush()


def main() -> None:
    tags = requests.get(OLLAMA_TAGS_URL, timeout=10).json()
    available = [m["model"] for m in tags.get("models", [])]
    if MODEL not in available:
        print(f"FAIL: {MODEL!r} not found in GET /api/tags — {available}")
        return
    print(f"OK: {MODEL!r} reachable via local Ollama daemon.\n")

    conditions = [False, True]
    trials: list[Trial] = []

    print(f"=== think=False vs think=True, N={N_PER_CONDITION} each, "
          f"model={MODEL}, temp={TEMPERATURE}, max_tokens={MAX_TOKENS} ===")
    total = len(conditions) * N_PER_CONDITION
    i = 0
    for think in conditions:
        for _ in range(N_PER_CONDITION):
            i += 1
            t = _run_trial(think)
            trials.append(t)
            _print_trial(i, total, t)
            time.sleep(CALL_DELAY_S)

    # -- Write CSV ------------------------------------------------------
    out_path = Path(__file__).resolve().parent / "ollama_thinking_fabrication_repro_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "think", "instruction", "prompt_chars", "content_chars",
            "thinking_chars", "fabricated", "empty", "done_reason",
            "eval_count", "wall_s", "error",
        ])
        for t in trials:
            writer.writerow([
                t.think, t.instruction, t.prompt_chars, t.content_chars,
                t.thinking_chars, t.fabricated, t.empty, t.done_reason,
                t.eval_count, f"{t.wall_s:.3f}", t.error,
            ])
    print(f"\nWrote {len(trials)} trial rows to {out_path}")

    # -- Summary ----------------------------------------------------------
    print("\n=== Summary ===")
    for think in conditions:
        cell = [t for t in trials if t.think == think and not t.error]
        errors = [t for t in trials if t.think == think and t.error]
        n = len(cell)
        if n == 0:
            print(f"think={think}: no valid trials (errors={len(errors)})")
            continue
        n_fab = sum(1 for t in cell if t.fabricated)
        n_empty = sum(1 for t in cell if t.empty)
        non_empty = [t for t in cell if not t.empty]
        n_fab_of_nonempty = sum(1 for t in non_empty if t.fabricated)
        mean_wall = sum(t.wall_s for t in cell) / n
        mean_thinking_chars = sum(t.thinking_chars for t in cell) / n
        print(f"\nthink={think}  (N={n}, errors={len(errors)})")
        print(f"  fabrication rate (of all trials):        {n_fab}/{n} = {n_fab/n:.1%}")
        print(f"  empty-content rate:                       {n_empty}/{n} = {n_empty/n:.1%}")
        print(f"  fabrication rate (of non-empty trials):  "
              f"{n_fab_of_nonempty}/{len(non_empty)} = "
              f"{(n_fab_of_nonempty/len(non_empty) if non_empty else 0):.1%}")
        print(f"  mean wall time:                            {mean_wall:.2f}s")
        print(f"  mean thinking_chars:                       {mean_thinking_chars:.0f}")

    print(
        "\nNote: fabrication rate 'of all trials' conflates two failure modes "
        "if empty-content rate rises under think=True — that's expected per "
        "§4.6.2 (thinking draws from the same num_predict budget). The "
        "'of non-empty trials' row is the fairer comparison for the "
        "fabrication-specific question this pilot is actually about."
    )


if __name__ == "__main__":
    main()
