"""
Ollama Cloud Prefix-Cache Probe
================================

Investigates whether Ollama Cloud does cross-request prefix-hash KV-cache
reuse, or whether its fast time-to-first-token is fully explained by
data-center GPU speed with no caching at all. Motivated by a live in-app
chat turn (chat_turns.id=580, 2026-07-18 18:31:02) that inferred "server-side
prefix caching is indeed active" from millisecond TTFT alone — a single
data point that's equally consistent with "no caching, just fast hardware".
See diagnostics/reports/ollama_cloud_prefix_cache_findings.md for the
write-up this script's output feeds.

Bypasses the rest of the app stack entirely (no ControllerAgent,
PromptBuilder, or OllamaRuntimeClient import) — talks to
http://localhost:11434/api/chat directly with `requests`, exactly the
transport OllamaRuntimeClient uses internally, so there is nothing in
between the measurement and the actual wire request.

Methodology
-----------
Condition A — repeated identical prefix, varying suffix. N calls share one
    fixed, byte-identical prefix block, each followed by a different short
    instruction. This is the shape a real growing conversation has.
Condition B — same total prompt length as A, but every call's content is
    freshly randomized — no repeated prefix across calls at all.
Condition C — the Condition-A prefix repeated after a multi-minute gap,
    to see whether any benefit in A survives a gap (cache TTL / eviction).

Metric caveat (a finding in itself, confirmed by a calibration call before
this script was written): Ollama Cloud's /api/chat response omits
`prompt_eval_duration`, `eval_duration`, and `load_duration` entirely —
fields a local Ollama daemon populates on every response. Only
`total_duration` (wall-clock-ish, ns) and `prompt_eval_count`/`eval_count`
(token counts, not durations) are present. So this script measures
wall-clock TTFT (time from request-sent to the first streamed NDJSON
line) via a local monotonic clock, plus the reported `total_duration`,
rather than the more precise `prompt_eval_duration` split the original
investigation prompt hoped to compare — that field simply isn't in the
Cloud response to compare.

If prefix caching is active: Condition A's TTFT should be roughly flat
regardless of the (large, fixed) prefix size, scaling only with the short
suffix — while Condition B's TTFT should scale with the full prompt length
on every single call, including the first. If no caching: both conditions
should show similar TTFT for the same total length, since prompt-eval cost
is a function of token count either way.

Run from the project root:
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_ollama_cloud_prefix_cache_probe.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
CLOUD_MODEL  = "gemma4:31b-cloud"
N_RUNS       = 5
NUM_PREDICT  = 8          # small on purpose — isolates prompt-eval-driven TTFT
                          # from generation time as much as possible
TEMPERATURE  = 0.0

# ~8,000-token fixed prefix (~32,000 chars at ~4 chars/token) — large enough
# that a real cache hit vs. miss should be visible in TTFT, small enough to
# keep the whole probe under a couple of minutes of wall time.
_FIXED_PREFIX = ("The quick brown fox jumps over the lazy dog. " * 700)[:32_000]


@dataclass
class CallResult:
    label:            str
    ttft_s:            float
    total_duration_ns: int | None
    prompt_eval_count: int | None
    eval_count:        int | None
    wall_s:            float


def _load_dictionary_words() -> list[str]:
    """
    Real English words (macOS /usr/share/dict/words), not random letter
    salad. Matters for this probe specifically: gibberish letter sequences
    tokenize far less efficiently under BPE than real English (each
    "word" fragments into more subword tokens), which silently inflates
    Condition B's token count relative to Condition A even when both are
    built to the same *character* length — confounding the very
    comparison this probe exists to make. Shuffled real words tokenize at
    a density much closer to natural prose while still guaranteeing no
    shared prefix across calls (the shuffle order is fresh every time).
    """
    path = "/usr/share/dict/words"
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [w.strip() for w in f if w.strip().isalpha()]


_DICTIONARY = _load_dictionary_words()


def _random_block(n_chars: int) -> str:
    """A block of shuffled real English words — no repeated substrings
    across calls, but tokenizes at roughly natural-prose density."""
    words = []
    length = 0
    while length < n_chars:
        w = random.choice(_DICTIONARY)
        words.append(w)
        length += len(w) + 1
    return " ".join(words)[:n_chars]


def _call(prompt: str, label: str) -> CallResult:
    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":    CLOUD_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   True,
            "options": {"num_predict": NUM_PREDICT, "temperature": TEMPERATURE},
        },
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()

    first_chunk_t: float | None = None
    last_line: dict = {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if first_chunk_t is None:
            first_chunk_t = time.time()
        last_line = json.loads(raw)

    t_end = time.time()
    if first_chunk_t is None:
        raise RuntimeError(f"[{label}] stream produced zero lines")

    return CallResult(
        label              = label,
        ttft_s             = first_chunk_t - t0,
        total_duration_ns  = last_line.get("total_duration"),
        prompt_eval_count  = last_line.get("prompt_eval_count"),
        eval_count         = last_line.get("eval_count"),
        wall_s             = t_end - t0,
    )


def _summarize(label: str, results: list[CallResult]) -> None:
    ttfts = [r.ttft_s for r in results]
    print(f"\n--- {label} ---")
    for r in results:
        print(
            f"  ttft={r.ttft_s:.3f}s  wall={r.wall_s:.3f}s  "
            f"total_duration={r.total_duration_ns and r.total_duration_ns/1e9:.3f}s  "
            f"prompt_tokens={r.prompt_eval_count}  gen_tokens={r.eval_count}"
        )
    print(
        f"  TTFT mean={statistics.mean(ttfts):.3f}s  "
        f"stdev={statistics.stdev(ttfts) if len(ttfts) > 1 else 0:.3f}s  "
        f"min={min(ttfts):.3f}s  max={max(ttfts):.3f}s"
    )


def main() -> None:
    health = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    models = [m["model"] for m in health.get("models", [])]
    if CLOUD_MODEL not in models:
        print(f"FAIL: {CLOUD_MODEL!r} not found in GET /api/tags — {models}")
        return
    print(f"OK: {CLOUD_MODEL!r} reachable via local Ollama daemon.\n")
    print(f"Fixed prefix length: {len(_FIXED_PREFIX)} chars (~{len(_FIXED_PREFIX)//4} tokens)")

    # -- Condition A: identical prefix, varying short suffix -----------------
    condition_a: list[CallResult] = []
    for i in range(N_RUNS):
        suffix = f"\n\nThis is call number {i}. Reply with just the number {i}."
        condition_a.append(_call(_FIXED_PREFIX + suffix, f"A[{i}]"))
        time.sleep(1)   # avoid hammering the endpoint back-to-back
    _summarize("Condition A — repeated identical prefix", condition_a)

    # -- Condition B: same total length, fully randomized each call ---------
    # Calibrated 2026-07-18: a real-word block of len(_FIXED_PREFIX)+80 chars
    # measured 7,443 prompt tokens vs. Condition A's 7,030 — the repeated
    # sentence in _FIXED_PREFIX compresses unusually well under BPE (exact
    # repeated substrings merge efficiently), so matching *tokens* rather
    # than characters needs a ~5.5% shorter char length here.
    target_len = int((len(_FIXED_PREFIX) + 80) * 0.945)
    condition_b: list[CallResult] = []
    for i in range(N_RUNS):
        condition_b.append(_call(_random_block(target_len), f"B[{i}]"))
        time.sleep(1)
    _summarize("Condition B — no shared prefix (randomized each call)", condition_b)

    # -- Condition C: same prefix, back-to-back vs. after a gap --------------
    print("\n--- Condition C — cache persistence across a gap ---")
    immediate = _call(_FIXED_PREFIX + "\n\nSay OK.", "C[immediate]")
    print(f"  immediate repeat: ttft={immediate.ttft_s:.3f}s")

    gap_s = 180
    print(f"  waiting {gap_s}s before repeating the same prefix...")
    time.sleep(gap_s)
    after_gap = _call(_FIXED_PREFIX + "\n\nSay OK.", "C[after_gap]")
    print(f"  after {gap_s}s gap:  ttft={after_gap.ttft_s:.3f}s")

    # -- Comparison ------------------------------------------------------
    mean_a = statistics.mean(r.ttft_s for r in condition_a)
    mean_b = statistics.mean(r.ttft_s for r in condition_b)
    print(f"\n=== Summary ===")
    print(f"Condition A mean TTFT (repeated prefix): {mean_a:.3f}s")
    print(f"Condition B mean TTFT (no shared prefix): {mean_b:.3f}s")
    print(f"Delta (B - A): {mean_b - mean_a:+.3f}s")
    print(
        "Note: both conditions use the same prompt LENGTH — a gap here that "
        "correlates specifically with prefix repetition (not just general "
        "variance) is the actual evidence this probe is built to isolate."
    )


if __name__ == "__main__":
    main()
