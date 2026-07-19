"""
Ollama num_ctx Honored Probe
=============================

Investigates whether Ollama Cloud actually applies the `options.num_ctx`
value OllamaRuntimeClient now sends on every request (added 2026-07-18
alongside context_profile.py's CLOUD_PROFILE=100,000 / LOCAL_PROFILE=8,000),
or silently clamps to some smaller server-side default — which would mean
the tail of a long conversation quietly falls off a cliff the app has no
visibility into, while still paying full cost to send it. See
diagnostics/reports/ollama_cloud_num_ctx_findings.md for the write-up this
script's output feeds.

Bypasses the rest of the app stack (no ControllerAgent/PromptBuilder/
OllamaRuntimeClient import) — raw `requests` calls to
http://localhost:11434/api/chat, the same transport OllamaRuntimeClient
uses internally, with `options.num_ctx` set explicitly per call exactly as
that client does.

Two independent checks, run in this order because they answer different
questions and shouldn't be conflated:

1. Metadata check — one call per prompt-length step, no needle, comparing
   Ollama's own reported `prompt_eval_count` against an independently
   predicted token count (via the same chars-per-token ratio established
   empirically in the prior prefix-cache probe: shuffled real dictionary
   words average ~4.24 chars/token). If `prompt_eval_count` keeps scaling
   linearly with input across all lengths tested, nothing is being dropped
   before evaluation. If it plateaus at some value regardless of larger
   input, that is direct, count-based proof of truncation — independent of
   whether the model can still *reason* over whatever it did see.

2. Needle-in-haystack behavioral check — a short, distinctive, unguessable
   fact planted at a controlled position (start, or ~70% through) inside a
   long filler block, asked back immediately after. Recall failure at a
   given length + position combination is evidence of either truncation
   (content silently dropped, most likely to show up first for an early
   needle at a large total length — i.e. front-truncation, if oldest
   content is dropped first) or "lost in the middle" degradation (content
   present but reasoning-over-length quality drops) — this script records
   position and length independently so the two are not conflated in the
   write-up.

A local-backend cross-check (gemma4:e4b-mlx, LOCAL_PROFILE's num_ctx=8000,
explicitly requested via the same options.num_ctx mechanism) is run at a
scaled-down grid bracketing 8K tokens, as a sanity check that this
methodology actually detects a *known* truncation boundary before trusting
a "cloud passes cleanly" result.

Run from the project root:
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_ollama_num_ctx_probe.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

CLOUD_MODEL     = "gemma4:31b-cloud"
CLOUD_NUM_CTX   = 100_000   # CLOUD_PROFILE.total_context_tokens
LOCAL_MODEL     = "gemma4:e4b-mlx"
LOCAL_NUM_CTX   = 8_000     # LOCAL_PROFILE.total_context_tokens

# Empirically established in diag_ollama_cloud_prefix_cache_probe.py:
# shuffled real-dictionary-word text averages ~4.24 chars/token against
# this same model family. Used only to convert a *target token count* into
# a char length to generate — not treated as exact, hence the metadata
# check (§1) comparing against Ollama's own reported count rather than
# trusting this ratio blindly.
CHARS_PER_TOKEN = 4.24

NEEDLE   = "The secret verification code is zebra-4471-quartz."
QUESTION = ("\n\nWhat is the secret verification code mentioned above? "
            "Reply with only the code and nothing else.")


@dataclass
class CallResult:
    label:             str
    target_tokens:     int
    prompt_eval_count: int | None
    ttft_s:            float
    total_duration_ns: int | None
    response_text:     str


def _load_dictionary_words() -> list[str]:
    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as f:
        return [w.strip() for w in f if w.strip().isalpha()]


_DICTIONARY = _load_dictionary_words()


def _filler(n_chars: int) -> str:
    words = []
    length = 0
    while length < n_chars:
        w = random.choice(_DICTIONARY)
        words.append(w)
        length += len(w) + 1
    return " ".join(words)[:n_chars]


def _build_haystack(target_tokens: int, needle_position: str) -> str:
    """
    needle_position: "start" (needle is the first sentence) or "late"
    (needle inserted at ~70% through the filler).
    """
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    filler_chars = max(target_chars - len(NEEDLE) - len(QUESTION), 100)

    if needle_position == "start":
        body = NEEDLE + " " + _filler(filler_chars)
    elif needle_position == "late":
        head_chars = int(filler_chars * 0.70)
        tail_chars = filler_chars - head_chars
        body = _filler(head_chars) + " " + NEEDLE + " " + _filler(tail_chars)
    else:
        raise ValueError(needle_position)

    return body + QUESTION


def _call(
    model:        str,
    prompt:       str,
    num_ctx:      int,
    label:        str,
    target_tokens: int,
    num_predict:  int = 20,
) -> CallResult:
    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":    model,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   True,
            "options": {
                "num_predict": num_predict,
                "temperature": 0.0,
                "num_ctx":     num_ctx,
            },
        },
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()

    first_chunk_t: float | None = None
    last_line: dict = {}
    text_parts: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if first_chunk_t is None:
            first_chunk_t = time.time()
        last_line = json.loads(raw)
        chunk = last_line.get("message", {}).get("content", "")
        if chunk:
            text_parts.append(chunk)

    if first_chunk_t is None:
        raise RuntimeError(f"[{label}] stream produced zero lines")

    return CallResult(
        label              = label,
        target_tokens      = target_tokens,
        prompt_eval_count  = last_line.get("prompt_eval_count"),
        ttft_s             = first_chunk_t - t0,
        total_duration_ns  = last_line.get("total_duration"),
        response_text      = "".join(text_parts),
    )


def _needle_found(response_text: str) -> bool:
    return "zebra-4471-quartz" in response_text.replace(" ", "").lower() \
        or "zebra-4471-quartz" in response_text.lower()


# ---------------------------------------------------------------------------
# 1. Metadata check — prompt_eval_count vs. predicted token count
# ---------------------------------------------------------------------------

def run_metadata_check(model: str, num_ctx: int, lengths: list[int], label: str) -> None:
    print(f"\n=== {label}: prompt_eval_count vs. target token count (num_ctx={num_ctx}) ===")
    print(f"{'target_tokens':>14} {'prompt_eval_count':>18} {'ratio':>8} {'ttft_s':>8}")
    for target in lengths:
        prompt = _build_haystack(target, "start")   # content shape doesn't matter here
        r = _call(model, prompt, num_ctx, f"{label}-meta-{target}", target, num_predict=5)
        ratio = (r.prompt_eval_count / target) if r.prompt_eval_count else 0.0
        print(f"{target:>14} {str(r.prompt_eval_count):>18} {ratio:>8.3f} {r.ttft_s:>8.2f}")


# ---------------------------------------------------------------------------
# 1b. num_ctx sweep at FIXED prompt length — the actual causal test.
#
# The metadata check above holds num_ctx fixed at CLOUD_NUM_CTX while
# varying prompt length — that only shows the server *can* handle up to
# 120K tokens of input; it never actually varies num_ctx, so on its own it
# cannot show whether num_ctx does anything at all. This sweep holds a
# single, fixed-length haystack constant and varies only the requested
# num_ctx value, comparing prompt_eval_count and needle recall (needle
# placed at "start", the position most exposed by front-truncation, the
# most common truncation strategy — drop oldest, keep most recent).
# ---------------------------------------------------------------------------

def run_num_ctx_sweep(model: str, fixed_tokens: int, num_ctx_values: list[int], label: str) -> None:
    print(f"\n=== {label}: num_ctx sweep at fixed {fixed_tokens}-token prompt, needle at start ===")
    prompt = _build_haystack(fixed_tokens, "start")
    print(f"{'num_ctx':>10} {'prompt_eval_count':>18} {'needle_found':>13} {'ttft_s':>8}")
    for num_ctx in num_ctx_values:
        r = _call(model, prompt, num_ctx, f"{label}-sweep-{num_ctx}", fixed_tokens)
        found = _needle_found(r.response_text)
        print(
            f"{num_ctx:>10} {str(r.prompt_eval_count):>18} {str(found):>13} {r.ttft_s:>8.2f}  "
            f"response={r.response_text.strip()[:60]!r}"
        )


# ---------------------------------------------------------------------------
# 2. Needle-in-haystack behavioral check
# ---------------------------------------------------------------------------

def run_needle_grid(
    model:      str,
    num_ctx:    int,
    lengths:    list[int],
    positions:  list[str],
    trials:     int,
    label:      str,
) -> list[tuple[int, str, int, bool, int | None]]:
    """Returns rows of (target_tokens, position, trial, found, prompt_eval_count)."""
    print(f"\n=== {label}: needle-in-haystack grid (num_ctx={num_ctx}) ===")
    rows = []
    for target in lengths:
        for position in positions:
            hits = 0
            for trial in range(trials):
                prompt = _build_haystack(target, position)
                r = _call(
                    model, prompt, num_ctx,
                    f"{label}-{target}-{position}-{trial}", target,
                )
                found = _needle_found(r.response_text)
                hits += int(found)
                rows.append((target, position, trial, found, r.prompt_eval_count))
                print(
                    f"  tokens={target:>7} pos={position:<6} trial={trial} "
                    f"found={found!s:<5} prompt_eval_count={r.prompt_eval_count} "
                    f"response={r.response_text.strip()[:60]!r}"
                )
            print(f"  -> tokens={target} pos={position}: {hits}/{trials} recalled")
    return rows


def main() -> None:
    tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
    models = [m["model"] for m in tags.get("models", [])]
    for m in (CLOUD_MODEL, LOCAL_MODEL):
        print(f"{m}: {'reachable' if m in models else 'NOT FOUND'}")

    # -- Cloud: metadata check across a range bracketing CLOUD_NUM_CTX ------
    cloud_meta_lengths = [10_000, 50_000, 90_000, 100_000, 110_000, 120_000]
    run_metadata_check(CLOUD_MODEL, CLOUD_NUM_CTX, cloud_meta_lengths, "CLOUD")

    # -- Cloud: num_ctx sweep at fixed length — the real causal test --------
    run_num_ctx_sweep(
        CLOUD_MODEL, fixed_tokens=50_000,
        num_ctx_values=[8_000, 30_000, 50_000, 100_000, 200_000],
        label="CLOUD",
    )

    # -- Cloud: needle grid --------------------------------------------------
    cloud_needle_lengths = [10_000, 90_000, 100_000, 120_000]
    cloud_rows = run_needle_grid(
        CLOUD_MODEL, CLOUD_NUM_CTX, cloud_needle_lengths,
        ["start", "late"], trials=3, label="CLOUD",
    )

    # -- Local: num_ctx sweep sanity check — must show truncation somewhere,
    # or the methodology itself (not just cloud's behavior) is suspect.
    run_num_ctx_sweep(
        LOCAL_MODEL, fixed_tokens=10_000,
        num_ctx_values=[2_000, 8_000, 20_000],
        label="LOCAL",
    )

    # -- Local: sanity-check cross-check against the known 8K boundary ------
    local_needle_lengths = [2_000, 6_000, 8_000, 10_000]
    local_rows = run_needle_grid(
        LOCAL_MODEL, LOCAL_NUM_CTX, local_needle_lengths,
        ["start", "late"], trials=2, label="LOCAL",
    )

    print("\n=== Raw rows (CLOUD) ===")
    for row in cloud_rows:
        print(row)
    print("\n=== Raw rows (LOCAL) ===")
    for row in local_rows:
        print(row)


if __name__ == "__main__":
    main()
