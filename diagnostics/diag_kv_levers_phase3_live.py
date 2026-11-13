"""
KV-Cache Levers, Phase 3 — live check against a running Localist backend
========================================================================

Drives the real backend (:8001, started by start_localist.sh) to confirm the
Phase 2 build end to end, not through mocks:

  --mode chat       Send a fixed 4-turn conversation through POST /task/stream
                    (the UI's chat path: Planner -> ControllerAgent ->
                    main_dispatch + working_state). Records time to the first
                    streamed answer token and total time per turn, oMLX's
                    /admin/api/stats cached/prompt token totals before and
                    after (oMLX only), and GET /settings/prefix-cache.
                    Run once with the original persona and once with the
                    persona that fills one cache block, and compare.

  --mode keepalive  Ollama only. GET /settings/prefix-cache, then
                    PUT /settings/ollama-keep-alive "30m" -> one turn ->
                    Ollama's /api/ps expires_at; PUT "off" -> one turn ->
                    expires_at again. Confirms the setting reaches Ollama.

Meant to run against a backend whose LOCALIST_MEMORY_DB / LOCALIST_WIKI_DIR
point at COPIES of the real data (see the Phase 3 runner script), because
chat turns write conversation history and may write episodes.

    python3 diagnostics/diag_kv_levers_phase3_live.py --mode chat --label original
    python3 diagnostics/diag_kv_levers_phase3_live.py --mode keepalive
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import uuid
from pathlib import Path

import requests

BACKEND = "http://127.0.0.1:8001"
OMLX = "http://127.0.0.1:8000"
OLLAMA = "http://127.0.0.1:11434"
OUT_DIR = Path(__file__).parent

# Neutral questions about the user's own vault: no personal statements (so the
# implicit-episode gate stays closed) and nothing that needs today's web.
QUESTIONS = [
    "What is the Localist build order?",
    "Summarize the Localist design philosophy in three bullets.",
    "Which part of that matters most for running models locally?",
    "Name one risk the build order doesn't address.",
]


def omlx_stats() -> dict | None:
    try:
        s = requests.get(f"{OMLX}/admin/api/stats", timeout=10).json()
        return {k: s.get(k) for k in ("total_prompt_tokens", "total_cached_tokens", "total_requests")}
    except Exception:
        return None


def prefix_cache() -> dict:
    return requests.get(f"{BACKEND}/settings/prefix-cache", timeout=10).json()


def chat_turn(conversation_id: str, instruction: str, title: str) -> dict:
    t0 = time.perf_counter()
    t_first = None
    chars = 0
    err = None
    with requests.post(f"{BACKEND}/task/stream", json={
        "instruction": instruction, "conversation_id": conversation_id,
        "conversation_title": title,
    }, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            ev = json.loads(raw[5:])
            if ev.get("type") == "token":
                if t_first is None:
                    t_first = time.perf_counter()
                chars += len(ev.get("token", ""))
            elif ev.get("type") == "answer_reset":
                chars = 0
            elif ev.get("type") == "error":
                err = ev.get("message")
            elif ev.get("type") == "task_complete":
                break
    t_end = time.perf_counter()
    return {"question": instruction,
            "first_token_s": round((t_first or t_end) - t0, 2),
            "total_s": round(t_end - t0, 2), "answer_chars": chars, "error": err}


def mode_chat(label: str) -> dict:
    status = prefix_cache()
    print(f"[{label}] prefix-cache: {status}")
    before = omlx_stats()
    conv = str(uuid.uuid4())
    turns = []
    for q in QUESTIONS:
        t = chat_turn(conv, q, f"KV cache live check ({label})")
        turns.append(t)
        print(f"  {t['first_token_s']:>6.2f}s first token  {t['total_s']:>6.2f}s total  "
              f"{t['answer_chars']} chars  {q!r}" + (f"  ERROR {t['error']}" if t["error"] else ""))
    time.sleep(8)   # let the trailing working_state / extraction calls finish
    after = omlx_stats()
    delta = None
    if before and after:
        p = after["total_prompt_tokens"] - before["total_prompt_tokens"]
        c = after["total_cached_tokens"] - before["total_cached_tokens"]
        delta = {"prompt_tokens": p, "cached_tokens": c,
                 "requests": after["total_requests"] - before["total_requests"],
                 "cache_efficiency_pct": round(100 * c / p, 1) if p else None}
        print(f"  oMLX over this run: {delta}")
    return {"label": label, "prefix_cache": status, "turns": turns, "omlx_delta": delta}


def ollama_expires() -> str | None:
    models = requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", [])
    return models[0].get("expires_at") if models else None


def mode_keepalive() -> dict:
    status = prefix_cache()
    print(f"[ollama] prefix-cache: {status}")
    out = {"prefix_cache": status, "steps": []}
    conv = str(uuid.uuid4())
    for choice, q in (("30m", QUESTIONS[0]), ("off", QUESTIONS[1])):
        put = requests.put(f"{BACKEND}/settings/ollama-keep-alive", json={"keep_alive": choice}, timeout=10).json()
        turn = chat_turn(conv, q, "KV cache live check (keep-alive)")
        time.sleep(8)
        exp = ollama_expires()
        now = dt.datetime.now().astimezone()
        mins = None
        if exp:
            try:
                mins = round((dt.datetime.fromisoformat(exp) - now).total_seconds() / 60, 1)
            except ValueError:
                pass
        step = {"choice": choice, "put_response": put, "turn": turn,
                "expires_at": exp, "minutes_until_unload": mins}
        out["steps"].append(step)
        print(f"  keep_alive={choice}: applied_live={put.get('applied_live')}  "
              f"/api/ps expires in ~{mins} min  (turn {turn['total_s']}s)")
    out["final_setting"] = requests.get(f"{BACKEND}/settings/ollama-keep-alive", timeout=10).json()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["chat", "keepalive"])
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    result = mode_chat(args.label) if args.mode == "chat" else mode_keepalive()
    name = f"kv_levers_phase3_{args.mode}_{args.label if args.mode == 'chat' else 'ollama'}.json"
    (OUT_DIR / name).write_text(json.dumps(
        {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **result}, indent=2))
    print(f"Results written to {OUT_DIR / name}")


if __name__ == "__main__":
    main()
