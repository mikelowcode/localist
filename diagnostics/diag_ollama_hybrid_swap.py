"""
Ollama Hybrid Inference, Phase 0 — measure before building
==========================================================

Phase 0 of the Ollama hybrid inference manager plan (2026-11-14): a
per-conversation router that answers on the local model (Gemma 4 12B) until
the conversation's size passes a user-set token cap, then moves that
conversation to the cloud model (gemma4:31b-cloud) for the rest of its life.
A new chat starts local again. This script sets the default cap and checks
that the swap is fast and leaves the embedding model alone.

Read-only with respect to Localist: no production code is imported or
changed, and the memory DB is opened read-only (sqlite URI mode=ro). It does
make real inference calls (local and Ollama Cloud — cloud calls count against
the account's usage) and it lets the local model unload naturally via a
1-minute keep_alive, exactly as the user's "Keep model loaded: 1 minute"
setting does. It never forces an unload or restarts Ollama.

Three parts, run one at a time with --part:

  sizes     Real conversation sizes from chat history (chat_turns, grouped by
            conversation_id). Size = running sum of len(content)//4 over
            every turn, the same estimate PromptBuilder uses. Reports the
            distribution of final sizes, the turn at which conversations
            cross each candidate cap, and — the key number for choosing the
            cap — what share of all turns would have run on the cloud model
            under each cap. Session-file attachments aren't stored in the DB,
            so real sizes are a lower bound.

  prefill   Prompt-size sweep, local vs cloud: ~2k/4k/8k/16k/32k-token
            prompts of real text (the architecture docs), a fresh nonce at
            the start of each so nothing is served from cache. Records TTFT,
            Ollama's prompt_eval_count vs our len//4 estimate (calibrates the
            estimate the cap will be measured with), prefill and load
            duration. A needle planted at the very start of the 32k prompt
            confirms the local model sees the whole prompt (no silent
            truncation). This is the latency argument for where the cap goes:
            where does local prefill stop being cheaper than going to cloud?

  swap      The swap itself, as production would do it. Embeds a query with
            the configured embedding model on every turn (as the real app
            does for RAG), then: 2 local turns → swap to cloud carrying the
            full history (~10k tokens) → 1 more cloud turn → "new chat" back
            on local within the 1-minute keep-alive (warm) → idle 70 s →
            local again (cold, after the 1-minute keep-alive expired).
            Records each step's TTFT and load_duration, the embedding call's
            load_duration (non-trivial = the swap evicted it), and /api/ps
            after every step.

Run from the project root (Ollama running; Localist app idle or closed so it
doesn't load models mid-run):
    source backend/.venv/bin/activate
    python3 diagnostics/diag_ollama_hybrid_swap.py --part sizes
    python3 diagnostics/diag_ollama_hybrid_swap.py --part prefill
    python3 diagnostics/diag_ollama_hybrid_swap.py --part swap

Results: diagnostics/ollama_hybrid_swap_<part>_results.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
import uuid
from pathlib import Path

import requests

ROOT       = Path(__file__).resolve().parent.parent
ENV_PATH   = ROOT / "backend" / ".env"
DOCS_DIR   = ROOT / "docs" / "architecture"
OLLAMA     = "http://localhost:11434"
LOCAL      = "gemma4:12b-mlx"
CLOUD      = "gemma4:31b-cloud"
KEEP_ALIVE = "1m"     # the user's "Keep model loaded" choice

CAPS         = [2_000, 4_000, 8_000, 12_000, 16_000, 24_000, 32_000]
SWEEP_TOKENS = [2_000, 4_000, 8_000, 16_000, 32_000]
NEEDLE       = "The vault code for this session is PERIWINKLE-4471."


def est_tokens(text: str) -> int:
    return len(text) // 4          # PromptBuilder._estimate_tokens


def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def save(part: str, data: dict) -> None:
    out = ROOT / "diagnostics" / f"ollama_hybrid_swap_{part}_results.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nSaved → {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------

def chat(model: str, messages: list[dict], max_tokens: int = 64) -> dict:
    """One streamed /api/chat call, payload shaped like OllamaRuntimeClient's."""
    payload = {
        "model": model, "messages": messages, "stream": True, "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    if not model.endswith("-cloud"):
        payload["keep_alive"] = KEEP_ALIVE
    t0 = time.perf_counter()
    ttft = None
    text = []
    final: dict = {}
    with requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(d["error"])
            piece = d.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(piece)
            if d.get("done"):
                final = d
    ns = 1e9
    return {
        "model":                model,
        "ttft_s":               round(ttft, 3) if ttft is not None else None,
        "total_s":              round(time.perf_counter() - t0, 3),
        "prompt_eval_count":    final.get("prompt_eval_count"),
        "eval_count":           final.get("eval_count"),
        "load_s":               round(final.get("load_duration", 0) / ns, 3),
        "prompt_eval_s":        round(final.get("prompt_eval_duration", 0) / ns, 3),
        "done_reason":          final.get("done_reason"),
        "reply":                "".join(text).strip(),
    }


def embed(model: str, text: str) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": text}, timeout=120)
    r.raise_for_status()
    d = r.json()
    return {
        "model":   model,
        "total_s": round(time.perf_counter() - t0, 3),
        "load_s":  round(d.get("load_duration", 0) / 1e9, 3),
    }


def ps() -> list[dict]:
    d = requests.get(f"{OLLAMA}/api/ps", timeout=10).json()
    return [
        {
            "name":           m.get("name"),
            "size_gb":        round(m.get("size", 0) / 1e9, 2),
            "context_length": m.get("context_length"),
            "expires_at":     m.get("expires_at"),
        }
        for m in d.get("models", [])
    ]


def corpus() -> str:
    return "\n\n".join(p.read_text() for p in sorted(DOCS_DIR.glob("*.md")))


def prompt_of(tokens: int, text: str, needle: bool = False) -> str:
    body = text[: tokens * 4]
    head = f"[run {uuid.uuid4().hex}]\n"
    if needle:
        head += NEEDLE + "\n\n"
    return head + body


# ---------------------------------------------------------------------------
# Part: sizes
# ---------------------------------------------------------------------------

def part_sizes() -> None:
    db = read_env().get("LOCALIST_MEMORY_DB")
    if not db:
        raise SystemExit("LOCALIST_MEMORY_DB not set in backend/.env")
    con = sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT conversation_id, role, content FROM chat_turns "
        "WHERE conversation_id != 'legacy' ORDER BY conversation_id, created_at, id"
    ).fetchall()
    con.close()

    convs: dict[str, list[tuple[str, int]]] = {}
    for cid, role, content in rows:
        convs.setdefault(cid, []).append((role, est_tokens(content or "")))

    finals, user_turns_total = [], 0
    cloud_turns = {c: 0 for c in CAPS}
    cross_turn: dict[int, list[int]] = {c: [] for c in CAPS}
    for turns in convs.values():
        running, user_idx = 0, 0
        crossed = {c: False for c in CAPS}
        for role, toks in turns:
            if role == "user":
                user_idx += 1
                user_turns_total += 1
                # The router decides when the user message arrives: history
                # so far plus this message.
                size_at_decision = running + toks
                for c in CAPS:
                    if not crossed[c] and size_at_decision > c:
                        crossed[c] = True
                        cross_turn[c].append(user_idx)
                    if crossed[c]:
                        cloud_turns[c] += 1
            running += toks
        finals.append(running)

    finals.sort()
    def pct(p: float) -> int:
        return finals[min(len(finals) - 1, int(p * len(finals)))]

    summary = {
        "conversations":    len(finals),
        "user_turns":       user_turns_total,
        "final_size_p50":   pct(0.50),
        "final_size_p75":   pct(0.75),
        "final_size_p90":   pct(0.90),
        "final_size_p99":   pct(0.99),
        "final_size_max":   finals[-1] if finals else 0,
        "per_cap": [
            {
                "cap":                    c,
                "conversations_crossing": len(cross_turn[c]),
                "share_conversations":    round(len(cross_turn[c]) / max(len(finals), 1), 3),
                "median_crossing_turn":   statistics.median(cross_turn[c]) if cross_turn[c] else None,
                "share_turns_on_cloud":   round(cloud_turns[c] / max(user_turns_total, 1), 3),
            }
            for c in CAPS
        ],
    }
    print(json.dumps(summary, indent=2))
    save("sizes", summary)


# ---------------------------------------------------------------------------
# Part: prefill
# ---------------------------------------------------------------------------

def part_prefill() -> None:
    text = corpus()
    q = "\n\nIn one short sentence: what is this text about?"
    results = []
    for model in (LOCAL, CLOUD):
        print(f"\n== {model}: warm-up")
        warm = chat(model, [{"role": "user", "content": f"[{uuid.uuid4().hex}] Say OK."}], 8)
        print(f"   load {warm['load_s']}s  ttft {warm['ttft_s']}s")
        results.append({"step": "warmup", **warm})
        for n in SWEEP_TOKENS:
            needle = (n == SWEEP_TOKENS[-1] and model == LOCAL)
            content = prompt_of(n, text, needle) + (
                "\n\nFirst, what is the vault code stated at the very top? Then one sentence on the topic."
                if needle else q
            )
            r = chat(model, [{"role": "user", "content": content}], 64)
            r.update({
                "step": "sweep", "target_tokens": n, "est_tokens": est_tokens(content),
                "est_over_actual": round(est_tokens(content) / r["prompt_eval_count"], 3)
                if r["prompt_eval_count"] else None,
            })
            if needle:
                r["needle_recalled"] = "PERIWINKLE-4471" in r["reply"].upper()
            print(
                f"   {n:>6} est={r['est_tokens']:>6} actual={r['prompt_eval_count']}  "
                f"ttft={r['ttft_s']}s  prefill={r['prompt_eval_s']}s  load={r['load_s']}s"
                + (f"  needle={r['needle_recalled']}" if needle else "")
            )
            results.append(r)
        results.append({"step": "ps_after", "model": model, "ps": ps()})
    save("prefill", {"local": LOCAL, "cloud": CLOUD, "results": results})


# ---------------------------------------------------------------------------
# Part: swap
# ---------------------------------------------------------------------------

def part_swap() -> None:
    env = read_env()
    embed_model = env.get("LOCALIST_EMBEDDING_MODEL_OLLAMA") or env.get("LOCALIST_EMBEDDING_MODEL")
    if not embed_model:
        raise SystemExit("No Ollama embedding model configured in backend/.env")
    text = corpus()
    steps = []

    def step(name: str, model: str, messages: list[dict]) -> dict:
        e = embed(embed_model, messages[-1]["content"][-500:])
        r = chat(model, messages, 96)
        rec = {"step": name, "embed": e, "chat": {k: v for k, v in r.items() if k != "reply"},
               "est_tokens": sum(est_tokens(m["content"]) for m in messages), "ps": ps()}
        loaded = [m["name"] for m in rec["ps"]]
        print(
            f"{name:<22} {model:<18} est={rec['est_tokens']:>6} ttft={r['ttft_s']}s "
            f"load={r['load_s']}s | embed load={e['load_s']}s | loaded={loaded}"
        )
        steps.append(rec)
        return r

    nonce = uuid.uuid4().hex
    hist = [{"role": "system", "content": f"[conv {nonce}] You are a concise assistant."}]

    # Two local turns, ~1.5k tokens each of real text.
    for i, chunk in enumerate((text[:6_000], text[6_000:12_000]), 1):
        hist.append({"role": "user", "content": chunk + "\n\nSummarise that in one sentence."})
        r = step(f"local turn {i}", LOCAL, hist)
        hist.append({"role": "assistant", "content": r["reply"]})

    # Conversation grows past a cap → swap to cloud, carrying full history.
    hist.append({"role": "user", "content": text[12_000:40_000] + "\n\nHow does this relate to the earlier parts?"})
    r = step("swap → cloud", CLOUD, hist)
    hist.append({"role": "assistant", "content": r["reply"]})
    hist.append({"role": "user", "content": "Name one open item mentioned so far."})
    r = step("cloud turn 2", CLOUD, hist)

    # New chat → back to local, within the 1-minute keep-alive.
    new = [{"role": "system", "content": f"[conv {uuid.uuid4().hex}] You are a concise assistant."},
           {"role": "user", "content": "New chat: what's a good name for a houseplant?"}]
    step("new chat → local warm", LOCAL, new)

    print("idle 70 s (1-minute keep-alive expires)…")
    time.sleep(70)
    steps.append({"step": "ps after idle", "ps": ps()})
    print(f"ps after idle: {[m['name'] for m in steps[-1]['ps']]}")

    new2 = [{"role": "system", "content": f"[conv {uuid.uuid4().hex}] You are a concise assistant."},
            {"role": "user", "content": "New chat: suggest a name for a goldfish."}]
    step("new chat → local cold", LOCAL, new2)

    save("swap", {"local": LOCAL, "cloud": CLOUD, "embed_model": embed_model,
                  "keep_alive": KEEP_ALIVE, "steps": steps})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["sizes", "prefill", "swap"])
    {"sizes": part_sizes, "prefill": part_prefill, "swap": part_swap}[ap.parse_args().part]()
