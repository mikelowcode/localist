"""
KV-Cache Levers, Phase 1 — measure before changing anything
============================================================

Phase 1 of the "make the existing KV-cache design hold up on every engine"
plan (2026-11-12). Read-only with respect to Localist: no production code
changes. Three independent parts, run one at a time with --part:

  a           Shared-prefix size sweep. Localist's main chat call and the
              working_state call share one leading system message (identity
              + persona, §3.7c shared-prefix fix). On Gemma 4 12B, oMLX's
              cache block is 1,024 tokens (forced: it must be a multiple of
              the model's 1,024-token sliding window — oMLX scheduler
              _align_block_size_with_rotating_window), and that system
              message is ~550 tokens, so oMLX caches none of it. This part
              grows the system message to ~1,100 and ~2,150 tokens (real
              persona + real wiki text, standing in for a longer persona)
              and measures, on both engines, a 6-turn conversation of real
              main_dispatch + working_state calls: cold first-turn cost,
              later-turn TTFT, and oMLX cached_tokens.

  b           Ollama keep-alive. Localist sends no keep_alive, so Ollama
              unloads after its 5-minute default and the next message pays a
              reload and a cold cache. Two turns, a 5.5-minute idle gap, a
              third turn — once as today (no keep_alive) and once with
              keep_alive="30m". Records /api/ps residency and engine RAM
              during the idle gap, and the third turn's TTFT/load time.
              Takes ~13 minutes, mostly waiting.

  c-omlx      oMLX 4-bit KV cache (turboquant_kv). Off vs on: peak/retained
              RAM and decode speed at 2k/8k/16k/30k prompt tokens (reusing
              diag_engine_bench_phase1's run_engine), plus a quality spot
              check — 4 real Localist questions answered at temperature 0
              both ways, saved side by side for a human to judge. The
              setting is switched through oMLX's admin API and ALWAYS
              restored to its original value afterwards (finally block).

  c-ollama    Ollama engine-memory profile under whatever OLLAMA_* env the
              running server has (read from the server process and recorded
              as the run label). Peak/retained RAM at 2k/8k/16k, plus
              multi-conversation retention: 4 conversations of ~3k tokens,
              then a return to the first — retained RAM after each, and
              whether the return reuses cache. Run once as-is, then again
              after restarting Ollama with e.g. OLLAMA_NUM_PARALLEL=1 or
              OLLAMA_KV_CACHE_TYPE=q8_0 (this script never changes Ollama's
              env or restarts it).

Methodology carried over from the engine benchmark (see
diag_engine_bench_phase1.py / phase2.py headers): RAM via `top` footprint
summed over the engine's process tree, one engine loaded at a time, real
data from this install, a per-run nonce at the very start of the system
message so runs can't reuse each other's cache. Within a run the main and
working_state calls carry the SAME nonce, so they share a prefix exactly as
they do in production. Thinking off on both engines.

Run from the project root (backend venv active, both engines running,
Localist desktop app closed):
    source backend/.venv/bin/activate
    python3 diagnostics/diag_kv_levers_phase1.py --part a
    python3 diagnostics/diag_kv_levers_phase1.py --part b
    python3 diagnostics/diag_kv_levers_phase1.py --part c-omlx
    python3 diagnostics/diag_kv_levers_phase1.py --part c-ollama
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import secrets
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_engine_bench_phase1 import (  # noqa: E402
    OLLAMA_URL, OMLX_MODEL, OMLX_URL, OllamaEngine, OMLXEngine, _pid_on_port, load_corpus,
    run_engine, snapshot,
)
from diag_engine_bench_phase2 import Data, WIKI_DIR  # noqa: E402

from localist.episodic_extractor import _WSU_TASK_INSTRUCTIONS  # noqa: E402
from localist.prompt_builder import PromptBuilder, RagSource, Turn  # noqa: E402

OUT_DIR = Path(__file__).parent
ASSISTANT_NAME = "Localist"
N_TURNS = 6
MAIN_MAX_TOKENS = 128
SIDE_MAX_TOKENS = 64

# Diagnostic-only builder with the persona ceiling lifted, so a longer
# system message can be measured. Production PromptBuilder is untouched.
PB = PromptBuilder()
PB._CEIL_PERSONA = 100_000


def stamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def save(name: str, payload: dict) -> Path:
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nRaw results written to {path}")
    return path


def ollama_env() -> dict:
    pid = _pid_on_port(11434)
    if not pid:
        return {}
    out = subprocess.run(["ps", "eww", "-p", str(pid)], capture_output=True, text=True).stdout
    return {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in out.split()
            if kv.startswith("OLLAMA_") and "=" in kv}


def omlx_probe(messages: list[dict]) -> dict:
    try:
        r = requests.post(f"{OMLX_URL}/admin/api/cache/probe",
                          json={"model_id": OMLX_MODEL, "messages": messages}, timeout=60).json()
        return {k: r.get(k) for k in ("total_tokens", "block_size", "total_blocks")}
    except Exception as exc:
        return {"error": repr(exc)}


# ---------------------------------------------------------------------------
# Real Localist call shapes with a configurable persona
# ---------------------------------------------------------------------------

def persona_variants(data: Data) -> dict[str, str]:
    """current / ~1.1k / ~2.15k-token system messages. The extension is real
    wiki prose (how Localist works, design philosophy), standing in for what
    a longer, user-written persona would contain."""
    base = data.persona or ""
    preferred = ("how-localist-works.md", "localist-design-philosophy.md")
    pages = [WIKI_DIR / n for n in preferred if (WIKI_DIR / n).exists()]
    pages += [WIKI_DIR / s.path for s in data.wiki if s.path not in preferred]
    extra = "\n\n".join(p.read_text() for p in pages)

    def sized(target_tokens: int) -> str:
        cur = PB._estimate_tokens(PB._slot1_system(base, ASSISTANT_NAME))
        need_chars = max(0, (target_tokens - cur)) * 4
        return base + "\n\n## Background\n" + extra[:need_chars]

    return {"current": base, "p1100": sized(1100), "p2150": sized(2150)}


class Run:
    """One conversation: one nonce shared by main + working_state calls."""

    def __init__(self, data: Data, persona: str, start_pair: int):
        self.data, self.persona, self.start = data, persona, start_pair
        self.nonce = secrets.token_hex(6)
        self.history: list[Turn] = []

    def system(self) -> str:
        return f"[run {self.nonce}]\n" + PB._slot1_system(self.persona, ASSISTANT_NAME)

    def pair(self, t: int) -> tuple[str, str]:
        return self.data.pairs[(self.start + t) % len(self.data.pairs)]

    def main_messages(self, engine: str, t: int) -> list[dict]:
        instruction, _ = self.pair(t)
        n = len(self.data.wiki)
        kw = dict(instruction=instruction, current_datetime=dt.datetime.now().astimezone(),
                  persona=self.persona, assistant_name=ASSISTANT_NAME,
                  episodic_summary=self.data.episodes,
                  rag_snippets=[self.data.wiki[(self.start + t) % n]],
                  working_memory=list(self.history))
        msgs = [{"role": "system", "content": self.system()}]
        if engine == "omlx":   # OMLXRuntimeClient's structured working-memory path
            _, user, turns = PB.build(**kw, emit_structured_working_memory=True)
            msgs += [{"role": "assistant" if x.role == "assistant" else "user", "content": x.content}
                     for x in turns or []]
        else:                  # OllamaRuntimeClient: one flattened user message
            _, user = PB.build(**kw)
        msgs.append({"role": "user", "content": user})
        return msgs

    def wsu_messages(self, t: int) -> list[dict]:
        u, a = self.pair(t)
        return [
            {"role": "system", "content": self.system() + "\n\n" + _WSU_TASK_INSTRUCTIONS},
            {"role": "user", "content": "Previous working state:\ncurrent_focus: NONE\nopen_loops: NONE\n"
                                        f"recent_decisions: NONE\n\nNew turn:\nUser: {u}\nAssistant: {a}"},
        ]

    def advance(self, t: int):
        u, a = self.pair(t)
        self.history += [Turn(role="user", content=u), Turn(role="assistant", content=a)]


def call(engine, msgs, max_tokens) -> dict:
    r = engine.stream(msgs, max_tokens)
    return {k: r.get(k) for k in ("ttft_s", "prompt_tokens", "cached_tokens", "prefill_s",
                                  "load_s", "completion_tokens", "thinking_chars", "finish_reason")}


def engine_key(engine) -> str:
    return "omlx" if engine.name == "omlx" else "ollama"


def load_only(engine, all_engines):
    for e in all_engines:
        e.unload_all()
    time.sleep(3)
    engine.stream([{"role": "user", "content": f"[load {secrets.token_hex(4)}] hi"}], 4)


# ---------------------------------------------------------------------------
# Part a — shared-prefix size sweep
# ---------------------------------------------------------------------------

def part_a(args):
    data = Data()
    variants = persona_variants(data)
    engines = [OllamaEngine(), OMLXEngine()]
    chosen = [e for e in engines if e.name in args.engines.split(",")]
    runs = []
    for engine in chosen:
        load_only(engine, engines)
        for vname, persona in variants.items():
            for rep in range(1, args.reps + 1):
                run = Run(data, persona, start_pair=200 + 40 * rep + 7 * len(runs))
                sys_tokens = None
                if engine.name == "omlx":
                    sys_tokens = omlx_probe([{"role": "system", "content": run.system()},
                                             {"role": "user", "content": "x"}])
                print(f"\n[{engine.name}] prefix={vname} rep={rep} system≈"
                      f"{PB._estimate_tokens(run.system())} tok (est) probe={sys_tokens}")
                calls = []
                for t in range(N_TURNS):
                    m = call(engine, run.main_messages(engine_key(engine), t), MAIN_MAX_TOKENS)
                    m.update(call="main", turn=t + 1)
                    w = call(engine, run.wsu_messages(t), SIDE_MAX_TOKENS)
                    w.update(call="working_state", turn=t + 1)
                    calls += [m, w]
                    run.advance(t)
                    print(f"  t{t + 1}: main tok={m['prompt_tokens']} cached={m['cached_tokens']} "
                          f"ttft={m['ttft_s']}s | wsu tok={w['prompt_tokens']} cached={w['cached_tokens']} "
                          f"ttft={w['ttft_s']}s")
                runs.append({"engine": engine.name, "prefix": vname, "rep": rep,
                             "system_tokens_est": PB._estimate_tokens(run.system()),
                             "omlx_probe": sys_tokens, "calls": calls,
                             "ram_after": snapshot(engine.pids())})
        engine.unload_all()
    save("kv_levers_phase1_a_results.json", {"meta": {"timestamp": stamp(), "args": vars(args)}, "runs": runs})

    print("\n" + "=" * 86 + "\nPART A SUMMARY (means over reps)\n" + "=" * 86)
    print(f"{'engine':7} {'prefix':8} {'sys tok':>7} {'main t1':>8} {'main t2-6':>9} {'wsu t1':>7} "
          f"{'wsu t2-6':>8} {'main cached':>11} {'wsu cached':>10}")
    for eng in [e.name for e in chosen]:
        for vname in variants:
            rs = [r for r in runs if r["engine"] == eng and r["prefix"] == vname]
            if not rs:
                continue

            def mean(sel, key):
                vals = [c[key] for r in rs for c in r["calls"] if sel(c) and c.get(key) is not None]
                return statistics.mean(vals) if vals else float("nan")
            print(f"{eng:7} {vname:8} {rs[0]['system_tokens_est']:>7} "
                  f"{mean(lambda c: c['call'] == 'main' and c['turn'] == 1, 'ttft_s'):>8.2f} "
                  f"{mean(lambda c: c['call'] == 'main' and c['turn'] > 1, 'ttft_s'):>9.2f} "
                  f"{mean(lambda c: c['call'] == 'working_state' and c['turn'] == 1, 'ttft_s'):>7.2f} "
                  f"{mean(lambda c: c['call'] == 'working_state' and c['turn'] > 1, 'ttft_s'):>8.2f} "
                  f"{mean(lambda c: c['call'] == 'main' and c['turn'] > 1, 'cached_tokens'):>11.0f} "
                  f"{mean(lambda c: c['call'] == 'working_state', 'cached_tokens'):>10.0f}")


# ---------------------------------------------------------------------------
# Part b — Ollama keep-alive
# ---------------------------------------------------------------------------

def part_b(args):
    data = Data()
    persona = data.persona or ""
    results = []
    for label, keep_alive in (("default (none sent)", None), ("keep_alive=30m", "30m")):
        engine = OllamaEngine()
        engine.keep_alive = keep_alive
        load_only(engine, [engine, OMLXEngine()])
        run = Run(data, persona, start_pair=600 + len(results) * 30)
        turns = []
        for t in range(2):
            r = call(engine, run.main_messages("ollama", t), MAIN_MAX_TOKENS)
            r.update(turn=t + 1)
            turns.append(r)
            run.advance(t)
        print(f"\n[{label}] t1 ttft={turns[0]['ttft_s']} t2 ttft={turns[1]['ttft_s']}; idling {args.idle_s}s ...")
        idle_samples = []
        t_idle0 = time.time()
        while time.time() - t_idle0 < args.idle_s:
            time.sleep(30)
            loaded = [m["name"] for m in engine.loaded()]
            snap = snapshot(engine.pids())
            idle_samples.append({"t_s": round(time.time() - t_idle0), "loaded": loaded,
                                 "engine_footprint_mb": snap["engine_footprint_mb"],
                                 "sys_used_gb": snap["sys_used_gb"]})
            print(f"  idle {idle_samples[-1]['t_s']:>4}s loaded={loaded} "
                  f"footprint={snap['engine_footprint_mb']:.0f}MB")
        r3 = call(engine, run.main_messages("ollama", 2), MAIN_MAX_TOKENS)
        r3.update(turn=3)
        turns.append(r3)
        print(f"  t3 after idle: ttft={r3['ttft_s']}s load_s={r3['load_s']}")
        results.append({"label": label, "keep_alive": keep_alive, "turns": turns, "idle": idle_samples})
        engine.keep_alive = None
        engine.unload_all()
    save("kv_levers_phase1_b_results.json",
         {"meta": {"timestamp": stamp(), "idle_s": args.idle_s, "ollama_env": ollama_env()}, "runs": results})
    print("\nPART B SUMMARY")
    for r in results:
        held = [s["engine_footprint_mb"] for s in r["idle"]]
        print(f"  {r['label']:20s} t2 ttft={r['turns'][1]['ttft_s']:.2f}s  t3-after-idle ttft="
              f"{r['turns'][2]['ttft_s']:.2f}s (load {r['turns'][2]['load_s']}s)  "
              f"still loaded at end of idle: {bool(r['idle'][-1]['loaded'])}  "
              f"RAM held while idle: {min(held):.0f}-{max(held):.0f} MB")


# ---------------------------------------------------------------------------
# Part c-omlx — 4-bit KV cache
# ---------------------------------------------------------------------------

def omlx_set_turboquant(enabled: bool):
    r = requests.put(f"{OMLX_URL}/admin/api/models/{OMLX_MODEL}/settings",
                     json={"turboquant_kv_enabled": enabled}, timeout=30)
    r.raise_for_status()


def omlx_turboquant_current() -> bool:
    p = Path.home() / ".omlx" / "model_settings.json"
    try:
        return bool(json.loads(p.read_text())["models"][OMLX_MODEL].get("turboquant_kv_enabled", False))
    except Exception:
        return False


def quality_answers(engine, data: Data) -> list[dict]:
    out = []
    for k in range(4):
        run = Run(data, data.persona or "", start_pair=50 + 113 * k)
        msgs = run.main_messages("omlx", 0)
        r = requests.post(f"{OMLX_URL}/v1/chat/completions", json={
            "model": OMLX_MODEL, "messages": msgs, "stream": False, "max_tokens": 200,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=300).json()
        out.append({"question": run.pair(0)[0][:300],
                    "answer": r["choices"][0]["message"].get("content", "")})
    return out


def part_c_omlx(args):
    data = Data()
    corpus = load_corpus()
    original = omlx_turboquant_current()
    sizes = [int(s) for s in args.sizes.split(",")]
    engines = [OllamaEngine(), OMLXEngine()]
    runs = {}
    try:
        for enabled in (False, True):
            label = "turboquant_on" if enabled else "turboquant_off"
            print(f"\n##### oMLX {label}")
            omlx_set_turboquant(enabled)
            engines[1].unload_all()
            runs[label] = run_engine(engines[1], engines, corpus, sizes, 1)
            load_only(engines[1], engines)
            runs[label]["quality"] = quality_answers(engines[1], data)
            engines[1].unload_all()
    finally:
        omlx_set_turboquant(original)
        engines[1].unload_all()
        print(f"\nturboquant_kv_enabled restored to {original}")
    save("kv_levers_phase1_c_omlx_results.json",
         {"meta": {"timestamp": stamp(), "sizes": sizes, "restored_to": original}, "runs": runs})
    print("\nPART C-oMLX SUMMARY")
    for label, run in runs.items():
        print(f"  {label}: idle {run['loaded_idle']['engine_footprint_mb']:.0f} MB")
        for t in run["trials"]:
            if t.get("ok"):
                print(f"    {t['prompt_tokens']:>6} tok  peak {t['peak_engine_mb']:>7.0f} MB  retained "
                      f"{t['retained']['engine_footprint_mb']:>7.0f} MB  decode {t.get('decode_tok_s')} tok/s  "
                      f"swapouts {t['swapouts_delta']}")
            else:
                print(f"    FAILED at {t['target_tokens']}: {t.get('error')}")
    print("\n  Quality spot check (temperature 0) — compare by eye:")
    for a, b in zip(runs["turboquant_off"]["quality"], runs["turboquant_on"]["quality"]):
        same = "IDENTICAL" if a["answer"].strip() == b["answer"].strip() else "differs"
        print(f"\n  Q: {a['question'][:120]!r}  [{same}]\n   off: {a['answer'][:300]!r}\n   on:  {b['answer'][:300]!r}")


# ---------------------------------------------------------------------------
# Part c-ollama — memory profile under the running server's env
# ---------------------------------------------------------------------------

def part_c_ollama(args):
    env = ollama_env()
    label = ",".join(f"{k}={v}" for k, v in sorted(env.items())
                     if k not in ("OLLAMA_MODELS", "OLLAMA_HOST")) or "defaults"
    print(f"Ollama env for this run: {label}")
    data = Data()
    corpus = load_corpus()
    engines = [OllamaEngine(), OMLXEngine()]
    ollama = engines[0]
    profile = run_engine(ollama, engines, corpus, [int(s) for s in args.sizes.split(",")], 1)

    # Multi-conversation retention: 4 conversations of ~3k tokens, return to #1.
    load_only(ollama, engines)
    docs = data.files
    convs = []
    retention = []
    for k in range(4):
        nonce = secrets.token_hex(6)
        doc = "\n\n".join(f.content for f in docs[k:k + 2])[:11000]
        sysmsg = f"[conversation {nonce}]\n" + PB._slot1_system(data.persona, ASSISTANT_NAME) + "\n\n" + doc
        convs.append(sysmsg)
        r = call(ollama, [{"role": "system", "content": sysmsg},
                          {"role": "user", "content": data.pairs[300 + k][0]}], 64)
        snap = snapshot(ollama.pids())
        retention.append({"conversation": k + 1, "ttft_s": r["ttft_s"], "prompt_tokens": r["prompt_tokens"],
                          "engine_footprint_mb": snap["engine_footprint_mb"]})
        print(f"  conv {k + 1}: tok={r['prompt_tokens']} ttft={r['ttft_s']}s footprint={snap['engine_footprint_mb']:.0f}MB")
    r = call(ollama, [{"role": "system", "content": convs[0]},
                      {"role": "user", "content": data.pairs[310][0]}], 64)
    snap = snapshot(ollama.pids())
    retention.append({"conversation": "return to 1", "ttft_s": r["ttft_s"], "prompt_tokens": r["prompt_tokens"],
                      "engine_footprint_mb": snap["engine_footprint_mb"]})
    print(f"  return to conv 1: ttft={r['ttft_s']}s (its cold ttft was {retention[0]['ttft_s']}s) "
          f"footprint={snap['engine_footprint_mb']:.0f}MB")
    ollama.unload_all()

    safe = args.out_label or "".join(c if c.isalnum() else "_" for c in label)[:80]
    save(f"kv_levers_phase1_c_ollama_{safe}.json",
         {"meta": {"timestamp": stamp(), "ollama_env": env}, "profile": profile, "retention": retention})
    print("\nPART C-OLLAMA SUMMARY")
    print(f"  idle {profile['loaded_idle']['engine_footprint_mb']:.0f} MB, context {profile['load_info'].get('context_length')}")
    for t in profile["trials"]:
        if t.get("ok"):
            print(f"    {t['prompt_tokens']:>6} tok  peak {t['peak_engine_mb']:>7.0f} MB  retained "
                  f"{t['retained']['engine_footprint_mb']:>7.0f} MB  decode {t.get('decode_tok_s')} tok/s  "
                  f"swapouts {t['swapouts_delta']}")
    print(f"  retention: " + "; ".join(f"{x['conversation']}: {x['engine_footprint_mb']:.0f}MB/{x['ttft_s']}s"
                                       for x in retention))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "b", "c-omlx", "c-ollama"])
    ap.add_argument("--engines", default="ollama,omlx", help="part a only")
    ap.add_argument("--reps", type=int, default=2, help="part a only")
    ap.add_argument("--idle-s", type=int, default=330, help="part b only")
    ap.add_argument("--sizes", default="2000,8000,16000,30000", help="part c only")
    ap.add_argument("--out-label", default=None, help="part c-ollama only: results file suffix")
    args = ap.parse_args()
    if args.part == "c-ollama" and args.sizes == "2000,8000,16000,30000":
        args.sizes = "2000,8000,16000"
    if _pid_on_port(8001):
        print("WARNING: Localist backend (:8001) is running — close the desktop app for cleaner numbers.")
    {"a": part_a, "b": part_b, "c-omlx": part_c_omlx, "c-ollama": part_c_ollama}[args.part](args)


if __name__ == "__main__":
    main()
