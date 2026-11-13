"""
Engine Benchmark, Phase 1 — oMLX vs Ollama, Gemma 4 12B: speed + RAM
=====================================================================

First phase of the "which local engine drives Localist best" benchmark
(oMLX vs local Ollama, both serving Gemma 4 12B). This phase measures each
engine *cold*, with no prefix-cache help, so later phases (prefix / KV-block
reuse) have a clean baseline to compare against:

  - RAM before the model is loaded, after it loads (idle), peak during a
    request, and retained a few seconds after the request finishes
  - how RAM grows with prompt length (MB per 1k prompt tokens = KV cost)
  - time to first token, prefill (prompt) tok/s, decode tok/s
  - swap-in/out activity during every request
  - thinking tokens (must be zero on both — thinking is forced off)

Methodology notes (read before trusting the numbers)
-----------------------------------------------------
- RAM is measured per process with `top -stats mem` (macOS "footprint":
  resident + compressed, includes Metal/unified allocations), summed over the
  engine's server process and all its children (Ollama spawns a runner per
  loaded model). psutil RSS is NOT used: it under-reports MLX-backed
  processes ~40x — see diag_local_working_memory_ram_probe.py.
- Only one engine has a model loaded at a time. Before each engine runs,
  every model on BOTH engines is unloaded; after it runs, its model is
  unloaded again. Anything else using RAM (browsers, the Localist backend's
  own embedding calls) still shows up in system-wide numbers — the
  per-engine footprint is the primary metric for that reason.
- Every request starts with a unique random nonce, so neither engine can
  reuse a cached prefix. Cache reuse is Phase 2's subject, not this one's.
- Prompt filler is real text from this install's `conversation_log`
  (read-only), with a different slice per trial. The filler is sized using
  a chars-per-token ratio calibrated against each engine's own tokenizer
  count; the engine-reported prompt token count is what gets recorded.
- Thinking is forced off on both (Ollama `think: false`, matching
  OllamaRuntimeClient; oMLX `chat_template_kwargs.enable_thinking: false`,
  which is also oMLX's default for this model). Temperature 0.2 and no
  num_ctx, matching what Localist sends.
- The two engines do NOT run byte-identical weights: Ollama's
  gemma4:12b-mlx is nvfp4 (7.7 GB), oMLX's gemma-4-12B-it-MLX-4bit is MLX
  affine 4-bit, group size 64 (6.3 GB on disk). Compare "overhead above
  weights" and KV growth, not raw totals alone.
- If Ollama's context window (reported by /api/ps) is smaller than a test
  size, Ollama truncates the prompt silently; that shows up as a lower
  engine-reported prompt token count and is flagged in the summary. That's
  a finding, not a bug in this script.

Run from the project root (backend venv active, both engines running):
    source backend/.venv/bin/activate
    python3 diagnostics/diag_engine_bench_phase1.py
    python3 diagnostics/diag_engine_bench_phase1.py --engines omlx --sizes 2000,8000
    python3 diagnostics/diag_engine_bench_phase1.py --engines ollama --ollama-num-ctx 32768 \
        --out diagnostics/engine_bench_phase1_results_ollama_ctx32k.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import secrets
import sqlite3
import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).parent / "engine_bench_phase1_results.json"

DEFAULT_SIZES = [2000, 8000, 16000, 30000]   # 30k, not 32k: leaves room for output in a 32k window
DEFAULT_TRIALS = 3
MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.2
SETTLE_S = 4
POLL_INTERVAL_S = 0.2
REQUEST_TIMEOUT_S = 900

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:12b-mlx"
OMLX_URL = "http://localhost:8000"
OMLX_MODEL = "gemma-4-12B-it-MLX-4bit"

SYSTEM_PROMPT = "You are Localist, a helpful local assistant."
INSTRUCTION = (
    "Write a detailed, multi-paragraph summary of the conversation log above: "
    "the main topics, decisions made, and any open questions."
)


# ---------------------------------------------------------------------------
# Real filler text
# ---------------------------------------------------------------------------

def memory_db_path() -> Path:
    env_db = os.environ.get("LOCALIST_MEMORY_DB")
    if not env_db:
        env_file = REPO_ROOT / "backend" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("LOCALIST_MEMORY_DB="):
                    env_db = line.split("=", 1)[1].strip().strip('"').strip("'")
    return Path(env_db) if env_db else REPO_ROOT / "backend" / "localist_memory.db"


def load_corpus() -> str:
    db = memory_db_path()
    uri = "file:" + str(db).replace(" ", "%20") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT role, content FROM conversation_log "
            "WHERE length(content) > 20 ORDER BY created_at ASC"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(f"[{r.upper()}] {c}" for r, c in rows)


def filler(corpus: str, n_chars: int, offset: int) -> str:
    start = offset % max(1, len(corpus) - n_chars)
    return corpus[start:start + n_chars]


def build_messages(nonce: str, body: str) -> list[dict]:
    return [
        # Nonce first, so no engine can match a cached prefix from an earlier call.
        {"role": "system", "content": f"[run {nonce}]\n{SYSTEM_PROMPT}"},
        {"role": "user", "content": f"[CONVERSATION LOG]\n{body}\n\n[INSTRUCTION]\n{INSTRUCTION}"},
    ]


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------

_TOP_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)([BKMG])\+?-?\s*$", re.MULTILINE)
_UNIT_MB = {"B": 1 / 1024 / 1024, "K": 1 / 1024, "M": 1.0, "G": 1024.0}


def footprint_mb(pids: list[int]) -> float | None:
    if not pids:
        return 0.0
    cmd = ["top", "-l", "1", "-stats", "pid,mem"]
    for p in pids:
        cmd += ["-pid", str(p)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    rows = _TOP_ROW_RE.findall(out)
    if not rows:
        return None
    return sum(float(v) * _UNIT_MB[u] for _, v, u in rows)


def sys_used_gb() -> float:
    return psutil.virtual_memory().used / 1e9


def memorystatus_level() -> int | None:
    """macOS free-memory percentage (what memory_pressure reports)."""
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                                  capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        return None


def swap_counters() -> tuple[int, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    vals = {}
    for line in out.splitlines():
        for key in ("Swapins", "Swapouts"):
            if line.startswith(key + ":"):
                vals[key] = int(line.split(":")[1].strip().rstrip("."))
    return vals.get("Swapins", 0), vals.get("Swapouts", 0)


def snapshot(pids: list[int]) -> dict:
    return {
        "engine_footprint_mb": round(footprint_mb(pids) or 0.0, 1),
        "sys_used_gb": round(sys_used_gb(), 3),
        "memorystatus_free_pct": memorystatus_level(),
    }


class Poller:
    def __init__(self, pids_fn):
        self.pids_fn = pids_fn
        self.peak_engine_mb = 0.0
        self.peak_sys_gb = 0.0
        self.min_free_pct: int | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            fp = footprint_mb(self.pids_fn())
            if fp is not None:
                self.peak_engine_mb = max(self.peak_engine_mb, fp)
            self.peak_sys_gb = max(self.peak_sys_gb, sys_used_gb())
            lvl = memorystatus_level()
            if lvl is not None:
                self.min_free_pct = lvl if self.min_free_pct is None else min(self.min_free_pct, lvl)
            self._stop.wait(POLL_INTERVAL_S)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=15)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _pid_on_port(port: int) -> int | None:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True, timeout=5).stdout.strip()
    return int(out.splitlines()[0]) if out else None


def _with_children(pid: int | None) -> list[int]:
    if pid is None:
        return []
    try:
        p = psutil.Process(pid)
        return [pid] + [c.pid for c in p.children(recursive=True)]
    except psutil.Error:
        return []


class OllamaEngine:
    name = "ollama"
    model = OLLAMA_MODEL

    def __init__(self, num_ctx: int | None = None):
        # None = send no num_ctx, exactly like OllamaRuntimeClient (Ollama
        # then picks its own default window, reported by /api/ps).
        self.num_ctx = num_ctx
        if num_ctx:
            self.name = f"ollama_ctx{num_ctx}"

    def pids(self) -> list[int]:
        return _with_children(_pid_on_port(11434))

    def loaded(self) -> list[dict]:
        return requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])

    def unload_all(self):
        for m in self.loaded():
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": m["name"], "keep_alive": 0}, timeout=60)
        for _ in range(30):
            if not self.loaded():
                return
            time.sleep(1)

    def load_info(self) -> dict:
        for m in self.loaded():
            if m["name"] == self.model:
                return {"reported_size_gb": round(m.get("size", 0) / 1e9, 2),
                        "reported_size_vram_gb": round(m.get("size_vram", 0) / 1e9, 2),
                        "context_length": m.get("context_length")}
        return {}

    def stream(self, messages: list[dict], max_tokens: int) -> dict:
        payload = {
            "model": self.model, "messages": messages, "stream": True, "think": False,
            "options": {"num_predict": max_tokens, "temperature": TEMPERATURE},
        }
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        if getattr(self, "keep_alive", None) is not None:
            payload["keep_alive"] = self.keep_alive
        t0 = time.perf_counter()
        t_first = None
        thinking_chars = content_chars = 0
        final = {}
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True,
                           timeout=REQUEST_TIMEOUT_S) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                msg = ev.get("message") or {}
                if msg.get("thinking"):
                    thinking_chars += len(msg["thinking"])
                if msg.get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    content_chars += len(msg["content"])
                if ev.get("done"):
                    final = ev
        t_end = time.perf_counter()
        ns = 1e9
        return {
            "ttft_s": round((t_first or t_end) - t0, 3),
            "total_s": round(t_end - t0, 3),
            "prompt_tokens": final.get("prompt_eval_count"),
            "cached_tokens": None,  # Ollama doesn't report it
            "prefill_s": round(final.get("prompt_eval_duration", 0) / ns, 3),
            "completion_tokens": final.get("eval_count"),
            "decode_s": round(final.get("eval_duration", 0) / ns, 3),
            "load_s": round(final.get("load_duration", 0) / ns, 3),
            "thinking_chars": thinking_chars,
            "reasoning_tokens": None,
            "content_chars": content_chars,
            "finish_reason": final.get("done_reason"),
        }


class OMLXEngine:
    name = "omlx"
    model = OMLX_MODEL

    def pids(self) -> list[int]:
        return _with_children(_pid_on_port(8000))

    def _models(self) -> list[dict]:
        return requests.get(f"{OMLX_URL}/admin/api/models", timeout=10).json()["models"]

    def unload_all(self):
        for m in self._models():
            if m.get("loaded") and m.get("engine_type") != "external" and m["id"] != "MarkItDown":
                requests.post(f"{OMLX_URL}/admin/api/models/{m['id']}/unload", timeout=60)
        for _ in range(30):
            if not any(m.get("loaded") and m.get("estimated_size", 0) > 0 for m in self._models()):
                return
            time.sleep(1)

    def load_info(self) -> dict:
        for m in self._models():
            if m["id"] == self.model:
                return {"reported_size_gb": round((m.get("actual_size") or 0) / 1e9, 2),
                        "estimated_size_gb": round((m.get("estimated_size") or 0) / 1e9, 2),
                        "context_length": m.get("model_context_length"),
                        "thinking_default": m.get("thinking_default")}
        return {}

    def stream(self, messages: list[dict], max_tokens: int) -> dict:
        payload = {
            "model": self.model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True},
        }
        t0 = time.perf_counter()
        t_first = None
        reasoning_chars = content_chars = 0
        usage: dict = {}
        finish = None
        with requests.post(f"{OMLX_URL}/v1/chat/completions", json=payload, stream=True,
                           timeout=REQUEST_TIMEOUT_S) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if rc:
                        reasoning_chars += len(rc)
                    if d.get("content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        content_chars += len(d["content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        t_end = time.perf_counter()
        return {
            "ttft_s": round((t_first or t_end) - t0, 3),
            "total_s": round(t_end - t0, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prefill_s": usage.get("prompt_eval_duration"),
            "completion_tokens": usage.get("completion_tokens"),
            "decode_s": usage.get("generation_duration"),
            "load_s": usage.get("model_load_duration"),
            "thinking_chars": reasoning_chars,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "content_chars": content_chars,
            "finish_reason": finish,
        }


ENGINES = {"ollama": OllamaEngine, "omlx": OMLXEngine}


def rates(res: dict) -> dict:
    out = {}
    if res.get("prompt_tokens") and res.get("prefill_s"):
        out["prefill_tok_s"] = round(res["prompt_tokens"] / res["prefill_s"], 1)
    if res.get("completion_tokens") and res.get("decode_s"):
        out["decode_tok_s"] = round(res["completion_tokens"] / res["decode_s"], 2)
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_engine(engine, all_engines, corpus: str, sizes: list[int], trials: int) -> dict:
    print(f"\n=== {engine.name} ({engine.model}) ===")
    for e in all_engines:
        e.unload_all()
    time.sleep(SETTLE_S)
    before = snapshot(engine.pids())
    print(f"[{engine.name}] unloaded baseline: {before}")

    # Load: tiny request, timed.
    swap0 = swap_counters()
    t0 = time.perf_counter()
    load_res = engine.stream(build_messages(secrets.token_hex(6), "hello"), 8)
    load_wall = round(time.perf_counter() - t0, 2)
    time.sleep(SETTLE_S)
    idle = snapshot(engine.pids())
    swap1 = swap_counters()
    info = engine.load_info()
    print(f"[{engine.name}] loaded in {load_wall}s (engine-reported load {load_res.get('load_s')}s); "
          f"idle: {idle}; info: {info}")

    # Calibrate chars/token against this engine's own tokenizer count.
    calib_chars = 16000
    cal = engine.stream(build_messages(secrets.token_hex(6), filler(corpus, calib_chars, 7)), 1)
    chars_per_tok = calib_chars / max(1, (cal.get("prompt_tokens") or calib_chars / 4))
    print(f"[{engine.name}] calibration: {cal.get('prompt_tokens')} tokens for {calib_chars} chars "
          f"-> {chars_per_tok:.2f} chars/token")

    trials_out = []
    for size in sizes:
        for trial in range(1, trials + 1):
            n_chars = int(size * chars_per_tok)
            offset = (size * 31 + trial * 100_003)
            msgs = build_messages(secrets.token_hex(6), filler(corpus, n_chars, offset))
            pre = snapshot(engine.pids())
            s0 = swap_counters()
            rec = {"engine": engine.name, "target_tokens": size, "trial": trial, "pre": pre}
            try:
                with Poller(engine.pids) as poll:
                    res = engine.stream(msgs, MAX_OUTPUT_TOKENS)
                s1 = swap_counters()
                time.sleep(SETTLE_S)
                rec.update(res)
                rec.update(rates(res))
                rec.update({
                    "ok": True,
                    "peak_engine_mb": round(poll.peak_engine_mb, 1),
                    "peak_sys_used_gb": round(poll.peak_sys_gb, 3),
                    "min_free_pct": poll.min_free_pct,
                    "retained": snapshot(engine.pids()),
                    "swapins_delta": s1[0] - s0[0],
                    "swapouts_delta": s1[1] - s0[1],
                })
            except Exception as exc:
                rec.update({"ok": False, "error": repr(exc)})
                if not engine.pids():
                    print(f"[{engine.name}] engine process gone after size={size} — stopping this engine")
                    trials_out.append(rec)
                    break
            trials_out.append(rec)
            print(f"[{engine.name}] size={size} t{trial}: "
                  + ", ".join(f"{k}={rec.get(k)}" for k in (
                      "prompt_tokens", "cached_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s",
                      "completion_tokens", "peak_engine_mb", "swapouts_delta", "thinking_chars",
                      "finish_reason", "error") if rec.get(k) is not None))

    engine.unload_all()
    return {
        "engine": engine.name, "model": engine.model, "load_info": info,
        "unloaded_baseline": before, "loaded_idle": idle,
        "load_wall_s": load_wall, "load_engine_reported_s": load_res.get("load_s"),
        "load_swapouts": swap1[1] - swap0[1],
        "chars_per_token": round(chars_per_tok, 3),
        "trials": trials_out,
    }


def linfit(xs: list[float], ys: list[float]) -> float | None:
    if len(set(xs)) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def summarize(runs: list[dict]) -> None:
    print("\n" + "=" * 78 + "\nSUMMARY (means over trials)\n" + "=" * 78)
    for run in runs:
        idle = run["loaded_idle"]["engine_footprint_mb"]
        base = run["unloaded_baseline"]["engine_footprint_mb"]
        print(f"\n{run['engine']}  ({run['model']})  {run['load_info']}")
        print(f"  footprint unloaded {base:.0f} MB -> loaded idle {idle:.0f} MB "
              f"(model load cost {idle - base:.0f} MB); load {run['load_wall_s']}s")
        ok = [t for t in run["trials"] if t.get("ok")]
        by = {}
        for t in ok:
            by.setdefault(t["target_tokens"], []).append(t)
        print(f"  {'target':>7} {'prompt':>7} {'TTFT s':>7} {'pre t/s':>8} {'dec t/s':>8} "
              f"{'peak MB':>8} {'+idle':>7} {'retain':>7} {'swapout':>8}")
        for size, ts in sorted(by.items()):
            m = lambda k: statistics.mean([t[k] for t in ts if t.get(k) is not None] or [0])
            ret = statistics.mean([t["retained"]["engine_footprint_mb"] for t in ts])
            print(f"  {size:>7} {m('prompt_tokens'):>7.0f} {m('ttft_s'):>7.2f} {m('prefill_tok_s'):>8.0f} "
                  f"{m('decode_tok_s'):>8.1f} {m('peak_engine_mb'):>8.0f} {m('peak_engine_mb') - idle:>7.0f} "
                  f"{ret:>7.0f} {sum(t['swapouts_delta'] for t in ts):>8}")
            if any((t.get("prompt_tokens") or 0) < 0.8 * size for t in ts):
                print(f"  !! engine-reported prompt tokens < 80% of target at {size}: prompt truncated?")
            if any(t.get("thinking_chars") or t.get("reasoning_tokens") for t in ts):
                print(f"  !! thinking output present at {size} despite thinking=off")
        slope = linfit([t["prompt_tokens"] for t in ok if t.get("prompt_tokens")],
                       [t["peak_engine_mb"] for t in ok if t.get("prompt_tokens")])
        if slope is not None:
            print(f"  peak-RAM growth: {slope * 1000:.1f} MB per 1k prompt tokens")
        fails = [t for t in run["trials"] if not t.get("ok")]
        if fails:
            print(f"  FAILED trials: {[(t['target_tokens'], t['trial'], t.get('error')) for t in fails]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="ollama,omlx")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--ollama-num-ctx", type=int, default=None,
                    help="send this num_ctx to Ollama (default: none, as Localist does)")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    all_engines = [OllamaEngine(args.ollama_num_ctx), OMLXEngine()]
    wanted = args.engines.split(",")
    chosen = [e for e in all_engines if e.name.split("_ctx")[0] in wanted]

    localist_up = _pid_on_port(8001) is not None
    if localist_up:
        print("WARNING: Localist backend (:8001) is running — its background calls can add noise "
              "to system-wide numbers and contend for the engine. Per-engine footprint is unaffected "
              "unless it sends requests to the engine under test.")

    corpus = load_corpus()
    print(f"Corpus: {len(corpus):,} chars of real conversation_log text")

    try:
        ollama_ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        ollama_ver = None
    meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "machine": {"ram_gb": round(psutil.virtual_memory().total / 2**30),
                    "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          capture_output=True, text=True).stdout.strip(),
                    "macos": platform.mac_ver()[0]},
        "ollama_version": ollama_ver,
        "localist_backend_running": localist_up,
        "sizes": sizes, "trials": args.trials, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE, "ollama_num_ctx": args.ollama_num_ctx,
    }

    runs = [run_engine(e, all_engines, corpus, sizes, args.trials) for e in chosen]
    out = Path(args.out)
    out.write_text(json.dumps({"meta": meta, "runs": runs}, indent=2))
    print(f"\nRaw results written to {out}")
    summarize(runs)


if __name__ == "__main__":
    main()
