"""
Local Working-Memory RAM Probe
================================

Answers the question behind the "raise LOCAL_PROFILE's working-memory ceiling
from 300 tokens / 5 turns" request: what does an additional 1,000 tokens of
working-memory context actually cost in RAM on *this* machine, for *both*
local backends (oMLX and local, non-`-cloud` Ollama), measured live rather
than assumed?

Methodology notes (read before trusting the numbers)
-----------------------------------------------------
- On Apple Silicon, `psutil`'s per-process RSS (`proc_pidinfo`'s resident
  set) does NOT include Metal/unified-memory GPU-resident allocations —
  confirmed empirically during this investigation: after loading
  gemma4:e4b-mlx (an 8.8GB on-disk model) into the Ollama runner process,
  psutil reported ~200MB RSS while `top`'s MEM column (macOS's memory
  "footprint": resident + compressed + purgeable) reported ~8.3GB for the
  *same pid at the same instant*. This script uses `top -l 1 -pid <pid>
  -stats mem` as the per-process ground truth, not psutil RSS, and cross-
  checks against system-wide `psutil.virtual_memory().used` deltas. Do not
  "simplify" this back to psutil RSS in a future edit — it silently
  under-reports by ~40x for MLX-backed processes.
- Working-memory filler is real conversational text pulled from this
  project's own `conversation_log` table (not synthetic/repeated filler) —
  an earlier probe in this project was invalidated by a token-density
  mismatch between synthetic and real text; this one avoids that by
  construction.
- Each (backend, token size) cell is run 3 times; report mean + spread, not
  a single sample.
- Watches for swap-in/out deltas (`vm_stat`) during each call, not just
  peak footprint — swapping is a slow, silent precursor to the same
  failure mode as an outright OOM/Metal-cap crash.

Run from the project root (backend venv active, both local backends up):
    cd /Users/michaelfilanc/Projects/lora-app-demo
    source backend/.venv/bin/activate
    python3 diagnostics/diag_local_working_memory_ram_probe.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "backend" / "localist_memory.db"
RESULTS_PATH = Path(__file__).parent / "local_working_memory_ram_probe_results.json"

TOKEN_SIZES = [300, 600, 1200, 2000, 3000]
TRIALS_PER_SIZE = 3
MAX_OUTPUT_TOKENS = 150  # fixed, so only input (working-memory) size varies
SETTLE_SECONDS = 3        # pause between trials to let the system settle
POLL_INTERVAL_S = 0.25

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b-mlx"

OMLX_CHAT_URL = "http://localhost:8000/v1/chat/completions"
OMLX_MODEL = "gemma-4-e4b-it-4bit"


# ---------------------------------------------------------------------------
# Real conversational filler
# ---------------------------------------------------------------------------

def load_real_turns() -> list[str]:
    con = sqlite3.connect(str(DB_PATH))
    try:
        rows = con.execute(
            "SELECT role, content FROM conversation_log "
            "WHERE length(content) > 20 ORDER BY created_at ASC"
        ).fetchall()
    finally:
        con.close()
    return [f"[{role.upper()}] {content}" for role, content in rows]


def build_working_memory_block(turns: list[str], target_tokens: int) -> tuple[str, int]:
    """Returns (block_text, actual_estimated_tokens). Real turns vary widely
    in length (up to ~1,300 tokens for a single long agent reply), so the
    delivered size can overshoot the nominal target — the actual token
    count is returned alongside the text so callers log truth, not intent."""
    target_chars = target_tokens * 4
    lines: list[str] = []
    total_chars = 0
    i = 0
    while total_chars < target_chars:
        line = turns[i % len(turns)]
        lines.append(line)
        total_chars += len(line)
        i += 1
    return "\n".join(lines), total_chars // 4


# ---------------------------------------------------------------------------
# macOS-aware per-process memory footprint
# ---------------------------------------------------------------------------

_TOP_MEM_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)([MGK])\s*$", re.MULTILINE)


def top_footprint_mb(pid: int) -> float | None:
    try:
        out = subprocess.run(
            ["top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    m = _TOP_MEM_RE.search(out)
    if not m:
        return None
    value, unit = float(m.group(2)), m.group(3)
    if unit == "G":
        return value * 1024
    if unit == "K":
        return value / 1024
    return value


def find_ollama_runner_pid() -> int | None:
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = p.info["cmdline"] or []
        if any("runner" in c for c in cl) and any("ollama" in c.lower() for c in cl):
            return p.info["pid"]
    return None


def find_omlx_pid() -> int | None:
    try:
        out = subprocess.run(
            ["lsof", "-iTCP:8000", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except Exception:
        pass
    return None


def read_vm_swap_counters() -> tuple[int, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    swapins = swapouts = 0
    for line in out.splitlines():
        if line.startswith("Swapins:"):
            swapins = int(line.split(":")[1].strip().rstrip("."))
        elif line.startswith("Swapouts:"):
            swapouts = int(line.split(":")[1].strip().rstrip("."))
    return swapins, swapouts


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

class Poller:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_proc_mb = 0.0
        self.peak_sys_used_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            fp = top_footprint_mb(self.pid)
            if fp is not None:
                self.peak_proc_mb = max(self.peak_proc_mb, fp)
            used_gb = psutil.virtual_memory().used / 1e9
            self.peak_sys_used_gb = max(self.peak_sys_used_gb, used_gb)
            time.sleep(POLL_INTERVAL_S)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Backend request shims
# ---------------------------------------------------------------------------

def send_ollama(working_block: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"[WORKING MEMORY]\n{working_block}\n\n[INSTRUCTION]\nBriefly summarize the above in two sentences."},
        ],
        "stream": False,
        "options": {"num_ctx": 8000, "num_predict": MAX_OUTPUT_TOKENS},
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def send_omlx(working_block: str) -> dict:
    payload = {
        "model": OMLX_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"[WORKING MEMORY]\n{working_block}\n\n[INSTRUCTION]\nBriefly summarize the above in two sentences."},
        ],
        "stream": False,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    r = requests.post(OMLX_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


BACKENDS = {
    "ollama_local": {"send": send_ollama, "find_pid": find_ollama_runner_pid},
    "omlx": {"send": send_omlx, "find_pid": find_omlx_pid},
}


def run_backend(name: str, turns: list[str]) -> list[dict]:
    cfg = BACKENDS[name]
    records = []
    pid = cfg["find_pid"]()
    if pid is None:
        print(f"[{name}] SKIPPED — could not find a running process for this backend")
        return records

    for size in TOKEN_SIZES:
        block, actual_tokens = build_working_memory_block(turns, size)
        for trial in range(1, TRIALS_PER_SIZE + 1):
            pid = cfg["find_pid"]() or pid
            swap_before = read_vm_swap_counters()
            t0 = time.time()
            try:
                try:
                    with Poller(pid) as poller:
                        resp = cfg["send"](block)
                except requests.exceptions.RequestException as transient:
                    # The externally-managed oMLX process has been observed
                    # restarting itself between requests (pid churn observed
                    # live during this investigation); one retry after a
                    # short wait distinguishes that startup race from a
                    # genuine capacity failure, which re-raises below.
                    print(f"[{name}] size={size} trial={trial} transient error, retrying once: {transient}")
                    time.sleep(5)
                    pid = cfg["find_pid"]() or pid
                    with Poller(pid) as poller:
                        resp = cfg["send"](block)
                elapsed = time.time() - t0
                swap_after = read_vm_swap_counters()
                record = {
                    "backend": name,
                    "target_tokens": size,
                    "actual_tokens": actual_tokens,
                    "trial": trial,
                    "elapsed_s": round(elapsed, 2),
                    "peak_proc_footprint_mb": round(poller.peak_proc_mb, 1),
                    "peak_sys_used_gb": round(poller.peak_sys_used_gb, 3),
                    "swapins_delta": swap_after[0] - swap_before[0],
                    "swapouts_delta": swap_after[1] - swap_before[1],
                    "ok": True,
                }
            except Exception as e:
                record = {
                    "backend": name,
                    "target_tokens": size,
                    "actual_tokens": actual_tokens,
                    "trial": trial,
                    "ok": False,
                    "error": str(e),
                }
                print(f"[{name}] size={size} trial={trial} FAILED: {e}")
                # Check if the process died — that IS the finding.
                if cfg["find_pid"]() is None:
                    print(f"[{name}] process disappeared after failure at size={size} — treating as ceiling, stopping this backend.")
                    records.append(record)
                    return records

            print(f"[{name}] size={size} trial={trial} -> {record}")
            records.append(record)
            time.sleep(SETTLE_SECONDS)
    return records


def summarize(records: list[dict]) -> None:
    print("\n=== SUMMARY (mean ± population stdev, per backend/size) ===")
    by_key: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        by_key.setdefault((r["backend"], r["target_tokens"]), []).append(r)

    for (backend, size), rs in sorted(by_key.items()):
        footprints = [r["peak_proc_footprint_mb"] for r in rs]
        sys_used = [r["peak_sys_used_gb"] for r in rs]
        swapouts = [r["swapouts_delta"] for r in rs]
        print(
            f"{backend:12s} size={size:5d}  "
            f"proc_footprint_mb: mean={statistics.mean(footprints):.1f} stdev={statistics.pstdev(footprints):.1f}  "
            f"sys_used_gb: mean={statistics.mean(sys_used):.3f} stdev={statistics.pstdev(sys_used):.3f}  "
            f"swapouts: {swapouts}"
        )


def main():
    turns = load_real_turns()
    print(f"Loaded {len(turns)} real conversation_log rows for filler text.")

    all_records: list[dict] = []
    for backend in BACKENDS:
        all_records.extend(run_backend(backend, turns))

    RESULTS_PATH.write_text(json.dumps(all_records, indent=2))
    print(f"\nRaw results written to {RESULTS_PATH}")
    summarize(all_records)


if __name__ == "__main__":
    main()
