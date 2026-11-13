"""
Engine Benchmark, Phase 2 — oMLX vs Ollama, Gemma 4 12B: prefix / KV-cache reuse
=================================================================================

Phase 2 of the "which local engine drives Localist best" benchmark (Phase 1 =
diag_engine_bench_phase1.py, cold speed + RAM). Question: does oMLX's
vLLM-style block-based KV cache (RAM hot tier + SSD tier) give it a real
advantage over Ollama *for the prompts Localist actually sends*?

Prompts are built with Localist's real PromptBuilder (real slot order,
ceilings and truncation), from this install's real data: persona.md, wiki
pages as RAG sources, raw documents as session files, episodes as episodic
memory, and consecutive user/assistant pairs from conversation_log as the
conversation. Side calls use the real working_state / implicit_extraction
system prompts and user-prompt shapes from episodic_extractor.py.

What was learned before writing this (live probes, 2026-11-12)
--------------------------------------------------------------
- Ollama reuses a shared token-level prefix (prefill 1.48s -> 0.13s when only
  the final message changed) but keeps ONE cached sequence, and its
  prompt_eval_count does NOT drop on a hit — reuse shows in timing only.
- oMLX reports cached_tokens directly. It reuses whole 1,024-token blocks of
  a shared prefix (2,184-token shared prefix -> 2,048 cached; ~970 -> 0) plus
  near-all of a prompt that extends the previous request.
- PromptBuilder renders [CURRENT DATETIME] (changes every call) FIRST in the
  user message, ahead of session files, memory, RAG and instruction. So on
  today's layout only the system message (and, on the oMLX path, the
  structured working-memory turns) can ever be cache hits.

Layouts (the same content, two orders)
--------------------------------------
- today:      exactly what each engine's runtime client sends now.
              Ollama: [system, user(datetime|files|memory|RAG|history|instr)]
              oMLX:   [system, *history turns, user(datetime|files|...|instr)]
- friendly:   a cache-friendly reorder, same on both engines:
              [system + session files, *history turns, user(datetime|...)]
              i.e. stable content first, volatile content last.

Scenarios (each with a per-conversation nonce at the very start of the
system message, so nothing is reused across scenario runs or from earlier
benchmarks — this also removes the small real-world reuse of the identical
system prompt across different conversations)
--------------------------------------------------------------------------
- A growing:     6 turns of one conversation, main call only.
- B interleaved: the same 6 turns, each followed by the working_state and
                 implicit_extraction side calls Localist makes after a turn.
- C return:      conversation X turns 1-2, then 3 different conversations
                 (one turn each), then X turn 3.
- D reload:      X turns 1-2, unload the model, reload it, then X turn 3.

Context sizes
-------------
- light: no session files (typical; real main_dispatch prompts in
         ~/Library/Logs/Localist/backend.log are median ~490, p90 ~2k
         tokens, plus the system message).
- heavy: two raw documents attached as session files (PromptBuilder caps
         each at 4k tokens), the "chatting about an uploaded doc" case.

Metrics per main call: TTFT, prefill seconds, prompt tokens, cached tokens
(oMLX only) and estimated reuse = 1 - TTFT / cold TTFT, where the cold
estimate comes from an in-run cold calibration of the same engine
(fresh nonce, no reuse possible) interpolated by prompt length. Engine RAM
footprint is snapshotted after each scenario (oMLX's hot cache costs RAM).

Run from the project root (backend venv active, both engines running; the
Localist desktop app / backend should be closed to cut noise):
    source backend/.venv/bin/activate
    python3 diagnostics/diag_engine_bench_phase2.py
    python3 diagnostics/diag_engine_bench_phase2.py --engines omlx --layouts today --sizes light --scenarios A
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import secrets
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_engine_bench_phase1 import (  # noqa: E402  (shared engine + RAM plumbing)
    OLLAMA_URL, OMLX_URL, OllamaEngine, OMLXEngine, _pid_on_port, memory_db_path,
    snapshot,
)

from localist.episodic_extractor import _EXTRACTION_SYSTEM, _build_wsu_system  # noqa: E402
from localist.prompt_builder import (  # noqa: E402
    EpisodeBullet, PromptBuilder, RagSource, SessionFile, Turn,
)

RESULTS_PATH = Path(__file__).parent / "engine_bench_phase2_results.json"
APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Localist"
WIKI_DIR = APP_SUPPORT / "wiki"
RAW_DIR = APP_SUPPORT / "raw"

MAIN_MAX_TOKENS = 128   # TTFT is the metric; keep decode short to bound runtime
SIDE_MAX_TOKENS = 64
N_TURNS = 6

PB = PromptBuilder()


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def _db():
    db = memory_db_path()
    return sqlite3.connect("file:" + str(db).replace(" ", "%20") + "?mode=ro", uri=True)


def load_pairs() -> list[tuple[str, str]]:
    """Consecutive (user, assistant) pairs from conversation_log."""
    con = _db()
    try:
        rows = con.execute(
            "SELECT role, content FROM conversation_log WHERE length(content) > 20 "
            "ORDER BY created_at ASC, id ASC").fetchall()
    finally:
        con.close()
    def answer_text(c: str) -> str:
        # Agent rows are stored as a repr'd dict: "{'answer': '...', ...}".
        try:
            v = ast.literal_eval(c)
            if isinstance(v, dict) and v.get("answer"):
                return str(v["answer"])
        except (ValueError, SyntaxError):
            pass
        return c

    pairs = []
    for (r1, c1), (r2, c2) in zip(rows, rows[1:]):
        if r1 == "user" and r2 in ("agent", "assistant") and len(c1) < 1500:
            pairs.append((c1, answer_text(c2)))
    return pairs


def load_episodes() -> list[EpisodeBullet]:
    con = _db()
    try:
        rows = con.execute(
            "SELECT content, episode_type, confidence FROM episodes "
            "WHERE content IS NOT NULL ORDER BY created_at DESC LIMIT 6").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    return [EpisodeBullet(content=c, episode_type=t or "project_fact", confidence=float(f or 0.8))
            for c, t, f in rows]


def load_persona() -> str | None:
    p = WIKI_DIR / "persona.md"
    return p.read_text() if p.exists() else None


def load_wiki_pages() -> list[RagSource]:
    skip = {"persona.md", "MEMORY.md", "index.md", "logs.md"}
    return [RagSource(path=str(p.relative_to(WIKI_DIR)), content=p.read_text()[:4000])
            for p in sorted(WIKI_DIR.rglob("*.md"))
            if p.name not in skip and "users" not in p.parts]


def load_session_files() -> list[SessionFile]:
    docs = sorted(RAW_DIR.glob("*.md"), key=lambda p: -p.stat().st_size)
    return [SessionFile(filename=p.name, content=p.read_text()) for p in docs]


class Data:
    def __init__(self):
        self.pairs = load_pairs()
        self.episodes = load_episodes()
        self.persona = load_persona()
        self.wiki = load_wiki_pages()
        self.files = load_session_files()
        if len(self.pairs) < 60:
            raise SystemExit(f"need >= 60 user/assistant pairs in conversation_log, found {len(self.pairs)}")


# ---------------------------------------------------------------------------
# Prompt construction (real PromptBuilder)
# ---------------------------------------------------------------------------

class Conversation:
    """One conversation: fixed nonce, fixed session files, sliding real turns."""

    def __init__(self, data: Data, start_pair: int, heavy: bool, file_offset: int = 0):
        self.data = data
        self.start = start_pair
        self.nonce = secrets.token_hex(6)
        self.files = (data.files[file_offset:file_offset + 2] or data.files[:2]) if heavy else None
        self.history: list[Turn] = []

    def pair(self, turn_idx: int) -> tuple[str, str]:
        return self.data.pairs[(self.start + turn_idx) % len(self.data.pairs)]

    def main_messages(self, engine_name: str, layout: str, turn_idx: int) -> list[dict]:
        instruction, _ = self.pair(turn_idx)
        n = len(self.data.wiki)
        rag = [self.data.wiki[(self.start + turn_idx) % n], self.data.wiki[(self.start + turn_idx + 1) % n]]
        common = dict(
            instruction=instruction,
            current_datetime=dt.datetime.now().astimezone(),
            persona=self.data.persona,
            episodic_summary=self.data.episodes,
            rag_snippets=rag,
            working_memory=list(self.history),
        )
        structured = layout == "friendly" or engine_name == "omlx"
        if layout == "friendly":
            system, user, turns = PB.build(**common, session_files=None, emit_structured_working_memory=True)
            files_slot = PB._slot_session_files(self.files)
            if files_slot:
                system = system + "\n\n" + files_slot
        elif structured:
            system, user, turns = PB.build(**common, session_files=self.files, emit_structured_working_memory=True)
        else:
            system, user = PB.build(**common, session_files=self.files)
            turns = []
        msgs = [{"role": "system", "content": f"[conversation {self.nonce}]\n{system}"}]
        for t in turns or []:
            msgs.append({"role": "assistant" if t.role == "assistant" else "user", "content": t.content})
        msgs.append({"role": "user", "content": user})
        return msgs

    def advance(self, turn_idx: int):
        u, a = self.pair(turn_idx)
        self.history += [Turn(role="user", content=u), Turn(role="assistant", content=a)]

    def side_calls(self, turn_idx: int) -> list[tuple[str, list[dict]]]:
        u, a = self.pair(turn_idx)
        wsu_user = (
            "Previous working state:\ncurrent_focus: NONE\nopen_loops: NONE\n"
            f"recent_decisions: NONE\n\nNew turn:\nUser: {u}\nAssistant: {a}"
        )
        ext_user = (f"A user said: {u!r}\nWrite one sentence about them starting with 'The user'. "
                    "If no durable fact is present, write: NONE")
        return [
            ("working_state", [{"role": "system", "content": _build_wsu_system(self.data.persona, None)},
                               {"role": "user", "content": wsu_user}]),
            ("implicit_extraction", [{"role": "system", "content": _EXTRACTION_SYSTEM},
                                     {"role": "user", "content": ext_user}]),
        ]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class ColdModel:
    """Cold (no-reuse) TTFT seconds as a function of prompt tokens, per engine
    (linear interpolation between in-run calibration points)."""

    def __init__(self):
        self.points: list[tuple[int, float]] = []

    def add(self, tokens: int, secs: float):
        self.points.append((tokens, secs))
        self.points.sort()

    def predict(self, tokens: int) -> float | None:
        pts = self.points
        if not pts:
            return None
        if tokens <= pts[0][0]:
            return pts[0][1] * tokens / pts[0][0]
        for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
            if tokens <= t1:
                return s0 + (s1 - s0) * (tokens - t0) / (t1 - t0)
        t_last, s_last = pts[-1]
        return s_last * tokens / t_last


def call(engine, msgs: list[dict], max_tokens: int) -> dict:
    res = engine.stream(msgs, max_tokens)
    return {k: res.get(k) for k in ("ttft_s", "prefill_s", "prompt_tokens", "cached_tokens",
                                    "completion_tokens", "thinking_chars", "finish_reason")}


def measure_main(engine, cold: ColdModel, conv: Conversation, layout: str, turn_idx: int,
                 tag: str) -> dict:
    rec = call(engine, conv.main_messages(engine.name.split("_ctx")[0], layout, turn_idx), MAIN_MAX_TOKENS)
    rec["call"] = tag
    rec["turn"] = turn_idx + 1
    # Reuse is estimated from wall-clock TTFT (measured here, identically on
    # both engines) — Ollama's own prompt_eval_duration was seen disagreeing
    # with its TTFT in the smoke run, so engine-reported prefill_s is kept
    # in the record but not used for the estimate.
    est = cold.predict(rec["prompt_tokens"] or 0)
    if est:
        rec["cold_ttft_est_s"] = round(est, 3)
        rec["reuse_est"] = round(max(0.0, min(1.0, 1 - rec["ttft_s"] / est)), 3)
    if rec.get("cached_tokens") is not None and rec.get("prompt_tokens"):
        rec["reuse_reported"] = round(rec["cached_tokens"] / rec["prompt_tokens"], 3)
    print(f"    {tag:14s} turn {turn_idx + 1}: tokens={rec['prompt_tokens']} cached={rec.get('cached_tokens')} "
          f"prefill={rec.get('prefill_s')}s cold_est={rec.get('cold_ttft_est_s')}s ttft={rec['ttft_s']}s reuse_est={rec.get('reuse_est')}")
    return rec


def reload_model(engine):
    engine.unload_all()
    time.sleep(3)
    engine.stream([{"role": "user", "content": f"[reload {secrets.token_hex(4)}] hi"}], 4)


def run_scenario(engine, cold: ColdModel, data: Data, layout: str, heavy: bool, scenario: str,
                 base: int) -> list[dict]:
    out: list[dict] = []
    x = Conversation(data, base, heavy, file_offset=0)
    if scenario in ("A", "B"):
        for t in range(N_TURNS):
            out.append(measure_main(engine, cold, x, layout, t, "main"))
            if scenario == "B":
                for label, msgs in x.side_calls(t):
                    r = call(engine, msgs, SIDE_MAX_TOKENS)
                    r.update({"call": label, "turn": t + 1})
                    out.append(r)
            x.advance(t)
    elif scenario == "C":
        for t in range(2):
            out.append(measure_main(engine, cold, x, layout, t, "main"))
            x.advance(t)
        for k in range(3):
            other = Conversation(data, base + 20 + 7 * k, heavy, file_offset=2 + k)
            out.append(measure_main(engine, cold, other, layout, 0, f"other_{k + 1}"))
        out.append(measure_main(engine, cold, x, layout, 2, "return"))
    elif scenario == "D":
        for t in range(2):
            out.append(measure_main(engine, cold, x, layout, t, "main"))
            x.advance(t)
        reload_model(engine)
        out.append(measure_main(engine, cold, x, layout, 2, "after_reload"))
    return out


def calibrate(engine, data: Data) -> ColdModel:
    """Cold (no-reuse) prefill points, using the heavy and light main prompts
    themselves with a fresh nonce each."""
    cold = ColdModel()
    for heavy in (False, True):
        for layout in ("today",):
            conv = Conversation(data, 400, heavy)
            for t in range(2):
                conv.advance(t)
            res = engine.stream(conv.main_messages(engine.name.split("_ctx")[0], layout, 2), 1)
            if res.get("prompt_tokens") and res.get("ttft_s"):
                cold.add(res["prompt_tokens"], res["ttft_s"])
    # a long point too, so heavy friendly prompts never extrapolate far
    conv = Conversation(data, 410, True, file_offset=1)
    msgs = conv.main_messages(engine.name.split("_ctx")[0], "today", 0)
    msgs[-1]["content"] += "\n\n" + "\n\n".join(f.content[:12000] for f in data.files[:2])
    res = engine.stream(msgs, 1)
    if res.get("prompt_tokens") and res.get("ttft_s"):
        cold.add(res["prompt_tokens"], res["ttft_s"])
    print(f"  cold calibration (tokens, TTFT s): {cold.points}")
    return cold


def summarize(runs: list[dict]) -> None:
    print("\n" + "=" * 92 + "\nSUMMARY — main-call reuse and TTFT\n" + "=" * 92)
    print(f"{'engine':8} {'layout':9} {'size':6} {'scen':5} {'call/turn':16} {'tokens':>7} {'cached':>7} "
          f"{'reuse':>6} {'TTFT s':>7} {'cold est':>8}")
    for run in runs:
        for key, recs in run["scenarios"].items():
            scen, size, layout = key.split("|")
            for r in recs:
                if r["call"] in ("working_state", "implicit_extraction"):
                    continue
                print(f"{run['engine']:8} {layout:9} {size:6} {scen:5} {r['call'] + ' t' + str(r['turn']):16} "
                      f"{r.get('prompt_tokens') or 0:>7} {str(r.get('cached_tokens', '-')):>7} "
                      f"{(r.get('reuse_reported') if r.get('reuse_reported') is not None else r.get('reuse_est')) or 0:>6.2f} "
                      f"{r['ttft_s']:>7.2f} {r.get('cold_ttft_est_s') or 0:>8.2f}")
        print(f"  {run['engine']} RAM after scenarios: "
              + ", ".join(f"{k}={v['engine_footprint_mb']:.0f}MB" for k, v in run["ram_after"].items()))
    print("\nMean main-call TTFT for turns >= 2 (scenarios A/B), seconds:")
    for run in runs:
        for key, recs in run["scenarios"].items():
            scen, size, layout = key.split("|")
            if scen not in ("A", "B"):
                continue
            ts = [r["ttft_s"] for r in recs if r["call"] == "main" and r["turn"] >= 2]
            if ts:
                print(f"  {run['engine']:8} {layout:9} {size:6} {scen}: {statistics.mean(ts):6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="ollama,omlx")
    ap.add_argument("--layouts", default="today,friendly")
    ap.add_argument("--sizes", default="light,heavy")
    ap.add_argument("--scenarios", default="A,B,C,D")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args()

    if _pid_on_port(8001):
        print("WARNING: Localist backend (:8001) is running — close the desktop app for cleaner numbers.")

    data = Data()
    print(f"data: {len(data.pairs)} turn pairs, {len(data.episodes)} episodes, {len(data.wiki)} wiki pages, "
          f"{len(data.files)} raw docs, persona={'yes' if data.persona else 'no'}")

    all_engines = [OllamaEngine(), OMLXEngine()]
    chosen = [e for e in all_engines if e.name in args.engines.split(",")]
    runs = []
    for engine in chosen:
        print(f"\n=== {engine.name} ({engine.model}) ===")
        for e in all_engines:
            e.unload_all()
        time.sleep(3)
        engine.stream([{"role": "user", "content": f"[load {secrets.token_hex(4)}] hi"}], 4)
        cold = calibrate(engine, data)
        run = {"engine": engine.name, "model": engine.model, "cold_points": cold.points,
               "loaded_idle": snapshot(engine.pids()), "scenarios": {}, "ram_after": {}}
        base = 100
        for layout in args.layouts.split(","):
            for size in args.sizes.split(","):
                for scen in args.scenarios.split(","):
                    key = f"{scen}|{size}|{layout}"
                    print(f"  -- scenario {scen}, {size}, layout={layout}")
                    base += 37  # different real conversation per scenario run
                    run["scenarios"][key] = run_scenario(engine, cold, data, layout, size == "heavy", scen, base)
                    run["ram_after"][key] = snapshot(engine.pids())
        engine.unload_all()
        runs.append(run)

    meta = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "localist_backend_running": _pid_on_port(8001) is not None,
            "main_max_tokens": MAIN_MAX_TOKENS, "side_max_tokens": SIDE_MAX_TOKENS, "turns": N_TURNS,
            "args": vars(args)}
    Path(args.out).write_text(json.dumps({"meta": meta, "runs": runs}, indent=2))
    print(f"\nRaw results written to {args.out}")
    summarize(runs)


if __name__ == "__main__":
    main()
