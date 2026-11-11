"""
Diagnostic: diag_openrouter_news_feed_garbage.py
================================================
2026-11-11. On the OpenRouter backend (nvidia/nemotron-3-ultra-550b-a55b:free),
News Feed "Tell me more about this news story" turns came back as garbage
("那些\"", "i.ζiusfrac.\n\nto.τ", " שח") or empty (→ _EMPTY_RESPONSE_FALLBACK),
while plain "Look up ..." web-search turns on the same backend answered
normally. The DB shows the tool calls succeeded; the packaged app keeps no
backend log, so finish_reason / usage / provider / the actual prompt were lost.

This script answers: is the garbage reproducible, and what does the raw
OpenRouter stream look like when it happens?

  Stage 1 (no OpenRouter quota): runs the REAL ControllerAgent on each case,
          with the same context the News Feed sends
          ({web_search_queries: [title], news_article_url: url}), against a
          throwaway temp-dir DB/wiki, and a capture-only runtime that records
          the main_dispatch prompt+system and returns "" for every other
          inference call. Tools run for real through the live localist-mcp
          (port 8003), so news_search / url_fetch enrichment behave as in the
          app. Each case is a fresh conversation (the 09:32 failure was the
          first turn of its conversation).
  Stage 2 (COSTS QUOTA: one request per case per repeat): sends each captured
          main_dispatch prompt to OpenRouter with exactly the payload
          OpenRouterRuntimeClient.infer_stream() builds (build_messages,
          stream, max_tokens 1024, temperature 0.2, reasoning enabled:false),
          and records every raw SSE event: finish_reason,
          native_finish_reason, provider, usage, reasoning deltas, content.

READ-ONLY with respect to Localist data: temp DB and wiki; user profile and
persona are not loaded. The OpenRouter key is read from the environment, else
backend/.env, else the packaged app's .env; it is never printed.

Not part of the pytest suite. A surprising result is the finding — do not edit
this script to make the output look better.

Usage:
    cd backend && source .venv/bin/activate
    python ../diagnostics/diag_openrouter_news_feed_garbage.py \
        [--cases all|news|control|<case>,<case>] [--repeat 1] \
        [--model nvidia/nemotron-3-ultra-550b-a55b:free] [--capture-only]
        [--reasoning off|default] [--out-suffix _reasoning_default]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
import unicodedata
import uuid
from typing import Any, Generator

import requests

_REPO = pathlib.Path(__file__).resolve().parents[1]
_BACKEND = _REPO / "backend"
sys.path.insert(0, str(_BACKEND / "src"))

_API = "https://openrouter.ai/api/v1"
_APP_ENV = pathlib.Path.home() / "Library/Application Support/Localist/.env"
_OUT_CSV = pathlib.Path(__file__).with_name("openrouter_news_feed_garbage_results.csv")
_OUT_RAW = pathlib.Path(__file__).with_name("openrouter_news_feed_garbage_raw.jsonl")

# The three News Feed clicks that failed on 2026-11-11 (titles/URLs from that
# day's news_brief_cache), plus the plain web-search turn that succeeded on
# OpenRouter at 09:45 as the control.
_CASES: dict[str, dict[str, Any]] = {
    "news_verge_gta": {
        "kind": "news",
        "title": "GTA VI will be available on Xbox Cloud Gaming for Xbox One consoles - The Verge",
        "url": "https://www.theverge.com/news/1030714/gta-vi-xbox-cloud-gaming-support",
    },
    "news_gsmarena_iphone": {
        "kind": "news",
        "title": "iPhone Air 2 and iPhone 18 dummy models leak - GSMArena.com news - GSMArena.com",
        "url": "https://www.gsmarena.com/iphone_air_2_and_iphone_18_dummy_models_leak-news-75421.php",
    },
    "news_ign_achievements": {
        "kind": "news",
        "title": "Achievements Could Be Making Gaming Worse for Everyone - IGN",
        "url": "https://www.ign.com/articles/phantom-blade-zero-doesnt-have-an-achievement-for-beating-its-hardest-mode-a-growing-trend-in-the-games-industry-is-to-blame",
    },
    "control_lookup": {
        "kind": "control",
        "instruction": "Look up GTA 6 preorder sales stats.",
    },
}


# ---------------------------------------------------------------------------
# Key / settings
# ---------------------------------------------------------------------------

def _env_value(path: pathlib.Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _setting(key: str) -> str:
    return (os.environ.get(key) or _env_value(_BACKEND / ".env", key)
            or _env_value(_APP_ENV, key))


# ---------------------------------------------------------------------------
# Stage 1 — capture the real prompts
# ---------------------------------------------------------------------------

class CaptureRuntime:
    """BaseRuntimeClient surface; records calls, spends no quota."""

    is_local = False
    wants_cache_warmup = False

    def __init__(self, model: str) -> None:
        self._model = model
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, prompt: str, system: str, max_tokens: int,
                temperature: float, label: str) -> str:
        self.calls.append({"method": method, "label": label, "prompt": prompt,
                           "system": system, "max_tokens": max_tokens,
                           "temperature": temperature})
        return ""

    def infer(self, prompt: str, system: str = "", max_tokens: int = 1024,
              temperature: float = 0.2, label: str = "", timeout: float | None = None,
              **_: Any) -> str:
        return self._record("infer", prompt, system, max_tokens, temperature, label)

    def infer_stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
                     temperature: float = 0.2, label: str = "",
                     timeout: float | None = None, **_: Any) -> Generator[str, None, None]:
        self._record("infer_stream", prompt, system, max_tokens, temperature, label)
        if label == "main_dispatch":
            yield "(captured)"   # non-empty so the turn completes normally

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("capture runtime does not embed")

    def health_check(self) -> dict:
        return {"reachable": True, "models": [self._model], "chat_model_found": True,
                "embed_model_found": None, "base_url": _API, "chat_model": self._model}


def capture(cases: list[str], model: str) -> dict[str, dict[str, Any]]:
    from localist.controller_agent import ControllerAgent
    from localist.conversational_agent import ConversationalAgent
    from localist.memory_manager import MemoryManager
    from localist.ollama_runtime_client import OllamaRuntimeClient
    from localist.wiki_agent import WikiAgent

    # Local embeddings as the app does for a chat-only backend (§16.32).
    embed_fn, embed_model = None, _setting("LOCALIST_EMBEDDING_MODEL_OLLAMA") or None
    if embed_model:
        emb = OllamaRuntimeClient(chat_model="embed-only", embedding_model=embed_model,
                                  base_url=_setting("LOCALIST_OLLAMA_URL") or "http://localhost:11434")
        if emb.health_check().get("embed_model_found"):
            embed_fn = emb.embed
    print(f"Embeddings: {'local Ollama ' + embed_model if embed_fn else 'keyword-only (BM25)'}")

    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="localist-diag-news-") as tmp:
        root = pathlib.Path(tmp)
        (root / "wiki").mkdir()
        rt = CaptureRuntime(model)
        mm = MemoryManager(db_path=root / "diag.db", embed_fn=embed_fn,
                           embedding_model_name=embed_model if embed_fn else None)
        wiki_agent = WikiAgent(runtime=rt, project_root=root, memory_manager=mm)
        conv_agent = ConversationalAgent(runtime=rt, memory_manager=mm, project_root=root)
        controller = ControllerAgent(
            runtime=rt, agents=[wiki_agent, conv_agent], memory_manager=mm,
            embed_fn=embed_fn, embedding_model_name=embed_model if embed_fn else None,
            calibrated_thresholds=None, episodic_write_approval=True,
            wiki_dir=root / "wiki",
        )

        for name in cases:
            case = _CASES[name]
            conversation_id = f"diag-{uuid.uuid4()}"
            context: dict[str, Any] = {"conversation_id": conversation_id,
                                       "wiki_dir": str(root / "wiki")}
            if case["kind"] == "news":
                # Exactly what PreviewsPanel.svelte handleAskAboutArticle sends.
                instruction = f'Tell me more about this news story: "{case["title"]}"'
                context.update({"web_search_queries": [case["title"]],
                                "news_article_url": case["url"]})
            else:
                instruction = case["instruction"]
            task = {"task_id": str(uuid.uuid4()), "instruction": instruction,
                    "context": context}
            before = len(rt.calls)
            t0 = time.monotonic()
            plan = controller.route_task(instruction, context)
            result = controller.handle_task_with_plan(task, plan, on_token=lambda _c: None)
            calls = rt.calls[before:]
            main = [c for c in calls if c["label"] == "main_dispatch"]
            meta = result.get("metadata") or {}
            steps = meta.get("workflow_steps") or []
            out[name] = {
                "instruction": instruction,
                "route": f"{plan.agent} p={getattr(plan, 'priority', '?')} tools={plan.tools_to_call}",
                "tools_fired": meta.get("tools_fired"),
                "workflow_steps": [{k: s.get(k) for k in ("tool_name", "parameters", "success")}
                                   for s in steps],
                "labels": [c["label"] for c in calls],
                "main": main[-1] if main else None,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            m = out[name]["main"]
            print(f"  {name:24s} route={out[name]['route']}  tools={out[name]['tools_fired']}"
                  f"  steps={[s['tool_name'] + ('' if s['success'] else '(FAIL)') for s in out[name]['workflow_steps']]}"
                  f"  main_prompt_chars={len(m['prompt']) + len(m['system']) if m else None}"
                  f"  labels={out[name]['labels']}")
    return out


# ---------------------------------------------------------------------------
# Stage 2 — send the captured prompts to OpenRouter, record the raw stream
# ---------------------------------------------------------------------------

def _garbage_score(text: str) -> dict[str, Any]:
    """Cheap signals; the human reading the CSV decides."""
    stripped = text.strip()
    letters = [ch for ch in stripped if ch.isalpha()]
    non_latin = [ch for ch in letters
                 if not unicodedata.name(ch, "").startswith(("LATIN", "DIGIT"))]
    return {
        "content_chars": len(text),
        "words": len(stripped.split()),
        "non_latin_letter_ratio": round(len(non_latin) / len(letters), 2) if letters else None,
        "looks_degenerate": (len(stripped) < 40) or (bool(letters) and len(non_latin) / len(letters) > 0.2),
    }


_REASONING = {"off": {"enabled": False}, "default": None}


def send(key: str, model: str, main: dict[str, Any], timeout: float,
         reasoning: str = "off") -> dict[str, Any]:
    from localist.openai_compat import build_messages

    payload = {   # mirrors OpenRouterRuntimeClient.infer_stream()
        "model":       model,
        "messages":    build_messages(main["prompt"], main["system"]),
        "stream":      True,
        "max_tokens":  main["max_tokens"],
        "temperature": main["temperature"],
    }
    # "off" is what the app sends; "default" omits the field (A/B only).
    if _REASONING[reasoning] is not None:
        payload["reasoning"] = _REASONING[reasoning]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "Localist diagnostic"}
    t0 = time.monotonic()
    resp = requests.post(f"{_API}/chat/completions", headers=headers, json=payload,
                         stream=True, timeout=timeout)
    rec: dict[str, Any] = {"status": resp.status_code}
    if resp.status_code != 200:
        rec["error_body"] = resp.text[:2000].replace(key, "<KEY>")
        rec["latency_s"] = round(time.monotonic() - t0, 2)
        return rec

    events: list[Any] = []
    content, reasoning = [], []
    ttft = None
    rec.update(finish_reason=None, native_finish_reason=None, provider=None,
               usage=None, mid_stream_error=None, comment_lines=0, generation_id=None)
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if line.startswith(":"):
            rec["comment_lines"] += 1
            continue
        if not line.startswith("data:"):
            events.append({"_non_data_line": line[:500]})
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            rec["saw_done"] = True
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            events.append({"_bad_json": data[:500]})
            continue
        events.append(obj)
        rec["generation_id"] = obj.get("id") or rec["generation_id"]
        rec["provider"] = obj.get("provider") or rec["provider"]
        if obj.get("error"):
            rec["mid_stream_error"] = obj["error"]
        if obj.get("usage"):
            rec["usage"] = obj["usage"]
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                ttft = ttft or time.monotonic()
                content.append(d["content"])
            if d.get("reasoning"):
                reasoning.append(d["reasoning"])
            if ch.get("finish_reason"):
                rec["finish_reason"] = ch["finish_reason"]
            if ch.get("native_finish_reason"):
                rec["native_finish_reason"] = ch["native_finish_reason"]
    resp.close()
    rec.setdefault("saw_done", False)
    rec["latency_s"] = round(time.monotonic() - t0, 2)
    rec["ttft_s"] = round(ttft - t0, 2) if ttft else None
    rec["content"] = "".join(content)
    rec["reasoning_text"] = "".join(reasoning)
    rec["event_count"] = len(events)
    rec["events"] = events
    rec.update(_garbage_score(rec["content"]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cases", default="all")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--reasoning", choices=["off", "default"], default="off",
                    help="off = production payload; default = omit the reasoning field (A/B)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for output files, so an A/B run doesn't overwrite the baseline")
    args = ap.parse_args()
    global _OUT_CSV, _OUT_RAW
    if args.out_suffix:
        _OUT_CSV = _OUT_CSV.with_name(_OUT_CSV.stem + args.out_suffix + _OUT_CSV.suffix)
        _OUT_RAW = _OUT_RAW.with_name(_OUT_RAW.stem + args.out_suffix + _OUT_RAW.suffix)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.cases == "all":
        cases = list(_CASES)
    elif args.cases in ("news", "control"):
        cases = [n for n, c in _CASES.items() if c["kind"] == args.cases]
    else:
        cases = [c.strip() for c in args.cases.split(",")]

    print(f"== Stage 1  capture real prompts (no quota)  cases={cases}")
    captured = capture(cases, args.model)
    if args.capture_only:
        _OUT_RAW.write_text("".join(json.dumps({"case": n, "stage": "capture", **c}) + "\n"
                                    for n, c in captured.items()))
        print(f"Wrote {_OUT_RAW}")
        return 0

    key = _setting("OPENROUTER_API_KEY")
    if not key:
        print("No OPENROUTER_API_KEY found (env, backend/.env, app .env).")
        return 2

    print(f"\n== Stage 2  OpenRouter  model={args.model}  "
          f"requests={sum(1 for n in cases if captured[n]['main']) * args.repeat}")
    rows: list[dict[str, Any]] = []
    with _OUT_RAW.open("w") as raw_out:
        for n in cases:
            raw_out.write(json.dumps({"case": n, "stage": "capture", **captured[n]}) + "\n")
        for rep in range(1, args.repeat + 1):
            for n in cases:
                main_call = captured[n]["main"]
                if not main_call:
                    print(f"  {n}: no main_dispatch captured — skipped")
                    continue
                rec = send(key, args.model, main_call, args.timeout, args.reasoning)
                raw_out.write(json.dumps({"case": n, "stage": "send", "repeat": rep,
                                          "model": args.model, **rec}) + "\n")
                usage = rec.get("usage") or {}
                row = {
                    "case": n, "repeat": rep, "model": args.model, "reasoning": args.reasoning,
                    "status": rec["status"],
                    "prompt_chars": len(main_call["prompt"]) + len(main_call["system"]),
                    "steps": ";".join(s["tool_name"] + ("" if s["success"] else "(FAIL)")
                                      for s in captured[n]["workflow_steps"]),
                    "provider": rec.get("provider"),
                    "finish_reason": rec.get("finish_reason"),
                    "native_finish_reason": rec.get("native_finish_reason"),
                    "saw_done": rec.get("saw_done"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "reasoning_chars": len(rec.get("reasoning_text") or ""),
                    "content_chars": rec.get("content_chars"),
                    "non_latin_letter_ratio": rec.get("non_latin_letter_ratio"),
                    "looks_degenerate": rec.get("looks_degenerate"),
                    "latency_s": rec.get("latency_s"), "ttft_s": rec.get("ttft_s"),
                    "mid_stream_error": json.dumps(rec.get("mid_stream_error"))[:300] if rec.get("mid_stream_error") else "",
                    "error_body": (rec.get("error_body") or "")[:300],
                    "content_head": (rec.get("content") or "")[:160].replace("\n", "⏎"),
                }
                rows.append(row)
                print(f"  [{rep}] {n:24s} status={row['status']} finish={row['finish_reason']}"
                      f"/{row['native_finish_reason']} provider={row['provider']} "
                      f"tok={row['prompt_tokens']}→{row['completion_tokens']} "
                      f"degenerate={row['looks_degenerate']} "
                      f"{row['error_body'] or row['mid_stream_error']}\n"
                      f"      content: {row['content_head']!r}")

    if rows:
        with _OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {_OUT_CSV}\nWrote {_OUT_RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
