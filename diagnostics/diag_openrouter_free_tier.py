"""
Diagnostic: diag_openrouter_free_tier.py
========================================
Phase 0 of the 2026-11-10 OpenRouter-backend plan. Ollama Cloud's free plan
became pay-as-you-go on 2026-11-09; OpenRouter's free tier
(google/gemma-4-31b-it:free) is the candidate replacement. Before any product
code is written, this answers, against the real API:

  Part A  Requests per turn. OpenRouter's free tier is counted in requests
          (20/min, 50/day; 1,000/day after a one-time $10 credit purchase),
          and one Localist turn is more than one model call (answer +
          gated implicit extraction + working-state update + classifiers).
          Runs the real ControllerAgent on a scripted conversation, with
          every runtime call counted, timed, and token-metered.
  Part B  API behaviour probes, one request each:
            B1  GET /api/v1/key     — usage, limit, is_free_tier (no quota)
            B2  GET endpoints       — which providers serve the free model
            B3  reasoning variants  — default / enabled:false / exclude:true:
                                      which one actually stops hidden
                                      reasoning (Ollama's think:false, §4.6.2)
            B4  streaming SSE       — chunk shape, empty-content completion,
                                      reasoning deltas, final usage chunk
            B5  data_collection:deny — does the free endpoint survive the
                                      no-training filter
            B6  native tools        — does the free endpoint return tool_calls
          Any 429/402 seen anywhere is recorded with its full body.

READ-ONLY with respect to Localist: Part A builds its own MemoryManager on a
throwaway temp-dir SQLite DB, with temp wiki/project dirs, so nothing reaches
the real (packaged-app) DB or wiki — see reference_dev_env_points_at_packaged_db.
The user profile and persona are NOT loaded, so nothing personal is sent to
OpenRouter; the scripted turns are synthetic. Part A's tool turn uses the live
localist-mcp (port 8003) if it is running. Embeddings, when available, come
from the local Ollama daemon (LOCALIST_EMBEDDING_MODEL_OLLAMA), never
OpenRouter.

COSTS QUOTA: about 6 requests for Part B and about 15-25 for Part A, against a
free-tier allowance of 50/day. Use --part to run one half.

Not part of the pytest suite. A surprising result is the finding — do not edit
this script to make the output look better.

Usage:
    cd backend
    source .venv/bin/activate
    export OPENROUTER_API_KEY=...      # or put it in backend/.env
    python ../diagnostics/diag_openrouter_free_tier.py [--part a|b|all]
        [--model google/gemma-4-31b-it:free] [--reasoning off|exclude|default]
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
import uuid
from typing import Any, Generator

import requests

_REPO = pathlib.Path(__file__).resolve().parents[1]
_BACKEND = _REPO / "backend"
_API = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemma-4-31b-it:free"
_CSV_OUT = pathlib.Path(__file__).with_name("openrouter_free_tier_results.csv")
_JSON_OUT = pathlib.Path(__file__).with_name("openrouter_free_tier_probes.json")

# Synthetic conversation for Part A — one of each turn kind that matters for
# request counting. Nothing personal.
_TURNS: list[tuple[str, str]] = [
    ("chit_chat",       "Hi! What kinds of things can you help me with?"),
    ("explicit_memory", "Remember that I prefer metric units."),
    ("implicit_fact",   "I'm planting tomatoes in my garden this weekend."),
    ("factual",         "How deep should I plant tomato seedlings?"),
    ("web_search",      "Search the web for the current OpenRouter free model rate limits."),
    ("follow_up",       "Thanks. Summarize what we discussed in two sentences."),
]

_REASONING_VARIANTS: dict[str, dict[str, Any] | None] = {
    "default": None,
    "off":     {"enabled": False},
    "exclude": {"exclude": True},
}


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _env_file_value(key: str) -> str:
    env_path = _BACKEND / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _setting(key: str) -> str:
    return os.environ.get(key) or _env_file_value(key)


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<OPENROUTER_API_KEY>") if key else text


# ---------------------------------------------------------------------------
# Call log — every HTTP request this script makes to /chat/completions
# ---------------------------------------------------------------------------

class CallLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.phase = ""
        self.turn = ""

    def add(self, **row: Any) -> None:
        row.setdefault("phase", self.phase)
        row.setdefault("turn", self.turn)
        self.rows.append(row)


# ---------------------------------------------------------------------------
# Minimal OpenRouter runtime client (diagnostic-only; Phase 2 builds the real
# one). Satisfies the BaseRuntimeClient surface ControllerAgent uses.
# ---------------------------------------------------------------------------

class DiagOpenRouterClient:
    is_local = False

    def __init__(self, key: str, model: str, reasoning: str, log: CallLog,
                 timeout: float = 120.0) -> None:
        self._key = key
        self._model = model
        self._reasoning = _REASONING_VARIANTS[reasoning]
        self._log = log
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "X-Title": "Localist diagnostic",
        }

    def _payload(self, prompt: str, system: str, max_tokens: int,
                 temperature: float, stream: bool) -> dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if self._reasoning is not None:
            body["reasoning"] = self._reasoning
        return body

    def _post(self, body: dict[str, Any], stream: bool, timeout: float | None):
        """POST with one bounded 429 retry; every attempt is logged."""
        for attempt in (1, 2):
            resp = requests.post(
                f"{_API}/chat/completions", headers=self._headers(), json=body,
                stream=stream, timeout=timeout or self._timeout,
            )
            if resp.status_code == 429 and attempt == 1:
                retry_after = min(float(resp.headers.get("Retry-After") or 10), 30)
                self._log.errors.append({
                    "status": 429, "phase": self._log.phase, "turn": self._log.turn,
                    "retry_after": resp.headers.get("Retry-After"),
                    "body": _redact(resp.text[:2000], self._key),
                    "headers": {k: v for k, v in resp.headers.items()
                                if k.lower().startswith(("x-ratelimit", "retry"))},
                })
                self._log.add(label="(429 retry)", method="http", status=429,
                              latency_s=0, prompt_chars=0)
                time.sleep(retry_after)
                continue
            return resp
        return resp

    def _fail(self, resp: requests.Response, label: str, t0: float,
              prompt_chars: int, method: str) -> RuntimeError:
        body = _redact(resp.text[:2000], self._key)
        self._log.errors.append({"status": resp.status_code, "phase": self._log.phase,
                                 "turn": self._log.turn, "label": label, "body": body})
        self._log.add(label=label, method=method, status=resp.status_code,
                      latency_s=round(time.monotonic() - t0, 2), prompt_chars=prompt_chars)
        return RuntimeError(f"OpenRouter HTTP {resp.status_code}: {body[:300]}")

    def infer(self, prompt: str, system: str = "", max_tokens: int = 1024,
              temperature: float = 0.2, label: str = "",
              timeout: float | None = None) -> str:
        t0 = time.monotonic()
        body = self._payload(prompt, system, max_tokens, temperature, stream=False)
        resp = self._post(body, stream=False, timeout=timeout)
        if resp.status_code != 200:
            raise self._fail(resp, label, t0, len(prompt) + len(system), "infer")
        data = resp.json()
        if "error" in data:
            self._log.errors.append({"status": "200+error", "label": label,
                                     "phase": self._log.phase, "turn": self._log.turn,
                                     "body": json.dumps(data["error"])[:2000]})
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        self._log.add(
            label=label or "(unlabelled)", method="infer", status=200,
            latency_s=round(time.monotonic() - t0, 2),
            prompt_chars=len(prompt) + len(system),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cost=usage.get("cost"), provider=data.get("provider"),
            finish_reason=choice.get("finish_reason"),
            empty_content=not content.strip(),
        )
        return content

    def infer_stream(self, prompt: str, system: str = "", max_tokens: int = 1024,
                     temperature: float = 0.2, label: str = "",
                     timeout: float | None = None) -> Generator[str, None, None]:
        t0 = time.monotonic()
        body = self._payload(prompt, system, max_tokens, temperature, stream=True)
        resp = self._post(body, stream=True, timeout=timeout)
        if resp.status_code != 200:
            raise self._fail(resp, label, t0, len(prompt) + len(system), "infer_stream")
        stats = _StreamStats()
        try:
            for delta in _iter_openrouter_sse(resp, stats):
                yield delta
        finally:
            usage = stats.usage or {}
            self._log.add(
                label=label or "(unlabelled)", method="infer_stream", status=200,
                latency_s=round(time.monotonic() - t0, 2),
                ttft_s=round(stats.first_token_at - t0, 2) if stats.first_token_at else None,
                prompt_chars=len(prompt) + len(system),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                cost=usage.get("cost"), provider=stats.provider,
                finish_reason=stats.finish_reason,
                empty_content=stats.content_chars == 0,
                reasoning_chunks=stats.reasoning_chunks,
                mid_stream_error=stats.error,
            )
            if stats.error:
                self._log.errors.append({"status": "mid-stream", "label": label,
                                         "phase": self._log.phase, "turn": self._log.turn,
                                         "body": stats.error})

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("OpenRouter client does not embed — embeddings stay local.")

    def health_check(self) -> dict:
        return {"reachable": True, "models": [self._model], "chat_model_found": True,
                "embed_model_found": None, "base_url": _API, "chat_model": self._model}


class _StreamStats:
    def __init__(self) -> None:
        self.first_token_at: float | None = None
        self.content_chars = 0
        self.reasoning_chunks = 0
        self.chunks = 0
        self.comment_lines = 0
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.provider: str | None = None
        self.error: str | None = None


def _iter_openrouter_sse(resp: requests.Response, stats: _StreamStats) -> Generator[str, None, None]:
    """Parse OpenRouter's OpenAI-style SSE, recording everything unusual."""
    # Decode bytes as UTF-8 explicitly: OpenRouter's text/event-stream has no
    # charset, so requests' decode_unicode=True falls back to ISO-8859-1 and
    # mangles non-ASCII (first Part B run: an en dash arrived as "â\x80\x93").
    for raw_bytes in resp.iter_lines():
        raw = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
        if raw == "":
            continue
        if raw.startswith(":"):          # OpenRouter keep-alive comments
            stats.comment_lines += 1
            continue
        if not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        stats.chunks += 1
        if "error" in obj:
            stats.error = json.dumps(obj["error"])[:2000]
        stats.provider = obj.get("provider") or stats.provider
        if obj.get("usage"):
            stats.usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning") or delta.get("reasoning_details"):
                stats.reasoning_chunks += 1
            if choice.get("finish_reason"):
                stats.finish_reason = choice["finish_reason"]
            text = delta.get("content") or ""
            if text:
                if stats.first_token_at is None:
                    stats.first_token_at = time.monotonic()
                stats.content_chars += len(text)
                yield text


# ---------------------------------------------------------------------------
# Part B — raw API probes
# ---------------------------------------------------------------------------

def _chat(key: str, body: dict[str, Any], timeout: float = 120.0) -> tuple[int, dict, float]:
    t0 = time.monotonic()
    resp = requests.post(
        f"{_API}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "Localist diagnostic"},
        json=body, timeout=timeout,
    )
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text[:2000]}
    return resp.status_code, data, round(time.monotonic() - t0, 2)


def _summarize_completion(status: int, data: dict, latency: float) -> dict[str, Any]:
    if status != 200 or "error" in data:
        return {"status": status, "latency_s": latency, "error": data.get("error", data)}
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "status": status, "latency_s": latency, "provider": data.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(msg.get("content") or ""),
        "content_head": (msg.get("content") or "")[:160],
        "has_reasoning_field": bool(msg.get("reasoning") or msg.get("reasoning_details")),
        "reasoning_chars": len(msg.get("reasoning") or ""),
        "tool_calls": msg.get("tool_calls"),
        "usage": usage,
    }


def run_part_b(key: str, model: str, reasoning: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    q = "In one sentence: why do tomatoes need full sun?"

    print("\n== B1  GET /api/v1/key")
    r = requests.get(f"{_API}/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    out["B1_key"] = {"status": r.status_code,
                     "data": r.json().get("data") if r.ok else _redact(r.text[:500], key)}
    print(json.dumps(out["B1_key"], indent=2))

    print("\n== B2  free-model endpoints")
    r = requests.get(f"{_API}/models/{model}/endpoints", timeout=30)
    eps = []
    if r.ok:
        for e in (r.json().get("data") or {}).get("endpoints", []):
            eps.append({k: e.get(k) for k in (
                "provider_name", "tag", "context_length", "max_completion_tokens",
                "quantization", "uptime_last_1d", "supports_implicit_caching")}
                | {"tools": "tools" in (e.get("supported_parameters") or []),
                   "reasoning_param": "reasoning" in (e.get("supported_parameters") or [])})
    out["B2_endpoints"] = {"status": r.status_code, "endpoints": eps}
    print(json.dumps(out["B2_endpoints"], indent=2))

    print("\n== B3  reasoning variants (3 requests)")
    out["B3_reasoning"] = {}
    for name, reasoning_param in _REASONING_VARIANTS.items():
        body: dict[str, Any] = {"model": model, "max_tokens": 400, "temperature": 0.2,
                                "messages": [{"role": "user", "content": q}]}
        if reasoning_param is not None:
            body["reasoning"] = reasoning_param
        res = _summarize_completion(*_chat(key, body))
        out["B3_reasoning"][name] = res
        print(f"  {name:8s} -> {json.dumps(res)[:400]}")

    print(f"\n== B4  streaming (reasoning={reasoning})")
    log = CallLog()
    log.phase, log.turn = "B4", "stream"
    client = DiagOpenRouterClient(key, model, reasoning, log)
    try:
        text = "".join(client.infer_stream(q, system="You are concise.", max_tokens=400,
                                           label="B4_stream"))
        out["B4_stream"] = {"row": log.rows[-1] if log.rows else None,
                            "content_head": text[:160], "errors": log.errors}
    except RuntimeError as exc:
        out["B4_stream"] = {"error": _redact(str(exc), key), "errors": log.errors}
    print(json.dumps(out["B4_stream"], indent=2, default=str))

    print("\n== B5  provider.data_collection = deny")
    body = {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": q}],
            "provider": {"data_collection": "deny"}}
    if _REASONING_VARIANTS[reasoning] is not None:
        body["reasoning"] = _REASONING_VARIANTS[reasoning]
    out["B5_deny"] = _summarize_completion(*_chat(key, body))
    print(json.dumps(out["B5_deny"], indent=2, default=str))

    print("\n== B6  native tool calling")
    body = {
        "model": model, "max_tokens": 300, "temperature": 0.0,
        "messages": [{"role": "user", "content": "What's the weather in Lisbon right now? Use a tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
    }
    if _REASONING_VARIANTS[reasoning] is not None:
        body["reasoning"] = _REASONING_VARIANTS[reasoning]
    out["B6_tools"] = _summarize_completion(*_chat(key, body))
    print(json.dumps(out["B6_tools"], indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# Part C — availability (added after the first Part B run, 2026-11-10, hit
# 429 "upstream_provider_shared_pool" on every request). 429s were confirmed
# NOT to consume free_model_daily_requests, so polling only costs quota on
# success. C1 polls the target model; C2 sends one tiny request to every
# free model that advertises tool support, to compare congestion.
# ---------------------------------------------------------------------------

def _tiny(key: str, model: str, reasoning: str) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "max_tokens": 20, "temperature": 0.0,
                            "messages": [{"role": "user", "content": "Reply with the word OK."}]}
    if _REASONING_VARIANTS[reasoning] is not None:
        body["reasoning"] = _REASONING_VARIANTS[reasoning]
    status, data, latency = _chat(key, body, timeout=60)
    meta = (data.get("error") or {}).get("metadata") or {}
    return {"model": model, "status": status if "error" not in data else (data["error"].get("code") or status),
            "latency_s": latency, "limit_source": meta.get("limit_source"),
            "provider": data.get("provider") or meta.get("provider_name"),
            "content": (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")[:40]}


def run_part_c(key: str, model: str, reasoning: str, poll_minutes: float,
               poll_interval: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    print(f"\n== C1  poll {model} every {poll_interval:.0f}s for {poll_minutes:.0f} min")
    polls = []
    deadline = time.monotonic() + poll_minutes * 60
    while time.monotonic() < deadline:
        r = _tiny(key, model, reasoning)
        r["t"] = time.strftime("%H:%M:%S")
        polls.append(r)
        print(f"  {r['t']} status={r['status']} {r['limit_source'] or ''} {r['latency_s']}s")
        time.sleep(poll_interval)
    ok = sum(1 for p in polls if p["status"] == 200)
    out["C1_poll"] = {"polls": polls, "ok": ok, "total": len(polls)}
    print(f"  -> {ok}/{len(polls)} succeeded")

    print("\n== C2  one request per free tool-capable model")
    models = requests.get(f"{_API}/models", timeout=30).json().get("data", [])
    free = [m["id"] for m in models if m["id"].endswith(":free")
            and "tools" in (m.get("supported_parameters") or [])]
    out["C2_free_models"] = []
    for mid in free:
        r = _tiny(key, mid, "default")
        out["C2_free_models"].append(r)
        print(f"  {mid:55s} status={r['status']} {r['limit_source'] or ''} "
              f"{r['latency_s']}s {r['content']!r}")
    return out


# ---------------------------------------------------------------------------
# Part A — requests per turn through the real ControllerAgent
# ---------------------------------------------------------------------------

def run_part_a(key: str, model: str, reasoning: str, log: CallLog) -> dict[str, Any]:
    sys.path.insert(0, str(_BACKEND / "src"))
    from localist.controller_agent import ControllerAgent
    from localist.conversational_agent import ConversationalAgent
    from localist.memory_manager import MemoryManager
    from localist.ollama_runtime_client import OllamaRuntimeClient
    from localist.warmup import run_cache_warmup
    from localist.wiki_agent import WikiAgent

    client = DiagOpenRouterClient(key, model, reasoning, log)

    # Local embeddings, exactly as Phase 1 will wire them: a local Ollama
    # client used only for embed(). Falls back to keyword-only (BM25).
    embed_fn, embed_model = None, _setting("LOCALIST_EMBEDDING_MODEL_OLLAMA") or None
    if embed_model:
        emb = OllamaRuntimeClient(chat_model="embed-only", embedding_model=embed_model,
                                  base_url=_setting("LOCALIST_OLLAMA_URL") or "http://localhost:11434")
        if emb.health_check().get("embed_model_found"):
            embed_fn = emb.embed
    print(f"Embeddings: {'local Ollama ' + embed_model if embed_fn else 'keyword-only (BM25)'}")

    summary: dict[str, Any] = {"embedding": embed_model if embed_fn else None, "turns": []}
    with tempfile.TemporaryDirectory(prefix="localist-diag-or-") as tmp:
        root = pathlib.Path(tmp)
        (root / "wiki").mkdir()
        mm = MemoryManager(db_path=root / "diag.db", embed_fn=embed_fn,
                           embedding_model_name=embed_model if embed_fn else None)
        wiki_agent = WikiAgent(runtime=client, project_root=root, memory_manager=mm)
        conv_agent = ConversationalAgent(runtime=client, memory_manager=mm, project_root=root)
        controller = ControllerAgent(
            runtime=client, agents=[wiki_agent, conv_agent], memory_manager=mm,
            embed_fn=embed_fn, embedding_model_name=embed_model if embed_fn else None,
            calibrated_thresholds=None, episodic_write_approval=False,
            wiki_dir=root / "wiki",
        )

        log.phase, log.turn = "startup", "cache_warmup"
        before = len(log.rows)
        run_cache_warmup(controller, client, _BACKEND / "templates")
        summary["startup_calls"] = len(log.rows) - before

        conversation_id = f"diag-{uuid.uuid4()}"
        for kind, instruction in _TURNS:
            log.phase, log.turn = "turn", kind
            before, t0 = len(log.rows), time.monotonic()
            context = {"conversation_id": conversation_id, "session_id": conversation_id}
            task = {"task_id": str(uuid.uuid4()), "instruction": instruction, "context": context}
            plan = controller.route_task(instruction, context)
            result = controller.handle_task_with_plan(task, plan, on_token=lambda _c: None)
            calls = log.rows[before:]
            turn = {
                "kind": kind, "route": f"{plan.agent} tools={plan.tools_to_call}",
                "status": result.get("status"), "error": result.get("error"),
                "calls": len([c for c in calls if c.get("method") != "http"]),
                "http_429_retries": len([c for c in calls if c.get("method") == "http"]),
                "labels": [c.get("label") for c in calls],
                "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
                "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
                "wall_s": round(time.monotonic() - t0, 1),
                "answer_head": (result.get("answer") or "")[:120],
            }
            summary["turns"].append(turn)
            print(f"  {kind:16s} calls={turn['calls']} 429s={turn['http_429_retries']} "
                  f"status={turn['status']} wall={turn['wall_s']}s labels={turn['labels']}")
    return summary


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--part", choices=["a", "b", "c", "all"], default="all",
                    help="'all' runs B and A; C (availability) runs only when asked")
    ap.add_argument("--poll-minutes", type=float, default=10.0)
    ap.add_argument("--poll-interval", type=float, default=30.0)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--reasoning", choices=list(_REASONING_VARIANTS), default="off",
                    help="reasoning setting used by Part A and B4-B6 (default: off)")
    args = ap.parse_args()

    key = _setting("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set (env or backend/.env).", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING)

    results: dict[str, Any] = {"model": args.model, "reasoning": args.reasoning,
                               "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log = CallLog()
    if args.part == "c":
        results["part_c"] = run_part_c(key, args.model, args.reasoning,
                                       args.poll_minutes, args.poll_interval)
        out_path = _JSON_OUT.with_name("openrouter_free_tier_availability.json")
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nAvailability results -> {out_path}")
        return 0
    if args.part in ("b", "all"):
        results["part_b"] = run_part_b(key, args.model, args.reasoning)
    if args.part in ("a", "all"):
        print("\n== Part A  requests per turn (real ControllerAgent, temp DB)")
        results["part_a"] = run_part_a(key, args.model, args.reasoning, log)
        turns = results["part_a"]["turns"]
        total = sum(t["calls"] for t in turns)
        results["part_a"]["calls_per_turn_mean"] = round(total / len(turns), 2) if turns else None
        results["part_a"]["calls_per_turn_max"] = max((t["calls"] for t in turns), default=None)
        results["part_a"]["errors"] = log.errors
        print(f"\n  total calls={total} over {len(turns)} turns "
              f"(mean {results['part_a']['calls_per_turn_mean']}, "
              f"max {results['part_a']['calls_per_turn_max']}), "
              f"+{results['part_a']['startup_calls']} at startup")
        if log.rows:
            fields = sorted({k for r in log.rows for k in r})
            with _CSV_OUT.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(log.rows)
            print(f"  per-call rows -> {_CSV_OUT}")

    _JSON_OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nAll results -> {_JSON_OUT}")
    if log.errors:
        print(f"{len(log.errors)} error response(s) recorded (see JSON).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
