"""
Localist — per-completion telemetry ("why did this reply end?")
==============================================================
Non-answer guard plan, Phase 2 (2026-11-11). Free OpenRouter models returned
junk that finished cleanly (finish_reason "stop", 1 completion token), and
nothing recorded finish_reason or token counts anywhere — the packaged app
keeps no backend log — so explaining it took a live diagnostic. Every
runtime client now records, for the last completion it finished, a small
backend-neutral dict:

    {
      "backend":           "openrouter" | "ollama" | "omlx",
      "model":             str,
      "finish_reason":     str | None,   # "stop", "length", … (Ollama: done_reason)
      "prompt_tokens":     int | None,
      "completion_tokens": int | None,
      "reasoning_tokens":  int | None,   # OpenRouter only, when reported
      "provider":          str | None,   # OpenRouter's upstream provider
    }

and exposes it as `runtime.last_completion` (None until a completion has
finished on the calling thread). ControllerAgent reads it right after the
main dispatch and stores it in the turn's metadata["completion"].

Thread-local on purpose: one runtime client serves the main answer, the
post-answer hooks (working state, implicit extraction) and background jobs
from different worker threads. A plain attribute would let any of those
overwrite the main answer's record before the controller reads it;
thread-local storage means the controller always reads the record of the
call its own thread just made. The streaming generators run on the
consuming thread (ConversationalAgent iterates them inline), so the record
is set on the right thread.

Not part of the BaseRuntimeClient Protocol's required members — callers use
getattr(runtime, "last_completion", None), so test doubles and any future
client without it keep working.
"""

from __future__ import annotations

import threading
from typing import Any


class CompletionRecorder:
    """Holds the calling thread's last completion record."""

    def __init__(self) -> None:
        self._local = threading.local()

    def clear(self) -> None:
        self._local.value = None

    def set(self, record: dict[str, Any]) -> None:
        self._local.value = record

    def get(self) -> dict[str, Any] | None:
        return getattr(self._local, "value", None)


def make_record(
    *,
    backend:           str,
    model:             str,
    finish_reason:     str | None = None,
    prompt_tokens:     int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens:  int | None = None,
    provider:          str | None = None,
) -> dict[str, Any]:
    return {
        "backend":           backend,
        "model":             model,
        "finish_reason":     finish_reason,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens":  reasoning_tokens,
        "provider":          provider,
    }


def record_from_openai_meta(backend: str, model: str, meta: dict[str, Any]) -> dict[str, Any]:
    """
    Build a record from the `meta` dict openai_compat.iter_sse_events()
    fills in (finish_reason, usage, provider) — shared by OpenRouter and
    oMLX, whose streams use the same OpenAI-style envelope.
    """
    usage = meta.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return make_record(
        backend           = backend,
        model             = meta.get("model") or model,
        finish_reason     = meta.get("finish_reason"),
        prompt_tokens     = usage.get("prompt_tokens"),
        completion_tokens = usage.get("completion_tokens"),
        reasoning_tokens  = details.get("reasoning_tokens"),
        provider          = meta.get("provider"),
    )
