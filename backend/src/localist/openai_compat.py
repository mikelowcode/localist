"""
Localist — OpenAI-compatible wire helpers
=========================================
Shared by the runtime clients that speak the OpenAI /chat/completions
format: OMLXRuntimeClient (local) and OpenRouterRuntimeClient (hosted).
Ollama speaks its own NDJSON format and doesn't use this module.

No backend-specific logic lives here: SSE parsing and message building
only. Each client keeps its own transport, headers and error handling.

UTF-8 decoding (docs/architecture/16-runtime-backend-layer.md §16.33)
--------------------------------------------------------------------
SSE lines are read as bytes and decoded as UTF-8 here, never through
`response.iter_lines(decode_unicode=True)`. That flag decodes with the
response's declared charset, and for a `text/event-stream` with no
charset `requests` falls back to ISO-8859-1. OpenRouter sends no charset,
so non-ASCII text arrived mangled (an en dash became "â\\x80\\x93", found
by diagnostics/diag_openrouter_free_tier.py, 2026-11-10). Lines that
already arrive as `str` (test fakes) pass through unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Iterator, Mapping, Protocol

logger = logging.getLogger(__name__)


class _TurnLike(Protocol):
    role: str
    content: str


# conversation_log stores role="agent" for completed agent turns; the
# OpenAI vocabulary is system/user/assistant/tool.
TURN_ROLE_TO_MESSAGE_ROLE: Mapping[str, str] = {"agent": "assistant"}


def build_messages(
    prompt: str,
    system: str = "",
    turns:  Iterable[_TurnLike] | None = None,
) -> list[dict[str, str]]:
    """
    system (if any) → one message per working-memory turn (oldest first) →
    the current prompt as the final user message.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in turns or []:
        role = TURN_ROLE_TO_MESSAGE_ROLE.get(turn.role, turn.role)
        messages.append({"role": role, "content": turn.content})
    messages.append({"role": "user", "content": prompt})
    return messages


def iter_sse_lines(response: Any) -> Iterator[str]:
    """Non-empty SSE lines from `response`, decoded as UTF-8."""
    for raw in response.iter_lines():
        if not raw:
            continue
        yield raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def iter_sse_events(response: Any) -> Iterator[dict[str, Any]]:
    """
    Parsed JSON objects from an OpenAI-style SSE stream, stopping at
    `data: [DONE]`. Comment lines (": OPENROUTER PROCESSING" keep-alives),
    non-data lines and malformed JSON are skipped.
    """
    for line in iter_sse_lines(response):
        if line.startswith(":") or not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            return
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("SSE: skipping non-JSON line: %s", line[:120])
            continue
        if isinstance(event, dict):
            yield event


def content_delta(event: Mapping[str, Any]) -> str:
    """The text content delta of one stream event ("" when there is none)."""
    choices = event.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("delta") or {}).get("content") or ""


def update_stream_meta(meta: dict[str, Any], event: Mapping[str, Any]) -> None:
    """
    Fold one stream event's end-of-reply facts into `meta`: finish_reason
    (and OpenRouter's native_finish_reason), usage (usually only on the
    final chunk), provider and model. Feeds completion_record (§16.35).
    """
    if event.get("usage"):
        meta["usage"] = event["usage"]
    if event.get("provider"):
        meta["provider"] = event["provider"]
    if event.get("model"):
        meta["model"] = event["model"]
    for choice in event.get("choices") or []:
        if choice.get("finish_reason"):
            meta["finish_reason"] = choice["finish_reason"]
        if choice.get("native_finish_reason"):
            meta["native_finish_reason"] = choice["native_finish_reason"]


def iter_content_deltas(response: Any, meta: dict[str, Any] | None = None) -> Iterator[str]:
    """
    Non-empty text content deltas from an OpenAI-style SSE stream. When
    `meta` is given, it is filled in place via update_stream_meta().
    """
    for event in iter_sse_events(response):
        if meta is not None:
            update_stream_meta(meta, event)
        text = content_delta(event)
        if text:
            yield text
