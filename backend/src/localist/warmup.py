"""
§3.7c Lever 3 — Startup cache warm-up.

Parses warmup_fixture.md and issues a single best-effort runtime.infer()
call at startup to promote block 0 of the Localist-shaped prompt prefix
to a cache-resident tier. PromptBuilder.build() assembles an identical
slot layout to production traffic so the prefill matches real requests.

This module has no FastAPI or pydantic dependencies and is safe to import
in test environments that lack those packages.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .prompt_builder import PromptBuilder, ToolResult, Turn

if TYPE_CHECKING:
    from .base_runtime_client import BaseRuntimeClient

logger = logging.getLogger(__name__)

_WARMUP_BUILDER = PromptBuilder()

# Matches metadata lines of the form "key: value" where key is a lowercase
# identifier (no spaces). Prevents content lines that happen to contain ": "
# from being misread as metadata.
_META_RE = re.compile(r"^([a-z][a-z_]*): (.*)$")


def parse_warmup_fixture(path: Path) -> tuple[list[ToolResult], list[Turn]]:
    """
    Parse warmup_fixture.md into (tool_results, turns).

    File format — two top-level sections separated by '## TOOL RESULTS' and
    '## WORKING MEMORY' headers. Within each section, entries are separated
    by lines containing only '---'. The first lines of each entry are
    'key: value' metadata (lowercase key, no spaces); a blank line separates
    metadata from body text.

    Raises ValueError when required headers are missing.
    Raises OSError when the file cannot be read.
    Callers should catch both; run_cache_warmup() does so.
    """
    text = path.read_text("utf-8")

    TOOL_HDR = "## TOOL RESULTS"
    WM_HDR   = "## WORKING MEMORY"
    if TOOL_HDR not in text or WM_HDR not in text:
        raise ValueError(
            f"warmup_fixture.md missing required section header(s): "
            f"'{TOOL_HDR}' and/or '{WM_HDR}'"
        )

    _, after_tool    = text.split(TOOL_HDR, 1)
    tool_section, wm_section = after_tool.split(WM_HDR, 1)

    def _parse_entry(raw: str) -> tuple[dict[str, str], str]:
        lines = raw.strip().split("\n")
        meta: dict[str, str] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _META_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
                i += 1
            elif line.strip() == "":
                i += 1
                break
            else:
                break
        body = "\n".join(lines[i:]).strip()
        return meta, body

    tool_results: list[ToolResult] = []
    for raw in re.split(r"\n---\n", tool_section):
        if not raw.strip():
            continue
        meta, body = _parse_entry(raw)
        if "tool" not in meta or not body:
            continue
        tool_results.append(ToolResult(
            tool_name  = meta["tool"],
            parameters = meta.get("params", ""),
            result     = body,
        ))

    turns: list[Turn] = []
    for raw in re.split(r"\n---\n", wm_section):
        if not raw.strip():
            continue
        meta, body = _parse_entry(raw)
        if "role" not in meta or not body:
            continue
        turns.append(Turn(role=meta["role"], content=body))

    return tool_results, turns


def prefix_cache_status(runtime: "BaseRuntimeClient", system_prompt: str) -> dict:
    """
    How much of the shared system message (identity + persona — the prefix
    every main_dispatch and working_state call starts with) the active
    engine can reuse from its prompt cache (§3.7e).

    - Block-based engines (oMLX) expose probe_prefix_cache(); only whole
      blocks are cached, so full_blocks == 0 means none of it ever is.
    - Token-level engines (local Ollama) set prefix_cache_granularity =
      "token": the whole prefix is reusable; its length is estimated
      (Ollama has no tokenize endpoint).
    - Anything else (hosted backends): "unknown", estimate only.

    Never raises.
    """
    estimate = _WARMUP_BUILDER._estimate_tokens(system_prompt)
    info = None
    probe = getattr(runtime, "probe_prefix_cache", None)
    if callable(probe):
        try:
            info = probe(system_prompt)
        except Exception as exc:   # optional diagnostic — never fatal
            logger.debug("prefix_cache_status: probe raised (%s)", exc)
    # Strict type checks: anything but real ints (e.g. a mock runtime's
    # auto-attributes) is treated as "no probe".
    if (isinstance(info, dict)
            and isinstance(info.get("prefix_tokens"), int)
            and isinstance(info.get("block_size"), int)
            and info["block_size"] > 0):
        tokens, block = info["prefix_tokens"], info["block_size"]
        full = tokens // block
        return {
            "granularity":   "block",
            "prefix_tokens": tokens,
            "block_size":    block,
            "full_blocks":   full,
            "cached_tokens": full * block,
            "measured":      True,
        }
    granularity = getattr(runtime, "prefix_cache_granularity", None)
    if granularity not in ("token", "block"):
        granularity = "unknown"
    return {
        "granularity":   granularity,
        "prefix_tokens": estimate,
        "block_size":    None,
        "full_blocks":   None,
        "cached_tokens": estimate if granularity == "token" else None,
        "measured":      False,
    }


def _log_prefix_cache_status(status: dict) -> None:
    if status["granularity"] == "block" and status["full_blocks"] == 0:
        logger.warning(
            "Prompt cache: the shared system message (identity + persona) is "
            "%d tokens, shorter than this model's %d-token cache block, so the "
            "engine can't cache any of it — every call re-reads it. Lengthen "
            "persona.md so the system message passes %d tokens (§3.7e).",
            status["prefix_tokens"], status["block_size"], status["block_size"],
        )
    elif status["granularity"] == "block":
        logger.info(
            "Prompt cache: shared system message %d tokens, %d-token blocks — "
            "%d block(s) / %d tokens cacheable.",
            status["prefix_tokens"], status["block_size"],
            status["full_blocks"], status["cached_tokens"],
        )
    else:
        logger.info(
            "Prompt cache: shared system message ~%d tokens (estimated), "
            "%s-level caching.", status["prefix_tokens"], status["granularity"],
        )


def run_cache_warmup(
    controller:    Any,
    runtime:       "BaseRuntimeClient",
    templates_dir: Path,
) -> dict | None:
    """
    One-shot startup cache warm-up (§3.7c Lever 3).

    Issues a single runtime.infer() call using a prompt built from
    warmup_fixture.md via PromptBuilder.build(). The KV-prefill promotes
    block 0 to a cache-resident tier (disk or hot). Any failure is logged
    as a warning; startup continues regardless. Never raises.

    Returns prefix_cache_status() for the warmed system message (§3.7e) —
    logged, and kept on the controller for GET /settings/prefix-cache — or
    None when the warm-up was skipped or its call failed.
    """
    fixture_path = templates_dir / "warmup_fixture.md"

    # Step 1 — load and parse the fixture.
    try:
        tool_results, working_memory = parse_warmup_fixture(fixture_path)
    except Exception as exc:
        logger.warning(
            "Cache warm-up skipped: fixture load/parse failed (%s: %s).",
            type(exc).__name__, exc,
        )
        return

    if not tool_results and not working_memory:
        logger.warning(
            "Cache warm-up skipped: fixture parsed to empty content — "
            "check warmup_fixture.md."
        )
        return

    # Step 2 — build the prompt. Persona is fetched via the controller's
    # existing lazy-load path; falls back to None on any failure. Same for
    # assistant_name — resolved via the controller's own helper so the warm
    # cache primes the identity string that will actually be used on the
    # first real request (may be stale for one warmup cycle after a live
    # name change; acceptable, see PUT /settings/assistant-name).
    try:
        persona = controller._load_persona()
        assistant_name = controller._current_assistant_name()
        system_prompt, user_prompt = _WARMUP_BUILDER.build(
            instruction      = "Summarise the key themes from the tool results above.",
            current_datetime = datetime.now().astimezone(),
            persona          = persona,
            assistant_name   = assistant_name,
            tool_results     = tool_results or None,
            working_memory   = working_memory or None,
        )
    except Exception as exc:
        logger.warning(
            "Cache warm-up skipped: prompt assembly failed (%s: %s).",
            type(exc).__name__, exc,
        )
        return

    # Step 3 — one inference call; completion content is irrelevant.
    try:
        t0 = time.perf_counter()
        runtime.infer(
            prompt     = user_prompt,
            system     = system_prompt,
            max_tokens = 16,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Cache warm-up complete — block 0 promoted to cache-resident tier "
            "(%.0f ms).", elapsed_ms,
        )
    except Exception as exc:
        logger.warning(
            "Cache warm-up call failed (%s: %s) — startup continues normally.",
            type(exc).__name__, exc,
        )
        return None

    # Step 4 — report how much of the shared prefix the engine can cache.
    # After the call, so a block-based engine has the model loaded to probe.
    status = prefix_cache_status(runtime, system_prompt)
    _log_prefix_cache_status(status)
    return status
