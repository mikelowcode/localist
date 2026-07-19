"""
LORA — ContextProfile
=======================
Backend-tier-aware ceilings for how much conversation history flows into
each turn's prompt.

Layer placement
---------------
  ControllerAgent (_execute_plan)  →  ContextProfile  →  MemoryManager.get_context_window()
                                                       →  PromptBuilder.build()
  OllamaRuntimeClient               →  ContextProfile  →  options.num_ctx

Why two profiles, not one global constant
------------------------------------------
The prior single hardcoded ceiling (5 turns / 300 tokens) was sized for the
16GB Apple Silicon local-inference case — RAM held the KV cache, so history
had to stay tiny. A cloud-hosted model (e.g. Gemma-4-31B via Ollama Cloud)
runs on someone else's hardware with a much larger real context window
(~128K); the host Mac's RAM is no longer the constraint, so there is no
reason to truncate history to the same degree. `is_local` (see
base_runtime_client.py) is the single flag every ceiling below keys off —
no per-backend special-casing beyond that one flag.

Values are deliberately chosen, not copied from the model's raw ceiling:

LOCAL_PROFILE — unchanged from the pre-existing hardcoded behavior
  (controller_agent.py's `limit=5, max_tokens=300` call and
  PromptBuilder's `_CEIL_WORKING=300`), just made an explicit, named value
  instead of a magic number scattered across two files.

CLOUD_PROFILE — budgeted against Gemma-4-31B's ~128K window, reserving
  headroom rather than spending the literal ceiling:
    - total_context_tokens=100_000 leaves ~28K under the model's real limit
      for framework/tokenizer overhead and margin against an inexact
      context-length assumption.
    - working_memory_tokens=60_000 is what's left after budgeting the
      other slots' worst case (session files up to 20K, persona/RAG/tool/
      graph/working-state slots combined ~3K, output generation ~4K) plus
      a safety margin, out of the 100K total.
    - working_memory_limit=None removes the turn-count cap entirely — the
      token ceiling above is the only thing trimming history under this
      profile, so a request with many *short* turns doesn't get cut off by
      turn count before it ever approaches the token budget.

Known trade-off, out of scope here: a 60K-token cloud prompt has a real
$/latency cost per request. This module does not add a cost-based
guardrail — RAM-bloat protection only ever applied to the local tier, and
under the cloud tier there's no RAM to protect. A cost/latency guardrail,
if ever wanted, is a separate concern.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextProfile:
    working_memory_tokens: int          # PromptBuilder Slot 6 ceiling / MemoryManager max_tokens
    working_memory_limit:  int | None   # rows fetched by get_context_window() before token trim; None = unbounded
    total_context_tokens:  int          # model input context length (Ollama options.num_ctx)


LOCAL_PROFILE = ContextProfile(
    working_memory_tokens = 300,
    working_memory_limit  = 5,
    total_context_tokens  = 8_000,
)

CLOUD_PROFILE = ContextProfile(
    working_memory_tokens = 60_000,
    working_memory_limit  = None,
    total_context_tokens  = 100_000,
)


def profile_for(is_local: bool) -> ContextProfile:
    """Return the ContextProfile for a runtime client's `is_local` flag."""
    return LOCAL_PROFILE if is_local else CLOUD_PROFILE
