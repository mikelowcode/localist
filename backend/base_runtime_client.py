"""
LORA — BaseRuntimeClient Protocol
==================================
The unified interface that every runtime backend must satisfy.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  BaseRuntimeClient (this contract)
                                         ↓
                               FoundryRuntimeClient  |  OMLXRuntimeClient  |  …

Architectural contract
----------------------
- This module defines the Protocol only.  Zero backend-specific logic.
- All concrete runtime clients (foundry_runtime_client.py,
  omlx_runtime_client.py, …) implement this interface.
- The Controller, Planner, Synthesizer, and all sub-agents are typed
  against BaseRuntimeClient — never against a concrete class.
- Adding a new backend requires only: (a) implementing this Protocol,
  (b) registering it in runtime_factory.py.  Nothing else changes.

Why a Protocol rather than an ABC?
-----------------------------------
Using typing.Protocol keeps the design structurally typed: any class that
implements the required methods satisfies the interface without inheriting
from it.  This preserves the existing FoundryRuntimeClient without forcing
a refactor of its class hierarchy, and makes mock/test runtimes trivial to
write.  @runtime_checkable enables isinstance() checks where needed.

Streaming design note
----------------------
infer_stream() is a synchronous generator rather than an async generator
for the same reason infer() is synchronous: the FastAPI layer wraps all
runtime calls in asyncio.to_thread().  Keeping the runtime layer sync
makes it usable both from synchronous agent code and from async FastAPI
handlers without any special handling inside the runtime itself.
If a backend natively supports async streaming, wrap it in a thread-safe
queue inside the concrete client — do not change this contract.
"""

from __future__ import annotations

from typing import Generator, Protocol, runtime_checkable


@runtime_checkable
class BaseRuntimeClient(Protocol):
    """
    Structural interface for all LORA runtime backends.

    Every concrete runtime client must implement all three methods below.
    Type-checkers (mypy, pyright) and isinstance() checks use this Protocol
    to verify conformance without requiring inheritance.

    Methods
    -------
    infer(prompt, system, max_tokens, temperature) → str
        Blocking single-turn completion.  Returns the full response string.
        Used by the Planner, Synthesizer, and all sub-agents.

    embed(text) → list[float]
        Dense embedding for the given text string.  Returns a float vector
        whose dimensionality depends on the backend model.
        Used by ResearchAgent for embedding-based re-ranking.

    infer_stream(prompt, system, max_tokens, temperature) → Generator[str, None, None]
        Synchronous generator that yields text chunks (tokens or small
        word-level pieces) as they are produced by the model.
        Used by the FastAPI streaming endpoint (POST /task/stream) to relay
        tokens to the Svelte UI without buffering the full response.
        Backends that do not natively stream may yield the full string as a
        single chunk — the streaming endpoint degrades gracefully to this.
    """

    def infer(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
    ) -> str:
        """
        Request a blocking chat completion.

        Parameters
        ----------
        prompt:
            The user-turn content.
        system:
            Optional system prompt.
        max_tokens:
            Hard cap on generated tokens.
        temperature:
            Sampling temperature (0.0 = deterministic, 1.0 = creative).

        Returns
        -------
        str
            The model's complete response text.

        Raises
        ------
        RuntimeError
            On any transport, timeout, or decode failure.
            All concrete implementations must normalise backend-specific
            exceptions to RuntimeError so callers get a consistent type.
        """
        ...

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector.  Dimensionality is backend-dependent.

        Raises
        ------
        RuntimeError
            On any transport or decode failure.
        """
        ...

    def infer_stream(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """
        Request a streaming chat completion.

        Yields text chunks as they arrive from the model.  The caller
        accumulates chunks or relays them directly to a client (e.g. SSE).

        Backends without native streaming support must still implement this
        method; they may yield the full infer() result as one chunk:

            def infer_stream(self, prompt, system="", max_tokens=1024,
                             temperature=0.2):
                yield self.infer(prompt, system, max_tokens, temperature)

        Parameters
        ----------
        prompt, system, max_tokens, temperature:
            Same semantics as infer().

        Yields
        ------
        str
            One text chunk per iteration.  Chunk granularity is
            backend-dependent (token, word, or sentence).

        Raises
        ------
        RuntimeError
            On any transport or decode failure.  May be raised after the
            generator has already yielded partial output.
        """
        ...
