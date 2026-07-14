"""
LORA — OllamaRuntimeClient
===========================
Concrete implementation of BaseRuntimeClient for a local Ollama server.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  OllamaRuntimeClient  →  Ollama HTTP API
                                                            (local only)

Architectural contract
----------------------
- Implements BaseRuntimeClient (base_runtime_client.py).
- No FastAPI imports.  No agent logic.  Pure transport + normalisation.
- Normalises Ollama's response shapes to the same types that
  FoundryRuntimeClient / OMLXRuntimeClient return, so the Controller and
  agents are completely unaware of which backend is active.
- All network/decode errors are caught and re-raised as RuntimeError —
  the same contract as FoundryRuntimeClient / OMLXRuntimeClient.

Ollama integration notes
--------------------------
Ollama exposes its own native HTTP API (not OpenAI-compatible), so the
transport code here is structurally different from Foundry/oMLX in two
places:

  - Model listing: GET /api/tags returns {"models": [{"model": "...", ...}]}
    — a "models" list keyed by "model" (plus a duplicate "name" field) —
    not the OpenAI-style {"data": [{"id": "..."}]} shape.
  - Streaming: POST /api/chat with "stream": true returns NDJSON (one raw
    JSON object per line), not an SSE "data: {...}" envelope.  Each line
    carries the next content delta at message.content, and the final line
    is marked "done": true.  This is NOT the same wire format as Foundry's
    SSE stream, so _iter_sse_chunks is not reusable here — this module
    implements its own line-delimited JSON parser (_iter_ndjson_chunks).

embed() calls Ollama's native POST /api/embed endpoint, which is also
structurally different from Foundry/oMLX:

  - Response shape: {"embeddings": [[...]], ...} — a plural "embeddings"
    key holding a list of vectors (batch-shaped), not the OpenAI-style
    {"data": [{"embedding": [...]}]} shape Foundry/oMLX use. This client
    only ever sends a single string as "input", so it extracts index 0
    of that list.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Iterator

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confirmed base URL — GET /api/tags returns HTTP 200 at this address.
# Override via the base_url constructor argument.
_DEFAULT_BASE_URL = "http://localhost:11434"

_CHAT_PATH  = "/api/chat"
_TAGS_PATH  = "/api/tags"
_EMBED_PATH = "/api/embed"

# No default chat model — Ollama serves many models of wildly different size
# (multi-GB local pulls vs. lightweight cloud models), so silently falling
# back to one is never safe. Empty string means "not configured"; the
# constructor raises ValueError rather than assuming a model.
DEFAULT_CHAT_MODEL = ""


# ---------------------------------------------------------------------------
# NDJSON streaming helper
# ---------------------------------------------------------------------------

def _iter_ndjson_chunks(response: requests.Response) -> Iterator[str]:
    """
    Yield text delta strings from an Ollama NDJSON chat stream.

    Each line from the stream is a standalone JSON object (not an SSE
    envelope), e.g.:
        {"model": "...", "message": {"role": "assistant", "content": "hi"}, "done": false}

    The stream ends on the line where "done" is true (which may also carry
    a trailing empty/absent content) — the loop stops there rather than
    waiting for connection close.

    Malformed lines and empty deltas are silently skipped.
    """
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug("NDJSON: skipping non-JSON line: %s", raw_line[:120])
            continue

        content = data.get("message", {}).get("content", "")
        if content:
            yield content

        if data.get("done"):
            return


# ---------------------------------------------------------------------------
# OllamaRuntimeClient
# ---------------------------------------------------------------------------

class OllamaRuntimeClient:
    """
    Concrete BaseRuntimeClient for a local Ollama inference backend.

    Satisfies the BaseRuntimeClient Protocol:
        def infer(self, prompt, system, max_tokens, temperature) -> str
        def embed(self, text) -> list[float]
        def infer_stream(self, prompt, system, max_tokens, temperature)
               -> Generator[str, None, None]

    Parameters
    ----------
    chat_model:
        Ollama model name for chat completions. Required — there is no
        default. Construction raises ValueError if left empty, since
        silently falling back to some specific local model would risk
        pulling in a multi-gigabyte model the caller never asked for.
    embedding_model:
        Ollama model name for embedding generation. Empty string (default)
        means no embedding model is configured — embed() raises
        NotImplementedError until one is set.
    base_url:
        Base URL of the Ollama HTTP server.  Defaults to _DEFAULT_BASE_URL.
    request_timeout:
        Seconds before a non-streaming request is considered hung.
    stream_timeout:
        Seconds before the first byte of a streaming response must arrive.

    Raises
    ------
    ValueError
        If chat_model is empty — no default chat model is assumed.
    """

    def __init__(
        self,
        chat_model:      str   = DEFAULT_CHAT_MODEL,
        embedding_model: str   = "",
        base_url:        str   = _DEFAULT_BASE_URL,
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
    ) -> None:
        if not chat_model:
            raise ValueError(
                "OllamaRuntimeClient requires an explicit chat_model — no default "
                "is assumed. Set LOCALIST_CHAT_MODEL (or pass chat_model explicitly) "
                "to a model name confirmed present via GET /api/tags."
            )

        self._chat_model      = chat_model
        self._embedding_model = embedding_model
        self._base_url        = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._stream_timeout  = stream_timeout

        self._chat_endpoint  = self._base_url + _CHAT_PATH
        self._tags_endpoint  = self._base_url + _TAGS_PATH
        self._embed_endpoint = self._base_url + _EMBED_PATH

        logger.info(
            "OllamaRuntimeClient initialised — chat: %s  embed: %s  base: %s",
            self._chat_model,
            self._embedding_model,
            self._base_url,
        )

    # -----------------------------------------------------------------------
    # BaseRuntimeClient interface
    # -----------------------------------------------------------------------

    def infer(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
    ) -> str:
        """
        Request a blocking chat completion from Ollama via NDJSON accumulation.

        Internally calls infer_stream() and joins all chunks — this ensures
        the streaming and non-streaming paths use exactly the same transport
        code and model parameters, preventing subtle behavioural divergence.

        Parameters
        ----------
        label:
            Optional caller identifier forwarded to infer_stream() for
            diagnostic correlation. Purely diagnostic — has no effect on
            the request itself.

        Returns
        -------
        str
            The fully accumulated model response.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, or stream decode failure.
        """
        chunks = list(self.infer_stream(
            prompt      = prompt,
            system      = system,
            max_tokens  = max_tokens,
            temperature = temperature,
            label       = label,
        ))
        result = "".join(chunks)
        logger.debug("infer() ← %d chars received.", len(result))
        return result

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector from Ollama via POST /api/embed.

        Raises NotImplementedError when no embedding model is configured —
        this Ollama client is currently chat-only. Set embedding_model at
        construction time when an embedding model becomes available.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            The embedding vector for the input.

        Raises
        ------
        NotImplementedError
            When embedding_model is empty (no model loaded).
        RuntimeError
            On any network error, non-200 status, or unexpected response shape.
        """
        if not self._embedding_model:
            raise NotImplementedError(
                "No embedding model configured for OllamaRuntimeClient. "
                "Set embedding_model to a valid model ID when one is available. "
                "Ensure ResearchAgent is called with use_embeddings=False (the default) "
                "until then."
            )

        payload = {
            "model": self._embedding_model,
            "input": text,
        }

        logger.debug("embed() → %s  input_chars=%d", self._embed_endpoint, len(text))

        try:
            response = requests.post(
                self._embed_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self._request_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._embed_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama embed request timed out after {self._request_timeout}s."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embed returned HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        try:
            data: dict = response.json()
            # Ollama's native embed response shape:
            # {"embeddings": [[...]], ...} — plural key, list of vectors.
            vector: list[float] = data["embeddings"][0]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama embed response shape: {exc}\n"
                f"Raw: {response.text[:400]}"
            ) from exc

        logger.debug("embed() ← vector dim=%d", len(vector))
        return vector

    def infer_stream(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
    ) -> Generator[str, None, None]:
        """
        Request a streaming chat completion from Ollama.

        Yields individual text chunks from the NDJSON stream as they arrive.
        The FastAPI streaming endpoint consumes this generator and relays
        chunks to the Svelte UI via Server-Sent Events.

        Parameters
        ----------
        label:
            Optional caller identifier for diagnostic correlation. Has no
            effect on the request itself.

        Yields
        ------
        str
            One text chunk per iteration.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, or NDJSON decode failure.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":    self._chat_model,
            "messages": messages,
            "stream":   True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        logger.debug(
            "infer_stream() → %s  max_tokens=%d  temp=%.2f  prompt_chars=%d  label=%s",
            self._chat_endpoint, max_tokens, temperature, len(prompt), label,
        )

        try:
            response = requests.post(
                self._chat_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=self._stream_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._chat_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama did not respond within {self._stream_timeout}s "
                f"(endpoint: {self._chat_endpoint})."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code} "
                f"from {self._chat_endpoint}: {response.text[:400]}"
            )

        try:
            yield from _iter_ndjson_chunks(response)
        except Exception as exc:
            raise RuntimeError(
                f"Error reading Ollama NDJSON stream: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def health_check(self) -> dict:
        """
        Verify that the Ollama service is reachable and the configured
        chat model is listed by GET /api/tags.

        Returns a dict with keys:
            reachable (bool), models (list[str]), chat_model_found (bool),
            embed_model_found (bool | None), base_url (str)

        embed_model_found is None when no embedding model is configured
        (not applicable), False when configured but not found, True when
        found.

        Does not raise — all failures are captured in the returned dict.
        """
        result: dict = {
            "reachable":         False,
            "models":            [],
            "chat_model_found":  False,
            # None = not applicable (no embedding model configured)
            "embed_model_found": None if not self._embedding_model else False,
            "base_url":          self._base_url,
        }

        try:
            resp = requests.get(
                self._tags_endpoint,
                timeout=self._request_timeout,
            )
            resp.raise_for_status()
            data   = resp.json()
            models = [m["model"] for m in data.get("models", [])]
            result.update({
                "reachable":        True,
                "models":           models,
                "chat_model_found": self._chat_model in models,
                "embed_model_found": (
                    self._embedding_model in models
                    if self._embedding_model else None
                ),
            })
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("OllamaRuntimeClient health_check() failed: %s", exc)

        return result

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OllamaRuntimeClient("
            f"chat_model={self._chat_model!r}, "
            f"embedding_model={self._embedding_model!r}, "
            f"base_url={self._base_url!r})"
        )


# ---------------------------------------------------------------------------
# Protocol conformance check (import-time, debug aid)
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    """Verify OllamaRuntimeClient satisfies BaseRuntimeClient at import time."""
    from base_runtime_client import BaseRuntimeClient
    # chat_model has no default (see class docstring) — pass a placeholder
    # purely to satisfy the constructor for this conformance check.
    assert isinstance(OllamaRuntimeClient(chat_model="placeholder"), BaseRuntimeClient), (
        "OllamaRuntimeClient does not satisfy the BaseRuntimeClient Protocol."
    )
    logger.debug("OllamaRuntimeClient protocol conformance check passed.")
