"""
LORA — OMLXRuntimeClient
=========================
Concrete implementation of BaseRuntimeClient for the oMLX inference backend.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  OMLXRuntimeClient  →  oMLX HTTP API
                                                          (local only)

Architectural contract
----------------------
- Implements BaseRuntimeClient (base_runtime_client.py).
- No FastAPI imports.  No agent logic.  Pure transport + normalisation.
- Normalises all oMLX-specific response shapes to the same types that
  FoundryRuntimeClient returns, so the Controller and agents are
  completely unaware of which backend is active.
- All network/decode errors are caught and re-raised as RuntimeError —
  the same contract as FoundryRuntimeClient.

oMLX integration notes
-----------------------
oMLX exposes an OpenAI-compatible local HTTP API by default.  The
endpoints and request/response shapes mirror the OpenAI spec, so most
of the transport code here is structurally identical to
FoundryRuntimeClient.  The differences are:

  - Port/URL resolution: oMLX uses a fixed port by default (see
    _DEFAULT_BASE_URL) rather than an ephemeral one, so no CLI
    subprocess is needed to discover it.
  - Model IDs: oMLX model identifiers follow a different convention
    (typically "<family>/<variant>", e.g. "mlx-community/Phi-4-mini").
    These are configured at construction time and must match the model
    IDs returned by GET /v1/models on your oMLX instance.
  - Streaming: oMLX supports SSE streaming with the same
    "data: {...}\ndata: [DONE]" envelope as OpenAI, so _iter_sse_chunks
    from foundry_runtime_client is reusable.  We import it directly to
    avoid duplicating the SSE parsing logic.

TODOs are marked with # TODO(omlx) and indicate where integration
details need to be filled in once the oMLX API surface is confirmed.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

import requests

# Reuse the SSE chunk iterator from FoundryRuntimeClient — the wire
# format is identical (OpenAI-compatible SSE envelope).
from foundry_runtime_client import _iter_sse_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confirmed base URL — GET /v1/models returns HTTP 200 at this address.
# Override via the base_url constructor argument or LORA_OMLX_URL env var.
_DEFAULT_BASE_URL      = "http://localhost:8000"

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_EMBEDDINGS_PATH       = "/v1/embeddings"

# Confirmed chat model ID — matches the "id" field returned by GET /v1/models.
DEFAULT_CHAT_MODEL      = "gemma-4-e4b-it-4bit"

# No embedding model is currently loaded in this oMLX instance.
# Set to a non-empty string when one becomes available; until then embed()
# raises NotImplementedError to give a clear signal rather than a bad request.
DEFAULT_EMBEDDING_MODEL = ""


# ---------------------------------------------------------------------------
# OMLXRuntimeClient
# ---------------------------------------------------------------------------

class OMLXRuntimeClient:
    """
    Concrete BaseRuntimeClient for the oMLX local inference backend.

    Satisfies the BaseRuntimeClient Protocol:
        def infer(self, prompt, system, max_tokens, temperature) -> str
        def embed(self, text) -> list[float]
        def infer_stream(self, prompt, system, max_tokens, temperature)
               -> Generator[str, None, None]

    Parameters
    ----------
    chat_model:
        oMLX model ID for chat completions.
    embedding_model:
        oMLX model ID for embedding generation.
    base_url:
        Base URL of the oMLX HTTP server.  Defaults to _DEFAULT_BASE_URL.
    request_timeout:
        Seconds before a non-streaming request is considered hung.
    stream_timeout:
        Seconds before the first byte of a streaming response must arrive.
    """

    def __init__(
        self,
        chat_model:      str   = DEFAULT_CHAT_MODEL,
        embedding_model: str   = DEFAULT_EMBEDDING_MODEL,
        base_url:        str   = _DEFAULT_BASE_URL,
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
    ) -> None:
        self._chat_model      = chat_model
        self._embedding_model = embedding_model
        self._base_url        = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._stream_timeout  = stream_timeout

        self._chat_endpoint  = self._base_url + _CHAT_COMPLETIONS_PATH
        self._embed_endpoint = self._base_url + _EMBEDDINGS_PATH

        logger.info(
            "OMLXRuntimeClient initialised — chat: %s  embed: %s  base: %s",
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
    ) -> str:
        """
        Request a blocking chat completion from oMLX via SSE accumulation.

        Internally calls infer_stream() and joins all chunks — this ensures
        the streaming and non-streaming paths use exactly the same transport
        code and model parameters, preventing subtle behavioural divergence.

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
        ))
        result = "".join(chunks)
        logger.debug("infer() ← %d chars received.", len(result))
        return result

    def embed(self, text: str) -> list[float]:
        """
        Request a dense embedding vector from oMLX.

        Raises NotImplementedError when no embedding model is configured —
        this oMLX instance is currently inference-only.  Set embedding_model
        at construction time when an embedding model becomes available.

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
                "No embedding model configured for OMLXRuntimeClient. "
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
                f"Cannot reach oMLX at {self._embed_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"oMLX embed request timed out after {self._request_timeout}s."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"oMLX embed returned HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        try:
            data: dict = response.json()
            # Standard OpenAI-compatible embedding response shape:
            # {"data": [{"embedding": [...], "index": 0}], ...}
            # TODO(omlx): If oMLX returns a different shape, update this path.
            vector: list[float] = data["data"][0]["embedding"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected oMLX embed response shape: {exc}\n"
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
    ) -> Generator[str, None, None]:
        """
        Request a streaming chat completion from oMLX.

        Yields individual text chunks from the SSE stream as they arrive.
        The FastAPI streaming endpoint consumes this generator and relays
        chunks to the Svelte UI via Server-Sent Events.

        Yields
        ------
        str
            One text chunk per iteration.

        Raises
        ------
        RuntimeError
            On any network error, non-200 status, or SSE decode failure.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self._chat_model,
            "messages":    messages,
            "stream":      True,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        # TODO(omlx): If oMLX requires additional request headers (e.g. an
        # Authorization token or a custom Accept header), add them here.
        headers = {"Content-Type": "application/json"}

        logger.debug(
            "infer_stream() → %s  max_tokens=%d  temp=%.2f  prompt_chars=%d",
            self._chat_endpoint, max_tokens, temperature, len(prompt),
        )

        try:
            response = requests.post(
                self._chat_endpoint,
                headers = headers,
                data    = json.dumps(payload),
                stream  = True,
                timeout = self._stream_timeout,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach oMLX at {self._chat_endpoint}. "
                f"Is the service running?  Detail: {exc}"
            ) from exc
        except requests.Timeout:
            raise RuntimeError(
                f"oMLX did not respond within {self._stream_timeout}s "
                f"(endpoint: {self._chat_endpoint})."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"oMLX returned HTTP {response.status_code} "
                f"from {self._chat_endpoint}: {response.text[:400]}"
            )

        # _iter_sse_chunks handles the OpenAI-compatible SSE envelope
        # ("data: {...}" lines, "data: [DONE]" sentinel) identically for
        # both Foundry and oMLX.  If oMLX uses a non-standard SSE format,
        # replace this with a custom iterator.
        # TODO(omlx): Verify the SSE envelope format matches OpenAI spec.
        try:
            yield from _iter_sse_chunks(response)
        except Exception as exc:
            raise RuntimeError(
                f"Error reading oMLX SSE stream: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def health_check(self) -> dict:
        """
        Verify that the oMLX service is reachable and the configured
        chat model is listed by GET /v1/models.

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
                self._base_url + "/v1/models",
                timeout=self._request_timeout,
            )
            resp.raise_for_status()
            data   = resp.json()
            models = [m["id"] for m in data.get("data", [])]
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
            logger.warning("OMLXRuntimeClient health_check() failed: %s", exc)

        return result

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OMLXRuntimeClient("
            f"chat_model={self._chat_model!r}, "
            f"embedding_model={self._embedding_model!r}, "
            f"base_url={self._base_url!r})"
        )


# ---------------------------------------------------------------------------
# Protocol conformance check (import-time, debug aid)
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    """Verify OMLXRuntimeClient satisfies BaseRuntimeClient at import time."""
    from base_runtime_client import BaseRuntimeClient
    assert isinstance(OMLXRuntimeClient(), BaseRuntimeClient), (
        "OMLXRuntimeClient does not satisfy the BaseRuntimeClient Protocol."
    )
    logger.debug("OMLXRuntimeClient protocol conformance check passed.")
