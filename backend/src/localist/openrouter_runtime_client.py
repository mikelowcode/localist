"""
Localist — OpenRouterRuntimeClient
==================================
Concrete BaseRuntimeClient for OpenRouter (https://openrouter.ai), a hosted
gateway to many models, including a free tier. Added after Ollama Cloud's
free plan ended (2026-11-09). Design and live findings:
docs/architecture/16-runtime-backend-layer.md §16.33,
diagnostics/reports/openrouter_free_tier_2026-11-10.md.

Layer placement
---------------
  ControllerAgent / Sub-agents  →  OpenRouterRuntimeClient  →  OpenRouter API
                                                                (remote)

Architectural contract
----------------------
- Implements BaseRuntimeClient. No FastAPI imports, no agent logic.
- Chat only. embed() raises: a chat-only backend's embeddings come from
  local Ollama (runtime_factory.embedding_host_for(), §16.32), so nothing
  is ever embedded remotely.
- is_local is False (the cloud context profile applies).
- Every failure is raised as OpenRouterError, a RuntimeError subclass, so
  callers keep the one exception type the contract promises. Its `kind`
  tells the failures apart for the UI (busy upstream vs. daily free limit
  vs. bad key ...).

Wire format
-----------
OpenAI-compatible /chat/completions over SSE; parsing and message building
are shared with OMLXRuntimeClient through openai_compat.py, which decodes
the stream as UTF-8 explicitly (OpenRouter's event stream declares no
charset, and requests' fallback mangled non-ASCII text).

Behaviour measured live in Phase 0 (2026-11-10)
-----------------------------------------------
- Reasoning: `reasoning: {"enabled": false}` stops hidden reasoning
  (reasoning_tokens 0, half the latency). `{"exclude": true}` only hides
  it. Sent on every request, the counterpart of Ollama's `think: false`
  (§4.6.2). With reasoning on and a small max_tokens, most free models
  spent the whole budget reasoning and returned empty content.
- 429s come in two kinds, told apart by error.metadata.limit_source:
  "upstream_provider_shared_pool" means the provider's shared free pool is
  saturated; nothing about this account. It was persistent for minutes
  (22/22 on Gemma 4 31B free), so it is retried once only. Account limits
  (free-models-per-min / -per-day) are reported as such.
- 429s did not consume free_model_daily_requests.
- GET /key reports free_model_daily_requests {used, limit, remaining}
  and is_free_tier, for free; health_check() passes it through.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator

import requests

from .openai_compat import build_messages, content_delta, iter_sse_events

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# The default free model (user decision 2026-11-10): Gemma 4 31B free was
# rate-limited upstream on every Phase 0 request; Nemotron 3 Ultra free
# answered every one. Gemma stays selectable.
DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Sent as X-Title for OpenRouter's app attribution. Nothing user-identifying
# goes in a header.
_APP_TITLE = "Localist"

# One retry for an upstream-pool 429 (the pool can clear between requests),
# after this many seconds unless Retry-After says otherwise, capped at
# _MAX_RETRY_WAIT_S so a stuck pool doesn't hold a turn for long.
_UPSTREAM_RETRY_WAIT_S = 2.0
_MAX_RETRY_WAIT_S      = 10.0

# GET /models is ~hundreds of KB; the picker list is cached this long.
_MODEL_LIST_TTL_S = 600.0

_UPSTREAM_POOL = "upstream_provider_shared_pool"


class OpenRouterError(RuntimeError):
    """
    A failed OpenRouter request. `kind` is one of:

      "no_key"        no API key configured
      "auth"          401/403: key missing, invalid, disabled
      "credits"       402: out of credits (also blocks free models)
      "daily_limit"   429 free-models-per-day (50, or 1,000 after a
                      one-time $10 credit purchase)
      "minute_limit"  429 free-models-per-min (20)
      "upstream_busy" 429 from the provider's shared free pool
      "rate_limited"  any other 429
      "no_endpoint"   404: unknown model, or no provider allowed for it
      "transport"     connection failure or timeout
      "stream"        an error reported mid-stream
      "other"         anything else

    `message` is meant for people; str(exc) carries the detail.
    """

    def __init__(self, kind: str, message: str, status: int | None = None,
                 detail: str = "") -> None:
        super().__init__(f"{message} {detail}".strip())
        self.kind    = kind
        self.message = message
        self.status  = status
        self.detail  = detail


def _error_body(response: requests.Response) -> tuple[str, dict[str, Any]]:
    """(error.message, error.metadata) from an OpenRouter error body."""
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        return response.text[:400], {}
    return str(error.get("message") or response.text[:400]), dict(error.get("metadata") or {})


def classify_error(status: int, message: str, metadata: dict[str, Any]) -> tuple[str, str]:
    """(kind, human-readable message) for an OpenRouter error response."""
    lowered = message.lower()
    if status in (401, 403):
        return "auth", "OpenRouter rejected the API key. Check it in Settings."
    if status == 402:
        return "credits", "Your OpenRouter account is out of credits."
    if status == 404:
        return "no_endpoint", (
            "OpenRouter has no provider for this model right now, or none your privacy "
            "settings allow."
        )
    if status == 429:
        if metadata.get("limit_source") == _UPSTREAM_POOL or "rate-limited upstream" in lowered:
            return "upstream_busy", (
                "This free model is busy at its provider right now. Try again shortly, or "
                "pick another model in Settings."
            )
        if "per-day" in lowered or "per day" in lowered:
            return "daily_limit", (
                "You've used today's free OpenRouter requests. Buying $10 of credits once "
                "raises the limit from 50 to 1,000 a day."
            )
        if "per-min" in lowered or "per min" in lowered:
            return "minute_limit", "Too many requests this minute. Try again in a moment."
        return "rate_limited", "OpenRouter is rate-limiting requests. Try again shortly."
    return "other", f"OpenRouter returned HTTP {status}."


class OpenRouterRuntimeClient:
    """
    Concrete BaseRuntimeClient for OpenRouter.

    Parameters
    ----------
    chat_model:
        OpenRouter model ID, e.g. "nvidia/nemotron-3-ultra-550b-a55b:free".
    api_key:
        OpenRouter API key (OPENROUTER_API_KEY). Empty means every call
        fails fast with kind "no_key", and health_check() reports
        unreachable.
    embedding_model:
        Accepted for factory symmetry and ignored: this backend never embeds.
    base_url, request_timeout, stream_timeout:
        As for the other clients.
    """

    def __init__(
        self,
        chat_model:      str   = DEFAULT_CHAT_MODEL,
        api_key:         str   = "",
        embedding_model: str   = "",
        base_url:        str   = _DEFAULT_BASE_URL,
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
    ) -> None:
        self._chat_model      = chat_model
        self._api_key         = api_key.strip()
        self._base_url        = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._stream_timeout  = stream_timeout
        self._chat_endpoint   = self._base_url + "/chat/completions"

        self.is_local = False
        # No startup/switch cache warm-up (warmup.py): it would spend one of
        # the free tier's daily requests (50/day) on every start and switch,
        # and the free endpoints measured in Phase 0 report no implicit
        # prompt caching for it to prime. Read by main._build_controller().
        self.wants_cache_warmup = False
        # The most recent failure's kind/message and when, for /health and
        # the chat view's banner; cleared by the next successful request.
        self.last_error: dict[str, Any] | None = None
        # `usage` from the last completed stream (tokens, cost).
        self.last_usage: dict[str, Any] | None = None

        self._model_list: list[dict[str, Any]] | None = None
        self._model_list_at = 0.0

        logger.info(
            "OpenRouterRuntimeClient initialised — chat: %s  base: %s  key: %s",
            self._chat_model, self._base_url, "set" if self._api_key else "MISSING",
        )

    def __repr__(self) -> str:
        return f"OpenRouterRuntimeClient(chat_model={self._chat_model!r})"

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
        timeout:     float | None = None,
    ) -> str:
        """Blocking completion: infer_stream() joined, one code path for both."""
        return "".join(self.infer_stream(
            prompt, system, max_tokens, temperature, label=label, timeout=timeout,
        ))

    def embed(self, text: str) -> list[float]:
        raise RuntimeError(
            "OpenRouterRuntimeClient is chat-only: embeddings come from local Ollama "
            "(runtime_factory.embedding_host_for(), §16.32)."
        )

    def infer_stream(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
        timeout:     float | None = None,
    ) -> Generator[str, None, None]:
        if not self._api_key:
            raise self._record(OpenRouterError(
                "no_key", "No OpenRouter API key is set. Add one in Settings.",
            ))

        payload = {
            "model":       self._chat_model,
            "messages":    build_messages(prompt, system),
            "stream":      True,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "reasoning":   {"enabled": False},
        }
        effective_timeout = timeout if timeout is not None else self._stream_timeout
        logger.debug(
            "infer_stream() → %s  model=%s  max_tokens=%d  prompt_chars=%d  label=%s",
            self._chat_endpoint, self._chat_model, max_tokens, len(prompt), label,
        )

        response = self._post(payload, effective_timeout)
        if response.status_code == 429:
            message, metadata = _error_body(response)
            kind, _ = classify_error(429, message, metadata)
            if kind == "upstream_busy":
                wait = _retry_wait(response)
                logger.warning(
                    "OpenRouter: %s busy upstream — retrying once in %.1fs — label=%s",
                    self._chat_model, wait, label,
                )
                response.close()
                time.sleep(wait)
                response = self._post(payload, effective_timeout)

        if response.status_code != 200:
            message, metadata = _error_body(response)
            kind, human = classify_error(response.status_code, message, metadata)
            raise self._record(OpenRouterError(
                kind, human, status=response.status_code,
                detail=f"(model {self._chat_model}: {metadata.get('raw') or message})",
            ))

        try:
            for event in iter_sse_events(response):
                if event.get("error"):
                    error = event["error"]
                    raise OpenRouterError(
                        "stream", "OpenRouter reported an error mid-response.",
                        detail=str(error.get("message") or error)[:400],
                    )
                if event.get("usage"):
                    self.last_usage = event["usage"]
                text = content_delta(event)
                if text:
                    yield text
        except OpenRouterError as exc:
            raise self._record(exc)
        except requests.RequestException as exc:
            raise self._record(OpenRouterError(
                "transport", "The connection to OpenRouter dropped mid-response.", detail=str(exc),
            )) from exc
        finally:
            response.close()
        self.last_error = None

    def health_check(self) -> dict:
        """
        Reachability = OpenRouter answers and accepts the key (GET /key,
        which costs no quota). chat_model_found = the model has at least one
        endpoint (GET /models/{id}/endpoints). Never raises.

        Extra keys: `key_info` (free-tier status, free requests left today,
        spend limit) and `last_error` (the most recent failed request's
        kind/message, or None).
        """
        result: dict[str, Any] = {
            "reachable":         False,
            "models":            [],
            "chat_model_found":  False,
            "embed_model_found": None,
            "base_url":          self._base_url,
            "chat_model":        self._chat_model,
            "key_info":          None,
            "last_error":        self.last_error,
        }
        if not self._api_key:
            result["error"] = "No OpenRouter API key is set."
            return result
        try:
            key_resp = requests.get(
                self._base_url + "/key", headers=self._headers(), timeout=self._request_timeout,
            )
            if key_resp.status_code in (401, 403):
                result["error"] = "OpenRouter rejected the API key."
                return result
            key_resp.raise_for_status()
            result["reachable"] = True
            result["key_info"] = _key_info((key_resp.json() or {}).get("data") or {})

            ep_resp = requests.get(
                f"{self._base_url}/models/{self._chat_model}/endpoints",
                headers=self._headers(), timeout=self._request_timeout,
            )
            endpoints = (
                ((ep_resp.json() or {}).get("data") or {}).get("endpoints") or []
                if ep_resp.status_code == 200 else []
            )
            result["chat_model_found"] = bool(endpoints)
            result["models"] = [self._chat_model] if endpoints else []
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("OpenRouterRuntimeClient health_check() failed: %s", exc)
        return result

    # -----------------------------------------------------------------------
    # OpenRouter-specific: the model catalog, for the Settings picker
    # -----------------------------------------------------------------------

    def list_models(self, free_only: bool = False) -> list[dict[str, Any]]:
        """
        Chat models that accept tools, as {id, name, free, context_length},
        free ones first then by name. GET /models is public and large, so it
        is cached for _MODEL_LIST_TTL_S. Raises OpenRouterError on failure.
        Localist doesn't send native tool calls today; "accepts tools" is
        used as the bar for a capable chat model.
        """
        now = time.monotonic()
        if self._model_list is None or now - self._model_list_at > _MODEL_LIST_TTL_S:
            try:
                resp = requests.get(self._base_url + "/models", timeout=self._request_timeout)
                resp.raise_for_status()
                entries = (resp.json() or {}).get("data") or []
            except Exception as exc:
                raise OpenRouterError("transport", "Couldn't load OpenRouter's model list.",
                                      detail=str(exc)) from exc
            models = []
            for m in entries:
                params = m.get("supported_parameters") or []
                outputs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
                if "tools" not in params or "text" not in outputs:
                    continue
                # OpenRouter's own routers (openrouter/free, openrouter/auto)
                # pick a different model per request, which defeats
                # Localist's prompt caching and makes behaviour turn-to-turn
                # unpredictable — not offered.
                if str(m.get("id", "")).startswith("openrouter/"):
                    continue
                pricing = m.get("pricing") or {}
                models.append({
                    "id":             m.get("id", ""),
                    "name":           m.get("name") or m.get("id", ""),
                    "free":           str(m.get("id", "")).endswith(":free")
                                      or (pricing.get("prompt") == "0" and pricing.get("completion") == "0"),
                    "context_length": m.get("context_length"),
                })
            models.sort(key=lambda m: (not m["free"], m["name"].lower()))
            self._model_list, self._model_list_at = models, now
        return [m for m in self._model_list if m["free"]] if free_only else list(self._model_list)

    def set_api_key(self, api_key: str) -> None:
        """
        Replace the key in place — PUT /settings/api-keys saving a new
        OPENROUTER_API_KEY while this backend is active, so it takes effect
        on the next request without rebuilding the controller.
        """
        self._api_key = api_key.strip()
        self.last_error = None

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
            "X-Title":       _APP_TITLE,
        }

    def _post(self, payload: dict[str, Any], timeout: float) -> requests.Response:
        try:
            return requests.post(
                self._chat_endpoint, headers=self._headers(), data=json.dumps(payload),
                stream=True, timeout=timeout,
            )
        except requests.ConnectionError as exc:
            raise self._record(OpenRouterError(
                "transport", "Can't reach OpenRouter. Check the internet connection.",
                detail=str(exc),
            )) from exc
        except requests.Timeout as exc:
            raise self._record(OpenRouterError(
                "transport", f"OpenRouter didn't respond within {timeout:.0f}s.",
            )) from exc

    def _record(self, exc: OpenRouterError) -> OpenRouterError:
        self.last_error = {"kind": exc.kind, "message": exc.message, "at": time.time()}
        logger.warning("OpenRouter request failed (%s): %s", exc.kind, exc)
        return exc


def validate_api_key(api_key: str, base_url: str = _DEFAULT_BASE_URL,
                     timeout: float = 10.0) -> dict[str, Any]:
    """
    Check a key before it's saved (GET /key, which costs no quota). Returns
    {"status", "message", "key_info"} where status is:
      "valid"       OpenRouter accepted it
      "rejected"    401/403: refuse the save
      "unverified"  OpenRouter unreachable or erroring: save anyway, warn
    Mirrors github_watch.validate_token()'s contract.
    """
    try:
        resp = requests.get(
            base_url.rstrip("/") + "/key",
            headers={"Authorization": f"Bearer {api_key.strip()}"}, timeout=timeout,
        )
    except requests.RequestException as exc:
        return {"status": "unverified", "key_info": None,
                "message": f"Couldn't reach OpenRouter to check the key ({exc.__class__.__name__})."}
    if resp.status_code in (401, 403):
        return {"status": "rejected", "key_info": None,
                "message": "OpenRouter rejected this API key. Check what you pasted."}
    if resp.status_code != 200:
        return {"status": "unverified", "key_info": None,
                "message": f"OpenRouter returned HTTP {resp.status_code} when checking the key."}
    try:
        info = _key_info((resp.json() or {}).get("data") or {})
    except ValueError:
        info = None
    return {"status": "valid", "key_info": info, "message": "Connected to OpenRouter."}


def _retry_wait(response: requests.Response) -> float:
    try:
        wait = float(response.headers.get("Retry-After") or _UPSTREAM_RETRY_WAIT_S)
    except ValueError:
        wait = _UPSTREAM_RETRY_WAIT_S
    return max(0.0, min(wait, _MAX_RETRY_WAIT_S))


def _key_info(data: dict[str, Any]) -> dict[str, Any]:
    """The parts of GET /key worth showing; nothing identifying (no label, user or workspace ID)."""
    free = data.get("free_model_daily_requests") or {}
    return {
        "is_free_tier":            data.get("is_free_tier"),
        "free_requests_used":      free.get("used"),
        "free_requests_limit":     free.get("limit"),
        "free_requests_remaining": free.get("remaining"),
        "limit_usd":               data.get("limit"),
        "limit_remaining_usd":     data.get("limit_remaining"),
        "limit_reset":             data.get("limit_reset"),
        "usage_usd":               data.get("usage"),
    }


if __name__ == "__main__":
    from .base_runtime_client import BaseRuntimeClient
    assert isinstance(OpenRouterRuntimeClient(), BaseRuntimeClient), (
        "OpenRouterRuntimeClient does not satisfy BaseRuntimeClient."
    )
    logger.debug("OpenRouterRuntimeClient protocol conformance check passed.")
