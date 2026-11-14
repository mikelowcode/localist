"""
Localist — Ollama hybrid inference (§16.37)
==========================================

Answers on a local Ollama model until a conversation grows past a token cap,
then moves that conversation to an Ollama Cloud model for the rest of its
life. A new conversation starts local again. Both models are served by the
same Ollama daemon, so a swap is just a different "model" field on the next
request — no restart, no controller rebuild, and the embedding model is
never touched (embed() always goes to the local-side client, and the app's
embedding source is MemoryManager.embed_fn anyway, §16.5).

Why the cap measures the whole conversation, not the prompt sent
----------------------------------------------------------------
Local Ollama reports no max_model_len, so context_profile.py budgets local
history at its 300-token floor. Measured on the trimmed prompt, a long
conversation would almost never cross the cap. The caller (ControllerAgent)
passes the conversation's full size — every logged turn, attached session
files, and the new message, len//4 each — to decide_tier().

Phase 0 measurements (2026-11-14, diagnostics/diag_ollama_hybrid_swap.py):
local Gemma 4 12B prefills ~650 tok/s (8K uncached = 13 s to first token);
gemma4:31b-cloud answers in 0.3–1.3 s at any size up to 32K. A mid-
conversation swap to cloud took 0.49 s, back to a warm local model 0.31 s,
to a cold one (after the 1-minute keep-alive) 2.2 s. The embedding model
stayed loaded throughout. Real conversations: p50 1.3K, p90 3.8K tokens;
a 4,000 cap moves ~4% of conversations (~7% of turns) to cloud.

Tier selection is per execution context
---------------------------------------
The active tier lives in a ContextVar, not on the client: concurrent
requests for different conversations share one client. use_tier() sets it
for a block; asyncio.to_thread copies the caller's context into the worker
thread, so every infer()/infer_stream() a turn makes — main answer, working
state, episodic extraction — follows that conversation's tier. Code that
never sets a tier (startup warm-up, news brief, ingestion) runs local.

Cloud failure
-------------
If the cloud model fails before producing a first token (sign-in, quota,
outage), a prompt small enough for the local model (≤ the cap) is retried
on local transparently. A larger one raises CloudTierUnavailableError so the
controller can rebuild the turn with local-sized history and retry it there
(sending a 60K-token cloud prompt to a ~650 tok/s local prefill isn't a
fallback). Either way the conversation stays latched to cloud, so the next
turn tries cloud again. A failure after the first token is not retried.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generator, Iterator

import requests

from .ollama_runtime_client import OllamaRuntimeClient, _is_cloud_model

logger = logging.getLogger(__name__)

TIER_LOCAL = "local"
TIER_CLOUD = "cloud"

DEFAULT_HYBRID_TOKEN_CAP = 4_000            # Phase 0, chosen 2026-11-14
HYBRID_TOKEN_CAP_PRESETS = (2_000, 4_000, 8_000, 16_000)
MIN_HYBRID_TOKEN_CAP     = 1_000
MAX_HYBRID_TOKEN_CAP     = 100_000

# Conversations remembered for the cloud latch. A conversation that falls
# out (or any after a restart) is simply re-measured on its next turn: if
# it's still over the cap it goes straight back to cloud.
_MAX_TRACKED_CONVERSATIONS = 512

_active_tier: contextvars.ContextVar[str] = contextvars.ContextVar(
    "localist_ollama_hybrid_tier", default=TIER_LOCAL,
)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4                    # same as PromptBuilder


def validate_token_cap(cap: int) -> int:
    """Return `cap` as an int, or raise ValueError if out of range."""
    try:
        value = int(cap)
    except (TypeError, ValueError):
        raise ValueError(f"Token cap must be a whole number, got {cap!r}.") from None
    if not MIN_HYBRID_TOKEN_CAP <= value <= MAX_HYBRID_TOKEN_CAP:
        raise ValueError(
            f"Token cap must be between {MIN_HYBRID_TOKEN_CAP:,} and "
            f"{MAX_HYBRID_TOKEN_CAP:,}, got {value:,}."
        )
    return value


def _capabilities(base_url: str, model: str) -> list[str] | None:
    """Ollama's capability list for `model` (POST /api/show), or None when
    it can't be read."""
    try:
        resp = requests.post(f"{base_url.rstrip('/')}/api/show", json={"model": model}, timeout=5)
        resp.raise_for_status()
        caps = resp.json().get("capabilities")
        return caps if isinstance(caps, list) else None
    except Exception:
        return None


def list_chat_models(base_url: str) -> dict:
    """
    Models the hybrid pickers can offer: everything Ollama serves (GET
    /api/tags) that can chat, split into local and cloud ("-cloud") models.
    Embedding-only models (capabilities without "completion", e.g.
    nomic-embed-text) are left out; a model whose capabilities can't be
    read is kept rather than hidden. Never raises.

    Returns {"reachable", "models" (all served), "local", "cloud", "error"}.
    """
    tags = OllamaRuntimeClient(chat_model="", base_url=base_url).health_check()
    result: dict = {
        "reachable": bool(tags.get("reachable")),
        "models":    list(tags.get("models", [])),
        "local":     [],
        "cloud":     [],
        "error":     tags.get("error"),
    }
    for model in result["models"]:
        caps = _capabilities(base_url, model)
        if caps is not None and "completion" not in caps:
            continue
        result["cloud" if _is_cloud_model(model) else "local"].append(model)
    return result


class CloudTierUnavailableError(RuntimeError):
    """The cloud model failed before its first token on a prompt too large
    to hand to the local model as-is. Subclasses RuntimeError, so every
    existing handler of runtime failures still catches it."""


@dataclass(frozen=True)
class TierDecision:
    tier:      str            # TIER_LOCAL or TIER_CLOUD
    model:     str
    size:      int            # conversation size the decision used (tokens)
    cap:       int
    swapped:   bool           # this turn moved the conversation to cloud


class ConversationTierTracker:
    """Which tier each conversation is on. Thread-safe. Once a conversation
    crosses the cap it is latched to cloud; it never moves back."""

    def __init__(self, token_cap: int = DEFAULT_HYBRID_TOKEN_CAP) -> None:
        self._cap    = validate_token_cap(token_cap)
        self._cloud: OrderedDict[str, None] = OrderedDict()
        self._lock   = threading.Lock()

    @property
    def cap(self) -> int:
        return self._cap

    @cap.setter
    def cap(self, value: int) -> None:
        # Read on every decision, so a Settings change applies next turn.
        # Conversations already on cloud stay there.
        self._cap = validate_token_cap(value)

    def decide(self, conversation_id: str | None, size: int) -> tuple[str, bool]:
        """Return (tier, swapped) for a turn of `conversation_id` whose
        conversation size, including the new message, is `size` tokens."""
        if not conversation_id:
            return TIER_LOCAL, False
        with self._lock:
            if conversation_id in self._cloud:
                self._cloud.move_to_end(conversation_id)
                return TIER_CLOUD, False
            if size > self._cap:
                self._cloud[conversation_id] = None
                while len(self._cloud) > _MAX_TRACKED_CONVERSATIONS:
                    self._cloud.popitem(last=False)
                return TIER_CLOUD, True
            return TIER_LOCAL, False

    def tier_of(self, conversation_id: str | None) -> str:
        with self._lock:
            return TIER_CLOUD if conversation_id in self._cloud else TIER_LOCAL

    def forget(self, conversation_id: str) -> None:
        with self._lock:
            self._cloud.pop(conversation_id, None)


class HybridOllamaRuntimeClient:
    """
    BaseRuntimeClient over two OllamaRuntimeClients on one daemon: `local`
    (a locally resident model) and `cloud` (an Ollama Cloud "-cloud" model).
    Every call goes to the tier set by use_tier() in the current context.
    """

    def __init__(
        self,
        local:     OllamaRuntimeClient,
        cloud:     OllamaRuntimeClient,
        token_cap: int = DEFAULT_HYBRID_TOKEN_CAP,
    ) -> None:
        if not local._chat_model or not cloud._chat_model:
            raise ValueError("Hybrid inference needs both a local and a cloud model.")
        if _is_cloud_model(local._chat_model):
            raise ValueError(f"Hybrid local model must not be a cloud model: {local._chat_model!r}")
        if not _is_cloud_model(cloud._chat_model):
            raise ValueError(f"Hybrid cloud model must be a '-cloud' model: {cloud._chat_model!r}")
        self._local   = local
        self._cloud   = cloud
        self.tracker  = ConversationTierTracker(token_cap)
        # Which client served this thread's last call, so last_completion
        # reports the record of the call this thread actually made.
        self._served  = threading.local()
        logger.info(
            "HybridOllamaRuntimeClient — local: %s  cloud: %s  token cap: %d",
            self.local_model, self.cloud_model, self.tracker.cap,
        )

    @classmethod
    def build(
        cls,
        local_model:     str,
        cloud_model:     str,
        token_cap:       int   = DEFAULT_HYBRID_TOKEN_CAP,
        embedding_model: str   = "",
        base_url:        str   = "http://localhost:11434",
        request_timeout: float = 30.0,
        stream_timeout:  float = 60.0,
        keep_alive:      str   = "off",
    ) -> "HybridOllamaRuntimeClient":
        common = dict(
            embedding_model = embedding_model,
            base_url        = base_url,
            request_timeout = request_timeout,
            stream_timeout  = stream_timeout,
            keep_alive      = keep_alive,
        )
        return cls(
            OllamaRuntimeClient(chat_model=local_model, **common),
            OllamaRuntimeClient(chat_model=cloud_model, **common),
            token_cap,
        )

    # ---------------------------------------------------------------------
    # Tier selection
    # ---------------------------------------------------------------------

    @property
    def local_model(self) -> str:
        return self._local._chat_model

    @property
    def cloud_model(self) -> str:
        return self._cloud._chat_model

    @property
    def active_tier(self) -> str:
        return _active_tier.get()

    @property
    def active_model(self) -> str:
        return self.cloud_model if self.active_tier == TIER_CLOUD else self.local_model

    def decide_tier(self, conversation_id: str | None, size: int) -> TierDecision:
        """Decide (and latch) the tier for one turn. Does not set it — wrap
        the turn in use_tier(decision.tier)."""
        tier, swapped = self.tracker.decide(conversation_id, size)
        if swapped:
            logger.info(
                "Hybrid: conversation %s is %d tokens (cap %d) — moving to %s.",
                conversation_id, size, self.tracker.cap, self.cloud_model,
            )
        return TierDecision(
            tier    = tier,
            model   = self.cloud_model if tier == TIER_CLOUD else self.local_model,
            size    = size,
            cap     = self.tracker.cap,
            swapped = swapped,
        )

    @contextlib.contextmanager
    def use_tier(self, tier: str) -> Iterator[None]:
        if tier not in (TIER_LOCAL, TIER_CLOUD):
            raise ValueError(f"Unknown tier {tier!r}")
        token = _active_tier.set(tier)
        try:
            yield
        finally:
            _active_tier.reset(token)

    def switch_tier(self, tier: str) -> None:
        """Change the tier for the rest of the enclosing use_tier() block —
        the controller's answer guard uses this to keep a turn local after
        the cloud model failed. Outside any use_tier() block the change
        lasts for the current context (a request's worker thread)."""
        if tier not in (TIER_LOCAL, TIER_CLOUD):
            raise ValueError(f"Unknown tier {tier!r}")
        _active_tier.set(tier)

    def _client(self) -> OllamaRuntimeClient:
        return self._cloud if self.active_tier == TIER_CLOUD else self._local

    # ---------------------------------------------------------------------
    # Attributes callers read off a runtime client
    # ---------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        # Unlike other clients this follows the active tier, so
        # context_profile.profile_for() gives a cloud-tier conversation the
        # cloud history budget and a local one the local budget.
        return self.active_tier == TIER_LOCAL

    @property
    def prefix_cache_granularity(self) -> str:
        return self._client().prefix_cache_granularity

    @property
    def keep_alive(self) -> str:
        return self._local.keep_alive

    @keep_alive.setter
    def keep_alive(self, choice: str) -> None:
        self._local.keep_alive = choice
        self._cloud.keep_alive = choice

    @property
    def last_completion(self) -> dict | None:
        client = getattr(self._served, "client", None)
        return client.last_completion if client is not None else None

    @property
    def last_fell_back(self) -> bool:
        """True when this thread's last call was meant for cloud but ran
        local because cloud failed (the turn's "answered locally" notice)."""
        return bool(getattr(self._served, "fell_back", False))

    @property
    def last_cloud_unavailable(self) -> bool:
        """True when this thread's last call raised CloudTierUnavailableError.
        Read by ControllerAgent's answer guard, which only sees the agent's
        failed result (the exception type doesn't survive agent.run())."""
        return bool(getattr(self._served, "cloud_unavailable", False))

    def reset_keep_alive_to_default(self) -> bool:
        return self._local.reset_keep_alive_to_default()

    # ---------------------------------------------------------------------
    # BaseRuntimeClient interface
    # ---------------------------------------------------------------------

    def infer(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
        timeout:     float | None = None,
    ) -> str:
        return "".join(self.infer_stream(
            prompt=prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, label=label, timeout=timeout,
        ))

    def infer_stream(
        self,
        prompt:      str,
        system:      str   = "",
        max_tokens:  int   = 1024,
        temperature: float = 0.2,
        label:       str   = "",
        timeout:     float | None = None,
    ) -> Generator[str, None, None]:
        kwargs = dict(
            prompt=prompt, system=system, max_tokens=max_tokens,
            temperature=temperature, label=label, timeout=timeout,
        )
        client = self._client()
        self._served.client            = client
        self._served.fell_back         = False
        self._served.cloud_unavailable = False

        stream = client.infer_stream(**kwargs)
        try:
            first = next(stream)
        except StopIteration:
            return
        except RuntimeError as exc:
            if client is not self._cloud:
                raise
            size = _estimate_tokens(system) + _estimate_tokens(prompt)
            if size > self.tracker.cap:
                self._served.cloud_unavailable = True
                raise CloudTierUnavailableError(
                    f"Cloud model {self.cloud_model} failed before answering "
                    f"({exc}); the {size:,}-token prompt needs rebuilding for "
                    f"the local model."
                ) from exc
            logger.warning(
                "Hybrid: cloud model %s failed (%s) — retrying this %d-token "
                "call (%s) on %s.", self.cloud_model, exc, size, label or "-",
                self.local_model,
            )
            self._served.client    = self._local
            self._served.fell_back = True
            yield from self._local.infer_stream(**kwargs)
            return

        yield first
        yield from stream

    def embed(self, text: str) -> list[float]:
        return self._local.embed(text)

    def health_check(self) -> dict:
        """The local client's health dict (chat_model / chat_model_found
        describe the local model, the tier every new conversation starts
        on) plus cloud_model / cloud_model_found and the hybrid settings."""
        result = self._local.health_check()
        models = result.get("models", [])
        result.update({
            "hybrid":            True,
            "cloud_model":       self.cloud_model,
            "cloud_model_found": self.cloud_model in models,
            "token_cap":         self.tracker.cap,
        })
        return result

    def __repr__(self) -> str:
        return (
            f"HybridOllamaRuntimeClient(local={self.local_model!r}, "
            f"cloud={self.cloud_model!r}, token_cap={self.tracker.cap})"
        )
