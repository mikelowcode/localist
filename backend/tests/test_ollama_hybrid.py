"""
Ollama hybrid inference, Phase 1 (2026-11-14, §16.37): HybridOllamaRuntimeClient
and ConversationTierTracker in isolation. Not wired into the app yet.

  H-1  Construction: local must be non-cloud, cloud must be "-cloud"; cap range.
  H-2  Tracker: under cap → local; over → cloud and latched for the rest of
       the conversation; a new conversation starts local; cap is live.
  H-3  Calls follow use_tier(); default is local; is_local and
       context_profile follow the tier; tiers don't leak across contexts.
  H-4  embed() and keep-alive reset always go to the local client.
  H-5  Cloud failure before the first token: small prompt → retried on local
       (last_fell_back); large prompt → CloudTierUnavailableError. Local
       failures and mid-stream failures are not retried.
  H-6  health_check() adds cloud_model / cloud_model_found.

All inference is mocked; no live Ollama needed.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from localist.base_runtime_client import BaseRuntimeClient
from localist.context_profile import CLOUD_PROFILE, profile_for
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.ollama_hybrid import (
    DEFAULT_HYBRID_TOKEN_CAP, MAX_HYBRID_TOKEN_CAP, MIN_HYBRID_TOKEN_CAP,
    TIER_CLOUD, TIER_LOCAL, CloudTierUnavailableError, ConversationTierTracker,
    HybridOllamaRuntimeClient, validate_token_cap,
)

LOCAL = "gemma4:12b-mlx"
CLOUD = "gemma4:31b-cloud"


def _fake_stream(name: str, fail: str | None = None):
    """infer_stream stand-in: yields "<name>:a", "<name>:b". fail="before"
    raises on the first next(); fail="after" raises after one chunk."""
    calls: list[dict] = []

    def infer_stream(**kwargs):
        calls.append(kwargs)
        if fail == "before":
            raise RuntimeError(f"{name} down")
        yield f"{name}:a"
        if fail == "after":
            raise RuntimeError(f"{name} dropped")
        yield f"{name}:b"

    infer_stream.calls = calls
    return infer_stream


@pytest.fixture
def hybrid():
    h = HybridOllamaRuntimeClient.build(LOCAL, CLOUD, embedding_model="nomic-embed-text:latest")
    h._local.infer_stream = _fake_stream("local")
    h._cloud.infer_stream = _fake_stream("cloud")
    return h


# H-1 -----------------------------------------------------------------------

def test_rejects_cloud_model_as_local():
    with pytest.raises(ValueError, match="must not be a cloud model"):
        HybridOllamaRuntimeClient.build(CLOUD, CLOUD)


def test_rejects_non_cloud_model_as_cloud():
    with pytest.raises(ValueError, match="must be a '-cloud' model"):
        HybridOllamaRuntimeClient.build(LOCAL, LOCAL)


@pytest.mark.parametrize("bad", [MIN_HYBRID_TOKEN_CAP - 1, MAX_HYBRID_TOKEN_CAP + 1, "lots", None])
def test_cap_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        validate_token_cap(bad)


def test_default_cap_is_4000(hybrid):
    assert DEFAULT_HYBRID_TOKEN_CAP == 4_000
    assert hybrid.tracker.cap == 4_000


def test_conforms_to_base_runtime_protocol(hybrid):
    assert isinstance(hybrid, BaseRuntimeClient)


# H-2 -----------------------------------------------------------------------

def test_under_cap_stays_local(hybrid):
    d = hybrid.decide_tier("c1", 4_000)
    assert (d.tier, d.model, d.swapped) == (TIER_LOCAL, LOCAL, False)


def test_over_cap_moves_to_cloud_and_latches(hybrid):
    d = hybrid.decide_tier("c1", 4_001)
    assert (d.tier, d.model, d.swapped) == (TIER_CLOUD, CLOUD, True)
    # Later turns stay on cloud and don't report a new swap, even if the
    # measured size dropped (e.g. a session file was removed).
    d2 = hybrid.decide_tier("c1", 100)
    assert (d2.tier, d2.swapped) == (TIER_CLOUD, False)


def test_new_conversation_starts_local(hybrid):
    hybrid.decide_tier("c1", 9_000)
    assert hybrid.decide_tier("c2", 50).tier == TIER_LOCAL


def test_no_conversation_id_is_always_local(hybrid):
    assert hybrid.decide_tier(None, 50_000).tier == TIER_LOCAL
    assert hybrid.decide_tier("", 50_000).tier == TIER_LOCAL


def test_cap_change_applies_next_decision_and_keeps_latches(hybrid):
    hybrid.decide_tier("c1", 5_000)          # latched at cap 4,000
    hybrid.tracker.cap = 8_000
    assert hybrid.decide_tier("c2", 5_000).tier == TIER_LOCAL
    assert hybrid.decide_tier("c1", 5_000).tier == TIER_CLOUD


def test_tracker_is_bounded_and_forgotten_conversations_are_remeasured(monkeypatch):
    from localist import ollama_hybrid
    monkeypatch.setattr(ollama_hybrid, "_MAX_TRACKED_CONVERSATIONS", 2)
    t = ConversationTierTracker(1_000)
    for cid in ("a", "b", "c"):
        t.decide(cid, 2_000)
    assert t.tier_of("a") == TIER_LOCAL       # evicted
    assert t.decide("a", 2_000) == (TIER_CLOUD, True)
    assert t.decide("a", 10) == (TIER_CLOUD, False)


# H-3 -----------------------------------------------------------------------

def test_default_tier_is_local(hybrid):
    assert hybrid.infer("hi") == "local:alocal:b"
    assert hybrid.is_local is True
    assert hybrid.active_model == LOCAL


def test_use_tier_routes_calls_and_restores(hybrid):
    with hybrid.use_tier(TIER_CLOUD):
        assert list(hybrid.infer_stream("hi")) == ["cloud:a", "cloud:b"]
        assert hybrid.is_local is False
        assert hybrid.active_model == CLOUD
        assert profile_for(hybrid) is CLOUD_PROFILE
    assert hybrid.active_tier == TIER_LOCAL
    assert profile_for(hybrid) is not CLOUD_PROFILE


def test_use_tier_rejects_unknown(hybrid):
    with pytest.raises(ValueError):
        with hybrid.use_tier("orbit"):
            pass


def test_tier_follows_asyncio_to_thread_but_not_other_contexts(hybrid):
    seen: dict[str, str] = {}

    async def run():
        with hybrid.use_tier(TIER_CLOUD):
            seen["to_thread"] = await asyncio.to_thread(hybrid.infer, "x")

    asyncio.run(run())
    t = threading.Thread(target=lambda: seen.__setitem__("plain", hybrid.infer("x")))
    t.start(); t.join()
    assert seen == {"to_thread": "cloud:acloud:b", "plain": "local:alocal:b"}


def test_infer_passes_arguments_through(hybrid):
    with hybrid.use_tier(TIER_CLOUD):
        hybrid.infer("p", system="s", max_tokens=7, temperature=0.5, label="L", timeout=3.0)
    assert hybrid._cloud.infer_stream.calls[-1] == dict(
        prompt="p", system="s", max_tokens=7, temperature=0.5, label="L", timeout=3.0,
    )


# H-4 -----------------------------------------------------------------------

def test_embed_always_uses_local_client(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid._local, "embed", lambda text: [1.0])
    monkeypatch.setattr(hybrid._cloud, "embed", lambda text: pytest.fail("cloud embed"))
    with hybrid.use_tier(TIER_CLOUD):
        assert hybrid.embed("q") == [1.0]


def test_keep_alive_applies_to_both_and_reset_targets_local(hybrid, monkeypatch):
    hybrid.keep_alive = "1m"
    assert hybrid._local.keep_alive == hybrid._cloud.keep_alive == "1m"
    assert hybrid.keep_alive == "1m"
    monkeypatch.setattr(hybrid._local, "reset_keep_alive_to_default", lambda: True)
    monkeypatch.setattr(hybrid._cloud, "reset_keep_alive_to_default", lambda: pytest.fail("cloud"))
    assert hybrid.reset_keep_alive_to_default() is True


def test_last_completion_reports_the_client_that_served(hybrid, monkeypatch):
    monkeypatch.setattr(type(hybrid._cloud), "last_completion",
                        property(lambda self: {"model": self._chat_model}))
    assert hybrid.last_completion is None
    with hybrid.use_tier(TIER_CLOUD):
        hybrid.infer("x")
    assert hybrid.last_completion == {"model": CLOUD}


# H-5 -----------------------------------------------------------------------

def test_cloud_failure_small_prompt_retries_on_local(hybrid):
    hybrid._cloud.infer_stream = _fake_stream("cloud", fail="before")
    with hybrid.use_tier(TIER_CLOUD):
        assert hybrid.infer("short prompt", label="episodic") == "local:alocal:b"
        assert hybrid.last_fell_back is True
    assert hybrid._local.infer_stream.calls[-1]["label"] == "episodic"
    # The next call tries cloud again and clears the flag.
    hybrid._cloud.infer_stream = _fake_stream("cloud")
    with hybrid.use_tier(TIER_CLOUD):
        assert hybrid.infer("x") == "cloud:acloud:b"
        assert hybrid.last_fell_back is False


def test_cloud_failure_large_prompt_raises_for_controller_rebuild(hybrid):
    hybrid._cloud.infer_stream = _fake_stream("cloud", fail="before")
    big = "w" * 4 * (hybrid.tracker.cap + 1)
    with hybrid.use_tier(TIER_CLOUD):
        with pytest.raises(CloudTierUnavailableError) as info:
            hybrid.infer(big)
    assert isinstance(info.value, RuntimeError)
    assert hybrid._local.infer_stream.calls == []


def test_local_failure_is_not_retried(hybrid):
    hybrid._local.infer_stream = _fake_stream("local", fail="before")
    with pytest.raises(RuntimeError, match="local down"):
        hybrid.infer("x")
    assert hybrid._cloud.infer_stream.calls == []


def test_cloud_failure_mid_stream_is_not_retried(hybrid):
    hybrid._cloud.infer_stream = _fake_stream("cloud", fail="after")
    got = []
    with hybrid.use_tier(TIER_CLOUD):
        with pytest.raises(RuntimeError, match="cloud dropped"):
            for chunk in hybrid.infer_stream("x"):
                got.append(chunk)
    assert got == ["cloud:a"]
    assert hybrid._local.infer_stream.calls == []


# H-6 -----------------------------------------------------------------------

def test_health_check_adds_cloud_fields(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid._local, "health_check", lambda: {
        "reachable": True, "models": [LOCAL, CLOUD], "chat_model": LOCAL,
        "chat_model_found": True,
    })
    h = hybrid.health_check()
    assert h["chat_model"] == LOCAL and h["chat_model_found"] is True
    assert h["hybrid"] is True
    assert (h["cloud_model"], h["cloud_model_found"], h["token_cap"]) == (CLOUD, True, 4_000)


def test_health_check_reports_missing_cloud_model(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid._local, "health_check", lambda: {
        "reachable": True, "models": [LOCAL], "chat_model": LOCAL, "chat_model_found": True,
    })
    assert hybrid.health_check()["cloud_model_found"] is False


# H-7 -----------------------------------------------------------------------

def test_list_chat_models_splits_and_drops_embedding_models(monkeypatch):
    from localist import ollama_hybrid
    monkeypatch.setattr(OllamaRuntimeClient, "health_check", lambda self: {
        "reachable": True, "models": [LOCAL, CLOUD, "nomic-embed-text:latest", "mystery:1b"],
    })
    caps = {LOCAL: ["completion"], CLOUD: ["completion", "tools"], "nomic-embed-text:latest": ["embedding"]}
    monkeypatch.setattr(ollama_hybrid, "_capabilities", lambda base, m: caps.get(m))
    out = ollama_hybrid.list_chat_models("http://localhost:11434")
    assert out["local"] == [LOCAL, "mystery:1b"]       # unreadable capabilities: kept
    assert out["cloud"] == [CLOUD]
    assert out["reachable"] is True
