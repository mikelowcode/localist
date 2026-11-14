"""
Ollama hybrid inference, Phase 3 (2026-11-14, §16.37): GET/PUT
/settings/ollama-hybrid and the hybrid fields on GET /health.

  S-1  GET returns the saved settings, cap presets/range, and whether the
       hybrid client is live.
  S-2  PUT turning hybrid on (Ollama active) validates, persists to .env and
       rebuilds the controller onto a HybridOllamaRuntimeClient; turning it
       off rebuilds back to the plain client.
  S-3  A cap-only change updates the live client in place (no rebuild,
       cloud latches kept).
  S-4  Validation: model kinds, both models to turn on, served by Ollama,
       Ollama reachable, cap range — nothing is persisted on a rejection.
  S-5  Another backend active: saved, not applied live, no rebuild.
  S-6  GET /health reports hybrid / cloud model / cap.

Ollama, the controller build and .env location are all stubbed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.ollama_hybrid import HybridOllamaRuntimeClient
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.runtime_factory import create_runtime

LOCAL = "gemma4:12b-mlx"
CLOUD = "gemma4:31b-cloud"
SERVED = [LOCAL, CLOUD, "nomic-embed-text:latest"]


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    prev = (main._state.settings, main._state.runtime, main._state.controller,
            main._state.memory_manager)
    main._state.settings = main.Settings(runtime_backend="ollama", chat_model_ollama=LOCAL)
    main._state.runtime = OllamaRuntimeClient(chat_model=LOCAL)
    main._state.controller = MagicMock()
    main._state.memory_manager = MagicMock()

    served = {"reachable": True, "models": list(SERVED), "local": [LOCAL], "cloud": [CLOUD], "error": None}
    monkeypatch.setattr(main, "_ollama_chat_models", lambda settings: served)

    builds: list[object] = []

    def fake_create_and_check(settings, backend):
        rt = create_runtime(backend, chat_model=main._resolve_chat_model(settings, backend),
                            **main._ollama_hybrid_kwargs(settings))
        builds.append(rt)
        return rt, {"reachable": True, "base_url": "http://localhost:11434"}

    monkeypatch.setattr(main, "_create_and_check_backend", fake_create_and_check)
    monkeypatch.setattr(main, "_build_controller",
                        lambda *a, **k: (MagicMock(), MagicMock(), MagicMock()))
    yield TestClient(main.app), tmp_path, served, builds
    (main._state.settings, main._state.runtime, main._state.controller,
     main._state.memory_manager) = prev


def _env(tmp_path) -> str:
    p = tmp_path / ".env"
    return p.read_text() if p.exists() else ""


# S-1 -----------------------------------------------------------------------

def test_get_defaults(api):
    client, *_ = api
    body = client.get("/settings/ollama-hybrid").json()
    assert body["enabled"] is False
    assert body["token_cap"] == 4_000
    assert body["cap_presets"] == [2_000, 4_000, 8_000, 16_000]
    assert (body["cap_min"], body["cap_max"]) == (1_000, 100_000)
    assert body["active"] is False
    assert (body["local_models"], body["cloud_models"], body["models_error"]) == ([LOCAL], [CLOUD], None)


def test_get_reports_unreachable_ollama(api):
    client, _, served, _ = api
    served.update(reachable=False, models=[], local=[], cloud=[])
    body = client.get("/settings/ollama-hybrid").json()
    assert body["local_models"] == [] and "isn't reachable" in body["models_error"]


# S-2 -----------------------------------------------------------------------

def test_enable_persists_and_rebuilds_onto_hybrid(api):
    client, tmp_path, _, builds = api
    r = client.put("/settings/ollama-hybrid", json={
        "enabled": True, "local_model": LOCAL, "cloud_model": CLOUD, "token_cap": 4_000,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["enabled"], body["active"], body["applied_live"]) == (True, True, True)
    assert isinstance(main._state.runtime, HybridOllamaRuntimeClient)
    assert main._state.runtime.cloud_model == CLOUD
    env = _env(tmp_path)
    for line in ("LOCALIST_OLLAMA_HYBRID_ENABLED=true",
                 f"LOCALIST_OLLAMA_HYBRID_LOCAL_MODEL={LOCAL}",
                 f"LOCALIST_OLLAMA_HYBRID_CLOUD_MODEL={CLOUD}",
                 "LOCALIST_OLLAMA_HYBRID_TOKEN_CAP=4000"):
        assert line in env
    assert len(builds) == 1


def test_disable_rebuilds_back_to_plain_client(api):
    client, tmp_path, _, builds = api
    client.put("/settings/ollama-hybrid", json={"enabled": True, "local_model": LOCAL, "cloud_model": CLOUD})
    r = client.put("/settings/ollama-hybrid", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["active"] is False
    assert type(main._state.runtime) is OllamaRuntimeClient
    assert "LOCALIST_OLLAMA_HYBRID_ENABLED=false" in _env(tmp_path)
    assert len(builds) == 2


# S-3 -----------------------------------------------------------------------

def test_cap_change_applies_in_place_and_keeps_latches(api):
    client, tmp_path, _, builds = api
    client.put("/settings/ollama-hybrid", json={"enabled": True, "local_model": LOCAL, "cloud_model": CLOUD})
    rt = main._state.runtime
    rt.decide_tier("c1", 5_000)                   # latched at 4,000
    r = client.put("/settings/ollama-hybrid", json={"token_cap": 8_000})
    assert r.status_code == 200 and r.json()["applied_live"] is True
    assert main._state.runtime is rt and len(builds) == 1
    assert rt.tracker.cap == 8_000
    assert rt.decide_tier("c1", 5_000).tier == "cloud"
    assert "LOCALIST_OLLAMA_HYBRID_TOKEN_CAP=8000" in _env(tmp_path)


# S-4 -----------------------------------------------------------------------

@pytest.mark.parametrize("payload,status,fragment", [
    ({"local_model": CLOUD}, 400, "is a cloud model"),
    ({"cloud_model": LOCAL}, 400, "not an Ollama Cloud"),
    ({"enabled": True}, 400, "both a local and a cloud model"),
    ({"enabled": True, "local_model": LOCAL}, 400, "both a local and a cloud model"),
    ({"enabled": True, "local_model": "gemma4:27b", "cloud_model": CLOUD}, 400, "Not served by Ollama: gemma4:27b"),
    ({"enabled": True, "local_model": "nomic-embed-text:latest", "cloud_model": CLOUD}, 400, "Not a chat model: nomic-embed-text:latest"),
    ({"token_cap": 999}, 400, "between 1,000 and 100,000"),
    ({"token_cap": 100_001}, 400, "between 1,000 and 100,000"),
])
def test_rejections_persist_nothing(api, payload, status, fragment):
    client, tmp_path, _, builds = api
    r = client.put("/settings/ollama-hybrid", json=payload)
    assert r.status_code == status
    assert fragment in r.json()["detail"]
    assert _env(tmp_path) == "" and builds == []
    assert main._state.settings.ollama_hybrid_enabled is False


def test_unreachable_ollama_blocks_enabling(api):
    client, tmp_path, served, builds = api
    served["reachable"] = False
    r = client.put("/settings/ollama-hybrid", json={"enabled": True, "local_model": LOCAL, "cloud_model": CLOUD})
    assert r.status_code == 502
    assert _env(tmp_path) == "" and builds == []


def test_models_can_be_saved_while_off(api):
    client, tmp_path, _, builds = api
    r = client.put("/settings/ollama-hybrid", json={"local_model": LOCAL, "cloud_model": CLOUD})
    assert r.status_code == 200
    assert r.json()["active"] is False and builds == []
    assert f"LOCALIST_OLLAMA_HYBRID_CLOUD_MODEL={CLOUD}" in _env(tmp_path)


# S-5 -----------------------------------------------------------------------

def test_other_backend_active_saves_without_applying(api):
    client, tmp_path, _, builds = api
    main._state.settings.runtime_backend = "openrouter"
    r = client.put("/settings/ollama-hybrid", json={"enabled": True, "local_model": LOCAL, "cloud_model": CLOUD})
    assert r.status_code == 200
    assert (r.json()["applied_live"], r.json()["active"]) == (False, False)
    assert builds == []
    assert "LOCALIST_OLLAMA_HYBRID_ENABLED=true" in _env(tmp_path)


# S-6 -----------------------------------------------------------------------

def test_health_reports_hybrid(api, monkeypatch):
    client, *_ = api
    h = HybridOllamaRuntimeClient.build(LOCAL, CLOUD, token_cap=8_000)
    monkeypatch.setattr(h._local, "health_check", lambda: {
        "reachable": True, "models": list(SERVED), "chat_model": LOCAL,
        "chat_model_found": True, "base_url": "http://localhost:11434",
    })
    main._state.runtime = h
    body = client.get("/health").json()
    assert body["hybrid"] is True
    assert (body["chat_model"], body["hybrid_cloud_model"]) == (LOCAL, CLOUD)
    assert body["hybrid_cloud_model_found"] is True
    assert body["hybrid_token_cap"] == 8_000


def test_health_plain_ollama_is_not_hybrid(api, monkeypatch):
    client, *_ = api
    monkeypatch.setattr(main._state.runtime, "health_check", lambda: {
        "reachable": True, "models": list(SERVED), "chat_model": LOCAL,
        "chat_model_found": True, "base_url": "http://localhost:11434",
    })
    body = client.get("/health").json()
    assert body["hybrid"] is False and body["hybrid_token_cap"] is None
