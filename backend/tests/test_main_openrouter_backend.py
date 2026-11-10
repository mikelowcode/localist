"""
OpenRouter as a registered runtime backend (docs/architecture/16-runtime-
backend-layer.md §16.34): the factory entry, switching to it (refused without
a key; embeddings from local Ollama; no cache warm-up), its chat-model pin,
its model catalog, and what GET /health passes through.

The OpenRouter client itself is real; its HTTP calls are faked at
openrouter_runtime_client.requests. The Ollama embedding host is a fake from
main.create_runtime, so nothing reaches localhost:11434. Same AppState/
_PROJECT_ROOT conventions as test_main_runtime_backend_switch.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main, runtime_factory
from localist import openrouter_runtime_client as orc
from localist.memory_manager import MemoryManager
from localist.openrouter_runtime_client import DEFAULT_CHAT_MODEL, OpenRouterRuntimeClient
from localist.threshold_calibration import CalibrationResult

NOMIC = "nomic-embed-text:latest"


def _settings(**overrides):
    defaults = dict(
        runtime_backend="ollama",
        chat_model=None,
        chat_model_omlx=None,
        chat_model_ollama="gemma4:e4b-mlx",
        chat_model_openrouter=None,
        embedding_model="",
        embedding_model_omlx="",
        embedding_model_ollama=NOMIC,
        omlx_url="http://localhost:8000",
        ollama_url="http://localhost:11434",
        request_timeout=30.0,
        stream_timeout=60.0,
        episodic_write_approval=False,
    )
    defaults.update(overrides)
    return main.Settings(**defaults)


class FakeHTTP:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


KEY = {"data": {"is_free_tier": True, "limit": None,
                "free_model_daily_requests": {"used": 3, "limit": 50, "remaining": 47}}}
CATALOG = {"data": [
    {"id": DEFAULT_CHAT_MODEL, "name": "Nemotron 3 Ultra (free)", "supported_parameters": ["tools"],
     "pricing": {"prompt": "0", "completion": "0"}, "context_length": 1000000},
    {"id": "acme/paid", "name": "Acme Paid", "supported_parameters": ["tools"],
     "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
]}


def _openrouter_get(url, **kwargs):
    if url.endswith("/key"):
        return FakeHTTP(200, KEY)
    if url.endswith("/endpoints"):
        return FakeHTTP(200, {"data": {"endpoints": [{"provider_name": "Nvidia"}]}})
    if url.endswith("/models"):
        return FakeHTTP(200, CATALOG)
    raise AssertionError(url)


def _create_runtime_with_fake_ollama(calls: list):
    """Real factory for openrouter; a fake, NOMIC-serving client for ollama."""
    real = runtime_factory.create_runtime

    def _create(backend, **kwargs):
        calls.append((backend, kwargs.get("chat_model")))
        if backend != "ollama":
            return real(backend, **kwargs)
        client = MagicMock(name=f"fake-ollama-{kwargs.get('embedding_model')}")
        client.health_check.return_value = {
            "reachable": True, "base_url": "http://ollama", "models": [NOMIC],
            "chat_model": kwargs.get("chat_model") or "", "chat_model_found": True,
            "embed_model_found": kwargs.get("embedding_model") == NOMIC, "error": None,
        }
        return client
    return _create


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orc.requests, "get", _openrouter_get)
    saved = {
        name: getattr(main._state, name) for name in (
            "settings", "runtime", "wiki_agent", "controller", "memory_manager",
            "templates_dir", "active_embedding_model_name", "active_embedding_backend",
            "embedder_runtime",
        )
    }
    main._state.settings       = _settings()
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "openrouter.db")
    main._state.templates_dir  = tmp_path
    main._state.runtime        = MagicMock(name="initial-ollama-runtime")
    main._state.wiki_agent     = MagicMock(name="initial-wiki-agent")
    main._state.controller     = MagicMock(name="initial-controller")
    embed_fn = MagicMock(name="embed-ollama")
    main._state.memory_manager.set_embedding_source(embed_fn, NOMIC)
    main._state.active_embedding_model_name = NOMIC
    main._state.active_embedding_backend    = "ollama"
    main._state.embedder_runtime            = None
    monkeypatch.setattr(main, "_calibrate_and_persist",
                        lambda mm, name, fn: CalibrationResult(gates={}))
    warmups: list = []
    monkeypatch.setattr(main, "_run_cache_warmup", lambda c, r, t: warmups.append(r))

    yield TestClient(main.app), warmups

    for name, value in saved.items():
        setattr(main._state, name, value)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_registered_and_chat_only(self):
        assert "openrouter" in runtime_factory.available_backends()
        assert runtime_factory.embedding_host_for("openrouter") == "ollama"

    def test_defaults_and_key_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-env")
        client = runtime_factory.create_runtime("openrouter", chat_model=None)
        assert isinstance(client, OpenRouterRuntimeClient)
        assert client._chat_model == DEFAULT_CHAT_MODEL
        assert client._api_key == "test-openrouter-key-env"
        assert client.is_local is False
        assert client.wants_cache_warmup is False

    def test_explicit_key_and_model_win(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-env")
        client = runtime_factory.create_runtime(
            "openrouter", chat_model="google/gemma-4-31b-it:free", openrouter_api_key="test-openrouter-key-arg",
        )
        assert client._chat_model == "google/gemma-4-31b-it:free"
        assert client._api_key == "test-openrouter-key-arg"


# ---------------------------------------------------------------------------
# POST /settings/runtime-backend
# ---------------------------------------------------------------------------

class TestSwitchToOpenRouter:
    def test_refused_without_a_key(self, client, monkeypatch):
        test_client, _ = client
        before = main._state.runtime

        resp = test_client.post("/settings/runtime-backend", json={"backend": "openrouter"})

        assert resp.status_code == 502
        assert "No OpenRouter API key" in resp.json()["detail"]
        assert main._state.runtime is before
        assert main._state.settings.runtime_backend == "ollama"

    def test_switch_chats_remotely_and_embeds_locally(self, client, monkeypatch, tmp_path):
        test_client, warmups = client
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-test")
        calls: list = []
        monkeypatch.setattr(main, "create_runtime", _create_runtime_with_fake_ollama(calls))

        resp = test_client.post("/settings/runtime-backend", json={"backend": "openrouter"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        runtime = main._state.runtime
        assert isinstance(runtime, OpenRouterRuntimeClient)
        # No pin: the factory default runs (the response's chat_model is the
        # resolved pin, None here, as for the other backends).
        assert runtime._chat_model == DEFAULT_CHAT_MODEL
        assert body["embedding_backend"] == "ollama"
        assert body["embedding_warning"] is None
        # Embeddings: an embed-only local Ollama client, never OpenRouter.
        assert ("ollama", "") in calls
        assert main._state.memory_manager.embed_fn is main._state.embedder_runtime.embed
        assert main._state.memory_manager._corpus_stale is False
        # No warm-up request spent on the free tier.
        assert warmups == []
        assert "LOCALIST_RUNTIME_BACKEND=openrouter" in (tmp_path / ".env").read_text()

    def test_switch_with_a_pinned_model(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-test")
        monkeypatch.setattr(main, "create_runtime", _create_runtime_with_fake_ollama([]))

        resp = test_client.post("/settings/runtime-backend",
                                json={"backend": "openrouter", "chat_model": "google/gemma-4-31b-it:free"})

        assert resp.status_code == 200, resp.text
        assert main._state.runtime._chat_model == "google/gemma-4-31b-it:free"
        assert main._state.settings.chat_model_openrouter == "google/gemma-4-31b-it:free"

    def test_other_backends_still_warm_up(self, client, monkeypatch):
        test_client, warmups = client
        monkeypatch.setattr(main, "create_runtime", _create_runtime_with_fake_ollama([]))
        test_client.post("/settings/runtime-backend", json={"backend": "ollama"})
        assert len(warmups) == 1


# ---------------------------------------------------------------------------
# Chat-model pin and model catalog
# ---------------------------------------------------------------------------

class TestPinAndCatalog:
    def test_pin_for_inactive_openrouter_is_persisted(self, client, tmp_path):
        test_client, _ = client
        resp = test_client.post("/settings/runtime-backend/openrouter/chat-model",
                                json={"chat_model": "acme/paid"})
        assert resp.status_code == 200, resp.text
        assert main._state.settings.chat_model_openrouter == "acme/paid"
        assert "LOCALIST_CHAT_MODEL_OPENROUTER=acme/paid" in (tmp_path / ".env").read_text()

    def test_catalog_with_details(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-test")
        body = test_client.get("/settings/runtime-backend/openrouter/models").json()
        assert body["reachable"] is True
        assert body["models"] == [DEFAULT_CHAT_MODEL, "acme/paid"]
        assert body["model_details"][0] == {
            "id": DEFAULT_CHAT_MODEL, "name": "Nemotron 3 Ultra (free)", "free": True,
            "context_length": 1000000,
        }
        assert body["chat_model"] == DEFAULT_CHAT_MODEL
        assert body["chat_model_found"] is True

    def test_catalog_free_only(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-test")
        body = test_client.get("/settings/runtime-backend/openrouter/models?free_only=true").json()
        assert body["models"] == [DEFAULT_CHAT_MODEL]

    def test_catalog_without_a_key_reports_unreachable(self, client):
        test_client, _ = client
        body = test_client.get("/settings/runtime-backend/openrouter/models").json()
        assert body["reachable"] is False
        assert body["models"] == []
        assert "No OpenRouter API key" in body["error"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthWhileOnOpenRouter:
    def test_passes_through_key_info_and_last_error(self, client):
        test_client, _ = client
        runtime = OpenRouterRuntimeClient(api_key="test-openrouter-key-test")
        runtime.last_error = {"kind": "upstream_busy", "message": "busy", "at": 1.0}
        main._state.runtime = runtime
        main._state.settings.runtime_backend = "openrouter"
        embedder = MagicMock(name="embedder")
        embedder.health_check.return_value = {"reachable": True, "embed_model_found": True}
        main._state.embedder_runtime = embedder

        body = test_client.get("/health").json()

        assert body["backend"] == "openrouter"
        assert body["healthy"] is True
        assert body["runtime_key_info"]["free_requests_remaining"] == 47
        assert body["runtime_last_error"]["kind"] == "upstream_busy"
        assert body["embedding_host"] == "ollama"
        assert body["embed_model_found"] is True

    def test_local_backends_report_none(self, client):
        test_client, _ = client
        main._state.runtime.health_check.return_value = {
            "reachable": True, "base_url": "http://ollama", "models": [], "chat_model": "x",
            "chat_model_found": True, "embed_model_found": True,
        }
        body = test_client.get("/health").json()
        assert body["runtime_key_info"] is None
        assert body["runtime_last_error"] is None
