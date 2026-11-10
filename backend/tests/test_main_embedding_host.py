"""
Embedding host for chat-only backends (docs/architecture/16-runtime-backend-layer.md
§16.32): a backend that can't embed (a hosted chat API) gets its embeddings
from local Ollama through a separate embed-only client — at startup, on a live
switch, from POST /settings/embedding-model, and on GET /health (warning while
Ollama isn't serving, recovering once it is).

No real chat-only backend exists yet, so a fake "chatonly" backend is added to
runtime_factory's registry for these tests. Same conventions as
test_main_embedding_per_backend.py: lifespan() never runs, AppState is
swapped directly, main.create_runtime is faked, and main._PROJECT_ROOT points
at tmp_path so the real backend/.env is never touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main, runtime_factory
from localist.memory_manager import MemoryManager
from localist.threshold_calibration import CalibrationResult

NOMIC = "nomic-embed-text:latest"   # has validated thresholds — no calibration


def _settings(**overrides):
    defaults = dict(
        runtime_backend="ollama",
        chat_model=None,
        chat_model_omlx="gemma-4-e4b-it-4bit",
        chat_model_ollama="gemma4:e4b-mlx",
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


class _Hosts:
    """Fake create_runtime with per-backend reachability and served embedding models."""

    def __init__(self, served: dict[str, set[str]], down: set[str] | None = None) -> None:
        self.served = served
        self.down = down or set()
        self.calls: list[dict] = []
        self.clients: list[MagicMock] = []

    def __call__(self, backend, **kwargs):
        self.calls.append({"backend": backend, **kwargs})
        client = MagicMock(name=f"fake-{backend}-{kwargs.get('embedding_model') or 'none'}")
        client.is_local = backend != "chatonly"

        def _health(backend=backend, embedding_model=kwargs.get("embedding_model", "")):
            reachable = backend not in self.down
            return {
                "reachable":         reachable,
                "base_url":          f"http://{backend}",
                "models":            sorted(self.served.get(backend, set())) if reachable else [],
                "chat_model":        kwargs.get("chat_model") or "",
                "chat_model_found":  reachable,
                "embed_model_found": (
                    reachable and bool(embedding_model)
                    and embedding_model in self.served.get(backend, set())
                ),
                "error":             None if reachable else "connection refused",
            }

        client.health_check.side_effect = _health
        self.clients.append(client)
        return client


@pytest.fixture()
def chatonly_registered(monkeypatch):
    registry = dict(runtime_factory._REGISTRY)
    registry["chatonly"] = lambda kwargs: MagicMock(name="unused")
    monkeypatch.setattr(runtime_factory, "_REGISTRY", registry)


@pytest.fixture()
def client(tmp_path, monkeypatch, chatonly_registered):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    saved = {
        name: getattr(main._state, name) for name in (
            "settings", "runtime", "wiki_agent", "controller", "memory_manager",
            "templates_dir", "active_embedding_model_name", "active_embedding_backend",
            "embedder_runtime",
        )
    }
    main._state.settings       = _settings()
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "embedding_host.db")
    main._state.templates_dir  = tmp_path
    main._state.runtime        = MagicMock(name="initial-runtime")
    main._state.wiki_agent     = MagicMock(name="initial-wiki-agent")
    main._state.controller     = MagicMock(name="initial-controller")
    main._state.active_embedding_model_name = None
    main._state.active_embedding_backend    = None
    main._state.embedder_runtime            = None
    monkeypatch.setattr(
        main, "_calibrate_and_persist",
        lambda mm, name, fn: CalibrationResult(gates={}),
    )

    yield TestClient(main.app)

    for name, value in saved.items():
        setattr(main._state, name, value)


def _start_embedding_on(backend: str, model: str = NOMIC) -> MagicMock:
    embed_fn = MagicMock(name=f"embed-{backend}")
    main._state.memory_manager.set_embedding_source(embed_fn, model)
    main._state.active_embedding_model_name = model
    main._state.active_embedding_backend    = backend
    return embed_fn


def _start_chatonly(hosts: _Hosts, embedding: bool) -> MagicMock:
    """_state as lifespan() leaves it with chatonly active and an embed-only Ollama client."""
    main._state.settings.runtime_backend = "chatonly"
    main._state.runtime = hosts("chatonly", chat_model="remote-model", embedding_model="")
    embedder = hosts("ollama", chat_model="", embedding_model=NOMIC)
    main._state.embedder_runtime = embedder
    if embedding:
        main._state.memory_manager.set_embedding_source(embedder.embed, NOMIC)
        main._state.active_embedding_model_name = NOMIC
        main._state.active_embedding_backend    = "ollama"
    return embedder


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestEmbeddingHostFor:
    def test_embedding_capable_backends_host_themselves(self):
        assert runtime_factory.embedding_host_for("omlx") == "omlx"
        assert runtime_factory.embedding_host_for(" Ollama ") == "ollama"

    def test_chat_only_backend_falls_back_to_local_ollama(self, chatonly_registered):
        assert runtime_factory.embedding_host_for("chatonly") == "ollama"


class TestResolveEmbeddingHost:
    def test_capable_backend_reuses_the_chat_runtime(self, monkeypatch):
        hosts = _Hosts({"ollama": {NOMIC}})
        monkeypatch.setattr(main, "create_runtime", hosts)
        runtime, health = MagicMock(), {"reachable": True}

        result = main._resolve_embedding_host(_settings(), "ollama", runtime, health)

        assert result == ("ollama", runtime, health, None)
        assert hosts.calls == []

    def test_chat_only_backend_gets_an_embed_only_ollama_client(self, monkeypatch, chatonly_registered):
        hosts = _Hosts({"ollama": {NOMIC}})
        monkeypatch.setattr(main, "create_runtime", hosts)

        host, host_runtime, host_health, embedder = main._resolve_embedding_host(
            _settings(), "chatonly", MagicMock(), {},
        )

        assert host == "ollama"
        assert embedder is host_runtime
        assert host_health["embed_model_found"] is True
        assert hosts.calls == [{
            "backend": "ollama", "chat_model": "", "embedding_model": NOMIC,
            "omlx_url": "http://localhost:8000", "ollama_url": "http://localhost:11434",
            "request_timeout": 30.0, "stream_timeout": 60.0,
        }]

    def test_startup_source_uses_the_host_pin(self, monkeypatch, chatonly_registered):
        hosts = _Hosts({"ollama": {NOMIC}})
        monkeypatch.setattr(main, "create_runtime", hosts)
        settings = _settings(runtime_backend="chatonly")
        host, host_runtime, host_health, _ = main._resolve_embedding_host(
            settings, "chatonly", MagicMock(), {},
        )

        embed_fn = main._configure_embedding_source(settings, host_runtime, host_health, host)

        assert embed_fn is host_runtime.embed
        assert main._derive_active_embedding_model_name(settings, embed_fn, host) == NOMIC


class TestEmbeddingHostWarning:
    def test_no_warning_when_the_chat_backend_is_its_own_host(self):
        assert main._embedding_host_warning(_settings(), "ollama", None) is None

    def test_not_running(self):
        warning = main._embedding_host_warning(_settings(), "ollama", {"reachable": False})
        assert "Ollama isn't running" in warning
        assert "keyword matching" in warning

    def test_no_model_set(self):
        warning = main._embedding_host_warning(
            _settings(embedding_model_ollama=""), "ollama", {"reachable": True},
        )
        assert "No embedding model is set for Ollama" in warning

    def test_model_not_installed(self):
        warning = main._embedding_host_warning(
            _settings(), "ollama", {"reachable": True, "embed_model_found": False},
        )
        assert NOMIC in warning

    def test_serving(self):
        assert main._embedding_host_warning(
            _settings(), "ollama", {"reachable": True, "embed_model_found": True},
        ) is None


# ---------------------------------------------------------------------------
# POST /settings/runtime-backend
# ---------------------------------------------------------------------------

class TestSwitchToChatOnlyBackend:
    def test_embeddings_move_to_local_ollama_without_going_stale(self, client, monkeypatch):
        _start_embedding_on("ollama")
        hosts = _Hosts({"ollama": {NOMIC}})
        monkeypatch.setattr(main, "create_runtime", hosts)

        resp = client.post("/settings/runtime-backend", json={"backend": "chatonly"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        chat_runtime, embedder = hosts.clients
        assert main._state.runtime is chat_runtime
        assert main._state.embedder_runtime is embedder
        assert main._state.memory_manager.embed_fn is embedder.embed
        assert main._state.active_embedding_backend == "ollama"
        assert body["embedding_backend"] == "ollama"
        assert body["embedding_model"] == NOMIC
        assert body["embedding_warning"] is None
        # Same model as before: nothing to re-embed.
        assert main._state.memory_manager._corpus_stale is False
        assert hosts.calls[1]["backend"] == "ollama"
        assert hosts.calls[1]["chat_model"] == ""

    def test_ollama_down_keeps_the_embedder_and_says_so(self, client, monkeypatch):
        kept = _start_embedding_on("ollama")
        monkeypatch.setattr(main, "create_runtime", _Hosts({"ollama": {NOMIC}}, down={"ollama"}))

        body = client.post("/settings/runtime-backend", json={"backend": "chatonly"}).json()

        assert main._state.memory_manager.embed_fn is kept
        assert "Ollama isn't running" in body["embedding_warning"]

    def test_switching_back_drops_the_separate_embedder(self, client, monkeypatch):
        _start_embedding_on("ollama")
        monkeypatch.setattr(main, "create_runtime", _Hosts({"ollama": {NOMIC}}))

        client.post("/settings/runtime-backend", json={"backend": "chatonly"})
        client.post("/settings/runtime-backend", json={"backend": "ollama"})

        assert main._state.embedder_runtime is None
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
        assert main._state.active_embedding_backend == "ollama"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthWithChatOnlyBackend:
    def test_ollama_down_and_keyword_only_warns(self, client, monkeypatch):
        hosts = _Hosts({"ollama": {NOMIC}}, down={"ollama"})
        _start_chatonly(hosts, embedding=False)

        body = client.get("/health").json()

        assert body["backend"] == "chatonly"
        assert body["embedding_host"] == "ollama"
        assert body["embed_model_found"] is False
        assert "Ollama isn't running" in body["embedding_warning"]
        assert main._state.active_embedding_backend is None

    def test_embedding_through_ollama_reports_in_use_without_warning(self, client):
        hosts = _Hosts({"ollama": {NOMIC}})
        _start_chatonly(hosts, embedding=True)

        body = client.get("/health").json()

        assert body["embed_model_found"] is True
        assert body["embedding_model"] == NOMIC
        assert body["active_embedding_backend"] == "ollama"
        assert body["embedding_warning"] is None

    def test_ollama_stopping_mid_session_warns(self, client):
        hosts = _Hosts({"ollama": {NOMIC}})
        _start_chatonly(hosts, embedding=True)
        hosts.down.add("ollama")

        body = client.get("/health").json()

        assert body["embed_model_found"] is False
        assert "Ollama isn't running" in body["embedding_warning"]

    def test_ollama_starting_later_turns_embeddings_on(self, client):
        hosts = _Hosts({"ollama": {NOMIC}}, down={"ollama"})
        embedder = _start_chatonly(hosts, embedding=False)
        assert client.get("/health").json()["embedding_warning"] is not None

        hosts.down.clear()
        body = client.get("/health").json()

        assert main._state.memory_manager.embed_fn is embedder.embed
        assert main._state.active_embedding_backend == "ollama"
        assert main._state.active_embedding_model_name == NOMIC
        assert body["embed_model_found"] is True
        assert body["embedding_warning"] is None
        # The chat runtime itself is untouched by the recovery.
        assert main._state.runtime is hosts.clients[0]

    def test_embedder_kept_from_another_backend_doesnt_warn(self, client):
        hosts = _Hosts({"ollama": {NOMIC}}, down={"ollama"})
        _start_chatonly(hosts, embedding=False)
        _start_embedding_on("omlx", "some-omlx-embedder")

        body = client.get("/health").json()

        assert body["embed_model_found"] is True
        assert body["active_embedding_backend"] == "omlx"
        assert body["embedding_warning"] is None

    def test_capable_backend_never_warns(self, client):
        main._state.runtime.health_check.return_value = {
            "reachable": True, "base_url": "http://ollama", "models": [], "chat_model": "x",
            "chat_model_found": True, "embed_model_found": False,
        }

        body = client.get("/health").json()

        assert body["embedding_host"] == "ollama"
        assert body["embedding_warning"] is None


# ---------------------------------------------------------------------------
# POST /settings/embedding-model
# ---------------------------------------------------------------------------

class TestEmbeddingEndpointWithChatOnlyBackend:
    def test_sets_the_ollama_pin_and_leaves_the_chat_runtime_alone(self, client, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text("LOCALIST_EMBEDDING_MODEL_OLLAMA=\n")
        hosts = _Hosts({"ollama": {NOMIC, "other-embed"}})
        _start_chatonly(hosts, embedding=True)
        chat_runtime = main._state.runtime
        monkeypatch.setattr(main, "create_runtime", hosts)

        resp = client.post("/settings/embedding-model", json={"model": "other-embed"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["backend"] == "ollama"
        assert main._state.runtime is chat_runtime
        candidate = hosts.clients[-1]
        assert main._state.embedder_runtime is candidate
        assert main._state.memory_manager.embed_fn is candidate.embed
        assert main._state.settings.embedding_model_ollama == "other-embed"
        assert main._state.active_embedding_backend == "ollama"
        assert "LOCALIST_EMBEDDING_MODEL_OLLAMA=other-embed" in (tmp_path / ".env").read_text()

    def test_ollama_down_is_refused(self, client, monkeypatch):
        hosts = _Hosts({"ollama": {NOMIC}}, down={"ollama"})
        _start_chatonly(hosts, embedding=False)
        monkeypatch.setattr(main, "create_runtime", hosts)

        resp = client.post("/settings/embedding-model", json={"model": NOMIC})

        assert resp.status_code == 502
        assert "'ollama'" in resp.json()["detail"]
