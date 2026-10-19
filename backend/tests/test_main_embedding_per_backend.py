"""
Per-backend embedding-model pins (docs/architecture/16-runtime-backend-layer.md
§16.25): resolution, the one-time legacy LOCALIST_EMBEDDING_MODEL migration,
the embedder following the target backend on a live switch, GET /health's
reporting of a kept embedder, and POST /settings/embedding-model writing the
active backend's pin.

Same conventions as test_main_runtime_backend_switch.py: lifespan() never
runs, AppState is swapped directly, main.create_runtime is faked, and
main._PROJECT_ROOT points at tmp_path so the real backend/.env is never
touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager
from localist.threshold_calibration import CalibrationResult

NOMIC = "nomic-embed-text:latest"          # has validated thresholds — no calibration
GEMMA = "embeddinggemma-300m-4bit"         # no thresholds — triggers first-time calibration


def _settings(**overrides):
    defaults = dict(
        runtime_backend="ollama",
        chat_model=None,
        chat_model_omlx="gemma-4-e4b-it-4bit",
        chat_model_ollama="gemma4:31b-cloud",
        embedding_model="",
        embedding_model_omlx="",
        embedding_model_ollama="",
        omlx_url="http://localhost:8000",
        ollama_url="http://localhost:11434",
        request_timeout=30.0,
        stream_timeout=60.0,
        episodic_write_approval=False,
    )
    defaults.update(overrides)
    return main.Settings(**defaults)


def _fake_create_runtime(served_embedding_models: dict[str, set[str]], calls: list | None = None):
    """
    Fake create_runtime whose client reports embed_model_found only when the
    embedding model it was built with is in served_embedding_models[backend].
    Records (backend, embedding_model) per call when `calls` is given.
    """
    def _create(backend, **kwargs):
        embedding_model = kwargs.get("embedding_model", "")
        if calls is not None:
            calls.append((backend, embedding_model))
        client = MagicMock(name=f"fake-runtime-{backend}-{embedding_model or 'none'}")
        client.health_check.return_value = {
            "reachable":         True,
            "base_url":          f"http://{backend}",
            "models":            ["chat-model"],
            "chat_model":        kwargs.get("chat_model") or "",
            "chat_model_found":  True,
            "embed_model_found": bool(embedding_model) and embedding_model in served_embedding_models.get(backend, set()),
            "error":             None,
        }
        return client
    return _create


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    saved = {
        name: getattr(main._state, name) for name in (
            "settings", "runtime", "wiki_agent", "controller", "memory_manager",
            "templates_dir", "active_embedding_model_name", "active_embedding_backend",
        )
    }
    main._state.settings       = _settings()
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "per_backend.db")
    main._state.templates_dir  = tmp_path
    main._state.runtime        = MagicMock(name="initial-runtime")
    main._state.wiki_agent     = MagicMock(name="initial-wiki-agent")
    main._state.controller     = MagicMock(name="initial-controller")
    main._state.active_embedding_model_name = None
    main._state.active_embedding_backend    = None

    calibrations: list = []
    monkeypatch.setattr(
        main, "_calibrate_and_persist",
        lambda mm, name, fn: calibrations.append(name) or CalibrationResult(gates={}),
    )

    yield TestClient(main.app), calibrations

    for name, value in saved.items():
        setattr(main._state, name, value)


def _start_embedding_on(backend: str, model: str) -> MagicMock:
    """Put _state into "embedding via `backend` with `model`" as lifespan() would."""
    embed_fn = MagicMock(name=f"embed-{backend}")
    main._state.memory_manager.set_embedding_source(embed_fn, model)
    main._state.active_embedding_model_name = model
    main._state.active_embedding_backend    = backend
    return embed_fn


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolveEmbeddingModel:
    @pytest.mark.parametrize("overrides, backend, expected", [
        ({},                                                          "ollama", ""),
        ({"embedding_model_ollama": NOMIC},                           "ollama", NOMIC),
        ({"embedding_model_ollama": NOMIC},                           "omlx",   ""),
        ({"embedding_model": NOMIC},                                  "omlx",   NOMIC),
        ({"embedding_model": NOMIC, "embedding_model_omlx": GEMMA},   "omlx",   GEMMA),
        ({"embedding_model_omlx": GEMMA},                             " OMLX ", GEMMA),
    ])
    def test_pin_wins_and_legacy_is_only_a_fallback(self, overrides, backend, expected):
        assert main._resolve_embedding_model(_settings(**overrides), backend) == expected

    def test_derived_name_follows_the_named_backend(self):
        settings = _settings(runtime_backend="ollama", embedding_model_ollama=NOMIC, embedding_model_omlx=GEMMA)
        embed_fn = MagicMock()

        assert main._derive_active_embedding_model_name(settings, embed_fn) == NOMIC
        assert main._derive_active_embedding_model_name(settings, embed_fn, "omlx") == GEMMA
        assert main._derive_active_embedding_model_name(settings, None, "omlx") is None


# ---------------------------------------------------------------------------
# One-time legacy migration (lifespan)
# ---------------------------------------------------------------------------

class TestMigrateLegacyEmbeddingModel:
    def test_served_legacy_model_becomes_this_backends_pin(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
        env_path = tmp_path / ".env"
        env_path.write_text(f"LOCALIST_RUNTIME_BACKEND=ollama\nLOCALIST_EMBEDDING_MODEL={NOMIC}\n")
        settings = _settings(embedding_model=NOMIC)

        main._migrate_legacy_embedding_model(settings, "ollama", {"embed_model_found": True})

        assert settings.embedding_model_ollama == NOMIC
        assert settings.embedding_model == ""
        assert settings.embedding_model_omlx == ""
        env = env_path.read_text()
        assert f"LOCALIST_EMBEDDING_MODEL_OLLAMA={NOMIC}\n" in env
        assert "LOCALIST_EMBEDDING_MODEL=\n" in env

    def test_unserved_legacy_model_is_left_alone(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
        settings = _settings(runtime_backend="omlx", embedding_model=NOMIC)

        main._migrate_legacy_embedding_model(settings, "omlx", {"embed_model_found": False})

        assert settings.embedding_model == NOMIC
        assert settings.embedding_model_omlx == ""
        assert not (tmp_path / ".env").exists()

    def test_existing_pin_is_never_overwritten(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
        settings = _settings(embedding_model=NOMIC, embedding_model_ollama="mxbai-embed-large:latest")

        main._migrate_legacy_embedding_model(settings, "ollama", {"embed_model_found": True})

        assert settings.embedding_model_ollama == "mxbai-embed-large:latest"
        assert settings.embedding_model == NOMIC
        assert not (tmp_path / ".env").exists()

    def test_env_write_failure_keeps_legacy_for_a_retry(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(main, "_write_env_var", MagicMock(side_effect=OSError("read-only")))
        settings = _settings(embedding_model=NOMIC)

        main._migrate_legacy_embedding_model(settings, "ollama", {"embed_model_found": True})

        # Pin applies in-process; the legacy value stays so the next start retries.
        assert settings.embedding_model_ollama == NOMIC
        assert settings.embedding_model == NOMIC


# ---------------------------------------------------------------------------
# Live runtime-backend switch
# ---------------------------------------------------------------------------

class TestSwitchEmbedderFollowsTargetPin:
    def test_target_pin_served_moves_the_embedder(self, client, monkeypatch):
        test_client, calibrations = client
        main._state.settings.embedding_model_ollama = NOMIC
        main._state.settings.embedding_model_omlx   = GEMMA
        _start_embedding_on("ollama", NOMIC)
        calls: list = []
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"omlx": {GEMMA}}, calls))

        resp = test_client.post("/settings/runtime-backend", json={"backend": "omlx"})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert calls[0] == ("omlx", GEMMA)
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
        assert main._state.active_embedding_model_name == GEMMA
        assert main._state.active_embedding_backend == "omlx"
        assert body["embedding_model"] == GEMMA
        assert body["embedding_backend"] == "omlx"
        assert body["embedding_warning"] is None
        # Handed to MemoryManager, whose provenance check owns re-embed/stale
        # handling (covered in test_embedding_provenance.py).
        assert main._state.memory_manager._embedding_model_name == GEMMA
        # GEMMA has no thresholds yet — first-time calibration ran for it.
        assert calibrations == [GEMMA]
        assert body["calibration"] is not None

    def test_target_without_a_pin_keeps_the_current_embedder(self, client, monkeypatch):
        test_client, calibrations = client
        main._state.settings.embedding_model_ollama = NOMIC
        kept = _start_embedding_on("ollama", NOMIC)
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"omlx": {GEMMA}}))

        body = test_client.post("/settings/runtime-backend", json={"backend": "omlx"}).json()

        assert main._state.memory_manager.embed_fn is kept
        assert main._state.active_embedding_model_name == NOMIC
        assert main._state.active_embedding_backend == "ollama"
        assert body["embedding_model"] == NOMIC
        assert body["embedding_backend"] == "ollama"
        assert body["embedding_warning"] is None
        assert calibrations == []

    def test_target_pin_not_served_keeps_the_embedder_and_warns(self, client, monkeypatch):
        test_client, _ = client
        main._state.settings.embedding_model_ollama = NOMIC
        main._state.settings.embedding_model_omlx   = GEMMA
        kept = _start_embedding_on("ollama", NOMIC)
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"omlx": set()}))

        body = test_client.post("/settings/runtime-backend", json={"backend": "omlx"}).json()

        assert main._state.memory_manager.embed_fn is kept
        assert body["embedding_backend"] == "ollama"
        assert GEMMA in body["embedding_warning"]
        assert NOMIC in body["embedding_warning"]

    def test_legacy_global_doesnt_leak_into_a_backend_that_lacks_it(self, client, monkeypatch):
        # Before pins, switching carried LOCALIST_EMBEDDING_MODEL into the other
        # backend's client. It's still the fallback there, but an unserved
        # fallback must keep the working embedder, not drop it.
        test_client, _ = client
        main._state.settings.embedding_model = NOMIC
        kept = _start_embedding_on("ollama", NOMIC)
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"omlx": {GEMMA}}))

        body = test_client.post("/settings/runtime-backend", json={"backend": "omlx"}).json()

        assert main._state.memory_manager.embed_fn is kept
        assert body["embedding_model"] == NOMIC
        assert body["embedding_warning"] is not None

    def test_switching_back_reuses_the_same_model_without_going_stale(self, client, monkeypatch):
        test_client, _ = client
        main._state.settings.embedding_model_ollama = NOMIC
        _start_embedding_on("ollama", NOMIC)
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"ollama": {NOMIC}, "omlx": set()}))

        test_client.post("/settings/runtime-backend", json={"backend": "omlx"})
        test_client.post("/settings/runtime-backend", json={"backend": "ollama"})

        assert main._state.active_embedding_backend == "ollama"
        assert main._state.memory_manager.embed_fn is main._state.runtime.embed
        assert main._state.memory_manager._corpus_stale is False


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthReportsTheRealEmbedder:
    def test_kept_embedder_from_another_backend_reports_in_use(self, client):
        test_client, _ = client
        main._state.settings.runtime_backend = "omlx"
        main._state.settings.embedding_model_ollama = NOMIC
        _start_embedding_on("ollama", NOMIC)
        main._state.runtime.health_check.return_value = {
            "reachable": True, "base_url": "http://omlx", "models": [], "chat_model": "x",
            "chat_model_found": True, "embed_model_found": False,
        }

        body = test_client.get("/health").json()

        assert body["embed_model_found"] is True
        assert body["active_embedding_backend"] == "ollama"
        assert body["active_embedding_model_name"] == NOMIC
        # embedding_model is the active backend's own resolved choice: none here.
        assert body["embedding_model"] == ""

    def test_keyword_only_reports_not_found(self, client):
        test_client, _ = client
        main._state.runtime.health_check.return_value = {
            "reachable": True, "base_url": "http://ollama", "models": [], "chat_model": "x",
            "chat_model_found": True, "embed_model_found": True,
        }

        body = test_client.get("/health").json()

        assert body["embed_model_found"] is False
        assert body["active_embedding_backend"] is None


# ---------------------------------------------------------------------------
# POST /settings/embedding-model
# ---------------------------------------------------------------------------

class TestEmbeddingEndpointWritesTheActiveBackendsPin:
    def test_sets_pin_retires_legacy_and_leaves_the_other_backend_alone(self, client, monkeypatch, tmp_path):
        test_client, _ = client
        env_path = tmp_path / ".env"
        env_path.write_text(f"LOCALIST_EMBEDDING_MODEL={NOMIC}\nLOCALIST_EMBEDDING_MODEL_OMLX={GEMMA}\n")
        main._state.settings.embedding_model      = NOMIC
        main._state.settings.embedding_model_omlx = GEMMA
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"ollama": {NOMIC}}))

        resp = test_client.post("/settings/embedding-model", json={"model": NOMIC})

        assert resp.status_code == 200, resp.text
        settings = main._state.settings
        assert settings.embedding_model_ollama == NOMIC
        assert settings.embedding_model == ""
        assert settings.embedding_model_omlx == GEMMA
        assert main._state.active_embedding_backend == "ollama"
        env = env_path.read_text()
        assert f"LOCALIST_EMBEDDING_MODEL_OLLAMA={NOMIC}\n" in env
        assert "LOCALIST_EMBEDDING_MODEL=\n" in env
        assert f"LOCALIST_EMBEDDING_MODEL_OMLX={GEMMA}\n" in env

    def test_clearing_means_none_for_this_backend_not_the_legacy_fallback(self, client, monkeypatch):
        test_client, _ = client
        main._state.settings.embedding_model        = NOMIC
        main._state.settings.embedding_model_ollama = NOMIC
        _start_embedding_on("ollama", NOMIC)
        monkeypatch.setattr(main, "create_runtime", _fake_create_runtime({"ollama": {NOMIC}}))

        test_client.post("/settings/embedding-model", json={"model": ""})

        assert main._resolve_embedding_model(main._state.settings, "ollama") == ""
        assert main._state.memory_manager.embed_fn is None
        assert main._state.active_embedding_backend is None
