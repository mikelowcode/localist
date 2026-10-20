"""
GET /health's gate_tiers / active_embedding_model_name fields
(PLAN_semantic_gating_calibration.md §6) — the Settings UI's trust badge
reads these on every page load/poll, not just transiently after a switch or
reembed response. Computed via planner.resolve_gate_tiers(), the exact
function Planner.__init__ uses, so the two can never disagree.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.memory_manager import MemoryManager
from localist.planner import _GATE_NAMES


@pytest.fixture()
def client(tmp_path):
    prev_settings = main._state.settings
    prev_runtime = main._state.runtime
    prev_memory = main._state.memory_manager
    prev_active_model_name = main._state.active_embedding_model_name

    main._state.settings = main.Settings(
        runtime_backend="ollama", chat_model=None, chat_model_omlx=None,
        chat_model_ollama=None, embedding_model="",
        omlx_url="http://localhost:8000",
        ollama_url="http://localhost:11434", request_timeout=30.0,
        stream_timeout=60.0, episodic_write_approval=False,
    )
    fake_runtime = MagicMock(name="fake-runtime")
    fake_runtime.health_check.return_value = {
        "reachable": True, "base_url": "http://localhost:11434",
        "models": [], "chat_model_found": True, "embed_model_found": False,
        "error": None,
    }
    main._state.runtime = fake_runtime
    main._state.memory_manager = MemoryManager(db_path=tmp_path / "health_gate_tiers.db")

    yield TestClient(main.app)

    main._state.settings = prev_settings
    main._state.runtime = prev_runtime
    main._state.memory_manager = prev_memory
    main._state.active_embedding_model_name = prev_active_model_name


class TestHealthGateTiers:
    def test_none_active_model_reports_all_gates_lexical_fallback(self, client):
        main._state.active_embedding_model_name = None

        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["active_embedding_model_name"] is None
        # Search gates are keyword-scored for every model (2026-10-20).
        assert body["gate_tiers"] == {"explicit_search_action": "keyword", "lookup_request": "keyword", "research_intent": "lexical-fallback", "episodic_relevance": "lexical-fallback"}

    def test_validated_model_reports_validated_tiers(self, client):
        main._state.active_embedding_model_name = "nomic-embed-text:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["gate_tiers"] == {"explicit_search_action": "keyword", "lookup_request": "keyword", "research_intent": "validated", "episodic_relevance": "validated"}

    def test_calibrated_model_reports_mixed_tiers_from_persisted_row(self, client):
        main._state.memory_manager.set_calibrated_thresholds(
            "mystery-model:latest", {"lookup_request": 0.6, "research_intent": 0.55},
        )
        main._state.active_embedding_model_name = "mystery-model:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        gate_tiers = resp.json()["gate_tiers"]
        # A persisted lookup_request value (pre-2026-10-20 row) is ignored:
        # the search gates are keyword-scored for every model.
        assert gate_tiers["lookup_request"] == "keyword"
        assert gate_tiers["research_intent"] == "auto-calibrated"
        # Not "disabled" -- PLAN_semantic_gating_calibration.md §9's
        # model-independent lexical/BM25 fallback tier covers every gate,
        # so a gate with neither a validated nor a persisted calibration
        # entry still gets real signal.
        assert gate_tiers["explicit_search_action"] == "keyword"
        assert gate_tiers["episodic_relevance"] == "lexical-fallback"

    def test_never_calibrated_unvalidated_model_reports_all_lexical_fallback(self, client):
        main._state.active_embedding_model_name = "totally-unknown-model:latest"

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["gate_tiers"] == {"explicit_search_action": "keyword", "lookup_request": "keyword", "research_intent": "lexical-fallback", "episodic_relevance": "lexical-fallback"}


class TestHealthChatModel:
    def test_chat_model_passed_through_from_runtime(self, client):
        # An explicit pin makes the missing model unrecoverable, so GET
        # /health's poll-time recovery (§16.22) leaves it as reported.
        main._state.settings.chat_model_ollama = "gemma4:31b-cloud"
        main._state.runtime.health_check.return_value = {
            "reachable": True, "base_url": "http://localhost:11434",
            "models": ["llama3.2:3b"], "chat_model": "gemma4:31b-cloud",
            "chat_model_found": False, "embed_model_found": False, "error": None,
        }

        body = client.get("/health").json()

        assert body["chat_model"] == "gemma4:31b-cloud"
        assert body["chat_model_found"] is False
        assert body["healthy"] is False

    def test_chat_model_defaults_to_empty_when_runtime_omits_it(self, client):
        body = client.get("/health").json()

        assert body["chat_model"] == ""
