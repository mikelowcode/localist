"""
Tests for POST /memory/reembed (main.py) — the manual corpus re-embed
endpoint backing MemoryManager.reembed_corpus() (docs/architecture/
16-runtime-backend-layer.md §16.4).

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_memory_episodes.py — the FastAPI lifespan is never triggered;
_state.memory_manager is swapped in per-test instead.
"""

import sqlite3
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main, memory_manager
from localist.memory_manager import MemoryManager, _EMBEDDING_DIM


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    yield TestClient(main.app), tmp_path
    main._state.memory_manager = prev_memory


def _embed_fn():
    return MagicMock(side_effect=lambda text: [0.1] * _EMBEDDING_DIM)


class TestReembedEndpoint:
    def test_reembeds_all_documents_and_returns_counts(self, client):
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "reembed.db", embed_fn=_embed_fn(),
            embedding_model_name="old-embedding-model:latest",
        )
        mm.index_document(path=tmp_path / "a.md", doc_type="wiki", content="alpha", embed=True)
        mm.index_document(path=tmp_path / "b.md", doc_type="wiki", content="beta", embed=True)
        # A model change leaves both documents without a vector for the new model.
        mm.set_embedding_source(_embed_fn(), "some-embedding-model:latest")
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 200
        body = resp.json()
        assert body == {
            "reembedded": 2,
            "total": 2,
            "already_current": 0,
            "model": "some-embedding-model:latest",
            "failed_ids": [],
            "chat_turns": {"reembedded": 0, "total": 0, "already_current": 0, "failed_ids": []},
            "episodes": {"reembedded": 0, "total": 0, "already_current": 0, "failed_ids": []},
            "calibration": None,
        }

    def test_also_reembeds_chat_turns_and_clears_their_stale_flag(self, client):
        """The Settings UI's one re-embed button calls only this endpoint —
        chat_turns must not be left on the previous model's vectors."""
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "reembed_chat.db", embed_fn=_embed_fn(),
            embedding_model_name="old-embedding-model:latest",
        )
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="assistant", content="hi there", conversation_id="c")
        new_embed_fn = MagicMock(side_effect=lambda text: [0.3] * _EMBEDDING_DIM)
        mm.set_embedding_source(new_embed_fn, "some-embedding-model:latest")
        assert mm._chat_turns_stale is True
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 200
        assert resp.json()["chat_turns"] == {
            "reembedded": 2, "total": 2, "already_current": 0, "failed_ids": [],
        }
        new_embed_fn.assert_any_call("hello")
        new_embed_fn.assert_any_call("hi there")
        assert test_client.get("/memory/stats").json()["chat_turns_stale"] is False

    def test_also_reembeds_episodes(self, client):
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "reembed_episodes.db", embed_fn=_embed_fn(),
            embedding_model_name="some-embedding-model:latest",
        )
        conn = sqlite3.connect(str(tmp_path / "reembed_episodes.db"))
        now = time.time()
        conn.execute(
            "INSERT INTO episodes (episode_type, subject, content, confidence, source, "
            "project_context, status, created_at, last_accessed) "
            "VALUES ('preference', 'dark-mode', 'The user prefers dark mode.', 1.0, "
            "'explicit', 'general', 'active', ?, ?)",
            (now, now),
        )
        conn.commit()
        conn.close()
        new_embed_fn = MagicMock(side_effect=lambda text: [0.3] * _EMBEDDING_DIM)
        mm._embed_fn = new_embed_fn
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 200
        assert resp.json()["episodes"] == {
            "reembedded": 1, "total": 1, "already_current": 0, "failed_ids": [],
        }
        new_embed_fn.assert_any_call("dark-mode. The user prefers dark mode.")

    def test_partial_failure_surfaces_failed_ids_and_keeps_chat_turns_stale(self, client, monkeypatch):
        monkeypatch.setattr(memory_manager, "_REEMBED_RETRY_DELAYS_S", (0.0, 0.0))
        test_client, tmp_path = client
        mm = MemoryManager(
            db_path=tmp_path / "reembed_partial.db", embed_fn=_embed_fn(),
            embedding_model_name="old-embedding-model:latest",
        )
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="assistant", content="boom", conversation_id="c")

        def flaky_embed(text):
            if text == "boom":
                raise RuntimeError("embed down")
            return [0.3] * _EMBEDDING_DIM

        mm.set_embedding_source(flaky_embed, "some-embedding-model:latest")
        assert mm._chat_turns_stale is True
        main._state.memory_manager = mm

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 200
        chat = resp.json()["chat_turns"]
        assert chat["reembedded"] == 1
        assert chat["total"] == 2
        assert len(chat["failed_ids"]) == 1
        assert test_client.get("/memory/stats").json()["chat_turns_stale"] is True

    def test_returns_503_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 503

    def test_returns_409_when_no_embedding_source_configured(self, client):
        test_client, tmp_path = client
        main._state.memory_manager = MemoryManager(db_path=tmp_path / "keyword_only.db")

        resp = test_client.post("/memory/reembed")

        assert resp.status_code == 409


class TestMemoryStatsCorpusStale:
    """GET /memory/stats exposes MemoryManager._corpus_stale, and it toggles
    False -> True -> False around a genuine mismatch / reembed_corpus() call
    (docs/architecture/16-runtime-backend-layer.md §16.4)."""

    def test_corpus_stale_false_by_default(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="some-embedding-model:latest",
        )
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is False

    def test_corpus_stale_true_when_flagged(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="some-embedding-model:latest",
        )
        mm._corpus_stale = True
        main._state.memory_manager = mm

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is True

    def test_reembed_clears_corpus_stale(self, client):
        test_client, tmp_path = client
        embed_fn = _embed_fn()
        mm = MemoryManager(
            db_path=tmp_path / "stats.db", embed_fn=embed_fn,
            embedding_model_name="some-embedding-model:latest",
        )
        mm.index_document(path=tmp_path / "a.md", doc_type="wiki", content="alpha", embed=True)
        mm._corpus_stale = True
        main._state.memory_manager = mm

        assert test_client.get("/memory/stats").json()["corpus_stale"] is True

        resp = test_client.post("/memory/reembed")
        assert resp.status_code == 200

        assert test_client.get("/memory/stats").json()["corpus_stale"] is False

    def test_returns_false_when_memory_manager_not_initialised(self, client):
        test_client, _ = client
        main._state.memory_manager = None

        resp = test_client.get("/memory/stats")

        assert resp.status_code == 200
        assert resp.json()["corpus_stale"] is False
        assert resp.json()["available"] is False
