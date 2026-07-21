"""
Tests for the episodic memory REST endpoints (main.py):
GET /memory/episodes, POST /memory/episodes/{id}/approve,
POST /memory/episodes/{id}/reject.

Follows the same TestClient + real-temp-file-MemoryManager pattern as
tests/test_main_task_chat_turns.py — the FastAPI lifespan is never
triggered; _state.memory_manager is swapped in per-test instead.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

import main
from memory_manager import MemoryManager, EpisodicMemoryWriter


@pytest.fixture()
def client(tmp_path):
    """
    TestClient against main.app with a real, temp-file-backed MemoryManager
    so pending/active/retracted episode rows can be seeded and inspected
    directly. Restores the previous _state.memory_manager afterward so this
    suite doesn't leak state into other test modules.
    """
    prev_memory = main._state.memory_manager

    main._state.memory_manager = MemoryManager(db_path=tmp_path / "main_episodes.db")

    yield TestClient(main.app)

    main._state.memory_manager = prev_memory


def _status(memory_manager: MemoryManager, episode_id: int) -> str:
    conn = sqlite3.connect(str(memory_manager._db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return row["status"]


def _insert_pending(memory_manager: MemoryManager) -> int:
    writer = EpisodicMemoryWriter(db_path=memory_manager._db_path)
    row_id = writer.insert(
        episode_type   = "project_fact",
        subject        = "staged fact",
        content        = "The user mentioned something offhand.",
        source         = "model_extracted",
        confidence     = 0.7,
        initial_status = "pending",
    )
    assert row_id is not None
    return row_id


class TestApproveEndpoint:

    def test_approve_pending_row_returns_updated_true_and_activates(self, client):
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/approve")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": episode_id, "status": "active", "updated": True}
        assert _status(main._state.memory_manager, episode_id) == "active"

    def test_approve_nonexistent_id_returns_updated_false(self, client):
        resp = client.post("/memory/episodes/999999/approve")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": 999999, "status": "active", "updated": False}

    def test_approve_already_active_row_returns_updated_false(self, client):
        # Ordinary explicit insert — active from the start, never pending.
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        row_id = writer.insert("preference", "x", "y.", "explicit")

        resp = client.post(f"/memory/episodes/{row_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["updated"] is False
        assert _status(main._state.memory_manager, row_id) == "active"


class TestRejectEndpoint:

    def test_reject_pending_row_returns_updated_true_and_retracts(self, client):
        episode_id = _insert_pending(main._state.memory_manager)

        resp = client.post(f"/memory/episodes/{episode_id}/reject")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": episode_id, "status": "retracted", "updated": True}
        assert _status(main._state.memory_manager, episode_id) == "retracted"

    def test_reject_nonexistent_id_returns_updated_false(self, client):
        resp = client.post("/memory/episodes/999999/reject")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"episode_id": 999999, "status": "retracted", "updated": False}

    def test_reject_already_retracted_row_returns_updated_false(self, client):
        episode_id = _insert_pending(main._state.memory_manager)
        first = client.post(f"/memory/episodes/{episode_id}/reject")
        assert first.json()["updated"] is True

        second = client.post(f"/memory/episodes/{episode_id}/reject")
        assert second.status_code == 200
        assert second.json()["updated"] is False


class TestGetEpisodesTotalCount:

    def test_total_reflects_full_count_not_capped_by_limit(self, client):
        # Regression guard: total used to be len(rows) (i.e. capped by
        # `limit`), which made status=pending&limit=1 — the pending-count
        # badge's query shape — always report 0 or 1 no matter how many
        # pending episodes actually existed. total must now come from
        # MemoryManager.count_episodes(), independent of limit/offset.
        for _ in range(3):
            _insert_pending(main._state.memory_manager)

        resp = client.get("/memory/episodes", params={"status": "pending", "limit": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["episodes"]) == 1   # page is still limited...
        assert body["total"] == 3           # ...but total is not

    def test_total_zero_when_nothing_matches(self, client):
        resp = client.get("/memory/episodes", params={"status": "pending", "limit": 1})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


class TestGetEpisodesTaskIdFilter:
    """
    task_id filtering (episode-browsing-ui-plan.md Phase 6) backs the
    Episode Browsing UI's per-turn "related memory" overlay.
    """

    def test_filters_to_matching_task_id_only(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert("preference", "a", "A.", "explicit", task_id="task-1")
        writer.insert("decision", "b", "B.", "explicit", task_id="task-2")

        resp = client.get("/memory/episodes", params={"task_id": "task-1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["episodes"][0]["task_id"] == "task-1"

    def test_no_task_id_returns_all(self, client):
        writer = EpisodicMemoryWriter(db_path=main._state.memory_manager._db_path)
        writer.insert("preference", "a", "A.", "explicit", task_id="task-1")
        writer.insert("decision", "b", "B.", "explicit", task_id="task-2")

        resp = client.get("/memory/episodes")

        assert resp.json()["total"] == 2
