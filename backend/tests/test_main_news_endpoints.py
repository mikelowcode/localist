"""
Tests for the Daily News Brief endpoints in main.py
(docs/daily-news-brief-plan.md §5/§6):

  GET  /news/preferences
  PUT  /news/preferences
  GET  /news/brief/preview
  POST /news/brief/open

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_memory_reembed.py. news_brief.build_brief() is mocked
(AsyncMock) at every call site that would otherwise reach real NewsAPI —
this is a pure wiring/persistence test, not a NewsAPI integration test
(that's news_brief.py's own test_news_brief.py, plus live verification).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from memory_manager import MemoryManager

_FAKE_SECTIONS = [
    {
        "key": "world", "label": "World", "error": None,
        "articles": [{
            "title": "Big Story", "description": "desc", "source": "Reuters",
            "published_at": "2026-07-22T00:00:00Z", "url": "https://example.com/a",
        }],
    },
    {"key": "national", "label": "National", "error": None, "articles": []},
]


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    mm = MemoryManager(db_path=tmp_path / "news_endpoints.db")
    main._state.memory_manager = mm
    yield TestClient(main.app), mm
    main._state.memory_manager = prev_memory


class TestGetPreferences:
    def test_returns_defaults_when_unset(self, client):
        test_client, _ = client
        resp = test_client.get("/news/preferences")

        assert resp.status_code == 200
        body = resp.json()
        assert body["home_country"] == "us"
        assert body["local_query"] is None
        assert body["topics"] == []
        assert "finance" in body["topic_pool"]

    def test_returns_stored_preferences(self, client):
        test_client, mm = client
        mm.set_news_preferences("gb", "London", ["finance", "technology", "sports"])

        resp = test_client.get("/news/preferences")

        body = resp.json()
        assert body["home_country"] == "gb"
        assert body["local_query"] == "London"
        assert body["topics"] == ["finance", "technology", "sports"]


class TestPutPreferences:
    def test_valid_request_roundtrips(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": "Seattle",
            "topics": ["finance", "technology", "sports"],
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["home_country"] == "us"
        assert body["local_query"] == "Seattle"
        assert body["topics"] == ["finance", "technology", "sports"]

    def test_wrong_topic_count_rejected(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": None,
            "topics": ["finance", "technology"],
        })
        assert resp.status_code == 422

    def test_unknown_topic_key_rejected(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "us", "local_query": None,
            "topics": ["finance", "technology", "not-a-real-topic"],
        })
        assert resp.status_code == 422

    def test_home_country_lowercased(self, client):
        test_client, _ = client
        resp = test_client.put("/news/preferences", json={
            "home_country": "US", "local_query": None,
            "topics": ["finance", "technology", "sports"],
        })
        assert resp.json()["home_country"] == "us"


class TestBriefPreview:
    def test_unavailable_when_no_cache(self, client):
        test_client, _ = client
        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        assert resp.status_code == 200
        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_unavailable_when_cache_is_from_a_previous_day(self, client):
        test_client, mm = client
        mm.set_news_brief_cache("2000-01-01", _FAKE_SECTIONS, "conv-old")

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_available_when_cache_matches_today(self, client):
        test_client, mm = client
        today = main._today_str()
        mm.set_news_brief_cache(today, _FAKE_SECTIONS, "conv-today")

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.get("/news/brief/preview")

        body = resp.json()
        assert body["available"] is True
        assert body["brief_date"] == today
        assert body["sections"][0]["key"] == "world"
        mock_build.assert_not_called()


class TestBriefOpen:
    def test_cache_miss_generates_and_writes_two_chat_turns(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ) as mock_build:
            resp = test_client.post("/news/brief/open", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["generated"] is True
        conversation_id = body["conversation_id"]
        mock_build.assert_awaited_once_with("us", None, ["finance", "technology", "sports"])

        rows, total = mm.get_chat_turns(conversation_id=conversation_id, limit=10)
        assert total == 2
        roles = sorted(r["role"] for r in rows)
        assert roles == ["assistant", "user"]

        assistant_row = next(r for r in rows if r["role"] == "assistant")
        assert "Big Story" in assistant_row["content"]
        assert assistant_row["sources"]
        assert assistant_row["sources"][0]["type"] == "web"
        assert assistant_row["sources"][0]["path"] == "https://example.com/a"

        cache = mm.get_news_brief_cache()
        assert cache["conversation_id"] == conversation_id
        assert cache["brief_date"] == main._today_str()

    def test_repeat_call_same_day_reopens_without_regenerating(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            first = test_client.post("/news/brief/open", json={})
        first_id = first.json()["conversation_id"]

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            second = test_client.post("/news/brief/open", json={})

        assert second.json()["generated"] is False
        assert second.json()["conversation_id"] == first_id
        mock_build.assert_not_called()

        # No duplicate chat_turns were written on the reopen.
        rows, total = mm.get_chat_turns(conversation_id=first_id, limit=10)
        assert total == 2

    def test_stale_cache_regenerates_with_a_new_conversation(self, client):
        test_client, mm = client
        mm.set_news_brief_cache("2000-01-01", _FAKE_SECTIONS, "conv-old")

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ) as mock_build:
            resp = test_client.post("/news/brief/open", json={})

        body = resp.json()
        assert body["generated"] is True
        assert body["conversation_id"] != "conv-old"
        mock_build.assert_awaited_once()


class TestBriefOpenWorkingMemorySeed:
    """
    conversation_log (via MemoryManager.add(), keyed by session_id) is a
    completely separate mechanism from chat_turns — it's what
    ControllerAgent._memory_key()/get_context_window() actually read for
    Slot 6 working-memory continuity (confirmed live, 2026-07-22: a
    follow-up question in the brief's chat_turns conversation had no idea
    the brief existed until this seeding was added). These tests are the
    regression guard for that fix.
    """

    def test_cache_miss_seeds_conversation_log_when_session_id_given(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            test_client.post("/news/brief/open", json={"session_id": "sess-1"})

        entries = mm.get_context_window(task_id="sess-1")
        roles = [e["role"] for e in entries]
        assert roles == ["user", "agent"]
        assert entries[0]["content"] == main._NEWS_BRIEF_USER_INSTRUCTION
        assert "Big Story" in entries[1]["content"]

    def test_no_session_id_writes_nothing_to_conversation_log(self, client):
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            test_client.post("/news/brief/open", json={})

        assert mm.get_context_window(task_id="global") == []

    def test_reopen_path_also_seeds_a_fresh_session_id(self, client):
        """A page reload between two presses gets a new SESSION_ID — the
        reopen (cache-hit) path must still seed that new session's working
        memory, not just the original generation's session."""
        test_client, mm = client
        mm.set_news_preferences("us", None, ["finance", "technology", "sports"])

        with patch.object(
            main.news_brief, "build_brief", new=AsyncMock(return_value=_FAKE_SECTIONS)
        ):
            test_client.post("/news/brief/open", json={"session_id": "sess-morning"})

        with patch.object(main.news_brief, "build_brief", new=AsyncMock()) as mock_build:
            resp = test_client.post("/news/brief/open", json={"session_id": "sess-after-reload"})

        assert resp.json()["generated"] is False
        mock_build.assert_not_called()

        entries = mm.get_context_window(task_id="sess-after-reload")
        assert [e["role"] for e in entries] == ["user", "agent"]
        assert "Big Story" in entries[1]["content"]
