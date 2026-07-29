"""
Tests for the GitHub Watch Feed endpoints in main.py:

  GET  /github/watch/preview
  POST /github/watch/refresh

Follows the same TestClient + real-temp-file-MemoryManager pattern as
test_main_news_endpoints.py. github_watch.build_watch_feed() is mocked
(AsyncMock) at every call site that would otherwise reach real GitHub —
this is a pure wiring/persistence test, not a GitHub integration test
(that's github_watch.py's own test_github_watch.py, plus live verification).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import main
from memory_manager import MemoryManager

_FAKE_REPOS = [
    {
        "key": "anthropics/claude-code", "label": "anthropics/claude-code",
        "repo_url": "https://github.com/anthropics/claude-code",
        "latest_release": {"tag_name": "v1.0.0"}, "error": None,
    },
]


@pytest.fixture()
def client(tmp_path):
    prev_memory = main._state.memory_manager
    mm = MemoryManager(db_path=tmp_path / "github_watch_endpoints.db")
    main._state.memory_manager = mm
    yield TestClient(main.app), mm
    main._state.memory_manager = prev_memory


class TestWatchPreview:
    def test_unavailable_when_no_cache(self, client):
        test_client, _ = client
        with patch.object(main.github_watch, "build_watch_feed", new=AsyncMock()) as mock_build:
            resp = test_client.get("/github/watch/preview")

        assert resp.status_code == 200
        assert resp.json()["available"] is False
        mock_build.assert_not_called()

    def test_available_when_cache_present(self, client):
        test_client, mm = client
        mm.set_github_watch_cache(_FAKE_REPOS)

        with patch.object(main.github_watch, "build_watch_feed", new=AsyncMock()) as mock_build:
            resp = test_client.get("/github/watch/preview")

        body = resp.json()
        assert body["available"] is True
        assert body["repos"][0]["key"] == "anthropics/claude-code"
        assert body["generated_at"] > 0
        mock_build.assert_not_called()


class TestWatchRefresh:
    def test_generates_and_caches(self, client):
        test_client, mm = client

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ) as mock_build:
            resp = test_client.post("/github/watch/refresh")

        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        mock_build.assert_awaited_once()

        cache = mm.get_github_watch_cache()
        assert cache["content"] == _FAKE_REPOS

    def test_repeat_call_always_regenerates(self, client):
        test_client, mm = client

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ):
            test_client.post("/github/watch/refresh")

        with patch.object(
            main.github_watch, "build_watch_feed", new=AsyncMock(return_value=_FAKE_REPOS)
        ) as mock_build:
            test_client.post("/github/watch/refresh")

        mock_build.assert_awaited_once()
