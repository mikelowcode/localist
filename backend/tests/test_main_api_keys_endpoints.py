"""
Tests for the optional API keys endpoints in main.py:

  GET /settings/api-keys
  PUT /settings/api-keys

Same TestClient pattern as test_main_github_watch_endpoints.py, with
_PROJECT_ROOT pointed at a temp dir so a save never touches a real .env.
github_watch.validate_token() is mocked wherever it would reach real GitHub;
its own behavior is covered in test_github_watch.py. The call to localist-mcp
(_reload_mcp_keys) is mocked by the client fixture, so these tests never reach
a real server on :8003; TestReloadMcpKeys covers the helper itself.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from fastapi.testclient import TestClient

from localist import main
from localist.mcp_server import main as mcp_main

_ENV_NAMES = ("LANGSEARCH_API_KEY", "BRAVE_API_KEY", "NEWSAPI_API_KEY", "GITHUB_TOKEN")
_VALID = {"status": "valid", "message": "", "expires_on": "January 24, 2027"}
_REJECTED = {"status": "rejected", "message": "GitHub rejected your token.", "expires_on": None}
_UNVERIFIED = {"status": "unverified", "message": "Couldn't reach GitHub.", "expires_on": None}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # delenv also restores each name afterward — the endpoint writes os.environ
    # directly, and importing main has already loaded the real .env.
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_reload_mcp_keys", AsyncMock(return_value=True))
    yield TestClient(main.app), tmp_path / ".env"


def _env_file(path) -> str:
    return path.read_text() if path.exists() else ""


class TestGetStatus:
    def test_all_unset(self, client):
        test_client, _ = client
        assert test_client.get("/settings/api-keys").json() == {
            "langsearch_api_key_set": False,
            "brave_api_key_set":      False,
            "newsapi_api_key_set":    False,
            "github_token_set":       False,
        }

    def test_reports_set_without_echoing_value(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        resp = test_client.get("/settings/api-keys")
        assert resp.json()["github_token_set"] is True
        assert "ghp_secret" not in resp.text


class TestPutGithubToken:
    def test_valid_token_is_saved_and_reports_expiry(self, client):
        test_client, env_path = client
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_VALID)) as check:
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp_good"})

        assert resp.status_code == 200
        check.assert_awaited_once_with("ghp_good")
        assert resp.json()["github_token_set"] is True
        assert "Connected to GitHub" in resp.json()["warning"]
        assert "January 24, 2027" in resp.json()["warning"]
        assert "ghp_good" not in resp.text
        assert "GITHUB_TOKEN=ghp_good\n" in _env_file(env_path)
        assert os.environ["GITHUB_TOKEN"] == "ghp_good"

    def test_valid_token_with_no_expiry(self, client):
        test_client, _ = client
        no_expiry = {**_VALID, "expires_on": None}
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=no_expiry)):
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp_good"})
        assert "Connected to GitHub." in resp.json()["warning"]
        assert "expires" not in resp.json()["warning"]

    def test_rejected_token_keeps_the_current_one(self, client, monkeypatch):
        test_client, env_path = client
        env_path.write_text("GITHUB_TOKEN=ghp_current\n")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_current")
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_REJECTED)):
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp_bad"})

        assert resp.status_code == 400
        assert resp.json()["detail"] == _REJECTED["message"]
        assert "ghp_bad" not in resp.text
        assert _env_file(env_path) == "GITHUB_TOKEN=ghp_current\n"
        assert os.environ["GITHUB_TOKEN"] == "ghp_current"

    def test_rejected_token_blocks_other_keys_in_same_save(self, client):
        test_client, env_path = client
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_REJECTED)):
            resp = test_client.put(
                "/settings/api-keys",
                json={"brave_api_key": "brave-new", "github_token": "ghp_bad"},
            )
        assert resp.status_code == 400
        assert _env_file(env_path) == ""
        assert "BRAVE_API_KEY" not in os.environ

    def test_unverified_token_is_saved_with_warning(self, client):
        test_client, env_path = client
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_UNVERIFIED)):
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp_maybe"})
        assert resp.status_code == 200
        assert "Couldn't reach GitHub." in resp.json()["warning"]
        assert "saved anyway" in resp.json()["warning"]
        assert "GITHUB_TOKEN=ghp_maybe\n" in _env_file(env_path)

    def test_clearing_skips_validation(self, client, monkeypatch):
        test_client, env_path = client
        env_path.write_text("GITHUB_TOKEN=ghp_current\n")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_current")
        with patch.object(main.github_watch, "validate_token", AsyncMock()) as check:
            resp = test_client.put("/settings/api-keys", json={"github_token": ""})
        check.assert_not_called()
        assert resp.status_code == 200
        assert resp.json()["github_token_set"] is False
        assert _env_file(env_path) == "GITHUB_TOKEN=\n"

    def test_other_keys_do_not_trigger_validation(self, client):
        test_client, env_path = client
        with patch.object(main.github_watch, "validate_token", AsyncMock()) as check:
            resp = test_client.put("/settings/api-keys", json={"brave_api_key": "brave-new"})
        check.assert_not_called()
        assert resp.status_code == 200
        assert "BRAVE_API_KEY=brave-new\n" in _env_file(env_path)

    def test_write_failure_does_not_claim_connected(self, client):
        test_client, _ = client
        with (
            patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_VALID)),
            patch.object(main, "_write_env_var", side_effect=OSError("disk full")),
        ):
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp_good"})
        assert resp.status_code == 200
        warning = resp.json()["warning"]
        assert "Failed writing GITHUB_TOKEN" in warning
        assert "Connected" not in warning
        assert resp.json()["github_token_set"] is False


class TestPutKeyFormat:
    def test_surrounding_whitespace_is_trimmed(self, client):
        test_client, env_path = client
        resp = test_client.put("/settings/api-keys", json={"brave_api_key": "  brave-new\n"})
        assert resp.status_code == 200
        assert _env_file(env_path) == "BRAVE_API_KEY=brave-new\n"

    @pytest.mark.parametrize("bad_value", [
        "abc\nLOCALIST_MEMORY_DB=/tmp/evil",
        "abc def",
        "abc\x00def",
        "abcdéf",
        "   ",
    ])
    def test_malformed_key_changes_nothing(self, client, bad_value):
        test_client, env_path = client
        env_path.write_text("BRAVE_API_KEY=current\n")
        resp = test_client.put(
            "/settings/api-keys",
            json={"newsapi_api_key": "news-new", "brave_api_key": bad_value},
        )
        assert resp.status_code == 400
        assert "BRAVE_API_KEY" in resp.json()["detail"]
        assert _env_file(env_path) == "BRAVE_API_KEY=current\n"
        assert "NEWSAPI_API_KEY" not in os.environ

    def test_malformed_github_token_never_reaches_github(self, client):
        test_client, _ = client
        with patch.object(main.github_watch, "validate_token", AsyncMock()) as check:
            resp = test_client.put("/settings/api-keys", json={"github_token": "ghp bad"})
        assert resp.status_code == 400
        check.assert_not_called()


class TestMcpReloadAfterSave:
    def test_successful_save_reloads_and_needs_no_restart_note(self, client):
        test_client, _ = client
        resp = test_client.put("/settings/api-keys", json={"brave_api_key": "brave-new"})
        main._reload_mcp_keys.assert_awaited_once()
        assert resp.json()["warning"] is None

    def test_reload_failure_says_how_to_restart(self, client):
        test_client, _ = client
        main._reload_mcp_keys.return_value = False
        resp = test_client.put("/settings/api-keys", json={"brave_api_key": "brave-new"})
        assert resp.status_code == 200
        assert "./start_localist.sh" in resp.json()["warning"]

    def test_reload_failure_on_desktop_never_mentions_the_script(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setattr(main.sys, "frozen", True, raising=False)
        main._reload_mcp_keys.return_value = False
        resp = test_client.put("/settings/api-keys", json={"brave_api_key": "brave-new"})
        assert "Quit and reopen Localist" in resp.json()["warning"]
        assert "start_localist" not in resp.json()["warning"]

    def test_refused_save_does_not_reload(self, client):
        test_client, _ = client
        with patch.object(main.github_watch, "validate_token", AsyncMock(return_value=_REJECTED)):
            test_client.put("/settings/api-keys", json={"github_token": "ghp_bad"})
        main._reload_mcp_keys.assert_not_called()

    def test_nothing_written_does_not_reload(self, client):
        test_client, _ = client
        with patch.object(main, "_write_env_var", side_effect=OSError("disk full")):
            resp = test_client.put("/settings/api-keys", json={"brave_api_key": "brave-new"})
        main._reload_mcp_keys.assert_not_called()
        assert "Failed writing BRAVE_API_KEY" in resp.json()["warning"]

    def test_partial_write_still_reloads(self, client):
        test_client, _ = client
        real_write = main._write_env_var

        def fail_brave(root, key, value):
            if key == "BRAVE_API_KEY":
                raise OSError("disk full")
            real_write(root, key, value)

        with patch.object(main, "_write_env_var", side_effect=fail_brave):
            resp = test_client.put(
                "/settings/api-keys",
                json={"brave_api_key": "brave-new", "newsapi_api_key": "news-new"},
            )
        main._reload_mcp_keys.assert_awaited_once()
        assert "Failed writing BRAVE_API_KEY" in resp.json()["warning"]


class TestReloadMcpKeys:
    def _response(self, status_code):
        return httpx.Response(
            status_code, json={},
            request=httpx.Request("POST", "http://localhost:8003/internal/reload-keys"),
        )

    def test_targets_reload_route_not_sse_endpoint(self):
        post = AsyncMock(return_value=self._response(200))
        with patch.object(httpx.AsyncClient, "post", post):
            assert asyncio.run(main._reload_mcp_keys()) is True
        url = post.await_args.args[0]
        assert url.endswith("/internal/reload-keys")
        assert "/sse" not in url

    def test_http_error_returns_false(self):
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=self._response(500))):
            assert asyncio.run(main._reload_mcp_keys()) is False

    def test_server_down_returns_false(self):
        down = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch.object(httpx.AsyncClient, "post", down):
            assert asyncio.run(main._reload_mcp_keys()) is False


class TestMcpReloadRoute:
    @pytest.fixture()
    def mcp_client(self, tmp_path, monkeypatch):
        for name in _ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(mcp_main, "get_backend_root", lambda: tmp_path)
        yield TestClient(mcp_main.app), tmp_path / ".env"

    def test_new_value_replaces_old_and_response_has_no_values(self, mcp_client, monkeypatch):
        test_client, env_path = mcp_client
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_old")
        env_path.write_text("GITHUB_TOKEN=ghp_new\n")
        resp = test_client.post("/internal/reload-keys")
        assert resp.status_code == 200
        assert resp.json() == {"reloaded": ["GITHUB_TOKEN"]}
        assert "ghp_new" not in resp.text
        assert os.environ["GITHUB_TOKEN"] == "ghp_new"

    def test_cleared_key_becomes_empty(self, mcp_client, monkeypatch):
        test_client, env_path = mcp_client
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_old")
        env_path.write_text("GITHUB_TOKEN=\n")
        test_client.post("/internal/reload-keys")
        assert os.environ["GITHUB_TOKEN"] == ""

    def test_key_missing_from_file_is_left_alone(self, mcp_client, monkeypatch):
        test_client, env_path = mcp_client
        monkeypatch.setenv("BRAVE_API_KEY", "from-shell")
        env_path.write_text("GITHUB_TOKEN=ghp_new\n")
        test_client.post("/internal/reload-keys")
        assert os.environ["BRAVE_API_KEY"] == "from-shell"

    def test_other_env_values_are_never_loaded(self, mcp_client, monkeypatch):
        test_client, env_path = mcp_client
        monkeypatch.delenv("LOCALIST_LOG_LEVEL", raising=False)
        env_path.write_text("LOCALIST_LOG_LEVEL=DEBUG\nGITHUB_TOKEN=ghp_new\n")
        test_client.post("/internal/reload-keys")
        assert "LOCALIST_LOG_LEVEL" not in os.environ

    def test_missing_env_file_is_a_no_op(self, mcp_client):
        test_client, _ = mcp_client
        resp = test_client.post("/internal/reload-keys")
        assert resp.status_code == 200
        assert resp.json() == {"reloaded": []}

    def test_covers_every_settings_key(self):
        assert set(mcp_main._RELOADABLE_KEY_NAMES) == set(main._API_KEY_ENV_NAMES.values())
