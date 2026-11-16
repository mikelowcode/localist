"""
Settings → Wikipedia (§14.21.8): GET/PUT /settings/wikipedia.

  W-1  GET reports both flags the way their consumers read them
       (enabled: on unless explicitly false; enrich: off unless explicitly true).
  W-2  PUT writes .env (other lines untouched, existing key replaced in place)
       and os.environ, so the very next request follows it — no restart.
  W-3  Partial PUTs touch only the supplied flag; enrich is kept while
       Wikipedia is off, and only acts when both are on.
  W-4  Bad input is rejected and nothing is written.

.env location is redirected to tmp_path; os.environ changes are undone.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist.mcp_tool_dispatcher import _wikipedia_enrich_enabled, _wikipedia_globally_enabled
from localist.planner import Planner

ENABLED = "LOCALIST_WIKIPEDIA_ENABLED"
ENRICH = "LOCALIST_WIKIPEDIA_ENRICH"


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    # setenv first so monkeypatch records the original (absent) state and
    # removes whatever PUT writes into os.environ at teardown.
    for var in (ENABLED, ENRICH):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    return TestClient(main.app), tmp_path


def _env(tmp_path) -> str:
    p = tmp_path / ".env"
    return p.read_text() if p.exists() else ""


def _routes_to_wikipedia() -> bool:
    rt = MagicMock()
    rt.infer.return_value = "no"
    rt.embed.return_value = [0.0] * 768
    plan = Planner(runtime=rt).route("what does wikipedia say about the transistor", context={})
    return plan.tools_to_call == ["wikipedia_search"]


# W-1 -----------------------------------------------------------------------

def test_get_defaults(api):
    client, _ = api
    assert client.get("/settings/wikipedia").json() == {"enabled": True, "enrich": False}


@pytest.mark.parametrize("enabled_raw, enrich_raw, expected", [
    ("off",   "yes",   {"enabled": False, "enrich": True}),
    ("FALSE", "1",     {"enabled": False, "enrich": True}),
    ("true",  "false", {"enabled": True,  "enrich": False}),
    ("junk",  "junk",  {"enabled": True,  "enrich": False}),
])
def test_get_reads_env_like_consumers(api, monkeypatch, enabled_raw, enrich_raw, expected):
    client, _ = api
    monkeypatch.setenv(ENABLED, enabled_raw)
    monkeypatch.setenv(ENRICH, enrich_raw)
    assert client.get("/settings/wikipedia").json() == expected
    assert _wikipedia_globally_enabled() is expected["enabled"]


# W-2 -----------------------------------------------------------------------

def test_put_writes_env_and_keeps_other_lines(api):
    client, tmp_path = api
    (tmp_path / ".env").write_text(
        "# my settings\nBRAVE_API_KEY=test-placeholder\nLOCALIST_WIKIPEDIA_ENRICH=false\n"
    )
    body = client.put("/settings/wikipedia", json={"enrich": True}).json()

    assert body == {"enabled": True, "enrich": True}
    env = _env(tmp_path)
    assert env == "# my settings\nBRAVE_API_KEY=test-placeholder\nLOCALIST_WIKIPEDIA_ENRICH=true\n"
    assert main.os.environ[ENRICH] == "true"


def test_put_appends_missing_key(api):
    client, tmp_path = api
    client.put("/settings/wikipedia", json={"enabled": False, "enrich": True})
    env = _env(tmp_path)
    assert "LOCALIST_WIKIPEDIA_ENABLED=false\n" in env
    assert "LOCALIST_WIKIPEDIA_ENRICH=true\n" in env


def test_enabled_switch_changes_routing_on_the_next_request(api):
    client, _ = api
    assert _routes_to_wikipedia() is True

    client.put("/settings/wikipedia", json={"enabled": False})
    assert _routes_to_wikipedia() is False

    client.put("/settings/wikipedia", json={"enabled": True})
    assert _routes_to_wikipedia() is True


def test_enrich_switch_changes_dispatcher_on_the_next_request(api):
    client, _ = api
    assert _wikipedia_enrich_enabled() is False
    client.put("/settings/wikipedia", json={"enrich": True})
    assert _wikipedia_enrich_enabled() is True
    client.put("/settings/wikipedia", json={"enrich": False})
    assert _wikipedia_enrich_enabled() is False


# W-3 -----------------------------------------------------------------------

def test_partial_put_leaves_other_flag_alone(api):
    client, tmp_path = api
    client.put("/settings/wikipedia", json={"enabled": False})
    env = _env(tmp_path)
    assert "LOCALIST_WIKIPEDIA_ENABLED=false" in env
    assert ENRICH not in env
    assert ENRICH not in main.os.environ


def test_enrich_kept_while_wikipedia_off_and_inactive(api):
    client, _ = api
    body = client.put("/settings/wikipedia", json={"enabled": False, "enrich": True}).json()
    assert body == {"enabled": False, "enrich": True}
    assert _wikipedia_enrich_enabled() is False          # master switch wins

    body = client.put("/settings/wikipedia", json={"enabled": True}).json()
    assert body == {"enabled": True, "enrich": True}
    assert _wikipedia_enrich_enabled() is True           # restored, not lost


def test_empty_put_changes_nothing(api):
    client, tmp_path = api
    body = client.put("/settings/wikipedia", json={}).json()
    assert body == {"enabled": True, "enrich": False}
    assert _env(tmp_path) == ""


# W-4 -----------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"enabled": "yes"}, {"enrich": 1}, {"enabled": "false"}, {"enrich": None, "enabled": "on"},
])
def test_non_boolean_rejected_and_nothing_written(api, payload):
    client, tmp_path = api
    resp = client.put("/settings/wikipedia", json=payload)
    assert resp.status_code == 422
    assert _env(tmp_path) == ""
    assert ENABLED not in main.os.environ and ENRICH not in main.os.environ
