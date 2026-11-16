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


# ---------------------------------------------------------------------------
# §14.21.9 — "Wikipedia not consulted" prompt note
# ---------------------------------------------------------------------------

from localist.controller_agent import (  # noqa: E402
    _WIKIPEDIA_NOT_CONSULTED, _WIKIPEDIA_TURNED_OFF, _wikipedia_not_consulted,
)
from localist.prompt_builder import PromptBuilder, ToolResult  # noqa: E402


def _tr(name, result, success=True):
    return ToolResult(tool_name=name, parameters="", result=result, success=success)


class TestWikipediaNotConsultedRule:

    @pytest.fixture(autouse=True)
    def _wikipedia_on(self, monkeypatch):
        monkeypatch.delenv(ENABLED, raising=False)

    @pytest.mark.parametrize("value", ["false", "off", "0"])
    def test_turned_off_wording_when_switched_off(self, monkeypatch, value):
        monkeypatch.setenv(ENABLED, value)
        note = _wikipedia_not_consulted("what does wikipedia say about x", [])
        assert note == [_WIKIPEDIA_TURNED_OFF]
        assert "turned off in Settings" in note[0].result
        assert "failed" not in note[0].result.lower()

    @pytest.mark.parametrize("instruction", [
        "what does Wikipedia say about the transistor",
        "according to wikipedia, who was Ada Lovelace?",
        "WIKIPEDIA's take on CRISPR",
    ])
    def test_note_when_named_and_no_wikipedia_text(self, instruction):
        assert _wikipedia_not_consulted(instruction, []) == [_WIKIPEDIA_NOT_CONSULTED]
        assert _wikipedia_not_consulted(instruction, [_tr("web_search", "• hit — example.com\n  [https://example.com]")]) \
            == [_WIKIPEDIA_NOT_CONSULTED]

    @pytest.mark.parametrize("instruction", [
        "what does my wiki say about the transistor", "who was Ada Lovelace", "",
    ])
    def test_no_note_when_wikipedia_not_named(self, instruction):
        assert _wikipedia_not_consulted(instruction, []) == []

    @pytest.mark.parametrize("result", [
        _tr("wikipedia_search", "• Transistor — Wikipedia\n  [https://en.wikipedia.org/wiki/Transistor]"),
        _tr("wikipedia_read", "Wikipedia article: Transistor\n[https://en.wikipedia.org/wiki/Transistor]"),
        _tr("url_fetch", "Title: Transistor\nSource: https://en.wikipedia.org/wiki/Transistor\nVia: Wikipedia API"),
        _tr("web_search", "• Transistor - Wikipedia — en.wikipedia.org\n  [https://en.wikipedia.org/wiki/Transistor]"),
        _tr("wikipedia_summary", "Wikipedia background …\n[https://en.wikipedia.org/wiki/X]"),
    ])
    def test_no_note_when_wikipedia_text_present(self, result):
        assert _wikipedia_not_consulted("what does wikipedia say about it", [result]) == []

    def test_failed_wikipedia_result_still_gets_note(self):
        miss = _tr("wikipedia_search", "ERROR: no Wikipedia article matched 'x'.", success=False)
        assert _wikipedia_not_consulted("what does wikipedia say about x", [miss]) == [_WIKIPEDIA_NOT_CONSULTED]

    def test_renders_in_tool_failed_slot(self):
        from datetime import datetime
        _, user_prompt = PromptBuilder().build(
            instruction="what does wikipedia say about the transistor",
            current_datetime=datetime.now().astimezone(),
            tool_failures=[_WIKIPEDIA_NOT_CONSULTED],
        )
        assert "[TOOL FAILED]" in user_prompt
        assert "wikipedia(): FAILED — no Wikipedia text was provided for this reply." in user_prompt
        assert "Do not say or imply that Wikipedia says" in user_prompt
