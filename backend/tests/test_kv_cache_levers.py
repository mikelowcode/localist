"""
KV-cache levers (2026-11-12, docs/architecture/03-unified-prompt-contract.md
§3.7e and 16-runtime-backend-layer.md §16.36):

  KV-1  Persona ceiling is 1,200 tokens: a persona that fills one 1,024-token
        cache block (~1,100 tokens) reaches the system message untruncated.
  KV-2  Ollama keep_alive: "off" (default) sends no keep_alive field; any
        other choice sends Ollama's value; invalid choices are rejected;
        the live attribute is read per request.
  KV-3  runtime_factory falls back to "off" on an invalid .env value instead
        of failing startup.
  KV-4  warmup.prefix_cache_status(): block-based probe -> full blocks;
        token-level (local Ollama) -> whole prefix; no probe / mock runtime ->
        "unknown"; warning logged when a block engine can cache nothing.
  KV-5  run_cache_warmup() returns the status after a successful call, None
        when skipped or failed.
  KV-6  GET/PUT /settings/ollama-keep-alive and GET /settings/prefix-cache.
  KV-7  OMLXRuntimeClient.probe_prefix_cache() subtracts the user turn and
        never raises.

All inference and HTTP is mocked; no live engine needed.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import main
from localist import ollama_runtime_client as _ollama_mod
from localist import omlx_runtime_client as _omlx_mod
from localist.ollama_runtime_client import (
    KEEP_ALIVE_CHOICES, OllamaRuntimeClient, normalize_keep_alive,
)
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.prompt_builder import PromptBuilder
from localist.runtime_factory import create_runtime
from localist.warmup import prefix_cache_status, run_cache_warmup


def _ndjson_response():
    class _Resp:
        status_code = 200

        def iter_lines(self, decode_unicode: bool = True):
            yield json.dumps({"message": {"content": "hi"}, "done": True})

    return _Resp()


def _sent_payload(client: OllamaRuntimeClient) -> dict:
    with patch.object(_ollama_mod.requests, "post", return_value=_ndjson_response()) as mock_post:
        list(client.infer_stream("prompt"))
    return json.loads(mock_post.call_args.kwargs["data"])


# ---------------------------------------------------------------------------
# KV-1  persona ceiling
# ---------------------------------------------------------------------------

def test_kv1_one_block_persona_is_not_truncated():
    pb = PromptBuilder()
    persona = "word " * 880   # 4,400 chars ≈ 1,100 tokens: fills a 1,024-token block
    system_prompt, _ = pb.build(instruction="x", persona=persona.strip(),
                                current_datetime=__import__("datetime").datetime(2026, 11, 12))
    assert "… [truncated]" not in system_prompt
    assert system_prompt.endswith(persona.strip())


def test_kv1_ceiling_value():
    assert PromptBuilder._CEIL_PERSONA == 1200


def test_kv1_load_persona_keeps_a_one_block_persona():
    """Regression (found in the Phase 3 live check): _load_persona() had its
    own fixed [:2000] slice, which cut every persona back to ~500 tokens
    before PromptBuilder's raised ceiling could apply."""
    from pathlib import Path
    from localist.controller_agent import ControllerAgent

    body = "You are {{ASSISTANT_NAME}}.\n" + ("Stable persona guidance line. " * 150)  # ~4,500 chars
    doc = MagicMock()
    doc.path = Path("/wiki/persona.md")
    doc.content = body
    mm = MagicMock()
    mm.get_assistant_name.return_value = "Localist"
    mm.query_corpus.return_value = [doc]
    runtime = MagicMock()
    runtime.is_local = True

    persona = ControllerAgent(runtime=runtime, agents=[], memory_manager=mm)._load_persona()

    assert len(persona) > 4000
    assert persona == body.replace("{{ASSISTANT_NAME}}", "Localist")[:4800]


# ---------------------------------------------------------------------------
# KV-2  Ollama keep_alive in the request payload
# ---------------------------------------------------------------------------

def test_kv2_off_by_default_sends_no_keep_alive():
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    assert client.keep_alive == "off"
    assert "keep_alive" not in _sent_payload(client)


@pytest.mark.parametrize("choice,expected", [
    ("15m", "15m"), ("30m", "30m"), ("1h", "1h"), ("4h", "4h"), ("always", -1),
])
def test_kv2_choice_sent_as_ollama_value(choice, expected):
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx", keep_alive=choice)
    assert _sent_payload(client)["keep_alive"] == expected


def test_kv2_live_attribute_change_applies_to_next_request():
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    client.keep_alive = "30m"
    assert _sent_payload(client)["keep_alive"] == "30m"
    client.keep_alive = "off"
    assert "keep_alive" not in _sent_payload(client)


def test_kv2_infer_carries_keep_alive_too():
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx", keep_alive="1h")
    with patch.object(_ollama_mod.requests, "post", return_value=_ndjson_response()) as mock_post:
        client.infer("prompt")
    assert json.loads(mock_post.call_args.kwargs["data"])["keep_alive"] == "1h"


@pytest.mark.parametrize("raw,expected", [
    (None, "off"), ("", "off"), ("OFF", "off"), (" 30m ", "30m"), ("always", "always"),
])
def test_kv2_normalize_accepts(raw, expected):
    assert normalize_keep_alive(raw) == expected


@pytest.mark.parametrize("raw", ["5m", "forever", "-1", "yes"])
def test_kv2_normalize_rejects(raw):
    with pytest.raises(ValueError):
        normalize_keep_alive(raw)


def test_kv2_constructor_rejects_invalid_choice():
    with pytest.raises(ValueError):
        OllamaRuntimeClient(chat_model="gemma4:12b-mlx", keep_alive="forever")


# ---------------------------------------------------------------------------
# KV-3  runtime_factory fallback
# ---------------------------------------------------------------------------

def test_kv3_factory_passes_keep_alive():
    rt = create_runtime("ollama", chat_model="gemma4:12b-mlx", ollama_keep_alive="4h")
    assert rt.keep_alive == "4h"


def test_kv3_factory_invalid_value_falls_back_to_off(caplog):
    with caplog.at_level(logging.WARNING):
        rt = create_runtime("ollama", chat_model="gemma4:12b-mlx", ollama_keep_alive="forever")
    assert rt.keep_alive == "off"
    assert "keep-alive" in caplog.text


def test_kv3_factory_default_is_off():
    rt = create_runtime("ollama", chat_model="gemma4:12b-mlx")
    assert rt.keep_alive == "off"


# ---------------------------------------------------------------------------
# KV-4  prefix_cache_status
# ---------------------------------------------------------------------------

def _probing_runtime(prefix_tokens: int, block_size: int):
    return SimpleNamespace(
        probe_prefix_cache=lambda system: {"prefix_tokens": prefix_tokens, "block_size": block_size},
    )


def test_kv4_block_engine_under_one_block_caches_nothing():
    status = prefix_cache_status(_probing_runtime(582, 1024), "sys")
    assert status["granularity"] == "block"
    assert status["full_blocks"] == 0
    assert status["cached_tokens"] == 0
    assert status["measured"] is True


def test_kv4_block_engine_one_full_block():
    status = prefix_cache_status(_probing_runtime(1084, 1024), "sys")
    assert (status["full_blocks"], status["cached_tokens"]) == (1, 1024)


def test_kv4_block_engine_smaller_blocks():
    status = prefix_cache_status(_probing_runtime(582, 512), "sys")
    assert (status["full_blocks"], status["cached_tokens"]) == (1, 512)


def test_kv4_local_ollama_is_token_level():
    rt = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    status = prefix_cache_status(rt, "x" * 400)
    assert status["granularity"] == "token"
    assert status["cached_tokens"] == status["prefix_tokens"] == 100
    assert status["measured"] is False


def test_kv4_cloud_ollama_is_unknown():
    rt = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
    assert prefix_cache_status(rt, "sys")["granularity"] == "unknown"


def test_kv4_probe_returning_none_falls_back_to_estimate():
    rt = SimpleNamespace(probe_prefix_cache=lambda system: None)
    status = prefix_cache_status(rt, "x" * 40)
    assert status["granularity"] == "unknown"
    assert status["prefix_tokens"] == 10


def test_kv4_probe_raising_is_contained():
    def boom(system):
        raise RuntimeError("probe down")
    status = prefix_cache_status(SimpleNamespace(probe_prefix_cache=boom), "sys")
    assert status["granularity"] == "unknown"


def test_kv4_mock_runtime_is_treated_as_unknown():
    """A MagicMock's auto-attributes must not be mistaken for probe data."""
    status = prefix_cache_status(MagicMock(), "sys")
    assert status["granularity"] == "unknown"
    assert status["block_size"] is None


# ---------------------------------------------------------------------------
# KV-5  run_cache_warmup returns + logs the status
# ---------------------------------------------------------------------------

_FIXTURE = """# fixture

## TOOL RESULTS

tool: web_search
params: "q"

Some result.

## WORKING MEMORY

role: user

hello
"""


def _templates(tmp_path):
    (tmp_path / "warmup_fixture.md").write_text(_FIXTURE, encoding="utf-8")
    return tmp_path


def _controller():
    ctrl = MagicMock()
    ctrl._load_persona.return_value = "persona text"
    ctrl._current_assistant_name.return_value = "Localist"
    return ctrl


def test_kv5_returns_status_and_warns_when_nothing_cacheable(tmp_path, caplog):
    rt = MagicMock()
    rt.infer.return_value = "ok"
    rt.probe_prefix_cache = lambda system: {"prefix_tokens": 582, "block_size": 1024}
    with caplog.at_level(logging.WARNING, logger="localist.warmup"):
        status = run_cache_warmup(_controller(), rt, _templates(tmp_path))
    assert status["full_blocks"] == 0
    assert "can't cache any of it" in caplog.text


def test_kv5_probe_sees_the_exact_warmed_system_prompt(tmp_path):
    seen = {}
    rt = MagicMock()
    rt.infer.return_value = "ok"
    rt.probe_prefix_cache = lambda system: seen.setdefault("system", system) and None
    run_cache_warmup(_controller(), rt, _templates(tmp_path))
    assert seen["system"] == rt.infer.call_args.kwargs["system"]


def test_kv5_returns_none_when_call_fails(tmp_path):
    rt = MagicMock()
    rt.infer.side_effect = RuntimeError("down")
    assert run_cache_warmup(_controller(), rt, _templates(tmp_path)) is None


def test_kv5_returns_none_when_fixture_missing(tmp_path):
    assert run_cache_warmup(_controller(), MagicMock(), tmp_path) is None


# ---------------------------------------------------------------------------
# KV-6  endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_PROJECT_ROOT", tmp_path)
    prev = (main._state.settings, main._state.runtime, main._state.controller)
    main._state.settings = main.Settings(runtime_backend="ollama", ollama_keep_alive="off")
    main._state.runtime = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    main._state.controller = MagicMock(prefix_cache_status=None)
    yield TestClient(main.app), tmp_path
    main._state.settings, main._state.runtime, main._state.controller = prev


def test_kv6_get_keep_alive_default(api):
    client, _ = api
    body = client.get("/settings/ollama-keep-alive").json()
    assert body["keep_alive"] == "off"
    assert body["choices"] == list(KEEP_ALIVE_CHOICES)


def test_kv6_put_keep_alive_persists_and_applies_live(api):
    client, root = api
    body = client.put("/settings/ollama-keep-alive", json={"keep_alive": "30m"}).json()
    assert body["keep_alive"] == "30m"
    assert body["applied_live"] is True
    assert main._state.runtime.keep_alive == "30m"
    assert main._state.settings.ollama_keep_alive == "30m"
    assert "LOCALIST_OLLAMA_KEEP_ALIVE=30m" in (root / ".env").read_text()


def test_kv6_put_keep_alive_rejects_invalid(api):
    client, root = api
    resp = client.put("/settings/ollama-keep-alive", json={"keep_alive": "forever"})
    assert resp.status_code == 400
    assert main._state.runtime.keep_alive == "off"
    assert not (root / ".env").exists()


def test_kv6_put_keep_alive_on_other_backend_saves_only(api):
    client, _ = api
    main._state.runtime = MagicMock(name="omlx-runtime")
    body = client.put("/settings/ollama-keep-alive", json={"keep_alive": "1h"}).json()
    assert body["applied_live"] is False
    assert main._state.settings.ollama_keep_alive == "1h"


def test_kv6_prefix_cache_unavailable_without_warmup(api):
    client, _ = api
    assert client.get("/settings/prefix-cache").json()["available"] is False


def test_kv6_prefix_cache_reports_controller_status(api):
    client, _ = api
    main._state.controller = MagicMock(prefix_cache_status={
        "granularity": "block", "prefix_tokens": 582, "block_size": 1024,
        "full_blocks": 0, "cached_tokens": 0, "measured": True,
    })
    body = client.get("/settings/prefix-cache").json()
    assert body["available"] is True
    assert (body["block_size"], body["full_blocks"]) == (1024, 0)
    assert body["persona_ceiling_tokens"] == 1200


# ---------------------------------------------------------------------------
# KV-7  OMLXRuntimeClient.probe_prefix_cache
# ---------------------------------------------------------------------------

def _probe_response(total_tokens, block_size=1024):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"total_tokens": total_tokens, "block_size": block_size}
    return resp


def test_kv7_probe_subtracts_user_turn():
    client = OMLXRuntimeClient(chat_model="gemma-4-12B-it-MLX-4bit")
    with patch.object(_omlx_mod.requests, "post",
                      side_effect=[_probe_response(590), _probe_response(9)]) as mock_post:
        info = client.probe_prefix_cache("system text")
    # 590 total - 9 (BOS + user turn) + 1 (BOS is shared) = 582
    assert info == {"prefix_tokens": 582, "block_size": 1024}
    first = mock_post.call_args_list[0].kwargs["json"]
    assert first["model_id"] == "gemma-4-12B-it-MLX-4bit"
    assert first["messages"][0] == {"role": "system", "content": "system text"}


def test_kv7_probe_failure_returns_none():
    client = OMLXRuntimeClient(chat_model="gemma-4-12B-it-MLX-4bit")
    with patch.object(_omlx_mod.requests, "post", side_effect=ConnectionError("down")):
        assert client.probe_prefix_cache("s") is None


def test_kv7_probe_model_not_loaded_returns_none():
    """oMLX answers 200 with no token counts when the model isn't loaded."""
    client = OMLXRuntimeClient(chat_model="gemma-4-12B-it-MLX-4bit")
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"model_loaded": False, "reason": "Model is not loaded"}
    with patch.object(_omlx_mod.requests, "post", return_value=resp):
        assert client.probe_prefix_cache("s") is None


# ---------------------------------------------------------------------------
# KV-8  switching back to "off" resets a loaded model's idle timer
#       (Ollama keeps the last-set keep_alive when a request omits it —
#       found in the Phase 3 live check)
# ---------------------------------------------------------------------------

def _ps_response(names):
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": n, "model": n} for n in names]}
    return resp


def test_kv8_reset_sends_default_when_model_loaded():
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    with patch.object(_ollama_mod.requests, "get", return_value=_ps_response(["gemma4:12b-mlx"])), \
         patch.object(_ollama_mod.requests, "post") as mock_post:
        assert client.reset_keep_alive_to_default() is True
    assert mock_post.call_args.kwargs["json"] == {"model": "gemma4:12b-mlx", "keep_alive": "5m"}
    assert mock_post.call_args.args[0].endswith("/api/generate")


def test_kv8_reset_never_loads_an_unloaded_model():
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    with patch.object(_ollama_mod.requests, "get", return_value=_ps_response(["nomic-embed-text:latest"])), \
         patch.object(_ollama_mod.requests, "post") as mock_post:
        assert client.reset_keep_alive_to_default() is False
    mock_post.assert_not_called()


def test_kv8_reset_skips_cloud_models_and_never_raises():
    assert OllamaRuntimeClient(chat_model="gemma4:31b-cloud").reset_keep_alive_to_default() is False
    client = OllamaRuntimeClient(chat_model="gemma4:12b-mlx")
    with patch.object(_ollama_mod.requests, "get", side_effect=ConnectionError("down")):
        assert client.reset_keep_alive_to_default() is False


def test_kv8_endpoint_resets_only_on_transition_to_off(api):
    client, _ = api
    calls = []
    main._state.runtime.reset_keep_alive_to_default = lambda: calls.append(1) or True
    client.put("/settings/ollama-keep-alive", json={"keep_alive": "off"})   # off -> off
    assert calls == []
    client.put("/settings/ollama-keep-alive", json={"keep_alive": "4h"})    # off -> 4h
    assert calls == []
    client.put("/settings/ollama-keep-alive", json={"keep_alive": "off"})   # 4h -> off
    assert calls == [1]
