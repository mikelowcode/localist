"""
OpenRouterRuntimeClient (docs/architecture/16-runtime-backend-layer.md §16.33)
and the shared OpenAI-compatible helpers (openai_compat.py).

HTTP is faked at the module's `requests.post` / `requests.get`. Error bodies
and stream lines mirror what the live API returned in Phase 0
(diagnostics/reports/openrouter_free_tier_2026-11-10.md).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from localist import openrouter_runtime_client as orc
from localist.base_runtime_client import BaseRuntimeClient
from localist.openai_compat import build_messages, iter_content_deltas, iter_sse_lines
from localist.openrouter_runtime_client import OpenRouterError, OpenRouterRuntimeClient

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status: int = 200, lines: list[bytes] | None = None,
                 body: dict | None = None, headers: dict | None = None) -> None:
        self.status_code = status
        self._lines = lines or []
        self._body = body
        self.headers = headers or {}
        self.closed = False

    @property
    def text(self) -> str:
        return json.dumps(self._body) if self._body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body

    def iter_lines(self, decode_unicode: bool = False):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


def _event(content: str = "", **extra) -> bytes:
    body = {"choices": [{"delta": {"content": content}}], **extra}
    return f"data: {json.dumps(body, ensure_ascii=False)}".encode("utf-8")


def _stream(*chunks: str, usage: dict | None = None) -> FakeResponse:
    lines = [b": OPENROUTER PROCESSING", b""]
    lines += [_event(c) for c in chunks]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}".encode())
    lines.append(b"data: [DONE]")
    return FakeResponse(200, lines)


def _error(status: int, message: str, **metadata) -> FakeResponse:
    error = {"message": message, "code": status}
    if metadata:
        error["metadata"] = metadata
    return FakeResponse(status, body={"error": error})


UPSTREAM_429 = dict(
    message="Provider returned error",
    raw=f"{MODEL} is temporarily rate-limited upstream.",
    provider_name="Nvidia",
    limit_source="upstream_provider_shared_pool",
)


@pytest.fixture()
def client():
    return OpenRouterRuntimeClient(chat_model=MODEL, api_key="test-openrouter-key-test")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(orc.time, "sleep", sleeps.append)
    return sleeps


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class TestOpenAICompat:
    def test_bytes_are_decoded_as_utf8(self):
        # The Phase 0 bug: requests' ISO-8859-1 fallback turned "–" into "â\x80\x93".
        resp = FakeResponse(200, [_event("6–8 hours"), b"data: [DONE]"])
        assert list(iter_content_deltas(resp)) == ["6–8 hours"]

    def test_str_lines_pass_through(self):
        resp = FakeResponse(200, ["data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]})])
        assert list(iter_sse_lines(resp)) == [resp._lines[0]]
        assert list(iter_content_deltas(resp)) == ["hi"]

    def test_comments_malformed_and_done(self):
        resp = FakeResponse(200, [b": keep-alive", b"data: {not json", _event("a"),
                                  b"data: [DONE]", _event("after-done")])
        assert list(iter_content_deltas(resp)) == ["a"]

    def test_build_messages_maps_agent_role(self):
        class T:
            def __init__(self, role, content):
                self.role, self.content = role, content
        msgs = build_messages("now?", "sys", [T("user", "q1"), T("agent", "a1")])
        assert msgs == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "now?"},
        ]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class TestContract:
    def test_satisfies_the_protocol(self, client):
        assert isinstance(client, BaseRuntimeClient)
        assert client.is_local is False

    def test_embed_never_goes_remote(self, client):
        with patch.object(orc.requests, "post") as post:
            with pytest.raises(RuntimeError, match="chat-only"):
                client.embed("text")
        post.assert_not_called()

    def test_errors_are_runtime_errors(self):
        assert issubclass(OpenRouterError, RuntimeError)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_request_shape(self, client):
        with patch.object(orc.requests, "post", return_value=_stream("ok")) as post:
            client.infer("hello", system="be brief", max_tokens=50, temperature=0.1)

        kwargs = post.call_args.kwargs
        payload = json.loads(kwargs["data"])
        assert post.call_args.args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert payload == {
            "model": MODEL,
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hello"}],
            "stream": True, "max_tokens": 50, "temperature": 0.1,
            "reasoning": {"enabled": False},
        }
        assert kwargs["headers"]["Authorization"] == "Bearer test-openrouter-key-test"
        assert kwargs["headers"]["X-Title"] == "Localist"
        assert kwargs["stream"] is True

    def test_yields_deltas_and_records_usage(self, client):
        usage = {"prompt_tokens": 31, "completion_tokens": 3, "cost": 0}
        with patch.object(orc.requests, "post", return_value=_stream("To", "ma", "toes", usage=usage)):
            chunks = list(client.infer_stream("q"))
        assert chunks == ["To", "ma", "toes"]
        assert client.last_usage == usage
        assert client.last_error is None

    def test_timeout_override(self, client):
        with patch.object(orc.requests, "post", return_value=_stream("x")) as post:
            client.infer("q", timeout=5.0)
        assert post.call_args.kwargs["timeout"] == 5.0

    def test_mid_stream_error(self, client):
        resp = FakeResponse(200, [_event("par"),
                                  b'data: {"error": {"message": "provider disconnected"}}'])
        with patch.object(orc.requests, "post", return_value=resp):
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert info.value.kind == "stream"
        assert "provider disconnected" in str(info.value)
        assert resp.closed

    def test_success_clears_the_last_error(self, client):
        with patch.object(orc.requests, "post", return_value=_error(401, "User not found.")):
            with pytest.raises(OpenRouterError):
                client.infer("q")
        assert client.last_error["kind"] == "auth"
        with patch.object(orc.requests, "post", return_value=_stream("ok")):
            client.infer("q")
        assert client.last_error is None


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_no_key_fails_fast(self):
        client = OpenRouterRuntimeClient(chat_model=MODEL, api_key="  ")
        with patch.object(orc.requests, "post") as post:
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert info.value.kind == "no_key"
        post.assert_not_called()

    def test_upstream_busy_is_retried_once_then_succeeds(self, client, no_sleep):
        responses = [_error(429, **UPSTREAM_429), _stream("ok")]
        with patch.object(orc.requests, "post", side_effect=responses) as post:
            assert client.infer("q") == "ok"
        assert post.call_count == 2
        assert no_sleep == [orc._UPSTREAM_RETRY_WAIT_S]

    def test_upstream_busy_twice_raises_upstream_busy(self, client):
        responses = [_error(429, **UPSTREAM_429), _error(429, **UPSTREAM_429)]
        with patch.object(orc.requests, "post", side_effect=responses) as post:
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert post.call_count == 2
        assert info.value.kind == "upstream_busy"
        assert "busy at its provider" in info.value.message
        assert client.last_error["kind"] == "upstream_busy"

    def test_retry_after_is_honoured_and_capped(self, client, no_sleep):
        busy = _error(429, **UPSTREAM_429)
        busy.headers = {"Retry-After": "120"}
        with patch.object(orc.requests, "post", side_effect=[busy, _stream("ok")]):
            client.infer("q")
        assert no_sleep == [orc._MAX_RETRY_WAIT_S]

    def test_daily_limit_is_not_retried(self, client, no_sleep):
        resp = _error(429, "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock "
                           "1000 free model requests per day")
        with patch.object(orc.requests, "post", return_value=resp) as post:
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert post.call_count == 1
        assert no_sleep == []
        assert info.value.kind == "daily_limit"
        assert "1,000 a day" in info.value.message

    @pytest.mark.parametrize("status,message,kind", [
        (429, "Rate limit exceeded: free-models-per-min.", "minute_limit"),
        (429, "Too many requests", "rate_limited"),
        (401, "User not found.", "auth"),
        (403, "Key is disabled", "auth"),
        (402, "Insufficient credits", "credits"),
        (404, "No endpoints found matching your data policy (Free model training).", "no_endpoint"),
        (500, "Internal error", "other"),
    ])
    def test_error_kinds(self, client, status, message, kind):
        with patch.object(orc.requests, "post", return_value=_error(status, message)):
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert info.value.kind == kind
        assert info.value.status == status

    def test_connection_failure(self, client):
        with patch.object(orc.requests, "post", side_effect=requests.ConnectionError("dns")):
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert info.value.kind == "transport"

    def test_timeout(self, client):
        with patch.object(orc.requests, "post", side_effect=requests.Timeout()):
            with pytest.raises(OpenRouterError) as info:
                client.infer("q", timeout=3)
        assert info.value.kind == "transport"
        assert "3s" in info.value.message

    def test_non_json_error_body(self, client):
        resp = FakeResponse(502)
        with patch.object(orc.requests, "post", return_value=resp):
            with pytest.raises(OpenRouterError) as info:
                client.infer("q")
        assert info.value.kind == "other"


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------

KEY_DATA = {
    "label": "test-key-label", "creator_user_id": "user_x", "workspace_id": "ws",
    "limit": 10, "limit_remaining": 10, "limit_reset": "weekly", "usage": 0,
    "is_free_tier": True,
    "free_model_daily_requests": {"used": 24, "limit": 50, "remaining": 26},
}


def _gets(key: FakeResponse, endpoints: FakeResponse | None = None):
    def _get(url, **kwargs):
        if url.endswith("/key"):
            return key
        assert url.endswith(f"/models/{MODEL}/endpoints")
        return endpoints
    return _get


class TestHealthCheck:
    def test_no_key(self):
        client = OpenRouterRuntimeClient(chat_model=MODEL, api_key="")
        with patch.object(orc.requests, "get") as get:
            health = client.health_check()
        get.assert_not_called()
        assert health["reachable"] is False
        assert "No OpenRouter API key" in health["error"]

    def test_rejected_key_is_unreachable(self, client):
        with patch.object(orc.requests, "get", side_effect=_gets(_error(401, "User not found."))):
            health = client.health_check()
        assert health["reachable"] is False
        assert "rejected" in health["error"]

    def test_healthy(self, client):
        endpoints = FakeResponse(200, body={"data": {"endpoints": [{"provider_name": "Nvidia"}]}})
        with patch.object(orc.requests, "get", side_effect=_gets(FakeResponse(200, body={"data": KEY_DATA}), endpoints)):
            health = client.health_check()
        assert health["reachable"] is True
        assert health["chat_model_found"] is True
        assert health["models"] == [MODEL]
        assert health["embed_model_found"] is None
        assert health["key_info"]["free_requests_remaining"] == 26
        assert health["key_info"]["is_free_tier"] is True
        # Nothing identifying is passed on.
        flat = json.dumps(health)
        assert "user_x" not in flat and "test-key-label" not in flat and "ws" not in health["key_info"].values()

    def test_model_without_endpoints_is_not_found(self, client):
        endpoints = FakeResponse(404, body={"error": {"message": "Model not found"}})
        with patch.object(orc.requests, "get", side_effect=_gets(FakeResponse(200, body={"data": KEY_DATA}), endpoints)):
            health = client.health_check()
        assert health["reachable"] is True
        assert health["chat_model_found"] is False

    def test_network_failure_never_raises(self, client):
        with patch.object(orc.requests, "get", side_effect=requests.ConnectionError("offline")):
            health = client.health_check()
        assert health["reachable"] is False
        assert "offline" in health["error"]

    def test_reports_the_last_failure(self, client):
        with patch.object(orc.requests, "post", side_effect=[_error(429, **UPSTREAM_429)] * 2):
            with pytest.raises(OpenRouterError):
                client.infer("q")
        with patch.object(orc.requests, "get", side_effect=requests.ConnectionError("x")):
            assert client.health_check()["last_error"]["kind"] == "upstream_busy"


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------

CATALOG = {"data": [
    {"id": "z/paid-model", "name": "Z Paid", "supported_parameters": ["tools"],
     "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "context_length": 128000},
    {"id": MODEL, "name": "NVIDIA: Nemotron 3 Ultra (free)", "supported_parameters": ["tools", "reasoning"],
     "pricing": {"prompt": "0", "completion": "0"}, "context_length": 1000000},
    {"id": "a/no-tools:free", "name": "A No Tools", "supported_parameters": ["temperature"],
     "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "g/music-preview", "name": "Music", "supported_parameters": ["tools"],
     "architecture": {"output_modalities": ["audio"]}, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "b/also-free:free", "name": "B Free", "supported_parameters": ["tools"],
     "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "openrouter/free", "name": "Free Models Router", "supported_parameters": ["tools"],
     "pricing": {"prompt": "0", "completion": "0"}},
]}


class TestListModels:
    def test_filters_and_orders(self, client):
        with patch.object(orc.requests, "get", return_value=FakeResponse(200, body=CATALOG)):
            models = client.list_models()
        assert [m["id"] for m in models] == ["b/also-free:free", MODEL, "z/paid-model"]
        assert [m["free"] for m in models] == [True, True, False]

    def test_free_only_and_cache(self, client):
        with patch.object(orc.requests, "get", return_value=FakeResponse(200, body=CATALOG)) as get:
            client.list_models()
            free = client.list_models(free_only=True)
        assert get.call_count == 1
        assert [m["id"] for m in free] == ["b/also-free:free", MODEL]

    def test_failure_raises(self, client):
        with patch.object(orc.requests, "get", side_effect=requests.ConnectionError("x")):
            with pytest.raises(OpenRouterError):
                client.list_models()


# ---------------------------------------------------------------------------
# validate_api_key() / set_api_key() (Phase 3, §16.34)
# ---------------------------------------------------------------------------

class TestValidateApiKey:
    def test_valid(self):
        with patch.object(orc.requests, "get", return_value=FakeResponse(200, body={"data": KEY_DATA})) as get:
            result = orc.validate_api_key(" test-openrouter-key-good ")
        assert result["status"] == "valid"
        assert result["key_info"]["free_requests_remaining"] == 26
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer test-openrouter-key-good"

    def test_rejected(self):
        with patch.object(orc.requests, "get", return_value=_error(401, "User not found.")):
            assert orc.validate_api_key("test-openrouter-key-bad")["status"] == "rejected"

    @pytest.mark.parametrize("effect", [requests.ConnectionError("down"), FakeResponse(503)])
    def test_unverified(self, effect):
        kwargs = {"side_effect": effect} if isinstance(effect, Exception) else {"return_value": effect}
        with patch.object(orc.requests, "get", **kwargs):
            assert orc.validate_api_key("test-openrouter-key-maybe")["status"] == "unverified"


class TestSetApiKey:
    def test_new_key_is_used_and_clears_the_last_error(self):
        client = OpenRouterRuntimeClient(chat_model=MODEL, api_key="")
        with pytest.raises(OpenRouterError):
            client.infer("q")
        assert client.last_error["kind"] == "no_key"

        client.set_api_key(" test-openrouter-key-new ")

        assert client.last_error is None
        with patch.object(orc.requests, "post", return_value=_stream("ok")) as post:
            assert client.infer("q") == "ok"
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-openrouter-key-new"

    def test_opts_out_of_cache_warmup(self, client):
        assert client.wants_cache_warmup is False
