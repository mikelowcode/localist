"""
OllamaRuntimeClient — _iter_ndjson_chunks NDJSON stream parsing, and the
per-call `timeout` override added to infer()/infer_stream().

Covers the 2026-07-17 fix for two previously-silent failure modes:
  1. A line carrying {"error": "..."} (Ollama's mid-stream error shape —
     rate limit, context-length overflow, moderation block, mid-generation
     crash) used to resolve to an empty content delta and be silently
     skipped, since the code only ever read data["message"]["content"].
  2. A stream that closes (response.iter_lines() exhausted) without ever
     sending a "done": true line used to make the generator finish
     normally with zero chunks yielded — indistinguishable from a genuine
     empty completion.

Confirmed live: repeated output_chars=0 completions with task still
marked COMPLETE during 2026-07-16 research-loop testing, correlating with
longer tool-result-heavy prompts. This fix makes the real Ollama-side
error visible (RuntimeError) instead of resolving silently; it does not
attempt to fix whatever the underlying Ollama-side error turns out to be.

TestTimeoutOverride (also 2026-07-17) covers a separate but related fix:
a gate-check call inside MCPToolDispatcher._evaluate_pricing_gate
(max_tokens=10) was observed stalling for the full 60s
LOCALIST_STREAM_TIMEOUT on a cloud-model-side hang. infer()/infer_stream()
now accept an optional `timeout` override so cheap classifier calls can
fail fast instead of sharing the full main-dispatch budget — see
mcp_tool_dispatcher.py's _RESEARCH_CLASSIFIER_TIMEOUT for the call site.

_iter_ndjson_chunks is tested directly against a fake requests.Response-
like object (status_code + iter_lines(decode_unicode=True)), mirroring
the fake-response pattern already used for OMLXRuntimeClient's SSE
parsing in test_omlx_runtime_client_concurrency.py — no real network call,
no OllamaRuntimeClient construction needed for those tests since
_iter_ndjson_chunks is a module-level function, not a method.
TestTimeoutOverride does construct a real OllamaRuntimeClient and patches
ollama_runtime_client.requests.post directly (same fake-response object,
reused as requests.post's return value) so it can assert on the timeout
kwarg actually passed to the transport call.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import ollama_runtime_client as _ollama_mod
from ollama_runtime_client import OllamaRuntimeClient, _iter_ndjson_chunks


def _fake_response(lines: list[str]):
    """Build a fake requests.Response-like object for _iter_ndjson_chunks.

    `lines` are the exact raw strings response.iter_lines() should yield —
    already-formatted (via json.dumps or, for malformed-line tests, a
    literal non-JSON string), not auto-encoded here, so each test can
    construct exactly the wire shape it needs.
    """

    class _FakeResponse:
        status_code = 200

        def iter_lines(self, decode_unicode: bool = True):
            yield from lines

    return _FakeResponse()


class TestHappyPath:
    def test_content_chunks_yielded_and_stops_at_done(self):
        response = _fake_response([
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "Hello"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": " world"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}),
        ])

        chunks = list(_iter_ndjson_chunks(response))
        assert chunks == ["Hello", " world"]

    def test_empty_content_deltas_are_skipped_without_raising(self):
        response = _fake_response([
            json.dumps({"message": {"content": ""}, "done": False}),
            json.dumps({"message": {"content": "answer"}, "done": True}),
        ])

        chunks = list(_iter_ndjson_chunks(response))
        assert chunks == ["answer"]

    def test_malformed_json_line_is_skipped_without_raising(self):
        """Pre-existing behavior, unchanged by the 2026-07-17 fix — only
        well-formed JSON lines are inspected for "error"/"done"."""
        response = _fake_response([
            "not valid json {{{",
            json.dumps({"message": {"content": "answer"}, "done": True}),
        ])

        chunks = list(_iter_ndjson_chunks(response))
        assert chunks == ["answer"]

    def test_line_without_message_error_or_done_is_skipped_without_raising(self):
        """A pure-metadata line (no "message", no "error", no "done") is a
        legitimate NDJSON shape, not an error — must not raise and must
        not stop the stream."""
        response = _fake_response([
            json.dumps({"model": "m", "created_at": "2026-07-17T00:00:00Z"}),
            json.dumps({"message": {"content": "answer"}, "done": True}),
        ])

        chunks = list(_iter_ndjson_chunks(response))
        assert chunks == ["answer"]

    def test_blank_lines_are_skipped(self):
        response = _fake_response([
            "",
            json.dumps({"message": {"content": "answer"}, "done": True}),
        ])

        chunks = list(_iter_ndjson_chunks(response))
        assert chunks == ["answer"]


class TestMidStreamError:
    def test_error_field_raises_runtime_error_containing_message(self):
        response = _fake_response([
            json.dumps({"error": "rate limit exceeded, please retry later"}),
        ])

        with pytest.raises(RuntimeError, match="rate limit exceeded, please retry later"):
            list(_iter_ndjson_chunks(response))

    def test_error_field_stops_generator_no_further_next_after_raise(self):
        """Content that arrived before the error line still streams
        normally (proves the fix doesn't buffer/delay legitimate chunks);
        once the error line is hit, the generator raises and is then
        exhausted — a subsequent next() must not resume or re-raise the
        same error, it must behave like any dead generator."""
        response = _fake_response([
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"error": "context length exceeded"}),
        ])

        gen = _iter_ndjson_chunks(response)
        assert next(gen) == "Hello"

        with pytest.raises(RuntimeError, match="context length exceeded"):
            next(gen)

        with pytest.raises(StopIteration):
            next(gen)

    def test_list_consumption_never_returns_partial_content_on_error(self):
        """The infer()/infer_stream() consumption pattern is
        `chunks = list(self.infer_stream(...))` — if the generator raises
        partway through, that assignment never completes, so the caller
        never receives a partial/truncated result silently; it gets a
        clean exception instead."""
        response = _fake_response([
            json.dumps({"message": {"content": "Partial answer that "}, "done": False}),
            json.dumps({"message": {"content": "should never reach the caller. "}, "done": False}),
            json.dumps({"error": "moderation block"}),
        ])

        with pytest.raises(RuntimeError, match="moderation block"):
            list(_iter_ndjson_chunks(response))


class TestIncompleteStream:
    def test_stream_ends_without_done_true_raises_runtime_error(self):
        """The stream just closes (iter_lines() exhausted) without ever
        sending "done": true — previously this looked identical to a
        successful empty completion (generator finishes normally, zero
        chunks yielded, no exception)."""
        response = _fake_response([
            json.dumps({"message": {"content": "partial answer, then silence"}, "done": False}),
        ])

        with pytest.raises(RuntimeError, match="incomplete or truncated"):
            list(_iter_ndjson_chunks(response))

    def test_completely_empty_stream_raises_runtime_error(self):
        """Zero lines at all (e.g. connection closed immediately) is the
        most extreme case of the same bug — must not resolve as a silent
        empty completion either."""
        response = _fake_response([])

        with pytest.raises(RuntimeError, match="incomplete or truncated"):
            list(_iter_ndjson_chunks(response))


class TestTimeoutOverride:
    """
    infer()/infer_stream()'s optional `timeout` parameter (2026-07-17).
    None (default) must preserve the existing self._stream_timeout
    behavior exactly; a float override must reach requests.post()'s
    `timeout` kwarg, not just be accepted and dropped.
    """

    def _client(self, stream_timeout: float = 42.0) -> OllamaRuntimeClient:
        return OllamaRuntimeClient(chat_model="test-model", stream_timeout=stream_timeout)

    def test_infer_stream_explicit_timeout_reaches_requests_post(self):
        client = self._client(stream_timeout=42.0)
        response = _fake_response([
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ])

        with patch.object(_ollama_mod.requests, "post", return_value=response) as mock_post:
            list(client.infer_stream("prompt", timeout=15.0))

        assert mock_post.call_args.kwargs["timeout"] == 15.0

    def test_infer_stream_omitted_timeout_uses_configured_stream_timeout(self):
        """No timeout override passed -> requests.post() gets
        self._stream_timeout exactly as before this parameter existed."""
        client = self._client(stream_timeout=42.0)
        response = _fake_response([
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ])

        with patch.object(_ollama_mod.requests, "post", return_value=response) as mock_post:
            list(client.infer_stream("prompt"))

        assert mock_post.call_args.kwargs["timeout"] == 42.0

    def test_infer_forwards_explicit_timeout_through_to_requests_post(self):
        """infer() delegates to infer_stream() — the override must survive
        that delegation, not get dropped at the non-streaming entry point."""
        client = self._client(stream_timeout=42.0)
        response = _fake_response([
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ])

        with patch.object(_ollama_mod.requests, "post", return_value=response) as mock_post:
            client.infer("prompt", timeout=15.0)

        assert mock_post.call_args.kwargs["timeout"] == 15.0

    def test_timeout_error_message_reflects_the_actual_override_in_effect(self):
        """The "Ollama did not respond within {timeout}s" error must report
        whatever timeout was actually used for that call, not always
        self._stream_timeout — regression guard for the hardcoded-message
        bug this parameter's addition also fixed."""
        client = self._client(stream_timeout=42.0)

        with patch.object(_ollama_mod.requests, "post", side_effect=_ollama_mod.requests.Timeout()):
            with pytest.raises(RuntimeError, match=r"did not respond within 15\.0s"):
                list(client.infer_stream("prompt", timeout=15.0))


class TestIsLocalAndNumCtx:
    """
    is_local (base_runtime_client.py) and options.num_ctx (2026-07-18):
    Ollama Cloud models are proxied through the same local daemon
    (base_url is always localhost:11434 either way — see
    docs/architecture/16-runtime-backend-layer.md §16.4's live-verified
    "gemma4:31b-cloud, proxied through ollama.com" configuration), so the
    only signal is the "-cloud" suffix on the model's tag.
    """

    def test_cloud_suffixed_model_is_not_local(self):
        client = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
        assert client.is_local is False

    def test_local_model_is_local(self):
        client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
        assert client.is_local is True

    def test_model_with_no_tag_is_local(self):
        """A bare model name with no ':tag' at all — not a cloud shape."""
        client = OllamaRuntimeClient(chat_model="gemma4")
        assert client.is_local is True

    def test_cloud_model_sends_cloud_num_ctx(self):
        from context_profile import CLOUD_PROFILE

        client = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
        response = _fake_response([
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ])

        with patch.object(_ollama_mod.requests, "post", return_value=response) as mock_post:
            list(client.infer_stream("prompt"))

        sent_payload = json.loads(mock_post.call_args.kwargs["data"])
        assert sent_payload["options"]["num_ctx"] == CLOUD_PROFILE.total_context_tokens

    def test_local_model_sends_local_num_ctx(self):
        from context_profile import LOCAL_PROFILE

        client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
        response = _fake_response([
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ])

        with patch.object(_ollama_mod.requests, "post", return_value=response) as mock_post:
            list(client.infer_stream("prompt"))

        sent_payload = json.loads(mock_post.call_args.kwargs["data"])
        assert sent_payload["options"]["num_ctx"] == LOCAL_PROFILE.total_context_tokens
