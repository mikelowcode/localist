"""
Per-completion telemetry (non-answer guard plan Phase 2, §16.35):
completion_record.py, openai_compat.update_stream_meta(), each runtime
client's `last_completion`, and ControllerAgent copying it into the turn's
metadata["completion"].

Motivation: on 2026-11-11 free OpenRouter models ended junk replies with a
clean finish_reason "stop" after 1 completion token, and nothing recorded
either fact. HTTP is faked at each client module's `requests.post`.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from localist import ollama_runtime_client as ollama_mod
from localist import omlx_runtime_client as omlx_mod
from localist import openrouter_runtime_client as openrouter_mod
from localist.completion_record import CompletionRecorder, make_record, record_from_openai_meta
from localist.controller_agent import AgentResult, ControllerAgent, TaskStatus
from localist.memory_manager import MemoryManager
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.omlx_runtime_client import OMLXRuntimeClient
from localist.openai_compat import update_stream_meta
from localist.openrouter_runtime_client import OpenRouterRuntimeClient
from localist.planner import RoutingPlan
from localist.prompt_builder import ToolResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStream:
    status_code = 200
    headers: dict = {}

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self) -> None:
        pass


def _sse(*events: dict) -> _FakeStream:
    return _FakeStream([f"data: {json.dumps(e)}" for e in events] + ["data: [DONE]"])


# The shape of the real 2026-11-11 junk reply: one empty delta that already
# carries finish_reason, then a usage chunk with completion_tokens=1.
_OPENROUTER_JUNK = (
    {"provider": "Nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
     "choices": [{"delta": {"content": ""}, "finish_reason": "stop",
                  "native_finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 1108, "completion_tokens": 1,
                              "completion_tokens_details": {"reasoning_tokens": 0}}},
)


# ---------------------------------------------------------------------------
# completion_record.py
# ---------------------------------------------------------------------------

class TestCompletionRecorder:
    def test_none_until_set_then_cleared(self):
        rec = CompletionRecorder()
        assert rec.get() is None
        rec.set({"model": "m"})
        assert rec.get() == {"model": "m"}
        rec.clear()
        assert rec.get() is None

    def test_records_are_per_thread(self):
        """The main answer's record must survive another thread's call
        (post-answer hooks / background jobs share the runtime client)."""
        rec = CompletionRecorder()
        rec.set({"label": "main_dispatch"})
        other = threading.Thread(target=lambda: rec.set({"label": "working_state"}))
        other.start()
        other.join()
        assert rec.get() == {"label": "main_dispatch"}

    def test_record_from_openai_meta(self):
        meta: dict = {}
        for event in _OPENROUTER_JUNK:
            update_stream_meta(meta, event)
        record = record_from_openai_meta("openrouter", "fallback-model", meta)
        assert record == make_record(
            backend="openrouter", model="nvidia/nemotron-3-ultra-550b-a55b:free",
            finish_reason="stop", prompt_tokens=1108, completion_tokens=1,
            reasoning_tokens=0, provider="Nvidia",
        )
        assert meta["native_finish_reason"] == "stop"

    def test_record_from_empty_meta_keeps_configured_model(self):
        record = record_from_openai_meta("omlx", "gemma-local", {})
        assert record["model"] == "gemma-local"
        assert record["finish_reason"] is None
        assert record["completion_tokens"] is None


# ---------------------------------------------------------------------------
# Runtime clients
# ---------------------------------------------------------------------------

class TestOpenRouterClientRecordsCompletion:
    def test_junk_reply_is_recorded(self):
        client = OpenRouterRuntimeClient(chat_model="nvidia/nemotron-3-ultra-550b-a55b:free",
                                         api_key="test-openrouter-key-test")
        with patch.object(openrouter_mod.requests, "post", return_value=_sse(*_OPENROUTER_JUNK)):
            assert client.infer("prompt") == ""
        record = client.last_completion
        assert record["backend"] == "openrouter"
        assert record["finish_reason"] == "stop"
        assert record["completion_tokens"] == 1
        assert record["provider"] == "Nvidia"

    def test_failed_request_clears_previous_record(self):
        client = OpenRouterRuntimeClient(chat_model="m", api_key="test-openrouter-key-test")
        with patch.object(openrouter_mod.requests, "post", return_value=_sse(*_OPENROUTER_JUNK)):
            client.infer("prompt")
        assert client.last_completion is not None

        failing = _FakeStream([])
        failing.status_code = 500
        failing.text = json.dumps({"error": {"message": "boom", "code": 500}})
        failing.json = lambda: {"error": {"message": "boom", "code": 500}}
        with patch.object(openrouter_mod.requests, "post", return_value=failing):
            with pytest.raises(Exception):
                client.infer("prompt")
        assert client.last_completion is None


class TestOMLXClientRecordsCompletion:
    def test_finish_reason_and_usage_recorded(self):
        omlx_mod._inflight_count = 0
        client = OMLXRuntimeClient(chat_model="gemma-local")
        stream = _sse(
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 50, "completion_tokens": 1024}},
        )
        with patch.object(omlx_mod.requests, "post", return_value=stream):
            assert client.infer("prompt") == "Hello"
        record = client.last_completion
        assert record["backend"] == "omlx"
        assert record["model"] == "gemma-local"
        assert record["finish_reason"] == "length"
        assert record["completion_tokens"] == 1024


class TestOllamaClientRecordsCompletion:
    def test_done_reason_and_eval_counts_recorded(self):
        client = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
        stream = _FakeStream([
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop",
                        "eval_count": 2, "prompt_eval_count": 300}),
        ])
        with patch.object(ollama_mod.requests, "post", return_value=stream):
            assert client.infer("prompt") == "Hi"
        assert client.last_completion == make_record(
            backend="ollama", model="gemma4:31b-cloud", finish_reason="stop",
            prompt_tokens=300, completion_tokens=2,
        )


# ---------------------------------------------------------------------------
# ControllerAgent → metadata["completion"]
# ---------------------------------------------------------------------------

class _RecordingRuntime:
    """Runtime double whose last_completion changes with every infer call,
    like the real clients (one record per finished completion)."""
    is_local = False

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.last_completion: dict | None = None

    def push(self, **fields) -> None:
        self.records.append(make_record(backend="openrouter", model="m", **fields))

    def infer(self, prompt, system="", max_tokens=1024, temperature=0.2, label="", timeout=None):
        self.last_completion = None   # post-answer hooks must not need a record
        return "no"

    def embed(self, text):
        return [0.0] * 768


def _agent_with(runtime: _RecordingRuntime, answers: list[str]):
    """Conversational agent double: each run() 'completes' on the runtime."""
    calls = {"n": 0}
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True

    def run(subtask):
        i = min(calls["n"], len(answers) - 1)
        runtime.last_completion = runtime.records[min(calls["n"], len(runtime.records) - 1)]
        calls["n"] += 1
        return AgentResult(subtask.subtask_id, "conversational_agent", TaskStatus.COMPLETE,
                           {"answer": answers[i], "sources": [], "grounded": False})
    agent.run.side_effect = run
    return agent


def _plan() -> RoutingPlan:
    return RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False,
                       priority=3, tools_to_call=["news_search"])


def _run_turn(tmp_path, runtime, answers):
    mm = MemoryManager(db_path=tmp_path / "t.db")
    ctrl = ControllerAgent(runtime=runtime, agents=[_agent_with(runtime, answers)],
                           memory_manager=mm)
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = [ToolResult(
        tool_name="news_search", parameters="q", success=True,
        result="• iPhone Air 2 and iPhone 18 dummy models leak — GSMArena — 2026-11-10",
    )]
    with patch("localist.controller_agent.MCPToolDispatcher", return_value=dispatcher):
        return ctrl.handle_task_with_plan(
            {"instruction": "Tell me more about this news story: \"iPhone Air 2\"",
             "task_id": "completion-test"},
            _plan(),
        )


class TestControllerStoresCompletion:
    def test_good_answer_carries_its_completion(self, tmp_path):
        rt = _RecordingRuntime()
        rt.push(finish_reason="stop", prompt_tokens=400, completion_tokens=310)
        result = _run_turn(tmp_path, rt, ["GSMArena reports dummy units of both phones have leaked."])

        md = result["metadata"]
        assert md["completion"]["finish_reason"] == "stop"
        assert md["completion"]["completion_tokens"] == 310
        assert "completion_first" not in md
        json.dumps(md)   # must stay JSON-serialisable for chat_turns

    def test_retried_turn_keeps_both_attempts(self, tmp_path):
        rt = _RecordingRuntime()
        rt.push(finish_reason="stop", completion_tokens=1)      # " שח"
        rt.push(finish_reason="stop", completion_tokens=290)    # real retry
        result = _run_turn(tmp_path, rt, [" שח", "GSMArena reports dummy units of both phones have leaked."])

        md = result["metadata"]
        assert md["non_answer_reason"] == "script_mismatch"
        assert md["completion_first"]["completion_tokens"] == 1
        assert md["completion"]["completion_tokens"] == 290

    def test_runtime_without_record_adds_nothing(self, tmp_path):
        """MagicMock runtimes (most tests) and clients without the attribute."""
        rt = MagicMock()
        rt.infer.return_value = "no"
        rt.embed.return_value = [0.0] * 768
        mm = MemoryManager(db_path=tmp_path / "t.db")
        conv = MagicMock()
        conv.name = "conversational_agent"
        conv.run.side_effect = lambda st: AgentResult(
            st.subtask_id, "conversational_agent", TaskStatus.COMPLETE,
            {"answer": "A complete answer here.", "sources": [], "grounded": False})
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        result = ctrl.handle_task_with_plan(
            {"instruction": "hello", "task_id": "t"},
            RoutingPlan(agent="conversational_agent", fetch_episodic=False,
                        fetch_rag=False, priority=6, tools_to_call=[]),
        )
        assert "completion" not in result["metadata"]
