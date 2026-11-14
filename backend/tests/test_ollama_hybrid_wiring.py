"""
Ollama hybrid inference, Phase 2 (2026-11-14, §16.37): the hybrid client wired
into runtime_factory, ControllerAgent and the keep-alive setting.

  W-1  runtime_factory builds a HybridOllamaRuntimeClient when enabled with
       valid models; an invalid hybrid config falls back to the plain client.
  W-2  A turn runs on the tier its conversation size picks: under the cap
       local, over it cloud (swapped once, then latched); a new conversation
       starts local; attached session files count toward the size. The
       turn's metadata carries "hybrid" (also on the early answer_ready
       payload the stream sends).
  W-3  A cloud conversation gets the cloud history budget; when cloud fails
       before its first token the answer is rebuilt with local-sized history
       and answered locally (fell_back), and the next turn tries cloud again.
  W-4  route_task() follows the conversation's existing latch.
  W-5  Non-hybrid runtimes add no "hybrid" metadata.
  W-6  "Keep model loaded" applies live to the hybrid client.

Inference and embeddings are faked; SQLite is a tmp_path DB.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localist import session_files
from localist.controller_agent import AgentResult, ControllerAgent, TaskStatus
from localist.memory_manager import MemoryManager
from localist.ollama_hybrid import TIER_CLOUD, TIER_LOCAL, HybridOllamaRuntimeClient
from localist.ollama_runtime_client import OllamaRuntimeClient
from localist.planner import RoutingPlan
from localist.runtime_factory import create_runtime

LOCAL = "gemma4:12b-mlx"
CLOUD = "gemma4:31b-cloud"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mm(tmp_path: Path) -> MemoryManager:
    return MemoryManager(db_path=tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _clear_session_files():
    session_files.clear()
    yield
    session_files.clear()


def _stream(name: str, state: dict):
    def infer_stream(**kwargs):
        if state.get(f"{name}_down"):
            raise RuntimeError(f"{name} unavailable")
        yield f"{name} answer"
    return infer_stream


@pytest.fixture()
def hybrid():
    h = HybridOllamaRuntimeClient.build(LOCAL, CLOUD, token_cap=4_000)
    h.state = {}
    h._local.infer_stream = _stream("local", h.state)
    h._cloud.infer_stream = _stream("cloud", h.state)
    h._local.embed = lambda text: [0.0] * 8
    return h


def _agent(runtime):
    """Conversational agent that answers through the runtime with the
    prebuilt prompt, recording the tier and prompt of every call."""
    calls: list[dict] = []
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True

    def run(subtask):
        calls.append({
            "tier":   runtime.active_tier,
            "prompt": subtask.context["_prebuilt_prompt"],
        })
        answer = "".join(runtime.infer_stream(
            prompt = subtask.context["_prebuilt_prompt"],
            system = subtask.context["_prebuilt_system"],
        ))
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "conversational_agent",
            status     = TaskStatus.COMPLETE,
            output     = {"answer": answer, "sources": [], "grounded": False},
        )

    agent.run.side_effect = run
    agent.calls = calls
    return agent


_PLAN = RoutingPlan(
    agent="conversational_agent", fetch_episodic=False, fetch_rag=False,
    priority=6, tools_to_call=[],
)


def _turn(ctrl, conversation_id: str, instruction: str = "Tell me more.", **callbacks):
    with patch("localist.controller_agent.MCPToolDispatcher"):
        return ctrl.handle_task_with_plan(
            {"instruction": instruction, "context": {"conversation_id": conversation_id}},
            _PLAN, **callbacks,
        )


def _grow(mm: MemoryManager, conversation_id: str, tokens: int) -> None:
    """Log earlier turns totalling ~`tokens` tokens (len//4)."""
    chunk = "word " * 400                        # 2,000 chars = 500 tokens
    for i in range(max(1, tokens // 500)):
        mm.add("user" if i % 2 == 0 else "assistant", chunk, task_id=conversation_id)


# W-1 -----------------------------------------------------------------------

def test_factory_builds_hybrid_when_enabled():
    rt = create_runtime(
        "ollama", chat_model=CLOUD, ollama_hybrid_enabled=True,
        ollama_hybrid_local_model=LOCAL, ollama_hybrid_cloud_model=CLOUD,
        ollama_hybrid_token_cap=8_000, ollama_keep_alive="1m",
    )
    assert isinstance(rt, HybridOllamaRuntimeClient)
    assert (rt.local_model, rt.cloud_model, rt.tracker.cap, rt.keep_alive) == (LOCAL, CLOUD, 8_000, "1m")


def test_factory_defaults_cap_when_unset():
    rt = create_runtime(
        "ollama", ollama_hybrid_enabled=True,
        ollama_hybrid_local_model=LOCAL, ollama_hybrid_cloud_model=CLOUD,
    )
    assert rt.tracker.cap == 4_000


@pytest.mark.parametrize("local,cloud", [("", CLOUD), (LOCAL, ""), (CLOUD, CLOUD), (LOCAL, LOCAL)])
def test_factory_falls_back_to_plain_client_on_invalid_hybrid(local, cloud):
    rt = create_runtime(
        "ollama", chat_model=LOCAL, ollama_hybrid_enabled=True,
        ollama_hybrid_local_model=local, ollama_hybrid_cloud_model=cloud,
    )
    assert type(rt) is OllamaRuntimeClient
    assert rt._chat_model == LOCAL


def test_factory_plain_client_when_disabled():
    rt = create_runtime(
        "ollama", chat_model=LOCAL, ollama_hybrid_enabled=False,
        ollama_hybrid_local_model=LOCAL, ollama_hybrid_cloud_model=CLOUD,
    )
    assert type(rt) is OllamaRuntimeClient


# W-2 -----------------------------------------------------------------------

def test_short_conversation_answers_local(mm, hybrid):
    agent = _agent(hybrid)
    ctrl = ControllerAgent(runtime=hybrid, agents=[agent], memory_manager=mm)
    result = _turn(ctrl, "c1")
    assert result["answer"] == "local answer"
    assert agent.calls[0]["tier"] == TIER_LOCAL
    h = result["metadata"]["hybrid"]
    assert (h["tier"], h["model"], h["swapped"], h["fell_back"], h["cap"]) == (
        TIER_LOCAL, LOCAL, False, False, 4_000,
    )


def test_long_conversation_moves_to_cloud_once_then_stays(mm, hybrid):
    _grow(mm, "c1", 4_500)
    agent = _agent(hybrid)
    ctrl = ControllerAgent(runtime=hybrid, agents=[agent], memory_manager=mm)

    first = _turn(ctrl, "c1")
    assert first["answer"] == "cloud answer"
    assert first["metadata"]["hybrid"]["swapped"] is True
    assert first["metadata"]["hybrid"]["size"] > 4_000

    second = _turn(ctrl, "c1")
    assert (second["metadata"]["hybrid"]["tier"], second["metadata"]["hybrid"]["swapped"]) == (TIER_CLOUD, False)

    other = _turn(ctrl, "c2")
    assert other["metadata"]["hybrid"]["tier"] == TIER_LOCAL
    assert hybrid.active_tier == TIER_LOCAL          # nothing leaks out of a turn


def test_session_files_count_toward_size(mm, hybrid):
    # Two files (each file is capped at 4,000 tokens), ~4,500 tokens together.
    for name in ("a.md", "b.md"):
        assert session_files.add_file(name, "fact " * 1_800) is None
    ctrl = ControllerAgent(runtime=hybrid, agents=[_agent(hybrid)], memory_manager=mm)
    assert _turn(ctrl, "c1")["metadata"]["hybrid"]["tier"] == TIER_CLOUD


def test_answer_ready_payload_carries_hybrid_metadata(mm, hybrid):
    _grow(mm, "c1", 4_500)
    seen: list[dict] = []
    ctrl = ControllerAgent(runtime=hybrid, agents=[_agent(hybrid)], memory_manager=mm)
    _turn(ctrl, "c1", on_answer_ready=seen.append)
    assert seen and seen[0]["metadata"]["hybrid"]["swapped"] is True


def test_blocking_handle_task_also_decides_tier(mm, hybrid):
    _grow(mm, "c1", 4_500)
    agent = _agent(hybrid)
    ctrl = ControllerAgent(runtime=hybrid, agents=[agent], memory_manager=mm)
    with patch.object(ctrl._planner, "route", return_value=_PLAN), \
         patch("localist.controller_agent.MCPToolDispatcher"):
        result = ctrl.handle_task({"instruction": "More?", "context": {"conversation_id": "c1"}})
    assert agent.calls[0]["tier"] == TIER_CLOUD
    assert result["metadata"]["hybrid"]["tier"] == TIER_CLOUD


# W-3 -----------------------------------------------------------------------

def test_cloud_tier_gets_full_history(mm, hybrid):
    _grow(mm, "c1", 4_500)
    agent = _agent(hybrid)
    ctrl = ControllerAgent(runtime=hybrid, agents=[agent], memory_manager=mm)
    _turn(ctrl, "c1")
    # Cloud budget (60K) keeps all ~4.5K tokens of history in the prompt.
    assert len(agent.calls[0]["prompt"]) // 4 > 4_000


def test_cloud_failure_rebuilds_with_local_history_and_retries_cloud_next_turn(mm, hybrid):
    _grow(mm, "c1", 4_500)
    agent = _agent(hybrid)
    statuses: list[str] = []
    ctrl = ControllerAgent(runtime=hybrid, agents=[agent], memory_manager=mm)

    hybrid.state["cloud_down"] = True
    result = _turn(ctrl, "c1", on_status=statuses.append)

    assert result["status"] == "complete"
    assert result["answer"] == "local answer"
    assert [c["tier"] for c in agent.calls] == [TIER_CLOUD, TIER_LOCAL]
    assert len(agent.calls[1]["prompt"]) < len(agent.calls[0]["prompt"]) // 2
    h = result["metadata"]["hybrid"]
    assert (h["tier"], h["model"], h["fell_back"]) == (TIER_LOCAL, LOCAL, True)
    assert any("Cloud model unavailable" in s for s in statuses)

    hybrid.state["cloud_down"] = False
    again = _turn(ctrl, "c1")
    assert again["answer"] == "cloud answer"
    assert (again["metadata"]["hybrid"]["tier"], again["metadata"]["hybrid"]["fell_back"]) == (TIER_CLOUD, False)


# W-4 -----------------------------------------------------------------------

def test_route_task_follows_existing_latch(mm, hybrid):
    ctrl = ControllerAgent(runtime=hybrid, agents=[_agent(hybrid)], memory_manager=mm)
    tiers: list[str] = []
    with patch.object(ctrl._planner, "route", side_effect=lambda **kw: tiers.append(hybrid.active_tier) or _PLAN):
        ctrl.route_task("hi", {"conversation_id": "c1"})
        hybrid.decide_tier("c1", 9_000)
        ctrl.route_task("hi", {"conversation_id": "c1"})
    assert tiers == [TIER_LOCAL, TIER_CLOUD]
    # Reading the latch doesn't consume the turn's "swapped" signal.
    assert hybrid.decide_tier("c1", 9_000).swapped is False


# W-5 -----------------------------------------------------------------------

def test_non_hybrid_runtime_has_no_hybrid_metadata(mm):
    rt = MagicMock()
    rt.infer.return_value = "no"
    rt.embed.return_value = [0.0] * 8
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.run.side_effect = lambda st: AgentResult(
        subtask_id=st.subtask_id, agent_name="conversational_agent",
        status=TaskStatus.COMPLETE, output={"answer": "fine", "sources": [], "grounded": False},
    )
    ctrl = ControllerAgent(runtime=rt, agents=[agent], memory_manager=mm)
    assert "hybrid" not in _turn(ctrl, "c1")["metadata"]


# W-6 -----------------------------------------------------------------------

def test_keep_alive_setting_applies_to_hybrid(hybrid, monkeypatch):
    from localist import main
    monkeypatch.setattr(main._state, "runtime", hybrid)
    assert main._apply_keep_alive_live("1m") is True
    assert hybrid._local.keep_alive == "1m"
