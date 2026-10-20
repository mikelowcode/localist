"""
Corpus documents kept out of RAG because they reach the prompt through their
own slot: the persona (Slot 1b) and the user profile (Slot 3b, scored line by
line against the per-model profile-fact threshold —
docs/architecture/03-unified-prompt-contract.md §3.6). Without the profile
exclusion, wiki/users/michael.md could arrive whole as a Slot 4 source,
bypassing that threshold.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localist.controller_agent import ControllerAgent, _excluded_from_rag
from localist.memory_manager import EpisodicMemoryReader, MemoryManager
from localist.planner import RoutingPlan
from localist.prompt_builder import ToolResult as _ToolResult
from tests.test_controller_phase4 import make_conv_agent, make_runtime


class TestExcludedFromRag:
    @pytest.mark.parametrize("path", [
        "/wiki/persona.md",
        "/Users/x/Library/Application Support/Localist/wiki/users/michael.md",
        "/Users/x/Projects/localist/backend/wiki/users/michael.md",   # another backend root
        Path("wiki/users/michael.md"),
    ])
    def test_excluded(self, path):
        assert _excluded_from_rag(path) is True

    @pytest.mark.parametrize("path", [
        "/backend/wiki/how-localist-works.md",
        "/backend/wiki/michael.md",              # not under users/
        "/backend/wiki/users/someone-else.md",
        "/backend/raw/michael.md",
        "/backend/wiki/users-guide/michael.md",
    ])
    def test_kept(self, path):
        assert _excluded_from_rag(path) is False


def _doc(path: str, content: str):
    doc = MagicMock()
    doc.path = Path(path)
    doc.content = content
    doc.doc_type = "wiki"
    doc.scored_by_embedding = True
    doc.relevance_score = 0.9
    return doc


_PROFILE = _doc("/backend/wiki/users/michael.md", "- Lives in PROFILE-MARKER-CITY")
_PAGE    = _doc("/backend/wiki/how-localist-works.md", "Localist routes with PAGE-MARKER-RULES.")


def _prompt_for(tmp_path, plan: RoutingPlan, tool_results=None) -> str:
    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = tool_results or []
    conv = make_conv_agent()
    ctrl = ControllerAgent(
        runtime=make_runtime(), agents=[conv],
        memory_manager=MemoryManager(db_path=tmp_path / "rag.db"),
    )
    patches = [
        patch.object(ctrl._planner, "route", return_value=plan),
        patch.object(MemoryManager, "query_corpus", return_value=[_PROFILE, _PAGE]),
        patch.object(EpisodicMemoryReader, "by_recency", return_value=[]),
        patch.object(EpisodicMemoryReader, "by_similarity", return_value=[]),
        patch("localist.controller_agent.MCPToolDispatcher", return_value=dispatcher),
    ]
    for p in patches:
        p.start()
    try:
        ctrl.handle_task({"instruction": "where do I live and how does Localist route"})
    finally:
        for p in reversed(patches):
            p.stop()
    return conv._received[0].context["_prebuilt_prompt"]


class TestProfileNeverARagSource:
    def test_rag_fetch_drops_the_profile_but_keeps_other_pages(self, tmp_path):
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=True, priority=4)
        prompt = _prompt_for(tmp_path, plan)
        assert "PAGE-MARKER-RULES" in prompt
        assert "PROFILE-MARKER-CITY" not in prompt

    def test_web_search_failure_fallback_drops_the_profile_too(self, tmp_path):
        plan = RoutingPlan(
            agent="conversational_agent", fetch_episodic=False, fetch_rag=False,
            priority=6, tools_to_call=["web_search"],
        )
        failed = _ToolResult(tool_name="web_search", parameters="q", result="ERROR", success=False)
        prompt = _prompt_for(tmp_path, plan, tool_results=[failed])
        assert "PAGE-MARKER-RULES" in prompt
        assert "PROFILE-MARKER-CITY" not in prompt
