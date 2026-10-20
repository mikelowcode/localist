"""
Memory check and the "episodic" chip (2026-10-20, diagnostics/reports/
memory_check_assessment_2026-10-20.md).

- Planner Priority 5's deterministic signal is a word-boundary phrase list
  (planner._EPISODIC_RECALL_RE) that no longer fires on generic words such as
  "before", "wrong" or "decided".
- RoutingPlan.episodic_recalled counts the relevance-selected episodes
  (semantic hits, graph neighbours) that reached the prompt — not recency
  priming — and is reported in the response metadata, where it drives the
  chat's "episodic" chip.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from localist.controller_agent import ControllerAgent
from localist.memory_manager import EpisodicMemoryReader, MemoryManager
from localist.planner import RoutingPlan, _EPISODIC_RECALL_RE
from localist.threshold_calibration_fixtures import EP_HARD_NEGATIVES_V3
from tests.test_controller_phase4 import _ep, make_conv_agent, make_runtime


class TestRecallPhrases:
    @pytest.mark.parametrize("text", EP_HARD_NEGATIVES_V3)
    def test_generic_words_no_longer_fire(self, text):
        assert _EPISODIC_RECALL_RE.search(text.lower()) is None

    @pytest.mark.parametrize("text", [
        "do you remember which gpu i bought?",
        "what did i tell you about my laptop",
        "remind me what we decided about the logo",
        "what is my workflow preference?",
        "what are my formatting preferences?",
        "how is my dev setup configured",
        "who am i",
        "we discussed this last week, what was the plan",
    ])
    def test_memory_phrasings_fire(self, text):
        assert _EPISODIC_RECALL_RE.search(text) is not None

    @pytest.mark.parametrize("text", ["remembering the alamo", "beforehand", "wrongly"])
    def test_whole_words_only(self, text):
        assert _EPISODIC_RECALL_RE.search(text) is None


class TestEpisodicRecalled:
    def _run(self, tmp_path, recency, similar):
        conv = make_conv_agent()
        ctrl = ControllerAgent(
            runtime=make_runtime(infer_return="no"), agents=[conv],
            memory_manager=MemoryManager(db_path=tmp_path / "m.db"),
        )
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)
        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=recency), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=similar):
            result = ctrl.handle_task({"instruction": "what console do i play on"})
        return result["metadata"]

    def test_semantic_hit_in_the_prompt_is_counted(self, tmp_path):
        meta = self._run(tmp_path, recency=[_ep(1, "preference", "likes dark mode")],
                         similar=[_ep(2, "project_fact", "plays on a playstation 5")])
        assert meta["episodic_recalled"] == 1
        # The memory check didn't fire; the chip now follows recall, not routing.
        assert meta["fetch_episodic"] is False

    def test_recency_priming_alone_is_not_counted(self, tmp_path):
        meta = self._run(tmp_path, recency=[_ep(1, "preference", "likes dark mode")], similar=[])
        assert meta["episodic_recalled"] == 0

    def test_semantic_hit_also_returned_by_recency_still_counts(self, tmp_path):
        pref = _ep(1, "preference", "likes dark mode")
        meta = self._run(tmp_path, recency=[pref], similar=[pref])
        assert meta["episodic_recalled"] == 1

    def test_no_memories_reports_zero(self, tmp_path):
        meta = self._run(tmp_path, recency=[], similar=[])
        assert meta["episodic_recalled"] == 0
