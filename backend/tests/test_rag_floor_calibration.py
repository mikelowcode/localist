"""
Per-model document RAG cutoff — hand-validated, calibrated, default
(diagnostics/reports/rag_cutoff_assessment_2026-10-20.md).

Covers resolve_rag_floor() precedence, calibrate_rag_floor() through a
temporary MemoryManager on the RAG_* battery, CalibrationResult's persisted
key, schema v21 storage and migration, the Planner's Priority 4 Path B /
Priority 3b and the controller's Slot 4 filter using the resolved cutoff, and
the calibration response. No live embedding model.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localist.memory_manager import _EMBEDDING_DIM, EpisodicMemoryReader, MemoryManager
from localist.planner import Planner, RoutingPlan, resolve_rag_floor
from localist.threshold_calibration import CalibrationResult, GateCalibration, calibrate_rag_floor
from localist.threshold_calibration_fixtures import RAG_DOCS, RAG_NEGATIVES, RAG_POSITIVES


def _clean(t: float) -> GateCalibration:
    return GateCalibration(threshold=t, degenerate=False, reason=None, min_positive_score=0.6,
                           max_negative_score=0.3, tpr_at_threshold=1.0, fpr_at_threshold=0.0,
                           positive_count=24, negative_count=20)


class TestPrecedence:
    @pytest.mark.parametrize("model, calibrated, floor", [
        ("nomic-embed-text:latest", {"rag_source_floor": 0.30}, 0.55),   # hand-validated wins
        ("embeddinggemma-2:latest", {"rag_source_floor": 0.72}, 0.72),   # calibrated
        ("unlisted-model",          {"profile_fact_threshold": 0.5}, 0.55),
        ("unlisted-model",          None, 0.55),
        (None,                      None, 0.55),
    ])
    def test_hand_then_calibrated_then_default(self, model, calibrated, floor):
        assert resolve_rag_floor(model, calibrated) == floor


def _topic(key: str) -> str:
    return key.removesuffix("-raw")


def _perfect_embedder():
    """One axis per topic: a topic's documents and its questions share it;
    negatives point along a spare axis. 768-dimensional, since
    index_document() stores only vectors of _EMBEDDING_DIM."""
    topics = sorted({_topic(k) for k in RAG_DOCS})
    dim = _EMBEDDING_DIM

    def axis(i, w=1.0):
        return [w if j == i else 0.0 for j in range(dim)]

    by_text = {content[:500]: axis(topics.index(_topic(k))) for k, (_t, content) in RAG_DOCS.items()}
    for query, answers in RAG_POSITIVES:
        by_text[query] = axis(topics.index(_topic(answers[0])))
    return lambda text: by_text.get(text, axis(dim - 1))


class TestCalibrateRagFloor:
    def test_perfect_embedder_calibrates_cleanly(self):
        g = calibrate_rag_floor(_perfect_embedder())
        assert g.degenerate is False
        assert g.tpr_at_threshold == 1.0 and g.fpr_at_threshold == 0.0
        assert g.positive_count == len(RAG_POSITIVES) and g.negative_count == len(RAG_NEGATIVES)

    def test_inseparable_embedder_is_degenerate(self):
        g = calibrate_rag_floor(lambda text: [1.0] + [0.0] * (_EMBEDDING_DIM - 1))
        assert g.degenerate is True and g.threshold is None

    def test_failing_embedder_is_degenerate_not_raised(self):
        def embed(text):
            raise RuntimeError("embedder down")
        g = calibrate_rag_floor(embed)
        assert g.degenerate is True

    def test_calibrate_thresholds_runs_it(self):
        from localist import threshold_calibration as tc
        with patch.object(tc, "calibrate_rag_floor", return_value=_clean(0.72)) as cal:
            result = tc.calibrate_thresholds(lambda text: [1.0, 0.0])
        cal.assert_called_once()
        assert result.thresholds["rag_source_floor"] == 0.72


class TestStorage:
    def test_round_trip(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "t.db")
        mm.set_calibrated_thresholds("m", {"rag_source_floor": 0.72, "profile_fact_threshold": 0.7})
        assert mm.get_calibrated_thresholds("m") == {"rag_source_floor": 0.72, "profile_fact_threshold": 0.7}

    def test_degenerate_floor_is_not_persisted(self):
        bad = GateCalibration(threshold=None, degenerate=True, reason="x", min_positive_score=None,
                              max_negative_score=None, tpr_at_threshold=None, fpr_at_threshold=None,
                              positive_count=0, negative_count=0)
        assert "rag_source_floor" not in CalibrationResult(rag_floor=bad).thresholds

    def test_v20_db_migrates_to_v21_and_keeps_rows(self, tmp_path):
        path = tmp_path / "v20.db"
        MemoryManager(db_path=path).set_calibrated_thresholds(
            "m", {"profile_fact_threshold": 0.7}, calibration_version=2,
        )
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE t20 (model TEXT PRIMARY KEY, explicit_search_action REAL, lookup_request REAL,
                              research_intent REAL, episodic_relevance REAL, calibrated_at REAL NOT NULL,
                              episodic_recall_floor REAL, profile_fact_threshold REAL,
                              calibration_version INTEGER);
            INSERT INTO t20 SELECT model, explicit_search_action, lookup_request, research_intent,
                                   episodic_relevance, calibrated_at, episodic_recall_floor,
                                   profile_fact_threshold, calibration_version
                            FROM embedding_model_thresholds;
            DROP TABLE embedding_model_thresholds;
            ALTER TABLE t20 RENAME TO embedding_model_thresholds;
            UPDATE schema_version SET version = 20;
        """)
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path)

        from localist.memory_manager import _SCHEMA_VERSION
        conn = sqlite3.connect(str(path))
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == _SCHEMA_VERSION == 21
        conn.close()
        assert mm.get_calibrated_thresholds("m") == {"profile_fact_threshold": 0.7}
        assert mm.get_calibration_version("m") == 2   # stale vs CALIBRATION_VERSION 3
        mm.set_calibrated_thresholds("m", {"rag_source_floor": 0.4})
        assert mm.get_calibrated_thresholds("m") == {"rag_source_floor": 0.4}


def _doc(score: float, name: str = "page.md"):
    d = MagicMock()
    d.path, d.name, d.doc_type = Path(f"/backend/wiki/{name}"), name, "wiki"
    d.content, d.relevance_score, d.scored_by_embedding = f"{name} body", score, True
    return d


class TestPlannerUsesTheFloor:
    def _planner(self, model, calibrated, top_score):
        mm = MagicMock()
        mm.query_corpus.return_value = [_doc(top_score)]
        rt = MagicMock(); rt.infer.return_value = "no"
        return Planner(runtime=rt, memory_manager=mm, embedding_model_name=model,
                       calibrated_thresholds=calibrated)

    def test_p4_routes_only_above_the_models_floor(self):
        high = self._planner("embeddinggemma-2:latest", {"rag_source_floor": 0.72}, 0.65)
        assert high._priority4_corpus("summarize the outline", "summarize the outline") is None
        nomic = self._planner("nomic-embed-text:latest", None, 0.65)
        plan = nomic._priority4_corpus("summarize the outline", "summarize the outline")
        assert plan is not None and plan.fetch_rag is True

    def test_p3b_treats_a_below_floor_hit_as_a_miss(self):
        from localist.planner import _FACTUAL_QUERY_KEYWORDS
        kw = sorted(_FACTUAL_QUERY_KEYWORDS)[0]
        p = self._planner("embeddinggemma-2:latest", {"rag_source_floor": 0.72}, 0.65)
        plan = p._priority3b_factual(f"{kw} the release date", f"{kw} the release date")
        assert plan is not None and "web_search" in plan.tools_to_call


class TestControllerFiltersByTheFloor:
    def test_rag_sources_below_the_floor_are_dropped(self, tmp_path):
        from localist.controller_agent import ControllerAgent
        from tests.test_controller_phase4 import make_conv_agent, make_runtime

        conv = make_conv_agent()
        ctrl = ControllerAgent(
            runtime=make_runtime(), agents=[conv], memory_manager=MemoryManager(db_path=tmp_path / "c.db"),
            embedding_model_name="embeddinggemma-2:latest", calibrated_thresholds={"rag_source_floor": 0.72},
        )
        hi, lo = _doc(0.80, "kept.md"), _doc(0.65, "dropped.md")
        hi.content, lo.content = "KEPT-MARKER", "DROPPED-MARKER"
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=True, priority=4)
        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(MemoryManager, "query_corpus", return_value=[hi, lo]), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[]):
            ctrl.handle_task({"instruction": "summarize the outline"})
        prompt = conv._received[0].context["_prebuilt_prompt"]
        assert "KEPT-MARKER" in prompt and "DROPPED-MARKER" not in prompt


class TestCalibrationResponse:
    @pytest.mark.parametrize("model, calibrated, in_use", [
        ("nomic-embed-text:latest", 0.57, 0.55),
        ("embeddinggemma-2:latest", 0.72, 0.72),
    ])
    def test_in_use_follows_precedence(self, model, calibrated, in_use):
        from localist import main
        resp = main._calibration_result_to_response(model, CalibrationResult(rag_floor=_clean(calibrated)))
        assert resp.rag_floor.threshold == calibrated
        assert resp.rag_floor_in_use == in_use
