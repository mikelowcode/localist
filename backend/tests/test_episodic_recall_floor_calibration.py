"""
Per-model episodic recall floor calibration — option B
(docs/architecture/16-runtime-backend-layer.md §16.28).

Covers the pure floor selection, CalibrationResult's persisted key, schema
v19 storage, the validated → calibrated → default precedence, and the
controller passing a calibrated floor. No live embedding model — the live
check (calibration reproduces the hand-validated floors) is recorded in
§16.28.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from localist.memory_manager import MemoryManager, EpisodicMemoryReader
from localist.planner import resolve_episodic_recall_floor
from localist.threshold_calibration import (
    CalibrationResult,
    GateCalibration,
    _recall_floor_from_scores,
    calibrate_recall_floor,
)
from localist.threshold_calibration_fixtures import RECALL_EPISODES, RECALL_POSITIVES


def _clean(floor: float) -> GateCalibration:
    return GateCalibration(
        threshold=floor, degenerate=False, reason=None, min_positive_score=0.4,
        max_negative_score=0.3, tpr_at_threshold=0.9, fpr_at_threshold=0.0,
        positive_count=31, negative_count=15,
    )


class TestFloorSelection:
    def test_picks_the_floor_separating_right_episodes_from_off_topic(self):
        positives = [(0.62, 1), (0.58, 1), (0.55, 2), (0.51, 1)]
        negatives = [0.40, 0.44, 0.47]
        g = _recall_floor_from_scores(positives, negatives)
        # 0.48–0.51 all give recall 100% / FP 0%; the highest wins (fewest wrong extras).
        assert g.threshold == 0.51
        assert (g.tpr_at_threshold, g.fpr_at_threshold, g.degenerate) == (1.0, 0.0, False)

    def test_right_episode_outside_top_five_never_counts(self):
        g = _recall_floor_from_scores([(0.9, 6), (0.6, 1)], [0.2])
        assert g.tpr_at_threshold == 0.5

    def test_overlapping_scores_are_degenerate(self):
        positives = [(0.50, 1), (0.52, 1), (0.54, 1), (0.56, 1)]
        negatives = [0.55, 0.57, 0.53, 0.51]
        g = _recall_floor_from_scores(positives, negatives)
        assert g.degenerate is True and g.threshold is None

    def test_empty_pool_is_degenerate(self):
        assert _recall_floor_from_scores([], [0.3]).degenerate is True

    def test_live_style_battery_with_a_perfect_embedder_calibrates_cleanly(self):
        # One-hot vector per episode; a positive query maps to its right
        # episode's vector, negatives to a direction no episode uses.
        keys = list(RECALL_EPISODES)
        dim = len(keys) + 1
        by_text = {}
        for i, (key, (_t, content)) in enumerate(RECALL_EPISODES.items()):
            from localist.threshold_calibration import recall_document_text
            by_text[recall_document_text(content)] = [1.0 if j == i else 0.0 for j in range(dim)]
        for query, right in RECALL_POSITIVES:
            i = keys.index(right)
            by_text[query] = [0.8 if j == i else 0.0 for j in range(dim - 1)] + [0.6]

        def embed(text):
            return by_text.get(text, [0.0] * (dim - 1) + [1.0])

        g = calibrate_recall_floor(embed)
        assert g.degenerate is False
        assert g.tpr_at_threshold == 1.0 and g.fpr_at_threshold == 0.0


class TestCalibrationResultThresholds:
    def test_clean_floor_is_persisted_alongside_gates(self):
        r = CalibrationResult(gates={"lookup_request": _clean(0.6)}, recall_floor=_clean(0.41))
        assert r.thresholds == {"lookup_request": 0.6, "episodic_recall_floor": 0.41}

    def test_degenerate_floor_is_left_out(self):
        bad = GateCalibration(
            threshold=None, degenerate=True, reason="x", min_positive_score=None,
            max_negative_score=None, tpr_at_threshold=None, fpr_at_threshold=None,
            positive_count=0, negative_count=0,
        )
        assert "episodic_recall_floor" not in CalibrationResult(recall_floor=bad).thresholds


class TestStorage:
    def test_round_trip(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "t.db")
        mm.set_calibrated_thresholds("m", {"lookup_request": 0.6, "episodic_recall_floor": 0.41})
        assert mm.get_calibrated_thresholds("m") == {"lookup_request": 0.6, "episodic_recall_floor": 0.41}

    def test_recalibration_clears_a_floor_that_went_degenerate(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "t.db")
        mm.set_calibrated_thresholds("m", {"episodic_recall_floor": 0.41})
        mm.set_calibrated_thresholds("m", {"lookup_request": 0.6})
        assert mm.get_calibrated_thresholds("m") == {"lookup_request": 0.6}

    def test_v18_db_gains_the_column_and_keeps_existing_rows(self, tmp_path):
        path = tmp_path / "v18.db"
        MemoryManager(db_path=path).set_calibrated_thresholds("m", {"lookup_request": 0.6})
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE t18 (model TEXT PRIMARY KEY, explicit_search_action REAL, lookup_request REAL,
                              research_intent REAL, episodic_relevance REAL, calibrated_at REAL NOT NULL);
            INSERT INTO t18 SELECT model, explicit_search_action, lookup_request, research_intent,
                                   episodic_relevance, calibrated_at FROM embedding_model_thresholds;
            DROP TABLE embedding_model_thresholds;
            ALTER TABLE t18 RENAME TO embedding_model_thresholds;
            UPDATE schema_version SET version = 18;
        """)
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path)

        assert mm.get_calibrated_thresholds("m") == {"lookup_request": 0.6}
        mm.set_calibrated_thresholds("m", {"episodic_recall_floor": 0.4})
        assert mm.get_calibrated_thresholds("m") == {"episodic_recall_floor": 0.4}


class TestPrecedence:
    @pytest.mark.parametrize("model, calibrated, floor", [
        ("nomic-embed-text:latest", {"episodic_recall_floor": 0.30}, 0.55),   # validated wins
        ("unlisted-model",          {"episodic_recall_floor": 0.41}, 0.41),   # calibrated
        ("unlisted-model",          {"lookup_request": 0.6},          0.45),   # no floor calibrated
        ("unlisted-model",          None,                             0.45),
        (None,                      None,                             0.45),
    ])
    def test_validated_then_calibrated_then_default(self, model, calibrated, floor):
        assert resolve_episodic_recall_floor(model, calibrated) == floor


class TestControllerUsesCalibratedFloor:
    def test_unlisted_model_recall_uses_its_calibrated_floor(self, tmp_path):
        from localist.controller_agent import ControllerAgent
        from localist.planner import RoutingPlan
        from tests.test_controller_phase4 import make_conv_agent, make_runtime

        mm   = MemoryManager(db_path=tmp_path / "c.db")
        conv = make_conv_agent()
        ctrl = ControllerAgent(
            runtime=make_runtime(infer_return="no"), agents=[conv], memory_manager=mm,
            embedding_model_name="unlisted-model",
            calibrated_thresholds={"episodic_recall_floor": 0.41},
        )
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=6)
        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[]) as by_similarity:
            ctrl.handle_task({"instruction": "what console do I play on"})

        assert by_similarity.call_args.kwargs["min_score"] == 0.41


class TestCalibrationResponseReportsTheFloorInUse:
    @pytest.mark.parametrize("model, calibrated, in_use", [
        ("nomic-embed-text:latest", 0.30, 0.55),   # hand-validated beats calibration
        ("unlisted-model",          0.41, 0.41),
    ])
    def test_in_use_follows_precedence(self, model, calibrated, in_use):
        from localist import main
        resp = main._calibration_result_to_response(model, CalibrationResult(recall_floor=_clean(calibrated)))
        assert resp.recall_floor.threshold == calibrated
        assert resp.recall_floor_in_use == in_use
