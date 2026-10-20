"""
Per-model profile-fact threshold — hand-validated, calibrated, default
(docs/architecture/03-unified-prompt-contract.md §3.6; measured in
diagnostics/reports/profile_fact_threshold_assessment_2026-10-20.md).

Covers calibrate_profile_threshold() on the PROFILE_* battery,
CalibrationResult's persisted key, schema v20 storage and migration, the
validated → calibrated → default precedence, the controller passing the
resolved threshold to _score_profile_facts() (on every route, P6 included),
and the calibration response's in-use value. No live embedding model.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from localist.memory_manager import MemoryManager, EpisodicMemoryReader
from localist.planner import resolve_profile_fact_threshold
from localist.threshold_calibration import (
    CalibrationResult,
    GateCalibration,
    calibrate_profile_threshold,
)
from localist.threshold_calibration_fixtures import (
    PROFILE_LINES,
    PROFILE_NEGATIVES,
    PROFILE_POSITIVES,
)


def _clean(threshold: float) -> GateCalibration:
    return GateCalibration(
        threshold=threshold, degenerate=False, reason=None, min_positive_score=0.5,
        max_negative_score=0.4, tpr_at_threshold=0.9, fpr_at_threshold=0.0,
        positive_count=30, negative_count=16,
    )


_DEGENERATE = GateCalibration(
    threshold=None, degenerate=True, reason="x", min_positive_score=None,
    max_negative_score=None, tpr_at_threshold=None, fpr_at_threshold=None,
    positive_count=0, negative_count=0,
)


def _perfect_embedder():
    """One-hot vector per profile line; a positive query points at its first
    relevant line, negatives at a direction no line uses."""
    keys = list(PROFILE_LINES)
    dim = len(keys) + 1
    by_text = {line: [1.0 if j == i else 0.0 for j in range(dim)]
               for i, (_section, line) in enumerate(PROFILE_LINES.values())}
    for query, relevant in PROFILE_POSITIVES:
        i = keys.index(relevant[0])
        by_text[query] = [0.8 if j == i else 0.0 for j in range(dim - 1)] + [0.6]
    return lambda text: by_text.get(text, [0.0] * (dim - 1) + [1.0])


class TestFixture:
    def test_every_relevant_key_is_a_profile_line(self):
        assert all(k in PROFILE_LINES for _q, rel in PROFILE_POSITIVES for k in rel)

    def test_battery_has_positives_and_negatives(self):
        assert len(PROFILE_POSITIVES) >= 20 and len(PROFILE_NEGATIVES) >= 10


class TestCalibrateProfileThreshold:
    def test_perfect_embedder_calibrates_cleanly(self):
        g = calibrate_profile_threshold(_perfect_embedder())
        assert g.degenerate is False
        assert g.tpr_at_threshold == 1.0 and g.fpr_at_threshold == 0.0
        assert g.positive_count == len(PROFILE_POSITIVES)
        assert g.negative_count == len(PROFILE_NEGATIVES)

    def test_embedder_that_cannot_separate_is_degenerate(self):
        g = calibrate_profile_threshold(lambda text: [1.0, 0.0])
        assert g.degenerate is True and g.threshold is None

    def test_embed_failures_thin_the_pool_without_raising(self):
        def embed(text):
            raise RuntimeError("down")
        g = calibrate_profile_threshold(embed)
        assert g.degenerate is True


class TestCalibrationResultThresholds:
    def test_clean_threshold_is_persisted_alongside_gates_and_floor(self):
        r = CalibrationResult(
            gates={"lookup_request": _clean(0.6)},
            recall_floor=_clean(0.41),
            profile_threshold=_clean(0.72),
        )
        assert r.thresholds == {
            "lookup_request": 0.6, "episodic_recall_floor": 0.41, "profile_fact_threshold": 0.72,
        }

    def test_degenerate_threshold_is_left_out(self):
        assert "profile_fact_threshold" not in CalibrationResult(profile_threshold=_DEGENERATE).thresholds

    def test_calibrate_thresholds_runs_it(self):
        from localist import threshold_calibration as tc
        with patch.object(tc, "calibrate_profile_threshold", return_value=_clean(0.6)) as cal:
            result = tc.calibrate_thresholds(lambda text: [1.0, 0.0])
        cal.assert_called_once()
        assert result.profile_threshold.threshold == 0.6


class TestStorage:
    def test_round_trip(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "t.db")
        mm.set_calibrated_thresholds("m", {"episodic_recall_floor": 0.41, "profile_fact_threshold": 0.72})
        assert mm.get_calibrated_thresholds("m") == {
            "episodic_recall_floor": 0.41, "profile_fact_threshold": 0.72,
        }

    def test_recalibration_clears_a_threshold_that_went_degenerate(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "t.db")
        mm.set_calibrated_thresholds("m", {"profile_fact_threshold": 0.72})
        mm.set_calibrated_thresholds("m", {"lookup_request": 0.6})
        assert mm.get_calibrated_thresholds("m") == {"lookup_request": 0.6}

    def test_v19_db_migrates_to_v20_and_keeps_existing_rows(self, tmp_path):
        path = tmp_path / "v19.db"
        MemoryManager(db_path=path).set_calibrated_thresholds(
            "m", {"lookup_request": 0.6, "episodic_recall_floor": 0.41},
        )
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE t19 (model TEXT PRIMARY KEY, explicit_search_action REAL, lookup_request REAL,
                              research_intent REAL, episodic_relevance REAL, calibrated_at REAL NOT NULL,
                              episodic_recall_floor REAL);
            INSERT INTO t19 SELECT model, explicit_search_action, lookup_request, research_intent,
                                   episodic_relevance, calibrated_at, episodic_recall_floor
                            FROM embedding_model_thresholds;
            DROP TABLE embedding_model_thresholds;
            ALTER TABLE t19 RENAME TO embedding_model_thresholds;
            UPDATE schema_version SET version = 19;
        """)
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path)

        conn = sqlite3.connect(str(path))
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        from localist.memory_manager import _SCHEMA_VERSION
        assert version == _SCHEMA_VERSION   # migrates through v20 to the latest
        assert mm.get_calibrated_thresholds("m") == {"lookup_request": 0.6, "episodic_recall_floor": 0.41}
        assert mm.get_calibration_version("m") is None   # stale: recalibrates when next active
        mm.set_calibrated_thresholds("m", {"profile_fact_threshold": 0.72})
        assert mm.get_calibrated_thresholds("m") == {"profile_fact_threshold": 0.72}


class TestPrecedence:
    @pytest.mark.parametrize("model, calibrated, threshold", [
        ("nomic-embed-text:latest",  {"profile_fact_threshold": 0.30}, 0.52),   # validated wins
        ("embeddinggemma-300m-4bit", {"profile_fact_threshold": 0.40}, 0.45),
        ("embeddinggemma:latest",    {"profile_fact_threshold": 0.40}, 0.45),
        ("unlisted-model",           {"profile_fact_threshold": 0.72}, 0.72),   # calibrated
        ("unlisted-model",           {"episodic_recall_floor": 0.41},  0.45),   # none calibrated
        ("unlisted-model",           None,                             0.45),
        (None,                       None,                             0.45),
    ])
    def test_validated_then_calibrated_then_default(self, model, calibrated, threshold):
        assert resolve_profile_fact_threshold(model, calibrated) == threshold


class TestControllerUsesResolvedThreshold:
    def _controller(self, tmp_path, model, calibrated):
        from localist.controller_agent import ControllerAgent
        from tests.test_controller_phase4 import make_conv_agent, make_runtime

        return ControllerAgent(
            runtime=make_runtime(infer_return="no"), agents=[make_conv_agent()],
            memory_manager=MemoryManager(db_path=tmp_path / "c.db"),
            embedding_model_name=model, calibrated_thresholds=calibrated,
        )

    def _run(self, ctrl, priority: int):
        from localist.planner import RoutingPlan
        ctrl._profile_lines = ["Lives in Portland"]
        ctrl._profile_embeddings = [[1.0, 0.0]]
        plan = RoutingPlan(agent="conversational_agent", fetch_episodic=False, fetch_rag=False, priority=priority)
        with patch.object(ctrl._planner, "route", return_value=plan), \
             patch.object(EpisodicMemoryReader, "by_recency", return_value=[]), \
             patch.object(EpisodicMemoryReader, "by_similarity", return_value=[]), \
             patch.object(ctrl, "_embed", return_value=[1.0, 0.0]), \
             patch.object(ctrl, "_score_profile_facts", wraps=ctrl._score_profile_facts) as score:
            ctrl.handle_task({"instruction": "where do I live"})
        return score

    def test_unlisted_model_uses_its_calibrated_threshold(self, tmp_path):
        ctrl = self._controller(tmp_path, "unlisted-model", {"profile_fact_threshold": 0.72})
        score = self._run(ctrl, priority=4)
        assert score.call_args.kwargs["threshold"] == 0.72

    def test_hand_validated_model_ignores_calibration(self, tmp_path):
        ctrl = self._controller(tmp_path, "nomic-embed-text:latest", {"profile_fact_threshold": 0.30})
        score = self._run(ctrl, priority=4)
        assert score.call_args.kwargs["threshold"] == 0.52

    def test_direct_answer_turns_still_score_profile_facts(self, tmp_path):
        # P6 keeps getting profile facts, bound by the threshold (user
        # decision 2026-10-20; §3.6 updated to match).
        ctrl = self._controller(tmp_path, "unlisted-model", {"profile_fact_threshold": 0.72})
        score = self._run(ctrl, priority=6)
        score.assert_called_once()
        assert score.call_args.kwargs["threshold"] == 0.72


class TestScoreProfileFactsHonoursThreshold:
    def test_line_below_threshold_is_dropped(self, tmp_path):
        from localist.controller_agent import ControllerAgent
        from tests.test_controller_phase4 import make_conv_agent, make_runtime

        ctrl = ControllerAgent(runtime=make_runtime(), agents=[make_conv_agent()],
                               memory_manager=MemoryManager(db_path=tmp_path / "s.db"))
        ctrl._profile_lines = ["close", "far"]
        ctrl._profile_embeddings = [[0.6, 0.8], [0.48, 0.877]]   # cosines vs [1, 0]: 0.60, ≈0.48
        facts = ctrl._score_profile_facts([1.0, 0.0], threshold=0.52)
        assert [f.content for f in facts] == ["close"]


class TestCalibrationResponseReportsTheThresholdInUse:
    @pytest.mark.parametrize("model, calibrated, in_use", [
        ("nomic-embed-text:latest", 0.30, 0.52),   # hand-validated beats calibration
        ("unlisted-model",          0.72, 0.72),
    ])
    def test_in_use_follows_precedence(self, model, calibrated, in_use):
        from localist import main
        resp = main._calibration_result_to_response(
            model, CalibrationResult(profile_threshold=_clean(calibrated)),
        )
        assert resp.profile_threshold.threshold == calibrated
        assert resp.profile_threshold_in_use == in_use

    def test_degenerate_falls_back_to_default(self):
        from localist import main
        resp = main._calibration_result_to_response(
            "unlisted-model", CalibrationResult(profile_threshold=_DEGENERATE),
        )
        assert resp.profile_threshold.degenerate is True
        assert resp.profile_threshold_in_use == 0.45
