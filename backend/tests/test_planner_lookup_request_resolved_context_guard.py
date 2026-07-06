"""
lookup_request resolved-context guard — feature-flagged, shadow-first.

Trigger: diagnostics/score_referential_followup_lookup_request.py found that
9/10 referential/follow-up phrases ("What do you make of all that?", "What
does that mean?", etc.) clear the lookup_request 0.60 gate, in the same band
as genuine lookup_request true positives — while explicit_search_action
stays well clear of its own 0.72 gate for the whole category (max 0.6247).
This guard withholds web_search when lookup_request is the *only* reason it
would fire AND the prior turn already fired a tool (self._last_turn_tools_fired,
the same state tracked for the P6-fallthrough classifier's Gate 1).

Covers:
  - Flag off (default): zero behavior change, even for the exact referential
    phrases the guard targets and even with a prior tool fired.
  - Literal keyword match is untouched regardless of prior-turn state or mode
    (the guard's suppression branch is never reached when a literal keyword
    already put "web_search" in tools).
  - explicit_search_action independently clearing its own gate is untouched
    regardless of prior-turn state or mode.
  - Active mode suppresses web_search when lookup_request fires alone AND
    the prior turn already fired a tool — using the real referential phrases
    from the diagnostic as fixtures (live-measured scores baked in, not
    synthetic stand-ins).
  - The guard does NOT suppress when no tool fired on the prior turn — the
    original lookup_request gate's designed case, which must remain intact.
  - Shadow mode evaluates and logs but produces identical tools_to_call to
    flag-off, for the same instruction/history pair.

All tests construct their own Planner instance (via _planner_with_mocked_semantic,
same pattern as test_planner_phase3.py's TestPriority3SemanticGating) and
manage the LOCALIST_LOOKUP_REQUEST_RESOLVED_CONTEXT_GUARD env var via
monkeypatch, so no state leaks between tests or into the rest of the suite.
"""

import logging
from unittest.mock import MagicMock

import pytest

from planner import Planner

_ENV_VAR = "LOCALIST_LOOKUP_REQUEST_RESOLVED_CONTEXT_GUARD"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_runtime(infer_return: str = "none"):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = [0.0] * 768
    return rt


def _planner_with_mocked_semantic(all_scores: dict[str, float]) -> "Planner":
    """
    Same helper as test_planner_phase3.py's TestPriority3SemanticGating
    (duplicated locally — this file is self-contained, matching the
    test_planner_tool_fallback_classifier.py convention): patches
    _semantic_search_intent to return fixed scores directly, bypassing
    embed_fn, for precise control over which group(s) cross their gate.
    """
    p = Planner(runtime=make_runtime())
    best_group = max(all_scores, key=lambda g: all_scores[g])
    best_score = all_scores[best_group]
    p._semantic_search_intent = MagicMock(
        return_value=(best_group, best_score, all_scores)
    )
    return p


# Real referential/follow-up phrases + live-measured scores from
# diagnostics/score_referential_followup_lookup_request.py (2026-07-04 run
# against the real mlx-community/embeddinggemma-300m-4bit model). Both cross
# lookup_request (>=0.60) but stay well clear of explicit_search_action
# (>=0.72, max here 0.6247) — the exact shape the guard targets: LR fires
# alone. Baked in as fixed scores (rather than re-embedding live) so this
# suite stays fast and deterministic, per the mocked-semantic pattern already
# established in test_planner_phase3.py.
_REFERENTIAL_FIXTURES: list[tuple[str, dict[str, float]]] = [
    (
        "What do you make of all that?",
        {
            "explicit_search_action": 0.5721,
            "lookup_request":         0.6158,
            "knowledge_request_open": 0.7421,
            "freshness_request":      0.6519,
        },
    ),
    (
        "What does that mean?",
        {
            "explicit_search_action": 0.6247,
            "lookup_request":         0.7028,
            "knowledge_request_open": 0.7903,
            "freshness_request":      0.6518,
        },
    ),
]


# ---------------------------------------------------------------------------
# Flag off (default) — zero behavior change
# ---------------------------------------------------------------------------

class TestFlagOff:

    def test_default_env_unset_is_off(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        p = Planner(runtime=make_runtime())
        assert p._resolved_context_guard_mode() == "off"

    def test_unrecognized_env_value_treated_as_off(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "banana")
        p = Planner(runtime=make_runtime())
        assert p._resolved_context_guard_mode() == "off"

    @pytest.mark.parametrize("utterance,scores", _REFERENTIAL_FIXTURES)
    def test_off_adds_web_search_even_with_prior_tool_fired(
        self, monkeypatch, utterance, scores
    ):
        """Flag off => zero behavior change, even for the exact referential
        phrases the guard targets and even with a prior tool fired."""
        monkeypatch.delenv(_ENV_VAR, raising=False)
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])  # prior turn fired a tool
        result = p._priority3_tool(utterance.lower())
        assert result is not None
        assert "web_search" in result.tools_to_call


# ---------------------------------------------------------------------------
# Literal keyword — untouched under all conditions
# ---------------------------------------------------------------------------

class TestLiteralKeywordUntouched:

    def test_literal_keyword_not_suppressed_active_mode_prior_tool_fired(
        self, monkeypatch
    ):
        """"latest" is a literal _WEB_SEARCH_KEYWORDS match, so tools already
        contains "web_search" before the semantic/guard block ever runs —
        the guard's suppression branch must never be reached."""
        monkeypatch.setenv(_ENV_VAR, "active")
        scores = {
            "explicit_search_action": 0.30,
            "lookup_request":         0.65,   # would also fire alone
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        }
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])  # prior turn fired a tool
        result = p._priority3_tool("what's the latest on that")
        assert result is not None
        assert "web_search" in result.tools_to_call


# ---------------------------------------------------------------------------
# explicit_search_action independent gate crossing — untouched
# ---------------------------------------------------------------------------

class TestExplicitSearchActionUntouched:

    def test_esa_alone_not_suppressed_active_mode_prior_tool_fired(
        self, monkeypatch
    ):
        monkeypatch.setenv(_ENV_VAR, "active")
        scores = {
            "explicit_search_action": 0.90,
            "lookup_request":         0.30,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        }
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])
        result = p._priority3_tool("can you check the internet for that")
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_esa_and_lr_both_fire_not_suppressed_active_mode_prior_tool_fired(
        self, monkeypatch
    ):
        """When explicit_search_action ALSO independently clears its gate
        (in addition to lookup_request), the guard must not suppress —
        lookup_request is not the sole justification here."""
        monkeypatch.setenv(_ENV_VAR, "active")
        scores = {
            "explicit_search_action": 0.90,
            "lookup_request":         0.65,
            "knowledge_request_open": 0.20,
            "freshness_request":      0.10,
        }
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])
        result = p._priority3_tool("can you check the internet for that")
        assert result is not None
        assert "web_search" in result.tools_to_call


# ---------------------------------------------------------------------------
# Active mode — suppresses lookup_request-only + prior tool fired
# ---------------------------------------------------------------------------

class TestActiveModeSuppression:

    @pytest.mark.parametrize("utterance,scores", _REFERENTIAL_FIXTURES)
    def test_active_suppresses_when_lookup_request_alone_and_prior_tool_fired(
        self, monkeypatch, utterance, scores
    ):
        monkeypatch.setenv(_ENV_VAR, "active")
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])  # prior turn fired a tool
        result = p._priority3_tool(utterance.lower())
        # No literal keyword and no file_op/url_fetch signal in these
        # utterances, so with web_search suppressed, tools ends up empty
        # and _priority3_tool returns None — the turn falls through toward
        # P3b/P4/P5/(P6-fallthrough classifier)/P6.
        assert result is None

    def test_active_suppression_falls_through_to_p6_via_route(self, monkeypatch):
        """End-to-end: route() on a referential phrase with a prior tool
        fired lands on the P6 direct-answer fallback, not web_search."""
        monkeypatch.setenv(_ENV_VAR, "active")
        utterance, scores = _REFERENTIAL_FIXTURES[0]
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])
        plan = p.route(utterance, context={})
        assert plan.tools_to_call == []
        assert plan.priority == 6


# ---------------------------------------------------------------------------
# No prior tool fired — guard must not suppress (original gate's own case)
# ---------------------------------------------------------------------------

class TestNoSuppressionWithoutPriorTool:

    @pytest.mark.parametrize("utterance,scores", _REFERENTIAL_FIXTURES)
    def test_active_does_not_suppress_when_no_prior_tool_fired(
        self, monkeypatch, utterance, scores
    ):
        monkeypatch.setenv(_ENV_VAR, "active")
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired([])  # first-turn-in-conversation / non-tool prior turn
        result = p._priority3_tool(utterance.lower())
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_fresh_planner_has_no_prior_tool_state_by_default(self, monkeypatch):
        """A newly constructed Planner (no record_tools_fired call at all)
        must behave identically to the explicit empty-list case above."""
        monkeypatch.setenv(_ENV_VAR, "active")
        utterance, scores = _REFERENTIAL_FIXTURES[0]
        p = _planner_with_mocked_semantic(scores)
        result = p._priority3_tool(utterance.lower())
        assert result is not None
        assert "web_search" in result.tools_to_call

    def test_active_evaluates_and_logs_without_suppressing(
        self, monkeypatch, caplog
    ):
        """Guard evaluates (active mode, lookup_request-only, no prior tool
        fired) and must log the evaluation even though it doesn't suppress —
        without this line, "guard never ran" and "guard ran but declined"
        are indistinguishable from logs alone."""
        monkeypatch.setenv(_ENV_VAR, "active")
        utterance, scores = _REFERENTIAL_FIXTURES[0]
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired([])  # no prior tool fired

        with caplog.at_level(logging.INFO, logger="planner"):
            result = p._priority3_tool(utterance.lower())

        assert result is not None
        assert "web_search" in result.tools_to_call
        assert any(
            "resolved-context guard (active) — evaluated, NOT suppressing"
            in r.message
            for r in caplog.records
        )
        assert not any(
            "suppressing web_search:" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Shadow mode — evaluates and logs, RoutingPlan unaffected
# ---------------------------------------------------------------------------

class TestShadowMode:

    @pytest.mark.parametrize("utterance,scores", _REFERENTIAL_FIXTURES)
    def test_shadow_identical_to_off_for_same_instruction_and_history(
        self, monkeypatch, utterance, scores
    ):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        p_off = _planner_with_mocked_semantic(scores)
        p_off.record_tools_fired(["web_search"])
        result_off = p_off._priority3_tool(utterance.lower())

        monkeypatch.setenv(_ENV_VAR, "shadow")
        p_shadow = _planner_with_mocked_semantic(scores)
        p_shadow.record_tools_fired(["web_search"])
        result_shadow = p_shadow._priority3_tool(utterance.lower())

        assert result_off is not None and result_shadow is not None
        assert result_off.tools_to_call == result_shadow.tools_to_call == ["web_search"]

    def test_shadow_runs_evaluation_and_logs_but_does_not_suppress(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(_ENV_VAR, "shadow")
        utterance, scores = _REFERENTIAL_FIXTURES[0]
        p = _planner_with_mocked_semantic(scores)
        p.record_tools_fired(["web_search"])

        with caplog.at_level(logging.INFO, logger="planner"):
            result = p._priority3_tool(utterance.lower())

        assert result is not None
        assert "web_search" in result.tools_to_call
        assert any(
            "resolved-context guard (shadow)" in r.message
            and "would_suppress=True" in r.message
            for r in caplog.records
        )
