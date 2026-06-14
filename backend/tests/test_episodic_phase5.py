"""
Phase 5 integration tests — LORA episodic extraction pipeline.

Covers:
  5.1 — Deterministic signal detection: all trigger categories,
        retraction, no-signal, case-insensitivity
  5.2 — Model-based extraction: content returned, NONE handling,
        inference failure graceful degradation
  5.3 — Confidence scoring: all five score bands
  5.4 — ControllerAgent wiring:
          explicit path  → process_explicit_signal() called, DB written
          implicit path  → post-response hook fires, DB written
          suppression    → write_episode=True prevents implicit hook
          retraction     → retract() called, no insert

All DB tests use real SQLite via tmp_path. Runtime calls use MagicMock.
"""

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memory_manager import (
    MemoryManager,
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
)
from episodic_extractor import (
    detect_explicit_signal,
    score_model_extraction,
    extract_content_from_instruction,
    extract_implicit_episode,
    process_explicit_signal,
    process_implicit_extraction,
    ExtractionSignal,
    ExtractionResult,
    _infer_type_from_content,
)
from controller_agent import ControllerAgent, TaskStatus, AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    MemoryManager(db_path=path)
    return path


@pytest.fixture()
def reader(db_path: Path) -> EpisodicMemoryReader:
    return EpisodicMemoryReader(db_path=db_path)


def make_runtime(infer_return="Test answer.", embed_return=None):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = embed_return or ([0.0] * 768)
    return rt


def make_conv_agent(answer="Test answer."):
    agent = MagicMock()
    agent.name = "conversational_agent"
    agent.can_handle.return_value = True
    agent.run.return_value = AgentResult(
        subtask_id = "t-0",
        agent_name = "conversational_agent",
        status     = TaskStatus.COMPLETE,
        output     = {"answer": answer, "sources": [], "grounded": False},
    )
    return agent


def all_active_episodes(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM episodes WHERE status='active' ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# 5.1 — Deterministic signal detection
# ---------------------------------------------------------------------------

class TestDetectExplicitSignal:

    def test_retraction_forget_that(self):
        sig = detect_explicit_signal("forget that preference about formatting")
        assert sig is not None
        assert sig.is_retraction is True
        assert sig.confidence    == 1.0

    def test_retraction_no_longer_true(self):
        sig = detect_explicit_signal("that's no longer true about the vault resolver")
        assert sig is not None
        assert sig.is_retraction is True

    def test_retraction_beats_insert_signal(self):
        """Retraction phrases checked before insert phrases."""
        sig = detect_explicit_signal("forget that — my preference is dark mode")
        assert sig.is_retraction is True

    def test_preference_remember_that(self):
        sig = detect_explicit_signal("remember that I prefer step-by-step instructions")
        assert sig is not None
        assert sig.is_retraction  is False
        assert sig.episode_type   == "preference"
        assert sig.source         == "explicit"
        assert sig.confidence     == 1.0

    def test_preference_my_preference_is(self):
        sig = detect_explicit_signal("my preference is to always use type hints")
        assert sig.episode_type == "preference"

    def test_correction_thats_wrong(self):
        sig = detect_explicit_signal("That's wrong — raw_path comes from context")
        assert sig.episode_type == "correction"

    def test_correction_correct_value(self):
        sig = detect_explicit_signal("the correct value is 768 dimensions")
        assert sig.episode_type == "correction"

    def test_task_completion_mark_complete(self):
        sig = detect_explicit_signal("mark complete: file ingestion pipeline")
        assert sig.episode_type == "task_completion"

    def test_task_completion_thats_done(self):
        sig = detect_explicit_signal("that's done — migration is finished")
        assert sig.episode_type == "task_completion"

    def test_decision_we_decided(self):
        sig = detect_explicit_signal("we decided to use SQLite for the memory layer")
        assert sig.episode_type == "decision"

    def test_workflow_always(self):
        sig = detect_explicit_signal("always upload source files before reviewing")
        assert sig.episode_type == "workflow"

    def test_project_fact_note_that(self):
        sig = detect_explicit_signal("note that oMLX runs on port 8080")
        assert sig.episode_type == "project_fact"

    def test_naming_convention_should_be_called(self):
        sig = detect_explicit_signal("it should be called oMLX not OMLX")
        assert sig.episode_type == "naming_convention"

    def test_no_signal_returns_none(self):
        assert detect_explicit_signal("What is the capital of France?") is None
        assert detect_explicit_signal("How does embedding work?")        is None
        assert detect_explicit_signal("summarise this document")         is None

    def test_case_insensitive(self):
        sig = detect_explicit_signal("REMEMBER THAT I PREFER DARK MODE")
        assert sig is not None
        assert sig.episode_type == "preference"


# ---------------------------------------------------------------------------
# 5.3 — Confidence scoring
# ---------------------------------------------------------------------------

class TestScoreModelExtraction:

    def test_empty_string_zero(self):
        assert score_model_extraction("") == 0.0

    def test_none_string_zero(self):
        assert score_model_extraction("NONE") == 0.0

    def test_none_lowercase_zero(self):
        assert score_model_extraction("none") == 0.0

    def test_hedging_might(self):
        assert score_model_extraction(
            "The user might prefer a dark colour scheme."
        ) == 0.6

    def test_hedging_perhaps(self):
        assert score_model_extraction(
            "Perhaps the user prefers concise answers."
        ) == 0.6

    def test_short_response_point_seven(self):
        # Fewer than 7 words
        assert score_model_extraction("User prefers dark mode.") == 0.7

    def test_proper_noun_point_nine(self):
        # "oMLX" starts with lowercase 'o' — not a proper noun by the isupper() rule.
        # Use "SQLite" (uppercase S) to reliably trigger the proper-noun branch.
        assert score_model_extraction(
            "User prefers SQLite as the persistent storage backend."
        ) == 0.9

    def test_number_point_nine(self):
        assert score_model_extraction(
            "The team completes 4 review cycles per release."
        ) == 0.9

    def test_default_point_eight(self):
        assert score_model_extraction(
            "The user prefers step-by-step instructions over inline diffs."
        ) == 0.8


# ---------------------------------------------------------------------------
# 5.2 — Model-based extraction
# ---------------------------------------------------------------------------

class TestExtractContentFromInstruction:

    def test_returns_content_and_confidence(self):
        rt = make_runtime(
            infer_return="User prefers step-by-step swap instructions."
        )
        content, confidence = extract_content_from_instruction(
            instruction  = "remember that I prefer step-by-step instructions",
            episode_type = "preference",
            runtime      = rt,
        )
        assert content    != ""
        assert 0.6 <= confidence <= 0.9

    def test_none_response_returns_empty(self):
        rt = make_runtime(infer_return="NONE")
        content, confidence = extract_content_from_instruction(
            "hi", "preference", rt
        )
        assert content    == ""
        assert confidence == 0.0

    def test_inference_failure_returns_empty(self):
        rt = MagicMock()
        rt.infer.side_effect = Exception("model offline")
        content, confidence = extract_content_from_instruction(
            "remember that I prefer dark mode", "preference", rt
        )
        assert content    == ""
        assert confidence == 0.0

    def test_prompt_contains_instruction(self):
        """Direct prompt construction includes the raw instruction text."""
        rt = make_runtime(infer_return="User prefers dark mode.")
        extract_content_from_instruction(
            "remember that I prefer dark mode", "preference", rt
        )
        call_prompt = rt.infer.call_args.kwargs.get("prompt") or \
                      rt.infer.call_args.args[0]
        assert "remember that I prefer dark mode" in call_prompt


class TestExtractImplicitEpisode:

    def test_returns_triple_on_hit(self):
        rt = make_runtime(
            infer_return=(
                "User always uploads source files before accepting "
                "generated code."
            )
        )
        result = extract_implicit_episode(
            "I'm building a project, can you review this code?",
            "Here is my review...",
            rt,
        )
        assert result is not None
        episode_type, content, confidence = result
        assert episode_type in {
            "preference", "correction", "decision", "workflow",
            "project_fact", "task_completion", "naming_convention",
        }
        assert content    != ""
        assert 0.6 <= confidence <= 0.9

    def test_none_response_returns_none(self):
        rt = make_runtime(infer_return="NONE")
        result = extract_implicit_episode("hi", "hello", rt)
        assert result is None

    def test_inference_failure_returns_none(self):
        rt = MagicMock()
        rt.infer.side_effect = Exception("timeout")
        result = extract_implicit_episode("hi", "hello", rt)
        assert result is None


class TestInferTypeFromContent:

    def test_preference(self):
        assert _infer_type_from_content(
            "User prefers step-by-step instructions."
        ) == "preference"

    def test_correction(self):
        assert _infer_type_from_content(
            "raw_path should be passed explicitly — the fuzzy match was wrong."
        ) == "correction"

    def test_decision(self):
        assert _infer_type_from_content(
            "The team decided to use SQLite for memory."
        ) == "decision"

    def test_workflow(self):
        assert _infer_type_from_content(
            "User always reviews source files before accepting code."
        ) == "workflow"

    def test_naming_convention(self):
        assert _infer_type_from_content(
            "The runtime is called oMLX, not OMLX."
        ) == "naming_convention"

    def test_task_completion(self):
        assert _infer_type_from_content(
            "The ingestion pipeline is completed and working."
        ) == "task_completion"

    def test_default_project_fact(self):
        assert _infer_type_from_content(
            "The embedding dimension is 768 floats."
        ) == "project_fact"


# ---------------------------------------------------------------------------
# process_explicit_signal — end-to-end with real DB
# ---------------------------------------------------------------------------

class TestProcessExplicitSignal:

    def test_writes_preference_episode(self, db_path, reader):
        rt = make_runtime(
            infer_return="User prefers step-by-step instructions over diffs."
        )
        result = process_explicit_signal(
            instruction     = "remember that I prefer step-by-step instructions",
            runtime         = rt,
            db_path         = db_path,
            project_context = "LORA",
        )
        assert result is not None
        assert result.episode_type == "preference"
        assert result.source       == "explicit"
        assert result.confidence   == 1.0

        records = reader.by_recency(project_context="LORA")
        assert len(records) == 1
        assert records[0].episode_type == "preference"
        assert records[0].confidence   == 1.0
        assert records[0].source       == "explicit"

    def test_no_signal_returns_none(self, db_path):
        rt = make_runtime()
        result = process_explicit_signal(
            instruction = "What is the capital of France?",
            runtime     = rt,
            db_path     = db_path,
        )
        assert result is None
        assert all_active_episodes(db_path) == []

    def test_retraction_returns_none_calls_retract(self, db_path, reader):
        # First write a record to retract
        writer = EpisodicMemoryWriter(db_path=db_path)
        writer.insert(
            episode_type    = "preference",
            subject         = "output format",
            content         = "User prefers dark mode.",
            source          = "explicit",
            project_context = "LORA",
        )
        assert len(reader.by_recency(project_context="LORA")) == 1

        # retract() does exact subject matching; the model extracts the
        # subject being retracted so it can match the stored record.
        rt = make_runtime(infer_return="output format")
        result = process_explicit_signal(
            instruction     = "forget that preference about output format",
            runtime         = rt,
            db_path         = db_path,
            project_context = "LORA",
        )
        assert result is None
        # The record should now be retracted (not active)
        assert reader.by_recency(project_context="LORA") == []

    def test_model_none_response_returns_none(self, db_path):
        rt = make_runtime(infer_return="NONE")
        result = process_explicit_signal(
            instruction = "remember that I prefer dark mode",
            runtime     = rt,
            db_path     = db_path,
        )
        assert result is None
        assert all_active_episodes(db_path) == []

    def test_explicit_confidence_always_1(self, db_path):
        """Confidence is always 1.0 for explicit signals regardless of
        model extraction confidence scoring."""
        rt = make_runtime(
            infer_return="User prefers concise answers."   # short → 0.7 normally
        )
        result = process_explicit_signal(
            instruction = "remember that I prefer concise answers",
            runtime     = rt,
            db_path     = db_path,
        )
        assert result is not None
        assert result.confidence == 1.0
        episodes = all_active_episodes(db_path)
        assert episodes[0]["confidence"] == 1.0

    def test_supersession_on_duplicate_subject(self, db_path):
        rt = make_runtime(
            infer_return="User prefers step-by-step instructions."
        )
        # First insert
        process_explicit_signal(
            instruction = "remember that I prefer step-by-step instructions",
            runtime     = rt,
            db_path     = db_path,
        )
        # Second insert — same subject, same type → supersedes
        rt2 = make_runtime(
            infer_return="User prefers unified diffs over step-by-step."
        )
        process_explicit_signal(
            instruction = "remember that I prefer step-by-step instructions",
            runtime     = rt2,
            db_path     = db_path,
        )
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT status FROM episodes ORDER BY id"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0]["status"] == "superseded"
        assert rows[1]["status"] == "active"


# ---------------------------------------------------------------------------
# process_implicit_extraction — end-to-end with real DB
# ---------------------------------------------------------------------------

class TestProcessImplicitExtraction:

    def test_writes_model_extracted_episode(self, db_path):
        rt = make_runtime(
            infer_return=(
                "User always reviews source files before "
                "accepting generated code."
            )
        )
        result = process_implicit_extraction(
            instruction     = "I'm building a project, can you review this code?",
            response        = "Here is my review...",
            runtime         = rt,
            db_path         = db_path,
            project_context = "LORA",
        )
        assert result is not None
        assert result.source == "model_extracted"
        assert 0.6 <= result.confidence <= 0.9

        episodes = all_active_episodes(db_path)
        assert len(episodes) == 1
        assert episodes[0]["source"] == "model_extracted"

    def test_none_response_writes_nothing(self, db_path):
        rt = make_runtime(infer_return="NONE")
        result = process_implicit_extraction(
            "hi", "hello", rt, db_path
        )
        assert result is None
        assert all_active_episodes(db_path) == []

    def test_low_confidence_discarded(self, db_path):
        """Score of 0.0 (NONE) and below 0.6 are both discarded."""
        rt = make_runtime(infer_return="NONE")
        result = process_implicit_extraction(
            "What is 2+2?", "4.", rt, db_path
        )
        assert result is None
        assert all_active_episodes(db_path) == []


# ---------------------------------------------------------------------------
# 5.4 — ControllerAgent wiring
# ---------------------------------------------------------------------------

class TestControllerWiring:

    def test_explicit_signal_writes_to_db(self, db_path, reader):
        mm = MemoryManager(db_path=db_path)
        rt = MagicMock()
        rt.embed.return_value = [0.0] * 768
        # Only one infer call expected: explicit extraction
        rt.infer.return_value = (
            "User prefers step-by-step swap instructions over diffs."
        )

        conv = make_conv_agent("Here are the steps...")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        result = ctrl.handle_task({
            "instruction": "remember that I prefer step-by-step instructions",
            "context":     {"project_context": "LORA"},
        })

        assert result["status"] == "complete"
        records = reader.by_recency(project_context="LORA")
        assert len(records) >= 1
        assert records[0].source     == "explicit"
        assert records[0].confidence == 1.0

    def test_implicit_hook_fires_on_plain_query(self, db_path):
        mm = MemoryManager(db_path=db_path)
        rt = MagicMock()
        rt.embed.return_value = [0.0] * 768
        rt.infer.side_effect = [
            # implicit extraction call (P5 is now keyword-based, no infer)
            "User always reviews source files before accepting generated code.",
        ]

        conv = make_conv_agent("Here is my code review.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "I'm building a project, can you review this code?",
            "context":     {"project_context": "LORA"},
        })

        episodes = all_active_episodes(db_path)
        assert len(episodes) >= 1
        assert episodes[0]["source"] == "model_extracted"

    def test_implicit_hook_suppressed_when_write_episode_true(self, db_path):
        """When write_episode=True (explicit signal), implicit hook must not run."""
        mm = MemoryManager(db_path=db_path)
        rt = MagicMock()
        rt.embed.return_value = [0.0] * 768
        # Only explicit extraction call — no implicit
        rt.infer.return_value = (
            "User prefers step-by-step instructions over diffs."
        )

        conv = make_conv_agent("Here are the steps.")
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({
            "instruction": "remember that I prefer step-by-step instructions",
            "context":     {"project_context": "LORA"},
        })

        # Only one infer call (explicit extraction); no second call for implicit
        assert rt.infer.call_count == 1

    def test_failed_agent_suppresses_implicit_hook(self, db_path):
        """If agent fails, post-response hook must not run."""
        mm = MemoryManager(db_path=db_path)
        rt = MagicMock()
        rt.embed.return_value = [0.0] * 768
        rt.infer.return_value = "no"   # P5 returns no

        conv = MagicMock()
        conv.name = "conversational_agent"
        conv.can_handle.return_value = True
        conv.run.return_value = AgentResult(
            subtask_id = "t-0",
            agent_name = "conversational_agent",
            status     = TaskStatus.FAILED,
            output     = {},
            error      = "Agent crashed.",
        )

        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=mm)
        ctrl.handle_task({"instruction": "What is LORA?"})

        assert all_active_episodes(db_path) == []

    def test_no_memory_manager_no_extraction(self):
        """Without MemoryManager, extraction is silently skipped."""
        rt = MagicMock()
        rt.embed.return_value = [0.0] * 768
        rt.infer.return_value = "no"

        conv = make_conv_agent()
        ctrl = ControllerAgent(runtime=rt, agents=[conv], memory_manager=None)
        result = ctrl.handle_task({
            "instruction": "remember that I prefer dark mode",
        })
        assert result["status"] == "complete"
