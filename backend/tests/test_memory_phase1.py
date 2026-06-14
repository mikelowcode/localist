"""
Phase 1 unit tests — LORA episodic memory substrate.

Covers:
  - EpisodicMemoryWriter: lifecycle transitions (insert, supersede, retract)
  - EpisodicMemoryReader: all three retrieval modes + last_accessed touch
  - format_episodic_summary: filtering, ordering, truncation, edge cases
  - MemoryManager.get_context_window: max_tokens ceiling enforcement

Each test class uses a fresh temporary SQLite DB (pytest tmp_path fixture)
so tests are fully isolated and leave no state on disk.
"""

import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from memory_manager import (
    MemoryManager,
    EpisodicMemoryWriter,
    EpisodicMemoryReader,
    EpisodeRecord,
    format_episodic_summary,
    VALID_EPISODE_TYPES,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Fresh, schema-initialised SQLite DB for each test."""
    path = tmp_path / "test_lora.db"
    MemoryManager(db_path=path)   # runs _init_db → schema v2
    return path


@pytest.fixture()
def writer(db_path: Path) -> EpisodicMemoryWriter:
    return EpisodicMemoryWriter(db_path=db_path)


@pytest.fixture()
def reader(db_path: Path) -> EpisodicMemoryReader:
    return EpisodicMemoryReader(db_path=db_path)


@pytest.fixture()
def mm(db_path: Path) -> MemoryManager:
    return MemoryManager(db_path=db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(db_path: Path, episode_id: int) -> sqlite3.Row:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    conn.close()
    return row


def _all_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM episodes ORDER BY id").fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# EpisodicMemoryWriter
# ---------------------------------------------------------------------------

class TestEpisodicMemoryWriter:

    def test_insert_returns_positive_id(self, writer):
        id_ = writer.insert(
            episode_type="preference",
            subject="test subject",
            content="Some content.",
            source="explicit",
        )
        assert isinstance(id_, int)
        assert id_ > 0

    def test_insert_sets_active_status(self, writer, db_path):
        id_ = writer.insert(
            episode_type="decision",
            subject="arch choice",
            content="Chose SQLite.",
            source="explicit",
        )
        row = _row(db_path, id_)
        assert row["status"] == "active"

    def test_insert_sets_confidence_default(self, writer, db_path):
        id_ = writer.insert(
            episode_type="preference",
            subject="conf default",
            content="Content.",
            source="explicit",
        )
        row = _row(db_path, id_)
        assert row["confidence"] == 1.0

    def test_insert_custom_confidence(self, writer, db_path):
        id_ = writer.insert(
            episode_type="project_fact",
            subject="runtime",
            content="oMLX 0.4.2.",
            source="model_extracted",
            confidence=0.85,
        )
        row = _row(db_path, id_)
        assert abs(row["confidence"] - 0.85) < 1e-6

    def test_supersession_on_duplicate_subject_type(self, writer, db_path):
        id1 = writer.insert(
            episode_type="preference",
            subject="format",
            content="Original preference.",
            source="explicit",
        )
        id2 = writer.insert(
            episode_type="preference",
            subject="format",
            content="Updated preference.",
            source="explicit",
        )
        assert id2 > id1
        row1 = _row(db_path, id1)
        row2 = _row(db_path, id2)
        assert row1["status"] == "superseded"
        assert row2["status"] == "active"

    def test_supersession_audit_trail_preserved(self, writer, db_path):
        """Both old and new records must be present after supersession."""
        writer.insert("correction", "subject A", "Old content.", "explicit")
        writer.insert("correction", "subject A", "New content.", "explicit")
        rows = _all_rows(db_path)
        assert len(rows) == 2
        statuses = {r["status"] for r in rows}
        assert statuses == {"active", "superseded"}

    def test_different_type_same_subject_no_supersession(self, writer, db_path):
        """Same subject but different episode_type must NOT supersede each other."""
        writer.insert("preference", "overlap subject", "Pref content.", "explicit")
        writer.insert("correction", "overlap subject", "Corr content.", "explicit")
        rows = _all_rows(db_path)
        assert all(r["status"] == "active" for r in rows)

    def test_retract_marks_active_as_retracted(self, writer, db_path):
        id_ = writer.insert(
            episode_type="workflow",
            subject="review process",
            content="Always reviews source first.",
            source="explicit",
        )
        count = writer.retract(subject="review process", episode_type="workflow")
        assert count == 1
        assert _row(db_path, id_)["status"] == "retracted"

    def test_retract_returns_zero_when_no_match(self, writer):
        count = writer.retract(subject="nonexistent", episode_type="preference")
        assert count == 0

    def test_retract_does_not_affect_other_records(self, writer, db_path):
        id1 = writer.insert("preference", "subj A", "Content A.", "explicit")
        id2 = writer.insert("preference", "subj B", "Content B.", "explicit")
        writer.retract(subject="subj A", episode_type="preference")
        assert _row(db_path, id1)["status"] == "retracted"
        assert _row(db_path, id2)["status"] == "active"

    def test_invalid_episode_type_raises(self, writer):
        with pytest.raises(ValueError, match="episode_type"):
            writer.insert("bogus_type", "s", "c", "explicit")

    def test_invalid_source_raises(self, writer):
        with pytest.raises(ValueError, match="source"):
            writer.insert("preference", "s", "c", "bad_source")

    def test_confidence_above_1_raises(self, writer):
        with pytest.raises(ValueError, match="confidence"):
            writer.insert("preference", "s", "c", "explicit", confidence=1.1)

    def test_confidence_below_0_raises(self, writer):
        with pytest.raises(ValueError, match="confidence"):
            writer.insert("preference", "s", "c", "explicit", confidence=-0.1)

    def test_all_valid_episode_types_accepted(self, writer):
        for ep_type in VALID_EPISODE_TYPES:
            id_ = writer.insert(ep_type, f"subject_{ep_type}", "Content.", "explicit")
            assert id_ > 0


# ---------------------------------------------------------------------------
# EpisodicMemoryReader
# ---------------------------------------------------------------------------

class TestEpisodicMemoryReader:

    def _seed(self, writer: EpisodicMemoryWriter, project: str = "LORA"):
        """Insert a standard set of episodes for retrieval tests."""
        writer.insert("correction",   "vault resolver",  "raw_path passed explicitly.",     "explicit",        confidence=1.0, project_context=project)
        writer.insert("preference",   "output format",   "Step-by-step swap preferred.",    "explicit",        confidence=1.0, project_context=project)
        writer.insert("decision",     "memory backend",  "SQLite committed.",               "explicit",        confidence=1.0, project_context=project)
        writer.insert("project_fact", "runtime version", "oMLX 0.4.2 Gemma 4B.",           "model_extracted", confidence=0.9, project_context=project)
        writer.insert("workflow",     "code review",     "Always uploads source first.",    "explicit",        confidence=1.0, project_context=project)

    # Mode 1 — exact subject match

    def test_by_subject_returns_matching_record(self, writer, reader):
        self._seed(writer)
        results = reader.by_subject("vault resolver")
        assert len(results) == 1
        assert results[0].episode_type == "correction"

    def test_by_subject_returns_episode_record_type(self, writer, reader):
        self._seed(writer)
        results = reader.by_subject("vault resolver")
        assert all(isinstance(r, EpisodeRecord) for r in results)

    def test_by_subject_excludes_retracted(self, writer, reader):
        self._seed(writer)
        writer.retract("vault resolver", "correction")
        assert reader.by_subject("vault resolver") == []

    def test_by_subject_excludes_superseded(self, writer, reader):
        self._seed(writer)
        # Supersede the existing correction
        writer.insert("correction", "vault resolver", "Updated content.", "explicit")
        results = reader.by_subject("vault resolver")
        assert len(results) == 1
        assert results[0].content == "Updated content."

    def test_by_subject_updates_last_accessed(self, writer, reader, db_path):
        self._seed(writer)
        before = time.time()
        results = reader.by_subject("vault resolver")
        after = time.time()
        assert results[0].last_accessed is not None
        assert before <= results[0].last_accessed <= after

    def test_by_subject_no_match_returns_empty(self, reader):
        assert reader.by_subject("nonexistent subject") == []

    # Mode 2 — type-filtered recency

    def test_by_recency_returns_only_prime_types(self, writer, reader):
        self._seed(writer)
        results = reader.by_recency(project_context="LORA")
        allowed = {"preference", "correction", "decision", "workflow"}
        assert all(r.episode_type in allowed for r in results)

    def test_by_recency_scoped_to_project(self, writer, reader):
        self._seed(writer, project="LORA")
        self._seed(writer, project="OTHER")
        results = reader.by_recency(project_context="LORA")
        assert all(r.project_context == "LORA" for r in results)

    def test_by_recency_max_5_results(self, writer, reader):
        # Insert 6 prime-type records for the same project
        for i in range(6):
            writer.insert(
                "preference", f"subject_{i}", f"Content {i}.",
                "explicit", project_context="LORA",
            )
        results = reader.by_recency(project_context="LORA")
        assert len(results) <= 5

    def test_by_recency_updates_last_accessed(self, writer, reader):
        self._seed(writer)
        before = time.time()
        results = reader.by_recency(project_context="LORA")
        after = time.time()
        for r in results:
            assert r.last_accessed is not None
            assert before <= r.last_accessed <= after

    # Mode 3 — semantic similarity

    def test_by_similarity_returns_episode_records(self, writer, reader):
        self._seed(writer)
        results = reader.by_similarity("sqlite memory backend")
        assert all(isinstance(r, EpisodeRecord) for r in results)

    def test_by_similarity_top_result_most_relevant(self, writer, reader):
        self._seed(writer)
        results = reader.by_similarity("sqlite memory backend")
        assert len(results) >= 1
        # The decision about SQLite should score highest
        assert results[0].episode_type == "decision"

    def test_by_similarity_respects_top_n(self, writer, reader):
        self._seed(writer)
        results = reader.by_similarity("the", top_n=2)
        assert len(results) <= 2

    def test_by_similarity_min_score_filters(self, writer, reader):
        self._seed(writer)
        # min_score=1.0 is unreachable by keyword scoring — must return empty
        results = reader.by_similarity("sqlite", min_score=1.0)
        assert results == []

    def test_by_similarity_excludes_inactive(self, writer, reader):
        self._seed(writer)
        writer.retract("memory backend", "decision")
        results = reader.by_similarity("sqlite memory backend")
        subjects = [r.subject for r in results]
        assert "memory backend" not in subjects


# ---------------------------------------------------------------------------
# format_episodic_summary
# ---------------------------------------------------------------------------

class TestFormatEpisodicSummary:

    def _make_record(self, **kwargs) -> EpisodeRecord:
        defaults = dict(
            id=1, episode_type="preference", subject="s", content="Content.",
            confidence=1.0, source="explicit", task_id=None,
            conversation_id=None, project_context="general",
            status="active", created_at=time.time(), last_accessed=None,
        )
        defaults.update(kwargs)
        return EpisodeRecord(**defaults)

    def test_empty_input_returns_empty_string(self):
        assert format_episodic_summary([]) == ""

    def test_label_present(self):
        ep = self._make_record()
        result = format_episodic_summary([ep])
        assert result.startswith("[EPISODIC MEMORY]")

    def test_bullet_format(self):
        ep = self._make_record(
            episode_type="correction", content="Some fact.", confidence=1.0
        )
        result = format_episodic_summary([ep])
        bullets = [l for l in result.splitlines() if l.startswith("- ")]
        assert len(bullets) == 1
        assert bullets[0] == "- Some fact. (correction, 1.0)"

    def test_max_bullets_enforced(self):
        episodes = [
            self._make_record(id=i, subject=f"s{i}", content=f"Content {i}.")
            for i in range(10)
        ]
        result = format_episodic_summary(episodes)
        bullets = [l for l in result.splitlines() if l.startswith("- ")]
        assert len(bullets) <= 5

    def test_below_confidence_threshold_excluded(self):
        ep = self._make_record(content="Low confidence fact.", confidence=0.5)
        assert format_episodic_summary([ep]) == ""

    def test_at_confidence_threshold_included(self):
        ep = self._make_record(content="Exactly at threshold.", confidence=0.7)
        result = format_episodic_summary([ep])
        assert "Exactly at threshold." in result

    def test_superseded_excluded(self):
        ep = self._make_record(content="Old fact.", status="superseded")
        assert format_episodic_summary([ep]) == ""

    def test_retracted_excluded(self):
        ep = self._make_record(content="Retracted fact.", status="retracted")
        assert format_episodic_summary([ep]) == ""

    def test_priority_order_correction_before_preference(self):
        corr = self._make_record(id=1, episode_type="correction",  content="Correction.", confidence=1.0)
        pref = self._make_record(id=2, episode_type="preference",  content="Preference.", confidence=1.0)
        result = format_episodic_summary([pref, corr])   # deliberately reversed
        lines = [l for l in result.splitlines() if l.startswith("- ")]
        assert "correction" in lines[0]
        assert "preference" in lines[1]

    def test_priority_order_full_chain(self):
        types_in_priority = [
            "correction", "decision", "preference", "workflow",
            "project_fact", "naming_convention", "task_completion",
        ]
        episodes = [
            self._make_record(id=i, episode_type=t, subject=f"s{i}", content=f"{t} content.")
            for i, t in enumerate(reversed(types_in_priority))
        ]
        result = format_episodic_summary(episodes, max_bullets=7)
        lines = [l for l in result.splitlines() if l.startswith("- ")]
        returned_types = [l.split("(")[1].split(",")[0] for l in lines]
        assert returned_types == types_in_priority

    def test_long_content_truncated_with_ellipsis(self):
        ep = self._make_record(content="A" * 100)
        result = format_episodic_summary([ep])
        bullet = [l for l in result.splitlines() if l.startswith("- ")][0]
        assert "…" in bullet

    def test_short_content_not_truncated(self):
        ep = self._make_record(content="Short content.")
        result = format_episodic_summary([ep])
        assert "Short content." in result
        assert "…" not in result

    def test_annotation_format(self):
        import re
        ep = self._make_record(episode_type="decision", confidence=0.85)
        result = format_episodic_summary([ep])
        bullet = [l for l in result.splitlines() if l.startswith("- ")][0]
        assert re.search(r'\(decision, \d+\.\d\)$', bullet)

    def test_custom_max_bullets(self):
        episodes = [
            self._make_record(id=i, subject=f"s{i}", content=f"Content {i}.")
            for i in range(5)
        ]
        result = format_episodic_summary(episodes, max_bullets=2)
        bullets = [l for l in result.splitlines() if l.startswith("- ")]
        assert len(bullets) == 2

    def test_custom_min_confidence(self):
        ep = self._make_record(content="Medium confidence.", confidence=0.75)
        # With stricter threshold, this record should be excluded
        assert format_episodic_summary([ep], min_confidence=0.8) == ""
        # With looser threshold, included
        assert format_episodic_summary([ep], min_confidence=0.7) != ""


# ---------------------------------------------------------------------------
# MemoryManager.get_context_window — max_tokens ceiling
# ---------------------------------------------------------------------------

class TestGetContextWindowMaxTokens:

    def test_no_max_tokens_returns_all(self, mm):
        task = "test_no_ceiling"
        for i in range(5):
            mm.add(role="user", content="A" * 100, task_id=task)
        result = mm.get_context_window(task_id=task)
        assert len(result) == 5

    def test_within_budget_returns_all(self, mm):
        task = "test_within"
        for i in range(5):
            mm.add(role="user", content="A" * 100, task_id=task)
        # 5 × 100 chars = 500 chars = 125 tokens → fits in 300
        result = mm.get_context_window(task_id=task, max_tokens=300)
        assert len(result) == 5

    def test_tight_budget_drops_oldest(self, mm):
        task = "test_tight"
        for i in range(5):
            mm.add(role="user", content="A" * 100, task_id=task)
        # budget 60 tokens = 240 chars → 2 entries of 100 chars fit, 3 do not
        result = mm.get_context_window(task_id=task, max_tokens=60)
        assert len(result) == 2

    def test_newest_entries_survive(self, mm):
        task = "test_newest"
        contents = [f"Entry {i}: " + "A" * 90 for i in range(5)]
        for c in contents:
            mm.add(role="user", content=c, task_id=task)
        all_entries = mm.get_context_window(task_id=task)
        trimmed    = mm.get_context_window(task_id=task, max_tokens=60)
        assert trimmed == all_entries[-len(trimmed):]

    def test_chronological_order_preserved(self, mm):
        task = "test_order"
        for i in range(3):
            mm.add(role="user", content=f"Message {i}", task_id=task)
        result = mm.get_context_window(task_id=task, max_tokens=300)
        contents = [e["content"] for e in result]
        assert contents == ["Message 0", "Message 1", "Message 2"]

    def test_zero_budget_returns_empty(self, mm):
        task = "test_zero"
        mm.add(role="user", content="Any content.", task_id=task)
        result = mm.get_context_window(task_id=task, max_tokens=0)
        assert result == []

    def test_legacy_limit_arg_unaffected(self, mm):
        task = "test_legacy"
        for i in range(5):
            mm.add(role="user", content="A" * 10, task_id=task)
        result = mm.get_context_window(task_id=task, limit=3)
        assert len(result) == 3
