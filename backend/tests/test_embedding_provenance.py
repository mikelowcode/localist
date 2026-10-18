"""
Embedding-provenance tracking for MemoryManager's corpus (document_index)
and episodes stores — the confirmed follow-up named in docs/architecture/
16-runtime-backend-layer.md §16.4, applying the same detect-and-fail-safe
pattern shipped for Planner (resolve_gate_tiers() / _semantic_gating_disabled)
to stored vectors instead of threshold constants.

Split (decided): episodes auto-re-embeds in place on a detected mismatch
(small, bounded cost); the wiki/raw corpus stays a manual, explicitly-
triggered refresh via reembed_corpus() / POST /memory/reembed (potentially
large/expensive — must never silently delay or cost money at every boot).

Rows are inserted directly via raw sqlite3 (bypassing MemoryManager's own
write paths and EpisodicMemoryWriter's validation) so each scenario — fresh,
pre-existing-migration, genuine mismatch — can be constructed precisely and
independently of any embed_fn actually being called during setup.
"""

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localist import memory_manager
from localist.memory_manager import (
    MemoryManager,
    _pack_embedding,
    _unpack_embedding,
    _EMBEDDING_DIM,
    _SCHEMA_VERSION,
)


_MODEL_A = "some-embedding-model:latest"
_OTHER = "nomic-embed-text"


def _stub_embed_fn(vector_value: float = 0.1):
    """embed_fn stub returning a fixed-length, content-independent vector."""
    return MagicMock(side_effect=lambda text: [vector_value] * _EMBEDDING_DIM)


def _build_schema(path: Path) -> None:
    """Materialise a fresh, current-version schema with no provenance tracking active."""
    MemoryManager(db_path=path)


def _insert_document(path: Path, *, name: str, embedded: bool) -> None:
    conn = sqlite3.connect(str(path))
    blob = _pack_embedding([0.1] * _EMBEDDING_DIM) if embedded else None
    conn.execute(
        """
        INSERT INTO document_index
            (name, path, doc_type, content, token_set, embedding, content_hash, indexed_at)
        VALUES (?, ?, 'wiki', 'some content', 'some content', ?, 'hash', ?)
        """,
        (name, f"/wiki/{name}.md", blob, time.time()),
    )
    conn.commit()
    conn.close()


def _insert_episode(path: Path, *, subject: str, content: str = "content", embedded: bool) -> None:
    conn = sqlite3.connect(str(path))
    blob = _pack_embedding([0.1] * _EMBEDDING_DIM) if embedded else None
    conn.execute(
        """
        INSERT INTO episodes
            (episode_type, subject, content, confidence, source, status, created_at, embedding)
        VALUES ('preference', ?, ?, 1.0, 'user_explicit', 'active', ?, ?)
        """,
        (subject, content, time.time(), blob),
    )
    conn.commit()
    conn.close()


def _set_provenance(path: Path, store: str, model: str) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO embedding_provenance (store, model) VALUES (?, ?)", (store, model)
    )
    conn.commit()
    conn.close()


def _get_provenance(path: Path, store: str) -> str | None:
    conn = sqlite3.connect(str(path))
    row = conn.execute(
        "SELECT model FROM embedding_provenance WHERE store = ?", (store,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _set_cache_row_valid(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        INSERT INTO retrieval_cache (query_hash, top_n, result_json, created_at, valid)
        VALUES (?, 5, '[]', ?, 1)
        """,
        (str(uuid.uuid4()), time.time()),
    )
    conn.commit()
    conn.close()


def _cache_valid_count(path: Path) -> int:
    conn = sqlite3.connect(str(path))
    count = conn.execute("SELECT COUNT(*) FROM retrieval_cache WHERE valid = 1").fetchone()[0]
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Fresh DB, no prior embeddings
# ---------------------------------------------------------------------------

class TestFreshDatabaseNoProvenance:
    def test_no_provenance_row_created_when_no_data_exists(self, tmp_path, caplog):
        path = tmp_path / "fresh.db"
        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        assert _get_provenance(path, "corpus") is None
        assert _get_provenance(path, "episodes") is None
        assert caplog.text == ""

    def test_first_index_document_call_seeds_corpus_provenance(self, tmp_path, caplog):
        path = tmp_path / "fresh.db"
        embed_fn = _stub_embed_fn()
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)
        assert _get_provenance(path, "corpus") is None

        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            mm.index_document(
                path=tmp_path / "doc.md", doc_type="wiki", content="hello world", embed=True,
            )

        assert _get_provenance(path, "corpus") == _MODEL_A
        assert mm._corpus_stale is False
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# Migration case — pre-existing embedded rows, no provenance row at all
# ---------------------------------------------------------------------------

class TestMigrationSeeding:
    def test_corpus_pre_existing_embeddings_seeded_silently(self, tmp_path, caplog):
        path = tmp_path / "migrate.db"
        _build_schema(path)
        _insert_document(path, name="old-doc", embedded=True)
        assert _get_provenance(path, "corpus") is None

        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            mm = MemoryManager(
                db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A,
            )

        assert _get_provenance(path, "corpus") == _MODEL_A
        assert mm._corpus_stale is False
        assert caplog.text == ""

    def test_episodes_pre_existing_embeddings_seeded_silently(self, tmp_path, caplog):
        path = tmp_path / "migrate.db"
        _build_schema(path)
        _insert_episode(path, subject="old-episode", embedded=True)
        assert _get_provenance(path, "episodes") is None

        embed_fn = _stub_embed_fn()
        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)

        assert _get_provenance(path, "episodes") == _MODEL_A
        assert caplog.text == ""
        # Seeded, not re-embedded — no model call for the pre-existing row.
        embed_fn.assert_not_called()

    def test_no_data_no_row_defers_instead_of_seeding(self, tmp_path):
        """Zero embedded rows in either store: nothing to compare against yet
        — no provenance row should be created for a store with no data."""
        path = tmp_path / "migrate.db"
        _build_schema(path)
        _insert_document(path, name="unembedded-doc", embedded=False)

        MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        assert _get_provenance(path, "corpus") is None


# ---------------------------------------------------------------------------
# Self-heal — schema_version already current but embedding_provenance table
# missing (an earlier buggy migration bumped the version without creating it;
# _init_db's version-gate never re-triggers _migrate() in that state).
# ---------------------------------------------------------------------------

class TestMissingProvenanceTableSelfHeals:
    def test_construction_does_not_raise_and_table_is_recreated(self, tmp_path):
        path = tmp_path / "corrupted.db"
        _build_schema(path)

        conn = sqlite3.connect(str(path))
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == _SCHEMA_VERSION
        conn.execute("DROP TABLE embedding_provenance")
        conn.commit()
        conn.close()

        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        conn = sqlite3.connect(str(path))
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_provenance'"
        ).fetchone() is not None
        conn.close()
        assert mm._corpus_stale is False

    def test_seeds_provenance_normally_after_self_heal(self, tmp_path):
        """The self-heal shouldn't just avoid crashing — the rest of
        _check_embedding_provenance()'s logic (migration-seeding here) must
        still run correctly against the freshly recreated table."""
        path = tmp_path / "corrupted.db"
        _build_schema(path)
        _insert_document(path, name="old-doc", embedded=True)

        conn = sqlite3.connect(str(path))
        conn.execute("DROP TABLE embedding_provenance")
        conn.commit()
        conn.close()

        MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        assert _get_provenance(path, "corpus") == _MODEL_A


# ---------------------------------------------------------------------------
# Genuine mismatch — 'corpus'
# ---------------------------------------------------------------------------

class TestGenuineCorpusMismatch:
    def test_mismatch_warns_and_sets_corpus_stale(self, tmp_path, caplog):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_document(path, name="doc", embedded=True)
        _set_provenance(path, "corpus", _OTHER)

        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            mm = MemoryManager(
                db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A,
            )

        assert mm._corpus_stale is True
        assert "corpus embeddings were produced by" in caplog.text
        assert _OTHER in caplog.text
        assert _MODEL_A in caplog.text
        # Provenance is left recording the OLD model — it still describes
        # what's actually on disk until reembed_corpus() runs.
        assert _get_provenance(path, "corpus") == _OTHER

    def test_mismatch_flushes_retrieval_cache(self, tmp_path):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_document(path, name="doc", embedded=True)
        _set_provenance(path, "corpus", _OTHER)
        _set_cache_row_valid(path)
        assert _cache_valid_count(path) == 1

        MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        assert _cache_valid_count(path) == 0

    def test_query_corpus_falls_back_to_keyword_only_when_stale(self, tmp_path):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_document(path, name="apple-pie-recipe", embedded=True)
        _set_provenance(path, "corpus", _OTHER)

        embed_fn = _stub_embed_fn()
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)
        assert mm._corpus_stale is True
        embed_fn.reset_mock()

        results = mm.query_corpus("apple pie", use_embeddings=True)

        # embed_fn must NOT be called for the re-rank path while stale —
        # same fail-safe-to-keyword-only posture as embed_fn being None.
        embed_fn.assert_not_called()
        assert len(results) == 1
        assert results[0].name == "apple-pie-recipe"


# ---------------------------------------------------------------------------
# Genuine mismatch — 'episodes'
# ---------------------------------------------------------------------------

class TestGenuineEpisodesMismatch:
    def test_mismatch_triggers_auto_reembed(self, tmp_path, caplog):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_episode(path, subject="likes-coffee", content="Michael likes coffee", embedded=True)
        _insert_episode(path, subject="likes-tea", content="Michael likes tea", embedded=True)
        _set_provenance(path, "episodes", _OTHER)

        embed_fn = _stub_embed_fn(vector_value=0.9)
        with caplog.at_level(logging.WARNING, logger="localist.memory_manager"):
            MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)

        assert "episodes embeddings were produced by" in caplog.text
        assert embed_fn.call_count == 2
        embed_fn.assert_any_call("likes-coffee. Michael likes coffee")
        embed_fn.assert_any_call("likes-tea. Michael likes tea")

        assert _get_provenance(path, "episodes") == _MODEL_A

        conn = sqlite3.connect(str(path))
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM episodes").fetchall()]
        conn.close()
        assert len(blobs) == 2
        for blob in blobs:
            assert _unpack_embedding(blob) == pytest.approx([0.9] * _EMBEDDING_DIM)

    def test_no_manual_step_needed_no_stale_flag(self, tmp_path):
        """Unlike corpus, episodes carries no persistent staleness flag —
        the mismatch is corrected in place before __init__ even returns."""
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_episode(path, subject="s", embedded=True)
        _set_provenance(path, "episodes", _OTHER)

        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)

        assert not hasattr(mm, "_episodes_stale")
        assert _get_provenance(path, "episodes") == _MODEL_A


# ---------------------------------------------------------------------------
# reembed_episodes() — the manual counterpart to the startup auto-correct,
# called by POST /memory/reembed alongside the corpus and chat_turns
# ---------------------------------------------------------------------------

class TestReembedEpisodes:
    def test_reembeds_every_episode_and_advances_provenance(self, tmp_path):
        path = tmp_path / "reembed_episodes.db"
        _build_schema(path)
        _insert_episode(path, subject="likes-coffee", content="Michael likes coffee", embedded=True)
        _insert_episode(path, subject="likes-tea", content="Michael likes tea", embedded=True)
        _set_provenance(path, "episodes", _MODEL_A)

        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)
        new_embed_fn = _stub_embed_fn(vector_value=0.4)
        mm._embed_fn = new_embed_fn

        result = mm.reembed_episodes()

        assert result == {"reembedded": 2, "total": 2, "model": _MODEL_A, "failed_ids": []}
        new_embed_fn.assert_any_call("likes-coffee. Michael likes coffee")
        assert _get_provenance(path, "episodes") == _MODEL_A
        conn = sqlite3.connect(str(path))
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM episodes").fetchall()]
        conn.close()
        for blob in blobs:
            assert _unpack_embedding(blob) == pytest.approx([0.4] * _EMBEDDING_DIM)

    @pytest.mark.usefixtures("no_retry_sleep")
    def test_partial_failure_leaves_provenance_unchanged(self, tmp_path):
        path = tmp_path / "reembed_episodes_partial.db"
        _build_schema(path)
        _insert_episode(path, subject="ok", content="fine", embedded=True)
        _insert_episode(path, subject="bad", content="boom", embedded=True)
        _set_provenance(path, "episodes", _OTHER)

        mm = MemoryManager(db_path=path)  # no embed source -> startup check skipped
        def flaky_embed(text):
            if "boom" in text:
                raise RuntimeError("embed down")
            return [0.5] * _EMBEDDING_DIM

        mm._embed_fn = flaky_embed
        mm._embedding_model_name = _MODEL_A

        result = mm.reembed_episodes()

        assert result["reembedded"] == 1
        assert result["total"] == 2
        assert _get_provenance(path, "episodes") == _OTHER

    def test_raises_without_embed_fn(self, tmp_path):
        mm = MemoryManager(db_path=tmp_path / "keyword_only.db")
        with pytest.raises(RuntimeError):
            mm.reembed_episodes()


# ---------------------------------------------------------------------------
# set_embedding_source() — the live-switch counterpart to passing embed_fn/
# embedding_model_name at construction time (POST /settings/embedding-model)
# ---------------------------------------------------------------------------

class TestSetEmbeddingSource:
    def test_switching_to_a_new_model_reembeds_episodes_in_place(self, tmp_path):
        path = tmp_path / "live_switch.db"
        _build_schema(path)
        _insert_episode(path, subject="likes-coffee", content="Michael likes coffee", embedded=True)
        _set_provenance(path, "episodes", _OTHER)

        mm = MemoryManager(db_path=path)  # no embed source configured yet
        assert mm.embed_fn is None

        new_embed_fn = _stub_embed_fn(vector_value=0.7)
        mm.set_embedding_source(new_embed_fn, _MODEL_A)

        assert mm.embed_fn is new_embed_fn
        assert _get_provenance(path, "episodes") == _MODEL_A
        new_embed_fn.assert_any_call("likes-coffee. Michael likes coffee")

    def test_switching_flags_mismatched_corpus_stale(self, tmp_path):
        path = tmp_path / "live_switch_corpus.db"
        _build_schema(path)
        _insert_document(path, name="doc-a", embedded=True)
        _set_provenance(path, "corpus", _OTHER)

        mm = MemoryManager(db_path=path)
        assert mm._corpus_stale is False

        mm.set_embedding_source(_stub_embed_fn(), _MODEL_A)

        assert mm._corpus_stale is True

    def test_clearing_sets_embed_fn_none_and_skips_provenance_check(self, tmp_path):
        path = tmp_path / "live_switch_clear.db"
        _build_schema(path)
        _insert_episode(path, subject="s", embedded=True)
        _set_provenance(path, "episodes", _MODEL_A)

        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)
        mm.set_embedding_source(None, None)

        assert mm.embed_fn is None
        # Provenance untouched — clearing isn't a "new model", nothing to compare.
        assert _get_provenance(path, "episodes") == _MODEL_A


# ---------------------------------------------------------------------------
# reembed_corpus() — manual refresh
# ---------------------------------------------------------------------------

class TestReembedCorpus:
    def test_clears_staleness_updates_provenance_flushes_cache(self, tmp_path):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_document(path, name="doc-a", embedded=True)
        _insert_document(path, name="doc-b", embedded=True)
        _set_provenance(path, "corpus", _OTHER)
        _set_cache_row_valid(path)

        embed_fn = _stub_embed_fn(vector_value=0.42)
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)
        assert mm._corpus_stale is True
        assert _cache_valid_count(path) == 0   # already flushed by the mismatch detection itself
        _set_cache_row_valid(path)              # simulate a fresh (still-stale-scored) cache entry

        result = mm.reembed_corpus()

        assert result == {"reembedded": 2, "total": 2, "model": _MODEL_A, "failed_ids": []}
        assert mm._corpus_stale is False
        assert _get_provenance(path, "corpus") == _MODEL_A
        assert _cache_valid_count(path) == 0

        conn = sqlite3.connect(str(path))
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM document_index").fetchall()]
        conn.close()
        for blob in blobs:
            assert _unpack_embedding(blob) == pytest.approx([0.42] * _EMBEDDING_DIM)

    def test_works_when_not_currently_stale(self, tmp_path):
        """A manual 'just refresh it' call must work even when nothing is stale."""
        path = tmp_path / "fresh.db"
        embed_fn = _stub_embed_fn(vector_value=0.7)
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)
        mm.index_document(path=tmp_path / "doc.md", doc_type="wiki", content="hi", embed=True)
        assert mm._corpus_stale is False

        result = mm.reembed_corpus()

        assert result["reembedded"] == 1
        assert result["total"] == 1
        assert mm._corpus_stale is False
        assert _get_provenance(path, "corpus") == _MODEL_A

    def test_query_corpus_uses_embeddings_again_after_reembed(self, tmp_path):
        path = tmp_path / "mismatch.db"
        _build_schema(path)
        _insert_document(path, name="doc", embedded=True)
        _set_provenance(path, "corpus", _OTHER)

        embed_fn = _stub_embed_fn()
        mm = MemoryManager(db_path=path, embed_fn=embed_fn, embedding_model_name=_MODEL_A)
        assert mm._corpus_stale is True

        mm.reembed_corpus()
        embed_fn.reset_mock()

        mm.query_corpus("doc", use_embeddings=True)

        embed_fn.assert_called()


# ---------------------------------------------------------------------------
# Re-embed retry + partial-failure handling (docs/architecture/
# 16-runtime-backend-layer.md §16.4) — shared by reembed_corpus(),
# reembed_chat_turns() and reembed_episodes()
# ---------------------------------------------------------------------------

_VEC = [0.6] * _EMBEDDING_DIM


@pytest.fixture()
def no_retry_sleep(monkeypatch):
    """Keep the retry schedule's attempt count but skip its real sleeps."""
    monkeypatch.setattr(memory_manager, "_REEMBED_RETRY_DELAYS_S", (0.0, 0.0))


def _doc_ids(path: Path) -> list[int]:
    conn = sqlite3.connect(str(path))
    ids = [row[0] for row in conn.execute("SELECT id FROM document_index ORDER BY id")]
    conn.close()
    return ids


def _corpus_mm_on_other_model(tmp_path, n_docs: int) -> tuple[Path, MemoryManager]:
    """n_docs embedded under _OTHER, MemoryManager now active on _MODEL_A (stale)."""
    path = tmp_path / "corpus_retry.db"
    _build_schema(path)
    for i in range(n_docs):
        _insert_document(path, name=f"doc-{i}", embedded=True)
    _set_provenance(path, "corpus", _OTHER)
    mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)
    assert mm._corpus_stale is True
    return path, mm


@pytest.mark.usefixtures("no_retry_sleep")
class TestReembedRetry:
    def test_transient_failure_is_retried_and_store_completes(self, tmp_path):
        path, mm = _corpus_mm_on_other_model(tmp_path, n_docs=2)
        mm._embed_fn = MagicMock(side_effect=[RuntimeError("timeout"), _VEC, _VEC])

        result = mm.reembed_corpus()

        assert result["reembedded"] == 2
        assert result["failed_ids"] == []
        assert mm._embed_fn.call_count == 3
        assert mm._corpus_stale is False
        assert _get_provenance(path, "corpus") == _MODEL_A

    def test_persistent_failure_keeps_store_stale_and_reports_the_row(self, tmp_path):
        path, mm = _corpus_mm_on_other_model(tmp_path, n_docs=2)
        first_id, second_id = _doc_ids(path)
        err = RuntimeError("embed down")
        mm._embed_fn = MagicMock(side_effect=[err, err, err, _VEC])

        result = mm.reembed_corpus()

        assert result["reembedded"] == 1
        assert result["total"] == 2
        assert result["failed_ids"] == [first_id]
        assert mm._corpus_stale is True
        assert _get_provenance(path, "corpus") == _OTHER
        # The row that succeeded keeps its new vector, so a retry only has
        # the failure left to fix.
        conn = sqlite3.connect(str(path))
        blob = conn.execute(
            "SELECT embedding FROM document_index WHERE id = ?", (second_id,),
        ).fetchone()[0]
        conn.close()
        assert _unpack_embedding(blob) == pytest.approx(_VEC)

    def test_rerun_after_partial_failure_completes_the_store(self, tmp_path):
        path, mm = _corpus_mm_on_other_model(tmp_path, n_docs=2)
        err = RuntimeError("embed down")
        mm._embed_fn = MagicMock(side_effect=[err, err, err, _VEC])
        mm.reembed_corpus()
        assert mm._corpus_stale is True

        mm._embed_fn = _stub_embed_fn(0.6)
        result = mm.reembed_corpus()

        assert result["failed_ids"] == []
        assert mm._corpus_stale is False
        assert _get_provenance(path, "corpus") == _MODEL_A

    def test_same_model_refresh_with_a_failure_does_not_mark_stale(self, tmp_path):
        """Provenance already names the active model, so the failed row's old
        vector is still in the right space — degrading to keyword-only would
        be a regression, not a safeguard."""
        path = tmp_path / "same_model.db"
        _build_schema(path)
        _insert_document(path, name="doc-a", embedded=True)
        _insert_document(path, name="doc-b", embedded=True)
        _set_provenance(path, "corpus", _MODEL_A)
        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_MODEL_A)
        err = RuntimeError("embed down")
        mm._embed_fn = MagicMock(side_effect=[err, err, err, _VEC])

        result = mm.reembed_corpus()

        assert len(result["failed_ids"]) == 1
        assert mm._corpus_stale is False
        assert _get_provenance(path, "corpus") == _MODEL_A

    def test_aborts_after_consecutive_failures_and_reports_every_skipped_row(self, tmp_path):
        path, mm = _corpus_mm_on_other_model(tmp_path, n_docs=5)
        mm._embed_fn = MagicMock(side_effect=RuntimeError("embed down"))

        result = mm.reembed_corpus()

        attempts_per_row = 1 + len(memory_manager._REEMBED_RETRY_DELAYS_S)
        assert mm._embed_fn.call_count == (
            memory_manager._REEMBED_ABORT_AFTER_CONSECUTIVE_FAILURES * attempts_per_row
        )
        assert result["reembedded"] == 0
        assert result["failed_ids"] == _doc_ids(path)
        assert mm._corpus_stale is True

    def test_wrong_dimension_is_not_retried(self, tmp_path):
        path, mm = _corpus_mm_on_other_model(tmp_path, n_docs=2)
        mm._embed_fn = MagicMock(side_effect=[[0.1] * 3, _VEC])

        result = mm.reembed_corpus()

        assert mm._embed_fn.call_count == 2
        assert result["failed_ids"] == [_doc_ids(path)[0]]
        assert mm._corpus_stale is True

    def test_chat_turns_persistent_failure_keeps_store_stale(self, tmp_path):
        path = tmp_path / "chat_retry.db"
        mm = MemoryManager(db_path=path, embed_fn=_stub_embed_fn(), embedding_model_name=_OTHER)
        mm.add_chat_turn(task_id="t", role="user", content="hello", conversation_id="c")
        mm.add_chat_turn(task_id="t", role="assistant", content="hi there", conversation_id="c")
        assert _get_provenance(path, "chat_turns") == _OTHER

        mm._embedding_model_name = _MODEL_A
        err = RuntimeError("embed down")
        mm._embed_fn = MagicMock(side_effect=[err, err, err, _VEC])

        result = mm.reembed_chat_turns()

        assert result["reembedded"] == 1
        assert len(result["failed_ids"]) == 1
        assert mm._chat_turns_stale is True
        assert _get_provenance(path, "chat_turns") == _OTHER
