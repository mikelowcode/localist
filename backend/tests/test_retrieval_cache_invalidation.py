"""
The retrieval cache must be invalidated, durably, whenever document_index
changes. remove_document() and index_document() used to run
_invalidate_cache() after their commit and then close the connection, so the
invalidation was rolled back and a cached query_corpus() result kept being
served after the index changed.
"""

from __future__ import annotations

import sqlite3

from localist.memory_manager import MemoryManager


def _valid_cache_entries(db_path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM retrieval_cache WHERE valid = 1").fetchone()[0]
    finally:
        conn.close()


def _mm_with_cached_query(tmp_path):
    db = tmp_path / "cache.db"
    mm = MemoryManager(db_path=db)   # no embed_fn: keyword retrieval, still cached
    page = tmp_path / "zylophone.md"
    mm.index_document(path=page, doc_type="wiki", content="Zylophone tuning guide for brass reeds.")
    assert [d.path.name for d in mm.query_corpus("zylophone tuning", max_results=3)] == ["zylophone.md"]
    assert _valid_cache_entries(db) >= 1
    return mm, db, page


class TestCacheInvalidatedOnIndexChange:
    def test_remove_document_invalidates_the_cache(self, tmp_path):
        mm, db, page = _mm_with_cached_query(tmp_path)
        mm.remove_document(page)
        assert _valid_cache_entries(db) == 0

    def test_index_document_invalidates_the_cache(self, tmp_path):
        mm, db, _page = _mm_with_cached_query(tmp_path)
        mm.index_document(path=tmp_path / "other.md", doc_type="wiki", content="Unrelated notes.")
        assert _valid_cache_entries(db) == 0

    def test_new_document_appears_in_a_previously_cached_query(self, tmp_path):
        mm, _db, _page = _mm_with_cached_query(tmp_path)
        mm.index_document(
            path=tmp_path / "zylophone-advanced.md", doc_type="wiki",
            content="Advanced zylophone tuning: zylophone tuning for concert zylophone players.",
        )
        names = {d.path.name for d in mm.query_corpus("zylophone tuning", max_results=3)}
        assert "zylophone-advanced.md" in names
