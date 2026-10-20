"""
Keeping the document index free of duplicates and dead rows (2026-10-20).

A source checkout and the packaged app shared one database while indexing
separate copies of the same files, so every document appeared twice and could
fill several RAG slots with one piece of content. Raw files deleted outside the
app, and generated wiki housekeeping files, also lingered as rows.

- query_corpus() keeps one result per content_hash.
- prune_missing_documents() removes rows whose file is gone (raw had no
  equivalent of reconcile_wiki()'s orphan pass).
- reconcile_wiki() removes indexed housekeeping files (index.md, logs.md,
  MEMORY.md).
"""

from __future__ import annotations

from localist.memory_manager import MemoryManager


def _mm(tmp_path):
    return MemoryManager(db_path=tmp_path / "dup.db")


class TestQueryCorpusDedupe:
    def test_identical_content_under_two_paths_is_returned_once(self, tmp_path):
        mm = _mm(tmp_path)
        body = "Zylophone tuning guide for brass reeds and concert pitch."
        mm.index_document(path=tmp_path / "a" / "wiki" / "zylophone.md", doc_type="wiki", content=body)
        mm.index_document(path=tmp_path / "b" / "wiki" / "zylophone.md", doc_type="wiki", content=body)
        mm.index_document(path=tmp_path / "a" / "wiki" / "other.md", doc_type="wiki",
                          content="Zylophone history notes, a different page.")

        results = mm.query_corpus("zylophone tuning", max_results=3)

        assert len(results) == 2
        assert len({r.content for r in results}) == 2

    def test_different_content_is_never_collapsed(self, tmp_path):
        mm = _mm(tmp_path)
        for i in range(3):
            mm.index_document(path=tmp_path / f"p{i}.md", doc_type="raw", content=f"zylophone note number {i}")
        assert len(mm.query_corpus("zylophone note", max_results=3)) == 3


class TestPruneMissingDocuments:
    def test_rows_for_deleted_files_are_removed(self, tmp_path):
        mm = _mm(tmp_path)
        kept, gone = tmp_path / "raw" / "kept.md", tmp_path / "raw" / "gone.md"
        kept.parent.mkdir()
        for p in (kept, gone):
            p.write_text(f"{p.stem} zylophone notes")
            mm.index_document(path=p, doc_type="raw", content=p.read_text())
        gone.unlink()

        assert mm.prune_missing_documents("raw") == ["gone"]   # document names are file stems
        assert [d.name for d in mm.get_all_documents(doc_type="raw")] == ["kept"]

    def test_only_the_given_doc_type_is_touched(self, tmp_path):
        mm = _mm(tmp_path)
        mm.index_document(path=tmp_path / "missing-wiki.md", doc_type="wiki", content="wiki zylophone")
        assert mm.prune_missing_documents("raw") == []
        assert len(mm.get_all_documents(doc_type="wiki")) == 1


class TestReconcileWikiHousekeeping:
    def test_indexed_housekeeping_files_are_removed(self, tmp_path):
        mm = _mm(tmp_path)
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "page.md").write_text("---\ntitle: Page\n---\n\nZylophone page.")
        for name in ("index.md", "logs.md", "MEMORY.md"):
            (wiki / name).write_text(f"# {name}")
            mm.index_document(path=wiki / name, doc_type="wiki", content=f"# {name}")

        summary = mm.reconcile_wiki(wiki)

        assert sorted(summary["housekeeping_removed"]) == ["MEMORY", "index", "logs"]
        assert [d.name for d in mm.get_all_documents(doc_type="wiki")] == ["page"]
