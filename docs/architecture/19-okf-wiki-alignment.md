## 19. OKF (Open Knowledge Framework) Wiki Alignment

### 19.1 Motivation

The user asked LORA to compare the wiki corpus against Google's OKF spec, as
adopted by LangChain's OpenWiki 0.2
([blog post](https://www.langchain.com/blog/openwiki-0-2-adds-okf-support)).
LORA's first attempt at this comparison (in-chat, not from this repo) was
not well-grounded: it claimed the existing front matter already reflected
OKF's influence, citing fields (`type`, `created`, `updated`) that aren't
OKF's actual distinguishing fields, and it never checked for `index.md`/
`logs.md` — OKF's two required structural files — which didn't exist
anywhere in `wiki/`. The article was re-fetched directly and the codebase
was read to establish ground truth before any change was made.

**OKF spec, as fetched from the article:**
- YAML front matter: `type` (**required** — "identifying concept of the
  doc"); `title`, `description`, `resource` (canonical URI), `tags`,
  `timestamp` all optional; other producer-defined keys allowed alongside.
- `index.md` — one per wiki directory, summarizing that directory's files
  as headed, bulleted lists (title + description per entry).
- `logs.md` — a changelog: dated headers (`YYYY-MM-DD`), entries typed
  Creation/Update/Initialization, linking to the files changed.

### 19.2 Pre-existing state this had to reconcile, not just build on

Exploration before implementation found this wasn't "add OKF to a blank
slate" — three things already existed and needed reconciling:

1. **Three independent, mutually inconsistent front-matter conventions**:
   `SCHEMA.md` (a literal, never-replaced placeholder — `title, type,
   created, updated`), the actual prompt template the model copies,
   `_EXAMPLE` in `wiki_agent.py` (`type, status, source` — no `title` at
   all), and the 7 real pages on disk (`title, type, query, created,
   updated`). None of the three agreed with each other before this change.
2. **No file-exclusion mechanism existed anywhere in the indexing
   pipeline.** `memory_manager.index_directory()`, `build_graph.py`'s wiki
   walk, and `wiki_agent.py`'s page loaders swept every `.md` file under
   `wiki_dir` indiscriminately — confirmed via `MEMORY.md` (no front matter
   at all) already being fully exposed as a `doc_type="wiki"` RAG document,
   a `graph_nodes` row, and a `resolve_graph_target()` candidate stem.
   Adding `index.md`/`logs.md` without an exclusion mechanism would have
   repeated that same mistake on purpose.
3. **`_apply_changes()` writes are flat and 100% model-authored** — no
   metadata is programmatically injected; whatever front matter the model
   produces is exactly what lands on disk.

### 19.3 `META_WIKI_FILENAMES` — one shared exclusion constant

`wiki_doc.py` (confirmed a pure, dependency-free leaf module — only
`re`/`yaml`/stdlib — already imported by both `build_graph.py` and
`wiki_agent.py`) gained:

```python
META_WIKI_FILENAMES: frozenset[str] = frozenset({"index.md", "logs.md", "MEMORY.md"})
```

Rather than duplicating a filename denylist across four modules, this one
constant is imported everywhere a wiki-directory walk needs to skip
structural/generated files. Including `MEMORY.md` closes the pre-existing
`MEMORY.md` exposure gap found during exploration, at zero extra cost since
the mechanism had to be built anyway.

**Four enforcement points:**
1. `wiki_agent.py`'s `_load_wiki_pages()` / `_load_wiki_pages_from_index()`
   — these feed `build_wiki_context()` (model-visible "EXISTING WIKI
   PAGES") and `diff_target` lookups; a meta file must never be either.
2. `build_graph.py`'s wiki-file walk — keeps `index.md`/`logs.md`/
   `MEMORY.md` out of `graph_nodes`/`list_graph_node_stems()`, so they
   can never become a Planner P1b/P1c/P3c resolution target.
3. `memory_manager.index_directory()` gained an optional
   `exclude: frozenset[str] = frozenset()` parameter (default empty — zero
   behavior change for existing `raw_dir` callers); the wiki-specific call
   sites (`main.py`'s startup seed, `reconcile_wiki()`'s internal call)
   pass `exclude=META_WIKI_FILENAMES`, keeping meta files out of
   `document_index`/`query_corpus()` RAG retrieval entirely.
4. `GET /files/wiki` (`main.py`) — excludes meta filenames from the file
   listing Feature A's wiki-pin picker reuses; `POST /chat/pin-wiki-page`
   also rejects a meta filename directly (400), as defense in depth beyond
   the picker UI already not surfacing them.

### 19.4 Front matter — reconciled, not layered on top

`SCHEMA.md`'s placeholder content was replaced with the real, OKF-aligned
convention (`type` required; `title`/`description`/`resource`/`tags`/
`timestamp` OKF-optional; `status`/`created`/`updated`/`query` kept as this
project's own allowed producer-defined extras — no existing field removed).
`wiki_agent.py`'s `_EXAMPLE` template (the thing that actually governs model
output, not `SCHEMA.md`) was expanded to include the five new fields with
worked example values, and both `build_user_prompt()`/`build_slim_prompt()`
gained a new numbered rule instructing the model to fill them with real,
specific values (never the literal placeholder text) — this shifted the
existing link-target rule from "7." to "8." and the H3-subsection rule from
"2." to "3." (test coverage updated to match: renamed
`test_build_slim_prompt_contains_rule7_identical_to_user_prompt` →
`test_build_slim_prompt_contains_link_rule_identical_to_user_prompt`, rule
number updated 7→8; `_RULE_SUBSTRINGS`/`test_prompt_rules_1_through_6_unchanged`
→ `test_prompt_rules_1_through_7_unchanged`).

**description/tags are model-generated at write time** (per user decision)
— no new inference call, since the same call that already produces a page
now also produces these fields.

### 19.5 `index.md` / `logs.md` — deterministic, never model-authored

Both are pure functions of on-disk state, computed in `WikiAgent._finalize()`
(the existing shared post-write tail for both the ingest and diff-only
paths) and in `apply_pending_diff()` (the review-then-apply UI's write
path, §17.7) — never produced by the model's own inference. Asking the
model to hand-author a changelog/index risks drift and hallucinated links,
and neither existing write path has an obvious place to route a second
"also rewrite index.md" model turn.

- **`WikiAgent._regenerate_index_md(wiki_pages, wiki_dir)`** — parses each
  page's front matter via the existing `parse_wiki_doc()`, groups by
  `type` (alphabetically), and writes `wiki_dir/index.md` as headed bullet
  lists (`[[stem]] — title: description`). A page missing `type` groups
  under `UNSPECIFIED`; missing `title`/`description` fall back to the
  page's stem / no description line, so a not-yet-backfilled page never
  breaks generation. Fully overwritten on every call (not merged).
- **`WikiAgent._append_logs_md(entries, wiki_dir)`** — `entries` is a list
  of `(kind, page_name)` pairs, `kind` one of `"Creation"`/`"Update"`
  (derived in `_finalize()` from which `written` page names came from
  `actions.new_pages` vs. `actions.diffs`; always `"Update"` from
  `apply_pending_diff()`, which only ever patches an existing page).
  Appends a dated section (`## YYYY-MM-DD`, reusing the existing
  `date.today().isoformat()` convention already used in Revision History)
  with one `- {kind}: [[page_name]]` bullet per entry. Creates the file
  with a `# Wiki Changelog` header on first write; every later call is a
  pure append — no re-parsing of `logs.md`'s own structure, so a
  hand-edited `logs.md` can never confuse a future write. No-op if
  `entries` is empty (nothing was actually written).

Both are gated on `auto_apply`/an actual disk write — a chat turn that only
*proposes* a diff (the default, since `auto_apply` is never set from chat
context) does not touch either file, same as it doesn't touch the disk
page itself.

### 19.6 One-time backfill — `backend/backfill_okf_frontmatter.py`

Mirrors `build_graph.py`'s existing offline `__main__` script pattern (no
new idiom). For each real page under `wiki/` (skipping
`META_WIKI_FILENAMES`), parses existing front matter via `parse_wiki_doc()`
and adds any of `title`/`description`/`resource`/`tags`/`timestamp` that
are missing, deriving values heuristically — no inference call, no model
involved, since this is a one-time pass over pages that already exist (new
pages get these fields from the model directly, §19.4):
- `description` — first sentence of the `## Summary` section.
- `tags` — up to 5 kebab-cased bullet labels under `### Extracted Concepts`.
- `resource` — the filename referenced in the existing `query` field
  (stripping a leading `"Analyze "` verb this project's ingestion queries
  always carry, so `resource` holds just the filename, matching OKF's
  "canonical URI of the underlying asset" intent).
- `timestamp` — the existing `updated` value, stringified via
  `.isoformat()` rather than reused as the same `datetime.date` object —
  an early version of this script left the same object reference in the
  dict twice, which made PyYAML emit anchor/alias syntax
  (`timestamp: &id001 2026-06-16` / `updated: *id001`) instead of two
  plain scalars. Fixed by stringifying, plus a `_NoAliasDumper`
  (`ignore_aliases` forced `True`) as defense in depth against any other
  repeated-object case.

No field is ever removed or overwritten — only missing fields are added,
appended after the OKF fields in a fixed order, with every original field
preserved in its original position after that.

**Run against the real corpus 2026-07-21**: all 7 existing pages updated
(6 pages gained `description`/`resource`/`tags`/`timestamp`; `lora-persona.md`
— which has no `## Summary`/`### Extracted Concepts` sections, being a
persona doc rather than a research note — gained only `title`/`timestamp`,
since `description`/`tags` correctly derived to nothing rather than
fabricating placeholder text). Verified via diff against a pre-run backup
(since `wiki/` is `.gitignore`d, with no version-control safety net) that
only front matter changed — no body content lost beyond a single
leading-blank-line normalization already inherent to `parse_wiki_doc()`'s
body extraction.

### 19.7 Test suite

969 → 987 passed, 0 failed. New/changed coverage:
- `test_wiki_doc.py` — `META_WIKI_FILENAMES` contains the expected three names.
- `test_wiki_agent.py` — `_load_wiki_pages()` excludes meta filenames;
  `_regenerate_index_md()` groups by type, falls back cleanly on missing
  front matter, and fully overwrites on each call; `_append_logs_md()`
  creates-then-appends (never overwrites) and no-ops on empty entries; an
  end-to-end `_run_diff_only()`-via-`run()` case confirms both files are
  correctly populated after an `auto_apply=True` write; the two renumbered
  rule-text regression tests (§19.4) updated to match the new rule
  ordering rather than being weakened.
- `test_build_graph.py` — a fixture `wiki_dir` containing all three meta
  filenames confirms `list_graph_node_stems()` excludes them.
- `test_memory_phase1.py` — `TestReconcileWiki` gained a meta-filename
  exclusion case; new `TestIndexDirectoryExclude` class covers the new
  `exclude` parameter directly, including a regression guard that omitting
  `exclude` (existing `raw_dir` callers) is completely unaffected.
- `test_files_wiki_endpoint.py` (new file) — `GET /files/wiki` excludes
  meta filenames from its response.
- `test_chat_pin_wiki_page.py` — extended with a parametrized case
  confirming all three meta filenames are rejected (400) even if one
  happens to exist on disk under that name.
