## 8. Graph Retrieval Layer

### 8.1 Scope

**Implemented (Phases A, B, and C):**

- `wiki_doc.py` — shared frontmatter/body/link parsing helper consumed by
  `controller_agent.py` and `build_graph.py`.
- `memory_manager.py` v2→v3 schema migration — `graph_nodes` and `graph_edges`
  tables; `_SCHEMA_VERSION = 3`.
- `build_graph.py` — offline two-pass link-graph builder.
- `_validate_links()` in `wiki_agent.py` — write-time link validation wired
  between XML parsing and journaling.
- `memory_manager.py` — three new graph read methods: `resolve_node_by_stem()`,
  `get_backlinks()`, `get_outgoing_links()`, plus `list_graph_node_stems()` (added
  during Planner wiring once a gap was found — no existing method listed all stems).
  New result type `GraphEdgeResult`.
- `prompt_builder.py` — new `[GRAPH RESULT]` slot (`_slot_graph()`), positioned
  after Tool Results, before Working Memory. New input dataclasses
  `GraphQueryResult`/`GraphLinkEntry` (deliberately separate from
  `memory_manager.GraphEdgeResult` — `prompt_builder.py` remains free of any
  `memory_manager` import, preserving its pure-Python constraint). New
  `_CEIL_GRAPH = 300` ceiling. This slot is the one documented exception to the
  module's clean-omission contract: it is emitted whenever a graph query resolves
  a target page, even with zero edges, and is omitted only when resolution itself
  fails.
- `planner.py` — new standalone functions `extract_graph_query()` and
  `resolve_graph_target()` (three deterministic extraction patterns; three-tier
  stem-based name resolution: exact/substring, then token-overlap with a 2-token
  minimum and 0.5 ratio threshold, then ambiguous/no-match fallthrough — never a
  tiebreak). New `RoutingPlan` field `graph_query: tuple[str, int, str] | None`.
  New method `_priority3c_graph_query()`, checked in `route()` **before**
  `_priority3_tool()` — see ordering-correction note below. P3c's own inline guard
  checks `_FILE_OP_KEYWORDS`/`_FETCH_KEYWORDS`/the URL regex directly; when either
  fires, P3c defers and normal `_priority3_tool()` evaluation proceeds.
- `controller_agent.py` — new Step 5c in `_execute_plan()`: fetches
  `get_backlinks()`/`get_outgoing_links()` when `plan.graph_query` is set, translates
  `GraphEdgeResult` → `GraphLinkEntry`/`GraphQueryResult` (using `link_text`, not
  `target_path`, as the display name for unresolved targets, to preserve original
  casing per the locked output format), and passes the result into
  `PromptBuilder.build()`'s new `graph_result` parameter. The "pure/minimal"
  guarantee (graph-query turns never combine with RAG/episodic/profile context)
  requires no extra guard code — it falls out for free because P3c's `RoutingPlan`
  already sets `fetch_rag`/`fetch_episodic` to `False` and
  `tools_to_call` to `[]`; confirmed end-to-end with a dedicated leak-marker test.
- `build_graph.py` — fixed: the `__main__` block previously called `MemoryManager()`
  with no path argument, which resolved to `MemoryManager`'s bare default
  (`lora_memory.db`) rather than the live backend's actual database
  (`localist_memory.db`, per `main.py:254`). Found via live manual testing, not by
  any automated test. Fixed by hardcoding `_BACKEND_DIR / "localist_memory.db"`.

**Locked-design ordering correction (found during implementation):** The design's
requirement that graph-query win over a web_search-only match is only satisfiable
if P3c is checked **before** `_priority3_tool()` runs — not after.
`_priority3_tool()` returns a plan whenever *any* of its three signals match,
including web_search alone; if P3c ran after it, a web_search-only match would
cause `route()` to return before P3c ever ran. The implemented ordering checks P3c
first, with P3c's inline guard (checking only `_FILE_OP_KEYWORDS`/`_FETCH_KEYWORDS`/
URL-regex — deliberately not `_WEB_SEARCH_KEYWORDS`) deferring to P3 only when
file_op or url_fetch signals are present. Locked in by `test_p3c_beats_web_search_p3`
in `tests/test_planner_phase3.py`, which would fail under the naive "after
`_priority3_tool()`" ordering.

**Explicitly deferred:**

- Phase D — automatic promotion from episodic memory to graph: not started.

### 8.2 Design Decisions

**Link graph, not LLM extraction.** Phase B parses existing `[[wiki-link]]`
references deterministically — no inference call, no entity/relationship
extraction. Rationale: matches the **Predictable** constraint (§1); edges only
exist where a human or WikiAgent explicitly linked two pages; avoids the
validation burden of LLM-extracted entities before the graph schema has any
real production usage. Richer NER/relationship extraction is a possible future
Phase C, not cancelled.

**Offline script, not WikiAgent post-ingest hook.** `build_graph.py` runs
manually and touches only `graph_nodes`/`graph_edges`. Rationale: WikiAgent's
system prompt is a protected XML-only contract (§3.5); embedding a hook would
carry the `/ingest` → `retrieval_cache` invalidation blast radius for an
unrelated concern; an offline script keeps both responsibilities isolated.
Migration path: the `build_graph()` function's signature is caller-agnostic —
a future WikiAgent post-ingest hook could call it without changing the function
itself, only the call site.

**Whole-corpus clear between passes, not per-document.** `build_graph.py`
calls `clear_graph_edges()` once between the node-upsert pass and the
edge-upsert pass. Ensures stale edges from since-removed `[[...]]` links never
survive a rebuild. `clear_graph_for_doc()` is implemented for future
per-document incremental updates but not called by the offline script.

**`doc_path` uses absolute resolved paths.** `str(Path(p).resolve())`,
consistent with the existing `document_index.path` convention in
`MemoryManager`. Future Phase C retrieval code can look up graph nodes the same
way it already looks up indexed documents.

**`raw/` documents are not graph nodes.** Only curated wiki pages are nodes.
A `[[...]]` link whose normalized target coincidentally matches a filename in
`raw/` still counts as unresolved.

### 8.3 Schema

`_SCHEMA_VERSION` is now **3** (v2→v3 migration added to `memory_manager.py`).

```sql
CREATE TABLE IF NOT EXISTS graph_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_path        TEXT    NOT NULL UNIQUE,
    node_type       TEXT,
    title           TEXT,
    source_doc_path TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_path
    ON graph_nodes(doc_path);

CREATE TABLE IF NOT EXISTS graph_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id  INTEGER NOT NULL REFERENCES graph_nodes(id),
    target_path     TEXT    NOT NULL,
    target_node_id  INTEGER REFERENCES graph_nodes(id),
    target_resolved INTEGER NOT NULL DEFAULT 0,
    link_text       TEXT    NOT NULL,
    source_doc_path TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source
    ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target_path
    ON graph_edges(target_path);
CREATE INDEX IF NOT EXISTS idx_graph_edges_resolved
    ON graph_edges(target_resolved);
```

`target_node_id` is nullable. Unresolved links are written with
`target_node_id = NULL` and `target_resolved = 0`. When the target page is
later created and the script rerun, `upsert_graph_edge()` updates the existing
row in-place — enabling automatic resolution on the next rebuild without a
separate cleanup pass.

### 8.4 Resolution Rule

```
link_text.lower().replace(" ", "-")
```

Applied identically by:

| Layer | Location |
|---|---|
| Write-time (WikiAgent) | `wiki_agent._validate_links()`, lines 663/665 |
| Read-time (builder) | `build_graph._normalize()` |

A link resolves if the normalized form matches the **stem** of an existing
`graph_nodes.doc_path` — matching how `wiki_pages` keys are built in
`wiki_agent.py` (`p.stem`), so resolution is apples-to-apples across both
layers.

Unresolved links are never dropped. A `target_resolved=False` row is always
written, enabling corpus gap analysis from the graph tables directly.

### 8.5 wiki_doc.py — Shared Parsing Helper

New module `backend/wiki_doc.py`:

```python
@dataclass(frozen=True)
class WikiLink:
    link_text: str
    target_path: str   # same as link_text; Phase B normalizes independently

@dataclass(frozen=True)
class ParsedWikiDoc:
    frontmatter: dict[str, Any]   # PyYAML 6.0 — ISO dates parse as datetime.date
    body: str
    links: list[WikiLink]         # all [[...]] in body; not section-scoped

def parse_wiki_doc(content: str) -> ParsedWikiDoc: ...
def load_wiki_doc(path: Path) -> ParsedWikiDoc: ...
```

`links` contains every `[[...]]` reference in the body. Section scoping is
`_validate_links()`'s concern at write time; the helper is scope-agnostic by
design so a future caller can impose any scoping it needs.

**Regression closed by this module:**
- `_load_persona()` previously truncated raw file content at 2,000 characters
  with no frontmatter awareness. Now operates on `body` only.
- `_load_user_profile()` had no frontmatter-skip logic. Now calls
  `load_wiki_doc()` and parses `body` lines.
- Both fixes verified zero-behavior-change for `lora-persona.md` and
  `wiki/users/michael.md`, neither of which has frontmatter today.

### 8.6 WikiAgent Link Validation

`_validate_links(actions, wiki_pages) -> dict[page_name, list[target]]` added
to `wiki_agent.py` and wired into `run()` between XML parsing and journaling.

**Scope:** `### Mapped Pages` (H3) and `## Related Pages` (H2) sections only.

**Behavior:** For each `[[link]]` in the scanned sections, if the normalized
form does not match an existing page stem or a self-proposed page name, the
link is flagged. Page content reaching disk is **never modified**. Flagged
links are logged at WARNING level and returned as
`AgentResult.output["unresolved_links"]`. This is intentional layered defense:
the read-time graph builder independently detects any unresolved link regardless
of what the write-time check catches.

A complementary write-time rule lives in the WikiAgent prompt templates themselves. Rule 7, added to both `build_user_prompt()` and `build_slim_prompt()` in `wiki_agent.py`, instructs the model to use the verbatim `page_name` as the `[[...]]` link target rather than a paraphrased title or longer description, reducing how often `_validate_links()` has anything to flag. This is a model-prompting measure only — `_validate_links()`'s normalization rule and section scope are unchanged, and it continues to flag every link that does not resolve exactly as before. Rule 7 reduces false positives at the source; it does not change what counts as resolved.

### 8.7 Validation-Run Results

`python build_graph.py` run against the real 5-document `wiki/` corpus
(2026-06-19 session).

| Metric | Count |
|---|---|
| Nodes | 5 |
| Edges | 11 |
| Resolved | 8 |
| Unresolved | 3 |

**Per-page breakdown:**

| Source page | Resolved edges | Unresolved edges |
|---|---|---|
| `how-localist-works` | 4 (→ `localist-build-order`, `localist-master-project-outline`, `localist-software-stack`, `lora-persona`) | 0 |
| `localist-build-order` | 1 (→ `localist-master-project-outline`) | 1 |
| `localist-master-project-outline` | 2 (→ `localist-build-order`, `localist-software-stack`) | 2 |
| `localist-software-stack` | 1 (→ `localist-master-project-outline`) | 0 |
| `lora-persona` | 0 | 0 |

**The three unresolved cases (precisely characterized):**

1. `localist-software-stack-overview` — from `localist-build-order.md`'s
   `[[Localist Software Stack Overview]]`. **Word-count mismatch**, not a casing
   issue. The actual page stem is `localist-software-stack`; the link text has
   an extra word ("Overview"). Will not resolve via the narrow normalization
   rule. This is the expected, correct behavior — not a defect in the
   normalization logic.

2. `localist-design-philosophy` — from `localist-master-project-outline.md`.
   Genuinely nonexistent page, proposed in that file's "Proposed New Pages"
   section but never created.

3. `localist-wiki-evolution-ideas` — from `localist-master-project-outline.md`.
   Same: genuinely nonexistent page.

**Incidental finding (recorded, not acted on):** `how-localist-works.md` is
the only page in the corpus whose `[[...]]` link targets are already correctly
kebab-cased, matching their target filenames exactly. Every other page exhibits
the title-case defect described in §8.8 Open Item 1. This suggests the model
can produce correct kebab-case link generation under at least some conditions —
relevant evidence for the prompt-tightening follow-up but not acted on here.

### 8.8 Open Items (Explicitly Deferred)

*Cross-reference (2026-06-21): Slot 5b (`[GRAPH RESULT]`) is now documented canonically in §3.2 and §3.3, not only in §8.1 Scope. The documentation gap from Phase C is closed.*

**Open Item 1 — WikiAgent prompt wording (highest-priority follow-up).**
WikiAgent's prompt does not state that `[[...]]` link targets must equal an
existing or self-proposed `page_name` verbatim. The real corpus confirms this
is a live defect (title-case vs. kebab-case throughout; word-count mismatch in
`localist-build-order.md`). A prompt-tightening change to Rule 5 and/or the
`_EXAMPLE` block was **agreed in principle (2026-06-19 session) but not
scheduled or implemented.** Recommended as a small standalone follow-up kept
separate from this build so it can be tested in isolation.

**Open Item 2 — `wiki/users/michael.md` frontmatter.** No decision has been
made about whether this file will ever receive OKF-style frontmatter.
`_load_user_profile()`'s frontmatter-skip logic handles it correctly if added,
per test coverage — but the decision to add frontmatter to that file has not
been made.

**Open Item 3 — Phase C retrieval path.** Implemented and live-verified
(2026-06-20 session); see session-log entry for detail.

**Open Item 4 — Phase D automatic promotion.** Not started, unchanged.

**Open Item 5 — Future LLM-based entity/relationship extraction.** Whether
this lives inside WikiAgent or remains a separate offline process is the same
structural question already resolved for link-parsing (offline), but has not
been decided for the richer extraction case.

**Open Item 6 — RAG frontmatter regression. CLOSED 2026-06-21.**

*Root cause (identified via read-only diagnostic, 2026-06-21):* `parse_model_xml()` in
`wiki_agent.py` extracted `content` from `create_page` actions via
`action.findtext("content")` (and the `__CONTENT_N__` placeholder path from
`_shield_content_blocks()`) without `.strip()`. Unlike `page_name`/`page_type` — both
stripped two lines above in the same function — `content` was assigned raw. Gemma's
generated XML consistently places a newline immediately after the opening `<content>` tag
(the few-shot `_EXAMPLE` template does not show one); that leading `\n` was written
verbatim to disk, becoming line 0 and pushing the real `---` frontmatter fence to line 1.
`parse_wiki_doc()` checks only `lines[0].rstrip("\r\n") == "---"` for fence detection;
when line 0 is a stray blank, the frontmatter branch is never entered and the entire raw
content — YAML block included — passes through as `body`, reaching `[CONTEXT]` via the
already-correct Step 4 call site (`parse_wiki_doc(doc.content).body[:2000]`). That
2026-06-19 fix was correctly placed; it was defeated by malformed input it had no way to
detect.

*Confirmed affected (both on-disk and in `document_index`):* four model-generated
`research-note` docs — `how-localist-works.md`, `localist-build-order.md`,
`localist-master-project-outline.md`, `localist-software-stack.md` — all written with a
leading blank line by Gemma, all returning `parse_wiki_doc().body == content` (full raw
file with YAML block intact).

*Confirmed unaffected:* `lora-persona.md` and `wiki/users/michael.md` (human-authored,
never pass through `parse_model_xml()`). Both verified byte-identical (`body == content`,
`frontmatter == {}`, `fence_idx = None`) via direct fresh disk-read in a follow-up
confirmation pass. The persona-cache call site in `_load_persona()` also verified
unaffected — `parse_wiki_doc()` takes the `fence_idx = None` path for that file.

*Fix, two layers (locked together — symptom-only fix would leave the malformed files
silently producing stale output on the next ingest cycle):*
1. **Write-time** (`wiki_agent.py`, `parse_model_xml()`): `.strip()` added to
   `raw_content` before assignment into `entry["content"]`, covering both the
   `__CONTENT_N__` placeholder path and the direct-`findtext` path identically.
   Prevents future model-generated pages from carrying a stray leading/trailing blank
   line to disk.
2. **Read-time** (`wiki_doc.py`, `parse_wiki_doc()`): `fence_idx` detection hardened
   to tolerate exactly one leading blank line before the `---` opening fence (bounded,
   not unbounded, to avoid masking unrelated malformed-doc cases). Existing
   no-closing-fence fallback (`frontmatter = {}`, `body = content`) preserved exactly
   unchanged. Fixes the four already-affected files immediately on next RAG fetch —
   no re-indexing required (`document_index.content` stores raw file text; `parse_wiki_doc()`
   runs at read time in Step 4, not at index time; confirmed by re-reading `index_document()`
   and the Step 4 call site fresh).

*Live verification:* query `"localist build order phases development roadmap"` against
live `localist_memory.db` returned all previously-affected docs with clean `[CONTEXT]`
bodies — each starts at `## Summary` with no `---`, `title:`, `type:`, or `query:` YAML
lines. Actual excerpt captured as evidence.

*Test suite:* 279 → 286 (+7: 4 in `test_wiki_doc.py` — leading-blank-parses-frontmatter,
body-clean, no-close-fence-fallback-unchanged, standard-fence-at-line-zero-unaffected;
3 in `test_wiki_agent.py` — strips-leading-newline, strips-trailing-whitespace,
strips-trailing-only), 0 failures.

**Open Item 7 — `build_graph.py` manual-trigger gap. CLOSED 2026-07-01.**

*Originally:* no automated trigger (no hook, no CI step, no runbook) ran
`build_graph.py` after wiki content changes. This is what allowed the live P3c
resolution failure to go undetected until manual testing — the graph was simply
never built against the production database.

*Decision:* three options were considered — a WikiAgent post-ingest hook, a
startup check, and a runbook-only note. Runbook-only was rejected: it is the
same manual-reminder approach that already failed once, and that failure is
literally how this item was discovered. The two remaining options were
combined rather than choosing one — the hook covers the primary ingestion
path live, while the startup check is the safety net for drift from any other
source (manual file edits, restores, between-session drift). The startup
check specifically would have caught the original incident immediately at
boot rather than on a failed live query.

*Fix (two parts, two separate Claude Code prompts, one file each):* Part B
(`main.py`, applied first) — `build_graph` imported and called in `lifespan()`
right after the "MemoryManager ready" log line, wrapped in try/except and
logging a warning (non-fatal) rather than propagating the exception — a
deliberate departure from the unguarded `index_directory()` precedent in the
same function, since graph state is lower-stakes and already degrades
gracefully. Part A (`wiki_agent.py`, applied second) — `build_graph` imported
and called immediately after the existing `index_document()` loop, still
inside the existing `if self._memory_manager is not None and written:` guard
so it only fires when pages were actually written, using the same non-fatal
try/except pattern.

*Live-verified:* Part B — a backend restart showed `"Graph rebuilt at
startup — nodes=5 edges=11 resolved=8 unresolved=3"`, matching the known-good
Phase C baseline. Part A — a real WikiAgent write produced `"Graph rebuilt
after write — nodes=6 edges=13 resolved=10 unresolved=3"`, independently
matching a standalone manual `build_graph.py` run against the same database
exactly; test artifacts were cleaned up afterward and the pre-test state
(nodes=5/edges=11/resolved=8/unresolved=3) was confirmed restored via `git
status` and a final clean re-run.

*Test suite:* 447 passed throughout both prompts, 0 regressions, 0 new tests
(both changes are additive/non-fatal with no new branching logic requiring
dedicated coverage).

*Note:* both call sites use `build_graph()`'s full-rebuild behavior (clears
and re-walks all `graph_edges`) rather than an incremental update — fine at
current corpus size (~6 wiki files), not filed as a numbered open item, just
noted as a forward-looking scaling consideration.

**Open Item 8 — `raw/`-in-RAG via `force_rag` bypass. CLOSED 2026-06-21.**

*Originally:* an unscoped inline observation from the 2026-06-19 evening live-testing session
(not a numbered Open Item at the time — logged as "flagged for evaluation in a future session").
Promoted to a tracked item and closed in the same 2026-06-21 session.

*Root cause:* `controller_agent.py` Step 4's `query_corpus()` call passed no `doc_type` filter,
so `wiki` and `raw` documents were ranked together in a single pool. The Step 4 filter condition
`if (plan.force_rag or doc.relevance_score >= 0.55)` meant that when `plan.force_rag=True` (set
by Priority 4a in `planner.py`, the identity-question route triggered by keywords such as "who
are you", "what can you do", "what is localist"), every top-3 `query_corpus()` result was
included in `[CONTEXT]` with no quality floor. `raw/` source files — structurally different from
curated wiki pages and not intended as direct grounding material for identity questions — could
backfill `[CONTEXT]` slots at scores as low as 0.0070.

*Live reproduction (three real P4a-triggering queries against the running backend —
`"What is Localist?"`, `"Who are you?"`, `"What can you do?"`):* `raw/` files reached `[CONTEXT]`
on every test, always via the `force_rag` bypass (no `raw/` result in any test would have cleared
the 0.55 threshold on its own merit — scores 0.0070–0.4206). Worst case: on `"Who are you?"`,
`lora-persona.md` scored highest (0.5023) but was excluded by the existing persona-exclusion guard,
leaving both remaining `[CONTEXT]` slots backfilled entirely by `raw/` files
(`raw/how-localist-works.md` at 0.4206, `raw/Localist Master Project Outline.md` at 0.4166).

*Design decision:* `raw/` files remain fully eligible for RAG in the normal (non-identity) routing
path, unchanged. For the `force_rag=True` identity-route path specifically, `[CONTEXT]` must never
be backfilled with `raw/` material; the persona doc or other curated wiki content should fill those
slots instead.

*Fix:* `controller_agent.py` Step 4's `query_corpus()` call now passes
`doc_type="wiki" if plan.force_rag else None` — restricting the candidate pool at the source for
the identity-route path rather than adding a second filter pass after the fact. No changes to
`memory_manager.py` — `query_corpus()`'s existing `doc_type` parameter already supported this.

*Live-verified post-fix:* Same three reproduction queries re-run — no `doc_type='raw'` document
appeared in any of the three. Normal (non-identity) RAG path confirmed unaffected:
`query_corpus(doc_type=None)` returns `raw/` documents as before; the `doc_type="wiki"` filter is
applied only when `force_rag=True`.

*Test suite:* 286 → 288 (+2 tests in `test_controller_phase4.py`, class
`TestForceRagDocTypeFilter`: `test_force_rag_true_calls_query_corpus_with_wiki_doc_type` and
`test_force_rag_false_calls_query_corpus_with_no_doc_type_filter`), 0 failures.

**Open Item 9 — Empty `[CONTEXT]` on identity-route queries. CLOSED 2026-06-22.**

*Originally:* observed in the same live-verification pass as Open Item 8's fix (2026-06-21) — for
two of the three identity-route reproduction queries (`"Who are you?"` and `"What can you do?"`),
`query_corpus(doc_type="wiki")` returned only `lora-persona.md` as a relevant wiki candidate, which
the existing persona-exclusion guard then removed, leaving `[CONTEXT]` empty. Logged as open, no
fix direction decided, pending a live-tested diagnostic pass across more identity-phrasing variants.

*Diagnostic pass (2026-06-22):* all 13 `_IDENTITY_KEYWORDS` phrasings from `planner.py` were run
through a read-only probe against the live backend, capturing each query's full top-3
`query_corpus(doc_type="wiki")` result set plus a direct cosine-similarity score against
`lora-persona.md` specifically (independent of whether persona made the top-3). Result: 11 of 13
phrasings returned populated `[CONTEXT]` (1–2 survivors after persona exclusion); the same two
phrasings from the original observation (`"Who are you?"`, `"What can you do?"`) remained empty.
Persona similarity for the two empty cases (0.490, 0.484) was solidly mid-range, ruling out
"persona's score is unusually dominant" as the mechanism — both cases returned only one document
in their top-3 entirely, with that document being `lora-persona.md`.

*Two candidate mechanisms were proposed and disproven before the actual root cause was found —
preserved here deliberately, not smoothed over, per this project's standing discipline of stating
plainly when an informal description turns out wrong on fresh investigation:*

1. *Keyword-Jaccard bottleneck (disproven).* Hypothesis: `query_corpus()`'s two-stage pipeline
   (rank all docs by keyword Jaccard overlap, re-rank the top `2×max_results` by cosine) was
   producing a shrunken candidate pool for these two low-keyword-overlap phrasings. Direct
   inspection of `query_corpus()` disproved this: `pool = scored[:max_results*2]` and
   `top = scored[:max_results]` are unconditional slices with no internal threshold, dedupe, or
   early-exit — the function's own logic guarantees exactly `max_results` results whenever at
   least that many documents of the requested `doc_type` exist, regardless of score values. A
   live corpus-size check (`document_count(doc_type="wiki")` = 6) confirmed the corpus itself
   was never the constraint either.
2. *Relative-path cache drift (disproven).* A first live trace of the two failing queries showed
   `_check_cache()` returning a hit, with a cached payload whose paths appeared to be short
   filenames (`lora-persona.md`) rather than the absolute paths `document_index` currently stores
   — suggesting a path-format migration had silently broken cache hydration. A second, deeper
   trace disproved this directly: the short filenames were a display artifact of the trace
   script itself (printing `Path(e["path"]).name` instead of the full stored path); the underlying
   cache payload always contained correct, matching absolute paths. `git log` confirmed
   `index_document()` has used `Path(path).resolve()` since the very first commit that introduced
   `MemoryManager` — there was never a relative-path era for this table.

*Actual root cause:* `_query_hash(query, top_n)` in `memory_manager.py` hashed only the query
string and `max_results` — `doc_type` was never part of the cache key. `query_corpus()` calls this
hash with the same `query`/`max_results` regardless of `doc_type`, so a `retrieval_cache` entry
written for one `doc_type` (e.g. `None`, wiki+raw combined) could be served as a hit for a later
call with a different `doc_type` (e.g. `"wiki"`). `_hydrate_cache_result()` then filters the
already-hydrated cached docs down to the requested `doc_type` *after* retrieval, silently dropping
any cached docs of the wrong type. Both originally-failing queries had real, valid (`valid=1`)
cache entries written for `doc_type=None` at an earlier point — `"Who are you?"` on 2026-06-18,
`"What can you do?"` on 2026-06-21 — each containing 3 absolute paths (a mix of `wiki/` and `raw/`
docs). On a `doc_type="wiki"` call, only the single `wiki/` doc in each cached payload survived
the post-hoc filter, and that doc was `lora-persona.md` in both cases — which the persona-exclusion
guard then removed, yielding empty `[CONTEXT]`. This is not specific to P4a or to identity
questions: any caller of `query_corpus()` that varies `doc_type` across calls sharing the same
query text and `max_results` is exposed to the same collision. It happened to surface through the
P4a route because P4a is the only caller that forces `doc_type="wiki"` on text that other routes
or earlier sessions may have queried with `doc_type=None`.

*Fix:* `_query_hash()`'s signature extended to `_query_hash(query: str, top_n: int, doc_type: str
| None)`, with `doc_type` folded into the hashed string. Its one call site, inside `query_corpus()`,
updated to pass `doc_type` through. No other method (`_write_cache`, `_check_cache`,
`_hydrate_cache_result`) required modification — `_write_cache` already accepted a pre-computed
hash string and `_check_cache`/`_hydrate_cache_result` are agnostic to how the hash was derived.
No schema change — `doc_type` enters the hash input only, not a stored column. Existing cache rows
computed under the old 2-field hash become unreachable under the new 3-field key and are left in
place rather than purged; this is harmless and intentional — a fresh 3-field-keyed cache miss now
falls through correctly to a real keyword+embedding re-rank for any query previously polluted by a
cross-`doc_type` collision.

*Separately found, separately fixed (not folded into this root cause, by deliberate choice — see
§10's precedent for treating co-occurring failure shapes independently):* `backfill_embeddings.py`
writes directly to `document_index.embedding` via its own raw `sqlite3.Connection`, bypassing
`MemoryManager` and never calling `_invalidate_cache()`. A single `UPDATE retrieval_cache SET
valid = 0` was added once after the script's embedding-update loop completes (not per-row),
matching the script's existing raw-SQL pattern rather than refactoring it to construct a
`MemoryManager`.

*Live-verified, in stages, against three different conditions before the real one was confirmed —
preserved here as a worked example of the project's "verify the mechanism, not just a
symptom-correlation" discipline, the second such pattern this arc surfaced after mount-staleness:*

1. A first re-run returned 3 docs for both queries — but against a freshly-reindexed, *empty*
   database using the keyword-only fallback path (no embed model loaded), which is a different
   code branch than the one that produced the original bug. Confirmed the fix's mechanism in
   isolation; did not confirm it against the original failure's actual conditions.
2. A second re-run, intended to use the real database, was discovered to have connected to
   `lora_memory.db` — a known stray, empty, unreferenced database left over from an earlier
   wrong-target `build_graph.py` run (see §8, Validation-Run Results) — rather than the real
   production database. This was caught before being accepted as evidence, the same discipline
   applied to source-file mount staleness now applied to database-file ambiguity.
3. A corrected final run confirmed, from source (`main.py` → `backend/.env`'s
   `LOCALIST_MEMORY_DB` setting → resolved working-directory path), the real database path
   (`backend/localist_memory.db`); confirmed the original two stale `retrieval_cache` rows
   (same `query_hash`, same `created_at` timestamps as originally traced) were still present and
   still `valid=1` in that real database; computed both the old 2-field hash and the new 3-field
   hash for both queries side by side, showing them to be different values (non-collision
   demonstrated directly, not inferred); and re-ran both queries with the real `EmbeddingEngine`
   against the real database, returning 3 documents each at cosine-similarity-range scores
   (0.39–0.49, as opposed to the 0.0–0.05 range a keyword-only fallback would produce — confirmed
   explicitly to rule out a repeat of stage 1's branch ambiguity).

*Known, accepted gap:* verification in stage 3 constructed a standalone `MemoryManager` pointed at
the confirmed real database and real `embed_fn`, rather than exercising the actual running FastAPI
backend end-to-end through its HTTP endpoint — the backend was not running at verification time.
`controller_agent.py`'s P4a branch is a thin wrapper around the identical `query_corpus()` call
shape that was tested, so divergence risk is low, but this was not a full HTTP-level confirmation
and is recorded as such rather than overstated.

*Test suite:* 339 → 342 (+3 tests in `tests/test_memory_phase1.py`, class `TestQueryHash`:
hash differs for `doc_type=None` vs `"wiki"`, differs for `"wiki"` vs `"raw"`, and is stable for
identical inputs), 0 failures.

**Open Item 10 — `_priority4a_identity()` missing `priority` field. CLOSED 2026-06-21.**

*Originally:* an unanalyzed observation noticed during the same live-reproduction pass used for
Open Item 8 — `_priority4a_identity()` was described informally as "returning `priority=4` in
its RoutingPlan but live runs showed `priority=6`." On fresh investigation this description was
inaccurate in its premise: the function does not set `priority` at all.

*Root cause:* `_priority4a_identity()` in `planner.py` constructs its `RoutingPlan` return value
without passing a `priority=` argument. `RoutingPlan.priority` defaults to `6` — the same default
used by `_priority6_direct()`, an unrelated fallback at the opposite end of the routing chain. Every
other `_priorityN_*` method in `planner.py` sets this field explicitly (priorities 1, 2, 3, 3, 4);
`_priority4a_identity()` was the sole outlier, so every identity-route plan silently inherited the
P6 default.

*Impact:* purely metadata/observability. `plan.priority` is consumed in exactly one place in the
entire codebase — `controller_agent.py`'s `ControllerResult.metadata` dict — with no influence on
actual routing control flow. `route()`'s evaluation order, and the returned `agent`, `fetch_rag`,
and `force_rag` were all already correct. Only the reported `priority` value in response metadata
was wrong: every identity question's response metadata reported `"priority": 6` when it should
have reported `"priority": 4`.

*Fix:* `priority = 4` added to the `RoutingPlan(...)` construction in `_priority4a_identity()`,
matching the function's name and its documented position in the evaluation order. The
`RoutingPlan.priority` default (`6`) is unchanged — it is correct and intentional for
`_priority6_direct()`'s use; the bug was the missing explicit override in P4a.

*Live-verified:* re-ran the three reproduction queries (`"What is Localist?"`, `"Who are you?"`,
`"What can you do?"`) — all three now report `priority=4` (previously `6`); `force_rag=True` and
`agent='conversational_agent'` unchanged, confirming no behavioral change, only the metadata
correction.

*Test suite:* 288 → 289 (+1: `test_p4a_identity_returns_priority_4` in
`tests/test_planner_phase3.py`, class `TestPlannerPriorities`), 0 failures.

*Note: P4a, and the `force_rag` mechanism these three Open Items (8, 9, 10) describe, were removed entirely on 2026-06-26 — see Open Item 12 below.*

**Open Item 11 — Fabricated tool-call syntax in generation output. OPEN, mechanism unknown,
2026-06-22.**

*Originally:* a single live turn produced a fabricated tool-call string as the model's entire
visible output, in place of a synthesized answer. Instruction: `"Do a web search then tell when
Microsoft's first formal investment in OpenAI was?"`. Backend logs confirmed routing, LangSearch
dispatch, and prompt assembly all completed correctly — `[TOOL RESULTS]` in the assembled user
prompt contained three real search results before generation. The model's raw completion was:

```
<|toolcall>call:websearch{query: "when was microsoft's first formal investment in openai"}<tool_call|>
```

This tag matches no real format used anywhere in this codebase. `OMLXRuntimeClient.infer()`'s
chat-completions payload contains no `tools` or `tool_choice` field at all — confirmed by direct
inspection of `omlx_runtime_client.py` — so there is no real tool-calling contract for the model
to be honoring or malforming. The string was invented by the model, most likely reflecting
tool-call-shaped patterns present in its training data despite this harness never exposing that
capability.

*Diagnostic (read-only, same day):* a standalone script (`diagnostics/diag_toolcall_fabrication.py`)
reconstructed the exact system prompt, `[TOOL RESULTS]` block, and `[WORKING MEMORY]` block from
the incident's backend log as a fixed fixture, varying only the final `[INSTRUCTION]` line across
4 phrasing variants — including the original instruction verbatim (Variant A) — at 5 repeat runs
each (20 total live `OMLXRuntimeClient.infer()` calls, `temperature=0.30`, `max_tokens=1024`,
matching the incident's real call parameters). Variants tested: (A) original exact phrasing, (B)
search reframed as already-done ("Based on the search results..."), (C) no mention of search at
all, (D) explicit statement that search already happened.

*Result:* 0/20 fabrications. Every run across all four variants correctly treated `[TOOL RESULTS]`
as already-resolved search content and produced a grounded (if sometimes hedged/inconclusive)
answer rather than fabricating a tool-call string. This closes the original phrasing hypothesis —
the literal instruction "do a web search" is not, on its own, a reliable trigger — but does not
explain the original incident, which did occur once, live, under what appears to be the same
prompt shape.

*Mechanism: unknown.* The diagnostic fixture is a faithful reconstruction of what the backend log
*displayed*, but is not a guaranteed faithful reconstruction of full live session state at the
exact moment of the incident — e.g. the true stored `[WORKING MEMORY]` turn content (persisted via
`MemoryManager.get_context_window()`) could in principle diverge from what a finite log excerpt
showed, and that possibility has not been ruled out. No diagnostic has yet tested temperatures
other than 0.30, run counts beyond 20 per variant, or working-memory content other than the one
fixture pulled from the original log excerpt.

*Status:* not reproduced; not root-caused; no fix direction proposed or evaluated. Logged as a
single confirmed live occurrence with unknown recurrence rate. Per this project's standard
discipline for under-specified findings, this should not be treated as fix-ready until either (a)
it recurs and a fuller live state capture is available, or (b) a wider diagnostic sweep (higher
run count, varied temperature, varied working-memory content) establishes a non-zero reproduction
rate. A passive detection guard in `conversational_agent.py` (flagging this output pattern at
generation time and logging the full real prompt that produced it) has been suggested as a future
non-fix instrumentation step, not yet scheduled or implemented.

*Cross-reference (2026-06-23):* §9.5 Open Item 4 confirms, via live diagnostic, a structurally
different but topically related issue on a different call (`extract_working_state_update()`,
`max_tokens=200`) — the model emits a `reasoning_content` delta stream that consumes the full
token budget before any parseable output reaches `content`. This is **not** offered as an
explanation for this item's fabricated tool-call string, which occurred on the main conversational
call (`max_tokens=1024`) under different parameters and remains independently unreproduced and
unexplained. Noted only because both findings involve this model/serving setup producing unexpected
output shaped around its own internal process, on calls this codebase's parsers were not written
expecting. Do not treat Open Item 4 as having root-caused this item.

*Second live occurrence, 2026-06-24, 12:34 — different trigger shape, real backend log captured
directly (not reconstructed from a screenshot/chat excerpt).* Instruction:
`"What do you know about LangSmith Engine?"`. Unlike the original incident, **no tool fired**:
Priority 3's semantic gate scored `knowledge_request_open` highest (0.643) with `gate_fired=False`,
so the plan carried `tools=[]`. The conversational call (`temp=0.30, max_tokens=1024,
prompt_chars=610`, full `[TOOL RESULTS]` block absent from the prompt — there was none to include)
returned, as the model's entire visible answer:

```
<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|>
```

This is the **inverse trigger condition** from the original incident, not a repeat of it. The
2026-06-22 case fabricated a tool-call string *after* a real `web_search` had already run and
real results were sitting in `[TOOL RESULTS]` — fabrication there meant ignoring grounded content
already provided. This 2026-06-24 case fabricated the *same shaped* string when **no tool was ever
offered or dispatched for that turn** — `tools=[]` — on a topic outside the model's training
knowledge. Read naturally, this looks less like a malformed reaction to tool output already present
and more like the model attempting to request a tool call that this harness simply does not expose
(`OMLXRuntimeClient.infer()`'s payload has no `tools`/`tool_choice` field, confirmed previously and
still true). Both incidents share the same malformed delimiter pattern (`<|tool_call...` /
`...<tool_call|>`, never a real matched tag pair in any format this codebase uses), which is itself
notable — two independent live incidents, twelve days apart, different trigger shapes, producing
near-identical syntactically-broken tool-call tokens suggests the *string itself* is something
the base model reaches for, rather than something assembled fresh from prompt content each time.
This is offered as an observation, not a confirmed mechanism.

*New finding not present in the original incident: propagation into a second, independent call.*
The fabricated string was stored verbatim as that turn's answer in `[WORKING MEMORY]`
(`Turn -2 [agent]: {'answer': '\n<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|>', ...}`).
The Tier 2 working-state-update call for that same turn — a separate `infer_stream()` call,
`temp=0.00`, prompt built from this same contaminated working-memory content — returned a near-
identical string (`'\n<|tool_call>call:web_search{query:<|"|>LangSmith Engine<|"|>}<tool_call|><eos>'`),
and `extract_working_state_update()` correctly logged this as `PARSE_FAILURE` (`missing label(s)`)
rather than silently accepting it — the existing parse-failure guard from Open Item 4's diagnostic
work did its job here. This establishes that a fabrication in the main conversational answer can
**propagate into a second, structurally unrelated call** simply by virtue of being stored as normal
turn history and later re-read as context — a blast-radius fact, not a root-cause fact. It does not
mean Open Item 11 and Open Item 4 share a mechanism (they remain logged separately, per the
cross-reference above); it means Open Item 11's failure mode, once it occurs, is not necessarily
contained to the single turn it occurs on.

*Adjacent, unverified observation — not part of this finding, logged separately so it isn't lost:*
the same live chat session reportedly included a model-generated remark about oMLX cache state
("cache is building with each turn"). No `/admin/api/cache/probe` call or dashboard read appears
anywhere in the captured backend log for this session, so this claim cannot be checked against
real cache telemetry from the evidence in hand. Flagged because, if accurate as a description of
what the model said, it would be a third instance of the same class of behavior as this item and
Open Item 4 — the model narrating something about its own serving/runtime internals that it has no
actual introspection path to — but on a different surface (plain conversational prose instead of
malformed tool-call tokens) and with no raw evidence yet captured. Not logged as its own Open Item
pending an actual occurrence with backend log coverage.

*Status (updated 2026-06-24):* now two confirmed live occurrences, not one — still not root-caused,
still no fix direction proposed or evaluated, recurrence rate still unknown (n=2 live, against
indeterminate live turn volume). The original diagnostic's 0/20 isolation result is **not**
contradicted by this new incident, since the new incident's prompt shape (`tools=[]`, no
`[TOOL RESULTS]` block) was never one of the four variants tested — the diagnostic sweep covered
only the "tool already ran" trigger shape. A natural next diagnostic step (not yet scheduled) would
extend `diag_toolcall_fabrication.py` with a fifth variant matching this incident's actual shape:
no tool dispatched, `[TOOL RESULTS]` absent, topic outside training knowledge, `temp=0.30`. Two data
points with different trigger shapes still does not license a unified mechanism claim — it licenses
widening the diagnostic, which remains the next concrete step if this is picked up before it
recurs again.

*Third, fourth, and fifth live occurrences, 2026-06-25 — deliberate live reproduction attempt,
three turns, real backend logs, isolating priming vs. instruction phrasing as candidate variables.*
Following the 2026-06-24 update's open question (what trigger shape actually produces this), three
live turns were run specifically to test whether fabrication requires priming from an immediately
preceding real tool-dispatch turn, or is driven by something else.

**Turn 1 ("Test A"), 10:49 — priming present, turn lands on Priority 6, `[TOOL RESULTS]` absent.**
Instruction: `"Can you look up Apples price hike for the MacBook Neo and iPad?"`. Preceded in the
same session by a real `tools=['web_search']` turn (`"What's the latest Apple News?"`) with results
delivered normally. On the test turn: `lookup_request` scored 0.593 (`gate_fired=False`); Priority 4
missed (`top_score=0.424`); Priority 5 no match; **Priority 6 — direct answer fallback**, `tools=[]`.
Assembled user prompt contained only `[WORKING MEMORY]` + `[INSTRUCTION]` — no `[TOOL RESULTS]`, no
`[CONTEXT]`. Model's entire output:

```
<|toolcall>call:web search{queries:[<|"|>Apple price hike MacBook Neo iPad<|"|>,<|"|>MacBook Neo price change<|"|>,<|"|>iPad price increase<|"|>]}<toolcall|>
```

**Turn 2 ("Test B"), 11:11 — priming present, turn lands on Priority 4 (corpus), `[TOOL RESULTS]`
absent but `[CONTEXT]`/`[USER PROFILE]`/`[WORKING STATE]` all present and populated, topic-mismatched.**
Instruction: `"Can you look up their next-generation in-house Microsoft AI models?"`, following a real
`tools=['web_search']` turn (`"What's the latest Microsoft news?"`) in the same session. `lookup_request`
scored 0.598 (`gate_fired=False`); **Priority 4 matched via corpus score (0.582 ≥ 0.550)** — `tools=[]`,
`fetch_rag=True`. The RAG hit pulled two Localist-architecture wiki docs (`localist-master-project-
outline.md`, `localist-software-stack.md`) that have no topical relevance to Microsoft's AI models —
matched on shared technical vocabulary ("AI models," embeddings) rather than subject. `prompt_chars=4874`,
including real prior-turn search results in `[WORKING MEMORY]`. Chat-pane tag: `P4 · Vault ◈ grounded`.
Model's entire output:

```
<|toolcall>call:websearch{query: "next-generation in-house Microsoft AI models Build 2026"}<tool_call|>
```

**Turn 3 ("B1"), 11:17 — no priming (fresh task, no preceding turn in working memory at all), turn
lands on Priority 4 (corpus), same topic-mismatch shape as Test B.** Instruction: `"Can you look up
Microsoft's next-generation in-house AI models?"` — first and only turn in this task; `Turn -1` is the
sole `[WORKING MEMORY]` entry, no prior agent response, fresh `mem_key`. `lookup_request` scored 0.598
(`gate_fired=False`); Priority 4 matched via corpus score (0.584 ≥ 0.550) — `tools=[]`, `fetch_rag=True`,
pulling the same two irrelevant Localist-architecture docs. `prompt_chars=3883`. Chat-pane tag: `P4 ·
Vault ◈ grounded`. Model's entire output:

```
<|toolcall>call:websearch{query:<|"|>Microsoft next-generation in-house AI models<|"|>}<tool_call|>
```

*Interpretation.* Turn 3 (B1) is the decisive result: it reproduces fabrication with **no priming
turn at all**, ruling out "immediately preceded by a real tool-dispatch turn" as a necessary
condition — Test A had priming with an empty downstream prompt, Test B had priming with a populated
(but topically irrelevant) downstream prompt, and B1 had neither priming nor relevant context, yet
produced the same failure. The one factor constant across all three of today's reproductions, the
2026-06-22 original incident, and the 2026-06-24 second incident is **`tools=[]` on the turn that
produced the fabrication** — no exception across five live occurrences to date. The three 2026-06-25
turns additionally share an instruction phrased with an explicit "look up" verb, and a `lookup_request`
semantic score consistently in a narrow 0.593–0.598 band — below the 0.65 gate threshold but well
above a clean miss — across all three, despite three different downstream routing outcomes (Priority
6 empty fallback; Priority 4 RAG hit with irrelevant content; Priority 4 RAG hit with irrelevant
content and no priming). This is read as suggestive that "look up"-phrased instructions landing on a
`tools=[]` turn are a stronger candidate trigger than priming, tool-result-emptiness, or RAG-content
relevance individually — each of which varied across the three turns while the outcome did not.

*This remains a hypothesis, not a confirmed mechanism.* Promoted here from "candidate" to "leading
hypothesis" on the strength of three converging live data points plus one clean disconfirmation
(B1 against the priming theory), per this project's standard for distinguishing hypothesis-consistent-
evidence from confirmed mechanism. Not yet tested: (a) whether the "look up" phrasing is doing real
work versus any instruction landing on `tools=[]`-with-lookup-shaped-semantic-score regardless of
literal verb choice — the originally-proposed B2 variant (priming held constant, non-"look up"
phrasing) was not run this session and remains a natural next check if this is revisited; (b) whether
the 0.59–0.60 score band itself is load-bearing (a near-miss specifically) versus any `lookup_request`
score below 0.65; (c) whether the system prompt's "Your Tools" section framing — "Web search fires
automatically on factual queries" — is contributing by setting an expectation the model then
"completes" via fabricated syntax when that automatic firing doesn't happen on a given turn; this is
plausible given the consistent malformed-but-tool-call-shaped string across all five occurrences, but
untested.

*Status (updated 2026-06-25, superseded later same day — see the generation-time backstop and gate
threshold entries cross-referenced below):* five confirmed live occurrences total (2026-06-22 ×1,
2026-06-24 ×1, 2026-06-25 ×3). Reproduction rate within today's deliberate three-turn attempt: 3/3.
Leading hypothesis: instructions using explicit lookup/search phrasing, landing on a turn where
`tools=[]` regardless of cause (Priority 6 fallback or a Priority 4 RAG hit that doesn't satisfy the
lookup intent), reliably produce fabricated tool-call syntax as the entire model output. Still not
root-caused at the mechanism level (why the model reaches for this specific malformed string remains
unexplained — see the cross-session observation above that the same broken delimiter pattern recurs
across unrelated trigger shapes). A two-part fix was implemented and live-verified later the same
day: see "Gate-Calibration Fix" and "Generation-Time Backstop" entries immediately below.

**Gate-Calibration Fix (Prompt 1), 2026-06-25.** `_SEARCH_INTENT_TEMPLATES["lookup_request"]` in
`planner.py` was missing coverage for the "Can/Could you + look up/look into + [specific object]"
question-form frame that all three of today's reproductions used — the existing five templates were
all bare imperatives with a vague pronoun object. Four templates were added (`"can you look up"`,
`"can you look that up for me"`, `"could you look up"`, `"can you look into this for me"`), with
`_SEMANTIC_GATE_THRESHOLDS` deliberately left unchanged at first, on the reasoning that this looked
like a paraphrase-coverage gap rather than a miscalibration. Live re-verification of the three
original utterances showed real but insufficient movement: 0.593→0.608, 0.598→0.617, 0.598→0.621 —
all three remained below the 0.65 threshold, and two of three still fabricated on re-test (the third
hit a stale query-cache from an earlier same-day run, not a new confound).

Given this evidence — three consistent live measurements, each landing 0.029–0.042 short of
threshold — and per §10.4 Open Item 3's own stated revisit criterion ("revisit if live false
negatives are observed," now satisfied), `_SEMANTIC_GATE_THRESHOLDS["lookup_request"]` was lowered
from 0.65 to 0.60 (`explicit_search_action` at 0.68 left untouched). **Known, named risk:** the
original 18-utterance diagnostic's per-utterance scores for `lookup_request`'s 7 adversarial
negatives are not available in this document or in any retained diagnostic output, so the new
threshold's negative-side margin is unverified. This is an accepted risk consistent with this
project's existing "shippable-but-not-fully-validated" posture for these thresholds; the named
mitigation is that any live false positive on `lookup_request` (gate fires when no search was
intended) is the signal to revisit this value.

**Full live re-verification, all three utterances, post-threshold-fix:**

| Utterance | Score | gate_fired | Result |
|---|---|---|---|
| "Can you look up Apple's price hike for the MacBook Neo and iPad?" | 0.608 | True | Real `web_search` dispatch, 3 real results, grounded answer, no fabrication |
| "Can you look up Microsoft's next-generation in-house AI models?" | 0.617 | True | Real `web_search` dispatch, 3 real results (Microsoft MAI/Build 2026 announcements), grounded answer, no fabrication |
| "Can you look up their next-generation in-house Microsoft AI models?" | 0.621 | True | Real `web_search` dispatch, same real results, grounded answer, no fabrication |

All three routed via `_priority3_tool()`'s semantic-gate path, confirming Priority 3 evaluates and
short-circuits `route()` before Priority 4 is ever reached on these turns. Test suite (file-scoped,
`tests/test_planner_phase3.py`): 65 → 69 (template addition) → 71 (threshold adjustment + two new
boundary tests), 0 failures throughout. Note: these are file-scoped counts, not full-suite figures —
the last confirmed full-suite total remains 339 (2026-06-22); a full-suite re-run to establish the
current project-wide total has not yet been done.

**Generation-Time Backstop (Prompt 2), 2026-06-25 — closes Open Item 11's user-facing impact, not
mechanism.** The gate-calibration fix reduces exposure for one phrasing family but does not address
generation-time behavior on any `tools=[]` turn regardless of cause. A detection-and-substitution
guard was added directly to `conversational_agent.py`, the call site all five live incidents shared.

Placement was confirmed by tracing real code, not assumed: `controller_agent._dispatch()` writes
each agent's `AgentResult` to memory via `memory.add_agent_result()` immediately, before
`_execute_plan()`'s implicit-extraction and working-state-update post-hooks read the same
`results[0].output["answer"]` value — confirming the only point early enough to prevent propagation
into working memory and Tier 2 extraction is inside `ConversationalAgent.run()` itself, before it
returns.

Detection: `_is_fabricated_toolcall()`, a module-level regex
(`<\|?tool_?call.*?call:web.*?tool_?call\|>`, case-insensitive, dotall), matched against all seven
real fabricated strings observed across the five live incidents to date — covering delimiter
variants (`toolcall`/`tool_call`) and call-target variants (`websearch`/`web_search`/`web search`).
Verified against five negative-control strings, including an adversarial near-miss ("You can call
the web_search tool if needed.") that contains both "call" and "web_search" as separate words
without the contiguous `call:web` substring or the `<|tool...tool_call|>` bracketing — correctly not
matched. No real tool-calling contract exists in any runtime client in this codebase, so any match
is unambiguously fabrication.

On detection, at both the prebuilt-prompt call site (all five live incidents) and the legacy RAG
call site (no live incidents, but identical structural exposure — included for consistency): `answer`
is replaced with a fixed fallback message ("I don't have live search results for that — here's what
I know from training, which may be stale or incomplete."), and `output["grounded"]`/`output["sources"]`
are forced to `False`/`[]` regardless of what they would otherwise have been — confirmed by dedicated
tests that the guard overrides a real `plan.fetch_rag=True` on the prebuilt path and a real
corpus-hit-derived `grounded=True` on the legacy path. No retry is attempted. New test file
`tests/test_conversational_agent_toolcall_guard.py`: 0 → 36, 0 failures.

**What this closes and what it does not.** This closes Open Item 11's user-facing impact: a turn that
fabricates this pattern can no longer surface the malformed string to the user, store it in working
memory, or have it re-read as context by a later turn — the propagation behavior documented above and
re-confirmed live during this same fix's verification pass (the Apple-utterance fabrication appearing
as `Turn -2 [agent]` context on the following turn, before the threshold fix was applied) is now
structurally prevented at the source. This does **not** close Open Item 11's "mechanism unknown"
status — why the model reaches for this specific malformed string when it does remains unexplained.
The model may still attempt to emit the pattern internally; this guard ensures it never reaches the
user or persists anywhere.

**Live verification of the backstop is explicitly limited, not papered over.** Fabrication is
non-deterministic and cannot be reliably forced on demand, unlike the gate-calibration fix's
live-verifiable score. The 36 mocked-runtime tests are the primary confirmation that the guard works
mechanically. If a live recurrence is observed in normal use going forward, the check is: confirm the
returned answer is the fallback message and `grounded=False`/`sources=[]` for that turn.

**Status: Open Item 11's user-facing impact closed (2026-06-25); mechanism remains open and
unexplained.** Both halves of the two-prompt plan (gate calibration; generation-time backstop) are
implemented and verified to the extent each could be.

---

**Open Item 12 — Removal of Priority 4a (`_priority4a_identity()`). CLOSED 2026-06-26.**

*Motivation:* Michael's view was that P4a was unnecessary scope creep once `lora-persona.md`
was rebuilt past 500 tokens, and that the original "I am Gemma 4" incident that P4a was
built to address was most likely caused by the persona document being too short to provide
adequate grounding — not by any structural gap in the priority ladder. This causal claim is
recorded as Michael's stated hypothesis, not as confirmed root cause; the original incident
was deliberately not re-diagnosed as part of this removal.

*Structural removal:*

- **`backend/planner.py`**: Deleted `_priority4a_identity()` method and its section header
  (~50 lines), its call site in `route()` (4 lines), the `_IDENTITY_KEYWORDS` frozenset (13
  phrases), `force_rag: bool = False` from `RoutingPlan`, and `force_rag`-related text from
  docstrings. Also updated `_priority3c_graph_query()`'s RoutingPlan construction (removed
  `force_rag=False`) and its docstring.
- **`backend/controller_agent.py`**: Three `force_rag` consumers simplified:
  `doc_type = "wiki" if plan.force_rag else None` kwarg dropped entirely from the Step 4
  `query_corpus()` call (now defaults to `None`); `if (plan.force_rag or doc.relevance_score
  >= 0.55)` filter reduced to `if doc.relevance_score >= 0.55` (threshold now unconditional);
  `or plan.force_rag  # P4a identity route` removed from `_should_inject_profile`.
- **Confirmed zero remaining functional references:** `grep -rn "force_rag" backend/` returns
  zero results outside of test docstrings describing the removed behavior.

*Tests removed (3):* `test_p4a_identity_returns_priority_4` (`test_planner_phase3.py`
`TestPlannerPriorities`) and both tests in `TestForceRagDocTypeFilter`
(`test_controller_phase4.py`). All three asserted behavior of code that no longer exists;
all were deleted, not adapted. Two incidental fixture fixes were also required and made
(`force_rag=False` removed from two `_make_*_plan()` helpers in `test_controller_phase4.py`
that would have raised `TypeError` post-removal) — not counted in the deletion total.

*Tests added (16):* 13 in new class `TestFormerP4aIdentityPhrasingsRouteToPSix`
(`test_planner_phase3.py`) — one per former `_IDENTITY_KEYWORDS` phrase, asserting the
discovered routing outcome (not assumed). Plus one confirming `doc_type` is absent from
the Step 4 `query_corpus()` call (`TestQueryCorpusNeverReceivesDocType`), one confirming
the relevance threshold is unconditionally enforced (`TestRelevanceThresholdUnconditional`,
doc at score 0.40 excluded with no bypass), and one confirming `RoutingPlan(force_rag=True)`
now raises `TypeError` (`TestRoutingPlanNoForceRagField`).

*Test suite delta:* 405 (baseline) → 402 (3 tests deleted) → 418 (16 tests added), 0 failures.

*Live-verification findings:*

Unit tests (no `embed_fn`, no `MemoryManager`): all 13 former identity phrasings resolved
to `priority=6`, `fetch_rag=False`, `fetch_episodic=False`, `agent=conversational_agent`.
P4 Path B is skipped without MemoryManager; P3 semantic gate does not fire without
`embed_fn`. All 13 phrases reach P6 in the unit-test baseline.

Live backend (real `embed_fn` present): three spot-checked queries showed a divergence from
the unit-test baseline:
- `"What is Localist?"` → priority=6. Corpus top_score=0.547 (below 0.55 threshold; P4
  miss). Semantic gate: best=knowledge_request_open(0.598), gate_fired=False. Received a
  hedging response ("I don't have live search results for that — here's what I know from
  training..."). `how-localist-works.md` was NOT in `[CONTEXT]`.
- `"Who are you?"` → priority=3. Semantic gate: lookup_request=0.631 (≥ 0.60 threshold),
  gate_fired=True. `web_search` dispatched. Response correctly identified as LORA.
- `"What can you do?"` → priority=3. Semantic gate: lookup_request=0.666 (≥ 0.60 threshold),
  gate_fired=True. `web_search` dispatched. Response correctly identified as LORA.

*Interpretation of the P3 routing result:* this is NOT a regression caused by this removal.
`route()`'s evaluation order has always run Priority 3 before Priority 4a — confirmed
directly by reading `route()`'s call order in `planner.py` (P3c → P3 → P3b → P4 → P5 →
P6, with P4a never having existed between P3b and P4 from the routing engine's perspective
once removed). Any phrasing that clears P3's semantic gate would have been caught by P3
regardless of P4a's presence, because P4a never had the opportunity to evaluate those turns
in the old ladder. What this live test surfaced is a pre-existing condition of the semantic
search-intent classifier — cross-reference §10.4 Open Item 3 (thresholds derived from only
18 diagnostic utterances, explicitly flagged for revisiting "if live false-positive signals
are observed"). This session's result is now one such observed instance.

*One finding directly attributable to this removal:* `"What is Localist?"` reached P6 and
missed the corpus threshold narrowly (top_score=0.547 vs. 0.55 cutoff), receiving a hedging
response instead of a grounded one. Under the old ladder, P4a's `force_rag` bypass would
have included `how-localist-works.md` regardless of score. This is the one real, narrow
behavioral change caused by removing P4a — recorded plainly.

*Two open follow-ups, explicitly undecided at the time of this entry:*
1. Whether the `lookup_request` 0.60 threshold should be revisited given this newly observed
   false-positive instance against identity-shaped queries — a change to the semantic
   classifier, not to the routing ladder. Cross-reference §10.4 Open Item 3.
2. Whether the 0.547-vs-0.55 near-miss on `"What is Localist?"` warrants action (e.g.
   lowering the P4 Path B threshold, or a targeted corpus boost for that document) or is an
   acceptable cost of the restored, un-padded routing design.
Neither had been decided at the time of this entry.

*Follow-up 1 update (2026-06-26):* Resolved via `_SEARCH_NEGATIVE_FILTER` expansion rather
than threshold adjustment. Five identity/capability phrases ("who are you", "what are you",
"what can you do", "what can you help with", "what do you do") added to the negative filter,
blocking the false-positive collision before the embedding call. The 0.60 threshold was not
changed. See §10.4 Open Item 3 — Update 2026-06-26 for the full fix record.

*Follow-up 2* remains open and unscheduled.

*Status:* CLOSED. The removal is complete and live-verified.

