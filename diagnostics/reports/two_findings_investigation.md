# Two-Findings Investigation — Discrimination Band & Corpus Duplication

**Status: observational only.** No threshold, no corpus, no production code
was changed as part of this investigation. Where an obvious next step
surfaced, it is named explicitly as a **recommendation for a separate
follow-up prompt**, not implemented here. The originally-reported 0.028
corpus-miss score is out of scope throughout — it is considered closed/noise
per the prior session and is not revisited below.

Companion script: `diagnostics/diag_discrimination_band.py` (new, read-only,
does not modify `diag_compare_embedding_models.py`). Its raw output is also
saved standalone at `diagnostics/reports/discrimination_band.md`. Finding 2
required no new script — it was answered by reading `memory_manager.py` /
`wiki_agent.py` / `main.py` and diffing on-disk file pairs; commands are
inlined below so they're reproducible.

---

## Finding 1 — Narrow discrimination band

### What changed vs. the original run

The original run had **one** true-negative sample (cookies, ~0.39–0.41) and
compared it against **one** broad true-positive sample ("What is this
project about?", 0.55) — a gap of ~0.14–0.16 on n=1 vs n=1. This run widens
both sets to 8 queries each:

- True-negatives: cooking, sports, history, small talk, an unrelated
  well-known software product (Excel), math, travel, biology.
- True-positives: each written to target one specific, unambiguous corpus
  document (build order, design philosophy, LORA's tools, Michael's
  location, runtime backends, project vision, software stack, the
  MEMORY.md snapshot) — not vague/broad queries this time.

### Results (real 16-doc corpus, live MLX + live Ollama)

| | MLX true-neg | MLX true-pos | Ollama true-neg | Ollama true-pos |
|---|---|---|---|---|
| mean | 0.3584 | 0.6898 | 0.3982 | 0.7014 |
| min | 0.2880 | 0.5386 | 0.3664 | 0.5832 |
| max | 0.4654 | 0.7853 | 0.4351 | 0.7574 |
| stddev | 0.0545 | 0.0796 | 0.0229 | 0.0723 |

**Gap between the two distributions:**
- MLX: **no overlap**, clean gap of 0.0732 (neg.max 0.4654 vs pos.min 0.5386).
- Ollama: **no overlap**, clean gap of 0.1481 (neg.max 0.4351 vs pos.min 0.5832).

With a properly sized sample in each bucket, the band is **not actually
narrow** — it was an artifact of n=1 sampling in the original run, plus the
original true-positive query being unusually broad/vague (0.55 sat close to
the negative band; every one of the 8 sharper true-positive queries here
scored 0.54–0.79, comfortably clear of the negative band's 0.29–0.47).
Ollama's separation is roughly 2x wider than MLX's.

*(Aside, not requested but visible in the raw output: MLX's expected-doc
hit rate was 5/8 and Ollama's 4/8 on the true-positive set — e.g. "What is
the MEMORY.md human-readable snapshot?" top-1'd to `how-localist-works`
under MLX instead of `MEMORY`. That's a retrieval-accuracy observation, not
a discrimination-band one; flagging only so it isn't mistaken for
supporting evidence of the band question.)*

### Corpus-homogeneity vs. model-calibration check

Three control documents with zero topical relationship to Localist were
embedded (Lorem Ipsum, a generic chocolate-chip-cookie recipe, a generic
paragraph about Jupiter/astronomy), and the same 8 true-negative queries
were re-scored against them:

| | MLX true-neg vs. real corpus | MLX true-neg vs. control docs | Ollama true-neg vs. real corpus | Ollama true-neg vs. control docs |
|---|---|---|---|---|
| mean | 0.3584 | 0.4176 | 0.3982 | 0.4725 |
| mean delta (real − control) | — | **−0.0592** | — | **−0.0744** |

The delta is **negative for both models** — true-negative queries scored
*higher* against the topically unrelated control docs than against the real
corpus, not lower. Per the check's own logic (a positive drop would support
corpus-homogeneity; a near-zero-or-negative drop supports model-calibration),
this points toward **model score calibration**, not corpus homogeneity: both
embedding models appear to produce a "floor" of ~0.3–0.5 cosine similarity
for almost any short natural-language query against almost any block of
prose, independent of real topical relevance. That floor is not unique to
Localist's thematically narrow corpus.

Supporting detail: the one control doc that *did* have genuine topical
overlap with a query (the cookie-recipe control doc vs. the literal cookie
query) scored 0.67–0.74 — sharply higher than any other true-negative
score against any document, real or control. That's the models correctly
recognizing an actual match. It reinforces that the ~0.3–0.5 range seen
elsewhere is a baseline/floor behavior for unrelated text, not a sign the
models can't discriminate at all.

### Conclusion (observational)

The originally-reported "narrow band" does not replicate at n=8/n=8 — the
two distributions are cleanly separated for both models. The moderate
absolute floor score (~0.3–0.5) that even unrelated text produces looks like
a property of these embedding models' score calibration in general, not a
symptom of this specific corpus's topical narrowness. No threshold
conclusion is drawn from this — that remains a separate decision, and
whatever threshold is chosen should be picked with reference to this wider,
cleanly-separated distribution rather than the single low-n data point that
originally prompted concern.

Full per-query table: `diagnostics/reports/discrimination_band.md`.

---

## Finding 2 — Corpus duplication (wiki vs. raw)

### 1. Is wiki/raw dual-indexing intentional?

**Yes — confirmed intentional by design, not an indexing bug.**

- `backend/main.py:384-394` indexes `wiki_dir` and `raw_dir` as two
  **separate**, deliberate calls at startup:
  `memory_manager.index_directory(wiki_dir, doc_type="wiki", ...)` and
  `memory_manager.index_directory(raw_dir, doc_type="raw", ...)`. Both are
  meant to coexist in `document_index` simultaneously; there is no dedup
  step between them.
- `backend/memory_manager.py:1278-1293`'s `index_document()` docstring
  states the semantics plainly: `doc_type: "wiki" for pages in wiki/, "raw"
  for files in raw/"`. `query_corpus()`'s docstring (line ~1780) likewise
  documents `doc_type=None` (both) as the default, with `"wiki"`/`"raw"`
  as opt-in filters — implying both-by-default is the intended baseline
  behavior, not an oversight.
- `backend/wiki_agent.py` (`WikiAgent.run()`, `_resolve_raw_path()`, ~line
  1068 onward) confirms the actual pipeline: a `raw_path` (original
  uploaded/source file) is read, an LLM restructures it into a schema'd
  wiki page (YAML frontmatter, `## Summary`, `### Extracted Concepts`,
  `### Mapped Pages`, `## Related Pages`, `## Revision History`), and that
  page is written to `wiki_dir` **and separately indexed there** —
  explicitly leaving the original raw file in place and indexed under
  `doc_type="raw"`. Wiki = curated distillation; raw = original source.
  This is the WikiAgent ingestion pipeline working as designed, not two
  copies of the same content by accident.

### 2. Content comparison — actual diffs, not filenames

Three pairs, diffed directly on disk (`diff backend/wiki/X backend/raw/Y`):

**`wiki/how-localist-works.md` (44 lines) vs. `raw/how-localist-works.md` (5 lines)**
Genuinely different, related by topic. Raw is 5 lines of first-person LORA
persona prose ("I am LORA, a local research assistant... I am not ChatGPT
and I am not made by Google..."). Wiki is a fully restructured 44-line
research-note with YAML frontmatter, a `## Summary`, an `### Extracted
Concepts` bullet list, `### Mapped Pages` wiki-links, and a `## Revision
History` stamped "created from how-localist-works.md." Not a duplicate —
wiki is a structural distillation of raw's prose into the wiki schema.

**`wiki/localist-design-philosophy.md` (41 lines) vs. `raw/Localist Design
Philosophy Proposal.md` (41 lines)** — same line count, different content.
Raw is prose with numbered `## 1. Local-First Sovereignty` / `## 2. Hybrid
Inference Model` / ... sections and full paragraphs per pillar. Wiki
compresses each of the 5 pillars into single bullet points under
`### Extracted Concepts`, adds `### Mapped Pages` / `## Related Pages`
wiki-links (`[[michael]]`, `[[localist-software-stack]]`,
`[[how-localist-works]]`) that don't exist in raw at all, and a
`## Revision History` line. Coincidental matching line count, genuinely
different structure and content density.

**`wiki/users/michael.md` (29 lines) vs. `raw/michael.md` (23 lines)**
Overlapping facts (Localist Runtime description, Michael's workflow with
Claude Code) but meaningfully different: raw has "Lives in southern
California" (absent from wiki's version), first-person voice ("I plan
architecture..."); wiki has third-person voice ("Plans architecture..."),
adds session-workflow and prompt-style preference bullets not present in
raw. Related, overlapping, but not identical — each has content the other
lacks.

**Verdict across all 3 pairs: none are byte-identical or near-identical.**
Every pair is "wiki = distilled/restructured version, raw = original
source," exactly matching the WikiAgent pipeline's intended behavior. This
is coexistence-by-design, not a duplication bug.

### 3. Does any live query path already filter/dedupe by `doc_type`?

**No — not in any user-facing chat-turn retrieval path.** Checked every
`query_corpus()` call site:

- `controller_agent.py:778` (persona load) — `doc_type` not passed (defaults
  to `None`, both types), then filters by filename (`"lora-persona" in
  str(d.path)`) after the fact.
- `controller_agent.py:1214` (Step 3b, web_search-failed corpus fallback) —
  no `doc_type` passed.
- `controller_agent.py:1252` (Step 4, RAG fetch on `plan.fetch_rag`) — no
  `doc_type` passed.
- `planner.py:1353` (Priority 3b factual-miss check) — no `doc_type` passed.
- `planner.py:1423` (Priority 4 corpus-score check) — no `doc_type` passed.

The only two `doc_type="wiki"`-filtered call sites in the whole backend are
`memory_manager.py`'s own `reconcile_wiki()` (lines ~1475/1478, wiki-file
resync/orphan-cleanup housekeeping) and `wiki_agent.py:1608`
(`_load_wiki_pages_from_index()`, WikiAgent's own page-cache load) — neither
is a retrieval path a chat turn goes through. (`docs/architecture/
08-graph-retrieval-layer.md` documents an earlier `force_rag` →
`doc_type="wiki"` mechanism on a P4a route; that call site no longer exists
in `controller_agent.py` today — a `grep` for `force_rag` in that file
returns nothing — so that filtering appears to have been since removed or
refactored away, and is not currently live.)

**Conclusion: the duplication observed in the diagnostic script's
unfiltered top-K ranking is not a diagnostic-only artifact.** Every live
production retrieval call (persona load, RAG fetch, web_search fallback,
Planner P3b/P4 corpus checks) queries with `doc_type=None` today, so
wiki/raw pairs compete for the same top-K slots in real chat turns exactly
as seen in `embedding_model_comparison.md`.

### A related methodology note surfaced while checking (3), relevant to both findings

`query_corpus()` (`memory_manager.py:1751-1879`) does **not** do a pure
embedding rank over the whole corpus in production, unlike both diagnostic
scripts. It's two-stage: (1) rank *all* documents by keyword/Jaccard overlap,
(2) only re-rank the **top `2 × max_results`** keyword candidates by
embedding cosine similarity (`max_results` is 1 or 3 at every real call
site above, so only the top 2–6 keyword hits ever get an embedding score
at all). Separately, `index_document()` only embeds `content[:500]` (first
~500 characters) when building the persisted `embedding` column
(`memory_manager.py:1322-1337`), whereas both diagnostic scripts embed
each document's **full** content directly via `embed_fn`. Both diagnostics
therefore measure a cleaner, more exhaustive, full-text embedding
comparison than what production retrieval actually executes — production
is gated by a keyword pre-filter and truncated-to-500-chars document
embeddings. This doesn't invalidate either finding above, but it's a
caveat worth carrying into any future decision that generalizes diagnostic
numbers to production behavior.

---

## Explicitly out of scope / not decided here

- No similarity-threshold change is proposed, for either finding.
- No corpus deduplication or cleanup was performed or is proposed.
- No changes to `memory_manager.py`, `query_corpus()`, `wiki_agent.py`, or
  the Planner.
- The 0.028 miss from the original run was not revisited.

## Recommendations for separate follow-up prompts (not implemented here)

1. If wiki/raw dual-indexing-with-no-dedup is intended to stay permanent
   (looks that way from the code), consider whether production retrieval
   should have a `doc_type` preference/tie-break (e.g. prefer `wiki` over
   `raw` when both clear threshold for the same underlying topic) so a
   3-result RAG fetch doesn't spend 2 of its 3 slots on a wiki/raw pair
   about the same subject. This is a product/ranking decision, not
   something to implement from this observational prompt.
2. The keyword-prefilter-then-embed-rerank gating in `query_corpus()`
   (point 3 above) means any future embedding-model comparison intended to
   predict real production behavior should query through `query_corpus()`
   itself (or replicate its two-stage + 500-char-truncation logic) rather
   than doing a full unfiltered embedding rank, to avoid measuring a
   friendlier retrieval process than what's actually live.
