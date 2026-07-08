## 6. Build-Order Checklist

The dependency chain is strict. Each item depends on all items above it.
No item is begun until all items above it are complete and tested.

> **Session progress** — Phases 1–7 complete, plus KV-Cache Prompt Refactor,
> LangSearch integration, HTTP Fetcher service, Priority 4 rewrite,
> Priority 3b, persona rewrite, episodic memory bug fixes, Localist rebrand,
> Localist UI overhaul (provenance bar, episodic memory panel, full rebrand),
> Fetcher service restored (lxml, readability-lxml pinned in requirements.txt),
> and Graph Retrieval Layer Phase A/B (wiki_doc.py shared parsing helper,
> graph schema v3 migration, offline link-graph builder, WikiAgent link validation).
> Test suite: **224 tests, 0 failures** across 9 test files.
>
> **Files added/modified (all phases):**
> `memory_manager.py`, `prompt_builder.py`, `planner.py`,
> `episodic_extractor.py`, `tool_dispatcher.py`, `controller_agent.py`,
> `conversational_agent.py`, `wiki_agent.py`, `main.py`,
> `wiki/lora-persona.md`, `backfill_embeddings.py`, `embedding_engine.py`,
> `fetcher/__init__.py`, `fetcher/main.py`, `fetcher/models.py`,
> `fetcher/client.py`, `fetcher/extractor.py`,
> `wiki_doc.py` (new), `build_graph.py` (new), `requirements.txt`,
> `tests/test_memory_phase1.py`, `tests/test_prompt_builder.py`,
> `tests/test_planner_phase3.py`, `tests/test_controller_phase4.py`,
> `tests/test_episodic_phase5.py`, `tests/test_tool_dispatcher_phase6.py`,
> `tests/test_integration_phase7.py`,
> `tests/test_wiki_doc.py` (new), `tests/test_wiki_agent.py` (new),
> `tests/test_build_graph.py` (new).
>
> **Post-Phase-7 architectural changes (all reflected above):**
>
> *KV-Cache Prompt Refactor:*
> - Slot ordering redesigned: static-first, volatile-last
> - Persona moved to stable system message (Slot 1b)
> - `PromptBuilder.build()` gained `persona=` parameter
> - `[USER]` label renamed `[INSTRUCTION]`; slots renumbered 3–7
> - `ControllerAgent._load_persona()` added: loads and caches persona once per session
> - KV-cache efficiency: 79.7% confirmed in live session (oMLX dashboard)
>
> *LangSearch integration:*
> - `ToolDispatcher._execute_single_search()` replaced with real LangSearch HTTP call
> - `load_dotenv()` added to `main.py` so `LANGSEARCH_API_KEY` loads at server startup
> - `LANGSEARCH_API_KEY` added to `backend/.env`
>
> *Priority 3b:*
> - New priority between P3 and P4: factual keyword + corpus miss → web search
> - `_FACTUAL_QUERY_KEYWORDS` frozenset added to `planner.py`
> - `_priority3b_factual()` method added to `Planner`
>
> *Priority 4 rewrite:*
> - Scoring-based RAG injection eliminated entirely
> - Priority 4 now fires only on explicit wiki/vault trigger keywords
> - `_WIKI_QUERY_KEYWORDS` frozenset added; `_priority4_corpus()` rewritten
> - `_CORPUS_SCORE_THRESHOLD` no longer used for routing (retained for P3b)
>
> *`_WEB_SEARCH_KEYWORDS` expansion:*
> - `"current"` replaced with multi-word phrases: `"current price"`, `"current version"`,
>   `"current ceo"`, `"current status"`, `"current rate"`
> - `_any_whole_word()` helper added with `\b` regex anchors for single-word keywords
>
> *`_FILE_OP_KEYWORDS` fix:*
> - `"read"` replaced with `"read the file"`, `"read file"` to prevent false
>   positive on `"read this link"` / `"read this URL"`
> - `"open"` replaced with `"open the file"` (same pattern)
>
> *HTTP Fetcher service:*
> - Standalone FastAPI microservice on port 8002
> - Three endpoints: `/fetch`, `/extract`, `/api`; plus `/health`
> - `url_fetch` tool added to `ToolDispatcher`
> - `_FETCH_KEYWORDS` frozenset + URL regex added to `_priority3_tool()`
> - `LOCALIST_FETCHER_URL` added to `backend/.env`
>
> *Persona rewrite:*
> - `wiki/lora-persona.md` rewritten: second-person voice, trust hierarchy,
>   tool awareness (LangSearch + Fetcher), honor code
> - Removed third-person documentation register; added direct behavioral instructions
>
> *Episodic memory bug fixes:*
> - Priority 5 personal reference keywords added: `"my name"`, `"do you remember"`,
>   `"who am i"`, `"what do you know about me"`, etc.
> - Explicit extraction subject normalization: `subject` now derived from normalized
>   `content` string, not raw instruction (see §2.8)
>
> *Graph Retrieval Layer Phase A/B:*
> - `wiki_doc.py` added: `parse_wiki_doc()` / `load_wiki_doc()` returns `ParsedWikiDoc(frontmatter, body, links)`; PyYAML parses ISO dates as `datetime.date`; 12 tests in `tests/test_wiki_doc.py`
> - `controller_agent.py`: `_load_persona()` and `_load_user_profile()` now strip frontmatter via `parse_wiki_doc()` / `load_wiki_doc()` before operating on body; verified zero-behavior-change for current `lora-persona.md` and `wiki/users/michael.md`; `PyYAML>=6.0` added to `requirements.txt`
> - `wiki_agent.py`: `_validate_links()` added — scans Mapped Pages (H3) and Related Pages (H2) only; normalization `link_text.lower().replace(" ", "-")`; wired between `parse_model_xml()` and journaling; flagged links appear in `AgentResult.output["unresolved_links"]` and are logged as warnings; page content is never modified; 8 tests in `tests/test_wiki_agent.py` (new); `_FakeRuntime` established as convention for `run()` tests
> - `memory_manager.py`: `graph_nodes` and `graph_edges` tables added as v2→v3 migration (`_SCHEMA_VERSION = 3`); four new public methods: `upsert_graph_node()`, `upsert_graph_edge()`, `clear_graph_for_doc()`, `clear_graph_edges()`; 6 new tests in `TestGraphSchema` class in `tests/test_memory_phase1.py`
> - `build_graph.py` added: offline two-pass link-graph builder; same normalization rule as `_validate_links()`; same-page-same-target duplicate links collapse to one edge row per `(source_doc_path, target_path)` pair; whole-corpus `clear_graph_edges()` between passes; `doc_path` uses absolute resolved paths matching `document_index.path` convention; 10 tests in `tests/test_build_graph.py` (new)
> - Validation run against real 5-document corpus: 5 nodes, 11 edges, 8 resolved, 3 unresolved — see §8.7

---

### Phase 1 — Memory Substrate

- [x] **1.1** Add `episodes` table to `MemoryManager` SQLite schema
- [x] **1.2** Write and run migration script against existing `lora_memory.db`
- [x] **1.3** Implement `EpisodicMemoryWriter`: insert, supersede, retract
- [x] **1.4** Implement `EpisodicMemoryReader`: all three retrieval modes (§2.6)
- [x] **1.5** Implement summarization contract (§2.7) as `format_episodic_summary()`
- [x] **1.6** Add `max_tokens` parameter to `get_context_window()` with 300-token ceiling
- [x] **1.7** Unit tests: lifecycle transitions, retrieval modes, summarization output

---

### Phase 2 — Prompt Contract

- [x] **2.1** Implement `PromptBuilder` class with all seven slot methods
- [x] **2.2** Implement token ceiling enforcement for slots 3, 4, 5, 6
- [x] **2.3** Implement clean omission of empty optional slots (no empty labels)
- [x] **2.4** Replace prompt assembly in `ConversationalAgent` with `PromptBuilder.build()`
- [x] **2.5** Replace prompt assembly in `WikiAgent` with `PromptBuilder.build()`
- [x] **2.6** Unit tests: slot ordering, ceiling enforcement, empty slot omission, round-trip output

---

### Phase 3 — Planner Rewrite

- [x] **3.1** Implement `RoutingPlan` dataclass
- [x] **3.2** Implement Priority 1–4 as deterministic rule evaluations (no inference)
- [x] **3.3** Implement Priority 5 episodic relevance — deterministic keyword check
- [x] **3.4** Implement Priority 6 direct answer fallback
- [x] **3.5** Implement compound instruction detection and sequencing
- [x] **3.6** Replace existing `Planner` inference-based routing with new rule engine
- [x] **3.7** Integration tests: each priority level fires correctly, compound cases sequence correctly

---

### Phase 4 — Controller Integration

- [x] **4.1** Update `ControllerAgent.handle_task()` to consume `RoutingPlan`
- [x] **4.2** Implement the 7-step execution contract (§4.4)
- [x] **4.3** Wire `PromptBuilder.build()` as the single prompt assembly point
- [x] **4.4** End-to-end integration test: ingest path, RAG path, direct answer path

---

### Phase 5 — Episodic Extraction Pipeline

- [x] **5.1** Implement deterministic signal detection (explicit memory commands)
- [x] **5.2** Implement model-based extraction call with direct prompt construction
- [x] **5.3** Implement confidence scoring for model-extracted episodes (0.6–0.9 range)
- [x] **5.4** Wire extraction pipeline into `ControllerAgent` post-response hook
- [x] **5.5** Integration tests: explicit signals produce confidence=1.0 records, model
             extraction produces correctly typed and scored records
- [x] **5.6** Subject normalization: explicit extraction derives subject from normalized
             content string, not raw instruction (§2.8)

---

### Phase 6 — Tool Dispatcher

- [x] **6.1** Define `ToolResult` dataclass and tool dispatcher interface
- [x] **6.2** Implement `web_search` tool — LangSearch API integration
- [x] **6.3** Implement `file_op` tool (read, write, append — sandboxed)
- [x] **6.4** Implement `url_fetch` tool — calls Fetcher service `/extract`
- [x] **6.5** Wire tool results into Slot 5 via `PromptBuilder`
- [x] **6.6** Integration tests: tool results appear in correct slot, token ceiling enforced

---

### Phase 7 — Final Integration

- [x] **7.1** Full pipeline test: instruction → Planner → fetches → PromptBuilder → agent → response
- [x] **7.2** Episodic extraction fires correctly on real conversations
- [x] **7.3** Working memory window enforces 300-token ceiling across session
- [x] **7.4** Persona loaded from wiki and injected into system message as Slot 1b
- [x] **7.5** All agents use `PromptBuilder.build()`. No agent assembles its own prompt string.
- [x] **7.6** Prompt logging enabled: every inference call writes its assembled prompt to debug log

---

### Fetcher Service

- [x] **F.1** Standalone FastAPI service on port 8002
- [x] **F.2** `POST /fetch` endpoint — raw HTTP fetch
- [x] **F.3** `POST /extract` endpoint — readability extraction
- [x] **F.4** `POST /api` endpoint — JSON REST fetch
- [x] **F.5** `url_fetch` tool wired into `ToolDispatcher`
- [x] **F.6** URL regex + explicit keyword triggers in `_priority3_tool()`
- [x] **F.7** End-to-end verified: GitHub release URL fetched and summarized correctly

---

