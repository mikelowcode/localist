# Localist Framework

Localist Framework is a local-first, agentic general assistant running entirely on macOS Apple Silicon. The system maintains persistent memory across sessions, fetches and reads live web content, searches indexed documents, and routes every query through a deterministic priority engine before a single token of inference is spent.

Localist is inference-engine-agnostic. It ships with oMLX, Ollama, and Azure AI Foundry support out of the box, swappable via a single config variable; MLX-LM, LM Studio, and other local runtimes can be added the same way. The Ollama backend also supports Ollama Cloud models (e.g. `gemma4:31b-cloud`), which proxy chat completions through Ollama's cloud API over the same local daemon — the one case where inference leaves the device; embeddings always run locally via `EmbeddingEngine` regardless of which chat backend is active.

---

## Architecture

The frontend (SvelteKit) sends requests over HTTP to the main backend — a FastAPI application on port 8001. The backend's `ControllerAgent` receives each task, runs it through the `Planner` (a priority-ordered rule engine), and dispatches to the appropriate agent: `ConversationalAgent` for answers and `WikiAgent` for document ingestion. When a query requires a tool, `MCPToolDispatcher` opens an MCP session (SSE transport) to **localist-mcp**, a standalone MCP server on port 8003 that exposes `web_search` (LangSearch), `fetch_url` (HTML extraction via `readability-lxml`), and the `file_op` tools (`read_file`/`write_file`/`append_file`). The legacy `ToolDispatcher` and the standalone Fetcher microservice (port 8002) have both been retired — their functionality now lives on localist-mcp. All inference goes through a `BaseRuntimeClient`-conforming runtime, selected at startup via `LOCALIST_RUNTIME_BACKEND` and constructed by `runtime_factory.py`: `OMLXRuntimeClient` (OpenAI-compatible HTTP API), `OllamaRuntimeClient` (Ollama's native `/api/chat`, NDJSON streaming — works for both local Ollama models and Ollama Cloud models proxied through the same local daemon), or `FoundryRuntimeClient` (Azure AI Foundry). Swapping backends is a config change, not a code change. Vector embeddings are always local via `EmbeddingEngine`, independent of the active chat backend. Episodic memory and vector embeddings are stored in a SQLite database with WAL mode enabled and survive server restarts.

```
Localist UI  (localist-ui/)
     │  HTTP
     ▼
FastAPI — port 8001  (backend/main.py)
     │
     ▼
ControllerAgent  →  Planner  →  RoutingPlan
     │
     ├──► ConversationalAgent  →  PromptBuilder  →  Runtime (oMLX / Ollama / Foundry)
     │         │
     │         ├── MemoryManager (SQLite episodic + RAG)
     │         └── MCPToolDispatcher
     │               │  MCP / SSE
     │               ▼
     │         localist-mcp — port 8003  (backend/mcp_server/)
     │               ├── web_search  (LangSearch API)
     │               ├── fetch_url   (readability-lxml extraction)
     │               └── read_file / write_file / append_file
     │
     └──► WikiAgent  →  Runtime (oMLX / Ollama / Foundry)
```

---

## Prerequisites

- macOS Apple Silicon
- Python 3.13
- A runtime backend — one of:
  - oMLX running a chat model (default `gemma-4-e4b-it-4bit`) on port 8000
  - [Ollama](https://ollama.com) running locally (default port 11434) with a model pulled (`ollama pull <model>`) — including Ollama Cloud models, via `ollama signin` + `ollama pull <model>-cloud`
  - Azure AI Foundry (`LOCALIST_RUNTIME_BACKEND=foundry`)
- Node.js (for Localist UI)

---

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running Localist

Use the **Localist CLI** launcher to start all three services with a single command:

```bash
./start_localist.sh
```

This starts the Localist backend (port 8001), the localist-mcp server (port 8003), and the frontend (port 5173), tails all three logs to the terminal with `[backend]`, `[mcp]`, and `[frontend]` prefixes, and shuts them all down cleanly on Ctrl+C.

To stop all services from a separate terminal:

```bash
./start_localist.sh --stop
```

localist-mcp is a standalone MCP (Model Context Protocol) server exposed over SSE transport. It is required for `web_search`, `fetch_url`, and the `file_op` tools (`read_file`/`write_file`/`append_file`) — the main backend's `MCPToolDispatcher` calls it over an MCP session. The standalone Fetcher microservice (port 8002) has been retired; its extraction logic now lives on localist-mcp as the `fetch_url` tool. See `backend/mcp_server/main.py`.

---

## Configuration

Copy `backend/.env.example` to `backend/.env` and set values as needed. The only required variable for full functionality is `LANGSEARCH_API_KEY`; all others have working defaults.

| Variable | Default | Description |
|---|---|---|
| `LOCALIST_RUNTIME_BACKEND` | `foundry` | `foundry`, `omlx`, or `ollama` — selects the active `BaseRuntimeClient` implementation |
| `LOCALIST_CHAT_MODEL` | *(none)* | Chat model ID, interpreted by whichever backend is active (e.g. `gemma-4-e4b-it-4bit` for omlx, `gemma4:e4b-mlx` or a `-cloud` model for ollama) |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING` |
| `LOCALIST_OMLX_URL` | `http://localhost:8000` | oMLX server URL (omlx backend only) |
| `LOCALIST_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL — used for both local and Ollama Cloud models, which proxy through the same local daemon (ollama backend only) |
| `LANGSEARCH_API_KEY` | *(none)* | LangSearch API key, read by localist-mcp. Required for `web_search`; without it the tool returns a clean failure (no hallucination fallback) and `ConversationalAgent` falls back to the corpus. |
| `LOCALIST_MCP_URL` | `http://localhost:8003` | localist-mcp server URL |
| `LOCALIST_MCP_PROJECT_ROOT` | `backend/` | Base path for localist-mcp's `file_op` tools; writes/reads are sandboxed to a `generated_files/` subdirectory of this path |
| `LOCALIST_GENERATED_DIR` | `backend/generated_files` | Path the main backend lists under the Files tab's "Generated Files" pane (`GET /files/generated`) — should match where `file_op` actually writes |
| `LOCALIST_EMBEDDING_ENGINE_ENABLED` | `true` | Enable local embedding engine for RAG |
| `LOCALIST_WIKI_DIR` | `./wiki` | Path to wiki document directory |
| `LOCALIST_RAW_DIR` | `./raw` | Path to raw documents directory |
| `LOCALIST_EPISODIC_WRITE_APPROVAL` | `false` | When `true`, `model_extracted` (implicit) episodes are staged with `status='pending'` instead of going live immediately, and must be approved or rejected via `POST /memory/episodes/{id}/approve`/`.../reject` before they're eligible for retrieval. Explicit ("remember that...") episodes are never gated. |

---

## How It Works

### Routing

Every query passes through `Planner`, a deterministic rule engine with priority levels evaluated in order. No inference is used for routing decisions.

| Priority | Trigger | Route |
|---|---|---|
| P1 | `raw_path` in context, or ingest keyword | `WikiAgent` |
| P1b | Diff keyword (`update page`, `revise page`, etc.) + resolvable target page | `WikiAgent` with `diff_target` set (no raw file) — falls back to `ConversationalAgent` for clarification if the target page can't be resolved |
| P2 | Memory keyword (`remember`, `forget`, `prefer`) | `ConversationalAgent` + write episode |
| P3 | Tool signal: web search keyword, URL present, file keyword | `ConversationalAgent` + tool call |
| P3b | Factual question keyword + corpus score below threshold | `ConversationalAgent` + `web_search` |
| P4 | Explicit vault keyword (`check the wiki`, `vault`, etc.) **or** corpus score ≥ 0.55 | `ConversationalAgent` + RAG fetch |
| P5 | Episodic relevance keyword (`my preference`, `last time`, etc.) | `ConversationalAgent` + episodic fetch |
| P6 | Fallback | `ConversationalAgent`, direct answer |

### Tools

All tools are served over MCP (SSE transport) by the **localist-mcp** server (port 8003, `backend/mcp_server/`). `MCPToolDispatcher` opens one MCP session per dispatch call and reuses it across every tool invocation made within that call.

**Web search** — fires automatically at P3b when the query looks factual and the local corpus has no strong hit. Calls the `web_search` MCP tool (LangSearch API under the hood), returns the top three results with titles, URLs, and truncated body text. When `LANGSEARCH_API_KEY` is unset, the tool returns a clean failure — there is no inference-hallucination fallback — and `ConversationalAgent`'s existing corpus fallback grounds the answer instead.

**Page fetch** — triggered when a URL appears in the message or the user says "fetch this link", "summarize this URL", etc. Calls the `fetch_url` MCP tool, which downloads the page and uses `readability-lxml` to extract clean article text. Returns title, source URL, word count, and body text. This replaces the retired standalone Fetcher microservice (formerly port 8002).

**File operations** — sandboxed `read_file`/`write_file`/`append_file` MCP tools, rooted at `LOCALIST_MCP_PROJECT_ROOT/generated_files/`. Triggered by explicit file-operation phrasing (`"write a file"`, `"save it as X.md"`, `"append to X.md"`, etc.). `write_file` refuses empty/whitespace-only content rather than silently creating a 0-byte file, and versions on filename collision (`notes.md` → `notes_2.md`, up to 10 versions) instead of overwriting. When the content to save doesn't yet exist in the instruction — e.g. *"write a haiku about the sea and save it as haiku.md"* — the Planner defers the file write until after the answer is generated, then dispatches it and appends a deterministic `*(Saved to haiku.md)*` / `*(Could not save — {reason})*` confirmation line to the response. Files written this way, and any other `file_op` output, are listed in the Localist UI's Files tab under "Generated Files."

**Wiki ingestion** — processes a raw document into structured wiki pages via `WikiAgent`. Triggered by `raw_path` in request context or ingestion keywords.

**Wiki diff updates** — proposes a targeted diff against an *existing* wiki page with no raw file involved, via `WikiAgent`'s `diff_target` path (P1b, e.g. *"update page localist-software-stack to reflect X"*). The proposed diff renders in the Localist UI as a reviewable block with Apply/Discard actions — nothing is written until the user explicitly clicks Apply, which calls `POST /wiki/apply-diff`. The apply step re-matches the diff against the page's current on-disk content (not by line number — by content, so a stale or hand-edited page fails the apply cleanly with a 409 instead of corrupting anything) and, on success, reindexes the page and rebuilds the wiki link graph. See `docs/architecture/17-wiki-agent-diff-target.md`.

### Memory

Localist maintains two memory stores in SQLite:

**Episodic memory** stores typed facts extracted from conversation — preferences, corrections, decisions, workflows, naming conventions, project facts, and task completions. Each episode has a subject, content, type, confidence score, and status (`pending` → `active` → `superseded`/`retracted`; episodes are never hard-deleted). Supersession (updating rather than duplicating) is handled by matching on subject and type, and only fires for `active` writes — a staged (`pending`) write never supersedes, or is superseded by, anything until it's approved.

Extraction runs on two paths:
- **Explicit**: the user says "remember that" or "my preference is" — the episode is stored immediately at confidence 1.0, `status='active'`, subject derived from the normalized content string. Explicit writes always bypass the write-approval gate below — that's already direct user consent.
- **Implicit**: after every response, the controller checks whether the exchange contains memorable content and extracts an episode at confidence 0.6–0.9 if so.

**Retrieval** uses three modes on `EpisodicMemoryReader`: `by_subject` (exact match), `by_recency` (type-filtered session priming — scoped to preference/correction/decision/workflow — cached per `project_context` on `ControllerAgent` and invalidated on the next write), and `by_similarity`/`best_match` (real cosine similarity against embeddings stored on every `insert()`, falling back to keyword overlap for un-embedded rows). `by_similarity` runs across every episode type, which is what makes `project_fact`/`task_completion`/`naming_convention` episodes reachable at all — `by_recency` alone never surfaces them.

**Retraction** ("forget that...") resolves the target episode via `best_match()` — semantic similarity against the retraction phrase, not an exact string match against the stored `subject` — since a model-extracted retraction query rarely matches the original subject's wording verbatim. Falls back to an exact `(subject, episode_type)` match across all valid types when no embedding is available.

**Content safety**: every `insert()` scans `subject` and `content` through `content_safety.py` — a lightweight, non-ML pattern matcher for prompt-injection phrases, credential/secret-looking text, and invisible Unicode — before writing. Flagged content is rejected and logged rather than stored, since episode content is replayed back into every future prompt.

**Write-approval gate** (`LOCALIST_EPISODIC_WRITE_APPROVAL`, default off): when enabled, implicit (`model_extracted`) episodes are staged with `status='pending'` instead of going live immediately. `POST /memory/episodes/{id}/approve` or `.../reject` resolves them; only `active` episodes are retrievable or appear in `MEMORY.md`. The Localist UI Memory tab surfaces a "Pending (N)" filter with inline Approve/Reject actions and a sidebar badge, so pending items are visible without opening the tab.

A human-readable snapshot of active episodic memory auto-regenerates at `wiki/MEMORY.md` on every write — one scrollable file, grouped by date, showing type, confidence, project context, source, and content. `backfill_episode_embeddings.py` is a one-off script for embedding pre-existing episode rows and rebuilding `MEMORY.md` for data written before embedding support existed.

**Corpus (RAG)** stores vector embeddings of wiki pages and ingested documents. `ConversationalAgent` queries the corpus when `fetch_rag=True` in the routing plan and injects matching passages into the prompt. Embeddings use `mlx-community/embeddinggemma-300m-4bit` (768-dimensional, local).

**User profile** stores durable facts about the user in `wiki/users/michael.md` — identity, active projects, preferences, working patterns, and committed decisions. The profile is loaded once per session at first request, embedded line-by-line using `mlx-community/embeddinggemma-300m-4bit`, and cached for the process lifetime. On turns where the corpus or episodic memory is queried, the top relevant profile lines are scored against the current instruction (cosine similarity, threshold 0.45) and injected into the prompt as a `[USER PROFILE]` block. Only lines that score above the threshold are included — the full profile is never injected wholesale.

### Prompt Layout

The prompt builder assembles a fixed 7-slot layout optimized for KV-cache efficiency:

1. **Identity** — static system role (never changes between turns)
2. **Persona** — loaded from `wiki/lora-persona.md`, cached per session
3. **Episodic memory + User profile** — retrieved episodes and
   relevance-scored user profile facts (two independent sub-budgets:
   150 tokens episodic, 100 tokens profile)
4. **RAG / context** — retrieved corpus passages
5. **Tool results** — output from `web_search`, `url_fetch`, `file_op` (a
   failed tool call renders in its own budget-isolated block rather than
   being silently dropped, so the model can see and hedge on a failure
   instead of fabricating success)
6. **Working memory** — recent conversation turns
7. **Instruction** — the current user message

Slots 1–2 are static across turns, maximizing KV-cache reuse. Measured cache efficiency on turn 2: 79.7%. Worst-case prompt (all slots populated): approximately 2,450 tokens against an 8,000-token context window.

---

## Project Structure

```
localist/
├── backend/
│   ├── main.py                      # FastAPI app — HTTP entry point, port 8001
│   ├── controller_agent.py          # Task orchestration, agent dispatch, episodic extraction trigger
│   ├── planner.py                   # Deterministic routing rule engine (P1–P6)
│   ├── conversational_agent.py      # Primary agent: prompt assembly, RAG, tool result injection
│   ├── wiki_agent.py                # Document ingestion agent — raw file → structured wiki pages
│   ├── prompt_builder.py            # 7-slot KV-cache-optimized prompt assembler
│   ├── mcp_tool_dispatcher.py       # Opens an MCP/SSE session to localist-mcp and dispatches web_search, url_fetch, file_op
│   ├── memory_manager.py            # SQLite-backed episodic + RAG memory interface
│   ├── episodic_extractor.py        # Explicit and implicit episode extraction pipeline
│   ├── content_safety.py            # Pattern-based scanner (prompt injection / credentials / invisible Unicode) run before every episode write
│   ├── backfill_episode_embeddings.py  # One-off: embeds pre-existing episode rows, rebuilds wiki/MEMORY.md
│   ├── embedding_engine.py          # Local mlx embedding engine (768-dim)
│   ├── omlx_runtime_client.py       # oMLX HTTP transport (OpenAI-compatible)
│   ├── ollama_runtime_client.py     # Ollama HTTP transport (native /api/chat, NDJSON streaming); also serves Ollama Cloud models
│   ├── runtime_factory.py           # Constructs the active runtime from LOCALIST_RUNTIME_BACKEND
│   ├── base_runtime_client.py       # BaseRuntimeClient protocol definition
│   ├── mcp_server/
│   │   ├── main.py                  # localist-mcp FastAPI/FastMCP app — port 8003, SSE transport
│   │   ├── file_ops.py              # read_file/write_file/append_file, sandboxed to LOCALIST_MCP_PROJECT_ROOT
│   │   ├── url_fetch.py             # fetch_url MCP tool — readability-lxml extraction (replaces retired Fetcher microservice)
│   │   └── web_search.py            # web_search MCP tool — LangSearch integration
│   ├── wiki/
│   │   ├── lora-persona.md          # LORA persona — loaded into Slot 1b of every prompt
│   │   ├── users/
│   │   │   └── michael.md           # User profile — line-level embeddings, Slot 3b injection
│   │   ├── MEMORY.md                # Auto-regenerated human-readable snapshot of active episodic memory
│   │   └── *.md                     # Indexed wiki pages
│   ├── tests/
│   │   ├── test_planner_phase3.py   # Planner routing unit tests
│   │   ├── test_controller_phase4.py # ControllerAgent integration tests
│   │   ├── test_memory_phase1.py    # Episodic memory substrate: writer/reader lifecycle, similarity retrieval, write-approval gate, content safety
│   │   ├── test_episodic_phase5.py  # Episodic memory extraction tests
│   │   ├── test_content_safety.py   # content_safety.py pattern coverage
│   │   ├── test_main_memory_episodes.py  # GET/POST /memory/episodes REST endpoint tests
│   │   ├── test_tool_dispatcher_phase6.py  # MCPToolDispatcher unit tests (web_search, url_fetch, file_op)
│   │   ├── test_integration_phase7.py      # Full pipeline integration tests
│   │   ├── test_mcp_server.py       # localist-mcp server tests — file_ops, fetch_url, web_search tools
│   │   └── test_mcp_tool_dispatcher.py     # MCPToolDispatcher session/dispatch tests
│   ├── localist_memory.db           # SQLite database (episodic + embeddings)
│   ├── requirements.txt
│   └── .env                         # Local configuration (not committed)
└── localist-ui/                     # Localist UI
```

---

## Development

Run the full test suite from `backend/` with the virtual environment active:

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v
```

Tests are organized by phase and cover each layer independently:
- **Phase 1** (`test_memory_phase1.py`) — episodic memory substrate: writer/reader lifecycle, all three retrieval modes, cosine-similarity scoring, write-approval gate (`pending`/`approve`/`reject`), content safety scanning
- **Phase 3** (`test_planner_phase3.py`) — routing rule engine, all priority levels
- **Phase 4** (`test_controller_phase4.py`) — controller dispatch, RAG injection, prompt assembly, `by_recency()` cache invalidation
- **Phase 5** (`test_episodic_phase5.py`) — episodic extraction, supersession, confidence scoring, semantic-match retraction
- **Phase 6** (`test_tool_dispatcher_phase6.py`) — MCPToolDispatcher: LangSearch integration, file ops, url_fetch
- **Phase 7** (`test_integration_phase7.py`) — full pipeline from instruction to response
- Content safety (`test_content_safety.py`) — `content_safety.py` pattern coverage (prompt injection, credentials, invisible Unicode)
- Memory REST API (`test_main_memory_episodes.py`) — `GET`/`POST /memory/episodes` endpoints, including approve/reject
- MCP migration (`test_mcp_server.py`, `test_mcp_tool_dispatcher.py`) — localist-mcp server tools and dispatcher session handling, covering the retired Fetcher microservice and legacy ToolDispatcher's replacement

All tests use mocks for inference and SQLite; no oMLX server or live API keys are required.

---

## Roadmap

- **Localist CLI** — ✅ `./start_localist.sh` launches all three services
  (backend, localist-mcp, frontend); `--stop` kills them cleanly
- **MCP migration** — ✅ tools (`web_search`, `fetch_url`, `file_op`) served
  over MCP/SSE by the standalone localist-mcp server (port 8003); the
  legacy `ToolDispatcher` and standalone Fetcher microservice (port 8002)
  are both retired
- **Identity continuity** — ✅ LORA correctly identifies itself; identity
  questions route via P3 semantic gate or P6 direct answer backed by
  `lora-persona.md`
- **User profile** — ✅ `wiki/users/michael.md`; line-level embedding
  and cosine-scored injection into Slot 3b
- **Generate-then-save file operations** — ✅ a `file_op` instruction whose
  content isn't literally present (e.g. "write a haiku and save it as
  X.md") defers the write until the answer is generated, then dispatches
  it and appends a deterministic saved/failed confirmation line;
  `write_file` refuses empty content and versions on collision instead of
  overwriting
- ** Graph retrieval layer — ✅ concept relationship reasoning via SQLite 
  node/edge tables (schema v6) and hybrid graph + RAG retrieval; 
  build-trigger mechanism and ambient (implicit) graph usage still open
- **Ollama runtime backend** — ✅ `OllamaRuntimeClient`, interchangeable
  with oMLX/Foundry via `LOCALIST_RUNTIME_BACKEND=ollama`; supports both
  local Ollama models and Ollama Cloud models (proxied through the same
  local daemon at `localhost:11434`) with no code change between the two;
  embeddings stay 100% local via `EmbeddingEngine` regardless of which
  chat backend is selected
- **Wiki diff updates** — ✅ `WikiAgent.diff_target` path (P1b routing) proposes
  a targeted diff against an existing wiki page with no raw file required;
  review-then-apply UI (`POST /wiki/apply-diff`) lets the user approve before
  anything writes; `apply_unified_diff` matches hunks by content rather than
  model-authored line numbers, live-verified end to end including a caught
  and repaired on-disk corruption during testing. Open: `wiki/` is gitignored
  with no rollback mechanism for any write (diff or ingest) — candidate fixes
  not yet scoped; a bullet/diff-marker collision on unchanged context lines
  fails safely (409) but isn't generalized-away yet
- **Episodic memory hardening** — ✅ `by_similarity()`/`best_match()` now
  do real cosine retrieval against stored embeddings instead of a
  keyword-only path that was never actually wired into live retrieval;
  `by_recency()` results cached per `project_context` on `ControllerAgent`
  and invalidated on the next write; retraction resolves the target
  episode by semantic match (`best_match()`, precise `retract_by_id()`)
  instead of an exact subject string, with a keyword-only fallback; every
  `insert()` is scanned by `content_safety.py` for prompt-injection,
  credential-looking, and invisible-Unicode content before being stored;
  `wiki/MEMORY.md` auto-regenerates a human-readable snapshot on every
  write; `LOCALIST_EPISODIC_WRITE_APPROVAL` stages implicit
  (`model_extracted`) writes behind a pending → approve/reject step
  (`POST /memory/episodes/{id}/approve|reject`), live-verified end to end
- **Localist UI redesign** — Memory tab now supports a "Pending" filter
  with inline Approve/Reject actions and a sidebar notification badge
  (see Episodic memory hardening above); broader rework for general
  episode browsing and tool result display still open
- **macOS app packaging** — bundle as a native `.app` via PyInstaller +
  Tauri so Localist Framework can run without a terminal
