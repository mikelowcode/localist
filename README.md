# Localist Framework

Localist Framework is a local-first, agentic general assistant running entirely on macOS Apple Silicon. The system maintains persistent memory across sessions, fetches and reads live web content, searches indexed documents, and routes every query through a deterministic priority engine before a single token of inference is spent.

Localist is inference-engine-agnostic. It ships with oMLX support out of the box and is designed to support MLX-LM, Ollama, LM Studio, and other local runtimes.

---

## Architecture

The frontend (SvelteKit) sends requests over HTTP to the main backend — a FastAPI application on port 8001. The backend's `ControllerAgent` receives each task, runs it through the `Planner` (a priority-ordered rule engine), and dispatches to the appropriate agent: `ConversationalAgent` for answers and `WikiAgent` for document ingestion. When a query requires a tool, `MCPToolDispatcher` opens an MCP session (SSE transport) to **localist-mcp**, a standalone MCP server on port 8003 that exposes `web_search` (LangSearch), `fetch_url` (HTML extraction via `readability-lxml`), and the `file_op` tools (`read_file`/`write_file`/`append_file`). The legacy `ToolDispatcher` and the standalone Fetcher microservice (port 8002) have both been retired — their functionality now lives on localist-mcp. All inference goes through `OMLXRuntimeClient`, which speaks the OpenAI-compatible HTTP API exposed by oMLX. Episodic memory and vector embeddings are stored in a SQLite database with WAL mode enabled and survive server restarts.

```
Localist UI  (localist-ui/)
     │  HTTP
     ▼
FastAPI — port 8001  (backend/main.py)
     │
     ▼
ControllerAgent  →  Planner  →  RoutingPlan
     │
     ├──► ConversationalAgent  →  PromptBuilder  →  OMLXRuntimeClient
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
     └──► WikiAgent  →  OMLXRuntimeClient
```

---

## Prerequisites

- macOS Apple Silicon
- Python 3.13
- oMLX running `gemma-4-e4b-it-4bit` on port 8000
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
| `LOCALIST_RUNTIME_BACKEND` | `omlx` | `omlx` or `foundry` |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING` |
| `LOCALIST_OMLX_URL` | `http://localhost:8000` | oMLX server URL |
| `LANGSEARCH_API_KEY` | *(none)* | LangSearch API key, read by localist-mcp. Required for `web_search`; without it the tool returns a clean failure (no hallucination fallback) and `ConversationalAgent` falls back to the corpus. |
| `LOCALIST_MCP_URL` | `http://localhost:8003` | localist-mcp server URL |
| `LOCALIST_MCP_PROJECT_ROOT` | `backend/` | Base path for localist-mcp's `file_op` tools; writes/reads are sandboxed to a `generated_files/` subdirectory of this path |
| `LOCALIST_GENERATED_DIR` | `backend/generated_files` | Path the main backend lists under the Files tab's "Generated Files" pane (`GET /files/generated`) — should match where `file_op` actually writes |
| `LOCALIST_EMBEDDING_ENGINE_ENABLED` | `true` | Enable local embedding engine for RAG |
| `LOCALIST_WIKI_DIR` | `./wiki` | Path to wiki document directory |
| `LOCALIST_RAW_DIR` | `./raw` | Path to raw documents directory |

---

## How It Works

### Routing

Every query passes through `Planner`, a deterministic rule engine with seven priority levels evaluated in order. No inference is used for routing decisions.

| Priority | Trigger | Route |
|---|---|---|
| P1 | `raw_path` in context, or ingest keyword | `WikiAgent` |
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

### Memory

Localist maintains two memory stores in SQLite:

**Episodic memory** stores typed facts extracted from conversation — preferences, corrections, decisions, workflows, naming conventions, project facts, and task completions. Each episode has a subject, content, type, confidence score, and status. Supersession (updating rather than duplicating) is handled by matching on subject and type. Episodes are never hard-deleted; status transitions manage lifecycle.

Extraction runs on two paths:
- **Explicit**: the user says "remember that" or "my preference is" — the episode is stored immediately at confidence 1.0, subject derived from the normalized content string.
- **Implicit**: after every response, the controller checks whether the exchange contains memorable content and extracts an episode at confidence 0.6–0.9 if so.

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
│   ├── embedding_engine.py          # Local mlx embedding engine (768-dim)
│   ├── omlx_runtime_client.py       # oMLX HTTP transport (OpenAI-compatible)
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
│   │   └── *.md                     # Indexed wiki pages
│   ├── tests/
│   │   ├── test_planner_phase3.py   # Planner routing unit tests
│   │   ├── test_controller_phase4.py # ControllerAgent integration tests
│   │   ├── test_episodic_phase5.py  # Episodic memory extraction tests
│   │   ├── test_tool_dispatcher_phase6.py  # MCPToolDispatcher unit tests (web_search, url_fetch, file_op)
│   │   ├── test_integration_phase7.py      # Full pipeline integration tests
│   │   ├── test_mcp_server.py       # localist-mcp server tests — file_ops, fetch_url, web_search tools
│   │   └── test_mcp_tool_dispatcher.py     # MCPToolDispatcher session/dispatch tests
│   ├── lora_memory.db               # SQLite database (episodic + embeddings)
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
- **Phase 3** (`test_planner_phase3.py`) — routing rule engine, all priority levels
- **Phase 4** (`test_controller_phase4.py`) — controller dispatch, RAG injection, prompt assembly
- **Phase 5** (`test_episodic_phase5.py`) — episodic extraction, supersession, confidence scoring
- **Phase 6** (`test_tool_dispatcher_phase6.py`) — MCPToolDispatcher: LangSearch integration, file ops, url_fetch
- **Phase 7** (`test_integration_phase7.py`) — full pipeline from instruction to response
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
- **Graph retrieval layer** — planned; concept relationship reasoning
  via SQLite node/edge tables and hybrid graph + RAG retrieval
- **Localist UI redesign** — functional but minimal; planned rework for
  memory inspection, episode browsing, and tool result display
- **macOS app packaging** — bundle as a native `.app` via PyInstaller +
  Tauri so Localist Framework can run without a terminal
