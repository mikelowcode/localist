# Localist Framework

Localist Framework is a local-first, agentic general assistant running entirely on macOS Apple Silicon. The system maintains persistent memory across sessions, fetches and reads live web content, searches indexed documents, and routes every query through a deterministic priority engine before a single token of inference is spent.

Localist is inference-engine-agnostic. It ships with oMLX support out of the box and is designed to support MLX-LM, Ollama, LM Studio, and other local runtimes.

---

## Architecture

The frontend (SvelteKit) sends requests over HTTP to the main backend — a FastAPI application on port 8001. The backend's `ControllerAgent` receives each task, runs it through the `Planner` (a priority-ordered rule engine), and dispatches to the appropriate agent: `ConversationalAgent` for answers and `WikiAgent` for document ingestion. When a query requires live web content, `ToolDispatcher` calls the LangSearch API or posts to the fetcher microservice on port 8002, which fetches and extracts clean article text. All inference goes through `OMLXRuntimeClient`, which speaks the OpenAI-compatible HTTP API exposed by oMLX. Episodic memory and vector embeddings are stored in a SQLite database with WAL mode enabled and survive server restarts.

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
     │         └── ToolDispatcher
     │               ├── LangSearch API  (web_search)
     │               ├── Fetcher — port 8002  (url_fetch)
     │               └── FileSystem  (file_op)
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

Use the **Localist CLI** launcher to start both services with a single command:

```bash
./start_localist.sh
```

This starts the Localist backend on port 8001 and the fetcher microservice on port 8002, tails both logs to the terminal with `[backend]` and `[fetcher]` prefixes, and shuts both down cleanly on Ctrl+C.

To stop both services from a separate terminal:

```bash
./start_localist.sh --stop
```

The fetcher is a standalone FastAPI service. It is only required if you intend to use the `url_fetch` tool (drop a URL into chat). The main backend degrades gracefully if the fetcher is unreachable.

---

## Configuration

Copy `backend/.env.example` to `backend/.env` and set values as needed. The only required variable for full functionality is `LANGSEARCH_API_KEY`; all others have working defaults.

| Variable | Default | Description |
|---|---|---|
| `LOCALIST_RUNTIME_BACKEND` | `omlx` | `omlx` or `foundry` |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, or `WARNING` |
| `LOCALIST_OMLX_URL` | `http://localhost:8000` | oMLX server URL |
| `LANGSEARCH_API_KEY` | *(none)* | LangSearch API key. Web search is disabled and falls back to an inference stub when absent. |
| `LOCALIST_FETCHER_URL` | `http://localhost:8002` | Fetcher microservice URL |
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

**Web search** — fires automatically at P3b when the query looks factual and the local corpus has no strong hit. Calls LangSearch API, returns the top three results with titles, URLs, and truncated body text. Falls back to an inference stub when no API key is configured.

**Page fetch** — triggered when a URL appears in the message or the user says "fetch this link", "summarize this URL", etc. Posts to the fetcher microservice, which downloads the page and uses `readability-lxml` to extract clean article text. Returns title, source URL, word count, and body text.

**File operations** — sandboxed read, write, and append on local files. Triggered by explicit file-operation phrasing.

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
5. **Tool results** — output from `web_search`, `url_fetch`, `file_op`
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
│   ├── tool_dispatcher.py           # Executes web_search, url_fetch, file_op tool calls
│   ├── memory_manager.py            # SQLite-backed episodic + RAG memory interface
│   ├── episodic_extractor.py        # Explicit and implicit episode extraction pipeline
│   ├── embedding_engine.py          # Local mlx embedding engine (768-dim)
│   ├── omlx_runtime_client.py       # oMLX HTTP transport (OpenAI-compatible)
│   ├── runtime_factory.py           # Constructs the active runtime from LOCALIST_RUNTIME_BACKEND
│   ├── base_runtime_client.py       # BaseRuntimeClient protocol definition
│   ├── fetcher/
│   │   ├── main.py                  # Fetcher FastAPI app — port 8002
│   │   ├── client.py                # Async HTTP client (httpx)
│   │   ├── extractor.py             # readability-lxml extraction pipeline
│   │   └── models.py                # Pydantic v2 request/response models
│   ├── wiki/
│   │   ├── lora-persona.md          # LORA persona — loaded into Slot 1b of every prompt
│   │   ├── users/
│   │   │   └── michael.md           # User profile — line-level embeddings, Slot 3b injection
│   │   └── *.md                     # Indexed wiki pages
│   ├── tests/
│   │   ├── test_planner_phase3.py   # Planner routing unit tests
│   │   ├── test_controller_phase4.py # ControllerAgent integration tests
│   │   ├── test_episodic_phase5.py  # Episodic memory extraction tests
│   │   ├── test_tool_dispatcher_phase6.py  # ToolDispatcher unit tests
│   │   └── test_integration_phase7.py      # Full pipeline integration tests
│   ├── lora_memory.db               # SQLite database (episodic + embeddings)
│   ├── requirements.txt
│   └── .env                         # Local configuration (not committed)
└── lora-ui/                         # Localist UI
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
- **Phase 6** (`test_tool_dispatcher_phase6.py`) — LangSearch integration, file ops, url_fetch
- **Phase 7** (`test_integration_phase7.py`) — full pipeline from instruction to response

All tests use mocks for inference and SQLite; no oMLX server or live API keys are required.

---

## Roadmap

- **Localist CLI** — ✅ `./start_localist.sh` launches both services;
  `--stop` kills them cleanly
- **Identity continuity** — ✅ LORA correctly identifies itself; identity
  questions route via P3 semantic gate or P6 direct answer backed by
  `lora-persona.md`
- **User profile** — ✅ `wiki/users/michael.md`; line-level embedding
  and cosine-scored injection into Slot 3b
- **Graph retrieval layer** — planned; concept relationship reasoning
  via SQLite node/edge tables and hybrid graph + RAG retrieval
- **Localist UI redesign** — functional but minimal; planned rework for
  memory inspection, episode browsing, and tool result display
- **macOS app packaging** — bundle as a native `.app` via PyInstaller +
  Tauri so Localist Framework can run without a terminal
