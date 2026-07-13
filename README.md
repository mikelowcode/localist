# Localist Framework

A local-first, agentic general assistant running entirely on macOS Apple Silicon. Persistent memory across sessions, live web search/fetch, indexed document search, and a deterministic priority-based router — no inference spent deciding how to route a query.

Inference-engine-agnostic: ships with oMLX, Ollama (including Ollama Cloud models), and Azure AI Foundry, swappable via one config variable. Embeddings always run locally regardless of chat backend.

---

## Architecture

SvelteKit frontend → FastAPI backend (port 8001). The backend's `ControllerAgent` runs each task through `Planner` (a priority-ordered rule engine) and dispatches to `ConversationalAgent` (answers/tools) or `WikiAgent` (document ingestion). Tool calls go through `MCPToolDispatcher` to **localist-mcp** (port 8003), a standalone MCP server exposing `web_search`, `fetch_url`, and `file_op` tools. All inference runs through a `BaseRuntimeClient` implementation selected via `LOCALIST_RUNTIME_BACKEND`. Episodic memory and embeddings live in SQLite (WAL mode), surviving restarts.

```
Localist UI ──HTTP──► FastAPI :8001
                          │
                    ControllerAgent → Planner → RoutingPlan
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                    ▼
 ConversationalAgent                    WikiAgent
   │        │                         (ingestion/diff)
   │        └── MCPToolDispatcher ──► localist-mcp :8003
   │                                    ├─ web_search
   │                                    ├─ fetch_url
   │                                    └─ file_op
   └── MemoryManager (SQLite episodic + RAG)
```

---

## Prerequisites

- macOS Apple Silicon, Python 3.13, Node.js
- One runtime backend: oMLX (chat model on :8000), [Ollama](https://ollama.com) (local or Ollama Cloud, :11434), or Azure AI Foundry

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
./start_localist.sh         # starts backend, localist-mcp, and frontend
./start_localist.sh --stop  # stops all three
```

## Configuration

Copy `backend/.env.example` to `backend/.env`. Only `LANGSEARCH_API_KEY` is required for full functionality; everything else has a working default.

| Variable | Default | Description |
|---|---|---|
| `LOCALIST_RUNTIME_BACKEND` | `foundry` | `foundry`, `omlx`, or `ollama` |
| `LOCALIST_CHAT_MODEL` | *(none)* | Model ID for the active backend |
| `LANGSEARCH_API_KEY` | *(none)* | Enables `web_search`; without it, falls back to corpus |
| `LOCALIST_MCP_URL` | `http://localhost:8003` | localist-mcp server URL |
| `LOCALIST_EPISODIC_WRITE_APPROVAL` | `false` | Gate implicit memory writes behind approve/reject |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING` |

See `backend/.env.example` for the full list (embedding engine, wiki/raw directories, MCP project root, etc.).

---

## How It Works

**Routing** — `Planner` evaluates priority rules (P1–P6) in order, no inference required: raw file/ingest → `WikiAgent`; diff keywords → targeted wiki diff; memory keywords → episode write; tool signals (URL, file, search) → tool dispatch; factual gaps → web search; corpus match → RAG; fallback → direct answer.

**Tools** — served over MCP/SSE by localist-mcp: `web_search` (LangSearch), `fetch_url` (readability-lxml extraction), and sandboxed `file_op` (`read_file`/`write_file`/`append_file`, versioned on collision). File writes for not-yet-generated content are deferred until after the answer, then confirmed inline.

**Memory** — two SQLite-backed stores. Episodic memory captures typed facts (preferences, decisions, corrections, etc.) with confidence scores and a `pending → active → superseded/retracted` lifecycle; retrieval by subject, recency, or cosine similarity; retraction via semantic match; every write scanned for prompt-injection/credential content before storing. A human-readable snapshot regenerates at `wiki/MEMORY.md`. The corpus (RAG) stores embeddings of wiki pages and documents. A user profile (`wiki/users/michael.md`) is embedded line-by-line and injected only where relevant (cosine ≥ 0.45).

**Prompt layout** — fixed 7-slot structure (identity, persona, episodic+profile, RAG, tool results, working memory, instruction) optimized for KV-cache reuse.

---

## Project Structure

```
localist/
├── backend/
│   ├── main.py                  # FastAPI entry, port 8001
│   ├── controller_agent.py      # Task orchestration
│   ├── planner.py               # Routing rules (P1–P6)
│   ├── conversational_agent.py  # Prompt assembly, RAG, tools
│   ├── wiki_agent.py            # Document ingestion / diff
│   ├── prompt_builder.py        # 7-slot prompt assembler
│   ├── mcp_tool_dispatcher.py   # MCP/SSE client to localist-mcp
│   ├── memory_manager.py        # Episodic + RAG memory
│   ├── episodic_extractor.py    # Episode extraction
│   ├── content_safety.py        # Pre-write content scanner
│   ├── embedding_engine.py      # Local embedding engine
│   ├── runtime_factory.py       # Backend selection
│   ├── mcp_server/              # localist-mcp — port 8003
│   ├── wiki/                    # Persona, user profile, indexed pages, MEMORY.md
│   ├── tests/                   # Unit + integration tests by phase
│   └── requirements.txt
└── localist-ui/                 # Frontend
```

## Development

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/ -v
```

Tests are organized by phase (memory substrate, routing, controller dispatch, extraction, tool dispatcher, integration, content safety, REST API) and mock inference/SQLite — no live server or API keys required.

---

## Roadmap

**Done:** Localist CLI launcher, MCP migration (tools off legacy dispatcher/Fetcher), identity continuity, user profile injection, generate-then-save file ops, graph retrieval layer (SQLite schema v6), Ollama runtime backend (incl. Cloud), wiki diff updates with review/apply UI, episodic memory hardening (real cosine retrieval, write-approval gate, semantic retraction).

**Open:** generalize the bullet/diff-marker collision edge case; rollback mechanism for wiki writes (currently gitignored, no undo); broader Localist UI rework for episode browsing and tool result display; macOS `.app` packaging via PyInstaller + Tauri.
