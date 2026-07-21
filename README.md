# Localist Framework

A local-first, agentic general assistant built primarily for macOS Apple Silicon. Persistent memory across sessions, live web search/fetch, indexed document search, and a deterministic priority-based router — no inference spent deciding how to route a query.

Inference-engine-agnostic: ships with oMLX, Ollama (including Ollama Cloud models), and Azure AI Foundry, swappable via one config variable or live at runtime with no restart. Embeddings always run locally regardless of chat backend — via MLX EmbeddingGemma (Apple Silicon only) or, with the Ollama backend, via any locally-served Ollama embedding model (e.g. `nomic-embed-text`), which also makes the framework usable on non-Apple-Silicon hardware.

---

## Architecture

SvelteKit frontend → FastAPI backend (port 8001). The backend's `ControllerAgent` runs each task through `Planner` (a priority-ordered rule engine, plus an explicit `/chart`/`/research` slash-command bypass ahead of it) and dispatches to `ConversationalAgent` (answers/tools) or `WikiAgent` (document ingestion). Tool calls go through `MCPToolDispatcher` to **localist-mcp** (port 8003), a standalone MCP server exposing `web_search`, `fetch_url`, `file_op`, and `generate_chart` tools. All inference runs through a `BaseRuntimeClient` implementation selected via `LOCALIST_RUNTIME_BACKEND` — swappable live, without a restart, via the Settings UI or `POST /settings/runtime-backend`. Episodic memory and embeddings live in SQLite (WAL mode), surviving restarts.

```
Localist UI ──HTTP──► FastAPI :8001
                          │
              Planner Priority 0 — /chart, /research
                    (explicit tool bypass)
                          │
                    ControllerAgent → Planner → RoutingPlan
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                    ▼
 ConversationalAgent                    WikiAgent
   │        │                         (ingestion/diff)
   │        └── MCPToolDispatcher ──► localist-mcp :8003
   │                                    ├─ web_search (research loop upgrade)
   │                                    ├─ fetch_url
   │                                    ├─ file_op
   │                                    └─ generate_chart
   └── MemoryManager (SQLite episodic + RAG)
```

---

## Prerequisites

- Python 3.13, Node.js
- One runtime backend: oMLX (chat model on :8000, macOS Apple Silicon only), [Ollama](https://ollama.com) (local or Ollama Cloud, :11434, any OS), or Azure AI Foundry
- MLX EmbeddingGemma (the default local embedding model) requires Apple Silicon; on other platforms, set `LOCALIST_EMBEDDING_MODEL` to an Ollama-served embedding model instead (or fall back to keyword-only retrieval)

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First backend startup downloads the local embedding model (`mlx-community/embeddinggemma-300m-4bit`, ~400MB) from the Hugging Face Hub and caches it — a one-time cost that needs internet access. Without it (or on non-Apple-Silicon hardware, where it can't run at all), episodic memory and RAG retrieval still work in keyword-only mode, or via an Ollama-served embedding model instead (see `LOCALIST_EMBEDDING_MODEL` below).

## Running

```bash
./start_localist.sh         # starts backend, localist-mcp, and frontend
./start_localist.sh --stop  # stops all three
```

## Configuration

Copy `backend/.env.example` to `backend/.env`. Only an API key for the active `web_search` provider — `LANGSEARCH_API_KEY` by default, or `BRAVE_API_KEY` if you switch providers — is required for full functionality; everything else has a working default.

| Variable | Default | Description |
|---|---|---|
| `LOCALIST_RUNTIME_BACKEND` | `foundry` | `foundry`, `omlx`, or `ollama` — also swappable live at runtime, see below |
| `LOCALIST_CHAT_MODEL` | *(none)* | Chat model ID override — wins over any per-backend pin below; required for `ollama` if no pin is set either (fails fast at startup if unset) |
| `LOCALIST_CHAT_MODEL_OLLAMA` / `_OMLX` / `_FOUNDRY` | *(none)* | Per-backend chat model pin, used when `LOCALIST_CHAT_MODEL` is unset — lets each backend remember its own model choice independently, including across a live runtime-backend switch |
| `LOCALIST_EMBEDDING_MODEL` | *(none)* | Embedding model ID for the active backend (`foundry`/`ollama`); if set and found, takes precedence over MLX EmbeddingGemma |
| `SEARCH_PROVIDER` | `langsearch` | `web_search` provider: `langsearch` or `brave` |
| `LANGSEARCH_API_KEY` | *(none)* | Required when `SEARCH_PROVIDER=langsearch`; without it, `web_search` fails and falls back to corpus |
| `BRAVE_API_KEY` | *(none)* | Required when `SEARCH_PROVIDER=brave`; without it, `web_search` fails and falls back to corpus |
| `LOCALIST_MCP_URL` | `http://localhost:8003` | localist-mcp server URL |
| `LOCALIST_EPISODIC_WRITE_APPROVAL` | `false` | Gate implicit memory writes behind approve/reject |
| `LOCALIST_RESEARCH_LOOP_ENABLED` | `false` | Upgrade `web_search` to a bounded search/evaluate/fetch/reformulate loop for price/spec-lookup queries — a request can also always force this loop directly with a leading `/research`, regardless of this flag (see Tools below) |
| `LOCALIST_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING` |

See `backend/.env.example` for the full list (embedding engine, wiki/raw directories, MCP project root, etc.).

**Live runtime-backend switching** — the active backend doesn't require a restart to change: the Settings UI (or `POST /settings/runtime-backend` directly) health-checks the target backend, swaps it in, and persists the choice to `.env`, all while the server keeps running. Each backend remembers its own chat-model pin (`LOCALIST_CHAT_MODEL_OLLAMA`/`_OMLX`/`_FOUNDRY` above), so switching back to a backend you'd previously configured doesn't lose that choice.

---

## How It Works

**Routing** — `Planner` evaluates priority rules (P0–P6) in order, no inference required: an explicit `/chart` or `/research` slash command → force that tool directly (P0, ahead of everything else); raw file/ingest → `WikiAgent`; diff keywords, or a pinned wiki page (see Attachments below) with any diff phrasing anywhere in the instruction → targeted wiki diff; memory keywords → episode write; tool signals (URL, file, search, chart) → tool dispatch; factual gaps → web search; corpus match → RAG; fallback → direct answer.

**Slash commands** — `/chart <data>` and `/research <question>` bypass the normal detection paths and force that tool directly, even on input that wouldn't otherwise trigger it (a bare `/chart` with no data still reaches the tool and degrades gracefully; `/research` runs the full search/evaluate/fetch/reformulate loop even when `LOCALIST_RESEARCH_LOOP_ENABLED` is off). An explicit, user-invoked escape hatch — normal (non-slash) instructions are routed exactly as before.

**Tools** — served over MCP/SSE by localist-mcp: `web_search` (LangSearch or Brave), `fetch_url` (readability-lxml extraction), sandboxed `file_op` (`read_file`/`write_file`/`append_file`, versioned on collision), and `generate_chart` (bar/line/pie charts rendered server-side and as an interactive Chart.js widget in the UI). File writes for not-yet-generated content are deferred until after the answer, then confirmed inline. A bounded research loop (search → evaluate → fetch → reformulate, capped at 3 iterations, with a relevance-aware gate that checks the candidate text actually answers the question asked rather than just containing pricing-shaped content) can upgrade `web_search` for price/spec-lookup queries a single search snippet can't resolve — automatically above a semantic-intent threshold when `LOCALIST_RESEARCH_LOOP_ENABLED` is on (off by default), or always on-demand via `/research`.

**Attachments** — the paperclip in the chat UI uploads a local file into an ephemeral, session-scoped cache injected into every subsequent prompt; a second, bookmark-icon control pins an *existing wiki page* into the same cache instead, so asking LORA to propose a diff against a specific page hands it the real, current file content rather than the model's own (possibly stale) memory of it. Either kind bypasses Planner routing and wiki indexing entirely for as long as it's attached, and clears on backend restart.

**Memory** — two SQLite-backed stores. Episodic memory captures typed facts (preferences, decisions, corrections, etc.) with confidence scores and a `pending → active → superseded/retracted` lifecycle; retrieval by subject, recency, or cosine similarity; retraction via semantic match; every write scanned for prompt-injection/credential content before storing. A human-readable snapshot regenerates at `wiki/MEMORY.md`. The corpus (RAG) stores embeddings of wiki pages and documents, each following an OKF (Open Knowledge Framework)-aligned front-matter convention (`type` required; `title`/`description`/`resource`/`tags`/`timestamp` optional); a per-directory `index.md` and a dated `logs.md` changelog are deterministically regenerated from on-disk state after every write, never model-authored. `MEMORY.md`/`index.md`/`logs.md` are structural/generated files, always excluded from RAG indexing, the graph, and the attachment picker. A user profile (`wiki/users/michael.md`) is embedded line-by-line and injected only where relevant (cosine ≥ 0.45).

**Prompt layout** — fixed 7-slot structure (identity, persona, episodic+profile, RAG, tool results, working memory, instruction) optimized for KV-cache reuse. Local working-memory budget is now sized from the active model's real context window (oMLX's reported `max_model_len`) rather than a fixed turn count; on oMLX specifically, working-memory turns are sent as discrete messages — mirroring oMLX's own client — instead of flattened into one string, for genuine cross-turn KV-cache reuse.

---

## Project Structure

```
localist/
├── backend/
│   ├── main.py                  # FastAPI entry, port 8001
│   ├── controller_agent.py      # Task orchestration
│   ├── planner.py               # Routing rules (P0–P6)
│   ├── conversational_agent.py  # Prompt assembly, RAG, tools
│   ├── wiki_agent.py            # Document ingestion / diff
│   ├── prompt_builder.py        # 7-slot prompt assembler
│   ├── mcp_tool_dispatcher.py   # MCP/SSE client to localist-mcp; research loop
│   ├── memory_manager.py        # Episodic + RAG memory
│   ├── episodic_extractor.py    # Episode extraction
│   ├── content_safety.py        # Pre-write content scanner
│   ├── embedding_engine.py      # Local embedding engine
│   ├── runtime_factory.py       # Backend selection (foundry/omlx/ollama), live-swappable
│   ├── chart_tool_schema.py     # generate_chart argument extraction/validation
│   ├── mcp_server/              # localist-mcp — port 8003 (web_search, fetch_url, file_op, generate_chart)
│   ├── wiki/                    # Persona, user profile, indexed pages, MEMORY.md/index.md/logs.md
│   ├── tests/                   # Unit + integration tests by phase
│   └── requirements.txt
├── diagnostics/                 # Read-only live-verification scripts (not part of the test suite)
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

**Done:** Localist CLI launcher, MCP migration (tools off legacy dispatcher/Fetcher), identity continuity, user profile injection, generate-then-save file ops, graph retrieval layer (SQLite schema v6), Ollama runtime backend (incl. Cloud) with real `/api/embed` support and a cross-platform local embedding path, chat-model fail-fast validation, wiki diff updates with review/apply UI (plus a pre-write snapshot safety net with a 30-day-TTL undo path, closing the prior no-rollback gap), episodic memory hardening (real cosine retrieval, write-approval gate, semantic retraction), KaTeX-rendered math in chat output, oMLX-specific multi-turn prompt caching (working memory sent as discrete per-turn messages, sized from the model's real context window), live-switchable runtime backend with per-backend chat-model pinning (no restart required), a diagnostic-first `generate_chart` tool (interactive Chart.js rendering in the UI), a bounded research loop with a relevance-aware answer gate for price/spec-lookup queries, explicit `/chart`/`/research` slash commands that bypass normal tool-detection paths on demand, chat attachments that can pin an existing wiki page (not just upload a local file) with a Planner short-circuit routing a pinned-page diff request straight to `WikiAgent`, and OKF (Open Knowledge Framework)-aligned wiki front matter/structure (reconciled `type`/`title`/`description`/`resource`/`tags`/`timestamp` convention, deterministically-generated `index.md`/`logs.md`, and a one-time backfill of the existing corpus).

**Open:** generalize the bullet/diff-marker collision edge case; multi-diff wiki-update turns (UI only exercises the single-diff case); broader Localist UI rework for episode browsing and tool result display; macOS `.app` packaging via PyInstaller + Tauri.
