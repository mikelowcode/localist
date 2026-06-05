# LORA — Local Reasoning Agent

A local-first, multi-agent research system. All inference runs on local hardware via Azure AI Foundry or oMLX. No external cloud APIs are called during normal operation.

---

## Project structure

```
lora-app-demo/
├── backend/
│   ├── main.py                      # FastAPI app — HTTP boundary only
│   ├── controller_agent.py          # Central coordinator (Planner, Synthesizer, MemoryManager)
│   ├── base_runtime_client.py       # BaseRuntimeClient Protocol — shared interface for all backends
│   ├── runtime_factory.py           # Constructs the active runtime from LORA_RUNTIME_BACKEND
│   ├── foundry_runtime_client.py    # Azure AI Foundry transport (auto-resolves ephemeral port)
│   ├── omlx_runtime_client.py       # oMLX transport (OpenAI-compatible local API)
│   ├── wiki_agent.py                # Write path: raw file → structured wiki pages
│   ├── research_agent.py            # Read path: question → iterative research report
│   ├── SCHEMA.md                    # Wiki page schema (edit to match your corpus)
│   ├── templates/                   # Page templates used by WikiAgent
│   │   ├── system.md
│   │   ├── concept.md
│   │   └── research-note.md
│   ├── wiki/                        # Generated wiki pages (starts empty)
│   ├── raw/                         # Drop raw .md / .txt files here
│   ├── logs/                        # Runtime log output
│   ├── requirements.txt
│   ├── .env.example                 # Copy to .env and fill in values
│   └── api.http                     # REST Client test file (VSCode)
└── .vscode/
    ├── settings.json
    ├── launch.json                  # Debug configurations
    └── extensions.json
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- [Azure AI Foundry](https://ai.azure.com/) installed locally **or** [oMLX](https://github.com/ml-explore/mlx-examples) running at `http://localhost:8000`
- VSCode (recommended) with the extensions in `.vscode/extensions.json`

### 2. Set up the Python environment

```bash
cd ~/Projects/lora-app-demo/backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — key variables listed below
```

| Variable | Default | Description |
|---|---|---|
| `LORA_RUNTIME_BACKEND` | `foundry` | `foundry` or `omlx` |
| `LORA_CHAT_MODEL` | `Phi-4-mini-instruct-generic-gpu:5` | Chat model ID (backend-specific) |
| `LORA_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model ID (Foundry only) |
| `LORA_FOUNDRY_URL` | *(auto-resolved)* | Override Foundry base URL |
| `LORA_OMLX_URL` | `http://localhost:8000` | oMLX server base URL |
| `LORA_WIKI_DIR` | `./wiki` | Absolute path to wiki directory |
| `LORA_RAW_DIR` | `./raw` | Absolute path to raw files directory |
| `LORA_AUTO_APPLY` | `false` | WikiAgent: write pages to disk immediately |
| `LORA_LOG_LEVEL` | `INFO` | Root log level |

### 4. Start your inference backend

**Foundry:**
```bash
foundry service start
```

**oMLX:**
```bash
# Start your oMLX server — confirm it's live at http://localhost:8000/v1/models
```

### 5. Start the LORA backend

```bash
# From backend/ with .venv active:
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Or use the **"LORA: FastAPI (uvicorn)"** launch config in VSCode (`F5`).

### 6. Test the API

Open `backend/api.http` in VSCode and click **Send Request**, or use curl:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Research task
curl -X POST http://127.0.0.1:8000/task \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "What do we know about attention mechanisms?",
    "context": {"query": "attention mechanisms"}
  }'

# Wiki ingestion
curl -X POST http://127.0.0.1:8000/task \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Ingest this raw file into the wiki.",
    "context": {
      "raw_path": "/abs/path/to/raw/paper.md",
      "auto_apply": true
    }
  }'
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Backend reachability + model availability |
| `GET` | `/agents` | List registered agent names |
| `POST` | `/task` | Submit a task (blocking, returns full result) |
| `POST` | `/task/stream` | Submit a task, stream answer as SSE |

### POST /task — request body

```json
{
  "task_id": "optional-uuid",
  "instruction": "Your research question or ingestion command.",
  "context": {
    "query":          "focused sub-question for ResearchAgent",
    "raw_path":       "/abs/path/to/raw/file.md",
    "auto_apply":     false,
    "max_sources":    5,
    "max_iterations": 3,
    "use_embeddings": false
  }
}
```

### POST /task/stream — SSE event format

```
data: {"type": "status",  "message": "Planning task…"}
data: {"type": "token",   "token": "The "}
data: {"type": "token",   "token": "answer "}
data: {"type": "sources", "sources": [...]}
data: {"type": "done",    "task_id": "...", "status": "complete"}
data: [DONE]
```

> **Note:** Streaming currently replays the completed answer word-by-word after the full pipeline finishes. True token-level streaming during synthesis is listed under [What still needs to be built](#what-still-needs-to-be-built).

---

## Architecture

```
Svelte UI  (not yet built)
    │  HTTP / SSE
    ▼
FastAPI  (main.py)
    │  task_dict
    ▼
ControllerAgent  (controller_agent.py)
    │
    ├── Planner  ──────────────────► RuntimeClient.infer()
    │       │ [SubTask list]
    │       ▼
    ├── _dispatch()
    │       ├──► WikiAgent        ──► RuntimeClient.infer()
    │       └──► ResearchAgent    ──► RuntimeClient.infer() / embed()
    │
    └── Synthesizer ───────────────► RuntimeClient.infer()
                                            │
                               ┌────────────┴─────────────┐
                               ▼                           ▼
                    FoundryRuntimeClient        OMLXRuntimeClient
                    (Azure AI Foundry)          (oMLX local API)
```

The `RuntimeFactory` selects the active backend at startup based on `LORA_RUNTIME_BACKEND`. All agents and the controller are typed against `BaseRuntimeClient` and are completely unaware of which backend is running.

---

## Agent overview

### WikiAgent (`wiki_agent.py`)

The **write path**. Accepts a raw `.md` or `.txt` file and uses the model to produce structured wiki actions: new `RESEARCH_NOTE`, `CONCEPT`, or `SYSTEM` pages, and unified diffs for existing pages.

- Set `auto_apply: true` in context to write immediately; leave `false` to review proposed actions first.
- Page content is validated against `SCHEMA.md` and the templates in `templates/`.
- Optionally appends a `JournalEntry` to a `.jsonl` file for audit trails.

### ResearchAgent (`research_agent.py`)

The **read path**. Accepts a research question and iteratively retrieves, reads, and synthesises information from the wiki and raw directories.

- Decomposes queries into sub-queries, retrieves relevant documents by keyword overlap, and extracts discrete claims with source references.
- Set `use_embeddings: true` to re-rank retrieved documents by embedding cosine similarity (requires an embedding model in your backend).
- Returns a structured Markdown report with claims, sources, detected gaps, and sub-query provenance.

---

## Adding a new runtime backend

1. Write a class that satisfies `BaseRuntimeClient` (`base_runtime_client.py`) — implement `infer()`, `embed()`, and `infer_stream()`.
2. Add a factory function and registry entry in `runtime_factory.py`.
3. Add any new config fields to `Settings` in `main.py` and pass them through `create_runtime()`.

Nothing else in the stack changes.

---

## What still needs to be built

- **Svelte UI** — the frontend layer. The SSE event format and `/task`, `/health`, `/agents` endpoints are stable and ready to consume.
- **True streaming synthesis** — currently the pipeline completes fully before tokens are replayed. Wire `infer_stream()` into the synthesis step so tokens arrive in real time.
- **Persistent MemoryManager** — the current in-process list is evicted after 200 entries. Swap the storage backend for SQLite or a vector store without changing the `MemoryManager` interface.
- **oMLX embedding support** — `OMLXRuntimeClient.embed()` raises `NotImplementedError` until an embedding model is confirmed available. Once one is loaded, set `LORA_EMBEDDING_MODEL` and enable `use_embeddings: true` in ResearchAgent calls.