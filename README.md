# LORA — Local Reasoning Agent

A local-first, multi-agent research system powered by Azure AI Foundry.
All inference runs on local hardware. No external cloud APIs are called during normal operation.

---

## Project structure

```
lora-app-demo/
├── backend/
│   ├── main.py                        # FastAPI app — HTTP boundary
│   ├── controller_agent.py            # Central coordinator
│   ├── foundry_runtime_client.py      # Azure AI Foundry transport layer
│   ├── wiki_agent.py                  # Write path: raw file → wiki pages
│   ├── research_agent.py              # Read path: question → research report
│   ├── agent_wiki_loop_streaming.py   # Standalone wiki ingestion logic (replace stub)
│   ├── SCHEMA.md                      # Wiki page schema (edit to match your corpus)
│   ├── templates/                     # Page templates used by WikiAgent
│   │   ├── system.md
│   │   ├── concept.md
│   │   └── research-note.md
│   ├── wiki/                          # Generated wiki pages (starts empty)
│   ├── raw/                           # Drop raw .md / .txt files here
│   ├── logs/                          # Runtime log output
│   ├── requirements.txt
│   ├── .env.example                   # Copy to .env and fill in values
│   └── api.http                       # REST Client test file (VSCode)
└── .vscode/
    ├── settings.json
    ├── launch.json                    # Debug configurations
    └── extensions.json
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- [Azure AI Foundry](https://ai.azure.com/) installed and running locally
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
# Edit .env — set LORA_CHAT_MODEL and LORA_EMBEDDING_MODEL to match
# the model IDs shown in `foundry service status`
```

### 4. Drop in your real wiki agent logic

Replace `agent_wiki_loop_streaming.py` with your tested standalone
implementation. The stub exports the correct interface so the server
starts cleanly before you do this.

### 5. Start Foundry

```bash
foundry service start
```

### 6. Start the LORA backend

```bash
# From backend/ with .venv active:
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Or use the **"LORA: FastAPI (uvicorn)"** launch config in VSCode (`F5`).

### 7. Test the API

Open `backend/api.http` in VSCode and click **Send Request** on any block,
or use curl:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Research task
curl -X POST http://127.0.0.1:8000/task \
  -H "Content-Type: application/json" \
  -d '{"instruction": "What do we know about attention mechanisms?", "context": {"query": "attention mechanisms"}}'
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Foundry reachability + model availability |
| `GET`  | `/agents` | List registered agent names |
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
    ├── Planner  ──────────────────► FoundryRuntimeClient.infer()
    │       │ [SubTask list]
    │       ▼
    ├── _dispatch()
    │       ├──► WikiAgent      ──► agent_wiki_loop_streaming
    │       └──► ResearchAgent  ──► FoundryRuntimeClient.infer() / embed()
    │
    └── Synthesizer ───────────────► FoundryRuntimeClient.infer()
```

---

## What still needs to be built

- **`agent_wiki_loop_streaming.py`** — replace the stub with your real implementation.
- **Svelte UI** — frontend layer (see project summary for spec).
- **Streaming synthesis** — wire `_iter_sse_chunks` into the `/task/stream` pipeline for true real-time token delivery during synthesis.
- **Persistent MemoryManager** — swap the in-process list for SQLite or a vector store.

See `LORA_project_summary.md` for the full roadmap.
