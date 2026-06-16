"""
LORA — FastAPI Backend
======================
The HTTP boundary between the Svelte UI and the agent core.

Layer placement
---------------
  Svelte UI  →  FastAPI (this module)  →  ControllerAgent  →  Sub-agents / Runtime
                                       →  MemoryManager    →  SQLite (local file)

Architectural contract
----------------------
- This is the ONLY module that imports FastAPI, Pydantic, or anything HTTP-related.
- All agent logic lives in controller_agent.py, wiki_agent.py, conversational_agent.py.
- All model inference flows through the active RuntimeClient.
- This module constructs the runtime, MemoryManager, agents, and controller
  once at startup and holds them as app-level state.  No singleton is
  constructed per-request.
- Long-running synchronous calls (controller.handle_task, runtime.infer) are
  dispatched to a thread pool via asyncio.to_thread so they never block the
  event loop.

Endpoints
---------
  POST /task
      Submit a task.  Accepts a TaskRequest body, calls handle_task(),
      returns a TaskResponse.  The response always includes task_id and
      status so the caller can correlate.

  POST /task/stream
      Submit a task whose synthesis answer is streamed back token-by-token
      as Server-Sent Events (SSE).  Planning and sub-agent dispatch still
      run synchronously in the background; only the final synthesis call
      streams.  The event stream closes with a [DONE] sentinel.

  GET /health
      Calls runtime.health_check() and returns its dict.  Returns HTTP 200
      even when the runtime is unreachable so the UI can display a degraded
      state rather than a hard error page.  A separate "healthy" boolean in
      the body signals true service health to automated monitors.

  GET /agents
      Returns the list of registered agent names.  Useful for the UI to
      show which capabilities are active without parsing log files.

  GET /memory/stats
      Returns MemoryManager statistics: document counts, DB size, embedding
      coverage, cache state.  Always HTTP 200 — degraded values are shown
      when the MemoryManager is not initialised.

  GET /files/raw
      List all .md and .txt files in the raw/ directory.

  GET /files/wiki
      List all .md files in the wiki/ directory.

  GET /files/content?path=<absolute_path>
      Return the plain-text content of a file.  Only paths inside raw_dir
      or wiki_dir are permitted — anything else returns HTTP 403.

  POST /files/upload
      Accept a multipart .md or .txt file upload and save it to raw/.
      Immediately indexes the file in MemoryManager.

Running locally
---------------
  uvicorn main:app --reload --host 127.0.0.1 --port 8001

Environment / configuration
----------------------------
All tuneable values are in the ``Settings`` class (pydantic-settings).  They
can be overridden via environment variables or a .env file:

  LORA_RUNTIME_BACKEND             Runtime backend: "foundry" | "omlx" (default "foundry")
  LORA_CHAT_MODEL                  Chat model ID (interpreted by the active backend)
  LORA_FOUNDRY_URL                 Override auto-resolved Foundry base URL (foundry only)
  LORA_OMLX_URL                    oMLX server base URL (omlx only, default http://localhost:8000)
  LORA_LOG_LEVEL                   Root log level (default INFO)
  LORA_WIKI_DIR                    Absolute path to the wiki directory
  LORA_RAW_DIR                     Absolute path to the raw files directory
  LORA_SCHEMA_PATH                 Absolute path to SCHEMA.md
  LORA_TEMPLATES_DIR               Absolute path to the templates directory
  LORA_AUTO_APPLY                  Whether WikiAgent writes to disk immediately (bool)
  LORA_STREAM_TIMEOUT              Streaming timeout in seconds (float)
  LORA_REQUEST_TIMEOUT             Non-streaming timeout in seconds (float)
  LORA_MEMORY_DB                   Absolute path to the SQLite memory DB file.
                                   Defaults to <project_root>/lora_memory.db
  LORA_EMBEDDING_ENGINE_ENABLED    Load the standalone MLX-LM EmbeddingEngine at
                                   startup (bool, default True).  Set False to run
                                   in keyword-only mode without loading the model.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import datetime
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project imports (agent core + runtime)
# ---------------------------------------------------------------------------

from base_runtime_client import BaseRuntimeClient
from controller_agent import ControllerAgent
from conversational_agent import ConversationalAgent
from embedding_engine import EmbeddingEngine
from memory_manager import MemoryManager
from runtime_factory import create_runtime
from wiki_agent import WikiAgent

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    All configuration for the LORA backend.
    Override any field via environment variable or .env file.
    """
    model_config = SettingsConfigDict(env_prefix="LORA_", env_file=".env", extra="ignore")

    # Runtime backend selection
    runtime_backend: str = "foundry"

    # Model ID — chat only; embeddings are handled by EmbeddingEngine (MLX-LM),
    # not by the runtime backend.
    chat_model:       str = "Phi-4-mini-instruct-generic-gpu:5"

    # Foundry network (foundry backend only)
    foundry_url:      str | None = None

    # oMLX network (omlx backend only)
    omlx_url:         str = "http://localhost:8000"

    # Shared network timeouts
    stream_timeout:   float = 60.0
    request_timeout:  float = 30.0

    # Paths — resolved at startup; defaults are relative to project root
    wiki_dir:        str | None = None
    raw_dir:         str | None = None
    schema_path:     str | None = None
    templates_dir:   str | None = None

    # MemoryManager
    memory_db:                str | None = None   # None → <project_root>/lora_memory.db

    # EmbeddingEngine — standalone MLX-LM embedding, backend-agnostic.
    # Set False to skip model load and run MemoryManager in keyword-only mode.
    embedding_engine_enabled: bool = True

    # Agent behaviour
    auto_apply:      bool = False

    # Logging
    log_level:       str = "INFO"


# ---------------------------------------------------------------------------
# App-level state (constructed once at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds singletons that live for the lifetime of the process."""

    def __init__(self) -> None:
        self.runtime:           BaseRuntimeClient | None = None
        self.controller:        ControllerAgent   | None = None
        self.memory_manager:    MemoryManager     | None = None
        self.embedding_engine:  EmbeddingEngine   | None = None
        self.settings:          Settings          | None = None
        # Resolved at startup by lifespan()
        self.wiki_dir:          Path | None = None
        self.raw_dir:           Path | None = None
        self.schema_path:       Path | None = None
        self.templates_dir:     Path | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Construct the runtime, MemoryManager, agents, and controller once at
    startup.  Runs in the main thread before the first request is accepted.
    """
    settings = Settings()
    _state.settings = settings

    # Configure logging
    logging.basicConfig(
        level   = getattr(logging, settings.log_level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    logger.info("LORA backend starting up.")

    project_root = Path(__file__).resolve().parent

    # -- Runtime -------------------------------------------------------------

    runtime = create_runtime(
        backend         = settings.runtime_backend,
        chat_model      = settings.chat_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        request_timeout = settings.request_timeout,
        stream_timeout  = settings.stream_timeout,
    )
    _state.runtime = runtime

    health = runtime.health_check()
    if health["reachable"]:
        logger.info(
            "%s runtime reachable at %s — chat_found=%s  embed_found=%s",
            settings.runtime_backend.upper(),
            health["base_url"],
            health["chat_model_found"],
            health["embed_model_found"],
        )
    else:
        logger.warning(
            "%s runtime NOT reachable at startup (%s). "
            "Requests will fail until the service is running.",
            settings.runtime_backend.upper(),
            health.get("base_url"),
        )

    # -- Resolve path defaults -----------------------------------------------

    wiki_dir      = Path(settings.wiki_dir)      if settings.wiki_dir      else project_root / "wiki"
    raw_dir       = Path(settings.raw_dir)       if settings.raw_dir       else project_root / "raw"
    schema_path   = Path(settings.schema_path)   if settings.schema_path   else project_root / "SCHEMA.md"
    templates_dir = Path(settings.templates_dir) if settings.templates_dir else project_root / "templates"
    memory_db     = Path(settings.memory_db)     if settings.memory_db     else project_root / "lora_memory.db"

    # -- EmbeddingEngine (standalone MLX-LM, backend-agnostic) ---------------
    #
    # Embeddings are now independent of the runtime backend.  EmbeddingEngine
    # loads mlx-community/embeddinggemma-300m-4bit via mlx_lm at startup.
    # On failure it logs a warning and sets available=False — MemoryManager
    # then falls back to keyword-only retrieval without surfacing errors.

    embed_fn = None
    if settings.embedding_engine_enabled:
        embedding_engine = EmbeddingEngine()
        _state.embedding_engine = embedding_engine
        if embedding_engine.available:
            embed_fn = embedding_engine.embed
            logger.info("EmbeddingEngine ready — embeddings enabled.")
        else:
            logger.warning(
                "EmbeddingEngine failed to load — MemoryManager will run "
                "in keyword-only mode.  Install mlx-lm and retry."
            )
    else:
        logger.info(
            "EmbeddingEngine disabled (LORA_EMBEDDING_ENGINE_ENABLED=false) — "
            "MemoryManager will run in keyword-only mode."
        )

    memory_manager = MemoryManager(db_path=memory_db, embed_fn=embed_fn)
    _state.memory_manager = memory_manager

    # Seed the document index from disk on startup.  index_directory() is
    # idempotent — unchanged files are skipped via content-hash comparison.
    # This ensures the index is always current even after pages were written
    # while the server was down.
    if wiki_dir.exists():
        n_wiki = memory_manager.index_directory(wiki_dir, doc_type="wiki", embed=bool(embed_fn))
        logger.info("MemoryManager: indexed %d wiki pages from %s.", n_wiki, wiki_dir)
    else:
        logger.warning("MemoryManager: wiki_dir does not exist yet (%s) — skipping seed.", wiki_dir)

    if raw_dir.exists():
        n_raw = memory_manager.index_directory(raw_dir, doc_type="raw", embed=bool(embed_fn))
        logger.info("MemoryManager: indexed %d raw files from %s.", n_raw, raw_dir)
    else:
        logger.info("MemoryManager: raw_dir does not exist yet (%s) — skipping seed.", raw_dir)

    stats = memory_manager.stats()
    logger.info(
        "MemoryManager ready — wiki=%d  raw=%d  db_size=%.1f KB  embeddings=%.0f%%",
        stats["wiki_docs"], stats["raw_docs"],
        stats["db_size_kb"], stats["embeddings_pct"],
    )

    # -- Agents --------------------------------------------------------------

    wiki_agent = WikiAgent(
        runtime        = runtime,
        project_root   = project_root,
        memory_manager = memory_manager,
    )
    conversational_agent = ConversationalAgent(
        runtime        = runtime,
        memory_manager = memory_manager,
        project_root   = project_root,
    )

    # -- Store resolved paths in state so endpoints can inject them ----------

    _state.wiki_dir      = wiki_dir
    _state.raw_dir       = raw_dir
    _state.schema_path   = schema_path
    _state.templates_dir = templates_dir

    # -- Controller ----------------------------------------------------------

    controller = ControllerAgent(
        runtime        = runtime,
        agents         = [wiki_agent, conversational_agent],
        memory_manager = memory_manager,
    )
    _state.controller = controller

    logger.info(
        "ControllerAgent ready — agents: %s",
        [wiki_agent.name, conversational_agent.name],
    )

    yield  # — application runs —

    logger.info("LORA backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LORA — Local Reasoning Agent",
    description = "Multi-agent research system — WikiAgent + corpus-aware ConversationalAgent (RAG).",
    version     = "0.4.0",
    lifespan    = lifespan,
)

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173",
                         "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    """
    Payload accepted by POST /task and POST /task/stream.

    ``instruction`` is the only required field.  ``context`` is passed
    through verbatim to the controller and then on to each agent:

        {
            "query":      "What do we know about attention mechanisms?",
            "raw_path":   "/abs/path/to/paper.md",
            "auto_apply": false
        }
    """
    task_id:     str              = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str              = Field(..., min_length=1)
    context:     dict[str, Any]   = Field(default_factory=dict)
    metadata:    dict[str, Any]   = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Serialised ControllerResult returned by POST /task."""
    task_id:  str
    status:   str
    answer:   str
    sources:  list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any]       = Field(default_factory=dict)
    error:    str | None           = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    healthy:           bool
    reachable:         bool
    base_url:          str
    models:            list[str]  = Field(default_factory=list)
    chat_model_found:  bool       = False
    embed_model_found: bool       = False
    error:             str | None = None


class AgentsResponse(BaseModel):
    """Response body for GET /agents."""
    agents: list[str]


class MemoryStatsResponse(BaseModel):
    """Response body for GET /memory/stats."""
    db_path:          str
    db_size_kb:       float
    wiki_docs:        int
    raw_docs:         int
    conv_log_rows:    int
    cache_valid:      int
    cache_invalid:    int
    embeddings_pct:   float
    available:        bool   # False when MemoryManager is not initialised


class FileEntry(BaseModel):
    """Metadata for a single file in raw/ or wiki/."""
    name:     str    # stem without extension
    filename: str    # filename with extension, e.g. "my-doc.md"
    path:     str    # absolute path — passed as context.raw_path on ingest
    size:     int    # bytes
    modified: str    # ISO-8601 UTC timestamp


class FilesResponse(BaseModel):
    """Response body for GET /files/raw and GET /files/wiki."""
    files: list[FileEntry]


class FileContentResponse(BaseModel):
    """Response body for GET /files/content."""
    path:    str
    content: str


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require_controller() -> ControllerAgent:
    if _state.controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialised.")
    return _state.controller


def _require_runtime() -> BaseRuntimeClient:
    if _state.runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialised.")
    return _state.runtime


def _enrich_context(context: dict[str, Any]) -> dict[str, Any]:
    """
    Merge app-level path defaults into the caller-supplied context dict.
    Caller-supplied values always win — this only fills gaps.
    """
    defaults: dict[str, Any] = {}

    if _state.wiki_dir:
        defaults["wiki_dir"] = str(_state.wiki_dir)
    if _state.raw_dir:
        defaults["raw_dir"] = str(_state.raw_dir)
    if _state.schema_path:
        defaults["schema_path"] = str(_state.schema_path)
    if _state.templates_dir:
        defaults["templates_dir"] = str(_state.templates_dir)
    if _state.settings:
        defaults["auto_apply"] = _state.settings.auto_apply

    return {**defaults, **context}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/task",
    response_model       = TaskResponse,
    summary              = "Submit a task (blocking)",
    response_description = "The completed task result.",
)
async def post_task(request: TaskRequest) -> TaskResponse:
    """
    Submit an instruction to the LORA multi-agent system.

    The call blocks until the full pipeline (plan → dispatch → synthesize)
    completes.  Use POST /task/stream to receive tokens incrementally.
    """
    controller = _require_controller()

    task_dict = {
        "task_id":     request.task_id,
        "instruction": request.instruction,
        "context":     _enrich_context(request.context),
        "metadata":    request.metadata,
    }

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            controller.handle_task, task_dict
        )
    except Exception as exc:
        logger.exception("Unhandled error in POST /task for task %s.", request.task_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TaskResponse(**result)


@app.post(
    "/task/stream",
    summary = "Submit a task and stream the synthesis answer (SSE)",
)
async def post_task_stream(request: TaskRequest) -> StreamingResponse:
    """
    Submit a task and receive the synthesis answer as a Server-Sent Events stream.

    Event format:

        data: {"type": "status",  "message": "Planning..."}
        data: {"type": "token",   "token": "The"}
        data: {"type": "sources", "sources": [...]}
        data: {"type": "done",    "task_id": "...", "status": "complete"}
        data: [DONE]
    """
    controller = _require_controller()
    runtime    = _require_runtime()

    task_dict = {
        "task_id":     request.task_id,
        "instruction": request.instruction,
        "context":     _enrich_context(request.context),
        "metadata":    request.metadata,
    }

    return StreamingResponse(
        _stream_task(controller, runtime, task_dict, request.task_id),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Runtime service health check",
)
async def get_health() -> HealthResponse:
    """
    Check whether the active runtime backend is reachable and the configured
    models are available.  Always returns HTTP 200.
    """
    runtime = _require_runtime()
    raw: dict[str, Any] = await asyncio.to_thread(runtime.health_check)

    # Embedding availability is now independent of the runtime backend —
    # it comes from EmbeddingEngine (MLX-LM), not from oMLX's model list.
    embedding_engine = _state.embedding_engine
    embed_available  = embedding_engine is not None and embedding_engine.available

    return HealthResponse(
        healthy           = bool(raw.get("reachable") and raw.get("chat_model_found")),
        reachable         = bool(raw.get("reachable", False)),
        base_url          = str(raw.get("base_url", "")),
        models            = raw.get("models", []),
        chat_model_found  = bool(raw.get("chat_model_found", False)),
        embed_model_found = embed_available,
        error             = raw.get("error"),
    )


@app.get(
    "/agents",
    response_model = AgentsResponse,
    summary        = "List registered agents",
)
async def get_agents() -> AgentsResponse:
    """Return the names of all agents currently registered with the controller."""
    controller = _require_controller()
    return AgentsResponse(agents=list(controller._agents.keys()))


@app.get(
    "/memory/stats",
    response_model = MemoryStatsResponse,
    summary        = "MemoryManager statistics",
)
async def get_memory_stats() -> MemoryStatsResponse:
    """
    Return MemoryManager statistics.

    Always HTTP 200.  When the MemoryManager is not initialised (e.g. startup
    failure), all numeric fields are 0 and ``available`` is False.
    """
    mm = _state.memory_manager
    if mm is None:
        return MemoryStatsResponse(
            db_path        = "",
            db_size_kb     = 0.0,
            wiki_docs      = 0,
            raw_docs       = 0,
            conv_log_rows  = 0,
            cache_valid    = 0,
            cache_invalid  = 0,
            embeddings_pct = 0.0,
            available      = False,
        )

    raw: dict[str, Any] = await asyncio.to_thread(mm.stats)
    return MemoryStatsResponse(
        db_path        = raw["db_path"],
        db_size_kb     = raw["db_size_kb"],
        wiki_docs      = raw["wiki_docs"],
        raw_docs       = raw["raw_docs"],
        conv_log_rows  = raw["conv_log_rows"],
        cache_valid    = raw["cache_valid"],
        cache_invalid  = raw["cache_invalid"],
        embeddings_pct = raw["embeddings_pct"],
        available      = True,
    )




# ---------------------------------------------------------------------------
# File management endpoints
# ---------------------------------------------------------------------------

def _file_entry(p: "Path") -> FileEntry:
    """Build a FileEntry from a Path."""
    stat = p.stat()
    return FileEntry(
        name     = p.stem,
        filename = p.name,
        path     = str(p.resolve()),
        size     = stat.st_size,
        modified = datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=datetime.timezone.utc
        ).isoformat(),
    )


@app.get(
    "/files/raw",
    response_model = FilesResponse,
    summary        = "List raw files",
)
async def get_files_raw() -> FilesResponse:
    """Return metadata for every .md and .txt file in the raw/ directory."""
    if _state.raw_dir is None:
        raise HTTPException(status_code=503, detail="raw_dir not configured.")
    raw_dir = _state.raw_dir
    if not raw_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p)
        for p in sorted(raw_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/wiki",
    response_model = FilesResponse,
    summary        = "List wiki pages",
)
async def get_files_wiki() -> FilesResponse:
    """Return metadata for every .md file in the wiki/ directory."""
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")
    wiki_dir = _state.wiki_dir
    if not wiki_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p)
        for p in sorted(wiki_dir.iterdir())
        if p.is_file() and p.suffix == ".md"
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/content",
    response_model = FileContentResponse,
    summary        = "Read file content",
)
async def get_file_content(path: str) -> FileContentResponse:
    """
    Return the plain-text content of a file by absolute path.

    Only paths inside raw_dir or wiki_dir are permitted — anything else
    returns HTTP 403.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if not any(str(target).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        content = await asyncio.to_thread(target.read_text, "utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc

    return FileContentResponse(path=str(target), content=content)


@app.post(
    "/files/upload",
    response_model = FileEntry,
    summary        = "Upload a raw file",
)
async def post_file_upload(file: UploadFile = File(...)) -> FileEntry:
    """
    Accept a multipart file upload and save it to raw/.

    Only .md and .txt files are accepted.  If a file with the same name
    already exists it is overwritten.  Returns the FileEntry for the
    saved file.
    """
    if _state.raw_dir is None:
        raise HTTPException(status_code=503, detail="raw_dir not configured.")

    filename = file.filename or "upload.md"
    suffix   = Path(filename).suffix.lower()
    if suffix not in {".md", ".txt"}:
        raise HTTPException(
            status_code=422,
            detail=f"Only .md and .txt files are accepted, got: {suffix}",
        )

    raw_dir = _state.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename

    try:
        contents = await file.read()
        await asyncio.to_thread(dest.write_bytes, contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    # Index the newly uploaded file immediately so ConversationalAgent can find it
    # without waiting for the next server restart seed pass.
    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(
                _state.memory_manager.index_document,
                dest,
                "raw",
                None,
                False,
            )
        except Exception as exc:
            logger.warning(
                "MemoryManager.index_document failed for upload %s: %s", filename, exc
            )

    return _file_entry(dest)

# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

async def _stream_task(
    controller: ControllerAgent,
    runtime:    BaseRuntimeClient,
    task_dict:  dict[str, Any],
    task_id:    str,
) -> AsyncIterator[str]:
    """
    Async generator that drives the streaming endpoint.

    Pipeline
    --------
    1.  Emit a "Planning..." status event.
    2.  Run the full handle_task() pipeline in a thread pool.
    3.  Emit a "Streaming answer…" status event.
    4.  Replay the completed answer word-by-word as token events.
    5.  Emit sources, done, and the [DONE] sentinel.
    """

    def _sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield _sse({"type": "status", "message": "Planning task…", "task_id": task_id})

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            controller.handle_task, task_dict
        )
    except Exception as exc:
        logger.exception("Error during planning/dispatch for task %s.", task_id)
        yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    if result.get("status") == "failed":
        yield _sse({
            "type":    "error",
            "message": result.get("error", "Task failed during planning or dispatch."),
            "task_id": task_id,
        })
        yield "data: [DONE]\n\n"
        return

    yield _sse({"type": "status", "message": "Streaming answer…", "task_id": task_id})

    answer: str = result.get("answer", "")
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else " " + word
        yield _sse({"type": "token", "token": chunk})
        await asyncio.sleep(0)

    yield _sse({"type": "sources",  "sources": result.get("sources", [])})
    yield _sse({
        "type":     "done",
        "task_id":  task_id,
        "status":   result.get("status", "complete"),
        "metadata": result.get("metadata", {}),
    })
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = 500,
        content     = {
            "detail": str(exc),
            "path":   str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host      = "127.0.0.1",
        port      = 8001,
        reload    = True,
        log_level = "info",
    )