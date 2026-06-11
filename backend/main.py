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
- All agent logic lives in controller_agent.py, wiki_agent.py, research_agent.py.
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

Running locally
---------------
  uvicorn main:app --reload --host 127.0.0.1 --port 8001

Environment / configuration
----------------------------
All tuneable values are in the ``Settings`` class (pydantic-settings).  They
can be overridden via environment variables or a .env file:

  LORA_RUNTIME_BACKEND    Runtime backend: "foundry" | "omlx" (default "foundry")
  LORA_CHAT_MODEL         Chat model ID (interpreted by the active backend)
  LORA_EMBEDDING_MODEL    Embedding model ID (interpreted by the active backend)
  LORA_FOUNDRY_URL        Override auto-resolved Foundry base URL (foundry only)
  LORA_OMLX_URL           oMLX server base URL (omlx only, default http://localhost:8000)
  LORA_LOG_LEVEL          Root log level (default INFO)
  LORA_WIKI_DIR           Absolute path to the wiki directory
  LORA_RAW_DIR            Absolute path to the raw files directory
  LORA_SCHEMA_PATH        Absolute path to SCHEMA.md
  LORA_TEMPLATES_DIR      Absolute path to the templates directory
  LORA_AUTO_APPLY         Whether WikiAgent writes to disk immediately (bool)
  LORA_STREAM_TIMEOUT     Streaming timeout in seconds (float)
  LORA_REQUEST_TIMEOUT    Non-streaming timeout in seconds (float)
  LORA_MEMORY_DB          Absolute path to the SQLite memory DB file.
                          Defaults to <project_root>/lora_memory.db
  LORA_MEMORY_EMBED       Whether to embed documents at startup (bool, default True).
                          Set False to skip embedding on the initial index pass —
                          useful when the embedding model is not yet available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project imports (agent core + runtime)
# ---------------------------------------------------------------------------

from base_runtime_client import BaseRuntimeClient
from controller_agent import ControllerAgent
from memory_manager import MemoryManager
from research_agent import ResearchAgent
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

    # Model IDs — interpreted by the active backend
    chat_model:       str = "Phi-4-mini-instruct-generic-gpu:5"
    embedding_model:  str = "text-embedding-3-small"

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
    memory_db:       str | None = None   # None → <project_root>/lora_memory.db
    memory_embed:    bool = True          # embed documents during startup index pass

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
        self.runtime:        BaseRuntimeClient | None = None
        self.controller:     ControllerAgent   | None = None
        self.memory_manager: MemoryManager     | None = None
        self.settings:       Settings          | None = None


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
        embedding_model = settings.embedding_model,
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

    # -- MemoryManager -------------------------------------------------------
    #
    # The embed_fn passed to MemoryManager is runtime.embed when the runtime
    # has an embedding model configured and reachable.  If embed() would raise
    # NotImplementedError (oMLX without an embedding model loaded) or if the
    # runtime is unreachable, we pass None so MemoryManager falls back to
    # keyword-only retrieval without surfacing errors at startup.

    embed_fn = None
    if settings.memory_embed and health.get("embed_model_found"):
        embed_fn = runtime.embed
        logger.info("MemoryManager: embedding enabled via %s.", settings.runtime_backend)
    else:
        logger.info(
            "MemoryManager: embedding disabled at startup "
            "(memory_embed=%s  embed_model_found=%s).",
            settings.memory_embed,
            health.get("embed_model_found"),
        )

    memory_manager = MemoryManager(db_path=memory_db, embed_fn=embed_fn)
    _state.memory_manager = memory_manager

    # Seed the document index from disk on startup.  index_directory() is
    # idempotent — unchanged files are skipped via content-hash comparison.
    # This ensures the index is always current even after pages were written
    # while the server was down.
    if wiki_dir.exists():
        n_wiki = memory_manager.index_directory(wiki_dir, doc_type="wiki", embed=settings.memory_embed)
        logger.info("MemoryManager: indexed %d wiki pages from %s.", n_wiki, wiki_dir)
    else:
        logger.warning("MemoryManager: wiki_dir does not exist yet (%s) — skipping seed.", wiki_dir)

    if raw_dir.exists():
        n_raw = memory_manager.index_directory(raw_dir, doc_type="raw", embed=settings.memory_embed)
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
    research_agent = ResearchAgent(
        runtime        = runtime,
        project_root   = project_root,
        memory_manager = memory_manager,
    )

    # -- Store resolved paths in state so endpoints can inject them ----------

    _state.wiki_dir      = wiki_dir
    _state.raw_dir       = raw_dir
    _state.schema_path   = schema_path
    _state.templates_dir = templates_dir

    # -- Controller ----------------------------------------------------------

    controller = ControllerAgent(
        runtime        = runtime,
        agents         = [wiki_agent, research_agent],
        memory_manager = memory_manager,
    )
    _state.controller = controller

    logger.info(
        "ControllerAgent ready — agents: %s",
        [wiki_agent.name, research_agent.name],
    )

    yield  # — application runs —

    logger.info("LORA backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LORA — Local Reasoning Agent",
    description = "Multi-agent research system with persistent SQLite memory.",
    version     = "0.2.0",
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

    if hasattr(_state, "wiki_dir") and _state.wiki_dir:
        defaults["wiki_dir"] = str(_state.wiki_dir)
    if hasattr(_state, "raw_dir") and _state.raw_dir:
        defaults["raw_dir"] = str(_state.raw_dir)
    if hasattr(_state, "schema_path") and _state.schema_path:
        defaults["schema_path"] = str(_state.schema_path)
    if hasattr(_state, "templates_dir") and _state.templates_dir:
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

    return HealthResponse(
        healthy           = bool(raw.get("reachable") and raw.get("chat_model_found")),
        reachable         = bool(raw.get("reachable", False)),
        base_url          = str(raw.get("base_url", "")),
        models            = raw.get("models", []),
        chat_model_found  = bool(raw.get("chat_model_found", False)),
        embed_model_found = bool(raw.get("embed_model_found") or False),
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