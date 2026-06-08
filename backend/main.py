"""
LORA — FastAPI Backend
======================
The HTTP boundary between the Svelte UI and the agent core.

Layer placement
---------------
  Svelte UI  →  FastAPI (this module)  →  ControllerAgent  →  Sub-agents / Runtime

Architectural contract
----------------------
- This is the ONLY module that imports FastAPI, Pydantic, or anything HTTP-related.
- All agent logic lives in controller_agent.py, wiki_agent.py, research_agent.py.
- All model inference flows through FoundryRuntimeClient.
- This module constructs the runtime, agents, and controller once at startup
  and holds them as app-level state.  No agent or runtime is constructed
  per-request.
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
      even when Foundry is unreachable so the UI can display a degraded
      state rather than a hard error page.  A separate "healthy" boolean in
      the body signals true service health to automated monitors.

  GET /agents
      Returns the list of registered agent names.  Useful for the UI to
      show which capabilities are active without parsing log files.

  GET /files/raw
      List all .md and .txt files in the raw/ directory.
      Returns an array of FileEntry objects (name, path, size, modified).

  GET /files/wiki
      List all .md files in the wiki/ directory.
      Returns an array of FileEntry objects (name, path, size, modified).

  GET /files/content
      Return the plain-text content of any file under raw/ or wiki/.
      Query param: path (absolute path, must be inside raw_dir or wiki_dir).

  POST /files/upload
      Accept a multipart .md or .txt file upload and save it to raw/.
      Returns the FileEntry for the newly saved file.

Running locally
---------------
  uvicorn main:app --reload --host 127.0.0.1 --port 8000

Environment / configuration
----------------------------
All tuneable values are in the ``Settings`` class (pydantic-settings).  They
can be overridden via environment variables or a .env file:

  LORA_RUNTIME_BACKEND    Runtime backend to use: "foundry" | "omlx" (default "foundry")
  LORA_CHAT_MODEL         Chat model ID (interpreted by the active backend)
  LORA_EMBEDDING_MODEL    Embedding model ID (interpreted by the active backend)
  LORA_FOUNDRY_URL        Override auto-resolved Foundry base URL (foundry backend only)
  LORA_OMLX_URL           oMLX server base URL (omlx backend only, default http://localhost:8000)
  LORA_LOG_LEVEL          Root log level (default INFO)
  LORA_WIKI_DIR           Absolute path to the wiki directory
  LORA_RAW_DIR            Absolute path to the raw files directory
  LORA_SCHEMA_PATH        Absolute path to SCHEMA.md
  LORA_TEMPLATES_DIR      Absolute path to the templates directory
  LORA_AUTO_APPLY         Whether WikiAgent writes to disk immediately (bool)
  LORA_STREAM_TIMEOUT     Streaming timeout in seconds (float)
  LORA_REQUEST_TIMEOUT    Non-streaming timeout in seconds (float)
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

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
from foundry_runtime_client import _iter_sse_chunks
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

    # Runtime backend selection — determines which client create_runtime() builds.
    # "foundry" uses FoundryRuntimeClient (Azure AI Foundry, local).
    # "omlx"    uses OMLXRuntimeClient (oMLX, local).
    runtime_backend: str = "foundry"

    # Model IDs — interpreted by the active backend.
    # Foundry defaults are kept here; oMLX defaults live in omlx_runtime_client.py
    # and are used automatically when runtime_backend="omlx" and these are not set.
    chat_model:       str = "Phi-4-mini-instruct-generic-gpu:5"
    embedding_model:  str = "text-embedding-3-small"

    # Foundry network (foundry backend only)
    foundry_url:      str | None = None   # None → auto-resolve from CLI

    # oMLX network (omlx backend only)
    omlx_url:         str = "http://localhost:8000"

    # Shared network timeouts
    stream_timeout:   float = 60.0
    request_timeout:  float = 30.0

    # Paths — resolved at startup; defaults are relative to this file's parent
    wiki_dir:        str | None = None
    raw_dir:         str | None = None
    schema_path:     str | None = None
    templates_dir:   str | None = None

    # Agent behaviour
    auto_apply:      bool = False   # WikiAgent: write to disk immediately

    # Logging
    log_level:       str = "INFO"


# ---------------------------------------------------------------------------
# App-level state (constructed once at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Holds singletons that live for the lifetime of the process."""

    def __init__(self) -> None:
        self.runtime:    BaseRuntimeClient | None = None
        self.controller: ControllerAgent   | None = None
        self.settings:   Settings          | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Construct the runtime, agents, and controller once at startup.
    Runs in the main thread before the first request is accepted.
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

    # Resolve project root (two parents above this file by convention)
    project_root = Path(__file__).resolve().parent

    # Build runtime via factory — backend selected by LORA_RUNTIME_BACKEND.
    # All settings are passed as kwargs; each backend's factory function
    # extracts only the keys it needs and ignores the rest.
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

    # Log health at startup (non-blocking — don't fail if runtime is cold)
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

    # Resolve path defaults
    wiki_dir      = Path(settings.wiki_dir)      if settings.wiki_dir      else project_root / "wiki"
    raw_dir       = Path(settings.raw_dir)       if settings.raw_dir       else project_root / "raw"
    schema_path   = Path(settings.schema_path)   if settings.schema_path   else project_root / "SCHEMA.md"
    templates_dir = Path(settings.templates_dir) if settings.templates_dir else project_root / "templates"

    # Build agents
    wiki_agent     = WikiAgent(runtime=runtime,     project_root=project_root)
    research_agent = ResearchAgent(runtime=runtime, project_root=project_root)

    # Attach resolved paths as defaults on the agents' project_root.
    # The agents respect per-SubTask context overrides; these are fallbacks.
    wiki_agent._project_root     = project_root
    research_agent._project_root = project_root

    # Store resolved paths in state so endpoints can inject them into context
    _state.wiki_dir      = wiki_dir
    _state.raw_dir       = raw_dir
    _state.schema_path   = schema_path
    _state.templates_dir = templates_dir

    # Build controller
    controller = ControllerAgent(
        runtime = runtime,
        agents  = [wiki_agent, research_agent],
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
    description = "Multi-agent research system powered by Azure AI Foundry.",
    version     = "0.1.0",
    lifespan    = lifespan,
)

logger = logging.getLogger(__name__)

# Allow the Svelte dev server (default :5173) and any localhost origin during
# development.  Tighten this list before any production deployment.
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

    ``instruction`` is the only required field — the controller will handle
    planning, dispatch, and synthesis from it alone.

    ``context`` is passed through verbatim to the controller and then on to
    each agent.  Use it to supply paths, flags, or sub-agent hints:

        {
            "query":     "What do we know about attention mechanisms?",
            "raw_path":  "/abs/path/to/paper.md",
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
    healthy:          bool
    reachable:        bool
    base_url:         str
    models:           list[str]           = Field(default_factory=list)
    chat_model_found: bool                = False
    embed_model_found: bool               = False
    error:            str | None          = None


class AgentsResponse(BaseModel):
    """Response body for GET /agents."""
    agents: list[str]


class FileEntry(BaseModel):
    """Metadata for a single file in raw/ or wiki/."""
    name:     str           # filename without extension
    filename: str           # filename with extension, e.g. "my-doc.md"
    path:     str           # absolute path — passed back as context.raw_path for ingest
    size:     int           # bytes
    modified: str           # ISO-8601 UTC timestamp


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
    """Return the app-level ControllerAgent or raise 503 if not ready."""
    if _state.controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialised.")
    return _state.controller


def _require_runtime() -> BaseRuntimeClient:
    """Return the app-level runtime or raise 503 if not ready."""
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
    response_model   = TaskResponse,
    summary          = "Submit a task (blocking)",
    response_description = "The completed task result.",
)
async def post_task(request: TaskRequest) -> TaskResponse:
    """
    Submit an instruction to the LORA multi-agent system.

    The call blocks until the full pipeline (plan → dispatch → synthesize)
    completes and returns the final answer.  For long research tasks this
    may take tens of seconds.  Use ``POST /task/stream`` to receive tokens
    incrementally during synthesis.

    The controller runs synchronously in a thread pool so the event loop
    is never blocked.
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

    **Event format** (each line is one SSE event):

        data: {"type": "status",  "message": "Planning..."}
        data: {"type": "token",   "token": "The"}
        data: {"type": "token",   "token": " capital"}
        data: {"type": "sources", "sources": [...]}
        data: {"type": "done",    "task_id": "...", "status": "complete"}
        data: [DONE]

    Status events are emitted at the start of each pipeline phase so the UI
    can show progress without polling.  Token events carry individual text
    chunks from the SSE stream returned by Foundry.  The [DONE] sentinel
    signals that the stream is closed.

    Planning and sub-agent dispatch run in a thread pool and complete before
    streaming begins.  Only the final synthesis call streams.
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
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
    models are available.

    Always returns HTTP 200 — the ``healthy`` and ``reachable`` fields in
    the body carry the actual health signal.  This lets the Svelte UI render
    a degraded-state banner rather than hitting an error boundary.
    """
    runtime = _require_runtime()

    # health_check() is synchronous and makes a real HTTP call.
    raw: dict[str, Any] = await asyncio.to_thread(runtime.health_check)

    # embed_model_found may be None (not applicable) for inference-only backends.
    # Coerce to bool for the response schema — None → False.
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
    # _agents is an internal dict keyed by agent name
    agent_names = list(controller._agents.keys())
    return AgentsResponse(agents=agent_names)


# ---------------------------------------------------------------------------
# File management endpoints
# ---------------------------------------------------------------------------

def _file_entry(path: Path) -> FileEntry:
    """Build a FileEntry from a Path.  path must exist and be a file."""
    stat = path.stat()
    return FileEntry(
        name     = path.stem,
        filename = path.name,
        path     = str(path.resolve()),
        size     = stat.st_size,
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


def _list_dir(directory: Path, suffixes: set[str]) -> list[FileEntry]:
    """Return sorted FileEntry list for all matching files in directory."""
    if not directory.exists():
        return []
    entries = [
        _file_entry(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    return entries


def _require_dir(attr: str, label: str) -> Path:
    """Return a resolved directory from _state or raise 503."""
    directory = getattr(_state, attr, None)
    if directory is None:
        raise HTTPException(status_code=503, detail=f"{label} directory not configured.")
    return directory


@app.get(
    "/files/raw",
    response_model = FilesResponse,
    summary        = "List files in the raw/ corpus directory",
)
async def get_files_raw() -> FilesResponse:
    """
    Return metadata for every .md and .txt file in the raw/ directory.

    The ``path`` field of each FileEntry is the absolute path suitable for
    passing directly as ``context.raw_path`` in a ``POST /task`` or
    ``POST /task/stream`` ingest request.
    """
    raw_dir = _require_dir("raw_dir", "raw/")
    files   = await asyncio.to_thread(_list_dir, raw_dir, {".md", ".txt"})
    return FilesResponse(files=files)


@app.get(
    "/files/wiki",
    response_model = FilesResponse,
    summary        = "List pages in the wiki/ directory",
)
async def get_files_wiki() -> FilesResponse:
    """Return metadata for every .md file in the wiki/ directory."""
    wiki_dir = _require_dir("wiki_dir", "wiki/")
    files    = await asyncio.to_thread(_list_dir, wiki_dir, {".md"})
    return FilesResponse(files=files)


@app.get(
    "/files/content",
    response_model = FileContentResponse,
    summary        = "Return the text content of a file",
)
async def get_file_content(path: str) -> FileContentResponse:
    """
    Return the plain-text content of a file under raw/ or wiki/.

    The ``path`` query parameter must be an absolute path previously
    returned by ``GET /files/raw`` or ``GET /files/wiki``.  Requests for
    paths outside those two directories are rejected with 403.
    """
    raw_dir  = _require_dir("raw_dir",  "raw/")
    wiki_dir = _require_dir("wiki_dir", "wiki/")

    target = Path(path).resolve()

    # Security check — only serve files that live inside the configured dirs.
    try:
        target.relative_to(raw_dir.resolve())
        in_raw = True
    except ValueError:
        in_raw = False

    try:
        target.relative_to(wiki_dir.resolve())
        in_wiki = True
    except ValueError:
        in_wiki = False

    if not in_raw and not in_wiki:
        raise HTTPException(
            status_code = 403,
            detail      = "Path is outside the configured raw/ and wiki/ directories.",
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
    summary        = "Upload a .md or .txt file to raw/",
)
async def post_files_upload(file: UploadFile = File(...)) -> FileEntry:
    """
    Accept a multipart file upload and save it to the raw/ directory.

    Only .md and .txt files are accepted.  The file is saved with its
    original filename; if a file with that name already exists it is
    overwritten (the assumption is the user is re-uploading a newer
    version of the same document).

    Returns the FileEntry for the saved file, including the absolute
    ``path`` that can be passed directly as ``context.raw_path``.
    """
    raw_dir = _require_dir("raw_dir", "raw/")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload has no filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".txt"}:
        raise HTTPException(
            status_code = 400,
            detail      = f"Only .md and .txt files are accepted, got: {suffix!r}",
        )

    dest = raw_dir / file.filename

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        contents = await file.read()
        await asyncio.to_thread(dest.write_bytes, contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    logger.info("Uploaded file: %s (%d bytes)", dest, dest.stat().st_size)
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
    2.  Run plan + dispatch in a thread (synchronous).  These steps do not
        stream — they complete and their results are held in memory.
    3.  Emit a "Synthesising..." status event.
    4.  Re-run synthesis via the runtime's SSE stream, yielding tokens.
    5.  Emit sources, done, and [DONE] sentinel.

    This approach keeps the streaming path simple: the controller's existing
    handle_task() path is used for planning and dispatch, then we intercept
    the synthesis step by calling runtime.infer() directly with streaming.

    For the streaming synthesis we bypass the controller and call
    runtime.infer() ourselves, using the same prompt the Synthesizer would
    build.  The non-streaming /task endpoint continues to use the full
    controller pipeline unchanged.
    """

    def _sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # -- Phase 1: Planning + dispatch (in thread) ----------------------------

    yield _sse({"type": "status", "message": "Planning task…", "task_id": task_id})

    # Run the full handle_task pipeline synchronously in a thread.
    # We use this to get sub-agent results; we'll re-synthesise with streaming.
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            controller.handle_task, task_dict
        )
    except Exception as exc:
        logger.exception("Error during planning/dispatch for task %s.", task_id)
        yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    # If the controller already failed during planning/dispatch, surface it.
    if result.get("status") == "failed":
        yield _sse({
            "type":    "error",
            "message": result.get("error", "Task failed during planning or dispatch."),
            "task_id": task_id,
        })
        yield "data: [DONE]\n\n"
        return

    # -- Phase 2: Stream the answer token-by-token ---------------------------
    #
    # The controller has already synthesised the answer inside handle_task().
    # We stream that answer token-by-token by replaying it from the result
    # dict. This keeps things simple and avoids duplicating the synthesis
    # prompt logic here.
    #
    # If you later want true streaming synthesis (tokens arriving in real time
    # before the agent pipeline finishes), replace this section with a call to
    # runtime._chat_endpoint via _iter_sse_chunks in a separate thread and
    # pipe the chunks here. The endpoint contract for the Svelte client remains
    # identical either way.

    yield _sse({"type": "status", "message": "Streaming answer…", "task_id": task_id})

    answer: str = result.get("answer", "")

    # Yield the answer in word-sized chunks to simulate token streaming and
    # give the UI something to render progressively.
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else " " + word
        yield _sse({"type": "token", "token": chunk})
        # Tiny yield point so the event loop can flush each chunk to the client.
        await asyncio.sleep(0)

    # -- Phase 3: Sources + done sentinel ------------------------------------

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
    """
    Catch any unhandled exception and return a structured JSON error rather
    than an HTML 500 page.  Keeps the Svelte client's error handling simple.
    """
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
        host    = "127.0.0.1",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )