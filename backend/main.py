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

  POST /memory/reembed
      Manually re-embed the wiki/raw corpus with the active embedding model
      and clear MemoryManager's corpus_stale flag (docs/architecture/
      16-runtime-backend-layer.md §16.4). Idempotent; safe to call whether
      or not the corpus is currently stale.

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

  LOCALIST_RUNTIME_BACKEND             Runtime backend: "foundry" | "omlx" (default "foundry")
  LOCALIST_CHAT_MODEL                  Chat model ID override (wins over any per-backend pin below)
  LOCALIST_CHAT_MODEL_OMLX             Per-backend chat model pin for "omlx"
  LOCALIST_CHAT_MODEL_OLLAMA           Per-backend chat model pin for "ollama"
  LOCALIST_CHAT_MODEL_FOUNDRY          Per-backend chat model pin for "foundry"
  LOCALIST_FOUNDRY_URL                 Override auto-resolved Foundry base URL (foundry only)
  LOCALIST_OMLX_URL                    oMLX server base URL (omlx only, default http://localhost:8000)
  LOCALIST_OLLAMA_URL                  Ollama server base URL (ollama only, default http://localhost:11434)
  LOCALIST_LOG_LEVEL                   Root log level (default INFO)
  LOCALIST_WIKI_DIR                    Absolute path to the wiki directory
  LOCALIST_RAW_DIR                     Absolute path to the raw files directory
  LOCALIST_GENERATED_DIR               Absolute path to the generated files directory
  LOCALIST_SCHEMA_PATH                 Absolute path to SCHEMA.md
  LOCALIST_TEMPLATES_DIR               Absolute path to the templates directory
  LOCALIST_AUTO_APPLY                  Whether WikiAgent writes to disk immediately (bool)
  LOCALIST_STREAM_TIMEOUT              Streaming timeout in seconds (float)
  LOCALIST_REQUEST_TIMEOUT             Non-streaming timeout in seconds (float)
  LOCALIST_MEMORY_DB                   Absolute path to the SQLite memory DB file.
                                       Defaults to <project_root>/localist_memory.db
  LOCALIST_EMBEDDING_MODEL             Runtime-backend embedding model ID (foundry/ollama
                                       only; omlx does not yet wire this through). Empty
                                       string (default) = not configured, falls back to
                                       EmbeddingEngine.
  LOCALIST_EMBEDDING_ENGINE_ENABLED    Load the standalone MLX-LM EmbeddingEngine at
                                       startup (bool, default True).  Set False to run
                                       in keyword-only mode without loading the model.
                                       Ignored when LOCALIST_EMBEDDING_MODEL is set and
                                       found by the active runtime backend — the runtime
                                       backend's own embed() is used instead.  Also a
                                       no-op on non-Apple-Silicon platforms regardless of
                                       its value, since EmbeddingEngine's mlx_lm
                                       dependency only runs on Apple Silicon.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import datetime
import json
import logging
import mimetypes
import os
import platform
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project imports (agent core + runtime)
# ---------------------------------------------------------------------------

from base_runtime_client import BaseRuntimeClient
from build_graph import build_graph
from context_profile import check_local_ram_headroom, profile_for
from controller_agent import ControllerAgent, TaskStatus, _MEMORY_MD_PATH
from conversational_agent import ConversationalAgent
from embedding_engine import EmbeddingEngine
from memory_manager import MemoryManager, EpisodicMemoryWriter
from runtime_factory import available_backends, create_runtime
import session_files
import wiki_maintenance_log
from warmup import run_cache_warmup as _run_cache_warmup
from wiki_agent import WikiAgent, read_text_file, sweep_expired_snapshots
from wiki_doc import META_WIKI_FILENAMES

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    All configuration for the Localist Framework backend.
    Override any field via environment variable or .env file.
    """
    model_config = SettingsConfigDict(env_prefix="LOCALIST_", env_file=".env", extra="ignore")

    # Runtime backend selection
    runtime_backend: str = "foundry"

    # Model ID — chat only; embeddings are handled by EmbeddingEngine (MLX-LM)
    # unless embedding_model below is set and found by the active runtime
    # backend, in which case the runtime backend's own embed() is used instead.
    chat_model:       str | None = None

    # Per-backend chat-model pins (LOCALIST_CHAT_MODEL_OMLX / _OLLAMA / _FOUNDRY).
    # Used by _resolve_chat_model() when chat_model above is unset; lets a
    # live runtime-backend switch remember which model to use per backend
    # instead of carrying one backend's model id into another's client.
    chat_model_omlx:    str | None = None
    chat_model_ollama:  str | None = None
    chat_model_foundry: str | None = None

    # Runtime-backend embedding model ID (foundry/ollama only; empty string =
    # not configured, falls back to EmbeddingEngine).
    embedding_model:  str = ""

    # Foundry network (foundry backend only)
    foundry_url:      str | None = None

    # oMLX network (omlx backend only)
    omlx_url:         str = "http://localhost:8000"

    # Ollama network (ollama backend only)
    ollama_url:       str = "http://localhost:11434"

    # Shared network timeouts
    stream_timeout:   float = 60.0
    request_timeout:  float = 30.0

    # Paths — resolved at startup; defaults are relative to project root
    wiki_dir:        str | None = None
    raw_dir:         str | None = None
    generated_dir:   str | None = None
    schema_path:     str | None = None
    templates_dir:   str | None = None

    # MemoryManager
    memory_db:                str | None = None   # None → <project_root>/localist_memory.db

    # EmbeddingEngine — standalone MLX-LM embedding, backend-agnostic.
    # Set False to skip model load and run MemoryManager in keyword-only mode.
    embedding_engine_enabled: bool = True

    # Agent behaviour
    auto_apply:      bool = False

    # Episodic memory — write-approval gate for model_extracted (implicit)
    # episodes. When True, those writes are staged as "pending" instead of
    # going live immediately, and must be approved via
    # POST /memory/episodes/{id}/approve (or rejected via .../reject).
    # Explicit (user-said "remember that...") episodes are never gated.
    episodic_write_approval: bool = False

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
        self.wiki_agent:        WikiAgent         | None = None
        self.memory_manager:    MemoryManager     | None = None
        self.embedding_engine:  EmbeddingEngine   | None = None
        self.settings:          Settings          | None = None
        # Name of the embedding model actually backing embed_fn, resolved by
        # _derive_active_embedding_model_name() alongside _configure_embedding_
        # source() — see planner.py's _TUNED_EMBEDDING_MODEL guard. None means
        # keyword-only mode (no embedding source at all).
        self.active_embedding_model_name: str | None = None
        # Resolved at startup by lifespan()
        self.wiki_dir:          Path | None = None
        self.raw_dir:           Path | None = None
        self.generated_dir:     Path | None = None
        self.schema_path:       Path | None = None
        self.templates_dir:     Path | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Embedding source selection (pulled out of lifespan() for testability)
# ---------------------------------------------------------------------------

def _configure_embedding_source(
    settings: Settings,
    runtime:  BaseRuntimeClient,
    health:   dict,
) -> tuple[Any, EmbeddingEngine | None]:
    """
    Decide and construct which embedding source lifespan() should use, in
    three-tier precedence order:

      1. The active runtime backend's own embed() — used when
         settings.embedding_model is set AND health["embed_model_found"]
         (from the health check already run in lifespan()) is truthy.
         Platform-agnostic.
      2. EmbeddingEngine (standalone MLX-LM) — attempted only when enabled
         AND this platform is Apple Silicon, since mlx_lm cannot run
         elsewhere.
      3. Neither — MemoryManager falls back to keyword-only retrieval.

    Tiers 1 and 2 are mutually exclusive: loading both would hold two
    embedding models in memory for no benefit.

    Returns
    -------
    tuple[Callable | None, EmbeddingEngine | None]
        (embed_fn, embedding_engine). embedding_engine is the constructed
        EmbeddingEngine instance whenever tier 2 was attempted (whether or
        not it ended up available), so the caller can stash it on app
        state; otherwise None.

    This is a standalone function (not inlined in lifespan()) purely so the
    branch selection can be unit-tested without running the full startup
    sequence (real runtime construction, directory indexing, graph build,
    etc.) — lifespan() itself is not exercised by the test suite.
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")

    if settings.embedding_model and health.get("embed_model_found"):
        logger.info(
            "Runtime-backend embeddings ready — backend=%s model=%s. "
            "EmbeddingEngine will not be loaded.",
            settings.runtime_backend.upper(), settings.embedding_model,
        )
        return runtime.embed, None

    if settings.embedding_engine_enabled and is_apple_silicon:
        embedding_engine = EmbeddingEngine()
        if embedding_engine.available:
            logger.info("EmbeddingEngine ready — embeddings enabled.")
            return embedding_engine.embed, embedding_engine
        logger.warning(
            "EmbeddingEngine failed to load — MemoryManager will run "
            "in keyword-only mode.  Install mlx-lm and retry."
        )
        return None, embedding_engine

    if settings.embedding_engine_enabled and not is_apple_silicon:
        logger.info(
            "EmbeddingEngine skipped — mlx_lm requires Apple Silicon, this platform "
            "is %s/%s. MemoryManager will run in keyword-only mode.",
            platform.system(), platform.machine(),
        )
        return None, None

    logger.info(
        "EmbeddingEngine disabled (LOCALIST_EMBEDDING_ENGINE_ENABLED=false) — "
        "MemoryManager will run in keyword-only mode."
    )
    return None, None


def _derive_active_embedding_model_name(
    settings:         Settings,
    embed_fn:         Any,
    embedding_engine: EmbeddingEngine | None,
) -> str | None:
    """
    Name the embedding model actually backing `embed_fn`, mirroring
    _configure_embedding_source()'s three-tier precedence:

      1. Runtime-backend embed (embedding_engine is None, embed_fn is not)
         -> settings.embedding_model.
      2. EmbeddingEngine (embedding_engine is not None) -> its model_path,
         but only if it loaded successfully (embedding_engine.available);
         a construction attempt that failed to load names no model.
      3. Keyword-only (embed_fn is None, embedding_engine is None) -> None.

    Consumed by Planner's _TUNED_EMBEDDING_MODEL guard (docs/architecture/
    16-runtime-backend-layer.md §16.4) so a mismatched embedding model
    disables semantic gating instead of silently producing thresholds with
    no validated meaning.
    """
    if embedding_engine is not None:
        return embedding_engine.model_path if embedding_engine.available else None
    if embed_fn is not None:
        return settings.embedding_model
    return None


# ---------------------------------------------------------------------------
# Controller construction (extracted so lifespan() and a live runtime-backend
# switch share one code path — see _build_controller() below).
# ---------------------------------------------------------------------------

def _build_controller(
    settings:              Settings,
    runtime:               BaseRuntimeClient,
    memory_manager:        MemoryManager,
    embed_fn:              Any,
    project_root:          Path,
    templates_dir:         Path,
    embedding_model_name:  str | None = None,
) -> tuple[WikiAgent, ConversationalAgent, ControllerAgent]:
    """
    Construct WikiAgent, ConversationalAgent, and ControllerAgent for a given
    runtime, and warm its persona cache. Used both at startup (lifespan())
    and by a live runtime-backend switch, since ControllerAgent captures its
    runtime by value (via its own Synthesizer/_RulePlanner) — there is no way
    to rebind an existing ControllerAgent to a new runtime, only to build a
    fresh one.
    """
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
    controller = ControllerAgent(
        runtime                 = runtime,
        agents                  = [wiki_agent, conversational_agent],
        memory_manager          = memory_manager,
        embed_fn                = embed_fn,
        embedding_model_name    = embedding_model_name,
        episodic_write_approval = settings.episodic_write_approval,
    )
    _run_cache_warmup(controller, runtime, templates_dir)
    return wiki_agent, conversational_agent, controller


# ---------------------------------------------------------------------------
# Live runtime-backend switching support
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent

# Serializes any sequence that reads-then-swaps _state.runtime/wiki_agent/
# controller, so two concurrent switch/pin requests can't interleave their
# health-check → rebuild → swap steps. An asyncio.Lock, not threading.Lock —
# held across `await asyncio.to_thread(...)` below, and a plain threading.Lock
# there would block the whole event loop on a second requester's acquire
# instead of just queuing it behind the first (docs/architecture/
# 16-runtime-backend-layer.md §16.5).
_runtime_switch_lock = asyncio.Lock()

_CHAT_MODEL_SETTINGS_FIELD: dict[str, str] = {
    "omlx":    "chat_model_omlx",
    "ollama":  "chat_model_ollama",
    "foundry": "chat_model_foundry",
}

_CHAT_MODEL_ENV_KEY: dict[str, str] = {
    "omlx":    "LOCALIST_CHAT_MODEL_OMLX",
    "ollama":  "LOCALIST_CHAT_MODEL_OLLAMA",
    "foundry": "LOCALIST_CHAT_MODEL_FOUNDRY",
}


def _resolve_chat_model(settings: Settings, backend: str) -> str | None:
    """
    Resolve the chat model to use for `backend`, one source of truth shared
    by lifespan() and the runtime-backend endpoints.

    Precedence: settings.chat_model (global override) > the per-backend pin
    for `backend` > None (falls through to runtime_factory.py's own
    hardcoded per-backend default).
    """
    if settings.chat_model:
        return settings.chat_model
    field = _CHAT_MODEL_SETTINGS_FIELD.get(backend.strip().lower())
    return getattr(settings, field) if field else None


def _write_env_var(project_root: Path, key: str, value: str) -> None:
    """
    Set `key=value` in project_root/.env, preserving every other line
    (comments, blank lines, unrelated keys) byte-for-byte. Replaces the
    existing `key=...` line if present, otherwise appends a new one.
    Written atomically via a temp file + os.replace() so a crash or
    concurrent read never observes a half-written .env.
    """
    env_path = project_root / ".env"
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text().splitlines(keepends=True)

    new_line = f"{key}={value}\n"
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#") and stripped.split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line)

    fd, tmp_path = tempfile.mkstemp(dir=str(project_root), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        os.replace(tmp_path, env_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


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

    project_root = _PROJECT_ROOT

    # -- Runtime -------------------------------------------------------------

    runtime = create_runtime(
        backend         = settings.runtime_backend,
        chat_model      = _resolve_chat_model(settings, settings.runtime_backend),
        embedding_model = settings.embedding_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        ollama_url      = settings.ollama_url,
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

    if getattr(runtime, "is_local", True):
        local_profile = profile_for(runtime)
        logger.info(
            "LOCAL_PROFILE working_memory_tokens=%d (max_model_len=%s)",
            local_profile.working_memory_tokens,
            getattr(runtime, "max_model_len", "unknown"),
        )
        ram = check_local_ram_headroom()
        if ram["warning"]:
            logger.warning(
                "LOCAL_PROFILE RAM headroom check: %s — this machine's current "
                "load matches the swap-under-load condition measured 2026-07-19 "
                "(see diagnostics/reports/local_working_memory_ram_findings.md); "
                "expect swap activity under the working-memory ceiling.",
                ram["message"],
            )
        else:
            logger.info("LOCAL_PROFILE RAM headroom check: %s", ram["message"])

    # -- Resolve path defaults -----------------------------------------------

    wiki_dir      = Path(settings.wiki_dir)      if settings.wiki_dir      else project_root / "wiki"
    raw_dir       = Path(settings.raw_dir)       if settings.raw_dir       else project_root / "raw"
    generated_dir = Path(settings.generated_dir) if settings.generated_dir else project_root / "generated_files"
    schema_path   = Path(settings.schema_path)   if settings.schema_path   else project_root / "SCHEMA.md"
    templates_dir = Path(settings.templates_dir) if settings.templates_dir else project_root / "templates"
    memory_db     = Path(settings.memory_db)     if settings.memory_db     else project_root / "localist_memory.db"

    # -- Embedding source selection -------------------------------------------
    # See _configure_embedding_source() for the three-tier precedence rules
    # (runtime-backend embed / EmbeddingEngine-if-Apple-Silicon / keyword-only).

    embed_fn, embedding_engine = _configure_embedding_source(settings, runtime, health)
    if embedding_engine is not None:
        _state.embedding_engine = embedding_engine
    _state.active_embedding_model_name = _derive_active_embedding_model_name(
        settings, embed_fn, embedding_engine,
    )

    memory_manager = MemoryManager(
        db_path               = memory_db,
        embed_fn              = embed_fn,
        embedding_model_name  = _state.active_embedding_model_name,
    )
    _state.memory_manager = memory_manager

    # Seed the document index from disk on startup.  index_directory() is
    # idempotent — unchanged files are skipped via content-hash comparison.
    # This ensures the index is always current even after pages were written
    # while the server was down.
    if wiki_dir.exists():
        n_wiki = memory_manager.index_directory(
            wiki_dir, doc_type="wiki", embed=bool(embed_fn), exclude=META_WIKI_FILENAMES,
        )
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

    if wiki_dir.exists():
        try:
            graph_summary = build_graph(wiki_dir, memory_manager)
            logger.info(
                "Graph rebuilt at startup — nodes=%d edges=%d resolved=%d unresolved=%d",
                graph_summary["nodes"], graph_summary["edges"],
                graph_summary["resolved"], graph_summary["unresolved"],
            )
        except Exception as exc:
            logger.warning("Graph build failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Graph build skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    if wiki_dir.exists():
        try:
            reconcile_summary = memory_manager.reconcile_wiki(wiki_dir)
            logger.info(
                "Wiki reconciled at startup — reindexed=%d orphans_removed=%d%s",
                reconcile_summary["reindexed"],
                reconcile_summary["orphans_removed"],
                f" ({', '.join(reconcile_summary['orphan_names'])})"
                    if reconcile_summary["orphan_names"] else "",
            )
        except Exception as exc:
            logger.warning("Wiki reconciliation failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Wiki reconciliation skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    if wiki_dir.exists():
        try:
            pruned = sweep_expired_snapshots(wiki_dir)
            for p in pruned:
                wiki_maintenance_log.log_snapshot_pruned(p.name, str(p))
            logger.info("Wiki snapshot TTL sweep at startup — pruned=%d", len(pruned))
        except Exception as exc:
            logger.warning("Wiki snapshot TTL sweep failed at startup (non-fatal): %s", exc)
    else:
        logger.info("Wiki snapshot TTL sweep skipped — wiki_dir does not exist yet (%s).", wiki_dir)

    # -- Store resolved paths in state so endpoints can inject them ----------

    _state.wiki_dir      = wiki_dir
    _state.raw_dir       = raw_dir
    _state.generated_dir = generated_dir
    _state.schema_path   = schema_path
    _state.templates_dir = templates_dir

    # -- Agents + Controller --------------------------------------------------

    logger.info(
        "Episodic write-approval gate: %s",
        "ON" if settings.episodic_write_approval else "OFF",
    )

    wiki_agent, conversational_agent, controller = _build_controller(
        settings, runtime, memory_manager, embed_fn, project_root, templates_dir,
        _state.active_embedding_model_name,
    )
    _state.wiki_agent = wiki_agent
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
    task_id:            str              = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction:        str              = Field(..., min_length=1)
    context:            dict[str, Any]   = Field(default_factory=dict)
    metadata:           dict[str, Any]   = Field(default_factory=dict)
    conversation_id:    str              = Field(..., min_length=1)
    conversation_title: str | None       = Field(default=None)


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
    backend:           str
    base_url:          str
    models:            list[str]  = Field(default_factory=list)
    chat_model_found:  bool       = False
    embed_model_found: bool       = False
    error:             str | None = None


class AgentsResponse(BaseModel):
    """Response body for GET /agents."""
    agents: list[str]


class RuntimeBackendSwitchRequest(BaseModel):
    """
    Payload accepted by POST /settings/runtime-backend.

    chat_model is optional — if provided, it also becomes that backend's
    persisted pin (see _resolve_chat_model()), not a one-shot override.
    """
    backend:    str
    chat_model: str | None = None


class RuntimeBackendSwitchResponse(BaseModel):
    """Response body for POST /settings/runtime-backend."""
    backend:          str
    chat_model:       str | None = None
    persisted:        bool
    reachable:        bool
    base_url:         str
    models:           list[str]  = Field(default_factory=list)
    chat_model_found: bool       = False
    error:            str | None = None
    warning:          str | None = None


class RuntimeBackendModelsResponse(BaseModel):
    """Response body for GET /settings/runtime-backend/{backend}/models."""
    reachable:        bool
    base_url:         str
    models:           list[str]  = Field(default_factory=list)
    chat_model_found: bool       = False
    error:            str | None = None


class ChatModelPinRequest(BaseModel):
    """Payload accepted by POST /settings/runtime-backend/{backend}/chat-model."""
    chat_model: str = Field(..., min_length=1)


class ChatModelPinResponse(BaseModel):
    """Response body for POST /settings/runtime-backend/{backend}/chat-model."""
    backend:      str
    chat_model:   str
    persisted:    bool
    applied_live: bool


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
    corpus_stale:     bool   # True when the wiki/raw corpus needs POST /memory/reembed
    available:        bool   # False when MemoryManager is not initialised


class ReembedCorpusResponse(BaseModel):
    """
    Response body for POST /memory/reembed — the manual, explicitly-
    triggered wiki/raw corpus re-embed (docs/architecture/
    16-runtime-backend-layer.md §16.4's confirmed embedding-provenance
    follow-up). Episodes get no equivalent endpoint — a genuine embedding-
    model mismatch there is auto-corrected at startup, not left pending.
    """
    reembedded: int
    total:      int
    model:      str | None


class EpisodeItem(BaseModel):
    """A single episode record returned by GET /memory/episodes."""
    id:              int
    episode_type:    str
    subject:         str
    content:         str
    confidence:      float
    source:          str
    task_id:         str | None = None
    project_context: str | None = None
    status:          str
    created_at:      float
    last_accessed:   float | None = None

class EpisodesResponse(BaseModel):
    """Response body for GET /memory/episodes."""
    episodes: list[EpisodeItem]
    total:    int
    offset:   int
    limit:    int


class EpisodeApprovalResponse(BaseModel):
    """
    Response body for POST /memory/episodes/{id}/approve and .../reject.

    updated=False (rather than a 404/409) means the id doesn't exist or
    wasn't in "pending" status (already resolved, or never staged) —
    kept idempotent and simple for a single-user local app.
    """
    episode_id: int
    status:     Literal["active", "retracted"]
    updated:    bool


class FileEntry(BaseModel):
    """Metadata for a single file in raw/, wiki/, or generated_files/."""
    name:     str    # stem without extension
    filename: str    # filename with extension, e.g. "my-doc.md"
    path:     str    # absolute path — passed as context.raw_path on ingest
    size:     int    # bytes
    modified: str    # ISO-8601 UTC timestamp
    type:     Literal["raw", "wiki", "generated"]


class FilesResponse(BaseModel):
    """Response body for GET /files/raw and GET /files/wiki."""
    files: list[FileEntry]


class FileContentResponse(BaseModel):
    """Response body for GET /files/content."""
    path:    str
    content: str


class FileDeleteResponse(BaseModel):
    """Response body for DELETE /files."""
    path:    str
    deleted: bool


class ChatHistorySettingsResponse(BaseModel):
    """Response body for GET/PUT /chat/history/settings."""
    eviction_preset: str | None = None


class ChatHistorySettingsRequest(BaseModel):
    """Payload accepted by PUT /chat/history/settings."""
    eviction_preset: Literal["7d", "30d", "90d", "forever"]


class ChatTurnItem(BaseModel):
    """A single chat_turns record returned by GET /chat/history."""
    id:                 int
    task_id:            str
    role:               str
    content:            str
    sources:            list[dict[str, Any]] = Field(default_factory=list)
    status_message:     str | None = None
    metadata:           dict[str, Any]       = Field(default_factory=dict)
    conversation_id:    str
    conversation_title: str | None = None
    created_at:         float


class ChatHistoryResponse(BaseModel):
    """Response body for GET /chat/history."""
    turns:  list[ChatTurnItem]
    total:  int
    offset: int
    limit:  int


class ConversationSummary(BaseModel):
    """One row per distinct conversation, for the sidebar list."""
    conversation_id:    str
    conversation_title: str | None = None
    last_created_at:    float
    first_created_at:   float


class ConversationListResponse(BaseModel):
    """Response body for GET /chat/history/conversations."""
    conversations: list[ConversationSummary]


class ApplyDiffRequest(BaseModel):
    """
    Payload accepted by POST /wiki/apply-diff.

    task_id identifies the chat turn whose persisted metadata.pending_diffs
    entry should be marked "applied" on success (see
    MemoryManager.mark_diff_applied()) — the round-tripped page_name/diff
    are what actually gets written; task_id only updates the durable
    review-then-apply UI state.
    """
    task_id:   str = Field(..., min_length=1)
    page_name: str = Field(..., min_length=1)
    diff:      str = Field(..., min_length=1)


class ApplyDiffResponse(BaseModel):
    """Response body for POST /wiki/apply-diff (success only — failures raise HTTPException)."""
    success:   bool = True
    page_name: str


class PinWikiPageRequest(BaseModel):
    """Payload accepted by POST /chat/pin-wiki-page."""
    stem: str = Field(..., min_length=1)


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


def _require_wiki_agent() -> WikiAgent:
    if _state.wiki_agent is None:
        raise HTTPException(status_code=503, detail="WikiAgent not initialised.")
    return _state.wiki_agent


def _require_memory_manager() -> MemoryManager:
    if _state.memory_manager is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")
    return _state.memory_manager


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


def _persist_chat_turn(
    role:               str,
    content:            str,
    task_id:            str,
    conversation_id:    str,
    sources:            list[dict[str, Any]] | None = None,
    status_message:     str | None = None,
    metadata:           dict[str, Any] | None = None,
    conversation_title: str | None = None,
) -> None:
    """
    Best-effort write of one chat turn to the chat_turns table.

    No-ops silently when no memory_manager is configured. Never raises —
    a chat_turns write failure must not break the actual task response,
    since the source of truth for an in-flight answer is the SSE stream /
    TaskResponse, not this table.

    Parameters
    ----------
    conversation_id :
        Groups turns by conversation. Required.
    conversation_title :
        Optional human-readable title for the conversation.
    """
    if _state.memory_manager is None:
        return
    try:
        _state.memory_manager.add_chat_turn(
            task_id            = task_id,
            role               = role,
            content            = content,
            conversation_id    = conversation_id,
            sources            = sources,
            status_message     = status_message,
            metadata           = metadata,
            conversation_title = conversation_title,
        )
    except Exception:
        logger.warning("Failed to persist chat turn (role=%s, task_id=%s).", role, task_id, exc_info=True)


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

    _persist_chat_turn(
        "user", request.instruction, request.task_id, request.conversation_id,
        conversation_title = request.conversation_title,
    )

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            controller.handle_task, task_dict
        )
    except Exception as exc:
        logger.exception("Unhandled error in POST /task for task %s.", request.task_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _persist_chat_turn(
        "assistant", result.get("answer", ""), request.task_id, request.conversation_id,
        sources  = result.get("sources"),
        metadata = result.get("metadata"),
    )

    return TaskResponse(**result)


@app.post(
    "/task/stream",
    summary = "Submit a task and stream the synthesis answer (SSE)",
)
async def post_task_stream(request: TaskRequest) -> StreamingResponse:
    """
    Submit a task and receive the synthesis answer as a Server-Sent Events stream.

    Event format:

        data: {"type": "status",        "message": "Planning..."}
        data: {"type": "token",         "token": "The"}
        data: {"type": "sources",       "sources": [...]}
        data: {"type": "done",          "task_id": "...", "status": "complete"}
        data: {"type": "task_complete", "task_id": "..."}
        data: [DONE]

    'done' fires as soon as the visible answer is ready (may precede
    background memory writes by up to ~20-30s). 'task_complete' fires only
    after the full pipeline — including post-answer episodic/working-state
    hooks — has finished, and always precedes [DONE]. Clients that submit
    a new task while a prior one's background writes are still running can
    cause overlapping calls against a single-instance local model backend,
    so the client should gate the next submission on 'task_complete', not
    'done'.
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
        _stream_task(
            controller, runtime, task_dict, request.task_id,
            request.conversation_id, request.conversation_title,
        ),
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

    # Mirrors the embed_fn precedence lifespan() establishes at startup: the
    # runtime backend's own embed() wins when an embedding_model is configured
    # and the health check confirms it's actually present, since in that case
    # lifespan() never loads EmbeddingEngine at all (_state.embedding_engine
    # stays None). Only fall back to EmbeddingEngine's own availability when
    # the runtime-backend path isn't the one actually wired to MemoryManager.
    settings = _state.settings
    if settings is not None and settings.embedding_model and raw.get("embed_model_found"):
        embed_available = True
    else:
        embedding_engine = _state.embedding_engine
        embed_available  = embedding_engine is not None and embedding_engine.available

    return HealthResponse(
        healthy           = bool(raw.get("reachable") and raw.get("chat_model_found")),
        reachable         = bool(raw.get("reachable", False)),
        backend           = settings.runtime_backend if settings is not None else "",
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


def _validate_backend_name(backend: str) -> str:
    """Normalize and reject an unknown backend name before anything is touched."""
    normalized = backend.strip().lower()
    if normalized not in available_backends():
        raise HTTPException(
            status_code = 422,
            detail      = (
                f"Unknown runtime backend: {backend!r}. "
                f"Supported backends: {', '.join(available_backends())}."
            ),
        )
    return normalized


def _require_settings() -> Settings:
    if _state.settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialised.")
    return _state.settings


def _create_and_check_backend(settings: Settings, backend: str) -> tuple[BaseRuntimeClient, dict]:
    """Build a runtime client for `backend` and health-check it. Blocking — run via asyncio.to_thread."""
    chat_model = _resolve_chat_model(settings, backend)
    candidate_runtime = create_runtime(
        backend         = backend,
        chat_model      = chat_model,
        embedding_model = settings.embedding_model,
        foundry_url     = settings.foundry_url,
        omlx_url        = settings.omlx_url,
        ollama_url      = settings.ollama_url,
        request_timeout = settings.request_timeout,
        stream_timeout  = settings.stream_timeout,
    )
    return candidate_runtime, candidate_runtime.health_check()


@app.post(
    "/settings/runtime-backend",
    response_model = RuntimeBackendSwitchResponse,
    summary        = "Live-switch the active runtime backend",
)
async def switch_runtime_backend(request: RuntimeBackendSwitchRequest) -> RuntimeBackendSwitchResponse:
    """
    Live-switch the active runtime backend. The target backend must answer
    health_check() successfully before anything is mutated — an unreachable
    target leaves the current backend running untouched.

    An optional chat_model on the request pins that backend's chat model as
    a side effect (persisted Settings field + .env), independent of whether
    the switch itself succeeds.
    """
    backend  = _validate_backend_name(request.backend)
    settings = _require_settings()

    if request.chat_model:
        setattr(settings, _CHAT_MODEL_SETTINGS_FIELD[backend], request.chat_model)
        _write_env_var(_PROJECT_ROOT, _CHAT_MODEL_ENV_KEY[backend], request.chat_model)

    chat_model = _resolve_chat_model(settings, backend)

    async with _runtime_switch_lock:
        candidate_runtime, health = await asyncio.to_thread(
            _create_and_check_backend, settings, backend,
        )

        if not health.get("reachable"):
            raise HTTPException(
                status_code = 502,
                detail      = (
                    f"Runtime backend {backend!r} is not reachable at "
                    f"{health.get('base_url')!r} — current backend left untouched."
                ),
            )

        memory_manager = _require_memory_manager()
        # Deliberately NOT re-derived from candidate_runtime — a chat-backend switch changes
        # inference only, never the embedding source. Re-coupling this would risk silently
        # dropping a working embedder when switching to a backend that doesn't wire embeddings
        # (oMLX today, per §16.4's open gap), even though nothing about embeddings was supposed
        # to change. See docs/architecture/16-runtime-backend-layer.md §16.5.
        embed_fn = memory_manager.embed_fn
        # Read fresh from _state, not captured once — the embedding source
        # itself isn't re-derived here either (see comment above), but this
        # follows the same "always resolve from _state at request time" rule
        # as _state.runtime, in case that ever changes.
        embedding_model_name = _state.active_embedding_model_name

        wiki_agent, _conversational_agent, controller = await asyncio.to_thread(
            _build_controller,
            settings, candidate_runtime, memory_manager, embed_fn,
            _PROJECT_ROOT, _state.templates_dir, embedding_model_name,
        )

        _state.runtime    = candidate_runtime
        _state.wiki_agent = wiki_agent
        _state.controller = controller
        settings.runtime_backend = backend

        persisted: bool = True
        warning:   str | None = None
        try:
            _write_env_var(_PROJECT_ROOT, "LOCALIST_RUNTIME_BACKEND", backend)
        except OSError as exc:
            persisted = False
            warning = (
                f"Runtime switched to {backend!r} in-process, but writing .env failed "
                f"({exc}) — it will revert to the previous backend on next restart."
            )

    return RuntimeBackendSwitchResponse(
        backend          = backend,
        chat_model       = chat_model,
        persisted        = persisted,
        reachable        = bool(health.get("reachable", False)),
        base_url         = str(health.get("base_url", "")),
        models           = health.get("models", []),
        chat_model_found = bool(health.get("chat_model_found", False)),
        error            = health.get("error"),
        warning          = warning,
    )


@app.get(
    "/settings/runtime-backend/{backend}/models",
    response_model = RuntimeBackendModelsResponse,
    summary        = "List models available on a runtime backend without switching to it",
)
async def get_runtime_backend_models(backend: str) -> RuntimeBackendModelsResponse:
    """
    Build a throwaway client for `backend`, health-check it, and return its
    reported models. Never touches _state — not even a read — so this is
    safe to call for the currently-inactive backend(s) as a "what's
    available there" lookup.
    """
    normalized = _validate_backend_name(backend)
    settings   = _require_settings()

    _candidate_runtime, health = await asyncio.to_thread(
        _create_and_check_backend, settings, normalized,
    )

    return RuntimeBackendModelsResponse(
        reachable        = bool(health.get("reachable", False)),
        base_url         = str(health.get("base_url", "")),
        models           = health.get("models", []),
        chat_model_found = bool(health.get("chat_model_found", False)),
        error            = health.get("error"),
    )


@app.post(
    "/settings/runtime-backend/{backend}/chat-model",
    response_model = ChatModelPinResponse,
    summary        = "Pin a chat model for a specific runtime backend",
)
async def set_runtime_backend_chat_model(
    backend: str, request: ChatModelPinRequest,
) -> ChatModelPinResponse:
    """
    Persist a chat-model pin for `backend`. Always writes the Settings field
    and .env, regardless of which backend is currently active. If `backend`
    is the active backend, also live-rebuilds the controller against it so
    the pin takes effect immediately rather than only on the next switch.
    """
    normalized = _validate_backend_name(backend)
    settings   = _require_settings()

    setattr(settings, _CHAT_MODEL_SETTINGS_FIELD[normalized], request.chat_model)
    _write_env_var(_PROJECT_ROOT, _CHAT_MODEL_ENV_KEY[normalized], request.chat_model)

    applied_live = False
    if normalized == settings.runtime_backend.strip().lower():
        async with _runtime_switch_lock:
            candidate_runtime, health = await asyncio.to_thread(
                _create_and_check_backend, settings, normalized,
            )

            if not health.get("reachable"):
                raise HTTPException(
                    status_code = 502,
                    detail      = (
                        f"Chat model pin saved for {normalized!r}, but it is not "
                        f"reachable at {health.get('base_url')!r} — live rebuild skipped."
                    ),
                )

            memory_manager = _require_memory_manager()
            # Deliberately NOT re-derived from candidate_runtime — a chat-backend switch changes
            # inference only, never the embedding source. Re-coupling this would risk silently
            # dropping a working embedder when switching to a backend that doesn't wire embeddings
            # (oMLX today, per §16.4's open gap), even though nothing about embeddings was supposed
            # to change. See docs/architecture/16-runtime-backend-layer.md §16.5.
            embed_fn = memory_manager.embed_fn
            # Read fresh from _state, not captured once — the embedding source
            # itself isn't re-derived here either (see comment above), but this
            # follows the same "always resolve from _state at request time" rule
            # as _state.runtime, in case that ever changes.
            embedding_model_name = _state.active_embedding_model_name

            wiki_agent, _conversational_agent, controller = await asyncio.to_thread(
                _build_controller,
                settings, candidate_runtime, memory_manager, embed_fn,
                _PROJECT_ROOT, _state.templates_dir, embedding_model_name,
            )

            _state.runtime    = candidate_runtime
            _state.wiki_agent = wiki_agent
            _state.controller = controller
            applied_live = True

    return ChatModelPinResponse(
        backend      = normalized,
        chat_model   = request.chat_model,
        persisted    = True,
        applied_live = applied_live,
    )


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
            corpus_stale   = False,
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
        corpus_stale   = mm._corpus_stale,
        available      = True,
    )


@app.post(
    "/memory/reembed",
    response_model = ReembedCorpusResponse,
    summary        = "Manually re-embed the wiki/raw corpus with the active embedding model",
)
async def reembed_corpus() -> ReembedCorpusResponse:
    """
    Explicit, manually-triggered corpus re-embed — the counterpart to
    episodes' automatic startup re-embed (docs/architecture/
    16-runtime-backend-layer.md §16.4). Wiki/raw corpora can be arbitrarily
    large, so unlike episodes a detected embedding-model mismatch never
    triggers this automatically; call it after switching embedding models
    to clear MemoryManager's corpus_stale flag and restore embedding-based
    re-ranking in query_corpus().

    Idempotent — safe to call whether or not the corpus is currently
    flagged stale (a "just refresh it" operation).
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")
    if mm.embed_fn is None:
        raise HTTPException(
            status_code = 409,
            detail      = "No embedding source configured — nothing to re-embed with.",
        )

    result = await asyncio.to_thread(mm.reembed_corpus)
    return ReembedCorpusResponse(**result)


@app.get(
    "/memory/episodes",
    response_model = EpisodesResponse,
    summary        = "List stored episodes",
)
async def get_memory_episodes(
    status:          str      = "active",
    project_context: str | None = None,
    episode_type:    str | None = None,
    limit:           int      = 50,
    offset:          int      = 0,
) -> EpisodesResponse:
    """
    Return a paginated list of episodes from the episodic memory store.

    Query parameters
    ----------------
    status          : "active" (default) | "pending" | "superseded" |
                      "retracted" | "all". "pending" surfaces episodes
                      staged by the episodic_write_approval gate awaiting
                      POST /memory/episodes/{id}/approve or .../reject.
    project_context : filter by project context string
    episode_type    : filter by episode type
    limit           : max results (default 50, max 200)
    offset          : pagination offset (default 0)
    """
    mm = _state.memory_manager
    if mm is None:
        return EpisodesResponse(episodes=[], total=0, offset=offset, limit=limit)

    rows: list[dict] = await asyncio.to_thread(
        mm.list_episodes,
        status          = status,
        project_context = project_context,
        episode_type    = episode_type,
        limit           = limit,
        offset          = offset,
    )
    # Total is the full matching-row count (mm.count_episodes()), not
    # len(rows) — the latter is silently capped by `limit` and would make
    # e.g. ?status=pending&limit=1 (used for a pending-count badge) always
    # report 0 or 1 regardless of how many pending episodes actually exist.
    total: int = await asyncio.to_thread(
        mm.count_episodes,
        status          = status,
        project_context = project_context,
        episode_type    = episode_type,
    )

    return EpisodesResponse(
        episodes = [EpisodeItem(**row) for row in rows],
        total    = total,
        offset   = offset,
        limit    = limit,
    )


@app.post(
    "/memory/episodes/{episode_id}/approve",
    response_model = EpisodeApprovalResponse,
    summary        = "Approve a pending episode (write-approval gate)",
)
async def approve_memory_episode(episode_id: int) -> EpisodeApprovalResponse:
    """
    Transition a pending episode to active — the "yes" path of the
    episodic_write_approval gate. Once active, the episode is eligible for
    by_recency()/by_similarity() and appears in MEMORY.md immediately.

    Idempotent: approving an id that's already active/retracted, or that
    doesn't exist, returns updated=False rather than an error.
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")

    writer = EpisodicMemoryWriter(
        db_path=getattr(mm, "_db_path", None), memory_md_path=_MEMORY_MD_PATH,
    )
    count = await asyncio.to_thread(writer.approve, episode_id)
    return EpisodeApprovalResponse(
        episode_id = episode_id,
        status     = "active",
        updated    = count > 0,
    )


@app.post(
    "/memory/episodes/{episode_id}/reject",
    response_model = EpisodeApprovalResponse,
    summary        = "Reject a pending episode (write-approval gate)",
)
async def reject_memory_episode(episode_id: int) -> EpisodeApprovalResponse:
    """
    Transition a pending episode to retracted — the "no" path of the
    episodic_write_approval gate. A rejected episode never becomes live
    memory.

    Idempotent: rejecting an id that's already active/retracted, or that
    doesn't exist, returns updated=False rather than an error.
    """
    mm = _state.memory_manager
    if mm is None:
        raise HTTPException(status_code=503, detail="MemoryManager not initialised.")

    writer = EpisodicMemoryWriter(
        db_path=getattr(mm, "_db_path", None), memory_md_path=_MEMORY_MD_PATH,
    )
    count = await asyncio.to_thread(writer.reject, episode_id)
    return EpisodeApprovalResponse(
        episode_id = episode_id,
        status     = "retracted",
        updated    = count > 0,
    )


# ---------------------------------------------------------------------------
# File management endpoints
# ---------------------------------------------------------------------------

def _file_entry(p: "Path", type: Literal["raw", "wiki", "generated"]) -> FileEntry:
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
        type     = type,
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
        _file_entry(p, "raw")
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
    """
    Return metadata for every .md content page in the wiki/ directory.

    Excludes META_WIKI_FILENAMES (index.md, logs.md, MEMORY.md) — these are
    structural/generated files, never a page a user would pin as a diff
    target.
    """
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")
    wiki_dir = _state.wiki_dir
    if not wiki_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p, "wiki")
        for p in sorted(wiki_dir.iterdir())
        if p.is_file() and p.suffix == ".md" and p.name not in META_WIKI_FILENAMES
    ]
    return FilesResponse(files=files)


@app.get(
    "/files/generated",
    response_model = FilesResponse,
    summary        = "List generated files",
)
async def get_files_generated() -> FilesResponse:
    """Return metadata for every file in the generated_files/ directory."""
    if _state.generated_dir is None:
        raise HTTPException(status_code=503, detail="generated_dir not configured.")
    generated_dir = _state.generated_dir
    if not generated_dir.exists():
        return FilesResponse(files=[])
    files = [
        _file_entry(p, "generated")
        for p in sorted(generated_dir.iterdir())
        if p.is_file() and not p.name.startswith(".")
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
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
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


@app.get(
    "/files/download",
    summary = "Download a file",
)
async def get_file_download(path: str) -> FileResponse:
    """
    Stream a file back with a Content-Disposition: attachment header so the
    browser saves it (Safari's Downloads queue) instead of navigating to it.

    Same allowed-roots gate as /files/content, but path containment is
    checked with is_relative_to() rather than a raw string prefix — a
    prefix check would let /data/wiki_evil slip through for an allowed
    root of /data/wiki.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        path=target,
        filename=target.name,
        media_type=media_type,
    )


@app.delete(
    "/files",
    response_model = FileDeleteResponse,
    summary        = "Delete a file",
)
async def delete_file(path: str) -> FileDeleteResponse:
    """
    Delete a file from raw/, wiki/, or generated_files/ by absolute path.

    Same allowed-roots gate as /files/content and /files/download. Also
    purges any document_index row for the path — a no-op for generated
    files (never indexed), but necessary for raw/wiki so a deleted file
    doesn't linger in RAG retrieval.
    """
    if _state.raw_dir is None or _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="Directories not configured.")

    target = Path(path).resolve()
    allowed_roots = [
        _state.raw_dir.resolve(),
        _state.wiki_dir.resolve(),
    ]
    if _state.generated_dir is not None:
        allowed_roots.append(_state.generated_dir.resolve())
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside permitted directories.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        await asyncio.to_thread(target.unlink)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}") from exc

    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(_state.memory_manager.remove_document, target)
        except Exception as exc:
            logger.warning("remove_document failed for deleted file %s: %s", target, exc)

    return FileDeleteResponse(path=str(target), deleted=True)


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

    return _file_entry(dest, "raw")


# ---------------------------------------------------------------------------
# Review-then-apply wiki diffs
# ---------------------------------------------------------------------------

@app.post(
    "/wiki/apply-diff",
    response_model = ApplyDiffResponse,
    summary        = "Apply a previously-proposed wiki diff",
)
async def post_wiki_apply_diff(body: ApplyDiffRequest) -> ApplyDiffResponse:
    """
    Write a diff WikiAgent previously proposed (surfaced to the chat UI via
    a turn's metadata.pending_diffs) directly to disk — no fresh model
    call, no re-routing through the Planner.

    Content-based matching in apply_unified_diff() is the staleness check:
    if the target page changed on disk since the diff was proposed, the
    match legitimately fails and this raises 409 rather than corrupting
    the page. A missing target page raises 404.

    On success, best-effort marks the originating chat turn's persisted
    pending_diffs entry as "applied" (MemoryManager.mark_diff_applied())
    so a page reload reflects the write; failure to do so is logged but
    does not fail the request — the disk write already succeeded.
    """
    wiki_agent = _require_wiki_agent()
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")

    result = await asyncio.to_thread(
        wiki_agent.apply_pending_diff, body.page_name, body.diff, _state.wiki_dir,
    )

    if result.status != TaskStatus.COMPLETE:
        status_code = 404 if result.output.get("error_kind") == "not_found" else 409
        raise HTTPException(status_code=status_code, detail=result.error)

    if _state.memory_manager is not None:
        try:
            await asyncio.to_thread(
                _state.memory_manager.mark_diff_applied, body.task_id, body.page_name,
            )
        except Exception as exc:
            logger.warning(
                "mark_diff_applied failed for task_id=%s page_name=%s: %s",
                body.task_id, body.page_name, exc,
            )

    return ApplyDiffResponse(success=True, page_name=body.page_name)


# ---------------------------------------------------------------------------
# Chat file attachments (session-scoped, ephemeral, no wiki ingestion)
# ---------------------------------------------------------------------------

@app.post("/chat/files")
async def attach_chat_file(file: UploadFile = File(...)):
    """
    Upload a text file into the ephemeral session file cache.

    The file is read, decoded as UTF-8, and passed to session_files.add_file().
    Returns 200 + {filename, token_estimate} on success.
    Returns 400 + {detail} on rejection (type, size, or budget).
    Returns 422 on encoding failure (binary/non-UTF-8 file).
    """
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=422,
            detail=f"'{file.filename}' could not be read as UTF-8 text. Binary files are not supported.",
        )

    error = session_files.add_file(file.filename, content)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "filename":       file.filename,
        "token_estimate": len(content) // 4,
    }


@app.delete("/chat/files/{filename}")
async def detach_chat_file(filename: str):
    """
    Remove a file from the ephemeral session file cache by filename.

    Returns 200 + {removed: true} if found and removed.
    Returns 404 if the filename was not in the cache.
    """
    removed = session_files.remove_file(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in session files.")
    return {"removed": True}


@app.post("/chat/pin-wiki-page")
async def pin_wiki_page(body: PinWikiPageRequest):
    """
    Pin an existing wiki page into the ephemeral session file cache by stem.

    Reads the page straight off disk (wiki_dir/{stem}.md) rather than via
    the graph index, since the graph is only rebuilt on an explicit trigger
    and can lag behind real files. Returns 200 + {filename, token_estimate,
    source} on success. Returns 404 if no such page exists on disk.
    Returns 400 + {detail} on rejection (size or budget), same as
    POST /chat/files.
    """
    if _state.wiki_dir is None:
        raise HTTPException(status_code=503, detail="wiki_dir not configured.")

    filename = f"{body.stem}.md"
    if filename in META_WIKI_FILENAMES:
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is a structural wiki file, not a pinnable page.",
        )

    page_path = _state.wiki_dir / filename
    if not page_path.is_file():
        raise HTTPException(status_code=404, detail=f"Wiki page '{body.stem}' not found.")

    content = await asyncio.to_thread(read_text_file, page_path)
    error = session_files.add_file(filename, content, source="wiki_pin")
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {
        "filename":       filename,
        "token_estimate": len(content) // 4,
        "source":         "wiki_pin",
    }


# ---------------------------------------------------------------------------
# Chat history settings  (Chat History Tab — eviction preset only)
# ---------------------------------------------------------------------------

@app.get(
    "/chat/history/settings",
    response_model = ChatHistorySettingsResponse,
    summary        = "Read the chat history eviction preset",
)
async def get_chat_history_settings() -> ChatHistorySettingsResponse:
    """
    Return the current chat_turns eviction preset.

    ``eviction_preset`` is None when the user has never set one.
    """
    mm = _require_memory_manager()
    preset = await asyncio.to_thread(mm.get_chat_history_eviction_preset)
    return ChatHistorySettingsResponse(eviction_preset=preset)


@app.put(
    "/chat/history/settings",
    response_model = ChatHistorySettingsResponse,
    summary        = "Set the chat history eviction preset",
)
async def put_chat_history_settings(
    request: ChatHistorySettingsRequest,
) -> ChatHistorySettingsResponse:
    """
    Set the chat_turns eviction preset.

    Does not trigger an eviction sweep — this endpoint only persists the
    preference. Returns the value re-read from the database to confirm the
    write landed.
    """
    mm = _require_memory_manager()
    await asyncio.to_thread(mm.set_chat_history_eviction_preset, request.eviction_preset)
    preset = await asyncio.to_thread(mm.get_chat_history_eviction_preset)
    return ChatHistorySettingsResponse(eviction_preset=preset)


@app.get(
    "/chat/history",
    response_model = ChatHistoryResponse,
    summary        = "List chat_turns, optionally full-text filtered",
)
async def get_chat_history(
    q:               str | None = None,
    limit:           int         = 50,
    offset:          int         = 0,
    conversation_id: str | None = None,
) -> ChatHistoryResponse:
    """
    Return a paginated list of chat_turns, newest first.

    Query parameters
    ----------------
    q               : optional full-text search string (matched via chat_turns_fts)
    limit           : max results (default 50, max 200)
    offset          : pagination offset (default 0)
    conversation_id : optional conversation_id filter — when provided, restricts
                      results to one conversation; when omitted, searches/lists
                      across all conversations.

    Read-only — no eviction/deletion happens here.
    """
    mm = _require_memory_manager()
    limit = min(limit, 200)

    rows, total = await asyncio.to_thread(
        mm.get_chat_turns, query=q, limit=limit, offset=offset,
        conversation_id=conversation_id,
    )

    return ChatHistoryResponse(
        turns  = [ChatTurnItem(**row) for row in rows],
        total  = total,
        offset = offset,
        limit  = limit,
    )


@app.get(
    "/chat/history/conversations",
    response_model = ConversationListResponse,
    summary        = "List distinct conversations, newest first",
)
async def get_conversations() -> ConversationListResponse:
    """
    Return one summary row per distinct conversation_id, ordered by
    last_created_at descending — used to populate the Chat tab's
    conversation sub-list in the sidebar.

    Read-only.
    """
    mm = _require_memory_manager()
    rows = await asyncio.to_thread(mm.get_conversations)
    return ConversationListResponse(
        conversations = [ConversationSummary(**row) for row in rows]
    )


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

async def _stream_task(
    controller:         ControllerAgent,
    runtime:            BaseRuntimeClient,
    task_dict:          dict[str, Any],
    task_id:            str,
    conversation_id:    str,
    conversation_title: str | None = None,
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

    _persist_chat_turn(
        "user", task_dict["instruction"], task_id, conversation_id,
        conversation_title = conversation_title,
    )

    # Route in a separate thread — some priority branches call embed_fn / infer().
    try:
        plan = await asyncio.to_thread(
            controller.route_task,
            task_dict["instruction"],
            task_dict.get("context", {}),
        )
    except Exception as exc:
        logger.exception("Error during routing for task %s.", task_id)
        yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    yield _sse({
        "type":    "status",
        "message": f"Routed to {plan.agent}",
        "task_id": task_id,
    })

    # Execute the precomputed plan with real per-token streaming.
    #
    # Bridge design: asyncio.Queue + loop.call_soon_threadsafe
    # --------------------------------------------------------
    # ConversationalAgent.run() calls on_token(chunk) from a worker thread
    # (via asyncio.to_thread).  call_soon_threadsafe schedules a put_nowait
    # on the asyncio.Queue from that thread so we can await items on the
    # event loop side without crossing the thread boundary per-get.  This
    # avoids the overhead of wrapping every queue.get() in asyncio.to_thread.
    # For agents that never call on_token (e.g. WikiAgent), the queue stays
    # empty and the drain loop terminates immediately once the producer task
    # is done — no stall.

    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_token(chunk: str) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, {"_kind": "token", "chunk": chunk})

    def on_status(message: str) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, {"_kind": "status", "message": message})

    def on_answer_ready(result_dict: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(
            event_queue.put_nowait, {"_kind": "answer_ready", "result": result_dict}
        )

    def _drain_item(item: dict[str, Any]) -> str:
        if item["_kind"] == "status":
            return _sse({"type": "status", "message": item["message"], "task_id": task_id})
        return _sse({"type": "token", "token": item["chunk"]})

    producer_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(
        asyncio.to_thread(
            controller.handle_task_with_plan,
            task_dict,
            plan,
            on_token=on_token,
            on_status=on_status,
            on_answer_ready=on_answer_ready,
        )
    )

    # Tracks whether sources+done were already emitted via on_answer_ready.
    # When True, the post-hook [DONE] sentinel closes the stream without
    # re-emitting those events; failure events are only logged, not sent.
    answer_ready_emitted = False

    # Drain events while producer runs
    while not producer_task.done():
        try:
            item = await asyncio.wait_for(event_queue.get(), timeout=0.05)
            if item["_kind"] == "answer_ready":
                # Answer is complete — emit sources+done immediately so the
                # client unblocks before memory hooks finish.
                answer_ready_emitted = True
                rd = item["result"]
                yield _sse({"type": "sources", "sources": rd.get("sources", [])})
                yield _sse({
                    "type":     "done",
                    "task_id":  task_id,
                    "status":   rd.get("status", "complete"),
                    "metadata": rd.get("metadata", {}),
                    "answer":   rd.get("answer", ""),
                })
                _persist_chat_turn(
                    "assistant", rd.get("answer", ""), task_id, conversation_id,
                    sources  = rd.get("sources"),
                    metadata = rd.get("metadata"),
                )
            elif not answer_ready_emitted:
                # Relay token/status events only before 'done' is sent; silently
                # drop post-done hook status events (e.g. "Updating working memory…")
                # to avoid flickering the task status back to 'planning'.
                yield _drain_item(item)
        except asyncio.TimeoutError:
            pass

    # Collect result / surface exception
    try:
        result: dict[str, Any] = await producer_task
    except Exception as exc:
        if answer_ready_emitted:
            # Error occurred in post-answer hooks — answer already sent, so do
            # not attempt to emit an error event over an already-closed stream.
            logger.exception(
                "Error in post-answer hooks for task %s (answer already sent).", task_id
            )
        else:
            logger.exception("Error during planning/dispatch for task %s.", task_id)
            yield _sse({"type": "error", "message": str(exc), "task_id": task_id})
        # The pipeline (including post-answer hooks) has stopped running one
        # way or another — signal task_complete so the client can re-enable
        # input rather than waiting indefinitely.
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    # Drain any events queued between the last poll and task completion.
    # Skip answer_ready (already handled) and post-done hook events.
    while not event_queue.empty():
        item = event_queue.get_nowait()
        if item["_kind"] not in ("answer_ready",) and not answer_ready_emitted:
            yield _drain_item(item)

    if answer_ready_emitted:
        # sources+done were already sent early; the pipeline (including
        # post-answer episodic/working-state hooks) has now actually
        # finished, since we're past `await producer_task`. Signal that
        # distinctly so the client can tell "answer visible" apart from
        # "fully done" and re-enable input only now.
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    if result.get("status") == "failed":
        yield _sse({
            "type":    "error",
            "message": result.get("error", "Task failed during planning or dispatch."),
            "task_id": task_id,
        })
        yield _sse({"type": "task_complete", "task_id": task_id})
        yield "data: [DONE]\n\n"
        return

    _persist_chat_turn(
        "assistant", result.get("answer", ""), task_id, conversation_id,
        sources  = result.get("sources"),
        metadata = result.get("metadata"),
    )

    yield _sse({"type": "sources",  "sources": result.get("sources", [])})
    yield _sse({
        "type":     "done",
        "task_id":  task_id,
        "status":   result.get("status", "complete"),
        "metadata": result.get("metadata", {}),
        "answer":   result.get("answer", ""),
    })
    # No on_answer_ready path was taken (e.g. non-conversational agent) —
    # 'done' above already reflects the fully-resolved pipeline, but emit
    # task_complete too so the client's completion signal is uniform across
    # both paths.
    yield _sse({"type": "task_complete", "task_id": task_id})
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