"""
Localist MCP Server
====================
Standalone MCP (Model Context Protocol) service on port 8003.

Exposes file_op capabilities as three MCP tools — read_file, write_file,
append_file — plus fetch_url (Phase 2: ports the retired standalone Fetcher
microservice's /extract path in-process), web_search (Phase 3: ports the
LangSearch integration in-process, no runtime.infer() hallucination
fallback), and generate_chart (renders a bar/line/pie chart from structured
data server-side via matplotlib) — over SSE transport, using the official
`mcp` Python SDK's FastMCP. See backend/mcp_tool_dispatcher.py for the
dispatch seam.

Endpoints
---------
  GET  /health   — {"status": "ok"}
  GET  /sse       — MCP SSE stream (mounted from FastMCP)
  POST /messages/ — MCP message endpoint (mounted from FastMCP)

Configuration
-------------
  LOCALIST_MCP_PROJECT_ROOT   Sandbox root for file_op tools. Defaults to
                               backend/ (parent of this package) — see
                               mcp_server/file_ops.py.
  LOCALIST_LOG_LEVEL           Root log level (default INFO).
  SEARCH_PROVIDER               Which web_search provider is active:
                               "langsearch" (default) or "brave". See
                               mcp_server/web_search.py.
  LANGSEARCH_API_KEY           Required for web_search when
                               SEARCH_PROVIDER=langsearch — see
                               mcp_server/web_search.py. Loaded from
                               backend/.env below, same as backend/main.py —
                               this is a separate process, so it does not
                               inherit backend/main.py's own load_dotenv().
  BRAVE_API_KEY                Required for web_search when
                               SEARCH_PROVIDER=brave — see
                               mcp_server/web_search.py.

Start
-----
    uvicorn mcp_server.main:app --host 127.0.0.1 --port 8003

Or from backend/ with venv activated:
    python -m uvicorn mcp_server.main:app --host 127.0.0.1 --port 8003 --reload
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

from mcp_server import chart as _chart, file_ops, url_fetch as _url_fetch, web_search as _web_search

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = os.environ.get("LOCALIST_LOG_LEVEL", "INFO").upper(),
    format = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("localist-mcp")


# ── MCP tools ────────────────────────────────────────────────────────────────

mcp = FastMCP(name="localist-mcp")


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.read_file(path)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.write_file(path, content)


@mcp.tool()
def append_file(path: str, content: str, turn_id: str | None = None) -> str:
    """Append content to a UTF-8 text file. path is resolved relative to project_root and sandboxed."""
    return file_ops.append_file(path, content, turn_id)


@mcp.tool()
async def fetch_url(url: str, timeout: float = 10.0) -> dict:
    """Fetch a URL and extract clean article text (title, author, date, cleaned_text, word_count)."""
    return await _url_fetch.fetch_url(url, timeout)


@mcp.tool()
async def web_search(query: str) -> dict:
    """Run one web search query via the configured search provider (SEARCH_PROVIDER=langsearch|brave)."""
    return await _web_search.web_search(query)


@mcp.tool()
def generate_chart(chart_type: str, labels: list[str], datasets: list[dict], title: str = "") -> dict:
    """Render a bar/line/pie chart from structured data and save it as a PNG. Returns summary, png_path, and chart_config."""
    return _chart.generate_chart(chart_type, labels, datasets, title)


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Localist MCP Server starting on port 8003 — project_root=%s",
        file_ops.get_project_root(),
    )
    yield
    logger.info("Localist MCP Server shutting down.")


app = FastAPI(
    title       = "Localist MCP Server",
    description = "MCP tool server for Localist — file_op tools (read_file/write_file/append_file), fetch_url, web_search, and generate_chart.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173",
                         "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount the MCP SSE app at root — exposes GET /sse and POST /messages/.
# Registered after /health so the explicit route takes precedence over the mount.
app.mount("/", mcp.sse_app())


# ── Dev entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mcp_server.main:app",
        host      = "127.0.0.1",
        port      = 8003,
        reload    = True,
        log_level = "info",
    )
