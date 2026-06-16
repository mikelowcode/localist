"""
LORA Fetcher Service
====================
Standalone FastAPI service on port 8002.

Endpoints
---------
POST /fetch    — Raw HTTP fetch (HTML + headers)
POST /extract  — Fetch + readability extraction (clean article text)
POST /api      — Fetch a JSON REST endpoint

Start
-----
    uvicorn fetcher.main:app --host 127.0.0.1 --port 8002

Or from backend/ with venv activated:
    python -m uvicorn fetcher.main:app --host 127.0.0.1 --port 8002 --reload
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fetcher.client import fetch as _fetch
from fetcher.extractor import extract as _extract
from fetcher.models import (
    ApiRequest, ApiResponse,
    ErrorResponse,
    ExtractRequest, ExtractResponse,
    FetchRequest, FetchResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = os.environ.get("LORA_LOG_LEVEL", "INFO").upper(),
    format = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("fetcher")


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LORA Fetcher Service starting on port 8002.")
    yield
    logger.info("LORA Fetcher Service shutting down.")


app = FastAPI(
    title       = "LORA Fetcher Service",
    description = "HTTP fetch, extraction, and JSON API tool for LORA.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Error helper ─────────────────────────────────────────────────────────────

def _error(url: str, code: str, message: str,
           detail: str = "", status: int = 502) -> JSONResponse:
    logger.warning("fetcher error: url=%r  code=%r  message=%r", url, code, message)
    return JSONResponse(
        status_code = status,
        content     = ErrorResponse(
            url        = url,
            error_code = code,
            message    = message,
            detail     = detail,
        ).model_dump(),
    )


def _handle_httpx_error(url: str, exc: Exception) -> JSONResponse:
    """Map httpx exceptions to structured ErrorResponse."""
    if isinstance(exc, httpx.TimeoutException):
        return _error(url, "timeout",
                      "Request timed out.", str(exc), 504)
    if isinstance(exc, httpx.ConnectError):
        return _error(url, "connection_error",
                      "Could not connect to host.", str(exc), 502)
    if isinstance(exc, httpx.HTTPStatusError):
        code = ("http_client_error" if exc.response.status_code < 500
                else "http_server_error")
        return _error(url, code,
                      f"Target returned HTTP {exc.response.status_code}.",
                      str(exc), 502)
    return _error(url, "connection_error",
                  "HTTP request failed.", str(exc), 502)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/fetch", response_model=FetchResponse)
async def endpoint_fetch(req: FetchRequest):
    """
    Raw HTTP fetch. Returns status, headers, and raw HTML.
    Useful for inspection and debugging.
    """
    try:
        raw = await _fetch(req.url, req.timeout, req.headers)
    except Exception as exc:
        return _handle_httpx_error(req.url, exc)

    return FetchResponse(
        url               = raw.url,
        status_code       = raw.status_code,
        content_type      = raw.content_type,
        html              = raw.content.decode("utf-8", errors="replace"),
        headers           = raw.headers,
        fetch_duration_ms = raw.fetch_duration_ms,
    )


@app.post("/extract", response_model=ExtractResponse)
async def endpoint_extract(req: ExtractRequest):
    """
    Fetch a URL and extract clean article text via readability-lxml.
    This is the primary endpoint called by LORA's ToolDispatcher.
    Returns full content — PromptBuilder enforces Slot 6 truncation.
    """
    try:
        raw = await _fetch(req.url, req.timeout)
    except Exception as exc:
        return _handle_httpx_error(req.url, exc)

    try:
        content = _extract(raw.content, raw.url)
    except ValueError as exc:
        return _error(req.url, "extraction_failed",
                      "Content extraction failed.", str(exc), 422)

    return ExtractResponse(
        url               = raw.url,
        title             = content.title,
        author            = content.author,
        date_published    = content.date_published,
        cleaned_text      = content.cleaned_text,
        word_count        = content.word_count,
        fetch_duration_ms = raw.fetch_duration_ms,
        extractor_used    = "readability-lxml",
    )


@app.post("/api", response_model=ApiResponse)
async def endpoint_api(req: ApiRequest):
    """
    Fetch a JSON REST endpoint. Returns parsed JSON data.
    Returns HTTP 422 if the response is not application/json.
    """
    try:
        raw = await _fetch(req.url, req.timeout, req.headers)
    except Exception as exc:
        return _handle_httpx_error(req.url, exc)

    if "application/json" not in raw.content_type:
        return _error(
            req.url, "not_json",
            f"Expected application/json, got {raw.content_type!r}.",
            status = 422,
        )

    try:
        data = json.loads(raw.content)
    except json.JSONDecodeError as exc:
        return _error(req.url, "not_json",
                      "Response claimed JSON but failed to parse.",
                      str(exc), 422)

    return ApiResponse(
        url               = raw.url,
        status_code       = raw.status_code,
        content_type      = raw.content_type,
        data              = data,
        fetch_duration_ms = raw.fetch_duration_ms,
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"healthy": True, "service": "lora-fetcher", "port": 8002}
