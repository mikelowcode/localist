"""
LORA Fetcher — Pydantic models
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, field_validator

# ── Shared ──────────────────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    url:     str
    timeout: float             = 10.0
    headers: dict[str, str]    = {}

    @field_validator("timeout")
    @classmethod
    def _clamp_timeout(cls, v: float) -> float:
        return max(1.0, min(v, 30.0))


class ErrorResponse(BaseModel):
    url:        str
    error_code: str   # connection_error | timeout | http_client_error |
                      # http_server_error | extraction_failed | not_json
    message:    str
    detail:     str   = ""


# ── /fetch ───────────────────────────────────────────────────────────────────

class FetchResponse(BaseModel):
    url:              str
    status_code:      int
    content_type:     str
    html:             str
    headers:          dict[str, str]
    fetch_duration_ms: float


# ── /extract ─────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    url:     str
    timeout: float = 10.0

    @field_validator("timeout")
    @classmethod
    def _clamp_timeout(cls, v: float) -> float:
        return max(1.0, min(v, 30.0))


class ExtractResponse(BaseModel):
    url:              str
    title:            str
    author:           str        = ""
    date_published:   str        = ""
    cleaned_text:     str
    word_count:       int
    fetch_duration_ms: float
    extractor_used:   str        = "readability-lxml"


# ── /api ─────────────────────────────────────────────────────────────────────

class ApiRequest(BaseModel):
    url:     str
    timeout: float             = 10.0
    headers: dict[str, str]    = {}

    @field_validator("timeout")
    @classmethod
    def _clamp_timeout(cls, v: float) -> float:
        return max(1.0, min(v, 30.0))


class ApiResponse(BaseModel):
    url:               str
    status_code:       int
    content_type:      str
    data:              Any
    fetch_duration_ms: float
