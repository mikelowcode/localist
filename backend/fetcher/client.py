"""
LORA Fetcher — async HTTP client (httpx)

Single responsibility: perform the HTTP GET and return raw bytes + metadata.
All content parsing lives in extractor.py and main.py.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Browser-like UA reduces bot-blocking on common sites.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_BASE_HEADERS = {
    "User-Agent": _DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class RawResponse:
    url:               str
    status_code:       int
    content_type:      str
    content:           bytes
    headers:           dict[str, str]
    fetch_duration_ms: float


async def fetch(
    url:             str,
    timeout:         float             = 10.0,
    extra_headers:   dict[str, str]    = {},
) -> RawResponse:
    """
    Perform an async HTTP GET. Returns RawResponse.

    Raises
    ------
    httpx.TimeoutException        — caller maps to error_code="timeout"
    httpx.ConnectError            — caller maps to error_code="connection_error"
    httpx.HTTPStatusError         — caller maps to http_client_error / http_server_error
    httpx.RequestError            — caller maps to error_code="connection_error"
    """
    headers = {**_BASE_HEADERS, **extra_headers}
    t0 = time.perf_counter()

    logger.debug("fetch() → %s  timeout=%.1fs", url, timeout)

    async with httpx.AsyncClient(
        follow_redirects = True,
        timeout          = httpx.Timeout(timeout),
    ) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    content_type = response.headers.get("content-type", "")

    logger.info(
        "fetch() ← %s  status=%d  content_type=%r  "
        "bytes=%d  duration=%.0fms",
        url, response.status_code, content_type,
        len(response.content), elapsed_ms,
    )

    return RawResponse(
        url               = str(response.url),
        status_code       = response.status_code,
        content_type      = content_type,
        content           = response.content,
        headers           = dict(response.headers),
        fetch_duration_ms = elapsed_ms,
    )
