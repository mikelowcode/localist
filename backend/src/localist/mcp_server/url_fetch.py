"""
Localist MCP Server — url_fetch tool implementation
=======================================================
Ports the retired standalone Fetcher microservice's /extract path
(backend/fetcher/client.py's fetch() + backend/fetcher/extractor.py's
extract()) verbatim, in-process — no HTTP hop to a separate service.

Only /extract is ported. /fetch (raw HTML) and /api (JSON passthrough) are
not — nothing in the codebase calls them, url_fetch only ever needs
"readable content," and porting unused capability speculatively is out of
scope for this phase.

Error taxonomy mirrors fetcher/models.py's ErrorResponse.error_code exactly
(connection_error | timeout | http_client_error | http_server_error |
extraction_failed), plus one Localist-only addition, javascript_required
(§14.17: an HTTP 200 page that is just a JavaScript app shell, with no
readable text until a browser runs its scripts — kept distinct from
extraction_failed's paywall/login-wall case so the model can say so
honestly), carried as a "ERROR: <code> — <message> (<detail>)"
string raised from fetch_url() so the MCP protocol layer's isError path
picks it up — see mcp_tool_dispatcher.py's _normalize_mcp_error_text(),
reused unchanged from the Phase 1 follow-up, which strips FastMCP's own
"Error executing tool <name>: " wrapper back down to this exact shape.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx
import lxml.html
from readability import Document

from . import msn_resolve

logger = logging.getLogger(__name__)

# Browser-like UA reduces bot-blocking on common sites (ported from fetcher/client.py).
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_BASE_HEADERS = {
    "User-Agent": _DEFAULT_USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# Timeout clamp — same bounds as fetcher/models.py's ExtractRequest validator.
_MIN_TIMEOUT: float = 1.0
_MAX_TIMEOUT: float = 30.0


# ---------------------------------------------------------------------------
# Ported from fetcher/client.py — async HTTP GET
# ---------------------------------------------------------------------------

@dataclass
class RawResponse:
    url:               str
    status_code:       int
    content_type:      str
    content:           bytes
    headers:           dict[str, str]
    fetch_duration_ms: float


async def _fetch(url: str, timeout: float = 10.0) -> RawResponse:
    """
    Perform an async HTTP GET. Returns RawResponse.

    Raises
    ------
    httpx.TimeoutException        — caller maps to error_code="timeout"
    httpx.ConnectError            — caller maps to error_code="connection_error"
    httpx.HTTPStatusError         — caller maps to http_client_error / http_server_error
    httpx.RequestError            — caller maps to error_code="connection_error"
    """
    t0 = time.perf_counter()

    logger.debug("fetch_url: fetch() → %s  timeout=%.1fs", url, timeout)

    async with httpx.AsyncClient(
        follow_redirects = True,
        timeout          = httpx.Timeout(timeout),
    ) as client:
        response = await client.get(url, headers=_BASE_HEADERS)
        response.raise_for_status()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    content_type = response.headers.get("content-type", "")

    logger.info(
        "fetch_url: fetch() ← %s  status=%d  content_type=%r  bytes=%d  duration=%.0fms",
        url, response.status_code, content_type, len(response.content), elapsed_ms,
    )

    return RawResponse(
        url               = str(response.url),
        status_code       = response.status_code,
        content_type      = content_type,
        content           = response.content,
        headers           = dict(response.headers),
        fetch_duration_ms = elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Ported from fetcher/extractor.py — readability extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractedContent:
    title:          str
    author:         str
    date_published: str
    cleaned_text:   str
    word_count:     int


def _extract(html: bytes, url: str = "") -> ExtractedContent:
    """
    Extract clean article content from raw HTML bytes.

    Raises
    ------
    ValueError
        If readability produces empty content (e.g. login walls, paywalls).
    """
    html_str = html.decode("utf-8", errors="replace")
    doc    = Document(html_str, url=url)
    title  = (doc.title() or "").strip()
    summary_html = doc.summary(html_partial=True)

    if not summary_html:
        raise ValueError("readability returned empty content — possible paywall or login wall")

    root         = lxml.html.fromstring(summary_html)
    cleaned_text = _clean_text(root.text_content())

    if not cleaned_text:
        raise ValueError("extraction produced empty text after tag stripping")

    try:
        full_doc = lxml.html.fromstring(html)
        author   = _extract_meta(full_doc, ["author", "article:author"])
        date     = _extract_meta(full_doc, [
            "article:published_time", "datePublished", "pubdate", "date",
        ])
    except Exception:
        author = ""
        date   = ""

    word_count = len(cleaned_text.split())

    logger.debug(
        "fetch_url: extract() ← title=%r  words=%d  author=%r  date=%r",
        title, word_count, author, date,
    )

    return ExtractedContent(
        title          = title,
        author         = author,
        date_published = date,
        cleaned_text   = cleaned_text,
        word_count     = word_count,
    )


def _clean_text(raw: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = re.sub(r"\s+", " ", raw)
    return text.strip()


# ---------------------------------------------------------------------------
# JavaScript app-shell detection (§14.17)
# ---------------------------------------------------------------------------

# Calibrated 2026-10-21 against real pages: an MSN article shell had 0
# visible body chars with 77% of its bytes in <script>; real articles
# (Digital Foundry, Engadget) had 26k / 3.4k visible chars at 14% / 20%.
_JS_SHELL_MAX_VISIBLE_CHARS: int   = 200
_JS_SHELL_MIN_SCRIPT_SHARE:  float = 0.5
# Extraction that "succeeds" with fewer words than this is re-checked —
# catches stubs like "You need to enable JavaScript to run this app."
_JS_SHELL_TINY_WORD_COUNT:   int   = 40
# Mount nodes React/Vue/Next/Nuxt render into; empty in an unrendered shell.
_SPA_MOUNT_IDS: tuple[str, ...] = ("root", "app", "__next", "__nuxt")


def _looks_like_js_shell(html: bytes) -> bool:
    """
    True when a page has almost no visible body text AND carries a positive
    JavaScript-app signal: scripts make up most of its bytes, an SPA mount
    node is empty, or a <noscript> asks for JavaScript. The second clause
    is what keeps a bare paywall/login page (no text, no scripts) on
    extraction_failed rather than misreporting it as a JavaScript site.
    """
    try:
        doc = lxml.html.fromstring(html)
    except Exception:
        return False

    script_bytes = sum(len(lxml.html.tostring(s)) for s in doc.xpath("//script"))
    script_share = script_bytes / len(html) if html else 0.0
    noscript_asks_for_js = any(
        "javascript" in n.text_content().lower() for n in doc.xpath("//noscript")
    )

    for node in doc.xpath("//script|//style|//noscript|//template"):
        node.drop_tree()

    body         = doc.find("body")
    visible_text = _clean_text(body.text_content()) if body is not None else ""
    if len(visible_text) >= _JS_SHELL_MAX_VISIBLE_CHARS:
        return False

    empty_mount = any(
        not _clean_text(node.text_content())
        for mount_id in _SPA_MOUNT_IDS
        for node in doc.xpath(f'//*[@id="{mount_id}"]')
    )

    return script_share >= _JS_SHELL_MIN_SCRIPT_SHARE or empty_mount or noscript_asks_for_js


def _javascript_required_error() -> "FetchUrlError":
    return FetchUrlError(
        "javascript_required",
        "This site loads its content with JavaScript and can't be read by the fetcher.",
        "the page is an app shell with no readable text until a browser runs its scripts",
    )


def _extract_meta(doc: "lxml.html.HtmlElement", names: list[str]) -> str:
    """Extract the first matching meta tag content — checks name= and property=."""
    for name in names:
        for attr in ("name", "property"):
            nodes = doc.xpath(f'//meta[@{attr}="{name}"]/@content')
            if nodes:
                return str(nodes[0]).strip()
    return ""


# ---------------------------------------------------------------------------
# fetch_url — the MCP tool entry point
# ---------------------------------------------------------------------------

class FetchUrlError(ValueError):
    """Raised on any fetch/extraction failure — str(self) always starts with 'ERROR:'."""

    def __init__(self, error_code: str, message: str, detail: str = ""):
        self.error_code = error_code
        self.message    = message
        self.detail     = detail
        text = f"ERROR: {error_code} — {message}"
        if detail:
            text += f" ({detail})"
        super().__init__(text)


def _clamp_timeout(v: float) -> float:
    return max(_MIN_TIMEOUT, min(v, _MAX_TIMEOUT))


async def fetch_url(url: str, timeout: float = 10.0) -> dict:
    """
    Fetch a URL and extract clean article content — the /extract path,
    ported in-process. Raises FetchUrlError (a ValueError) on any failure,
    carrying the fetcher/models.py-style error_code in its message.

    MSN article URLs are resolved first (§14.17, _fetch_msn_article): those
    pages are JavaScript shells, but MSN's content API names the original
    publisher. A resolved result carries an extra "via" key; anything the
    resolver can't handle falls through to the normal fetch below.
    """
    clamped = _clamp_timeout(timeout)

    msn_ref = msn_resolve.parse_article_url(url)
    if msn_ref is not None:
        resolved = await _fetch_msn_article(url, msn_ref, clamped)
        if resolved is not None:
            return resolved

    return await _fetch_and_extract(url, clamped)


async def _fetch_and_extract(url: str, timeout: float) -> dict:
    """One plain fetch + readability extraction — fetch_url's core path."""
    try:
        raw = await _fetch(url, timeout)
    except httpx.TimeoutException as exc:
        raise FetchUrlError("timeout", "Request timed out.", str(exc)) from exc
    except httpx.ConnectError as exc:
        raise FetchUrlError("connection_error", "Could not connect to host.", str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        code = "http_client_error" if exc.response.status_code < 500 else "http_server_error"
        raise FetchUrlError(
            code, f"Target returned HTTP {exc.response.status_code}.", str(exc)
        ) from exc
    except Exception as exc:
        raise FetchUrlError("connection_error", "HTTP request failed.", str(exc)) from exc

    try:
        content = _extract(raw.content, raw.url)
    except ValueError as exc:
        if _looks_like_js_shell(raw.content):
            raise _javascript_required_error() from exc
        raise FetchUrlError("extraction_failed", "Content extraction failed.", str(exc)) from exc

    if content.word_count < _JS_SHELL_TINY_WORD_COUNT and _looks_like_js_shell(raw.content):
        raise _javascript_required_error()

    return {
        "url":               raw.url,
        "title":             content.title,
        "author":            content.author,
        "date_published":    content.date_published,
        "cleaned_text":      content.cleaned_text,
        "word_count":        content.word_count,
        "fetch_duration_ms": raw.fetch_duration_ms,
    }


# ---------------------------------------------------------------------------
# MSN republished articles (§14.17)
# ---------------------------------------------------------------------------

# The original publisher's page wins unless it extracts to less than this
# share of MSN's own copy — a publisher paywall teaser, say, where MSN's
# syndicated copy is the full article.
_MSN_ORIGINAL_MIN_SHARE: float = 0.5


async def _fetch_msn_article(
    url:     str,
    ref:     msn_resolve.MsnArticleRef,
    timeout: float,
) -> dict | None:
    """
    Read an MSN article via its content API: fetch the original publisher's
    page (sourceHref) when there is one, else — or when that fails or comes
    back much thinner — use MSN's own copy of the body. None when the API
    lookup fails or yields nothing readable, so fetch_url falls through to
    fetching the MSN URL itself.
    """
    start   = time.perf_counter()
    article = await msn_resolve.lookup(ref, timeout, _BASE_HEADERS)
    if article is None:
        return None

    via = "msn.com" + (f" (republished from {article.provider})" if article.provider else "")
    msn_words = len(article.body_text.split())

    if article.source_href:
        try:
            original = await _fetch_and_extract(article.source_href, timeout)
        except FetchUrlError as exc:
            logger.info(
                "fetch_url: MSN original %s unreadable (%s) — using MSN's copy.",
                article.source_href, exc.error_code,
            )
        else:
            if original["word_count"] >= _MSN_ORIGINAL_MIN_SHARE * msn_words:
                logger.debug(
                    "fetch_url: MSN %s resolved to original %s (%d words)",
                    url, original["url"], original["word_count"],
                )
                return {**original, "via": via}
            logger.info(
                "fetch_url: MSN original %s gave %d words vs MSN's %d — using MSN's copy.",
                original["url"], original["word_count"], msn_words,
            )

    if not msn_words:
        return None

    if article.source_href:
        via += f"; original: {article.source_href}"
    return {
        "url":               url,
        "title":             article.title,
        "author":            article.author,
        "date_published":    article.date_published,
        "cleaned_text":      article.body_text,
        "word_count":        msn_words,
        "fetch_duration_ms": (time.perf_counter() - start) * 1000,
        "via":               via,
    }
