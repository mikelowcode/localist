"""
Localist MCP Server — MSN republished-article lookup
=======================================================
MSN article pages (msn.com/<locale>/…/ar-<ID>) are JavaScript app shells:
HTTP 200 with no readable text until a browser runs their scripts, so
url_fetch's readability extraction gets nothing from them (§14.17). Most
MSN articles are syndicated from another outlet, and MSN's public content
API — the same one the page's own scripts call — returns, keyless:

    sourceHref         the original publisher's URL
    provider.name      the original publisher's name
    title / authors / publishedDateTime
    body               the full article HTML

This module only identifies MSN article URLs and performs that lookup.
Deciding what to fetch with the result (original first, MSN's own body as
the fallback) is url_fetch.fetch_url's job, so all page fetching and
extraction stays in one place.

Every failure here returns None rather than raising: the lookup is an
enhancement, and the caller falls through to fetching the MSN URL itself
(which then reports javascript_required honestly).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import lxml.html

logger = logging.getLogger(__name__)

_CONTENT_API_URL: str = "https://assets.msn.com/content/view/v2/Detail/{locale}/{article_id}"
_DEFAULT_LOCALE:  str = "en-us"

_LOCALE_RE     = re.compile(r"^[a-z]{2}-[a-z]{2}$")
_ARTICLE_ID_RE = re.compile(r"^ar-([A-Za-z0-9]+)$")


@dataclass(frozen=True)
class MsnArticleRef:
    locale:     str
    article_id: str


@dataclass
class MsnArticle:
    title:          str
    author:         str
    date_published: str
    provider:       str
    source_href:    str   # original publisher URL, "" when absent or itself MSN
    body_text:      str   # whitespace-normalised text of MSN's own copy


def _is_msn_host(host: str) -> bool:
    host = host.lower()
    return host == "msn.com" or host.endswith(".msn.com")


def parse_article_url(url: str) -> MsnArticleRef | None:
    """
    Return the locale + article ID for an MSN article URL, else None.
    Only `ar-<ID>` article pages qualify — video (`vi-`), gallery (`gm-`)
    and section pages are left to the normal fetch path.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not _is_msn_host(parsed.hostname or ""):
        return None

    segments = [s for s in parsed.path.split("/") if s]
    if not segments:
        return None
    match = _ARTICLE_ID_RE.match(segments[-1])
    if not match:
        return None

    locale = segments[0].lower() if _LOCALE_RE.match(segments[0].lower()) else _DEFAULT_LOCALE
    return MsnArticleRef(locale=locale, article_id=match.group(1))


async def lookup(ref: MsnArticleRef, timeout: float, headers: dict[str, str]) -> MsnArticle | None:
    """Query MSN's content API for one article. None on any failure."""
    api_url = _CONTENT_API_URL.format(locale=ref.locale, article_id=ref.article_id)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.info("msn_resolve: content API lookup failed for %s: %s", api_url, exc)
        return None

    if not isinstance(data, dict):
        return None

    source_href = str(data.get("sourceHref") or "").strip()
    parsed_source = urlparse(source_href) if source_href else None
    if (
        parsed_source is None
        or parsed_source.scheme not in ("http", "https")
        or _is_msn_host(parsed_source.hostname or "")
    ):
        source_href = ""

    provider = data.get("provider") if isinstance(data.get("provider"), dict) else {}
    authors  = [
        str(a.get("name", "")).strip()
        for a in (data.get("authors") or [])
        if isinstance(a, dict) and a.get("name")
    ]

    article = MsnArticle(
        title          = str(data.get("title") or "").strip(),
        author         = ", ".join(authors),
        date_published = str(data.get("publishedDateTime") or "").strip(),
        provider       = str(provider.get("name") or "").strip(),
        source_href    = source_href,
        body_text      = _body_text(str(data.get("body") or "")),
    )
    logger.debug(
        "msn_resolve: %s → provider=%r source=%r body_words=%d",
        ref.article_id, article.provider, article.source_href, len(article.body_text.split()),
    )
    return article


def _body_text(body_html: str) -> str:
    if not body_html.strip():
        return ""
    try:
        text = lxml.html.fromstring(body_html).text_content()
    except Exception:
        return ""
    return re.sub(r"\s+", " ", text).strip()
