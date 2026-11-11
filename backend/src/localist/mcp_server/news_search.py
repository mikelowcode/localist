"""
Localist MCP Server — news_search tool implementation
========================================================
NewsAPI.org (/v2/everything) is the first-tier provider for news-shaped
queries ("latest news on X", "headlines about Y") — a purpose-built news
index with publish dates and source attribution, unlike the generic
web_search tool (LangSearch/Brave), which has no concept of article
freshness. See news-query-routing plan, §4.1.

Auth is via the X-Api-Key header rather than the ?apiKey= query param —
functionally equivalent, but keeps the key out of any request logging that
captures URLs (same rationale as web_search's choice of Authorization
header for LangSearch).

Free Developer-tier terms (100 req/day, 24h publish delay, last-month-only
article window, explicitly not licensed for production) are a natural fit
here since Localist runs single-user on localhost and never leaves it —
see NEWSAPI_API_KEY's .env.example comment.

The env var is read lazily inside news_search() rather than cached at
import time, same reasoning as web_search.py: this process does not
inherit backend/main.py's own load_dotenv() call.
"""

from __future__ import annotations

import html
import logging
import os
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_NEWSAPI_ENDPOINT: str = "https://newsapi.org/v2/everything"
_NEWSAPI_PAGE_SIZE: int = 5
_NEWSAPI_SUMMARY_CHARS: int = 600
_NEWSAPI_CONTENT_CHARS: int = 500

# Shown above the results when a caller pinned a URL that NewsAPI didn't
# return (2026-11-11): without it the model reads another outlet's copy —
# e.g. a Slashdot "firehose" repost of a Verge story — as the clicked
# article itself. The dispatcher fetches the clicked URL directly (§14.19).
_PIN_MISS_NOTE: str = (
    "Note: the article the user selected ({url}) was not among these "
    "results; they are other reports on the same story."
)


def _clean_url(url: str) -> str:
    """NewsAPI sometimes returns HTML-escaped URLs (``&amp;`` for ``&``) —
    seen live on a Slashdot firehose link, which then 403'd when fetched."""
    return html.unescape((url or "").strip())


def _normalize_article_url(url: str) -> str:
    """
    Comparison key for URL pinning. Two links to the same article often
    differ only cosmetically, so exact string equality silently dropped the
    pin. Ignored: scheme, a leading ``www.``, host case, a trailing ``/``,
    the ``#fragment``, and ``utm_*`` tracking parameters. Other query
    parameters are kept — some sites identify the article by them
    (``?id=…``).
    """
    parts = urlsplit(_clean_url(url))
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ])
    return f"{host}{path}?{query}" if query else f"{host}{path}"


async def news_search(query: str, url: str | None = None) -> dict:
    """
    Run one news_search query via NewsAPI's /v2/everything endpoint.

    Parameters
    ----------
    query : the search text (usually a headline or topic).
    url   : optional — when a caller already knows the exact article (e.g.
            the user clicked a specific story in the Live Feed panel), pass
            its URL to pin the result to that one article rather than
            trusting the query text to find it again among near-duplicate
            coverage. Matching ignores cosmetic differences (see
            _normalize_article_url). If none of the returned articles match,
            falls back to the normal unfiltered top-5 behavior — an older
            clicked article may simply be outside NewsAPI dev-tier's
            last-month window, or NewsAPI may only return another outlet's
            copy — prefixed with a note saying so (_PIN_MISS_NOTE). The
            dispatcher fetches the clicked URL itself either way (§14.19).

    Returns
    -------
    dict with keys:
      query        : the input query, echoed back
      result_text  : formatted bullet list, or "" when there are no
                     results (caller decides what "no results" means —
                     see is_miss below)
      result_count : number of articles returned
      pinned       : True when ``url`` was given and matched a result.
      is_miss      : True when NewsAPI itself reports no usable result
                     (status != "ok", or totalResults == 0) — the precise
                     "fall through to Brave" trigger condition from §4.1.
                     False for a genuine transport/HTTP error, which
                     raises instead (indistinguishable from any other
                     tool-call failure, same as web_search).

    Raises
    ------
    ValueError
        "ERROR: NEWSAPI_API_KEY not configured" if the key is unset/empty,
        or "ERROR: news_search failed — <exc>" on any network/HTTP/
        parsing error (transport failure, not a zero-result response —
        those two are deliberately distinct, see is_miss above).
    """
    api_key = os.environ.get("NEWSAPI_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: NEWSAPI_API_KEY not configured")

    headers = {
        "X-Api-Key": api_key,
    }
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": _NEWSAPI_PAGE_SIZE,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_NEWSAPI_ENDPOINT, headers=headers, params=params)
            data = resp.json()
    except Exception as exc:
        logger.warning("news_search: NewsAPI failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: news_search failed — {exc}") from exc

    if data.get("status") != "ok" or data.get("totalResults", 0) == 0:
        logger.info(
            "news_search: NewsAPI miss for query=%r status=%s totalResults=%s.",
            query, data.get("status"), data.get("totalResults"),
        )
        return {"query": query, "result_text": "", "result_count": 0, "is_miss": True}

    articles = data.get("articles", [])

    # Pin to the one already-known article when a caller (e.g. the Live Feed
    # panel's "Ask about this") supplies its URL — otherwise a query built
    # from a headline can resolve to several near-duplicate stories and the
    # model would have no way to tell which one the user actually clicked.
    pinned = False
    if url:
        wanted = _normalize_article_url(url)
        matched = [
            a for a in articles
            if _normalize_article_url(a.get("url") or "") == wanted
        ]
        if matched:
            articles = matched
            pinned = True

    results: list[search_format.SearchResult] = []
    for article in articles[:_NEWSAPI_PAGE_SIZE]:
        title        = article.get("title", "").strip()
        description  = (article.get("description") or "").strip()
        source_name  = article.get("source", {}).get("name", "").strip()
        published_at = article.get("publishedAt", "").strip().split("T", 1)[0]
        article_url  = _clean_url(article.get("url") or "")

        # `content` is NewsAPI's own short body snippet (also truncated on
        # the dev tier) — surfaced only in the pinned single-article case
        # so the multi-result formatting other callers rely on is unchanged.
        extra = None
        if pinned:
            content = (article.get("content") or "").strip()
            if content:
                extra = search_format.truncate_summary(content, _NEWSAPI_CONTENT_CHARS)

        results.append(search_format.SearchResult(
            title=title,
            summary=description,
            url=article_url,
            source=source_name or None,
            published_at=published_at or None,
            extra=extra,
        ))

    result_text = search_format.format_results(
        results, per_result_budget=_NEWSAPI_SUMMARY_CHARS
    )
    if url and not pinned:
        result_text = _PIN_MISS_NOTE.format(url=_clean_url(url)) + "\n\n" + result_text
    logger.info(
        "news_search: NewsAPI complete for query=%r results=%d result_chars=%d pinned=%s.",
        query, len(articles), len(result_text), pinned if url else "n/a",
    )
    return {
        "query":        query,
        "result_text":  result_text,
        "result_count": len(articles),
        "is_miss":      False,
        "pinned":       pinned,
    }
