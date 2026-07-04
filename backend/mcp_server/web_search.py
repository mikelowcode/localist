"""
Localist MCP Server — web_search tool implementation
========================================================
Ports the real-LangSearch branch of ToolDispatcher._execute_single_search
verbatim (payload, endpoint, response parsing, truncation, formatting) —
same proven-working request/response contract, no redesign.

Phase 3 locked decision: the legacy fallback that called runtime.infer() to
generate plausible-sounding bullet points when LANGSEARCH_API_KEY was unset
is removed entirely — model-hallucinated content indistinguishable from a
real search result to every downstream consumer. Missing API key raises a
clean error here; nothing calls inference on this path. Confirmed via grep
that _WEB_SEARCH_FALLBACK_SYSTEM (tool_dispatcher.py) has no other callers,
so nothing else depends on the removed behaviour.

Async (httpx) rather than sync (requests) — brings this tool in line with
fetch_url's async style from Phase 2 and removes a sync HTTP call that was
previously blocking inside what may be an async context. The
request/response contract itself is unchanged.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_LANGSEARCH_ENDPOINT: str = "https://api.langsearch.com/v1/web-search"
_LANGSEARCH_COUNT: int = 3


async def web_search(query: str) -> dict:
    """
    Run one web_search query via the LangSearch API.

    Raises
    ------
    ValueError
        "ERROR: LANGSEARCH_API_KEY not configured" if the key is unset/empty
        (no inference fallback — see module docstring), or
        "ERROR: web_search failed — <exc>" on any network/HTTP/parsing error.
    """
    api_key = os.environ.get("LANGSEARCH_API_KEY", "")

    if not api_key:
        raise ValueError("ERROR: LANGSEARCH_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "query":     query,
        "summary":   True,
        "count":     _LANGSEARCH_COUNT,
        "freshness": "noLimit",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(_LANGSEARCH_ENDPOINT, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("web_search: LangSearch failed for query=%r: %s", query, exc)
        raise ValueError(f"ERROR: web_search failed — {exc}") from exc

    pages = data.get("data", {}).get("webPages", {}).get("value", [])

    if not pages:
        return {"query": query, "result_text": "No results found.", "result_count": 0}

    lines: list[str] = []
    for page in pages[:_LANGSEARCH_COUNT]:
        name    = page.get("name", "").strip()
        snippet = page.get("snippet", "").strip()
        url     = page.get("displayUrl", page.get("url", "")).strip()
        # Prefer summary over snippet when available
        body    = page.get("summary") or snippet
        # Truncate body to keep Slot 6 within budget
        body    = body[:300].rsplit(" ", 1)[0] if len(body) > 300 else body
        lines.append(f"• {name}\n  {body}\n  [{url}]")

    result_text = "\n\n".join(lines)
    logger.info(
        "web_search: LangSearch complete for query=%r results=%d result_chars=%d.",
        query, len(pages), len(result_text),
    )
    return {"query": query, "result_text": result_text, "result_count": len(pages)}
