"""
Localist MCP Server — Wikipedia tool implementations (wikipedia_search,
wikipedia_read)
========================================================
Read-only lookups against Wikipedia's free public MediaWiki Action API
(https://<lang>.wikipedia.org/w/api.php). No account, no key, no Wikimedia
Enterprise — see docs/architecture/14-localist-mcp-tool-layer.md §14.21.

Why the Action API (not the REST /api/rest_v1 summary endpoint): one
endpoint covers search + intro extracts + descriptions + disambiguation
flags + canonical URLs + an article's external (cited) links, so each
tool is a single request. TextExtracts' `explaintext` gives clean plain
text with no HTML scraping.

Good-citizen rules (Wikimedia APIs/Rate_limits, enforced since 2026-03):
  - A compliant User-Agent naming the client and a contact URL. Without
    one, an unauthenticated client is limited to 10 req/min per IP; with
    one, 200 req/min. The contact is the public repo URL — never a
    person's email. No version number: the Python package version
    (pyproject) is not the release version, and a wrong one is worse
    than none.
  - At most _MAX_CONCURRENT requests in flight (Wikimedia asks for <= 3).
  - On HTTP 429, honor Retry-After (5 s if absent) with ONE retry, only
    when the wait is <= _MAX_RETRY_AFTER_S — a chat turn should never
    stall longer than that on an optional source.

Language: WIKIPEDIA_LANG (default "en"), read lazily on every call — same
reasoning as every other mcp_server tool (this process doesn't inherit
backend/main.py's load_dotenv()). An invalid value falls back to "en".

Wikipedia text is CC BY-SA 4.0: every result carries an attribution line
(`Source: Wikipedia — <url> (CC BY-SA 4.0)`) that callers must keep.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import weakref
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from . import search_format

logger = logging.getLogger(__name__)

_USER_AGENT: str = (
    f"Localist (https://github.com/mikelowcode/localist) httpx/{httpx.__version__}"
)

_DEFAULT_LANG: str = "en"
_LANG_RE = re.compile(r"^[a-z]{2,3}(-[a-z]{2,8})*$")

_TIMEOUT_S: float = 10.0
_MAX_CONCURRENT: int = 3
_MAX_RETRY_AFTER_S: float = 5.0

_SEARCH_COUNT: int = 5
_BEST_SUMMARY_CHARS: int = 1200      # best match's intro in wikipedia_search
_OTHER_SUMMARY_CHARS: int = 200      # every other search hit
_READ_CONTENT_CHARS: int = 6000      # wikipedia_read body cap
_READ_INTRO_CHARS: int = 1500
_MAX_REFERENCES: int = 10
_EXTLINKS_FETCH: int = 60            # fetched, then filtered down to _MAX_REFERENCES

# Hosts that are never useful as "go read the cited source" candidates:
# archive copies of something already listed, and Wikimedia's own sites.
_SKIP_REFERENCE_HOSTS: tuple[str, ...] = (
    "web.archive.org", "archive.org", "archive.today", "archive.ph",
    "archive.is", "wikipedia.org", "wikimedia.org", "wikidata.org",
    "wikisource.org", "wiktionary.org", "mediawiki.org",
)

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _lang() -> str:
    raw = (os.environ.get("WIKIPEDIA_LANG") or _DEFAULT_LANG).strip().lower()
    if not _LANG_RE.match(raw):
        logger.warning("wikipedia: invalid WIKIPEDIA_LANG=%r — using %r.", raw, _DEFAULT_LANG)
        return _DEFAULT_LANG
    return raw


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _semaphore() -> asyncio.Semaphore:
    """
    One concurrency limiter per event loop. A module-level Semaphore would
    bind to whichever loop first contended on it and then fail under any
    other loop (tests run each call under a fresh asyncio.run()).
    """
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        _semaphores[loop] = sem
    return sem


def _retry_after_seconds(resp: httpx.Response) -> float:
    raw = resp.headers.get("retry-after", "")
    try:
        return max(0.0, float(raw))
    except ValueError:
        # Absent, or an HTTP-date we don't bother parsing: Wikimedia's
        # guidance is to wait at least 5 s.
        return _MAX_RETRY_AFTER_S


async def _api_get(
    tool:    str,
    params:  dict[str, str],
    *,
    lang:    str | None = None,
    timeout: float | None = None,
) -> dict:
    """
    One Action API GET with the shared headers, concurrency cap and the
    single 429 retry. Returns the parsed JSON body. Raises
    ValueError("ERROR: <tool> failed — ...") on any failure, including an
    API-level {"error": ...} body (the Action API returns those as 200).
    """
    lang = lang or _lang()
    timeout = timeout or _TIMEOUT_S
    query = {"action": "query", "format": "json", "formatversion": "2", **params}
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

    try:
        async with _semaphore():
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(_api_url(lang), params=query, headers=headers)
                if resp.status_code == 429:
                    wait = _retry_after_seconds(resp)
                    if wait > _MAX_RETRY_AFTER_S:
                        raise ValueError(
                            f"ERROR: {tool} failed — Wikipedia rate limit "
                            f"(retry after {wait:.0f}s)"
                        )
                    logger.info("%s: HTTP 429, retrying once after %.1fs.", tool, wait)
                    await asyncio.sleep(wait)
                    resp = await client.get(_api_url(lang), params=query, headers=headers)
                    if resp.status_code == 429:
                        raise ValueError(f"ERROR: {tool} failed — Wikipedia rate limit")
                resp.raise_for_status()
                data = resp.json()
    except ValueError as exc:
        if str(exc).startswith("ERROR:"):
            raise
        logger.warning("%s: failed params=%r: %s", tool, params, exc)
        raise ValueError(f"ERROR: {tool} failed — {exc}") from exc
    except Exception as exc:
        logger.warning("%s: failed params=%r: %s", tool, params, exc)
        raise ValueError(f"ERROR: {tool} failed — {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"ERROR: {tool} failed — unexpected response shape")
    if "error" in data:
        info = (data.get("error") or {}).get("info") or data["error"]
        raise ValueError(f"ERROR: {tool} failed — {info}")
    return data


def attribution(url: str) -> str:
    """The CC BY-SA attribution line every Wikipedia-derived result carries."""
    return f"Source: Wikipedia — {url} (CC BY-SA 4.0)"


def _is_disambiguation(page: dict) -> bool:
    return "disambiguation" in (page.get("pageprops") or {})


def _page_url(page: dict, lang: str) -> str:
    """
    The article's canonical URL, with parentheses percent-encoded.
    Disambiguated titles ("Mercury (planet)") put parentheses in the URL,
    and MCPToolDispatcher's _URL_RE stops at ")" — the raw form would be
    chained as ".../Mercury_(planet". %28/%29 is the same page.
    """
    url = page.get("canonicalurl") or page.get("fullurl")
    if not url:
        title = (page.get("title") or "").replace(" ", "_")
        url = f"https://{lang}.wikipedia.org/wiki/{title}"
    return url.replace("(", "%28").replace(")", "%29")


def _clean(text: str) -> str:
    """Collapse the blank-line runs TextExtracts leaves behind."""
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def filter_references(urls: list[str], limit: int = _MAX_REFERENCES) -> list[str]:
    """
    Keep http(s) links that point at an actual cited source: drop archive
    copies and Wikimedia's own sites, dedupe, preserve order (extlinks come
    back roughly in citation order), cap at `limit`.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url:
            continue
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        host = parsed.netloc.lower().split(":")[0]
        if any(host == h or host.endswith("." + h) for h in _SKIP_REFERENCE_HOSTS):
            continue
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        kept.append(url)
        if len(kept) >= limit:
            break
    return kept


async def wikipedia_search(query: str) -> dict:
    """
    Search Wikipedia and return the top matches plus the best match's intro.

    One Action API call: generator=search (top _SEARCH_COUNT hits, ranked)
    with each hit's plain-text intro, short description, canonical URL and
    disambiguation flag.

    Returns
    -------
    dict with keys:
      query, result_text, result_count, is_miss — same contract as
        github_search/news_search; result_text uses the shared
        search_format bullet shape, so every hit's URL is on its own
        `[url]` line (MCPToolDispatcher's fetch-chaining depends on it).
        The best match comes first with a longer summary; when it is a
        disambiguation page the text says so, and the remaining hits act
        as the candidate list. Ends with the attribution line.
      best_match — None on a miss, else {title, url, description, summary,
        disambiguation (bool)}.

    Raises
    ------
    ValueError
        "ERROR: wikipedia_search failed — <reason>" on any network/HTTP/
        API error.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("ERROR: wikipedia_search failed — empty query")

    lang = _lang()
    data = await _api_get("wikipedia_search", {
        "generator":   "search",
        "gsrsearch":   query,
        "gsrlimit":    str(_SEARCH_COUNT),
        "gsrnamespace": "0",
        "prop":        "extracts|description|pageprops|info",
        "exintro":     "1",
        "explaintext": "1",
        "exlimit":     "max",
        "inprop":      "url",
        "ppprop":      "disambiguation",
    })

    pages = sorted(
        (data.get("query") or {}).get("pages") or [],
        key=lambda p: p.get("index", 1_000),
    )
    pages = [p for p in pages if not p.get("missing")]
    if not pages:
        logger.info("wikipedia_search: no results for query=%r.", query)
        return {
            "query": query, "result_text": "", "result_count": 0,
            "is_miss": True, "best_match": None,
        }

    best = pages[0]
    best_disambig = _is_disambiguation(best)
    best_url = _page_url(best, lang)
    best_summary = _clean(best.get("extract", ""))

    results: list[search_format.SearchResult] = []
    for i, page in enumerate(pages):
        disambig = _is_disambiguation(page)
        description = page.get("description") or ""
        summary = _clean(page.get("extract", ""))
        if i == 0:
            budget_summary = search_format.truncate_summary(summary, _BEST_SUMMARY_CHARS)
            if disambig:
                budget_summary = (
                    "(Disambiguation page — this name refers to several "
                    "topics; the other results below are candidates.)\n"
                    + budget_summary
                )
        else:
            budget_summary = search_format.truncate_summary(summary, _OTHER_SUMMARY_CHARS)
        if description and not disambig:
            lead = description.rstrip(". ")
            budget_summary = f"{lead}. {budget_summary}" if budget_summary else lead
        results.append(search_format.SearchResult(
            title   = page.get("title", ""),
            summary = budget_summary,
            url     = _page_url(page, lang),
            source  = "Wikipedia",
        ))

    # The best match carries its own (larger) budget above; the shared
    # formatter's per-result budget must not re-cut it.
    body = search_format.format_results(
        results, per_result_budget=_BEST_SUMMARY_CHARS + 300,
    )
    result_text = f"{body}\n\n{attribution(best_url)}"

    logger.info(
        "wikipedia_search: complete for query=%r results=%d best=%r disambiguation=%s.",
        query, len(pages), best.get("title"), best_disambig,
    )
    return {
        "query":        query,
        "result_text":  result_text,
        "result_count": len(pages),
        "is_miss":      False,
        "best_match": {
            "title":          best.get("title", ""),
            "url":            best_url,
            "description":    best.get("description") or "",
            "summary":        best_summary,
            "disambiguation": best_disambig,
        },
    }


def _split_sections(text: str) -> tuple[str, list[str]]:
    """Intro = text before the first `== Heading ==`; plus every heading."""
    match = _HEADING_RE.search(text)
    intro = text[: match.start()] if match else text
    headings = [m.group(2) for m in _HEADING_RE.finditer(text)]
    return _clean(intro), headings


async def wikipedia_read(title: str) -> dict:
    """
    Read one Wikipedia article by title (redirects followed).

    Returns
    -------
    dict with keys:
      title (after redirects), url, description,
      kind — "article" | "disambiguation",
      intro — the lead section (capped at _READ_INTRO_CHARS),
      sections — every section heading, in order,
      content — the plain-text body with `== Heading ==` markers kept,
        capped at _READ_CONTENT_CHARS,
      truncated — whether content was cut,
      references — up to _MAX_REFERENCES cited source URLs (archive copies
        and Wikimedia sites removed), in citation order,
      attribution — the CC BY-SA line callers must keep.

    Raises
    ------
    ValueError
        "ERROR: wikipedia_read — '<title>' not found" for a missing page,
        or "ERROR: wikipedia_read failed — <reason>" on any network/HTTP/
        API error.
    """
    title = (title or "").strip()
    if not title:
        raise ValueError("ERROR: wikipedia_read failed — empty title")

    lang = _lang()
    data = await _api_get("wikipedia_read", {
        "titles":           title,
        "redirects":        "1",
        "prop":             "extracts|description|pageprops|info|extlinks",
        "explaintext":      "1",
        "exsectionformat":  "wiki",
        "inprop":           "url",
        "ppprop":           "disambiguation",
        "ellimit":          str(_EXTLINKS_FETCH),
    })

    pages = (data.get("query") or {}).get("pages") or []
    page = pages[0] if pages else None
    if page is None or page.get("missing") or page.get("invalid"):
        raise ValueError(f"ERROR: wikipedia_read — {title!r} not found")

    url = _page_url(page, lang)
    text = _clean(page.get("extract", ""))
    intro, sections = _split_sections(text)
    intro = search_format.truncate_summary(intro, _READ_INTRO_CHARS)

    truncated = len(text) > _READ_CONTENT_CHARS
    if truncated:
        text = search_format.truncate_summary(text, _READ_CONTENT_CHARS)

    references = filter_references(
        [link.get("url", "") for link in page.get("extlinks") or []]
    )

    kind = "disambiguation" if _is_disambiguation(page) else "article"
    logger.info(
        "wikipedia_read: complete for title=%r resolved=%r kind=%s chars=%d "
        "truncated=%s references=%d.",
        title, page.get("title"), kind, len(text), truncated, len(references),
    )
    return {
        "title":       page.get("title", title),
        "url":         url,
        "description": page.get("description") or "",
        "kind":        kind,
        "intro":       intro,
        "sections":    sections,
        "content":     text,
        "truncated":   truncated,
        "references":  references,
        "attribution": attribution(url),
    }


# ---------------------------------------------------------------------------
# wikipedia.org links read through the API (fetch_url resolver, §14.21)
# ---------------------------------------------------------------------------

_WIKIPEDIA_HOST_RE = re.compile(
    r"^(?P<lang>[a-z]{2,3}(?:-[a-z]{2,8})*)(?:\.m)?\.wikipedia\.org$"
)

# Sections that are lists of links/citations, not prose — dropped from the
# article text fetch_url hands back (the plain-text extract leaves them as
# empty or near-empty headings anyway).
_TRAILING_SECTIONS: frozenset[str] = frozenset({
    "see also", "references", "notes", "external links", "further reading",
    "bibliography", "citations", "sources", "footnotes", "works cited",
    "notes and references", "references and notes",
    # German / French / Spanish / Italian / Portuguese — the common
    # WIKIPEDIA_LANG alternatives;
    # any other language's heading-only tail is caught by the empty-
    # section trim in _arrange_article.
    "siehe auch", "einzelnachweise", "anmerkungen", "weblinks", "literatur",
    "quellen", "voir aussi", "notes et références", "références",
    "liens externes", "bibliographie", "véase también", "referencias",
    "notas", "enlaces externos", "bibliografía",
    "note", "voci correlate", "altri progetti", "collegamenti esterni",
    "bibliografia", "referências", "ligações externas", "ver também",
})

# Namespaces whose pages aren't articles; the plain fetch handles them.
_NON_ARTICLE_PREFIXES: tuple[str, ...] = (
    "special:", "file:", "image:", "media:", "category:", "template:",
    "help:", "portal:", "user:", "talk:", "user talk:", "wikipedia:",
    "draft:", "module:", "mediawiki:", "timedtext:", "book:",
)


@dataclass(frozen=True)
class ArticleRef:
    lang:     str
    title:    str
    fragment: str   # section anchor, spaces not underscores; "" if none


def parse_article_url(url: str) -> ArticleRef | None:
    """
    An article reference from a Wikipedia link, or None when `url` isn't
    one: desktop and mobile hosts (en.wikipedia.org, en.m.wikipedia.org),
    /wiki/<Title> and /w/index.php?title=<Title>. A specific old revision
    (oldid/diff), any non-view action, and non-article namespaces
    (Special:, File:, Talk:, ...) return None so the normal page fetch
    handles them.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.hostname or "").lower()
    match = _WIKIPEDIA_HOST_RE.match(host)
    if not match:
        return None

    query = parse_qs(parsed.query)
    if parsed.path.startswith("/wiki/"):
        raw_title = parsed.path[len("/wiki/"):]
    elif parsed.path in ("/w/index.php", "/w/index.php/"):
        raw_title = (query.get("title") or [""])[0]
    else:
        return None
    if any(k in query for k in ("oldid", "diff", "curid")):
        return None
    if (query.get("action") or ["view"])[0] != "view":
        return None

    title = unquote(raw_title).replace("_", " ").strip()
    if not title or title.lower().startswith(_NON_ARTICLE_PREFIXES):
        return None
    fragment = unquote(parsed.fragment).replace("_", " ").strip()
    return ArticleRef(lang=match.group("lang"), title=title, fragment=fragment)


def _sections(text: str) -> list[tuple[int, str, str]]:
    """
    Split a plain-text extract (exsectionformat=wiki) into
    (level, heading, block) tuples. The lead comes first with level 1 and
    heading "". Each block includes its own heading line.
    """
    out: list[tuple[int, str, str]] = []
    matches = list(_HEADING_RE.finditer(text))
    lead_end = matches[0].start() if matches else len(text)
    out.append((1, "", text[:lead_end].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((len(m.group(1)), m.group(2).strip(), text[m.start():end].strip()))
    return out


def _arrange_article(text: str, fragment: str) -> tuple[str, str]:
    """
    Drop the trailing link/citation sections (and their subsections), and,
    when the link pointed at a section (#fragment) that exists, move that
    section and its subsections up to right after the lead, so it survives
    any later prompt truncation. Returns (text, matched heading or "").
    """
    sections = _sections(text)
    kept: list[tuple[int, str, str]] = []
    skip_level: int | None = None
    for level, heading, block in sections:
        if skip_level is not None and level > skip_level:
            continue
        skip_level = None
        if heading.lower() in _TRAILING_SECTIONS:
            skip_level = level
            continue
        kept.append((level, heading, block))

    matched = ""
    if fragment:
        want = fragment.lower()
        idx = next((i for i, (_, h, _) in enumerate(kept) if h and h.lower() == want), None)
        if idx is not None:
            level = kept[idx][0]
            end = idx + 1
            while end < len(kept) and kept[end][0] > level:
                end += 1
            focus = kept[idx:end]
            rest = kept[:idx] + kept[end:]
            kept = rest[:1] + focus + rest[1:]
            matched = kept[1][1]

    # Heading-only sections at the end (link lists the extract left empty,
    # in a language the set above doesn't cover) carry nothing.
    while len(kept) > 1 and not _HEADING_RE.sub("", kept[-1][2]).strip():
        kept.pop()

    body = "\n\n".join(block for _, _, block in kept if block)
    return _clean(body), matched


async def fetch_article(ref: ArticleRef, timeout: float | None = None) -> dict | None:
    """
    The full plain text of the article a Wikipedia link points at, in
    fetch_url's response shape (url, title, author, date_published,
    cleaned_text, word_count, plus "via"), or None when the API can't
    supply readable text — fetch_url then fetches the page normally.
    Never raises.

    Redirects are followed and the canonical URL returned. The text keeps
    `== Heading ==` markers, drops reference/link sections, leads with the
    linked section when the URL had a matching #fragment, and ends with
    the CC BY-SA attribution line.
    """
    try:
        data = await _api_get("wikipedia_fetch", {
            "titles":          ref.title,
            "redirects":       "1",
            "prop":            "extracts|description|pageprops|info|revisions",
            "explaintext":     "1",
            "exsectionformat": "wiki",
            "inprop":          "url",
            "ppprop":          "disambiguation",
            "rvprop":          "timestamp",
        }, lang=ref.lang, timeout=timeout)
    except ValueError as exc:
        logger.info("wikipedia_fetch: API failed for %r (%s) — normal fetch.", ref.title, exc)
        return None

    pages = (data.get("query") or {}).get("pages") or []
    page = pages[0] if pages else None
    if not page or page.get("missing") or page.get("invalid"):
        return None
    raw = _clean(page.get("extract", ""))
    if not raw:
        return None

    text, matched = _arrange_article(raw, ref.fragment)
    if not text:
        return None

    url = _page_url(page, ref.lang)
    if matched:
        url += "#" + matched.replace(" ", "_")
    revisions = page.get("revisions") or []
    edited = (revisions[0].get("timestamp") or "")[:10] if revisions else ""

    via = "Wikipedia API"
    if _is_disambiguation(page):
        via += " — disambiguation page (the topics below share this name)"
    if edited:
        via += f"; last edited {edited}"
    if matched:
        via += f"; linked section \"{matched}\" shown first"

    cleaned = f"{text}\n\n{attribution(_page_url(page, ref.lang))}"
    return {
        "url":               url,
        "title":             page.get("title") or ref.title,
        "author":            "Wikipedia contributors",
        "date_published":    "",
        "cleaned_text":      cleaned,
        "word_count":        len(text.split()),
        "via":               via,
    }
