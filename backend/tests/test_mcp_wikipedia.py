"""
Tests for mcp_server/wikipedia.py — wikipedia_search / wikipedia_read over
Wikipedia's free public Action API (§14.21). Every HTTP call is mocked;
no test touches the network.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from localist.mcp_server import wikipedia
from localist.mcp_server.main import mcp as mcp_app


_API = "https://en.wikipedia.org/w/api.php"


def _response(body: dict | list, status_code: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code, json=body, headers=headers or {},
        request=httpx.Request("GET", _API),
    )


def _search_body(pages: list[dict]) -> dict:
    return {"batchcomplete": True, "query": {"pages": pages}}


def _page(title: str, index: int, extract: str = "", description: str = "",
          disambiguation: bool = False, **extra) -> dict:
    page = {
        "pageid": index, "ns": 0, "title": title, "index": index,
        "extract": extract, "description": description,
        "canonicalurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }
    if disambiguation:
        page["pageprops"] = {"disambiguation": ""}
    page.update(extra)
    return page


@pytest.fixture(autouse=True)
def _default_lang(monkeypatch):
    monkeypatch.delenv("WIKIPEDIA_LANG", raising=False)


@pytest.fixture
def no_sleep():
    with patch.object(wikipedia.asyncio, "sleep", AsyncMock()) as fake:
        yield fake


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

class TestRequestShape:
    def test_user_agent_names_client_and_public_repo(self):
        fake_get = AsyncMock(return_value=_response(_search_body([_page("Apple M4", 1, "x")])))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            asyncio.run(wikipedia.wikipedia_search("apple m4"))
        headers = fake_get.call_args.kwargs["headers"]
        assert headers["User-Agent"].startswith(
            "Localist (https://github.com/mikelowcode/localist) httpx/"
        )
        assert "@" not in headers["User-Agent"]

    def test_search_uses_action_api_generator_search(self):
        fake_get = AsyncMock(return_value=_response(_search_body([_page("Apple M4", 1, "x")])))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            asyncio.run(wikipedia.wikipedia_search("apple m4"))
        url = fake_get.call_args.args[0]
        params = fake_get.call_args.kwargs["params"]
        assert url == _API
        assert params["action"] == "query"
        assert params["generator"] == "search"
        assert params["gsrsearch"] == "apple m4"
        assert params["explaintext"] == "1"
        assert params["formatversion"] == "2"

    def test_wikipedia_lang_selects_edition(self, monkeypatch):
        monkeypatch.setenv("WIKIPEDIA_LANG", "de")
        fake_get = AsyncMock(return_value=_response(_search_body([_page("Berlin", 1, "x")])))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(wikipedia.wikipedia_search("berlin"))
        assert fake_get.call_args.args[0] == "https://de.wikipedia.org/w/api.php"
        assert result["best_match"]["title"] == "Berlin"

    @pytest.mark.parametrize("bad", ["english", "e n", "../x", "EN!", ""])
    def test_invalid_lang_falls_back_to_en(self, monkeypatch, bad):
        monkeypatch.setenv("WIKIPEDIA_LANG", bad)
        assert wikipedia._lang() == "en"

    def test_lang_is_case_normalised(self, monkeypatch):
        monkeypatch.setenv("WIKIPEDIA_LANG", "FR")
        assert wikipedia._lang() == "fr"


# ---------------------------------------------------------------------------
# wikipedia_search
# ---------------------------------------------------------------------------

class TestWikipediaSearch:
    def test_best_match_first_with_url_lines_and_attribution(self):
        pages = [
            _page("Apple M3", 2, "The Apple M3 is a chip.", "Chip"),
            _page("Apple M4", 1, "The Apple M4 is a series of ARM-based chips.",
                  "System-on-a-chip designed by Apple Inc."),
        ]
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_search_body(pages)))):
            result = asyncio.run(wikipedia.wikipedia_search("apple m4"))

        assert result["is_miss"] is False
        assert result["result_count"] == 2
        best = result["best_match"]
        assert best["title"] == "Apple M4"
        assert best["url"] == "https://en.wikipedia.org/wiki/Apple_M4"
        assert best["disambiguation"] is False
        assert best["summary"].startswith("The Apple M4")

        text = result["result_text"]
        assert text.startswith("• Apple M4 — Wikipedia\n")
        # description's own trailing period isn't doubled
        assert "designed by Apple Inc. The Apple M4" in text
        assert "[https://en.wikipedia.org/wiki/Apple_M4]" in text
        assert "[https://en.wikipedia.org/wiki/Apple_M3]" in text
        assert text.index("Apple M4 —") < text.index("Apple M3 —")
        assert text.endswith(
            "Source: Wikipedia — https://en.wikipedia.org/wiki/Apple_M4 (CC BY-SA 4.0)"
        )

    def test_disambiguation_best_match_is_flagged(self):
        pages = [
            _page("Mercury", 1, "Mercury most commonly refers to:\n\nMercury (planet)",
                  "Topics referred to by the same term", disambiguation=True),
            _page("Mercury (planet)", 2, "Mercury is the first planet from the Sun.", "Planet"),
        ]
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_search_body(pages)))):
            result = asyncio.run(wikipedia.wikipedia_search("mercury"))

        assert result["best_match"]["disambiguation"] is True
        assert "Disambiguation page" in result["result_text"]
        # a disambiguation page's generic description isn't prepended
        assert "Topics referred to by the same term" not in result["result_text"]
        assert "Mercury (planet) — Wikipedia" in result["result_text"]

    def test_parenthesised_title_url_survives_dispatcher_url_regex(self):
        from localist.mcp_tool_dispatcher import _URL_RE

        pages = [_page("Mercury (planet)", 1, "Mercury is the first planet.")]
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_search_body(pages)))):
            result = asyncio.run(wikipedia.wikipedia_search("mercury planet"))

        expected = "https://en.wikipedia.org/wiki/Mercury_%28planet%29"
        assert result["best_match"]["url"] == expected
        assert _URL_RE.findall(result["result_text"])[0] == expected

    def test_other_hits_get_short_summaries(self):
        long_text = "word " * 400
        pages = [_page("A", 1, long_text), _page("B", 2, long_text)]
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_search_body(pages)))):
            result = asyncio.run(wikipedia.wikipedia_search("a"))
        block_a, block_b = result["result_text"].split("\n\n")[:2]
        assert len(block_a) > 1000
        assert len(block_b) < 400

    def test_no_pages_is_miss(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response({"batchcomplete": True}))):
            result = asyncio.run(wikipedia.wikipedia_search("zzqxj nothing"))
        assert result == {
            "query": "zzqxj nothing", "result_text": "", "result_count": 0,
            "is_miss": True, "best_match": None,
        }

    def test_empty_query_raises_without_network_call(self):
        fake_get = AsyncMock()
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="ERROR: wikipedia_search failed — empty query"):
                asyncio.run(wikipedia.wikipedia_search("   "))
        fake_get.assert_not_called()


# ---------------------------------------------------------------------------
# wikipedia_read
# ---------------------------------------------------------------------------

_ARTICLE = (
    "The Apple M4 is a series of chips.\n\n\n"
    "== Design ==\n\nBuilt on TSMC N3E.\n\n"
    "=== CPU ===\n\nTen cores.\n\n"
    "== References =="
)


class TestWikipediaRead:
    def _read(self, page: dict):
        body = {"query": {"pages": [page]}}
        fake_get = AsyncMock(return_value=_response(body))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            return asyncio.run(wikipedia.wikipedia_read("apple m4")), fake_get

    def test_article_intro_sections_content_references(self):
        page = _page(
            "Apple M4", 1, _ARTICLE, "System-on-a-chip",
            extlinks=[
                {"url": "https://www.apple.com/newsroom/m4/"},
                {"url": "https://web.archive.org/web/2024/https://www.apple.com/newsroom/m4/"},
                {"url": "https://en.wikipedia.org/wiki/TSMC"},
                {"url": "//www.theverge.com/m4"},
                {"url": "https://www.apple.com/newsroom/m4"},   # dup (trailing slash)
                {"url": "ftp://example.org/file"},
            ],
        )
        result, fake_get = self._read(page)

        params = fake_get.call_args.kwargs["params"]
        assert params["titles"] == "apple m4"
        assert params["redirects"] == "1"
        assert "extlinks" in params["prop"]

        assert result["title"] == "Apple M4"
        assert result["kind"] == "article"
        assert result["intro"] == "The Apple M4 is a series of chips."
        assert result["sections"] == ["Design", "CPU", "References"]
        assert "== Design ==" in result["content"]
        assert result["truncated"] is False
        assert result["references"] == [
            "https://www.apple.com/newsroom/m4/",
            "https://www.theverge.com/m4",
        ]
        assert result["attribution"] == (
            "Source: Wikipedia — https://en.wikipedia.org/wiki/Apple_M4 (CC BY-SA 4.0)"
        )

    def test_long_article_is_truncated(self):
        page = _page("Long", 1, "intro\n\n== A ==\n\n" + "word " * 3000)
        result, _ = self._read(page)
        assert result["truncated"] is True
        assert len(result["content"]) <= wikipedia._READ_CONTENT_CHARS + 1
        assert result["content"].endswith("…")

    def test_disambiguation_kind(self):
        page = _page("Mercury", 1, "Mercury may refer to:", disambiguation=True)
        result, _ = self._read(page)
        assert result["kind"] == "disambiguation"

    def test_missing_page_raises_not_found(self):
        page = {"ns": 0, "title": "Nope Nope", "missing": True}
        with pytest.raises(ValueError, match="ERROR: wikipedia_read — 'apple m4' not found"):
            self._read(page)

    def test_empty_title_raises_without_network_call(self):
        fake_get = AsyncMock()
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="empty title"):
                asyncio.run(wikipedia.wikipedia_read(""))
        fake_get.assert_not_called()

    def test_reference_cap(self):
        links = [{"url": f"https://site{i}.example.com/a"} for i in range(30)]
        result, _ = self._read(_page("X", 1, "x", extlinks=links))
        assert len(result["references"]) == wikipedia._MAX_REFERENCES


# ---------------------------------------------------------------------------
# Errors and rate limiting
# ---------------------------------------------------------------------------

class TestErrorsAndRateLimit:
    def test_transport_error_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: wikipedia_search failed —"):
                asyncio.run(wikipedia.wikipedia_search("q"))

    def test_http_500_raises_clean_message(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response({}, 500))):
            with pytest.raises(ValueError, match="ERROR: wikipedia_read failed —"):
                asyncio.run(wikipedia.wikipedia_read("q"))

    def test_api_error_body_raises(self):
        body = {"error": {"code": "badvalue", "info": "Unrecognized value"}}
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            with pytest.raises(ValueError, match="Unrecognized value"):
                asyncio.run(wikipedia.wikipedia_search("q"))

    def test_non_json_body_raises_clean_message(self):
        resp = httpx.Response(200, text="<html>oops</html>", request=httpx.Request("GET", _API))
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=resp)):
            with pytest.raises(ValueError, match="ERROR: wikipedia_search failed —"):
                asyncio.run(wikipedia.wikipedia_search("q"))

    def test_429_short_retry_after_retries_once(self, no_sleep):
        ok = _response(_search_body([_page("A", 1, "x")]))
        limited = _response({}, 429, {"Retry-After": "2"})
        fake_get = AsyncMock(side_effect=[limited, ok])
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(wikipedia.wikipedia_search("a"))
        assert result["is_miss"] is False
        assert fake_get.call_count == 2
        no_sleep.assert_awaited_once_with(2.0)

    def test_429_without_retry_after_waits_five_seconds(self, no_sleep):
        ok = _response(_search_body([_page("A", 1, "x")]))
        fake_get = AsyncMock(side_effect=[_response({}, 429), ok])
        with patch.object(httpx.AsyncClient, "get", fake_get):
            asyncio.run(wikipedia.wikipedia_search("a"))
        no_sleep.assert_awaited_once_with(5.0)

    def test_429_long_retry_after_gives_up_without_waiting(self, no_sleep):
        fake_get = AsyncMock(return_value=_response({}, 429, {"Retry-After": "120"}))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="rate limit"):
                asyncio.run(wikipedia.wikipedia_search("a"))
        assert fake_get.call_count == 1
        no_sleep.assert_not_awaited()

    def test_429_twice_gives_up(self, no_sleep):
        limited = _response({}, 429, {"Retry-After": "1"})
        fake_get = AsyncMock(side_effect=[limited, limited])
        with patch.object(httpx.AsyncClient, "get", fake_get):
            with pytest.raises(ValueError, match="ERROR: wikipedia_search failed — Wikipedia rate limit"):
                asyncio.run(wikipedia.wikipedia_search("a"))
        assert fake_get.call_count == 2

    def test_concurrency_capped_at_three(self):
        in_flight = 0
        peak = 0

        async def slow_get(*args, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _response(_search_body([_page("A", 1, "x")]))

        async def run_many():
            await asyncio.gather(*(wikipedia.wikipedia_search(f"q{i}") for i in range(8)))

        with patch.object(httpx.AsyncClient, "get", side_effect=slow_get):
            asyncio.run(run_many())
            asyncio.run(run_many())   # a second event loop must not break the limiter
        assert peak == 3


class TestFilterReferences:
    def test_skips_subdomains_of_wikimedia_hosts(self):
        urls = [
            "https://commons.wikimedia.org/x",
            "https://de.wikipedia.org/wiki/X",
            "https://www.wikidata.org/wiki/Q1",
            "https://notwikipedia.org.example.com/ok",
        ]
        assert wikipedia.filter_references(urls) == ["https://notwikipedia.org.example.com/ok"]


# ---------------------------------------------------------------------------
# In-process MCP registration
# ---------------------------------------------------------------------------

async def _call_tool(name: str, arguments: dict) -> tuple[str, bool]:
    async with create_connected_server_and_client_session(mcp_app) as session:
        result = await session.call_tool(name, arguments)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        return text, result.isError


class TestMCPRegistration:
    def test_wikipedia_search_tool_success(self):
        body = _search_body([_page("Apple M4", 1, "The Apple M4 is a chip.")])
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            text, is_error = asyncio.run(_call_tool("wikipedia_search", {"query": "apple m4"}))
        assert is_error is False
        data = json.loads(text)
        assert data["best_match"]["title"] == "Apple M4"

    def test_wikipedia_read_tool_success(self):
        body = {"query": {"pages": [_page("Apple M4", 1, _ARTICLE)]}}
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            text, is_error = asyncio.run(_call_tool("wikipedia_read", {"title": "Apple M4"}))
        assert is_error is False
        assert json.loads(text)["sections"] == ["Design", "CPU", "References"]

    def test_wikipedia_read_missing_surfaces_as_is_error(self):
        body = {"query": {"pages": [{"ns": 0, "title": "X", "missing": True}]}}
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            text, is_error = asyncio.run(_call_tool("wikipedia_read", {"title": "X"}))
        assert is_error is True
        assert "not found" in text


# ---------------------------------------------------------------------------
# Phase 3 — wikipedia.org links read through the API (fetch_url resolver)
# ---------------------------------------------------------------------------

from localist.mcp_server import url_fetch  # noqa: E402


class TestParseArticleUrl:

    @pytest.mark.parametrize("url, lang, title, fragment", [
        ("https://en.wikipedia.org/wiki/Transistor", "en", "Transistor", ""),
        ("https://en.m.wikipedia.org/wiki/Transistor", "en", "Transistor", ""),
        ("http://de.wikipedia.org/wiki/Berlin", "de", "Berlin", ""),
        ("https://en.wikipedia.org/wiki/Mercury_(planet)", "en", "Mercury (planet)", ""),
        ("https://en.wikipedia.org/wiki/Mercury_%28planet%29", "en", "Mercury (planet)", ""),
        ("https://en.wikipedia.org/wiki/Transistor#History", "en", "Transistor", "History"),
        ("https://en.wikipedia.org/wiki/Apple_M4#Design_and_CPU", "en", "Apple M4", "Design and CPU"),
        ("https://en.wikipedia.org/w/index.php?title=Transistor", "en", "Transistor", ""),
        ("https://zh-yue.wikipedia.org/wiki/X", "zh-yue", "X", ""),
        ("https://EN.Wikipedia.org/wiki/Caf%C3%A9", "en", "Café", ""),
    ])
    def test_article_links(self, url, lang, title, fragment):
        ref = wikipedia.parse_article_url(url)
        assert ref == wikipedia.ArticleRef(lang=lang, title=title, fragment=fragment)

    @pytest.mark.parametrize("url", [
        "https://example.com/wiki/Transistor",
        "https://wikipedia.org/wiki/Transistor",                # portal, no language
        "https://en.wikipedia.org/",
        "https://en.wikipedia.org/wiki/",
        "https://en.wikipedia.org/wiki/Special:Random",
        "https://en.wikipedia.org/wiki/File:Transistor.jpg",
        "https://en.wikipedia.org/wiki/Talk:Transistor",
        "https://en.wikipedia.org/wiki/Category:Transistors",
        "https://en.wikipedia.org/w/index.php?title=Transistor&oldid=123",
        "https://en.wikipedia.org/w/index.php?title=Transistor&action=history",
        "https://en.wikipedia.org/w/index.php?title=Transistor&diff=1&oldid=2",
        "https://commons.wikimedia.org/wiki/Main_Page",
        "https://en.wikipedia.org.evil.example/wiki/Transistor",
        "ftp://en.wikipedia.org/wiki/Transistor",
        "not a url",
    ])
    def test_not_article_links(self, url):
        assert wikipedia.parse_article_url(url) is None


_FULL_ARTICLE = (
    "Lead paragraph.\n\n"
    "== History ==\n\nHistory text.\n\n"
    "=== Early days ===\n\nEarly text.\n\n"
    "== Design ==\n\nDesign text.\n\n"
    "== See also ==\n\nOther page\n\n"
    "== References ==\n\n"
    "=== Citations ===\n\ncite\n\n"
    "== External links =="
)


class TestArrangeArticle:
    def test_drops_reference_sections_and_their_subsections(self):
        text, matched = wikipedia._arrange_article(_FULL_ARTICLE, "")
        assert matched == ""
        assert "See also" not in text
        assert "References" not in text
        assert "Citations" not in text and "cite" not in text
        assert "External links" not in text
        assert text.startswith("Lead paragraph.")
        assert text.index("== History ==") < text.index("== Design ==")

    def test_fragment_section_moves_after_lead_with_subsections(self):
        text, matched = wikipedia._arrange_article(_FULL_ARTICLE, "design")
        assert matched == "Design"
        assert text.index("Lead paragraph.") < text.index("== Design ==") < text.index("== History ==")
        assert text.index("== History ==") < text.index("=== Early days ===")

    def test_fragment_with_subsections_keeps_them_together(self):
        text, matched = wikipedia._arrange_article(_FULL_ARTICLE, "History")
        assert matched == "History"
        assert text.index("== History ==") < text.index("=== Early days ===") < text.index("== Design ==")

    def test_german_reference_sections_dropped(self):
        text, _ = wikipedia._arrange_article(
            "Lead.\n\n== Geschichte ==\n\nText.\n\n== Literatur ==\n\nBuch\n\n"
            "== Weblinks ==\n\n== Einzelnachweise ==", "",
        )
        assert text == "Lead.\n\n== Geschichte ==\n\nText."

    def test_trailing_heading_only_sections_trimmed_any_language(self):
        text, _ = wikipedia._arrange_article(
            "Lead.\n\n== Storia ==\n\nTesto.\n\n== Collegamenti esterni ==\n\n== Note ==", "",
        )
        assert text == "Lead.\n\n== Storia ==\n\nTesto."

    def test_unknown_fragment_leaves_order(self):
        text, matched = wikipedia._arrange_article(_FULL_ARTICLE, "Nope")
        assert matched == ""
        assert text.index("== History ==") < text.index("== Design ==")


def _fetch_page(**extra) -> dict:
    page = _page("Transistor", 1, _FULL_ARTICLE, "Semiconductor device",
                 revisions=[{"timestamp": "2026-11-14T07:36:07Z"}])
    page.update(extra)
    return {"query": {"pages": [page]}}


class TestFetchArticle:
    def test_returns_fetch_url_shape(self):
        fake_get = AsyncMock(return_value=_response(_fetch_page()))
        ref = wikipedia.ArticleRef("en", "Transistor", "")
        with patch.object(httpx.AsyncClient, "get", fake_get):
            result = asyncio.run(wikipedia.fetch_article(ref, timeout=7.0))

        assert fake_get.call_args.args[0] == "https://en.wikipedia.org/w/api.php"
        params = fake_get.call_args.kwargs["params"]
        assert params["titles"] == "Transistor" and params["redirects"] == "1"
        assert result["url"] == "https://en.wikipedia.org/wiki/Transistor"
        assert result["title"] == "Transistor"
        assert result["author"] == "Wikipedia contributors"
        assert result["via"] == "Wikipedia API; last edited 2026-11-14"
        assert result["cleaned_text"].startswith("Lead paragraph.")
        assert "References" not in result["cleaned_text"]
        assert result["cleaned_text"].endswith(
            "Source: Wikipedia — https://en.wikipedia.org/wiki/Transistor (CC BY-SA 4.0)"
        )
        assert result["word_count"] > 0

    def test_link_language_wins_over_wikipedia_lang(self, monkeypatch):
        monkeypatch.setenv("WIKIPEDIA_LANG", "en")
        fake_get = AsyncMock(return_value=_response(_fetch_page()))
        with patch.object(httpx.AsyncClient, "get", fake_get):
            asyncio.run(wikipedia.fetch_article(wikipedia.ArticleRef("fr", "Transistor", "")))
        assert fake_get.call_args.args[0] == "https://fr.wikipedia.org/w/api.php"

    def test_fragment_goes_into_url_and_via(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_fetch_page()))):
            result = asyncio.run(wikipedia.fetch_article(wikipedia.ArticleRef("en", "Transistor", "Design")))
        assert result["url"] == "https://en.wikipedia.org/wiki/Transistor#Design"
        assert 'linked section "Design" shown first' in result["via"]

    def test_disambiguation_noted_in_via(self):
        body = _fetch_page(pageprops={"disambiguation": ""})
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            result = asyncio.run(wikipedia.fetch_article(wikipedia.ArticleRef("en", "Transistor", "")))
        assert "disambiguation page" in result["via"]

    @pytest.mark.parametrize("body", [
        {"query": {"pages": [{"ns": 0, "title": "X", "missing": True}]}},
        {"query": {"pages": [{"ns": 0, "title": "X", "extract": "   "}]}},
        {"query": {"pages": []}},
        {"error": {"info": "bad"}},
    ])
    def test_nothing_readable_returns_none(self, body):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(body))):
            assert asyncio.run(wikipedia.fetch_article(wikipedia.ArticleRef("en", "X", ""))) is None

    def test_transport_error_returns_none(self):
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("x"))):
            assert asyncio.run(wikipedia.fetch_article(wikipedia.ArticleRef("en", "X", ""))) is None


class TestFetchUrlWikipediaResolver:
    def test_wikipedia_link_uses_api_not_scrape(self):
        scrape = AsyncMock()
        with patch.object(url_fetch, "_fetch_and_extract", scrape), \
             patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_response(_fetch_page()))):
            result = asyncio.run(url_fetch.fetch_url("https://en.m.wikipedia.org/wiki/Transistor"))
        scrape.assert_not_called()
        assert result["via"].startswith("Wikipedia API")
        assert result["fetch_duration_ms"] >= 0
        assert result["url"] == "https://en.wikipedia.org/wiki/Transistor"

    def test_api_failure_falls_through_to_normal_fetch(self):
        scraped = {"url": "https://en.wikipedia.org/wiki/Transistor", "title": "Transistor - Wikipedia",
                   "author": "", "date_published": "", "cleaned_text": "scraped", "word_count": 1,
                   "fetch_duration_ms": 1.0}
        scrape = AsyncMock(return_value=scraped)
        with patch.object(url_fetch, "_fetch_and_extract", scrape), \
             patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("x"))):
            result = asyncio.run(url_fetch.fetch_url("https://en.wikipedia.org/wiki/Transistor"))
        scrape.assert_awaited_once()
        assert result is scraped

    def test_non_article_wikipedia_link_skips_api(self):
        scrape = AsyncMock(return_value={"cleaned_text": "x"})
        api = AsyncMock()
        with patch.object(url_fetch, "_fetch_and_extract", scrape), \
             patch.object(httpx.AsyncClient, "get", api):
            asyncio.run(url_fetch.fetch_url("https://en.wikipedia.org/wiki/Special:Random"))
        api.assert_not_called()
        scrape.assert_awaited_once()
