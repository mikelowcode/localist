"""
Phase 2 of the Wikipedia integration (§14.21): Planner routing
(P3-wikipedia keyword gate, "/wikipedia" P0 slash command,
LOCALIST_WIKIPEDIA_ENABLED) and MCPToolDispatcher._run_wikipedia_search
(search -> read best match, web_search fallback). MCP calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from localist.mcp_tool_dispatcher import MCPToolDispatcher, _WIKIPEDIA_READ_PROMPT_CHARS
from localist.planner import Planner


def make_runtime(infer_return="no"):
    rt = MagicMock()
    rt.infer.return_value = infer_return
    rt.embed.return_value = [0.0] * 768
    return rt


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("LOCALIST_WIKIPEDIA_ENABLED", raising=False)
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "test-placeholder")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlannerWikipediaRoute:

    @pytest.mark.parametrize("instruction", [
        "what does wikipedia say about the transistor",
        "Search Wikipedia for Ada Lovelace",
        "wikipedia's article on the Apollo program",
        "tell me about the Battle of Hastings according to wikipedia",
    ])
    def test_naming_wikipedia_routes_to_wikipedia_search(self, instruction):
        plan = Planner(runtime=make_runtime()).route(instruction, context={})
        assert plan.tools_to_call == ["wikipedia_search"]
        assert plan.priority == 3
        assert plan.tool_signal_source == "keyword"
        assert plan.agent == "conversational_agent"

    @pytest.mark.parametrize("instruction", [
        "what does my wiki say about the transistor",
        "search the wiki for my notes on Ada Lovelace",
        "tell me about wikis in general",
    ])
    def test_bare_wiki_never_routes_to_wikipedia(self, instruction):
        plan = Planner(runtime=make_runtime()).route(instruction, context={})
        assert "wikipedia_search" not in plan.tools_to_call

    def test_pasted_wikipedia_url_is_a_fetch_not_a_search(self):
        plan = Planner(runtime=make_runtime()).route(
            "summarize https://en.wikipedia.org/wiki/Transistor", context={},
        )
        assert "wikipedia_search" not in plan.tools_to_call

    def test_news_about_wikipedia_stays_news(self):
        plan = Planner(runtime=make_runtime()).route(
            "latest news about wikipedia", context={},
        )
        assert plan.tools_to_call == ["news_search"]

    def test_github_repo_for_wikipedia_stays_github(self):
        plan = Planner(runtime=make_runtime()).route(
            "find a github repo for a wikipedia client", context={},
        )
        assert plan.tools_to_call == ["github_search"]

    @pytest.mark.parametrize("value", ["false", "0", "off", "NO"])
    def test_flag_off_disables_keyword_route(self, monkeypatch, value):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENABLED", value)
        plan = Planner(runtime=make_runtime()).route(
            "what does wikipedia say about the transistor", context={},
        )
        assert "wikipedia_search" not in plan.tools_to_call

    def test_flag_unset_or_true_is_on(self, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENABLED", "true")
        assert Planner._wikipedia_enabled() is True
        monkeypatch.delenv("LOCALIST_WIKIPEDIA_ENABLED")
        assert Planner._wikipedia_enabled() is True


class TestPlannerWikipediaSlashCommand:

    @pytest.mark.parametrize("instruction", [
        "/wikipedia the transistor", "/Wikipedia Ada Lovelace", "/wikipedia",
    ])
    def test_slash_wikipedia_forces_tool(self, instruction):
        plan = Planner(runtime=make_runtime()).route(instruction, context={})
        assert plan.tools_to_call == ["wikipedia_search"]
        assert plan.priority == 0
        assert plan.tool_signal_source == "slash_command"

    def test_slash_wikipedia_ignores_the_flag(self, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENABLED", "false")
        plan = Planner(runtime=make_runtime()).route("/wikipedia the transistor", context={})
        assert plan.tools_to_call == ["wikipedia_search"]

    def test_mid_sentence_slash_does_not_match_p0(self):
        plan = Planner(runtime=make_runtime()).route(
            "can I type /wikipedia somewhere", context={},
        )
        assert plan.priority != 0


# ---------------------------------------------------------------------------
# Dispatcher — query derivation
# ---------------------------------------------------------------------------

class TestDeriveWikipediaQuery:

    @pytest.mark.parametrize("instruction, expected", [
        ("what does wikipedia say about the transistor", "the transistor"),
        ("What does Wikipedia say about Ada Lovelace?", "Ada Lovelace"),
        ("search wikipedia for the Apollo program", "the Apollo program"),
        ("wikipedia's article on the Battle of Hastings", "the Battle of Hastings"),
        ("the wikipedia page for Rust (programming language)", "Rust (programming language)"),
        ("tell me about Marie Curie according to wikipedia", "Marie Curie"),
        ("who was Alan Turing? check wikipedia", "Alan Turing"),
        ("/wikipedia the transistor", "the transistor"),
        ("/Wikipedia   quantum computing  ", "quantum computing"),
        ("look up photosynthesis on wikipedia", "photosynthesis"),
        ("/wikipedia", ""),
        ("wikipedia", ""),
    ])
    def test_strips_wikipedia_framing(self, instruction, expected):
        assert MCPToolDispatcher._derive_wikipedia_query(instruction, {}) == expected

    def test_context_query_wins(self):
        assert MCPToolDispatcher._derive_wikipedia_query(
            "whatever wikipedia", {"wikipedia_query": "Transistor"}
        ) == "Transistor"


# ---------------------------------------------------------------------------
# Dispatcher — _run_wikipedia_search
# ---------------------------------------------------------------------------

_FAKE_SESSION = object()


@pytest.fixture()
def dispatcher(tmp_path):
    rt = MagicMock(spec=["infer"])
    rt.infer.return_value = "• Fallback result."
    d = MCPToolDispatcher(runtime=rt, project_root=tmp_path)
    with patch.object(MCPToolDispatcher, "_open_session", return_value=_FAKE_SESSION):
        yield d


def _search_payload(title="Transistor", disambiguation=False, is_miss=False):
    if is_miss:
        return {"query": "q", "result_text": "", "result_count": 0,
                "is_miss": True, "best_match": None}
    url = f"https://en.wikipedia.org/wiki/{title}"
    return {
        "query": "q",
        "result_text": f"• {title} — Wikipedia\n  Intro.\n  [{url}]\n\nSource: Wikipedia — {url} (CC BY-SA 4.0)",
        "result_count": 1, "is_miss": False,
        "best_match": {"title": title, "url": url, "description": "",
                       "summary": "Intro.", "disambiguation": disambiguation},
    }


def _read_payload(title="Transistor", content="A transistor is a device.", refs=None):
    url = f"https://en.wikipedia.org/wiki/{title}"
    return {
        "title": title, "url": url, "description": "", "kind": "article",
        "intro": content, "sections": [], "content": content, "truncated": False,
        "references": refs if refs is not None else [
            f"https://example{i}.com/ref" for i in range(8)
        ],
        "attribution": f"Source: Wikipedia — {url} (CC BY-SA 4.0)",
    }


def _dispatch(dispatcher, fake_call, instruction="what does wikipedia say about the transistor"):
    with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
        return dispatcher.dispatch(
            tools_to_call=["wikipedia_search"], instruction=instruction, context={},
        )


class TestRunWikipediaSearch:

    def test_hit_searches_then_reads_best_match(self, dispatcher):
        calls: list[tuple[str, dict]] = []

        async def fake_call(session, name, arguments):
            calls.append((name, arguments))
            if name == "wikipedia_search":
                return json.dumps(_search_payload()), False
            assert name == "wikipedia_read"
            return json.dumps(_read_payload()), False

        results = _dispatch(dispatcher, fake_call)

        assert calls == [
            ("wikipedia_search", {"query": "the transistor"}),
            ("wikipedia_read", {"title": "Transistor"}),
        ]
        assert [r.tool_name for r in results] == ["wikipedia_search", "wikipedia_read"]
        assert all(r.success for r in results)
        assert results[0].workflow_id and results[0].workflow_id == results[1].workflow_id

        read = results[1].result
        assert read.startswith("Wikipedia article: Transistor\n[https://en.wikipedia.org/wiki/Transistor]")
        assert "A transistor is a device." in read
        assert "Cited sources:" in read
        assert read.count("- https://example") == 5
        assert read.endswith("(CC BY-SA 4.0)")

    def test_read_body_capped_for_the_prompt(self, dispatcher):
        long_body = "word " * 3000

        async def fake_call(session, name, arguments):
            if name == "wikipedia_search":
                return json.dumps(_search_payload()), False
            return json.dumps(_read_payload(content=long_body, refs=[])), False

        results = _dispatch(dispatcher, fake_call)
        read = results[1].result
        assert "Cited sources:" not in read
        body = read.split("\n\n", 1)[1].rsplit("\n\nSource:", 1)[0]
        assert len(body) <= _WIKIPEDIA_READ_PROMPT_CHARS + 1
        assert body.endswith("…")

    def test_disambiguation_best_match_is_not_read(self, dispatcher):
        names: list[str] = []

        async def fake_call(session, name, arguments):
            names.append(name)
            return json.dumps(_search_payload("Mercury", disambiguation=True)), False

        results = _dispatch(dispatcher, fake_call, "what does wikipedia say about mercury")
        assert names == ["wikipedia_search"]
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].workflow_id is None

    def test_failed_read_keeps_search_alone(self, dispatcher):
        async def fake_call(session, name, arguments):
            if name == "wikipedia_search":
                return json.dumps(_search_payload()), False
            return "Error executing tool wikipedia_read: ERROR: wikipedia_read failed — boom", True

        results = _dispatch(dispatcher, fake_call)
        assert [r.tool_name for r in results] == ["wikipedia_search"]
        assert results[0].success is True

    def test_miss_falls_back_to_web_search(self, dispatcher):
        seen: list[tuple[str, str]] = []

        async def fake_call(session, name, arguments):
            seen.append((name, arguments.get("query", "")))
            if name == "wikipedia_search":
                return json.dumps(_search_payload(is_miss=True)), False
            assert name == "web_search"
            return json.dumps({
                "query": arguments["query"], "result_text": "• Web result", "result_count": 1,
            }), False

        results = _dispatch(dispatcher, fake_call, "search wikipedia for zylophonic widgets")

        assert [r.tool_name for r in results] == [
            "wikipedia_search", "wikipedia_search:web_fallback",
        ]
        assert results[0].success is False
        assert "no Wikipedia article matched" in results[0].result
        assert results[1].success is True
        assert results[1].result == "• Web result"
        assert seen == [
            ("wikipedia_search", "zylophonic widgets"),
            ("web_search", "zylophonic widgets"),
        ]
        assert results[0].workflow_id == results[1].workflow_id

    def test_tool_error_falls_back_to_web_search(self, dispatcher):
        async def fake_call(session, name, arguments):
            if name == "wikipedia_search":
                return "Error executing tool wikipedia_search: ERROR: wikipedia_search failed — Wikipedia rate limit", True
            return json.dumps({"query": "q", "result_text": "• Web result", "result_count": 1}), False

        results = _dispatch(dispatcher, fake_call)
        assert results[0].result == "ERROR: wikipedia_search failed — Wikipedia rate limit"
        assert results[1].tool_name == "wikipedia_search:web_fallback"
        assert results[1].success is True

    def test_mcp_call_raising_falls_back(self, dispatcher):
        async def fake_call(session, name, arguments):
            if name == "wikipedia_search":
                raise ConnectionRefusedError("refused")
            return json.dumps({"query": "q", "result_text": "• Web result", "result_count": 1}), False

        results = _dispatch(dispatcher, fake_call)
        assert results[0].success is False
        assert "localist-mcp unreachable" in results[0].result
        assert results[1].tool_name == "wikipedia_search:web_fallback"

    def test_no_topic_makes_no_calls(self, dispatcher):
        async def fake_call(session, name, arguments):
            raise AssertionError("no MCP call expected")

        results = _dispatch(dispatcher, fake_call, "/wikipedia")
        assert len(results) == 1
        assert results[0].success is False
        assert "no topic given" in results[0].result


# ---------------------------------------------------------------------------
# Phase 3 — _URL_RE keeps balanced parentheses (§14.21)
# ---------------------------------------------------------------------------

from localist.mcp_tool_dispatcher import _URL_RE  # noqa: E402


class TestUrlRegexParentheses:

    @pytest.mark.parametrize("text, expected", [
        ("[https://en.wikipedia.org/wiki/Mercury_(planet)]",
         "https://en.wikipedia.org/wiki/Mercury_(planet)"),
        ("see https://en.wikipedia.org/wiki/Rust_(programming_language) for more",
         "https://en.wikipedia.org/wiki/Rust_(programming_language)"),
        ("(see https://en.wikipedia.org/wiki/Mercury_(planet))",
         "https://en.wikipedia.org/wiki/Mercury_(planet)"),
        ("[Mercury](https://en.wikipedia.org/wiki/Mercury_(planet))",
         "https://en.wikipedia.org/wiki/Mercury_(planet)"),
        ("See the pricing page (https://vendor.example/pricing) for details.",
         "https://vendor.example/pricing"),
        ("[https://www.t-mobile.com/cell-phones/brand/apple]",
         "https://www.t-mobile.com/cell-phones/brand/apple"),
        ("https://a.example/x(1)(2)/y", "https://a.example/x(1)(2)/y"),
        ("https://a.example/broken(path", "https://a.example/broken"),
    ])
    def test_balanced_parentheses_kept_wrapping_ones_dropped(self, text, expected):
        assert _URL_RE.search(text).group(0) == expected

    def test_extract_first_url_keeps_wikipedia_title(self, tmp_path):
        d = MCPToolDispatcher(runtime=MagicMock(), project_root=tmp_path)
        text = "• Mercury (planet) — en.wikipedia.org\n  body\n  [https://en.wikipedia.org/wiki/Mercury_(planet)]"
        assert d._extract_first_url(text, set()) == "https://en.wikipedia.org/wiki/Mercury_(planet)"


# ---------------------------------------------------------------------------
# Phase 4 — confident-match rule + web_search background (flag, default off)
# ---------------------------------------------------------------------------

from localist import wikipedia_match  # noqa: E402


class TestWikipediaMatchRule:

    @pytest.mark.parametrize("query, title", [
        ("who was Ada Lovelace", "Ada Lovelace"),
        ("Mercury planet facts", "Mercury (planet)"),
        ("Rust programming language", "Rust (programming language)"),
        ("what is CRISPR", "CRISPR"),
        ("explain the café culture", "Café culture"),          # accent folding
        ("Ursula K. Le Guin", "Ursula K. Le Guin"),
    ])
    def test_attaches(self, query, title):
        d = wikipedia_match.decide(query, title, False)
        assert d.attach is True and d.reason == "match"

    @pytest.mark.parametrize("query, title, disambig, reason", [
        ("mercury", "Mercury", True, "disambiguation"),
        ("iPhone 17 price", "IPhone 17", False, "freshness"),
        ("latest Xbox news", "Xbox", False, "freshness"),
        ("weather in Boston", "Boston", False, "freshness"),
        ("is Costco open on Sunday", "Costco", False, "low_coverage"),
        ("Apple earnings Q3", "Apple Inc.", False, "title_not_in_query"),
        ("what is a black hole", "Binary black hole", False, "title_not_in_query"),
        ("what is the", "Thing", False, "empty_query"),
    ])
    def test_rejects(self, query, title, disambig, reason):
        d = wikipedia_match.decide(query, title, disambig)
        assert d.attach is False
        assert d.reason == reason

    def test_qualifier_words_count_toward_coverage_but_are_not_required(self):
        assert wikipedia_match.decide("mercury facts", "Mercury (planet)", False).coverage == 0.5
        assert wikipedia_match.decide("mercury planet facts", "Mercury (planet)", False).coverage == 0.67


def _web_payload(text="• Web result"):
    return json.dumps({"query": "q", "result_text": text, "result_count": 1}), False


class TestWebSearchWikipediaBackground:

    def _run(self, dispatcher, wiki_payload, instruction="who was Ada Lovelace"):
        names: list[str] = []

        async def fake_call(session, name, arguments):
            names.append(name)
            if name == "wikipedia_search":
                if isinstance(wiki_payload, Exception):
                    raise wiki_payload
                return json.dumps(wiki_payload), False
            assert name == "web_search"
            return _web_payload()

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(["web_search"], instruction, {})
        return results, names

    def test_flag_off_by_default_makes_no_wikipedia_call(self, dispatcher, monkeypatch):
        monkeypatch.delenv("LOCALIST_WIKIPEDIA_ENRICH", raising=False)
        results, names = self._run(dispatcher, _search_payload("Ada Lovelace"))
        assert names == ["web_search"]
        assert [r.tool_name for r in results] == ["web_search"]

    def test_flag_on_confident_match_appends_summary(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        results, names = self._run(dispatcher, _search_payload("Ada Lovelace"))
        assert sorted(names) == ["web_search", "wikipedia_search"]
        assert [r.tool_name for r in results] == ["web_search", "wikipedia_summary"]
        summary = results[1]
        assert summary.success is True
        assert summary.result.startswith("Wikipedia background (encyclopedic; may not cover recent events): Ada Lovelace")
        assert "\n[https://en.wikipedia.org/wiki/Ada Lovelace]\nIntro.\n" in summary.result
        assert summary.result.endswith("(CC BY-SA 4.0)")
        assert results[0].workflow_id and results[0].workflow_id == summary.workflow_id

    def test_flag_on_no_confident_match_adds_nothing(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        results, _ = self._run(dispatcher, _search_payload("Binary black hole"), "what is a black hole")
        assert [r.tool_name for r in results] == ["web_search"]

    @pytest.mark.parametrize("payload", [
        _search_payload(is_miss=True),
        _search_payload("Mercury", disambiguation=True),
        ConnectionRefusedError("refused"),
    ])
    def test_flag_on_miss_or_error_is_silent(self, dispatcher, monkeypatch, payload):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        results, _ = self._run(dispatcher, payload, "mercury")
        assert [r.tool_name for r in results] == ["web_search"]
        assert all(r.success for r in results)

    def test_wikipedia_disabled_overrides_enrich(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENABLED", "false")
        results, names = self._run(dispatcher, _search_payload("Ada Lovelace"))
        assert names == ["web_search"]

    def test_only_first_query_gets_background(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        names: list[str] = []

        async def fake_call(session, name, arguments):
            names.append(name)
            if name == "wikipedia_search":
                return json.dumps(_search_payload("Ada Lovelace")), False
            return _web_payload()

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            dispatcher.dispatch(["web_search"], "x", {"web_search_queries": ["Ada Lovelace", "Charles Babbage"]})
        assert names.count("wikipedia_search") == 1
        assert names.count("web_search") == 2

    def test_skipped_when_search_top_hit_was_the_same_article(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENRICH", "true")
        url = "https://en.wikipedia.org/wiki/Ada_Lovelace"

        async def fake_call(session, name, arguments):
            if name == "wikipedia_search":
                payload = _search_payload("Ada Lovelace")
                payload["best_match"]["url"] = url
                payload["result_text"] = payload["result_text"].replace(
                    "https://en.wikipedia.org/wiki/Ada Lovelace", url)
                return json.dumps(payload), False
            if name == "web_search":
                return _web_payload(f"• Ada Lovelace - Wikipedia\n  body\n  [{url}]")
            assert name == "fetch_url"
            return json.dumps({
                "url": url, "title": "Ada Lovelace", "author": "", "date_published": "",
                "cleaned_text": "Augusta Ada King…", "word_count": 3, "fetch_duration_ms": 1.0,
                "via": "Wikipedia API",
            }), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(["web_search"], "who was Ada Lovelace", {})
        assert [r.tool_name for r in results] == ["web_search", "url_fetch"]


# ---------------------------------------------------------------------------
# Phase 5 — research loop: fact vs pricing, Wikipedia background, citations
# ---------------------------------------------------------------------------

from localist.mcp_tool_dispatcher import (  # noqa: E402
    _MAX_RESEARCH_ITERATIONS,
    _RESEARCH_FACT_GATE_SYSTEM_PROMPT,
    _RESEARCH_FACT_REFORMULATE_SYSTEM_PROMPT,
    _RESEARCH_GATE_SYSTEM_PROMPT,
    _RESEARCH_MAX_CITATIONS,
)


class TestResearchKind:

    @pytest.mark.parametrize("instruction", [
        "what does the Basic plan cost", "How much is ChatGPT Plus?",
        "Notion pricing tiers", "Look up the specs on the RTX 4090",
        "is it under $50", "monthly subscription for Spotify",
        "What's the going rate for a plumber", "M4 Pro benchmark scores",
    ])
    def test_pricing(self, instruction):
        assert MCPToolDispatcher._research_kind(instruction) == "pricing"

    @pytest.mark.parametrize("instruction", [
        "/research the history of the transistor", "who invented the telephone",
        "why did the Byzantine Empire fall", "the Marshall Plan's effects on Europe",
    ])
    def test_factual(self, instruction):
        assert MCPToolDispatcher._research_kind(instruction) == "factual"


class TestDeriveInitialQuerySlash:
    @pytest.mark.parametrize("instruction, expected", [
        ("/research the history of the transistor", "the history of the transistor"),
        ("/Research what is CRISPR", "CRISPR"),
        ("/wikipedia Ada Lovelace", "Ada Lovelace"),
        ("history of /research tools", "history of /research tools"),
    ])
    def test_leading_slash_command_stripped(self, dispatcher, instruction, expected):
        assert dispatcher._derive_initial_query(instruction, {}) == expected


_ARTICLE_URL = "https://en.wikipedia.org/wiki/History_of_the_transistor"
_CITES = ["https://cite1.example/a", "https://cite2.example/b", "https://cite3.example/c"]


class TestResearchLoopWikipedia:
    """The factual research path with Wikipedia (§14.21 Phase 5)."""

    def _setup(self, dispatcher, *, gate_yes_on=None, wiki=True, wiki_error=False):
        """gate_yes_on: substring of the gated text that makes the gate say yes."""
        calls: list[tuple[str, dict]] = []
        prompts: list[dict] = []

        def infer(**kw):
            prompts.append(kw)
            if kw["system"] == _RESEARCH_FACT_GATE_SYSTEM_PROMPT:
                return "yes" if gate_yes_on and gate_yes_on in kw["prompt"] else "no"
            if kw["system"] == _RESEARCH_FACT_REFORMULATE_SYSTEM_PROMPT:
                return f"reformulated {len(prompts)}"
            raise AssertionError(f"unexpected system prompt: {kw['system'][:60]}")

        dispatcher._runtime.infer.side_effect = infer

        async def fake_call(session, name, arguments):
            calls.append((name, arguments))
            if name == "web_search":
                q = arguments["query"]
                return json.dumps({"query": q, "result_text": f"• hit for {q}\n  [https://web.example/{len(calls)}]",
                                   "result_count": 1}), False
            if name == "fetch_url":
                u = arguments["url"]
                return json.dumps({"url": u, "title": "t", "author": "", "date_published": "",
                                   "cleaned_text": f"page body of {u}", "word_count": 50,
                                   "fetch_duration_ms": 1.0}), False
            if name == "wikipedia_search":
                if wiki_error:
                    raise ConnectionRefusedError("refused")
                if not wiki:
                    return json.dumps(_search_payload(is_miss=True)), False
                payload = _search_payload("History of the transistor")
                payload["best_match"]["url"] = _ARTICLE_URL
                payload["best_match"]["description"] = "Overview of transistor development"
                return json.dumps(payload), False
            if name == "wikipedia_read":
                return json.dumps(_read_payload("History of the transistor", refs=_CITES)), False
            raise AssertionError(name)

        return calls, prompts, fake_call

    def _run(self, dispatcher, fake_call, instruction="/research the history of the transistor"):
        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            return dispatcher.dispatch(["research"], instruction, {})

    def test_background_added_and_citation_answers(self, dispatcher):
        calls, prompts, fake_call = self._setup(dispatcher, gate_yes_on="cite1.example")
        results = self._run(dispatcher, fake_call)

        names = [r.tool_name for r in results]
        assert names == ["web_search", "wikipedia_summary", "url_fetch", "url_fetch:wikipedia_citation"]
        assert results[1].result.startswith("Wikipedia background")
        assert "cite1.example" in results[3].result
        assert len({r.workflow_id for r in results}) == 1
        # the query sent to both search and Wikipedia is the topic, not "/research ..."
        assert ("web_search", {"query": "the history of the transistor"}) in calls
        assert ("wikipedia_search", {"query": "the history of the transistor"}) in calls
        assert all(p["system"] != _RESEARCH_GATE_SYSTEM_PROMPT for p in prompts)

    def test_citations_capped_and_background_reaches_reformulation(self, dispatcher):
        calls, prompts, fake_call = self._setup(dispatcher)   # gate never passes
        results = self._run(dispatcher, fake_call)

        cited = [r for r in results if r.tool_name == "url_fetch:wikipedia_citation"]
        assert len(cited) == _RESEARCH_MAX_CITATIONS
        assert [c.parameters for c in cited] == [f"url={u!r}" for u in _CITES[:_RESEARCH_MAX_CITATIONS]]
        assert sum(1 for n, _ in calls if n == "web_search") == _MAX_RESEARCH_ITERATIONS
        assert sum(1 for n, _ in calls if n == "wikipedia_search") == 1
        reform = [p for p in prompts if p["system"] == _RESEARCH_FACT_REFORMULATE_SYSTEM_PROMPT]
        assert reform and all(
            "Background (Wikipedia): History of the transistor — Overview of transistor development"
            in p["prompt"] for p in reform
        )
        assert results[-1].tool_name == "research"
        assert "without finding a direct answer" in results[-1].result

    def test_answer_on_first_search_needs_no_citation(self, dispatcher):
        calls, _, fake_call = self._setup(dispatcher, gate_yes_on="hit for")
        results = self._run(dispatcher, fake_call)
        assert [r.tool_name for r in results] == ["web_search", "wikipedia_summary"]
        # citations are looked up with the background, before any gate runs
        assert [n for n, _ in calls].count("wikipedia_read") == 1
        assert not any(n == "fetch_url" for n, _ in calls)

    def test_no_confident_match_runs_plain_factual_loop(self, dispatcher):
        calls, _, fake_call = self._setup(dispatcher, wiki=False, gate_yes_on="page body")
        results = self._run(dispatcher, fake_call)
        assert [r.tool_name for r in results] == ["web_search", "url_fetch"]
        assert not any(n == "wikipedia_read" for n, _ in calls)

    def test_wikipedia_error_is_silent(self, dispatcher):
        _, _, fake_call = self._setup(dispatcher, wiki_error=True, gate_yes_on="page body")
        results = self._run(dispatcher, fake_call)
        assert [r.tool_name for r in results] == ["web_search", "url_fetch"]

    def test_wikipedia_disabled_skips_lookup(self, dispatcher, monkeypatch):
        monkeypatch.setenv("LOCALIST_WIKIPEDIA_ENABLED", "false")
        calls, _, fake_call = self._setup(dispatcher, gate_yes_on="page body")
        self._run(dispatcher, fake_call)
        assert not any(n.startswith("wikipedia") for n, _ in calls)

    def test_pricing_request_never_touches_wikipedia(self, dispatcher):
        calls: list[str] = []

        def infer(**kw):
            assert kw["system"] == _RESEARCH_GATE_SYSTEM_PROMPT
            return "yes"

        dispatcher._runtime.infer.side_effect = infer

        async def fake_call(session, name, arguments):
            calls.append(name)
            return json.dumps({"query": arguments["query"], "result_text": "Pro: $20/month",
                               "result_count": 1}), False

        with patch.object(MCPToolDispatcher, "_call_mcp_tool", side_effect=fake_call):
            results = dispatcher.dispatch(["research"], "/research how much does Notion Pro cost", {})
        assert calls == ["web_search"]
        assert [r.tool_name for r in results] == ["web_search"]

    def test_background_dropped_when_article_itself_was_read(self, dispatcher):
        calls, _, fake_call = self._setup(dispatcher, gate_yes_on="page body")

        async def search_hits_article(session, name, arguments):
            if name == "web_search":
                calls.append((name, arguments))
                return json.dumps({"query": arguments["query"],
                                   "result_text": f"• History of the transistor\n  [{_ARTICLE_URL}]",
                                   "result_count": 1}), False
            if name == "fetch_url":
                calls.append((name, arguments))
                return json.dumps({"url": _ARTICLE_URL, "title": "History of the transistor",
                                   "author": "", "date_published": "", "cleaned_text": "page body",
                                   "word_count": 50, "fetch_duration_ms": 1.0,
                                   "via": "Wikipedia API"}), False
            return await fake_call(session, name, arguments)

        results = self._run(dispatcher, search_hits_article)
        assert [r.tool_name for r in results] == ["web_search", "url_fetch"]
