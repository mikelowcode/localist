"""
Phase 1 tests — localist-mcp server (mcp_server/).
Phase 2 adds fetch_url coverage (mcp_server/url_fetch.py) — ports the
retired standalone Fetcher microservice's /extract path in-process.
Phase 3 adds web_search coverage (mcp_server/web_search.py) — ports the
LangSearch integration in-process, no runtime.infer() fallback.

Covers:
  - file_ops.read_file / write_file / append_file: sandboxing, truncation,
    error raising (ported behaviour from ToolDispatcher._file_read/_write/_append)
  - url_fetch.fetch_url: success, timeout clamping, connection error, HTTP
    4xx/5xx, and extraction_failed (paywall/empty content) — error taxonomy
    ported from fetcher/models.py's ErrorResponse.error_code
  - web_search.web_search: results found (bullet formatting matches the
    legacy shape exactly), empty results, missing API key (clean error, no
    inference call), network/timeout error
  - All tools as registered on the FastMCP instance, exercised through
    an in-process MCP client session (mcp.shared.memory) — no network server
    required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server import file_ops, url_fetch, web_search
from mcp_server.main import mcp as mcp_app


# ---------------------------------------------------------------------------
# file_ops — direct unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_project_root():
    """Every test sets its own root explicitly; avoid state leaking between tests."""
    yield
    file_ops.set_project_root(Path(__file__).resolve().parent.parent)


class TestFileOpsRead:
    def test_read_returns_file_content(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "notes.md").write_text("hello world", encoding="utf-8")
        assert file_ops.read_file("notes.md") == "hello world"

    def test_read_missing_file_raises(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="file not found"):
            file_ops.read_file("ghost.md")

    def test_read_truncates_long_content(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        big = "x" * (file_ops._MAX_FILE_READ_CHARS + 500)
        (tmp_path / "big.txt").write_text(big, encoding="utf-8")
        result = file_ops.read_file("big.txt")
        assert result.endswith("\n… [truncated]")
        assert len(result) == file_ops._MAX_FILE_READ_CHARS + len("\n… [truncated]")

    def test_read_path_traversal_blocked(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            file_ops.read_file("../../etc/passwd")


class TestFileOpsWrite:
    def test_write_creates_file(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        result = file_ops.write_file("out/result.md", "# Result\nContent here.")
        assert result.startswith("OK: wrote")
        assert (tmp_path / "out" / "result.md").read_text(encoding="utf-8") == "# Result\nContent here."

    def test_write_path_traversal_blocked(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            file_ops.write_file("../escape.md", "nope")


class TestFileOpsAppend:
    def test_append_to_existing_file(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "log.txt").write_text("Line 1.\n", encoding="utf-8")
        result = file_ops.append_file("log.txt", "Line 2.\n")
        assert result.startswith("OK: appended")
        assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "Line 1.\nLine 2.\n"

    def test_append_creates_parent_dirs(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        file_ops.append_file("a/b/c/deep.txt", "content")
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# url_fetch.fetch_url — direct unit tests
# ---------------------------------------------------------------------------

_SAMPLE_ARTICLE_HTML = b"""
<html><head><title>Test Article Title</title>
<meta name="author" content="Jane Doe">
<meta property="article:published_time" content="2026-01-01">
</head>
<body>
<article>
<h1>Test Article Title</h1>
<p>This is the first paragraph of a reasonably long test article used to
verify that the readability extraction pipeline correctly identifies the
main content block and strips away any surrounding navigation or
boilerplate markup that a typical web page would include around the
actual article body text.</p>
<p>This is a second paragraph adding more substantive content so that the
extractor has enough signal to treat this block as the primary article
content rather than discarding it as noise or a login wall placeholder.</p>
</article>
</body></html>
"""

_EMPTY_HTML = b"<html><head><title>Login</title></head><body></body></html>"


def _raw_response(content: bytes, url: str = "https://example.com/article") -> url_fetch.RawResponse:
    return url_fetch.RawResponse(
        url               = url,
        status_code       = 200,
        content_type      = "text/html",
        content           = content,
        headers           = {},
        fetch_duration_ms = 12.3,
    )


class TestFetchUrlSuccess:
    def test_success_returns_expected_fields(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))):
            result = asyncio.run(url_fetch.fetch_url("https://example.com/article"))

        assert result["title"] == "Test Article Title"
        assert result["author"] == "Jane Doe"
        assert result["date_published"] == "2026-01-01"
        assert "reasonably long test article" in result["cleaned_text"]
        assert result["word_count"] > 0
        assert result["url"] == "https://example.com/article"
        assert result["fetch_duration_ms"] == 12.3

    def test_timeout_is_clamped_before_reaching_fetch(self):
        fake_fetch = AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))
        with patch.object(url_fetch, "_fetch", fake_fetch):
            asyncio.run(url_fetch.fetch_url("https://example.com/article", timeout=100.0))
        fake_fetch.assert_called_once_with("https://example.com/article", 30.0)

        fake_fetch.reset_mock()
        with patch.object(url_fetch, "_fetch", fake_fetch):
            asyncio.run(url_fetch.fetch_url("https://example.com/article", timeout=0.1))
        fake_fetch.assert_called_once_with("https://example.com/article", 1.0)


class TestFetchUrlErrors:
    def test_timeout_maps_to_timeout_code(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.TimeoutException("timed out"))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/slow"))
        assert exc_info.value.error_code == "timeout"
        assert str(exc_info.value).startswith("ERROR: timeout —")

    def test_connect_error_maps_to_connection_error_code(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://unreachable.example"))
        assert exc_info.value.error_code == "connection_error"
        assert str(exc_info.value).startswith("ERROR: connection_error —")

    def test_http_404_maps_to_http_client_error_code(self):
        request  = httpx.Request("GET", "https://example.com/missing")
        response = httpx.Response(404, request=request)
        error    = httpx.HTTPStatusError("404", request=request, response=response)
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=error)):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/missing"))
        assert exc_info.value.error_code == "http_client_error"
        assert "404" in str(exc_info.value)

    def test_http_500_maps_to_http_server_error_code(self):
        request  = httpx.Request("GET", "https://example.com/broken")
        response = httpx.Response(500, request=request)
        error    = httpx.HTTPStatusError("500", request=request, response=response)
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=error)):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/broken"))
        assert exc_info.value.error_code == "http_server_error"

    def test_empty_extraction_maps_to_extraction_failed_code(self):
        """Paywall/login-wall page — readability produces no usable content."""
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_EMPTY_HTML))):
            with pytest.raises(url_fetch.FetchUrlError) as exc_info:
                asyncio.run(url_fetch.fetch_url("https://example.com/paywalled"))
        assert exc_info.value.error_code == "extraction_failed"
        assert str(exc_info.value).startswith("ERROR: extraction_failed —")


# ---------------------------------------------------------------------------
# web_search.web_search — direct unit tests
# ---------------------------------------------------------------------------

def _langsearch_response(pages: list[dict], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", web_search._LANGSEARCH_ENDPOINT)
    return httpx.Response(
        status_code,
        json    = {"data": {"webPages": {"value": pages}}},
        request = request,
    )


class TestWebSearchSuccess:
    def test_results_formatted_matching_legacy_bullet_shape(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        pages = [
            {
                "name":        "oMLX Release Notes",
                "snippet":     "fallback snippet",
                "summary":     "x" * 350,  # forces truncation
                "displayUrl":  "example.com/omlx",
                "url":         "https://example.com/omlx",
            }
        ]
        response = _langsearch_response(pages)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("oMLX release notes"))

        assert result["query"] == "oMLX release notes"
        assert result["result_count"] == 1
        text = result["result_text"]
        assert text.startswith("• oMLX Release Notes\n  ")
        assert text.endswith("[example.com/omlx]")
        # body truncated to <=300 chars on a word boundary, no raw 350-char run
        body_line = text.splitlines()[1]
        assert len(body_line.strip()) <= 300

    def test_prefers_summary_over_snippet(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        pages = [{
            "name": "Title", "snippet": "snippet text", "summary": "summary text",
            "url": "https://example.com",
        }]
        response = _langsearch_response(pages)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("q"))
        assert "summary text" in result["result_text"]
        assert "snippet text" not in result["result_text"]

    def test_empty_results_returns_success_not_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            result = asyncio.run(web_search.web_search("nothing found query"))
        assert result["result_text"] == "No results found."
        assert result["result_count"] == 0


class TestWebSearchErrors:
    def test_missing_api_key_raises_clean_error_without_network_call(self, monkeypatch):
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
        fake_post = AsyncMock()
        with patch.object(httpx.AsyncClient, "post", fake_post):
            with pytest.raises(ValueError, match="LANGSEARCH_API_KEY not configured"):
                asyncio.run(web_search.web_search("anything"))
        fake_post.assert_not_called()

    def test_empty_string_api_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "")
        with pytest.raises(ValueError, match="LANGSEARCH_API_KEY not configured"):
            asyncio.run(web_search.web_search("anything"))

    def test_connection_error_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))

    def test_timeout_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.TimeoutException("timed out"))):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))

    def test_http_error_status_wraps_as_clean_error(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([], status_code=500)
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            with pytest.raises(ValueError, match="ERROR: web_search failed —"):
                asyncio.run(web_search.web_search("q"))


# ---------------------------------------------------------------------------
# MCP tool wiring — in-process client session (no network)
# ---------------------------------------------------------------------------

async def _call_tool(name: str, arguments: dict) -> tuple[str, bool]:
    async with create_connected_server_and_client_session(mcp_app) as session:
        result = await session.call_tool(name, arguments)
        text = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        return text, result.isError


class TestMCPToolsInProcess:
    def test_read_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "notes.md").write_text("hi from mcp", encoding="utf-8")
        text, is_error = asyncio.run(_call_tool("read_file", {"path": "notes.md"}))
        assert is_error is False
        assert text == "hi from mcp"

    def test_read_file_tool_error_surfaces_as_is_error(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(_call_tool("read_file", {"path": "ghost.md"}))
        assert is_error is True
        assert "file not found" in text

    def test_write_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(
            _call_tool("write_file", {"path": "out.md", "content": "written via mcp"})
        )
        assert is_error is False
        assert "OK: wrote" in text
        assert (tmp_path / "out.md").read_text(encoding="utf-8") == "written via mcp"

    def test_append_file_tool_success(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        (tmp_path / "log.txt").write_text("first\n", encoding="utf-8")
        text, is_error = asyncio.run(
            _call_tool("append_file", {"path": "log.txt", "content": "second\n"})
        )
        assert is_error is False
        assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "first\nsecond\n"

    def test_path_traversal_blocked_over_mcp(self, tmp_path: Path):
        file_ops.set_project_root(tmp_path)
        text, is_error = asyncio.run(
            _call_tool("read_file", {"path": "../../etc/passwd"})
        )
        assert is_error is True
        assert "path traversal" in text

    def test_fetch_url_tool_success(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(return_value=_raw_response(_SAMPLE_ARTICLE_HTML))):
            text, is_error = asyncio.run(
                _call_tool("fetch_url", {"url": "https://example.com/article"})
            )
        assert is_error is False
        data = json.loads(text)
        assert data["title"] == "Test Article Title"
        assert data["word_count"] > 0

    def test_fetch_url_tool_error_surfaces_as_is_error(self):
        with patch.object(url_fetch, "_fetch", AsyncMock(side_effect=httpx.ConnectError("refused"))):
            text, is_error = asyncio.run(
                _call_tool("fetch_url", {"url": "https://unreachable.example"})
            )
        assert is_error is True
        assert "connection_error" in text

    def test_web_search_tool_success(self, monkeypatch):
        monkeypatch.setenv("LANGSEARCH_API_KEY", "test-key")
        response = _langsearch_response([{"name": "T", "snippet": "s", "url": "https://e.com"}])
        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
            text, is_error = asyncio.run(_call_tool("web_search", {"query": "test query"}))
        assert is_error is False
        data = json.loads(text)
        assert data["result_count"] == 1

    def test_web_search_tool_missing_api_key_surfaces_as_is_error(self, monkeypatch):
        monkeypatch.delenv("LANGSEARCH_API_KEY", raising=False)
        text, is_error = asyncio.run(_call_tool("web_search", {"query": "test query"}))
        assert is_error is True
        assert "LANGSEARCH_API_KEY not configured" in text
