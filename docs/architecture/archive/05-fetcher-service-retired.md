## 5. Fetcher Service (Retired)

> **Retired in Phase 2 (2026-07-03).** The standalone Fetcher microservice
> described below ran as a separate FastAPI process on port 8002 from
> project inception through Phase 1. In Phase 2, its `/extract` path
> (`client.py`'s `fetch()` + `extractor.py`'s `extract()`) was ported
> verbatim, in-process, into the `fetch_url` MCP tool on `localist-mcp`
> (port 8003) — see §14. `/fetch` (raw HTML) and `/api` (JSON passthrough)
> were **not** ported: nothing in the codebase called them, and `url_fetch`
> only ever needed `/extract`'s "get me readable content" behavior. The
> standalone service, its directory (`backend/fetcher/`), and the
> `LOCALIST_FETCHER_URL` environment variable were all deleted in Phase 4
> once nothing referenced them. Port 8002 is unbound; `start_localist.sh`
> no longer manages a fetcher process (§13).

This section is kept as the historical record of the pre-Phase-2 design,
in case any of its implementation notes are useful for a future standalone
service. It no longer describes anything live in the running system.

**What it was:** a standalone FastAPI microservice on port 8002 with three
endpoints — `POST /fetch` (raw HTML fetch, for debugging), `POST /extract`
(fetch + `readability-lxml` extraction — the one `ToolDispatcher`'s
`url_fetch` actually called), and `POST /api` (strict JSON REST passthrough).
Error handling used a structured `ErrorResponse` with an `error_code`
taxonomy (`connection_error`, `timeout`, `http_client_error`,
`http_server_error`, `extraction_failed`, `not_json`) — that same taxonomy
(minus `not_json`, since `/api` was never ported) is preserved in
`fetch_url`'s error shape today; see §14.

Implementation notes worth preserving: `readability-lxml` 0.8.4.1 expects a
decoded string, not bytes (`html.decode("utf-8", errors="replace")` before
`Document()`); a browser-like User-Agent reduces bot-blocking; the URL
regex in Planner's `_priority3_tool()` fires on any `http://`/`https://` in
the instruction, so "what's the difference between http and https?" also
triggers `url_fetch` — an edge case still present today, unrelated to the
retirement, worth monitoring if it becomes noisy.

