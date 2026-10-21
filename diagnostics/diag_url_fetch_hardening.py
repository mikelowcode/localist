"""
Diagnostic: diag_url_fetch_hardening.py
========================================
§14.17 (2026-10-21): does the local fetcher still read MSN links via their
original publisher, report JavaScript shells honestly, and leave ordinary
pages alone?

Runs the real fetch_url() (mcp_server/url_fetch.py) in-process against live
URLs: the two MSN article links from the 2026-10-21 session that started this
work, an MSN section page (a JavaScript shell with no article ID, so not
resolvable), and the Digital Foundry article that fetched fine between them.
Prints the outcome (ok, or the error code), word count, the URL actually read,
the Via: value, and timing per URL.

  Expected today:  MSN articles   ok, original publisher, via "msn.com (…)"
                   MSN section    javascript_required
                   Digital Fdry   ok, no via

A different outcome is the finding — e.g. MSN changing its content API or a
publisher starting to block the fetcher — not a bug in this script.

--with-ollama additionally drives MCPToolDispatcher._run_url_fetch() so a
failed fetch goes through the real Ollama Cloud web_fetch fallback, showing
exactly the text the model would see. Needs OLLAMA_API_KEY in the
environment; makes one Ollama Cloud call per failing URL.

READ-ONLY. Network only, no DB, no running services (works while the
installed app holds ports 8001/8003). Not part of the pytest suite.

Usage:
    cd backend
    source .venv/bin/activate
    python ../diagnostics/diag_url_fetch_hardening.py [--with-ollama] [--url URL ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from localist.mcp_server import ollama_web_fetch, url_fetch

DEFAULT_URLS = [
    "https://www.msn.com/en-us/news/other/the-next-xbox-might-cost-1800-report-says/"
    "ar-AA2fGabX?ocid=BingNewsSerp",
    "https://www.msn.com/en-us/gaming/rockstar-games/"
    "rockstar-officially-comments-on-gta-6-s-release-date/ar-AA2eaRx2?ocid=BingNewsSerp",
    "https://www.msn.com/en-us/news",
    "https://www.digitalfoundry.net/features/gta-6-an-extended-look-analysis-perhaps-the-most-"
    "impressive-showcase-weve-ever-seen-in-real-time-rendering",
]


async def fetch_direct(url: str) -> dict:
    start = time.perf_counter()
    try:
        data = await url_fetch.fetch_url(url)
    except url_fetch.FetchUrlError as exc:
        return {"outcome": exc.error_code, "words": 0, "read": "", "via": "",
                "secs": time.perf_counter() - start}
    return {"outcome": "ok", "words": data["word_count"], "read": data["url"],
            "via": data.get("via", ""), "secs": time.perf_counter() - start}


async def fetch_via_dispatcher(url: str) -> str:
    from localist.mcp_tool_dispatcher import MCPToolDispatcher

    tools = {"fetch_url": url_fetch.fetch_url, "ollama_web_fetch": ollama_web_fetch.ollama_web_fetch}

    async def call_in_process(self, session, name, arguments):
        try:
            return json.dumps(await tools[name](arguments["url"])), False
        except Exception as exc:
            return f"Error executing tool {name}: {exc}", True

    MCPToolDispatcher._call_mcp_tool = call_in_process
    dispatcher = MCPToolDispatcher.__new__(MCPToolDispatcher)
    result = await dispatcher._run_url_fetch(object(), None, f"fetch {url}", {})
    head = result.result.split("\n\n", 1)[0].replace("\n", " / ")
    return f"{result.tool_name} success={result.success} | {head[:300]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--with-ollama", action="store_true",
                        help="also run the dispatcher path with the real Ollama fallback")
    parser.add_argument("--url", action="append", help="URL(s) to check instead of the defaults")
    args = parser.parse_args()
    urls = args.url or DEFAULT_URLS

    print(f"{'outcome':<20} {'words':>5} {'secs':>5}  url → read [via]")
    for url in urls:
        r = asyncio.run(fetch_direct(url))
        read = "" if not r["read"] or r["read"] == url else f" → {r['read']}"
        via  = f"  [via {r['via']}]" if r["via"] else ""
        print(f"{r['outcome']:<20} {r['words']:>5} {r['secs']:>5.1f}  {url[:90]}{read}{via}")

    if args.with_ollama:
        print("\nDispatcher path (what the model sees):")
        for url in urls:
            print(f"- {url[:90]}\n  {asyncio.run(fetch_via_dispatcher(url))}")


if __name__ == "__main__":
    main()
