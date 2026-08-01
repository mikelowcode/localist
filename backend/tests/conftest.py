"""
Shared pytest fixtures for the backend test suite.

Env isolation for SEARCH_PROVIDER / BRAVE_API_KEY / LANGSEARCH_API_KEY /
LOCALIST_RESEARCH_LOOP_ENABLED / OLLAMA_API_KEY / WEB_SEARCH_PROVIDER
-----------------------------------------------------------------------
mcp_server/main.py calls load_dotenv() at import time (needed so the real
localist-mcp service picks up config from backend/.env when launched
normally). Whenever any test — in this file or transitively via another
test module's import — imports mcp_server.main within the same pytest
process, that load_dotenv() call populates os.environ with whatever is
actually in backend/.env, including real values for these vars.
Those values then leak into every other test in the session (import
happens once, at collection time, and os.environ is process-global), and
can silently override provider dispatch or even trigger genuine live
calls to the real search-provider APIs when a real key is present.

This is a generic leak vector, not one specific to these vars — any
LOCALIST_*/provider-ish flag read via os.environ.get() will leak the same
way the moment it's set in backend/.env, as confirmed 2026-07-16 when
LOCALIST_RESEARCH_LOOP_ENABLED=true (set for live testing) started
failing test_planner_phase3.py::TestPriority3SemanticGating::
test_literal_keyword_still_fires_with_embed_fn. The fixture below is kept
scoped to the vars actually known to cause problems rather than
broadening it to strip all LOCALIST_* flags speculatively — add to the
tuple as each new leak is confirmed, the same way this one was.

OLLAMA_API_KEY / WEB_SEARCH_PROVIDER added 2026-07-31, proactively rather
than after a confirmed in-process leak (no test currently exercises
ollama_web_search/ollama_web_fetch through the real in-process FastMCP
session the way test_mcp_server.py does for web_search/fetch_url — see
docs/architecture/14-localist-mcp-tool-layer.md §14.16). The identical
leak *did* already happen once for OLLAMA_API_KEY, just through a
subprocess-based fixture this autouse fixture doesn't cover (see the
"does not protect subprocess-based fixtures" note below) —
test_tool_dispatcher_phase6.py's test_web_search_missing_key_triggers_
corpus_fallback made a live, authenticated call to the real Ollama Cloud
API before that fixture was fixed by hand. Closing this gap here too
before any in-process test for either new tool gets a chance to hit it
the same way.

The fixture below strips all of them before every test runs, so each
test's starting environment is deterministic regardless of import order
and regardless of what's actually stored in backend/.env. Tests that
need a specific value set it explicitly with monkeypatch.setenv
(see tests/test_mcp_server.py) — that happens after this fixture's
deletion within the same test, so explicit test-level values always win.

This does not protect subprocess-based fixtures (e.g.
test_tool_dispatcher_phase6.py's localist_mcp_server*) — those spawn a
separate process that loads backend/.env fresh via its own
load_dotenv() call and must pin any vars they need to control explicitly
in the child's env dict.
"""

import pytest


@pytest.fixture(autouse=True)
def _clean_search_provider_env(monkeypatch):
    for var in (
        "SEARCH_PROVIDER", "BRAVE_API_KEY", "LANGSEARCH_API_KEY",
        "LOCALIST_RESEARCH_LOOP_ENABLED", "OLLAMA_API_KEY",
        "WEB_SEARCH_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)
