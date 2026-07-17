"""
Shared pytest fixtures for the backend test suite.

Env isolation for SEARCH_PROVIDER / BRAVE_API_KEY / LANGSEARCH_API_KEY /
LOCALIST_RESEARCH_LOOP_ENABLED
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

This is a generic leak vector, not one specific to these four vars — any
LOCALIST_* flag read via os.environ.get() will leak the same way the
moment it's set in backend/.env, as confirmed 2026-07-16 when
LOCALIST_RESEARCH_LOOP_ENABLED=true (set for live testing) started
failing test_planner_phase3.py::TestPriority3SemanticGating::
test_literal_keyword_still_fires_with_embed_fn. The fixture below is kept
scoped to the vars actually known to cause problems rather than
broadening it to strip all LOCALIST_* flags speculatively — add to the
tuple as each new leak is confirmed, the same way this one was.

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
        "LOCALIST_RESEARCH_LOOP_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
