"""
Shared pytest fixtures for the backend test suite.

Env isolation for SEARCH_PROVIDER / BRAVE_API_KEY / LANGSEARCH_API_KEY
-----------------------------------------------------------------------
mcp_server/main.py calls load_dotenv() at import time (needed so the real
localist-mcp service picks up config from backend/.env when launched
normally). Whenever any test — in this file or transitively via another
test module's import — imports mcp_server.main within the same pytest
process, that load_dotenv() call populates os.environ with whatever is
actually in backend/.env, including real values for these three vars.
Those values then leak into every other test in the session (import
happens once, at collection time, and os.environ is process-global), and
can silently override provider dispatch or even trigger genuine live
calls to the real search-provider APIs when a real key is present.

The fixture below strips all three before every test runs, so each
test's starting environment is deterministic regardless of import order
and regardless of what's actually stored in backend/.env. Tests that
need a specific provider/key set it explicitly with monkeypatch.setenv
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
    for var in ("SEARCH_PROVIDER", "BRAVE_API_KEY", "LANGSEARCH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
