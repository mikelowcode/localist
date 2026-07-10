"""
WikiAgent × Ollama Cloud Diagnostic
====================================

Live-verification script: ingests one raw file through WikiAgent using the
Ollama Cloud model (`gemma4:31b-cloud`), routed through the local Ollama
daemon at http://localhost:11434 (same daemon, no separate cloud client —
Ollama proxies cloud models over its normal HTTP API).

This is the first wiki_agent ingestion run since the runtime/tooling changes
described in `raw/Localist Runtime and Tooling Update 2026-07.md` — the wiki
corpus predates the Ollama runtime backend, MCP consolidation, and retirement
of the legacy ToolDispatcher/Fetcher, so `wiki/localist-software-stack.md`
is known to be stale going in. This run is expected to propose either an
apply_diff against that page or new pages/concepts covering the update.

auto_apply is hardcoded to False — this prints proposed actions for review,
it does not write to backend/wiki/.  Re-run with the constant flipped only
after reviewing the output.

Run from the project root:
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_wiki_agent_ollama_cloud.py
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------------------------------------------------------
# Path setup — allow imports from backend/
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ollama_runtime_client import OllamaRuntimeClient  # noqa: E402
from wiki_agent import WikiAgent  # noqa: E402
from controller_agent import ControllerAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLOUD_CHAT_MODEL = "gemma4:31b-cloud"
OLLAMA_BASE_URL  = "http://localhost:11434"

RAW_PATH = BACKEND_DIR / "raw" / "Localist Runtime and Tooling Update 2026-07.md"

AUTO_APPLY = False  # keep False until output has been reviewed


def main() -> None:
    if not RAW_PATH.exists():
        print(f"FAIL: raw file not found: {RAW_PATH}")
        sys.exit(1)

    runtime = OllamaRuntimeClient(
        chat_model=CLOUD_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    health = runtime.health_check()
    print("--- Ollama health_check ---")
    print(json.dumps(health, indent=2))
    if not health.get("reachable"):
        print("FAIL: Ollama daemon not reachable at", OLLAMA_BASE_URL)
        sys.exit(1)
    if not health.get("chat_model_found"):
        print(
            f"WARNING: '{CLOUD_CHAT_MODEL}' not in GET /api/tags model list — "
            "proceeding anyway in case tags lags a fresh pull."
        )

    wiki_agent = WikiAgent(runtime=runtime, project_root=BACKEND_DIR)
    controller = ControllerAgent(runtime=runtime, agents=[wiki_agent])

    result = controller.handle_task({
        "task_id": str(uuid.uuid4()),
        "instruction": "Ingest the raw file and create a wiki research note.",
        "context": {
            "raw_path": str(RAW_PATH),
            "wiki_dir": str(BACKEND_DIR / "wiki"),
            "schema_path": str(BACKEND_DIR / "SCHEMA.md"),
            "templates_dir": str(BACKEND_DIR / "templates"),
            "auto_apply": AUTO_APPLY,
        },
    })

    print("\n--- WikiAgent result ---")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
