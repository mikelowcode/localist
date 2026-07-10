"""
WikiAgent Diff-Only Path Diagnostic
====================================

Live-verification script: exercises WikiAgent._run_diff_only() (the
standalone-diff-instruction path added alongside this script — see the
"standalone diff instructions to WikiAgent" scope doc) against the Ollama
Cloud model (`gemma4:31b-cloud`), routed through the local Ollama daemon at
http://localhost:11434.

This closes the loop on the finding that motivated the diff-only path:
diag_wiki_agent_ollama_cloud.py ingested
`raw/Localist Runtime Tooling Upate.md` and produced a new research note,
but proposed zero diffs against the stale `wiki/localist-software-stack.md`
page it should have updated — because the ingest prompt only ever treats
diffs as optional, and the model didn't propose one. This script targets
that same page directly, with no raw file involved.

Calls WikiAgent.run() directly (SubTask.context={"diff_target": ...}),
bypassing ControllerAgent/Planner entirely — Priority 1b routing (the
instruction-keyword + graph-stem-resolution path that gets a live request
to WikiAgent with diff_target set) is already covered by
tests/test_planner_phase3.py::TestPlannerP1bDiff against an in-memory
MemoryManager fixture. What that layer cannot verify is what the *model*
actually proposes when asked to diff a real page live, which is the only
thing this script adds. Calling controller.handle_task() here would exercise
routing this script isn't trying to check, and would require a MemoryManager
seeded with real graph nodes just to reach WikiAgent at all.

auto_apply is hardcoded to False — this prints the proposed diff for
review, it does not write to backend/wiki/. Re-run with the constant
flipped only after reviewing the output.

Run from the project root:
    cd /Users/michaelfilanc/Projects/lora-app-demo
    python3 diagnostics/diag_wiki_agent_diff_only.py
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
from controller_agent import SubTask  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLOUD_CHAT_MODEL = "gemma4:31b-cloud"
OLLAMA_BASE_URL  = "http://localhost:11434"

DIFF_TARGET = "localist-software-stack"

INSTRUCTION = (
    "Update the localist-software-stack page to reflect the new Ollama "
    "runtime backend (OllamaRuntimeClient, also serving Ollama Cloud models "
    "over the same local daemon) and the MCP tool layer consolidation — "
    "localist-mcp on port 8003 now exposes web_search/fetch_url/file_op, "
    "and the legacy in-process ToolDispatcher plus the standalone Fetcher "
    "microservice (former port 8002) are both retired."
)

WIKI_DIR = BACKEND_DIR / "wiki"

AUTO_APPLY = False  # reset to safe default after live verification (2026-07-09) — no
# rollback exists for wiki/ writes (gitignored, see docs/architecture/17-wiki-agent-
# diff-target.md §17.7); flip only deliberately, review output before re-flipping.


def main() -> None:
    target_path = WIKI_DIR / f"{DIFF_TARGET}.md"
    if not target_path.exists():
        print(f"FAIL: diff_target page not found on disk: {target_path}")
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

    subtask = SubTask(
        subtask_id  = str(uuid.uuid4()),
        agent_name  = "wiki_agent",
        instruction = INSTRUCTION,
        context     = {
            "diff_target":   DIFF_TARGET,
            "wiki_dir":      str(WIKI_DIR),
            "schema_path":   str(BACKEND_DIR / "SCHEMA.md"),
            "templates_dir": str(BACKEND_DIR / "templates"),
            "auto_apply":    AUTO_APPLY,
        },
    )

    result = wiki_agent.run(subtask)

    print("\n--- WikiAgent AgentResult ---")
    print(json.dumps(
        {
            "subtask_id": result.subtask_id,
            "agent_name": result.agent_name,
            "status":     result.status.value,
            "output":     result.output,
            "error":      result.error,
        },
        indent=2, default=str,
    ))

    diffs = result.output.get("diffs") if result.output else None
    if diffs:
        print(f"\n{len(diffs)} diff(s) proposed against '{DIFF_TARGET}'.")
    else:
        print(
            f"\nNo diff proposed against '{DIFF_TARGET}' — this IS the "
            "finding if it occurs; do not treat it as a script bug."
        )


if __name__ == "__main__":
    main()
