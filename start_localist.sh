#!/usr/bin/env bash
# start_localist.sh — Localist Framework service launcher
#
# Part of: Localist CLI
#
# Starts:
#   • Localist backend   (FastAPI / uvicorn) — port 8001
#   • Localist fetcher   (FastAPI / uvicorn) — port 8002
#
# The inference engine (oMLX, MLX-LM, Ollama, LM Studio, etc.) is managed
# separately. Localist is inference-engine-agnostic.
#
# Usage:
#   ./start_localist.sh          — start both services
#   ./start_localist.sh --stop   — kill any running instances on ports 8001/8002
#
# Logs:
#   logs/backend.log
#   logs/fetcher.log
#
# Ctrl+C stops both services cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_PYTHON="$BACKEND_DIR/.venv/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
# --stop flag: kill any running instances and exit
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping Localist services..."
    lsof -ti tcp:8001 | xargs kill -TERM 2>/dev/null && echo "  backend (8001) stopped." || echo "  backend (8001) not running."
    lsof -ti tcp:8002 | xargs kill -TERM 2>/dev/null && echo "  fetcher (8002) stopped." || echo "  fetcher (8002) not running."
    exit 0
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    echo "Run: cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
    echo "WARNING: $BACKEND_DIR/.env not found — environment variables may be missing."
fi

# Warn if ports already in use (do not abort — let uvicorn surface the error)
for PORT in 8001 8002; do
    if lsof -ti tcp:$PORT &>/dev/null; then
        echo "WARNING: port $PORT is already in use. Run ./start_localist.sh --stop first."
    fi
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

echo ""
echo "  ██╗      ██████╗  ██████╗ █████╗ ██╗     ██╗███████╗████████╗"
echo "  ██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██║██╔════╝╚══██╔══╝"
echo "  ██║     ██║   ██║██║     ███████║██║     ██║███████╗   ██║   "
echo "  ██║     ██║   ██║██║     ██╔══██║██║     ██║╚════██║   ██║   "
echo "  ███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║███████║   ██║   "
echo "  ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝   "
echo ""
echo "  Localist Framework — local-first agentic assistant"
echo ""
echo "  Backend  → http://127.0.0.1:8001  (log: logs/backend.log)"
echo "  Fetcher  → http://127.0.0.1:8002  (log: logs/fetcher.log)"
echo ""
echo "  Ctrl+C to stop both services."
echo ""

# ---------------------------------------------------------------------------
# Launch services
# Both run from backend/ so import paths resolve correctly.
# ---------------------------------------------------------------------------
cd "$BACKEND_DIR"

"$VENV_PYTHON" -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 8001 \
    --reload \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

"$VENV_PYTHON" -m uvicorn fetcher.main:app \
    --host 127.0.0.1 \
    --port 8002 \
    --reload \
    > "$LOG_DIR/fetcher.log" 2>&1 &
FETCHER_PID=$!

# ---------------------------------------------------------------------------
# Cleanup on Ctrl+C or unexpected exit
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Stopping Localist services..."
    kill -TERM "$BACKEND_PID" 2>/dev/null && echo "  backend stopped."
    kill -TERM "$FETCHER_PID" 2>/dev/null && echo "  fetcher stopped."
    kill "$TAIL_BACKEND_PID" "$TAIL_FETCHER_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Tail both logs interleaved with service prefix
# ---------------------------------------------------------------------------
tail -f "$LOG_DIR/backend.log" | sed 's/^/[backend] /' &
TAIL_BACKEND_PID=$!

tail -f "$LOG_DIR/fetcher.log" | sed 's/^/[fetcher] /' &
TAIL_FETCHER_PID=$!

# Wait — if either service exits unexpectedly, surface it
wait "$BACKEND_PID" "$FETCHER_PID"

kill "$TAIL_BACKEND_PID" "$TAIL_FETCHER_PID" 2>/dev/null
echo ""
echo "A Localist service exited unexpectedly. Check logs/ for details."
exit 1
