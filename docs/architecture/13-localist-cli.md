## 13. Localist CLI (start_localist.sh)

### 13.1 Overview

`start_localist.sh` manages three local services as a single lifecycle unit:
backend (FastAPI/uvicorn, port 8001), `localist-mcp` (FastAPI/uvicorn, port
8003), and the Localist UI dev server (SvelteKit/Vite, port 5173). All
three start together, log to `logs/backend.log`, `logs/mcp_server.log`, and
`logs/frontend.log` with prefixed interleaved tailing, and stop together on
Ctrl+C or via `./start_localist.sh --stop`. The inference engine (oMLX,
MLX-LM, Ollama, LM Studio, etc.) remains managed separately, per the
existing engine-agnostic design principle (§1).

> **Note:** through Phase 1, this script managed a third service — the
> standalone Fetcher microservice on port 8002 — in place of `localist-mcp`.
> Phase 2 retired the Fetcher (§5) and swapped it for `localist-mcp`; see
> §14 for what that service does.

### 13.2 Service Launch

Backend and `localist-mcp` both run from `cwd = backend/` — required for
import resolution (`main:app` and `mcp_server.main:app`). The frontend
runs from `localist-ui/` via a subshell, so it doesn't disturb that shared
`cwd`.

Preflight checks confirm `backend/.venv/bin/python` and
`localist-ui/node_modules` exist before launch, failing fast with a
remediation command if either is missing.

### 13.3 Reload Directory Scoping

Backend's uvicorn invocation passes `--reload-exclude 'mcp_server/*'`;
`localist-mcp`'s passes `--reload-dir mcp_server`. Effect: editing backend
code reloads only the backend process; editing `backend/mcp_server/*.py`
reloads only the `localist-mcp` process. Both processes still resolve
imports from `cwd = backend/` — only the reload-watch scope is separated.

Design constraint to revisit if it becomes relevant: the current glob-based
exclude only matches direct children of `mcp_server/`, not deeper nested
paths. If `backend/mcp_server/` grows subpackages, this scoping will need
to be revisited.

### 13.4 Port Configuration

The frontend port is pinned via `localist-ui/vite.config.ts`
(`server.port: 5173`, `server.strictPort: true`) rather than left to
Vite's default-with-fallback behavior, so the script's port-based
assumptions (`--stop`, preflight warnings, banner URL) stay valid.

### 13.5 Test Coverage

This tooling has no automated test suite. Verification is live/manual:
service startup, log prefixing, Ctrl+C/`--stop` cleanup, and reload-scope
isolation.

### 13.6 Dev-Server Warmup — Chat UI Scroll Regression Fix (2026-07-06)

**Status: RESOLVED, live-verified (single session — 2026-07-06).** No prior
note in this document tracked the scroll/input-bar regression as an open
item; it was previously tracked only in `sessions-log.md` (2026-07-03,
2026-07-05 entries). This is the first entry for it in this file, added
alongside the fix rather than superseding anything here.

The regression (chat message list growing past the viewport, nothing to
scroll, input bar mispositioned/hidden) correlated with the point where the
Vite frontend service was folded into `start_localist.sh`'s unified
start/stop lifecycle (§13.1) — every restart of the full stack now also
cold-starts the Vite dev server, and it was on that cold start, specifically
on the first navigation to the chat route, that the symptom appeared. This
session is the first live-verified confirmation of that causal link; before
this it was a hypothesis only (see `sessions-log.md`, 2026-07-05 entry,
explicitly flagged there as unverified).

Root cause: Vite compiles modules on demand by default — `ChatPanel.svelte`,
its route parent `src/routes/conversation/[id]/+page.svelte`, and its child
`MarkdownRenderer.svelte` were only transformed the first time the chat
route was actually requested, so that first hit after every cold start paid
a compile-on-demand delay, during which the scroll/overflow symptom
manifested.

Fix: `localist-ui/vite.config.ts` (lines 16-24) now sets `server.warmup`,
listing those three files under `clientFiles` so Vite pre-transforms them
at dev-server startup instead of on first navigation:

```ts
warmup: {
	clientFiles: [
		'./src/routes/conversation/[id]/+page.svelte',
		'./src/lib/components/ChatPanel.svelte',
		'./src/lib/components/MarkdownRenderer.svelte'
	]
}
```

Deployment note: this is a `vite.config.ts` change, which Vite's HMR does
not pick up — it requires a full server restart (`./start_localist.sh --stop`
then `./start_localist.sh`), not a targeted kill of just the frontend
process. Per §13.1's unified-lifecycle design, killing one of the three
managed services directly is treated by the script as an unexpected crash
and cascades teardown of the other two — the stop/start pair is the correct
restart path here, not a manual process kill.

Confirmation: post-restart, first request to the chat route resolved in
362ms with no cold-transform delay, and the scroll behavior was confirmed
working normally on a fresh browser load post-restart. This is single-session
confirmation only — not yet re-verified across a second or third independent
cold start.

**Separate, still-open item — not fixed by the above:**
`[id]/+page.svelte`'s `loadConversationHistory()` (§12.5) throws on every SSR
render (`Cannot call fetch eagerly during server-side rendering with
relative URL ...`) because it's invoked from a top-level `$:` reactive block
rather than `onMount`/a `load` function. The error is caught and swallowed —
history falls back to a client-side re-fetch post-hydration — and it does
not suppress `ChatPanel`'s SSR-rendered CSS, so it is unrelated to the scroll
regression above. Confirmed via `logs/frontend.log`, firing on every
request. Remains open.

