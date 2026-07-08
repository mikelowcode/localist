# Localist Framework — Canonical Architecture Specification

> **Status: Authoritative**
> This document is the canonical reference for Localist Framework's substrate architecture.
> No implementation begins until it is reflected here. No deviation from this
> specification is made without updating this document first.

> **This file is now an index.** Full section content lives in
> `docs/architecture/NN-*.md` (one file per numbered section below). Retired
> sections live in `docs/architecture/archive/`. Read this index first; only
> open the specific section file you need. Edits to substance go in the
> section file, never back into this index — see "Keeping this index current"
> at the bottom.

---

## Core Constraints

Localist Framework is a **local-first, agentic general assistant**. Every architectural
decision is evaluated against five constraints:

| Constraint | Meaning |
|---|---|
| **Local** | All inference, embeddings, memory, and tools run on-device. No cloud calls except explicit user-initiated web search or page fetch. |
| **Sparse** | Memory is high-value semantic events, not transcripts. Prompts carry only what is needed. |
| **Predictable** | The same input produces the same routing decision. Inference is used for reasoning, not for control flow, except where explicitly specified. |
| **Minimal** | System prompts are small. Persona lives in the wiki. Agents are single-purpose. |
| **Auditable** | Every prompt can be logged and read. Every memory write has provenance. Every routing decision has a named rule. |

These constraints are not preferences. They are the identity of the system.

---

## Sections

| # | Section | Status | Last updated | Summary |
|---|---|---|---|---|
| 2 | [Episodic Memory Schema](docs/architecture/02-episodic-memory-schema.md) | Authoritative | 2026-06-14 | Defines the episode as the unit of durable memory — a sparse, typed, provenanced fact extracted from conversation (not a transcript or turn log) — plus its table schema, field reference, type taxonomy, lifecycle rules, retrieval modes, and summarization contract. |
| 3 | [Unified Prompt Contract](docs/architecture/03-unified-prompt-contract.md) | Authoritative | 2026-06-25 | Specifies the fixed multi-slot prompt layout (identity, persona, session files, episodic memory, RAG, tool results, working state, instruction), the aggregate token budget, `PromptBuilder`'s interface, and the ongoing KV-cache / stable-prefix investigation into cache efficiency under the current runtime. |
| 4 | [Planner Routing Model](docs/architecture/04-planner-routing-model.md) | Authoritative | 2026-07-07 | The deterministic, priority-ordered decision tree that routes each instruction to a tool, retrieval path, or direct-answer route; covers `RoutingPlan`/`ControllerResult` schemas, compound-instruction handling, the tool dispatcher, and Gemma 4B-specific behavioral constraints. |
| 5 | [Fetcher Service](docs/architecture/archive/05-fetcher-service-retired.md) | **Retired** (2026-07-03) | 2026-07-03 | The standalone FastAPI microservice (port 8002) that performed URL fetch + readability extraction from project inception through Phase 1. Retired in Phase 2 — its logic was ported in-process into the `fetch_url` MCP tool on `localist-mcp` (§14). Kept for historical reference only. |
| 6 | [Build-Order Checklist](docs/architecture/06-build-order-checklist.md) | Authoritative | 2026-06-14 | The strict, dependency-ordered build phases (memory substrate → prompt contract → planner → controller → episodic extraction → tool dispatcher → final integration) plus running session-progress notes. |
| 7 | [Localist UI](docs/architecture/07-localist-ui.md) | Authoritative | 2026-07-07 | The SvelteKit frontend: routes, provenance bar, episodic memory panel, API proxy, status bar and live-turn/SSE streaming behavior, chat history persistence, and the file browser's generated-files listing. |
| 8 | [Graph Retrieval Layer](docs/architecture/08-graph-retrieval-layer.md) | Authoritative | 2026-07-01 | The offline wiki link-graph (schema, `wiki_doc.py` shared parser, WikiAgent link validation, resolution rule) plus an extensive dated log of validation runs and open items, several since closed. |
| 9 | [Slot 6A — Structured Working State](docs/architecture/09-slot-6a-structured-working-state.md) | Authoritative | 2026-07-06 | The `[WORKING STATE]` prompt slot and its `WorkingMemoryState` dataclass, tracking current project and active artifacts across turns; schema, test suite, and open items (Tier 2 fields, render wiring). |
| 10 | [Semantic Search-Intent Classifier](docs/architecture/10-semantic-search-intent-classifier.md) | Authoritative | 2026-06-28 | Replaces literal-keyword web-search gating with an embedding-similarity classifier, added after a confirmed false-negative incident; covers design decisions, live re-verification, and a running log of threshold-tuning open items. |
| 11 | [Session File Attachments](docs/architecture/11-session-file-attachments.md) | Authoritative | 2026-07-06 | User-uploaded files injected into every subsequent prompt via Slot SF for the session's duration, bypassing planner routing and wiki ingestion entirely; backend cache, API endpoints, UI, and prompt-assembly integration. |
| 12 | [Chat History Tab](docs/architecture/12-chat-history-tab.md) | Authoritative | 2026-07-02 | Durable, searchable, user-manageable persistence of chat turns, distinct from both episodic memory and session files; schema, backend API, frontend, and live verification. |
| 13 | [Localist CLI](docs/architecture/13-localist-cli.md) (`start_localist.sh`) | Authoritative | 2026-07-06 | The single-lifecycle launcher managing backend, `localist-mcp`, and the UI dev server together (start/stop/log-tail as one unit), including the dev-server-warmup fix for a chat-scroll regression. |
| 14 | [localist-mcp / MCP Tool Layer](docs/architecture/14-localist-mcp-tool-layer.md) | Authoritative | 2026-07-07 | The MCP tool server exposing `fetch_url`, `web_search`, and file-op tools; `MCPToolDispatcher`, configuration, port topology, and test coverage. |
| 15 | [P6-Fallthrough Classifier & lookup_request Guard](docs/architecture/15-p6-fallthrough-classifier.md) | **Draft** (feature-flagged, not fully live) | 2026-07-08 | Two feature-flagged mechanisms — a shadow-mode classifier for tool-need fallthrough and a resolved-context guard for `lookup_request` — running live for trial data collection but not yet reflected in §4.2's routing tree. |
| 16 | [Runtime Backend Layer](docs/architecture/16-runtime-backend-layer.md) (`BaseRuntimeClient`) | Authoritative | 2026-07-08 | The structural-typing `BaseRuntimeClient` protocol and `runtime_factory.create_runtime()` swap-only backend selection; covers the oMLX, MLX-LM/Foundry, and newly-added Ollama backends. |

*Section 1 (System Identity) was folded into "Core Constraints" above — its original 17 lines didn't warrant a standalone file.*

---

## Keeping this index current

When you finish a change to any `docs/architecture/NN-*.md` file:

1. Update that row's **Last updated** date.
2. Re-read the **Summary** cell — rewrite it if the change was material, leave it if cosmetic.
3. Re-check **Status** — move to `Retired` (and relocate the file to `docs/architecture/archive/`) if the change makes the section dead; otherwise leave as `Authoritative` or `Draft`.

Do not add substantive content to this file directly — it stays index-sized so it's cheap to read in full every session.

---

*End of Localist Framework Canonical Architecture Specification (index)*
