## 4. Planner Routing Model

### 4.1 Design Principles

The Planner is a **rule engine**, not a classifier and not a free-form
inference call. It evaluates a priority-ordered set of conditions against
the instruction and current context. The first matching condition wins.

The Planner **never answers**. It produces a `RoutingPlan`. The
`ControllerAgent` executes the plan.

### 4.2 Priority-Ordered Decision Tree

Conditions are evaluated in strict priority order. The first match wins.
All lower priorities are skipped.

---

**PRIORITY 1 — INGEST SIGNAL**

| | |
|---|---|
| **Condition** | Explicit file path present in context (`raw_path` key) OR ingest keyword detected in instruction (`"ingest"`, `"process this file"`, `"add to wiki"`, `"index this"`) |
| **Action** | Route to `WikiAgent`. Set `fetch_rag = False`, `fetch_episodic = False`. |
| **Rationale** | Ingest is never ambiguous. Fast-pathing prevents any possibility of incorrect agent scheduling. |

---

**PRIORITY 2 — EXPLICIT MEMORY COMMAND**

| | |
|---|---|
| **Condition** | Explicit memory signal detected: `"remember that"`, `"my preference is"`, `"that's wrong"`, `"the correct value is"`, `"forget that"`, `"mark complete"`, `"that's no longer true"` |
| **Action** | Route to `EpisodicMemoryWriter` first (extract and store, with subject normalization per §2.8). Then proceed to Priority 4 or 6 for the response. Set `write_episode = True`. |
| **Rationale** | These signals are deterministic and safe. The memory write always precedes the response. |

---

**PRIORITY 3 — TOOL SIGNAL**

| | |
|---|---|
| **Condition** | Web search keywords (`"latest"`, `"current price"`, `"current version"`, `"current ceo"`, `"current status"`, `"current rate"`, `"news"`, `"recent"`); OR file operation keywords (`"read the file"`, `"read file"`, `"open the file"`, `"create a file"`); OR URL fetch keywords (`"fetch this"`, `"fetch the url"`, `"read this link"`, `"read this url"`, `"open this link"`, `"summarize this url"`, `"summarize this link"`, `"extract this"`); OR any `http://` or `https://` URL present in the instruction. |
| **Action** | Dispatch appropriate tool(s). Populate `RoutingPlan.tools_to_call`. Tool results populate Slot 5 before ConversationalAgent runs. |
| **Rationale** | Tool results are the freshest possible evidence and must be gathered before any RAG retrieval. |
| **Notes** | All single-word keywords use `_any_whole_word()` with `\b` regex anchors to prevent substring false positives. Multi-word phrases (`"current version"`, `"read the file"`) carry no false-positive risk. The URL regex (`https?://`) automatically triggers `url_fetch` when any link is dropped into the instruction. |

---

*Correction 2026-07-06 — bare `"today"`/`"write"`/`"save"` pruned from the keyword lists above.* Live traffic showed the deterministic keyword branch firing independently of, and before, the semantic gate's judgment: an utterance like "Did you know I added a new file read / write / append tool to my Localist app today?" correctly scored non-search by the semantic gate (`lookup_request=0.532 < 0.65`, `gate_fired=False`) but still triggered both `web_search` (on bare `"today"`) and `file_op` (on bare `"write"`) via the keyword branch, which runs unconditionally regardless of the semantic result. Fix: removed `"today"` from `_WEB_SEARCH_KEYWORDS` and `"write"`/`"save"` from `_FILE_OP_KEYWORDS` — all three were too common in ordinary conversational sentences to serve as reliable single-word tool signals. No replacement phrases were added; instructions that used to hit these bare words now fall through to P3b/P4/P5/§15.1's P6 classifier, which is the intended effect, not a gap to be immediately recaptured. Two tests keyed to the removed words (`test_file_op_guard_defers_to_p3`, `test_p3c_beats_web_search_p3` in `test_planner_phase3.py`) were rewritten to use surviving keywords (`"create a file"`, `"recent"`) — same P3c-ordering behavior under test, just a different trigger word; full suite restored to the 572 passed / 2 failed baseline (the 2 being the pre-existing, unrelated failures tracked in §14.6).

---

**PRIORITY 3b — FACTUAL QUERY + CORPUS MISS**

| | |
|---|---|
| **Condition** | Instruction contains a factual query keyword (`"when did"`, `"what year"`, `"who founded"`, `"who invented"`, `"who created"`, `"where was"`, `"how many"`, `"what is the"`, `"which company"`, `"who was the first"`, `"what was the first"`) AND `MemoryManager.query_corpus()` returns no result with `relevance_score >= 0.55`. |
| **Action** | Schedule `web_search` via `tools_to_call`. Route to `ConversationalAgent`. |
| **Rationale** | Factual queries about the external world should go to web search when the corpus has no strong hit. Corpus is checked first to avoid unnecessary API calls when the answer is already in the vault. |
| **Notes** | Requires `MemoryManager` to be available. Skipped entirely when no MemoryManager is present. When corpus returns a hit (score ≥ 0.55), Priority 3b returns `None` and evaluation falls through to Priority 4. |

---

**PRIORITY 4 — CORPUS SIGNAL**

| | |
|---|---|
| **Condition** | **Path A:** Instruction contains an explicit wiki/vault trigger keyword (`"check the wiki"`, `"search the wiki"`, `"from the wiki"`, `"in the wiki"`, `"vault"`, etc.). **Path B:** `MemoryManager.query_corpus()` returns a top result with `relevance_score >= 0.55`. Either path is sufficient to match. |
| **Action** | Run RAG retrieval. Set `fetch_rag = True`. Snippets populate slot 4. Path A also sets `fetch_episodic = True`. Path B sets `fetch_episodic = False` (episodic is evaluated independently at P5). |
| **Rationale** | Path A keeps routing deterministic for explicit wiki requests. Path B restores score-based RAG injection for natural-language corpus queries that carry no trigger keyword (e.g. "summarize the Localist Master Project Outline"). Without Path B, ingested documents are unreachable unless the user knows to say "check the wiki". |

---

**PRIORITY 5 — EPISODIC RELEVANCE**

| | |
|---|---|
| **Condition** | Instruction contains a personal reference or episodic relevance keyword. Personal reference keywords (always return `fetch_episodic=True` immediately): `"my name"`, `"do you remember"`, `"who am i"`, `"what do you know about me"`, `"my preference"`, `"my setup"`, `"what did i tell you"`, `"what have i told you"`. General episodic keywords: `"preference"`, `"preferences"`, `"remember"`, `"remembered"`, `"you know about me"`, `"what do you know"`, `"decision"`, `"decisions"`, `"decided"`, `"correction"`, `"corrections"`, `"wrong"`, `"workflow"`, `"workflows"`, `"last time"`, `"previously"`, `"before"`, `"my project"`, `"my environment"`. |
| **Action** | Run episodic retrieval. Set `fetch_episodic = True`. Bullets populate Slot 3. |
| **Rationale** | Deterministic keyword matching is faster and cheaper than a model-based relevance call. Personal reference phrases are unambiguous and bypass keyword evaluation — they always fetch episodic memory. |
| **Session flag:** | Once episodic bullets have been injected this session, `mark_episodic_injected()` is called. **Session flag caching:** Once episodic bullets have been injected this session, the relevance inference call is skipped on subsequent turns — but keyword evaluation still runs. A turn with no episodic keyword returns `None` and falls through to P6. The flag suppresses the inference cost only, not the routing decision. |

---

**PRIORITY 6 — DIRECT ANSWER**

| | |
|---|---|
| **Condition** | None of the above triggered. |
| **Action** | Route to `ConversationalAgent` with Slots 1–3 only (system + working memory + instruction). |
| **Rationale** | General knowledge questions need no retrieval. The model answers from its own weights plus working memory. |

---

*Note: Priority 4a (`_priority4a_identity()`) was removed on 2026-06-26. Identity-style questions now fall through to P4 Path B (corpus scoring, threshold ≥ 0.55) or P6 (direct answer fallback) depending on corpus score. See Open Item 12 (§8.8) for the full removal record.*

---

### 4.3 Priority 5 — Deterministic Episodic Relevance Check

Priority 5 uses a deterministic keyword check. No inference call is made.

**Implementation:** Scan the lowercased instruction for membership in
`_EPISODIC_KEYWORDS` and `_PERSONAL_REF_KEYWORDS` (defined in `planner.py`).
Personal reference keywords return `fetch_episodic=True` immediately.
General episodic keywords also return `fetch_episodic=True` on first match.

**Caching rule:**
- Once episodic bullets have been injected this session (`_episodic_injected = True`),
  the inference call is skipped on subsequent turns.
- Keyword evaluation still runs regardless of the flag. A turn with no matching
  episodic keyword returns `fetch_episodic = False` and falls through to P6.
- The flag suppresses inference cost only — it does not force `fetch_episodic = True`
  unconditionally.

**Why inference was removed:** Gemma 4B (`gemma-4-e4b-it-4bit`) requires
`max_tokens ≥ 300` to produce reliable output on binary classification
prompts. Below this threshold the model consistently returns a bare newline.
A 300-token budget for a yes/no routing decision is incompatible with the
**Sparse** and **Predictable** constraints in §1.

### 4.4 RoutingPlan Structure

```python
@dataclass
class RoutingPlan:
    agent:             str            # "wiki_agent" | "conversational_agent"
    fetch_episodic:    bool           # True → retrieve from episodes table
    fetch_rag:         bool           # True → query_corpus() before responding
    tools_to_call:     list[str]      # tool names in dispatch order; [] if none
    write_episode:     bool           # True → EpisodicMemoryWriter runs first
    episode_type:      str | None     # type hint for extraction; None if not write
    compound:          bool           # True → multiple signal types detected
    priority:          int            # 1–6; which priority rule matched (default 6)
    file_op_deferred:  bool           # True → file_op content must be generated
                                       # first; see §4.4b. Default False.
    file_op_path:      str | None     # deferred file_op destination; None unless
                                       # file_op_deferred
    file_op_action:    str | None     # "write" | "append"; None unless
                                       # file_op_deferred
```

**Execution contract for `ControllerAgent.handle_task()`:**

1. Receive `RoutingPlan` from Planner.
2. If `write_episode`: run `EpisodicMemoryWriter`, wait for completion.
3. If `tools_to_call`: dispatch tools in listed order, collect results.
4. If `fetch_rag`: run `MemoryManager.query_corpus()`, collect snippets for Slot 4.
   RAG results are filtered by `relevance_score >= 0.55` unconditionally (still filtered
   for `lora-persona.md` exclusion). Maximum 3 sources.
5. If `fetch_episodic`: run episodic retrieval, collect bullets for Slot 3.
6. Call `PromptBuilder.build()` with all collected content; persona is loaded
   from `_load_persona()` (cached) and passed as `persona=` for Slot 1b.
7. Call `RoutingPlan.agent` with the assembled prompt.

The Planner never calls agents, never calls tools, and never touches the
database. It is pure decision logic.

### 4.4a ControllerResult — API Response Schema

`ControllerAgent.handle_task()` returns a `ControllerResult` dict that is
serialised directly to the HTTP response by `main.py`.

```python
{
    "task_id":  str,
    "status":   "complete" | "failed",
    "answer":   str,
    "sources":  list[SourceItem],   # see below
    "metadata": ResponseMetadata,   # see below
    "error":    str | None,
}
```

**`SourceItem`** — typed source reference:
```python
{
    "path": str,              # absolute path on disk
    "type": "wiki" | "raw",  # classified by path prefix
    "name": str,              # human-readable title derived from filename
}
```

**`ResponseMetadata`** — routing provenance:
```python
{
    "agent":          str,         # agent that produced the answer
    "priority":       int,         # 1–6; which Planner rule matched
    "fetch_rag":      bool,        # True if RAG retrieval ran
    "fetch_episodic": bool,        # True if episodic memory was injected
    "tools_fired":    list[str],   # tool names that executed this turn
    "grounded":       bool,        # True if any corpus context was injected
    "file_op_deferred": bool,      # True if a deferred file_op ran this turn
                                    # (§4.4b) — tools_fired stays [] for it,
                                    # since file_op never entered tools_to_call
}
```

This metadata is emitted in the SSE stream as the `"done"` event payload
and consumed by Localist UI's provenance bar (see §7.3).

### 4.4b Deferred file_op — Content-Present vs. Generation-Required (2026-07-07)

**New capability, not a bug fix.** Builds directly on §14.7 Open Item 1
(silent empty-file write, RESOLVED) and Open Item 3 (`[TOOL FAILED]` slot,
RESOLVED) — both of those fixes made a content-less `file_op` fail loudly
instead of silently; this feature is what stops the failure from happening
in the first place for the single largest class of instruction that caused
it: "write a haiku about the sea and save it as haiku.md," where the content
doesn't exist anywhere in the instruction for `_derive_file_op_content()`
(§4.6/§14.3) to extract, because the model has to compose it first.

**Design.** `Planner._priority3_tool()` (`planner.py`) now imports
`_derive_file_op_action/_path/_content()` and `_FILE_OP_PATH_PATTERNS`
directly from `mcp_tool_dispatcher.py` rather than re-implementing them, so
the planning-time content-present check can never drift from what dispatch
time actually derives. Once a `file_op` match is found (literal
`_FILE_OP_KEYWORDS` hit, or a destination phrase like `"save it as X.ext"`
via `_FILE_OP_PATH_PATTERNS`), a `write`/`append` action additionally checks
whether content is already derivable from the instruction (fenced block,
quoted span, or `"with the content"`/`"containing"`/`"that says"` phrasing):

- **Content present** → unchanged old behavior: `"file_op"` is added to
  `tools_to_call`, dispatched pre-generation exactly as before.
- **Content generation-required** → `"file_op"` is deliberately *not* added
  to `tools_to_call` (there is nothing to dispatch yet). Instead
  `RoutingPlan.file_op_deferred = True`, with `file_op_path`/`file_op_action`
  pre-resolved via the same derivation functions, so a later controller step
  can dispatch once the answer exists. A bare `read` action never needs
  content and always takes the immediate-dispatch path regardless.

**Controller-side dispatch (`controller_agent.py`, `_execute_plan` Step
7b).** Runs after the agent's answer has been generated, before the
early-completion signal fires (the SSE `'done'` event). `_extract_file_op_content()`
strips model meta-commentary framing the answer before it's saved:
an optional leading title-style label line (`"Haiku about the sea:\n\n"`)
and an optional trailing parenthetical aside (`"(Saved to haiku.md)"`),
each strip guarded so it can never zero out real content if the pattern
false-positives. The extracted content is dispatched through the same
`MCPToolDispatcher` path the content-present case already used (§4.6),
and a deterministic, code-generated result line — never model-narrated, so
it can't be fabricated — is appended to the answer: `*(Saved to {filename})*`
on success (filename read from `write_file`'s own `"OK: wrote N characters
to {name}"` response text, so a version-collision rename like `haiku_2.md`
is reported correctly) or `*(Could not save — {reason})*` on failure.

**Content-extraction bug caught live during verification, fixed same
session.** The first live test (task `a44232b3-9f87-422e-b689-d24c01c8e9c9`,
13:50:51, "write a haiku about the sea and save it as haiku.md") wrote the
model's *entire* answer — including its own trailing aside — verbatim to
disk: `write_file` received `content='Blue expanse so wide,\nWaves crash on
the sandy shore,\nSalt wind fills the air.\n\n*(This haiku has been
generated and is ready to be saved as \`haiku.md\`.)*'` (153 characters).
Root cause: the model wrapped its whole aside in markdown italics with a
backticked filename — `*(...)*` — and the trailing-parenthetical regex only
matched a bare `(...)`, not one wrapped in `*...*`. Fixed by making the
regex's surrounding `\*` optional. Re-run immediately after the fix
(`controller_agent.py` reload at 13:52:12; task `b176493e-5d22-4882-876b-79541d20be67`,
13:52:48, same instruction) wrote a clean 78-character haiku with no
meta-commentary. New test `test_markdown_italicized_trailing_aside_is_stripped`
in `TestExtractFileOpContent` (`test_controller_phase4.py`) locks this in.

**Three further live repros, same session, after the italics fix:**

- **Version-cap failure** (task `7f4efc0f-23f5-43eb-a61c-1c0dd56eb500`,
  13:54:58, same haiku.md instruction repeated past `write_file`'s
  10-version collision cap — see §14.4/§4.6) — `write_file` returned
  `"ERROR: version cap reached — 10 versions of 'haiku.md' already exist"`;
  `_execute_plan: deferred file_op dispatched — action=write path=haiku.md
  success=False.` The model's clean (non-fabricated) haiku content was
  generated correctly; only the save failed, and failed loudly.
- **Content-present regression check** (task `bf84eac2-5a83-4dea-8001-b5400409cfe4`,
  13:55:49, "create a file called regression_check.md with the content
  hello world from water") — confirmed the content-present path still
  dispatches immediately, not deferred: `Planner: Priority 3 — file_op
  signal detected (kw='create a file' dest_match=None action='write'
  content_present=True)`, `tools=['file_op']` in the plan, 28 characters
  written in the same turn.
- **Clean success, no regression** (task `a411f1aa-1d09-46c1-99bf-3b7fff6e21d5`,
  14:05:07, "Write a haiku about autumn and save it as autumn.md") — 86
  characters written with no meta-commentary; this file is still on disk at
  `backend/generated_files/autumn.md` as of this writing. Two further clean
  repros followed the same session (`moon.md`, 14:11:45; `ocean.md`,
  14:46:44), the latter also used for §7.3's provenance-badge verification.

**Test suite:** 3 new `Planner` tests
(`test_p3_file_op_content_present_quoted_dispatches_immediately`,
`test_p3_file_op_content_present_fenced_dispatches_immediately`,
`test_p3_file_op_generation_required_is_deferred`, all in
`test_planner_phase3.py`) plus 15 new `controller_agent.py` tests across
`TestExtractFileOpContent` (8), `TestFileOpConfirmationLine` (4), and
`TestDeferredFileOpDispatch` (3, in `test_controller_phase4.py`) — 595
tests total, 593 passed / 2 failed, same 2 pre-existing network-dependent
failures tracked in §14.6, unchanged before and after.

**Blocker hit mid-session, resolved same session.** `test_file_op_guard_defers_to_p3`
(`test_planner_phase3.py`, part of §21's 2026-07-06 keyword prune — see
`sessions-log.md`) still asserted against `"save the results"`, a phrase
that only worked as a `file_op` trigger back when `"save"` was in
`_FILE_OP_KEYWORDS`; that keyword was already pruned in this same
uncommitted working tree, so the test was exercising dead wording rather
than the P3c-deferral behavior it was meant to cover. Rewritten to
`"create a file with the results"` and re-asserted against the new
`file_op_deferred`/`tools_to_call` split rather than the old single
`"file_op" in tools_to_call` check — the graph-query result ("the
results") is exactly the generation-required case this feature exists for.

**Known coverage gap, left for backlog, not fixed this session.** No test
(and no live repro) exercises an `append`-shaped generation-required
instruction — e.g. "write a haiku and append it to notes.md" — end to end.
`_derive_file_op_action()`'s `append` keyword group (`"append"`, `"add to
the file"`, `"add this to"`) is exercised by existing dispatch-time tests,
and the deferred-split logic itself is action-agnostic (it only branches on
whether `action in ("write", "append")` needs content), so this is assessed
as low-risk, not a known-broken path — but it has not been live-verified,
unlike the `write` case above.

### 4.5 Compound Instruction Handling

A compound instruction triggers two or more priority conditions simultaneously.

**Tool + Ingest compound**
Example: *"Search for the latest oMLX release notes and update the wiki."*

Triggers: Priority 1 (ingest) and Priority 3 (tool).

Resolution: Tool call executes first. Result is passed as `raw_path`
context to `WikiAgent`. The `RoutingPlan` sets
`tools_to_call = ["web_search"]`, `agent = "wiki_agent"`,
`compound = True`.

**Episodic + RAG compound**
Example: *"What did we decide about the vault resolver?"*

Triggers: Potentially Priority 4 (explicit wiki query) and Priority 5
(episodic, if stored as a decision).

Resolution: Both `fetch_rag=True` and `fetch_episodic=True` are set on
the same `RoutingPlan`. Both retrievals run before the agent call.

### 4.6 Tool Dispatcher

> **Superseded (Phases 1–4, 2026-07-03).** `ToolDispatcher` (`tool_dispatcher.py`)
> was the original in-process implementation described below through 2026-06-28.
> It has been fully migrated to `MCPToolDispatcher` calling out to the
> `localist-mcp` service over MCP, and the legacy class was deleted in Phase 4
> once nothing referenced it. See §14 for the current architecture — this
> section is kept as the historical record of the pre-MCP design.

`MCPToolDispatcher` (`mcp_tool_dispatcher.py`) executes tool calls specified
in a `RoutingPlan` and returns `ToolResult` objects for injection into
Slot 5. All three registered tools are served by the `localist-mcp` MCP
server (port 8003) — see §14 for the full tool contracts, transport, and
error-shape details; the table below is kept at the same level of detail
the original `ToolDispatcher` table had, pointed at the real
implementation.

**Registered tools:**

| Tool name | Trigger | Implementation |
|---|---|---|
| `web_search` | P3 web keywords or P3b factual + corpus miss | MCP tool `web_search` on `localist-mcp` — LangSearch API (`https://api.langsearch.com/v1/web-search`), ported verbatim from the original implementation. Returns top 3 results as formatted bullets. Missing `LANGSEARCH_API_KEY` now raises a clean error (`success=False`) — the old inference-stub hallucination fallback was removed in Phase 3 (see §4.6.1). Max 3 queries per dispatch call. |
| `file_op` | P3 file keywords (`"read the file"`, `"write"`, `"open the file"`, `"save"`, `"create a file"`) | MCP tools `read_file`/`write_file`/`append_file` on `localist-mcp`. Paths resolved relative to a sandbox root and validated — no path traversal outside it permitted. Max 4000 chars on read. |
| `url_fetch` | P3 URL fetch keywords or any `https?://` URL in instruction | MCP tool `fetch_url` on `localist-mcp` — readability extraction ported in-process from the retired standalone Fetcher microservice (§5). Returns title, source URL, word count, and full extracted text. PromptBuilder enforces Slot 5 ceiling. |

**LangSearch integration (unchanged request/response contract, now async):**
- Endpoint: `POST https://api.langsearch.com/v1/web-search`
- Auth: `Authorization: Bearer {LANGSEARCH_API_KEY}` (from `backend/.env`, loaded by `localist-mcp`'s own `load_dotenv()` call — a separate process from the main backend, so it does not inherit the backend's dotenv load)
- Request: `{"query": q, "summary": true, "count": 3, "freshness": "noLimit"}`
- Result format: `• {name}\n  {body[:300]}\n  [{displayUrl}]` per result
- Call made via `httpx.AsyncClient`, not the legacy synchronous `requests` call

#### 4.6.1 Corpus fallback on `web_search` failure (added 2026-06-28)

**Design constraint.** `Planner.route()` commits to one priority branch per turn before any tool executes. It has no way to know in advance whether `web_search` will fail, so a routing-layer fix is not possible — the fallback lives in `_execute_plan()`, after tool dispatch and before final answer generation.

**`ToolResult.success` field.** `ToolResult` in `prompt_builder.py` gained a `success: bool = True` field, defaulting `True`. All pre-existing construction sites in `tool_dispatcher.py` required zero changes. The two `web_search` exception-handling branches — the LangSearch API exception path and the inference-stub exception path — now set `success=False` alongside the existing `result = f"ERROR: ..."` string. The string is retained for logging and Slot 5 display; the boolean is the structured signal `_execute_plan()` checks.

**Step 3b in `_execute_plan()`.** Inserted between Step 3 (tool dispatch) and Step 4 (RAG fetch). If any dispatched result has `tool_name == "web_search"` and `success == False`, `_execute_plan()` calls `self._memory_manager.query_corpus()` directly using the original instruction (`max_results=3`, `use_embeddings=True`). Results with `relevance_score ≥ 0.55` that do not match `lora-persona.md` are wrapped as `RagSource` objects and injected into `rag_sources` — the same list that Step 4 populates for normal P4 routes, and that PromptBuilder reads as Slot 4 RAG context. The 0.55 threshold and persona-exclusion guard are identical to those applied in Step 4; corpus fallback is intentionally a like-for-like substitute for normal RAG grounding. If no results clear the threshold, `rag_sources` stays empty and the pipeline falls through unchanged to its existing honest "I don't have live results" framing. Scoped to `web_search` failures only — `file_op` and `url_fetch` failures are explicitly out of scope for this mechanism. (The `research` tool's bounded search loop also feeds this same check via its own synthetic `tool_name="research"` failure result on exhaustion, without changing the check itself — see §18.)

**Verification status (superseded — see update below).** 436/436 tests passed before and after this change (confirmed by Claude Code). Live verification is partial: a LangSearch outage occurred once on the same day *before* this fix shipped (confirmed during fabrication-correction-fix testing) and once *after* it shipped, but the second occurrence was a LangSearch SUCCESS (3 real results returned), not a failure — the new fallback code path has **not yet been exercised under real failure conditions**. This is an open verification gap; the fix is not confirmed-working under live outage conditions.

#### 4.6.1 Update — Corpus Fallback Live-Verified Under Real Failure (2026-07-03)

**The verification gap above is closed.** Phase 3 (web_search migration to
`localist-mcp`, 2026-07-03) live-verified Step 3b end-to-end under a real
failure condition: with `LANGSEARCH_API_KEY` forced empty, a `web_search`
instruction produced `success=False`, `_execute_plan()`'s log confirmed
`"web_search failed — corpus fallback found 3 relevant source(s)"`, and
the assembled prompt's `[CONTEXT]` slot contained real indexed wiki
content (`how-localist-works.md`) that visibly grounded the model's
answer. Zero `runtime.infer()` calls occurred on this path, confirmed by
log inspection between the tool failure and final prompt assembly.

**Locked behavior change, not a preservation.** Prior to Phase 3, a missing
`LANGSEARCH_API_KEY` triggered a fallback that called `runtime.infer()` to
generate plausible-sounding bullet points and returned them as
`success=True` — model-hallucinated content indistinguishable from a real
search result to every downstream consumer, including this very corpus
fallback (a hallucinated "success" would never have triggered Step 3b at
all). That fallback was removed entirely in Phase 3: a missing API key now
always produces a clean `success=False` `ToolResult`
(`"ERROR: LANGSEARCH_API_KEY not configured"`), with no inference call
anywhere on that path. This is what makes Step 3b reliably reachable under
the exact condition it was designed for.

Open Item 5 in §10.4 (SUCCESS with irrelevant results) is a distinct,
still-open failure mode — see that section; it is not addressed by this
update.

### 4.7 Gemma 4B Behavioral Constraints

Live testing revealed several behavioral constraints of `gemma-4-e4b-it-4bit`
that affect prompt and inference call design. These are architectural
constraints, not implementation details.

**Binary classification floor (`max_tokens`)**
Gemma 4B returns a bare newline (`'\n'`) on binary yes/no classification
tasks when `max_tokens < 300`. All bounded inference calls that expect short
output must use `max_tokens ≥ 200` or be replaced with deterministic Python
logic. The preference is always deterministic Python over a model call for
binary decisions.

**Extraction call minimum (`max_tokens`)**
The episodic extraction call requires `max_tokens = 200` to reliably produce
a one-sentence output.

**PromptBuilder `[USER]\n` wrapper incompatibility**
The `[USER]\n` slot label combined with imperative instructions causes Gemma 4B
to return bare newlines on short-budget inference calls. Extraction calls
construct their user prompt directly rather than passing through
`PromptBuilder.build()`. This is a documented architectural exception.

**Temperature**
`temperature = 0.0` produces degenerate output on extraction tasks. All
bounded extraction calls use `temperature = 0.1` as the minimum viable value.

**Separate normalization prompt incompatibility**
A standalone normalization prompt (`max_tokens=60`, `temperature=0.1`)
reliably returns `'\n'` from Gemma 4B 4-bit — insufficient output budget
for the model to produce a complete sentence. Subject normalization must
derive from the already-normalized `content` string produced by the main
extraction call, not from a separate model call. See §2.8.

**Structured-output field label "SUMMARY" triggers EOS at position 1 (word-level sensitivity)**
Controlled A/B testing at `temperature=0.0` on the Slot 6A working-state
extraction prompt revealed that adding a `SUMMARY:` field as a fourth
structured-output label causes Gemma 4B to emit near-100%-probability EOS
at the first output token — producing zero content — on every sample (0/3
success rate). The same prompt with only three labels (`FOCUS:`,
`OPEN_LOOPS:`, `DECISIONS:`) succeeded on every sample (3/3). All other
variables were held constant: system prompt structure, user prompt format,
`max_tokens`, and `temperature=0.0`.

This is evidence for word-level prompt sensitivity, not a token-budget issue.
The `SUMMARY` label itself — not the added length — appears to trigger the
failure. `max_tokens` was unchanged between the A and B conditions.

**Working theory (unverified hypothesis):** "SUMMARY" carries
document-closing semantics from pretraining — summary sections
characteristically appear near the end of documents, making EOS a
high-probability continuation after that token. This hypothesis is consistent
with all observed evidence but the underlying mechanism has not been confirmed.
Do not treat it as established fact.

**Distinction from the nearby temperature finding above:** The entry
"Temperature — `temperature = 0.0` produces degenerate output on extraction
tasks" refers to episodic extraction tasks and concerns output quality across
a general extraction contract. This finding concerns a specific structured-output
prompt where a single field label name drives near-certain EOS independently of
temperature or token budget. These are not the same root cause and should not
be conflated. See §9.2 for the full diagnostic arc and the decision to remove
the SUMMARY field entirely from Slot 6A Tier 2.

**Implication for the rest of the codebase:** Any structured-output prompt that
includes a `SUMMARY:` or similar document-closing field label should be treated
as a risk for this failure mode, particularly at `temperature=0.0`.

