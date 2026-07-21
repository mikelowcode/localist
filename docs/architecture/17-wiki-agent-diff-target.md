## 17. WikiAgent Diff-Target Path (Standalone Wiki Updates)

### 17.1 Problem this closes

Prior to this addition, `wiki_agent.py`'s only entry point was ingestion: `WikiAgent.run()`
unconditionally required `context["raw_path"]`, and diff proposals against existing pages only
ever happened as an optional side effect of ingesting a fresh raw file — the model was free to
skip them, and routinely did (confirmed live: ingesting a raw file describing the new Ollama
runtime backend and MCP tool layer produced a new research note but zero diffs against the stale
`localist-software-stack.md`, which should have been updated in the same pass). There was also no
routing path at all for an instruction like "update page X to reflect Y" with no file attached —
`conversational_agent.py` has no wiki-write capability, and `planner.py`'s Priority 1 only matches
on `raw_path` presence or ingest keywords.

### 17.2 `diff_target` context field

`SubTask.context` gains an optional key: `diff_target: str` — an existing wiki page stem (e.g.
`"localist-software-stack"`). `WikiAgent.run()` branches before path resolution:

- `raw_path` present → ingest path (unchanged).
- `diff_target` present, `raw_path` absent → `_run_diff_only()`.
- Both present → `raw_path` wins; `diff_target` is ignored and logged at debug level.
- Neither present → fails exactly as before (`raw_path` required).

`_run_diff_only()` resolves `wiki_dir`/`schema_path`/`templates_dir` the same way the ingest path
does, looks up `diff_target` in the loaded `wiki_pages` dict (fails with a clear error if the page
doesn't exist — does not silently fall through to `create_page`), and calls a new
`build_diff_prompt()` / `DIFF_SYSTEM_PROMPT` pair: schema + the single target page's full current
content + the instruction's free-text description of the desired change, under an
`apply_diff`-only `<actions>` contract (`create_page` actions returned on this path are discarded
with a warning, not silently written).

The shared post-inference tail — link validation, journaling, disk apply, `MemoryManager`
reindex, graph rebuild, result assembly — was factored out of `run()` into `_finalize()`, called
by both the ingest and diff-only paths, so the two entry points cannot drift on write/journal/index
semantics.

### 17.3 Planner routing — Priority 1b

`planner.py` gains a new priority, evaluated between P1 (ingest) and P2 (explicit memory
commands):

- `_DIFF_KEYWORDS`: `"update the wiki"`, `"update page"`, `"revise page"`, `"modify page"`,
  `"apply a diff to"` — narrow, explicit lead phrases, deliberately conservative (same
  false-positive-avoidance posture as `_WEB_SEARCH_KEYWORDS`/`_BACKLINK_LEAD_PHRASES`).
- `extract_diff_query()`: longest-first lead-phrase stripping (mirrors
  `_BACKLINK_LEAD_PHRASES`/`_OUTGOING_LEAD_PHRASES`'s pattern in `extract_graph_query()`), returns
  the remainder after the matched phrase, or `None` if no `_DIFF_KEYWORDS` phrase matches.
- Target resolution reuses `resolve_graph_target()` — the same three-tier deterministic pipeline
  (symmetric substring → token-overlap → give up) Priority 3c already uses for graph-query name
  resolution — fed by `memory_manager.list_graph_node_stems()`. No second fuzzy-matcher was
  introduced.
- On successful resolution: `RoutingPlan(agent="wiki_agent", diff_target=resolved_stem, ...)`.
  `RoutingPlan` gains a `diff_target: str | None = None` field, mirroring the existing
  `graph_query` field pattern.
- On a keyword match with failed/ambiguous resolution, or no `MemoryManager` available to resolve
  against: routes to `conversational_agent` with no tools/retrieval so the model can ask which page
  is meant. The turn is claimed either way — it deliberately does not fall through to P2+ on an
  unresolved target.
- P1b sits ahead of P2 unconditionally: both P1 and P1b are "write to storage" intents and stay on
  the same deterministic, early footing (decision made explicitly, not a default).

`controller_agent.py`'s `_execute_plan` threads `plan.diff_target` into the `wiki_agent`
`SubTask.context` at the same construction site that already carries `raw_path` for the ingest
path.

### 17.4 `apply_unified_diff` — content-based hunk location

The diff-only path's first live run (`gemma4:31b-cloud`, target
`localist-software-stack`) surfaced a pre-existing latent bug: `apply_unified_diff()` trusted the
model-authored `@@ -N,M +N,M @@` header as a literal line position (`idx = orig_start - 1 +
offset`). LLMs are not reliable at counting exact 1-indexed line numbers across a full-page prompt
— in the reproduced failure, the model's diff correctly reproduced real file text verbatim in its
`-`/`+` lines, but the header's claimed position (`line 21`) pointed at unrelated content
elsewhere in the file. This is a general weakness of asking any model to hand-author diff line
numbers rather than deriving them from a real diff tool, not specific to this model or prompt —
expected to recur on any page without a fix.

**Fix — `_locate_hunk()`:** hunk application now searches `result_lines` for a contiguous run
matching the hunk's `before` block (context + removed lines) by content, re-run against the
*current* state of `result_lines` for each hunk in sequence (not a single offset carried across
hunks). `orig_start` is retained only as a disambiguation hint when `before` matches at more than
one position — the nearest match to the hint is chosen deterministically, never the first match by
default. A `before` block matching nowhere in the file still raises the original `ValueError`
(genuine content mismatch, e.g. the page changed since the model read it) — this failure path is
unchanged; only the position-finding step changed.

**Second live-run finding — bullet/diff-marker collision:** a follow-up live run surfaced a second
failure mode: when a hunk touches a markdown bullet line, the bullet's own `"- "` prefix can
collapse into the diff format's `-`/`+` marker character, corrupting the parsed `before`/`after`
split. Fix: `_extract_hunk_lines()` retries once, on primary content-match failure, under the
assumption a bullet marker was collapsed into the diff marker. `build_diff_prompt()`'s rule 3 also
gained explicit guidance plus a worked example to reduce recurrence at the generation source, not
just recovery after the fact.

### 17.5 Live verification

Full chain verified live end to end, `gemma4:31b-cloud` via `OllamaRuntimeClient` against local
Ollama proxying to Ollama Cloud (see §16.2 for the runtime client itself):

1. Chat instruction `"Update page localist-software-stack to reflect the new Ollama runtime
   backend and MCP tool layer."` → P1b matched, `diff_target='localist-software-stack'` resolved
   cleanly via Tier 1 substring match → routed to `wiki_agent` → one `apply_diff` action returned,
   content-accurate (Ollama/`OllamaRuntimeClient`, `localist-mcp` port 8003 with
   `web_search`/`fetch_url`/`file_op`, ToolDispatcher/Fetcher retirement — all correctly derived
   from the target page's actual stale content) → `applied: false` as designed, since chat's
   `context` is always `{}` (`ChatPanel.svelte` → `submitTask(text, {}, ...)`), so `auto_apply`
   defaults `False` for every chat-originated diff instruction today. No UI affordance yet to set
   `auto_apply: true` from chat — out of scope for this pass, same follow-on status as the
   original diff_target UI wiring (§17.2 note).
2. `diagnostics/diag_wiki_agent_diff_only.py` (direct `WikiAgent.run()` call, bypassing planner —
   routing is covered by `TestPlannerP1bDiff` unit tests, this script's job is checking live model
   + apply behavior) with `AUTO_APPLY=True`: first run failed on the `apply_unified_diff` line-number
   bug above (`applied: false`, `diff_errors: ["localist-software-stack"]`, nothing written — safe
   failure). After the `_locate_hunk()`/`_extract_hunk_lines()` fix: `applied: true`,
   `diff_errors: []`, and `wiki/localist-software-stack.md` read back directly to confirm correct,
   well-formed on-disk content (Ollama, `localist-mcp`/port 8003, ToolDispatcher/Fetcher retirement
   all present; no garbling from the bullet-collision recovery path).

**Known gap, not yet closed:** `backend/wiki/` is listed in `.gitignore` (confirmed via `git
check-ignore -v`) — every `auto_apply=True` write, diff or fresh ingest, currently has no rollback
mechanism at all. This was surfaced during this session's live testing (an earlier claim that the
directory was git-tracked and revertible was incorrect) and is tracked as an open item below, not
yet fixed.

### 17.6 Test suite

613 tests passed after this change (up from the pre-existing baseline). New coverage:

- `test_wiki_agent.py`: `diff_target`-only dispatch without touching `_resolve_raw_path()`,
  disk-apply on the diff-only path, `create_page` discard-with-warning, missing-target failure,
  `raw_path`-wins-when-both-present, diff prompt shape (no "RAW FILE TO INGEST" section), plus
  four cases added for the `apply_unified_diff` fix: wrong-line-number recovery, ambiguous-match
  nearest-to-hint tiebreak, bullet-marker-collision recovery, and genuine-mismatch (still raises,
  still writes nothing).
- `test_planner_phase3.py::TestPlannerP1bDiff`: resolvable target, ambiguous target (routes to
  `conversational_agent` for clarification), no-keyword regression guard, P1-beats-P1b (ingest
  signal present takes priority), P1b-beats-P2 (diff keyword outranks memory-command keyword), and
  no-`MemoryManager` fallback.

### 17.7 Review-then-apply UI — Added 2026-07-09 (closes the UI wiring gap below)

Rather than exposing `auto_apply` as a chat-side toggle (no review step, and `wiki/` still has no
rollback — see the still-open safety-net item), the chosen design is review-then-apply:
`wiki_agent` proposes, the UI renders the proposal as a distinct block, and a write only happens on
an explicit user click.

**Backend — structured diff pass-through.** `Synthesizer.synthesize()` previously discarded
`AgentResult.output["diffs"]` entirely, re-summarizing every sub-agent output into prose via a
second inference call and returning only `metadata = {"subtask_count": ...}`. A new
`_build_wiki_diff_result()` branch, mirroring the existing `_build_conversational_result()`
special-case pattern, intercepts single-result `wiki_agent` dispatches with a non-empty `diffs`
list and passes the raw `{page_name, diff}` pairs through verbatim as
`metadata.pending_diffs`, alongside a short prose summary for chat readability. This method also
threads `plan.priority` into `metadata.priority` — fixing a provenance-chip bug the live UI test
surfaced (every wiki-diff turn was displaying "P6 · Inference" regardless of actual routing,
because the field was never set on this path).

**Backend — stateless apply endpoint.** New `POST /wiki/apply-diff` (`{page_name, diff}` body).
No new persisted "pending" state — the diff text round-tripped back from the client on Apply *is*
the full state needed. New `WikiAgent.apply_pending_diff()` reuses the existing content-matching
`apply_unified_diff()` (§17.4) plus the same disk-write/reindex/graph-rebuild tail `_finalize()`
already had for `auto_apply=True`, with no fresh model call. This doubles as the staleness check
for free: if the target page changed between proposal and click, the content match legitimately
fails and the endpoint returns 409 rather than corrupting anything.

**Frontend.** Chat renders each `pending_diffs` entry as its own reviewable block — colored
diff lines, separated from the prose answer — with Apply/Discard actions. Discard is client-only
(no request). Apply calls the new endpoint and, on success, flips the block to a persisted
"Applied" badge that survives a page reload (rather than reverting to Apply/Discard on next
render).

**Live verification, three additional bugs found and fixed beyond original scope, plus one
confirmed-open edge case:**

1. **Provenance chip mislabeling (fixed).** `_build_wiki_diff_result()` initially didn't set
   `metadata.priority`, so every wiki-diff turn showed "P6 · Inference" in the UI regardless of
   which priority actually routed it. Fixed by threading `plan` through into the new branch.
2. **Mid-file line merging from missing trailing newlines (fixed — this one briefly corrupted the
   live wiki file).** A model's final `+` line in a hunk frequently lacks a trailing newline even
   when the replaced line is mid-file, not the last line of the page — this silently glues the
   replacement onto the start of the next line (observed: `...usage-based.### Mapped Pages`, two
   sentences fused with no line break). Caught live via a real Apply click against
   `wiki/localist-software-stack.md`, which briefly left the file in a corrupted state; repaired
   immediately and confirmed clean afterward. Fixed with a post-apply newline-normalization pass in
   `apply_unified_diff()`.
3. **Bullet/diff-marker collision on context lines (confirmed, not fixed — safe, open
   limitation).** §17.4's bullet-collision fix covered removed/added lines; live testing confirmed
   the same collision can also occur on unchanged context lines within a hunk. Unlike the two bugs
   above, this fails safely today — the system correctly rejects the apply with a 409 rather than
   writing corrupted content — so it's being left as a known, non-corrupting limitation rather than
   fixed in this pass. Revisit if it starts blocking real diffs often enough to be worth generalizing
   `_extract_hunk_lines()`'s recovery further.

**Test suite:** 636/636 passed after this addition (up from 613). Frontend typechecks clean
(2 pre-existing unrelated errors, unchanged). Live end-to-end verification performed in a real
browser session, including the corruption-and-repair above; the 9 test conversations generated
during this verification pass were removed from chat history afterward (36 unrelated pre-existing
conversations left untouched).

### 17.8 Open items

- **Wiki write safety net — CLOSED, see §17.9.**
- **Bullet/diff-marker collision on context lines.** See §17.7 point 3 — confirmed, fails safely
  (409, no corruption), intentionally left unfixed for now.
- **Multi-diff turns.** The model has only ever proposed one diff per instruction to date; the
  `pending_diffs` data shape accommodates a list, but the UI only renders/exercises the single-diff
  case.

### 17.9 Wiki write safety net — Added 2026-07-10 (closes the item above)

Candidate fixes weighed: pre-write snapshot/backup step vs. bringing `wiki/` under git tracking.
Snapshot chosen — no git dependency, no lock/merge concerns if anything else touches the directory,
rollback is just "copy the last snapshot back."

**Design.** `wiki_dir / ".snapshots"` (subdirectory of the already-gitignored `wiki/`, no new
ignore rule needed; not surfaced in the UI or API). Naming:
`{page_name}.v{N}.{YYYYMMDDTHHMMSS}.md`, `N` = existing-snapshot-count-for-that-page + 1. 30-day
TTL via two independent cleanup paths, since prune-on-write alone only fires for pages that get
edited again and would leave orphaned snapshots for abandoned pages indefinitely:

- **Prune-on-write** — `_snapshot_page()`/`_prune_page_snapshots()`, called from both existing
  overwrite paths (`_apply_changes()`'s diffs loop, and `apply_pending_diff()` — the latter beyond
  the original prompt's literal wiring instructions, extended deliberately since it's the actual
  UI-driven path that caused §17.7's corruption). Non-fatal try/except, matching `_write_journal()`'s
  style.
- **Startup sweep** — `sweep_expired_snapshots()`, wired into `lifespan()` alongside the existing
  disk→DB wiki-reconciliation block, same non-fatal pattern.

TTL is runtime-overridable via `LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS` (uncached env read, default
2,592,000s / 30 days) — following `planner.py`'s `_tool_fallback_mode()` convention for standalone
tunables, since no precedent exists for this class of env var in `Settings`/`.env.example`.

All snapshot removals — both prune-on-write and startup-sweep — write to `wiki_maintenance.log` via
`wiki_maintenance_log.log_snapshot_pruned()`, one shared audit trail regardless of trigger.
(Prune-on-write was initially wired to the plain application logger only; the asymmetry was caught
during live verification below and unified same-day.) `_snapshot_page()` logs at INFO on both
success and failure — the success line was added after live verification found no way to confirm a
snapshot fired from backend logs alone, short of a filesystem check.

**Live verification — 2026-07-10, four steps against the running backend
(ports 8001/8003/5173), `LOCALIST_WIKI_SNAPSHOT_TTL_SECONDS=30` for test speed:**

1. Real diff-apply against `localist-build-order` → `Snapshotted 'localist-build-order' ->
   .../localist-build-order.v1.<ts>.md` logged at INFO; file confirmed on disk with correct
   pre-patch content.
2. Second apply 34s later → prune-on-write removed `v1`, left only `v2`. (Surfaced the
   audit-log asymmetry above — prune-on-write hadn't yet been wired to `wiki_maintenance.log`
   at this point; flagged rather than fabricating a log line, fixed same day.)
3. Manually backdated snapshot + restart, no intervening write to that page → startup sweep
   pruned it plus the now-expired `v2` from step 2 (`pruned=2`), both logged to
   `wiki_maintenance.log` with `snapshot_pruned` entries.
4. TTL override unset, restart confirmed effective TTL reverted to 30 days via direct runtime
   check, not assumption.

Two harmless test-artifact lines appended to `localist-build-order.md`'s Revision History by the
live-fire diff-applies in steps 1–2 were reverted by hand afterward (byte-for-byte match against
the pre-verification content).

**Test suite:** 578 → 653 passed, 0 failed, across four incremental sessions (snapshot + sweep
core; success-logging + TTL env override; audit-log parity for prune-on-write). No existing test
modified or removed at any step.

### 17.10 Priority 1c — Pinned-Wiki-Page Diff Short-Circuit — Added 2026-07-21

Builds on §11.8's wiki-page pinning (a `SessionFile` can carry
`source == "wiki_pin"`, populated via `POST /chat/pin-wiki-page`). Two gaps
in Priority 1b (§17.3) motivated this: (1) `extract_diff_query()`'s
`startswith()` lead-phrase requirement means an instruction like "Read my
live repo then propose diffs to update the wiki corpus" never matches P1b at
all, since the diff phrase isn't at the start; (2) even when a lead phrase
does match, if the extracted remainder doesn't resolve against
`list_graph_node_stems()`, P1b bounces the user to `conversational_agent` for
clarification — friction that shouldn't exist when the user has already
pinned the exact page they mean.

**New tier, `Planner._priority1c_pinned_diff()`**, evaluated between P1
(ingest) and P1b in `_route_impl()`. Fires only when exactly one
`wiki_pin`-sourced `SessionFile` is currently attached (checked via a direct
`import session_files as _session_files` in `planner.py`, mirroring the same
pattern already used by `controller_agent.py`/`conversational_agent.py` — no
constructor plumbing needed, since `session_files` is a process-global
cache, not a per-request dependency) and `_DIFF_KEYWORDS` matches anywhere in
the instruction (substring `in` check, not the lead-phrase `startswith()`
P1b uses — no remainder needs to be parsed out, since the pin already
disambiguates the target). Zero or 2+ pins → returns `None`, falling through
to P1b unchanged (P1b itself received zero code changes).

On match: `diff_target` is set directly from the pinned file's stem
(`Path(sf.filename).stem`, since a pin is always cached as `"{stem}.md"`),
skipping `resolve_graph_target()` entirely. `RoutingPlan` gains
`diff_target_source: str | None = None` — left `None` by every existing
path (including P1b), set to `"pinned"` only by this new tier, for
observability parity with the existing `tool_signal_source` attribution
field.

**Compound tool dispatch.** P1c also calls `self._priority3_tool(lowered,
instruction)` directly (reusing P3's full tool-signal detection — url_fetch,
web_search, file_op, deferred file_op — rather than duplicating any keyword
logic) and folds whatever it finds into the same `RoutingPlan` alongside
`diff_target`, so "fetch this URL and update the pinned page to reflect it"
produces one plan carrying both `diff_target` and `tools_to_call`. This is a
compound shape no existing priority tier produced for the diff path before
(`_detect_compound()`, §4.5, only ever pairs ingest with web_search).

**New plumbing to make the tool result actually reach the model.**
`WikiAgent` builds its diff-only prompt entirely internally and never reads
`_prebuilt_prompt`/`_prebuilt_system` — the mechanism every other agent uses
to see tool output. (This gap already existed for the ingest+web_search
compound path too, pre-dating this change; left unfixed there, out of
scope.) Fixed only for the diff-only path: `ControllerAgent._execute_plan()`
now writes a `subtask_context["tool_context"]` key — the joined `.result`
text of every successful dispatched tool call — right alongside the
existing `diff_target` wiring, only when non-empty (no empty-string key
clutter when a tool fails or nothing was dispatched).
`WikiAgent._run_diff_only()` reads `ctx.get("tool_context")` and passes it to
`build_diff_prompt()`, which — when present — splices a `# FETCHED CONTEXT`
section between the target page's content and the task instructions.
Omitted entirely (byte-for-byte identical output) when `tool_context` is
`None`.

**Test suite:** 958 → 967 passed, 0 failed. New coverage:
`test_planner_phase3.py::TestPlannerP1cPinnedDiff` (keyword-anywhere
short-circuit, tool-need compound routing, 2+ pins falls through unaffected,
0 pins confirmed via the tier's own guard), `test_controller_phase4.py::
TestPinnedDiffToolContextWiring` (`tool_context` wired on success, absent
when no successful tool result exists), and
`test_wiki_agent.py`'s diff-only section (`build_diff_prompt()`'s
`# FETCHED CONTEXT` section present/absent, and an end-to-end
`_run_diff_only()` case confirming the fetched text reaches the actual
model-facing prompt via a capturing runtime fake).

**Follow-up fix, same day — `_DIFF_KEYWORDS` coverage gap.** First live use
surfaced that P1c's own guard (exactly one pin + a `_DIFF_KEYWORDS`
substring) still failed to fire on two real follow-up phrasings — "Apply
diffs updating my new tool route logic." and "Propose diffs for the
localist-runtime-tooling-update wiki file." — despite a page being
unambiguously pinned. Neither matched the original five-phrase set (`"update
the wiki"`, `"update page"`, `"revise page"`, `"modify page"`, `"apply a
diff to"`); both requests fell all the way through to Priority 6
(conversational fallback), confirmed via the debug trail
(`Priority 3 — gate_fired=False` → `Priority 4 — corpus miss` → `Priority 5
— no episodic keyword matched` → `Priority 6 — direct answer fallback`).
Fixed by adding `"apply diffs"`, `"apply this diff"`, `"propose diffs"`,
`"propose a diff"` to `_DIFF_KEYWORDS` — shared by both P1b and P1c since
both consume the same frozenset, so P1b's lead-phrase matching gains the
same coverage. Regression tests added for both exact phrasings.
**Test suite: 967 → 969 passed, 0 failed.**
