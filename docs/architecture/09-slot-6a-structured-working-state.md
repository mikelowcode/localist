## 9. Slot 6A — Structured Working State

### 9.1 Scope

**Implemented:**

- `WorkingMemoryState` dataclass in `prompt_builder.py` — two Tier 1 fields:
  `current_project: str | None` and `active_artifacts: list[str]`.
- `_slot6a_working_state()` and `_CEIL_WORKING_STATE = 100` in `prompt_builder.py` —
  renders `[WORKING STATE]` block with clean omission when both fields are empty/None.
- Step 5d in `controller_agent._execute_plan()` — assembles `WorkingMemoryState`
  deterministically from current-turn RAG sources and `task.context["project_context"]`;
  gated on `plan.graph_query is None` to suppress rendering on P3c turns.
- `working_state` SQLite table (`memory_manager.py`, v3→v4 migration).
- `WorkingStateRecord` dataclass and `WorkingStateStore` class in `memory_manager.py` —
  `get()`, `upsert()`, `clear()` methods.
- Tier 2 extraction in `episodic_extractor.py` — `_WORKING_STATE_UPDATE_SYSTEM` (3-field
  structured prompt), `extract_working_state_update()` (parse + return tuple), and
  `process_working_state_update()` (orchestrate + upsert). Wired into
  `controller_agent.py` post-response hook; runs regardless of `plan.graph_query`.
- v4→v5 migration (`memory_manager.py`) — removes `turn_summaries_json` column dropped
  after live diagnostic testing confirmed 4-field format failure (see §9.2).
- `tests/test_episodic_phase5.py` — covers full Tier 2 extraction arc including the
  new `test_exactly_three_labels_required` test added in this session.
- `tests/test_memory_phase1.py` — updated for v5 schema (version assertion, column
  checks, `upsert()` call sites).

**Explicitly deferred or not yet wired:**

- Tier 2 fields (`current_focus`, `open_loops`, `recent_decisions`) are stored and
  updated but **do not yet appear in the rendered `[WORKING STATE]` output**. The
  update pipeline is complete end-to-end; the render side is a separate future step.
- A deterministic pre-gate on Tier 2 updates (analogous to `_has_implicit_signal()`
  for episodic extraction) has not been decided. See §9.5 Open Item 1.
- Automatic promotion of working state to episodic memory — not started.

### 9.2 Design Decisions

**Literal/interpretive split with Slot 6 (Working Memory).** Slot 6 carries
raw conversational turns: what was said, by whom, in what order. Slot 6A carries
structured state about what is happening and what has been decided — a layer of
interpretation above the raw transcript. The split respects the Predictable
constraint: Slot 6 is deterministic; Slot 6A's Tier 1 fields are deterministic;
only Slot 6A's Tier 2 fields involve inference, and that inference runs in a
post-response hook, not in the routing or prompt-assembly path.

**Deterministic Tier 1 first, model-assisted Tier 2 second.** The system
establishes a working rendering slot (`[WORKING STATE]`) before Tier 2 is
wired in. This allows the slot to carry real value in early deployment (active
RAG sources are immediately visible in the assembled prompt) while Tier 2 is
validated in parallel. It also avoids the risk of a Tier 2 extraction failure
silently leaving the rendered slot empty — Tier 1 renders unconditionally when
either of its fields is non-empty.

**P3c render-gating (no `[WORKING STATE]` on graph-query turns).** Graph-query
turns (P3c route) are a deliberate no-interpretation zone: structural,
non-personalized, deterministic relative to graph state. Injecting `[WORKING
STATE]` on the same turn would introduce interpretive, session-specific content
into what should be a clean, repeatable graph answer — blurring the boundary
by accident rather than by deliberate future design. The gate is a single
`if plan.graph_query is None:` check in `controller_agent.py` Step 5d.

**Update-vs-render distinction.** Tier 2 update (`process_working_state_update()`)
runs in the post-response hook regardless of `plan.graph_query`. A graph-query
turn produces real conversational state — the user issued an instruction and the
assistant responded — and that state is worth persisting so the next non-P3c
turn can render it. Suppressing the update on P3c turns would discard valid
session state to maintain a render boundary that applies only to a single slot's
output.

**SUMMARY field removed, not deferred.** The Slot 6A Tier 2 prompt was
originally designed with four output fields: `FOCUS`, `OPEN_LOOPS`, `DECISIONS`,
`SUMMARY`. Live diagnostic testing at `temperature=0.0` showed a categorical
failure with the 4-field format: on every sample (0/3), the model emitted bare
EOS with zero output content. The 3-field format (removing only `SUMMARY`) succeeded
on every sample (3/3), all else held constant. The SUMMARY field was therefore
permanently removed from the system prompt, the extraction function, the
`WorkingStateRecord` dataclass, the `upsert()` signature, the database schema
(v4→v5 migration), and all test fixtures. This was a removal decision, not a
deferral — the field will not be re-added unless the underlying failure mode is
understood and a workaround exists. The working theory for the failure (see §4.7)
is that "SUMMARY" carries document-closing pretraining semantics that dominate EOS
probability at position 1; this is an unverified hypothesis consistent with the
evidence, not a confirmed mechanism.

**SQLite DROP COLUMN safety check.** The v4→v5 migration drops
`turn_summaries_json` using `ALTER TABLE working_state DROP COLUMN`. This
requires SQLite ≥ 3.35.0; the project uses 3.50.4, so the operation is safe.
A PRAGMA guard (`PRAGMA table_info(working_state)` before executing the DROP)
makes the migration idempotent: fresh databases opened via the v2→v3 or v3→v4
path have `working_state` pre-created by `_init_db()` without `turn_summaries_json`
(since `_init_db()` already reflects the v5 schema), so the column is absent
when the v4→v5 block runs. The conditional prevents a spurious
`sqlite3.OperationalError: no such column` on those paths.

### 9.3 Schema

`_SCHEMA_VERSION` is now **5** (up from 3 at the time §8 was written; v3→v4
created `working_state`, v4→v5 dropped `turn_summaries_json`).

**`working_state` table (current, v5):**
```sql
CREATE TABLE IF NOT EXISTS working_state (
    mem_key                TEXT    PRIMARY KEY,
    current_focus          TEXT,
    open_loops_json        TEXT    NOT NULL DEFAULT '[]',
    recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
    updated_at             REAL    NOT NULL
);
```

**`WorkingStateRecord` (Python dataclass, `memory_manager.py`):**
```python
@dataclass
class WorkingStateRecord:
    mem_key:          str
    current_focus:    str | None
    open_loops:       list[str]
    recent_decisions: list[str]
    updated_at:       float
```

**v3→v4 migration (added `working_state` with `turn_summaries_json`):**
```python
if from_version < 4:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_state (
            mem_key                TEXT    PRIMARY KEY,
            current_focus          TEXT,
            open_loops_json        TEXT    NOT NULL DEFAULT '[]',
            recent_decisions_json  TEXT    NOT NULL DEFAULT '[]',
            turn_summaries_json    TEXT    NOT NULL DEFAULT '[]',
            updated_at             REAL    NOT NULL
        );
    """)
```

**v4→v5 migration (dropped `turn_summaries_json`, conditional via PRAGMA):**
```python
if from_version < 5:
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(working_state)"
    ).fetchall()}
    if "turn_summaries_json" in cols:
        conn.executescript(
            "ALTER TABLE working_state DROP COLUMN turn_summaries_json;"
        )
```

`_init_db()` pre-creates `working_state` in the v5 form (without
`turn_summaries_json`) for fresh databases, so the PRAGMA guard above handles
both upgrade paths (v4→v5: column exists, DROP executes) and fresh/v2/v3 paths
(column never existed, DROP skipped).

### 9.4 Test Suite

Final state as of 2026-06-21: **318 tests, 0 failures** across all test files.

Per-file delta confirmed for the SUMMARY-removal session (the last session in
the Slot 6A build arc):

| File | Before | After | Delta |
|---|---|---|---|
| `test_episodic_phase5.py` | 61 | 62 | +1 (`test_exactly_three_labels_required`) |
| `test_memory_phase1.py` | unchanged count | unchanged count | tests updated for v5 (version assertion, column checks, `upsert()` signatures) |
| `test_tool_dispatcher_phase6.py` | pre-existing test broken | fixed | +0 net (fix, not new test) |

The total rose from approximately 314 (pre-SUMMARY-removal, with the tool ceiling
test already broken) to 318 after the session. Per-file breakdowns across the
full Slot 6A build arc (prior to the SUMMARY-removal session) are not
reconstructed here — the total of 318 passing, 0 failures is the authoritative
figure.

### 9.5 Open Items

**Open Item 1 — Tier 2 pre-gate (unresolved, flagged for live observation).**
The episodic extraction pipeline has a deterministic pre-gate
(`_has_implicit_signal()`) that avoids an inference call on turns with no
plausible implicit signal. Tier 2's `process_working_state_update()` has no
equivalent gate — it calls the model on every completed non-episodic turn,
including P6 (direct answer) turns. Whether a similar pre-gate is warranted
depends on how often Tier 2 updates produce useful state changes vs. `NONE`
returns on short, low-context turns. No decision has been made; flagged for
live observation rather than speculative pre-optimization.

**Live timing data (2026-06-28, n=2).** Two live conversational turns were timed via `TIMING` instrumentation added to `_execute_plan()` (see §7.7). Both satisfied the post-dispatch gate. Elapsed time from `working_state_start` to `working_state_end`:

| Turn | Elapsed | `process_working_state_update()` outcome |
|---|---|---|
| 1 | 23.134 s | CHANGED |
| 2 | 18.835 s | CHANGED |

Both produced real state changes. n=2 argues against the pre-gate trigger condition having been observed yet — neither call was a `NONE` return that a pre-gate would have prevented. The item remains open; these two data points represent an upper-end sample of the cost distribution (calls that produce changes), not a steady-state average. An `on_status("Updating working memory…")` SSE event was added as an immediate visibility mitigation (see §7.7) so the user is no longer presented with a silent 20+ second wait while this call runs.

*Update 2026-06-29.* The `on_answer_ready` early-completion fix (§7.7) removes the user-visible consequence of this latency — `'done'` now fires before the hooks run, so the input re-enables immediately and the 18–23s cost is no longer presented to the user as a wait. Open Item 1's pre-gate question (whether `process_working_state_update()` should be gated on a signal that its output will produce a meaningful state change, to avoid inference cost on low-value turns) remains genuinely open and unaddressed by this fix — the hooks still run on every qualifying turn at the same cost, just off the user-visible response path.

**Open Item 2 — Tier 2 render wiring not yet done.** `current_focus`,
`open_loops`, and `recent_decisions` are stored and updated per turn but
never appear in the rendered `[WORKING STATE]` block. Wiring them in requires
reading the stored `WorkingStateRecord` in `controller_agent.py` Step 5d and
merging those fields into the `WorkingMemoryState` passed to
`PromptBuilder.build()` — or a separate dataclass path for Tier 2 fields.
Not started; the schema and extraction pipeline are ready when this step is
tackled.

**Open Item 3 — SUMMARY/EOS mechanism hypothesis unverified.** The working
theory that "SUMMARY" triggers early EOS via document-closing pretraining
semantics is consistent with the diagnostic evidence but the mechanism is not
confirmed. Anyone considering a structured-output prompt with similar field
labels elsewhere in this codebase should treat §4.7's documented finding as
a risk, not a certain prediction, and run their own diagnostic at the target
temperature and prompt shape before committing.

**Open Item 4 — `extract_working_state_update()` PARSE_FAILURE root-caused:
reasoning-token exhaustion against `max_tokens=200`. CONFIRMED by live
diagnostic, FIXED 2026-06-23 (same day, bundled with §3.7c shared-prefix
fix below). See full status note at end of this entry.**

*Originally:* logged via newly-added `WSU_DIAG` read-only diagnostic
instrumentation (no gating logic, no behavior change) added to
`process_working_state_update()` in a prior session, in response to a single
live incident where a Tier 2 call returned an unparseable bare `'\n'`.

*Live session evidence (2026-06-23):* a real session of 4 conversational
turns — a simple knowledge question, a no-tool lookup, and a tool-call turn —
produced **4/4 `PARSE_FAILURE` outcomes**, zero `CHANGED`, zero
`UNCHANGED_NONE`, zero `INFER_FAILURE`. Every failure returned the identical
bare `'\n'` raw response. Every call's `infer_elapsed_s` fell in the same
12–14s band (11.958–13.972s) regardless of turn content or prompt length
(`prompt_chars` ranged 334–1905 across the four calls). This moved the
finding from "single historical incident, mechanism unknown" (§8.8 Open Item
11 — see cross-reference below) to deterministically reproducible under this
specific call's exact conditions.

*Isolation diagnostic (read-only, same day, direct-to-oMLX, bypassing the
FastAPI app entirely):* a standalone script sent the exact
`_WORKING_STATE_UPDATE_SYSTEM` system message and a reconstructed user prompt
matching the real call shape directly to `http://localhost:8000/v1/chat/completions`,
mirroring `OMLXRuntimeClient.infer_stream()`'s real request/response contract
exactly (`stream=True`, OpenAI-compatible SSE envelope, `choices[0].delta.content`)
after an initial mis-shaped non-streaming test attempt failed with `KeyError`
and was corrected by reading `omlx_runtime_client.py`'s real `infer_stream()`
and `foundry_runtime_client.py`'s real `_iter_sse_chunks()` source directly
rather than assuming the response shape.

Three cases run, `temperature=0.0` throughout:
1. **All-NONE previous state, `max_tokens=200`** (reproduction attempt): bare
   `'\n'` returned, `finish_reason: "length"`. Reproduced the live failure
   exactly.
2. **Realistic non-empty previous state** (`current_focus`, `open_loops`,
   `recent_decisions` all populated with real-looking values), `max_tokens=200`:
   bare `'\n'` returned, `finish_reason: "length"`. Identical failure despite
   non-empty state — **this rules out the all-NONE-previous-state-confuses-the-model
   hypothesis directly**, rather than merely deprioritizing it.
3. **Same all-NONE prompt as Case 1, `max_tokens=2000`**: succeeded.
   `finish_reason: "stop"`. Returned a clean, fully parseable three-line
   response: `FOCUS: Explaining the Localist Framework / OPEN_LOOPS: NONE /
   DECISIONS: NONE`.

*Mechanism, visible directly in the raw SSE stream (not inferred):* the model
is emitting a distinct `reasoning_content` delta stream — a multi-step
internal "Thinking Process" narrating its own analysis of the previous state,
the new turn, and each of FOCUS/OPEN_LOOPS/DECISIONS in turn — entirely
separate from the `content` delta stream the parser reads. In both 200-token
failures, the reasoning trace was still mid-step (step 3–5 of its own
6-step internal outline) when the token ceiling was hit; the only token(s)
that reached `content` before the forced cutoff was the stray `'\n'` the
parser then correctly rejected. In the 2000-token success case, the
*identical* reasoning trace ran to its own natural conclusion (its own final
step: *"Format Output: Apply the strict three-line format"*) and only then
did the delta stream switch from `reasoning_content` to `content`, emitting
the three labeled lines cleanly. The reasoning trace alone consumed roughly
500–600 tokens before any `content` token appeared in the successful case —
this is the real floor this call needs for reasoning, before the three-line
answer's own token cost is added on top.

*This explicitly rules out, by direct evidence rather than deprioritization:*
- The SUMMARY/EOS mechanism from Open Item 3 — no "SUMMARY" token appears
  anywhere in `_WORKING_STATE_UPDATE_SYSTEM`, and the failure is a hard
  `finish_reason: "length"` truncation, not an EOS emission.
- Previous-state emptiness as a causal factor (Case 2, above).
- Turn-content dependence — the live session's 4/4 spanned three
  meaningfully different turn shapes with identical failure behavior.

*Not yet established:*
- The true minimum viable `max_tokens` for this call. 2000 was chosen as
  generous headroom for the isolation test, not as a tuned value — it is
  very likely far larger than necessary. No bisection or systematic search
  has been run.
- Whether reasoning-trace length is roughly constant across turn content, or
  varies meaningfully with longer/more complex turns (e.g. a turn involving
  a tool-call result, which the live session's 4th turn was, but which has
  not yet been isolation-tested directly — only reproduced live, not
  isolation-tested with non-trivial response content).
- Whether this same reasoning-token-exhaustion mechanism is present (and
  currently silently absorbed by larger headroom) in any of this codebase's
  other bounded extraction calls — `extract_content_from_instruction()`
  (`max_tokens=60`) and `extract_implicit_episode()` have not been checked
  against this same diagnostic approach. Given `extract_content_from_instruction()`'s
  budget is smaller than the WSU call's, it is at least as exposed in
  principle; not yet verified empirically either way.
- Whether this is a property of the `gemma-4-e4b-it-4bit` model file itself,
  an oMLX 0.4.2 serving configuration/default, or something that changed
  between sessions — no comparison against a non-reasoning call configuration
  has been attempted, and no version/config history has been checked.

*Cross-reference:* §8.8 Open Item 11 (fabricated tool-call syntax, mechanism
unknown) is a structurally different failure on the *main conversational*
call (`max_tokens=1024`, not this call's `max_tokens=200`) and should **not**
be conflated with this finding — they were produced by different calls with
different parameters, and Open Item 11 remains unreproduced and
unexplained on its own terms. The two are noted together only because both
involve this model/serving setup producing output around its own internal
process (an invented tool-call string; an exposed reasoning trace) in a way
this codebase's call sites were not written expecting. Whether they share a
deeper common cause (e.g. reasoning-capable generation behavior this harness
doesn't yet account for anywhere) is an open question, not a finding.

*Status:* root cause confirmed by direct live diagnostic, **fix implemented and
verified, 2026-06-23 (same day).** `max_tokens` raised from 200 to 1024 on this
call — chosen to match the main conversational call's existing budget, not a
newly invented value, and confirmed comfortably above the ~570–600-token
reasoning floor measured above. Implemented as part of the same Claude Code
prompt as the system-message-sharing fix below (Open Item 4 and the §3.7c
shared-prefix fix were deliberately bundled — both touch
`extract_working_state_update()`'s call site). 73/73 tests passing
post-implementation (66 pre-fix → 72 from the sharing/budget change → 73
after one additional realistic-length verification test was requested and
added). No live re-verification of the *fix itself* (i.e. running real turns
and confirming `WSU_DIAG` no longer logs `PARSE_FAILURE` at the new budget)
has been done yet — the isolation diagnostic proved the mechanism and that
2000 tokens resolves it; 1024 tokens has not been independently
isolation-tested, only reasoned to be sufficient by analogy to the
measured ~570–600 token floor plus the main call's own budget convention.
This is a real, if probably minor, gap — flagged rather than assumed away.

**Update 2026-07-05 — value history: `1024` → `300` (2026-07-05, over-aggressive
first pass) → `750` (2026-07-05, current).** During the same-day runtime-concurrency
investigation (§7.7 Update 2026-07-05; full account in `sessions-log.md` §16),
`extract_working_state_update()`'s `max_tokens` was dropped from 1024 to 300 as
a first-pass reduction, then flagged before shipping: 300 sits below the
~570–600-token reasoning-trace floor measured above and would reproduce the
same `PARSE_FAILURE` this Open Item originally fixed. Corrected same-day to
**750** — comfortable margin above the measured floor, deliberately below the
prior 1024 (which was never a tuned value, only a reused convention borrowed
from the main conversational call's budget — see *Status* above). The current
value sent to `runtime.infer()` is 750, confirmed by an updated assertion in
`tests/test_episodic_phase5.py` (`test_max_tokens_is_1024` — name unchanged,
assertion updated to `== 750`). Full suite held at baseline throughout this
change (566/2 pre-existing → 572/2 pre-existing after the concurrency work in
§7.7; the 2 failures are a pre-existing flaky pair asserting on live
`web_search`/LangSearch content, unrelated to this call).

The items below, all logged at diagnosis time, remain genuinely open even
after the 2026-06-23 fix and were not addressed by either it or the
2026-07-05 value change:
- The true minimum viable `max_tokens` — 750 (current) is a chosen safety
  margin above the measured ~570–600-token floor, not a tuned or bisected
  value. Only 200, 300, 750, and 2000 have ever been tried; no systematic
  search has been run.
- Whether reasoning-trace length varies meaningfully across turn content
  (e.g. tool-result turns) — still untested.
- Whether `extract_content_from_instruction()` (`max_tokens=60`) or
  `extract_implicit_episode()` share this exposure — still unchecked.
- Whether this is a model-file property, an oMLX serving default, or
  something that changed recently — still unestablished.

**Update 2026-07-06 — same mechanism confirmed and fixed at two sibling call
sites.** `extract_implicit_episode()` and `extract_content_from_instruction()`
(both `episodic_extractor.py`) were live-diagnosed and found to exhibit the
identical reasoning-token-exhaustion signature this Open Item root-caused for
`extract_working_state_update()` — `finish_reason: "length"` after a
~730–870-char hidden `reasoning_content` trace consumed the entire
`max_tokens=200` budget before any content token was emitted, reproduced on
the input `"LORA is the assistant persona. Localist is the project name."`.
This directly resolves the "still unchecked" bullet immediately above. Both
call sites' `max_tokens` raised 200 → 750 (`episodic_extractor.py:380` and
`:499`), matching this Open Item's fix exactly in kind and magnitude. Full
suite held at baseline (572/2 pre-existing, same 2 unrelated failures, before
and after); no test asserted on the literal `200` value. Live-verified
against the real production model: 10/10 runs (5 at `temperature=0.10`, 5 at
`temperature=0.0`) returned `finish_reason: "stop"` with non-empty parsed
output; `temperature=0.0` was fully deterministic/byte-identical across all 5
runs. Full diagnostic and fix detail: see `sessions-log.md` §18 (2026-07-06).
This is an additive fix at separate call sites, not a supersession of this
Open Item's own 2026-06-23/2026-07-05 fix for
`extract_working_state_update()`, which remains correct and complete for its
own call site.

**Open Item 5 — model claims a durable memory write occurred independent of
whether extraction actually succeeded (new, 2026-07-06; not resolved by the
`max_tokens` fix above).** In the incident that prompted the 2026-07-06
diagnostic, the main conversational response to "LORA is the assistant
persona. Localist is the project name." stated "I have updated my context" —
durable-write language — despite `extract_implicit_episode()` having silently
failed (bare/near-bare output, no episode written) on that exact turn. The
`max_tokens` fix above eliminates the specific failure that exposed this gap
on this occasion, but nothing in the prompt contract ties the model's
language about memory persistence to whether the post-response extraction
hook actually ran and succeeded — the two are architecturally decoupled (the
extraction call happens after the response has already been generated and
streamed to the user). If a future extraction call fails for any other reason
(inference error, a different reasoning-exhaustion case, a NONE
misclassification), the model would be expected to make the same false claim
again. Not yet scoped: no design proposed for closing this gap (e.g.
deferring/softening durable-write language until confirmed, or decoupling the
response from any implied write guarantee).

**Open Item 6 — Memory Tab: delete individual episode entries (not yet
designed, 2026-07-06).** No UI mechanism currently exists to remove a stored
episodic memory entry — e.g. a stale or duplicated fact accumulated from
repeated testing (see `sessions-log.md` §19 for the live trace that surfaced
this: "The user's name is Michael." appearing twice in `[EPISODIC MEMORY]`
retrieval, both at `confidence=1.0`). Scoped as a future feature only; no
design proposed yet.

