# Explicit-memory write gate — semantic gating rejected, deterministic pattern adopted (2026-07-23)

## Trigger

Live bug, second one found the same day as `episodic_relevance_semantic_gate_2026-07-23.md`:
"I want you to remember I'm participating in a Claude Impact Lab on August 6th."
never wrote a durable episode. Root cause: two independent gates both require
the literal phrase `"remember that"`, and this instruction says "remember
**I'm**" (no "that"):

- `planner.py` Priority 2 (`_priority2_memory()` / `_MEMORY_KEYWORDS`) decides
  `write_episode=True` — missed, so `controller_agent.py` never even called
  `process_explicit_signal()`.
- `episodic_extractor.py`'s `_EXPLICIT_SIGNALS` / `detect_explicit_signal()`
  (called *inside* `process_explicit_signal()`, independently of Planner) —
  would also have missed it even if the first gate had fired, since it does
  its own separate `"remember that"`-only scan.

Both had to be fixed for the write to actually happen end-to-end.

## Attempt 1 — semantic gating (mirroring the Priority 5 fix)

Tried first since it worked well for the same-day Priority 5 retrieval fix.
Templates and method: `diagnostics/score_episodic_relevance_templates.py`'s
sibling methodology, run interactively against the live tuned
`mlx-community/embeddinggemma-300m-4bit` model. **Result: no viable
threshold found in two rounds — rejected on measured evidence.**

**Round 1** — bare "remember"/"keep in mind" imperative templates:

```
remember that I am
I want you to remember this about me
please remember that I
keep in mind that I
make a note that I
do not forget that I
for future reference I
```

Min true positive: 0.5919 ("For future reference, I use Ollama not oMLX.").
Max negative: **0.7401** ("Do you remember my name?") — a recall *question*
scored higher than a genuine write command. Negative-max > positive-min:
no threshold separates them.

**Round 2** — re-anchored away from bare "remember" (the presumed collision
source) toward "make a note"/"keep in mind"/"log"/"jot down" phrasing:

```
make a note that I am
keep in mind that I am
jot this down about me
log that I am
for future reference I am
going forward remember that I
write this down about me
```

Min true positive: 0.5901. Max negative (non-"do/can-you-remember" family):
**0.6842** ("I remember when we talked about this before.") — still
inverted. Even isolating the "do/can you remember" family as a candidate
deterministic pre-filter (matching the codebase's existing
`_RESEARCH_NEGATIVE_FILTER`/`_SEARCH_NEGATIVE_FILTER` pattern) didn't rescue
this round — "I remember when..." (ordinary reminiscing, not a recall
question) still collided on pure embedding similarity alone.

**Conclusion:** unlike Priority 5's problem (a personal-event request vs. a
generic task — semantically distant classes), this is write-intent vs.
recall-intent about the *same* personal-fact vocabulary and pronouns
("remember", "I", first-person). Short-phrase cosine similarity does not
reliably encode imperative-vs-interrogative mood or write-vs-read speech act
here. Not a threshold-tuning problem — a technique-fit problem. Recorded so
a future attempt doesn't re-walk the same two rounds expecting a different
result.

## Attempt 2 — deterministic pattern rule (adopted)

Bare word `"remember"`, excluded when:
1. preceded by an interrogative — `do|does|did|can|could|would|will|what do
   (you)? remember` (recall questions: "do you remember...", "what do you
   remember...", "will you remember to feed the cat?"),
2. the instruction ends in `"?"` (any remaining question shape),
3. the phrase `"i remember"` appears (user reminiscing/stating a fact about
   their own memory, not directing the assistant — "I remember when we
   talked about this before").

Implemented as `planner._has_explicit_remember_signal()` — a single shared
function, imported into `episodic_extractor.py` so both gates use the exact
same rule (no drift risk between the two independently-triggered checks).

Also added two literal, zero-collision-risk phrases directly to both
`planner._MEMORY_KEYWORDS` and `episodic_extractor._EXPLICIT_SIGNALS`:
`"keep in mind"`, `"make a note"` (neither collides with
`_RETRACTION_SIGNALS` or any existing entry).

### Result — same battery, both attempts compared

| | Attempt 1 (semantic, best round) | Attempt 2 (pattern) |
|---|---|---|
| True positives caught | inverted, no threshold works | 6/6 |
| True negatives excluded | inverted, no threshold works | 12/12 |

Full pass/fail detail (positives and negatives) captured interactively
during implementation; not scripted into a standalone diagnostic file since
the rule has no tunable threshold to re-verify later — it's exercised
directly by the unit tests added alongside this fix
(`tests/test_planner_phase3.py::TestExplicitRememberPattern`,
`tests/test_episodic_extractor*` for the `detect_explicit_signal()` side).

## Follow-up — "don't forget" retraction collision — CLOSED same day (2026-07-23)

Originally found while scoping the fix above and deliberately left
unaddressed (this section used to describe it as out of scope): `episodic_
extractor._RETRACTION_SIGNALS` contains the substring `"forget that"`,
checked *before* `_EXPLICIT_SIGNALS` in `detect_explicit_signal()`. "Don't
forget that I have a dentist appointment" — a request to please *remember*
something — substring-matched `"forget that"` and routed to **retraction**
(delete), the opposite of what the user means. A real, higher-stakes bug
(silent data deletion, not just a missed write).

Closed per explicit follow-up request the same day. Fix: `planner.
_MEMORY_NEGATED_FORGET` (`\b(don'?t|do not|never)\s+forget\b`) is checked in
`detect_explicit_signal()` *before* the `_RETRACTION_SIGNALS` loop runs at
all, short-circuiting a negated-forget instruction straight to an
insert-type `ExtractionSignal` (`episode_type="preference"`, matching
"remember that"'s existing mapping). Folded into
`_has_explicit_remember_signal()` (now covers both the bare-"remember" and
negated-"forget" cases) rather than the original scoped-out plan of a
separate function, since both funnel into the same Planner P2
`write_episode` decision and the same shared-function-avoids-drift rationale
applies equally.

Deliberately narrow: only `"forget"` is handled (matching what was actually
reported/found). `_RETRACTION_SIGNALS`' other phrases — `"ignore that"`,
`"disregard that"`, `"scratch that"` — were not reported as colliding and
are far less naturally used with a `"don't"` negation in this sense
("don't ignore that" reads as emphatic-affirm-the-opposite, not "remember
this", the way "don't forget" unambiguously does); left untouched rather
than guessing at fixes nobody asked for.

Verified: genuine (unnegated) `"forget that"` still retracts correctly
(regression-tested); `"don't forget that X"` and `"don't forget X"` (no
"that") both now insert instead of deleting; `"never forget that X"` /
`"do not forget X"` variants also covered.
