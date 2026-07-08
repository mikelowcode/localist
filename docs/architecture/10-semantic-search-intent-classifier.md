## 10. Semantic Search-Intent Classifier

### 10.1 Scope

Before this change, `_priority3_tool()` in `planner.py` triggered `web_search` only when the
instruction contained a literal keyword from `_WEB_SEARCH_KEYWORDS` — a small frozenset of
recency and freshness terms ("latest", "today", "news", "current price", and similar). This
produced a confirmed false negative in a real session.

**The incident:** the user asked "What do you know about APC (Auto Prefix Cache)?" The corpus
and episodic stores lacked APC-specific content; the model correctly stated it lacked specific
information. The user then issued the follow-up: "Go ahead and look it up." No keyword from
`_WEB_SEARCH_KEYWORDS` appeared in that instruction. `_priority3_tool()` returned `None`;
routing fell to Priority 6 (direct answer); and the model responded with fabricated claims about
APC — falsely presented as coming from "Web Search" despite no search tool ever being invoked.

A second, distinct gap was found in the same incident: the full original APC instruction that
preceded the follow-up — "Why don't you do a web search for APC and then tell me if you still
stand by your previous answer?" — contains the literal phrase "web search". That instruction
failed because of a keyword-coverage miss, not for any semantic reason.

This section documents the resulting two-part fix: an embedding-based semantic classifier
layered onto Priority 3 to catch natural-language search-action phrasings, and a literal-keyword
addition to `_WEB_SEARCH_KEYWORDS` to close the separate coverage gap. Each fix targets a
distinct failure mode; they were applied at different layers for that reason.

### 10.2 Design Decisions

**Two distinct failure shapes; two distinct fixes.** "Go ahead and look it up." is a
bare-affirmative follow-up with no topic keyword; no literal phrase reliably covers the space of
such instructions, which require semantic generalization. "Why don't you do a web search for
APC..." contains the exact literal phrase "web search" and failed only because that phrase was
absent from `_WEB_SEARCH_KEYWORDS`. Fixing the first gap by extending the keyword list would
have required open-ended enumeration with no principled stopping point. Fixing the second gap by
tuning the semantic threshold down to capture that one sentence (actual score: 0.638 vs. the 0.68
threshold; margin of 0.023 above the closest false positive) would have changed gate behavior for
all future utterances without any broader justification. Conflating these two failure shapes would
have produced a worse fix in either direction.

**Per-group cosine similarity against four canonical template groups, using the EmbeddingGemma
model already resident in the process.** `_SEARCH_INTENT_TEMPLATES` defines four named groups —
`explicit_search_action`, `lookup_request`, `knowledge_request_open`, `freshness_request` — with
21 template strings total (5+9+4+3; `lookup_request` expanded from 5 to 9 on 2026-06-25, then its 4 added templates replaced by Candidate Set 1 on 2026-06-28 — see §10.4 Open Item 3 updates). At startup, `Planner.__init__()` embeds all 21 using the
same EmbeddingGemma model (`mlx-community/embeddinggemma-300m-4bit`, 768-dimensional) that
`EmbeddingEngine` already uses for corpus retrieval. The `embed_fn` callable is threaded into
`Planner` as a new optional constructor parameter (`embed_fn: Callable[[str], list[float]] | None
= None`), passed from the `main.py` lifespan function through `ControllerAgent`. `MemoryManager`
already holds this callable for corpus scoring; `Planner` receives its own copy of the same
already-initialized function rather than reaching into `MemoryManager._embed_fn` as a shortcut.
A reach-through pattern exists elsewhere in the codebase (`controller_agent.py`'s `/embed`
endpoint helper); it was noted and deliberately not replicated here.

**Gating uses per-group scores from `all_scores`, not the global argmax.** `_semantic_search_intent()`
returns a 3-tuple `(best_group, best_score, all_scores)` where `all_scores` is a dict mapping
each group name to its own maximum cosine similarity across that group's templates. The gate in
`_priority3_tool()` evaluates each gating group against its own score independently:

```python
semantic_triggered = any(
    all_scores.get(group, 0.0) >= threshold
    for group, threshold in _SEMANTIC_GATE_THRESHOLDS.items()
)
```

This is load-bearing. If gating were evaluated on `best_group` only, a non-gating group winning
the argmax would suppress the gate even when a gating group independently cleared its own
threshold. In the live diagnostic evaluation, `knowledge_request_open` won the argmax in 3 of 7
adversarial negative test cases — demonstrating how frequently this scenario arises in practice.

**Only two of the four groups gate routing; the other two are informational only.** `_SEMANTIC_GATE_THRESHOLDS`
contains exactly two entries: `explicit_search_action` (≥ 0.72; raised from 0.68 on 2026-06-28 —
**UNDER OBSERVATION**, not finalized; see §10.4 Open Item 3 Update 2026-06-28) and `lookup_request`
(≥ 0.60; original value was 0.65, lowered 2026-06-25, templates partially revised 2026-06-28 —
see §10.4 Open Item 3 updates).
`knowledge_request_open` and `freshness_request` are computed and logged on every turn but are
excluded from gating. The evidence for `knowledge_request_open`: a live diagnostic pass found
that "Explain this code to me." scored 0.795 on that group — higher than 5 of the 10 real
positive search-intent paraphrases tested in the same pass — because that group's canonical
templates ("tell me about this", "what do you know about this", "what is this", "explain this to
me") are generically conversational phrasings that collide with ordinary non-search chat. For
`freshness_request`: one adversarial negative scored inside the positive range during evaluation,
and the group has not been independently stress-tested at a larger sample size. Both groups remain
in the computation pipeline and are emitted to the debug log; neither may gate `tools_to_call`
without a separate evaluation pass. See §10.4, Open Item 1.

**Thresholds were derived from live-backend diagnostics, not tuned to fit any incident utterance.**
The values 0.68 (`explicit_search_action`) and 0.65 (`lookup_request`, original) were determined from a
structured evaluation pass run before the gating logic was written: 10 positive search-intent
paraphrases, 7 adversarial negatives, and 1 negative-filter case (18 utterances total) submitted
against the live EmbeddingGemma model. The second incident instruction ("Why don't you do a web
search for APC...") was not used to tune these numbers — it scored 0.638, fell below the 0.68
threshold, and was fixed at the literal-keyword layer specifically to avoid post-hoc threshold
adjustment for one known utterance. `lookup_request` was subsequently lowered from 0.65 to 0.60
on 2026-06-25 after confirmed live false negatives (see §10.4 Open Item 3 update).
`explicit_search_action` was subsequently raised from 0.68 to 0.72 on 2026-06-28 after two
adversarial negatives scored ESA 0.69–0.70 via the single bare-verb template "go look it up"
colliding with "look at"/"look into" phrasing; the 0.72 value is **under observation**, not
finalized (see §10.4 Open Item 3 Update 2026-06-28).

**A negative filter short-circuits before the embedding call.** `_SEARCH_NEGATIVE_FILTER` is a
frozenset of 18 phrases (9 original + 5 added 2026-06-26 + 4 added 2026-06-27) identifying
meta-instructions that reference the conversation itself or the search tool — "did you search",
"what tool did you use", "search my previous messages", and similar — rather than requesting a
world-facing search, plus five identity/capability phrases ("who are you", "what are you",
"what can you do", "what can you help with", "what do you do") added after a confirmed
false-positive collision with the four 2026-06-25 `lookup_request` templates (see §10.4 Open
Item 3 update 2026-06-26), plus four greeting-form phrases ("hey lora", "hi there", "hey there",
"what's up") added after "Hey LORA!" scored 0.612 on `lookup_request` in live use (see §10.4
Open Item 3 update 2026-06-27). Bare "hi" and "hey" were assessed and deliberately excluded:
under the filter's `phrase in lowered` substring mechanism, "hi" collides with common words
("history", "this", "high", "vehicle", etc.) and "hey" collides with "they". When any of these
phrases appears in the lowered instruction, `_semantic_search_intent()` returns `None` immediately
without invoking `embed_fn`. Verified live: "Did you search for that already?" triggered the
filter and produced no embedding call.

**"web search" and "do a search" added to `_WEB_SEARCH_KEYWORDS`.** The second incident gap was
closed by adding these two phrases to the existing `_WEB_SEARCH_KEYWORDS` frozenset, matched via
the existing `_any_whole_word()` boundary function with no new matching logic. "search for" was
considered and deliberately excluded: it matches sentences like "search for a workaround in my
own code" with no search-tool intent, and adding it would reintroduce the over-broad literal-match
problem this arc was correcting.

**Priority 3's semantic embedding call and Priority 4's corpus-retrieval embedding call are not
shared.** Each computes a separate `embed_fn` invocation on the same query string. Sharing the
computed query vector across both call sites was considered — on any turn where P3 semantic
evaluation and P4 corpus retrieval both run, the vector is identical — and was explicitly
deferred, not rejected, pending latency profiling on the 16 GB development machine. The
EmbeddingGemma model is resident in memory; per-call overhead has not been measured at
production-style turn rates. Revisit only if profiling shows the duplicate call contributes
meaningfully to observed latency. This is the same accept-now/optimize-later posture applied to
other unmeasured-cost decisions in this codebase.

### 10.3 Live Verification — Original Incident, Recreated

The following is a live, two-turn recreation of the original incident, run against the deployed
backend after all four slots of the fix arc were applied.

**Turn 1 — "What do you know about APC (Auto Prefix Cache)?"** The semantic gate did not fire:
`knowledge_request_open` scored 0.624; neither `explicit_search_action` nor `lookup_request`
cleared its threshold. Corpus and episodic retrieval both missed on APC specifically. The model
correctly stated it lacked specific information rather than fabricating claims. **This turn's
correct, non-hallucinating response is not attributable to this fix.** The gate did not fire;
credit for the model's behavior here lies with prompt wording (e.g. the behavioral constraint
"you do not simulate certainty"), not with the classifier.

**Turn 2 — "Go ahead and look up APC (Auto Prefix Cache)."** This is a live, unscripted
paraphrase — not identical to any string in the diagnostic dataset and not identical to the
original incident's exact wording. `lookup_request` scored 0.740, clearing the 0.65 threshold
(the threshold was subsequently lowered to 0.60 on 2026-06-25; 0.740 clears both values).
`tools_to_call = ['web_search']`. LangSearch returned three real, correctly-disambiguated results
identifying APC as automatic prefix caching in LLM inference serving — including an arXiv paper
and a Chinese-language vLLM technical article independently confirming the same expansion. These
results directly contradicted the original incident's fabricated claims (networking/routing,
database indexing, compression/streaming), confirming those claims were not merely unsupported
but factually wrong.

This is the first live, non-scripted confirmation that the semantic fix generalizes beyond the
diagnostic dataset's curated test strings.

### 10.4 Open Items

**Open Item 1 — `freshness_request` gating status unresolved.** One adversarial negative
example scored inside the positive range during the Diagnostic 2 evaluation pass; the group has
not been stress-tested at a larger sample size. Remains informational-only — not a confirmed-safe
candidate for routing gating — until a separate evaluation pass establishes a defensible
threshold and negative margin.

**Open Item 2 — Embedding call sharing deferred.** Priority 3's semantic classification and
Priority 4's corpus-retrieval check each invoke `embed_fn` independently on the same query
string. Sharing a single computed embedding across both call sites was scoped and explicitly
deferred pending real latency data on the 16 GB development machine. Revisit only if profiling
shows the cost matters in practice; do not optimize without measurement.

**Open Item 3 — Threshold sample size.** 0.68 (`explicit_search_action`) and 0.65 (`lookup_request`,
original) are derived from a single diagnostic pass: 10 positive paraphrases and 7 adversarial
negatives. Treat as shippable-but-not-fully-validated. Revisit if live false positives (gate fires
when no search was intended) or false negatives (gate misses a clear search instruction) are
observed.

**Open Item 3 — Update 2026-06-25 (false negatives; `lookup_request` template expansion +
threshold lowering).** Three live "Can you look up [topic]?" utterances — "Can you look up
Apple's price hike for the MacBook Neo and iPad?", "Can you look up Microsoft's next-generation
in-house AI models?", "Can you look up their next-generation in-house Microsoft AI models?" —
each scored below the 0.65 gate (0.593, 0.598, 0.598) despite being unambiguous lookup
requests, because the original five `lookup_request` templates were all bare imperatives ("look
up this", "look that up", etc.) and did not cover the "Can/Could you + look up/look into +
[specific object]" question-form frame. Two fixes applied:

- **Template expansion (update A):** Four new templates added to `lookup_request` — "can you
  look up", "can you look that up for me", "could you look up", "can you look into this for me".
  Total expanded from 5 to 9. Post-addition, the same three utterances scored 0.608, 0.621, and
  0.617 respectively — real, consistent improvement (+0.015 to +0.023) but still below the 0.65
  gate. **These four templates were replaced by Candidate Set 1 on 2026-06-28** after live
  diagnostics confirmed they produced a threshold-unfixable false-positive surface (6/14
  adversarial negatives scoring 0.81–0.90 — above every true positive). See Update 2026-06-28
  below.
- **Threshold lowering (update B):** `lookup_request` threshold lowered from 0.65 to 0.60. The
  remaining 0.03–0.04 gap was consistent enough across all three utterances to satisfy the Open
  Item 3 "live false negatives observed" revisit criterion; template coverage alone could not close
  it. `explicit_search_action` (0.68) deliberately not changed. Known accepted risk: the original
  18-utterance diagnostic pass did not retain per-utterance scores for `lookup_request`'s
  adversarial negatives, so the margin to the new 0.60 line was unknown; any live false positive
  on `lookup_request` was named as the trigger to re-examine.

Six tests added to `TestPriority3SemanticGating` in `test_planner_phase3.py` covering the new
templates, unchanged original templates, updated threshold values, and boundary behavior at both
sides of the new 0.60 line.

**Open Item 3 — Update 2026-06-26 (false positives; `_SEARCH_NEGATIVE_FILTER` expansion).**
The accepted risk named in update B materialized. The named trigger was observed: after P4a
(`_priority4a_identity()`) was removed from the routing ladder (see §8.8 Open Item 12), live
verification showed "Who are you?" routing to priority=3 with `lookup_request=0.631 (≥ 0.60)`,
`web_search` dispatched — a wasted search call on a pure identity question. Two further
identity/capability utterances ("What can you do?", `lookup_request=0.666`) shared the same
false-positive pattern.

*Root-cause trace:*

- `_FACTUAL_QUERY_KEYWORDS` was ruled out as the mechanism: structurally unreachable on this
  routing path (P3b evaluates after P3, and `_FACTUAL_QUERY_KEYWORDS` phrases do not appear
  in these utterances lexically).
- Two diagnostic scripts (`diagnostics/score_lookup_request_templates.py`) isolated the
  mechanism to the four 2026-06-25-added question-form templates specifically: "can you look
  up", "can you look that up for me", "could you look up", "can you look into this for me".
  These share a modal-auxiliary question frame ("what/who + are/can/do + you") with the
  identity/capability utterances, producing syntactic (not semantic) similarity. The original
  five bare-imperative `lookup_request` templates never crossed 0.60 for any of the three
  tested utterances.
- Per-template breakdown confirmed: for "Who are you?", scores 0.630 / 0.604 (two new
  templates above gate), 0.588 / 0.588 (other two new templates), ≤ 0.522 (all five
  originals). For "What are you?", four new templates all above gate (0.603–0.660). For
  "What can you do?", four new templates above gate (0.652–0.672).

*Fix:* Five phrases added to `_SEARCH_NEGATIVE_FILTER` — "who are you", "what are you",
"what can you do", "what can you help with", "what do you do" — blocking the embedding call
entirely before it reaches the gating logic. Selected over template-rewording or per-template
thresholds as the narrowest reversible option: each phrase is a literal substring match,
independently removable, with no impact on the gate logic itself.

*Verification chain:*

1. **Unit tests (+7):** `TestIdentityCapabilityNegativeFilter` in `test_planner_phase3.py` —
   five tests confirming `_semantic_search_intent()` returns `None` for each phrase; one
   confirming `_priority3_tool("who are you?")` returns `None` end-to-end; one non-regression
   test confirming the 2026-06-25 incident utterances still fire the gate (lookup_request=0.62
   mocked, `web_search` in tools_to_call).
2. **Dedicated live-verification prompt:** Real `EmbeddingEngine`, real `Planner.__init__()`
   with real `_template_embeddings`. Group A (5 identity/capability phrases): all five filter
   fired, `_semantic_search_intent` returned `None`, `tools_to_call = []`. Group B (3 original
   incident utterances): filter not fired, scores 0.6077 / 0.6172 / 0.6208 (≥ 0.60, matching
   the §8.8 OI12 record to within rounding), `web_search` dispatched in all three cases.
3. **Unprompted real-traffic confirmation:** Two organic turns the same session independently
   triggered the filter correctly — not from a targeted test.

*What remains open:*

- The general negative-side margin of the 0.60 threshold remains unverified for the full
  adversarial set; this update patched five specific observed collisions reactively, not
  systematically. Any new live false positive on `lookup_request` remains the trigger to
  re-examine the threshold or the template set.
- Identity-adjacent siblings ("what's your name", "are you an AI", "what model are you")
  were raised and explicitly deferred; their scores have not been tested. Not blocked, but
  not covered.

**Open Item 3 — Update 2026-06-27 (greeting false positives; `_SEARCH_NEGATIVE_FILTER`
expansion).** "Hey LORA!" — a user greeting to open a session — produced `lookup_request=0.612`
(≥ 0.60), dispatching a spurious `web_search` call. This is the named trigger from update B:
a confirmed live false positive on `lookup_request`.

*Diagnostic arc (two passes, one script):* `diagnostics/score_greeting_collisions.py` was run
in two phases.

- **Breadth pass:** The original probe set (exact_repeat / isolation / common_greeting /
  known_anchor groups, 20 utterances) showed every short greeting clustering in a 0.60–0.65
  band on `lookup_request`. The "LORA" token was specifically ruled out as the cause: bare
  "Hey" (without "LORA") scored 0.648 — higher than "Hey LORA!" (0.612). "Hi" (0.636) and
  "Hello" (0.612) also cleared the gate. Confirmed a broad class effect, not a token artifact.

- **Length-controlled extension:** A three-track comparison (greetings vs. non-greeting small
  talk vs. `lookup_request` templates, matched at 1–4 word lengths) addressed whether "short
  strings in general collide" or "greetings specifically collide." The comparison table showed
  the greeting track (mean 0.623→0.602) running ~0.03 above the small-talk track (mean
  0.595→0.545) at every word count, with both tracks decaying with length. The initial read
  was an additive length+greeting effect. Per-utterance inspection revised this: the gap was
  produced by specific lead tokens ("hey", "hi", "what's up") scoring anomalously, not by
  greeting-ness as a semantic category — "good morning" (0.586–0.594) and "hello" (0.601–0.612)
  did not clear the gate reliably. The operative mechanism is lexical-token-specific, not
  category-level. This revision is part of the record; the table itself is not wrong, but the
  additive-effect interpretation was superseded by the per-utterance data.

*Fix:* Four phrases added to `_SEARCH_NEGATIVE_FILTER` — "hey lora", "hi there", "hey there",
"what's up" — blocking the embedding call before it reaches the gating logic. Selected over
threshold adjustment or template-set change as the narrowest reversible option, consistent with
the 2026-06-26 precedent. Bare "hi" and "hey" were assessed and deliberately excluded: under
the filter's `phrase in lowered` substring mechanism, "hi" collides with "history", "this",
"high", "vehicle", and similar; "hey" collides with "they". The multi-word forms carry no
collisions found in testing.

*Verification:*

1. **Unit tests (+11):** `TestGreetingFalsePositiveFilter` in `test_planner_phase3.py` — 4
   membership tests (one per new phrase); 4 behavioral tests confirming `_semantic_search_intent()`
   returns `None` for the confirmed-live utterance forms ("Hey LORA!", "hi there", "what's up?",
   "hey lora?", the last confirming the substring check is not tail-anchored); 1 non-regression
   test confirming genuine `lookup_request` utterances still fire the gate; 2 documented-gap
   tests asserting bare "hi" and "hey" are *not* in the filter, with docstrings pointing to
   this open item and the collision data.
2. **Live diagnostic (both passes):** Real `EmbeddingEngine`, real `Planner.__init__()`. The
   known-anchor utterances reproduced prior scores (1.000 on their respective groups). The
   four new filter phrases produced `None` returns confirmed by the first-pass output.

*What remains open:*

- **Bare "hi" and "hey" still unfiltered.** Collision data documented in the `planner.py`
  comment block and in `TestGreetingFalsePositiveFilter`'s documented-gap tests. Pending
  either a word-boundary-matched filter path or a different mechanism. Distinct from the
  identity-adjacent siblings deferred in the 2026-06-26 update — both are known false-positive
  candidates not yet added, but they arise from different data and different collision
  constraints.
- **Why these specific tokens collide is unknown.** The 300m EmbeddingGemma model places
  "hi", "hey", and "what's up" anomalously close to the `lookup_request` template group;
  no structural explanation was found in the diagnostic data. Logged only.
- **General 0.60 negative-side margin still unverified.** This is the second confirmed
  false-positive batch on `lookup_request` since the 2026-06-25 threshold lowering (the
  first was the 2026-06-26 identity/capability batch). The same unverified-margin risk
  named in that update persists; this update adds a second data point to the same open
  problem rather than introducing a new one.

**Open Item 3 — Update 2026-06-28 (lookup_request template replacement — Candidate Set 1;
explicit_search_action threshold raised 0.68 → 0.72).** Two changes shipped to `planner.py`,
both backed by diagnostic reports in `diagnostics/reports/` dated 2026-06-28.

**Change 1 — `lookup_request` template replacement.** The four templates added 2026-06-25 ("can
you look up", "can you look that up for me", "could you look up", "can you look into this for me")
produced a threshold-unfixable false-positive surface: 6 of 14 tested adversarial phrasings in
the "can/could/would you + verb" family scored 0.81–0.90 against those templates — above every
confirmed true positive's score. These four templates were replaced with Candidate Set 1
(object-specificity fix), which anchors on concrete queryable objects rather than the bare
modal-question scaffold. The current live `lookup_request` templates (production values as of
2026-06-28, read from `planner.py` directly):

- *(original 5, unchanged)* `"look up this"`, `"look that up"`, `"go ahead and look it up"`,
  `"find information on this"`, `"find out about this"`
- *(Candidate Set 1, replacing the 4 removed templates)* `"can you look up the release date for
  this"`, `"could you look up what year this happened"`, `"can you look up information about the
  latest Apple products"`, `"could you find out the current stock price for me"`

Effect on the three 2026-06-25 incident utterances ("Can you look up Apple's price hike for the
MacBook Neo and iPad?", "Can you look up their next-generation in-house Microsoft AI models?",
"Can you look up Microsoft's next-generation in-house AI models?"): all three remain gate-positive
(LR 0.7653 / 0.6522 / 0.6409, all ≥ 0.60 threshold). Cat A live false positives (3/3): all now
score below 0.60 under Set 1. Cat D adversarial false positives at 0.60: 13/14 → 6/14 remaining.
Source: `diagnostics/reports/lookup_request_template_rework_2026-06-28.md` and
`diagnostics/reports/full_pertable_lr_set1_esa_2026-06-28.md`.

**KNOWN ACCEPTED RESIDUAL — 6/14 adversarial phrasings remain gate-positive under Set 1.** These
fire via the modal-question scaffold and are not eliminated by object-specificity alone. By
category (per `diagnostics/reports/full_pertable_lr_set1_esa_2026-06-28.md`):

- D-verb-swap ×4: "Can you help me with this?", "Could you check this for me?", "Would you look
  at this?", "Can you tell me about this?"
- D-modal-swap ×2: "Will you look into this?", "Do you mind looking at this?"

Each can be individually patched via `_SEARCH_NEGATIVE_FILTER` if confirmed as a live false
positive. They are not pre-emptively added because `_SEARCH_NEGATIVE_FILTER` uses substring
matching, and conservative addition prevents silent suppression of legitimate queries.

**Change 2 — `explicit_search_action` threshold raised 0.68 → 0.72.** Two adversarial negatives
scored ESA 0.69–0.70 via the single bare-verb template "go look it up" — whose bare "look" token
produces syntactic overlap with "look at"/"look into" phrasing. Zero cost to true positives: the
three 2026-06-25 incident utterances all scored ESA ≤ 0.5785, well below either threshold.
Source: `diagnostics/reports/explicit_search_action_margin_assessment_2026-06-28.md`.

**PROVISIONAL STATUS — `explicit_search_action` threshold (0.72) is under observation.** Per
Michael's stated intent, this is being shipped to observe live behavior for several days before
being treated as settled. Any confirmed live false negative on `explicit_search_action` (gate
misses a genuine explicit-search instruction in the 0.68–0.72 band) is the trigger to revisit.
Not a permanently closed item.

Tests (+9 net in `test_planner_phase3.py`, file-scoped count 101 → 110): 2 stale-comment-only
fixes (pass/fail unchanged); 2 existing tests updated for the template and threshold changes
(flagged pass→fail in docstrings); new class `TestSet1TemplateFix20260628` (8 tests): 3 Cat C
true-positive gate assertions, 2 Cat D fixed-false-positive assertions (no longer fire under
Set 1), 3 ESA threshold boundary tests (0.73 fires, 0.69 does not, 0.85 fires).

---

**Open Item 4 — Live near-miss on `explicit_search_action` threshold, compound
instruction (2026-06-23).** A live, unscripted turn — "Look up karpathy llm
wiki then propose ways it implement it into Localist design." — scored
`explicit_search_action: 0.618`, the highest of all four groups
(`lookup_request: 0.597`, `knowledge_request_open: 0.462`,
`freshness_request: 0.404`), but fell short of the then-current 0.68 gating threshold (raised
to 0.72 on 2026-06-28 — see Update 2026-06-28 above — making this utterance 0.102 below the
current threshold, a wider gap than when this item was first logged).
`tools_to_call` was not populated; no LangSearch call occurred. Priority 4
matched instead via corpus score (0.638 ≥ 0.550), routing to
`conversational_agent` with wiki-only RAG context. The model correctly
stated it did not have the Karpathy material and asked the user to supply
it, rather than fabricating a claim about Karpathy's content — the
fail-safe (prompt-level "you do not simulate certainty" framing, not the
classifier) held, consistent with Turn 1 of the §10.3 live recreation.

This is structurally similar to §10.3 Turn 1 (a real informational-intent
turn scoring sub-threshold, correctly falling back to an honest non-answer)
but is a distinct data point: a different group won the argmax
(`explicit_search_action` here vs. `knowledge_request_open` in the §10.3
recreation), and the instruction was compound — a lookup clause ("Look up
karpathy llm wiki") joined to a proposal clause ("propose ways... into
Localist design") in a single 81-character instruction. Whether the
proposal clause's embedding signal diluted the lookup clause's score below
what it would have scored alone is a plausible mechanism, not a confirmed
one — no isolated test of the lookup clause alone has been run.

**Not actioned.** Per Open Item 3's standing posture and the project's
single-occurrence discipline (a single proposed mechanism is a hypothesis
to verify, not a finding to act on), no threshold change, no compound-
instruction-splitting logic, and no new gating behavior follows from this
one turn. Logged so it counts toward Open Item 3's "revisit if live false
negatives are observed" criterion — this is one such occurrence, not yet a
pattern. Revisit if additional compound or near-threshold instructions are
observed scoring in the 0.60–0.72 band for `explicit_search_action` (updated
from the original 0.60–0.68 band to track the 0.68 → 0.72 raise on
2026-06-28 — see Update 2026-06-28 above).

**Open Item 5 — `web_search` SUCCESS with irrelevant results; no fallback mechanism (2026-06-28).** A live turn asking "Tell me about Localist Framework?" scored `lookup_request=0.670` (≥ 0.65 gating threshold), routing correctly to `web_search` via Priority 3. LangSearch returned 3 real results and the call SUCCEEDED — no error, no `success=False`, so the Step 3b corpus fallback introduced in §4.6.1 did not fire. The returned results were entirely irrelevant: generic uses of "localism" and "localist" in unrelated academic and ML contexts, not information about this project. The model's response reflected the irrelevant web content.

This is a distinct failure mode from §4.6.1: that fix handles tool FAILURE (search returns an error or throws an exception); this case is tool SUCCESS with semantically irrelevant results, for which no fallback mechanism exists today.

**Routing-destination question, not threshold-tuning.** Project-specific questions about the Localist Framework itself are structurally better served by corpus/RAG than by generic web search, regardless of classifier gate accuracy — the project is not publicly indexed, so a web search for "Localist Framework" will reliably surface unrelated content. This is a routing-destination problem: the classifier gates correctly on lookup intent, but sends the query to the wrong tool for this class of subject matter. It is not a false-positive problem (the gate should not have fired at all). The distinction matters for deciding the right fix: threshold tuning would suppress a correctly-gated query, whereas destination logic would route certain classes of query to the corpus even when the search gate fires. Michael's explicit choice was to file this under §10.4 alongside the threshold/classifier open items rather than as a separate routing-architecture item, since the boundary between "tune the classifier" and "add destination logic" is unresolved.

No action taken. Single occurrence; not yet a pattern. Logged per the project's single-occurrence discipline.

**Open Item 6 — P3 semantic gate short-circuits before P4 corpus evaluation; `_WIKI_QUERY_KEYWORDS`
lacks coverage for "wiki files"-style phrasings. Both unresolved as of 2026-06-28.** Confirmed
live during the 2026-06-28 incident that originated the `lookup_request` template diagnostic.
Two structural facts about the routing ladder:

1. **P3 short-circuits before P4.** When Priority 3's semantic gate fires, `route()` returns
   immediately — Priority 4 corpus evaluation is never reached. For instructions with lookup
   intent directed at local corpus content (e.g. "Can you read my wiki files?"), this means the
   corpus that contains the answer is not consulted even when a matching document exists. The
   2026-06-28 Candidate Set 1 fix reduces the probability of false-positive P3 fires on these
   phrasings (Cat A LR scores dropped below 0.60 under Set 1), but does not address the
   structural ordering.

2. **`_WIKI_QUERY_KEYWORDS` lacks coverage for "wiki files"-style phrasings.** Priority 4 Path A
   fires on explicit wiki/vault trigger keywords ("check the wiki", "search the wiki", "from the
   wiki", "in the wiki", "vault", etc.). A phrasing like "my wiki files" does not match any
   current `_WIKI_QUERY_KEYWORDS` entry, so P4 Path A cannot catch it even when P3 does not
   fire. P4 Path B coverage (corpus score ≥ 0.55) is not guaranteed.

Neither root cause was addressed by the 2026-06-28 template change; that change targeted the
false-positive collisions that made incorrect P3 routing likely, not the structural ordering or
keyword-coverage gap that makes P4 the correct destination for this phrasing class. Not
scheduled. Logging here so the originating incident's unresolved structural causes are not
conflated with the shipped threshold/template fix.

### 10.5 Test Suite

Current state: **436 + 9 = ~445 tests, 0 failures** (436 verified fresh 2026-06-27; +9 net in
`test_planner_phase3.py` from the 2026-06-28 session, file-scoped count confirmed 101 → 110;
full-suite re-run not performed for that session).

The classifier was built across four sequential slots (all in `backend/tests/test_planner_phase3.py`),
then extended in two later sessions:

| Slot / Session | Purpose | Before | After | Net |
|---|---|---|---|---|
| Diagnostic 1 (2026-06-22) | `_semantic_search_intent()` scaffold; `embed_fn` wiring; logging only, no routing change | 318 | 329 | +11 |
| Diagnostic 2 (2026-06-22) | Expand return type to `(best_group, best_score, all_scores)`; per-group score logging | 329 | 331 | +2 |
| Fix 1 (2026-06-22) | Live gating via `_SEMANTIC_GATE_THRESHOLDS`; first routing change in `_priority3_tool()` | 331 | 336 | +5 |
| Fix 2 (2026-06-22) | "web search" and "do a search" added to `_WEB_SEARCH_KEYWORDS` | 336 | 339 | +3 |
| OI 3 update A+B (2026-06-25) | `lookup_request` template expansion (5→9) and threshold lowering (0.65→0.60); 6 new tests in `TestPriority3SemanticGating` | 339 | 345 | +6 |
| P4a removal (2026-06-26) | `_priority4a_identity()` / `force_rag` removed; −3 deleted, +16 added across `test_planner_phase3.py` and `test_controller_phase4.py` — see §8.8 OI 12 for full breakdown | 405* | 418 | +13 |
| OI 3 update 2026-06-26 | `_SEARCH_NEGATIVE_FILTER` identity/capability additions; `TestIdentityCapabilityNegativeFilter` in `test_planner_phase3.py` | 418 | 425 | +7 |
| OI 3 update 2026-06-27 | `_SEARCH_NEGATIVE_FILTER` greeting-form additions; `TestGreetingFalsePositiveFilter` in `test_planner_phase3.py` (4 membership + 4 behavioral short-circuit + 1 non-regression + 2 documented-gap) | 425 | 436 | +11 |
| OI 3 update 2026-06-28 | `lookup_request` Candidate Set 1 template replacement; ESA threshold 0.68→0.72; `TestSet1TemplateFix20260628` (8 new tests); 2 stale-comment fixes; 2 pass→fail updates in existing tests — all in `test_planner_phase3.py` (file-scoped: 101→110; full-suite not re-run) | 436 | ~445 | +9 |
| **Total** | | **318** | **~445** | **+127** |

\* The P4a-removal row uses 405 as its before-count because that was the confirmed baseline at the start of that session. The gap between 345 (OI 3 update A+B) and 405 reflects tests added across unrelated sessions (§8, §9, and other §8.8 close-outs) not tracked in this table.

Fix 1's net of +5 reflects 7 new tests in `TestPriority3SemanticGating` minus 2 tests removed
from its predecessor class `TestPriority3ToolUnaffectedBySemantic`, whose premise — "semantic
signal never affects routing" — became false after that slot.

