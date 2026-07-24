# Episodic-relevance semantic gate — threshold assessment (2026-07-23)

## Trigger

Live failure: the instruction "Help me prepare for the upcoming Claude Impact
Lab on August 6th." did not trigger episodic-memory retrieval, even though a
matching episode existed in the episodes bank. Root cause: Planner Priority 5
(`_priority5_episodic()`) gated `fetch_episodic` behind a small hardcoded
keyword list (`_EPISODIC_KEYWORDS` — "remember", "preference", "decision",
"my project", etc.) with no semantic fallback. This phrasing matched none of
them.

Separately, the `_episodic_injected` session-flag comments (`planner.py:967-969`
and the old `_priority5_episodic()` docstring) claimed that once episodic
bullets had been injected once this session, "all further Priority 5 checks
return True without [keyword] evaluation" — stale; the actual code always
requires a keyword match regardless of that flag. Corrected in the same
change as this gate (docs only, no behavior change there).

## Decision

Add a semantic fallback to Priority 5, structurally mirroring
`_semantic_search_intent()` (Priority 3's pattern: precomputed template
embeddings, cosine similarity, a tuned threshold) but kept as an **entirely
separate** mechanism — own template list, own threshold constant, own
instance state (`_episodic_template_embeddings`), own scoring method
(`_episodic_semantic_relevance()`). Deliberately *not* added as a new group
inside `_SEARCH_INTENT_TEMPLATES` / `_SEMANTIC_GATE_THRESHOLDS`: those carry
P3-specific negative-filter/tie-break machinery (`_ALL_SEARCH_NEGATIVE_FILTERS`,
`_resolve_negative_filter_conflict`) tuned for web-search-intent collisions
that have nothing to do with episodic relevance — reusing that path risked a
P3-specific filter match silently suppressing an unrelated P5 signal.

Still gated by the existing `_semantic_gating_disabled` guard (embedding
model mismatch against `_TUNED_EMBEDDING_MODEL`) — the "cosine thresholds
aren't portable across embedding models" rationale applies here too, and this
is the same guard, not a new one.

## Method

`diagnostics/score_episodic_relevance_templates.py` — embeds a candidate
template set plus a battery of hand-written positive/negative utterances
against the live `mlx-community/embeddinggemma-300m-4bit` model (the same
one every other threshold in `planner.py` is tuned against) and reports
best-template cosine score per utterance.

**Round 1** — bare "help me" / "remind me" / "get ready" verb-phrase
templates. Found a real collision: "Remind me how photosynthesis works."
(0.6952) and "Help me write a cover letter for a job application." (0.6724)
scored *above* several genuine positives (min true positive 0.6644 excluding
one outlier) — the same "threshold-unfixable, bare-verb collision" failure
shape already documented for `lookup_request`'s 2026-06-25 incident and
`research_intent`'s v1→v2 rework elsewhere in this file.

**Round 2** — re-anchored every template on an explicit personal/calendar
referent ("my upcoming event", "my calendar", "we planned/discussed/decided",
"my schedule", "my appointment") rather than a bare imperative verb, the same
"object-specificity fix" already applied to `lookup_request`/`research_intent`.
Final template set (7 phrases):

```
help me prepare for my upcoming event
what do I have coming up on my calendar
remind me what we planned for this
catch me up on what we discussed before
what is on my schedule this week
help me get ready for my appointment
what did we decide about this before
```

## Result — threshold = 0.70

10 positives / 20 negatives (20-utterance battery, not a large formal
diagnostic corpus — small-N, revisit if live behavior disagrees, same posture
already used for `explicit_search_action`'s threshold history in this file):

- **8/10 true positives** cleared 0.70, including the exact triggering
  utterance (0.7742).
- **19/20 true negatives** correctly excluded.
- 2 false negatives: "Catch me up on my project status." (0.6833) and "What
  did I say I would do about the server migration?" (0.5924) — both still
  reach the existing deterministic keyword path if phrased with an explicit
  keyword; not blocked, just not caught by this gate.
- 1 false positive: "Can you help me plan a birthday party for my friend?"
  (0.7406) — arguably a defensible true positive too (episodic memory about
  the friend's preferences would genuinely help here), not treated as a real
  miss.

No single threshold in the tested range achieves full separation — same
conclusion already reached for `research_intent`'s threshold assessment. 0.70
was chosen because it fully excludes every unambiguous negative in the
battery (max negative 0.6749) while keeping 8/10 positives, including the
original failing case. Revisit directly if live false positives or false
negatives are observed, per the same "ship to observe" posture already used
for other thresholds in this file.
