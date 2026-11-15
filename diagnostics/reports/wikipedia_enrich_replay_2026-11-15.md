# Wikipedia background-summary replay — 2026-11-15

**Question.** Should `web_search` turns get a Wikipedia background summary by
default (`LOCALIST_WIKIPEDIA_ENRICH`, §14.21 Phase 4)? The summary is attached
only when `localist.wikipedia_match.decide()` calls the best Wikipedia hit a
confident match.

**Decision input.** We need to know how often the rule attaches a summary, how
often the attached article is the wrong one, and how often it fires on real use.

**Script.** `diagnostics/diag_wikipedia_enrich_replay.py`. It runs the same rule
as the live code against live Wikipedia, using the free Action API with one
request per unique query. It only reads the chat database. Per-query details
for sets A and B stay outside the repo, so only their aggregates appear here.

## Rule (unchanged throughout; not tuned on these results)

A summary is attached only when all four hold:

1. The best hit is not a disambiguation page.
2. Every content word of the title (minus any "(qualifier)") is in the query.
3. The title covers ≥ 50% of the query's content words.
4. The query has no freshness word (latest, today, price, news, weather,
   score, schedule, update, …).

## Results

| Set | Queries | Attached | Wrong article | Right but unhelpful |
|---|---|---|---|---|
| A: real `web_search`/research turns | 9 | 1 (11%) | 0 | 0 |
| B: every user request, as if it were a search | 120 | 2 (2%) | 0 | 1 (a greeting) |
| C+: synthetic encyclopedic searches | 30 | 24 (80%) | 0 | 0 |
| C−: synthetic searches that need no summary | 30 | 4 (13%) | 2 | 2 |

**Why queries in A and B didn't attach:**
- **A:** no hit 5, title not in query 2, freshness 1.
- **B:** no hit 98, title not in query 13, freshness 6, disambiguation 1.

"No hit" is Wikipedia's search returning nothing for a whole conversational
sentence (it needs every word to match). Those sentences would fail the
coverage rule anyway.

**Set C+ misses (6).** In each case Wikipedia's top hit was a neighbouring
article, and the rule correctly declined it:

- "what is a black hole" → *Binary black hole*
- "how does photosynthesis work" → *Artificial photosynthesis*
- "who was Nikola Tesla" → *Nikola Tesla Museum*
- "who invented the telephone" → *The Telephone Cases*
- "what is the Krebs cycle" → *Citric acid cycle*
- "what is a transformer neural network" → *Transformer (deep learning)*
  (rejected on coverage, at 0.33)

**Set C− attachments (4):**

- Wrong article:
  - "who won the game last night" → *The Last Night (video game)*
  - "Mercury retrograde dates" → *Mercury Retrograde*
- Right article, but it doesn't answer the question:
  - "Xbox Game Pass games leaving this month" → *Xbox Game Pass*
  - "how many calories in a banana" → *Banana*

## Reading

- **Precision:** 2 wrong articles out of 31 attachments overall (~6%). All
  of them come from the synthetic negatives. None of the user's own
  requests got a wrong article.
- **Reach on real use is low.** One of the 9 real searches would have
  gained a summary. This user's searches mostly go through the dedicated
  news, Hacker News and GitHub routes, which this flag does not touch.
- **Overlap with Phase 3.** When the search's top hit is the Wikipedia
  article itself, enrichment already reads the full article through the
  Wikipedia API. The summary is then skipped as a duplicate (added after
  the first live check showed it).

**Default: off.** Reach is too low to justify a ~6% wrong-article rate in
everyone's prompts. The flag stays available as an opt-in. Revisit with the
research loop (Phase 5), where background matters more, or with more stored
`web_search` turns.

## Live check

The check ran with the flag on and `backend/.env` loaded, so search went to
Brave, with no Ollama calls.

- "who was Ada Lovelace" and "Battle of Hastings": the Wikipedia lookup
  ran alongside the Brave search. In both cases Brave's top hit was the
  Wikipedia article itself. Enrichment read that article in full through
  the API (Phase 3), so the summary was skipped as a duplicate. Each turn
  took about 0.7 s.
- "iPhone 17 price": no summary (freshness).

This adds to the case for default off. For the encyclopedic questions
where the rule fires, a web search usually ranks the Wikipedia article
first anyway, and Phase 3 already turns that into a clean full-article
read.
