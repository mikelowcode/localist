"""
Wikipedia background-summary match rule (§14.21, Phase 4)
==========================================================
Decides whether the best Wikipedia hit for a web_search query is confidently
the thing the query is about, so its summary can ride along with the web
results. Deterministic, no inference — shared by
MCPToolDispatcher._wikipedia_background (live, behind
LOCALIST_WIKIPEDIA_ENRICH) and diagnostics/diag_wikipedia_enrich_replay.py
(the replay that decides that flag's default), so both measure the same rule.

The rule — all must hold:
  1. The best match is not a disambiguation page.
  2. Every content word of the title, with any "(qualifier)" removed, is in
     the query: "Mercury (planet)" needs "mercury". A qualifier word may
     appear in the query but isn't required.
  3. The title covers at least MIN_COVERAGE of the query's content words —
     so "weather in Boston" doesn't attach "Boston": the summary has to be
     about most of what was asked, not one word of it.
  4. The query isn't asking for something an encyclopedia summary can't
     hold: a live/latest/price/schedule question (FRESHNESS_WORDS).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MIN_COVERAGE: float = 0.5

# Stopwords and question/command filler: never "content". Kept small and
# explicit — the coverage ratio is the real guard; this only stops "what",
# "the", "of" from diluting or inflating it.
STOPWORDS: frozenset[str] = frozenset("""
a an the of in on at to for from by with about and or vs versus is are was were be been
what whats who whos whom which when where why how does do did can could would should will
tell me give show find search look up lookup explain describe please i my you your it its
this that these those there their they some any much many more most than as into over
""".split())

# A query asking for live or changing facts gets no encyclopedia summary:
# the page can't hold the answer, and its stale numbers might be mistaken
# for it.
FRESHNESS_WORDS: frozenset[str] = frozenset("""
latest today tonight yesterday tomorrow now current currently recent recently
news price prices pricing cost costs deal deals sale buy cheapest weather forecast
score scores schedule tickets stock stocks release update updates live near
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_QUALIFIER_RE = re.compile(r"\s*\(([^)]*)\)\s*$")


def _fold(text: str) -> str:
    """Lowercase and strip accents, so "Café" matches "cafe"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN_RE.findall(_fold(text)):
        tok = tok.replace("'s", "").replace("'", "")
        if tok:
            out.append(tok)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


@dataclass(frozen=True)
class MatchDecision:
    attach:   bool
    reason:   str           # "match" or why not
    coverage: float         # title content words / query content words


def decide(query: str, title: str, disambiguation: bool) -> MatchDecision:
    """Apply the rule in the module docstring to one (query, best match)."""
    q_tokens = content_tokens(query)
    q_set = set(q_tokens)
    if not q_set:
        return MatchDecision(False, "empty_query", 0.0)
    if q_set & FRESHNESS_WORDS:
        return MatchDecision(False, "freshness", 0.0)
    if disambiguation:
        return MatchDecision(False, "disambiguation", 0.0)

    qualifier = _QUALIFIER_RE.search(title)
    base = _QUALIFIER_RE.sub("", title) if qualifier else title
    t_set = set(content_tokens(base))
    if not t_set:
        return MatchDecision(False, "empty_title", 0.0)
    if not t_set <= q_set:
        return MatchDecision(False, "title_not_in_query", 0.0)

    covered = t_set | (set(content_tokens(qualifier.group(1))) & q_set if qualifier else set())
    coverage = len(covered & q_set) / len(q_set)
    if coverage < MIN_COVERAGE:
        return MatchDecision(False, "low_coverage", round(coverage, 2))
    return MatchDecision(True, "match", round(coverage, 2))
