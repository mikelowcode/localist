# Daily News Brief — Header Button → Chat Conversation

**Status:** Built and live-verified, 2026-07-22 (see §11); frontend redesigned same day
after live use (see §12)
**Author:** scoped with Michael, 2026-07-22
**Depends on:** §14.9 (`news_search` / NewsAPI integration, already built —
`docs/architecture/14-localist-mcp-tool-layer.md`), `MemoryManager.add_chat_turn()`
(§12/§20, `docs/architecture/12-chat-history-tab.md` / `20-episode-browsing-ui.md`)

## 1. Goal

A "Daily Brief" feature surfaced as a single button in the app's top header bar (next
to the `● Ollama` runtime-status badge), curating news across three fixed geographic
tiers — **World**, **National**, **Local** — plus three user-selected special-interest
topics. Pressing it materializes the brief as a real conversation in Localist's
existing chat history, not a standalone panel — so the user can immediately ask the
model follow-up questions about what's in the brief, with the brief content already
sitting in that conversation's turns.

This is deliberately **not** routed through Planner/`MCPToolDispatcher` — it has no
chat-turn instruction to route, no tool-call ambiguity to resolve. It's a dedicated
REST feature, same tier as `/chat/history` or `/wiki/*`, that happens to write its
output into the `chat_turns` table like a normal conversation.

## 2. Reconciling the category scheme against what NewsAPI actually supports

Read both `/docs/get-started#top-headlines` and the full `/docs/endpoints/top-headlines`
reference before scoping this. Two real gaps exist between the requested category
scheme and the API's actual surface:

| Category | NewsAPI reality | Mapping |
|---|---|---|
| **World** | No "world" category exists. `category` ∈ `{business, entertainment, general, health, science, sports, technology}` only; `country` is a single code per call, not a scope selector. | `GET /v2/top-headlines?category=general` (no `country`) — closest proxy to "global general news." **Confirmed live, 2026-07-22 (not just unverified): this does not return a cross-country mix.** With no `country` param, NewsAPI returned 5/5 US-outlet results (NY Post, MLB.com, Ars Technica, Yahoo, CBS News) — identical to the National section's `country=us` results for a US `home_country`. World and National are effectively duplicative for a US-based user today; there is no available NewsAPI fix for this (it's the API's actual behavior, not a bug in this integration) — documented as a known limitation rather than silently shipped as if resolved. |
| **National** | `country` is exactly one 2-letter ISO 3166-1 code per call (e.g. `us`) — no concept above or below country level. | `GET /v2/top-headlines?country={home_country}`. Needs a `home_country` preference, default `"us"`. |
| **Local** | **No city/region granularity exists in this API at all.** | Approximated via `GET /v2/everything?q={local_query}&sortBy=publishedAt` — a free-text keyword the user sets (e.g. `"Seattle"`), matched against NewsAPI's full source archive. Keyword matching, not real local-outlet curation — labeled as an approximation in the UI, not implied to be genuine local-news coverage. |

## 3. Special-interest topic pool

NewsAPI's `category` enum is fixed and small, so the pool splits into native-category
topics (reliable) and keyword-approximated topics (same caveat as Local):

| Topic | Backing call |
|---|---|
| Finance | `top-headlines?category=business` |
| Technology | `top-headlines?category=technology` |
| Science | `top-headlines?category=science` |
| Health | `top-headlines?category=health` |
| Sports | `top-headlines?category=sports` |
| Entertainment | `top-headlines?category=entertainment` |
| Video Games *(keyword-approximated)* | `everything?q="video games"&sortBy=publishedAt` |
| Politics *(keyword-approximated)* | `everything?q=politics&sortBy=publishedAt` |
| Crypto *(keyword-approximated)* | `everything?q=cryptocurrency OR crypto&sortBy=publishedAt` |

9 members, user picks exactly 3. Adding/removing a member later is a map-entry change,
not a schema change.

## 4. Data model — two new tables (`memory_manager.py`, schema v10)

Follows the existing `chat_history_settings` single-row pattern exactly (not the
`Settings(BaseSettings)`/env-config pattern, which is process config, not per-user
preference data):

```sql
CREATE TABLE news_preferences (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    home_country    TEXT    NOT NULL DEFAULT 'us',
    local_query     TEXT,                          -- NULL until the user sets one
    topics_json     TEXT    NOT NULL DEFAULT '[]',  -- exactly 3 keys from §3's pool
    updated_at      REAL    NOT NULL
);

CREATE TABLE news_brief_cache (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    brief_date      TEXT    NOT NULL,   -- 'YYYY-MM-DD', local date this cache is valid for
    content_json    TEXT    NOT NULL,   -- full rendered brief (see §6), feeds the preview popover
    conversation_id TEXT,               -- the conversation created for today's brief; NULL until first generation
    generated_at    REAL    NOT NULL
);
```

`conversation_id` is what makes "press the button again → reopen today's existing
conversation, don't create a duplicate" possible without a separate lookup table.

Standard migration path: bump `_SCHEMA_VERSION` to 10, add both `CREATE TABLE`s to the
fresh-install DDL block in `_init_db()`, add a matching `if from_version < 10:` block
in `_migrate()`.

## 5. Preferences endpoints

- `GET /news/preferences` — current row, or defaults (`home_country="us"`,
  `local_query=None`, `topics=[]`) if unset.
- `PUT /news/preferences` — sets `home_country`/`local_query`/`topics`; validates
  `topics` is exactly 3 entries, each a real key from §3's pool. No `news_brief_cache`
  side effect — changing preferences doesn't retroactively touch an already-generated
  brief; it takes effect on the next generation (i.e. tomorrow, or immediately if
  today's cache hasn't been generated yet).

## 6. The brief itself — two endpoints split by side effect

**`GET /news/brief/preview`** — hover/idle state only.
- Returns cached `content_json` if `brief_date == today`, else an explicit
  "not generated yet" empty state.
- **Never calls NewsAPI.** Hovering the header button must never spend API quota —
  this endpoint only ever reads `news_brief_cache`.
- Frontend renders the returned sections in a popover under the button with a
  lightweight progressive-reveal (typewriter-style) animation — purely cosmetic, since
  this is already-fetched static text, not a live model stream. No new streaming
  infrastructure needed; this does not reuse the SSE chat-streaming mechanism.

**`POST /news/brief/open`** — the button's click handler.
- If `news_brief_cache.brief_date == today` **and** `conversation_id` is set: return
  `{conversation_id, generated: false}` immediately. No NewsAPI calls, no new
  `chat_turns` rows — this is the "reopen today's brief" path, satisfying "pressing the
  button again shouldn't create a duplicate conversation."
- Else (stale or missing cache): fetch all up-to-6 sections (World, National, Local,
  ×3 topics) per §2/§3's mapping, with **per-section failure containment** — one
  section's NewsAPI call failing degrades that section to an empty/error state rather
  than failing the whole brief (same posture as `_run_research_loop`'s per-iteration
  handling). Format the successful sections into one markdown message. `uuid4()` a new
  `conversation_id`. Write **two** rows via the existing `MemoryManager.add_chat_turn()`:
  - `role="user"`, content a synthetic instruction (e.g. *"Give me today's news
    brief"*) — gives the conversation the normal alternating-turn shape the rest of
    the UI already expects, and gives a later follow-up question a natural anchor
    turn to build on, rather than an orphaned assistant-only conversation.
  - `role="assistant"`, content the formatted sections, `sources` populated from every
    article's URL across all sections — same provenance convention a normal grounded
    answer already populates (`RagSource`-style), not a special case.
  - Update `news_brief_cache`: `brief_date=today`, `conversation_id=<new id>`,
    `content_json=<raw sections>`, `generated_at=now`.
  - Return `{conversation_id, generated: true}`.
- Frontend navigates to `conversation_id` in both cases; the conversation loads through
  the existing conversation-loading path unchanged — no special-case rendering needed
  for a "brief" conversation vs. any other. **This covers display only** — see the
  correction immediately below for what actually makes a follow-up question work.

**No inference cost to generate a brief.** The assistant turn is pure data formatting
from the fetched NewsAPI sections — zero `runtime.infer()` calls. The model only
engages once the user asks a real question in that conversation, through the normal
chat pipeline, completely unchanged by any of this.

**Correction, found during live verification (2026-07-22): `chat_turns` is not what
feeds the model's working memory.** The original version of this plan assumed writing
to `chat_turns` was sufficient for "the brief is already in chat history so the user
can ask about it" — this is wrong, and was caught by actually asking a real follow-up
question end to end (§9, step 6) rather than trusting the design on paper. `chat_turns`
only backs history *display*/search (§12/§20's Chat History Tab and Episode Browsing
UI). The model's actual Slot 6 working memory is populated from a completely separate
table, `conversation_log`, read via `MemoryManager.get_context_window(task_id=...)` and
written via `MemoryManager.add()` — keyed not by `conversation_id` but by
`ControllerAgent._memory_key()`, which prefers `context["session_id"]` (a value the
frontend generates once per page load, `tasks.ts`'s `SESSION_ID` — shared across
whatever conversations are opened in that same browser tab, not scoped per persisted
conversation at all). Confirmed live: a follow-up question sent right after a brief
conversation was created had no idea the brief existed, until `POST /news/brief/open`
was also made to write the same exchange into `conversation_log` under the caller's
current `session_id` (now a field on the request body) — on **every** call, hit or
miss, since a page reload between two button presses gets a fresh `session_id` that
never saw an earlier generation. This is now wired in and live-verified working (§9).

## 7. Rate-limit budget

Free tier is 100 req/day. One full generation (cache miss) costs ≤6 NewsAPI calls;
cached same-day reopens and hover previews cost zero. Comfortably inside budget even
with several presses across a day — no caching layer beyond the simple date check
above is needed at this scale (same conclusion §14.9 reached for `news_search`).

## 8. Frontend

> **Superseded 2026-07-22 — see §12.** The header-button + hover-popover design below was
> built and briefly live, but live use showed the popover too small and its content
> truncated for real headlines. It was replaced by a persistent, collapsible right-side
> "Previews" tab; the header button was removed outright. This section is kept for
> history — §12 is the current design.

- **Header button** (top bar, next to the `● Ollama` badge): newspaper icon + label.
  States: *idle* (hover → preview popover if `GET /news/brief/preview` returns
  content) → *loading* (spinner; shown only on a real cache-miss generation, i.e. the
  first press of a new day) → navigates away on success.
- **Preview popover**: renders `GET /news/brief/preview`'s sections with the
  progressive-reveal animation described in §6. Empty state ("No brief generated yet
  today — click to generate") when nothing is cached.
- **Preferences UI**: home-country picker, local-query text input, 3-of-9 topic
  checkboxes (§3) — a small settings surface, likely under the existing Settings nav
  item rather than a new top-level route.
- No standalone `/news` route or article-card panel — superseded entirely by the
  chat-conversation delivery model above.

## 9. Explicitly out of scope

- Push/email delivery — no such infrastructure exists anywhere in this app; stays
  strictly pull-based (button press), matching the rest of Localist's design.
- A background scheduler pre-generating the brief before the user opens the app — not
  needed given the on-demand-cached, click-triggered model.
- Real local-news curation — NewsAPI cannot do this; the keyword approximation (§2) is
  a stated compromise, not a fix.
- `top-headlines`'s `sources` param — mutually exclusive with `country`/`category`, and
  no source-curation UI is scoped here.
- Real SSE token streaming for the preview popover — explicitly rejected in favor of a
  cosmetic-only reveal animation, since there's no live generation happening on a cache
  hit.
- An LLM-authored intro/summary line above the raw sections — explicitly rejected;
  brief generation stays zero-inference.

## 10. Build order

1. **Schema v10** — `news_preferences` + `news_brief_cache` (with `conversation_id`)
   in `memory_manager.py`, following the `chat_history_settings` migration pattern
   exactly; `MemoryManager` get/set methods for both tables.
2. **`backend/news_brief.py`** — NewsAPI call builders for World/National/Local + all
   9 topic-pool entries (§2/§3), each returning a normalized section shape; per-section
   error containment (a failed call returns an empty/error section, never raises past
   this layer). Own `NEWSAPI_API_KEY` read (main backend process, not `mcp_server/` —
   this feature has no chat/tool-dispatch involvement, so routing it through
   `localist-mcp`/MCP SSE would be unnecessary indirection; duplicating the small
   request-building logic already in `mcp_server/news_search.py` matches this
   codebase's established cross-process-duplication convention).
3. **`main.py` endpoints** — `GET`/`PUT /news/preferences`, `GET /news/brief/preview`,
   `POST /news/brief/open` (§5/§6), the latter's cache-miss path calling
   `MemoryManager.add_chat_turn()` twice (synthetic user turn, then the formatted
   assistant turn with `sources` populated) and updating `news_brief_cache`.
4. **Frontend** — header button (three states), preview popover with the
   progressive-reveal animation, preferences UI (likely under Settings). Navigation on
   `POST /news/brief/open`'s response reuses the existing conversation-loading path —
   no new conversation-rendering logic.
5. **Tests** — preference validation (exactly-3, valid pool keys, defaults);
   `news_brief_cache` hit/miss/date-rollover logic, especially the
   same-day-reopen-vs-regenerate boundary; per-section failure containment (one topic
   failing doesn't fail the whole brief); the two-`chat_turns`-row insertion shape
   (alternating roles, `sources` populated, `conversation_id` shared); preview-endpoint
   read-only guarantee (a test asserting it never calls NewsAPI regardless of cache
   state).
6. **Live verification** — confirm the World mapping's actual behavior
   (`category=general`, no `country`) returns something globally meaningful rather than
   silently narrowing (the one real unknown left after doc research alone, §2); one
   real end-to-end button press (cold cache) confirming a new conversation appears with
   correctly formatted, sourced content; one repeat press same day confirming it
   reopens the same conversation rather than duplicating; one real follow-up question
   in that conversation confirming the model can see the brief content in its working
   memory.

## 11. Live Verification (2026-07-22)

All against the real running stack (backend + `localist-mcp` + real NewsAPI, no mocks),
schema v10 applied via the same self-heal pattern documented in
`docs/architecture/16-runtime-backend-layer.md` §16.4 (this build hit the identical
`--reload`-races-a-multi-step-schema-edit drift live, on the `news_preferences`/
`news_brief_cache` tables this time — self-heal extended, fired correctly, no manual DB
surgery needed).

- **World mapping confirmed, not just verified-as-non-crashing**: real query returned
  5/5 US-outlet results with no `country` param — see §2's updated finding. World and
  National are duplicative for a US `home_country` today; no NewsAPI-side fix exists.
- **Cold press**: `POST /news/brief/open` fetched all 6 sections (~1.45s), created a new
  conversation with correctly alternating `user`/`assistant` `chat_turns` rows, markdown
  correctly sectioned, `sources` populated with `type: "web"` entries pointing at real
  article URLs.
- **Warm press (same day)**: returned the same `conversation_id`, `generated: false`,
  ~0.01s — zero NewsAPI calls, confirmed by response time alone.
- **Follow-up question — failed on the first attempt, fixed, then passed.** The first
  real attempt asked the model about an article from the brief; it had no memory of it
  at all. Root-caused to the `chat_turns`-vs-`conversation_log` gap described in §6's
  correction note. After wiring `session_id`-keyed `conversation_log` seeding into
  `POST /news/brief/open`, the identical follow-up question correctly recalled the
  specific article, its source, and a technical detail from its body — confirmed via a
  real `/task` call sharing the same `session_id` the brief was opened with.

## 12. Frontend redesign: header popover → right-side "Previews" tab (2026-07-22)

Full component/store-level detail lives in `docs/architecture/07-localist-ui.md` §7.14; this
section records why, since §8 above is now superseded history.

**Why.** A screenshot from the user showed the popover rendering real content, but at
280px×220px with truncated lines — not usable for actually reading a headline. Rather
than just resize the popover, the request was to relocate the preview into a proper
collapsible right-side tab ("Previews") with reserved block slots for future daily-update
sources (GitHub, Hacker News) — scoped as layout only, no live wiring for those two yet.

**What changed:**
- `PreviewsPanel.svelte` (new) + `previewsPanel.ts` (new store) — a persistent third
  `#app-shell` grid column, collapsed by default to a 40px vertical tab, expanding to
  320px on click. No drag-resize (unlike the left sidebar) since nothing asked for one.
- The News block inside it shows up to 3 full article links per section, not one
  truncated line — live-verified by the user in Safari (article links correctly open in
  a new tab).
- Two reserved placeholder blocks, **GitHub** and **Hacker News** ("Coming soon", no live
  data) — intentionally scoped now, deferred integration later.
- The header button was removed from `StatusBar.svelte` entirely, including its
  hover-popover state machine and progressive-reveal timer (§6/§8's cosmetic-reveal
  animation is now dead code, removed with it — the panel just renders the live store
  value directly, no animation). The only remaining trigger is a plain-text underlined
  link inside the panel reading "Daily News Brief Refresh," wired to the same
  `openNewsBrief()` → navigate-to-conversation flow.

**Two Settings bugs found and fixed along the way** (full root-cause detail at
`docs/architecture/07-localist-ui.md` §7.14, not duplicated here): a native browser
address-autofill dropdown appearing on the "Local area" field (fixed with
`autocomplete="off"`), and a genuine Svelte reactivity bug where a store-derived
`$: { ... }` block sharing scope with `bind:value` targets caused every keystroke to be
silently reverted — fixed by replacing the continuous reactive mirror with a one-time
`onMount` sync, since the preferences form was always meant to be freely editable until
an explicit Save.

Verified via `svelte-check`/`vite build` (0 errors across all revisions) plus the user's
own live click-through in Safari — no automated browser tooling is available in this
environment.
