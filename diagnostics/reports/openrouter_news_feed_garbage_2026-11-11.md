# OpenRouter News Feed junk replies: findings (2026-11-11)

Script: `diagnostics/diag_openrouter_news_feed_garbage.py`. Raw output:
- `openrouter_news_feed_garbage_results.csv` and `openrouter_news_feed_garbage_raw.jsonl`: the 4 baseline requests
- the same two files with a `_reasoning_default` suffix: the 2 reasoning-on requests

Fix: non-answer guard (§4.6.2, 2026-11-11 update) and per-reply completion record (§16.35).

## 1. What was seen in the app

Clicking News Feed items ("Tell me more about this news story: …") on the OpenRouter backend
(`nvidia/nemotron-3-ultra-550b-a55b:free`) produced these replies, all from `chat_turns`:

| Turn | Reply |
|---|---|
| 1452 | `那些"` |
| 1454 | `i.ζiusfrac.\n\nto.τ` |
| 1467 | ` שח` |
| 1469 | Empty, so the user saw `_EMPTY_RESPONSE_FALLBACK` |

On every one of these turns the tools had succeeded. The IGN turn also had the full article fetched
(1,648 words). `/health` reported no `runtime_last_error`. The packaged app writes no backend log, so
`finish_reason`, token counts and the prompt were not recorded anywhere.

Turn 1471, later the same day, on `nvidia/nemotron-3.5-lightning:free`: `"Let me search for this news
story… searching for iPhone Air 2…"` and nothing more, although the search result was in its prompt.

## 2. Method

1. **Stage 1, no quota.** Run the real ControllerAgent with the News Feed context
   (`web_search_queries: [title]`, `news_article_url: url`) on a temp DB, with live localist-mcp tools
   and a capture-only runtime. This records the exact `main_dispatch` prompt.
2. **Stage 2.** Replay each captured prompt with the production payload: `stream`, `max_tokens` 1024,
   temperature 0.3, `reasoning: {"enabled": false}`. Record every raw SSE event.

Cases: the three failing articles, plus a control prompt ("Look up GTA 6 preorder sales stats.") that
had worked on OpenRouter at 09:45.

## 3. Results

| Case | Reasoning | Prompt chars | Tools in prompt | Finish | Tokens in→out | Reply |
|---|---|---|---|---|---|---|
| Verge GTA VI | off | 1,307 | NewsAPI snippet only | stop | 383→8 | `"The.\n\n\n\n\n"` |
| GSMArena iPhone | off | 1,321 | NewsAPI snippet only | stop | 429→309 | Good summary |
| IGN achievements | off | 4,706 | news + full article | stop | 1108→1 | Empty |
| **Control** web search | off | 2,084 | search + page fetch | stop | 620→1 | Empty |
| IGN achievements | default (on) | 4,706 | news + full article | stop | 1108→609 | Good summary |
| **Control** web search | default (on) | 2,084 | search + page fetch | stop | 620→4 | Empty (reasoning text was `2))`) |

Every request came back HTTP 200 from provider `Nvidia`, with `finish_reason` and
`native_finish_reason` both `stop`, `[DONE]` received, and no `error` event.

## 4. Conclusions

- **The model failed, not the prompt or the app.** 4 of 6 requests were junk or empty, including the
  control. The prompts were ordinary and shorter than ones that succeeded. Localist's SSE parsing
  matches the raw events.
- **The News Feed path isn't special.** It only looked that way because those turns happened to land
  in a bad stretch.
- **Reasoning on vs. off: inconclusive.** It fixed one failure and not the other (N=2).
- **The failure is silent.** A junk reply ends exactly like a good one, so only content inspection can
  catch it. That is the motivation for `classify_non_answer()`.
- **Side finding, not fixed here.** For the Verge item, NewsAPI's pinned-URL match returned a Slashdot
  aggregator copy, so the URL fetch never got the article. That's why oMLX also answered that it
  couldn't find the story.

## 5. Later the same day (guard live smokes)

The Phase 1 live smoke (real ControllerAgent with `OpenRouterRuntimeClient`, temp DB) found Ultra
producing different junk:
- 1,000+ characters of token soup: `")))-Re-Re-Re…"`, `"E(P\n\n(PAN\n\nThe%%%%…"`
- short soup: `")V)ide(CA)())Document)Das)"`

That led to the `degenerate` rule. Final smoke:
- Ultra: `"11\n1"` → reset → retry → junk again → fallback, with `non_answer_recovered: false` recorded.
- Lightning: two real answers, neither flagged. Its completion record read `stop`, 429→666 tokens,
  provider Nvidia.

## 6. Cost

About 12 of the day's 50 free requests: 6 for the diagnostic and about 6 for the live smokes.
