# OpenRouter free tier — Phase 0 findings (2026-11-10)

Script: `diagnostics/diag_openrouter_free_tier.py`. Raw output:
`openrouter_free_tier_probes.json`, `openrouter_free_tier_results.csv`,
`openrouter_free_tier_availability.json`.

## 1. Gemma 4 31B free is unavailable (today)

- `google/gemma-4-31b-it:free` (sole endpoint: Google AI Studio) returned 429 on
  22/22 requests over ~12 minutes (C1: every 30 s for 10 min), with
  `limit_source: upstream_provider_shared_pool`. `gemma-4-26b-a4b-it:free` same.
- 429s do **not** consume `free_model_daily_requests` (0/50 used after all of them).
- Decision (user, 2026-11-10): default free model is
  `nvidia/nemotron-3-ultra-550b-a55b:free`; Gemma stays selectable. Re-check Gemma
  with `--part c` on another day.

## 2. Nemotron 3 Ultra free works end to end

| Probe | Result |
|---|---|
| B2 endpoint | Nvidia only; 1M context, 65,536 max output, uptime 100% (1 d); tools + reasoning params |
| B3 reasoning | default: reasoning field present, 2.9 s. `{"enabled": false}`: no reasoning, 1.4 s, `reasoning_tokens` 0. `{"exclude": true}`: hides it but the model still reasons (1.8 s). → use `enabled:false` |
| B4 streaming | clean; TTFT 0.49 s; final chunk carries `usage` (incl. `cost: 0`) |
| B5 `data_collection: deny` | **404** "No endpoints found matching your data policy (Free model training)". The only free endpoint may train on prompts. Matches the user's decision: allow, disclose in Settings |
| B6 native tools | correct `tool_calls`, `finish_reason: tool_calls` (Localist doesn't use native tools today) |

## 3. Requests per turn (Part A, real ControllerAgent, temp DB)

| Turn | Calls | Labels |
|---|---|---|
| chit-chat | 2 | main_dispatch, working_state |
| explicit memory | 2 | explicit extraction, main_dispatch |
| implicit fact | 2 | main_dispatch, working_state (implicit gate didn't fire) |
| factual | 2 | main_dispatch, working_state |
| web search | 3 | query/classifier, main_dispatch, working_state |
| follow-up | 2 | main_dispatch, working_state |

- **Mean 2.17, max 3 requests per turn; +1 per startup/backend switch (cache warm-up).**
- 50/day ≈ 20–23 turns/day; 1,000/day (after a one-time $10 purchase) ≈ 450.
- No 429s, no empty completions, no errors across 14 calls. Main answers: TTFT
  0.7–1.9 s, ~30–40 tokens/s. Explicit memory ("metric units") was honoured on the
  next turn; local Ollama embeddings (`nomic-embed-text`) worked alongside the
  cloud chat model, which validates Phase 1's design.

## 4. Implications for Phases 2–4

- **SSE must be decoded as UTF-8 explicitly.** OpenRouter's `text/event-stream`
  has no charset, so `requests`' `iter_lines(decode_unicode=True)` falls back to
  ISO-8859-1 ("–" arrived as "â\x80\x93"). `omlx_runtime_client.py:74` uses the same
  pattern: check whether oMLX is affected when the SSE helper is shared.
- Send `reasoning: {"enabled": false}` (the equivalent of Ollama's `think:false`).
  With default reasoning and a tiny `max_tokens`, most free models returned empty
  content (C2).
- Distinguish 429 kinds by `error.metadata.limit_source`: an
  `upstream_provider_shared_pool` 429 means "model busy upstream"; a 429 or 402
  on the account means "daily free limit reached". Show different banners. No
  `Retry-After` header was sent.
- `GET /api/v1/key` returns `free_model_daily_requests {used, limit, remaining}`
  and `is_free_tier`, so Settings can show "N left today" without counting locally.
  The counter updates with some lag.
- A key can carry its own `limit` (this one: $10 weekly). Paid models would hit it
  with a 402 or 403.
- One free model (`thinkingmachines/inkling-2-small:free`) returned 403. The model
  picker must cope with listed-but-refused models.
