# Plan: Retire MLX EmbeddingGemma (Both as a Dependency and as a Config Option)

## Context (read this first — you have no memory of the session that produced this plan)

This is the Localist Framework repo (`lora-app-demo`). Read `CLAUDE.md` and `LOCALIST-Architecture.md`
first, per repo convention.

**Decision already made (not up for debate in this plan):** MLX EmbeddingGemma
(`mlx-community/embeddinggemma-300m-4bit`, loaded via `mlx_embeddings`) is being retired — both as a
packaging dependency (`backend/pyproject.toml`'s `[mlx]` extra) and as a user-facing embedding
option (`EmbeddingEngine`, `LOCALIST_EMBEDDING_ENGINE_ENABLED`). This was already reflected in
`README.md` in a prior session (embeddings are now documented as "Ollama-served, or zero-config
BM25 keyword-only" — no MLX option). **This plan is the follow-up: actually remove the code.**

**Why this matters more than a routine dependency removal:** `planner.py`'s entire hand-tuned
semantic-gating threshold system (`_SEMANTIC_GATE_THRESHOLDS`, `_RESEARCH_INTENT_THRESHOLD`,
`_EPISODIC_RELEVANCE_THRESHOLD`, and the whole five-... now four-tier `resolve_gate_tiers()`
resolution built across the last several sessions — see `PLAN_semantic_gating_calibration.md`) was
calibrated against `_TUNED_EMBEDDING_MODEL = "mlx-community/embeddinggemma-300m-4bit"` specifically.
Deleting the model support without deciding what happens to the "tuned" tier would leave dead,
unreachable code and stale comments referencing a model no longer used. **§2 below is the one real
design decision this plan requires — everything else is mechanical removal.**

## Full inventory (file:line, from a dedicated research pass — verify line numbers before editing,
## they will have drifted)

### 1. `backend/pyproject.toml`
- `:36-42` — the `[mlx]` extra: `mlx-embeddings>=0.1.0; sys_platform == 'darwin' and platform_machine == 'arm64'`. Only package in it.
- No other extra (`dev`, `packaging`, `ocr`, `chart`) references `mlx`.

### 2. `backend/src/localist/embedding_engine.py`
- Entire 258-line file is single-purpose (MLX EmbeddingGemma wrapper) — **delete the whole file**, nothing inside it is reused elsewhere.
- Key constants for cross-reference while removing callers: `_DEFAULT_MODEL = "mlx-community/embeddinggemma-300m-4bit"` (:62), `_EXPECTED_DIM = 768` (:65), class `EmbeddingEngine` (:73+) with `.embed()`, `.available`, `.embed_fn` property, `._load()`.

### 3. `backend/src/localist/main.py` — highest mechanical risk
- `:164` — `from .embedding_engine import EmbeddingEngine` (delete import).
- `:109-133` — module docstring block documenting the three-tier precedence including `EmbeddingEngine` — rewrite to two-tier (runtime-backend embed vs. keyword-only).
- `:197-198`, `:211` — comments referencing EmbeddingEngine.
- `:242` — `Settings.embedding_engine_enabled: bool = False` (`LOCALIST_EMBEDDING_ENGINE_ENABLED`) — delete field. `SettingsConfigDict(..., extra="ignore")` means an existing user `.env` with this key set will just be silently ignored post-upgrade — confirmed safe, not a startup crash, but see §3 of Open Decisions re: whether that silent downgrade deserves a log line.
- `:237-241` — comment block for that field — delete.
- `:270` — `AppState.embedding_engine: EmbeddingEngine | None = None` — delete field and every read/write of `_state.embedding_engine`.
- `:271-274` — comment referencing `planner.py`'s tuned-model guard — will need rewriting per §2's resolution.
- `_configure_embedding_source()` (~`:292-361`): delete the tier-2 MLX branch (`:336-345`, `if settings.embedding_engine_enabled and is_apple_silicon: embedding_engine = EmbeddingEngine(); ...`) and its Apple-Silicon-skip branch (`:347-354`). What remains: tier 1 (runtime-backend embed) → tier 2 (was tier 3) keyword-only.
- `_derive_active_embedding_model_name()` (~`:362-386`): delete the `if embedding_engine is not None: ...` branch (`:383-384`).
- Lifespan wiring (`:650-657`): delete the `embedding_engine` half of the `_configure_embedding_source()` return-tuple unpacking and the `_state.embedding_engine = embedding_engine` assignment. **Check whether `_configure_embedding_source()`'s return signature changes from a 2-tuple to something else** — if `embedding_engine` was only ever used for the `_state.embedding_engine` assignment and naming, the function can just return `embed_fn` alone (or keep the tuple shape for now if `_derive_active_embedding_model_name()` still needs a second value — check its actual signature before deciding).
- `:861` — comment "EmbeddingEngine fallback naming" — delete/rewrite.
- `GET /health` (~`:1584-1592`): delete the `embedding_engine = _state.embedding_engine; embed_available = embedding_engine is not None and embedding_engine.available` branch — `embed_available` becomes purely `settings.embedding_model and raw.get("embed_model_found")`.
- `POST /settings/embedding-model` (`set_embedding_model`, ~`:1875-1962`): the "clearing tier 1 falls back to EmbeddingEngine" branch (`:1937-1962`) must become "clearing tier 1 falls back to keyword-only" — i.e. clearing the model just sets `embed_fn = None` unconditionally, no fallback tier left to check. Rewrite the docstring at `:1875-1880` accordingly.

### 4. `backend/src/localist/planner.py` — the design decision (see §2 Open Decisions below for the actual choice)
- `:803` — `_TUNED_EMBEDDING_MODEL: str = "mlx-community/embeddinggemma-300m-4bit"`.
- `:780-802` — "Tuned-embedding-model guard" comment block (cites the 0.7119-vs-0.578 mismatch diagnostic).
- `:725` — `_SEMANTIC_GATE_THRESHOLDS` dict (tuned-tier defaults for explicit_search_action/lookup_request).
- `:945` — `_RESEARCH_INTENT_THRESHOLD: float = 0.65` (tuned-tier default).
- `:777` — `_EPISODIC_RELEVANCE_THRESHOLD: float = 0.70` (tuned-tier default).
- `:869-876` — `_VALIDATED_MODEL_THRESHOLDS` dict — currently one entry, `nomic-embed-text:latest`. **This survives and likely becomes the new top tier** (see §2).
- `:1197-1198` — `resolve_gate_tiers()`'s load-bearing line: `if embedding_model_name is None or embedding_model_name == _TUNED_EMBEDDING_MODEL: return {name: "tuned" for name in _GATE_NAMES}`. This is the exact line that needs to change per §2's decision.
- `Planner.__init__` (~`:1219-1400`): docstring at `:1267-1284` explaining the guard; `:1322-1324` sets defaults to the tuned-tier constants; `:1332` calls `resolve_gate_tiers()`; `:1334-1383` is the per-gate override branch, currently gated by `if embedding_model_name is not None and embedding_model_name != _TUNED_EMBEDDING_MODEL:` — this condition's shape will also change per §2.
- Comment cross-references at `:1977-1981`, `:2202`, `:2295` (lexical-fallback / gate-naming context, not functional, but reference the tuned/None relationship — update wording after §2 lands).

### 5. `start_localist.sh`
- `:11` — cosmetic mention of "MLX-LM" in an engine list (leave — refers to oMLX chat backend, unrelated).
- `:67-96` — the entire first-run interactive prompt block ("First-run prompt: local embedding model (EmbeddingGemma via MLX)") — gating conditions, the `import mlx_embeddings` probe, the `y/N` prompt, writing `LOCALIST_EMBEDDING_ENGINE_ENABLED` to `.env`. **Delete this whole block.**

### 6. Documentation
- `THIRD_PARTY_LICENSES.md`: `:21` (mlx-embeddings table row), `:25-30` (kept-status rationale), `:105-110` (mlx-family package table — note `mlx`/`mlx-lm`/`mlx-metal`/`mlx-audio`/`mlx-vlm` are incidental to the dev venv from other tools, not this repo's own dependency — only the `mlx-embeddings` row is this repo's concern), `:167` (transitive-deps note), `:194-208` (the Gemma Terms of Use section for the downloaded model weights), `:214` (a stale open item referencing a README line number that's since moved — re-verify against current README.md, may already be resolved).
- `CLAUDE.md`: `:10` (engine list, leave — oMLX/MLX-LM chat backend, not embeddings), `:20` (install example — drop `mlx` from the extras list), `:24,27` (extras description paragraph), `:79-94` (the dense three-tier-precedence paragraph — full rewrite to two-tier, matching whatever main.py's new precedence prose says).
- `docs/architecture/16-runtime-backend-layer.md`: §16.4 (`:204-308` roughly) is the *current-state* description and needs a real rewrite (not just an appended changelog note) since it describes the mechanism as it exists today, not a historical event — see its own file's convention (CLAUDE.md: "Edits to substance go in the individual section file... update its row in LOCALIST-Architecture.md"). Add a new dated §16.18 (or next available number) entry documenting the retirement itself (what was removed, why, what replaced the tuned tier per §2), following the file's own append-only-changelog-entries-but-rewrite-current-state-sections convention already established by past sessions. Leave §16.14/§16.15/§16.17's own historical changelog prose untouched (they're dated historical record, not current-state description) — only fix line/section cross-references if any now point at deleted code.
- `LOCALIST-Architecture.md`: append a new dated summary to §16's index row (its own established convention — never edit prior dated entries, prepend a new one), bump the last-updated date.

### 7. Tests — the largest mechanical surface, budget real time here
- `backend/tests/test_main_embedding_source_selection.py` — dedicated to `_configure_embedding_source()`'s three tiers (`TestEmbeddingEngineTier` etc., ~170+ lines). **Needs a real rewrite**, not a patch: delete the EmbeddingEngine-tier test class entirely, keep/adapt the runtime-backend-tier and keyword-only-tier tests for the new two-tier shape.
- `backend/tests/test_planner_phase3.py` — ~7+ module-level hits of `_TUNED_EMBEDDING_MODEL`/`_SEMANTIC_GATE_THRESHOLDS`/`embeddinggemma`, but many more individual assertions depend on them via shared fixtures/helper functions (`TestTunedEmbeddingModelGuard`, `TestValidatedModelThresholds`, `TestCalibratedModelThresholds`, `TestLexicalFallbackTier` — all built across `PLAN_semantic_gating_calibration.md`'s sessions). **This whole file needs a careful pass**: `TestTunedEmbeddingModelGuard`'s very premise (a model that equals `_TUNED_EMBEDDING_MODEL` gets special treatment) goes away entirely per §2's resolution — likely delete that test class outright and confirm its scenarios are still covered by whatever the new top tier's test class already asserts.
- `backend/tests/test_main_embedding_model_switch.py` — `:212` hardcodes `fake_engine.model_path = "mlx-community/embeddinggemma-300m-4bit"`; `:252-254` comments re: EmbeddingEngine fallback in the switch endpoint — the "clearing falls back to EmbeddingEngine" test scenario needs to become "clearing falls back to keyword-only" (or be deleted if that scenario no longer exists post-removal).
- `backend/tests/test_main_health_gate_tiers.py` — imports `_TUNED_EMBEDDING_MODEL`, asserts `active_embedding_model_name == _TUNED_EMBEDDING_MODEL` and `gate_tiers` all `"tuned"` for it — rewrite per §2.
- `backend/tests/test_embedding_provenance.py`, `test_chat_turns_semantic_search.py` — both have a local `_TUNED = "mlx-community/embeddinggemma-300m-4bit"` constant used as "some real embedding model name" fixture data — likely fine to just repoint this constant at `nomic-embed-text:latest` (or a synthetic fake name) rather than rewrite the tests' actual logic, **provided the test doesn't depend on tuned-tier-specific behavior** (check case by case).
- `backend/tests/test_main_chat_turns_semantic_endpoint.py`, `test_main_memory_reembed.py` — several hardcoded `embedding_model_name="mlx-community/embeddinggemma-300m-4bit"` fixture values — same "just repoint the constant" treatment likely applies.
- `backend/tests/test_build_graph.py`, `test_wiki_doc.py` — EmbeddingGemma appears only inside **sample wiki-page fixture content** (e.g. a sample doc mentioning "Localist Software Stack Overview"), not functional test logic — low priority, but worth a pass so fixture content doesn't describe a retired feature as current.
- `backend/tests/test_mcp_server.py`, `test_memory_phase1.py` — passing comment references only, no action needed.
- **Do not guess at exact line counts before starting** — re-grep at execution time; the estimate from this plan's research pass was "~12 files, 40-60+ individual references," which undercounts assertions inside parametrized/shared-fixture test classes.

### 8. `backend/packaging/`
- Already clean — the frozen PyInstaller build is base-only and already excludes MLX (confirmed via `localist-backend.spec` comments and `packaging/README.md`). No spec changes needed. Deleting `embedding_engine.py` will simply remove a residual "missing module named mlx_embeddings (delayed, optional)" line from the build's own warning log — cosmetic, no action required beyond the source deletion itself.

### 9. Frontend (`localist-ui/`)
- `localist-ui/src/routes/settings/+page.svelte:556` — a comment near the Embedding Model card explaining Ollama-as-alternative-to-MLX — update wording (no longer "alternative to", since MLX is gone, Ollama is just *the* local option).
- `localist-ui/src/routes/settings/+page.svelte:683` — **a real, user-facing status string**: `'Cosine-similarity retrieval active (mlx-community/embeddinggemma-300m-4bit).'`, shown when the (soon-to-not-exist) tuned model is active. This needs a real rewrite tied to §2's resolution — likely becomes generic ("Cosine-similarity retrieval active ({model}).") reading the actual active model name rather than a hardcoded string.
- Generated/build output (`.svelte-kit/output/`, `build/`) will self-regenerate on next `npm run build` — no manual edit.

## 2. The one real design decision: what replaces the "tuned" tier?

Today's five-tier resolution (from `PLAN_semantic_gating_calibration.md`) is: **tuned → validated →
auto-calibrated → lexical-fallback → disabled**, resolved per gate. "Tuned" means the original,
hand-picked constants (`_SEMANTIC_GATE_THRESHOLDS` etc.) computed directly against
`mlx-community/embeddinggemma-300m-4bit` before the multi-model system existed at all. "Validated"
means a per-model entry in the newer `_VALIDATED_MODEL_THRESHOLDS` dict — same rigor (hand-reviewed
via a real diagnostic run), just a different, more general storage shape that came later. They were
never meaningfully different in trust level, only in when they were added to the codebase.

**Recommendation: delete the "tuned" tier outright, don't replace it with a new pinned model.**

Collapse to four tiers: **validated → auto-calibrated → lexical-fallback → disabled**. Concretely:

- Delete `_TUNED_EMBEDDING_MODEL`, `_SEMANTIC_GATE_THRESHOLDS`, `_RESEARCH_INTENT_THRESHOLD`,
  `_EPISODIC_RELEVANCE_THRESHOLD` as module constants — their specific numeric values were measured
  against a model that no longer exists in this codebase, so they have no ongoing meaning to
  preserve.
- `_VALIDATED_MODEL_THRESHOLDS` already has a real entry (`nomic-embed-text:latest`) — it becomes
  the top tier as-is, no new diagnostic work required.
- `resolve_gate_tiers()`'s `:1197-1198` special-case line goes away entirely — `embedding_model_name
  is None` (true keyword-only) flows through the *same* general per-gate loop as any other
  non-matching name, which — since `_VALIDATED_MODEL_THRESHOLDS.get(None, {})` is naturally empty —
  correctly resolves to `"lexical-fallback"` for all four gates (assuming `_LEXICAL_FALLBACK_
  THRESHOLDS` still covers all four, which it does per the prior session's work).
- **This is also a real, independent bug fix, not just a refactor**: today, a true keyword-only
  session gets labeled `"tuned"` by `resolve_gate_tiers()`, which the Settings UI's trust-badge
  logic (`GATE_TIER_LABEL`/`GATE_TIER_BADGE_CLASS` in `+page.svelte`) renders as a green
  "hand-validated" badge — actively misleading, since no cosine scoring ever ran at all in that
  case. Collapsing to four tiers makes a keyword-only session correctly show "keyword fallback"
  instead. Flag this explicitly in the commit/changelog — it's a nice side benefit, not an
  incidental one.

**Alternative considered and not recommended:** pick a new model to be the pinned "tuned" baseline
(e.g. promote `nomic-embed-text:latest` from `_VALIDATED_MODEL_THRESHOLDS` into a renamed
`_TUNED_EMBEDDING_MODEL` constant, or run fresh diagnostics against some other model). Rejected
because: (a) it requires new data-collection/diagnostic work this plan's scope doesn't include, (b)
it delays the actual retirement for no material trust improvement — "validated" already carries the
same review rigor "tuned" did, just via a dict entry instead of a bare module constant, (c) it
reintroduces exactly the kind of "assume one canonical model" framing this multi-session project has
been deliberately moving away from.

**Confirm before executing:** if this recommendation is wrong (e.g. there's a reason to keep a
pinned zero-config "tuned" baseline independent of `_VALIDATED_MODEL_THRESHOLDS`), everything in §3
of `planner.py`'s inventory above changes shape — resolve this first, before touching any code.

## 3. Suggested execution order

1. **Confirm §2's decision** (or override it) before writing any code.
2. `planner.py` — implement §2's resolution. This is the file everything else's tests key off of;
   get it right and stable first, with `test_planner_phase3.py` updated alongside it (not after) so
   there's a passing baseline before touching `main.py`.
3. `main.py` — collapse `_configure_embedding_source()`/`_derive_active_embedding_model_name()` to
   two tiers, remove `Settings.embedding_engine_enabled`/`AppState.embedding_engine`, update the
   `GET /health` and `POST /settings/embedding-model` handlers. Update
   `test_main_embedding_source_selection.py`, `test_main_embedding_model_switch.py`,
   `test_main_health_gate_tiers.py` alongside.
4. Delete `backend/src/localist/embedding_engine.py`.
5. `backend/pyproject.toml` — remove the `[mlx]` extra.
6. `start_localist.sh` — remove the first-run MLX prompt block.
7. Sweep the remaining test files with hardcoded `embeddinggemma`/`_TUNED_EMBEDDING_MODEL` fixture
   values (§7's list) — repoint or delete per the case-by-case judgment noted there.
8. Frontend: `+page.svelte:683`'s hardcoded status string, `:556`'s comment.
9. Run the full backend suite (`cd backend && source .venv/bin/activate && python -m pytest tests/
   -v`) and `cd localist-ui && npm run check` — both must be clean before docs.
10. Docs last, once the code is settled and the actual final shape is known: `THIRD_PARTY_LICENSES.md`,
    `CLAUDE.md`, `docs/architecture/16-runtime-backend-layer.md` (§16.4 rewrite + new dated entry),
    `LOCALIST-Architecture.md` index row.
11. Rebuild the frozen PyInstaller backend (`cd backend/packaging && source ../.venv-packaging/bin/activate
    && pyinstaller -y localist-backend.spec`) if there's a running desktop dev app to verify against —
    same process used in prior sessions.

## Open decisions to confirm before/during building

1. **§2 above** — collapse to four tiers (recommended) vs. pin a new "tuned" model. The single
   highest-leverage decision in this whole plan.
2. **`_configure_embedding_source()`'s return shape** — currently returns `(embed_fn,
   embedding_engine)`; once tier 2 is deleted, confirm whether `_derive_active_embedding_model_name()`
   and the lifespan wiring still need two return values or can collapse to just `embed_fn`. Don't
   assume — check every call site first (main.py has at least 4: `lifespan()`, and both live-switch
   endpoints per `PLAN_semantic_gating_calibration.md`'s own inventory of `_build_controller()` call
   sites).
3. **Existing users' `.env` with `LOCALIST_EMBEDDING_ENGINE_ENABLED=true`** — confirmed safe at the
   settings-parsing level (`extra="ignore"`), but silently downgrades anyone who had it enabled to
   keyword-only (or their runtime-backend embed, if separately configured) with no visible warning.
   Decide whether `main.py`'s startup path should log a one-time warning when it detects this now-dead
   env var still set, so an upgrading user isn't confused about why retrieval quality changed. Not
   required, but cheap and user-friendly.
4. **Frontend status string** (`+page.svelte:683`) — confirm the replacement wording once §2 is
   settled; it currently hardcodes the tuned model's name specifically, so its replacement shape
   depends on whether there's still a "top tier" concept worth calling out by name or whether it
   should just report whatever model is actually active generically.
