<script lang="ts">
  import { isTauri } from '@tauri-apps/api/core';
  import { APP_VERSION } from '$lib/version';
  import { onMount, tick } from 'svelte';
  import { health, checkHealth, backendReady } from '$lib/stores/server';
  import { apiUrl } from '$lib/api';
  import { modelConfig, RUNTIME_BACKENDS, RUNTIME_BACKEND_LABELS, type RuntimeBackend } from '$lib/stores/model';
  import {
    runtimeBackendSwitchLoading,
    runtimeBackendSwitchError,
    runtimeBackendSwitchWarning,
    switchRuntimeBackend,
    fetchBackendModels,
    pinChatModel,
    type ChatModelSource
  } from '$lib/stores/runtimeBackendSwitch';
  import {
    embeddingModelSwitchLoading,
    embeddingModelSwitchError,
    setEmbeddingModel
  } from '$lib/stores/embeddingModelSwitch';
  import { confirmDialog } from '$lib/confirmDialog';
  import BackendWaitingNote from '$lib/components/BackendWaitingNote.svelte';
  import { theme } from '$lib/stores/theme';
  import { fontSize, FONT_SIZES } from '$lib/stores/fontSize';
  import { browser } from '$app/environment';
  import {
    retentionSettings,
    retentionSettingsLoading,
    retentionSettingsError,
    loadRetentionSettings,
    setRetentionPreset,
    setEpisodeRetentionPreset,
    type EvictionPreset
  } from '$lib/stores/retentionSettings';
  import {
    reembedLoading,
    reembedError,
    reembedCorpus,
    type CalibrationResponse,
    type GateCalibrationResponse,
    type ReembedCorpusResponse,
    type ReembedStoreCounts
  } from '$lib/stores/reembedCorpus';
  import {
    newsPreferences,
    newsPreferencesLoading,
    newsPreferencesError,
    loadNewsPreferences,
    setNewsPreferences
  } from '$lib/stores/newsPreferences';
  import {
    assistantName,
    assistantNameLoading,
    assistantNameError,
    loadAssistantName,
    setAssistantName
  } from '$lib/stores/assistantName';
  import {
    githubWatchPins,
    githubWatchPinsLoading,
    githubWatchPinsError,
    loadGithubWatchPins,
    setGithubWatchPins
  } from '$lib/stores/githubWatchPins';
  import {
    apiKeysStatus,
    apiKeysLoading,
    apiKeysError,
    loadApiKeysStatus,
    setApiKeys
  } from '$lib/stores/apiKeys';

  const EVICTION_PRESETS: { value: EvictionPreset; label: string }[] = [
    { value: '7d',      label: '7 days' },
    { value: '30d',     label: '30 days' },
    { value: '90d',     label: '90 days' },
    { value: 'forever', label: 'Forever' }
  ];

  // Every loader below fetches once. Waiting for backendReady() keeps an
  // early fetch (Settings opened while the packaged app's backend sidecar is
  // still starting) from failing silently and leaving every card on its
  // defaults until the page is reopened.
  onMount(() => {
    void backendReady().then(() => {
      loadRetentionSettings();
      loadMemoryStats();
      void loadNewsPreferences().then(() => {
        homeCountryInput = $newsPreferences.home_country;
        localQueryInput  = $newsPreferences.local_query ?? '';
        selectedTopics   = $newsPreferences.topics;
      });
      void loadAssistantName().then(() => {
        assistantNameInput = $assistantName.assistant_name;
      });
      void loadGithubWatchPins().then(() => {
        pinnedRepos = $githubWatchPins.repos;
      });
      void loadApiKeysStatus();
    });
  });

  // Primitive, so it only invalidates when it actually flips — reactive
  // blocks keyed on it don't re-run on every 15s health poll. Same signal
  // backendReady() waits on: never set by the startup grace running out.
  $: backendUp = $health.ever_ok;

  // Corpus/chat-history staleness — fetched once on mount and again after a
  // re-embed completes, so the badges reflect the server's current
  // MemoryManager state without adding a dedicated poll. Episodes carry no
  // stale flag (auto-corrected at startup), so there's no badge for them.
  let corpusStale = false;
  let chatTurnsStale = false;
  let reembedResult: string | null = null;
  let reembedWarning: string | null = null;

  async function loadMemoryStats() {
    try {
      const res = await fetch(apiUrl('/api/memory/stats'));
      if (!res.ok) return;
      const data = await res.json();
      corpusStale = Boolean(data.corpus_stale);
      chatTurnsStale = Boolean(data.chat_turns_stale);
    } catch {
      // Leave last-known state — a transient fetch failure here isn't
      // worth surfacing on its own card.
    }
  }

  // Live semantic-gating threshold calibration (PLAN_semantic_gating_
  // calibration.md §2/§6) — short labels for the four gate names, matching
  // the example message format the plan specifies.
  const GATE_ABBREV: Record<string, string> = {
    explicit_search_action: 'ESA',
    lookup_request:         'LR',
    research_intent:        'RI',
    episodic_relevance:     'episodic'
  };

  function formatCalibrationSummary(calibration: CalibrationResponse | null): string {
    if (!calibration) return '';
    // A hand-validated gate overrides whatever calibration measured for it.
    const validated = Object.keys(calibration.gates)
      .filter((name) => calibration.gate_tiers?.[name] === 'validated')
      .map((name) => GATE_ABBREV[name] ?? name);
    const clean = Object.entries(calibration.gates)
      .filter(([name, g]) => !g.degenerate && g.threshold !== null && calibration.gate_tiers?.[name] !== 'validated')
      .map(([name, g]) => `${GATE_ABBREV[name] ?? name}=${g.threshold!.toFixed(2)}`);
    const recall =
      formatMeasuredThreshold('Memory recall floor', calibration.recall_floor_in_use, calibration.recall_floor) +
      formatMeasuredThreshold('Profile threshold', calibration.profile_threshold_in_use, calibration.profile_threshold);
    const handPicked = validated.length ? ` Hand-validated: ${validated.join(', ')}.` : '';
    if (clean.length) {
      return ` Recalibrated semantic gating for ${calibration.model} (${clean.join(', ')}).${handPicked}${recall}`;
    }
    if (validated.length) {
      return ` Semantic gating for ${calibration.model} uses hand-validated thresholds (${validated.join(', ')}).${recall}`;
    }
    return ` No gate calibrated cleanly for ${calibration.model} — semantic gating stays disabled for it.${recall}`;
  }

  // A per-model threshold that isn't a gate — the episodic recall floor
  // (§16.28, min score for recalling a memory) or the profile-fact threshold
  // (§3.6, min score for a profile line). Says so when a hand-validated
  // value overrides the calibrated one.
  function formatMeasuredThreshold(
    label: string,
    inUse: number | null | undefined,
    cal: GateCalibrationResponse | null | undefined
  ): string {
    if (inUse == null) return '';
    const calibrated = cal && !cal.degenerate && cal.threshold !== null ? cal.threshold : null;
    if (calibrated !== null && Math.abs(calibrated - inUse) < 1e-9) {
      return ` ${label} ${inUse.toFixed(2)}.`;
    }
    if (calibrated !== null) {
      return ` ${label} ${inUse.toFixed(2)} (hand-validated; calibration measured ${calibrated.toFixed(2)}).`;
    }
    return ` ${label} ${inUse.toFixed(2)} (calibration inconclusive).`;
  }

  function formatReembedResult(result: ReembedCorpusResponse): { summary: string; warning: string | null } {
    const stores: [string, ReembedStoreCounts][] = [
      ['Documents', {
        reembedded: result.reembedded, total: result.total,
        already_current: result.already_current, failed_ids: result.failed_ids ?? []
      }]
    ];
    if (result.chat_turns) stores.push(['Chat history', result.chat_turns]);
    if (result.episodes) stores.push(['Episodes', result.episodes]);

    // Only items without a vector for the active model are embedded (§16.26),
    // so "3 new, 38 already current" rather than "3/3".
    const describe = (s: ReembedStoreCounts) => {
      if (s.already_current === undefined) return `${s.reembedded}/${s.total}`;
      if (s.total === 0) return 'up to date';
      return `${s.reembedded} of ${s.total} embedded` +
        (s.already_current ? `, ${s.already_current} already current` : '');
    };
    const summary = stores.map(([label, s]) => `${label}: ${describe(s)}`).join(' · ') + '.';

    const failures = stores.filter(([, s]) => s.failed_ids.length > 0);
    if (!failures.length) return { summary, warning: null };
    const failedCount = failures.reduce((n, [, s]) => n + s.failed_ids.length, 0);
    const detail = failures.map(([label, s]) => `${label.toLowerCase()}: ${s.failed_ids.length}`).join(', ');
    return {
      summary,
      warning: `${failedCount} item(s) couldn't be embedded (${detail}). Check that the embedding ` +
        'model is loaded, then run Re-embed Memory again — only the remaining items are redone.'
    };
  }

  async function handleReembedCorpus() {
    const proceed = await confirmDialog(
      'Embed any memory that has no vector for the active embedding model yet? This covers ' +
      'wiki/raw documents, chat history and episodes. Items already embedded with this model ' +
      'are kept, so it can take a while only the first time a model is used on a large corpus. ' +
      'This also re-runs semantic-gating threshold calibration for the active model.'
    );
    if (!proceed) return;

    reembedResult = null;
    reembedWarning = null;
    const result = await reembedCorpus();
    if (result) {
      const { summary, warning } = formatReembedResult(result);
      reembedResult = summary + formatCalibrationSummary(result.calibration);
      reembedWarning = warning;
      await loadMemoryStats();
      await checkHealth();
    }
  }

  // Local form state — mirrors store, saved on blur/change
  let backendUrl = '';

  // Initialize from store
  $: if (browser) {
    backendUrl = localStorage.getItem('localist-backend-url') ?? 'http://127.0.0.1:8000';
  }

  function saveBackendUrl() {
    if (browser) localStorage.setItem('localist-backend-url', backendUrl);
  }

  // The backend whose models the Chat Model dropdown below is scoped to.
  // Deliberately independent of $modelConfig.backend (the real active
  // backend, synced from health polling) — clicking a segmented-control
  // button previews that backend's models immediately, even before (or
  // without) confirming an actual switch to it.
  let selectedUiBackend: RuntimeBackend = $modelConfig.backend as RuntimeBackend;
  let availableModels: string[] = [];
  let previewChatModel = '';
  let chatModelSource: ChatModelSource = 'default';
  let modelsFetchToken = 0;

  async function refreshModelsForSelected() {
    const token = ++modelsFetchToken;
    const result = await fetchBackendModels(selectedUiBackend);
    if (token !== modelsFetchToken) return;
    availableModels  = result.models;
    previewChatModel = result.chatModel;
    chatModelSource  = result.chatModelSource;
  }

  // The model the dropdown shows as selected. For the active backend this
  // is GET /health's chat_model, refreshed on every poll, so a poll-time
  // recovery (§16.22) shows up without reopening Settings. A previewed
  // backend has no health of its own, so it uses what the models endpoint
  // says it would run.
  $: chatModel =
    selectedUiBackend === $modelConfig.backend && $health.chat_model
      ? $health.chat_model
      : previewChatModel;
  $: chatModelMissing = !!chatModel && availableModels.length > 0 && !availableModels.includes(chatModel);

  $: if (selectedUiBackend && backendUp) refreshModelsForSelected();

  // selectedUiBackend starts from $modelConfig.backend, which before the
  // first health response is only the browser's last-remembered backend —
  // wrong if the backend was switched outside this UI (curl, or a restart
  // reverting to .env). Re-sync once to the real active backend when health
  // first lands, unless the user has already picked one.
  let selectedUiBackendSynced = false;
  $: if ($health.ever_ok && !selectedUiBackendSynced) {
    selectedUiBackendSynced = true;
    selectedUiBackend = $modelConfig.backend as RuntimeBackend;
  }

  async function selectRuntimeBackend(backend: RuntimeBackend) {
    selectedUiBackendSynced = true;
    selectedUiBackend = backend;
    if (backend === $modelConfig.backend) return; // already active — nothing to switch

    const proceed = await confirmDialog(
      `Switch the active runtime backend to ${RUNTIME_BACKEND_LABELS[backend]}? ` +
      `In-flight requests will keep running on the current backend unaffected, ` +
      `but this switch itself can take several seconds.`
    );
    if (!proceed) return;

    void switchRuntimeBackend(backend);
  }

  // No optimistic update: after the pin, re-read what the backend actually
  // resolves (a global override still wins over the pin), and on failure put
  // the <select> back on the model still in use.
  async function onChatModelSelectChange(e: Event) {
    const select = e.target as HTMLSelectElement;
    if (!select.value) return;
    const ok = await pinChatModel(selectedUiBackend, select.value);
    if (ok && selectedUiBackend === $modelConfig.backend) await checkHealth();
    await refreshModelsForSelected();
    await tick(); // let chatModel/chatModelMissing recompute
    select.value = chatModelMissing ? '' : chatModel;
  }

  // After a switch that kept the previous embedder (§16.25), name the
  // backend the embeddings still come from.
  $: embeddingViaOtherBackend =
    $health.active_embedding_backend && $health.active_embedding_backend !== $health.backend
      ? ` via ${RUNTIME_BACKEND_LABELS[$health.active_embedding_backend as RuntimeBackend] ?? $health.active_embedding_backend}`
      : '';

  // Embedding model — for whichever backend is actually live
  // ($modelConfig.backend), never the Chat Model card's preview-only
  // selectedUiBackend: POST /settings/embedding-model sets the active
  // backend's own embedding pin (§16.25; oMLX supported since §16.27).
  // Both backends list chat and embedding models in one catalog, so ids
  // containing "embed" are offered first, the rest after.
  let embeddingCandidates: string[] = [];
  let otherModels: string[] = [];
  let embeddingModelsFetchToken = 0;
  let embeddingModelResult: string | null = null;

  async function refreshEmbeddingModels() {
    const backend = $modelConfig.backend as RuntimeBackend;
    const token = ++embeddingModelsFetchToken;
    const { models } = await fetchBackendModels(backend);
    if (token !== embeddingModelsFetchToken) return;
    embeddingCandidates = models.filter((m) => m.toLowerCase().includes('embed'));
    otherModels = models.filter((m) => !m.toLowerCase().includes('embed'));
  }

  $: activeBackendLabel =
    RUNTIME_BACKEND_LABELS[$modelConfig.backend as RuntimeBackend] ?? $modelConfig.backend;
  // Embeddings kept from the other backend after a switch (§16.25): this
  // backend has no usable embedding model of its own yet.
  $: embeddingsFromOtherBackend =
    $health.active_embedding_backend && $health.active_embedding_backend !== $modelConfig.backend
      ? (RUNTIME_BACKEND_LABELS[$health.active_embedding_backend as RuntimeBackend] ?? $health.active_embedding_backend)
      : null;

  $: if ($modelConfig.backend && backendUp) refreshEmbeddingModels();

  async function handleEmbeddingModelPick(value: string) {
    embeddingModelResult = null;
    const body = await setEmbeddingModel(value);
    if (body) {
      embeddingModelResult = (value
        ? `Embedding model set to ${value}.`
        : 'Embedding model cleared — back to keyword-only.')
        + formatCalibrationSummary(body.calibration);
      await checkHealth();
      await loadMemoryStats();
    }
  }

  // Trust-badge tiers (PLAN_semantic_gating_calibration.md §6) — refreshed
  // on every health poll, not just transiently after a switch/reembed, so
  // the badge is correct on page load too. A model can be "validated" for
  // one gate and "disabled" for another (per-gate resolution, never a
  // single all-or-nothing verdict) — the badge shows the WORST tier across
  // all four gates, with a tooltip listing the per-gate breakdown.
  const GATE_TIER_RANK: Record<string, number> = {
    disabled: 0, 'lexical-fallback': 1, 'auto-calibrated': 2, validated: 3, tuned: 4
  };
  const GATE_TIER_LABEL: Record<string, string> = {
    tuned: 'hand-validated',
    validated: 'hand-validated',
    'auto-calibrated': 'auto-calibrated',
    'lexical-fallback': 'keyword fallback',
    disabled: 'gating disabled'
  };
  const GATE_TIER_BADGE_CLASS: Record<string, string> = {
    tuned: 'badge-success',
    validated: 'badge-success',
    'auto-calibrated': 'badge-warning',
    'lexical-fallback': 'badge-accent',
    disabled: 'badge-muted'
  };

  $: gateTierEntries = Object.entries($health.gate_tiers ?? {});
  $: worstGateTier = gateTierEntries.length
    ? gateTierEntries.reduce(
        (worst, [, tier]) => (GATE_TIER_RANK[tier] < GATE_TIER_RANK[worst] ? tier : worst),
        'tuned'
      )
    : null;
  $: gateTierTooltip = gateTierEntries
    .map(([gate, tier]) => `${GATE_ABBREV[gate] ?? gate}: ${GATE_TIER_LABEL[tier] ?? tier}`)
    .join('\n');

  function onEmbeddingModelSelectChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    void handleEmbeddingModelPick(value);
  }

  let checking = false;
  async function runHealthCheck() {
    checking = true;
    await checkHealth();
    checking = false;
  }

  // Streaming / episodic write-approval — UI-only preferences for now; no
  // backend endpoint exposes either as a live-switchable setting yet (see
  // CLAUDE_CODE_PROMPT.md's request to flag missing backend data). Persisted
  // so the toggle state survives a reload, same pattern as backendUrl above.
  let streaming = true;
  let episodicApproval = false;

  $: if (browser) {
    streaming = localStorage.getItem('localist-streaming') !== '0';
    episodicApproval = localStorage.getItem('localist-episodic-approval') === '1';
  }

  function toggleStreaming() {
    streaming = !streaming;
    if (browser) localStorage.setItem('localist-streaming', streaming ? '1' : '0');
  }

  function toggleEpisodicApproval() {
    episodicApproval = !episodicApproval;
    if (browser) localStorage.setItem('localist-episodic-approval', episodicApproval ? '1' : '0');
  }

  // Daily News Brief preferences (docs/daily-news-brief-plan.md §5) — local
  // form state mirrors the store, synced whenever it reloads, saved
  // explicitly via the Save button below (not on every keystroke/toggle,
  // since exactly-3-topics is only checkable once the user is done picking).
  let homeCountryInput = 'us';
  let localQueryInput  = '';
  let selectedTopics: string[] = [];
  let newsSaveResult: string | null = null;

  function toggleTopic(key: string) {
    newsSaveResult = null;
    if (selectedTopics.includes(key)) {
      selectedTopics = selectedTopics.filter((t) => t !== key);
    } else if (selectedTopics.length < 3) {
      selectedTopics = [...selectedTopics, key];
    }
  }

  // The backend requires exactly 3 topics on every PUT (main.py enforces
  // this — it's not a frontend-only convention), and Local area/Home
  // country are saved in that same request. So an incomplete topic
  // selection silently blocks a city-only edit too — the Save button below
  // is disabled on this condition so that's no longer discoverable only
  // after a failed click.
  $: topicsIncomplete = selectedTopics.length !== 3;

  async function handleSaveNewsPreferences() {
    newsSaveResult = null;
    if (topicsIncomplete) {
      newsPreferencesError.set('Select exactly 3 special-interest topics.');
      return;
    }
    const ok = await setNewsPreferences(
      homeCountryInput.trim().toLowerCase(),
      localQueryInput.trim() || null,
      selectedTopics
    );
    if (ok) {
      const city = $newsPreferences.local_query;
      newsSaveResult = city
        ? `Saved — Local area set to "${city}".`
        : 'Saved — Local area cleared.';
    }
  }

  // Pinned GitHub repos — release-only tracking independent of GitHub's
  // Watch/subscriptions relationship (no PR/issue email noise). Local list
  // mirrors the store, synced whenever it reloads, saved explicitly via the
  // Save button (whole-list replace, same as News Preferences' topics).
  const PINNED_REPO_RE = /^[\w.-]+\/[\w.-]+$/;
  let pinnedRepos: string[] = [];
  let pinnedRepoInput = '';
  let pinnedRepoAddError: string | null = null;
  let pinnedReposSaveResult: string | null = null;

  function addPinnedRepo() {
    pinnedReposSaveResult = null;
    const trimmed = pinnedRepoInput.trim();
    if (!trimmed) return;
    if (!PINNED_REPO_RE.test(trimmed)) {
      pinnedRepoAddError = 'Expected the form "owner/repo", e.g. ollama/ollama.';
      return;
    }
    if (pinnedRepos.some((r) => r.toLowerCase() === trimmed.toLowerCase())) {
      pinnedRepoAddError = 'Already pinned.';
      return;
    }
    if (pinnedRepos.length >= 20) {
      pinnedRepoAddError = 'Up to 20 pinned repos.';
      return;
    }
    pinnedRepos = [...pinnedRepos, trimmed];
    pinnedRepoInput = '';
    pinnedRepoAddError = null;
  }

  function removePinnedRepo(repo: string) {
    pinnedRepos = pinnedRepos.filter((r) => r !== repo);
    pinnedReposSaveResult = null;
  }

  $: pinnedReposUnchanged =
    JSON.stringify(pinnedRepos) === JSON.stringify($githubWatchPins.repos);

  async function handleSavePinnedRepos() {
    pinnedReposSaveResult = null;
    const ok = await setGithubWatchPins(pinnedRepos);
    if (ok) {
      pinnedReposSaveResult =
        pinnedRepos.length > 0
          ? `Saved — ${pinnedRepos.length} repo${pinnedRepos.length === 1 ? '' : 's'} pinned.`
          : 'Saved — no repos pinned.';
    }
  }

  // Assistant name — form state mirrors the store, saved explicitly via the
  // Save button (not on every keystroke), same convention as News Preferences.
  let assistantNameInput = 'Localist';
  let assistantNameSaveResult: string | null = null;

  $: assistantNameUnchanged =
    assistantNameInput.trim() === '' ||
    assistantNameInput.trim() === $assistantName.assistant_name;

  async function handleSaveAssistantName() {
    assistantNameSaveResult = null;
    const trimmed = assistantNameInput.trim();
    if (!trimmed) {
      assistantNameError.set('Assistant name must not be empty.');
      return;
    }
    const ok = await setAssistantName(trimmed);
    if (ok) {
      assistantNameSaveResult = `Saved — now called "${$assistantName.assistant_name}".`;
    }
  }

  // Optional API keys — inputs start blank regardless of whether a key is
  // already configured (the backend never echoes a saved value back), and
  // an empty field is left out of the PUT payload so saving the other two
  // fields can't accidentally clear one you didn't touch.
  let langsearchKeyInput = '';
  let braveKeyInput = '';
  let newsapiKeyInput = '';
  let apiKeysSaveResult: string | null = null;

  async function handleSaveApiKeys() {
    apiKeysSaveResult = null;
    const update: { langsearch_api_key?: string; brave_api_key?: string; newsapi_api_key?: string } = {};
    if (langsearchKeyInput.trim()) update.langsearch_api_key = langsearchKeyInput.trim();
    if (braveKeyInput.trim()) update.brave_api_key = braveKeyInput.trim();
    if (newsapiKeyInput.trim()) update.newsapi_api_key = newsapiKeyInput.trim();
    if (Object.keys(update).length === 0) return;

    const warning = await setApiKeys(update);
    if (warning !== null) {
      langsearchKeyInput = '';
      braveKeyInput = '';
      newsapiKeyInput = '';
      apiKeysSaveResult = warning || 'Saved.';
    }
  }
</script>

<svelte:head>
  <title>Settings — Localist</title>
</svelte:head>

<div class="settings-page">
  <div class="settings-inner">

    <BackendWaitingNote subject="settings" />

    <!-- Runtime backend -->
    <section class="settings-card">
      <div class="card-title">Runtime Backend</div>
      <div class="segmented">
        {#each RUNTIME_BACKENDS as b}
          <button
            type="button"
            class="seg-btn"
            class:active={$modelConfig.backend === b}
            class:previewing={selectedUiBackend === b && selectedUiBackend !== $modelConfig.backend}
            disabled={$runtimeBackendSwitchLoading}
            on:click={() => selectRuntimeBackend(b)}
          >{RUNTIME_BACKEND_LABELS[b]}</button>
        {/each}
        {#if $runtimeBackendSwitchLoading}
          <span class="spinner" aria-hidden="true" />
        {/if}
      </div>
      <p class="card-hint">
        Live-switches the active backend via <code>POST /settings/runtime-backend</code>
        (docs/architecture/16-runtime-backend-layer.md §16.5) — health-checked before anything
        changes, and persisted to <code>.env</code> so it survives a restart. Switching takes a
        few seconds (mostly model-load time on the new backend); in-flight requests keep running
        on the backend that was active when they started.
      </p>
      {#if $runtimeBackendSwitchWarning}
        <p class="card-hint" style="color:var(--warning)">{$runtimeBackendSwitchWarning}</p>
      {/if}
      {#if $runtimeBackendSwitchError}
        <p class="card-hint" style="color:var(--error)">{$runtimeBackendSwitchError}</p>
      {/if}
    </section>

    <!-- Memory re-embed -->
    <section class="settings-card">
      <div class="card-title">Re-embed Memory</div>
      <p class="card-hint">
        Embeds wiki/raw documents, chat history and episodes that have no vector for the active
        embedding model yet, via <code>POST /memory/reembed</code> (docs/architecture/16-runtime-backend-layer.md §16.26).
        Vectors are kept per model, so switching back to a model you've used before needs nothing;
        this is only needed when switching to a model that hasn't embedded everything yet — until
        then, document retrieval and chat-history search fall back to keyword-only ranking. Items that fail
        are retried automatically; any still failing are reported below, and running it again
        redoes only those. Also re-runs semantic-gating threshold calibration for the active
        model (see the Embedding Model card's trust badge below) — useful to retry a model that
        calibrated poorly, or after the calibration method itself improves.
      </p>
      {#if corpusStale || chatTurnsStale}
        <div class="stale-badges">
          {#if corpusStale}
            <span class="badge badge-warning">
              Document embeddings out of date — re-ranking is running keyword-only.
            </span>
          {/if}
          {#if chatTurnsStale}
            <span class="badge badge-warning">
              Chat history embeddings out of date — history search is running keyword-only.
            </span>
          {/if}
        </div>
      {/if}
      <div class="field-row">
        <button
          type="button"
          class="btn-secondary"
          disabled={$reembedLoading}
          on:click={handleReembedCorpus}
        >
          {#if $reembedLoading}
            <span class="spinner" aria-hidden="true" />
            Re-embedding…
          {:else}
            Re-embed Memory Now
          {/if}
        </button>
      </div>
      {#if reembedResult}
        <p class="card-hint">{reembedResult}</p>
      {/if}
      {#if reembedWarning}
        <p class="card-hint" style="color:var(--warning)">{reembedWarning}</p>
      {/if}
      {#if $reembedError}
        <p class="card-hint" style="color:var(--error)">{$reembedError}</p>
      {/if}
    </section>

    <!-- Chat model -->
    <section class="settings-card">
      <div class="card-title">Chat Model — {RUNTIME_BACKEND_LABELS[selectedUiBackend]}</div>
      <div class="field-group">
        <select
          id="chat-model"
          aria-label={`Chat model for ${RUNTIME_BACKEND_LABELS[selectedUiBackend]}`}
          class="settings-input"
          disabled={$runtimeBackendSwitchLoading || availableModels.length === 0}
          on:change={onChatModelSelectChange}
        >
          {#if chatModelMissing}
            <option value="" disabled selected>{chatModel} (not found on this backend)</option>
          {:else}
            <option value="" disabled selected={!chatModel}>
              {availableModels.length ? 'Select a model…' : 'No models reported by this backend'}
            </option>
          {/if}
          {#each availableModels as m}
            <option value={m} selected={m === chatModel}>{m}</option>
          {/each}
        </select>
        {#if chatModelSource === 'override'}
          <p class="field-hint" style="color:var(--warning)">
            <code>LOCALIST_CHAT_MODEL</code> in <code>.env</code> overrides the model picked here for every
            backend. Remove that line and restart to use per-backend choices.
          </p>
        {/if}
        {#if selectedUiBackend === $modelConfig.backend}
          <p class="field-hint">Applies immediately — {RUNTIME_BACKEND_LABELS[selectedUiBackend]} is the active backend.</p>
        {:else}
          <p class="field-hint">Saved for {RUNTIME_BACKEND_LABELS[selectedUiBackend]} — takes effect the next time you switch to it.</p>
        {/if}
      </div>
    </section>

    <!-- Embedding model, for the active backend (§16.25, §16.27) -->
    <section class="settings-card">
      <div class="card-title">
        Embedding Model — {activeBackendLabel}
        {#if worstGateTier}
          <span
            class="badge {GATE_TIER_BADGE_CLASS[worstGateTier]}"
            title={gateTierTooltip}
          >
            {GATE_TIER_LABEL[worstGateTier]}
          </span>
        {/if}
      </div>
      <div class="field-group">
        <select
          id="embedding-model"
          aria-label={`Embedding model for ${activeBackendLabel}`}
          class="settings-input"
          disabled={$embeddingModelSwitchLoading}
          on:change={onEmbeddingModelSelectChange}
        >
          <option value="" selected={!$health.embedding_model}>
            None — keyword-only fallback
          </option>
          {#if embeddingCandidates.length}
            <optgroup label="Embedding models">
              {#each embeddingCandidates as m}
                <option value={m} selected={m === $health.embedding_model}>{m}</option>
              {/each}
            </optgroup>
          {/if}
          {#if otherModels.length}
            <optgroup label="Other models">
              {#each otherModels as m}
                <option value={m} selected={m === $health.embedding_model}>{m}</option>
              {/each}
            </optgroup>
          {/if}
        </select>
        {#if embeddingsFromOtherBackend}
          <p class="field-hint" style="color:var(--warning)">
            Embeddings still come from {embeddingsFromOtherBackend}
            ({$health.active_embedding_model_name}) — pick a model here to use {activeBackendLabel}'s own.
          </p>
        {/if}
        <p class="field-hint">
          Routes embeddings through {activeBackendLabel}'s
          <code>{$modelConfig.backend === 'omlx' ? '/v1/embeddings' : '/api/embed'}</code> via
          <code>POST /settings/embedding-model</code>; leave it unset to run in keyword-only (BM25)
          mode instead. Saved for {activeBackendLabel} only — each backend keeps its own choice, and
          switching backends switches to it. Vectors are kept per model, so going back to a model
          that has already embedded everything needs no re-embed; a model that hasn't leaves
          documents and chat history keyword-only until "Re-embed Memory Now" fills the gaps
          (episodes fill themselves). A model used for the first time is calibrated for semantic
          gating automatically. The badge shows the least-trusted tier across the four gates it
          drives — hover for the per-gate breakdown.
        </p>
      </div>
      {#if $embeddingModelSwitchLoading}
        <span class="spinner" aria-hidden="true" />
      {/if}
      {#if embeddingModelResult}
        <p class="card-hint">{embeddingModelResult}</p>
      {/if}
      {#if $embeddingModelSwitchError}
        <p class="card-hint" style="color:var(--error)">{$embeddingModelSwitchError}</p>
      {/if}
    </section>

    <!-- Streaming / episodic approval toggles -->
    <section class="settings-card row">
      <div class="card-title">Streaming responses</div>
      <button
        type="button"
        class="switch"
        class:on={streaming}
        aria-label="Toggle streaming responses"
        on:click={toggleStreaming}
      ><span class="switch-knob" /></button>
    </section>

    <section class="settings-card row">
      <div class="card-title">Episodic write-approval</div>
      <button
        type="button"
        class="switch"
        class:on={episodicApproval}
        aria-label="Toggle episodic write-approval"
        on:click={toggleEpisodicApproval}
      ><span class="switch-knob" /></button>
    </section>

    <!-- Appearance -->
    <section class="settings-card row">
      <div class="card-title">Theme — currently <strong>{$theme}</strong></div>
      <button
        type="button"
        class="switch"
        class:on={$theme === 'light'}
        aria-label="Toggle light/dark theme"
        on:click={theme.toggle}
      ><span class="switch-knob" /></button>
    </section>

    <!-- Font size -->
    <section class="settings-card">
      <div class="card-title">Font Size</div>
      <p class="card-desc">Scales text throughout the app. Some UI chrome and labels are a fixed size and won't change.</p>
      <div class="segmented">
        {#each FONT_SIZES as f}
          <button
            type="button"
            class="seg-btn"
            class:active={$fontSize === f.value}
            on:click={() => fontSize.set(f.value)}
          >{f.label}</button>
        {/each}
      </div>
    </section>

    <!-- Backend connection -->
    <section class="settings-card">
      <div class="card-title">Backend Connection</div>
      <p class="card-desc">Configure the FastAPI server that the UI connects to.</p>

      <div class="field-group">
        <label class="field-label" for="backend-url">Server URL</label>
        <div class="field-row">
          <input
            id="backend-url"
            type="text"
            class="settings-input"
            bind:value={backendUrl}
            on:blur={saveBackendUrl}
            placeholder="http://127.0.0.1:8000"
            spellcheck="false"
          />
          <button
            class="btn-secondary"
            on:click={runHealthCheck}
            disabled={checking}
            aria-label="Test connection"
          >
            {#if checking}
              <span class="spinner" aria-hidden="true" />
              Checking…
            {:else}
              Test connection
            {/if}
          </button>
        </div>

        <div class="health-readout">
          <div class="health-row">
            <span class="health-label">Status</span>
            <span
              class="badge"
              class:badge-success={$health.healthy}
              class:badge-error={!$health.reachable && !$health.checking}
              class:badge-warning={$health.reachable && !$health.healthy}
              class:badge-muted={$health.checking}
            >
              {#if $health.checking}checking…
              {:else if $health.healthy}online
              {:else if $health.reachable}degraded
              {:else}offline{/if}
            </span>
          </div>
          <div class="health-row">
            <span class="health-label">Embeddings</span>
            <span
              class="badge"
              class:badge-success={$health.embed_model_found}
              class:badge-warning={!$health.embed_model_found}
              title={$health.embed_model_found
                ? `Cosine-similarity retrieval active (${$health.active_embedding_model_name ?? $health.embedding_model}${embeddingViaOtherBackend}).`
                : 'Running in keyword-only fallback — check backend logs for the load error.'}
            >
              {$health.embed_model_found ? 'ready' : 'keyword-only fallback'}
            </span>
          </div>
          {#if $health.base_url}
            <div class="health-row">
              <span class="health-label">Resolved URL</span>
              <span class="health-value text-mono">{$health.base_url}</span>
            </div>
          {/if}
          {#if $health.models.length > 0}
            <div class="health-row">
              <span class="health-label">Available models</span>
              <div class="model-chips">
                {#each $health.models as m}
                  <span class="badge badge-muted">{m}</span>
                {/each}
              </div>
            </div>
          {/if}
          {#if $health.error}
            <div class="health-row error-row">
              <span class="health-label">Error</span>
              <span class="health-value text-mono" style="color:var(--error)">{$health.error}</span>
            </div>
          {/if}
        </div>
      </div>
    </section>

    <!-- Assistant name -->
    <section class="settings-card">
      <div class="card-title">Assistant Name</div>
      <p class="card-desc">
        What the assistant calls itself in chat — updates the system prompt
        and persona on save; takes effect on your very next message.
      </p>

      <label class="news-field-label" for="assistant-name-input">Name</label>
      <input
        id="assistant-name-input"
        class="news-text-input"
        type="text"
        maxlength="60"
        bind:value={assistantNameInput}
        placeholder="Localist"
      />

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$assistantNameLoading || assistantNameUnchanged}
        on:click={handleSaveAssistantName}
      >Save</button>

      {#if assistantNameSaveResult}
        <p class="card-hint">{assistantNameSaveResult}</p>
      {/if}
      {#if $assistantNameError}
        <p class="card-hint" style="color:var(--error)">{$assistantNameError}</p>
      {/if}
    </section>

    <!-- Optional API keys (BRAVE/LANGSEARCH/NEWSAPI — .env-backed, used by localist-mcp) -->
    <section class="settings-card">
      <div class="card-title">API Keys</div>
      <p class="card-desc">
        Optional keys for web search and news tools, saved to backend/.env. Only
        LANGSEARCH_API_KEY is required for full web_search functionality — the others enable
        Brave as a search fallback and the daily news brief. Values are never shown once saved;
        leave a field blank to keep its current key unchanged.
      </p>

      <label class="news-field-label" for="langsearch-key-input">
        LANGSEARCH_API_KEY {#if $apiKeysStatus.langsearch_api_key_set}<span class="card-hint">(configured)</span>{/if}
      </label>
      <input
        id="langsearch-key-input"
        class="news-text-input"
        type="password"
        autocomplete="off"
        bind:value={langsearchKeyInput}
        placeholder={$apiKeysStatus.langsearch_api_key_set ? '••••••••' : 'Not set'}
      />

      <label class="news-field-label" for="brave-key-input" style="margin-top: var(--sp-3)">
        BRAVE_API_KEY {#if $apiKeysStatus.brave_api_key_set}<span class="card-hint">(configured)</span>{/if}
      </label>
      <input
        id="brave-key-input"
        class="news-text-input"
        type="password"
        autocomplete="off"
        bind:value={braveKeyInput}
        placeholder={$apiKeysStatus.brave_api_key_set ? '••••••••' : 'Not set'}
      />

      <label class="news-field-label" for="newsapi-key-input" style="margin-top: var(--sp-3)">
        NEWSAPI_API_KEY {#if $apiKeysStatus.newsapi_api_key_set}<span class="card-hint">(configured)</span>{/if}
      </label>
      <input
        id="newsapi-key-input"
        class="news-text-input"
        type="password"
        autocomplete="off"
        bind:value={newsapiKeyInput}
        placeholder={$apiKeysStatus.newsapi_api_key_set ? '••••••••' : 'Not set'}
      />

      <button
        type="button"
        class="seg-btn news-save-btn"
        style="margin-top: var(--sp-3)"
        disabled={$apiKeysLoading || (!langsearchKeyInput.trim() && !braveKeyInput.trim() && !newsapiKeyInput.trim())}
        on:click={handleSaveApiKeys}
      >Save</button>

      {#if apiKeysSaveResult}
        <p class="card-hint">{apiKeysSaveResult}</p>
      {/if}
      {#if $apiKeysError}
        <p class="card-hint" style="color:var(--error)">{$apiKeysError}</p>
      {/if}
    </section>

    <!-- Data retention (chat history + episodes — independent as of §20.12) -->
    <section class="settings-card">
      <div class="card-title">Data Retention</div>

      <p class="card-desc">Chat History — how long raw conversation turns are kept before they're permanently deleted.</p>
      <div class="segmented">
        {#each EVICTION_PRESETS as p}
          <button
            type="button"
            class="seg-btn"
            class:active={$retentionSettings.eviction_preset === p.value}
            disabled={$retentionSettingsLoading}
            on:click={() => setRetentionPreset(p.value)}
          >{p.label}</button>
        {/each}
      </div>

      <p class="card-desc" style="margin-top: var(--sp-4)">
        Episodic Memory — how long learned facts/preferences are kept before being retracted
        (reversible — see the Episodes tab's "Reactivate" button). Defaults to Forever: a fact
        doesn't stop being true just because time passed with no chat about it.
      </p>
      <div class="segmented">
        {#each EVICTION_PRESETS as p}
          <button
            type="button"
            class="seg-btn"
            class:active={$retentionSettings.episode_eviction_preset === p.value}
            disabled={$retentionSettingsLoading}
            on:click={() => setEpisodeRetentionPreset(p.value)}
          >{p.label}</button>
        {/each}
      </div>

      {#if $retentionSettingsError}
        <p class="card-hint" style="color:var(--error)">{$retentionSettingsError}</p>
      {/if}
    </section>

    <!-- Daily News Brief -->
    <section class="settings-card">
      <div class="card-title">Daily News Brief</div>
      <p class="card-desc">
        Top Stories (anchored to your home country) is fixed; Local is
        keyword-matched against a place you set (NewsAPI has no true
        local-news coverage). Pick exactly 3 special-interest topics.
      </p>

      <label class="news-field-label" for="news-home-country">Home country (2-letter code)</label>
      <input
        id="news-home-country"
        class="news-text-input"
        type="text"
        maxlength="2"
        bind:value={homeCountryInput}
        placeholder="us"
      />

      <label class="news-field-label" for="news-local-query">Local area (e.g. a city name)</label>
      <input
        id="news-local-query"
        class="news-text-input"
        type="text"
        autocomplete="off"
        bind:value={localQueryInput}
        placeholder="e.g. Seattle"
      />

      <div class="news-field-label">
        Special-interest topics ({selectedTopics.length}/3 selected)
      </div>
      <div class="news-topic-grid">
        {#each Object.entries($newsPreferences.topic_pool) as [key, label]}
          <button
            type="button"
            class="news-topic-chip"
            class:active={selectedTopics.includes(key)}
            disabled={!selectedTopics.includes(key) && selectedTopics.length >= 3}
            on:click={() => toggleTopic(key)}
          >{label}</button>
        {/each}
      </div>
      {#if topicsIncomplete}
        <p class="card-hint" style="color:var(--text-tertiary)">
          Pick exactly 3 topics to enable Save — this also saves your Local
          area and Home country changes above; they're sent together in one
          request.
        </p>
      {/if}

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$newsPreferencesLoading || topicsIncomplete}
        title={topicsIncomplete ? 'Select exactly 3 topics to save' : undefined}
        on:click={handleSaveNewsPreferences}
      >Save</button>

      {#if newsSaveResult}
        <p class="card-hint">{newsSaveResult}</p>
      {/if}
      {#if $newsPreferencesError}
        <p class="card-hint" style="color:var(--error)">{$newsPreferencesError}</p>
      {/if}
    </section>

    <!-- Pinned GitHub Repos -->
    <section class="settings-card">
      <div class="card-title">Pinned GitHub Repos</div>
      <p class="card-desc">
        Track release notes for repos without clicking "Watch" on GitHub —
        pinning here doesn't subscribe you to that repo's PR/issue emails.
        Shows up alongside any repos you do watch in the Live Feed panel's
        GitHub block.
      </p>

      <label class="news-field-label" for="pinned-repo-input">Add a repo (owner/repo)</label>
      <div class="pinned-repo-add-row">
        <input
          id="pinned-repo-input"
          class="news-text-input"
          type="text"
          autocomplete="off"
          bind:value={pinnedRepoInput}
          placeholder="e.g. ollama/ollama"
          on:input={() => (pinnedRepoAddError = null)}
          on:keydown={(e) => e.key === 'Enter' && addPinnedRepo()}
        />
        <button type="button" class="seg-btn" on:click={addPinnedRepo}>Add</button>
      </div>
      {#if pinnedRepoAddError}
        <p class="card-hint" style="color:var(--error)">{pinnedRepoAddError}</p>
      {/if}

      {#if pinnedRepos.length > 0}
        <div class="news-topic-grid">
          {#each pinnedRepos as repo (repo)}
            <span class="pinned-repo-chip">
              {repo}
              <button
                type="button"
                class="pinned-repo-remove"
                aria-label={`Remove ${repo}`}
                on:click={() => removePinnedRepo(repo)}
              >&times;</button>
            </span>
          {/each}
        </div>
      {/if}

      <button
        type="button"
        class="seg-btn news-save-btn"
        disabled={$githubWatchPinsLoading || pinnedReposUnchanged}
        on:click={handleSavePinnedRepos}
      >Save</button>

      {#if pinnedReposSaveResult}
        <p class="card-hint">{pinnedReposSaveResult}</p>
      {/if}
      {#if $githubWatchPinsError}
        <p class="card-hint" style="color:var(--error)">{$githubWatchPinsError}</p>
      {/if}
    </section>

    <!-- Version -->
    <section class="settings-card row">
      <div class="card-title">Version</div>
      <div class="card-value text-mono">{APP_VERSION} · {isTauri() ? 'desktop' : 'web'}</div>
    </section>

  </div>
</div>

<style>
  .settings-page {
    flex: 1;
    overflow-y: auto;
    padding: var(--sp-8);
  }

  .settings-inner {
    max-width: 560px;
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
  }

  /* Card */
  .settings-card {
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: var(--sp-4) var(--sp-5);
  }

  .settings-card.row {
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
  }

  .card-title {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .card-title .badge {
    margin-left: var(--sp-2);
    vertical-align: middle;
  }

  .stale-badges {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-2);
  }

  .card-desc {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-top: calc(var(--sp-3) * -1);
  }

  .card-value {
    font-size: 12.5px;
    color: var(--text-secondary);
  }

  .card-hint {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    line-height: 1.6;
  }
  .card-hint code {
    font-size: 10.5px;
  }

  /* Fields */
  .field-group {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .field-label {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-secondary);
  }

  .field-hint {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    line-height: 1.5;
  }

  .field-row {
    display: flex;
    gap: var(--sp-2);
  }

  .settings-input {
    flex: 1;
    padding: var(--sp-2) var(--sp-3);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    min-width: 0;
  }

  /* Buttons */
  .btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    border-radius: var(--radius);
    white-space: nowrap;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 11px; height: 11px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Health readout */
  .health-readout {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .health-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--sp-4);
    font-size: var(--text-xs);
  }

  .health-label {
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  .health-value {
    color: var(--text-secondary);
    text-align: right;
    word-break: break-all;
  }

  .error-row .health-value { color: var(--error); }

  .model-chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    justify-content: flex-end;
  }

  .news-field-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: var(--sp-3);
  }

  .news-text-input {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: var(--text-sm);
    padding: var(--sp-2) var(--sp-3);
    margin-top: var(--sp-1);
  }

  .news-topic-grid {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-2);
    margin-top: var(--sp-2);
  }

  .news-topic-chip {
    padding: var(--sp-1) var(--sp-3);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .news-topic-chip:hover:not(:disabled) { background: var(--bg-hover); }
  .news-topic-chip.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .news-topic-chip:disabled:not(.active) { cursor: not-allowed; opacity: 0.5; }

  .news-save-btn {
    margin-top: var(--sp-3);
    align-self: flex-start;
  }

  .pinned-repo-add-row {
    display: flex;
    gap: var(--sp-2);
    align-items: center;
  }
  .pinned-repo-add-row .news-text-input { flex: 1; }

  .pinned-repo-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-1);
    padding: var(--sp-1) var(--sp-2) var(--sp-1) var(--sp-3);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: var(--bg-raised);
    color: var(--text-secondary);
    font-size: 12px;
    font-family: var(--font-mono);
  }

  .pinned-repo-remove {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0 var(--sp-1);
  }
  .pinned-repo-remove:hover { color: var(--error); }
</style>
