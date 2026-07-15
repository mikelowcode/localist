<script lang="ts">
  import { onMount } from 'svelte';
  import { health, checkHealth } from '$lib/stores/server';
  import { modelConfig, RUNTIME_BACKENDS, RUNTIME_BACKEND_LABELS, type RuntimeBackend } from '$lib/stores/model';
  import {
    runtimeBackendSwitchLoading,
    runtimeBackendSwitchError,
    switchRuntimeBackend,
    fetchBackendModels,
    pinChatModel
  } from '$lib/stores/runtimeBackendSwitch';
  import { theme } from '$lib/stores/theme';
  import { browser } from '$app/environment';
  import {
    chatHistorySettings,
    chatHistorySettingsLoading,
    chatHistorySettingsError,
    loadChatHistorySettings,
    setChatHistoryEvictionPreset,
    type EvictionPreset
  } from '$lib/stores/chatHistorySettings';

  const EVICTION_PRESETS: { value: EvictionPreset; label: string }[] = [
    { value: '7d',      label: '7 days' },
    { value: '30d',     label: '30 days' },
    { value: '90d',     label: '90 days' },
    { value: 'forever', label: 'Forever' }
  ];

  onMount(() => {
    loadChatHistorySettings();
  });

  // Local form state — mirrors store, saved on blur/change
  let backendUrl = '';
  let chatModel  = '';

  // Initialize from store
  $: if (browser) {
    backendUrl = localStorage.getItem('lora-backend-url') ?? 'http://127.0.0.1:8000';
    chatModel  = $modelConfig.chat_model;
  }

  function saveBackendUrl() {
    if (browser) localStorage.setItem('lora-backend-url', backendUrl);
  }

  // The backend whose models the Chat Model dropdown below is scoped to.
  // Deliberately independent of $modelConfig.backend (the real active
  // backend, synced from health polling) — clicking a segmented-control
  // button previews that backend's models immediately, even before (or
  // without) confirming an actual switch to it.
  let selectedUiBackend: RuntimeBackend = $modelConfig.backend as RuntimeBackend;
  let availableModels: string[] = [];
  let modelsFetchToken = 0;

  async function refreshModelsForSelected() {
    const token = ++modelsFetchToken;
    const models = await fetchBackendModels(selectedUiBackend);
    if (token === modelsFetchToken) availableModels = models;
  }

  $: if (selectedUiBackend) refreshModelsForSelected();

  function selectRuntimeBackend(backend: RuntimeBackend) {
    selectedUiBackend = backend;
    if (backend === $modelConfig.backend) return; // already active — nothing to switch

    const proceed = confirm(
      `Switch the active runtime backend to ${RUNTIME_BACKEND_LABELS[backend]}? ` +
      `In-flight requests will keep running on the current backend unaffected, ` +
      `but this switch itself can take several seconds.`
    );
    if (!proceed) return;

    void switchRuntimeBackend(backend);
  }

  async function handleChatModelPick(value: string) {
    if (!value) return;
    chatModel = value;
    modelConfig.update((s) => ({ ...s, chat_model: value }));
    await pinChatModel(selectedUiBackend, value);
  }

  function onChatModelSelectChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    void handleChatModelPick(value);
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
    streaming = localStorage.getItem('lora-streaming') !== '0';
    episodicApproval = localStorage.getItem('lora-episodic-approval') === '1';
  }

  function toggleStreaming() {
    streaming = !streaming;
    if (browser) localStorage.setItem('lora-streaming', streaming ? '1' : '0');
  }

  function toggleEpisodicApproval() {
    episodicApproval = !episodicApproval;
    if (browser) localStorage.setItem('lora-episodic-approval', episodicApproval ? '1' : '0');
  }
</script>

<svelte:head>
  <title>Settings — Localist</title>
</svelte:head>

<div class="settings-page">
  <div class="settings-inner">

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
      {#if $runtimeBackendSwitchError}
        <p class="card-hint" style="color:var(--error)">{$runtimeBackendSwitchError}</p>
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
          <option value="" disabled selected={!chatModel || !availableModels.includes(chatModel)}>
            {availableModels.length ? 'Select a model…' : 'No models reported by this backend'}
          </option>
          {#each availableModels as m}
            <option value={m} selected={m === chatModel}>{m}</option>
          {/each}
        </select>
        {#if selectedUiBackend === $modelConfig.backend}
          <p class="field-hint">Applies immediately — {RUNTIME_BACKEND_LABELS[selectedUiBackend]} is the active backend.</p>
        {:else}
          <p class="field-hint">Saved for {RUNTIME_BACKEND_LABELS[selectedUiBackend]} — takes effect the next time you switch to it.</p>
        {/if}
      </div>
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
                ? 'Cosine-similarity retrieval active (mlx-community/embeddinggemma-300m-4bit).'
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

    <!-- Chat history eviction -->
    <section class="settings-card">
      <div class="card-title">Chat History</div>
      <p class="card-desc">Choose how long chat turns are kept before they're automatically evicted.</p>
      <div class="segmented">
        {#each EVICTION_PRESETS as p}
          <button
            type="button"
            class="seg-btn"
            class:active={$chatHistorySettings.eviction_preset === p.value}
            disabled={$chatHistorySettingsLoading}
            on:click={() => setChatHistoryEvictionPreset(p.value)}
          >{p.label}</button>
        {/each}
      </div>
      {#if $chatHistorySettingsError}
        <p class="card-hint" style="color:var(--error)">{$chatHistorySettingsError}</p>
      {/if}
    </section>

    <!-- Version -->
    <section class="settings-card row">
      <div class="card-title">Version</div>
      <div class="card-value text-mono">0.2.0 · web</div>
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
</style>
