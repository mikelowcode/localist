<script lang="ts">
  import { health, checkHealth } from '$lib/stores/server';
  import { modelConfig } from '$lib/stores/model';
  import { theme } from '$lib/stores/theme';
  import { browser } from '$app/environment';

  // Local form state — mirrors store, saved on blur/change
  let backendUrl = '';
  let chatModel  = '';
  let embedModel = '';

  // Initialize from store
  $: if (browser) {
    backendUrl = localStorage.getItem('lora-backend-url') ?? 'http://127.0.0.1:8000';
    chatModel  = $modelConfig.chat_model;
    embedModel = $modelConfig.embedding_model;
  }

  function saveBackendUrl() {
    if (browser) localStorage.setItem('lora-backend-url', backendUrl);
  }

  function saveModelConfig() {
    modelConfig.set({
      ...$modelConfig,
      chat_model:      chatModel,
      embedding_model: embedModel
    });
  }

  let checking = false;
  async function runHealthCheck() {
    checking = true;
    await checkHealth();
    checking = false;
  }
</script>

<svelte:head>
  <title>Settings — Localist</title>
</svelte:head>

<div class="settings-page">
  <div class="settings-inner">
    <h1 class="settings-title">Settings</h1>

    <!-- Backend connection -->
    <section class="settings-section">
      <h2 class="section-heading">Backend Connection</h2>
      <p class="section-desc">Configure the FastAPI server that the UI connects to.</p>

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

        <!-- Health status readout -->
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

    <div class="divider" />

    <!-- Model selection -->
    <section class="settings-section">
      <h2 class="section-heading">Model Configuration</h2>
      <p class="section-desc">Model identifiers must match the IDs returned by the active backend.</p>

      <div class="field-group">
        <label class="field-label" for="chat-model">Chat model ID</label>
        <input
          id="chat-model"
          type="text"
          class="settings-input"
          bind:value={chatModel}
          on:blur={saveModelConfig}
          placeholder="Phi-4-mini-instruct-generic-gpu:5"
          spellcheck="false"
        />
      </div>

      <div class="field-group">
        <label class="field-label" for="embed-model">Embedding model ID</label>
        <input
          id="embed-model"
          type="text"
          class="settings-input"
          bind:value={embedModel}
          on:blur={saveModelConfig}
          placeholder="text-embedding-3-small"
          spellcheck="false"
        />
        <p class="field-hint">Leave empty if your backend does not provide embeddings (use_embeddings will be disabled).</p>
      </div>
    </section>

    <div class="divider" />

    <!-- Theme -->
    <section class="settings-section">
      <h2 class="section-heading">Appearance</h2>

      <div class="theme-row">
        <div class="theme-info">
          <span class="field-label">Theme</span>
          <span class="field-hint" style="margin:0">
            Currently: <strong>{$theme}</strong>
          </span>
        </div>
        <button
          class="btn-secondary theme-toggle-btn"
          on:click={theme.toggle}
          aria-label="Toggle theme"
        >
          {#if $theme === 'dark'}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
            Switch to light
          {:else}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
            Switch to dark
          {/if}
        </button>
      </div>
    </section>

    <div class="divider" />

    <!-- About -->
    <section class="settings-section">
      <h2 class="section-heading">About</h2>
      <dl class="about-list">
        <div class="about-row">
          <dt>System</dt>
          <dd>Localist Framework</dd>
        </div>
        <div class="about-row">
          <dt>UI Version</dt>
          <dd class="text-mono">0.1.0</dd>
        </div>
        <div class="about-row">
          <dt>Framework</dt>
          <dd class="text-mono">SvelteKit</dd>
        </div>
      </dl>
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
    max-width: 640px;
    display: flex;
    flex-direction: column;
    gap: var(--sp-6);
  }

  .settings-title {
    font-size: var(--text-2xl);
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: var(--sp-2);
  }

  /* Section */
  .settings-section {
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
  }

  .section-heading {
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--text-primary);
  }

  .section-desc {
    font-size: var(--text-sm);
    color: var(--text-tertiary);
    margin-top: calc(var(--sp-2) * -1);
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

  /* Theme toggle */
  .theme-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
  }

  .theme-info {
    display: flex;
    flex-direction: column;
    gap: var(--sp-1);
  }

  .theme-toggle-btn {
    flex-shrink: 0;
  }

  /* About */
  .about-list {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .about-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: var(--text-sm);
    gap: var(--sp-4);
  }

  .about-row dt { color: var(--text-tertiary); }
  .about-row dd { color: var(--text-secondary); }
</style>
