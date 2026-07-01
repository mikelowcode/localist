<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    chatHistorySettings,
    chatHistorySettingsLoading,
    chatHistorySettingsError,
    loadChatHistorySettings,
    setChatHistoryEvictionPreset,
    type EvictionPreset
  } from '$lib/stores/chatHistorySettings';
  import {
    chatTurns,
    chatTurnsTotal,
    chatTurnsLoading,
    chatTurnsError,
    chatHistoryQuery,
    chatHistoryOffset,
    CHAT_HISTORY_PAGE_SIZE,
    loadChatTurns,
    resetChatHistoryOffset
  } from '$lib/stores/chatHistoryList';

  const PRESETS: { value: EvictionPreset; label: string }[] = [
    { value: '7d',      label: '7 days' },
    { value: '30d',     label: '30 days' },
    { value: '90d',     label: '90 days' },
    { value: 'forever', label: 'Forever' }
  ];

  onMount(() => {
    loadChatHistorySettings();
    loadChatTurns();   // empty query = unfiltered recent turns
  });

  function handlePresetChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    if (value === '7d' || value === '30d' || value === '90d' || value === 'forever') {
      setChatHistoryEvictionPreset(value);
    }
  }

  // -- Search (debounced) --------------------------------------------------

  let searchInput = '';
  let debounceTimer: ReturnType<typeof setTimeout>;

  function handleSearchInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      chatHistoryQuery.set(searchInput);
      resetChatHistoryOffset();
      loadChatTurns();
    }, 300);
  }

  onDestroy(() => clearTimeout(debounceTimer));

  // -- Pagination ------------------------------------------------------------

  function goPrev() {
    chatHistoryOffset.update((o) => Math.max(0, o - CHAT_HISTORY_PAGE_SIZE));
    loadChatTurns();
  }

  function goNext() {
    chatHistoryOffset.update((o) => o + CHAT_HISTORY_PAGE_SIZE);
    loadChatTurns();
  }

  // -- Formatting ------------------------------------------------------------

  function formatCreatedAt(ts: number): string {
    return new Date(ts * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit'
    });
  }

  function truncate(content: string, max = 280): string {
    return content.length <= max ? content : content.slice(0, max) + '…';
  }

  function roleBadgeClass(role: string): string {
    if (role === 'user')      return 'badge-accent';
    if (role === 'assistant') return 'badge-success';
    return 'badge-muted';
  }
</script>

<svelte:head>
  <title>History — Localist</title>
</svelte:head>

<div class="settings-page">
  <div class="settings-inner">
    <h1 class="settings-title">Chat History</h1>

    <!-- Retention settings -->
    <section class="settings-section">
      <h2 class="section-heading">Retention</h2>
      <p class="section-desc">Choose how long chat turns are kept before they're automatically evicted.</p>

      <div class="field-group">
        <label class="field-label" for="eviction-preset">Retention policy</label>
        <div class="field-row">
          <select
            id="eviction-preset"
            class="settings-input"
            value={$chatHistorySettings.eviction_preset ?? ''}
            on:change={handlePresetChange}
            disabled={$chatHistorySettingsLoading}
          >
            <option value="" disabled selected={$chatHistorySettings.eviction_preset === null}>
              Choose a retention policy…
            </option>
            {#each PRESETS as p}
              <option value={p.value}>{p.label}</option>
            {/each}
          </select>
          {#if $chatHistorySettingsLoading}
            <span class="text-tertiary text-sm">Loading…</span>
          {/if}
        </div>
        {#if $chatHistorySettingsError}
          <p class="field-hint" style="color:var(--error)">{$chatHistorySettingsError}</p>
        {/if}
      </div>
    </section>

    <div class="divider" />

    <!-- Turn list -->
    <section class="settings-section">
      <h2 class="section-heading">Turns</h2>

      <input
        type="search"
        class="settings-input search-input"
        placeholder="Search chat history…"
        bind:value={searchInput}
        on:input={handleSearchInput}
        aria-label="Search chat history"
      />

      {#if $chatTurnsLoading}
        <div class="state-msg">
          <span class="dot dot-muted dot-pulse" />
          Loading…
        </div>

      {:else if $chatTurnsError}
        <div class="state-msg state-error">{$chatTurnsError}</div>

      {:else if $chatTurnsTotal === 0 && $chatHistoryQuery === ''}
        <div class="state-msg">No chat turns yet. They'll appear here as you use the assistant.</div>

      {:else if $chatTurnsTotal === 0}
        <div class="state-msg">No results for your search.</div>

      {:else}
        <div class="turn-list">
          {#each $chatTurns as turn (turn.id)}
            <div class="turn-card">
              <div class="turn-header">
                <span class="badge {roleBadgeClass(turn.role)}">{turn.role}</span>
                <span class="turn-date">{formatCreatedAt(turn.created_at)}</span>
              </div>
              <p class="turn-content">{truncate(turn.content)}</p>
              {#if turn.sources.length > 0}
                <div class="turn-footer">
                  <span class="meta-item">
                    ⬡ {turn.sources.length} source{turn.sources.length === 1 ? '' : 's'}
                  </span>
                </div>
              {/if}
            </div>
          {/each}
        </div>

        <div class="pagination-row">
          <span class="page-info">
            {$chatHistoryOffset + 1}–{Math.min($chatHistoryOffset + CHAT_HISTORY_PAGE_SIZE, $chatTurnsTotal)}
            of {$chatTurnsTotal}
          </span>
          <div class="pagination-buttons">
            <button
              class="btn-secondary"
              on:click={goPrev}
              disabled={$chatHistoryOffset === 0}
            >Previous</button>
            <button
              class="btn-secondary"
              on:click={goNext}
              disabled={$chatHistoryOffset + CHAT_HISTORY_PAGE_SIZE >= $chatTurnsTotal}
            >Next</button>
          </div>
        </div>
      {/if}
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
    align-items: center;
    gap: var(--sp-2);
  }

  .settings-input {
    flex: 1;
    padding: var(--sp-2) var(--sp-3);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    min-width: 0;
  }

  /* Search */
  .search-input {
    width: 100%;
  }

  /* State messages */
  .state-msg {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-sm);
    color: var(--text-tertiary);
    padding: var(--sp-6) 0;
    justify-content: center;
  }
  .state-error { color: var(--error); }

  /* Turn list */
  .turn-list {
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .turn-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-3) var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .turn-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
  }

  .turn-date {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .turn-content {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .turn-footer {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
  }

  .meta-item {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  /* Pagination */
  .pagination-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
  }

  .page-info {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-tertiary);
  }

  .pagination-buttons {
    display: flex;
    gap: var(--sp-2);
  }

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

  .btn-secondary:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
