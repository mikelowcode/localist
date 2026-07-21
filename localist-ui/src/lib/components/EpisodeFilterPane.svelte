<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    episodeFilters,
    conversations,
    conversationsLoading,
    loadConversationsList,
    resetEpisodeOffset,
    loadEpisodeTurns,
    type SearchMode
  } from '$lib/stores/episodeBrowser';

  let searchInput = '';
  let debounceTimer: ReturnType<typeof setTimeout>;

  function applyAndReload() {
    resetEpisodeOffset();
    loadEpisodeTurns();
  }

  function handleSearchInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      episodeFilters.update((f) => ({ ...f, query: searchInput }));
      applyAndReload();
    }, 300);
  }

  function handleModeChange(mode: SearchMode) {
    episodeFilters.update((f) => ({ ...f, mode }));
    applyAndReload();
  }

  function handleConversationChange(e: Event) {
    const value = (e.target as HTMLSelectElement).value;
    episodeFilters.update((f) => ({ ...f, conversationId: value || null }));
    applyAndReload();
  }

  function toUnixSeconds(dateStr: string): number | null {
    if (!dateStr) return null;
    const ms = new Date(dateStr).getTime();
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  }

  function handleDateFromChange(e: Event) {
    const value = (e.target as HTMLInputElement).value;
    episodeFilters.update((f) => ({ ...f, dateFrom: toUnixSeconds(value) }));
    applyAndReload();
  }

  function handleDateToChange(e: Event) {
    const value = (e.target as HTMLInputElement).value;
    // End-of-day for the "to" bound so a same-day range isn't empty.
    const ms = value ? new Date(value).getTime() + 24 * 60 * 60 * 1000 - 1 : NaN;
    episodeFilters.update((f) => ({ ...f, dateTo: Number.isNaN(ms) ? null : Math.floor(ms / 1000) }));
    applyAndReload();
  }

  function handleHasToolResultChange(e: Event) {
    const checked = (e.target as HTMLInputElement).checked;
    episodeFilters.update((f) => ({ ...f, hasToolResult: checked }));
    applyAndReload();
  }

  function conversationLabel(c: { conversation_title: string | null; last_created_at: number }): string {
    if (c.conversation_title) return c.conversation_title;
    const ts = new Date(c.last_created_at * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
    });
    return `Untitled — ${ts}`;
  }

  onMount(() => {
    loadConversationsList();
  });

  onDestroy(() => clearTimeout(debounceTimer));
</script>

<div class="filter-pane">
  <div class="filter-section">
    <label class="filter-label" for="episode-search">Search</label>
    <input
      id="episode-search"
      type="search"
      class="filter-input"
      placeholder="Search episodes…"
      bind:value={searchInput}
      on:input={handleSearchInput}
    />
    <div class="mode-toggle" role="group" aria-label="Search mode">
      <button
        type="button"
        class="mode-btn"
        class:active={$episodeFilters.mode === 'keyword'}
        on:click={() => handleModeChange('keyword')}
      >Keyword</button>
      <button
        type="button"
        class="mode-btn"
        class:active={$episodeFilters.mode === 'semantic'}
        on:click={() => handleModeChange('semantic')}
      >Semantic</button>
    </div>
  </div>

  <div class="divider" />

  <div class="filter-section">
    <label class="filter-label" for="episode-conversation">Conversation</label>
    <select
      id="episode-conversation"
      class="filter-input"
      on:change={handleConversationChange}
      disabled={$conversationsLoading}
    >
      <option value="">All conversations</option>
      {#each $conversations as c (c.conversation_id)}
        <option value={c.conversation_id}>{conversationLabel(c)}</option>
      {/each}
    </select>
  </div>

  <div class="divider" />

  <div class="filter-section">
    <span class="filter-label">Date range</span>
    <div class="date-row">
      <input type="date" class="filter-input" aria-label="From date" on:change={handleDateFromChange} />
      <input type="date" class="filter-input" aria-label="To date" on:change={handleDateToChange} />
    </div>
  </div>

  <div class="divider" />

  <div class="filter-section">
    <label class="checkbox-row">
      <input type="checkbox" on:change={handleHasToolResultChange} />
      <span>Has tool result</span>
    </label>
  </div>
</div>

<style>
  .filter-pane {
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
    padding: var(--sp-5) var(--sp-4);
    overflow-y: auto;
    height: 100%;
  }

  .filter-section {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .filter-label {
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-tertiary);
  }

  .filter-input {
    width: 100%;
    padding: var(--sp-2) var(--sp-3);
    font-size: var(--text-sm);
    font-family: var(--font-body);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
  }

  .mode-toggle {
    display: flex;
    gap: var(--sp-1);
  }

  .mode-btn {
    flex: 1;
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    padding: var(--sp-1) var(--sp-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--bg-active);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all var(--dur-fast) var(--ease);
  }
  .mode-btn:hover { color: var(--text-secondary); }
  .mode-btn.active {
    color: #fff;
    border-color: var(--accent);
    background: var(--accent);
  }

  .date-row {
    display: flex;
    gap: var(--sp-2);
  }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-sm);
    color: var(--text-secondary);
    cursor: pointer;
  }

  .divider {
    height: 1px;
    background: var(--border-soft);
  }
</style>
