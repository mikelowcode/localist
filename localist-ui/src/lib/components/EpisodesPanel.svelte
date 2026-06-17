<script lang="ts">
  import { onMount } from 'svelte';
  import {
    episodesStore,
    loadEpisodes,
    EPISODE_TYPES,
    TYPE_LABELS,
    TYPE_COLORS,
    type EpisodeItem,
  } from '$lib/stores/episodes';

  $: state     = $episodesStore;
  $: episodes  = state.episodes;
  $: loading   = state.loading;
  $: error     = state.error;

  let typeFilter = '';

  function applyFilter(type: string) {
    typeFilter = type;
    loadEpisodes({ typeFilter: type, offset: 0 });
  }

  function formatDate(ts: number): string {
    return new Date(ts * 1000).toLocaleDateString([], {
      month: 'short', day: 'numeric', year: 'numeric',
    });
  }

  function formatConfidence(c: number): string {
    return Math.round(c * 100) + '%';
  }

  function chipStyle(type: string): string {
    const col = TYPE_COLORS[type] ?? TYPE_COLORS['context'];
    return `background:${col.bg};color:${col.color};border-color:${col.border};`;
  }

  onMount(() => {
    loadEpisodes({ typeFilter: '' });
  });
</script>

<div class="episodes-panel">
  <!-- Header -->
  <div class="panel-header">
    <h2 class="panel-title">Episodic Memory</h2>
    <button
      class="refresh-btn"
      on:click={() => loadEpisodes({ typeFilter, offset: 0 })}
      disabled={loading}
      aria-label="Refresh episodes"
      title="Refresh"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round"
           class:spinning={loading}>
        <polyline points="23 4 23 10 17 10"/>
        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
      </svg>
    </button>
  </div>

  <!-- Type filters -->
  <div class="filter-row">
    <button
      class="filter-chip"
      class:active={typeFilter === ''}
      on:click={() => applyFilter('')}
    >All</button>
    {#each EPISODE_TYPES as type}
      <button
        class="filter-chip"
        class:active={typeFilter === type}
        style={typeFilter === type ? chipStyle(type) : ''}
        on:click={() => applyFilter(type)}
      >{TYPE_LABELS[type]}</button>
    {/each}
  </div>

  <!-- Content -->
  <div class="episode-list">
    {#if loading && episodes.length === 0}
      <div class="state-msg">
        <span class="dot dot-muted dot-pulse" />
        Loading…
      </div>

    {:else if error}
      <div class="state-msg state-error">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        {error}
      </div>

    {:else if episodes.length === 0}
      <div class="state-msg">
        No episodes stored yet. LORA will learn from your conversations.
      </div>

    {:else}
      {#each episodes as ep (ep.id)}
        <div class="episode-card fade-in">
          <div class="episode-header">
            <span
              class="type-chip"
              style={chipStyle(ep.episode_type)}
            >{TYPE_LABELS[ep.episode_type] ?? ep.episode_type}</span>
            <span class="episode-subject">{ep.subject}</span>
            <span class="episode-date">{formatDate(ep.created_at)}</span>
          </div>
          <p class="episode-content">{ep.content}</p>
          <div class="episode-footer">
            <span class="meta-item" title="Confidence">
              ◈ {formatConfidence(ep.confidence)}
            </span>
            {#if ep.project_context}
              <span class="meta-item">
                ⬡ {ep.project_context}
              </span>
            {/if}
            <span class="meta-item meta-source" title={ep.source}>
              via {ep.source}
            </span>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .episodes-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  /* Header */
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--sp-5) var(--sp-6) var(--sp-4);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .panel-title {
    font-size: var(--text-md);
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius);
    background: transparent;
    color: var(--text-tertiary);
    transition: color var(--dur-fast) var(--ease),
                background var(--dur-fast) var(--ease);
  }
  .refresh-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .spinning { animation: spin 0.8s linear infinite; }

  /* Filter row */
  .filter-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    padding: var(--sp-3) var(--sp-6);
    border-bottom: 1px solid var(--border-soft);
    flex-shrink: 0;
  }

  .filter-chip {
    font-size: 11px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg-active);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all var(--dur-fast) var(--ease);
  }
  .filter-chip:hover { color: var(--text-secondary); }
  .filter-chip.active {
    color: var(--text-primary);
    border-color: var(--accent-mid);
    background: var(--accent-dim);
  }

  /* Episode list */
  .episode-list {
    flex: 1;
    overflow-y: auto;
    padding: var(--sp-4) var(--sp-6);
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .state-msg {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-sm);
    color: var(--text-tertiary);
    padding: var(--sp-8) 0;
    justify-content: center;
  }
  .state-error { color: var(--error); }

  /* Episode card */
  .episode-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-3) var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
    transition: border-color var(--dur-fast) var(--ease);
  }
  .episode-card:hover { border-color: #3a3a3a; }

  .episode-header {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    flex-wrap: wrap;
  }

  .type-chip {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 7px;
    border-radius: 999px;
    border: 1px solid transparent;
    font-weight: 500;
    letter-spacing: 0.03em;
    flex-shrink: 0;
  }

  .episode-subject {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-primary);
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .episode-date {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .episode-content {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0;
  }

  .episode-footer {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    flex-wrap: wrap;
  }

  .meta-item {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  .meta-source {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 180px;
  }
</style>
