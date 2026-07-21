<script lang="ts">
  import {
    episodeTurns,
    episodeTurnsTotal,
    episodeTurnsLoading,
    episodeTurnsError,
    episodeOffset,
    episodeFilters,
    selectedEpisodeId,
    loadEpisodeTurns,
    PAGE_SIZE,
    type EpisodeTurn
  } from '$lib/stores/episodeBrowser';

  function formatDate(ts: number): string {
    return new Date(ts * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
    });
  }

  function truncate(content: string, max = 140): string {
    return content.length <= max ? content : content.slice(0, max) + '…';
  }

  function toolBadges(turn: EpisodeTurn): string[] {
    const badges: string[] = [];
    if (turn.metadata?.chart) badges.push('Chart');
    if (turn.metadata?.pending_diffs?.length) badges.push('Diff');
    if (turn.metadata?.workflow_id) badges.push('Workflow');
    return badges;
  }

  function selectTurn(id: number) {
    selectedEpisodeId.set(id);
  }

  function goPrev() {
    episodeOffset.update((o) => Math.max(0, o - PAGE_SIZE));
    loadEpisodeTurns();
  }

  function goNext() {
    episodeOffset.update((o) => o + PAGE_SIZE);
    loadEpisodeTurns();
  }
</script>

<div class="episode-list-pane">
  <div class="list-scroll">
    {#if $episodeTurnsLoading && $episodeTurns.length === 0}
      <div class="state-msg">
        <span class="dot dot-muted dot-pulse" />
        Loading…
      </div>
    {:else if $episodeTurnsError}
      <div class="state-msg state-error">{$episodeTurnsError}</div>
    {:else if $episodeTurnsTotal === 0}
      <div class="state-msg">No episodes match these filters.</div>
    {:else}
      {#each $episodeTurns as turn (turn.id)}
        {@const badges = toolBadges(turn)}
        <button
          type="button"
          class="episode-list-item"
          class:selected={$selectedEpisodeId === turn.id}
          on:click={() => selectTurn(turn.id)}
        >
          <div class="item-header">
            <span class="role-badge role-{turn.role}">{turn.role === 'user' ? 'You' : 'LORA'}</span>
            <span class="item-date">{formatDate(turn.created_at)}</span>
          </div>
          <p class="item-content">{truncate(turn.content)}</p>
          {#if badges.length > 0 || turn.score !== null}
            <div class="item-footer">
              {#each badges as b}
                <span class="tool-badge">{b}</span>
              {/each}
              {#if turn.score !== null}
                <span class="score-badge">match {Math.round(turn.score * 100)}%</span>
              {/if}
            </div>
          {/if}
        </button>
      {/each}
    {/if}
  </div>

  {#if $episodeTurnsTotal > 0}
    <div class="pagination-row">
      <span class="page-info">
        {$episodeOffset + 1}–{Math.min($episodeOffset + PAGE_SIZE, $episodeTurnsTotal)} of {$episodeTurnsTotal}
      </span>
      <div class="pagination-buttons">
        <button class="btn-secondary" on:click={goPrev} disabled={$episodeOffset === 0}>Previous</button>
        <button
          class="btn-secondary"
          on:click={goNext}
          disabled={$episodeOffset + PAGE_SIZE >= $episodeTurnsTotal}
        >Next</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .episode-list-pane {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    border-right: 1px solid var(--border);
  }

  .list-scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-3);
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .state-msg {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-sm);
    color: var(--text-tertiary);
    padding: var(--sp-8) var(--sp-4);
    justify-content: center;
    text-align: center;
  }
  .state-error { color: var(--error); }

  .episode-list-item {
    display: flex;
    flex-direction: column;
    gap: var(--sp-1);
    text-align: left;
    padding: var(--sp-3);
    border-radius: var(--radius);
    border: 1px solid transparent;
    background: var(--bg-panel);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
  }
  .episode-list-item:hover { background: var(--bg-hover); }
  .episode-list-item.selected {
    background: var(--accent-dim);
    border-color: var(--accent);
  }

  .item-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
  }

  .role-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 999px;
    letter-spacing: 0.03em;
  }
  .role-user { background: var(--accent-dim); color: var(--accent); }
  .role-assistant { background: var(--success-dim); color: var(--success); }

  .item-date {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .item-content {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 0;
    word-break: break-word;
  }

  .item-footer {
    display: flex;
    gap: var(--sp-2);
    flex-wrap: wrap;
  }

  .tool-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--warning-dim);
    color: var(--warning);
  }

  .score-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--bg-active);
    color: var(--text-tertiary);
  }

  .pagination-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-4);
    padding: var(--sp-3);
    border-top: 1px solid var(--border-soft);
    flex-shrink: 0;
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
    padding: var(--sp-1) var(--sp-3);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: var(--text-xs);
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
