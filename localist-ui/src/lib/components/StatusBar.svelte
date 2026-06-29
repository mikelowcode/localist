<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { health, connectivityLabel, agents } from '$lib/stores/server';
  import { modelLabel } from '$lib/stores/model';

  let agentsOpen = false;
  let agentsWrapperEl: HTMLElement;

  function handleWindowClick(e: MouseEvent) {
    if (agentsOpen && agentsWrapperEl && !agentsWrapperEl.contains(e.target as Node)) {
      agentsOpen = false;
    }
  }

  function handleWindowKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && agentsOpen) {
      agentsOpen = false;
    }
  }

  onMount(() => {
    window.addEventListener('click', handleWindowClick);
    window.addEventListener('keydown', handleWindowKeydown);
  });

  onDestroy(() => {
    if (browser) {
      window.removeEventListener('click', handleWindowClick);
      window.removeEventListener('keydown', handleWindowKeydown);
    }
  });
</script>

<header class="statusbar" aria-label="System status">
  <!-- Page slot (left side — filled by child layouts) -->
  <div class="statusbar-left">
    <slot />
  </div>

  <!-- Status indicators (right side) -->
  <div class="statusbar-right">

    <!-- Active agents -->
    {#if $agents.loaded && $agents.agents.length > 0}
      <div class="agents-wrap" bind:this={agentsWrapperEl}>
        <button
          class="status-chip agents-chip"
          aria-expanded={agentsOpen}
          aria-haspopup="true"
          on:click={() => (agentsOpen = !agentsOpen)}
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
            <circle cx="9" cy="7" r="4"/>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
          </svg>
          <span>{$agents.agents.length} agent{$agents.agents.length !== 1 ? 's' : ''}</span>
        </button>
        {#if agentsOpen}
          <div class="agents-popover" role="menu" aria-label="Active agents">
            {#each $agents.agents as name}
              <div class="agents-popover-item" role="menuitem">{name}</div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Model name -->
    {#if $modelLabel !== '—'}
      <div class="status-chip model-chip" title="Active model: {$modelLabel}">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
          <line x1="8" y1="21" x2="16" y2="21"/>
          <line x1="12" y1="17" x2="12" y2="21"/>
        </svg>
        <span class="model-name">{$modelLabel}</span>
      </div>
    {/if}

    <!-- Connectivity -->
    <div
      class="status-chip connectivity-chip"
      class:chip-online={$connectivityLabel === 'online'}
      class:chip-degraded={$connectivityLabel === 'degraded'}
      class:chip-offline={$connectivityLabel === 'offline'}
      class:chip-checking={$connectivityLabel === 'checking'}
      title="Backend: {$connectivityLabel}"
      aria-label="Backend status: {$connectivityLabel}"
    >
      <span
        class="dot"
        class:dot-success={$connectivityLabel === 'online'}
        class:dot-warning={$connectivityLabel === 'degraded'}
        class:dot-error={$connectivityLabel === 'offline'}
        class:dot-muted={$connectivityLabel === 'checking'}
        class:dot-pulse={$connectivityLabel === 'checking' || $connectivityLabel === 'online'}
        aria-hidden="true"
      />
      <span>{$connectivityLabel}</span>
    </div>
  </div>
</header>

<style>
  .statusbar {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--sp-6);
    height: var(--topbar-h);
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: var(--sp-4);
  }

  .statusbar-left {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: var(--sp-3);
  }

  .statusbar-right {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    flex-shrink: 0;
  }

  /* Generic chip */
  .status-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-1);
    padding: 3px 10px;
    border-radius: 100px;
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    font-weight: 500;
    letter-spacing: 0.03em;
    background: var(--bg-active);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    user-select: none;
    white-space: nowrap;
  }

  /* Connectivity states */
  .chip-online {
    color: var(--success);
    background: var(--success-dim);
    border-color: #3ecf8e33;
  }
  .chip-degraded {
    color: var(--warning);
    background: var(--warning-dim);
    border-color: #f5a62333;
  }
  .chip-offline {
    color: var(--error);
    background: var(--error-dim);
    border-color: #e0525233;
  }
  .chip-checking {
    color: var(--text-tertiary);
    background: var(--bg-active);
  }

  /* Model chip */
  .model-chip {
    max-width: 200px;
    overflow: hidden;
  }
  .model-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Agents chip + popover */
  .agents-wrap {
    position: relative;
  }

  .agents-chip {
    cursor: pointer;
    /* Reset button defaults */
    background: var(--bg-active);
    border: 1px solid var(--border);
    outline: none;
  }

  .agents-chip:hover {
    background: var(--bg-raised);
    border-color: var(--border-focus);
  }

  .agents-chip:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .agents-popover {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 100;
    min-width: 160px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-2) 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .agents-popover-item {
    padding: var(--sp-2) var(--sp-4);
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-secondary);
    white-space: nowrap;
  }
</style>
