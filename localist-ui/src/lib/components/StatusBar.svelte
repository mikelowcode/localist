<script lang="ts">
  import { page } from '$app/stores';
  import { connectivityLabel } from '$lib/stores/server';
  import { runtimeBackendLabel } from '$lib/stores/model';
  import { pendingCount } from '$lib/stores/episodes';
  import { toggleSidebarCollapsed } from '$lib/stores/sidebar';

  const TITLES: Record<string, string> = {
    '/conversation': 'Conversation',
    '/memory':       'Episodic Memory',
    '/files':        'Files',
    '/episodes':     'Episodes',
    '/settings':     'Settings',
    '/research':     'Research'
  };

  $: path = $page.url.pathname;
  $: isChat  = path.startsWith('/conversation');
  $: isMemory = path.startsWith('/memory');
  $: screenTitle =
    Object.entries(TITLES).find(([prefix]) => path.startsWith(prefix))?.[1] ?? '';
</script>

<header class="appbar" aria-label="System status">
  <button
    class="sbtoggle"
    on:click={toggleSidebarCollapsed}
    aria-label="Show/hide sidebar"
    title="Show/hide sidebar"
  >
    <span class="sbtoggle-icon" aria-hidden="true" />
  </button>

  <div class="appbar-title">{screenTitle}</div>

  <div class="appbar-right">
    {#if isChat}
      <span
        class="pill-mono chip-online"
        class:chip-degraded={$connectivityLabel === 'degraded'}
        class:chip-offline={$connectivityLabel === 'offline'}
        class:chip-checking={$connectivityLabel === 'checking'}
        title="Backend: {$connectivityLabel}"
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
        {$runtimeBackendLabel}
      </span>
    {:else if isMemory && $pendingCount > 0}
      <span class="pill-mono pending-chip">{$pendingCount} pending</span>
    {/if}
  </div>
</header>

<style>
  .appbar {
    height: var(--topbar-h);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    padding: 0 var(--sp-4);
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
  }

  .sbtoggle {
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: var(--bg-active);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background var(--dur-fast) var(--ease);
  }
  .sbtoggle:hover { background: var(--bg-hover); }

  .sbtoggle-icon {
    width: 13px;
    height: 9px;
    border: 1.4px solid var(--text-secondary);
    border-radius: 2px;
    position: relative;
    display: inline-block;
  }
  .sbtoggle-icon::before {
    content: '';
    position: absolute;
    left: 4px;
    top: -1.4px;
    bottom: -1.4px;
    border-left: 1.4px solid var(--text-secondary);
  }

  .appbar-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .appbar-right {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    flex-shrink: 0;
  }

  .chip-online { color: var(--success); }
  .chip-degraded { color: var(--warning); }
  .chip-offline { color: var(--error); }
  .chip-checking { color: var(--text-tertiary); }

  .pending-chip {
    color: var(--warning);
    background: var(--warning-dim);
  }
</style>
