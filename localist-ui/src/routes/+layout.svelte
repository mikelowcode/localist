<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import StatusBar from '$lib/components/StatusBar.svelte';
  import { startHealthPolling, stopHealthPolling } from '$lib/stores/server';
  import { theme } from '$lib/stores/theme';
  import { sidebarWidth, sidebarCollapsed } from '$lib/stores/sidebar';

  // #app-shell is the CSS grid container that sizes the sidebar column; it
  // lives in app.html outside this component's own DOM subtree, so it's
  // reached directly rather than through a Svelte binding.
  $: if (browser) {
    const shell = document.getElementById('app-shell');
    if (shell) {
      shell.style.gridTemplateColumns = `${$sidebarCollapsed ? 0 : $sidebarWidth}px 1fr`;
    }
  }

  onMount(() => {
    // Apply saved theme
    if (browser) {
      document.documentElement.setAttribute('data-theme', $theme);
    }
    startHealthPolling();
  });

  onDestroy(() => {
    stopHealthPolling();
  });
</script>

<Sidebar />

<div class="main-column">
  <StatusBar />
  <main class="page-area">
    <slot />
  </main>
</div>

<style>
  .main-column {
    grid-column: 2;
    grid-row: 1 / -1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
    min-height: 0;
  }

  .page-area {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
</style>
