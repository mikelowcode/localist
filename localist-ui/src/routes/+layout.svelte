<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import StatusBar from '$lib/components/StatusBar.svelte';
  import PreviewsPanel from '$lib/components/PreviewsPanel.svelte';
  import ComposeDocumentPanel from '$lib/components/ComposeDocumentPanel.svelte';
  import { startHealthPolling, stopHealthPolling } from '$lib/stores/server';
  import { theme } from '$lib/stores/theme';
  import { sidebarWidth, sidebarCollapsed } from '$lib/stores/sidebar';
  import { previewsPanelCollapsed } from '$lib/stores/previewsPanel';
  import { composeDocument } from '$lib/stores/composeDocument';
  import { loadAssistantName } from '$lib/stores/assistantName';

  // #app-shell is the CSS grid container that sizes the sidebar/previews/
  // compose-document columns; it lives in app.html outside this component's
  // own DOM subtree, so it's reached directly rather than through a Svelte
  // binding. The 4th (compose document) track is 0px whenever no compose
  // session is active — unlike the previews track, it has no "collapsed but
  // discoverable" strip state, since it's contextual to an in-progress
  // document rather than an always-present utility panel.
  $: if (browser) {
    const shell = document.getElementById('app-shell');
    if (shell) {
      shell.style.gridTemplateColumns =
        `${$sidebarCollapsed ? 0 : $sidebarWidth}px 1fr ` +
        `${$previewsPanelCollapsed ? 'var(--previews-w-collapsed)' : 'var(--previews-w)'} ` +
        `${$composeDocument.active ? 'var(--previews-w)' : '0px'}`;
    }
  }

  onMount(() => {
    // Apply saved theme
    if (browser) {
      document.documentElement.setAttribute('data-theme', $theme);
    }
    startHealthPolling();
    loadAssistantName();
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

<PreviewsPanel />
<ComposeDocumentPanel />

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
