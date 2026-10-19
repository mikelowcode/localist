<script lang="ts">
  import { onMount } from 'svelte';
  import EpisodeFilterPane from '$lib/components/EpisodeFilterPane.svelte';
  import EpisodeList from '$lib/components/EpisodeList.svelte';
  import EpisodeDetailPane from '$lib/components/EpisodeDetailPane.svelte';
  import BackendWaitingNote from '$lib/components/BackendWaitingNote.svelte';
  import { loadEpisodeTurns } from '$lib/stores/episodeBrowser';
  import { backendReady } from '$lib/stores/server';

  // Fetches once — wait for the backend's first answer so a page opened
  // before the packaged sidecar is listening doesn't stay empty (§7,
  // Settings startup race).
  onMount(() => {
    void backendReady().then(loadEpisodeTurns);
  });
</script>

<svelte:head>
  <title>Episodes — Localist</title>
</svelte:head>

<BackendWaitingNote subject="episodes" padded />

<div class="episodes-page">
  <div class="pane pane-filters">
    <EpisodeFilterPane />
  </div>
  <div class="pane pane-list">
    <EpisodeList />
  </div>
  <div class="pane pane-detail">
    <EpisodeDetailPane />
  </div>
</div>

<style>
  .episodes-page {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 220px 340px 1fr;
    overflow: hidden;
  }

  .pane {
    min-height: 0;
    overflow: hidden;
  }

  .pane-filters {
    border-right: 1px solid var(--border);
  }

  @media (max-width: 900px) {
    .episodes-page {
      grid-template-columns: 180px 280px 1fr;
    }
  }
</style>
