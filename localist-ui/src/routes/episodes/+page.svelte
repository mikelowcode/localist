<script lang="ts">
  import { onMount } from 'svelte';
  import EpisodeFilterPane from '$lib/components/EpisodeFilterPane.svelte';
  import EpisodeList from '$lib/components/EpisodeList.svelte';
  import EpisodeDetailPane from '$lib/components/EpisodeDetailPane.svelte';
  import { loadEpisodeTurns } from '$lib/stores/episodeBrowser';

  onMount(() => {
    loadEpisodeTurns();
  });
</script>

<svelte:head>
  <title>Episodes — Localist</title>
</svelte:head>

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
