<script lang="ts">
  import EpisodesPanel from '$lib/components/EpisodesPanel.svelte';
</script>

<svelte:head>
  <title>Memory — Localist</title>
</svelte:head>

<div class="page-inner">
  <EpisodesPanel />
</div>

<style>
  .page-inner {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    height: 100%;
  }
</style>
