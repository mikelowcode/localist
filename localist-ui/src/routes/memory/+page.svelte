<script lang="ts">
  import BackendWaitingNote from '$lib/components/BackendWaitingNote.svelte';
  import EpisodesPanel from '$lib/components/EpisodesPanel.svelte';
</script>

<svelte:head>
  <title>Memory — Localist</title>
</svelte:head>

<div class="page-inner">
  <BackendWaitingNote subject="episodes" padded />
  <EpisodesPanel />
</div>

<style>
  .page-inner {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
</style>
