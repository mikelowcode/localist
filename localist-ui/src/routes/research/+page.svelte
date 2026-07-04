<script lang="ts">
  import ResearchView from '$lib/components/ResearchView.svelte';
</script>

<svelte:head>
  <title>Research — Localist</title>
</svelte:head>

<div class="page-inner">
  <ResearchView />
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
