<script lang="ts">
  import ResearchView from '$lib/components/ResearchView.svelte';
</script>

<svelte:head>
  <title>Research — LORA</title>
</svelte:head>

<div class="page-inner">
  <ResearchView />
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
