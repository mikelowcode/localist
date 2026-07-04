<script lang="ts">
  import FileBrowser from '$lib/components/FileBrowser.svelte';
</script>

<svelte:head>
  <title>Files — Localist</title>
</svelte:head>

<div class="page-inner">
  <FileBrowser />
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
