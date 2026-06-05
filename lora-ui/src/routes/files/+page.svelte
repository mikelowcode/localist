<script lang="ts">
  import FileBrowser from '$lib/components/FileBrowser.svelte';
</script>

<svelte:head>
  <title>Files — LORA</title>
</svelte:head>

<div class="page-inner">
  <FileBrowser />
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
