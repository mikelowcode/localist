<script lang="ts">
  import ChatPanel from '$lib/components/ChatPanel.svelte';
</script>

<svelte:head>
  <title>Conversation — LORA</title>
</svelte:head>

<div class="page-inner">
  <ChatPanel />
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
