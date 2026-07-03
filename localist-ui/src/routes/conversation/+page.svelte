<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { currentConversationId } from '$lib/stores/conversation';

  // Redirect client-side (onMount + goto), not via a +page.ts load()/redirect() —
  // currentConversationId is backed by localStorage, which only exists in the
  // browser. A universal load() also runs during SSR, where it would see a
  // fresh, non-persisted id and redirect to the wrong conversation.
  onMount(() => {
    goto(`/conversation/${get(currentConversationId)}`, { replaceState: true });
  });
</script>
