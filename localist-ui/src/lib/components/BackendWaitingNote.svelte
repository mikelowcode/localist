<script lang="ts">
  // Shown by pages whose loaders wait on backendReady() (stores/server.ts)
  // until the backend first answers. Hidden once it has; switches to the
  // warning variant after the startup grace, while the page keeps waiting.
  import { health } from '$lib/stores/server';

  /** What will load once the backend answers, e.g. "settings", "episodes". */
  export let subject: string;
  /** Add page padding, for pages without a padded content column. */
  export let padded = false;
</script>

{#if !$health.ever_ok}
  <p class="backend-waiting" class:backend-waiting-late={$health.startup_grace_over} class:padded role="status">
    <span class="spinner" aria-hidden="true" />
    {#if $health.startup_grace_over}
      The Localist backend still hasn't answered — {subject} will load as soon as it does.
      If this persists, check that the backend is running.
    {:else}
      Waiting for the Localist backend to start… {subject} will load as soon as it answers.
    {/if}
  </p>
{/if}

<style>
  .backend-waiting {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    margin: 0;
    color: var(--text-secondary);
  }

  .padded {
    padding: var(--sp-3) var(--sp-4);
    border-bottom: 1px solid var(--border);
  }

  .backend-waiting-late {
    color: var(--warning);
  }

  .spinner {
    display: inline-block;
    flex-shrink: 0;
    width: 11px; height: 11px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
