<script lang="ts">
  // Explains, in the chat view itself, why a message won't reach a model —
  // previously the only signal was the appbar chip turning yellow/red, so a
  // first-run user with a reachable backend but a missing chat model just
  // saw chat silently fail (docs/architecture/16-runtime-backend-layer.md
  // §16.21). Driven entirely by the existing 15s health poll; clears itself
  // as soon as a poll comes back healthy.
  import { health, checkHealth, isStillStarting } from '$lib/stores/server';
  import { runtimeBackendLabel } from '$lib/stores/model';

  // Mirrors main.py's _NON_CHAT_MODEL_MARKERS — utility and embedding
  // entries a backend lists alongside its chat models.
  const NON_CHAT_MODEL_MARKERS = ['embed', 'rerank', 'markitdown'];

  $: chatModels = $health.models.filter(
    (m) => !NON_CHAT_MODEL_MARKERS.some((marker) => m.toLowerCase().includes(marker))
  );

  type Problem = 'api-down' | 'backend-down' | 'no-chat-models' | 'model-missing' | null;

  // Nothing until the first poll has answered, or while the backend is still
  // starting (see isStillStarting), so the banner never flashes on a normal
  // launch.
  $: problem = ((): Problem => {
    if ($health.last_checked === null || isStillStarting($health)) return null;
    if (!$health.api_ok) return 'api-down';
    if (!$health.reachable) return 'backend-down';
    if ($health.chat_model_found) return null;
    return chatModels.length === 0 ? 'no-chat-models' : 'model-missing';
  })();

  $: tone = problem === 'api-down' || problem === 'backend-down' ? 'error' : 'warning';
</script>

{#if problem}
  <div class="model-banner" class:tone-error={tone === 'error'} role="status" aria-live="polite">
    <div class="model-banner-text">
      {#if problem === 'api-down'}
        <strong>Can't reach the Localist backend.</strong>
        Try restarting Localist. This message clears on its own once the backend answers.
      {:else if problem === 'backend-down'}
        <strong>Can't reach {$runtimeBackendLabel}</strong>{#if $health.base_url}&nbsp;at
          <code>{$health.base_url}</code>{/if}.
        Start {$runtimeBackendLabel} and this message clears on its own, or switch backends in
        <a href="/settings">Settings</a>.
      {:else if problem === 'no-chat-models'}
        <strong>{$runtimeBackendLabel} is running, but no chat model is loaded.</strong>
        Load a chat model in {$runtimeBackendLabel}, then click Check again.
      {:else if problem === 'model-missing'}
        <strong>
          {$runtimeBackendLabel} is running, but
          {#if $health.chat_model}the chat model <code>{$health.chat_model}</code> isn't available{:else}no chat model is selected{/if}.
        </strong>
        Available: {#each chatModels as m, i}<code>{m}</code>{i < chatModels.length - 1 ? ', ' : ''}{/each}.
        <a href="/settings#chat-model">Choose a chat model in Settings</a>.
      {/if}
    </div>
    <button class="model-banner-retry" on:click={checkHealth} disabled={$health.checking}>
      {$health.checking ? 'Checking…' : 'Check again'}
    </button>
  </div>
{/if}

<style>
  .model-banner {
    display: flex;
    align-items: flex-start;
    gap: var(--sp-3);
    margin: var(--sp-3) var(--sp-8) 0;
    padding: var(--sp-2) var(--sp-3);
    border-radius: var(--radius);
    border: 1px solid color-mix(in srgb, var(--warning) 40%, transparent);
    background: var(--warning-dim);
    color: var(--text-primary);
    font-size: var(--text-sm);
    line-height: 1.5;
    flex-shrink: 0;
  }
  .model-banner.tone-error {
    border-color: color-mix(in srgb, var(--error) 40%, transparent);
    background: var(--error-dim);
  }
  .model-banner-text {
    flex: 1;
    min-width: 0;
  }
  .model-banner strong {
    font-weight: 600;
  }
  .model-banner code {
    font-family: var(--font-mono);
    font-size: 0.95em;
  }
  .model-banner a {
    color: var(--text-accent);
  }
  .model-banner-retry {
    flex-shrink: 0;
    padding: 2px var(--sp-2);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    font-size: var(--text-xs);
    cursor: pointer;
  }
  .model-banner-retry:hover:not(:disabled) {
    color: var(--text-primary);
  }
  .model-banner-retry:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>
