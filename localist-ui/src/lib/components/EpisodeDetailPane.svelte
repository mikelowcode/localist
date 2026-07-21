<script lang="ts">
  import { episodeTurns, selectedEpisodeId } from '$lib/stores/episodeBrowser';
  import ChartRenderer from '$lib/components/ChartRenderer.svelte';
  import DiffBlock from '$lib/components/DiffBlock.svelte';
  import WorkflowSteps from '$lib/components/WorkflowSteps.svelte';
  import EpisodeAnnotations from '$lib/components/EpisodeAnnotations.svelte';

  $: selected = $episodeTurns.find((t) => t.id === $selectedEpisodeId) ?? null;

  function formatDate(ts: number): string {
    return new Date(ts * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit'
    });
  }

  // Mirrors ChatPanel.svelte's handleDiffApplied, but syncs episodeTurns
  // (this pane's own source of truth) instead of chatHistoryStore/tasksStore.
  function handleDiffApplied(taskId: string, pageName: string) {
    episodeTurns.update((turns) =>
      turns.map((t) => {
        if (t.task_id !== taskId || !t.metadata?.pending_diffs) return t;
        return {
          ...t,
          metadata: {
            ...t.metadata,
            pending_diffs: t.metadata.pending_diffs.map((d) =>
              d.page_name === pageName ? { ...d, status: 'applied' as const } : d
            )
          }
        };
      })
    );
  }
</script>

<div class="detail-pane">
  {#if selected}
    <div class="detail-header">
      <span class="role-badge role-{selected.role}">{selected.role === 'user' ? 'You' : 'LORA'}</span>
      <span class="detail-date">{formatDate(selected.created_at)}</span>
    </div>

    <div class="detail-content">{selected.content}</div>

    {#if selected.metadata?.chart}
      <div class="detail-section">
        <h3 class="detail-section-title">Chart</h3>
        <ChartRenderer config={selected.metadata.chart.chart_config} />
      </div>
    {/if}

    {#if selected.metadata?.pending_diffs && selected.metadata.pending_diffs.length > 0}
      <div class="detail-section">
        <h3 class="detail-section-title">
          Proposed diff{selected.metadata.pending_diffs.length > 1 ? 's' : ''}
        </h3>
        <DiffBlock
          taskId={selected.task_id}
          diffs={selected.metadata.pending_diffs}
          onApplied={(pageName) => handleDiffApplied(selected?.task_id ?? '', pageName)}
        />
      </div>
    {/if}

    {#if selected.metadata?.workflow_steps && selected.metadata.workflow_steps.length > 0}
      <div class="detail-section">
        <h3 class="detail-section-title">Research steps</h3>
        <WorkflowSteps steps={selected.metadata.workflow_steps} />
      </div>
    {/if}

    {#if selected.sources && selected.sources.length > 0}
      <div class="detail-section">
        <h3 class="detail-section-title">Sources</h3>
        <div class="source-list">
          {#each selected.sources as src}
            <span class="source-chip">{src.name ?? src.path}</span>
          {/each}
        </div>
      </div>
    {/if}

    <div class="detail-section">
      <h3 class="detail-section-title">Metadata</h3>
      <div class="meta-grid">
        <span class="meta-key">conversation</span>
        <span class="meta-val">{selected.conversation_title ?? selected.conversation_id}</span>
        {#if selected.score !== null}
          <span class="meta-key">match score</span>
          <span class="meta-val">{Math.round(selected.score * 100)}%</span>
        {/if}
      </div>
    </div>

    <div class="detail-section">
      <h3 class="detail-section-title">Related memory</h3>
      {#key selected.task_id}
        <EpisodeAnnotations taskId={selected.task_id} />
      {/key}
    </div>
  {:else}
    <div class="detail-empty">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true" style="opacity:0.2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      <p>Select an episode to view its details</p>
    </div>
  {/if}
</div>

<style>
  .detail-pane {
    display: flex;
    flex-direction: column;
    gap: var(--sp-5);
    height: 100%;
    overflow-y: auto;
    padding: var(--sp-6);
  }

  .detail-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    color: var(--text-tertiary);
    font-size: var(--text-sm);
  }

  .detail-header {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
  }

  .role-badge {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    letter-spacing: 0.03em;
  }
  .role-user { background: var(--accent-dim); color: var(--accent); }
  .role-assistant { background: var(--success-dim); color: var(--success); }

  .detail-date {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-muted);
  }

  .detail-content {
    font-size: var(--text-base);
    color: var(--text-primary);
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .detail-section {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
    padding-top: var(--sp-4);
    border-top: 1px solid var(--border-soft);
  }

  .detail-section-title {
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-tertiary);
  }

  .source-list {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-2);
  }

  .source-chip {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    background: var(--bg-active);
    color: var(--text-secondary);
  }

  .meta-grid {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: var(--sp-1) var(--sp-3);
    font-size: var(--text-xs);
    font-family: var(--font-mono);
  }

  .meta-key { color: var(--text-muted); }
  .meta-val { color: var(--text-secondary); }
</style>
