<script lang="ts">
  // Renders one turn's proposed wiki diff(s) with review-then-apply
  // controls (Apply / Discard). Extracted from ChatPanel.svelte so the
  // Episode Browsing UI's detail pane (episode-browsing-ui-plan.md Phase 5)
  // can reuse the exact same rendering/apply logic instead of duplicating
  // it — the multi-diff case (a turn proposing 2+ page diffs) was already
  // fully supported end to end before this extraction (see
  // docs/architecture/17-wiki-agent-diff-target.md §17.11); this component
  // just gives that logic a second call site.
  //
  // Callers own their own source-of-truth sync after a successful apply
  // (ChatPanel syncs chatHistoryStore + tasksStore; the episode detail pane
  // syncs episodeBrowser's episodeTurns) via the onApplied callback —
  // this component only tracks its own local applying/error/discarded UI
  // state, keyed by page_name (each instance is already scoped to one
  // taskId).
  import { applyDiff } from '$lib/stores/wiki';

  export let taskId: string;
  export let diffs: { page_name: string; diff: string; status: 'pending' | 'applied' }[];
  export let onApplied: (pageName: string) => void = () => {};

  interface DiffUiState { applying: boolean; error: string | null; discarded: boolean; }
  let diffState: Record<string, DiffUiState> = {};

  function diffLineClass(line: string): string {
    if (line.startsWith('+')) return 'diff-line-add';
    if (line.startsWith('-')) return 'diff-line-del';
    if (line.startsWith('@@')) return 'diff-line-hunk';
    return 'diff-line-ctx';
  }

  async function handleApply(diff: { page_name: string; diff: string }) {
    const key = diff.page_name;
    diffState = { ...diffState, [key]: { applying: true, error: null, discarded: false } };

    const result = await applyDiff(taskId, diff.page_name, diff.diff);

    if (!result.success) {
      diffState = {
        ...diffState,
        [key]: { applying: false, error: result.error ?? 'Apply failed.', discarded: false }
      };
      return;
    }

    diffState = { ...diffState, [key]: { applying: false, error: null, discarded: false } };
    onApplied(diff.page_name);
  }

  function handleDiscard(diff: { page_name: string }) {
    const key = diff.page_name;
    diffState = { ...diffState, [key]: { applying: false, error: null, discarded: true } };
  }
</script>

{#each diffs as diff (diff.page_name)}
  {@const state = diffState[diff.page_name]}
  {#if !state?.discarded}
    <div class="diff-block">
      <div class="diff-block-header">
        <span class="diff-page-name">📄 {diff.page_name}.md</span>
        {#if diff.status === 'applied'}
          <span class="diff-badge diff-badge-applied">✓ Applied</span>
        {:else if state?.applying}
          <span class="diff-badge diff-badge-pending">Applying…</span>
        {:else}
          <span class="diff-badge diff-badge-pending">Pending review</span>
        {/if}
      </div>
      <div class="diff-body">
        {#each diff.diff.split('\n') as line}
          <div class="diff-line {diffLineClass(line)}">{line || ' '}</div>
        {/each}
      </div>
      {#if state?.error}
        <p class="diff-error">{state.error}</p>
      {/if}
      {#if diff.status !== 'applied'}
        <div class="diff-actions">
          <button
            class="diff-btn diff-btn-apply"
            on:click={() => handleApply(diff)}
            disabled={state?.applying}
          >
            {state?.applying ? 'Applying…' : 'Apply'}
          </button>
          <button
            class="diff-btn diff-btn-discard"
            on:click={() => handleDiscard(diff)}
            disabled={state?.applying}
          >
            Discard
          </button>
        </div>
      {/if}
    </div>
  {/if}
{/each}

<style>
  .diff-block {
    margin-top: var(--sp-3);
    padding-top: var(--sp-3);
    border-top: 1px solid var(--border-soft);
  }

  .diff-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
    margin-bottom: var(--sp-2);
  }

  .diff-page-name {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-secondary);
    font-weight: 600;
  }

  .diff-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 7px;
    border-radius: 999px;
    letter-spacing: 0.03em;
    font-weight: 500;
    border: 1px solid transparent;
    flex-shrink: 0;
  }

  .diff-badge-pending { background: var(--bg-active); color: var(--text-tertiary); border-color: var(--border); }
  .diff-badge-applied { background: #1a2a1a;          color: #7ecf7e;             border-color: #2d4a2d; }

  .diff-body {
    max-height: 320px;
    overflow-y: auto;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-2) 0;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.5;
  }

  .diff-line {
    padding: 0 var(--sp-3);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .diff-line-add  { background: rgba(126, 207, 126, 0.12); color: #7ecf7e; }
  .diff-line-del  { background: rgba(207, 126, 126, 0.12); color: #cf7e7e; }
  .diff-line-hunk { color: var(--text-tertiary); }
  .diff-line-ctx  { color: var(--text-secondary); }

  .diff-error {
    font-size: var(--text-xs);
    color: var(--error);
    font-family: var(--font-mono);
    margin-top: var(--sp-2);
  }

  .diff-actions {
    display: flex;
    gap: var(--sp-2);
    margin-top: var(--sp-2);
  }

  .diff-btn {
    font-size: var(--text-xs);
    font-weight: 500;
    padding: var(--sp-1) var(--sp-3);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }

  .diff-btn:disabled { opacity: 0.5; }

  .diff-btn-apply {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .diff-btn-apply:hover:not(:disabled) { background: #6fa3ff; }

  .diff-btn-discard {
    background: var(--bg-raised);
    color: var(--text-secondary);
  }
  .diff-btn-discard:hover:not(:disabled) { background: var(--bg-active); }
</style>
