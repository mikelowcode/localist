<script lang="ts">
  // Renders a research-loop workflow's step chain (metadata.workflow_steps —
  // see mcp_tool_dispatcher.py's _run_research_loop() and controller_agent.py's
  // workflow_id/workflow_steps extraction, docs/architecture/18-research-loop.md
  // §18.10). Read-only — no apply/discard actions, unlike DiffBlock; a
  // workflow step is a historical record of what the loop tried, not
  // something to act on.
  export let steps: { tool_name: string; parameters: string; success: boolean; result: string }[];

  function toolLabel(toolName: string): string {
    if (toolName === 'web_search') return 'Search';
    if (toolName === 'url_fetch') return 'Fetch';
    if (toolName === 'research') return 'Research';
    return toolName;
  }
</script>

<div class="workflow-steps">
  {#each steps as step, i (i)}
    <div class="step">
      <div class="step-connector" aria-hidden="true">
        <span class="step-dot" class:step-dot-fail={!step.success} />
        {#if i < steps.length - 1}
          <span class="step-line" />
        {/if}
      </div>
      <div class="step-body">
        <div class="step-header">
          <span class="step-tool">{toolLabel(step.tool_name)}</span>
          {#if !step.success}
            <span class="step-badge step-badge-fail">Failed</span>
          {/if}
        </div>
        {#if step.parameters}
          <p class="step-params">{step.parameters}</p>
        {/if}
        <p class="step-result">{step.result}</p>
      </div>
    </div>
  {/each}
</div>

<style>
  .workflow-steps {
    display: flex;
    flex-direction: column;
  }

  .step {
    display: flex;
    gap: var(--sp-3);
  }

  .step-connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
    width: 12px;
  }

  .step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    margin-top: 5px;
    flex-shrink: 0;
  }
  .step-dot-fail { background: var(--error); }

  .step-line {
    flex: 1;
    width: 1px;
    background: var(--border);
    margin: var(--sp-1) 0;
  }

  .step-body {
    flex: 1;
    min-width: 0;
    padding-bottom: var(--sp-4);
  }

  .step-header {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }

  .step-tool {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .step-badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 1px 6px;
    border-radius: var(--radius-sm);
  }
  .step-badge-fail {
    background: var(--error-dim);
    color: var(--error);
  }

  .step-params {
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    margin: var(--sp-1) 0 0;
    word-break: break-word;
  }

  .step-result {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    margin: var(--sp-2) 0 0;
    white-space: pre-wrap;
    word-break: break-word;
  }
</style>
