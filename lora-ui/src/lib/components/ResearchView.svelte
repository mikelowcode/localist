<script lang="ts">
  import { tasksStore, submitTask, type Task, type Source } from '$lib/stores/tasks';
  import { tick } from 'svelte';

  let query = '';
  let submitting = false;
  let currentTask: Task | null = null;
  let taskHistory: Task[] = [];

  $: streaming = $tasksStore.streaming;

  // Watch active task and keep local reference in sync
  $: {
    const id = $tasksStore.active_task_id;
    if (id && $tasksStore.tasks[id]) {
      currentTask = $tasksStore.tasks[id];
    }
  }

  async function handleSubmit() {
    const q = query.trim();
    if (!q || submitting || streaming) return;
    submitting = true;
    query = '';
    currentTask = null;

    const context = { query: q };
    await submitTask(q, context);
    submitting = false;
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function confidenceColor(conf: string): string {
    if (conf === 'high')   return 'badge-success';
    if (conf === 'medium') return 'badge-warning';
    return 'badge-muted';
  }

  function formatElapsed(task: Task): string {
    if (!task.completed_at) return '';
    const ms = task.completed_at - task.started_at;
    return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
  }

  // Parse the report into sections for structured display
  function parseSections(report: string): Array<{ heading: string; body: string }> {
    const sections: Array<{ heading: string; body: string }> = [];
    const lines = report.split('\n');
    let current: { heading: string; body: string } | null = null;

    for (const line of lines) {
      if (line.startsWith('## ')) {
        if (current) sections.push(current);
        current = { heading: line.slice(3).trim(), body: '' };
      } else if (current) {
        current.body += line + '\n';
      }
    }
    if (current) sections.push(current);
    return sections;
  }
</script>

<div class="research-panel">
  <!-- Query input -->
  <div class="query-bar">
    <div class="query-wrap" class:active={!streaming}>
      <svg class="query-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <input
        type="text"
        bind:value={query}
        on:keydown={handleKeydown}
        placeholder="Enter a research question…"
        disabled={streaming || submitting}
        class="query-input"
        aria-label="Research query"
      />
      <button
        class="query-btn"
        on:click={handleSubmit}
        disabled={!query.trim() || streaming || submitting}
      >
        {#if streaming || submitting}
          <span class="spinner" aria-hidden="true" />
          <span>Researching…</span>
        {:else}
          Research
        {/if}
      </button>
    </div>
  </div>

  <!-- Results area -->
  <div class="results-area">
    {#if !currentTask}
      <!-- Empty state -->
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
          </svg>
        </div>
        <p class="empty-title">No research results yet</p>
        <p class="empty-sub">Enter a research question above to query the agent corpus.</p>
      </div>
    {:else}
      <!-- Active / completed task view -->
      <div class="result-content">

        <!-- Query header -->
        <div class="result-header fade-in">
          <div class="result-query">
            <span class="badge badge-muted query-badge">query</span>
            <h2 class="query-text">{currentTask.instruction}</h2>
          </div>
          <div class="result-meta">
            {#if currentTask.status === 'planning'}
              <span class="badge badge-accent">
                <span class="dot dot-success dot-pulse" style="margin-right:4px" />
                {currentTask.status_message}
              </span>
            {:else if currentTask.status === 'streaming'}
              <span class="badge badge-accent">
                <span class="dot dot-success dot-pulse" style="margin-right:4px" />
                streaming
              </span>
            {:else if currentTask.status === 'complete'}
              <span class="badge badge-success">complete · {formatElapsed(currentTask)}</span>
            {:else if currentTask.status === 'failed'}
              <span class="badge badge-error">failed</span>
            {/if}
          </div>
        </div>

        <div class="result-grid">
          <!-- Main report -->
          <div class="report-col">
            {#if currentTask.answer}
              {@const sections = parseSections(currentTask.answer)}
              {#if sections.length > 0}
                {#each sections as section, i}
                  <div class="section-card card fade-in" style="animation-delay: {i * 60}ms">
                    <h3 class="section-heading">{section.heading}</h3>
                    <div
                      class="prose section-body"
                      class:cursor-blink={currentTask.status === 'streaming' && i === sections.length - 1}
                    >
                      {section.body.trim()}
                    </div>
                  </div>
                {/each}
              {:else}
                <!-- Fallback: raw text -->
                <div class="card">
                  <div
                    class="prose"
                    class:cursor-blink={currentTask.status === 'streaming'}
                  >
                    {currentTask.answer}
                  </div>
                </div>
              {/if}
            {:else if currentTask.status === 'planning' || currentTask.status === 'streaming'}
              <div class="planning-card card">
                <div class="planning-row">
                  <span class="dot dot-success dot-pulse" />
                  <span class="text-secondary text-sm">{currentTask.status_message}</span>
                </div>
                <div class="skeleton-lines">
                  <div class="skeleton-line" style="width:85%"/>
                  <div class="skeleton-line" style="width:70%"/>
                  <div class="skeleton-line" style="width:90%"/>
                  <div class="skeleton-line" style="width:55%"/>
                </div>
              </div>
            {/if}
          </div>

          <!-- Sidebar: sources + sub-queries -->
          <div class="meta-col">

            <!-- Sub-queries (from task context if available) -->

            <!-- Sources -->
            {#if currentTask.sources && currentTask.sources.length > 0}
              <div class="meta-card card-sm fade-in">
                <h4 class="meta-heading">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                  Sources
                  <span class="meta-count">{currentTask.sources.length}</span>
                </h4>
                <ul class="source-list">
                  {#each currentTask.sources as src}
                    <li class="source-item">
                      <span class="source-type-dot {src.type === 'wiki' ? 'dot-success' : 'dot-warning'}" />
                      <div class="source-info">
                        <span class="source-name">{src.name}</span>
                        <span class="source-meta">
                          <span class="badge badge-muted" style="padding:1px 5px;font-size:10px">{src.type}</span>
                          <span class="source-score">{(src.relevance_score * 100).toFixed(0)}%</span>
                        </span>
                      </div>
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}

            <!-- Task metadata -->
            {#if currentTask.status === 'complete'}
              <div class="meta-card card-sm fade-in">
                <h4 class="meta-heading">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10"/>
                    <polyline points="12 6 12 12 16 14"/>
                  </svg>
                  Task info
                </h4>
                <dl class="task-meta-list">
                  <div class="task-meta-row">
                    <dt>task id</dt>
                    <dd class="text-mono truncate">{currentTask.task_id.slice(0, 8)}…</dd>
                  </div>
                  <div class="task-meta-row">
                    <dt>elapsed</dt>
                    <dd>{formatElapsed(currentTask)}</dd>
                  </div>
                  <div class="task-meta-row">
                    <dt>status</dt>
                    <dd>{currentTask.status}</dd>
                  </div>
                </dl>
              </div>
            {/if}

          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .research-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  /* ── Query bar ─────────────────────────────── */
  .query-bar {
    flex-shrink: 0;
    padding: var(--sp-5) var(--sp-8) var(--sp-4);
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }

  .query-wrap {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-3) var(--sp-3) var(--sp-3) var(--sp-4);
    transition: border-color var(--dur-fast) var(--ease);
    max-width: 860px;
  }

  .query-wrap:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .query-icon {
    color: var(--text-tertiary);
    flex-shrink: 0;
  }

  .query-input {
    flex: 1;
    background: none;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-size: var(--text-base);
    padding: 0;
  }

  .query-input:disabled { opacity: 0.5; }

  .query-btn {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    background: var(--accent);
    color: #fff;
    font-size: var(--text-sm);
    font-weight: 500;
    border-radius: var(--radius);
    flex-shrink: 0;
    min-width: 100px;
    justify-content: center;
    transition: background var(--dur-fast) var(--ease);
  }

  .query-btn:hover:not(:disabled) { background: #6fa3ff; }

  .query-btn:disabled {
    background: var(--bg-active);
    color: var(--text-muted);
    opacity: 1;
  }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 11px; height: 11px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Results ───────────────────────────────── */
  .results-area {
    flex: 1;
    overflow-y: auto;
    padding: var(--sp-6) var(--sp-8);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    padding: var(--sp-16);
    text-align: center;
    color: var(--text-tertiary);
  }

  .empty-icon { opacity: 0.25; margin-bottom: var(--sp-2); }
  .empty-title { font-size: var(--text-lg); font-weight: 500; color: var(--text-secondary); }
  .empty-sub { font-size: var(--text-sm); max-width: 320px; line-height: 1.6; color: var(--text-tertiary); }

  /* Result layout */
  .result-content { display: flex; flex-direction: column; gap: var(--sp-6); }

  .result-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--sp-4);
  }

  .result-query {
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
    flex: 1;
  }

  .query-badge { align-self: flex-start; }
  .query-text { font-size: var(--text-xl); font-weight: 600; line-height: 1.3; }
  .result-meta { flex-shrink: 0; }

  /* Two-column grid */
  .result-grid {
    display: grid;
    grid-template-columns: 1fr 260px;
    gap: var(--sp-6);
    align-items: start;
  }

  @media (max-width: 900px) {
    .result-grid { grid-template-columns: 1fr; }
  }

  /* Section cards */
  .section-card { margin-bottom: var(--sp-4); }
  .section-heading {
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: var(--text-xs);
    margin-bottom: var(--sp-3);
  }
  .section-body {
    font-size: var(--text-sm);
    white-space: pre-wrap;
    line-height: 1.75;
  }

  /* Planning skeleton */
  .planning-card { display: flex; flex-direction: column; gap: var(--sp-4); }
  .planning-row { display: flex; align-items: center; gap: var(--sp-2); }
  .skeleton-lines { display: flex; flex-direction: column; gap: var(--sp-2); }
  .skeleton-line {
    height: 12px;
    background: var(--bg-active);
    border-radius: var(--radius-sm);
    animation: pulse 1.5s ease-in-out infinite;
  }

  /* Meta sidebar */
  .meta-col { display: flex; flex-direction: column; gap: var(--sp-4); }

  .meta-card {
    display: flex;
    flex-direction: column;
    gap: var(--sp-3);
  }

  .meta-heading {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-secondary);
  }

  .meta-count {
    margin-left: auto;
    background: var(--bg-active);
    color: var(--text-tertiary);
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 100px;
    font-family: var(--font-mono);
  }

  /* Source list */
  .source-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .source-item {
    display: flex;
    align-items: flex-start;
    gap: var(--sp-2);
  }

  .source-type-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 5px;
  }
  .dot-success { background: var(--success); }
  .dot-warning { background: var(--warning); }

  .source-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .source-name {
    font-size: var(--text-xs);
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: var(--font-mono);
  }

  .source-meta {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }

  .source-score {
    font-size: 10px;
    color: var(--text-tertiary);
    font-family: var(--font-mono);
  }

  /* Task meta DL */
  .task-meta-list { display: flex; flex-direction: column; gap: var(--sp-2); }
  .task-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--sp-2);
    font-size: var(--text-xs);
  }
  .task-meta-row dt {
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    flex-shrink: 0;
  }
  .task-meta-row dd {
    color: var(--text-secondary);
    font-family: var(--font-mono);
    text-align: right;
    min-width: 0;
  }
</style>
