<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { get } from 'svelte/store';
  import { tasksStore, submitTask, markDiffApplied, type PendingDiff } from '$lib/stores/tasks';
  import { chatHistoryStore, type Turn } from '$lib/stores/chatHistory';
  import { currentConversationId, isFirstTurnOfConversation } from '$lib/stores/conversation';
  import { applyDiff } from '$lib/stores/wiki';
  import MarkdownRenderer from '$lib/components/MarkdownRenderer.svelte';

  let instruction = '';
  let messagesEl: HTMLElement;
  let inputEl: HTMLTextAreaElement;
  let submitting = false;

  // Attached session files — display-only; source of truth is the backend cache.
  interface AttachedFile {
    filename: string;
    tokenEstimate: number;
  }
  let attachedFiles: AttachedFile[] = [];
  let fileInputEl: HTMLInputElement;
  let attachError: string | null = null;

  const ALLOWED_EXTENSIONS = new Set([
    '.md', '.txt', '.py', '.ts', '.js', '.svelte', '.json',
    '.yaml', '.yml', '.toml', '.sh', '.env', '.csv', '.xml',
    '.html', '.css', '.rs', '.go', '.rb', '.java', '.c', '.cpp',
    '.h', '.hpp', '.sql',
  ]);

  async function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';   // reset so the same file can be re-selected after removal
    if (!file) return;

    attachError = null;

    // Client-side extension check (defence in depth — server enforces the real gate)
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      attachError = `File type '${ext}' is not supported.`;
      return;
    }

    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch('/api/chat/files', { method: 'POST', body: form });
      const body = await res.json();
      if (!res.ok) {
        attachError = body.detail ?? `Upload failed (HTTP ${res.status}).`;
        return;
      }
      attachedFiles = [...attachedFiles, { filename: body.filename, tokenEstimate: body.token_estimate }];
    } catch (err) {
      attachError = 'Could not reach the server. Is the backend running?';
    }
  }

  async function removeAttachedFile(filename: string) {
    try {
      await fetch(`/api/chat/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    } catch {
      // Best-effort — backend cache may already be clear on restart
    }
    attachedFiles = attachedFiles.filter(f => f.filename !== filename);
    attachError = null;
  }

  $: activeTask = $tasksStore.active_task_id
    ? $tasksStore.tasks[$tasksStore.active_task_id]
    : null;

  // Reactively update the last assistant turn from the streaming store.
  // Uses get() to read the store without subscribing, avoiding an infinite loop.
  $: if (activeTask) {
    const current = get(chatHistoryStore);
    const idx = current.findLastIndex(
      (t) => t.role === 'assistant' && t.task_id === activeTask!.task_id
    );
    if (idx >= 0) {
      chatHistoryStore.update((turns) => {
        const next = [...turns];
        next[idx] = {
          ...next[idx],
          content:        activeTask!.answer,
          status:         activeTask!.status,
          sources:        activeTask!.sources,
          metadata:       activeTask!.metadata,
          status_message: activeTask!.status_message
        };
        return next;
      });
      scrollToBottom();
    }
  }

  // Pick up tasks injected by ingestFile() before navigation.
  // Uses get() to read the store without subscribing, avoiding an infinite loop.
  $: {
    const state = $tasksStore;
    if (state.active_task_id) {
      const task = state.tasks[state.active_task_id];
      if (
        task &&
        task.source === 'ingest' &&
        task.status === 'complete' &&
        !get(chatHistoryStore).some((t) => t.task_id === task.task_id)
      ) {
        chatHistoryStore.update((turns) => [
          ...turns,
          {
            role:      'user',
            content:   task.instruction,
            timestamp: task.started_at,
          },
          {
            role:           'assistant',
            content:        task.answer,
            task_id:        task.task_id,
            timestamp:      task.started_at + 1,  // +1ms ensures unique key
            status:         task.status,
            sources:        task.sources,
            metadata:       task.metadata,
            status_message: task.status_message,
          },
        ]);
        scrollToBottom();
      }
    }
  }

  async function scrollToBottom() {
    await tick();
    if (messagesEl) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  async function handleSubmit() {
    const text = instruction.trim();
    if (!text || submitting || $tasksStore.finalizing) return;

    submitting = true;
    instruction = '';
    autoResizeTextarea();

    const task_id = crypto.randomUUID();
    const now = Date.now();

    const conversationId = get(currentConversationId);
    let conversationTitle: string | undefined;
    if (get(isFirstTurnOfConversation)) {
      conversationTitle = text.length > 60 ? text.slice(0, 60) + '…' : text;
      isFirstTurnOfConversation.set(false);
    }

    chatHistoryStore.update((turns) => [
      ...turns,
      { role: 'user', content: text, task_id, timestamp: now }
    ]);
    await scrollToBottom();

    chatHistoryStore.update((turns) => [
      ...turns,
      {
        role: 'assistant',
        content: '',
        task_id,
        timestamp: now + 1,
        status: 'planning',
        status_message: 'Planning…',
        sources: []
      }
    ]);
    await scrollToBottom();

    await submitTask(text, {}, task_id, conversationId, conversationTitle);

    submitting = false;
    inputEl?.focus();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function autoResizeTextarea() {
    if (!inputEl) return;
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
  }

  function formatTime(ts: number): string {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Priority 3 in planner.py is a generic tool-signal priority (web_search,
  // file_op, or url_fetch) — not web-search-specific. Pick the label from
  // whichever tool actually fired instead of hardcoding "Web search".
  // file_op_deferred counts as a file_op match even though tools_fired is
  // empty for it (the write happens after generation completes — see
  // controller_agent.py's _execute_plan Step 7b — so it never lands in
  // plan.tools_to_call). Falls back to a generic label when none/multiple match.
  function p3Provenance(p: {
    tools_fired?:      string[];
    file_op_deferred?: boolean;
  }): { label: string; cls: string } {
    const fired       = p.tools_fired ?? [];
    const hasWebSearch = fired.includes('web_search');
    const hasFileOp    = fired.includes('file_op') || !!p.file_op_deferred;
    const hasUrlFetch  = fired.includes('url_fetch');
    const matchCount   = [hasWebSearch, hasFileOp, hasUrlFetch].filter(Boolean).length;

    if (matchCount === 1) {
      if (hasWebSearch) return { label: 'P3 · Web search',     cls: 'prov-web' };
      if (hasFileOp)    return { label: 'P3 · File operation', cls: 'prov-tool' };
      if (hasUrlFetch)  return { label: 'P3 · Page fetch',     cls: 'prov-tool' };
    }
    return { label: 'P3 · Tool', cls: 'prov-tool' };
  }

  // ── Review-then-apply wiki diffs ──────────────────────────────
  // Discard is deliberately client-only/ephemeral (no backend call, no
  // persistence) — keyed by "taskId:pageName" in local component state so
  // it doesn't fight the reactive metadata-sync block above, which only
  // ever copies activeTask.metadata (server-known state) onto chat turns.
  interface DiffUiState {
    applying: boolean;
    error: string | null;
    discarded: boolean;
  }
  let diffState: Record<string, DiffUiState> = {};

  function diffKey(taskId: string, pageName: string): string {
    return `${taskId}:${pageName}`;
  }

  function diffLineClass(line: string): string {
    if (line.startsWith('+')) return 'diff-line-add';
    if (line.startsWith('-')) return 'diff-line-del';
    if (line.startsWith('@@')) return 'diff-line-hunk';
    return 'diff-line-ctx';
  }

  async function handleApplyDiff(turn: Turn, diff: PendingDiff) {
    if (!turn.task_id) return;
    const key = diffKey(turn.task_id, diff.page_name);
    diffState = { ...diffState, [key]: { applying: true, error: null, discarded: false } };

    const result = await applyDiff(turn.task_id, diff.page_name, diff.diff);

    if (!result.success) {
      diffState = {
        ...diffState,
        [key]: { applying: false, error: result.error ?? 'Apply failed.', discarded: false }
      };
      return;
    }

    // Source of truth for rendering (works even if the task isn't tracked
    // in tasksStore any more — e.g. a reloaded historical conversation).
    chatHistoryStore.update((turns) =>
      turns.map((t) => {
        if (t.task_id !== turn.task_id || !t.metadata?.pending_diffs) return t;
        return {
          ...t,
          metadata: {
            ...t.metadata,
            pending_diffs: t.metadata.pending_diffs.map((d) =>
              d.page_name === diff.page_name ? { ...d, status: 'applied' as const } : d
            )
          }
        };
      })
    );
    // Also patch tasksStore's copy so a later reactive re-sync (see the
    // activeTask block above) doesn't stomp this back to "pending".
    markDiffApplied(turn.task_id, diff.page_name);

    diffState = { ...diffState, [key]: { applying: false, error: null, discarded: false } };
  }

  function handleDiscardDiff(turn: Turn, diff: PendingDiff) {
    if (!turn.task_id) return;
    const key = diffKey(turn.task_id, diff.page_name);
    diffState = { ...diffState, [key]: { applying: false, error: null, discarded: true } };
  }

  onMount(() => {
    inputEl?.focus();
  });
</script>

<div class="chat-panel">
  <!-- Messages -->
  <div class="messages" bind:this={messagesEl}>
    {#if $chatHistoryStore.length === 0}
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <p class="empty-title">Start a conversation</p>
        <p class="empty-sub">Ask LORA a question or give an instruction to get started.</p>
      </div>
    {:else}
      {#each $chatHistoryStore as turn (turn.timestamp)}
        <div class="turn turn-{turn.role} fade-in">
          <div class="turn-meta">
            <span class="turn-role">{turn.role === 'user' ? 'You' : 'LORA'}</span>
            <span class="turn-time">{formatTime(turn.timestamp)}</span>
          </div>

          <div class="bubble bubble-{turn.role}">
            {#if turn.role === 'assistant'}
              {#if turn.status === 'planning' || turn.status === 'streaming'}
                <span class="status-line">
                  <span class="dot dot-success dot-pulse" style="margin-right:6px" />
                  {turn.status_message ?? 'Working…'}
                </span>
              {/if}

              {#if turn.content}
                <MarkdownRenderer
                  content={turn.content}
                  streaming={turn.status === 'streaming'}
                />
              {:else if turn.status === 'planning'}
                <span class="placeholder-pulse">···</span>
              {/if}

              {#if turn.status === 'complete' && turn.metadata}
                {@const p = turn.metadata}
                <div class="provenance-bar">
                  <span class="prov-route">
                    {#if p.priority === 1}
                      <span class="prov-chip prov-direct">P1 · Direct</span>
                    {:else if p.priority === 2}
                      <span class="prov-chip prov-memory">P2 · Memory write</span>
                    {:else if p.priority === 3}
                      {@const p3 = p3Provenance(p)}
                      <span class="prov-chip {p3.cls}">{p3.label}</span>
                    {:else if p.priority === 4}
                      <span class="prov-chip prov-rag">P4 · Vault</span>
                    {:else if p.priority === 5}
                      <span class="prov-chip prov-episodic">P5 · Episodic</span>
                    {:else}
                      <span class="prov-chip prov-default">P6 · Inference</span>
                    {/if}
                  </span>
                  {#if p.tools_fired && p.tools_fired.length > 0}
                    {#each p.tools_fired as tool}
                      <span class="prov-chip prov-tool">⚙ {tool}</span>
                    {/each}
                  {/if}
                  {#if p.file_op_deferred && !(p.tools_fired && p.tools_fired.includes('file_op'))}
                    <span class="prov-chip prov-tool">⚙ file_op</span>
                  {/if}
                  {#if p.fetch_episodic}
                    <span class="prov-chip prov-episodic-mem">◎ episodic</span>
                  {/if}
                  {#if p.grounded}
                    <span class="prov-chip prov-grounded">◈ grounded</span>
                  {/if}
                </div>
              {/if}

              {#if turn.sources && turn.sources.length > 0 && turn.status === 'complete'}
                <div class="sources-row">
                  {#each turn.sources as src}
                    <span class="badge badge-muted source-badge" title={src.path}>
                      {src.type === 'wiki' ? '📄' : src.type === 'session' ? '📎' : '📁'} {src.name}
                    </span>
                  {/each}
                </div>
              {/if}

              {#if turn.status === 'complete' && turn.metadata?.pending_diffs}
                {#each turn.metadata.pending_diffs as diff (diff.page_name)}
                  {@const key = diffKey(turn.task_id ?? '', diff.page_name)}
                  {@const state = diffState[key]}
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
                            on:click={() => handleApplyDiff(turn, diff)}
                            disabled={state?.applying}
                          >
                            {state?.applying ? 'Applying…' : 'Apply'}
                          </button>
                          <button
                            class="diff-btn diff-btn-discard"
                            on:click={() => handleDiscardDiff(turn, diff)}
                            disabled={state?.applying}
                          >
                            Discard
                          </button>
                        </div>
                      {/if}
                    </div>
                  {/if}
                {/each}
              {/if}

              {#if turn.status === 'failed'}
                <span class="error-line">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                  </svg>
                  Task failed.
                </span>
              {/if}
            {:else}
              {turn.content}
            {/if}
          </div>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Input bar -->
  <div class="input-bar">
    <div class="input-wrap">
      <button
        class="attach-btn"
        on:click={() => fileInputEl.click()}
        disabled={submitting}
        aria-label="Attach a file"
        title="Attach file"
        type="button"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
        </svg>
      </button>
      <textarea
        bind:this={inputEl}
        bind:value={instruction}
        on:keydown={handleKeydown}
        on:input={autoResizeTextarea}
        placeholder="Ask LORA a question or give an instruction…"
        rows="1"
        class="chat-input"
        aria-label="Message input"
      />
      <button
        class="send-btn"
        on:click={handleSubmit}
        disabled={!instruction.trim() || $tasksStore.finalizing}
        aria-label="Send message"
        title={$tasksStore.finalizing ? 'Saving previous turn — send available shortly' : 'Send (Enter)'}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
    {#if attachedFiles.length > 0 || attachError}
      <div class="attached-files">
        {#each attachedFiles as f (f.filename)}
          <span class="file-pill">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
            <span class="file-pill-name" title={f.filename}>{f.filename}</span>
            <span class="file-pill-tokens">~{f.tokenEstimate.toLocaleString()}t</span>
            <button
              class="file-pill-remove"
              on:click={() => removeAttachedFile(f.filename)}
              aria-label="Remove {f.filename}"
              title="Remove"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </span>
        {/each}
        {#if attachError}
          <span class="attach-error">{attachError}</span>
        {/if}
      </div>
    {/if}
    <p class="input-hint">
      <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
    </p>
    <input
      bind:this={fileInputEl}
      type="file"
      accept={[...ALLOWED_EXTENSIONS].join(',')}
      style="display:none"
      on:change={handleFileSelect}
      aria-hidden="true"
      tabindex="-1"
    />
  </div>
</div>

<style>
  .chat-panel {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* ── Messages ─────────────────────────────── */
  .messages {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-6) var(--sp-8);
    display: flex;
    flex-direction: column;
    gap: var(--sp-6);
    scroll-behavior: smooth;
  }

  /* Empty state */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    text-align: center;
    padding: var(--sp-16) var(--sp-8);
    color: var(--text-tertiary);
  }
  .empty-icon {
    opacity: 0.3;
    margin-bottom: var(--sp-2);
  }
  .empty-title {
    font-size: var(--text-md);
    font-weight: 500;
    color: var(--text-secondary);
  }
  .empty-sub {
    font-size: var(--text-sm);
    max-width: 340px;
    line-height: 1.6;
    color: var(--text-tertiary);
  }

  /* Turn */
  .turn {
    display: flex;
    flex-direction: column;
    gap: var(--sp-1);
    max-width: 780px;
  }

  .turn-user {
    align-self: flex-end;
    align-items: flex-end;
  }

  .turn-assistant {
    align-self: flex-start;
    align-items: flex-start;
  }

  .turn-meta {
    display: flex;
    align-items: baseline;
    gap: var(--sp-2);
    padding: 0 var(--sp-2);
  }

  .turn-role {
    font-size: var(--text-xs);
    font-weight: 600;
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .turn-time {
    font-size: var(--text-xs);
    color: var(--text-muted);
    font-family: var(--font-mono);
    opacity: 0;
    transition: opacity var(--dur-base) var(--ease);
  }

  .turn:hover .turn-time { opacity: 1; }

  /* Bubbles */
  .bubble {
    padding: var(--sp-3) var(--sp-4);
    border-radius: var(--radius-lg);
    font-size: var(--text-md);
    line-height: 1.65;
    max-width: 680px;
    word-break: break-word;
  }

  .bubble-user {
    background: var(--accent-dim);
    border: 1px solid var(--accent-mid);
    color: var(--text-primary);
    border-bottom-right-radius: var(--radius-sm);
  }

  .bubble-assistant {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-bottom-left-radius: var(--radius-sm);
    min-width: 120px;
  }

  .status-line {
    display: flex;
    align-items: center;
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    color: var(--text-tertiary);
    margin-bottom: var(--sp-2);
  }

  .placeholder-pulse {
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    letter-spacing: 0.15em;
    animation: pulse 1.5s ease-in-out infinite;
  }

  .error-line {
    display: flex;
    align-items: center;
    gap: var(--sp-1);
    font-size: var(--text-xs);
    color: var(--error);
    font-family: var(--font-mono);
    margin-top: var(--sp-2);
  }

  .sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    margin-top: var(--sp-3);
    padding-top: var(--sp-3);
    border-top: 1px solid var(--border-soft);
  }

  .source-badge {
    font-size: 11px;
    cursor: default;
  }

  /* ── Wiki diff review block ───────────────── */
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

  /* ── Input bar ────────────────────────────── */
  .input-bar {
    flex-shrink: 0;
    padding: var(--sp-4) var(--sp-8) var(--sp-5);
    border-top: 1px solid var(--border);
    background: var(--bg);
  }

  .input-wrap {
    display: flex;
    align-items: flex-end;
    gap: var(--sp-2);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-3) var(--sp-3) var(--sp-3) var(--sp-4);
    transition: border-color var(--dur-fast) var(--ease);
  }

  .input-wrap:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .chat-input {
    flex: 1;
    background: none;
    border: none;
    outline: none;
    resize: none;
    color: var(--text-primary);
    font-size: var(--text-base);
    line-height: 1.6;
    padding: 0;
    min-height: 22px;
    max-height: 180px;
    overflow-y: auto;
  }

  .chat-input:disabled { opacity: 0.5; }
  .chat-input::placeholder { color: var(--text-tertiary); }

  .send-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    flex-shrink: 0;
    transition:
      background var(--dur-fast) var(--ease),
      transform var(--dur-fast) var(--ease),
      opacity var(--dur-fast) var(--ease);
  }

  .send-btn:hover:not(:disabled) { background: #6fa3ff; }

  .send-btn:disabled {
    background: var(--bg-active);
    color: var(--text-muted);
    opacity: 1;
  }

  .input-hint {
    margin-top: var(--sp-2);
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-align: right;
  }

  .input-hint kbd {
    font-family: var(--font-mono);
    font-size: 10px;
    background: var(--bg-active);
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid var(--border);
    color: var(--text-tertiary);
  }

  .provenance-bar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-1);
    margin-top: var(--sp-3);
    padding-top: var(--sp-2);
    border-top: 1px solid var(--border-soft);
  }

  .prov-chip {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 7px;
    border-radius: 999px;
    letter-spacing: 0.03em;
    font-weight: 500;
    border: 1px solid transparent;
  }

  .prov-direct   { background: var(--bg-active); color: var(--text-tertiary); border-color: var(--border); }
  .prov-memory   { background: #1a2a1a;           color: #7ecf7e; border-color: #2d4a2d; }
  .prov-web      { background: #1a1a2e;           color: #7ea8cf; border-color: #2d3a5a; }
  .prov-rag      { background: #2a1a2e;           color: #b07ecf; border-color: #4a2d5a; }
  .prov-episodic { background: #2a2218;           color: #cfb07e; border-color: #5a4a2d; }
  .prov-default  { background: var(--bg-active); color: var(--text-tertiary); border-color: var(--border); }
  .prov-tool     { background: #1e1e1e;           color: #cf9a7e; border-color: #4a3020; }
  .prov-episodic-mem { background: #2a2218;       color: #cfb07e; border-color: #5a4a2d; }
  .prov-grounded { background: #1a2a1a;           color: #7ecf7e; border-color: #2d4a2d; }

  .attach-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius);
    color: var(--text-tertiary);
    flex-shrink: 0;
    transition: color var(--dur-fast) var(--ease);
  }
  .attach-btn:hover:not(:disabled) { color: var(--text-secondary); }
  .attach-btn:disabled { opacity: 0.4; }

  .attached-files {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--sp-1);
    padding: var(--sp-2) 0 0;
  }

  .file-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2px var(--sp-2);
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    max-width: 280px;
  }

  .file-pill-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 180px;
  }

  .file-pill-tokens {
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .file-pill-remove {
    display: flex;
    align-items: center;
    color: var(--text-muted);
    flex-shrink: 0;
    padding: 0;
    transition: color var(--dur-fast) var(--ease);
  }
  .file-pill-remove:hover { color: var(--error); }

  .attach-error {
    font-size: 11px;
    color: var(--error);
    font-family: var(--font-mono);
    padding: 0 var(--sp-1);
  }
</style>
