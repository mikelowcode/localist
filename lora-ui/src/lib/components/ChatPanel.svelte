<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { tasksStore, submitTask, type Task } from '$lib/stores/tasks';
  import MarkdownRenderer from '$lib/components/MarkdownRenderer.svelte';

  let instruction = '';
  let messagesEl: HTMLElement;
  let inputEl: HTMLTextAreaElement;
  let submitting = false;

  // Conversation history (pairs of user instruction + task result)
  interface Turn {
    role: 'user' | 'assistant';
    content: string;
    task_id?: string;
    timestamp: number;
    status?: Task['status'];
    sources?: Task['sources'];
    status_message?: string;
  }

  let turns: Turn[] = [];

  $: activeTask = $tasksStore.active_task_id
    ? $tasksStore.tasks[$tasksStore.active_task_id]
    : null;

  // Reactively update the last assistant turn from the streaming store
  $: if (activeTask) {
    const idx = turns.findLastIndex(
      (t) => t.role === 'assistant' && t.task_id === activeTask!.task_id
    );
    if (idx >= 0) {
      turns[idx] = {
        ...turns[idx],
        content: activeTask.answer,
        status: activeTask.status,
        sources: activeTask.sources,
        status_message: activeTask.status_message
      };
      turns = turns; // trigger reactivity
      scrollToBottom();
    }
  }

  // Pick up tasks injected by ingestFile() before navigation
  $: {
    const state = $tasksStore;
    if (state.active_task_id) {
      const task = state.tasks[state.active_task_id];
      if (
        task &&
        task.status === 'complete' &&
        !turns.some((t) => t.task_id === task.task_id)
      ) {
        turns = [
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
            status_message: task.status_message,
          },
        ];
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
    if (!text || submitting || $tasksStore.streaming) return;

    submitting = true;
    instruction = '';
    autoResizeTextarea();

    // Add user turn
    turns = [
      ...turns,
      { role: 'user', content: text, timestamp: Date.now() }
    ];
    await scrollToBottom();

    // Reserve assistant slot with empty answer
    const tempId = `pending-${Date.now()}`;
    turns = [
      ...turns,
      {
        role: 'assistant',
        content: '',
        task_id: tempId,
        timestamp: Date.now(),
        status: 'planning',
        status_message: 'Planning…',
        sources: []
      }
    ];
    await scrollToBottom();

    const task_id = await submitTask(text);

    // Update the reserved slot with the real task_id
    turns = turns.map((t) =>
      t.task_id === tempId ? { ...t, task_id } : t
    );

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

  onMount(() => {
    inputEl?.focus();
  });
</script>

<div class="chat-panel">
  <!-- Messages -->
  <div class="messages" bind:this={messagesEl}>
    {#if turns.length === 0}
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <p class="empty-title">Start a conversation</p>
        <p class="empty-sub">Ask a research question or give an instruction to the agent system.</p>
      </div>
    {:else}
      {#each turns as turn (turn.timestamp)}
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

              {#if turn.sources && turn.sources.length > 0 && turn.status === 'complete'}
                <div class="sources-row">
                  {#each turn.sources as src}
                    <span class="badge badge-muted source-badge" title={src.path}>
                      {src.type === 'wiki' ? '📄' : '📁'} {src.name}
                    </span>
                  {/each}
                </div>
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
      <textarea
        bind:this={inputEl}
        bind:value={instruction}
        on:keydown={handleKeydown}
        on:input={autoResizeTextarea}
        placeholder="Ask a research question or give an instruction…"
        rows="1"
        disabled={$tasksStore.streaming || submitting}
        class="chat-input"
        aria-label="Message input"
      />
      <button
        class="send-btn"
        on:click={handleSubmit}
        disabled={!instruction.trim() || $tasksStore.streaming || submitting}
        aria-label="Send message"
        title="Send (Enter)"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
    <p class="input-hint">
      <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
    </p>
  </div>
</div>

<style>
  .chat-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  /* ── Messages ─────────────────────────────── */
  .messages {
    flex: 1;
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
</style>
