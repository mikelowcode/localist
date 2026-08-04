<script lang="ts">
  // Compose Mode's document panel — a 4th #app-shell grid column (see
  // +layout.svelte), visible only while composeDocument.active is true for
  // the current conversation. Unlike PreviewsPanel (an always-present,
  // app-wide utility that collapses to a thin persistent strip), this panel
  // is contextual to an in-progress document: "off" means gone (0-width),
  // not collapsed-to-a-tab, since there's nothing to keep discoverable when
  // no compose session exists. Reopening (the composer-row toggle in
  // ChatPanel.svelte) restores the last draft via composeDocument's own
  // localStorage persistence.
  //
  // The draft is hand-editable here directly, and also grows via each
  // assistant turn's "Add to document" control (EditableTurnContent.svelte,
  // wired through ChatPanel.svelte) — both write through the same
  // composeDocument store. Saving reuses the same POST /files/generated
  // flow single-turn saves already use.
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import {
    composeDocument, setComposeDraft, toggleComposeMode, clearComposeDraft,
    composePanelWidth, COMPOSE_MIN_WIDTH, COMPOSE_MAX_WIDTH
  } from '$lib/stores/composeDocument';
  import { saveGeneratedFile } from '$lib/stores/files';

  let filename = '';
  let extension: 'md' | 'txt' = 'md';
  let saving = false;
  let error: string | null = null;
  let savedAs: string | null = null;
  let confirmingClear = false;

  // ── Resize (drag the left edge) ──────────────────────────────────────
  // Mirrors Sidebar.svelte's divider exactly, except this panel is on the
  // right side of the screen, so the sign is flipped: dragging the handle
  // left (negative dx) grows the panel, not shrinks it.
  let dragging = false;
  let dragStartX = 0;
  let dragStartW = 0;

  function startResize(e: MouseEvent): void {
    dragging = true;
    dragStartX = e.clientX;
    dragStartW = $composePanelWidth;
    e.preventDefault();
  }

  function onMove(e: MouseEvent): void {
    if (!dragging) return;
    const dx = e.clientX - dragStartX;
    const w = dragStartW - dx;
    composePanelWidth.set(Math.max(COMPOSE_MIN_WIDTH, Math.min(COMPOSE_MAX_WIDTH, w)));
  }

  function onUp(): void {
    dragging = false;
  }

  function onDividerKeydown(e: KeyboardEvent): void {
    if (e.key === 'ArrowLeft') {
      composePanelWidth.set(Math.min(COMPOSE_MAX_WIDTH, $composePanelWidth + 8));
    } else if (e.key === 'ArrowRight') {
      composePanelWidth.set(Math.max(COMPOSE_MIN_WIDTH, $composePanelWidth - 8));
    } else {
      return;
    }
    e.preventDefault();
  }

  onMount(() => {
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });

  onDestroy(() => {
    if (browser) {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    }
  });

  function handleDraftInput(e: Event) {
    setComposeDraft((e.target as HTMLTextAreaElement).value);
    savedAs = null;
  }

  function handleConfirmClear() {
    clearComposeDraft();
    confirmingClear = false;
    filename = '';
    error = null;
    savedAs = null;
  }

  async function handleSave() {
    const trimmedName = filename.trim();
    if (!trimmedName) {
      error = 'Enter a file name.';
      return;
    }
    if (!$composeDocument.draft.trim()) {
      error = 'Document is empty.';
      return;
    }

    saving = true;
    error = null;
    try {
      const entry = await saveGeneratedFile(trimmedName, extension, $composeDocument.draft);
      savedAs = entry.filename;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }
</script>

{#if $composeDocument.active}
  <div class="compose-panel">
    <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div
      class="compose-divider"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize document panel"
      aria-valuenow={$composePanelWidth}
      aria-valuemin={COMPOSE_MIN_WIDTH}
      aria-valuemax={COMPOSE_MAX_WIDTH}
      tabindex="0"
      on:mousedown={startResize}
      on:keydown={onDividerKeydown}
    />
    <div class="compose-panel-header">
      <span class="compose-panel-title">Document</span>
      <div class="compose-header-actions">
        {#if $composeDocument.draft.trim() && !confirmingClear}
          <button
            type="button"
            class="compose-clear-link"
            on:click={() => (confirmingClear = true)}
          >Clear</button>
        {/if}
        <button
          type="button"
          class="compose-close-btn"
          on:click={toggleComposeMode}
          aria-label="Close compose mode"
          title="Close (draft is kept)"
        >×</button>
      </div>
    </div>

    {#if confirmingClear}
      <div class="compose-clear-confirm">
        <span>Clear this document? This can't be undone.</span>
        <div class="compose-clear-confirm-actions">
          <button type="button" class="compose-clear-confirm-btn" on:click={handleConfirmClear}>Clear</button>
          <button type="button" class="compose-clear-cancel-btn" on:click={() => (confirmingClear = false)}>Cancel</button>
        </div>
      </div>
    {/if}

    <div class="compose-panel-body">
      <textarea
        class="compose-textarea"
        value={$composeDocument.draft}
        on:input={handleDraftInput}
        disabled={saving}
        placeholder="Compose your document here — type directly, or use a turn's + button in the chat to add its content."
        aria-label="Document draft"
      />
      <div class="compose-name-row">
        <input
          class="compose-name-input"
          type="text"
          bind:value={filename}
          on:input={() => (error = null)}
          placeholder="File name"
          disabled={saving}
          aria-label="File name"
        />
        <select class="compose-ext-select" bind:value={extension} disabled={saving} aria-label="File type">
          <option value="md">.md</option>
          <option value="txt">.txt</option>
        </select>
        <button
          class="compose-save-btn"
          type="button"
          on:click={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
      {#if error}
        <p class="compose-error">{error}</p>
      {/if}
      {#if savedAs}
        <p class="compose-success">Saved as <strong>{savedAs}</strong></p>
      {/if}
    </div>
  </div>
{/if}

<style>
  .compose-panel {
    position: relative;
    grid-column: 4;
    grid-row: 1 / -1;
    height: 100%;
    border-left: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .compose-divider {
    position: absolute;
    top: 0;
    left: -3px;
    width: 6px;
    height: 100%;
    cursor: col-resize;
    z-index: 6;
  }

  .compose-panel-header {
    flex-shrink: 0;
    height: var(--topbar-h);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--sp-4);
    border-bottom: 1px solid var(--border);
  }

  .compose-panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .compose-header-actions {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }

  .compose-clear-link {
    font-size: 11.5px;
    font-weight: 500;
    color: var(--text-tertiary);
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
  }
  .compose-clear-link:hover { color: var(--error); }

  .compose-close-btn {
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease);
  }
  .compose-close-btn:hover { background: var(--bg-hover); }

  .compose-clear-confirm {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-3);
    background: var(--error-dim);
    border-bottom: 1px solid var(--border);
    font-size: 11.5px;
    color: var(--text-primary);
  }

  .compose-clear-confirm-actions {
    display: flex;
    gap: var(--sp-1);
    flex-shrink: 0;
  }

  .compose-clear-confirm-btn,
  .compose-clear-cancel-btn {
    font-size: 11px;
    font-weight: 600;
    padding: 2px var(--sp-2);
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    cursor: pointer;
  }

  .compose-clear-confirm-btn {
    background: var(--error);
    color: #fff;
  }
  .compose-clear-confirm-btn:hover { opacity: 0.85; }

  .compose-clear-cancel-btn {
    background: var(--bg-raised);
    border-color: var(--border);
    color: var(--text-secondary);
  }
  .compose-clear-cancel-btn:hover { background: var(--bg-active); }

  .compose-panel-body {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
    padding: var(--sp-3);
    overflow: hidden;
  }

  .compose-textarea {
    flex: 1;
    min-height: 0;
    resize: none;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12.5px;
    font-family: var(--font-mono);
    line-height: 1.6;
    padding: var(--sp-2);
  }

  .compose-textarea:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .compose-name-row {
    flex-shrink: 0;
    display: flex;
    gap: var(--sp-2);
  }

  .compose-name-input {
    flex: 1;
    min-width: 0;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12px;
    padding: var(--sp-1) var(--sp-2);
  }

  .compose-name-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .compose-ext-select {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12px;
    padding: var(--sp-1) var(--sp-2);
    flex-shrink: 0;
  }

  .compose-save-btn {
    flex-shrink: 0;
    font-size: var(--text-xs);
    font-weight: 500;
    padding: var(--sp-1) var(--sp-3);
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }
  .compose-save-btn:hover:not(:disabled) { background: #6fa3ff; }
  .compose-save-btn:disabled { opacity: 0.5; }

  .compose-error {
    flex-shrink: 0;
    font-size: var(--text-xs);
    color: var(--error);
  }

  .compose-success {
    flex-shrink: 0;
    font-size: var(--text-xs);
    color: var(--success);
  }
</style>
