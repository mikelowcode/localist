<script lang="ts">
  // Renders an assistant turn's content, with an optional in-place editor:
  // hovering the turn reveals a pencil toggle in the bubble's top-right
  // corner; toggling it swaps the rendered markdown for a full-width
  // textarea (pre-filled with the same content) plus inline filename/
  // extension fields, so the user drafts edits directly in the chat turn
  // rather than in a cramped popup. Save/Cancel then replace the pencil
  // toggle, top-right, only while editing.
  //
  // Editing is export-only: `draft` is local scratch state, never written
  // back to the caller's `content` or chatHistoryStore — cancelling (or
  // just navigating away) discards it, and the live chat turn is
  // unaffected by what gets saved.
  import { tick } from 'svelte';
  import MarkdownRenderer from '$lib/components/MarkdownRenderer.svelte';
  import { saveGeneratedFile } from '$lib/stores/files';

  export let content: string;
  export let streaming: boolean = false;
  export let editable: boolean = true;   // gate the edit toggle — false while streaming/planning
  export let composeActive: boolean = false;   // Compose Mode is on — show the "Add to document" control
  export let alreadyAdded: boolean = false;    // this turn has already been added to the compose draft
  export let onAddToDocument: () => void = () => {};

  let editing = false;
  let draft = content;
  let filename = '';
  let extension: 'md' | 'txt' = 'md';
  let saving = false;
  let error: string | null = null;
  let savedAs: string | null = null;
  let textareaEl: HTMLTextAreaElement;

  function resizeTextarea() {
    if (!textareaEl) return;
    textareaEl.style.height = 'auto';
    textareaEl.style.height = Math.min(textareaEl.scrollHeight, 560) + 'px';
    error = null;
  }

  async function startEditing() {
    draft = content;
    filename = '';
    error = null;
    savedAs = null;
    editing = true;
    await tick();
    resizeTextarea();
  }

  function cancelEditing() {
    editing = false;
  }

  function handleTextareaKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') cancelEditing();
  }

  async function handleSave() {
    const trimmedName = filename.trim();
    if (!trimmedName) {
      error = 'Enter a file name.';
      return;
    }
    if (!draft.trim()) {
      error = 'Content must not be empty.';
      return;
    }

    saving = true;
    error = null;
    try {
      const entry = await saveGeneratedFile(trimmedName, extension, draft);
      savedAs = entry.filename;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }
</script>

<div class="editable-turn">
  {#if editing}
    <div class="turn-edit-controls">
      {#if savedAs}
        <button class="turn-ctrl-btn turn-ctrl-done" on:click={cancelEditing} type="button">
          Done
        </button>
      {:else}
        <button
          class="turn-ctrl-btn turn-ctrl-save"
          on:click={handleSave}
          disabled={saving}
          aria-label="Save as file"
          title="Save as file"
          type="button"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        </button>
        <button
          class="turn-ctrl-btn"
          on:click={cancelEditing}
          disabled={saving}
          aria-label="Cancel editing"
          title="Cancel"
          type="button"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      {/if}
    </div>

    {#if savedAs}
      <p class="turn-edit-success">Saved as <strong>{savedAs}</strong></p>
    {:else}
      <textarea
        class="turn-edit-textarea"
        bind:this={textareaEl}
        bind:value={draft}
        on:input={resizeTextarea}
        on:keydown={handleTextareaKeydown}
        disabled={saving}
        aria-label="Edit response content"
      />
      <div class="turn-edit-name-row">
        <input
          class="turn-edit-name-input"
          type="text"
          bind:value={filename}
          on:input={() => (error = null)}
          placeholder="File name"
          disabled={saving}
          aria-label="File name"
        />
        <select class="turn-edit-ext-select" bind:value={extension} disabled={saving} aria-label="File type">
          <option value="md">.md</option>
          <option value="txt">.txt</option>
        </select>
      </div>
      {#if error}
        <p class="turn-edit-error">{error}</p>
      {/if}
    {/if}
  {:else}
    <div class="turn-view-controls">
      {#if composeActive}
        <button
          class="turn-add-doc-btn"
          class:turn-add-doc-added={alreadyAdded}
          on:click={onAddToDocument}
          disabled={alreadyAdded}
          aria-label={alreadyAdded ? 'Already added to document' : 'Add to document'}
          title={alreadyAdded ? 'Added to document' : 'Add to document'}
          type="button"
        >
          {#if alreadyAdded}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          {:else}
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <line x1="12" y1="5" x2="12" y2="19"/>
              <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          {/if}
        </button>
      {/if}
      {#if editable}
        <button
          class="turn-edit-toggle"
          on:click={startEditing}
          aria-label="Edit and save as file"
          title="Edit and save as file"
          type="button"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
      {/if}
    </div>
    <MarkdownRenderer {content} {streaming} />
  {/if}
</div>

<style>
  .editable-turn {
    position: relative;
  }

  .turn-view-controls {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    gap: var(--sp-1);
  }

  .turn-edit-toggle,
  .turn-add-doc-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: var(--radius-sm);
    /* bg-raised, not bg-panel: the bubble itself is bg-panel, so a bg-panel
       button sitting on top of it was indistinguishable from the bubble
       surface — only a 1px border separated them, easy to lose against
       whatever text happens to sit underneath. The shadow adds a real
       depth cue on top of the color difference, since a flat color alone
       can still blend into text of a similar shade directly behind it. */
    background: var(--bg-raised);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    color: var(--text-secondary);
    opacity: 0;
    transition: color var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease),
      box-shadow var(--dur-fast) var(--ease), border-color var(--dur-fast) var(--ease);
  }

  :global(.turn:hover) .turn-edit-toggle,
  :global(.turn:hover) .turn-add-doc-btn {
    opacity: 1;
  }

  /* Hover keeps the same opaque background (still var(--bg-raised), never
     var(--bg-active)/var(--bg-hover) — those are translucent overlay tints
     meant to sit on an already-transparent element elsewhere in the app;
     used as this chip's *only* background, they let the text underneath
     show back through on hover, which reads as "the shadow disappeared."
     A stronger shadow + a lit border communicate hover instead, with the
     surface staying fully opaque the whole time. */
  .turn-edit-toggle:hover {
    color: var(--text-primary);
    box-shadow: var(--shadow-md);
    border-color: var(--border-focus);
  }
  .turn-add-doc-btn:hover:not(:disabled) {
    color: var(--text-primary);
    box-shadow: var(--shadow-md);
    border-color: var(--border-focus);
  }

  /* Same hover-gated opacity as the base state (not permanently visible) —
     an always-on checkmark overlapping a short single-line turn's own last
     word/line is a real, common overlap for terse answers, not just a
     hover-only cosmetic tradeoff. Only the color/border read as "added"
     once revealed. */
  .turn-add-doc-added {
    color: var(--success);
    border-color: var(--success-dim);
  }
  /* Same opaque-background rule as above — border/shadow only, no
     translucent background swap. */
  .turn-add-doc-added:hover:not(:disabled) {
    color: var(--success);
    box-shadow: var(--shadow-md);
    border-color: var(--success);
  }

  .turn-edit-controls {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    gap: var(--sp-1);
  }

  .turn-ctrl-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: var(--radius-sm);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    color: var(--text-tertiary);
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }

  .turn-ctrl-btn:hover:not(:disabled) { color: var(--text-secondary); background: var(--bg-active); }
  .turn-ctrl-btn:disabled { opacity: 0.5; }

  .turn-ctrl-save { color: var(--success); border-color: var(--success-dim); }
  .turn-ctrl-save:hover:not(:disabled) { color: var(--success); background: var(--success-dim); }

  .turn-ctrl-done {
    width: auto;
    padding: 0 var(--sp-2);
    font-size: var(--text-xs);
    font-weight: 500;
    color: #fff;
    background: var(--accent);
    border-color: var(--accent);
  }
  .turn-ctrl-done:hover { background: #6fa3ff; }

  .turn-edit-textarea {
    /* A fixed width, not 100%: .bubble/.turn are shrink-to-fit flex items
       with no explicit width (sized by their content, up to max-width), so
       a percentage width here can't resolve against them — browsers fall
       back to the textarea's tiny default intrinsic width (~20ch), which
       collapsed the whole bubble into a narrow, content-forced-tall column.
       A definite width gives the shrink-to-fit ancestors something concrete
       to size around instead, and max-width keeps it responsive below 600px. */
    width: 600px;
    max-width: 100%;
    box-sizing: border-box;
    resize: vertical;
    min-height: 160px;
    max-height: 560px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12.5px;
    font-family: var(--font-mono);
    line-height: 1.6;
    padding: var(--sp-2);
    padding-right: 32px;   /* keep the top-right controls clear of the resize handle/text */
  }

  .turn-edit-textarea:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .turn-edit-name-row {
    display: flex;
    gap: var(--sp-2);
    margin-top: var(--sp-2);
  }

  .turn-edit-name-input {
    flex: 1;
    min-width: 0;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12px;
    padding: var(--sp-1) var(--sp-2);
  }

  .turn-edit-name-input:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--border-focus);
  }

  .turn-edit-ext-select {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    font-size: 12px;
    padding: var(--sp-1) var(--sp-2);
    flex-shrink: 0;
  }

  .turn-edit-error {
    font-size: var(--text-xs);
    color: var(--error);
    margin-top: var(--sp-2);
  }

  .turn-edit-success {
    font-size: var(--text-sm);
    color: var(--success);
    padding-right: 60px;   /* clear the Done button */
  }
</style>
