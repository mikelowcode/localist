<script lang="ts">
  // Listing, upload, and ingest now live in the sidebar's expandable Files
  // sub-nav (see Sidebar.svelte) — this component is just the full-width
  // preview pane for whatever fileSelection.selectedFile currently holds.
  import {
    selectedFile, fileContent, fileContentLoading, fileContentError, closeFile
  } from '$lib/stores/fileSelection';
  import { ingest, isIngesting, ingestFile, resetIngest } from '$lib/stores/files';

  async function handleIngest(): Promise<void> {
    if ($isIngesting || !$selectedFile) return;
    resetIngest();
    await ingestFile($selectedFile);
  }
</script>

<!-- ════════════════════════════════════════════════════════════════════ -->

<div class="content-pane">
  {#if !$selectedFile}
    <div class="content-empty">
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.1"
           stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true" style="opacity:0.2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
      </svg>
      <p>Select a file from the sidebar to view its contents</p>
    </div>
  {:else}
    <div class="content-header">
      <div class="content-title-row">
        <span class="badge {$selectedFile.type === 'wiki' ? 'badge-success' : $selectedFile.type === 'raw' ? 'badge-warning' : 'badge-muted'}">
          {$selectedFile.type}
        </span>
        <h3 class="content-filename">{$selectedFile.name}</h3>
      </div>
      <button class="close-btn" on:click={closeFile} aria-label="Close file">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>

    <div class="content-body">
      {#if $fileContentLoading}
        <div class="content-loading">
          <span class="spinner" aria-hidden="true"></span>
          <span class="text-tertiary text-sm">Loading…</span>
        </div>
      {:else if $fileContentError}
        <div class="content-loading">
          <span class="text-sm" style="color: var(--error)">{$fileContentError}</span>
        </div>
      {:else}
        <pre class="file-content-pre"><code>{$fileContent}</code></pre>
      {/if}
    </div>

    {#if $selectedFile.type === 'raw'}
      <div class="content-footer">
        <button
          class="ingest-btn-lg"
          class:loading={$isIngesting}
          disabled={$isIngesting}
          on:click={handleIngest}
        >
          {#if $isIngesting}
            <span class="spinner accent-spinner" aria-hidden="true"></span>
            Ingesting…
          {:else}
            <svg viewBox="0 0 16 16" width="12" height="12" fill="none"
                 stroke="currentColor" stroke-width="1.8" aria-hidden="true">
              <path d="M8 1v9M5 7l3 3 3-3" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 12v2h12v-2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Ingest to wiki
          {/if}
        </button>
      </div>
    {:else if $selectedFile.type === 'generated'}
      <div class="content-footer">
        <a
          class="ingest-btn-lg"
          href={`/api/files/download?path=${encodeURIComponent($selectedFile.path)}`}
          download={$selectedFile.filename}
        >
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download
        </a>
      </div>
    {/if}
  {/if}
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<style>
  .content-pane {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    background: var(--bg);
  }

  .content-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    color: var(--text-tertiary);
    font-size: var(--text-sm);
    text-align: center;
    padding: 0 var(--sp-8);
  }

  .content-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--sp-4) var(--sp-6);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    background: var(--bg-panel);
    gap: var(--sp-4);
  }

  .content-title-row {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    min-width: 0;
  }

  .content-filename {
    font-size: 14px;
    font-weight: 600;
    font-family: var(--font-mono);
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: var(--radius-sm);
    background: none;
    color: var(--text-tertiary);
    flex-shrink: 0;
    transition:
      background var(--dur-fast) var(--ease),
      color var(--dur-fast) var(--ease);
  }

  .close-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .content-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }

  .content-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--sp-3);
    padding: var(--sp-8);
  }

  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }
  .accent-spinner { border-color: currentColor; border-top-color: transparent; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .file-content-pre {
    margin: 0;
    padding: var(--sp-6);
    border: none;
    border-radius: 0;
    background: none;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    line-height: 1.65;
    color: var(--text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .file-content-pre code {
    background: none;
    padding: 0;
    color: inherit;
    font-size: inherit;
  }

  .content-footer {
    flex-shrink: 0;
    padding: var(--sp-3) var(--sp-6);
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    display: flex;
    justify-content: flex-end;
  }

  .ingest-btn-lg {
    display: inline-flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    font-size: var(--text-sm);
    font-weight: 500;
    font-family: var(--font-body);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    text-decoration: none;
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }

  .ingest-btn-lg:hover:not(:disabled):not(.loading) {
    background: var(--accent-dim);
  }

  .ingest-btn-lg:disabled { opacity: 0.4; cursor: not-allowed; }
  .ingest-btn-lg.loading { opacity: 0.7; cursor: wait; }
</style>
