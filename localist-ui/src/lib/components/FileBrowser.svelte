<script lang="ts">
  import { onMount } from 'svelte';
  import {
    rawFiles, wikiFiles,
    rawLoading, wikiLoading,
    rawError, wikiError,
    ingest, isIngesting,
    loadRawFiles, loadWikiFiles,
    uploadFile, ingestFile, resetIngest,
    formatBytes,
    type FileEntry,
  } from '$lib/stores/files';

  // ── File content viewer (right pane) ──────────────────────────────────
  let selectedFile:   FileEntry | null = null;
  let fileContent:    string | null    = null;
  let contentLoading: boolean          = false;
  let contentError:   string | null    = null;

  async function openFile(file: FileEntry) {
    // Toggle off if already open
    if (selectedFile?.path === file.path && fileContent !== null) {
      selectedFile = null;
      fileContent  = null;
      contentError = null;
      return;
    }
    selectedFile   = file;
    fileContent    = null;
    contentError   = null;
    contentLoading = true;
    try {
      const res = await fetch(`/api/files/content?path=${encodeURIComponent(file.path)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Backend returns JSON { path, content } — not plain text
      const data: { path: string; content: string } = await res.json();
      fileContent = data.content;
    } catch (err) {
      contentError = err instanceof Error ? err.message : String(err);
    } finally {
      contentLoading = false;
    }
  }

  function closeFile() {
    selectedFile   = null;
    fileContent    = null;
    contentError   = null;
  }

  // ── Upload ────────────────────────────────────────────────────────────
  let dragging:      boolean       = false;
  let uploading:     boolean       = false;
  let uploadError:   string | null = null;
  let uploadSuccess: string | null = null;

  function onDragOver(e: DragEvent) { e.preventDefault(); dragging = true; }
  function onDragLeave()            { dragging = false; }

  async function onDrop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    if (e.dataTransfer?.files?.length) await handleUpload(e.dataTransfer.files);
  }

  async function onFileInput(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) await handleUpload(input.files);
    input.value = '';
  }

  async function handleUpload(files: FileList) {
    uploadError   = null;
    uploadSuccess = null;

    const list = Array.from(files);
    for (const f of list) {
      const ext = f.name.split('.').pop()?.toLowerCase();
      if (!['md', 'txt'].includes(ext ?? '')) {
        uploadError = `Unsupported type: .${ext}. Only .md and .txt are accepted.`;
        return;
      }
    }

    uploading = true;
    try {
      for (const f of list) {
        await uploadFile(f);   // refreshes rawFiles internally
      }
      uploadSuccess = `${list.length} file${list.length > 1 ? 's' : ''} uploaded.`;
    } catch (err) {
      uploadError = err instanceof Error ? err.message : String(err);
    } finally {
      uploading = false;
    }
  }

  // ── Ingest ────────────────────────────────────────────────────────────
  async function handleIngest(entry: FileEntry) {
    if ($isIngesting) return;
    resetIngest();
    await ingestFile(entry);
    // Navigation to /conversation happens inside ingestFile() on success.
  }

  function formatSize(bytes?: number): string {
    if (bytes === undefined) return '';
    return formatBytes(bytes);
  }

  onMount(() => {
    loadRawFiles();
    loadWikiFiles();
  });
</script>

<!-- ════════════════════════════════════════════════════════════════════ -->

<div class="file-browser">

  <!-- ── Left pane: listings + upload ─────────────────────────────── -->
  <div class="file-pane">

    <!-- Upload zone -->
    <div
      class="drop-zone"
      class:dragging
      on:dragover={onDragOver}
      on:dragleave={onDragLeave}
      on:drop={onDrop}
      role="region"
      aria-label="File upload zone"
    >
      <input
        type="file"
        id="file-input"
        multiple
        accept=".md,.txt"
        on:change={onFileInput}
        class="sr-only"
      />
      <label for="file-input" class="drop-label">
        {#if uploading}
          <span class="spinner" aria-hidden="true" />
          <span>Uploading…</span>
        {:else}
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.75"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="16 16 12 12 8 16"/>
            <line x1="12" y1="12" x2="12" y2="21"/>
            <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
          </svg>
          <span>Drop .md / .txt or <span class="upload-link">browse</span></span>
        {/if}
      </label>
      {#if uploadSuccess}
        <p class="upload-feedback success">{uploadSuccess}</p>
      {/if}
      {#if uploadError}
        <p class="upload-feedback error">{uploadError}</p>
      {/if}
    </div>

    <!-- Ingest progress banner -->
    {#if $ingest.phase === 'planning' || $ingest.phase === 'streaming'}
      <div class="ingest-banner ingest-progress">
        <span class="spinner accent-spinner" aria-hidden="true"></span>
        <span class="ingest-msg">
          {#if $ingest.phase === 'planning'}
            {$ingest.statusMsg || 'Planning…'}
          {:else}
            Ingesting — {$ingest.tokens.length} chunks received…
          {/if}
        </span>
        <span class="ingest-file">{$ingest.sourceFile?.filename ?? ''}</span>
      </div>
    {/if}

    {#if $ingest.phase === 'error'}
      <div class="ingest-banner ingest-error">
        <span class="ingest-msg">Ingest failed: {$ingest.error}</span>
        <button class="dismiss-btn" on:click={resetIngest}>Dismiss</button>
      </div>
    {/if}

    <!-- Wiki pages section -->
    <div class="file-section">
      <div class="section-header">
        <h3 class="section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
          Wiki Pages
        </h3>
        <span class="section-count">{$wikiFiles.length}</span>
        <button class="refresh-btn" on:click={loadWikiFiles}
                disabled={$wikiLoading} title="Refresh" aria-label="Refresh wiki pages">
          <svg viewBox="0 0 16 16" width="11" height="11" fill="none"
               stroke="currentColor" stroke-width="1.8">
            <path d="M2 8a6 6 0 1 0 1-3.2" stroke-linecap="round"/>
            <path d="M2 4v2h2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>

      {#if $wikiError}
        <p class="section-error">{$wikiError}</p>
      {/if}

      {#if $wikiLoading}
        <div class="loading-list">
          {#each [80, 60, 72, 55] as w}
            <div class="skeleton-row" style="width:{w}%"></div>
          {/each}
        </div>
      {:else if $wikiFiles.length === 0}
        <p class="empty-section">No wiki pages yet.</p>
      {:else}
        <ul class="file-list" role="list">
          {#each $wikiFiles as file (file.path)}
            <li>
              <button
                class="file-item"
                class:selected={selectedFile?.path === file.path}
                on:click={() => openFile(file)}
                aria-pressed={selectedFile?.path === file.path}
              >
                <svg class="file-icon" width="13" height="13" viewBox="0 0 24 24"
                     fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span class="file-name truncate">{file.name}</span>
                {#if file.size}
                  <span class="file-size">{formatSize(file.size)}</span>
                {/if}
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>

    <!-- Raw files section -->
    <div class="file-section">
      <div class="section-header">
        <h3 class="section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
          Raw Files
        </h3>
        <span class="section-count">{$rawFiles.length}</span>
        <button class="refresh-btn" on:click={loadRawFiles}
                disabled={$rawLoading} title="Refresh" aria-label="Refresh raw files">
          <svg viewBox="0 0 16 16" width="11" height="11" fill="none"
               stroke="currentColor" stroke-width="1.8">
            <path d="M2 8a6 6 0 1 0 1-3.2" stroke-linecap="round"/>
            <path d="M2 4v2h2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>

      {#if $rawError}
        <p class="section-error">{$rawError}</p>
      {/if}

      {#if $rawLoading}
        <div class="loading-list">
          {#each [65, 80, 45] as w}
            <div class="skeleton-row" style="width:{w}%"></div>
          {/each}
        </div>
      {:else if $rawFiles.length === 0}
        <p class="empty-section">No raw files yet. Upload .md or .txt files above.</p>
      {:else}
        <ul class="file-list" role="list">
          {#each $rawFiles as file (file.path)}
            {@const isActiveIngest = $ingest.sourceFile?.path === file.path && $isIngesting}
            <li class="raw-file-item">
              <button
                class="file-item"
                class:selected={selectedFile?.path === file.path}
                on:click={() => openFile(file)}
                aria-pressed={selectedFile?.path === file.path}
              >
                <svg class="file-icon" width="13" height="13" viewBox="0 0 24 24"
                     fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                  <polyline points="13 2 13 9 20 9"/>
                </svg>
                <span class="file-name truncate">{file.name}</span>
                {#if file.size}
                  <span class="file-size">{formatSize(file.size)}</span>
                {/if}
              </button>

              <button
                class="ingest-btn"
                class:loading={isActiveIngest}
                disabled={$isIngesting}
                on:click|stopPropagation={() => handleIngest(file)}
                title="Ingest {file.filename} into the wiki"
              >
                {#if isActiveIngest}
                  <span class="spinner accent-spinner" aria-hidden="true"></span>
                {:else}
                  <svg viewBox="0 0 16 16" width="11" height="11" fill="none"
                       stroke="currentColor" stroke-width="1.8" aria-hidden="true">
                    <path d="M8 1v9M5 7l3 3 3-3" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12v2h12v-2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                {/if}
                {isActiveIngest ? 'Ingesting…' : 'Ingest'}
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>

  </div><!-- /file-pane -->

  <!-- ── Right pane: file content viewer ──────────────────────────── -->
  <div class="content-pane">
    {#if !selectedFile}
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
        <p>Select a file to view its contents</p>
      </div>
    {:else}
      <div class="content-header">
        <div class="content-title-row">
          <span class="badge {selectedFile.type === 'wiki' ? 'badge-success' : 'badge-warning'}">
            {selectedFile.type}
          </span>
          <h3 class="content-filename">{selectedFile.name}</h3>
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
        {#if contentLoading}
          <div class="content-loading">
            <span class="spinner" aria-hidden="true"></span>
            <span class="text-tertiary text-sm">Loading…</span>
          </div>
        {:else if contentError}
          <div class="content-loading">
            <span class="text-sm" style="color: var(--error)">{contentError}</span>
          </div>
        {:else}
          <pre class="file-content-pre"><code>{fileContent}</code></pre>
        {/if}
      </div>

      <!-- Ingest shortcut in content pane for raw files -->
      {#if selectedFile.type === 'raw'}
        {@const isActiveIngest = $ingest.sourceFile?.path === selectedFile.path && $isIngesting}
        <div class="content-footer">
          <button
            class="ingest-btn-lg"
            class:loading={isActiveIngest}
            disabled={$isIngesting}
            on:click={() => handleIngest(selectedFile)}
          >
            {#if isActiveIngest}
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
      {/if}

    {/if}
  </div><!-- /content-pane -->

</div>

<!-- ════════════════════════════════════════════════════════════════════ -->

<style>
  .file-browser {
    display: grid;
    grid-template-columns: 280px 1fr;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* ── Left pane ──────────────────────────────────────────────────── */
  .file-pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-right: 1px solid var(--border);
    overflow-y: auto;
    background: var(--bg-panel);
  }

  /* Drop zone */
  .drop-zone {
    margin: var(--sp-4);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: var(--sp-5) var(--sp-4);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--sp-2);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease),
      background var(--dur-fast) var(--ease);
    flex-shrink: 0;
  }

  .drop-zone.dragging {
    border-color: var(--accent);
    background: var(--accent-glow);
  }

  .drop-label {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    cursor: pointer;
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    text-align: center;
  }

  .upload-link { color: var(--accent); text-decoration: underline; }

  .upload-feedback { font-size: var(--text-xs); text-align: center; }
  .upload-feedback.success { color: var(--success); }
  .upload-feedback.error   { color: var(--error); }

  /* Ingest progress / error banners */
  .ingest-banner {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    margin: 0 var(--sp-4) var(--sp-2);
    padding: var(--sp-2) var(--sp-3);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    flex-shrink: 0;
  }

  .ingest-progress {
    background: var(--accent-dim);
    border: 1px solid var(--accent-mid);
    color: var(--accent);
  }

  .ingest-error {
    background: var(--error-dim);
    border: 1px solid color-mix(in srgb, var(--error) 30%, transparent);
    color: var(--error);
    justify-content: space-between;
  }

  .ingest-msg  { flex: 1; font-family: var(--font-mono); }
  .ingest-file {
    font-family: var(--font-mono);
    font-size: 10px;
    opacity: 0.7;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 120px;
  }

  .dismiss-btn {
    background: none;
    border: 1px solid currentColor;
    border-radius: var(--radius-sm);
    padding: 2px 8px;
    font-size: var(--text-xs);
    cursor: pointer;
    color: inherit;
    flex-shrink: 0;
  }

  .dismiss-btn:hover {
    background: color-mix(in srgb, currentColor 15%, transparent);
  }

  /* Spinners */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }

  .accent-spinner {
    border-color: currentColor;
    border-top-color: transparent;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  /* Section chrome */
  .file-section {
    flex-shrink: 0;
    padding: var(--sp-2) 0 var(--sp-4);
    border-top: 1px solid var(--border-soft);
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-4);
    margin-bottom: var(--sp-1);
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-tertiary);
    flex: 1;
  }

  .section-count {
    font-size: 10px;
    font-family: var(--font-mono);
    background: var(--bg-active);
    color: var(--text-tertiary);
    padding: 1px 7px;
    border-radius: 100px;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    background: none;
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
    color: var(--text-tertiary);
    padding: 0;
    transition: color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }

  .refresh-btn:hover:not(:disabled) {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .refresh-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .section-error {
    padding: var(--sp-1) var(--sp-4);
    font-size: var(--text-xs);
    color: var(--error);
  }

  .empty-section {
    padding: var(--sp-2) var(--sp-4);
    font-size: var(--text-xs);
    color: var(--text-muted);
    line-height: 1.5;
  }

  /* File list */
  .file-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 0 var(--sp-2);
  }

  /* Raw file rows get flex to accommodate the ingest button */
  .raw-file-item {
    display: flex;
    align-items: center;
    gap: var(--sp-1);
    padding-right: var(--sp-2);
    border-radius: var(--radius-sm);
    transition: background var(--dur-fast) var(--ease);
  }

  .raw-file-item:hover { background: var(--bg-hover); }

  .file-item {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: var(--sp-2) var(--sp-2);
    border-radius: var(--radius-sm);
    background: none;
    color: var(--text-secondary);
    font-size: var(--text-xs);
    text-align: left;
    cursor: pointer;
    transition: color var(--dur-fast) var(--ease);
  }

  /* Don't add background on file-item inside raw-file-item; parent handles it */
  .raw-file-item .file-item:hover { background: none; color: var(--text-primary); }

  /* Wiki file items (standalone li) */
  li:not(.raw-file-item) .file-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .file-item.selected {
    background: var(--accent-glow);
    color: var(--text-accent);
  }

  .raw-file-item .file-item.selected { background: none; color: var(--text-accent); }

  .file-icon { flex-shrink: 0; opacity: 0.6; }
  .file-name { flex: 1; font-family: var(--font-mono); }
  .file-size { flex-shrink: 0; color: var(--text-muted); font-family: var(--font-mono); }

  /* Ingest button (compact, in list row) */
  .ingest-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 500;
    font-family: var(--font-body);
    border: 1px solid var(--accent-mid);
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    white-space: nowrap;
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }

  .ingest-btn:hover:not(:disabled):not(.loading) {
    background: var(--accent-dim);
  }

  .ingest-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .ingest-btn.loading  { opacity: 0.7; cursor: wait; }

  /* Loading skeletons */
  .loading-list {
    padding: var(--sp-2) var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-2);
  }

  .skeleton-row {
    height: 10px;
    background: var(--bg-active);
    border-radius: var(--radius-sm);
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  /* ── Right pane ─────────────────────────────────────────────────── */
  .content-pane {
    display: flex;
    flex-direction: column;
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
    font-size: var(--text-sm);
    font-weight: 500;
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

  /* Ingest shortcut footer in content pane */
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
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    background: transparent;
    color: var(--accent);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), opacity var(--dur-fast) var(--ease);
  }

  .ingest-btn-lg:hover:not(:disabled):not(.loading) {
    background: var(--accent-dim);
  }

  .ingest-btn-lg:disabled { opacity: 0.4; cursor: not-allowed; }
  .ingest-btn-lg.loading  { opacity: 0.7; cursor: wait; }
</style>
