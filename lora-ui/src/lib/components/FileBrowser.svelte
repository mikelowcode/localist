<script lang="ts">
  import { onMount } from 'svelte';

  interface FileEntry {
    name: string;
    path: string;
    size?: number;
    modified?: string;
    type: 'wiki' | 'raw';
  }

  let wikiFiles: FileEntry[] = [];
  let rawFiles: FileEntry[]  = [];
  let loading = true;
  let error: string | null = null;
  let selectedFile: FileEntry | null = null;
  let fileContent: string | null = null;
  let contentLoading = false;

  let dragging = false;
  let uploading = false;
  let uploadError: string | null = null;
  let uploadSuccess: string | null = null;

  const BASE = '/api';

  async function loadFiles() {
    loading = true;
    error = null;
    try {
      // Try fetching wiki and raw file listings from backend
      // The backend doesn't expose these endpoints yet, so we gracefully
      // handle 404 and show the empty state with upload capability.
      const [wikiRes, rawRes] = await Promise.all([
        fetch(`${BASE}/files/wiki`).catch(() => null),
        fetch(`${BASE}/files/raw`).catch(() => null)
      ]);

      if (wikiRes?.ok) {
        const data = await wikiRes.json();
        wikiFiles = (data.files ?? []).map((f: any) => ({ ...f, type: 'wiki' }));
      }
      if (rawRes?.ok) {
        const data = await rawRes.json();
        rawFiles = (data.files ?? []).map((f: any) => ({ ...f, type: 'raw' }));
      }
    } catch (err) {
      error = String(err);
    } finally {
      loading = false;
    }
  }

  async function openFile(file: FileEntry) {
    if (selectedFile?.path === file.path && fileContent !== null) {
      selectedFile = null;
      fileContent = null;
      return;
    }
    selectedFile = file;
    contentLoading = true;
    fileContent = null;
    try {
      const res = await fetch(`${BASE}/files/content?path=${encodeURIComponent(file.path)}`);
      if (res.ok) {
        fileContent = await res.text();
      } else {
        fileContent = `Could not load file content (HTTP ${res.status}).`;
      }
    } catch {
      fileContent = 'Network error loading file content.';
    } finally {
      contentLoading = false;
    }
  }

  // ── Drag-and-drop upload ─────────────────────
  function onDragOver(e: DragEvent) {
    e.preventDefault();
    dragging = true;
  }
  function onDragLeave() { dragging = false; }

  async function onDrop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    const files = e.dataTransfer?.files;
    if (files?.length) await uploadFiles(files);
  }

  async function onFileInput(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) await uploadFiles(input.files);
    input.value = '';
  }

  async function uploadFiles(files: FileList) {
    uploading = true;
    uploadError = null;
    uploadSuccess = null;

    for (const file of Array.from(files)) {
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!['md', 'txt'].includes(ext ?? '')) {
        uploadError = `Unsupported file type: .${ext}. Only .md and .txt files are accepted.`;
        uploading = false;
        return;
      }
    }

    try {
      const formData = new FormData();
      for (const file of Array.from(files)) {
        formData.append('files', file);
      }
      const res = await fetch(`${BASE}/files/upload`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        uploadSuccess = `${files.length} file${files.length > 1 ? 's' : ''} uploaded successfully.`;
        await loadFiles();
      } else {
        uploadError = `Upload failed: HTTP ${res.status}`;
      }
    } catch (err) {
      uploadError = `Upload error: ${String(err)}`;
    } finally {
      uploading = false;
    }
  }

  function formatSize(bytes?: number): string {
    if (bytes === undefined) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  onMount(loadFiles);
</script>

<div class="file-browser">

  <!-- Left: file listings -->
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
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="16 16 12 12 8 16"/>
            <line x1="12" y1="12" x2="12" y2="21"/>
            <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
          </svg>
          <span>Drop .md / .txt files or <span class="upload-link">browse</span></span>
        {/if}
      </label>
      {#if uploadSuccess}
        <p class="upload-feedback success">{uploadSuccess}</p>
      {/if}
      {#if uploadError}
        <p class="upload-feedback error">{uploadError}</p>
      {/if}
    </div>

    <!-- Section: Wiki pages -->
    <div class="file-section">
      <div class="section-header">
        <h3 class="section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
          Wiki Pages
        </h3>
        <span class="section-count">{wikiFiles.length}</span>
      </div>

      {#if loading}
        <div class="loading-list">
          {#each [80, 60, 72, 55] as w}
            <div class="skeleton-row" style="width:{w}%"/>
          {/each}
        </div>
      {:else if wikiFiles.length === 0}
        <p class="empty-section">No wiki pages yet.</p>
      {:else}
        <ul class="file-list" role="list">
          {#each wikiFiles as file}
            <li>
              <button
                class="file-item"
                class:selected={selectedFile?.path === file.path}
                on:click={() => openFile(file)}
                aria-pressed={selectedFile?.path === file.path}
              >
                <svg class="file-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
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

    <!-- Section: Raw files -->
    <div class="file-section">
      <div class="section-header">
        <h3 class="section-title">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
          Raw Files
        </h3>
        <span class="section-count">{rawFiles.length}</span>
      </div>

      {#if loading}
        <div class="loading-list">
          {#each [65, 80, 45] as w}
            <div class="skeleton-row" style="width:{w}%"/>
          {/each}
        </div>
      {:else if rawFiles.length === 0}
        <p class="empty-section">No raw files yet. Upload .md or .txt files above.</p>
      {:else}
        <ul class="file-list" role="list">
          {#each rawFiles as file}
            <li>
              <button
                class="file-item"
                class:selected={selectedFile?.path === file.path}
                on:click={() => openFile(file)}
                aria-pressed={selectedFile?.path === file.path}
              >
                <svg class="file-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                  <polyline points="13 2 13 9 20 9"/>
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

  </div>

  <!-- Right: file content viewer -->
  <div class="content-pane">
    {#if !selectedFile}
      <div class="content-empty">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="opacity:0.2">
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
          <span class="badge {selectedFile.type === 'wiki' ? 'badge-success' : 'badge-warning'}">{selectedFile.type}</span>
          <h3 class="content-filename">{selectedFile.name}</h3>
        </div>
        <button class="close-btn" on:click={() => { selectedFile = null; fileContent = null; }} aria-label="Close file">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
      <div class="content-body">
        {#if contentLoading}
          <div class="content-loading">
            <span class="spinner" aria-hidden="true" />
            <span class="text-tertiary text-sm">Loading…</span>
          </div>
        {:else}
          <pre class="file-content-pre"><code>{fileContent}</code></pre>
        {/if}
      </div>
    {/if}
  </div>

</div>

<style>
  .file-browser {
    display: grid;
    grid-template-columns: 280px 1fr;
    height: 100%;
    overflow: hidden;
  }

  /* ── Left pane ─────────────────────────────── */
  .file-pane {
    display: flex;
    flex-direction: column;
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
    transition: border-color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
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

  .upload-link {
    color: var(--accent);
    text-decoration: underline;
  }

  .upload-feedback {
    font-size: var(--text-xs);
    text-align: center;
  }
  .upload-feedback.success { color: var(--success); }
  .upload-feedback.error   { color: var(--error); }

  /* Spinner */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* File sections */
  .file-section {
    flex-shrink: 0;
    padding: var(--sp-2) 0 var(--sp-4);
    border-top: 1px solid var(--border-soft);
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
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
  }

  .section-count {
    font-size: 10px;
    font-family: var(--font-mono);
    background: var(--bg-active);
    color: var(--text-tertiary);
    padding: 1px 7px;
    border-radius: 100px;
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

  .file-item {
    width: 100%;
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
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }

  .file-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .file-item.selected {
    background: var(--accent-glow);
    color: var(--text-accent);
  }

  .file-icon { flex-shrink: 0; opacity: 0.6; }
  .file-name { flex: 1; font-family: var(--font-mono); }
  .file-size { flex-shrink: 0; color: var(--text-muted); font-family: var(--font-mono); }

  /* Loading skeletons */
  .loading-list { padding: var(--sp-2) var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-2); }
  .skeleton-row {
    height: 10px;
    background: var(--bg-active);
    border-radius: var(--radius-sm);
    animation: pulse 1.5s ease-in-out infinite;
  }

  /* ── Right pane ────────────────────────────── */
  .content-pane {
    display: flex;
    flex-direction: column;
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
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .close-btn:hover { background: var(--bg-hover); color: var(--text-primary); }

  .content-body { flex: 1; overflow-y: auto; }

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
</style>
