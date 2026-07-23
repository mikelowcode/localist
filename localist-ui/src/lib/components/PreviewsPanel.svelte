<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { goto } from '$app/navigation';
  import { previewsPanelCollapsed, togglePreviewsPanel } from '$lib/stores/previewsPanel';
  import {
    newsBriefPreview,
    fetchNewsBriefPreview,
    newsBriefOpening,
    newsBriefError,
    openNewsBrief,
    type NewsBriefArticle
  } from '$lib/stores/newsBrief';
  import { tasksStore, submitTask } from '$lib/stores/tasks';
  import { chatHistoryStore } from '$lib/stores/chatHistory';
  import { currentConversationId, isFirstTurnOfConversation } from '$lib/stores/conversation';

  // Fetched once on mount regardless of collapsed state, so the preview is
  // already populated the moment the user expands the tab — this replaces
  // StatusBar's old hover-triggered fetch (docs/daily-news-brief-plan.md §8),
  // which only had room to show one truncated line per section.
  onMount(() => {
    void fetchNewsBriefPreview();
  });

  async function handleOpenBrief(): Promise<void> {
    const conversationId = await openNewsBrief();
    if (conversationId) {
      // /open already refreshed the backend's news_brief_cache by the time it
      // responds, but this panel's own store is only ever populated once on
      // mount — without this, the Live Feed list stays stale (showing
      // whatever was cached at page-load) until a full browser reload.
      await fetchNewsBriefPreview();
      await goto(`/conversation/${conversationId}`);
    }
  }

  // Sends just one clicked article into the currently open conversation,
  // mirroring ChatPanel.svelte's handleSubmit (optimistic turns + submitTask)
  // rather than files.ts's ingestFile pattern, since the user is expected to
  // watch this stream in live rather than land on it after the fact.
  async function handleAskAboutArticle(article: NewsBriefArticle): Promise<void> {
    if ($tasksStore.finalizing) return;

    const instruction = `Tell me more about this news story: "${article.title}"`;
    const task_id = crypto.randomUUID();
    const now = Date.now();
    const conversationId = get(currentConversationId);

    let conversationTitle: string | undefined;
    if (get(isFirstTurnOfConversation)) {
      conversationTitle = instruction.length > 60 ? instruction.slice(0, 60) + '…' : instruction;
      isFirstTurnOfConversation.set(false);
    }

    chatHistoryStore.update((turns) => [
      ...turns,
      { role: 'user', content: instruction, task_id, timestamp: now },
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

    await submitTask(
      instruction,
      { web_search_queries: [article.title], news_article_url: article.url },
      task_id,
      conversationId,
      conversationTitle
    );
    await goto(`/conversation/${conversationId}`);
  }
</script>

{#if $previewsPanelCollapsed}
  <button
    type="button"
    class="previews-tab-collapsed"
    on:click={togglePreviewsPanel}
    aria-label="Expand Live Feed panel"
    title="Live Feed"
  >
    <span class="previews-tab-label">Live Feed</span>
  </button>
{:else}
  <div class="previews-panel">
    <div class="previews-panel-header">
      <span class="previews-panel-title">Live Feed</span>
      <button
        type="button"
        class="previews-collapse-btn"
        on:click={togglePreviewsPanel}
        aria-label="Collapse Live Feed panel"
        title="Collapse"
      >›</button>
    </div>

    <div class="previews-panel-body">
      <!-- News block — live, moved out of StatusBar's cramped hover popover -->
      <section class="preview-block">
        <button
          type="button"
          class="preview-block-refresh-link"
          on:click={handleOpenBrief}
          disabled={$newsBriefOpening}
        >{$newsBriefOpening ? 'Refreshing…' : 'Daily News Brief Refresh'}</button>
        <div class="preview-block-body">
          {#if $newsBriefPreview.sections.length === 0}
            <p class="preview-empty">No brief generated yet today — click the link above to generate.</p>
          {:else}
            {#each $newsBriefPreview.sections as section (section.key)}
              <div class="preview-news-section">
                <div class="preview-news-section-label">{section.label}</div>
                {#if section.error}
                  <p class="preview-news-unavailable">unavailable</p>
                {:else if section.articles.length === 0}
                  <p class="preview-news-unavailable">no articles found</p>
                {:else}
                  {#each section.articles.slice(0, 3) as article}
                    <div class="preview-news-article-row">
                      <a
                        class="preview-news-article"
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <span class="preview-news-article-title">{article.title}</span>
                        <span class="preview-news-article-source">{article.source}</span>
                      </a>
                      <button
                        type="button"
                        class="preview-news-article-ask"
                        on:click={() => handleAskAboutArticle(article)}
                        disabled={$tasksStore.finalizing}
                        title="Ask about this story in the current chat"
                      >Ask about this</button>
                    </div>
                  {/each}
                {/if}
              </div>
            {/each}
          {/if}
          {#if $newsBriefError}
            <p class="preview-error">{$newsBriefError}</p>
          {/if}
        </div>
      </section>

      <!-- Reserved blocks — layout only, not wired to a live API yet. -->
      <section class="preview-block preview-block-reserved">
        <div class="preview-block-header">
          <span class="preview-block-title">🐙 GitHub</span>
          <span class="preview-block-badge">Coming soon</span>
        </div>
        <p class="preview-empty">Daily activity from watched repos will appear here.</p>
      </section>

      <section class="preview-block preview-block-reserved">
        <div class="preview-block-header">
          <span class="preview-block-title">💬 Hacker News</span>
          <span class="preview-block-badge">Coming soon</span>
        </div>
        <p class="preview-empty">Top stories will appear here.</p>
      </section>
    </div>
  </div>
{/if}

<style>
  .previews-tab-collapsed,
  .previews-panel {
    grid-column: 3;
    grid-row: 1 / -1;
    height: 100%;
    border-left: 1px solid var(--border);
    background: var(--bg-panel);
  }

  /* ── Collapsed: a slim always-visible tab strip ─────────────── */
  .previews-tab-collapsed {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--sp-3) 0;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease);
  }
  .previews-tab-collapsed:hover { background: var(--bg-hover); }

  .previews-tab-label {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .previews-tab-collapsed:hover .previews-tab-label { color: var(--text-secondary); }

  /* ── Expanded panel ──────────────────────────────────────────── */
  .previews-panel {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }

  .previews-panel-header {
    flex-shrink: 0;
    height: var(--topbar-h);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--sp-4);
    border-bottom: 1px solid var(--border);
  }

  .previews-panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .previews-collapse-btn {
    width: 22px;
    height: 22px;
    border-radius: var(--radius-sm);
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 15px;
    line-height: 1;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease);
  }
  .previews-collapse-btn:hover { background: var(--bg-hover); }

  .previews-panel-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-4);
    display: flex;
    flex-direction: column;
    gap: var(--sp-4);
  }

  /* ── Blob blocks ─────────────────────────────────────────────── */
  .preview-block {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-3);
  }

  .preview-block-reserved { opacity: 0.6; }

  .preview-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--sp-2);
    margin-bottom: var(--sp-2);
  }

  .preview-block-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .preview-block-badge {
    font-size: 10.5px;
    color: var(--text-tertiary);
    background: var(--bg-active);
    border-radius: var(--radius-sm);
    padding: 2px var(--sp-2);
  }

  .preview-block-refresh-link {
    display: inline-block;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-accent);
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    margin-bottom: var(--sp-2);
    text-decoration: underline;
  }
  .preview-block-refresh-link:hover:not(:disabled) { color: var(--accent-2); }
  .preview-block-refresh-link:disabled {
    color: var(--text-tertiary);
    cursor: default;
    text-decoration: none;
  }

  .preview-empty {
    font-size: 12px;
    color: var(--text-tertiary);
    line-height: 1.5;
  }

  .preview-news-section { margin-bottom: var(--sp-3); }
  .preview-news-section:last-child { margin-bottom: 0; }

  .preview-news-section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: var(--sp-1);
  }

  .preview-news-unavailable {
    font-size: 12px;
    color: var(--text-tertiary);
    font-style: italic;
  }

  .preview-news-article-row {
    position: relative;
    padding: var(--sp-1) 0;
  }
  .preview-news-article-row:hover .preview-news-article-ask { opacity: 1; }

  .preview-news-article {
    display: flex;
    flex-direction: column;
    gap: 2px;
    text-decoration: none;
  }
  .preview-news-article:hover .preview-news-article-title { color: var(--text-accent); }

  .preview-news-article-title {
    font-size: 12.5px;
    line-height: 1.4;
    color: var(--text-primary);
  }

  .preview-news-article-source {
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .preview-news-article-ask {
    position: absolute;
    top: var(--sp-1);
    right: 0;
    font-size: 10.5px;
    font-weight: 600;
    color: var(--text-accent);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 2px var(--sp-2);
    cursor: pointer;
    opacity: 0;
    transition: opacity var(--dur-fast) var(--ease);
  }
  .preview-news-article-ask:hover:not(:disabled) {
    color: var(--accent-2);
    border-color: var(--accent-2);
  }
  .preview-news-article-ask:disabled {
    color: var(--text-tertiary);
    cursor: default;
  }

  .preview-error {
    font-size: 12px;
    color: var(--error);
    margin-top: var(--sp-2);
  }
</style>
