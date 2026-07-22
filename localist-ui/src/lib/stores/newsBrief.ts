/**
 * newsBrief.ts — Daily News Brief Live Feed panel store
 *
 * Two deliberately separate calls (docs/daily-news-brief-plan.md §6/§12):
 *   GET  /api/news/brief/preview — read-only, never calls NewsAPI. Feeds
 *                                  the Live Feed panel's News block.
 *   POST /api/news/brief/open    — the "Daily News Brief Refresh" link's
 *                                  click handler. Always fetches a fresh
 *                                  brief and returns a brand-new
 *                                  conversation_id — deliberately not
 *                                  idempotent within a day, since an
 *                                  earlier same-day-reopen design meant
 *                                  pressing a link literally labeled
 *                                  "Refresh" could silently navigate into
 *                                  an old conversation showing stale
 *                                  articles instead (confirmed live,
 *                                  2026-07-22).
 *
 * fetchPreview() is safe to call on expand/idle — it has no side effects.
 * openBrief() is the only function that triggers real NewsAPI calls.
 */

import { writable, type Writable } from 'svelte/store';
import { SESSION_ID } from '$lib/stores/tasks';

export interface NewsBriefArticle {
  title:        string;
  description:  string;
  source:       string;
  published_at: string;
  url:          string;
}

export interface NewsBriefSection {
  key:      string;
  label:    string;
  articles: NewsBriefArticle[];
  error:    string | null;
}

export interface NewsBriefPreview {
  available:  boolean;
  brief_date: string | null;
  sections:   NewsBriefSection[];
}

export const newsBriefPreview: Writable<NewsBriefPreview> =
  writable({ available: false, brief_date: null, sections: [] });
export const newsBriefPreviewLoading: Writable<boolean> = writable(false);

export const newsBriefOpening: Writable<boolean> = writable(false);
export const newsBriefError: Writable<string | null> = writable(null);

export async function fetchNewsBriefPreview(): Promise<void> {
  newsBriefPreviewLoading.set(true);
  try {
    const res = await fetch('/api/news/brief/preview');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: NewsBriefPreview = await res.json();
    newsBriefPreview.set(data);
  } catch (err) {
    console.warn('Failed to load news brief preview:', err);
  } finally {
    newsBriefPreviewLoading.set(false);
  }
}

/**
 * Triggers generation (if today's brief doesn't exist yet) or reopens the
 * existing one. Returns the conversation_id to navigate to, or null on
 * failure — the caller (StatusBar.svelte) is responsible for navigation so
 * this store stays UI-framework-decision-free, same separation
 * reembedCorpus.ts/runtimeBackendSwitch.ts already use.
 */
export async function openNewsBrief(): Promise<string | null> {
  newsBriefOpening.set(true);
  newsBriefError.set(null);
  try {
    const res = await fetch('/api/news/brief/open', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: SESSION_ID }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    const data: { conversation_id: string } = await res.json();
    return data.conversation_id;
  } catch (err) {
    newsBriefError.set(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    newsBriefOpening.set(false);
  }
}
