/**
 * newsBrief.ts — Daily News Brief header-button store
 *
 * Two deliberately separate calls (docs/daily-news-brief-plan.md §6):
 *   GET  /api/news/brief/preview — read-only, never calls NewsAPI. Feeds
 *                                  the header button's hover popover.
 *   POST /api/news/brief/open    — the click handler. Reopens today's
 *                                  conversation if it already exists,
 *                                  otherwise generates it (fetch + format,
 *                                  zero inference cost) and returns a new
 *                                  conversation_id either way.
 *
 * fetchPreview() is safe to call on hover/idle — it has no side effects.
 * openBrief() is the only function that can trigger NewsAPI calls or write
 * chat_turns, and only does so on a genuine cache miss.
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
    const data: { conversation_id: string; generated: boolean } = await res.json();
    return data.conversation_id;
  } catch (err) {
    newsBriefError.set(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    newsBriefOpening.set(false);
  }
}
