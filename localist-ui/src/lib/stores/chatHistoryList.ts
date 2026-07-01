/**
 * chatHistoryList.ts — Chat History Tab searchable turn list
 *
 * Fetches from GET /api/chat/history and exposes paginated, searchable
 * chat_turns state to history/+page.svelte.
 *
 * Kept separate from chatHistorySettings.ts (the eviction-preset setting)
 * and chatHistory.ts (the live in-session turn store) — three distinct
 * concerns per this project's existing separation.
 */

import { writable, get, type Writable } from 'svelte/store';

export interface ChatTurnItem {
  id:             number;
  task_id:        string;
  role:           string;
  content:        string;
  sources:        Record<string, unknown>[];
  status_message: string | null;
  metadata:       Record<string, unknown>;
  created_at:     number;
}

export const CHAT_HISTORY_PAGE_SIZE = 25;

export const chatTurns: Writable<ChatTurnItem[]> = writable([]);
export const chatTurnsTotal: Writable<number> = writable(0);
export const chatTurnsLoading: Writable<boolean> = writable(false);
export const chatTurnsError: Writable<string | null> = writable(null);

export const chatHistoryQuery: Writable<string> = writable('');
export const chatHistoryOffset: Writable<number> = writable(0);

/**
 * Reads the current chatHistoryQuery/chatHistoryOffset values and fetches
 * that page. On failure, chatTurns/chatTurnsTotal are left unchanged —
 * same no-optimistic-update convention as chatHistorySettings.ts.
 */
export async function loadChatTurns(): Promise<void> {
  chatTurnsLoading.set(true);
  chatTurnsError.set(null);

  const query  = get(chatHistoryQuery);
  const offset = get(chatHistoryOffset);

  const params = new URLSearchParams();
  if (query) params.set('q', query);   // omit entirely when empty
  params.set('limit',  String(CHAT_HISTORY_PAGE_SIZE));
  params.set('offset', String(offset));

  try {
    const res = await fetch(`/api/chat/history?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { turns: ChatTurnItem[]; total: number; offset: number; limit: number }
      = await res.json();
    chatTurns.set(data.turns);
    chatTurnsTotal.set(data.total);
  } catch (err) {
    chatTurnsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    chatTurnsLoading.set(false);
  }
}

/** Call whenever the search text changes so a new search starts at page 1. */
export function resetChatHistoryOffset(): void {
  chatHistoryOffset.set(0);
}
