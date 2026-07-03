/**
 * conversation.ts — conversation_id tracking for the Chat tab
 *
 * Owns the frontend-generated conversation_id that groups chat_turns
 * server-side (a separate concept from tasks.ts's SESSION_ID, which is a
 * page-load-scoped id for backend working-memory grouping — do not conflate
 * the two, and do not touch SESSION_ID here).
 *
 * conversation_id is persisted to localStorage so it survives page reloads;
 * a brand new id is only minted on first-ever load or when the user
 * explicitly starts a new conversation via startNewConversation() (the
 * future '+' button).
 */

import { writable, type Writable } from 'svelte/store';
import { browser } from '$app/environment';

const STORAGE_KEY = 'localist:conversationId';

function loadOrCreateConversationId(): string {
  if (!browser) return crypto.randomUUID();
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) return stored;
  const fresh = crypto.randomUUID();
  localStorage.setItem(STORAGE_KEY, fresh);
  return fresh;
}

export const currentConversationId: Writable<string> = writable(loadOrCreateConversationId());

// Write-through: any update to the store (including from startNewConversation)
// is persisted to localStorage.
currentConversationId.subscribe((id) => {
  if (browser) localStorage.setItem(STORAGE_KEY, id);
});

/**
 * Tracks whether the *next* submitted message is the first turn of the
 * current conversation_id. Starts true (covers true first load), is reset
 * to true by startNewConversation(), and must be flipped to false by the
 * caller the instant a title is actually sent (before the request goes out,
 * not after it resolves) so a rapid double-submit can't send a title twice.
 * Backend contract: conversation_title must be sent on exactly the first
 * turn of a conversation_id, and never after.
 */
export const isFirstTurnOfConversation: Writable<boolean> = writable(true);

// ── Start a brand new conversation ───────────────────────────
// Called by the future '+' button (sidebar prompt). Not wired to any UI yet.
export function startNewConversation(): string {
  const id = crypto.randomUUID();
  if (browser) localStorage.setItem(STORAGE_KEY, id);
  currentConversationId.set(id);
  isFirstTurnOfConversation.set(true);
  return id;
}
