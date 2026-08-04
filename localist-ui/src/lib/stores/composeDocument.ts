/**
 * composeDocument.ts — Compose Mode document draft, scoped per conversation
 *
 * Compose Mode lets a user accumulate a long document across several chat
 * turns (see docs/architecture/07-localist-ui.md, "Compose Mode") and edit
 * the whole thing as one artifact before saving via the existing
 * POST /files/generated flow (stores/files.ts's saveGeneratedFile()) — no
 * backend involvement of its own. Turn-by-turn "Add to document" wiring
 * lands in a later phase; this store only owns the draft's lifecycle.
 *
 * Scoped per conversation_id (mirrors chatHistoryStore's reset-on-
 * conversation-switch pattern in conversation/[id]/+page.svelte) and
 * persisted to localStorage per conversation, so switching conversations
 * swaps in that conversation's own draft rather than leaking one
 * conversation's in-progress document into another, and a reload doesn't
 * lose an in-progress document.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';
import { currentConversationId } from './conversation';

export interface ComposeState {
  active:       boolean;
  draft:        string;
  addedTaskIds: string[];   // turn keys (task_id, or timestamp fallback — see ChatPanel's provKey())
                             // already appended to draft, so "Add to document" is idempotent per turn
}

const EMPTY_STATE: ComposeState = { active: false, draft: '', addedTaskIds: [] };
const STORAGE_PREFIX = 'lora-compose-doc-';

function storageKey(conversationId: string): string {
  return STORAGE_PREFIX + conversationId;
}

function readState(conversationId: string): ComposeState {
  if (!browser || !conversationId) return { ...EMPTY_STATE, addedTaskIds: [] };
  try {
    const raw = localStorage.getItem(storageKey(conversationId));
    if (!raw) return { ...EMPTY_STATE, addedTaskIds: [] };
    const parsed = JSON.parse(raw);
    return {
      active:       !!parsed.active,
      draft:        typeof parsed.draft === 'string' ? parsed.draft : '',
      addedTaskIds: Array.isArray(parsed.addedTaskIds)
        ? parsed.addedTaskIds.filter((x: unknown) => typeof x === 'string')
        : [],
    };
  } catch {
    return { ...EMPTY_STATE, addedTaskIds: [] };
  }
}

export const composeDocument = writable<ComposeState>({ ...EMPTY_STATE });

let currentKey: string | null = null;

// Keep composeDocument scoped to whichever conversation is currently open.
currentConversationId.subscribe((id) => {
  if (id === currentKey) return;
  currentKey = id;
  composeDocument.set(readState(id));
});

// Write-through, same convention as conversation.ts/chatHistory-adjacent stores.
composeDocument.subscribe((state) => {
  if (!browser || !currentKey) return;
  localStorage.setItem(storageKey(currentKey), JSON.stringify(state));
});

export function toggleComposeMode(): void {
  composeDocument.update((s) => ({ ...s, active: !s.active }));
}

export function setComposeDraft(draft: string): void {
  composeDocument.update((s) => ({ ...s, draft }));
}

/**
 * Append a turn's content to the draft and mark it added — idempotent per
 * turnKey (a repeat call, e.g. from a stray double-click, is a no-op).
 * Always appends to whatever the draft currently is, never recomputed from
 * scratch, so a prior hand-edit is never clobbered by a later addition.
 */
export function addTurnToDocument(turnKey: string, content: string): void {
  composeDocument.update((s) => {
    if (s.addedTaskIds.includes(turnKey)) return s;
    const separator = s.draft.trim() ? '\n\n' : '';
    return {
      ...s,
      draft:        s.draft + separator + content,
      addedTaskIds: [...s.addedTaskIds, turnKey],
    };
  });
}

/**
 * Reset the draft and added-turn tracking, keeping compose mode active —
 * lets the user start a fresh document in the same session (e.g. after
 * saving one) rather than accumulating indefinitely. Callers (the panel's
 * Clear button) are responsible for confirming with the user first, since
 * this is destructive and unsaved draft text is not recoverable afterward.
 */
export function clearComposeDraft(): void {
  composeDocument.update((s) => ({ ...s, draft: '', addedTaskIds: [] }));
}

// ── Panel width (drag-resize) ────────────────────────────────────────────
// A layout preference, not document content — global rather than scoped per
// conversation (mirrors stores/sidebar.ts's sidebarWidth, same drag-handle
// pattern reused in ComposeDocumentPanel.svelte, mirrored from
// Sidebar.svelte's own divider).

const WIDTH_KEY = 'lora-compose-panel-width';
const DEFAULT_WIDTH = 320;   // matches --previews-w, the panel's prior fixed width
export const COMPOSE_MIN_WIDTH = 280;
export const COMPOSE_MAX_WIDTH = 800;

function readWidth(): number {
  if (!browser) return DEFAULT_WIDTH;
  const stored = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(stored) && stored >= COMPOSE_MIN_WIDTH && stored <= COMPOSE_MAX_WIDTH
    ? stored
    : DEFAULT_WIDTH;
}

export const composePanelWidth = writable<number>(readWidth());

composePanelWidth.subscribe((w) => {
  if (browser) localStorage.setItem(WIDTH_KEY, String(w));
});
