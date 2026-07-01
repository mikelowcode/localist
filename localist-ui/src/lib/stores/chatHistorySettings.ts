/**
 * chatHistorySettings.ts — Chat History Tab settings store
 *
 * Owns the chat_turns eviction-preset setting:
 *   GET /api/chat/history/settings — read the current preset
 *   PUT /api/chat/history/settings — set the preset
 *
 * This is a separate concern from chatHistory.ts (the live-session turn
 * store) and from the searchable chat_turns list (a later step) — this
 * file only ever touches chat_history_settings.
 */

import { writable, type Writable } from 'svelte/store';

export type EvictionPreset = '7d' | '30d' | '90d' | 'forever';

export interface ChatHistorySettingsState {
  eviction_preset: string | null;
}

export const chatHistorySettings: Writable<ChatHistorySettingsState> =
  writable({ eviction_preset: null });

export const chatHistorySettingsLoading: Writable<boolean> = writable(false);
export const chatHistorySettingsError: Writable<string | null> = writable(null);

export async function loadChatHistorySettings(): Promise<void> {
  chatHistorySettingsLoading.set(true);
  chatHistorySettingsError.set(null);
  try {
    const res = await fetch('/api/chat/history/settings');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: ChatHistorySettingsState = await res.json();
    chatHistorySettings.set(data);
  } catch (err) {
    chatHistorySettingsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    chatHistorySettingsLoading.set(false);
  }
}

/**
 * On failure, chatHistorySettings is left untouched — the dropdown must
 * reflect the server's actual last-known state, not the user's pending
 * selection, when the write didn't land.
 */
export async function setChatHistoryEvictionPreset(preset: EvictionPreset): Promise<void> {
  chatHistorySettingsLoading.set(true);
  chatHistorySettingsError.set(null);
  try {
    const res = await fetch('/api/chat/history/settings', {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ eviction_preset: preset }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: ChatHistorySettingsState = await res.json();
    chatHistorySettings.set(data);
  } catch (err) {
    chatHistorySettingsError.set(err instanceof Error ? err.message : String(err));
  } finally {
    chatHistorySettingsLoading.set(false);
  }
}
