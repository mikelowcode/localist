/**
 * previewBlocks.ts — per-box collapse state for the Live Feed panel's
 * individual blocks (News, GitHub Watch, Hacker News)
 *
 * Sibling to previewsPanel.ts, which owns the whole-panel collapsed state.
 * That one answers "do I want to see the Live Feed panel at all"; this one
 * answers "which of its boxes do I want expanded right now" — a separate,
 * per-block concern, so a user who cares about GitHub Watch but not the
 * Daily News Brief can collapse just the one they don't need without
 * losing the whole panel.
 *
 * Persisted the same way (localStorage, same "keeps state exactly as the
 * user left it across a reload" rationale) but as a single JSON-encoded
 * record rather than one key per block, so adding a future block never
 * requires a new top-level localStorage key.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';

export type PreviewBlockKey = 'news' | 'github' | 'hackerNews';

const STORAGE_KEY = 'localist-preview-blocks-collapsed';

// Defaults to all expanded — collapsing a box is an opt-in decluttering
// action, not the default state (contrast previewsPanel.ts, whose default
// is collapsed since the whole panel itself is new and low-traffic).
function readCollapsed(): Record<PreviewBlockKey, boolean> {
  const defaults: Record<PreviewBlockKey, boolean> = { news: false, github: false, hackerNews: false };
  if (!browser) return defaults;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaults;
    return { ...defaults, ...JSON.parse(raw) };
  } catch {
    return defaults;
  }
}

export const previewBlocksCollapsed = writable<Record<PreviewBlockKey, boolean>>(readCollapsed());

previewBlocksCollapsed.subscribe((state) => {
  if (browser) localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
});

export function togglePreviewBlock(key: PreviewBlockKey): void {
  previewBlocksCollapsed.update((state) => ({ ...state, [key]: !state[key] }));
}
