/**
 * sidebar.ts — sidebar collapse/resize state
 *
 * Shared between Sidebar.svelte (renders at this width, owns the drag
 * handle) and StatusBar.svelte (the appbar's show/hide toggle button) and
 * +layout.svelte (applies the width to the #app-shell grid track). Persisted
 * so a reload keeps the sidebar exactly as the user left it.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';

const WIDTH_KEY = 'lora-sidebar-width';
const COLLAPSED_KEY = 'lora-sidebar-collapsed';

const DEFAULT_WIDTH = 236;
export const MIN_WIDTH = 180;
export const MAX_WIDTH = 320;
export const COLLAPSE_THRESHOLD = 120;

function readWidth(): number {
  if (!browser) return DEFAULT_WIDTH;
  const stored = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(stored) && stored >= MIN_WIDTH && stored <= MAX_WIDTH
    ? stored
    : DEFAULT_WIDTH;
}

function readCollapsed(): boolean {
  return browser && localStorage.getItem(COLLAPSED_KEY) === '1';
}

export const sidebarWidth = writable<number>(readWidth());
export const sidebarCollapsed = writable<boolean>(readCollapsed());

sidebarWidth.subscribe((w) => {
  if (browser) localStorage.setItem(WIDTH_KEY, String(w));
});
sidebarCollapsed.subscribe((c) => {
  if (browser) localStorage.setItem(COLLAPSED_KEY, c ? '1' : '0');
});

export function toggleSidebarCollapsed(): void {
  sidebarCollapsed.update((c) => !c);
}
