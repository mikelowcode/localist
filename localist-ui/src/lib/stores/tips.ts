/**
 * tips.ts — Tips panel state
 *
 * Tracks which tip is currently shown (cycled through on click) and whether
 * the user has dismissed the Tips panel. Persisted so a reload keeps the
 * panel exactly as the user left it.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';
import { tips } from '$lib/data/tips';

const INDEX_KEY = 'localist-tips-index';
const DISMISSED_KEY = 'localist-tips-dismissed';

function readIndex(): number {
  if (!browser) return 0;
  const stored = Number(localStorage.getItem(INDEX_KEY));
  return Number.isInteger(stored) && stored >= 0 && stored < tips.length ? stored : 0;
}

function readDismissed(): boolean {
  return browser && localStorage.getItem(DISMISSED_KEY) === '1';
}

export const tipIndex = writable<number>(readIndex());
export const tipsDismissed = writable<boolean>(readDismissed());

tipIndex.subscribe((i) => {
  if (browser) localStorage.setItem(INDEX_KEY, String(i));
});
tipsDismissed.subscribe((d) => {
  if (browser) localStorage.setItem(DISMISSED_KEY, d ? '1' : '0');
});

export function cycleTip(): void {
  tipIndex.update((i) => (i + 1) % tips.length);
}

export function dismissTips(): void {
  tipsDismissed.set(true);
}

export function restoreTips(): void {
  tipsDismissed.set(false);
}
