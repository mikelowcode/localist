/**
 * previewsPanel.ts — right-side "Previews" tab collapse state
 *
 * Shared between PreviewsPanel.svelte (renders the tab, owns the toggle
 * button) and +layout.svelte (applies collapsed/expanded width to the
 * #app-shell grid track), same split as sidebar.ts uses for the left
 * sidebar. Persisted so a reload keeps the tab exactly as the user left it.
 *
 * Unlike the left sidebar, this tab has no drag-resize — just two fixed
 * widths (collapsed strip vs. expanded panel), so there's no width store,
 * only the collapsed boolean.
 */

import { writable } from 'svelte/store';
import { browser } from '$app/environment';

const COLLAPSED_KEY = 'lora-previews-panel-collapsed';

// Defaults to collapsed — this is a new, low-traffic panel and shouldn't
// eat horizontal space for existing users until they opt in by expanding it.
function readCollapsed(): boolean {
  if (!browser) return true;
  return localStorage.getItem(COLLAPSED_KEY) !== '0';
}

export const previewsPanelCollapsed = writable<boolean>(readCollapsed());

previewsPanelCollapsed.subscribe((c) => {
  if (browser) localStorage.setItem(COLLAPSED_KEY, c ? '1' : '0');
});

export function togglePreviewsPanel(): void {
  previewsPanelCollapsed.update((c) => !c);
}
