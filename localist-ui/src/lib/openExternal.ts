/**
 * openExternal.ts — opens a URL in the system browser, working both in the
 * packaged Tauri app and the plain :5173 dev flow.
 *
 * Same class of bug as confirmDialog.ts: the packaged app's WKWebView does
 * not treat `<a target="_blank">` as "open externally" the way a real
 * browser tab does — Tauri's webview has no default handler for it, so
 * clicking those links silently no-ops. `isTauri()` (from
 * @tauri-apps/api/core) distinguishes the two contexts at runtime — same
 * source serves both — routing to the opener plugin's openUrl() (permissioned
 * via src-tauri/capabilities/default.json's opener:default) only when
 * actually running inside Tauri; the dev/browser flow keeps using
 * window.open(), which already works there.
 */

import { isTauri } from '@tauri-apps/api/core';
import { openUrl } from '@tauri-apps/plugin-opener';

export async function openExternal(url: string): Promise<void> {
  if (isTauri()) {
    await openUrl(url);
    return;
  }
  window.open(url, '_blank', 'noopener,noreferrer');
}
