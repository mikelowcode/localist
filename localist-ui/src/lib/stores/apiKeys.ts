/**
 * apiKeys.ts — optional API keys settings store (BRAVE / LANGSEARCH / NEWSAPI / GITHUB)
 *
 * Talks to:
 *   GET /api/settings/api-keys — which keys are currently configured (booleans only)
 *   PUT /api/settings/api-keys — set one or more keys, .env-backed
 *
 * The backend never echoes a key's value back, only whether it's set — so
 * this store never holds a plaintext key either, just the *_set booleans.
 * Follows retentionSettings.ts's pattern: on failure, state is left
 * untouched (no optimistic update).
 *
 * The backend refuses a malformed key or a GitHub token that GitHub rejects
 * (a save then fails with nothing changed, and the message lands in
 * apiKeysError). On a successful save it tells localist-mcp to reload its keys,
 * so no restart is needed. The `warning` field carries anything worth showing:
 * "Connected to GitHub", a write failure, or restart instructions if that
 * reload failed.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface ApiKeysStatus {
  langsearch_api_key_set: boolean;
  brave_api_key_set: boolean;
  newsapi_api_key_set: boolean;
  github_token_set: boolean;
}

const DEFAULT_STATE: ApiKeysStatus = {
  langsearch_api_key_set: false,
  brave_api_key_set: false,
  newsapi_api_key_set: false,
  github_token_set: false
};

export const apiKeysStatus: Writable<ApiKeysStatus> = writable(DEFAULT_STATE);
export const apiKeysLoading: Writable<boolean> = writable(false);
export const apiKeysError: Writable<string | null> = writable(null);

export async function loadApiKeysStatus(): Promise<void> {
  apiKeysLoading.set(true);
  apiKeysError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/api-keys'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: ApiKeysStatus = await res.json();
    apiKeysStatus.set(data);
  } catch (err) {
    apiKeysError.set(err instanceof Error ? err.message : String(err));
  } finally {
    apiKeysLoading.set(false);
  }
}

export interface ApiKeysUpdate {
  langsearch_api_key?: string;
  brave_api_key?: string;
  newsapi_api_key?: string;
  github_token?: string;
}

/** Returns the save-success warning (e.g. "restart localist-mcp"), or null on failure. */
export async function setApiKeys(update: ApiKeysUpdate): Promise<string | null> {
  apiKeysLoading.set(true);
  apiKeysError.set(null);
  try {
    const res = await fetch(apiUrl('/api/settings/api-keys'), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(update)
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    const data: ApiKeysStatus & { warning: string | null } = await res.json();
    apiKeysStatus.set({
      langsearch_api_key_set: data.langsearch_api_key_set,
      brave_api_key_set: data.brave_api_key_set,
      newsapi_api_key_set: data.newsapi_api_key_set,
      github_token_set: data.github_token_set
    });
    return data.warning ?? '';
  } catch (err) {
    apiKeysError.set(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    apiKeysLoading.set(false);
  }
}
