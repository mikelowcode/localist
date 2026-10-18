/**
 * reembedCorpus.ts — manual re-embed of every embedded store (wiki/raw
 * corpus, chat history, episodes)
 *
 * Talks to POST /memory/reembed (docs/architecture/16-runtime-backend-layer.md
 * §16.4), the explicit counterpart to episodes' automatic startup re-embed.
 *
 * Follows runtimeBackendSwitch.ts's pattern: on failure, state is left
 * untouched (no optimistic update) — the UI must reflect the server's actual
 * last-known state, not a pending guess.
 */

import { writable, type Writable } from 'svelte/store';
import { apiUrl } from '$lib/api';

// One gate's outcome within CalibrationResponse — mirrors
// main.py's GateCalibrationResponse (PLAN_semantic_gating_calibration.md
// §2/§6).
export interface GateCalibrationResponse {
  threshold: number | null;
  degenerate: boolean;
  reason: string | null;
  positive_count: number;
  negative_count: number;
}

// Live semantic-gating threshold calibration result, attached to both
// POST /settings/embedding-model (first-time, on switch) and
// POST /memory/reembed (every explicit re-trigger) — see main.py's
// CalibrationResponse.
export interface CalibrationResponse {
  model: string;
  gates: Record<string, GateCalibrationResponse>;
}

// One store's outcome — mirrors main.py's ReembedStoreCounts. A non-empty
// failed_ids means rows were left on their previous embedding after retries
// (§16.4); the store wasn't advanced to the active model, and re-running
// the re-embed retries just those rows.
export interface ReembedStoreCounts {
  reembedded: number;
  total:      number;
  failed_ids: number[];
}

// POST /memory/reembed re-embeds every store: the top-level counts are the
// wiki/raw corpus, chat_turns/episodes the other two. Those two are optional
// only so an older backend's corpus-only response still parses.
export interface ReembedCorpusResponse {
  reembedded:  number;
  total:       number;
  model:       string | null;
  failed_ids?: number[];
  chat_turns?: ReembedStoreCounts | null;
  episodes?:   ReembedStoreCounts | null;
  calibration: CalibrationResponse | null;
}

export const reembedLoading: Writable<boolean> = writable(false);
export const reembedError: Writable<string | null> = writable(null);

export async function reembedCorpus(): Promise<ReembedCorpusResponse | null> {
  reembedLoading.set(true);
  reembedError.set(null);
  try {
    const res = await fetch(apiUrl('/api/memory/reembed'), { method: 'POST' });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return (await res.json()) as ReembedCorpusResponse;
  } catch (err) {
    reembedError.set(err instanceof Error ? err.message : String(err));
    return null;
  } finally {
    reembedLoading.set(false);
  }
}
