import { writable, derived, get } from 'svelte/store';
import { apiUrl } from '$lib/api';

export interface HealthState {
  healthy: boolean;
  reachable: boolean;
  backend: string;
  base_url: string;
  models: string[];
  // The chat model id the active runtime is configured with — named by
  // ModelStatusBanner when chat_model_found is false.
  chat_model: string;
  chat_model_found: boolean;
  embed_model_found: boolean;
  embedding_model: string;
  error?: string;
  last_checked: number | null;
  checking: boolean;
  // False when the last GET /api/health itself failed — the Localist
  // backend is down, as opposed to the runtime backend it talks to.
  api_ok: boolean;
  // True once GET /api/health has succeeded at least once this session.
  ever_ok: boolean;
  // True once STARTUP_GRACE_MS has passed since polling started. Until the
  // backend first answers AND while this is false, a failed health check
  // means "still starting" (the packaged app launches its backend sidecar
  // alongside the window), not "down" — ModelStatusBanner and the appbar
  // chip stay quiet instead of flashing an error on every launch.
  startup_grace_over: boolean;
  // active_embedding_model_name / gate_tiers (PLAN_semantic_gating_
  // calibration.md §6): the RESOLVED model name Planner compares against
  // its tuned model, and a per-gate trust tier ("tuned"/"validated"/
  // "auto-calibrated"/"disabled") for the Settings page's trust badge —
  // refreshed on every health poll, not just after a switch/reembed.
  active_embedding_model_name: string | null;
  gate_tiers: Record<string, string>;
}

export interface AgentsState {
  agents: string[];
  loaded: boolean;
}

const initialHealth: HealthState = {
  healthy: false,
  reachable: false,
  backend: '',
  base_url: '',
  models: [],
  chat_model: '',
  chat_model_found: false,
  embed_model_found: false,
  embedding_model: '',
  last_checked: null,
  checking: false,
  api_ok: true,
  ever_ok: false,
  startup_grace_over: false,
  active_embedding_model_name: null,
  gate_tiers: {}
};

export const health = writable<HealthState>(initialHealth);
export const agents = writable<AgentsState>({ agents: [], loaded: false });

// Derived: simple connectivity label
export const connectivityLabel = derived(health, ($h) => {
  if ($h.checking) return 'checking';
  if (isStillStarting($h)) return 'checking';
  if ($h.healthy)  return 'online';
  if ($h.reachable) return 'degraded';
  return 'offline';
});

export async function checkHealth(): Promise<void> {
  health.update((s) => ({ ...s, checking: true }));
  try {
    const res = await fetch(apiUrl('/api/health'));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    health.set({
      ...data,
      last_checked: Date.now(),
      checking: false,
      api_ok: true,
      ever_ok: true,
      startup_grace_over: startupGraceOver()
    });
  } catch (err) {
    health.update((s) => ({
      ...s,
      healthy: false,
      reachable: false,
      error: String(err),
      last_checked: Date.now(),
      checking: false,
      api_ok: false,
      startup_grace_over: startupGraceOver()
    }));
  }
}

export async function loadAgents(): Promise<void> {
  try {
    const res = await fetch(apiUrl('/api/agents'));
    if (!res.ok) return;
    const data = await res.json();
    agents.set({ agents: data.agents ?? [], loaded: true });
  } catch {
    // silently ignore — agents panel just shows empty
  }
}

// Poll health every 15s — but every 2s until the backend first answers,
// so a normal launch goes from "starting" to healthy in about a second
// instead of waiting out a full 15s interval (called once from root layout).
const POLL_MS = 15_000;
const STARTUP_POLL_MS = 2_000;
// How long a backend that has never answered counts as "still starting"
// rather than "down". The packaged app's sidecar normally answers within a
// few seconds; this leaves headroom for a cold first launch.
const STARTUP_GRACE_MS = 20_000;

let _pollTimer: ReturnType<typeof setTimeout> | null = null;
let _pollingStartedAt: number | null = null;

function startupGraceOver(): boolean {
  return _pollingStartedAt !== null && Date.now() - _pollingStartedAt >= STARTUP_GRACE_MS;
}

// The backend hasn't answered yet, but it's too early to call it down.
export function isStillStarting(h: HealthState): boolean {
  return !h.api_ok && !h.ever_ok && !h.startup_grace_over;
}

async function _pollTick(): Promise<void> {
  await checkHealth();
  loadAgents();
  if (_pollingStartedAt === null) return; // stopped while the check was in flight
  const h = get(health);
  const delay = !h.ever_ok && !h.startup_grace_over ? STARTUP_POLL_MS : POLL_MS;
  _pollTimer = setTimeout(_pollTick, delay);
}

export function startHealthPolling(): void {
  if (_pollingStartedAt !== null) return;
  _pollingStartedAt = Date.now();
  _pollTick();
}

export function stopHealthPolling(): void {
  if (_pollTimer) clearTimeout(_pollTimer);
  _pollTimer = null;
  _pollingStartedAt = null;
}
