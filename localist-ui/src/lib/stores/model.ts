import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';
import { health } from './server';

export interface ModelState {
  chat_model: string;
  embedding_model: string;
  backend: string;
}

const BACKEND_KEY = 'lora-runtime-backend';

export const RUNTIME_BACKENDS = ['omlx', 'ollama', 'foundry'] as const;
export type RuntimeBackend = (typeof RUNTIME_BACKENDS)[number];

export const RUNTIME_BACKEND_LABELS: Record<RuntimeBackend, string> = {
  omlx: 'oMLX',
  ollama: 'Ollama',
  foundry: 'Foundry'
};

function readStoredBackend(): string {
  if (!browser) return 'foundry';
  const stored = localStorage.getItem(BACKEND_KEY);
  return stored && (RUNTIME_BACKENDS as readonly string[]).includes(stored) ? stored : 'foundry';
}

export const modelConfig = writable<ModelState>({
  chat_model: '',
  embedding_model: '',
  backend: readStoredBackend()
});

modelConfig.subscribe((s) => {
  if (browser) localStorage.setItem(BACKEND_KEY, s.backend);
});

// Display label for the appbar's consolidated status chip and the Settings
// segmented control — cosmetic only; the real backend is selected at process
// startup via LOCALIST_RUNTIME_BACKEND (see CLAUDE.md), not switched at runtime.
export const runtimeBackendLabel = derived(modelConfig, ($m) =>
  RUNTIME_BACKEND_LABELS[$m.backend as RuntimeBackend] ?? $m.backend
);

// Derive display name from health store (first model listed, or chat model from config)
export const activeModelName = derived(health, ($h) => {
  if ($h.models.length > 0) return $h.models[0];
  return '—';
});

// Short label for the status bar — truncate long model IDs
export const modelLabel = derived(activeModelName, ($n) => {
  if (!$n || $n === '—') return '—';
  // Strip version suffix like ":5" for display
  const clean = $n.replace(/:\d+$/, '');
  // Truncate to 24 chars
  return clean.length > 24 ? clean.slice(0, 22) + '…' : clean;
});
