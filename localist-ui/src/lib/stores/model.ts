import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';
import { health } from './server';

export interface ModelState {
  backend: string;
}

const BACKEND_KEY = 'localist-runtime-backend';

export const RUNTIME_BACKENDS = ['omlx', 'ollama'] as const;
export type RuntimeBackend = (typeof RUNTIME_BACKENDS)[number];

export const RUNTIME_BACKEND_LABELS: Record<RuntimeBackend, string> = {
  omlx: 'oMLX',
  ollama: 'Ollama'
};

function readStoredBackend(): string {
  if (!browser) return 'omlx';
  const stored = localStorage.getItem(BACKEND_KEY);
  return stored && (RUNTIME_BACKENDS as readonly string[]).includes(stored) ? stored : 'omlx';
}

export const modelConfig = writable<ModelState>({
  backend: readStoredBackend()
});

modelConfig.subscribe((s) => {
  if (browser) localStorage.setItem(BACKEND_KEY, s.backend);
});

// localStorage is only a paint-before-first-health-check cache. Once a real
// health response has arrived, the actual active backend (which can now
// change at runtime via POST /settings/runtime-backend — see
// docs/architecture/16-runtime-backend-layer.md §16.5) is authoritative and
// overrides whatever the browser last remembered — a process restart reverts
// to .env, or someone may have switched it directly via curl.
health.subscribe(($health) => {
  if ($health.reachable && $health.backend) {
    modelConfig.update((s) => (s.backend === $health.backend ? s : { ...s, backend: $health.backend }));
  }
});

// Display label for the appbar's consolidated status chip and the Settings
// segmented control.
export const runtimeBackendLabel = derived(modelConfig, ($m) =>
  RUNTIME_BACKEND_LABELS[$m.backend as RuntimeBackend] ?? $m.backend
);

