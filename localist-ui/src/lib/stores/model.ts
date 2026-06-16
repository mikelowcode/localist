import { writable, derived } from 'svelte/store';
import { health } from './server';

export interface ModelState {
  chat_model: string;
  embedding_model: string;
  backend: string;
}

export const modelConfig = writable<ModelState>({
  chat_model: '',
  embedding_model: '',
  backend: 'foundry'
});

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
