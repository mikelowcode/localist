/**
 * episodes.ts — Localist episodic memory store
 *
 * Fetches from GET /api/memory/episodes and exposes paginated,
 * filterable episode state to EpisodesPanel.
 */

import { writable, derived } from 'svelte/store';

export interface EpisodeItem {
  id:              number;
  episode_type:    string;
  subject:         string;
  content:         string;
  confidence:      number;
  source:          string;
  task_id:         string | null;
  project_context: string | null;
  status:          string;
  created_at:      number;
  last_accessed:   number | null;
}

export interface EpisodesState {
  episodes:    EpisodeItem[];
  loading:     boolean;
  error:       string | null;
  offset:      number;
  limit:       number;
  total:       number;
  typeFilter:  string;   // "" = all types
}

const _initial: EpisodesState = {
  episodes:   [],
  loading:    false,
  error:      null,
  offset:     0,
  limit:      50,
  total:      0,
  typeFilter: '',
};

export const episodesStore = writable<EpisodesState>(_initial);

export const EPISODE_TYPES = [
  'preference',
  'correction',
  'decision',
  'workflow',
  'fact',
  'relationship',
  'context',
] as const;

// Human-readable labels for episode types
export const TYPE_LABELS: Record<string, string> = {
  preference:   'Preference',
  correction:   'Correction',
  decision:     'Decision',
  workflow:     'Workflow',
  fact:         'Fact',
  relationship: 'Relationship',
  context:      'Context',
};

// Colour tokens per episode type (matches prov-chip palette in ChatPanel)
export const TYPE_COLORS: Record<string, { bg: string; color: string; border: string }> = {
  preference:   { bg: '#1a2a1a', color: '#7ecf7e', border: '#2d4a2d' },
  correction:   { bg: '#2a1a1a', color: '#cf7e7e', border: '#4a2d2d' },
  decision:     { bg: '#1a1a2e', color: '#7ea8cf', border: '#2d3a5a' },
  workflow:     { bg: '#2a2218', color: '#cfb07e', border: '#5a4a2d' },
  fact:         { bg: '#2a1a2e', color: '#b07ecf', border: '#4a2d5a' },
  relationship: { bg: '#1e2a2a', color: '#7ecfcf', border: '#2d4a4a' },
  context:      { bg: '#1e1e1e', color: '#9a9a9a', border: '#2a2a2a' },
};

export async function loadEpisodes(opts: {
  typeFilter?: string;
  offset?: number;
  limit?: number;
} = {}): Promise<void> {
  episodesStore.update((s) => ({ ...s, loading: true, error: null }));

  const params = new URLSearchParams();
  params.set('status', 'active');
  params.set('limit',  String(opts.limit  ?? 50));
  params.set('offset', String(opts.offset ?? 0));
  if (opts.typeFilter) params.set('episode_type', opts.typeFilter);

  try {
    const res = await fetch(`/api/memory/episodes?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: { episodes: EpisodeItem[]; total: number; offset: number; limit: number }
      = await res.json();

    episodesStore.update((s) => ({
      ...s,
      loading:  false,
      episodes: data.episodes,
      total:    data.total,
      offset:   data.offset,
      limit:    data.limit,
    }));
  } catch (err) {
    episodesStore.update((s) => ({
      ...s,
      loading: false,
      error:   err instanceof Error ? err.message : String(err),
    }));
  }
}

export function resetEpisodes(): void {
  episodesStore.set(_initial);
}
