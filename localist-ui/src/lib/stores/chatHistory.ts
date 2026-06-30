import { writable } from 'svelte/store';
import type { Task } from '$lib/stores/tasks';

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
  task_id?: string;
  timestamp: number;
  status?: Task['status'];
  sources?: Task['sources'];
  status_message?: string;
  metadata?: Task['metadata'];
}

export const chatHistoryStore = writable<Turn[]>([]);
