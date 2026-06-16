import { writable, get } from 'svelte/store';

export type TaskStatus = 'idle' | 'planning' | 'streaming' | 'complete' | 'failed';

export interface Source {
  name: string;
  path: string;
  type: 'wiki' | 'raw';
  relevance_score: number;
}

export interface Task {
  task_id: string;
  instruction: string;
  status: TaskStatus;
  status_message: string;
  tokens: string[];        // raw streamed chunks
  answer: string;          // accumulated final answer
  sources: Source[];
  error?: string;
  started_at: number;
  completed_at?: number;
}

export interface TasksState {
  active_task_id: string | null;
  tasks: Record<string, Task>;
  streaming: boolean;
}

export const tasksStore = writable<TasksState>({
  active_task_id: null,
  tasks: {},
  streaming: false
});

const BASE = '/api';

// ── Create a new task entry ──────────────────────────────────
function createTask(task_id: string, instruction: string): Task {
  return {
    task_id,
    instruction,
    status: 'planning',
    status_message: 'Planning…',
    tokens: [],
    answer: '',
    sources: [],
    started_at: Date.now()
  };
}

// ── Patch a task in the store ────────────────────────────────
function patchTask(task_id: string, patch: Partial<Task>): void {
  tasksStore.update((s) => {
    const existing = s.tasks[task_id];
    if (!existing) return s;
    return {
      ...s,
      tasks: {
        ...s.tasks,
        [task_id]: { ...existing, ...patch }
      }
    };
  });
}

// ── Submit a task with SSE streaming ─────────────────────────
export async function submitTask(
  instruction: string,
  context: Record<string, unknown> = {}
): Promise<string> {
  const task_id = crypto.randomUUID();
  const task = createTask(task_id, instruction);

  tasksStore.update((s) => ({
    ...s,
    active_task_id: task_id,
    streaming: true,
    tasks: { ...s.tasks, [task_id]: task }
  }));

  const body = JSON.stringify({ task_id, instruction, context });

  try {
    const res = await fetch(`${BASE}/task/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });

    if (!res.ok) {
      throw new Error(`Server error: HTTP ${res.status}`);
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') {
          patchTask(task_id, { status: 'complete', completed_at: Date.now() });
          tasksStore.update((s) => ({ ...s, streaming: false }));
          return task_id;
        }

        let event: Record<string, unknown>;
        try {
          event = JSON.parse(raw);
        } catch {
          continue;
        }

        handleSSEEvent(task_id, event);
      }
    }
  } catch (err) {
    patchTask(task_id, {
      status: 'failed',
      error: String(err),
      completed_at: Date.now()
    });
    tasksStore.update((s) => ({ ...s, streaming: false }));
  }

  return task_id;
}

function handleSSEEvent(task_id: string, event: Record<string, unknown>): void {
  const type = event.type as string;

  switch (type) {
    case 'status': {
      patchTask(task_id, {
        status: 'planning',
        status_message: (event.message as string) ?? 'Working…'
      });
      break;
    }
    case 'token': {
      tasksStore.update((s) => {
        const t = s.tasks[task_id];
        if (!t) return s;
        const newTokens = [...t.tokens, event.token as string];
        return {
          ...s,
          tasks: {
            ...s.tasks,
            [task_id]: {
              ...t,
              status: 'streaming',
              status_message: 'Streaming answer…',
              tokens: newTokens,
              answer: newTokens.join('')
            }
          }
        };
      });
      break;
    }
    case 'sources': {
      patchTask(task_id, { sources: (event.sources as Source[]) ?? [] });
      break;
    }
    case 'done': {
      patchTask(task_id, {
        status: 'complete',
        completed_at: Date.now()
      });
      tasksStore.update((s) => ({ ...s, streaming: false }));
      break;
    }
    case 'error': {
      patchTask(task_id, {
        status: 'failed',
        error: (event.message as string) ?? 'Unknown error',
        completed_at: Date.now()
      });
      tasksStore.update((s) => ({ ...s, streaming: false }));
      break;
    }
  }
}

// ── Get a task by ID ─────────────────────────────────────────
export function getTask(task_id: string): Task | undefined {
  return get(tasksStore).tasks[task_id];
}

// ── Get the currently active task ───────────────────────────
export function getActiveTask(): Task | null {
  const s = get(tasksStore);
  if (!s.active_task_id) return null;
  return s.tasks[s.active_task_id] ?? null;
}
