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
  source?: 'chat' | 'ingest';
  metadata?: {
    priority?:         number;
    fetch_rag?:        boolean;
    fetch_episodic?:   boolean;
    tools_fired?:      string[];
    agent?:            string;
    grounded?:         boolean;
    file_op_deferred?: boolean;
  };
}

export interface TasksState {
  active_task_id: string | null;
  tasks: Record<string, Task>;
  // True from submit until the visible answer text finishes streaming
  // ('done' SSE event). Cleared before background memory writes
  // (episodic/working-state extraction) are done — do not gate the next
  // submission on this alone; see `finalizing`.
  streaming: boolean;
  // True from submit until the backend's 'task_complete' SSE event, which
  // fires only after the full pipeline (including post-answer
  // episodic/working-state hooks) has finished. This can trail `streaming`
  // going false by 10-30+ seconds. Gate the next submission on this, not
  // on `streaming`, to avoid overlapping calls into the single-instance
  // local model backend.
  finalizing: boolean;
}

export const tasksStore = writable<TasksState>({
  active_task_id: null,
  tasks: {},
  streaming: false,
  finalizing: false
});

const BASE = '/api';

// Stable for the lifetime of this page load. Groups conversation_log
// entries server-side so working memory persists across turns within
// a session. Regenerated on full page reload (by design) rather than
// persisted, so a fresh page load starts a fresh conversation from the
// backend's perspective.
const SESSION_ID = crypto.randomUUID();

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
// Resolves as soon as the 'done' SSE event is processed (store already
// patched), so the caller's submitting flag clears immediately. The reader
// loop keeps draining in the background until the [DONE] sentinel closes
// the stream, ensuring no bytes are abandoned mid-connection.
//
// `streaming` and `finalizing` both start true and are cleared at
// different points: `streaming` clears on 'done' (visible answer text is
// ready), `finalizing` clears on 'task_complete' (the full backend
// pipeline, including background episodic/working-state writes, has
// actually finished). The next submission must be gated on `finalizing`,
// not `streaming` — submitting while the prior turn's background writes
// are still running causes overlapping calls into the single-instance
// local model backend.
export function submitTask(
  instruction: string,
  context: Record<string, unknown> = {},
  task_id?: string,
  conversation_id?: string,
  conversation_title?: string
): Promise<string> {
  const id = task_id ?? crypto.randomUUID();
  const task = createTask(id, instruction);

  tasksStore.update((s) => ({
    ...s,
    active_task_id: id,
    streaming: true,
    finalizing: true,
    tasks: { ...s.tasks, [id]: task }
  }));

  const body = JSON.stringify({
    task_id: id,
    instruction,
    context: { session_id: SESSION_ID, ...context },
    conversation_id,
    conversation_title,
  });

  return new Promise<string>((resolve) => {
    (async () => {
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

          // A single network read can coalesce several SSE events (e.g. a
          // burst of 'token' events) into one chunk. Processing all of them
          // synchronously back-to-back with no yield point forces every
          // downstream store update/re-render into one uninterrupted JS
          // task. Yield once per processed line — but only when this chunk
          // actually has more than one 'data:' line — so the common case
          // (one event per read) pays no extra overhead.
          const multipleDataLines =
            lines.filter((l) => l.startsWith('data: ')).length > 1;

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const raw = line.slice(6).trim();
            if (raw === '[DONE]') {
              patchTask(id, { status: 'complete', completed_at: Date.now() });
              // task_complete should already have cleared `finalizing` — this
              // is a fail-safe in case the stream closes without it, so
              // input never stays disabled indefinitely.
              tasksStore.update((s) => ({ ...s, streaming: false, finalizing: false }));
              resolve(id);
              return;
            }

            let event: Record<string, unknown>;
            try {
              event = JSON.parse(raw);
            } catch {
              continue;
            }

            handleSSEEvent(id, event);

            // Resolve immediately after the store is patched by the 'done'
            // event — loop continues draining until [DONE] arrives.
            // resolve() is idempotent: subsequent calls after the first are
            // no-ops per the Promise spec.
            if ((event.type as string) === 'done') {
              resolve(id);
            }

            if (multipleDataLines) {
              await Promise.resolve();
            }
          }
        }
        // Stream closed without [DONE] (e.g. connection dropped) — fail-safe
        // reset so input doesn't stay disabled with nothing left to unblock it.
        tasksStore.update((s) => ({ ...s, streaming: false, finalizing: false }));
        resolve(id);
      } catch (err) {
        patchTask(id, {
          status: 'failed',
          error: String(err),
          completed_at: Date.now()
        });
        tasksStore.update((s) => ({ ...s, streaming: false, finalizing: false }));
        resolve(id); // always resolve with id, never reject — matches prior behaviour
      }
    })();
  });
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
      tasksStore.update((s) => {
        const t = s.tasks[task_id];
        if (!t) return s;
        const correctedAnswer = event.answer as string | undefined;
        const answerPatch =
          correctedAnswer && correctedAnswer !== t.answer
            ? { answer: correctedAnswer, tokens: [] as string[] }
            : {};
        return {
          ...s,
          streaming: false,
          tasks: {
            ...s.tasks,
            [task_id]: {
              ...t,
              ...answerPatch,
              status:       (event.status as Task['status']) ?? 'complete',
              metadata:     (event.metadata as Task['metadata']) ?? {},
              completed_at: Date.now(),
            },
          },
        };
      });
      break;
    }
    case 'error': {
      patchTask(task_id, {
        status: 'failed',
        error: (event.message as string) ?? 'Unknown error',
        completed_at: Date.now()
      });
      // An error means nothing further is coming for this task — no need
      // to wait for 'task_complete' (which still follows, but redundantly).
      tasksStore.update((s) => ({ ...s, streaming: false, finalizing: false }));
      break;
    }
    case 'task_complete': {
      // Fires only after the full backend pipeline — including background
      // episodic/working-state writes — has finished. This is the real
      // "safe to submit again" signal; 'done' alone is not (see TasksState
      // doc comment on `finalizing`).
      tasksStore.update((s) => ({ ...s, finalizing: false }));
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
