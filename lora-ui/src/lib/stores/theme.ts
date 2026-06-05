import { writable } from 'svelte/store';
import { browser } from '$app/environment';

export type Theme = 'dark' | 'light';

function createThemeStore() {
  const stored = browser ? (localStorage.getItem('lora-theme') as Theme | null) : null;
  const initial: Theme = stored ?? 'dark';

  const { subscribe, set, update } = writable<Theme>(initial);

  return {
    subscribe,
    set: (theme: Theme) => {
      if (browser) {
        localStorage.setItem('lora-theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
      }
      set(theme);
    },
    toggle: () => {
      update((current) => {
        const next: Theme = current === 'dark' ? 'light' : 'dark';
        if (browser) {
          localStorage.setItem('lora-theme', next);
          document.documentElement.setAttribute('data-theme', next);
        }
        return next;
      });
    }
  };
}

export const theme = createThemeStore();
