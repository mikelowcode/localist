import { writable } from 'svelte/store';
import { browser } from '$app/environment';

export type FontSize = 'sm' | 'md' | 'lg' | 'xl';

export const FONT_SIZES: { value: FontSize; label: string }[] = [
  { value: 'sm', label: 'Small' },
  { value: 'md', label: 'Medium' },
  { value: 'lg', label: 'Large' },
  { value: 'xl', label: 'X-Large' }
];

function createFontSizeStore() {
  const stored = browser ? (localStorage.getItem('lora-font-size') as FontSize | null) : null;
  const initial: FontSize = stored ?? 'md';

  const { subscribe, set } = writable<FontSize>(initial);

  return {
    subscribe,
    set: (size: FontSize) => {
      if (browser) {
        localStorage.setItem('lora-font-size', size);
        document.documentElement.setAttribute('data-font-size', size);
      }
      set(size);
    }
  };
}

export const fontSize = createFontSizeStore();
