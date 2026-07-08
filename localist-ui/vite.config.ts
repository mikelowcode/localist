import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '127.0.0.1',
		port: 5173,
		strictPort: true,
		proxy: {
			'/api': {
				target: 'http://127.0.0.1:8001',
				rewrite: (path) => path.replace(/^\/api/, '')
			}
		},
		// Pre-transform the chat route on dev-server startup instead of on first
		// navigation. ChatPanel.svelte is the largest/slowest-to-compile component;
		// leaving it to on-demand compilation meant the very first load after every
		// restart could paint before its scoped CSS (overflow/flex constraints on
		// .messages) was ready, letting the message list grow past the viewport.
		warmup: {
			clientFiles: [
				'./src/routes/conversation/[id]/+page.svelte',
				'./src/lib/components/ChatPanel.svelte',
				'./src/lib/components/MarkdownRenderer.svelte'
			]
		}
	}
});
