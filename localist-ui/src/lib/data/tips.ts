export type TipCategory = 'keyword' | 'slash-command' | 'ui-action';

export interface Tip {
	id: string;
	category: TipCategory;
	trigger: string;
	example: string;
	description: string;
}

export const tips: Tip[] = [
	// Slash commands
	{
		id: 'slash-chart',
		category: 'slash-command',
		trigger: '/chart',
		example: '/chart my monthly expenses',
		description: 'Forces the chart tool directly, skipping keyword matching.'
	},
	{
		id: 'slash-research',
		category: 'slash-command',
		trigger: '/research',
		example: '/research the history of the transistor',
		description: 'Forces the research tool directly, bypassing the usual gating.'
	},

	// Ingest
	{
		id: 'ingest',
		category: 'keyword',
		trigger: 'ingest / process this file / add to wiki / index this',
		example: 'ingest this file into the wiki',
		description: 'Starts wiki ingestion of a raw document.'
	},

	// Diff-only wiki update
	{
		id: 'diff',
		category: 'keyword',
		trigger: 'update the wiki / propose a diff',
		example: 'propose a diff to update the wiki page',
		description: 'Applies a targeted diff to an existing wiki page instead of full re-ingestion.'
	},

	// Memory
	{
		id: 'memory-remember',
		category: 'keyword',
		trigger: 'remember that / my preference is',
		example: 'remember that I prefer dark mode',
		description: 'Explicitly saves a preference or fact to episodic memory.'
	},
	{
		id: 'memory-correction',
		category: 'keyword',
		trigger: "that's wrong / the correct value is",
		example: "that's wrong, the correct value is 42",
		description: 'Corrects a previously remembered fact.'
	},
	{
		id: 'memory-forget',
		category: 'keyword',
		trigger: "forget that / that's no longer true",
		example: "forget that, it's no longer true",
		description: 'Removes or supersedes a previously saved memory.'
	},

	// Graph query
	{
		id: 'graph-backlinks',
		category: 'keyword',
		trigger: 'what links to / backlinks for',
		example: 'what pages link to Project Localist',
		description: 'Looks up backlinks in the wiki concept graph.'
	},
	{
		id: 'graph-outgoing',
		category: 'keyword',
		trigger: 'outgoing links for / what links from',
		example: 'show outgoing links for the Router page',
		description: 'Looks up outgoing links in the wiki concept graph.'
	},

	// Web search / freshness
	{
		id: 'web-search',
		category: 'keyword',
		trigger: 'latest / current price / recent',
		example: "what's the current price of gold",
		description: 'Triggers a live web search for up-to-date information.'
	},

	// Fetch URL
	{
		id: 'fetch-url',
		category: 'keyword',
		trigger: 'fetch this url / summarize this link',
		example: 'summarize this link: https://example.com/article',
		description: 'Fetches and extracts readable content from a URL.'
	},

	// File ops
	{
		id: 'file-op',
		category: 'keyword',
		trigger: 'read the file / create a file',
		example: 'read the file notes.md',
		description: 'Reads, writes, or creates a file in the sandboxed project root.'
	},

	// Chart keywords
	{
		id: 'chart-keyword',
		category: 'keyword',
		trigger: 'make a chart / plot this / visualize this',
		example: 'turn this into a bar chart',
		description: 'Generates a chart from data in the conversation.'
	},

	// News
	{
		id: 'news',
		category: 'keyword',
		trigger: 'news / headlines / breaking',
		example: "what's the breaking news on AI regulation",
		description: 'Fetches current news headlines on a topic.'
	},

	// GitHub release
	{
		id: 'github-release',
		category: 'keyword',
		trigger: 'release notes / changelog / latest release',
		example: 'what are the release notes for Svelte 5',
		description: 'Fetches release notes or a changelog for a GitHub project.'
	},

	// GitHub
	{
		id: 'github',
		category: 'keyword',
		trigger: 'github / repo / pull request',
		example: 'find the readme for the sveltejs/kit repo',
		description: 'Looks up GitHub repository info, issues, or pull requests.'
	},

	// Hacker News
	{
		id: 'hacker-news',
		category: 'keyword',
		trigger: 'hacker news / hn / y combinator',
		example: "what's trending on hacker news today",
		description: 'Fetches top stories from Hacker News.'
	},

	// Factual query
	{
		id: 'factual',
		category: 'keyword',
		trigger: 'when did / who invented / how many',
		example: 'who invented the transistor',
		description: 'Answers a direct factual question, falling back to web search on a corpus miss.'
	},

	// Corpus / wiki query
	{
		id: 'corpus-wiki',
		category: 'keyword',
		trigger: 'check the wiki / search the wiki / vault',
		example: 'check the wiki for my notes on onboarding',
		description: 'Searches your indexed wiki documents (RAG corpus).'
	},

	// Episodic memory recall
	{
		id: 'episodic-recall',
		category: 'keyword',
		trigger: 'what do you know about me / my preference',
		example: 'what do you know about me',
		description: 'Recalls previously saved preferences, decisions, or corrections.'
	},

	// Chat toolbar actions
	{
		id: 'attach-file',
		category: 'ui-action',
		trigger: 'Attach a file (+ button)',
		example: 'Click the + button next to the chat input',
		description: 'Uploads a text/code file or image/PDF into the conversation as a pill; pills with a file path also offer "Ingest into wiki".'
	},
	{
		id: 'pin-wiki-page',
		category: 'ui-action',
		trigger: 'Pin a wiki page (pin button)',
		example: 'Click the pin button and pick a page from your wiki',
		description: 'Pins one or more wiki pages to the conversation so edits can be proposed as diffs to that page.'
	},
	{
		id: 'edit-save-response',
		category: 'ui-action',
		trigger: 'Edit and save as file (per-response button)',
		example: 'Click "Edit and save as file" under any response',
		description: 'Opens that response in an inline editor where you can tweak the text, name it, pick .md or .txt, and save it as a generated file.'
	},
	{
		id: 'generated-files-location',
		category: 'ui-action',
		trigger: 'Files tab → Generated',
		example: 'Open the Files tab and look under Generated',
		description: 'Every file saved via "Edit and save as file" (or from the compose-document panel) lands in the Files tab\'s Generated section.'
	},
	{
		id: 'compose-document',
		category: 'ui-action',
		trigger: 'Compose a document across turns (document button)',
		example: 'Click the document button, then "Add to document" under any reply',
		description: 'Opens a side panel that collects assistant replies into one draft you can hand-edit and save as a file, across multiple turns.'
	},

	// Memory tab
	{
		id: 'memory-nav',
		category: 'ui-action',
		trigger: 'Memory tab',
		example: 'Open the Memory tab and filter by type or status',
		description: 'Browse everything remembered as chip-filterable cards — by type (Preference, Correction, Decision, Workflow, Fact, Relationship, Context) or status (Pending, Retracted, Superseded, Everything).'
	},
	{
		id: 'memory-pending-approval',
		category: 'ui-action',
		trigger: 'Pending memories → Approve / Reject',
		example: 'Open Memory → Pending, then Approve or Reject a card',
		description: 'Memories awaiting review sit under the Pending filter until you click Approve or Reject on their card; a badge on the Memory tab shows how many are waiting. (Settings has an "Episodic write-approval" toggle for this — currently a saved preference, not yet backend-enforced.)'
	},

	// Settings
	{
		id: 'settings-theme',
		category: 'ui-action',
		trigger: 'Settings → Theme',
		example: 'Open Settings and toggle Theme',
		description: 'Switches light/dark mode — the same switch also lives in the sidebar footer.'
	},
	{
		id: 'settings-font-size',
		category: 'ui-action',
		trigger: 'Settings → Font Size',
		example: 'Open Settings and pick a Font Size',
		description: 'Scales text throughout the app immediately; some fixed-size UI chrome won\'t change.'
	},
	{
		id: 'settings-data-retention',
		category: 'ui-action',
		trigger: 'Settings → Data Retention',
		example: 'Open Settings and set Chat History / Episodic Memory retention',
		description: 'Choose how long chat turns and learned memories are kept (7/30/90 days or Forever) before chat history is deleted or memories are retracted (reversible from the Memory tab).'
	},
	{
		id: 'settings-github-pins',
		category: 'ui-action',
		trigger: 'Settings → Pinned GitHub Repos',
		example: 'Open Settings and add a repo as owner/repo',
		description: 'Pins up to 20 repos to show in the Live Feed panel; this does not subscribe you to any GitHub notifications.'
	},
	{
		id: 'settings-news-brief',
		category: 'ui-action',
		trigger: 'Settings → Daily News Brief',
		example: 'Open Settings, set your country/area, and pick 3 topics',
		description: 'Configures your Top Stories (home-country) and Local (keyword-matched) news, built from exactly 3 chosen topics.'
	},
	{
		id: 'settings-api-keys',
		category: 'ui-action',
		trigger: 'Settings → API Keys',
		example: 'Open Settings and paste your LangSearch/Brave/NewsAPI key',
		description: 'Only LANGSEARCH_API_KEY is required for web_search; Brave adds a search fallback and NewsAPI powers the Daily News Brief. Keys save to backend/.env and are never echoed back.'
	},
	{
		id: 'settings-assistant-name',
		category: 'ui-action',
		trigger: 'Settings → Assistant Name',
		example: 'Open Settings, type a new name, and click Save',
		description: 'Renames the assistant\'s persona in the system prompt — takes effect on your very next message.'
	},
	{
		id: 'settings-runtime-backend',
		category: 'ui-action',
		trigger: 'Settings → Runtime Backend',
		example: 'Open Settings and pick a different Runtime Backend',
		description: 'Live-switches the inference engine (e.g. oMLX ↔ Ollama) without a restart, after a confirm dialog; the choice is health-checked and persisted to .env.'
	},
	{
		id: 'settings-embedding-model',
		category: 'ui-action',
		trigger: 'Settings → Embedding Model',
		example: 'Open Settings and choose an Embedding Model (Ollama only)',
		description: 'Live-switches which model embeds your corpus, or falls back to keyword-only search if set to None; applies immediately and flags your corpus for re-embedding.'
	}
];
