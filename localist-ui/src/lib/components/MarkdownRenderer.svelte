<script lang="ts">
  /**
   * MarkdownRenderer.svelte
   *
   * Lightweight Markdown renderer for Localist agent output.
   * Handles the exact subset the model produces. Kept dependency-free
   * except for KaTeX, which renders the LaTeX math notation models
   * routinely emit (e.g. "$\rightarrow$") — that's source syntax for a
   * symbol, not literal text, so it's rendered rather than hand-rolled.
   *
   * Supported syntax:
   *   ## / ### headings
   *   **bold**  *italic*  `inline code`
   *   - / * unordered lists  (nested one level via leading spaces)
   *   1. ordered lists
   *   > blockquotes
   *   ```lang … ``` fenced code blocks
   *   --- horizontal rules
   *   $$…$$ / $…$ LaTeX math (KaTeX; single-$ only when it contains a
   *     backslash command, so plain currency like "$5" is never touched)
   *   Blank lines → paragraph breaks
   *
   * Security: raw text is HTML-escaped before any substitution runs,
   * so user-supplied or model-generated content cannot inject tags.
   * Math spans are escaped out to KaTeX (throwOnError: false, trust:
   * false) before that pass, so malformed LaTeX degrades to an inline
   * error span instead of breaking rendering, and KaTeX's own HTML
   * output is never re-escaped.
   *
   * Props:
   *   content  — the Markdown string to render
   *   streaming — when true, appends a blinking cursor after the last node
   */

  import { onDestroy } from 'svelte';
  import katex from 'katex';
  import 'katex/dist/katex.min.css';

  export let content:   string  = '';
  export let streaming: boolean = false;

  // ---------------------------------------------------------------------------
  // Escape
  // ---------------------------------------------------------------------------

  function escape(text: string): string {
    return text
      .replace(/&/g,  '&amp;')
      .replace(/</g,  '&lt;')
      .replace(/>/g,  '&gt;')
      .replace(/"/g,  '&quot;')
      .replace(/'/g,  '&#39;');
  }

  // ---------------------------------------------------------------------------
  // Math (KaTeX)
  // ---------------------------------------------------------------------------

  // expr is text that has already passed through escape() (see call sites
  // below), so entities like &lt; survive into the TeX source verbatim
  // rather than as literal `<` — an accepted limitation, since the LaTeX
  // models actually emit here (arrows, quote commands, Greek letters) never
  // contains those characters.
  function renderMath(expr: string, displayMode: boolean): string {
    try {
      return katex.renderToString(expr, { throwOnError: false, trust: false, displayMode });
    } catch {
      // KaTeX couldn't degrade gracefully even with throwOnError: false —
      // fall back to the literal source so rendering never breaks.
      const delim = displayMode ? '$$' : '$';
      return `${delim}${expr}${delim}`;
    }
  }

  // ---------------------------------------------------------------------------
  // Inline formatting  (runs AFTER escaping)
  // ---------------------------------------------------------------------------

  function inlineFormat(text: string): string {
    // Math is extracted to placeholder tokens before the bold/italic/code
    // regexes run, then swapped back in at the end — otherwise KaTeX's own
    // HTML output (full of literal < > " ') would get mangled by the later
    // substitutions or re-escaped downstream. A Private Use Area sentinel
    // is used because it can't appear in normal model output and none of
    // the other regexes below can match it.
    const mathStash: string[] = [];
    const stash = (html: string): string => {
      const token = `${mathStash.length}`;
      mathStash.push(html);
      return token;
    };

    let out = text
      // $$display math$$ — unambiguous, currency is never doubled like this.
      .replace(/\$\$([^\n]+?)\$\$/g, (_, expr) => stash(renderMath(expr, true)))
      // $inline math$ — collides with plain currency ("$5", "$10 total"),
      // so only treat it as math when it contains a backslash command;
      // plain currency never does, and every LaTeX symbol a model emits
      // this way (\rightarrow, \'\, \alpha, …) does.
      .replace(/\$([^\n$]+?)\$/g, (match, expr) =>
        expr.includes('\\') ? stash(renderMath(expr, false)) : match
      )
      // **bold** / __bold__
      .replace(/\*\*(.+?)\*\*/g,   '<strong>$1</strong>')
      .replace(/__(.+?)__/g,        '<strong>$1</strong>')
      // *italic* / _italic_   (not preceded/followed by another * or _)
      .replace(/\*([^*\n]+?)\*/g,   '<em>$1</em>')
      .replace(/_([^_\n]+?)_/g,     '<em>$1</em>')
      // `inline code`
      .replace(/`([^`\n]+?)`/g,     '<code>$1</code>')
      // [text](url)  →  plain text only (no external links in agent output)
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

    if (mathStash.length > 0) {
      out = out.replace(/(\d+)/g, (_, i) => mathStash[Number(i)]);
    }

    return out;
  }

  // ---------------------------------------------------------------------------
  // Block-level parser
  // ---------------------------------------------------------------------------

  type Block =
    | { type: 'h1';      text: string }
    | { type: 'h2';      text: string }
    | { type: 'h3';      text: string }
    | { type: 'h4';      text: string }
    | { type: 'hr' }
    | { type: 'blockquote'; lines: string[] }
    | { type: 'code';    lang: string; lines: string[] }
    | { type: 'ul';      items: ListItem[] }
    | { type: 'ol';      items: ListItem[] }
    | { type: 'p';       lines: string[] };

  interface ListItem {
    indent: number;
    text:   string;
    ordered?: boolean;
  }

  function parse(raw: string): Block[] {
    const lines = raw.split('\n');
    const blocks: Block[] = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trimStart();

      // ── Fenced code block ───────────────────────────────────────────
      if (trimmed.startsWith('```')) {
        const lang = trimmed.slice(3).trim();
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // consume closing ```
        blocks.push({ type: 'code', lang, lines: codeLines });
        continue;
      }

      // ── Headings ────────────────────────────────────────────────────
      if (trimmed.startsWith('#### ')) {
        blocks.push({ type: 'h4', text: trimmed.slice(5).trim() });
        i++; continue;
      }
      if (trimmed.startsWith('### ')) {
        blocks.push({ type: 'h3', text: trimmed.slice(4).trim() });
        i++; continue;
      }
      if (trimmed.startsWith('## ')) {
        blocks.push({ type: 'h2', text: trimmed.slice(3).trim() });
        i++; continue;
      }
      if (trimmed.startsWith('# ')) {
        blocks.push({ type: 'h1', text: trimmed.slice(2).trim() });
        i++; continue;
      }

      // ── Horizontal rule ─────────────────────────────────────────────
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        blocks.push({ type: 'hr' });
        i++; continue;
      }

      // ── Blockquote ──────────────────────────────────────────────────
      if (trimmed.startsWith('> ')) {
        const bqLines: string[] = [];
        while (i < lines.length && lines[i].trimStart().startsWith('> ')) {
          bqLines.push(lines[i].trimStart().slice(2));
          i++;
        }
        blocks.push({ type: 'blockquote', lines: bqLines });
        continue;
      }

      // ── Unordered list ──────────────────────────────────────────────
      if (/^(\s*)([-*+])\s/.test(line)) {
        const items: ListItem[] = [];
        while (i < lines.length && /^(\s*)([-*+])\s/.test(lines[i])) {
          const m = lines[i].match(/^(\s*)([-*+])\s(.*)$/)!;
          items.push({ indent: m[1].length, text: m[3], ordered: false });
          i++;
        }
        blocks.push({ type: 'ul', items });
        continue;
      }

      // ── Ordered list ────────────────────────────────────────────────
      if (/^(\s*)\d+\.\s/.test(line)) {
        const items: ListItem[] = [];
        while (i < lines.length && /^(\s*)\d+\.\s/.test(lines[i])) {
          const m = lines[i].match(/^(\s*)(\d+)\.\s(.*)$/)!;
          items.push({ indent: m[1].length, text: m[3], ordered: true });
          i++;
        }
        blocks.push({ type: 'ol', items });
        continue;
      }

      // ── Blank line — skip ───────────────────────────────────────────
      if (trimmed === '') {
        i++; continue;
      }

      // ── Paragraph — accumulate until blank line or block marker ─────
      const pLines: string[] = [];
      while (
        i < lines.length &&
        lines[i].trim() !== '' &&
        !/^(#{1,4} |```|> |[-*+] |\d+\. |---|===)/.test(lines[i].trimStart())
      ) {
        pLines.push(lines[i].trim());
        i++;
      }
      if (pLines.length > 0) {
        blocks.push({ type: 'p', lines: pLines });
      } else {
        // The guard regex above matches some prefixes with no
        // corresponding block handler (e.g. "===", or "---text" that
        // isn't a clean horizontal rule) — without this, `i` never
        // advances for such a line and the outer while loop spins
        // forever, freezing the tab (see the h1 case fixed above, which
        // hit this exact trap before h1 got its own branch). Treat any
        // line that reaches here unconsumed as a one-line paragraph so
        // the parser always makes forward progress.
        pLines.push(lines[i].trim());
        i++;
        blocks.push({ type: 'p', lines: pLines });
      }
    }

    return blocks;
  }

  // ---------------------------------------------------------------------------
  // HTML renderer
  // ---------------------------------------------------------------------------

  function renderList(items: ListItem[], ordered: boolean): string {
    if (items.length === 0) return '';
    const tag = ordered ? 'ol' : 'ul';
    let html = `<${tag}>`;
    for (const item of items) {
      html += `<li>${inlineFormat(escape(item.text))}</li>`;
    }
    html += `</${tag}>`;
    return html;
  }

  function blocksToHtml(blocks: Block[]): string {
    let html = '';
    for (const block of blocks) {
      switch (block.type) {
        case 'h1':
          html += `<h1>${inlineFormat(escape(block.text))}</h1>`;
          break;
        case 'h2':
          html += `<h2>${inlineFormat(escape(block.text))}</h2>`;
          break;
        case 'h3':
          html += `<h3>${inlineFormat(escape(block.text))}</h3>`;
          break;
        case 'h4':
          html += `<h4>${inlineFormat(escape(block.text))}</h4>`;
          break;
        case 'hr':
          html += '<hr>';
          break;
        case 'blockquote':
          html += `<blockquote>${
            block.lines.map(l => inlineFormat(escape(l))).join('<br>')
          }</blockquote>`;
          break;
        case 'code':
          html += `<pre><code class="language-${escape(block.lang)}">${
            block.lines.map(escape).join('\n')
          }</code></pre>`;
          break;
        case 'ul':
          html += renderList(block.items, false);
          break;
        case 'ol':
          html += renderList(block.items, true);
          break;
        case 'p':
          html += `<p>${
            block.lines.map(l => inlineFormat(escape(l))).join(' ')
          }</p>`;
          break;
      }
    }
    return html;
  }

  // ---------------------------------------------------------------------------
  // Reactive HTML
  // ---------------------------------------------------------------------------

  // While streaming, re-parsing the full accumulated answer on every single
  // token is O(n) per token / O(n^2) over a stream (see sessions-log.md,
  // 2026-07-06 diagnostic). Throttle to at most once per PARSE_THROTTLE_MS
  // while streaming — still progressive, just capped — and always force one
  // final parse the moment streaming ends, so the displayed content can
  // never be stale or truncated behind a pending throttle window.
  const PARSE_THROTTLE_MS = 75;

  let html = '';
  let lastParseAt = -Infinity;
  let pendingTimer: ReturnType<typeof setTimeout> | null = null;

  function runParse(text: string): void {
    html = blocksToHtml(parse(text));
    lastParseAt = Date.now();
  }

  function clearPending(): void {
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      pendingTimer = null;
    }
  }

  $: {
    if (!streaming) {
      clearPending();
      runParse(content ?? '');
    } else {
      const elapsed = Date.now() - lastParseAt;
      if (elapsed >= PARSE_THROTTLE_MS) {
        runParse(content ?? '');
      } else if (!pendingTimer) {
        pendingTimer = setTimeout(() => {
          pendingTimer = null;
          runParse(content ?? '');
        }, PARSE_THROTTLE_MS - elapsed);
      }
    }
  }

  onDestroy(clearPending);
</script>

<div
  class="md-render prose"
  class:cursor-blink={streaming}
>
  {@html html}
</div>

<style>
  /*
   * All base styles live in app.css (.prose).
   * These overrides only handle things specific to agent output rendering.
   */

  .md-render {
    /* Let the parent control width/overflow */
    min-width: 0;
    word-break: break-word;
  }

  /* Tighten heading margins inside chat bubbles / section cards */
  .md-render :global(h1) {
    font-size: var(--text-lg);
    font-weight: 700;
    color: var(--text-primary);
    margin-top: var(--sp-4);
    margin-bottom: var(--sp-2);
    padding-bottom: var(--sp-1);
    border-bottom: 1px solid var(--border-soft);
  }

  .md-render :global(h2) {
    font-size: var(--text-md);
    font-weight: 600;
    color: var(--text-primary);
    margin-top: var(--sp-4);
    margin-bottom: var(--sp-2);
    padding-bottom: var(--sp-1);
    border-bottom: 1px solid var(--border-soft);
  }

  .md-render :global(h3) {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-secondary);
    margin-top: var(--sp-3);
    margin-bottom: var(--sp-1);
  }

  .md-render :global(h4) {
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-tertiary);
    margin-top: var(--sp-3);
    margin-bottom: var(--sp-1);
  }

  .md-render :global(p) {
    margin-bottom: var(--sp-3);
    font-size: var(--text-sm);
    line-height: 1.75;
    color: var(--text-primary);
  }

  .md-render :global(p:last-child) { margin-bottom: 0; }

  .md-render :global(ul),
  .md-render :global(ol) {
    padding-left: var(--sp-5);
    margin-bottom: var(--sp-3);
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .md-render :global(li) {
    font-size: var(--text-sm);
    line-height: 1.65;
    color: var(--text-primary);
  }

  .md-render :global(ul li) { list-style-type: disc; }
  .md-render :global(ol li) { list-style-type: decimal; }

  .md-render :global(strong) {
    font-weight: 600;
    color: var(--text-primary);
  }

  .md-render :global(em) {
    font-style: italic;
    color: var(--text-secondary);
  }

  .md-render :global(code) {
    font-family: var(--font-mono);
    font-size: 0.88em;
    background: var(--bg-active);
    color: var(--text-accent);
    padding: 1px 5px;
    border-radius: var(--radius-sm);
  }

  .md-render :global(pre) {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
    overflow-x: auto;
    margin: var(--sp-3) 0;
  }

  .md-render :global(pre code) {
    background: none;
    padding: 0;
    font-size: var(--text-xs);
    line-height: 1.6;
    color: var(--text-secondary);
  }

  .md-render :global(blockquote) {
    border-left: 3px solid var(--accent-mid);
    padding-left: var(--sp-4);
    margin: var(--sp-3) 0;
    color: var(--text-secondary);
    font-style: italic;
  }

  .md-render :global(hr) {
    border: none;
    border-top: 1px solid var(--border);
    margin: var(--sp-4) 0;
  }
</style>
