<script lang="ts">
  import { page } from '$app/stores';

  interface NavItem {
    href: string;
    label: string;
    icon: string;
  }

  const nav: NavItem[] = [
    { href: '/conversation', label: 'Conversation', icon: 'chat' },
    { href: '/research',     label: 'Research',     icon: 'search' },
    { href: '/files',        label: 'Files',        icon: 'folder' },
    { href: '/settings',     label: 'Settings',     icon: 'settings' }
  ];

  $: active = $page.url.pathname;
</script>

<aside class="sidebar" aria-label="Main navigation">
  <!-- Wordmark -->
  <div class="wordmark">
    <span class="wordmark-logo">L</span>
    <span class="wordmark-text">LORA</span>
  </div>

  <!-- Nav links -->
  <nav>
    <ul>
      {#each nav as item}
        {@const isActive = active.startsWith(item.href)}
        <li>
          <a
            href={item.href}
            class="nav-link"
            class:active={isActive}
            aria-current={isActive ? 'page' : undefined}
          >
            <span class="nav-icon" aria-hidden="true">
              {#if item.icon === 'chat'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
              {:else if item.icon === 'search'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
              {:else if item.icon === 'folder'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                </svg>
              {:else if item.icon === 'settings'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
              {/if}
            </span>
            <span class="nav-label">{item.label}</span>
            {#if isActive}
              <span class="active-bar" aria-hidden="true" />
            {/if}
          </a>
        </li>
      {/each}
    </ul>
  </nav>

  <!-- Footer label -->
  <div class="sidebar-footer">
    <span class="text-tertiary text-xs">v0.1.0</span>
  </div>
</aside>

<style>
  .sidebar {
    grid-column: 1;
    grid-row: 1 / -1;
    display: flex;
    flex-direction: column;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    width: var(--sidebar-w);
    height: 100vh;
    overflow: hidden;
    position: relative;
    z-index: 10;
  }

  /* Wordmark */
  .wordmark {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    padding: 0 var(--sp-5);
    height: var(--topbar-h);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    user-select: none;
  }

  .wordmark-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--accent);
    color: #fff;
    font-size: 13px;
    font-weight: 700;
    border-radius: 5px;
    letter-spacing: -0.03em;
    flex-shrink: 0;
  }

  .wordmark-text {
    font-size: var(--text-sm);
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--text-primary);
  }

  /* Nav */
  nav {
    flex: 1;
    padding: var(--sp-3) 0;
    overflow-y: auto;
  }

  ul {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 var(--sp-2);
  }

  .nav-link {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    padding: var(--sp-2) var(--sp-3);
    border-radius: var(--radius);
    color: var(--text-secondary);
    font-size: var(--text-sm);
    font-weight: 400;
    text-decoration: none;
    transition:
      color var(--dur-fast) var(--ease),
      background var(--dur-fast) var(--ease);
    overflow: hidden;
  }

  .nav-link:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .nav-link.active {
    color: var(--text-accent);
    background: var(--accent-glow);
    font-weight: 500;
  }

  .nav-icon {
    display: flex;
    align-items: center;
    flex-shrink: 0;
    opacity: 0.7;
    transition: opacity var(--dur-fast) var(--ease);
  }

  .nav-link:hover .nav-icon,
  .nav-link.active .nav-icon {
    opacity: 1;
  }

  .nav-label {
    flex: 1;
  }

  /* Active accent bar — left edge */
  .active-bar {
    position: absolute;
    left: 0;
    top: 20%;
    bottom: 20%;
    width: 2.5px;
    background: var(--accent);
    border-radius: 0 2px 2px 0;
  }

  /* Footer */
  .sidebar-footer {
    padding: var(--sp-4) var(--sp-5);
    border-top: 1px solid var(--border-soft);
    flex-shrink: 0;
  }
</style>
