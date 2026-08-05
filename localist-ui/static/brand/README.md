# Localist Logo

Monochrome "place marker" mark — a map pin with an open center, standing for local-first: everything the agent needs stays on your machine, not the cloud.

## Files
- `icon.svg` — mark only, transparent background, `fill="currentColor"`. Use for favicons, nav icons, anywhere you need to set the color via CSS.
- `logomark-tile.svg` — mark on its dark rounded-square tile (120×120). Use as the app icon / avatar.
- `lockup-light.svg` / `lockup-dark.svg` — tile + "LOCALIST" wordmark, for light and dark page backgrounds respectively.

## Colors
- Tile background: `#111318`
- Mark: `#F5F4F0`
- `icon.svg` has no fixed color — it inherits `color` from its parent, so it works on any background.

## Sizing
- Minimum clear size: 16px (favicon). Below that the open center may fill in — keep as a solid dot if so.
- Tile corner radius is 26/120 (~21.7%) of the tile size; scale proportionally if you resize the tile.

## Svelte usage
```svelte
<script>
  import Icon from './icon.svg?raw'; // or import as component via svelte-svg tooling
</script>

<span style="color: white; width: 24px; height: 24px; display: inline-block;">
  {@html Icon}
</span>
```
Or reference directly as an `<img src="/logomark-tile.svg">` for the app icon.

## Don'ts
- Don't recolor the tile — it should stay near-black so the mark reads as "local"/on-device.
- Don't add a gradient or drop shadow.
- Don't stretch non-uniformly.

---
Used in this project at `localist-ui/static/brand/`: `logomark-tile.svg` as the
browser favicon / apple-touch-icon (`src/app.html`), and `icon.svg`'s path data
inlined directly in `src/lib/components/Sidebar.svelte`'s wordmark mark.
