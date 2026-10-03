# Tauri shell (Phase C of native `.app` packaging)

Wraps the Phase A static frontend build in a native macOS shell, and
spawns both Phase B PyInstaller bundles (`backend/packaging/dist/
localist-backend/`, `.../localist-mcp/`) as child processes on launch —
killing them on quit.

## Prerequisite: build the sidecars first

Tauri doesn't build these itself — `bundle.resources` in `tauri.conf.json`
just references `../../backend/packaging/dist/` directly, so it must
already exist:

```bash
cd backend/packaging
source ../.venv-packaging/bin/activate   # see backend/packaging/README.md
pyinstaller localist-backend.spec
pyinstaller localist-mcp.spec
```

## Why not Tauri's sidecar/`externalBin` mechanism

Checked against Tauri's own docs before building this: `externalBin` is
single-binary only. Our PyInstaller builds are onedir (an executable plus
a large `_internal/` dependency folder it needs alongside it) — that
doesn't fit. Instead, each onedir folder is bundled whole via the more
general `bundle.resources` (which does support directory-structure-
preserving folder bundling), resolved to a real path at runtime via
`app.path().resolve(..., BaseDirectory::Resource)`, and spawned with
plain Rust `std::process::Command` — not the shell plugin's JS-invokable
`Command`/sidecar API, whose capability system only allow-lists fixed
command names, not runtime-resolved paths. Nothing here is invoked from
JavaScript — spawning is app-lifecycle-driven — so no shell plugin or
capability JSON is needed at all; see `src/lib.rs`.

## Build

```bash
cd localist-ui
npx tauri build           # release — also produces a .dmg
npx tauri build --debug   # debug — same bundling, unoptimized, faster
```

Output: `src-tauri/target/{release,debug}/bundle/macos/Localist.app`.

## A real bug this caught: `RunEvent::ExitRequested` doesn't fire here

`src/lib.rs`'s cleanup handler originally matched only on
`RunEvent::ExitRequested` — Tauri's docs describe it as the normal
"about to exit" hook. Live-tested (AppleScript `tell application
"Localist" to quit`, the same Apple Event Cmd+Q/Dock-quit send): this
app's default configuration goes straight to `RunEvent::Exit` with no
`ExitRequested` at all. Confirmed by adding temporary `eprintln!`
instrumentation and watching a real quit — both sidecar processes were
left running (`ps aux` showed them still alive after the app itself had
exited) until the handler was changed to match both events. Fixed; kept
matching both since `.take()` on the stored `Option<Child>` makes
handling either (or both) harmless.

## What's verified so far (Phase C)

Both debug and release `.app` builds: launched via `open` (the real user
flow, not a dev shortcut) and directly, both sidecars start automatically
and respond on their real ports, the real webview makes genuine periodic
`GET /health`/`GET /agents` calls that reach the packaged backend with no
CORS errors (confirms `main.py`'s `tauri://localhost` origin fix), a
screenshot confirmed the real frontend renders correctly (not blank), and
quitting via AppleScript reliably leaves zero orphaned processes. Not yet
covered: first-run config UX (Phase D — a fresh `.app` still defaults to
`omlx`, unreachable without setup, same as source-tree today), and wiring
`tauri build` to trigger the PyInstaller build itself rather than assuming
it's already done.

Code signing/notarization (Phase E) is a deliberate scope decision, not a
gap: no Apple Developer ID certificate is available for this project, so
the `.dmg`/`.app` ship unsigned. Gatekeeper flags them on first launch —
documented for end users in the root `README.md`'s "Native macOS app"
section (`xattr -cr`; right-click → Open does not bypass this on current
macOS). `tauri.conf.json`'s `bundle` block has no signing identity or
entitlements config for the same reason.

## A real bug this caught: `tauri build`'s ad-hoc signature breaks once resources are added

`tauri build` ad-hoc-signs `Localist.app` before `bundle.resources`
(the sidecar folders) are copied in, which invalidates the signature's
resource seal — `codesign --verify --deep --strict` fails with "code has
no resources but signature indicates they must be present". Live-tested:
a `.dmg` built from that broken signature triggers macOS's **"is damaged
and can't be opened"** error on a real quarantined download — the harsher
failure mode current macOS uses for an invalid signature, distinct from
(and not fixable by) the normal "unidentified developer" gate that
`xattr -cr` resolves. The fix is to re-sign ad-hoc *after* resources are
in place and rebuild the DMG from the corrected `.app`:

```bash
APP=target/release/bundle/macos/Localist.app
DMG=target/release/bundle/dmg/Localist_0.1.0_aarch64.dmg
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict --verbose=4 "$APP"   # should report "valid on disk"
rm -f "$DMG"
hdiutil create -volname "Localist" -srcfolder "$APP" -ov -format UDZO "$DMG"
```

This is automated as a build step in `.github/workflows/release-macos.yml`
("Re-sign .app and rebuild DMG") — a local `tauri build` still needs this
run manually afterward, same as before CI picked it up.

## Releasing a new version

No auto-update path exists yet — every release is a fresh manual
download/install for users (drag to Applications, `xattr -cr`, relaunch).
To cut a new one:

1. Bump the version number in **both** `localist-ui/package.json` and
   `localist-ui/src-tauri/tauri.conf.json` (`"version"` field in each —
   they must match).
2. Commit that bump.
3. Tag the commit and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   This is what triggers `.github/workflows/release-macos.yml` — pushing
   the branch alone does not.
4. The workflow builds the sidecars, builds + re-signs the `.app`/`.dmg`,
   and creates (or updates) a **draft** release for that tag. It does not
   publish automatically.
5. Review the draft on GitHub (download and test the DMG if in doubt),
   then click **Publish release** when satisfied.

Re-running the workflow against a tag that already has a draft release
updates that draft's assets but does **not** overwrite its title/body —
confirmed live: a workflow body-text edit landed in the build but the
existing draft kept its old text until the draft was deleted
(`gh release delete vX.Y.Z --yes --cleanup-tag=false`, keeping the tag)
and the workflow re-run (`gh workflow run release-macos.yml --ref vX.Y.Z`)
to recreate it from scratch. Only matters if you edit the workflow's
release body/notes between runs on the same tag — a plain rebuild (new
commit, new tag) is unaffected.
