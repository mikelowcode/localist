# §25 — Machine Migration Checklist (New Mac, e.g. M6)

**Status: Authoritative** | **Last updated:** 2026-10-03

Step-by-step checklist for moving Localist development from one Apple
Silicon Mac to another (motivating case: a new Mac Mini M6, but applies to
any Mac → Mac move). Written so a future Claude Code session can execute
it directly, in order, without re-deriving which parts of the project are
portable and which aren't.

## 25.1 What a plain repo copy gets right, and what it misses

Copying `~/Projects/lora-app-demo/` (or cloning the git remote fresh) onto
the new machine brings over the entire source tree, all documentation,
`backend/wiki/`/`backend/raw/`/`backend/generated_files/` (personal data,
gitignored but real files on disk), and `backend/.env`. That's most of the
project, but three categories of state do **not** travel correctly in a
plain copy, and one category lives entirely outside the repo directory:

1. **Python virtual environments** (`backend/.venv/`, `backend/.venv-packaging/`)
   — these bake in absolute interpreter paths at creation time; copying
   the directory instead of recreating it is unreliable even across two
   machines of the same architecture. Recreate, don't copy.
2. **Node dependencies** (`localist-ui/node_modules/`) — safe to
   reinstall fresh rather than copy; smaller and avoids any
   binary-native-module mismatch.
3. **Rust build cache** (`localist-ui/src-tauri/target/`) — Cargo's own
   build output, meant to be regenerated, not copied. The Rust toolchain
   itself (`rustup`/`cargo`) is a separate system-level install, not part
   of the repo at all.
4. **The packaged desktop app's real data store**,
   `~/Library/Application Support/Localist/` — this is **outside**
   `~/Projects/lora-app-demo/` entirely (see §16's `paths.py`: a frozen
   `Localist.app` always reads/writes this fixed path, regardless of
   which build or download produced the `.app`). A plain repo copy will
   not bring this over; it needs its own explicit step if the user wants
   that history on the new machine (see §25.5).

## 25.1a The project directory can be renamed on the new machine

Confirmed by grepping the whole repo for the current directory name
(`lora-app-demo`): nothing functional depends on it. `paths.py`'s
`get_backend_root()`/`get_resource_root()` resolve from the file's own
location (`Path(__file__).resolve().parent...`) or from `sys.frozen`/
`sys._MEIPASS` when packaged — never from an assumed parent folder name.
`start_localist.sh`, `tauri.conf.json`'s `bundle.resources` paths, and
both `pyproject.toml`/`package.json` configs are the same way: relative
to their own location, not the repo's directory name. The project can be
renamed to anything (e.g. `localist-app`) when set up on the new machine,
with no functional breakage.

What *does* reference the current name, all cosmetic (safe to ignore, not
worth fixing up on rename):

- Comment headers in `PLAN_retire_mlx_embeddinggemma.md`/
  `PLAN_semantic_gating_calibration.md` and one docstring example in
  `memory_manager.py` — descriptive text, not executed logic.
- Usage-example comments in several `diagnostics/*.py` scripts
  (`# cd /Users/.../lora-app-demo`) — instructional text inside a comment
  block, not a path the script actually uses at runtime.
- `.claude/settings.local.json` — one literal absolute-path permission
  rule baked in by Claude Code itself. This is local tooling
  configuration, not project code; it will simply go stale/unused on a
  renamed or relocated directory rather than break anything. No action
  needed — Claude Code regenerates entries like this as needed on the new
  machine.

Also outside the repo: **Ollama** itself (a separate macOS app) and its
model weights (`~/.ollama`, not project-scoped). Re-pulling cloud-model
references doesn't re-download weights, but the Ollama daemon must still
be installed and running on the new machine for either local or cloud
models to work.

## 25.2 Step-by-step migration

Run these **on the new machine**, in order, after the repo itself has
been copied or cloned into place (e.g. via a direct volume copy, Time
Machine restore, AirDrop, or `git clone` from the remote plus a manual
copy of the gitignored `backend/wiki/`, `backend/raw/`,
`backend/generated_files/`, and `backend/.env` from the old machine).

### Step 1 — Verify the repo landed intact

```bash
cd ~/Projects/lora-app-demo
git status
git log --oneline -5
```

Confirm the branch (`public-beta` or `main`) and recent commits match
what's expected — this catches a partial copy before anything else is
built on top of it.

### Step 2 — Install system-level prerequisites

These are not part of the repo and must be installed fresh on the new
machine:

- **Xcode Command Line Tools**: `xcode-select --install` (needed for
  Rust's linker and for Python C-extension builds)
- **Rust toolchain**: install via [rustup.rs](https://rustup.rs) —
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.sh | sh`
- **Python 3.11+** (3.13 is what CI/the author's machine use —
  `backend/pyproject.toml` requires `>=3.11`): via `python.org`,
  Homebrew, or pyenv — whichever the user already prefers
- **Node.js 20+**: via Homebrew (`brew install node`), nvm, or
  `nodejs.org`
- **Ollama**: download from [ollama.com](https://ollama.com), install,
  launch once so the daemon is running. If any local-tier models were
  used on the old machine (not just cloud models), re-pull them:
  `ollama pull <model-name>`. Cloud models (e.g. `gemma4:31b-cloud`) need
  no local weights, just the daemon running and the account
  authenticated, if auth was configured.

### Step 3 — Recreate the backend Python environments

```bash
cd ~/Projects/lora-app-demo/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ocr,chart,dev]"   # Apple Silicon: full local stack
deactivate

python3 -m venv .venv-packaging
source .venv-packaging/bin/activate
pip install -e ".[dev,packaging]"   # extras-free — see backend/packaging/README.md for why
deactivate
```

### Step 4 — Reinstall frontend dependencies

```bash
cd ~/Projects/lora-app-demo/localist-ui
npm ci   # uses the committed package-lock.json
```

### Step 5 — Verify the source-tree app runs

```bash
cd ~/Projects/lora-app-demo
./start_localist.sh
```

Confirm backend (8001), localist-mcp (8003), and the frontend dev server
(5173) all come up cleanly and the UI loads real data from
`backend/wiki/`/`backend/raw/` (if those were copied over). `Ctrl+C` or
`./start_localist.sh --stop` to shut down once confirmed.

### Step 6 — Verify the packaged desktop build still works

Only needed if desktop packaging work continues on the new machine (not
just source-tree development):

```bash
cd ~/Projects/lora-app-demo/backend/packaging
source ../.venv-packaging/bin/activate
pyinstaller -y localist-backend.spec
pyinstaller -y localist-mcp.spec
deactivate

cd ~/Projects/lora-app-demo/localist-ui
npm run tauri:build
```

Then re-sign and rebuild the DMG per §25's sibling doc
(`localist-ui/src-tauri/README.md`'s "Releasing a new version" section)
— **do not skip this**: `tauri build`'s ad-hoc signature breaks once
`bundle.resources` is copied in (see that README's "A real bug this
caught" entry), and a DMG built without the re-sign step will fail with
macOS's "is damaged and can't be opened" error on any machine, including
this one.

### Step 7 — GitHub Actions / release workflow

No migration action needed — `.github/workflows/release-macos.yml` runs
on GitHub's own macOS runners, not the developer's local machine. Tagging
and pushing from the new machine works identically to the old one, once
`git` is configured with the right remote and credentials
(`gh auth login` if using the `gh` CLI for release management, as this
project's workflow log does).

## 25.3 Verifying nothing was lost

After Steps 1-5, confirm the personal data categories actually came over,
not just the code:

```bash
cd ~/Projects/lora-app-demo/backend
ls wiki/ raw/ generated_files/   # should show real content, not empty dirs
ls -la localist_memory.db        # should exist with a real file size, not 0 bytes
```

If any of these are empty or missing, the gitignored personal-data
directories weren't actually copied — they need a manual copy from the
old machine (they are deliberately excluded from `git`, see the root
`.gitignore`'s "Localist private runtime directories" section and
`PRIVACY.md`).

## 25.4 A tempting shortcut that doesn't work: copying `.venv`/`node_modules`/`target` directly

Noted explicitly because it's the obvious first instinct and looks like
it should save time: copying `backend/.venv/`, `backend/.venv-packaging/`,
`localist-ui/node_modules/`, or `localist-ui/src-tauri/target/` wholesale
(instead of recreating/reinstalling them per Steps 3-4) risks silent
breakage — absolute paths baked into venv activation scripts, native
Node addons built for a slightly different toolchain version, and Cargo
build artifacts that may not even be architecture-portable depending on
what changed between the two machines' Rust/Xcode versions. Both target
machines in the documented case are Apple Silicon (no x86_64/arm64
cross-architecture concern), but the path-baking issue alone makes a
direct copy the wrong call. Reinstalling is fast enough (minutes, not
hours) that there's no real time savings being given up.

## 25.5 Bringing over the packaged desktop app's real data store

Only relevant if the user has been using a packaged `Localist.app` (not
just the source-tree `start_localist.sh` flow) and wants that history —
wiki pages, episodic memory, chat history — on the new machine too. This
data lives at a fixed path **outside** the project repo (see §25.1,
point 4) and needs its own explicit transfer:

```bash
# On the OLD machine, after quitting Localist.app:
tar -czf localist-app-support-backup.tar.gz \
  -C ~/Library/"Application Support" Localist

# Transfer localist-app-support-backup.tar.gz to the new machine (AirDrop,
# USB drive, scp, etc.), then on the NEW machine, before first launching
# the packaged app there:
mkdir -p ~/Library/"Application Support"
tar -xzf localist-app-support-backup.tar.gz -C ~/Library/"Application Support"
```

Verify after extraction:

```bash
ls -la ~/Library/"Application Support"/Localist/
# expect: .env, wiki/, raw/, generated_files/, chat_uploads/, localist_memory.db
```

If the new machine's first `Localist.app` launch happens *before* this
restore step, it will create a fresh, empty directory at that same path
(per §16's `paths.py` default) — restoring afterward requires quitting
the app first and overwriting that fresh-but-empty directory, not merging
into it.

## 25.6 Open items

None yet — this checklist has not been live-run end-to-end against a real
second machine as of this writing (2026-10-03, written ahead of an
anticipated Mac Mini M6 purchase). The first real run should correct
anything this write-up got wrong about install prerequisites or ordering,
the same way every other section in this spec gets corrected by live use
rather than left as untested prose.
