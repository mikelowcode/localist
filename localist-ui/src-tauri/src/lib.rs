// Spawns the two frozen Localist services (localist-backend :8001,
// localist-mcp :8003 — see backend/packaging/) as child processes on
// launch, kills them on quit.
//
// Not Tauri's sidecar/externalBin mechanism: that's single-binary only
// (confirmed against Tauri's docs), and each PyInstaller build here is
// onedir — an executable plus a large _internal/ dependency folder it
// needs alongside it. Instead, each onedir folder is bundled whole via
// `bundle.resources` (tauri.conf.json), resolved to a real filesystem
// path at runtime, and spawned with plain std::process::Command. Nothing
// here is invoked from JavaScript — spawning is app-lifecycle-driven, not
// user/webview-triggered — so this needs no shell plugin or capability
// JSON; Tauri's capability system only gates what the webview can invoke
// over IPC.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::path::BaseDirectory;
use tauri::Manager;
use tauri_plugin_log::{RotationStrategy, Target, TargetKind};

// One folder for every log the desktop app produces (§7.30): this shell's
// app.log, plus the sidecars' backend.log / mcp.log (written by the Python
// side — backend/src/localist/log_file.py — which is told this same path
// via LOCALIST_LOG_DIR). Settings → "Show logs" opens it.
const LOG_MAX_BYTES: u128 = 5 * 1024 * 1024;
const LOG_KEEP_FILES: usize = 3;

fn log_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path()
        .home_dir()
        .ok()
        .map(|home| home.join("Library").join("Logs").join("Localist"))
}

struct Sidecars {
    backend: Option<Child>,
    mcp: Option<Child>,
}

fn spawn_sidecar(
    app: &tauri::AppHandle,
    resource_relative: &str,
    label: &str,
    log_dir: Option<&Path>,
) -> Option<Child> {
    let path = match app.path().resolve(resource_relative, BaseDirectory::Resource) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("{label}: could not resolve bundled resource path: {e}");
            return None;
        }
    };

    // Launched via `open`/Finder/Dock (LaunchServices), this process' own
    // stdio is not a terminal — without an explicit Stdio here, the child
    // inherits those same fds verbatim (Command's default). Reproduced
    // live: with real wiki/raw data to index (real startup log volume),
    // the backend sidecar died within ~1s every time under a LaunchServices
    // launch while the identical binary launched directly from a shell
    // (inheriting a real tty) started fine every time — an empty first-run
    // corpus (near-zero startup output) didn't trigger it either way. Null
    // stdio removes the inherited-fd variable entirely rather than relying
    // on this process always having a well-behaved stdout/stderr to give.
    // Logging still reaches disk: each sidecar writes its own rotating log
    // file (see log_dir above) rather than through these fds.
    let mut command = Command::new(&path);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = log_dir {
        command.env("LOCALIST_LOG_DIR", dir);
    }
    match command.spawn() {
        Ok(child) => {
            log::info!("{label}: started (pid {})", child.id());
            Some(child)
        }
        // Expected during `tauri dev` alongside an already-running
        // start_localist.sh (port already bound) — log and continue
        // rather than crashing the app, same as every other
        // platform/opt-in gate in this codebase.
        Err(e) => {
            log::warn!("{label}: failed to start ({path:?}): {e}");
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Release builds log too (§7.30) — before this, a sidecar that
            // failed to start left no trace anywhere.
            let logs = log_dir(app.handle());
            let mut targets = Vec::new();
            if let Some(dir) = &logs {
                targets.push(Target::new(TargetKind::Folder {
                    path: dir.clone(),
                    file_name: Some("app".into()),
                }));
            }
            if cfg!(debug_assertions) {
                targets.push(Target::new(TargetKind::Stdout));
            }
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .clear_targets()
                    .targets(targets)
                    .level(log::LevelFilter::Info)
                    .max_file_size(LOG_MAX_BYTES)
                    .rotation_strategy(RotationStrategy::KeepSome(LOG_KEEP_FILES))
                    .build(),
            )?;
            log::info!("Localist {} starting", app.package_info().version);

            let backend = spawn_sidecar(
                app.handle(),
                "localist-backend/localist-backend",
                "localist-backend",
                logs.as_deref(),
            );
            let mcp = spawn_sidecar(
                app.handle(),
                "localist-mcp/localist-mcp",
                "localist-mcp",
                logs.as_deref(),
            );
            app.manage(Mutex::new(Sidecars { backend, mcp }));

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Not automatic on macOS just because the parent process
            // exited — an orphaned backend/mcp process surviving app quit
            // (ports staying bound) would be a real, user-visible bug, not
            // a theoretical one. Handles both ExitRequested and Exit —
            // live-tested (AppleScript "quit", the same Apple Event
            // Cmd+Q/Dock-quit send): this app's default configuration
            // goes straight to RunEvent::Exit with no ExitRequested at
            // all, not the "cancelable pre-exit hook" its docs describe
            // as the normal case. take() makes handling both harmless if
            // some other quit path does fire ExitRequested first.
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                if let Some(sidecars) = app_handle.try_state::<Mutex<Sidecars>>() {
                    let mut sidecars = sidecars.lock().unwrap();
                    if let Some(mut child) = sidecars.backend.take() {
                        let _ = child.kill();
                    }
                    if let Some(mut child) = sidecars.mcp.take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
