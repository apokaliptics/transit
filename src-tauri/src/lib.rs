mod capture;
mod downloader;
mod ocr;
mod pipeline;
mod translate;

use std::sync::atomic::{AtomicBool, Ordering};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex as StdMutex};
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, Position, Size, WebviewWindow};
use tokio::sync::Mutex;
use xcap::Monitor;

/// Shared application state.
struct AppState {
    running: Arc<AtomicBool>,
    pipeline_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    capture_region: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
    snip_restore_bounds: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
    control_bounds: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
}

/// Start the translation pipeline.
#[tauri::command]
async fn start_translation(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    // If already running, do nothing.
    if state.running.load(Ordering::Relaxed) {
        return Ok("Already running".to_string());
    }

    state.running.store(true, Ordering::Relaxed);

    // Reset the diff checker for a fresh session
    capture::reset_diff();

    let window = overlay_window(&app)?;
    let running = state.running.clone();
    let capture_region = state.capture_region.clone();

    let handle = tokio::spawn(async move {
        pipeline::run_pipeline(window, running, capture_region).await;
    });

    let mut pipeline = state.pipeline_handle.lock().await;
    *pipeline = Some(handle);

    Ok("Translation started".to_string())
}

/// Stop the translation pipeline.
#[tauri::command]
async fn stop_translation(
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    state.running.store(false, Ordering::Relaxed);

    let mut pipeline = state.pipeline_handle.lock().await;
    if let Some(handle) = pipeline.take() {
        let _ = handle.await;
    }

    Ok("Translation stopped".to_string())
}

/// Set click-through mode on or off.
#[tauri::command]
async fn set_click_through(
    window: WebviewWindow,
    enabled: bool,
) -> Result<String, String> {
    window
        .set_ignore_cursor_events(enabled)
        .map_err(|e| format!("set_ignore_cursor_events error: {e}"))?;

    Ok(format!("Click-through set to {enabled}"))
}

/// Start a new snipping-style capture selection session.
#[tauri::command]
async fn begin_new_capture(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let window = overlay_window(&app)?;
    let control = app
        .get_webview_window("control")
        .ok_or_else(|| "Control window not found".to_string())?;

    let current_bounds = get_window_bounds(&control)?;

    // Persist control module bounds once so unlock can always return there.
    {
        let has_locked_region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?
            .is_some();

        if !has_locked_region {
            let mut control = state
                .control_bounds
                .lock()
                .map_err(|e| format!("Control bounds lock error: {e}"))?;
            *control = Some(pipeline::CaptureRegion {
                x: current_bounds.0,
                y: current_bounds.1,
                width: current_bounds.2,
                height: current_bounds.3,
            });
        }
    }

    {
        let mut restore = state
            .snip_restore_bounds
            .lock()
            .map_err(|e| format!("Snip restore lock error: {e}"))?;
        *restore = Some(pipeline::CaptureRegion {
            x: current_bounds.0,
            y: current_bounds.1,
            width: current_bounds.2,
            height: current_bounds.3,
        });
    }

    {
        let mut region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?;
        *region = None;
    }

    let monitor_bounds = get_monitor_bounds_at_point(current_bounds.0, current_bounds.1)?;

    window.show().map_err(|e| format!("show overlay error: {e}"))?;
    window
        .set_ignore_cursor_events(false)
        .map_err(|e| format!("set_ignore_cursor_events error: {e}"))?;

    window
        .set_position(Position::Physical(PhysicalPosition::new(
            monitor_bounds.0,
            monitor_bounds.1,
        )))
        .map_err(|e| format!("set_position error: {e}"))?;

    window
        .set_size(Size::Physical(PhysicalSize::new(
            monitor_bounds.2,
            monitor_bounds.3,
        )))
        .map_err(|e| format!("set_size error: {e}"))?;

    window.set_focus().map_err(|e| format!("set_focus error: {e}"))?;
    window
        .emit("overlay-mode", "snip")
        .map_err(|e| format!("emit overlay-mode error: {e}"))?;

    capture::reset_diff();

    Ok("Snip capture mode started".to_string())
}

/// Finalize a snipping-style selection and lock translation to that region.
#[tauri::command]
async fn finish_new_capture(
    window: WebviewWindow,
    state: tauri::State<'_, AppState>,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<String, String> {
    if width < 8 || height < 8 {
        return Err("Selected region is too small".to_string());
    }

    let win_pos = window
        .outer_position()
        .map_err(|e| format!("outer_position: {e}"))?;

    let absolute = pipeline::CaptureRegion {
        x: win_pos.x + x,
        y: win_pos.y + y,
        width,
        height,
    };

    {
        let mut region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?;
        *region = Some(absolute);
    }

    // Keep the translation overlay directly on top of the selected text region.
    window
        .set_position(Position::Physical(PhysicalPosition::new(absolute.x, absolute.y)))
        .map_err(|e| format!("set_position error: {e}"))?;

    window
        .set_size(Size::Physical(PhysicalSize::new(absolute.width, absolute.height)))
        .map_err(|e| format!("set_size error: {e}"))?;

    capture::reset_diff();

    Ok(format!(
        "Locked region at ({}, {}) size {}x{}",
        absolute.x, absolute.y, absolute.width, absolute.height
    ))
}

/// Cancel snip mode and restore previous window bounds without changing region lock.
#[tauri::command]
async fn cancel_new_capture(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let window = overlay_window(&app)?;
    restore_window_bounds(&window, &state)?;
    window.hide().map_err(|e| format!("hide overlay error: {e}"))?;
    Ok("Snip capture canceled".to_string())
}

/// Unlock a previously selected capture region and return to control module bounds.
#[tauri::command]
async fn unlock_capture(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let window = overlay_window(&app)?;
    {
        let mut region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?;
        *region = None;
    }

    let control = state
        .control_bounds
        .lock()
        .map_err(|e| format!("Control bounds lock error: {e}"))?
        .to_owned();

    if let Some(bounds) = control {
        window
            .set_position(Position::Physical(PhysicalPosition::new(bounds.x, bounds.y)))
            .map_err(|e| format!("unlock set_position error: {e}"))?;

        window
            .set_size(Size::Physical(PhysicalSize::new(bounds.width, bounds.height)))
            .map_err(|e| format!("unlock set_size error: {e}"))?;
    }

    window.hide().map_err(|e| format!("hide overlay error: {e}"))?;

    if let Some(control_window) = app.get_webview_window("control") {
        let _ = control_window.show();
        let _ = control_window.set_focus();
    }

    capture::reset_diff();
    Ok("Capture unlocked".to_string())
}

/// Initialize the translation engine from the local data directory.
#[tauri::command]
fn init_engine(app: AppHandle, language_pair: String) -> Result<String, String> {
    let local_data_dir = app.path().app_local_data_dir()
        .map_err(|e| format!("Could not determine local data directory: {}", e))?;
    
    let model_dir = local_data_dir.join("models").join(&language_pair);

    if model_dir.exists() {
        safe_init_engine(&model_dir)?;
        Ok("Engine initialized".to_string())
    } else {
        Err(format!("Model files not found at {:?}", model_dir))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            running: Arc::new(AtomicBool::new(false)),
            pipeline_handle: Mutex::new(None),
            capture_region: Arc::new(StdMutex::new(None)),
            snip_restore_bounds: Arc::new(StdMutex::new(None)),
            control_bounds: Arc::new(StdMutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            start_translation,
            stop_translation,
            set_click_through,
            begin_new_capture,
            finish_new_capture,
            cancel_new_capture,
            unlock_capture,
            downloader::download_models,
            downloader::check_models,
            init_engine,
        ])
        .setup(|app| {
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.hide();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn overlay_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    app.get_webview_window("overlay")
        .ok_or_else(|| "Overlay window not found".to_string())
}

fn safe_init_engine(model_dir: &std::path::Path) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(|| translate::init(model_dir))) {
        Ok(result) => result,
        Err(_) => Err("Model initialization panicked. Reinstall model files for this language.".to_string()),
    }
}

fn get_window_bounds(window: &WebviewWindow) -> Result<(i32, i32, u32, u32), String> {
    let position = window
        .outer_position()
        .map_err(|e| format!("outer_position: {e}"))?;

    let size = window
        .outer_size()
        .map_err(|e| format!("outer_size: {e}"))?;

    Ok((position.x, position.y, size.width, size.height))
}

fn get_monitor_bounds_at_point(x: i32, y: i32) -> Result<(i32, i32, u32, u32), String> {
    let monitors = Monitor::all().map_err(|e| format!("Failed to list monitors: {e}"))?;

    let monitor = monitors
        .into_iter()
        .find(|m| {
            let mx = m.x().unwrap_or(0);
            let my = m.y().unwrap_or(0);
            let mw = m.width().unwrap_or(0) as i32;
            let mh = m.height().unwrap_or(0) as i32;
            x >= mx && y >= my && x < mx + mw && y < my + mh
        })
        .ok_or_else(|| "No monitor found at window position".to_string())?;

    Ok((
        monitor.x().unwrap_or(0),
        monitor.y().unwrap_or(0),
        monitor.width().unwrap_or(0),
        monitor.height().unwrap_or(0),
    ))
}

fn restore_window_bounds(window: &WebviewWindow, state: &tauri::State<'_, AppState>) -> Result<(), String> {
    let restore = state
        .snip_restore_bounds
        .lock()
        .map_err(|e| format!("Snip restore lock error: {e}"))?
        .take();

    if let Some(bounds) = restore {
        window
            .set_position(Position::Physical(PhysicalPosition::new(bounds.x, bounds.y)))
            .map_err(|e| format!("restore set_position error: {e}"))?;

        window
            .set_size(Size::Physical(PhysicalSize::new(bounds.width, bounds.height)))
            .map_err(|e| format!("restore set_size error: {e}"))?;
    }

    Ok(())
}
