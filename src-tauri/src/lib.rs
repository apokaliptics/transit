mod capture;
mod downloader;
mod ocr;
mod pipeline;
mod translate;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, PhysicalSize, Position, Size, WebviewWindow,
};
use tokio::sync::Mutex;
use xcap::Monitor;

/// Shared application state.
struct AppState {
    running: Arc<AtomicBool>,
    pipeline_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    capture_region: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
    snip_restore_bounds: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
    control_bounds: Arc<StdMutex<Option<pipeline::CaptureRegion>>>,
    language_pair: Arc<StdMutex<String>>,
    capture_scale: Arc<StdMutex<f64>>,
    ocr_backend: Arc<StdMutex<String>>,
}

/// Start the translation pipeline.
#[tauri::command]
async fn start_translation(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    language_pair: Option<String>,
    ocr_backend: Option<String>,
) -> Result<String, String> {
    if let Some(pair) = language_pair {
        let mut selected = state
            .language_pair
            .lock()
            .map_err(|e| format!("Language pair lock error: {e}"))?;
        *selected = pair;
    }

    if let Some(backend) = ocr_backend {
        let mut selected_backend = state
            .ocr_backend
            .lock()
            .map_err(|e| format!("OCR backend lock error: {e}"))?;
        *selected_backend = backend;
    }

    {
        let mut pipeline = state.pipeline_handle.lock().await;
        if state.running.load(Ordering::Relaxed) {
            let stale_running = match pipeline.as_ref() {
                Some(handle) => handle.is_finished(),
                None => true,
            };

            if !stale_running {
                return Ok("Already running".to_string());
            }

            log::warn!("Pipeline marked running but task is stale; resetting state");
            if let Some(handle) = pipeline.take() {
                let _ = handle.await;
            }
            state.running.store(false, Ordering::Relaxed);
        } else if pipeline
            .as_ref()
            .map(|handle| handle.is_finished())
            .unwrap_or(false)
        {
            let _ = pipeline.take();
        }
    }

    let selected_language = state
        .language_pair
        .lock()
        .map_err(|e| format!("Language pair lock error: {e}"))?
        .clone();
    let model_dir = models_root_dir(&app)?.join(&selected_language);
    if !model_dir.exists() {
        return Err(format!("Model files not found at {:?}", model_dir));
    }
    safe_init_engine(&model_dir)?;

    state.running.store(true, Ordering::Relaxed);

    // Reset the diff checker for a fresh session
    capture::reset_diff();

    let window = overlay_window(&app)?;
    let _ = window.emit(
        "ocr-status",
        pipeline::OcrStatusPayload {
            state: "warning".to_string(),
            language_pair: selected_language.clone(),
            message: "Capture pipeline starting...".to_string(),
            backend: state
                .ocr_backend
                .lock()
                .map(|value| value.clone())
                .unwrap_or_else(|_| "auto".to_string()),
            language_tag: "".to_string(),
            used_profile_fallback: false,
        },
    );

    let running = state.running.clone();
    let capture_region = state.capture_region.clone();
    let language_pair = state.language_pair.clone();
    let capture_scale = state.capture_scale.clone();
    let ocr_backend = state.ocr_backend.clone();

    let handle = tokio::spawn(async move {
        pipeline::run_pipeline(
            window,
            running.clone(),
            capture_region,
            language_pair,
            capture_scale,
            ocr_backend,
        )
        .await;
        running.store(false, Ordering::Relaxed);
    });

    let mut pipeline = state.pipeline_handle.lock().await;
    *pipeline = Some(handle);

    Ok("Translation started".to_string())
}

/// Stop the translation pipeline.
#[tauri::command]
async fn stop_translation(state: tauri::State<'_, AppState>) -> Result<String, String> {
    state.running.store(false, Ordering::Relaxed);

    let mut pipeline = state.pipeline_handle.lock().await;
    if let Some(handle) = pipeline.take() {
        let _ = handle.await;
    }

    Ok("Translation stopped".to_string())
}

/// Set click-through mode on or off.
#[tauri::command]
async fn set_click_through(app: AppHandle, enabled: bool) -> Result<String, String> {
    let window = overlay_window(&app)?;
    window
        .set_ignore_cursor_events(enabled)
        .map_err(|e| format!("set_ignore_cursor_events error: {e}"))?;

    emit_click_through_state(&app, enabled);

    if enabled {
        if let Some(control_window) = app.get_webview_window("control") {
            let _ = control_window.show();
            let _ = control_window.set_focus();
        }
    }

    Ok(format!("Click-through set to {enabled}"))
}

/// Start a new snipping-style capture selection session on the caller's monitor.
#[tauri::command]
async fn begin_new_capture(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
    caller: WebviewWindow,
) -> Result<String, String> {
    let window = overlay_window(&app)?;
    let caller_bounds = get_window_bounds(&caller)?;

    // Persist control module bounds once so unlock can always return there.
    if caller.label() == "control" {
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
                x: caller_bounds.0,
                y: caller_bounds.1,
                width: caller_bounds.2,
                height: caller_bounds.3,
            });
        }
    }

    {
        let mut restore = state
            .snip_restore_bounds
            .lock()
            .map_err(|e| format!("Snip restore lock error: {e}"))?;
        *restore = Some(pipeline::CaptureRegion {
            x: caller_bounds.0,
            y: caller_bounds.1,
            width: caller_bounds.2,
            height: caller_bounds.3,
        });
    }

    {
        let mut region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?;
        *region = None;
    }

    let monitor_bounds = get_monitor_bounds_for_window(&caller)?;

    window
        .show()
        .map_err(|e| format!("show overlay error: {e}"))?;
    window
        .set_ignore_cursor_events(false)
        .map_err(|e| format!("set_ignore_cursor_events error: {e}"))?;
    emit_click_through_state(&app, false);

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

    window
        .set_focus()
        .map_err(|e| format!("set_focus error: {e}"))?;
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
    scale_factor: f64,
) -> Result<String, String> {
    if width < 8 || height < 8 {
        return Err("Selected region is too small".to_string());
    }

    let win_pos = window
        .outer_position()
        .map_err(|e| format!("outer_position: {e}"))?;

    let scale = if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    };

    let scaled_x = (x as f64 * scale).round() as i32;
    let scaled_y = (y as f64 * scale).round() as i32;
    let scaled_width = (width as f64 * scale).round().max(1.0) as u32;
    let scaled_height = (height as f64 * scale).round().max(1.0) as u32;

    let absolute = pipeline::CaptureRegion {
        x: win_pos.x + scaled_x,
        y: win_pos.y + scaled_y,
        width: scaled_width,
        height: scaled_height,
    };

    {
        let mut stored_scale = state
            .capture_scale
            .lock()
            .map_err(|e| format!("Capture scale lock error: {e}"))?;
        *stored_scale = scale;
    }

    {
        let mut region = state
            .capture_region
            .lock()
            .map_err(|e| format!("Capture region lock error: {e}"))?;
        *region = Some(absolute);
    }

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
    let _ = window.set_ignore_cursor_events(false);
    emit_click_through_state(&app, false);
    restore_window_bounds(&window, &state)?;
    window
        .hide()
        .map_err(|e| format!("hide overlay error: {e}"))?;
    Ok("Snip capture canceled".to_string())
}

/// Unlock a previously selected capture region and return to control module bounds.
#[tauri::command]
async fn unlock_capture(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let window = overlay_window(&app)?;
    let _ = window.set_ignore_cursor_events(false);
    emit_click_through_state(&app, false);
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
            .set_position(Position::Physical(PhysicalPosition::new(
                bounds.x, bounds.y,
            )))
            .map_err(|e| format!("unlock set_position error: {e}"))?;

        window
            .set_size(Size::Physical(PhysicalSize::new(
                bounds.width,
                bounds.height,
            )))
            .map_err(|e| format!("unlock set_size error: {e}"))?;
    }

    window
        .hide()
        .map_err(|e| format!("hide overlay error: {e}"))?;

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
    let model_dir = models_root_dir(&app)?.join(&language_pair);

    if model_dir.exists() {
        safe_init_engine(&model_dir)?;
        Ok("Engine initialized".to_string())
    } else {
        Err(format!("Model files not found at {:?}", model_dir))
    }
}

#[tauri::command]
fn check_ocr_support(language_pair: String) -> Result<bool, String> {
    ocr::is_language_supported(&language_pair)
}

#[tauri::command]
fn check_ocr_backend_support(backend: String) -> Result<bool, String> {
    let parsed = ocr::OcrBackend::from_str(&backend);
    ocr::is_backend_available(parsed)
}

#[tauri::command]
fn set_ocr_backend(state: tauri::State<'_, AppState>, backend: String) -> Result<String, String> {
    let parsed = ocr::OcrBackend::from_str(&backend);
    let mut selected_backend = state
        .ocr_backend
        .lock()
        .map_err(|e| format!("OCR backend lock error: {e}"))?;
    *selected_backend = parsed.as_str().to_string();
    Ok(format!("OCR backend set to {}", parsed.as_str()))
}

#[tauri::command]
fn get_ocr_backend(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let selected_backend = state
        .ocr_backend
        .lock()
        .map_err(|e| format!("OCR backend lock error: {e}"))?;
    Ok(selected_backend.clone())
}

#[derive(serde::Serialize)]
struct OcrDiagnostics {
    backend: String,
    backend_available: bool,
    language_pair: String,
    language_supported: bool,
    fallback_policy: String,
}

#[derive(serde::Serialize)]
struct PipelineState {
    running: bool,
    has_task: bool,
    task_finished: bool,
    has_capture_region: bool,
}

#[tauri::command]
fn get_ocr_diagnostics(language_pair: String, backend: String) -> Result<OcrDiagnostics, String> {
    let parsed = ocr::OcrBackend::from_str(&backend);
    let backend_available = ocr::is_backend_available(parsed)?;
    let language_supported = ocr::is_language_supported_for_backend(parsed, &language_pair)?;

    Ok(OcrDiagnostics {
        backend: parsed.as_str().to_string(),
        backend_available,
        language_pair,
        language_supported,
        fallback_policy: parsed.fallback_policy().to_string(),
    })
}

#[tauri::command]
async fn get_pipeline_state(state: tauri::State<'_, AppState>) -> Result<PipelineState, String> {
    let running = state.running.load(Ordering::Relaxed);

    let (has_task, task_finished) = {
        let pipeline = state.pipeline_handle.lock().await;
        let has_task = pipeline.is_some();
        let task_finished = pipeline
            .as_ref()
            .map(|handle| handle.is_finished())
            .unwrap_or(false);
        (has_task, task_finished)
    };

    let has_capture_region = state
        .capture_region
        .lock()
        .map_err(|e| format!("Capture region lock error: {e}"))?
        .is_some();

    Ok(PipelineState {
        running,
        has_task,
        task_finished,
        has_capture_region,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            running: Arc::new(AtomicBool::new(false)),
            pipeline_handle: Mutex::new(None),
            capture_region: Arc::new(StdMutex::new(None)),
            snip_restore_bounds: Arc::new(StdMutex::new(None)),
            control_bounds: Arc::new(StdMutex::new(None)),
            language_pair: Arc::new(StdMutex::new("ja-en".to_string())),
            capture_scale: Arc::new(StdMutex::new(1.0)),
            ocr_backend: Arc::new(StdMutex::new("auto".to_string())),
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
            check_ocr_support,
            check_ocr_backend_support,
            set_ocr_backend,
            get_ocr_backend,
            get_ocr_diagnostics,
            get_pipeline_state,
            init_engine,
        ])
        .setup(|app| {
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.set_content_protected(true);
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

fn emit_click_through_state(app: &AppHandle, enabled: bool) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.emit("click-through-changed", enabled);
    }

    if let Some(control) = app.get_webview_window("control") {
        let _ = control.emit("click-through-changed", enabled);
    }
}

pub(crate) fn models_root_dir(app: &AppHandle) -> Result<PathBuf, String> {
    // Store models next to bundled resources so they live on the install disk chosen by the user.
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Could not determine resources directory: {e}"))?;

    Ok(resource_dir.join("models"))
}

fn safe_init_engine(model_dir: &std::path::Path) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(|| translate::init(model_dir))) {
        Ok(result) => result,
        Err(payload) => {
            let panic_message = if let Some(msg) = payload.downcast_ref::<&str>() {
                (*msg).to_string()
            } else if let Some(msg) = payload.downcast_ref::<String>() {
                msg.clone()
            } else {
                "non-string panic payload".to_string()
            };

            Err(format!(
                "Model initialization panicked: {panic_message}. Reinstall model files for this language."
            ))
        }
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

fn get_monitor_bounds_for_window(window: &WebviewWindow) -> Result<(i32, i32, u32, u32), String> {
    if let Some(monitor) = window
        .current_monitor()
        .map_err(|e| format!("current_monitor: {e}"))?
    {
        let position = monitor.position();
        let size = monitor.size();
        return Ok((position.x, position.y, size.width, size.height));
    }

    let (x, y, _, _) = get_window_bounds(window)?;
    get_monitor_bounds_at_point(x, y)
}

fn restore_window_bounds(
    window: &WebviewWindow,
    state: &tauri::State<'_, AppState>,
) -> Result<(), String> {
    let restore = state
        .snip_restore_bounds
        .lock()
        .map_err(|e| format!("Snip restore lock error: {e}"))?
        .take();

    if let Some(bounds) = restore {
        window
            .set_position(Position::Physical(PhysicalPosition::new(
                bounds.x, bounds.y,
            )))
            .map_err(|e| format!("restore set_position error: {e}"))?;

        window
            .set_size(Size::Physical(PhysicalSize::new(
                bounds.width,
                bounds.height,
            )))
            .map_err(|e| format!("restore set_size error: {e}"))?;
    }

    Ok(())
}
