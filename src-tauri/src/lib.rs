mod capture;
mod downloader;
mod ocr;
mod pipeline;
mod translate;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Manager, WebviewWindow};
use tokio::sync::Mutex;

/// Shared application state.
struct AppState {
    running: Arc<AtomicBool>,
    pipeline_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

/// Start the translation pipeline.
#[tauri::command]
async fn start_translation(
    window: WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    // If already running, do nothing.
    if state.running.load(Ordering::Relaxed) {
        return Ok("Already running".to_string());
    }

    state.running.store(true, Ordering::Relaxed);

    // Reset the diff checker for a fresh session
    capture::reset_diff();

    let running = state.running.clone();

    let handle = tokio::spawn(async move {
        pipeline::run_pipeline(window, running).await;
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

/// Initialize the translation engine from the local data directory.
#[tauri::command]
fn init_engine(app: AppHandle, language_pair: String) -> Result<String, String> {
    let local_data_dir = app.path().app_local_data_dir()
        .map_err(|e| format!("Could not determine local data directory: {}", e))?;
    
    let model_dir = local_data_dir.join("models").join(&language_pair);

    if model_dir.exists() {
        translate::init(&model_dir)?;
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
        })
        .invoke_handler(tauri::generate_handler![
            start_translation,
            stop_translation,
            set_click_through,
            downloader::download_models,
            downloader::check_models,
            init_engine,
        ])
        .setup(|app| {
            // Check if default models (ja-en) exist and auto-init if they do
            if let Ok(local_data_dir) = app.path().app_local_data_dir() {
                let default_model_dir = local_data_dir.join("models").join("ja-en");
                if default_model_dir.exists() {
                    match translate::init(&default_model_dir) {
                        Ok(()) => log::info!("Auto-initialized ja-en translation engine"),
                        Err(e) => log::warn!("Failed auto-init: {e}"),
                    }
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
