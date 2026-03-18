use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, WebviewWindow};

use crate::capture;
use crate::ocr;
use crate::translate;

/// Payload sent to the frontend via Tauri events.
#[derive(Clone, serde::Serialize)]
pub struct TranslationPayload {
    pub text: String,
    pub source_text: String,
}

/// Region locked by the user for focused capture.
#[derive(Clone, Copy, Debug)]
pub struct CaptureRegion {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Runs the capture → OCR → translate → emit loop.
///
/// The loop runs every ~400ms and only processes frames where the content changed.
/// It stops when `running` is set to `false`.
pub async fn run_pipeline(
    window: WebviewWindow,
    running: Arc<AtomicBool>,
    capture_region: Arc<Mutex<Option<CaptureRegion>>>,
) {
    log::info!("Translation pipeline started");

    while running.load(Ordering::Relaxed) {
        let locked_region = match capture_region.lock() {
            Ok(guard) => *guard,
            Err(e) => {
                log::warn!("Capture region lock error: {e}");
                None
            }
        };

        // Use the selected snip region when available; otherwise fallback to window bounds.
        let (x, y, width, height) = if let Some(region) = locked_region {
            (region.x, region.y, region.width, region.height)
        } else {
            match get_window_bounds(&window) {
                Ok(bounds) => bounds,
                Err(e) => {
                    log::warn!("Failed to get window bounds: {e}");
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    continue;
                }
            }
        };

        // Step 1: Capture the screen region and check for changes
        let capture_result = tokio::task::spawn_blocking(move || {
            capture::capture_region(x, y, width, height)
        })
        .await;

        let image = match capture_result {
            Ok(Ok(Some(img))) => img,
            Ok(Ok(None)) => {
                // No change detected, skip
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Ok(Err(e)) => {
                log::warn!("Capture error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Err(e) => {
                log::warn!("Capture task join error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
        };

        // Step 2: Run OCR
        let ocr_result = tokio::task::spawn_blocking(move || {
            ocr::recognize_text(&image)
        })
        .await;

        let source_text = match ocr_result {
            Ok(Ok(text)) if !text.trim().is_empty() => text,
            Ok(Ok(_)) => {
                // Empty text, nothing to translate
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Ok(Err(e)) => {
                log::warn!("OCR error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Err(e) => {
                log::warn!("OCR task join error: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
        };

        log::info!("OCR detected: {}", &source_text);

        // Step 3: Translate
        let source_clone = source_text.clone();
        let translate_result = tokio::task::spawn_blocking(move || {
            translate::translate(&source_clone)
        })
        .await;

        let translated = match translate_result {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => {
                log::warn!("Translation error: {e}");
                // Still emit the source text so the user sees something
                source_text.clone()
            }
            Err(e) => {
                log::warn!("Translation task join error: {e}");
                source_text.clone()
            }
        };

        log::info!("Translated: {}", &translated);

        // Step 4: Emit to frontend
        let payload = TranslationPayload {
            text: translated,
            source_text,
        };

        if let Err(e) = window.emit("translation-update", &payload) {
            log::warn!("Failed to emit translation event: {e}");
        }

        tokio::time::sleep(std::time::Duration::from_millis(180)).await;
    }

    log::info!("Translation pipeline stopped");
}

/// Gets the window outer position and size as (x, y, width, height).
fn get_window_bounds(window: &WebviewWindow) -> Result<(i32, i32, u32, u32), String> {
    let position = window
        .outer_position()
        .map_err(|e| format!("outer_position: {e}"))?;

    let size = window
        .outer_size()
        .map_err(|e| format!("outer_size: {e}"))?;

    Ok((
        position.x,
        position.y,
        size.width,
        size.height,
    ))
}
