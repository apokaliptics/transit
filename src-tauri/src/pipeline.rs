use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tauri::{Emitter, WebviewWindow};

use crate::capture;
use crate::ocr;
use crate::translate;

fn cache_insert(
    cache: &mut HashMap<String, String>,
    order: &mut VecDeque<String>,
    key: String,
    value: String,
    capacity: usize,
) {
    if !cache.contains_key(&key) {
        order.push_back(key.clone());
    }
    cache.insert(key, value);

    while cache.len() > capacity {
        if let Some(oldest) = order.pop_front() {
            cache.remove(&oldest);
        } else {
            break;
        }
    }
}

fn make_line_id(source_text: &str, left: i32, top: i32, width: u32, height: u32) -> String {
    let mut hasher = DefaultHasher::new();
    source_text.trim().to_lowercase().hash(&mut hasher);
    (left / 4).hash(&mut hasher);
    (top / 4).hash(&mut hasher);
    (width / 4).hash(&mut hasher);
    (height / 4).hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn overlap_ratio(a: &ocr::OcrLine, b: &ocr::OcrLine) -> f32 {
    let ax1 = a.left;
    let ay1 = a.top;
    let ax2 = a.left + a.width as i32;
    let ay2 = a.top + a.height as i32;

    let bx1 = b.left;
    let by1 = b.top;
    let bx2 = b.left + b.width as i32;
    let by2 = b.top + b.height as i32;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    if ix1 >= ix2 || iy1 >= iy2 {
        return 0.0;
    }

    let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
    let a_area = ((ax2 - ax1).max(0) * (ay2 - ay1).max(0)) as f32;
    let b_area = ((bx2 - bx1).max(0) * (by2 - by1).max(0)) as f32;
    let union = a_area + b_area - inter;

    if union <= 0.0 { 0.0 } else { inter / union }
}

fn dedupe_lines(lines: Vec<ocr::OcrLine>) -> Vec<ocr::OcrLine> {
    let mut kept: Vec<ocr::OcrLine> = Vec::new();

    for line in lines {
        if line.text.trim().is_empty() {
            continue;
        }

        let mut is_duplicate = false;
        for existing in &mut kept {
            let same_text = existing.text.trim().eq_ignore_ascii_case(line.text.trim());
            let overlaps = overlap_ratio(existing, &line) > 0.6;

            if same_text && overlaps {
                let existing_area = existing.width.saturating_mul(existing.height);
                let line_area = line.width.saturating_mul(line.height);
                if line_area > existing_area {
                    *existing = line.clone();
                }
                is_duplicate = true;
                break;
            }
        }

        if !is_duplicate {
            kept.push(line);
        }
    }

    kept
}

#[derive(Clone, serde::Serialize)]
pub struct TranslatedLine {
    pub line_id: String,
    pub source_text: String,
    pub translated_text: String,
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
}

/// Payload sent to the frontend via Tauri events.
#[derive(Clone, serde::Serialize)]
pub struct TranslationPayload {
    pub frame_id: u64,
    pub text: String,
    pub source_text: String,
    pub lines: Vec<TranslatedLine>,
    pub capture_scale: f64,
    pub ocr_backend: String,
}

#[derive(Clone, serde::Serialize)]
pub struct OcrStatusPayload {
    pub state: String,
    pub language_pair: String,
    pub message: String,
    pub backend: String,
    pub language_tag: String,
    pub used_profile_fallback: bool,
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
    language_pair: Arc<Mutex<String>>,
    capture_scale: Arc<Mutex<f64>>,
    ocr_backend: Arc<Mutex<String>>,
) {
    log::info!("Translation pipeline started");

    let mut translation_cache: HashMap<String, String> = HashMap::new();
    let mut cache_order: VecDeque<String> = VecDeque::new();
    const CACHE_CAPACITY: usize = 512;
    let mut frame_id: u64 = 0;

    while running.load(Ordering::Relaxed) {
        let locked_region = match capture_region.lock() {
            Ok(guard) => *guard,
            Err(e) => {
                log::warn!("Capture region lock error: {e}");
                None
            }
        };

        // Translation is selection-driven; do not fallback to the overlay window bounds.
        let (x, y, width, height) = if let Some(region) = locked_region {
            (region.x, region.y, region.width, region.height)
        } else {
            tokio::time::sleep(std::time::Duration::from_millis(180)).await;
            continue;
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

        let current_language = match language_pair.lock() {
            Ok(guard) => guard.clone(),
            Err(e) => {
                log::warn!("Language lock error: {e}");
                "en".to_string()
            }
        };

        let backend_value = match ocr_backend.lock() {
            Ok(guard) => guard.clone(),
            Err(e) => {
                log::warn!("OCR backend lock error: {e}");
                "auto".to_string()
            }
        };
        let backend = ocr::OcrBackend::from_str(&backend_value);
        let backend_label = backend.as_str().to_string();

        // Step 2: Run OCR
        let ocr_language = current_language.clone();
        let ocr_backend_value = backend;
        let ocr_result = tokio::task::spawn_blocking(move || {
            ocr::recognize_text_with_backend(&image, &ocr_language, ocr_backend_value)
        })
        .await;

        let ocr_output = match ocr_result {
            Ok(Ok(result)) if !result.text.trim().is_empty() => result,
            Ok(Ok(_)) => {
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "no-text".to_string(),
                        language_pair: current_language.clone(),
                        message: "No readable text detected in selected region".to_string(),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                // Empty text, nothing to translate
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Ok(Err(e)) => {
                log::warn!("OCR error: {e}");
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "error".to_string(),
                        language_pair: current_language.clone(),
                        message: format!("OCR failed: {e}"),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Err(e) => {
                log::warn!("OCR task join error: {e}");
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "error".to_string(),
                        language_pair: current_language.clone(),
                        message: format!("OCR task failed: {e}"),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
        };

        let source_text = ocr_output.text.clone();
        let effective_language_tag = ocr_output.language_tag.clone();
        let used_profile_fallback = ocr_output.used_profile_fallback;

        log::info!("OCR detected: {}", &source_text);

        // Step 3: Translate per OCR line to keep location mapping accurate.
        let mut ordered_lines = dedupe_lines(ocr_output.lines.clone());
        ordered_lines.sort_by_key(|line| (line.top, line.left));

        let mut translated_lines = Vec::with_capacity(ordered_lines.len());
        for line in &ordered_lines {
            if line.text.trim().is_empty() {
                continue;
            }

            if line.width < 6 || line.height < 6 {
                continue;
            }

            let line_source = line.text.clone();
            let line_id = make_line_id(&line_source, line.left, line.top, line.width, line.height);

            if let Some(cached) = translation_cache.get(&line_source) {
                translated_lines.push(TranslatedLine {
                    line_id,
                    source_text: line_source.clone(),
                    translated_text: cached.clone(),
                    left: line.left,
                    top: line.top,
                    width: line.width,
                    height: line.height,
                });
                continue;
            }

            let source_for_task = line_source.clone();
            let line_translate_result = tokio::task::spawn_blocking(move || {
                translate::translate(&source_for_task)
            })
            .await;

            let line_translated = match line_translate_result {
                Ok(Ok(text)) if !text.trim().is_empty() => text,
                Ok(Ok(_)) => line_source.clone(),
                Ok(Err(e)) => {
                    log::warn!("Line translation error: {e}");
                    line_source.clone()
                }
                Err(e) => {
                    log::warn!("Line translation join error: {e}");
                    line_source.clone()
                }
            };

            cache_insert(
                &mut translation_cache,
                &mut cache_order,
                line_source.clone(),
                line_translated.clone(),
                CACHE_CAPACITY,
            );

            translated_lines.push(TranslatedLine {
                line_id,
                source_text: line_source,
                translated_text: line_translated,
                left: line.left,
                top: line.top,
                width: line.width,
                height: line.height,
            });
        }

        let translated = if translated_lines.is_empty() {
            source_text.clone()
        } else {
            translated_lines
                .iter()
                .map(|line| line.translated_text.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };

        log::info!("Translated: {}", &translated);

        let scale = match capture_scale.lock() {
            Ok(guard) => *guard,
            Err(e) => {
                log::warn!("Capture scale lock error: {e}");
                1.0
            }
        };

        // Step 4: Emit to frontend
        frame_id = frame_id.wrapping_add(1);
        let payload = TranslationPayload {
            frame_id,
            text: translated,
            source_text,
            lines: translated_lines,
            capture_scale: if scale.is_finite() && scale > 0.0 { scale } else { 1.0 },
            ocr_backend: backend_label.clone(),
        };

        if let Err(e) = window.emit("translation-update", &payload) {
            log::warn!("Failed to emit translation event: {e}");
        }

        let _ = window.emit(
            "ocr-status",
            OcrStatusPayload {
                state: if used_profile_fallback { "warning".to_string() } else { "ok".to_string() },
                language_pair: current_language,
                message: if used_profile_fallback {
                    "Text detected, but OCR language fallback was used".to_string()
                } else {
                    "Text detected".to_string()
                },
                backend: backend_label,
                language_tag: effective_language_tag,
                used_profile_fallback,
            },
        );

        tokio::time::sleep(std::time::Duration::from_millis(180)).await;
    }

    log::info!("Translation pipeline stopped");
}
