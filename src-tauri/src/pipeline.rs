use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
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

#[derive(Clone, Debug)]
struct ParagraphBlock {
    lines: Vec<ocr::OcrLine>,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    total_height: u32,
    total_left: i64,
    total_center_x: i64,
}

impl ParagraphBlock {
    fn new(line: ocr::OcrLine) -> Self {
        let right = line.left + line.width as i32;
        let bottom = line.top + line.height as i32;
        let center_x = line.left + (line.width as i32 / 2);

        Self {
            lines: vec![line.clone()],
            left: line.left,
            top: line.top,
            right,
            bottom,
            total_height: line.height,
            total_left: line.left as i64,
            total_center_x: center_x as i64,
        }
    }

    fn push(&mut self, line: ocr::OcrLine) {
        let right = line.left + line.width as i32;
        let bottom = line.top + line.height as i32;
        let center_x = line.left + (line.width as i32 / 2);

        self.left = self.left.min(line.left);
        self.top = self.top.min(line.top);
        self.right = self.right.max(right);
        self.bottom = self.bottom.max(bottom);
        self.total_height = self.total_height.saturating_add(line.height);
        self.total_left += line.left as i64;
        self.total_center_x += center_x as i64;
        self.lines.push(line);
        self.lines.sort_by_key(|entry| (entry.top, entry.left));
    }

    fn width(&self) -> u32 {
        (self.right - self.left).max(0) as u32
    }

    fn height(&self) -> u32 {
        (self.bottom - self.top).max(0) as u32
    }

    fn average_height(&self) -> f32 {
        self.total_height as f32 / self.lines.len().max(1) as f32
    }

    fn average_left(&self) -> f32 {
        self.total_left as f32 / self.lines.len().max(1) as f32
    }

    fn average_center_x(&self) -> f32 {
        self.total_center_x as f32 / self.lines.len().max(1) as f32
    }

    fn last_line(&self) -> &ocr::OcrLine {
        self.lines
            .last()
            .expect("ParagraphBlock always contains at least one line")
    }

    fn source_text(&self) -> String {
        join_lines_into_paragraph(&self.lines)
    }
}

fn join_lines_into_paragraph(lines: &[ocr::OcrLine]) -> String {
    let mut paragraph = String::new();

    for line in lines {
        let segment = line.text.split_whitespace().collect::<Vec<_>>().join(" ");

        if segment.is_empty() {
            continue;
        }

        if paragraph.is_empty() {
            paragraph.push_str(&segment);
            continue;
        }

        let starts_with_punctuation = segment
            .chars()
            .next()
            .map(|ch| ",.;:!?)]}".contains(ch))
            .unwrap_or(false);

        if paragraph.ends_with('-')
            && segment
                .chars()
                .next()
                .map(|ch| ch.is_alphanumeric())
                .unwrap_or(false)
        {
            paragraph.pop();
            paragraph.push_str(&segment);
        } else if starts_with_punctuation {
            paragraph.push_str(&segment);
        } else {
            paragraph.push(' ');
            paragraph.push_str(&segment);
        }
    }

    paragraph
}

fn horizontal_overlap_ratio(block: &ParagraphBlock, line: &ocr::OcrLine) -> f32 {
    let line_right = line.left + line.width as i32;
    let overlap = (block.right.min(line_right) - block.left.max(line.left)).max(0) as f32;
    let reference_width = block.width().min(line.width).max(1) as f32;
    overlap / reference_width
}

fn block_match_score(block: &ParagraphBlock, line: &ocr::OcrLine) -> Option<f32> {
    let last_line = block.last_line();
    let last_bottom = last_line.top + last_line.height as i32;
    let avg_height = block.average_height().max(line.height as f32);
    let vertical_gap = line.top - last_bottom;
    let max_vertical_gap = ((avg_height * 1.1).round() as i32).max(14);

    if vertical_gap > max_vertical_gap {
        return None;
    }

    if vertical_gap < -((avg_height * 0.75).round() as i32) {
        return None;
    }

    let overlap_ratio = horizontal_overlap_ratio(block, line);
    let left_diff = (line.left as f32 - block.average_left()).abs();
    let line_center_x = line.left as f32 + (line.width as f32 / 2.0);
    let center_diff = (line_center_x - block.average_center_x()).abs();
    let max_left_diff = ((avg_height * 2.5).round() as i32).max(28);
    let max_center_diff =
        ((block.width().max(line.width) as f32 * 0.45).round() as i32).max(max_left_diff);

    let horizontally_aligned = overlap_ratio >= 0.2
        || left_diff <= max_left_diff as f32
        || center_diff <= max_center_diff as f32;

    if !horizontally_aligned {
        return None;
    }

    let same_row_offset = (line.top - last_line.top).abs();
    if same_row_offset <= ((avg_height * 0.35).round() as i32)
        && overlap_ratio < 0.15
        && left_diff > max_left_diff as f32
    {
        return None;
    }

    let gap_penalty = vertical_gap.max(0) as f32 / (max_vertical_gap as f32 + 1.0);
    let left_penalty = left_diff / (max_left_diff as f32 + 1.0);
    let center_penalty = center_diff / (max_center_diff as f32 + 1.0);

    Some(overlap_ratio * 2.0 + (1.0 - gap_penalty) - left_penalty * 0.7 - center_penalty * 0.5)
}

fn group_lines_into_blocks(mut lines: Vec<ocr::OcrLine>) -> Vec<ParagraphBlock> {
    lines.retain(|line| !line.text.trim().is_empty() && line.width >= 6 && line.height >= 6);
    lines.sort_by_key(|line| (line.top, line.left));

    let mut blocks: Vec<ParagraphBlock> = Vec::new();

    for line in lines {
        let best_match = blocks
            .iter()
            .enumerate()
            .filter_map(|(idx, block)| block_match_score(block, &line).map(|score| (idx, score)))
            .max_by(|(_, left_score), (_, right_score)| left_score.total_cmp(right_score))
            .map(|(idx, _)| idx);

        if let Some(idx) = best_match {
            blocks[idx].push(line);
        } else {
            blocks.push(ParagraphBlock::new(line));
        }
    }

    blocks.sort_by_key(|block| (block.top, block.left));
    blocks
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
        let capture_result =
            tokio::task::spawn_blocking(move || capture::capture_region(x, y, width, height)).await;

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

        let ocr_source_text = ocr_output.text.clone();
        let effective_language_tag = ocr_output.language_tag.clone();
        let used_profile_fallback = ocr_output.used_profile_fallback;

        log::info!("OCR detected: {}", &ocr_source_text);

        // Step 3: Group OCR lines into paragraph-like blocks before translation.
        let paragraph_blocks = group_lines_into_blocks(dedupe_lines(ocr_output.lines.clone()));

        let mut translated_lines = Vec::with_capacity(paragraph_blocks.len());
        for block in &paragraph_blocks {
            let block_source = block.source_text();
            if block_source.trim().is_empty() {
                continue;
            }

            let block_width = block.width();
            let block_height = block.height();
            let line_id = make_line_id(
                &block_source,
                block.left,
                block.top,
                block_width,
                block_height,
            );

            if let Some(cached) = translation_cache.get(&block_source) {
                translated_lines.push(TranslatedLine {
                    line_id,
                    source_text: block_source.clone(),
                    translated_text: cached.clone(),
                    left: block.left,
                    top: block.top,
                    width: block_width,
                    height: block_height,
                });
                continue;
            }

            let source_for_task = block_source.clone();
            let block_translate_result =
                tokio::task::spawn_blocking(move || translate::translate(&source_for_task)).await;

            let block_translated = match block_translate_result {
                Ok(Ok(text)) if !text.trim().is_empty() => text,
                Ok(Ok(_)) => block_source.clone(),
                Ok(Err(e)) => {
                    log::warn!("Block translation error: {e}");
                    block_source.clone()
                }
                Err(e) => {
                    log::warn!("Block translation join error: {e}");
                    block_source.clone()
                }
            };

            cache_insert(
                &mut translation_cache,
                &mut cache_order,
                block_source.clone(),
                block_translated.clone(),
                CACHE_CAPACITY,
            );

            translated_lines.push(TranslatedLine {
                line_id,
                source_text: block_source,
                translated_text: block_translated,
                left: block.left,
                top: block.top,
                width: block_width,
                height: block_height,
            });
        }

        let source_text = if translated_lines.is_empty() {
            ocr_source_text.clone()
        } else {
            translated_lines
                .iter()
                .map(|line| line.source_text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        let translated = if translated_lines.is_empty() {
            ocr_source_text.clone()
        } else {
            translated_lines
                .iter()
                .map(|line| line.translated_text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
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
            capture_scale: if scale.is_finite() && scale > 0.0 {
                scale
            } else {
                1.0
            },
            ocr_backend: backend_label.clone(),
        };

        if let Err(e) = window.emit("translation-update", &payload) {
            log::warn!("Failed to emit translation event: {e}");
        }

        let _ = window.emit(
            "ocr-status",
            OcrStatusPayload {
                state: if used_profile_fallback {
                    "warning".to_string()
                } else {
                    "ok".to_string()
                },
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
