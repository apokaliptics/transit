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
}

#[derive(Clone, Debug)]
struct TranslationBlock {
    source_text: String,
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

fn is_cjk_text_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{3040}'..='\u{30ff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{f900}'..='\u{faff}'
            | '\u{ac00}'..='\u{d7af}'
    )
}

fn is_cjk_open_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '（' | '《' | '「' | '『' | '【' | '〔' | '〈' | '“' | '‘'
    )
}

fn is_cjk_close_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '）' | '》'
            | '」'
            | '』'
            | '】'
            | '〕'
            | '〉'
            | '”'
            | '’'
            | '。'
            | '，'
            | '、'
            | '！'
            | '？'
            | '；'
            | '：'
            | '…'
    )
}

fn should_elide_ocr_space(prev: char, next: char) -> bool {
    ((is_cjk_text_char(prev) || is_cjk_close_punctuation(prev))
        && (is_cjk_text_char(next)
            || is_cjk_open_punctuation(next)
            || is_cjk_close_punctuation(next)))
        || (is_cjk_open_punctuation(prev)
            && (is_cjk_text_char(next) || is_cjk_open_punctuation(next)))
}

fn normalize_translation_source(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut normalized = String::with_capacity(text.len());
    let mut last_emitted: Option<char> = None;
    let mut idx = 0usize;

    while idx < chars.len() {
        let ch = chars[idx];
        if ch.is_whitespace() {
            let mut next_idx = idx + 1;
            while next_idx < chars.len() && chars[next_idx].is_whitespace() {
                next_idx += 1;
            }

            if let (Some(prev), Some(&next)) = (last_emitted, chars.get(next_idx)) {
                if should_elide_ocr_space(prev, next) {
                    idx = next_idx;
                    continue;
                }

                if prev != ' ' {
                    normalized.push(' ');
                    last_emitted = Some(' ');
                }
            }

            idx = next_idx;
            continue;
        }

        normalized.push(ch);
        last_emitted = Some(ch);
        idx += 1;
    }

    normalized.trim().to_string()
}

fn is_probably_english_text(text: &str) -> bool {
    let mut ascii_alpha = 0usize;
    let mut cjk_chars = 0usize;
    let mut meaningful_chars = 0usize;

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() {
            ascii_alpha += 1;
            meaningful_chars += 1;
        } else if is_cjk_text_char(ch) {
            cjk_chars += 1;
            meaningful_chars += 1;
        } else if ch.is_ascii_digit() {
            meaningful_chars += 1;
        }
    }

    ascii_alpha >= 4
        && cjk_chars == 0
        && meaningful_chars > 0
        && ascii_alpha.saturating_mul(100) >= meaningful_chars.saturating_mul(65)
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

    normalize_translation_source(&paragraph)
}

fn normalize_ocr_text(text: &str) -> String {
    let joined = text
        .lines()
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    normalize_translation_source(&joined)
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

fn build_translation_blocks(
    paragraph_blocks: &[ParagraphBlock],
    all_lines: &[ocr::OcrLine],
    ocr_source_text: &str,
    capture_width: u32,
    capture_height: u32,
) -> Vec<TranslationBlock> {
    let line_blocks: Vec<TranslationBlock> = all_lines
        .iter()
        .filter_map(|line| translation_block_from_lines(std::slice::from_ref(line)))
        .collect();

    if !line_blocks.is_empty() {
        return line_blocks;
    }

    let mut blocks: Vec<TranslationBlock> = paragraph_blocks
        .iter()
        .filter_map(|block| translation_block_from_lines(&block.lines))
        .collect();

    if !blocks.is_empty() {
        return blocks;
    }

    let fallback_source = normalize_ocr_text(ocr_source_text);
    if fallback_source.is_empty() {
        return blocks;
    }

    let geometry_lines: Vec<&ocr::OcrLine> = all_lines
        .iter()
        .filter(|line| !line.text.trim().is_empty() && line.width > 0 && line.height > 0)
        .collect();

    let (left, top, width, height) = if geometry_lines.is_empty() {
        (0, 0, capture_width.max(8), capture_height.max(16))
    } else {
        let left = geometry_lines
            .iter()
            .map(|line| line.left)
            .min()
            .unwrap_or(0)
            .max(0);
        let top = geometry_lines
            .iter()
            .map(|line| line.top)
            .min()
            .unwrap_or(0)
            .max(0);
        let right = geometry_lines
            .iter()
            .map(|line| line.left + line.width as i32)
            .max()
            .unwrap_or(capture_width as i32);
        let bottom = geometry_lines
            .iter()
            .map(|line| line.top + line.height as i32)
            .max()
            .unwrap_or(capture_height as i32);

        (
            left,
            top,
            (right - left).max(8) as u32,
            (bottom - top).max(16) as u32,
        )
    };

    blocks.push(TranslationBlock {
        source_text: fallback_source,
        left,
        top,
        width,
        height,
    });

    blocks
}

fn translation_block_from_lines(lines: &[ocr::OcrLine]) -> Option<TranslationBlock> {
    if lines.is_empty() {
        return None;
    }

    let source_text = join_lines_into_paragraph(lines);
    if source_text.trim().is_empty() {
        return None;
    }

    let left = lines.iter().map(|line| line.left).min().unwrap_or(0).max(0);
    let top = lines.iter().map(|line| line.top).min().unwrap_or(0).max(0);
    let right = lines
        .iter()
        .map(|line| line.left + line.width as i32)
        .max()
        .unwrap_or(left);
    let bottom = lines
        .iter()
        .map(|line| line.top + line.height as i32)
        .max()
        .unwrap_or(top);

    Some(TranslationBlock {
        source_text,
        left,
        top,
        width: (right - left).max(8) as u32,
        height: (bottom - top).max(16) as u32,
    })
}

fn build_payload_text(
    translated_lines: &[TranslatedLine],
    fallback_source_text: &str,
) -> (String, String) {
    if translated_lines.is_empty() {
        return (
            fallback_source_text.to_string(),
            fallback_source_text.to_string(),
        );
    }

    let source_text = translated_lines
        .iter()
        .map(|line| line.source_text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let translated_text = translated_lines
        .iter()
        .map(|line| line.translated_text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    (source_text, translated_text)
}

fn build_translation_payload(
    frame_id: u64,
    translated_lines: &[TranslatedLine],
    fallback_source_text: &str,
    capture_scale: f64,
    backend_label: &str,
) -> TranslationPayload {
    let (source_text, translated_text) = build_payload_text(translated_lines, fallback_source_text);

    TranslationPayload {
        frame_id,
        text: translated_text,
        source_text,
        lines: translated_lines.to_vec(),
        capture_scale: if capture_scale.is_finite() && capture_scale > 0.0 {
            capture_scale
        } else {
            1.0
        },
        ocr_backend: backend_label.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_translation_source_collapses_cjk_spacing() {
        let text = "我 最 后 一 次 见 到 他 是 在 海 滩 。";
        assert_eq!(
            normalize_translation_source(text),
            "我最后一次见到他是在海滩。"
        );
    }

    #[test]
    fn normalize_translation_source_preserves_latin_spacing() {
        let text = "The Sound Of.. 我 最 后";
        assert_eq!(normalize_translation_source(text), "The Sound Of.. 我最后");
    }

    #[test]
    fn english_detection_ignores_ascii_sentences() {
        assert!(is_probably_english_text("The Sound of Metal"));
        assert!(!is_probably_english_text("最后一次见到他"));
        assert!(!is_probably_english_text("Chapter 7 最后的海滩"));
    }
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
            let _ = window.emit(
                "ocr-status",
                OcrStatusPayload {
                    state: "warning".to_string(),
                    language_pair: "".to_string(),
                    message: "Capture region is not set yet".to_string(),
                    backend: "auto".to_string(),
                    language_tag: "".to_string(),
                    used_profile_fallback: false,
                },
            );
            tokio::time::sleep(std::time::Duration::from_millis(180)).await;
            continue;
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

        // Step 1: Capture the screen region and check for changes
        let capture_result =
            tokio::task::spawn_blocking(move || capture::capture_region(x, y, width, height)).await;

        let image = match capture_result {
            Ok(Ok(Some(img))) => img,
            Ok(Ok(None)) => {
                // No change detected, skip
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "warning".to_string(),
                        language_pair: current_language.clone(),
                        message: "No visual change detected in selected region".to_string(),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Ok(Err(e)) => {
                log::warn!("Capture error: {e}");
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "error".to_string(),
                        language_pair: current_language.clone(),
                        message: format!("Capture failed: {e}"),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
            Err(e) => {
                log::warn!("Capture task join error: {e}");
                let _ = window.emit(
                    "ocr-status",
                    OcrStatusPayload {
                        state: "error".to_string(),
                        language_pair: current_language.clone(),
                        message: format!("Capture task failed: {e}"),
                        backend: backend_label.clone(),
                        language_tag: "".to_string(),
                        used_profile_fallback: false,
                    },
                );
                tokio::time::sleep(std::time::Duration::from_millis(180)).await;
                continue;
            }
        };

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
        let deduped_lines = dedupe_lines(ocr_output.lines.clone());
        let paragraph_blocks = group_lines_into_blocks(deduped_lines.clone());
        let translation_blocks = build_translation_blocks(
            &paragraph_blocks,
            &deduped_lines,
            &ocr_source_text,
            width,
            height,
        );

        let scale = match capture_scale.lock() {
            Ok(guard) => *guard,
            Err(e) => {
                log::warn!("Capture scale lock error: {e}");
                1.0
            }
        };
        let next_frame_id = frame_id.wrapping_add(1);

        let mut translated_lines = Vec::with_capacity(translation_blocks.len());
        let mut translation_warning: Option<String> = None;
        let mut ignored_english_blocks = 0usize;
        for block in translation_blocks {
            let block_source = block.source_text;
            if is_probably_english_text(&block_source) {
                ignored_english_blocks += 1;
                continue;
            }

            let line_id = make_line_id(
                &block_source,
                block.left,
                block.top,
                block.width,
                block.height,
            );

            if let Some(cached) = translation_cache.get(&block_source) {
                translated_lines.push(TranslatedLine {
                    line_id,
                    source_text: block_source.clone(),
                    translated_text: cached.clone(),
                    left: block.left,
                    top: block.top,
                    width: block.width,
                    height: block.height,
                });
                continue;
            }

            let source_for_task = block_source.clone();
            let block_translate_result =
                tokio::task::spawn_blocking(move || translate::translate(&source_for_task)).await;

            let (block_translated, should_cache) = match block_translate_result {
                Ok(Ok(text)) if !text.trim().is_empty() => (text, true),
                Ok(Ok(_)) => {
                    translation_warning =
                        Some("Text detected, but translation returned an empty result".to_string());
                    (block_source.clone(), false)
                }
                Ok(Err(e)) => {
                    log::warn!("Block translation error: {e}");
                    translation_warning =
                        Some(format!("Text detected, but translation failed: {e}"));
                    (block_source.clone(), false)
                }
                Err(e) => {
                    log::warn!("Block translation join error: {e}");
                    translation_warning =
                        Some(format!("Text detected, but translation task failed: {e}"));
                    (block_source.clone(), false)
                }
            };

            if should_cache {
                cache_insert(
                    &mut translation_cache,
                    &mut cache_order,
                    block_source.clone(),
                    block_translated.clone(),
                    CACHE_CAPACITY,
                );
            }

            translated_lines.push(TranslatedLine {
                line_id,
                source_text: block_source,
                translated_text: block_translated,
                left: block.left,
                top: block.top,
                width: block.width,
                height: block.height,
            });

            let payload = build_translation_payload(
                next_frame_id,
                &translated_lines,
                &ocr_source_text,
                scale,
                &backend_label,
            );

            if let Err(e) = window.emit("translation-update", &payload) {
                log::warn!("Failed to emit translation event: {e}");
            }
        }

        let (_source_text, translated) = build_payload_text(&translated_lines, &ocr_source_text);

        log::info!("Translated: {}", &translated);

        // Step 4: Emit to frontend
        frame_id = next_frame_id;
        let payload = if translated_lines.is_empty() && ignored_english_blocks > 0 {
            TranslationPayload {
                frame_id,
                text: String::new(),
                source_text: String::new(),
                lines: Vec::new(),
                capture_scale: if scale.is_finite() && scale > 0.0 {
                    scale
                } else {
                    1.0
                },
                ocr_backend: backend_label.clone(),
            }
        } else {
            build_translation_payload(
                frame_id,
                &translated_lines,
                &ocr_source_text,
                scale,
                &backend_label,
            )
        };

        if let Err(e) = window.emit("translation-update", &payload) {
            log::warn!("Failed to emit translation event: {e}");
        }

        let _ = window.emit(
            "ocr-status",
            OcrStatusPayload {
                state: if translated_lines.is_empty() && ignored_english_blocks > 0 {
                    "ignored".to_string()
                } else if used_profile_fallback || translation_warning.is_some() {
                    "warning".to_string()
                } else {
                    "ok".to_string()
                },
                language_pair: current_language,
                message: if translated_lines.is_empty() && ignored_english_blocks > 0 {
                    "English text detected; skipping translation".to_string()
                } else {
                    match (used_profile_fallback, translation_warning) {
                    (true, Some(translation_warning)) => {
                        format!(
                            "Text detected, but OCR language fallback was used. {translation_warning}"
                        )
                    }
                    (true, None) => "Text detected, but OCR language fallback was used".to_string(),
                    (false, Some(translation_warning)) => translation_warning,
                    (false, None) => "Text detected".to_string(),
                    }
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
