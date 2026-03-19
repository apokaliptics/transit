use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tokenizers::Tokenizer;

/// Holds the loaded translation model and tokenizer.
/// Sessions are wrapped in Mutex because `session.run()` requires `&mut self`.
struct TranslationEngine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    decoder_start_token_id: i64,
    eos_token_id: i64,
    max_length: usize,
    beam_size: usize,
    length_penalty: f32,
}

static ENGINE: OnceLock<RwLock<Option<Arc<TranslationEngine>>>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct GenerationSettings {
    decoder_start_token_id: i64,
    eos_token_id: i64,
    max_length: usize,
    beam_size: usize,
    length_penalty: f32,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    decoder_start_token_id: Option<i64>,
    eos_token_id: Option<i64>,
    forced_eos_token_id: Option<i64>,
    pad_token_id: Option<i64>,
    max_length: Option<usize>,
    num_beams: Option<usize>,
    length_penalty: Option<f32>,
}

#[derive(Debug, Clone)]
struct Beam {
    token_ids: Vec<i64>,
    log_prob: f32,
}

#[derive(Debug, Clone)]
struct CompletedBeam {
    token_ids: Vec<i64>,
    score: f32,
}

fn engine_slot() -> &'static RwLock<Option<Arc<TranslationEngine>>> {
    ENGINE.get_or_init(|| RwLock::new(None))
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn commit_session(path: &Path, label: &str) -> Result<Session, String> {
    match catch_unwind(AssertUnwindSafe(|| {
        Session::builder()
            .map_err(|e| format!("Session builder error: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| format!("Thread config error: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("Failed to load {label}: {e}"))
    })) {
        Ok(result) => result,
        Err(payload) => {
            let panic_msg = panic_payload_to_string(payload);
            Err(format!("{label} panicked during load: {panic_msg}"))
        }
    }
}

fn sanitize_precompiled_normalizers(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let is_precompiled = map
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t == "Precompiled")
                .unwrap_or(false);

            if is_precompiled {
                if let Some(field) = map.get_mut("precompiled_charsmap") {
                    if field.is_null() {
                        *field = serde_json::Value::String(String::new());
                    }
                }
                if let Some(field) = map.get_mut("precompiled") {
                    if field.is_null() {
                        *field = serde_json::Value::String(String::new());
                    }
                }
            }

            for child in map.values_mut() {
                sanitize_precompiled_normalizers(child);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                sanitize_precompiled_normalizers(item);
            }
        }
        _ => {}
    }
}

fn is_invalid_precompiled_normalizer(map: &serde_json::Map<String, serde_json::Value>) -> bool {
    map.get("type")
        .and_then(|value| value.as_str())
        .map(|kind| kind == "Precompiled")
        .unwrap_or(false)
        && map
            .get("precompiled_charsmap")
            .map(|value| value.is_null() || value.as_str() == Some(""))
            .unwrap_or(true)
}

fn strip_invalid_precompiled_normalizer_nodes(
    value: &mut serde_json::Value,
    stripped_any: &mut bool,
) -> bool {
    match value {
        serde_json::Value::Object(map) => {
            if is_invalid_precompiled_normalizer(map) {
                *stripped_any = true;
                return true;
            }

            for child in map.values_mut() {
                if strip_invalid_precompiled_normalizer_nodes(child, stripped_any) {
                    *child = serde_json::Value::Null;
                }
            }

            false
        }
        serde_json::Value::Array(items) => {
            let mut retained = Vec::with_capacity(items.len());
            for mut item in std::mem::take(items) {
                if !strip_invalid_precompiled_normalizer_nodes(&mut item, stripped_any) {
                    retained.push(item);
                }
            }
            *items = retained;
            false
        }
        _ => false,
    }
}

fn strip_invalid_precompiled_normalizers(value: &mut serde_json::Value) -> bool {
    let mut stripped_any = false;
    let _ = strip_invalid_precompiled_normalizer_nodes(value, &mut stripped_any);
    stripped_any
}

fn load_tokenizer_from_bytes(bytes: &[u8], label: &str) -> Result<Tokenizer, String> {
    match catch_unwind(AssertUnwindSafe(|| Tokenizer::from_bytes(bytes))) {
        Ok(Ok(tokenizer)) => Ok(tokenizer),
        Ok(Err(error)) => Err(format!("Failed to load {label}: {error}")),
        Err(payload) => {
            let panic_msg = panic_payload_to_string(payload);
            Err(format!("{label} panicked during load: {panic_msg}"))
        }
    }
}

fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer, String> {
    match catch_unwind(AssertUnwindSafe(|| Tokenizer::from_file(tokenizer_path))) {
        Ok(Ok(tokenizer)) => return Ok(tokenizer),
        Ok(Err(primary_error)) => {
            let primary_msg = primary_error.to_string();
            if !primary_msg.contains("Precompiled")
                || !primary_msg.contains("expected a borrowed string")
            {
                return Err(format!("Failed to load tokenizer: {primary_error}"));
            }
        }
        Err(payload) => {
            let panic_msg = panic_payload_to_string(payload);
            log::warn!("Tokenizer::from_file panicked; attempting sanitizer fallback: {panic_msg}");
        }
    }

    log::warn!(
        "Tokenizer deserialize failed with known Precompiled null-field issue, applying compatibility sanitizer"
    );

    let raw = fs::read_to_string(tokenizer_path)
        .map_err(|e| format!("Failed to read tokenizer for fallback: {e}"))?;

    let mut json: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("Failed to parse tokenizer JSON for fallback: {e}"))?;

    sanitize_precompiled_normalizers(&mut json);

    let normalized = serde_json::to_vec(&json)
        .map_err(|e| format!("Failed to serialize fallback tokenizer JSON: {e}"))?;

    match load_tokenizer_from_bytes(&normalized, "sanitized tokenizer fallback") {
        Ok(tokenizer) => Ok(tokenizer),
        Err(sanitized_error) => {
            if !strip_invalid_precompiled_normalizers(&mut json) {
                return Err(sanitized_error);
            }

            log::warn!(
                "Sanitized tokenizer still failed; stripping invalid Precompiled normalizer nodes with missing charsmap"
            );

            let stripped = serde_json::to_vec(&json)
                .map_err(|e| format!("Failed to serialize stripped tokenizer JSON: {e}"))?;

            load_tokenizer_from_bytes(&stripped, "stripped tokenizer fallback")
                .map_err(|stripped_error| format!("{sanitized_error}; {stripped_error}"))
        }
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<i64> {
    tokenizer.token_to_id(token).map(|id| id as i64)
}

fn fallback_generation_settings(tokenizer: &Tokenizer) -> Result<GenerationSettings, String> {
    let eos_token_id = token_id(tokenizer, "</s>")
        .or_else(|| token_id(tokenizer, "<eos>"))
        .ok_or_else(|| "Could not resolve EOS token id from tokenizer".to_string())?;

    let decoder_start_token_id = token_id(tokenizer, "<pad>")
        .or_else(|| token_id(tokenizer, "<s>"))
        .unwrap_or(eos_token_id);

    Ok(GenerationSettings {
        decoder_start_token_id,
        eos_token_id,
        max_length: 512,
        beam_size: 4,
        length_penalty: 1.0,
    })
}

fn load_generation_settings(
    model_dir: &Path,
    tokenizer: &Tokenizer,
) -> Result<GenerationSettings, String> {
    let fallback = fallback_generation_settings(tokenizer)?;
    let generation_config_path = model_dir.join("generation_config.json");

    if !generation_config_path.exists() {
        return Ok(fallback);
    }

    let raw = fs::read_to_string(&generation_config_path)
        .map_err(|e| format!("Failed to read generation_config.json: {e}"))?;
    let config: GenerationConfig = serde_json::from_str(&raw)
        .map_err(|e| format!("Failed to parse generation_config.json: {e}"))?;

    Ok(GenerationSettings {
        decoder_start_token_id: config
            .decoder_start_token_id
            .or(config.pad_token_id)
            .unwrap_or(fallback.decoder_start_token_id),
        eos_token_id: config
            .eos_token_id
            .or(config.forced_eos_token_id)
            .unwrap_or(fallback.eos_token_id),
        max_length: config.max_length.unwrap_or(fallback.max_length),
        beam_size: config.num_beams.unwrap_or(fallback.beam_size).max(1),
        length_penalty: config.length_penalty.unwrap_or(fallback.length_penalty),
    })
}

fn generated_token_count(token_ids: &[i64], eos_token_id: i64) -> usize {
    let mut count = token_ids.len().saturating_sub(1);
    if token_ids.last().copied() == Some(eos_token_id) {
        count = count.saturating_sub(1);
    }
    count.max(1)
}

fn length_penalized_score(
    log_prob: f32,
    token_ids: &[i64],
    eos_token_id: i64,
    length_penalty: f32,
) -> f32 {
    let length = generated_token_count(token_ids, eos_token_id) as f32;
    let penalty = ((5.0 + length) / 6.0).powf(length_penalty.max(0.0));
    log_prob / penalty
}

fn top_k_log_probs(logits: &[f32], k: usize) -> Vec<(i64, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logsumexp = max_logit
        + logits
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .sum::<f32>()
            .ln();

    let mut top: Vec<(i64, f32)> = Vec::with_capacity(k.min(logits.len()));
    for (token_id, &logit) in logits.iter().enumerate() {
        let log_prob = logit - logsumexp;
        if top.len() < top.capacity() {
            top.push((token_id as i64, log_prob));
            top.sort_by(|left, right| right.1.total_cmp(&left.1));
            continue;
        }

        if top
            .last()
            .map(|(_, score)| log_prob > *score)
            .unwrap_or(true)
        {
            top.pop();
            top.push((token_id as i64, log_prob));
            top.sort_by(|left, right| right.1.total_cmp(&left.1));
        }
    }

    top
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sanitize_precompiled_normalizers_replaces_null_fields() {
        let mut value = json!({
            "normalizer": {
                "type": "Precompiled",
                "precompiled_charsmap": null,
                "precompiled": null
            }
        });

        sanitize_precompiled_normalizers(&mut value);

        assert_eq!(value["normalizer"]["precompiled_charsmap"], json!(""));
        assert_eq!(value["normalizer"]["precompiled"], json!(""));
    }

    #[test]
    fn strip_invalid_precompiled_normalizers_drops_null_precompiled_nodes() {
        let mut value = json!({
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    { "type": "NFC" },
                    { "type": "Precompiled", "precompiled_charsmap": null }
                ]
            }
        });

        assert!(strip_invalid_precompiled_normalizers(&mut value));
        assert_eq!(
            value["normalizer"]["normalizers"],
            json!([{ "type": "NFC" }])
        );
    }

    #[test]
    fn top_k_log_probs_returns_descending_candidates() {
        let top = top_k_log_probs(&[0.0, 1.5, -2.0, 1.0], 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 3);
        assert!(top[0].1 >= top[1].1);
    }
}

/// Initialize the translation engine by loading model files from the given directory.
///
/// Expected files in `model_dir`:
/// - `encoder_model.onnx`
/// - `decoder_model.onnx` (or `decoder_model_merged.onnx`)
/// - `tokenizer.json`
pub fn init(model_dir: &Path) -> Result<(), String> {
    log::info!("Loading translation model from {:?}", model_dir);

    let encoder_path = model_dir.join("encoder_model.onnx");
    let merged_decoder_path = model_dir.join("decoder_model_merged.onnx");
    let plain_decoder_path = model_dir.join("decoder_model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !encoder_path.exists() {
        return Err(format!(
            "Encoder model not found: {}",
            encoder_path.display()
        ));
    }
    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer not found: {}", tokenizer_path.display()));
    }
    if !merged_decoder_path.exists() && !plain_decoder_path.exists() {
        return Err(
            "Decoder model not found (expected decoder_model_merged.onnx or decoder_model.onnx)"
                .to_string(),
        );
    }

    let encoder = commit_session(&encoder_path, "encoder model")?;

    // Prefer plain decoder for stability; use merged as fallback.
    let decoder = if plain_decoder_path.exists() {
        match commit_session(&plain_decoder_path, "decoder_model.onnx") {
            Ok(session) => session,
            Err(plain_error) => {
                if merged_decoder_path.exists() {
                    log::warn!(
                        "Plain decoder failed, trying merged decoder fallback: {plain_error}"
                    );
                    commit_session(&merged_decoder_path, "decoder_model_merged.onnx")?
                } else {
                    return Err(format!(
                        "Failed to load decoder_model.onnx and no merged fallback exists: {plain_error}"
                    ));
                }
            }
        }
    } else {
        commit_session(&merged_decoder_path, "decoder_model_merged.onnx")?
    };

    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let generation = load_generation_settings(model_dir, &tokenizer)?;

    log::info!(
        "Generation settings resolved: decoder_start_token_id={}, eos_token_id={}, max_length={}, beam_size={}, length_penalty={}",
        generation.decoder_start_token_id,
        generation.eos_token_id,
        generation.max_length,
        generation.beam_size,
        generation.length_penalty
    );

    let engine = Arc::new(TranslationEngine {
        encoder: Mutex::new(encoder),
        decoder: Mutex::new(decoder),
        tokenizer,
        decoder_start_token_id: generation.decoder_start_token_id,
        eos_token_id: generation.eos_token_id,
        max_length: generation.max_length,
        beam_size: generation.beam_size,
        length_penalty: generation.length_penalty,
    });

    let mut slot = engine_slot()
        .write()
        .map_err(|e| format!("Engine write lock error: {e}"))?;
    *slot = Some(engine);

    log::info!("Translation engine loaded successfully");
    Ok(())
}

/// Translate the given text. The engine must be initialized first via `init()`.
pub fn translate(text: &str) -> Result<String, String> {
    let engine = {
        let slot = engine_slot()
            .read()
            .map_err(|e| format!("Engine read lock error: {e}"))?;
        slot.as_ref()
            .cloned()
            .ok_or("Translation engine not initialized")?
    };

    if text.trim().is_empty() {
        return Ok(String::new());
    }

    let encoding = engine
        .tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization error: {e}"))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();

    let seq_len = input_ids.len();

    let input_ids_value = Value::from_array(
        ndarray::Array2::from_shape_vec((1, seq_len), input_ids.clone())
            .map_err(|e| format!("Shape error: {e}"))?,
    )
    .map_err(|e| format!("Value creation error: {e}"))?;

    let attention_mask_value = Value::from_array(
        ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone())
            .map_err(|e| format!("Shape error: {e}"))?,
    )
    .map_err(|e| format!("Value creation error: {e}"))?;

    // Keep the encoder session guard alive while decoder consumes encoder outputs.
    let mut encoder_session = engine
        .encoder
        .lock()
        .map_err(|e| format!("Encoder lock error: {e}"))?;

    let encoder_outputs = encoder_session
        .run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
        ])
        .map_err(|e| format!("Encoder run error: {e}"))?;

    let encoder_hidden = &encoder_outputs["last_hidden_state"];
    let beam_size = engine.beam_size.max(1);
    let mut active_beams = vec![Beam {
        token_ids: vec![engine.decoder_start_token_id],
        log_prob: 0.0,
    }];
    let mut completed_beams: Vec<CompletedBeam> = Vec::new();

    for _ in 0..engine.max_length {
        if active_beams.is_empty() {
            break;
        }

        let mut candidate_beams = Vec::with_capacity(active_beams.len().saturating_mul(beam_size));

        for beam in &active_beams {
            let dec_len = beam.token_ids.len();

            let decoder_input_value = Value::from_array(
                ndarray::Array2::from_shape_vec((1, dec_len), beam.token_ids.clone())
                    .map_err(|e| format!("Decoder shape error: {e}"))?,
            )
            .map_err(|e| format!("Value creation error: {e}"))?;

            let enc_attn_mask_value = Value::from_array(
                ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone())
                    .map_err(|e| format!("Shape error: {e}"))?,
            )
            .map_err(|e| format!("Value creation error: {e}"))?;

            let next_tokens = {
                let mut decoder_session = engine
                    .decoder
                    .lock()
                    .map_err(|e| format!("Decoder lock error: {e}"))?;

                let decoder_outputs = decoder_session
                    .run(ort::inputs![
                        "input_ids" => decoder_input_value,
                        "encoder_attention_mask" => enc_attn_mask_value,
                        "encoder_hidden_states" => encoder_hidden,
                    ])
                    .map_err(|e| format!("Decoder run error: {e}"))?;

                let logits_value = &decoder_outputs["logits"];
                let (logits_shape, logits_data) = logits_value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Extract logits error: {e}"))?;

                let shape_dims: &[i64] = &**logits_shape;
                if shape_dims.len() < 3 {
                    return Err(format!("Unexpected logits rank: {:?}", shape_dims));
                }

                let vocab_size = shape_dims[2] as usize;
                let logits_seq_len = shape_dims[1] as usize;
                if vocab_size == 0 || logits_seq_len == 0 {
                    return Err("Decoder returned empty logits tensor".to_string());
                }

                let last_pos = logits_seq_len - 1;
                let offset = last_pos * vocab_size;
                let last_logits = &logits_data[offset..offset + vocab_size];
                top_k_log_probs(last_logits, beam_size)
            };

            for (token_id, token_log_prob) in next_tokens {
                let mut next_ids = beam.token_ids.clone();
                next_ids.push(token_id);
                candidate_beams.push(Beam {
                    token_ids: next_ids,
                    log_prob: beam.log_prob + token_log_prob,
                });
            }
        }

        if candidate_beams.is_empty() {
            break;
        }

        candidate_beams.sort_by(|left, right| right.log_prob.total_cmp(&left.log_prob));

        let mut next_active_beams = Vec::with_capacity(beam_size);
        for candidate in candidate_beams {
            if candidate.token_ids.last().copied() == Some(engine.eos_token_id) {
                completed_beams.push(CompletedBeam {
                    score: length_penalized_score(
                        candidate.log_prob,
                        &candidate.token_ids,
                        engine.eos_token_id,
                        engine.length_penalty,
                    ),
                    token_ids: candidate.token_ids,
                });
            } else if next_active_beams.len() < beam_size {
                next_active_beams.push(candidate);
            }

            if next_active_beams.len() >= beam_size {
                break;
            }
        }

        active_beams = next_active_beams;
    }

    completed_beams.extend(active_beams.into_iter().map(|beam| CompletedBeam {
        score: length_penalized_score(
            beam.log_prob,
            &beam.token_ids,
            engine.eos_token_id,
            engine.length_penalty,
        ),
        token_ids: beam.token_ids,
    }));

    let best_beam = completed_beams
        .into_iter()
        .max_by(|left, right| left.score.total_cmp(&right.score))
        .ok_or_else(|| "Beam search produced no output".to_string())?;

    let output_ids: Vec<u32> = best_beam
        .token_ids
        .iter()
        .skip(1)
        .copied()
        .filter(|&id| id != engine.eos_token_id)
        .map(|id| id as u32)
        .collect();

    let decoded = engine
        .tokenizer
        .decode(&output_ids, true)
        .map_err(|e| format!("Decode error: {e}"))?;

    Ok(decoded)
}
