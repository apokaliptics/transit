use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::panic::{catch_unwind, AssertUnwindSafe};
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
}

static ENGINE: OnceLock<RwLock<Option<Arc<TranslationEngine>>>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct GenerationSettings {
    decoder_start_token_id: i64,
    eos_token_id: i64,
    max_length: usize,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    decoder_start_token_id: Option<i64>,
    eos_token_id: Option<i64>,
    forced_eos_token_id: Option<i64>,
    pad_token_id: Option<i64>,
    max_length: Option<usize>,
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

fn is_invalid_precompiled_normalizer(
    map: &serde_json::Map<String, serde_json::Value>,
) -> bool {
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
    match catch_unwind(AssertUnwindSafe(|| {
        Tokenizer::from_file(tokenizer_path)
    })) {
        Ok(Ok(tokenizer)) => return Ok(tokenizer),
        Ok(Err(primary_error)) => {
            let primary_msg = primary_error.to_string();
            if !primary_msg.contains("Precompiled") || !primary_msg.contains("expected a borrowed string") {
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

            load_tokenizer_from_bytes(&stripped, "stripped tokenizer fallback").map_err(
                |stripped_error| format!("{sanitized_error}; {stripped_error}"),
            )
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
    })
}

fn load_generation_settings(model_dir: &Path, tokenizer: &Tokenizer) -> Result<GenerationSettings, String> {
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
    })
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
        return Err(format!("Encoder model not found: {}", encoder_path.display()));
    }
    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer not found: {}", tokenizer_path.display()));
    }
    if !merged_decoder_path.exists() && !plain_decoder_path.exists() {
        return Err("Decoder model not found (expected decoder_model_merged.onnx or decoder_model.onnx)".to_string());
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
        "Generation settings resolved: decoder_start_token_id={}, eos_token_id={}, max_length={}",
        generation.decoder_start_token_id,
        generation.eos_token_id,
        generation.max_length
    );

    let engine = Arc::new(TranslationEngine {
        encoder: Mutex::new(encoder),
        decoder: Mutex::new(decoder),
        tokenizer,
        decoder_start_token_id: generation.decoder_start_token_id,
        eos_token_id: generation.eos_token_id,
        max_length: generation.max_length,
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
        slot
            .as_ref()
            .cloned()
            .ok_or("Translation engine not initialized")?
    };

    if text.trim().is_empty() {
        return Ok(String::new());
    }

    // Tokenize input
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

    // Create input Value tensors for encoder
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

    // Run encoder — keep the lock guard alive as long as encoder_outputs exists
    let mut encoder_session = engine.encoder.lock()
        .map_err(|e| format!("Encoder lock error: {e}"))?;

    let encoder_outputs = encoder_session
        .run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
        ])
        .map_err(|e| format!("Encoder run error: {e}"))?;

    // Autoregressive decoding
    let mut decoder_input_ids: Vec<i64> = vec![engine.decoder_start_token_id];

    for _ in 0..engine.max_length {
        let dec_len = decoder_input_ids.len();

        let decoder_input_value = Value::from_array(
            ndarray::Array2::from_shape_vec((1, dec_len), decoder_input_ids.clone())
                .map_err(|e| format!("Decoder shape error: {e}"))?,
        )
        .map_err(|e| format!("Value creation error: {e}"))?;

        let enc_attn_mask_value = Value::from_array(
            ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone())
                .map_err(|e| format!("Shape error: {e}"))?,
        )
        .map_err(|e| format!("Value creation error: {e}"))?;

        // Extract encoder hidden states and pass them to the decoder
        let encoder_hidden = &encoder_outputs["last_hidden_state"];

        // Run decoder — extract best_id within this block so outputs can be dropped
        let best_id = {
            let mut decoder_session = engine.decoder.lock()
                .map_err(|e| format!("Decoder lock error: {e}"))?;

            let decoder_outputs = decoder_session
                .run(ort::inputs![
                    "input_ids" => decoder_input_value,
                    "encoder_attention_mask" => enc_attn_mask_value,
                    "encoder_hidden_states" => encoder_hidden,
                ])
                .map_err(|e| format!("Decoder run error: {e}"))?;

            // Get logits for the last token — try_extract_tensor returns (&Shape, &[f32])
            let logits_value = &decoder_outputs["logits"];
            let (logits_shape, logits_data) = logits_value
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract logits error: {e}"))?;

            // Shape is [batch=1, seq_len, vocab_size]
            let shape_dims: &[i64] = &**logits_shape;
            let vocab_size = shape_dims[2] as usize;
            let logits_seq_len = shape_dims[1] as usize;

            // Get logits for the last position — greedy decoding
            let last_pos = logits_seq_len - 1;
            let offset = last_pos * vocab_size;
            let mut best_id = 0i64;
            let mut best_score = f32::NEG_INFINITY;

            for v in 0..vocab_size {
                let score = logits_data[offset + v];
                if score > best_score {
                    best_score = score;
                    best_id = v as i64;
                }
            }

            best_id
        }; // decoder_session and decoder_outputs dropped here

        if best_id == engine.eos_token_id {
            break;
        }

        decoder_input_ids.push(best_id);
    }

    // Decode output token IDs (skip the initial pad token)
    let output_ids: Vec<u32> = decoder_input_ids[1..]
        .iter()
        .map(|&id| id as u32)
        .collect();

    let decoded = engine
        .tokenizer
        .decode(&output_ids, true)
        .map_err(|e| format!("Decode error: {e}"))?;

    Ok(decoded)
}
