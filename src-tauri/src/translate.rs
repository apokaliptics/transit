use ort::session::Session;
use ort::value::Value;
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
}

static ENGINE: OnceLock<RwLock<Option<Arc<TranslationEngine>>>> = OnceLock::new();

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

fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer, String> {
    match Tokenizer::from_file(tokenizer_path) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(primary_error) => {
            let primary_msg = primary_error.to_string();
            if !primary_msg.contains("Precompiled") || !primary_msg.contains("expected a borrowed string") {
                return Err(format!("Failed to load tokenizer: {primary_error}"));
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

            Tokenizer::from_bytes(&normalized)
                .map_err(|e| format!("Failed to load sanitized tokenizer fallback: {e}"))
        }
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

    let engine = Arc::new(TranslationEngine {
        encoder: Mutex::new(encoder),
        decoder: Mutex::new(decoder),
        tokenizer,
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
    let pad_token_id = engine
        .tokenizer
        .token_to_id("</s>")
        .unwrap_or(0) as i64;

    let eos_token_id = pad_token_id;

    let max_length = 512;
    let mut decoder_input_ids: Vec<i64> = vec![pad_token_id];

    for _ in 0..max_length {
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

        if best_id == eos_token_id {
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
