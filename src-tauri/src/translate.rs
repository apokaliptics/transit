use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

/// Holds the loaded translation model and tokenizer.
/// Sessions are wrapped in Mutex because `session.run()` requires `&mut self`.
struct TranslationEngine {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
}

static ENGINE: OnceLock<Result<TranslationEngine, String>> = OnceLock::new();

/// Initialize the translation engine by loading model files from the given directory.
///
/// Expected files in `model_dir`:
/// - `encoder_model.onnx`
/// - `decoder_model.onnx` (or `decoder_model_merged.onnx`)
/// - `tokenizer.json`
pub fn init(model_dir: &Path) -> Result<(), String> {
    ENGINE.get_or_init(|| {
        log::info!("Loading translation model from {:?}", model_dir);

        let encoder_path = model_dir.join("encoder_model.onnx");
        let decoder_path = if model_dir.join("decoder_model_merged.onnx").exists() {
            model_dir.join("decoder_model_merged.onnx")
        } else {
            model_dir.join("decoder_model.onnx")
        };
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !encoder_path.exists() {
            return Err(format!("Encoder model not found: {}", encoder_path.display()));
        }
        if !decoder_path.exists() {
            return Err(format!("Decoder model not found: {}", decoder_path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(format!("Tokenizer not found: {}", tokenizer_path.display()));
        }

        let encoder = Session::builder()
            .map_err(|e| format!("Session builder error: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| format!("Thread config error: {e}"))?
            .commit_from_file(&encoder_path)
            .map_err(|e| format!("Failed to load encoder: {e}"))?;

        let decoder = Session::builder()
            .map_err(|e| format!("Session builder error: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| format!("Thread config error: {e}"))?
            .commit_from_file(&decoder_path)
            .map_err(|e| format!("Failed to load decoder: {e}"))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        log::info!("Translation engine loaded successfully");

        Ok(TranslationEngine {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
        })
    });

    // Check if initialization succeeded
    match ENGINE.get() {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e.clone()),
        None => Err("Engine not initialized".to_string()),
    }
}

/// Translate the given text. The engine must be initialized first via `init()`.
pub fn translate(text: &str) -> Result<String, String> {
    let engine = ENGINE
        .get()
        .ok_or("Translation engine not initialized")?
        .as_ref()
        .map_err(|e| e.clone())?;

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
