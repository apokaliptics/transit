use futures_util::StreamExt;
use reqwest::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, serde::Serialize)]
struct DownloadProgress {
    file: String,
    downloaded: u64,
    total: Option<u64>,
}

async fn resolve_content_length(client: &Client, url: &str) -> Option<u64> {
    // Some hosts omit content-length on streaming GET responses; try HEAD first.
    match client.head(url).send().await {
        Ok(resp) if resp.status().is_success() => resp.content_length(),
        _ => None,
    }
}

pub async fn download_file(
    app: &AppHandle,
    client: &Client,
    url: &str,
    filename: &str,
    save_path: &Path,
) -> Result<(), String> {
    log::info!("Downloading {} to {:?}", url, save_path);

    let res = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !res.status().is_success() {
        return Err(format!("HTTP Error {}: {}", res.status(), url));
    }

    let total_size = match res.content_length() {
        Some(len) => Some(len),
        None => resolve_content_length(client, url).await,
    };
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    let mut file = File::create(save_path).map_err(|e| format!("File creation failed: {e}"))?;

    let emit_interval = 2; // Keep UI responsive without flooding IPC.
    let mut chunk_count = 0;

    let _ = app.emit("download-progress", DownloadProgress {
        file: filename.to_string(),
        downloaded: 0,
        total: total_size,
    });

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| format!("Chunk error: {e}"))?;
        file.write_all(&chunk)
            .map_err(|e| format!("Write error: {e}"))?;

        downloaded += chunk.len() as u64;
        chunk_count += 1;

        if chunk_count % emit_interval == 0 {
            let _ = app.emit("download-progress", DownloadProgress {
                file: filename.to_string(),
                downloaded,
                total: total_size,
            });
        }
    }

    let _ = app.emit("download-progress", DownloadProgress {
        file: filename.to_string(),
        downloaded,
        total: total_size,
    });

    Ok(())
}

async fn try_download_variant(
    app: &AppHandle,
    client: &Client,
    base_url: &str,
    filename1: &str,
    filename2: &str,
    save_dir: &Path,
) -> Result<(), String> {
    let url1 = format!("{base_url}/{filename1}");
    let save_path1 = save_dir.join(filename1);
    
    match download_file(app, client, &url1, filename1, &save_path1).await {
        Ok(_) => Ok(()),
        Err(e) => {
            log::warn!("Failed to download {}, trying fallback {}. Error: {}", filename1, filename2, e);
            let url2 = format!("{base_url}/{filename2}");
            let save_path2 = save_dir.join(filename2);
            download_file(app, client, &url2, filename2, &save_path2).await
        }
    }
}

#[tauri::command]
pub async fn download_models(app: AppHandle, language_pair: String) -> Result<String, String> {
    let local_data_dir = app.path().app_local_data_dir()
        .map_err(|e| format!("Could not determine local data directory: {}", e))?;
    
    let model_dir = local_data_dir.join("models").join(&language_pair);
    
    if !model_dir.exists() {
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| format!("Failed to create model directory: {e}"))?;
    }

    let client = Client::new();
    let base_url = format!("https://huggingface.co/onnx-community/opus-mt-{}/resolve/main", language_pair);
    let onnx_base_url = format!("{}/onnx", base_url);

    // 1. Download Tokenizer
    download_file(
        &app,
        &client,
        &format!("{base_url}/tokenizer.json"),
        "tokenizer.json",
        &model_dir.join("tokenizer.json")
    ).await?;

    // 2. Download Encoder Model
    download_file(
        &app,
        &client,
        &format!("{onnx_base_url}/encoder_model.onnx"),
        "encoder_model.onnx",
        &model_dir.join("encoder_model.onnx")
    ).await?;

    // 3. Download Decoder Model (try merged first, fallback to regular)
    try_download_variant(
        &app,
        &client,
        &onnx_base_url,
        "decoder_model_merged.onnx",
        "decoder_model.onnx",
        &model_dir
    ).await?;

    Ok(model_dir.to_string_lossy().to_string())
}

#[tauri::command]
pub fn check_models(app: AppHandle, language_pair: String) -> Result<bool, String> {
    let local_data_dir = app.path().app_local_data_dir()
        .map_err(|e| format!("Could not determine local data directory: {}", e))?;
    
    let model_dir = local_data_dir.join("models").join(&language_pair);
    
    let has_tokenizer = model_dir.join("tokenizer.json").exists();
    let has_encoder = model_dir.join("encoder_model.onnx").exists();
    let has_decoder = model_dir.join("decoder_model_merged.onnx").exists() || 
                      model_dir.join("decoder_model.onnx").exists();
                      
    Ok(has_tokenizer && has_encoder && has_decoder)
}
