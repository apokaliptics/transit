use image::RgbaImage;

/// Performs OCR on an RGBA image using the Windows.Media.Ocr API.
///
/// Returns the recognized text as a single string with lines separated by newlines.
pub fn recognize_text(img: &RgbaImage) -> Result<String, String> {
    #[cfg(windows)]
    {
        windows_ocr(img)
    }
    #[cfg(not(windows))]
    {
        let _ = img;
        Err("OCR is only supported on Windows in this build".to_string())
    }
}

#[cfg(windows)]
fn windows_ocr(img: &RgbaImage) -> Result<String, String> {
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;


    let width = img.width();
    let height = img.height();

    // Convert RGBA → BGRA (Windows expects BGRA8)
    let mut bgra_pixels = Vec::with_capacity((width * height * 4) as usize);
    for pixel in img.pixels() {
        let [r, g, b, a] = pixel.0;
        bgra_pixels.push(b);
        bgra_pixels.push(g);
        bgra_pixels.push(r);
        bgra_pixels.push(a);
    }

    // Create a SoftwareBitmap from the BGRA pixel data
    let bitmap = SoftwareBitmap::CreateCopyFromBuffer(
        &create_buffer(&bgra_pixels)?,
        BitmapPixelFormat::Bgra8,
        width as i32,
        height as i32,
    )
    .map_err(|e| format!("Failed to create SoftwareBitmap: {e}"))?;

    // Create OCR engine using user profile languages
    let engine = OcrEngine::TryCreateFromUserProfileLanguages()
        .map_err(|e| format!("Failed to create OcrEngine: {e}"))?;

    // Perform OCR
    let result = engine
        .RecognizeAsync(&bitmap)
        .map_err(|e| format!("RecognizeAsync failed: {e}"))?
        .get()
        .map_err(|e| format!("OCR get result failed: {e}"))?;

    // Extract all lines of text
    let lines = result
        .Lines()
        .map_err(|e| format!("Failed to get lines: {e}"))?;

    let mut text_parts = Vec::new();
    for line in &lines {
        if let Ok(text) = line.Text() {
            let s = text.to_string_lossy();
            if !s.is_empty() {
                text_parts.push(s);
            }
        }
    }

    Ok(text_parts.join("\n"))
}

/// Creates an IBuffer from a byte slice using DataWriter.
#[cfg(windows)]
fn create_buffer(
    data: &[u8],
) -> Result<windows::Storage::Streams::IBuffer, String> {
    use windows::Storage::Streams::DataWriter;

    let writer = DataWriter::new()
        .map_err(|e| format!("DataWriter::new failed: {e}"))?;

    writer
        .WriteBytes(data)
        .map_err(|e| format!("WriteBytes failed: {e}"))?;

    let buffer = writer
        .DetachBuffer()
        .map_err(|e| format!("DetachBuffer failed: {e}"))?;

    Ok(buffer)
}
