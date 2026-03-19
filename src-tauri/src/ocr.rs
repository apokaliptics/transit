use image::{DynamicImage, RgbaImage};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcrBackend {
    Auto,
    Windows,
    WindowsProfile,
}

impl OcrBackend {
    pub fn from_str(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "windows-profile" => Self::WindowsProfile,
            "windows" => Self::Windows,
            _ => Self::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Windows => "windows",
            Self::WindowsProfile => "windows-profile",
        }
    }

    pub fn fallback_policy(self) -> &'static str {
        match self {
            Self::Auto => "explicit-then-profile-fallback",
            Self::Windows => "explicit-only",
            Self::WindowsProfile => "profile-only",
        }
    }
}

fn language_tag_for_pair(language_pair: &str) -> &'static str {
    match language_pair {
        "ja" | "ja-en" => "ja",
        "zh" | "zh-en" => "zh-CN",
        "ko" | "ko-en" => "ko",
        _ => "en",
    }
}

pub fn is_language_supported(language_pair: &str) -> Result<bool, String> {
    #[cfg(windows)]
    {
        use windows::core::HSTRING;
        use windows::Globalization::Language;
        use windows::Media::Ocr::OcrEngine;

        let tag = language_tag_for_pair(language_pair);
        let h = HSTRING::from(tag);
        let lang = Language::CreateLanguage(&h)
            .map_err(|e| format!("Failed to create language tag {tag}: {e}"))?;

        OcrEngine::IsLanguageSupported(&lang)
            .map_err(|e| format!("Failed to query OCR language support: {e}"))
    }
    #[cfg(not(windows))]
    {
        let _ = language_pair;
        Ok(false)
    }
}

pub fn is_backend_available(backend: OcrBackend) -> Result<bool, String> {
    match backend {
        OcrBackend::Auto => Ok(true),
        OcrBackend::Windows | OcrBackend::WindowsProfile => {
            #[cfg(windows)]
            {
                Ok(true)
            }
            #[cfg(not(windows))]
            {
                Ok(false)
            }
        }
    }
}

pub fn is_language_supported_for_backend(backend: OcrBackend, language_pair: &str) -> Result<bool, String> {
    match backend {
        OcrBackend::Windows => is_language_supported(language_pair),
        OcrBackend::Auto => is_language_supported(language_pair),
        OcrBackend::WindowsProfile => {
            // Profile mode does not target a specific language tag and relies on user profile OCR languages.
            Ok(true)
        }
    }
}

/// Single OCR line with bounding box in capture-image pixel coordinates.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OcrLine {
    pub text: String,
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
}

/// OCR output with full text and per-line geometry.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OcrResult {
    pub text: String,
    pub lines: Vec<OcrLine>,
    pub language_tag: String,
    pub used_profile_fallback: bool,
}

/// Performs OCR on an RGBA image using the Windows.Media.Ocr API.
///
/// Returns the recognized text as a single string with lines separated by newlines.
pub fn recognize_text_with_backend(
    img: &RgbaImage,
    language_pair: &str,
    backend: OcrBackend,
) -> Result<OcrResult, String> {
    #[cfg(windows)]
    {
        match backend {
            OcrBackend::Auto | OcrBackend::Windows | OcrBackend::WindowsProfile => {
                windows_ocr(img, language_pair, backend)
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = img;
        let _ = language_pair;
        let _ = backend;
        Err("OCR is only supported on Windows in this build".to_string())
    }
}

#[cfg(windows)]
fn windows_ocr(img: &RgbaImage, language_pair: &str, backend: OcrBackend) -> Result<OcrResult, String> {
    use windows::core::HSTRING;
    use windows::Globalization::Language;
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine;

    let (prepared, prep_scale) = preprocess_for_ocr(img, language_pair);
    let width = prepared.width();
    let height = prepared.height();

    // Convert RGBA → BGRA (Windows expects BGRA8)
    let mut bgra_pixels = Vec::with_capacity((width * height * 4) as usize);
    for pixel in prepared.pixels() {
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

    let language_tag = language_tag_for_pair(language_pair);

    let language_hstring = HSTRING::from(language_tag);
    let lang = Language::CreateLanguage(&language_hstring)
        .map_err(|e| format!("Failed to create language tag {language_tag}: {e}"))?;

    let mut used_profile_fallback = false;
    let engine = match backend {
        OcrBackend::Windows => OcrEngine::TryCreateFromLanguage(&lang)
            .map_err(|e| format!("Failed to create explicit-language OcrEngine for {language_tag}: {e}"))?,
        OcrBackend::WindowsProfile => OcrEngine::TryCreateFromUserProfileLanguages()
            .map_err(|e| format!("Failed to create profile-language OcrEngine: {e}"))?,
        OcrBackend::Auto => match OcrEngine::TryCreateFromLanguage(&lang) {
            Ok(e) => e,
            Err(explicit_err) => {
                log::warn!(
                    "Failed to create OcrEngine for language {language_tag}: {explicit_err}; falling back to user profile languages"
                );
                used_profile_fallback = true;
                OcrEngine::TryCreateFromUserProfileLanguages()
                    .map_err(|e| format!("Failed to create OcrEngine: {e}"))?
            }
        },
    };

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
    let mut out_lines = Vec::new();
    for line in &lines {
        let text = match line.Text() {
            Ok(t) => t,
            Err(_) => continue,
        };

        let s = text.to_string_lossy();
        if s.is_empty() {
            continue;
        }

        text_parts.push(s.clone());

        let words = line
            .Words()
            .map_err(|e| format!("Failed to read OCR words: {e}"))?;

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        for word in &words {
            let rect = word
                .BoundingRect()
                .map_err(|e| format!("Failed to read OCR word bounds: {e}"))?;

            let left = rect.X.floor() as i32;
            let top = rect.Y.floor() as i32;
            let right = (rect.X + rect.Width).ceil() as i32;
            let bottom = (rect.Y + rect.Height).ceil() as i32;

            min_x = min_x.min(left);
            min_y = min_y.min(top);
            max_x = max_x.max(right);
            max_y = max_y.max(bottom);
        }

        let (left, top, width, height) = if min_x <= max_x && min_y <= max_y {
            let raw_left = min_x.max(0) as f64 / prep_scale;
            let raw_top = min_y.max(0) as f64 / prep_scale;
            let raw_width = (max_x - min_x).max(0) as f64 / prep_scale;
            let raw_height = (max_y - min_y).max(0) as f64 / prep_scale;

            (
                raw_left.round().max(0.0) as i32,
                raw_top.round().max(0.0) as i32,
                raw_width.round().max(0.0) as u32,
                raw_height.round().max(0.0) as u32,
            )
        } else {
            (0, 0, 0, 0)
        };

        out_lines.push(OcrLine {
            text: s,
            left,
            top,
            width,
            height,
        });
    }

    Ok(OcrResult {
        text: text_parts.join("\n"),
        lines: out_lines,
        language_tag: language_tag.to_string(),
        used_profile_fallback,
    })
}

fn preprocess_for_ocr(img: &RgbaImage, language_pair: &str) -> (RgbaImage, f64) {
    let cjk_mode = matches!(language_pair, "ja" | "ja-en" | "zh" | "zh-en" | "ko" | "ko-en");

    let source = DynamicImage::ImageRgba8(img.clone());
    let mut scale = 1.0f64;

    // Upscale small captures to improve OCR recall on compact UI fonts.
    let min_side = img.width().min(img.height());
    if min_side > 0 && min_side < 420 {
        scale = 2.0;
    } else if min_side >= 420 && min_side < 720 {
        scale = 1.5;
    }

    let mut processed = if cjk_mode {
        // CJK often benefits from normalized luminance and stronger edge contrast.
        source.grayscale().adjust_contrast(24.0)
    } else {
        source.adjust_contrast(12.0)
    };

    if scale > 1.0 {
        let new_w = (img.width() as f64 * scale).round() as u32;
        let new_h = (img.height() as f64 * scale).round() as u32;
        processed = processed.resize(new_w.max(1), new_h.max(1), image::imageops::FilterType::CatmullRom);
    }

    (processed.to_rgba8(), scale)
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
