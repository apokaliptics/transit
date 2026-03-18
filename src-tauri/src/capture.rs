use image::RgbaImage;
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use xcap::Monitor;

/// Stores the previous frame hash for diff detection.
static PREV_HASH: Mutex<Option<Vec<u8>>> = Mutex::new(None);

/// Captures the screen region at the given coordinates and returns the image
/// only if the content has changed since the last capture.
pub fn capture_region(x: i32, y: i32, width: u32, height: u32) -> Result<Option<RgbaImage>, String> {
    // Find the monitor that contains the window
    let monitors = Monitor::all().map_err(|e| format!("Failed to list monitors: {e}"))?;

    let monitor = monitors
        .into_iter()
        .find(|m| {
            let mx = m.x().unwrap_or(0);
            let my = m.y().unwrap_or(0);
            let mw = m.width().unwrap_or(0) as i32;
            let mh = m.height().unwrap_or(0) as i32;
            x >= mx && y >= my && x < mx + mw && y < my + mh
        })
        .ok_or_else(|| "No monitor found at window position".to_string())?;

    // Capture the entire monitor
    let screenshot = monitor
        .capture_image()
        .map_err(|e| format!("Screen capture failed: {e}"))?;

    // Calculate crop coordinates relative to the monitor
    let mon_x = monitor.x().unwrap_or(0);
    let mon_y = monitor.y().unwrap_or(0);
    let rel_x = (x - mon_x).max(0) as u32;
    let rel_y = (y - mon_y).max(0) as u32;

    // Clamp dimensions to stay within the screenshot
    let crop_w = width.min(screenshot.width().saturating_sub(rel_x));
    let crop_h = height.min(screenshot.height().saturating_sub(rel_y));

    if crop_w == 0 || crop_h == 0 {
        return Ok(None);
    }

    // Crop to the window region
    let cropped = image::imageops::crop_imm(&screenshot, rel_x, rel_y, crop_w, crop_h).to_image();

    // Hash the pixel data for diff detection
    let raw_bytes = cropped.as_raw();
    let mut hasher = Sha256::new();
    hasher.update(raw_bytes);
    let hash = hasher.finalize().to_vec();

    // Compare with previous hash
    let mut prev = PREV_HASH.lock().map_err(|e| format!("Lock error: {e}"))?;
    if prev.as_ref() == Some(&hash) {
        // Content hasn't changed
        return Ok(None);
    }

    // Content changed — update the stored hash and return the image
    *prev = Some(hash);
    Ok(Some(cropped))
}

/// Resets the diff checker (e.g. when starting a new session).
pub fn reset_diff() {
    if let Ok(mut prev) = PREV_HASH.lock() {
        *prev = None;
    }
}
