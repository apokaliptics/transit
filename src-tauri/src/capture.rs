use image::RgbaImage;
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use xcap::Monitor;

/// Stores the previous frame hash for diff detection.
static PREV_HASH: Mutex<Option<Vec<u8>>> = Mutex::new(None);

/// Captures the screen region at the given coordinates and returns the image
/// only if the content has changed since the last capture.
pub fn capture_region(x: i32, y: i32, width: u32, height: u32) -> Result<Option<RgbaImage>, String> {
    let monitors = Monitor::all().map_err(|e| format!("Failed to list monitors: {e}"))?;

    if width == 0 || height == 0 {
        return Ok(None);
    }

    let req_left = x;
    let req_top = y;
    let req_right = x + width as i32;
    let req_bottom = y + height as i32;

    let mut composed = RgbaImage::new(width, height);
    let mut any_intersection = false;

    for monitor in monitors {
        let mon_x = monitor.x().unwrap_or(0);
        let mon_y = monitor.y().unwrap_or(0);
        let mon_w = monitor.width().unwrap_or(0) as i32;
        let mon_h = monitor.height().unwrap_or(0) as i32;
        let mon_right = mon_x + mon_w;
        let mon_bottom = mon_y + mon_h;

        let inter_left = req_left.max(mon_x);
        let inter_top = req_top.max(mon_y);
        let inter_right = req_right.min(mon_right);
        let inter_bottom = req_bottom.min(mon_bottom);

        if inter_left >= inter_right || inter_top >= inter_bottom {
            continue;
        }

        let screenshot = monitor
            .capture_image()
            .map_err(|e| format!("Screen capture failed: {e}"))?;

        let src_x = (inter_left - mon_x) as u32;
        let src_y = (inter_top - mon_y) as u32;
        let inter_w = (inter_right - inter_left) as u32;
        let inter_h = (inter_bottom - inter_top) as u32;

        let piece = image::imageops::crop_imm(&screenshot, src_x, src_y, inter_w, inter_h).to_image();

        let dst_x = (inter_left - req_left) as i64;
        let dst_y = (inter_top - req_top) as i64;
        image::imageops::replace(&mut composed, &piece, dst_x, dst_y);
        any_intersection = true;
    }

    if !any_intersection {
        return Err("Requested capture region does not intersect any monitor".to_string());
    }

    // Hash the pixel data for diff detection
    let raw_bytes = composed.as_raw();
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
    Ok(Some(composed))
}

/// Resets the diff checker (e.g. when starting a new session).
pub fn reset_diff() {
    if let Ok(mut prev) = PREV_HASH.lock() {
        *prev = None;
    }
}
