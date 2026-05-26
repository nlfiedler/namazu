//! Image decoding (with EXIF orientation applied) and the MobileNetV2 input
//! preprocess.

use std::path::Path;

use image::{DynamicImage, RgbImage, imageops, imageops::FilterType};
use ndarray::Array4;

use crate::{correct_orientation, get_image_orientation};

use super::engine::MAX_DIM;

/// MobileNetV2 standard input edge (square).
pub const CLASSIFIER_INPUT: usize = 224;
/// Resize short edge to this before center-crop (ONNX Model Zoo convention).
pub const CLASSIFIER_RESIZE: u32 = 256;

/// ImageNet normalization (RGB per-channel mean / std).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Decode the image and apply EXIF orientation so the result is in
/// displayed orientation.
///
/// A strict `max_image_width`/`max_image_height` limit pegged at
/// [`MAX_DIM`] is set on the decoder so a pathological PNG (e.g. a
/// decompression bomb) is rejected before any pixel buffer is
/// allocated, rather than being caught after decode by the dimension
/// check in `engine::process`. `image::ImageError::Limits` is the
/// error variant returned in that case.
pub fn decode_oriented(path: &Path) -> Result<DynamicImage, image::ImageError> {
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DIM);
    limits.max_image_height = Some(MAX_DIM);
    // Generous headroom over an `MAX_DIM × MAX_DIM × 4-byte` RGBA buffer
    // so well-formed inputs at the dimension limit still decode.
    limits.max_alloc = Some(
        (MAX_DIM as u64)
            .saturating_mul(MAX_DIM as u64)
            .saturating_mul(4)
            .saturating_add(64 * 1024 * 1024),
    );
    reader.limits(limits);
    let mut img = reader.decode()?;
    if let Ok(o) = get_image_orientation(path) {
        img = correct_orientation(o, img);
    }
    Ok(img)
}

/// MobileNetV2 preprocess: short-edge-256 resize (bilinear) → 224×224
/// center-crop → float32 in `[0, 1]` → per-channel ImageNet normalization →
/// NCHW tensor `[1, 3, 224, 224]`.
///
/// Takes a pre-converted `RgbImage` so the caller can do the `to_rgb8()` once
/// and share the buffer with the face pipeline.
pub fn classifier_input(img: &RgbImage) -> Array4<f32> {
    let (w, h) = img.dimensions();
    let short = w.min(h).max(1);
    let scale = CLASSIFIER_RESIZE as f32 / short as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(CLASSIFIER_INPUT as u32);
    let new_h = ((h as f32 * scale).round() as u32).max(CLASSIFIER_INPUT as u32);
    let resized = imageops::resize(img, new_w, new_h, FilterType::Triangle);

    let crop_x = (new_w - CLASSIFIER_INPUT as u32) / 2;
    let crop_y = (new_h - CLASSIFIER_INPUT as u32) / 2;
    let cropped = imageops::crop_imm(
        &resized,
        crop_x,
        crop_y,
        CLASSIFIER_INPUT as u32,
        CLASSIFIER_INPUT as u32,
    )
    .to_image();

    let mut arr = Array4::<f32>::zeros((1, 3, CLASSIFIER_INPUT, CLASSIFIER_INPUT));
    for (x, y, px) in cropped.enumerate_pixels() {
        for c in 0..3 {
            arr[[0, c, y as usize, x as usize]] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn classifier_input_has_expected_shape() {
        let img = RgbImage::new(400, 300);
        let arr = classifier_input(&img);
        assert_eq!(arr.shape(), &[1, 3, CLASSIFIER_INPUT, CLASSIFIER_INPUT]);
    }

    #[test]
    fn black_pixel_becomes_negative_mean_over_std() {
        let img = RgbImage::new(400, 300);
        let arr = classifier_input(&img);
        let expected = [-MEAN[0] / STD[0], -MEAN[1] / STD[1], -MEAN[2] / STD[2]];
        for c in 0..3 {
            let v = arr[[0, c, 100, 100]];
            assert!(
                (v - expected[c]).abs() < 1e-5,
                "channel {c}: got {v}, expected {}",
                expected[c]
            );
        }
    }

    #[test]
    fn small_image_is_upscaled_not_padded() {
        // 100x80 input should still produce a 224x224 crop.
        let img = RgbImage::new(100, 80);
        let arr = classifier_input(&img);
        assert_eq!(arr.shape(), &[1, 3, CLASSIFIER_INPUT, CLASSIFIER_INPUT]);
    }
}
