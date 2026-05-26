//! 5-point face alignment via least-squares similarity transform.
//!
//! Given five detected landmarks (left-eye, right-eye, nose, left-mouth,
//! right-mouth) in displayed-orientation pixel coordinates, this module
//! computes the 2D similarity transform (scale + rotation + translation)
//! that maps them onto the standard ArcFace 112×112 reference template,
//! then inverse-warps the source image with bilinear sampling into a
//! 112×112 RGB buffer suitable for MobileFaceNet.
//!
//! The transform has four unknowns (`a`, `b` form scale·rotation;
//! `c`, `d` are translation):
//!
//! ```text
//! [target_x]   [a -b] [src_x]   [c]
//! [target_y] = [b  a] [src_y] + [d]
//! ```
//!
//! With five point pairs we have ten equations; the closed-form
//! least-squares solution is what `similarity_transform` returns.

use image::{DynamicImage, Rgb, RgbImage};

/// Edge of the aligned face (ArcFace standard).
pub const ALIGNED_SIZE: u32 = 112;

/// Standard ArcFace 5-point reference template, in 112×112 coordinates.
const REFERENCE: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

/// Closed-form least-squares 2D similarity transform mapping `src` to the
/// ArcFace template. Returns `(a, b, c, d)`.
pub fn similarity_transform(src: &[(f32, f32); 5]) -> (f32, f32, f32, f32) {
    let n = src.len() as f32;
    let (sx_x, sx_y) = src
        .iter()
        .fold((0.0_f32, 0.0_f32), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let (sd_x, sd_y) = REFERENCE
        .iter()
        .fold((0.0_f32, 0.0_f32), |acc, &(x, y)| (acc.0 + x, acc.1 + y));
    let mx_x = sx_x / n;
    let mx_y = sx_y / n;
    let md_x = sd_x / n;
    let md_y = sd_y / n;

    let mut numer_a = 0.0_f32;
    let mut numer_b = 0.0_f32;
    let mut denom = 0.0_f32;
    for i in 0..5 {
        let cx = src[i].0 - mx_x;
        let cy = src[i].1 - mx_y;
        let ctx = REFERENCE[i].0 - md_x;
        let cty = REFERENCE[i].1 - md_y;
        numer_a += cx * ctx + cy * cty;
        numer_b += cx * cty - cy * ctx;
        denom += cx * cx + cy * cy;
    }
    let a = numer_a / denom;
    let b = numer_b / denom;
    let c = md_x - a * mx_x + b * mx_y;
    let d = md_y - b * mx_x - a * mx_y;
    (a, b, c, d)
}

/// Warp the source image into a 112×112 aligned face buffer using the
/// inverse of the similarity transform from `landmarks` to the ArcFace
/// template. Out-of-bounds samples come back black.
pub fn align_face(img: &DynamicImage, landmarks: &[(f32, f32); 5]) -> RgbImage {
    let (a, b, c, d) = similarity_transform(landmarks);
    // Inverse of [[a -b][b a]] is (1/s²) * [[a b][-b a]], where s² = a² + b².
    let scale_sq = a * a + b * b;
    let inv = 1.0 / scale_sq;

    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut out = RgbImage::new(ALIGNED_SIZE, ALIGNED_SIZE);

    for py in 0..ALIGNED_SIZE {
        for px in 0..ALIGNED_SIZE {
            let dx = px as f32 - c;
            let dy = py as f32 - d;
            let sx = inv * (a * dx + b * dy);
            let sy = inv * (-b * dx + a * dy);
            out.put_pixel(px, py, sample_bilinear(&rgb, sx, sy, w, h));
        }
    }
    out
}

fn sample_bilinear(img: &RgbImage, x: f32, y: f32, w: u32, h: u32) -> Rgb<u8> {
    if !(0.0..=(w as f32 - 1.0)).contains(&x) || !(0.0..=(h as f32 - 1.0)).contains(&y) {
        return Rgb([0, 0, 0]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0).0;
    let p01 = img.get_pixel(x0, y1).0;
    let p10 = img.get_pixel(x1, y0).0;
    let p11 = img.get_pixel(x1, y1).0;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = (1.0 - dx) * (1.0 - dy) * p00[c] as f32
            + dx * (1.0 - dy) * p10[c] as f32
            + (1.0 - dx) * dy * p01[c] as f32
            + dx * dy * p11[c] as f32;
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn identity_transform_when_landmarks_match_template() {
        let (a, b, c, d) = similarity_transform(&REFERENCE);
        assert!(approx(a, 1.0, 1e-5), "a={a}");
        assert!(approx(b, 0.0, 1e-5), "b={b}");
        assert!(approx(c, 0.0, 1e-4), "c={c}");
        assert!(approx(d, 0.0, 1e-4), "d={d}");
    }

    #[test]
    fn doubled_landmarks_produce_half_scale_transform() {
        let mut scaled = REFERENCE;
        for p in &mut scaled {
            p.0 *= 2.0;
            p.1 *= 2.0;
        }
        let (a, b, _c, _d) = similarity_transform(&scaled);
        assert!(approx(a, 0.5, 1e-5), "a={a}");
        assert!(approx(b, 0.0, 1e-5), "b={b}");
    }

    #[test]
    fn translated_landmarks_produce_translation_only() {
        let mut shifted = REFERENCE;
        for p in &mut shifted {
            p.0 += 50.0;
            p.1 += 30.0;
        }
        let (a, b, c, d) = similarity_transform(&shifted);
        assert!(approx(a, 1.0, 1e-5));
        assert!(approx(b, 0.0, 1e-5));
        assert!(approx(c, -50.0, 1e-3), "c={c}");
        assert!(approx(d, -30.0, 1e-3), "d={d}");
    }

    #[test]
    fn rotated_landmarks_round_trip_through_inverse() {
        // Rotate template by 90° (clockwise) and shift; recover the transform
        // and verify applying it returns to the template.
        let theta = std::f32::consts::FRAC_PI_2;
        let (s, co) = theta.sin_cos();
        let mut rotated = REFERENCE;
        for p in &mut rotated {
            let (x, y) = (p.0, p.1);
            p.0 = co * x - s * y + 10.0;
            p.1 = s * x + co * y - 5.0;
        }
        let (a, b, c, d) = similarity_transform(&rotated);
        for i in 0..5 {
            let (x, y) = rotated[i];
            let tx = a * x - b * y + c;
            let ty = b * x + a * y + d;
            assert!(approx(tx, REFERENCE[i].0, 1e-3), "i={i} tx={tx}");
            assert!(approx(ty, REFERENCE[i].1, 1e-3), "i={i} ty={ty}");
        }
    }

    #[test]
    fn align_face_produces_correct_size() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 480, Rgb([100, 150, 200])));
        let landmarks = REFERENCE; // identity transform, but sampling will be out-of-bounds
        let aligned = align_face(&img, &landmarks);
        assert_eq!(aligned.dimensions(), (ALIGNED_SIZE, ALIGNED_SIZE));
    }
}
