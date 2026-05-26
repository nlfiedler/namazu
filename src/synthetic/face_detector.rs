//! SCRFD face detection via ONNX Runtime.
//!
//! Loads `scrfd_2.5g.onnx`, runs detection on a displayed-orientation
//! image, and returns face boxes + 5 landmarks in displayed-pixel
//! coordinates. The output is internal in this slice; the HTTP response
//! still carries an empty `faces` array until Phase 5 adds alignment +
//! MobileFaceNet embedding + thumbnails.

use std::path::Path;
use std::sync::Mutex;

use image::{Rgb, RgbImage, imageops, imageops::FilterType};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

/// Square input edge (SCRFD-2.5g standard).
const INPUT_SIZE: usize = 640;
/// Anchors per spatial location, per the InsightFace SCRFD-2.5g recipe.
const NUM_ANCHORS: usize = 2;
/// Feature-map strides (each contributes one (score, bbox, kps) output trio).
const STRIDES: [usize; 3] = [8, 16, 32];

/// Spec-locked confidence threshold.
pub const SCORE_THRESHOLD: f32 = 0.5;
/// Spec-locked NMS IoU threshold.
pub const NMS_IOU: f32 = 0.4;

pub struct FaceDetector {
    /// `Session::run()` takes `&mut self`, so we serialize access here.
    /// ORT's intra-op parallelism handles utilizing all cores within a
    /// single inference.
    session: Mutex<Session>,
    /// Resolved output indices for (score, bbox, kps) per stride, parallel
    /// to `STRIDES`. Looked up once at load time so each request walks
    /// pre-indexed arrays.
    layout: OutputLayout,
}

struct OutputLayout {
    score: [usize; 3],
    bbox: [usize; 3],
    kps: [usize; 3],
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // bbox/landmarks consumed by Phase 5 (alignment + embedding).
pub struct DetectedFace {
    /// `[x, y, w, h]` in displayed-orientation pixel coordinates.
    pub bbox: [f32; 4],
    /// Five landmarks: left-eye, right-eye, nose, left-mouth, right-mouth.
    pub landmarks: [(f32, f32); 5],
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("load scrfd model ({path}): {source}")]
    Ort {
        path: String,
        #[source]
        source: ort::Error,
    },
    #[error(
        "scrfd model outputs do not match the expected layout for SCRFD-2.5g \
         (need score/bbox/kps at each of strides 8/16/32)"
    )]
    UnrecognizedOutputs,
}

impl FaceDetector {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let map_ort = |source| LoadError::Ort {
            path: path.display().to_string(),
            source,
        };
        let session = Session::builder()
            .map_err(map_ort)?
            .commit_from_file(path)
            .map_err(map_ort)?;
        let layout = discover_outputs(&session)?;
        Ok(Self {
            session: Mutex::new(session),
            layout,
        })
    }

    /// Run face detection on a displayed-orientation image.
    ///
    /// Takes a pre-converted `RgbImage` so the same buffer can be reused by
    /// the classifier and alignment stages without re-running `to_rgb8()`.
    pub fn detect(&self, img: &RgbImage) -> ort::Result<Vec<DetectedFace>> {
        let (w, h) = img.dimensions();
        let long_edge = w.max(h).max(1);
        let scale = INPUT_SIZE as f32 / long_edge as f32;
        let new_w = ((w as f32 * scale).round() as u32).min(INPUT_SIZE as u32).max(1);
        let new_h = ((h as f32 * scale).round() as u32).min(INPUT_SIZE as u32).max(1);
        let resized = imageops::resize(img, new_w, new_h, FilterType::Triangle);

        // Letterbox: top-left aligned, padded with zeros.
        let mut canvas =
            RgbImage::from_pixel(INPUT_SIZE as u32, INPUT_SIZE as u32, Rgb([0, 0, 0]));
        for (x, y, px) in resized.enumerate_pixels() {
            canvas.put_pixel(x, y, *px);
        }

        // Normalize: (px - 127.5) / 128.
        let mut arr = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for (x, y, px) in canvas.enumerate_pixels() {
            for c in 0..3 {
                arr[[0, c, y as usize, x as usize]] = (px[c] as f32 - 127.5) / 128.0;
            }
        }

        let tensor = Tensor::from_array(arr)?;
        let mut session = self.session.lock().expect("face detector session poisoned");
        let outputs = session.run(ort::inputs![tensor])?;

        let mut raw: Vec<RawDet> = Vec::new();
        for (i, &stride) in STRIDES.iter().enumerate() {
            let (score_shape, score_data) =
                outputs[self.layout.score[i]].try_extract_tensor::<f32>()?;
            let (bbox_shape, bbox_data) =
                outputs[self.layout.bbox[i]].try_extract_tensor::<f32>()?;
            let (kps_shape, kps_data) =
                outputs[self.layout.kps[i]].try_extract_tensor::<f32>()?;
            decode_stride(
                stride,
                score_shape,
                score_data,
                bbox_shape,
                bbox_data,
                kps_shape,
                kps_data,
                &mut raw,
            );
        }
        // `outputs` borrows from `session`; drop it before releasing the
        // session lock so the borrow checker is happy.
        drop(outputs);
        drop(session);

        let keep = nms(&raw, NMS_IOU);
        let inv = 1.0 / scale;
        let mut out: Vec<DetectedFace> = keep
            .into_iter()
            .map(|idx| {
                let d = &raw[idx];
                DetectedFace {
                    bbox: [
                        d.x1 * inv,
                        d.y1 * inv,
                        (d.x2 - d.x1) * inv,
                        (d.y2 - d.y1) * inv,
                    ],
                    landmarks: d.landmarks.map(|(lx, ly)| (lx * inv, ly * inv)),
                    score: d.score,
                }
            })
            .collect();
        // Sort descending by score; tiebreak by larger bbox area so output
        // is deterministic for fixture-based tests (per spec).
        out.sort_by(|a, b| {
            let area_a = a.bbox[2] * a.bbox[3];
            let area_b = b.bbox[2] * b.bbox[3];
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    area_b
                        .partial_cmp(&area_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        Ok(out)
    }
}

#[derive(Clone)]
struct RawDet {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    landmarks: [(f32, f32); 5],
    score: f32,
}

/// Inspect session outputs and classify each by anchor count + last dim.
///
/// The expected anchor counts are derived from [`INPUT_SIZE`]. The
/// `scrfd_2.5g.onnx` graph has a dynamic input (`[1, 3, ?, ?]`), but we always
/// letterbox the source into 640×640 before inference. If `INPUT_SIZE` ever
/// changes here, the layout discovery below has to change too.
fn discover_outputs(session: &Session) -> Result<OutputLayout, LoadError> {
    let expected_n: [i64; 3] = STRIDES.map(|s| {
        let cells = (INPUT_SIZE / s) as i64;
        cells * cells * NUM_ANCHORS as i64
    });
    let mut score: [Option<usize>; 3] = [None; 3];
    let mut bbox: [Option<usize>; 3] = [None; 3];
    let mut kps: [Option<usize>; 3] = [None; 3];

    for (idx, out) in session.outputs().iter().enumerate() {
        let Some(shape) = out.dtype().tensor_shape() else {
            continue;
        };
        // Accept [N, K] (canonical for this export) or [1, N, K] (older
        // variants that retain the batch dim). `Shape` derefs to `[i64]`.
        let (n, k) = match &shape[..] {
            [1, n, k] => (*n, *k),
            [n, k] => (*n, *k),
            _ => continue,
        };
        let Some(stride_idx) = expected_n.iter().position(|&e| e == n) else {
            continue;
        };
        match k {
            1 => score[stride_idx] = Some(idx),
            4 => bbox[stride_idx] = Some(idx),
            10 => kps[stride_idx] = Some(idx),
            _ => {}
        }
    }

    let unwrap = |arr: [Option<usize>; 3]| -> Result<[usize; 3], LoadError> {
        let mut out = [0usize; 3];
        for (i, v) in arr.iter().enumerate() {
            out[i] = v.ok_or(LoadError::UnrecognizedOutputs)?;
        }
        Ok(out)
    };
    Ok(OutputLayout {
        score: unwrap(score)?,
        bbox: unwrap(bbox)?,
        kps: unwrap(kps)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_stride(
    stride: usize,
    score_shape: &[i64],
    score_data: &[f32],
    bbox_shape: &[i64],
    bbox_data: &[f32],
    kps_shape: &[i64],
    kps_data: &[f32],
    out: &mut Vec<RawDet>,
) {
    let cells = INPUT_SIZE / stride;
    let stride_f = stride as f32;
    let n = (cells * cells * NUM_ANCHORS) as i64;

    // Each row of the flat slice is K elements per anchor (1 score, 4 bbox,
    // 10 kps), regardless of whether the model emits [N, K] or [1, N, K].
    let score_k = row_stride(score_shape, n).unwrap_or(1);
    let bbox_k = row_stride(bbox_shape, n).unwrap_or(4);
    let kps_k = row_stride(kps_shape, n).unwrap_or(10);

    let mut idx = 0;
    for cy in 0..cells {
        let ay = (cy * stride) as f32;
        for cx in 0..cells {
            let ax = (cx * stride) as f32;
            for _a in 0..NUM_ANCHORS {
                let s = score_data[idx * score_k];
                if s >= SCORE_THRESHOLD {
                    let bb_off = idx * bbox_k;
                    let kp_off = idx * kps_k;
                    let dx1 = bbox_data[bb_off] * stride_f;
                    let dy1 = bbox_data[bb_off + 1] * stride_f;
                    let dx2 = bbox_data[bb_off + 2] * stride_f;
                    let dy2 = bbox_data[bb_off + 3] * stride_f;
                    let mut lm = [(0.0_f32, 0.0_f32); 5];
                    for k in 0..5 {
                        lm[k] = (
                            ax + kps_data[kp_off + k * 2] * stride_f,
                            ay + kps_data[kp_off + k * 2 + 1] * stride_f,
                        );
                    }
                    out.push(RawDet {
                        x1: ax - dx1,
                        y1: ay - dy1,
                        x2: ax + dx2,
                        y2: ay + dy2,
                        landmarks: lm,
                        score: s,
                    });
                }
                idx += 1;
            }
        }
    }
}

/// Returns the per-anchor row length given an output shape and the expected
/// anchor count. Handles `[N, K]`, `[1, N, K]`, `[N]`, `[1, N]`.
fn row_stride(shape: &[i64], anchors: i64) -> Option<usize> {
    match shape {
        [n, k] if *n == anchors => Some(*k as usize),
        [a, n, k] if *a == 1 && *n == anchors => Some(*k as usize),
        [n] if *n == anchors => Some(1),
        [a, n] if *a == 1 && *n == anchors => Some(1),
        _ => None,
    }
}

/// Greedy NMS: sort by score desc, suppress later detections whose IoU with
/// any kept detection exceeds `iou_threshold`.
fn nms(dets: &[RawDet], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        dets[b]
            .score
            .partial_cmp(&dets[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut suppressed = vec![false; dets.len()];
    let mut keep = Vec::new();
    for i in 0..order.len() {
        let ai = order[i];
        if suppressed[ai] {
            continue;
        }
        keep.push(ai);
        for &aj in &order[(i + 1)..] {
            if suppressed[aj] {
                continue;
            }
            if iou(&dets[ai], &dets[aj]) > iou_threshold {
                suppressed[aj] = true;
            }
        }
    }
    keep
}

fn iou(a: &RawDet, b: &RawDet) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> RawDet {
        RawDet {
            x1,
            y1,
            x2,
            y2,
            landmarks: [(0.0, 0.0); 5],
            score,
        }
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let a = det(1.0, 0.0, 0.0, 10.0, 10.0);
        let b = det(1.0, 100.0, 100.0, 110.0, 110.0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let a = det(1.0, 0.0, 0.0, 10.0, 10.0);
        let b = a.clone();
        assert!((iou(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_half_overlap() {
        let a = det(1.0, 0.0, 0.0, 10.0, 10.0);
        let b = det(1.0, 5.0, 0.0, 15.0, 10.0);
        assert!((iou(&a, &b) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_highest_score_among_overlaps() {
        let dets = vec![
            det(0.9, 0.0, 0.0, 10.0, 10.0),
            det(0.7, 1.0, 1.0, 11.0, 11.0),
            det(0.8, 100.0, 100.0, 110.0, 110.0),
        ];
        let kept = nms(&dets, 0.4);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0], 0);
        assert_eq!(kept[1], 2);
    }

    #[test]
    fn nms_empty_input() {
        let kept = nms(&[], 0.4);
        assert!(kept.is_empty());
    }

    /// Integration: load the real SCRFD model + the committed fixture and
    /// assert the detector finds multiple faces.
    #[test]
    fn detects_faces_in_several_people_fixture() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = manifest.join("models").join("scrfd_2.5g.onnx");
        let fixture = manifest.join("tests").join("fixtures").join("several-people.jpg");
        if !model.exists() {
            eprintln!("skip: {} not present", model.display());
            return;
        }
        if !fixture.exists() {
            eprintln!("skip: {} not present", fixture.display());
            return;
        }
        let detector = FaceDetector::load(&model).expect("load scrfd");
        let img = image::ImageReader::open(&fixture)
            .expect("open fixture")
            .decode()
            .expect("decode fixture")
            .to_rgb8();
        let faces = detector.detect(&img).expect("detect");
        eprintln!("detected {} face(s):", faces.len());
        for f in &faces {
            eprintln!(
                "  score={:.3} bbox=[{:.1},{:.1} {:.1}x{:.1}]",
                f.score, f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3]
            );
        }
        assert!(
            faces.len() >= 2,
            "expected at least 2 faces in several-people.jpg, got {}",
            faces.len()
        );
    }

    #[test]
    fn detects_no_faces_in_cloud_fixture() {
        // Pure sky scene; nothing face-like. We use this rather than the
        // dog fixture because SCRFD (trained on human faces) sometimes
        // false-positives on animal faces.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = manifest.join("models").join("scrfd_2.5g.onnx");
        let fixture = manifest.join("tests").join("fixtures").join("cloud.jpg");
        if !model.exists() || !fixture.exists() {
            eprintln!("skip: prerequisites not present");
            return;
        }
        let detector = FaceDetector::load(&model).expect("load scrfd");
        let img = image::ImageReader::open(&fixture)
            .expect("open fixture")
            .decode()
            .expect("decode fixture")
            .to_rgb8();
        let faces = detector.detect(&img).expect("detect");
        assert!(
            faces.is_empty(),
            "expected 0 faces in cloud.jpg, got {}",
            faces.len()
        );
    }
}
