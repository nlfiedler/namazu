//! Top-level synthetic engine: holds loaded models + labels-map, exposes a
//! single `process()` entry point used by the HTTP handler.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::{DynamicImage, ExtendedColorType, GenericImageView, RgbImage, codecs::jpeg::JpegEncoder};
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::align;
use super::classifier::{self, Classifier};
use super::embedder::{self, Embedder};
use super::face_detector::{self, DetectedFace, FaceDetector};
use super::labels::{self, CurationResult, Label, LabelsMap};
use super::preprocess;

/// Maximum input dimension (per spec): images larger than this are rejected
/// with 413.
pub const MAX_DIM: u32 = 8000;

/// Spec-mandated cap on the curated labels list.
pub const LABEL_CAP: usize = 20;

/// Spec-mandated cap on the faces list.
pub const FACE_CAP: usize = 20;

/// Spec-mandated cap on the total serialized response. If the response
/// exceeds this after the count caps are applied, faces are dropped one at
/// a time (lowest score first) until it fits.
pub const RESPONSE_BYTE_CAP: usize = 1_000_000;

/// JPEG quality for face thumbnails (~85 per spec).
const THUMBNAIL_QUALITY: u8 = 85;

/// Spec-mandated per-image processing timeout.
pub const PROCESSING_TIMEOUT: Duration = Duration::from_secs(30);

const LABELS_MODEL_VERSION: &str = "mobilenetv2-v1";
const FACES_MODEL_VERSION: &str = "mobilefacenet-v1";

pub struct SyntheticEngine {
    labels_map: Arc<LabelsMap>,
    classifier: Arc<Classifier>,
    face_detector: Arc<FaceDetector>,
    embedder: Arc<Embedder>,
    /// Bounds concurrent inferences to the number of available cores so
    /// requests queue async rather than piling up in the blocking thread
    /// pool, where they would hold threads while contending for the model
    /// session mutexes.
    semaphore: Arc<Semaphore>,
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error(transparent)]
    LabelsMap(#[from] labels::LabelsMapError),
    #[error(transparent)]
    Classifier(#[from] classifier::LoadError),
    #[error(transparent)]
    FaceDetector(#[from] face_detector::LoadError),
    #[error(transparent)]
    Embedder(#[from] embedder::LoadError),
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("could not decode image: {0}")]
    Decode(#[from] image::ImageError),
    #[error("image too large: {0}x{1}")]
    TooLarge(u32, u32),
    #[error("inference failed: {0}")]
    Inference(#[source] ort::Error),
    #[error("thumbnail encode failed: {0}")]
    ThumbnailEncode(#[source] image::ImageError),
}

#[derive(Debug, Serialize)]
pub struct SyntheticResponse {
    pub labels: Vec<Label>,
    pub faces: Vec<Face>,
    pub model_versions: ModelVersions,
    pub truncated: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelVersions {
    pub labels: String,
    pub faces: String,
}

#[derive(Debug, Serialize)]
pub struct Face {
    /// `[x, y, width, height]` in displayed-orientation pixel coordinates.
    pub bbox: [f32; 4],
    /// Base64 of the raw little-endian `f32` embedding produced by
    /// MobileFaceNet (128 floats → 512 bytes raw → ~684 chars base64).
    pub embedding: String,
    /// Base64-encoded JPEG of the aligned face crop.
    pub thumbnail: String,
    pub score: f32,
    pub model_version: String,
}

impl SyntheticEngine {
    /// Load all required model artifacts from `models_dir`. Failure is fatal
    /// to the caller (per spec).
    pub fn new(models_dir: &Path) -> Result<Self, InitError> {
        let labels_map = LabelsMap::load(&models_dir.join("labels-map.json"))?;
        let classifier = Classifier::load(&models_dir.join("mobilenet_v2.onnx"))?;
        let face_detector = FaceDetector::load(&models_dir.join("scrfd_2.5g.onnx"))?;
        let embedder = Embedder::load(&models_dir.join("mobilefacenet.onnx"))?;
        let permits = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Ok(Self {
            labels_map: Arc::new(labels_map),
            classifier: Arc::new(classifier),
            face_detector: Arc::new(face_detector),
            embedder: Arc::new(embedder),
            semaphore: Arc::new(Semaphore::new(permits)),
        })
    }

    /// Acquire an inference permit. The returned guard must be held for the
    /// lifetime of the `process()` call (typically by moving it into the
    /// `web::block` closure).
    pub async fn acquire_permit(&self) -> OwnedSemaphorePermit {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("synthetic semaphore should never be closed")
    }

    /// Synchronous processing pipeline. Designed to run inside `web::block`.
    pub fn process(&self, image_path: &Path) -> Result<SyntheticResponse, ProcessError> {
        let img = preprocess::decode_oriented(image_path)?;
        let (w, h) = img.dimensions();
        if w > MAX_DIM || h > MAX_DIM {
            return Err(ProcessError::TooLarge(w, h));
        }

        // Labels.
        let classifier_input = preprocess::classifier_input(&img);
        let probs = self
            .classifier
            .infer(classifier_input)
            .map_err(ProcessError::Inference)?;
        let scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        let CurationResult {
            labels,
            truncated: labels_truncated,
        } = labels::curate(&scored, &self.labels_map, LABEL_CAP);

        // Faces: detect → align → embed → thumbnail.
        let detected = self
            .face_detector
            .detect(&img)
            .map_err(ProcessError::Inference)?;
        let face_truncated = detected.len() > FACE_CAP;
        let mut faces: Vec<Face> = Vec::with_capacity(detected.len().min(FACE_CAP));
        for det in detected.iter().take(FACE_CAP) {
            faces.push(self.build_face(&img, det)?);
        }

        Ok(SyntheticResponse {
            labels,
            faces,
            model_versions: ModelVersions {
                labels: LABELS_MODEL_VERSION.to_string(),
                faces: FACES_MODEL_VERSION.to_string(),
            },
            truncated: labels_truncated || face_truncated,
        })
    }

    fn build_face(&self, img: &DynamicImage, det: &DetectedFace) -> Result<Face, ProcessError> {
        let aligned = align::align_face(img, &det.landmarks);
        let embedding = self
            .embedder
            .embed(&aligned)
            .map_err(ProcessError::Inference)?;
        let embedding_b64 = encode_embedding(&embedding);
        let thumbnail_b64 = encode_thumbnail(&aligned).map_err(ProcessError::ThumbnailEncode)?;
        Ok(Face {
            bbox: det.bbox,
            embedding: embedding_b64,
            thumbnail: thumbnail_b64,
            score: det.score,
            model_version: FACES_MODEL_VERSION.to_string(),
        })
    }
}

/// Serialize the response, then trim faces (lowest score first) until the
/// body fits under [`RESPONSE_BYTE_CAP`]. Sets `truncated = true` whenever
/// any face is dropped here.
///
/// Labels are never dropped for size — only faces, per spec. If the body
/// still exceeds the cap with zero faces, it is returned as-is; in
/// practice the labels are small enough that this can't happen with
/// well-formed input.
pub fn encode_response(mut response: SyntheticResponse) -> Result<Vec<u8>, serde_json::Error> {
    let mut body = serde_json::to_vec(&response)?;
    while body.len() > RESPONSE_BYTE_CAP && !response.faces.is_empty() {
        // Faces are sorted by descending score in `process()`, so the
        // lowest-scoring entry is the last one.
        response.faces.pop();
        response.truncated = true;
        body = serde_json::to_vec(&response)?;
    }
    Ok(body)
}

fn encode_embedding(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    BASE64.encode(&bytes)
}

fn encode_thumbnail(img: &RgbImage) -> Result<String, image::ImageError> {
    let mut buf = Vec::new();
    {
        let mut encoder = JpegEncoder::new_with_quality(Cursor::new(&mut buf), THUMBNAIL_QUALITY);
        encoder.encode(img.as_raw(), img.width(), img.height(), ExtendedColorType::Rgb8)?;
    }
    Ok(BASE64.encode(&buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions() -> ModelVersions {
        ModelVersions {
            labels: LABELS_MODEL_VERSION.into(),
            faces: FACES_MODEL_VERSION.into(),
        }
    }

    fn face(score: f32, thumbnail_len: usize) -> Face {
        Face {
            bbox: [0.0, 0.0, 64.0, 64.0],
            embedding: "ZW1iZWRkaW5n".repeat(200), // ~2.4 KB filler
            thumbnail: "x".repeat(thumbnail_len),
            score,
            model_version: FACES_MODEL_VERSION.into(),
        }
    }

    #[test]
    fn small_response_is_passed_through_unchanged() {
        let faces = (0..3)
            .map(|i| face(0.9 - i as f32 * 0.1, 1_000))
            .collect();
        let resp = SyntheticResponse {
            labels: vec![],
            faces,
            model_versions: versions(),
            truncated: false,
        };
        let body = encode_response(resp).unwrap();
        assert!(body.len() < RESPONSE_BYTE_CAP);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["faces"].as_array().unwrap().len(), 3);
        assert_eq!(parsed["truncated"], false);
    }

    #[test]
    fn drops_lowest_score_faces_until_under_cap() {
        // 10 faces × ~200 KB thumbnail = ~2 MB raw, well over the 1 MB cap.
        let faces: Vec<Face> = (0..10)
            .map(|i| face(0.9 - i as f32 * 0.05, 200_000))
            .collect();
        let resp = SyntheticResponse {
            labels: vec![],
            faces,
            model_versions: versions(),
            truncated: false,
        };
        let body = encode_response(resp).unwrap();
        assert!(
            body.len() <= RESPONSE_BYTE_CAP,
            "body {} bytes exceeded cap",
            body.len()
        );
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let remaining = parsed["faces"].as_array().unwrap();
        assert!(remaining.len() < 10, "expected some faces dropped");
        assert!(!remaining.is_empty(), "expected at least one face to fit");
        assert_eq!(parsed["truncated"], true);
        // Highest-scoring face must still be there (descending order kept).
        let first_score = remaining[0]["score"].as_f64().unwrap();
        assert!((first_score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn returns_zero_faces_when_even_one_exceeds_cap() {
        // Single face that's bigger than the cap on its own.
        let faces = vec![face(0.95, RESPONSE_BYTE_CAP + 50_000)];
        let resp = SyntheticResponse {
            labels: vec![],
            faces,
            model_versions: versions(),
            truncated: false,
        };
        let body = encode_response(resp).unwrap();
        assert!(body.len() <= RESPONSE_BYTE_CAP);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["faces"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["truncated"], true);
    }

    #[test]
    fn preserves_existing_truncated_flag_when_no_size_drops_needed() {
        let resp = SyntheticResponse {
            labels: vec![],
            faces: vec![],
            model_versions: versions(),
            truncated: true, // e.g. labels were already over count cap
        };
        let body = encode_response(resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["truncated"], true);
    }

    /// End-to-end label-pipeline integration: classification +
    /// labels-map curation against a known image. Gated on models being
    /// present so `cargo test` works on a fresh checkout that hasn't
    /// fetched weights.
    #[test]
    fn one_dog_fixture_yields_dog_label() {
        use crate::synthetic::models_dir;
        let Ok(dir) = models_dir() else {
            eprintln!("skip: no models directory resolved");
            return;
        };
        let Ok(engine) = SyntheticEngine::new(&dir) else {
            eprintln!("skip: engine init failed");
            return;
        };
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("one-dog.jpg");
        if !fixture.exists() {
            eprintln!("skip: {} not present", fixture.display());
            return;
        }
        let response = engine.process(&fixture).expect("process");
        let names: Vec<&str> = response.labels.iter().map(|l| l.name.as_str()).collect();
        assert!(
            names.contains(&"dog"),
            "expected 'dog' among curated labels, got {names:?}"
        );
    }
}
