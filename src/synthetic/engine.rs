//! Top-level synthetic engine: holds loaded models + labels-map, exposes a
//! single `process()` entry point used by the HTTP handler.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use image::{DynamicImage, ExtendedColorType, GenericImageView, RgbImage, codecs::jpeg::JpegEncoder};
use serde::Serialize;

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

/// JPEG quality for face thumbnails (~85 per spec).
const THUMBNAIL_QUALITY: u8 = 85;

const LABELS_MODEL_VERSION: &str = "mobilenetv2-v1";
const FACES_MODEL_VERSION: &str = "mobilefacenet-v1";

pub struct SyntheticEngine {
    labels_map: Arc<LabelsMap>,
    classifier: Arc<Classifier>,
    face_detector: Arc<FaceDetector>,
    embedder: Arc<Embedder>,
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
        Ok(Self {
            labels_map: Arc::new(labels_map),
            classifier: Arc::new(classifier),
            face_detector: Arc::new(face_detector),
            embedder: Arc::new(embedder),
        })
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
