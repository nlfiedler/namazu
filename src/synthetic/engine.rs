//! Top-level synthetic engine: holds loaded models + labels-map, exposes a
//! single `process()` entry point used by the HTTP handler.
//!
//! v0 (this slice) only runs the label classifier; faces always come back as
//! an empty array. Subsequent slices fill in face detection, alignment, and
//! embedding.

use std::path::Path;
use std::sync::Arc;

use image::GenericImageView;
use serde::Serialize;

use super::classifier::{self, Classifier};
use super::labels::{self, CurationResult, Label, LabelsMap};
use super::preprocess;

/// Maximum input dimension (per spec): images larger than this are rejected
/// with 413.
pub const MAX_DIM: u32 = 8000;

/// Spec-mandated cap on the curated labels list.
pub const LABEL_CAP: usize = 20;

/// Spec-mandated cap on the faces list (kept here for completeness; no faces
/// emitted yet).
#[allow(dead_code)]
pub const FACE_CAP: usize = 20;

const LABELS_MODEL_VERSION: &str = "mobilenetv2-v1";
const FACES_MODEL_VERSION: &str = "mobilefacenet-v1";

pub struct SyntheticEngine {
    labels_map: Arc<LabelsMap>,
    classifier: Arc<Classifier>,
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error(transparent)]
    LabelsMap(#[from] labels::LabelsMapError),
    #[error(transparent)]
    Classifier(#[from] classifier::LoadError),
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
    #[error("could not decode image: {0}")]
    Decode(#[from] image::ImageError),
    #[error("image too large: {0}x{1}")]
    TooLarge(u32, u32),
    #[error("inference failed: {0}")]
    Inference(#[source] tract_onnx::prelude::TractError),
}

#[derive(Debug, Serialize)]
pub struct SyntheticResponse {
    pub labels: Vec<Label>,
    /// Always empty for now; populated in a later slice.
    pub faces: Vec<serde_json::Value>,
    pub model_versions: ModelVersions,
    pub truncated: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelVersions {
    pub labels: String,
    pub faces: String,
}

impl SyntheticEngine {
    /// Load all required model artifacts from `models_dir`. Failure is fatal
    /// to the caller (per spec).
    pub fn new(models_dir: &Path) -> Result<Self, InitError> {
        let labels_map = LabelsMap::load(&models_dir.join("labels-map.json"))?;
        let classifier = Classifier::load(&models_dir.join("mobilenet_v2.onnx"))?;
        Ok(Self {
            labels_map: Arc::new(labels_map),
            classifier: Arc::new(classifier),
        })
    }

    /// Synchronous processing pipeline. Designed to run inside `web::block`.
    pub fn process(&self, image_path: &Path) -> Result<SyntheticResponse, ProcessError> {
        let img = preprocess::decode_oriented(image_path)?;
        let (w, h) = img.dimensions();
        if w > MAX_DIM || h > MAX_DIM {
            return Err(ProcessError::TooLarge(w, h));
        }

        let input = preprocess::classifier_input(&img);
        let probs = self
            .classifier
            .infer(input)
            .map_err(ProcessError::Inference)?;
        let scored: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect();
        let CurationResult { labels, truncated } =
            labels::curate(&scored, &self.labels_map, LABEL_CAP);

        Ok(SyntheticResponse {
            labels,
            faces: Vec::new(),
            model_versions: ModelVersions {
                labels: LABELS_MODEL_VERSION.to_string(),
                faces: FACES_MODEL_VERSION.to_string(),
            },
            truncated,
        })
    }
}
