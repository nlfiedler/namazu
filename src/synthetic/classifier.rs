//! MobileNetV2 inference via ONNX Runtime (`ort`).

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

pub struct Classifier {
    /// `Session::run()` takes `&mut self`, so we serialize access here.
    /// ORT's intra-op parallelism keeps a single inference busy across all
    /// cores, so serializing requests doesn't sacrifice throughput.
    session: Mutex<Session>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("load mobilenet_v2.onnx ({path}): {source}")]
    Ort {
        path: String,
        #[source]
        source: ort::Error,
    },
}

impl Classifier {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let map_err = |source| LoadError::Ort {
            path: path.display().to_string(),
            source,
        };
        let session = Session::builder()
            .map_err(map_err)?
            .commit_from_file(path)
            .map_err(map_err)?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run inference and return softmax probabilities for the 1000 ImageNet
    /// classes. MobileNetV2 emits raw logits; we softmax here to get the
    /// `[0, 1]` scores the curation pipeline expects.
    pub fn infer(&self, input: Array4<f32>) -> ort::Result<Vec<f32>> {
        let tensor = Tensor::from_array(input)?;
        let mut session = self.session.lock().expect("classifier session poisoned");
        let outputs = session.run(ort::inputs![tensor])?;
        let (_shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(softmax(logits))
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.into_iter().map(|v| v / sum).collect()
    } else {
        // All -inf logits; degenerate but don't panic.
        vec![0.0; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let logits = [1.0_f32, 2.0, 3.0, 0.5];
        let p = softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let (idx, _) = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(idx, 2);
    }

    #[test]
    fn softmax_handles_constant_logits() {
        let logits = [3.0_f32; 5];
        let p = softmax(&logits);
        for v in &p {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }
}
