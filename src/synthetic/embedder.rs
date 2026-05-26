//! MobileFaceNet inference: 112×112 aligned face → 128-D L2-normalized
//! embedding.

use std::path::Path;
use std::sync::Mutex;

use image::RgbImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;

use super::align::ALIGNED_SIZE;

/// Output embedding dimensionality for the current MobileFaceNet checkpoint
/// (`mobilefacenet.onnx`, ArcFace-trained, 512-D). Kept for documentation;
/// the runtime trusts whatever the model emits.
#[allow(dead_code)]
pub const EMBEDDING_DIM: usize = 512;

pub struct Embedder {
    /// `Session::run()` takes `&mut self`; ORT's intra-op parallelism keeps
    /// a single inference busy on all cores.
    session: Mutex<Session>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("load mobilefacenet.onnx ({path}): {source}")]
    Ort {
        path: String,
        #[source]
        source: ort::Error,
    },
}

impl Embedder {
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

    /// Run MobileFaceNet on an aligned 112×112 RGB face and return the L2-
    /// normalized embedding. The output length is model-dependent (128 for
    /// this checkpoint).
    pub fn embed(&self, aligned: &RgbImage) -> ort::Result<Vec<f32>> {
        let (w, h) = (aligned.width(), aligned.height());
        debug_assert_eq!((w, h), (ALIGNED_SIZE, ALIGNED_SIZE));

        // ArcFace standard normalization: (px - 127.5) / 128.
        let mut arr = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        for (x, y, px) in aligned.enumerate_pixels() {
            for c in 0..3 {
                arr[[0, c, y as usize, x as usize]] = (px[c] as f32 - 127.5) / 128.0;
            }
        }

        let tensor = Tensor::from_array(arr)?;
        let mut session = self.session.lock().expect("embedder session poisoned");
        let outputs = session.run(ort::inputs![tensor])?;
        let (_shape, embedding) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(l2_normalize(embedding))
    }
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector_is_unchanged() {
        let v = [1.0_f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert_eq!(n[1], 0.0);
        assert_eq!(n[2], 0.0);
    }

    #[test]
    fn l2_normalize_makes_norm_one() {
        let v = [3.0_f32, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_zero() {
        let v = [0.0_f32; 5];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }
}
