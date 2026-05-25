//! MobileNetV2 inference wrapped around a tract typed runnable plan.

use std::path::Path;

use tract_onnx::prelude::*;

use super::preprocess::CLASSIFIER_INPUT;

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct Classifier {
    plan: Plan,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("load mobilenet_v2.onnx ({path}): {source}")]
    Tract {
        path: String,
        #[source]
        source: TractError,
    },
}

impl Classifier {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let map_err = |source| LoadError::Tract {
            path: path.display().to_string(),
            source,
        };
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(map_err)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(1, 3, CLASSIFIER_INPUT, CLASSIFIER_INPUT),
                ),
            )
            .map_err(map_err)?
            .into_optimized()
            .map_err(map_err)?
            .into_runnable()
            .map_err(map_err)?;
        Ok(Self { plan })
    }

    /// Run inference and return softmax probabilities for the 1000 ImageNet
    /// classes. The ONNX Model Zoo MobileNetV2 emits raw logits, so we apply
    /// softmax here to get the `[0, 1]` scores the curation pipeline expects.
    pub fn infer(&self, input: tract_ndarray::Array4<f32>) -> TractResult<Vec<f32>> {
        let tensor: Tensor = input.into();
        let result = self.plan.run(tvec!(tensor.into()))?;
        let view = result[0].to_array_view::<f32>()?;
        let logits: Vec<f32> = view.iter().copied().collect();
        Ok(softmax(&logits))
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
        // index 2 (logit 3.0) should be the largest.
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
