//! Synthetic data endpoint: ML-derived image classification + face recognition.
//!
//! See `doc/specs/0001-synthetic-data.md` for the contract this module
//! implements. v0 of this module only covers the labels-map loader and
//! curation pipeline; inference, the HTTP handler, and engine wiring land in
//! subsequent slices.

pub mod classifier;
pub mod engine;
pub mod handler;
pub mod labels;
pub mod preprocess;

pub use engine::SyntheticEngine;
pub use handler::post_synthetic;

use std::env;
use std::path::{Path, PathBuf};

/// Resolves the directory containing the ONNX model files and
/// `labels-map.json`.
///
/// Resolution order:
/// 1. `$NAMAZU_MODELS_PATH` if set (operator override; trusted as-is).
/// 2. `<directory-of-current-executable>/models` if it exists.
/// 3. `env!("CARGO_MANIFEST_DIR")/models` if it exists (dev / `cargo test`).
///
/// Existence of the files themselves is the engine's concern; this function
/// only locates the directory.
pub fn models_dir() -> Result<PathBuf, ModelsDirError> {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    resolve_models_dir(
        env::var("NAMAZU_MODELS_PATH").ok(),
        exe_dir.as_deref(),
        Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

fn resolve_models_dir(
    env_path: Option<String>,
    exe_dir: Option<&Path>,
    manifest_dir: &Path,
) -> Result<PathBuf, ModelsDirError> {
    if let Some(p) = env_path {
        return Ok(PathBuf::from(p));
    }
    if let Some(dir) = exe_dir {
        let cand = dir.join("models");
        if cand.is_dir() {
            return Ok(cand);
        }
    }
    let cand = manifest_dir.join("models");
    if cand.is_dir() {
        return Ok(cand);
    }
    Err(ModelsDirError::NotFound)
}

#[derive(Debug, thiserror::Error)]
pub enum ModelsDirError {
    #[error(
        "no models directory found: set NAMAZU_MODELS_PATH, place models/ next to the binary, or run from the cargo workspace"
    )]
    NotFound,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn env_var_takes_precedence_and_is_trusted_as_is() {
        // Even pointed at a path that doesn't exist on disk; resolution
        // returns it unchanged. The engine will report a clear error later.
        let got =
            resolve_models_dir(Some("/no/such/dir".into()), None, Path::new("/tmp")).unwrap();
        assert_eq!(got, PathBuf::from("/no/such/dir"));
    }

    #[test]
    fn falls_back_to_exe_dir_when_env_var_unset() {
        let tmp = tempdir();
        let exe_dir = tmp.path();
        fs::create_dir(exe_dir.join("models")).unwrap();
        let got = resolve_models_dir(None, Some(exe_dir), Path::new("/no/such/manifest")).unwrap();
        assert_eq!(got, exe_dir.join("models"));
    }

    #[test]
    fn falls_back_to_manifest_dir_when_exe_dir_has_no_models() {
        let exe_tmp = tempdir();
        let manifest_tmp = tempdir();
        fs::create_dir(manifest_tmp.path().join("models")).unwrap();
        let got =
            resolve_models_dir(None, Some(exe_tmp.path()), manifest_tmp.path()).unwrap();
        assert_eq!(got, manifest_tmp.path().join("models"));
    }

    #[test]
    fn errors_when_nothing_resolves() {
        let exe_tmp = tempdir();
        let manifest_tmp = tempdir();
        let err = resolve_models_dir(None, Some(exe_tmp.path()), manifest_tmp.path()).unwrap_err();
        assert!(matches!(err, ModelsDirError::NotFound));
    }

    // Minimal tempdir helper to avoid pulling in the `tempfile` crate just for
    // these tests. Uses a per-test unique directory under std::env::temp_dir.
    pub(super) struct TempDir(PathBuf);
    impl TempDir {
        pub fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    pub(super) fn tempdir() -> TempDir {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let p = env::temp_dir().join(format!("namazu-test-{pid}-{n}"));
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}
