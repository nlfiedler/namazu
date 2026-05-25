//! Fetch ML model artifacts listed in `model-manifest.json` into `models/`.
//!
//! Each entry is verified by SHA256 and byte size. Already-correct files are
//! left alone. Set `NAMAZU_SKIP_MODEL_FETCH=1` to skip all network access
//! (existing files are still verified; missing files emit warnings instead of
//! failing the build) — intended for offline development, sandboxed CI, and
//! Docker builds that pre-stage `models/` by other means.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST_FILE: &str = "model-manifest.json";
const MODELS_DIR: &str = "models";
const SKIP_ENV: &str = "NAMAZU_SKIP_MODEL_FETCH";

#[derive(Deserialize)]
struct Manifest {
    #[allow(dead_code)]
    version: String,
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct FileEntry {
    name: String,
    url: String,
    sha256: String,
    bytes: u64,
}

fn main() {
    println!("cargo:rerun-if-changed={MANIFEST_FILE}");
    println!("cargo:rerun-if-env-changed={SKIP_ENV}");

    if let Err(e) = run() {
        eprintln!("build.rs: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let manifest_path = PathBuf::from(MANIFEST_FILE);
    let raw = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
    let manifest: Manifest =
        serde_json::from_str(&raw).map_err(|e| format!("parse {MANIFEST_FILE}: {e}"))?;

    let models_dir = PathBuf::from(MODELS_DIR);
    fs::create_dir_all(&models_dir)
        .map_err(|e| format!("create {}: {e}", models_dir.display()))?;

    let skip_fetch = std::env::var(SKIP_ENV).map(|v| v == "1").unwrap_or(false);
    if skip_fetch {
        println!("cargo:warning={SKIP_ENV}=1 set; network fetch disabled");
    }

    for entry in &manifest.files {
        process_entry(entry, &models_dir, skip_fetch)?;
    }

    Ok(())
}

fn process_entry(entry: &FileEntry, models_dir: &Path, skip_fetch: bool) -> Result<(), String> {
    let dest = models_dir.join(&entry.name);

    if dest.exists() {
        match verify(&dest, &entry.sha256, entry.bytes) {
            Ok(()) => {
                println!("cargo:warning={}: ok ({} bytes)", entry.name, entry.bytes);
                return Ok(());
            }
            Err(why) => {
                if skip_fetch {
                    println!(
                        "cargo:warning={}: present but {why}; skipped per {SKIP_ENV}",
                        entry.name
                    );
                    return Ok(());
                }
                println!("cargo:warning={}: {why}; re-downloading", entry.name);
            }
        }
    } else if skip_fetch {
        println!(
            "cargo:warning={}: missing; skipped per {SKIP_ENV}",
            entry.name
        );
        return Ok(());
    }

    download(entry, &dest)?;
    verify(&dest, &entry.sha256, entry.bytes).map_err(|why| {
        // Don't leave a half-bad file on disk.
        let _ = fs::remove_file(&dest);
        format!("{}: downloaded file failed verification: {why}", entry.name)
    })?;
    println!("cargo:warning={}: downloaded ({} bytes)", entry.name, entry.bytes);
    Ok(())
}

fn verify(path: &Path, expected_sha: &str, expected_bytes: u64) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| format!("stat: {e}"))?;
    if meta.len() != expected_bytes {
        return Err(format!(
            "size mismatch (expected {} bytes, got {})",
            expected_bytes,
            meta.len()
        ));
    }
    let mut file = fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_sha) {
        return Err(format!(
            "sha256 mismatch (expected {expected_sha}, got {actual})"
        ));
    }
    Ok(())
}

fn download(entry: &FileEntry, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("tmp");
    // Best-effort cleanup of any prior tmp file.
    let _ = fs::remove_file(&tmp);

    println!("cargo:warning={}: fetching {}", entry.name, entry.url);
    let resp = ureq::get(&entry.url)
        .call()
        .map_err(|e| format!("{}: GET {} failed: {e}", entry.name, entry.url))?;

    let mut reader = resp.into_reader();
    let mut out =
        fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("{}: read body: {e}", entry.name))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        hasher.update(&buf[..n]);
        total += n as u64;
        if total > entry.bytes {
            let _ = fs::remove_file(&tmp);
            return Err(format!(
                "{}: server sent more than {} bytes",
                entry.name, entry.bytes
            ));
        }
    }
    out.sync_all()
        .map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(out);

    if total != entry.bytes {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "{}: server sent {} bytes, expected {}",
            entry.name, total, entry.bytes
        ));
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(&entry.sha256) {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "{}: sha256 mismatch (expected {}, got {actual})",
            entry.name, entry.sha256
        ));
    }

    fs::rename(&tmp, dest).map_err(|e| {
        format!(
            "{}: rename {} -> {}: {e}",
            entry.name,
            tmp.display(),
            dest.display()
        )
    })?;
    Ok(())
}
