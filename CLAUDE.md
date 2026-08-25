# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Namazu is a small actix-web blob server used by the [tanuki](https://github.com/nlfiedler/tanuki) project. Most of the server lives in one file (`src/main.rs`); the ML-backed `/synthetic` endpoint lives in its own `src/synthetic/` module (see Architecture below). A separate, intentionally tiny crate under `healthcheck/` builds the binary used by the Docker `HEALTHCHECK`. The healthcheck crate is excluded from the main workspace (see `Cargo.toml`'s `[workspace] exclude`) so it can build with a minimal dependency footprint.

## Common commands

```bash
# Run all tests (uses tests/blobs as ASSETS_PATH via cfg(test))
cargo test

# Run a single test
cargo test test_get_asset_find_jpeg
cargo test async_tests::test_put_asset_ok -- --nocapture

# Run the server locally (defaults: 127.0.0.1:3000, ASSETS_PATH=tmp/blobs)
RUST_LOG=info cargo run

# Build release binary
cargo build --release

# Build the Docker image (multi-stage: namazu + healthcheck)
docker build -t namazu-app .
```

The healthcheck binary lives in its own workspace and must be built separately when working on it: `cd healthcheck && cargo build`.

## Architecture

**Identifier encoding.** Asset IDs in URLs are Base64URL-encoded (no padding) relative paths like `2019-04-15/0830/f1t.jpg`. `blob_path()` is the single chokepoint that decodes the ID and rejects anything dangerous before any filesystem operation: non-UTF-8, control characters, `..` segments, and absolute/rooted paths (both Unix `/...` and Windows `\...`, `C:\...`, `\\server\share`). All four route handlers (`get_asset`, `put_asset`, `delete_asset`, `get_thumbnail`) funnel IDs through `blob_path()` and respond `400 Bad Request` on failure — when adding a new route that takes an ID, do the same. The decoded string is `trim()`-ed to tolerate trailing newlines that base64 tools commonly add.

**Caching and range support come from `actix_files::NamedFile`.** `get_asset` delegates ETag generation, `If-None-Match` → 304, and `Range` → 206 to `NamedFile`; do not hand-roll these. The `?attachment=...` query param flips `Content-Disposition` from `inline` to `attachment` and adds `X-Download-Options: noopen`.

**Thumbnails** (`get_thumbnail` + `create_thumbnail`) are computed synchronously inside `web::block` (CPU-bound work off the async runtime). The ETag is a strong tag derived from `width:height:identifier` and is checked *before* decoding the image, so 304s are cheap. EXIF orientation is read via `kamadak-exif` and applied with `correct_orientation` — when the image is sideways (orientation > 4) the thumbnail bounds are swapped before resizing so aspect ratio works out. On thumbnail failure other than `NotFound`, the handler 307-redirects to `/public/placeholder.svg` rather than returning an error.

**Video thumbnails/previews** are produced by shelling out to `ffmpeg` (must be on `PATH` at runtime — the Docker image installs it). `is_video()` dispatches by extension (mp4, mov, m4v, mkv, webm, avi, mpeg, mpg) inside `create_thumbnail` and `create_preview`; `extract_video_frame` runs `ffmpeg ... -vf thumbnail,scale=...` and pipes a single MJPEG frame to stdout. The `thumbnail` filter samples ~100 frames and picks the most representative one (avoids black/fade-in frames). `extract_video_frame` opens the source file once before invoking ffmpeg so a missing asset surfaces as `io::ErrorKind::NotFound` (→ 404) rather than getting lumped in with other ffmpeg failures (→ placeholder redirect). When ffmpeg itself isn't on `PATH`, the spawn-time NotFound is rewritten to a generic error so the handler returns a placeholder instead of a 404.

**PUT semantics.** `put_asset` uses `OpenOptions::create_new(true)` so re-uploads return `409 Conflict` — there is no overwrite path. On Unix it `chmod 0644`s the file after writing so external backup tooling can read it. The body is streamed (`web::Payload` + `tokio::io::AsyncWriteExt`) rather than buffered.

**The `/synthetic` endpoint** (`src/synthetic/`, wired in `main.rs` via `synthetic::post_synthetic`) runs ML inference — MobileNetV2 image classification, SCRFD face detection, and MobileFaceNet face embeddings — on a stored image blob. It shares `blob_path()` and `is_video()` with the other routes for ID validation and file-type dispatch, and reuses `correct_orientation`/`get_image_orientation` from `main.rs` for EXIF-aware decoding (`synthetic::preprocess::decode_oriented`). See `doc/specs/0001-synthetic-data.md` for the response contract.

**Model loading is fatal-at-startup, not per-request.** `synthetic::models_dir()` resolves the model directory in order: `$NAMAZU_MODELS_PATH` (trusted as-is), then `<exe-dir>/models`, then `$CARGO_MANIFEST_DIR/models` (dev/`cargo test`). `main()` calls this and `SyntheticEngine::new()` once before the server starts listening; either failing calls `std::process::exit(1)` rather than serving degraded responses. `build.rs` downloads the four artifacts listed in `model-manifest.json` (MobileNetV2, SCRFD-2.5g, MobileFaceNet, `labels-map.json`) from Tanuki's GitHub Releases at build time, SHA256-verifying each; `NAMAZU_SKIP_MODEL_FETCH=1` skips network access (existing files are still verified, missing ones just warn) for offline dev, sandboxed CI, and Docker builds that pre-stage `models/`.

**Concurrency is bounded by a semaphore sized to `available_parallelism()`, not a fixed constant**, because each `ort::Session::run()` takes `&mut self` (each of `Classifier`/`FaceDetector`/`Embedder` wraps its session in a `Mutex`) and ORT's own intra-op parallelism already saturates all cores per inference — queuing at the engine level avoids piling requests up on the blocking thread pool where they'd contend for those mutexes instead of running. `post_synthetic` acquires a permit before `web::block`, then moves it *into* the blocking closure so it stays held for the inference's full duration even if `tokio::time::timeout`'s 30s (`engine::PROCESSING_TIMEOUT`) fires and the awaiting future is dropped — otherwise ORT would keep chewing on an abandoned request.

**The processing pipeline enforces size limits before and after decode.** `decode_oriented` sets the image decoder's strict `max_image_width`/`max_image_height`/`max_alloc` limits to `MAX_DIM` (8000px) so a decompression-bomb-style image is rejected before any pixel buffer is allocated (surfaces as `image::ImageError::Limits`, mapped to `413` with `width`/`height: None`); a second check after decode catches images that passed the decoder's limits but still exceed `MAX_DIM` (`413` with dimensions populated). Labels are curated against `labels-map.json` (bucketed by mapped name, `is_person` entries and sub-threshold scores dropped, max score kept per bucket, capped at `LABEL_CAP` = 20) and never dropped for response size — only face count is. Detected faces are 5-point aligned (`align.rs`, least-squares similarity transform onto the 112×112 ArcFace template) before embedding, and the total response body is capped at `RESPONSE_BYTE_CAP` (1MB): `encode_response` pops faces lowest-score-first until it fits, setting `truncated: true`.

## Tests

Tests live alongside the code in `src/main.rs` under two modules: `tests` (sync, for pure functions) and `async_tests` (uses `actix_web::test`). The `cfg(test)` block at the top of `main.rs` redirects `DEFAULT_ASSETS_PATH` to `tests/blobs/`, where pre-staged fixtures live (`2019-04-15/0830/f1t.jpg`, `lorem-ipsum.txt`, `fighting_kittens.jpg`). `tests/fixtures/` holds source files copied into the blob store by mutating tests like `test_put_asset_ok` and `test_delete_asset_ok`. Those tests clean up after themselves; if a run is interrupted, leftover files in `tests/blobs/2019-04-15/0830/` (e.g. `f2t.jpg`, `f3t.jpg`) may need to be removed before re-running.

Many tests are gated by `cfg(unix)` / `cfg(windows)` because the path-rejection logic differs between platforms (Windows accepts both `/` and `\` separators, has drive letters, and UNC paths). Keep that pattern when adding tests for new path-validation rules.
