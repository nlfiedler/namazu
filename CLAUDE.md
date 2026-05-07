# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Namazu is a small actix-web blob server used by the [tanuki](https://github.com/nlfiedler/tanuki) project. The entire server is a single file (`src/main.rs`); a separate, intentionally tiny crate under `healthcheck/` builds the binary used by the Docker `HEALTHCHECK`. The healthcheck crate is excluded from the main workspace (see `Cargo.toml`'s `[workspace] exclude`) so it can build with a minimal dependency footprint.

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

## Tests

Tests live alongside the code in `src/main.rs` under two modules: `tests` (sync, for pure functions) and `async_tests` (uses `actix_web::test`). The `cfg(test)` block at the top of `main.rs` redirects `DEFAULT_ASSETS_PATH` to `tests/blobs/`, where pre-staged fixtures live (`2019-04-15/0830/f1t.jpg`, `lorem-ipsum.txt`, `fighting_kittens.jpg`). `tests/fixtures/` holds source files copied into the blob store by mutating tests like `test_put_asset_ok` and `test_delete_asset_ok`. Those tests clean up after themselves; if a run is interrupted, leftover files in `tests/blobs/2019-04-15/0830/` (e.g. `f2t.jpg`, `f3t.jpg`) may need to be removed before re-running.

Many tests are gated by `cfg(unix)` / `cfg(windows)` because the path-rejection logic differs between platforms (Windows accepts both `/` and `\` separators, has drive letters, and UNC paths). Keep that pattern when adding tests for new path-validation rules.
