//
// Copyright (c) 2025 Nathan Fiedler
//
use actix_files::NamedFile;
use actix_web::http::header;
use actix_web::{App, Either, HttpMessage, HttpRequest, HttpResponse, HttpServer, Responder, web};
use anyhow::{Error, anyhow};
use base64::{Engine as _, engine::general_purpose};
use log::{info, warn};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

mod synthetic;

#[cfg(test)]
pub(crate) const DEFAULT_ASSETS_PATH: &str = "tests/blobs";
#[cfg(not(test))]
const DEFAULT_ASSETS_PATH: &str = "tmp/blobs";

static ASSETS_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    let path = env::var("ASSETS_PATH").unwrap_or_else(|_| DEFAULT_ASSETS_PATH.to_owned());
    PathBuf::from(path)
});

fn has_control_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_control())
}

/// Convert the identifier to a file path within the assets store.
pub(crate) fn blob_path(encoded: &str) -> Result<PathBuf, Error> {
    let decoded = general_purpose::URL_SAFE_NO_PAD.decode(encoded)?;
    // Windows will raise unexpected errors if the path has any trailing EOL
    // characters which is very easy to do with base64 encoding.
    let as_string = str::from_utf8(&decoded)?.trim();
    if has_control_chars(as_string) {
        return Err(anyhow!("control characters not allowed"));
    }
    if as_string.contains("..") {
        return Err(anyhow!("relative paths not allowed"));
    }
    let rel_path = Path::new(as_string);
    if rel_path.has_root() {
        return Err(anyhow!("root path not allowed"));
    }
    // Note also that on Windows both slash and backslash can be used in the
    // same path without any issues. Only the trailing EOL causes problems.
    let mut full_path = ASSETS_PATH.to_path_buf();
    full_path.push(rel_path);
    Ok(full_path)
}

/// Return the last part of the path, converting to a String.
fn get_file_name(path: &Path) -> String {
    // ignore any paths that end in '..' and ignore any paths that failed UTF-8
    // translation; if normal conversion failed, use lossy conversion
    path.file_name()
        .and_then(|p| p.to_str())
        .map(|s| s.to_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// True if the path's extension matches a recognized video container.
pub(crate) fn is_video(filepath: &Path) -> bool {
    filepath
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .map(|e| {
            matches!(
                e.as_str(),
                "mp4" | "mov" | "m4v" | "mkv" | "webm" | "avi" | "mpeg" | "mpg" | "qt"
            )
        })
        .unwrap_or(false)
}

/// Extract a representative frame from a video as JPEG bytes via ffmpeg.
///
/// `vf` is the `-vf` filter chain (typically begins with `thumbnail` for
/// representative-frame selection). A missing source file produces an
/// `io::ErrorKind::NotFound` so the route handler can map it to a 404; any
/// other failure (ffmpeg not on PATH, decode error, unsupported codec) returns
/// a generic error so the handler redirects to the placeholder.
fn extract_video_frame(filepath: &Path, vf: &str) -> Result<Vec<u8>, Error> {
    use std::process::Command;
    // Surface a missing source as ErrorKind::NotFound before invoking ffmpeg —
    // ffmpeg's own exit status would otherwise be indistinguishable from any
    // other decode failure.
    let _ = std::fs::File::open(filepath)?;
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error"])
        .arg("-i")
        .arg(filepath)
        .args([
            "-frames:v", "1", "-vf", vf, "-f", "image2", "-vcodec", "mjpeg", "-",
        ])
        .output()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                anyhow!("ffmpeg binary not found in PATH")
            } else {
                anyhow!("failed to invoke ffmpeg: {e}")
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "ffmpeg exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    if output.stdout.is_empty() {
        return Err(anyhow!("ffmpeg produced no output"));
    }
    Ok(output.stdout)
}

/// Produce a thumbnail for the given asset that fits within the bounds given
/// while maintaining aspect ratio. Image assets are decoded with the `image`
/// crate; video assets are handled by ffmpeg's `thumbnail` filter.
///
/// If the asset cannot be decoded, or any other problem arises, returns an
/// error.
fn create_thumbnail(filepath: &Path, width: u32, height: u32) -> Result<Vec<u8>, Error> {
    if is_video(filepath) {
        let vf = format!(
            "thumbnail,scale='min(iw,{width})':'min(ih,{height})':force_original_aspect_ratio=decrease"
        );
        return extract_video_frame(filepath, &vf);
    }
    let mut cursor = std::io::Cursor::new(Vec::new());
    // The image crate does not recognize .jpe extension as jpeg, so use the
    // format guessing code based on the first few bytes.
    let mut img = image::ImageReader::open(filepath)?
        .with_guessed_format()?
        .decode()?;
    match get_image_orientation(filepath) {
        Ok(orientation) => {
            // c.f. https://magnushoff.com/articles/jpeg-orientation/
            if orientation > 4 {
                // image is sideways, need to swap new width/height
                img = img.thumbnail(height, width);
            } else {
                img = img.thumbnail(width, height);
            }
            img = correct_orientation(orientation, img);
        }
        Err(e) => {
            warn!("EXIF reading failed: {e:#}");
            img = img.thumbnail(width, height);
        }
    }
    // The image crate's JpegEncoder will use a quality factor of 75 by default,
    // which yields very good results (libvips uses the same default).
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
    Ok(cursor.into_inner())
}

/// Produce a JPEG preview of the given asset, constrained to the given
/// displayed width or height (after EXIF orientation correction for images),
/// with the other dimension scaled to maintain aspect ratio. Exactly one of
/// `width` or `height` must be `Some`. Image assets are decoded with the
/// `image` crate; video assets are handled by ffmpeg.
///
/// If the asset cannot be decoded, or any other problem arises, returns an
/// error.
fn create_preview(
    filepath: &Path,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<Vec<u8>, Error> {
    if is_video(filepath) {
        let vf = match (width, height) {
            (Some(w), None) => format!("thumbnail,scale='min(iw,{w})':-2"),
            (None, Some(h)) => format!("thumbnail,scale=-2:'min(ih,{h})'"),
            _ => return Err(anyhow!("must specify exactly one of width or height")),
        };
        return extract_video_frame(filepath, &vf);
    }
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut img = image::ImageReader::open(filepath)?
        .with_guessed_format()?
        .decode()?;
    let (w, h) = (img.width(), img.height());
    let orientation = get_image_orientation(filepath).ok();
    // When the image is sideways (orientation > 4), EXIF rotation swaps the
    // stored width and height for display. Compute the desired displayed size,
    // then map back to stored coordinates so thumbnail() resizes correctly
    // before correct_orientation rotates the result.
    let sideways = orientation.map(|o| o > 4).unwrap_or(false);
    let (disp_w, disp_h) = if sideways { (h, w) } else { (w, h) };
    let (target_disp_w, target_disp_h) = match (width, height) {
        (Some(tw), None) => (
            tw,
            (tw as u64 * disp_h as u64 / disp_w.max(1) as u64) as u32,
        ),
        (None, Some(th)) => (
            (th as u64 * disp_w as u64 / disp_h.max(1) as u64) as u32,
            th,
        ),
        _ => return Err(anyhow!("must specify exactly one of width or height")),
    };
    let (target_w, target_h) = if sideways {
        (target_disp_h, target_disp_w)
    } else {
        (target_disp_w, target_disp_h)
    };
    img = img.thumbnail(target_w, target_h);
    if let Some(o) = orientation {
        img = correct_orientation(o, img);
    }
    img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
    Ok(cursor.into_inner())
}

/// Maximum size of the JSON metadata response, in bytes. Files producing more
/// than this (typically due to oversized MakerNote/UserComment fields or a
/// pathological video container) get a 413 instead.
const METADATA_MAX_BYTES: usize = 256 * 1024;

/// Extract metadata as a JSON value from the asset file.
///
/// Videos are dispatched to `ffprobe`; everything else is treated as a
/// potential image and run through the EXIF reader. Files with no EXIF (plain
/// text, images without an EXIF header, etc.) produce an empty JSON object so
/// callers can treat the response uniformly. A missing source file surfaces as
/// `io::ErrorKind::NotFound` so the route handler can map it to 404.
fn extract_metadata(filepath: &Path) -> Result<serde_json::Value, Error> {
    if is_video(filepath) {
        return extract_video_metadata(filepath);
    }
    // Ensure missing files surface as NotFound rather than getting masked by
    // the EXIF reader's "no EXIF -> empty {}" fallback.
    let file = std::fs::File::open(filepath)?;
    match extract_image_metadata(file) {
        Ok(v) => Ok(v),
        Err(e) => {
            // Treat format-level EXIF errors (no header, malformed, not an
            // image) as "no metadata available" — but let real I/O errors
            // bubble up so they surface as 500 instead of a misleading 200.
            if let Some(exif::Error::Io(_)) = e.downcast_ref::<exif::Error>() {
                return Err(e);
            }
            log::debug!("no EXIF metadata for {}: {e:#}", filepath.display());
            // Fall back to the image crate so clients still get the basic
            // PixelXDimension/PixelYDimension pair for images that ship
            // without any EXIF header. Non-images (text files, etc.) will
            // fail here and produce an empty object as before.
            Ok(extract_image_dimensions(filepath)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())))
        }
    }
}

/// Read just the image header to recover width/height, formatted to match the
/// shape kamadak-exif would have produced for PixelXDimension/PixelYDimension.
fn extract_image_dimensions(filepath: &Path) -> Result<serde_json::Value, Error> {
    let (width, height) = image::ImageReader::open(filepath)?
        .with_guessed_format()?
        .into_dimensions()?;
    let mut map = serde_json::Map::new();
    map.insert("PixelXDimension".to_string(), pixel_dimension_entry(width));
    map.insert("PixelYDimension".to_string(), pixel_dimension_entry(height));
    Ok(serde_json::Value::Object(map))
}

fn pixel_dimension_entry(pixels: u32) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    entry.insert(
        "description".to_string(),
        serde_json::Value::String(format!("{pixels} pixels")),
    );
    entry.insert(
        "value".to_string(),
        serde_json::Value::Array(vec![serde_json::Value::Number(pixels.into())]),
    );
    serde_json::Value::Object(entry)
}

fn extract_video_metadata(filepath: &Path) -> Result<serde_json::Value, Error> {
    use std::process::Command;
    // Surface a missing source as ErrorKind::NotFound before invoking ffprobe.
    let _ = std::fs::File::open(filepath)?;
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(filepath)
        .output()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                anyhow!("ffprobe binary not found in PATH")
            } else {
                anyhow!("failed to invoke ffprobe: {e}")
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "ffprobe exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    // Bail before parsing if the extractor produced more bytes than we'd ever
    // be willing to return — avoids building a multi-MB serde_json::Value tree
    // just to 413 afterwards.
    if output.stdout.len() > METADATA_MAX_BYTES {
        return Err(MetadataTooLarge(output.stdout.len()).into());
    }
    // Parse + re-serialize so a corrupt extractor stdout becomes an error
    // instead of being relayed to the client as garbage.
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    Ok(parsed)
}

/// Marker error type so the route handler can map "extractor exceeded the
/// size cap" to a 413 instead of a generic 500.
#[derive(Debug)]
struct MetadataTooLarge(usize);

impl std::fmt::Display for MetadataTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "metadata exceeds {METADATA_MAX_BYTES} bytes (got {})", self.0)
    }
}

impl std::error::Error for MetadataTooLarge {}

fn extract_image_metadata(file: std::fs::File) -> Result<serde_json::Value, Error> {
    let mut buffer = std::io::BufReader::new(&file);
    let reader = exif::Reader::new();
    let exif = reader.read_from_container(&mut buffer)?;
    let mut map = serde_json::Map::new();
    for field in exif.fields() {
        // Only the primary IFD describes the image itself; the thumbnail IFD
        // contains a (potentially large) embedded JPEG that is useless to
        // downstream consumers.
        if field.ifd_num != exif::In::PRIMARY {
            continue;
        }
        // MakerNote is vendor-specific opaque bytes and can be hundreds of KB.
        if field.tag == exif::Tag::MakerNote {
            continue;
        }
        let key = field.tag.to_string();
        let description = field.display_value().with_unit(&exif).to_string();
        let raw_value = exif_value_to_json(&field.value);
        let mut entry = serde_json::Map::new();
        entry.insert(
            "description".to_string(),
            serde_json::Value::String(description),
        );
        entry.insert("value".to_string(), raw_value);
        map.insert(key, serde_json::Value::Object(entry));
    }
    Ok(serde_json::Value::Object(map))
}

// Convert a kamadak-exif `Value` into the JSON shape downstream consumers
// (notably tanuki's `parseImageTags`) expect: rationals as `[num, den]` so
// GPS coordinates can be reconstructed, integers as JSON numbers, ASCII as
// UTF-8 strings (lossy on non-UTF8 bytes). Single-element fields are kept
// inside the array — callers handle both.
fn exif_value_to_json(v: &exif::Value) -> serde_json::Value {
    use serde_json::Number;
    use serde_json::Value as J;
    let from_f64 =
        |n: f64| Number::from_f64(n).map(J::Number).unwrap_or(J::Null);
    match v {
        exif::Value::Byte(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::SByte(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::Short(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::SShort(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::Long(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::SLong(xs) => J::Array(xs.iter().map(|n| J::from(*n)).collect()),
        exif::Value::Float(xs) => {
            J::Array(xs.iter().map(|n| from_f64(*n as f64)).collect())
        }
        exif::Value::Double(xs) => J::Array(xs.iter().map(|n| from_f64(*n)).collect()),
        exif::Value::Rational(xs) => J::Array(
            xs.iter()
                .map(|r| J::Array(vec![J::from(r.num), J::from(r.denom)]))
                .collect(),
        ),
        exif::Value::SRational(xs) => J::Array(
            xs.iter()
                .map(|r| J::Array(vec![J::from(r.num), J::from(r.denom)]))
                .collect(),
        ),
        // EXIF ASCII is a sequence of NUL-separated byte strings. Decode each
        // lossily so non-UTF8 bytes don't fail the whole response.
        exif::Value::Ascii(parts) => J::Array(
            parts
                .iter()
                .map(|bs| J::String(String::from_utf8_lossy(bs).into_owned()))
                .collect(),
        ),
        // Undefined/Unknown carry opaque bytes; surface as a number array
        // for completeness. Most consumers ignore these and read `description`.
        exif::Value::Undefined(bs, _) => {
            J::Array(bs.iter().map(|n| J::from(*n)).collect())
        }
        exif::Value::Unknown(_, _, _) => J::Null,
    }
}

/// Extract the EXIF orientation value from the asset, if any.
pub(crate) fn get_image_orientation(filepath: &Path) -> Result<u16, Error> {
    let file = std::fs::File::open(filepath)?;
    let mut buffer = std::io::BufReader::new(&file);
    let reader = exif::Reader::new();
    let exif = reader.read_from_container(&mut buffer)?;
    let field = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .ok_or_else(|| anyhow!("no orientation field"))?;
    if let exif::Value::Short(data) = &field.value {
        return Ok(data[0]);
    }
    Err(anyhow!("not an image"))
}

/// Flip and/or rotate the image to have the correct rendering.
///
/// The orientation value should be as read from the EXIF header.
pub(crate) fn correct_orientation(orientation: u16, img: image::DynamicImage) -> image::DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.flipv().rotate90(),
        6 => img.rotate90(),
        7 => img.fliph().rotate90(),
        8 => img.rotate270(),
        _ => img,
    }
}

async fn index() -> &'static str {
    "See the README.md file for more information."
}

async fn get_asset(
    info: web::Path<String>,
    query: web::Query<HashMap<String, String>>,
) -> actix_web::Result<impl Responder> {
    let wants_attachment = query.get("attachment").is_some();
    if let Ok(filepath) = blob_path(&info) {
        // NamedFile will generate an ETag and respond to If-None-Match with 304
        // Not Modified, and respond to Range requests with 206 Partial Content
        // and a Content-Range header.
        match NamedFile::open(&filepath) {
            Ok(named_file) => {
                let responder = if wants_attachment {
                    // Add Content-Disposition to encourage the browser to save
                    // the file to disk rather than showing the content directly
                    // in the browser. The `download` attribute on the anchor
                    // (A) tag has no effect when the URL is for a different
                    // origin.
                    let filename = get_file_name(&filepath);
                    named_file
                        .set_content_disposition(header::ContentDisposition::attachment(filename))
                        .customize()
                        .insert_header(("X-Download-Options", "noopen"))
                        .insert_header(("Cache-Control", "public, max-age=31536000"))
                } else {
                    named_file
                        .customize()
                        .insert_header(("Cache-Control", "public, max-age=31536000"))
                };
                Ok(Either::Left(responder))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Ok(Either::Right(HttpResponse::NotFound().finish()))
            }
            Err(e) => Err(e.into()),
        }
    } else {
        // if path conversion fails, probably client error
        Ok(Either::Right(HttpResponse::BadRequest().finish()))
    }
}

// Produce a thumbnail for the asset of the requested size.
async fn get_thumbnail(req: HttpRequest) -> actix_web::Result<HttpResponse> {
    // => /thumbnail/{w}/{h}/{id}
    let width: u32 = req
        .match_info()
        .get("w")
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("invalid width"))?;
    let height: u32 = req
        .match_info()
        .get("h")
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("invalid height"))?;
    let identifier: String = req
        .match_info()
        .get("id")
        .ok_or_else(|| actix_web::error::ErrorBadRequest("missing id"))?
        .to_owned();
    let etag_value = format!("{}:{}:{}", width, height, &identifier);
    let etag: header::EntityTag = header::EntityTag::new_strong(etag_value);
    if none_match(&etag, &req) {
        if let Ok(filepath) = blob_path(&identifier) {
            let result = web::block(move || create_thumbnail(&filepath, width, height)).await?;
            match result {
                Ok(data) => Ok(HttpResponse::Ok()
                    .content_type("image/jpeg")
                    .append_header((header::CONTENT_LENGTH, data.len() as u64))
                    .append_header((header::ETAG, etag))
                    .body(data)),
                Err(e) => {
                    let is_not_found = e
                        .chain()
                        .filter_map(|cause| cause.downcast_ref::<io::Error>())
                        .any(|io_err| io_err.kind() == io::ErrorKind::NotFound);
                    if is_not_found {
                        return Ok(HttpResponse::NotFound().finish());
                    }
                    warn!("thumbnail generation failed: {e:#}");
                    Ok(HttpResponse::TemporaryRedirect()
                        .append_header((header::LOCATION, "/public/placeholder.svg"))
                        .finish())
                }
            }
        } else {
            // if path conversion fails, probably client error
            Ok(HttpResponse::BadRequest().finish())
        }
    } else {
        Ok(HttpResponse::NotModified().finish())
    }
}

// Produce a JPEG preview constrained to a displayed width or height (in pixels)
// supplied via `?width=N` or `?height=N`. Exactly one must be provided.
async fn get_preview(
    info: web::Path<String>,
    query: web::Query<HashMap<String, String>>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let width = query.get("width").and_then(|v| v.parse::<u32>().ok());
    let height = query.get("height").and_then(|v| v.parse::<u32>().ok());
    if width.is_some() == height.is_some() {
        // both or neither — ambiguous request
        return Ok(HttpResponse::BadRequest().finish());
    }
    let identifier = info.into_inner();
    // "p:" prefix keeps preview ETags disjoint from /thumbnail's "{w}:{h}:{id}".
    // Absent dimension is encoded as "_" so width=N and height=N can't collide.
    let etag_value = format!(
        "p:{}:{}:{}",
        width.map(|n| n.to_string()).unwrap_or_else(|| "_".into()),
        height.map(|n| n.to_string()).unwrap_or_else(|| "_".into()),
        &identifier,
    );
    let etag: header::EntityTag = header::EntityTag::new_strong(etag_value);
    if none_match(&etag, &req) {
        if let Ok(filepath) = blob_path(&identifier) {
            let result =
                web::block(move || create_preview(&filepath, width, height)).await?;
            match result {
                Ok(data) => Ok(HttpResponse::Ok()
                    .content_type("image/jpeg")
                    .append_header((header::CONTENT_LENGTH, data.len() as u64))
                    .append_header((header::ETAG, etag))
                    .body(data)),
                Err(e) => {
                    let is_not_found = e
                        .chain()
                        .filter_map(|cause| cause.downcast_ref::<io::Error>())
                        .any(|io_err| io_err.kind() == io::ErrorKind::NotFound);
                    if is_not_found {
                        return Ok(HttpResponse::NotFound().finish());
                    }
                    warn!("preview generation failed: {e:#}");
                    Ok(HttpResponse::TemporaryRedirect()
                        .append_header((header::LOCATION, "/public/placeholder.svg"))
                        .finish())
                }
            }
        } else {
            Ok(HttpResponse::BadRequest().finish())
        }
    } else {
        Ok(HttpResponse::NotModified().finish())
    }
}

/// Return a JSON blob describing everything we can extract from the asset:
/// ffprobe output for video, EXIF tags for images, an empty object otherwise.
async fn get_metadata(info: web::Path<String>) -> actix_web::Result<HttpResponse> {
    let Ok(filepath) = blob_path(&info) else {
        return Ok(HttpResponse::BadRequest().finish());
    };
    let result = web::block(move || {
        let json = extract_metadata(&filepath)?;
        let body = serde_json::to_vec(&json)?;
        if body.len() > METADATA_MAX_BYTES {
            return Err(MetadataTooLarge(body.len()).into());
        }
        Ok::<Vec<u8>, Error>(body)
    })
    .await?;
    match result {
        Ok(body) => Ok(HttpResponse::Ok()
            .content_type("application/json")
            .append_header((header::CONTENT_LENGTH, body.len() as u64))
            .body(body)),
        Err(e) => {
            let is_not_found = e
                .chain()
                .filter_map(|cause| cause.downcast_ref::<io::Error>())
                .any(|io_err| io_err.kind() == io::ErrorKind::NotFound);
            if is_not_found {
                return Ok(HttpResponse::NotFound().finish());
            }
            if let Some(MetadataTooLarge(size)) = e.downcast_ref::<MetadataTooLarge>() {
                warn!("metadata for {} exceeded size limit: {} bytes", info, size);
                let body = serde_json::json!({
                    "error": "metadata exceeds size limit",
                    "limit_bytes": METADATA_MAX_BYTES,
                    "actual_bytes": size,
                });
                let body = serde_json::to_vec(&body).unwrap_or_default();
                return Ok(HttpResponse::PayloadTooLarge()
                    .content_type("application/json")
                    .append_header((header::CONTENT_LENGTH, body.len() as u64))
                    .body(body));
            }
            warn!("metadata extraction failed for {}: {e:#}", info);
            Ok(HttpResponse::InternalServerError().finish())
        }
    }
}

// Returns true if `req` does not have an `If-None-Match` header matching `etag`.
fn none_match(etag: &header::EntityTag, req: &HttpRequest) -> bool {
    match req.get_header::<header::IfNoneMatch>() {
        Some(header::IfNoneMatch::Any) => false,
        Some(header::IfNoneMatch::Items(items)) => !items.iter().any(|item| item.weak_eq(etag)),
        None => true,
    }
}

// Receive the incoming file content as the body of the PUT request.
async fn put_asset(
    info: web::Path<String>,
    mut payload: web::Payload,
) -> actix_web::Result<HttpResponse> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    if let Ok(filepath) = blob_path(&info) {
        // store the file content to the given path
        let parent_dir = filepath
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
        tokio::fs::create_dir_all(parent_dir).await?;
        let mut file = match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&filepath)
            .await
        {
            Ok(file) => file,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                return Ok(HttpResponse::Conflict().finish());
            }
            Err(err) => return Err(err.into()),
        };
        // the body is a stream of Bytes objects
        while let Some(chunk) = payload.next().await {
            let data = chunk?;
            file.write_all(&data).await?;
        }
        // ensure file is readable by backup programs and the like
        #[cfg(target_family = "unix")]
        {
            use fs::Permissions;
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::set_permissions(&filepath, Permissions::from_mode(0o644)).await?;
        }
        info!("asset {} saved to {}", info, filepath.display());
        Ok(HttpResponse::Created().finish())
    } else {
        // if path conversion fails, probably client error
        Ok(HttpResponse::BadRequest().finish())
    }
}

async fn delete_asset(info: web::Path<String>) -> actix_web::Result<HttpResponse> {
    if let Ok(filepath) = blob_path(&info) {
        match fs::remove_file(&filepath) {
            Ok(_) => Ok(HttpResponse::Ok().finish()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(HttpResponse::NotFound().finish()),
            Err(e) => Err(e.into()),
        }
    } else {
        // if path conversion fails, probably client error
        Ok(HttpResponse::BadRequest().finish())
    }
}

#[actix_web::get("favicon.ico")]
async fn favicon() -> actix_web::Result<actix_files::NamedFile> {
    Ok(actix_files::NamedFile::open("./public/favicon.ico")?)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().ok();
    env_logger::init();
    std::fs::create_dir_all(ASSETS_PATH.as_path())?;
    let host = env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_owned());
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_owned());
    let addr = format!("{}:{}", host, port);

    let models_dir = synthetic::models_dir().unwrap_or_else(|e| {
        log::error!("synthetic: {e}");
        std::process::exit(1);
    });
    info!("synthetic: loading models from {}", models_dir.display());
    let engine = match synthetic::SyntheticEngine::new(&models_dir) {
        Ok(e) => web::Data::new(e),
        Err(e) => {
            log::error!("synthetic: failed to initialize engine: {e}");
            std::process::exit(1);
        }
    };

    info!("listening on {}", addr);
    HttpServer::new(move || {
        App::new()
            .wrap(actix_web::middleware::Logger::default())
            .app_data(engine.clone())
            .service(
                web::resource("/liveness")
                    .route(web::get().to(HttpResponse::Ok))
                    .route(web::head().to(HttpResponse::Ok)),
            )
            .service(
                actix_files::Files::new("/public", "./public")
                    .use_etag(true)
                    .use_last_modified(true),
            )
            .service(favicon)
            .route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail))
            .route("/preview/{id}", web::get().to(get_preview))
            .route("/metadata/{id}", web::get().to(get_metadata))
            .route("/synthetic/{id}", web::post().to(synthetic::post_synthetic))
            .route("/assets/{id}", web::get().to(get_asset))
            .route("/assets/{id}", web::head().to(get_asset))
            .route("/assets/{id}", web::put().to(put_asset))
            .route("/assets/{id}", web::delete().to(delete_asset))
            .route("/", web::get().to(index))
    })
    .bind(addr)?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_path_relative() {
        // echo -n 'foo/../bar' | base64 | tr '+/' '-_' | tr -d '='
        let result = blob_path("Zm9vLy4uL2Jhcg");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_path_controls() {
        // echo -en "foo\tbar" | base64 | tr '+/' '-_' | tr -d '='
        let result = blob_path("Zm9vCWJhcg");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_video() {
        assert!(is_video(Path::new("clip.mp4")));
        assert!(is_video(Path::new("clip.MOV")));
        assert!(is_video(Path::new("path/to/clip.webm")));
        assert!(is_video(Path::new("home.video.m4v")));
        assert!(is_video(Path::new("clip.qt")));
        assert!(!is_video(Path::new("photo.jpg")));
        assert!(!is_video(Path::new("notes.txt")));
        assert!(!is_video(Path::new("noext")));
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    use actix_web::http::header::{self, ContentType};
    use actix_web::{App, test};

    #[actix_web::test]
    async fn test_index_get() {
        let app = test::init_service(App::new().route("/", web::get().to(index))).await;
        let req = test::TestRequest::default()
            .insert_header(ContentType::plaintext())
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_get_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        let uri = "/assets/thisisnotbase64";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        )
    }

    #[actix_web::test]
    async fn test_get_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M"; // \windows
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw"; // c:\windows
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU"; // \\server\share
            let req = test::TestRequest::with_uri(uri).to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_get_asset_missing_file() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/assets/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        )
    }

    #[actix_web::test]
    async fn test_get_asset_trailing_newline() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2YxdC5qcGcK is 2019-04-15/0830/f1t.jpg\n
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGcK";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");
    }

    #[actix_web::test]
    async fn test_get_asset_find_jpeg() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc= is 2019-04-15/0830/f1t.jpg
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");
        let dispostion = resp.headers().get(header::CONTENT_DISPOSITION).unwrap();
        assert_eq!(dispostion, "inline; filename=\"f1t.jpg\"");

        // retrieve a second time with the same ETag to get a 304 response
        let etag = resp.headers().get(header::ETAG).unwrap();
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("If-None-Match", etag))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_MODIFIED
        );
    }

    #[actix_web::test]
    async fn test_get_asset_jpeg_attachment() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc= is 2019-04-15/0830/f1t.jpg
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc?attachment=yes";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");
        let dispostion = resp.headers().get(header::CONTENT_DISPOSITION).unwrap();
        assert_eq!(dispostion, "attachment; filename=\"f1t.jpg\"");
    }

    #[actix_web::test]
    async fn test_get_asset_plain_text() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "text/plain; charset=utf-8");
    }

    #[actix_web::test]
    async fn test_get_asset_range_request() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::get().to(get_asset))).await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/assets/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA";
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("Range", "bytes=500-999"))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::PARTIAL_CONTENT
        );
        let crange = resp.headers().get(header::CONTENT_RANGE).unwrap();
        #[cfg(unix)]
        assert_eq!(crange, "bytes 500-999/3129");
        #[cfg(windows)]
        assert_eq!(crange, "bytes 500-999/3138"); // Windows EOL
    }

    #[actix_web::test]
    async fn test_thumbnail_bad_encoding() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        let uri = "/thumbnail/480/320/thisisnotbase64";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        )
    }

    #[actix_web::test]
    async fn test_thumbnail_missing_file() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/thumbnail/480/320/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        )
    }

    #[actix_web::test]
    async fn test_thumbnail_not_image() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/thumbnail/480/320/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.headers().get(header::LOCATION).unwrap(),
            "/public/placeholder.svg"
        );
    }

    #[actix_web::test]
    async fn test_placeholder_image() {
        let app = test::init_service(
            App::new().service(
                actix_files::Files::new("/public", "./public")
                    .use_etag(true)
                    .use_last_modified(true),
            ),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/public/placeholder.svg";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/svg+xml");
        // actix_files::Files does not return content-length header?
    }

    #[actix_web::test]
    async fn test_thumbnail_valid_image() {
        let app = test::init_service(
            App::new().route("/thumbnail/{w}/{h}/{id}", web::get().to(get_thumbnail)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn is 2019-04-15/0830/fighting_kittens.jpg
        let uri = "/thumbnail/480/320/MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");

        // retrieve a second time with the same ETag to get a 304 response
        let etag = resp.headers().get(header::ETAG).unwrap();
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("If-None-Match", etag))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_MODIFIED
        );
    }

    #[actix_web::test]
    async fn test_preview_bad_encoding() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        let uri = "/preview/thisisnotbase64?height=300";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        )
    }

    #[actix_web::test]
    async fn test_preview_missing_dimensions() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn is 2019-04-15/0830/fighting_kittens.jpg
        let id = "MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn";

        // neither width nor height
        let req = test::TestRequest::with_uri(&format!("/preview/{id}")).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );

        // both width and height
        let req =
            test::TestRequest::with_uri(&format!("/preview/{id}?width=300&height=300")).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_preview_missing_file() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/preview/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw?height=300";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        )
    }

    #[actix_web::test]
    async fn test_preview_not_image() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/preview/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA?height=300";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.headers().get(header::LOCATION).unwrap(),
            "/public/placeholder.svg"
        );
    }

    #[actix_web::test]
    async fn test_preview_by_height() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn is 2019-04-15/0830/fighting_kittens.jpg
        let uri = "/preview/MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn?height=300";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        let ctype = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert_eq!(ctype, "image/jpeg");
        let etag = resp.headers().get(header::ETAG).unwrap().clone();
        // verify the displayed height matches the requested size
        let body = test::read_body(resp).await;
        let decoded = image::ImageReader::new(std::io::Cursor::new(body))
            .with_guessed_format()
            .expect("guess format")
            .decode()
            .expect("decode preview");
        assert_eq!(decoded.height(), 300);

        // retrieve a second time with the same ETag to get a 304 response
        let req = test::TestRequest::with_uri(uri)
            .insert_header(("If-None-Match", etag))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_redirection());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_MODIFIED
        );
    }

    #[actix_web::test]
    async fn test_preview_by_width() {
        let app = test::init_service(
            App::new().route("/preview/{id}", web::get().to(get_preview)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn is 2019-04-15/0830/fighting_kittens.jpg
        let uri = "/preview/MjAxOS0wNC0xNS8wODMwL2ZpZ2h0aW5nX2tpdHRlbnMuanBn?width=300";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        let body = test::read_body(resp).await;
        let decoded = image::ImageReader::new(std::io::Cursor::new(body))
            .with_guessed_format()
            .expect("guess format")
            .decode()
            .expect("decode preview");
        assert_eq!(decoded.width(), 300);
    }

    #[actix_web::test]
    async fn test_metadata_bad_encoding() {
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        let uri = "/metadata/thisisnotbase64";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_metadata_missing_file() {
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let uri = "/metadata/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        );
    }

    #[actix_web::test]
    async fn test_metadata_no_exif() {
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA== is 2019-04-15/0830/lorem-ipsum.txt
        let uri = "/metadata/MjAxOS0wNC0xNS8wODMwL2xvcmVtLWlwc3VtLnR4dA";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body = test::read_body(resp).await;
        let parsed: serde_json::Value =
            serde_json::from_slice(&body).expect("response is JSON");
        assert_eq!(parsed, serde_json::json!({}));
    }

    #[actix_web::test]
    async fn test_metadata_image_without_exif() {
        // Write a tiny PNG (the image crate's PNG encoder emits no EXIF
        // chunk) into the blob store, hit /metadata, then clean up. The
        // fallback should populate PixelXDimension/PixelYDimension from the
        // decoded header.
        let dest = std::path::PathBuf::from(DEFAULT_ASSETS_PATH)
            .join("2019-04-15/0830/dim-only.png");
        let img = image::RgbImage::from_pixel(7, 11, image::Rgb([255, 0, 0]));
        img.save(&dest).expect("write fixture PNG");
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2RpbS1vbmx5LnBuZw is 2019-04-15/0830/dim-only.png
        let uri = "/metadata/MjAxOS0wNC0xNS8wODMwL2RpbS1vbmx5LnBuZw";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        let body = test::read_body(resp).await;
        let _ = std::fs::remove_file(&dest);
        assert_eq!(status.as_u16(), actix_web::http::StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body).expect("response is JSON");
        assert_eq!(
            parsed,
            serde_json::json!({
                "PixelXDimension": { "description": "7 pixels", "value": [7] },
                "PixelYDimension": { "description": "11 pixels", "value": [11] },
            })
        );
    }

    #[actix_web::test]
    async fn test_metadata_image() {
        // dcp_1069.jpg is a pre-staged Kodak DC280 sample with a rich EXIF
        // header at tests/blobs/2019-04-15/0830/dcp_1069.jpg.
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL2RjcF8xMDY5LmpwZw is 2019-04-15/0830/dcp_1069.jpg
        let uri = "/metadata/MjAxOS0wNC0xNS8wODMwL2RjcF8xMDY5LmpwZw";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body = test::read_body(resp).await;
        let parsed: serde_json::Value =
            serde_json::from_slice(&body).expect("response is JSON");
        let obj = parsed.as_object().expect("metadata should be a JSON object");

        // Spot-check stable EXIF fields. Each tag is an object with a
        // `description` (the kamadak-exif display string, which wraps ASCII
        // in literal quotes and appends units to numerics) and a `value`
        // (the raw typed JSON: number arrays for integers/floats, [num,den]
        // arrays for rationals, string arrays for ASCII).
        let descr = |k: &str| -> Option<String> {
            obj.get(k)
                .and_then(|v| v.as_object())
                .and_then(|m| m.get("description"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        let value = |k: &str| -> Option<&serde_json::Value> {
            obj.get(k).and_then(|v| v.as_object()).and_then(|m| m.get("value"))
        };
        assert_eq!(descr("Make").as_deref(), Some("\"EASTMAN KODAK COMPANY\""));
        assert_eq!(
            descr("Model").as_deref(),
            Some("\"KODAK DC280 ZOOM DIGITAL CAMERA\"")
        );
        assert_eq!(descr("DateTimeOriginal").as_deref(), Some("2003-09-03 17:24:35"));
        assert_eq!(descr("ExposureTime").as_deref(), Some("1/125 s"));
        assert_eq!(descr("FNumber").as_deref(), Some("f/9.5"));
        assert_eq!(descr("PixelXDimension").as_deref(), Some("440 pixels"));
        assert_eq!(descr("ColorSpace").as_deref(), Some("sRGB"));

        // FNumber is a Rational; value should be a single [num, den] pair.
        let fnum_v = value("FNumber").and_then(|v| v.as_array()).expect("FNumber value");
        assert_eq!(fnum_v.len(), 1);
        let pair = fnum_v[0].as_array().expect("[num, den]");
        assert_eq!(pair.len(), 2);
        assert!(pair[0].as_u64().is_some() && pair[1].as_u64().is_some());

        // PixelXDimension is a Short/Long; value should be a single-element
        // number array.
        let pxw_v = value("PixelXDimension")
            .and_then(|v| v.as_array())
            .expect("PixelXDimension value");
        assert_eq!(pxw_v.len(), 1);
        assert!(pxw_v[0].as_u64().is_some());

        // Make is ASCII; value should be a string array containing the make.
        let make_v = value("Make").and_then(|v| v.as_array()).expect("Make value");
        assert!(
            make_v
                .iter()
                .any(|v| v.as_str() == Some("EASTMAN KODAK COMPANY")),
            "Make value should contain the raw ASCII string, got {make_v:?}"
        );

        // MakerNote and thumbnail-IFD entries must be stripped.
        assert!(!obj.contains_key("MakerNote"));
    }

    /// True if the named binary is on PATH (used to skip ffprobe-dependent
    /// tests in environments without ffmpeg installed).
    fn binary_on_path(name: &str) -> bool {
        std::process::Command::new(name)
            .arg("-version")
            .output()
            .is_ok()
    }

    #[actix_web::test]
    async fn test_metadata_video() {
        if !binary_on_path("ffprobe") {
            eprintln!("skipping test_metadata_video: ffprobe not on PATH");
            return;
        }
        // ooo_tracks.mp4 is a pre-staged H.264 sample at
        // tests/blobs/2019-04-15/0830/ooo_tracks.mp4.
        let app = test::init_service(
            App::new().route("/metadata/{id}", web::get().to(get_metadata)),
        )
        .await;
        // MjAxOS0wNC0xNS8wODMwL29vb190cmFja3MubXA0 is 2019-04-15/0830/ooo_tracks.mp4
        let uri = "/metadata/MjAxOS0wNC0xNS8wODMwL29vb190cmFja3MubXA0";
        let req = test::TestRequest::with_uri(uri).to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body = test::read_body(resp).await;
        let parsed: serde_json::Value =
            serde_json::from_slice(&body).expect("response is JSON");

        // Spot-check stable fields from ffprobe's -show_format / -show_streams
        // output. Numeric video properties surface as JSON numbers, while
        // ffprobe reports duration/bit_rate as strings.
        let streams = parsed
            .get("streams")
            .and_then(|v| v.as_array())
            .expect("streams array");
        assert!(!streams.is_empty(), "expected at least one stream");
        let video = streams
            .iter()
            .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("video"))
            .expect("expected a video stream");
        assert_eq!(
            video.get("codec_name").and_then(|v| v.as_str()),
            Some("h264")
        );
        assert_eq!(video.get("width").and_then(|v| v.as_u64()), Some(816));
        assert_eq!(video.get("height").and_then(|v| v.as_u64()), Some(608));

        let format = parsed.get("format").expect("format object");
        let format_name = format
            .get("format_name")
            .and_then(|v| v.as_str())
            .expect("format_name");
        assert!(
            format_name.contains("mp4"),
            "format_name should mention mp4, got {format_name}"
        );
        assert_eq!(
            format.get("duration").and_then(|v| v.as_str()),
            Some("3.500000")
        );
    }

    #[actix_web::test]
    async fn test_put_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let req = test::TestRequest::with_uri("/assets/thisisnotbase64")
            .method(actix_web::http::Method::PUT)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_put_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M"; // \windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw"; // c:\windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU"; // \\server\share
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::PUT)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_put_asset_conflict() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YxdC5qcGc")
            .method(actix_web::http::Method::PUT)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::CONFLICT
        );
    }

    fn checksum_file(infile: &Path) -> std::io::Result<String> {
        let mut file = std::fs::File::open(infile)?;
        let mut hasher = blake3::Hasher::new();
        std::io::copy(&mut file, &mut hasher)?;
        let digest = hasher.finalize();
        Ok(format!("{digest}"))
    }

    #[actix_web::test]
    async fn test_put_asset_ok() {
        // clean up from previous failed runs
        let expected_path = Path::new("./tests/blobs/2019-04-15/0830/f2t.jpg");
        if expected_path.exists() {
            std::fs::remove_file(expected_path).expect("delete file");
        }

        //
        // request should look something like this:
        //
        // PUT /assets/BASE64-ENCODED-PATH HTTP/1.1
        // Content-Type: xxx/yyy
        // Content-Length: nnn
        //
        // [raw data of the incoming file]
        let app =
            test::init_service(App::new().route("/assets/{id}", web::put().to(put_asset))).await;
        let filepath = "./tests/fixtures/f2t.jpg";
        let payload = std::fs::read(filepath).expect("file read");

        // MjAxOS0wNC0xNS8wODMwL2YydC5qcGc= is 2019-04-15/0830/f2t.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YydC5qcGc")
            .method(actix_web::http::Method::PUT)
            .append_header((header::CONTENT_TYPE, "image/jpeg"))
            .append_header((header::CONTENT_LENGTH, payload.len()))
            .set_payload(payload)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::CREATED);
        assert!(expected_path.exists());
        let digest = checksum_file(expected_path).expect("checksum");
        assert_eq!(
            digest,
            "72e32e7ef56e4b29d5f8897496c4b3dd8ca338a80ee026bb406e8d59f679d908"
        );
        std::fs::remove_file(expected_path).expect("delete file");
    }

    #[actix_web::test]
    async fn test_delete_asset_bad_encoding() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        let req = test::TestRequest::with_uri("/assets/thisisnotbase64")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::BAD_REQUEST
        );
    }

    #[actix_web::test]
    async fn test_delete_asset_root_paths() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;

        if cfg!(unix) {
            let uri = "/assets/L3Jvb3Qvc2VjcmV0L2ZpbGVz"; // /root/secret/files
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }

        if cfg!(windows) {
            let uri = "/assets/XHdpbmRvd3M"; // \windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/Yzpcd2luZG93cw"; // c:\windows
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );

            let uri = "/assets/XFxzZXJ2ZXJcc2hhcmU"; // \\server\share
            let req = test::TestRequest::with_uri(uri)
                .method(actix_web::http::Method::DELETE)
                .to_request();
            let resp = test::call_service(&app, req).await;
            assert!(resp.status().is_client_error());
            assert_eq!(
                resp.status().as_u16(),
                actix_web::http::StatusCode::BAD_REQUEST
            );
        }
    }

    #[actix_web::test]
    async fn test_delete_asset_missing_file() {
        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        // MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw== is 2019/08/17/0430/image.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS8wOC8xNy8wNDMwL2ltYWdlLmpwZw")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_client_error());
        assert_eq!(
            resp.status().as_u16(),
            actix_web::http::StatusCode::NOT_FOUND
        );
    }

    #[actix_web::test]
    async fn test_delete_asset_ok() {
        let source_path = Path::new("./tests/fixtures/f2t.jpg");
        // use a different destination to avoid Windows "used by another process" error
        let target_path = Path::new("./tests/blobs/2019-04-15/0830/f3t.jpg");
        std::fs::copy(source_path, target_path).expect("file copy");

        let app =
            test::init_service(App::new().route("/assets/{id}", web::delete().to(delete_asset)))
                .await;
        // MjAxOS0wNC0xNS8wODMwL2YzdC5qcGc= is 2019-04-15/0830/f3t.jpg
        let req = test::TestRequest::with_uri("/assets/MjAxOS0wNC0xNS8wODMwL2YzdC5qcGc")
            .method(actix_web::http::Method::DELETE)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), actix_web::http::StatusCode::OK);
        assert!(!target_path.exists());
    }
}
